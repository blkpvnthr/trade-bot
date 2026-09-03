"""
instance2_status_web.py — read-only status endpoint for trade-bot INSTANCE 2.

Mirrors the webhook.py health pattern (Flask @app.get("/")), but is STRICTLY
read-only: it NEVER constructs an Alpaca client and exposes NO order routes.
Liveness/metrics come only from this instance's own runtime files:
  - logs/run.<mode>.lock                  -> pid + started_utc (uptime, liveness)
  - system-data/dashboard_snapshot.json   -> equity / pnl / positions / freshness

Loads ONLY the dedicated .env.nonl0cal2 (via ENV_FILE), matching the
instance-isolation guarantee. Bind 127.0.0.1:8002 behind nginx.
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, Response

PROJECT_ROOT = Path(__file__).resolve().parent

# Instance-isolated env: load ONLY ENV_FILE (default to .env.nonl0cal2). No
# parent/shared .env, consistent with run.py's loader.
_ef = os.environ.get("ENV_FILE") or str(PROJECT_ROOT / ".env.nonl0cal2")
try:
    from dotenv import load_dotenv
    load_dotenv(_ef, override=True)
except Exception:
    pass

BOT_NAME = os.environ.get("INSTANCE_LABEL") or os.environ.get("INSTANCE") or "instance2"
MODE = os.environ.get("MODE", "paper").lower()
HOST_IP = os.environ.get("INSTANCE_HOST_IP", "107.170.25.88")
LOCK_PATH = PROJECT_ROOT / "logs" / ("run.%s.lock" % MODE)
SNAPSHOT_PATH = PROJECT_ROOT / "system-data" / "dashboard_snapshot.json"

app = Flask(__name__)


def _now():
    return datetime.now(timezone.utc)


def _human(secs):
    secs = int(secs)
    if secs < 60:
        return "%d secs" % secs
    m = secs // 60
    if m < 60:
        return "%d mins" % m
    h, m = m // 60, m % 60
    if h < 24:
        return "%dh %dm" % (h, m)
    d, h = h // 24, h % 24
    return "%dd %dh" % (d, h)


def _read_lock():
    info = {"pid": None, "started_utc": None, "running": False,
            "uptime_seconds": None, "uptime": "N/A"}
    try:
        for tok in LOCK_PATH.read_text().strip().split():
            if tok.startswith("pid="):
                info["pid"] = int(tok.split("=", 1)[1])
            elif tok.startswith("started_utc="):
                info["started_utc"] = tok.split("=", 1)[1]
    except Exception:
        return info
    if info["pid"]:
        try:
            os.kill(info["pid"], 0)
            info["running"] = True
        except OSError:
            info["running"] = False
    if info["started_utc"]:
        try:
            secs = (_now() - datetime.fromisoformat(info["started_utc"])).total_seconds()
            info["uptime_seconds"] = int(secs)
            info["uptime"] = _human(secs)
        except Exception:
            pass
    return info


def _read_snapshot():
    try:
        data = json.loads(SNAPSHOT_PATH.read_text())
    except Exception:
        return None, None, "N/A"
    upd = data.get("updated_at_utc")
    if not upd:
        return data, None, "N/A"
    try:
        age = (_now() - datetime.fromisoformat(upd)).total_seconds()
        return data, age, _human(age)
    except Exception:
        return data, None, "N/A"


def _status():
    lock = _read_lock()
    snap, age, age_h = _read_snapshot()
    acct = (snap or {}).get("account", {}) or {}
    return {
        "ok": True,
        "bot": BOT_NAME,
        "host": HOST_IP,
        "mode": MODE,
        "paper": MODE != "live",
        "running": lock["running"],
        "pid": lock["pid"],
        "uptime": lock["uptime"],
        "uptime_seconds": lock["uptime_seconds"],
        "snapshot_age": age_h,
        "snapshot_age_seconds": int(age) if age is not None else None,
        "equity": acct.get("equity"),
        "day_pnl": acct.get("day_pnl"),
        "cash": acct.get("cash"),
        "positions": len((snap or {}).get("positions", []) or []),
        "source": (snap or {}).get("source"),
        "env_file": _ef,
        "updated_at": (snap or {}).get("updated_at_utc"),
        "timestamp": _now().isoformat(),
    }


@app.get("/")
def index():
    s = _status()
    if "application/json" in request.headers.get("Accept", "") or request.args.get("format") == "json":
        return jsonify(s)
    state = "RUNNING" if s["running"] else "DOWN"
    dot = "#3fb950" if s["running"] else "#f85149"
    html = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{bot} - trade-bot status</title>
<style>body{{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
.card{{max-width:640px;margin:40px auto;background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px}}
h1{{margin:0 0 4px;font-size:20px;display:flex;align-items:center;gap:10px}}
.dot{{width:12px;height:12px;border-radius:50%;background:{dot};display:inline-block}}
.sub{{color:#8b949e;font-size:13px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse}} td{{padding:8px 4px;border-bottom:1px solid #21262d;font-size:14px}}
td.k{{color:#8b949e;width:42%}}</style></head>
<body><div class=card>
<h1><span class=dot></span>{bot} &middot; {state}</h1>
<div class=sub>trade-bot instance &middot; {host} &middot; {mode} mode</div>
<table>
<tr><td class=k>Status</td><td>{state}</td></tr>
<tr><td class=k>PID</td><td>{pid}</td></tr>
<tr><td class=k>Uptime</td><td>{uptime}</td></tr>
<tr><td class=k>Snapshot age</td><td>{snapshot_age}</td></tr>
<tr><td class=k>Equity</td><td>{equity}</td></tr>
<tr><td class=k>Day P&amp;L</td><td>{day_pnl}</td></tr>
<tr><td class=k>Positions</td><td>{positions}</td></tr>
<tr><td class=k>Host IP</td><td>{host}</td></tr>
<tr><td class=k>Env file</td><td>{env_file}</td></tr>
<tr><td class=k>Updated (UTC)</td><td>{updated_at}</td></tr>
</table></div></body></html>""".format(
        bot=s["bot"], state=state, dot=dot, host=s["host"], mode=s["mode"],
        pid=s["pid"], uptime=s["uptime"], snapshot_age=s["snapshot_age"],
        equity=s["equity"], day_pnl=s["day_pnl"], positions=s["positions"],
        env_file=os.path.basename(s["env_file"] or ""), updated_at=s["updated_at"])
    return Response(html, mimetype="text/html")


@app.get("/healthz")
def healthz():
    s = _status()
    return jsonify({"ok": True, "running": s["running"], "bot": s["bot"]}), (200 if s["running"] else 503)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8002, debug=False)
