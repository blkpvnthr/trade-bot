#!/usr/bin/env python3
"""
dashboard.py — phone-accessible trading dashboard (Flask, served behind nginx at /dash, token-gated).

Pages (all under DASH_PREFIX, default /dash):
  /            overview — today + this-month headline metrics
  /monthly             current month  (redirects to /monthly/<YYYY-MM>)
  /monthly/<YYYY-MM>   per-DAY metrics for that month + monthly totals; prev/next month nav
  /journal             recent round-trips (FIFO-matched), newest first

Data: Alpaca filled orders -> FIFO round-trips (realized P&L per closed trade) + the account
portfolio-history (daily equity / account P&L). Mobile-first HTML, no external assets.

Env (.env): APCA_API_KEY_ID, APCA_API_SECRET_KEY, DASH_TOKEN (required to view), DASH_PREFIX (/dash).
"""
import os
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, request, Response, redirect, jsonify, session

ROOT = Path(__file__).resolve().parent
PREFIX = os.environ.get("DASH_PREFIX", "/dash").rstrip("/")

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-4))


def _load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
from order_emailer import fetch_fills, round_trips, _occ   # reuse FIFO matcher

LOG_FILE = ROOT / "logs" / "run.log"
app = Flask(__name__)
DASH_TOKEN = os.environ.get("DASH_TOKEN", "")
DASH_PASSWORD = os.environ.get("DASH_PASSWORD", "")
app.secret_key = os.environ.get("DASH_SECRET") or ("qq-" + (DASH_PASSWORD or DASH_TOKEN or "dev"))
app.config["PREFERRED_URL_SCHEME"] = "https"   # OAuth redirect URIs must be https behind nginx
try:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
except Exception:
    pass
app.config.update(SESSION_COOKIE_NAME="qqdash", SESSION_COOKIE_PATH=(PREFIX or "/"),
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SECURE=True,
                  SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(days=60))
_CACHE = {}


@app.after_request
def _nocache(resp):
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=False)


def get_trips(ttl=60):
    now = time.time()
    if _CACHE.get("trips") is not None and now - _CACHE.get("trips_t", 0) < ttl:
        return _CACHE["trips"]
    try:
        trips = round_trips(fetch_fills(_client(), days=420))
        _CACHE["trips"], _CACHE["trips_t"] = trips, now     # only cache on SUCCESS
    except Exception as e:
        # Alpaca unavailable (throttle/auth) must NOT 500 the page — degrade to last-known.
        import logging as _lg
        _lg.getLogger("dashboard").warning("get_trips failed, keeping last-known: %s", str(e)[:120])
        trips = _CACHE.get("trips") or []
    return trips


def get_ph(ttl=300):
    now = time.time()
    if _CACHE.get("ph") is not None and now - _CACHE.get("ph_t", 0) < ttl:
        return _CACHE["ph"]
    import requests
    try:
        r = requests.get("https://api.alpaca.markets/v2/account/portfolio/history",
                         params={"period": "1A", "timeframe": "1D", "extended_hours": "false"},
                         headers={"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
                                  "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
                                  "User-Agent": "Mozilla/5.0 (QuantumQuant-Dash/1.0)"}, timeout=20)
        ph = r.json() if r.status_code < 300 else {}
    except Exception:
        ph = {}
    _CACHE["ph"], _CACHE["ph_t"] = ph, now
    return ph


def ph_by_date(ph):
    out = {}
    ts, eq, pl, plp = (ph.get("timestamp", []), ph.get("equity", []),
                       ph.get("profit_loss", []), ph.get("profit_loss_pct", []))
    for i, t in enumerate(ts):
        d = datetime.fromtimestamp(t, ET).date().isoformat()
        out[d] = {"equity": eq[i] if i < len(eq) else None,
                  "pl": pl[i] if i < len(pl) else None,
                  "plp": (plp[i] if i < len(plp) else None)}
    return out


# Account + positions are read from the bot's own snapshot (system-data/dashboard_snapshot.json),
# written each cycle by run.py:log_account_stats. The dashboard needs NO broker credentials and is
# immune to Alpaca auth/rate-limit issues — it shows whatever the bot last successfully read.
SNAP_PATH = ROOT / "system-data" / "dashboard_snapshot.json"


def _read_snapshot():
    try:
        import json as _json
        return _json.loads(SNAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def snapshot_age_sec():
    try:
        return max(0, int(time.time() - SNAP_PATH.stat().st_mtime))
    except Exception:
        return None


def get_account_snapshot(ttl=30):
    a = _read_snapshot().get("account", {})
    eq = float(a.get("equity", 0) or 0)
    le = float(a.get("last_equity", eq) or eq)
    return {"equity": eq, "last_equity": le, "cash": float(a.get("cash", 0) or 0),
            "day_pnl": float(a.get("day_pnl", eq - le) or 0),
            "day_pnl_pct": float(a.get("day_pnl_pct", ((eq - le) / le if le else 0.0)) or 0.0)}


def get_positions(ttl=30):
    from types import SimpleNamespace
    return [SimpleNamespace(**p) for p in _read_snapshot().get("positions", [])]


def _fmt_sym(sym):
    """OCC option symbols -> readable 'AAPL 312.5C 06/08'; equities unchanged."""
    if not _occ(sym):
        return sym
    und, exp, cp, strike = sym[:-15], sym[-15:-9], sym[-9], int(sym[-8:]) / 1000
    return f"{und} {strike:g}{cp} {exp[2:4]}/{exp[4:6]}"


def equity_series(ph, current=None):
    """[(datetime, equity)] from portfolio history (zeros/None dropped), with the live equity appended
    as the latest point so the chart includes today's intraday value."""
    out = []
    ts, eq = ph.get("timestamp", []), ph.get("equity", [])
    for i, e in enumerate(eq):
        if e is None or e <= 0:
            continue
        out.append((datetime.fromtimestamp(ts[i], ET), float(e)))
    if current and current > 0:
        now = datetime.now(ET)
        if out and out[-1][0].date() == now.date():
            out[-1] = (now, current)
        else:
            out.append((now, current))
    return out


def equity_svg(series, w=720, h=170):
    if len(series) < 2:
        return '<div class="sub">building equity history…</div>'
    vals = [v for _, v in series]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(series)
    pts = [(i / (n - 1) * w, h - (v - lo) / rng * h) for i, (_, v) in enumerate(series)]
    coords = " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    line = "M" + coords
    area = f"M{pts[0][0]:.1f},{h} L{coords} L{pts[-1][0]:.1f},{h} Z"
    c = "#3fb950" if vals[-1] >= vals[0] else "#f85149"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'style="width:100%;height:{h}px;display:block">'
            f'<path d="{area}" fill="{c}" fill-opacity="0.12"/>'
            f'<path d="{line}" fill="none" stroke="{c}" stroke-width="2" '
            f'vector-effect="non-scaling-stroke"/></svg>')


def equity_hybrid_series(ph, current=None):
    """Merge the clean DAILY portfolio-history curve (all days strictly before today) with
    today's INTRADAY per-cycle equity from run.log (the $100k paper window is already filtered
    out in udf_datafeed._equity_points). Returns [(unix_ts, equity)] ascending & unique.
    Fail-safe: on any error, falls back to the plain daily series."""
    try:
        now = datetime.now(ET)
        cutoff = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        out = {int(dt.timestamp()): float(v)
               for dt, v in equity_series(ph, current=None) if dt.date() < now.date()}
        import udf_datafeed as _udf
        for t, v in _udf._equity_points():
            if t >= cutoff:
                out[int(t)] = float(v)                  # today's intraday detail
        if current and current > 0:
            out[int(now.timestamp())] = float(current)  # pin the freshest live value as the last point
        merged = sorted(out.items())
        if len(merged) >= 2:
            return merged
    except Exception:
        pass
    return [(int(dt.timestamp()), float(v)) for dt, v in equity_series(ph, current=current)]


def equity_lwc(series, svg_fallback, h=170):
    """Render the account-equity curve with TradingView Lightweight Charts (open-source, CDN).
    Progressive enhancement: emits the existing SVG first, then upgrades it to an interactive
    chart if the library loads (so non-JS / failed-CDN still shows the sparkline).
    series = [(unix_ts, value)]."""
    import json as _json
    if len(series) < 2:
        return svg_fallback
    up = series[-1][1] >= series[0][1]
    line_c = "#3fb950" if up else "#f85149"
    top = "rgba(63,185,80,0.25)" if up else "rgba(248,81,73,0.25)"
    bot = "rgba(63,185,80,0.02)" if up else "rgba(248,81,73,0.02)"
    data = _json.dumps([{"time": int(t), "value": round(float(v), 2)} for t, v in series])
    return (
        f'<div id="eqc" style="height:{h}px">{svg_fallback}</div>'
        '<script src="https://unpkg.com/lightweight-charts@5/dist/lightweight-charts.standalone.production.js"></script>'
        '<script>(function(){'
        f'var data={data};'
        'var LWC=window.LightweightCharts;'
        'if(!(LWC&&LWC.createChart&&LWC.AreaSeries&&data.length>1))return;'
        'var el=document.getElementById("eqc");if(!el)return;el.innerHTML="";'
        f'var chart=LWC.createChart(el,{{height:{h},width:el.clientWidth,'
        'layout:{background:{type:"solid",color:"transparent"},textColor:"#8b949e",attributionLogo:false},'
        'grid:{vertLines:{visible:false},horzLines:{color:"rgba(255,255,255,0.05)"}},'
        'rightPriceScale:{borderVisible:false},'
        'timeScale:{borderVisible:false,timeVisible:true,secondsVisible:false},'
        'handleScroll:false,handleScale:false,'
        'localization:{priceFormatter:function(p){return "$"+p.toFixed(0);}}});'
        f'var s=chart.addSeries(LWC.AreaSeries,{{lineColor:"{line_c}",lineWidth:2,'
        f'topColor:"{top}",bottomColor:"{bot}",priceLineVisible:false,lastValueVisible:true}});'
        's.setData(data);chart.timeScale().fitContent();'
        f'new ResizeObserver(function(){{chart.resize(el.clientWidth,{h});}}).observe(el);'
        '})();</script>'
    )


def daily_from_trips(trips):
    days = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0, "gw": 0.0, "gl": 0.0, "vol": 0.0,
                                "best": None, "worst": None})
    for t in trips:
        d = t["exit_time"].astimezone(ET).date().isoformat()
        x = days[d]
        x["pnl"] += t["pnl"]; x["n"] += 1
        if t["pnl"] >= 0:
            x["wins"] += 1; x["gw"] += t["pnl"]
        else:
            x["gl"] += abs(t["pnl"])
        x["vol"] += t["exit_px"] * t["qty"] * (100.0 if _occ(t["symbol"]) else 1.0)
        x["best"] = t["pnl"] if x["best"] is None else max(x["best"], t["pnl"])
        x["worst"] = t["pnl"] if x["worst"] is None else min(x["worst"], t["pnl"])
    return days


_ORDER_RE = re.compile(
    r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\s+\w+\s+\[[^\]]*\]\s+ORDER (buy|sell) (\S+)\b.*?reason=(\S+)')


def reason_index(ttl=60):
    """Parse run.log 'ORDER <side> <SYM> ... reason=...' lines -> [(utc_time, side, symbol, reason)]."""
    now = time.time()
    if _CACHE.get("ridx") is not None and now - _CACHE.get("ridx_t", 0) < ttl:
        return _CACHE["ridx"]
    idx = []
    try:
        out = subprocess.run(["tail", "-n", "80000", str(LOG_FILE)],
                             capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            m = _ORDER_RE.match(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                idx.append((t, m.group(2), m.group(3), m.group(4)))
    except Exception:
        pass
    _CACHE["ridx"], _CACHE["ridx_t"] = idx, now
    return idx


def reason_at(idx, side, symbol, when, window=180):
    """Closest run.log reason for (side, symbol) within `window` seconds of `when`. None if no match."""
    if when is None:
        return None
    best, best_d = None, window + 1
    for (t, sd, sym, reason) in idx:
        if sd == side and sym == symbol:
            d = abs((t - when).total_seconds())
            if d < best_d:
                best_d, best = d, reason
    return best


def trip_logic(idx, t):
    """(entry_reason, exit_reason) for a round-trip; webhook options fall back to a TradingView label."""
    sym = t["symbol"]
    entry_t = t.get("entry_time") or (t["exit_time"] - timedelta(hours=t.get("hold_h", 0)))
    er = reason_at(idx, "buy", sym, entry_t)
    xr = reason_at(idx, "sell", sym, t["exit_time"])
    if _occ(sym):
        er = er or "TradingView webhook (option entry)"
        xr = xr or "TradingView webhook (option exit)"
    return er or "—", xr or "—"


def _pf(gw, gl):
    return (gw / gl) if gl > 1e-9 else (float("inf") if gw > 0 else 0.0)


def _money(v):
    if v is None:
        return "—"
    return f"${v:,.2f}" if abs(v) < 1e7 else f"${v:,.0f}"


def _pnl(v):
    if v is None:
        return '<span class="muted">—</span>'
    cls = "pos" if v >= 0 else "neg"
    return f'<span class="{cls}">{"+" if v >= 0 else "−"}${abs(v):,.2f}</span>'


def _pct(v):
    if v is None:
        return "—"
    cls = "pos" if v >= 0 else "neg"
    return f'<span class="{cls}">{v*100:+.2f}%</span>'


def _pf_str(pf):
    return "∞" if pf == float("inf") else f"{pf:.2f}"


# ── auth ──────────────────────────────────────────────────────────────────────
def _authed():
    if session.get("auth"):
        return True
    if DASH_TOKEN and (request.args.get("token") == DASH_TOKEN
                       or request.cookies.get("dash_token") == DASH_TOKEN):
        return True
    return False


def need_auth(f):
    @wraps(f)
    def w(*a, **k):
        if not _authed():
            if request.method == "POST" or request.path.endswith(("/trade", "/logs/tail")):
                return jsonify({"ok": False, "error": "unauthorized"}), 401
            return redirect(PREFIX + "/login")
        resp = app.make_response(f(*a, **k))
        if DASH_TOKEN and request.args.get("token") == DASH_TOKEN:
            resp.set_cookie("dash_token", DASH_TOKEN, max_age=86400 * 60, secure=True,
                            httponly=True, samesite="Lax")
        return resp
    return w


import dash_auth


def login_page(err=""):
    msg = f'<div class="neg" style="margin-bottom:10px">{err}</div>' if err else ""
    prov = dash_auth.providers()
    btns = ""
    if prov["google"]:
        btns += (f'<a class="btn" href="{PREFIX}/auth/google" style="display:block;background:#fff;'
                 'color:#202124;width:100%;margin-bottom:8px">Continue with Google</a>')
    if prov["github"]:
        btns += (f'<a class="btn" href="{PREFIX}/auth/github" style="display:block;background:#24292f;'
                 'color:#fff;width:100%;margin-bottom:8px">Continue with GitHub</a>')
    sep = '<div class="sub muted" style="margin:10px 0">or sign in with email</div>' if btns else ""
    return Response(f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Sign in · QuantumQuant</title>
<style>{CSS}</style></head><body><div class=wrap>
<div class="card" style="max-width:330px;margin:11vh auto;text-align:center">
<h1 style="margin-bottom:16px">QuantumQuant</h1>{msg}{btns}{sep}
<form method=post action="{PREFIX}/login">
<input class=minput type=email name=email placeholder="Email (optional)" autocomplete=username
style="width:100%;margin-bottom:8px">
<input class=minput type=password name=password placeholder=Password autocomplete=current-password
style="width:100%;margin-bottom:12px">
<button class="btn" style="background:#1f6feb;width:100%">Sign in</button></form>
</div></div></body></html>""", mimetype="text/html")


_ERR_MSG = {"oauth": "Sign-in failed — please try again.",
            "not_allowed": "That account is not authorized for this dashboard.",
            "oauth_unavailable": "That provider is not configured yet."}


@app.route(PREFIX + "/login", methods=["GET", "POST"])
def login():
    if _authed():
        return redirect(PREFIX + "/")
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        pw = request.form.get("password") or ""
        if email and dash_auth.email_login(email, pw):                  # email/password (allow-listed)
            return redirect(PREFIX + "/")
        if not email and DASH_PASSWORD and pw == DASH_PASSWORD:          # legacy single-password
            session.permanent = True
            session["auth"] = True
            return redirect(PREFIX + "/")
        return login_page("Incorrect email or password")
    return login_page(_ERR_MSG.get(request.args.get("err", ""), ""))


@app.route(PREFIX + "/logout")
def logout():
    session.clear()
    return redirect(PREFIX + "/login")


dash_auth.register_auth_routes(app, PREFIX)   # Google + GitHub OAuth callbacks (fail-safe)


# ── shared HTML chrome ────────────────────────────────────────────────────────
CSS = """
*{box-sizing:border-box}body{margin:0;background:#0d1117;color:#e6edf3;
font:15px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;-webkit-text-size-adjust:100%}
a{color:#58a6ff;text-decoration:none}.wrap{max-width:760px;margin:0 auto;padding:14px}
.nav{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.nav a{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 12px;font-weight:600}
.nav a.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
h1{font-size:18px;margin:4px 0 12px}h2{font-size:15px;color:#8b949e;margin:18px 0 8px;font-weight:600}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}
.card .k{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:22px;font-weight:700;margin-top:4px}
.pos{color:#3fb950}.neg{color:#f85149}.muted{color:#6e7681}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{padding:8px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{color:#8b949e;font-weight:600;position:sticky;top:0;background:#0d1117}
tr.tot td{font-weight:700;border-top:2px solid #30363d;background:#161b22}
tr.detail td{text-align:left;color:#9da7b1;font-size:11px;white-space:normal;word-break:break-word;
padding:2px 6px 9px;border-bottom:1px solid #21262d}
tr.main td{border-bottom:none}
.lbl{color:#6e7681;text-transform:uppercase;font-size:9px;border:1px solid #30363d;border-radius:4px;
padding:1px 4px;margin-right:4px;letter-spacing:.03em}.lbl.x{color:#d29922;border-color:#5a4a1a}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.sub{color:#6e7681;font-size:12px;margin-top:2px}
.logbox{background:#010409;border:1px solid #30363d;border-radius:10px;padding:10px;
height:72vh;overflow:auto;font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:pre-wrap;word-break:break-word;color:#c9d1d9}
tr.tap{cursor:pointer}tr.tap:active td{background:#1c2330}
.modalbg{display:none;position:fixed;inset:0;background:rgba(1,4,9,.7);z-index:50;
align-items:flex-end;justify-content:center}
.modal{background:#161b22;border:1px solid #30363d;border-radius:16px 16px 0 0;padding:18px;
width:100%;max-width:760px;animation:up .15s ease-out}
@keyframes up{from{transform:translateY(40px)}to{transform:none}}
.mhead{display:flex;justify-content:space-between;align-items:center;font-size:18px;font-weight:700;margin-bottom:14px}
.mclose{color:#8b949e;font-size:22px}.mrow{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.mrow span{color:#8b949e;width:42px}
.minput{flex:1;background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:13px;color:#e6edf3;font-size:18px}
.mbtns{display:flex;gap:10px;margin-top:8px}
.btn{flex:1;padding:15px;border:none;border-radius:12px;font-size:16px;font-weight:700;color:#fff}
.btn.buy{background:#238636}.btn.sell{background:#da3633}
.mresult{min-height:22px;font-size:14px;margin:6px 0}.hint{color:#6e7681;font-size:11px}
"""

MODAL = """
<div id=modal class=modalbg onclick="if(event.target==this)closeModal()">
 <div class=modal>
  <div class=mhead><span id=m-sym>—</span><a href=# class=mclose onclick="closeModal();return false">✕</a></div>
  <div class=mrow><span>Qty</span><input id=m-qty type=number step=any min=0 inputmode=decimal class=minput></div>
  <div class=mrow><span>Pass</span><input id=m-pw type=password autocomplete=current-password class=minput placeholder="Password to confirm order"></div>
  <div id=m-result class=mresult></div>
  <div class=mbtns><button type=button class="btn buy" onclick="doTrade('buy')">BUY</button>
  <button type=button class="btn sell" onclick="doTrade('sell')">SELL</button></div>
  <div class=hint>Market order, DAY. Password required. Tap a side twice to confirm.</div>
 </div>
</div>
<script>
var M_SYM='',PENDING=null;
function openTrade(s,q){M_SYM=s;document.getElementById('m-sym').textContent=s;
 document.getElementById('m-qty').value=q;document.getElementById('m-pw').value='';
 document.getElementById('m-result').textContent='';
 PENDING=null;document.getElementById('modal').style.display='flex';}
function closeModal(){document.getElementById('modal').style.display='none';PENDING=null;}
function doTrade(side){var q=document.getElementById('m-qty').value;
 var pw=document.getElementById('m-pw').value;var r=document.getElementById('m-result');
 if(!(parseFloat(q)>0)){r.innerHTML='<span class=neg>enter a qty</span>';return;}
 if(!pw){r.innerHTML='<span class=neg>enter your password</span>';return;}
 if(PENDING!==side){PENDING=side;r.innerHTML='<b>Tap '+side.toUpperCase()+' again to confirm '+side+' '+q+' '+M_SYM+'</b>';setTimeout(function(){PENDING=null;},4000);return;}
 PENDING=null;r.textContent='submitting…';
 var fd=new FormData();fd.append('symbol',M_SYM);fd.append('side',side);fd.append('qty',q);fd.append('password',pw);
 fetch('__PREFIX__/trade',{method:'POST',body:fd,credentials:'same-origin'}).then(function(x){return x.json();}).then(function(j){
  r.innerHTML=j.ok?'<span class=pos>\\u2713 '+side.toUpperCase()+' '+j.qty+' '+j.symbol+' \\u2014 '+j.status+(j.cancelled?' (freed '+j.cancelled+')':'')+'</span>':'<span class=neg>\\u2717 '+(j.error||'failed')+'</span>';
  if(j.ok){document.getElementById('m-pw').value='';setTimeout(function(){location.reload();},1500);}}).catch(function(){r.innerHTML='<span class=neg>\\u2717 network error</span>';});}
</script>
""".replace("__PREFIX__", PREFIX)


def page(title, body, active=""):
    def navlink(href, label, key):
        on = " on" if key == active else ""
        return f'<a class="navlink{on}" href="{PREFIX}{href}">{label}</a>'
    nav = (navlink("/", "Overview", "home") + navlink("/positions", "Positions", "positions")
           + navlink("/monthly", "Monthly", "monthly")
           + navlink("/journal", "Journal", "journal") + navlink("/logs", "Logs", "logs")
           + navlink("/backtest", "Strategy", "backtest") + navlink("/chart", "Chart", "chart"))
    return Response(f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{title} · QuantumQuant</title><style>{CSS}</style></head>
<body><div class=wrap><div class=nav>{nav}</div>{body}
<div class=sub>QuantumQuant · live · updated {datetime.now(ET):%Y-%m-%d %H:%M ET} · <a href="{PREFIX}/logout">sign out</a></div>
</div>{MODAL}</body></html>""", mimetype="text/html")


# ── TradingView Advanced Charts: UDF datafeed + chart page (additive, fail-safe) ──
try:
    from udf_datafeed import udf_bp

    @udf_bp.before_request
    def _udf_auth():
        if not _authed():
            return jsonify({"s": "error", "errmsg": "unauthorized"}), 401

    app.register_blueprint(udf_bp, url_prefix=PREFIX + "/udf")
except Exception as _e:
    import logging as _lg
    _lg.getLogger("dashboard").warning("UDF datafeed blueprint not registered: %s", _e)


@app.route(PREFIX + "/chart")
@need_auth
def chart_page():
    try:
        return Response((ROOT / "tv_chart.html").read_text(encoding="utf-8"), mimetype="text/html")
    except Exception as e:
        return Response(f"chart page error: {e}", mimetype="text/plain")


@app.route(PREFIX + "/static/<path:p>")
def dash_static(p):
    from flask import send_from_directory
    return send_from_directory(str(ROOT / "static"), p)


# ── routes ────────────────────────────────────────────────────────────────────
@app.route(PREFIX + "/")
@need_auth
def home():
    trips = get_trips()
    days = daily_from_trips(trips)
    acct = get_account_snapshot()
    ph = get_ph()
    today = datetime.now(ET).date().isoformat()
    ym = today[:7]
    td = days.get(today, {"pnl": 0, "n": 0, "wins": 0, "gw": 0, "gl": 0})
    m_pnl = sum(d["pnl"] for k, d in days.items() if k.startswith(ym))
    m_n = sum(d["n"] for k, d in days.items() if k.startswith(ym))
    m_w = sum(d["wins"] for k, d in days.items() if k.startswith(ym))
    m_gw = sum(d["gw"] for k, d in days.items() if k.startswith(ym))
    m_gl = sum(d["gl"] for k, d in days.items() if k.startswith(ym))
    series = equity_series(ph, current=acct["equity"])      # daily (SVG fallback)
    hybrid = equity_hybrid_series(ph, current=acct["equity"])  # daily + today's intraday
    chart = equity_lwc(hybrid, equity_svg(series))
    rng_lbl = ""
    if hybrid:
        vals = [v for _, v in hybrid]
        rng_lbl = (f"high {_money(max(vals))} · low {_money(min(vals))} · "
                   f"{datetime.fromtimestamp(hybrid[0][0], ET):%b %d} → now")
    cards = f"""
    <div class=cards>
      <div class=card><div class=k>Account equity</div><div class=v>{_money(acct['equity'])}</div><div class=sub>cash {_money(acct['cash'])}</div></div>
      <div class=card><div class=k>Today account P&L</div><div class=v>{_pnl(acct['day_pnl'])}</div><div class=sub>{_pct(acct['day_pnl_pct'])} vs prior close</div></div>
      <div class=card><div class=k>Today realized</div><div class=v>{_pnl(td['pnl'])}</div><div class=sub>{td['n']} trades · {(td['wins']/td['n']*100 if td['n'] else 0):.0f}% win</div></div>
    </div>
    <div class=card style="margin-top:10px">
      <div class=k>Account equity</div>
      {chart}
      <div class=sub>{rng_lbl}</div>
    </div>
    {positions_html(get_positions())}
    <h2>This month by day → <a href="{PREFIX}/monthly/{ym}">open {ym}</a></h2>
    """
    return page("Overview", cards, "home")


def positions_html(positions):
    age = snapshot_age_sec()
    if age is None:
        fresh = ('<div class="sub" style="color:#f85149;margin:-4px 0 8px">'
                 '⚠ no data yet — bot has not written a snapshot (is it running + authorized?)</div>')
    elif age > 420:
        fresh = (f'<div class="sub" style="color:#d29922;margin:-4px 0 8px">'
                 f'⚠ data {age // 60}m {age % 60}s old — bot may be offline or unauthorized</div>')
    else:
        fresh = f'<div class="sub muted" style="margin:-4px 0 8px">updated {age}s ago</div>'
    if not positions:
        return (f'<h2>Open positions</h2>{fresh}<div class="card"><span class="muted">Flat — no open positions.</span></div>')
    rows = ""
    tot_mv = tot_upl = 0.0
    for p in sorted(positions, key=lambda x: -abs(float(getattr(x, "market_value", 0) or 0))):
        mv = float(p.market_value or 0)
        upl = float(p.unrealized_pl or 0)
        uplpc = float(p.unrealized_plpc or 0)
        tot_mv += mv
        tot_upl += upl
        side = str(getattr(p, "side", "")).split(".")[-1].lower()
        tag = ' <span class="muted">opt</span>' if _occ(p.symbol) else (
            ' <span class="neg">short</span>' if side == "short" else "")
        rows += (f"<tr class=tap onclick=\"openTrade('{p.symbol}',{abs(float(p.qty)):g})\">"
                 f"<td>{_fmt_sym(p.symbol)}{tag}</td><td>{float(p.qty):g}</td>"
                 f"<td>${float(p.avg_entry_price):,.2f}</td><td>${float(p.current_price or 0):,.2f}</td>"
                 f"<td>{_money(mv)}</td><td>{_pnl(upl)} {_pct(uplpc)}</td></tr>")
    tot = (f"<tr class=tot><td>TOTAL</td><td>{len(positions)}</td><td></td><td></td>"
           f"<td>{_money(tot_mv)}</td><td>{_pnl(tot_upl)}</td></tr>")
    return (f"<h2>Open positions <span class=muted style=font-size:13px>· {len(positions)} · tap a row to trade</span></h2>{fresh}"
            f"<div class=scroll><table><thead><tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>Last</th>"
            f"<th>Value</th><th>Unreal P&L</th></tr></thead><tbody>{tot}{rows}</tbody></table></div>")


def month_nav(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    prev = f"{y-1}-12" if m == 1 else f"{y}-{m-1:02d}"
    nxt = f"{y+1}-01" if m == 12 else f"{y}-{m+1:02d}"
    return prev, nxt


@app.route(PREFIX + "/monthly")
@need_auth
def monthly_now():
    return redirect(f"{PREFIX}/monthly/{datetime.now(ET):%Y-%m}" + (f"?token={request.args.get('token')}" if request.args.get("token") else ""))


@app.route(PREFIX + "/monthly/<ym>")
@need_auth
def monthly(ym):
    trips = get_trips()
    days = daily_from_trips(trips)
    phd = ph_by_date(get_ph())
    keys = sorted([k for k in set(list(days) + list(phd)) if k.startswith(ym)])
    prev, nxt = month_nav(ym)
    # monthly totals
    t_pnl = sum(days[k]["pnl"] for k in keys if k in days)
    t_n = sum(days[k]["n"] for k in keys if k in days)
    t_w = sum(days[k]["wins"] for k in keys if k in days)
    t_gw = sum(days[k]["gw"] for k in keys if k in days)
    t_gl = sum(days[k]["gl"] for k in keys if k in days)
    t_vol = sum(days[k]["vol"] for k in keys if k in days)
    acct_pls = [phd[k]["pl"] for k in keys if k in phd and phd[k]["pl"] is not None]
    m_acct = sum(acct_pls)
    green = sum(1 for k in keys if k in days and days[k]["pnl"] > 0)
    red = sum(1 for k in keys if k in days and days[k]["pnl"] < 0)
    best = max((days[k]["pnl"] for k in keys if k in days), default=0.0)
    worst = min((days[k]["pnl"] for k in keys if k in days), default=0.0)
    eqs = [phd[k]["equity"] for k in keys if k in phd and phd[k]["equity"] is not None]
    cards = f"""
    <h1>{ym} <span class=muted style=font-size:13px>· <a href="{PREFIX}/monthly/{prev}">‹ {prev}</a> · <a href="{PREFIX}/monthly/{nxt}">{nxt} ›</a></span></h1>
    <div class=cards>
      <div class=card><div class=k>Account P&L (mo)</div><div class=v>{_pnl(m_acct)}</div></div>
      <div class=card><div class=k>Trades · Win%</div><div class=v>{t_n} · {(t_w/t_n*100 if t_n else 0):.0f}%</div><div class=sub>PF {_pf_str(_pf(t_gw,t_gl))} · vol {_money(t_vol)}</div></div>
      <div class=card><div class=k>Green / Red days</div><div class=v><span class=pos>{green}</span> / <span class=neg>{red}</span></div><div class=sub>best {_pnl(best)} · worst {_pnl(worst)}</div></div>
    </div>
    <h2>Each trading day</h2>
    """
    rows = ""
    for k in keys[::-1]:
        d = days.get(k, {"pnl": None, "n": 0, "wins": 0, "gw": 0, "gl": 0})
        a = phd.get(k, {})
        wr = f"{d['wins']/d['n']*100:.0f}%" if d["n"] else "—"
        pf = _pf_str(_pf(d["gw"], d["gl"])) if d["n"] else "—"
        dt = datetime.fromisoformat(k)
        rows += (f"<tr><td>{k[5:]}<span class=muted> {dt:%a}</span></td>"
                 f"<td>{_pnl(a.get('pl'))}</td><td>{_money(a.get('equity'))}</td>"
                 f"<td>{_pnl(d['pnl']) if d['n'] else '<span class=muted>—</span>'}</td>"
                 f"<td>{d['n'] or '—'}</td><td>{wr}</td><td>{pf}</td></tr>")
    tot = (f"<tr class=tot><td>TOTAL</td><td>{_pnl(m_acct)}</td><td>{_money(eqs[-1] if eqs else None)}</td>"
           f"<td>{_pnl(t_pnl)}</td><td>{t_n}</td><td>{(t_w/t_n*100 if t_n else 0):.0f}%</td><td>{_pf_str(_pf(t_gw,t_gl))}</td></tr>")
    table = (f"<div class=scroll><table><thead><tr><th>Day</th><th>Acct P&L</th><th>Equity</th>"
             f"<th>Realized</th><th>Trades</th><th>Win%</th><th>PF</th></tr></thead>"
             f"<tbody>{tot}{rows or '<tr><td colspan=7 class=muted>no data this month</td></tr>'}</tbody></table></div>")
    return page(f"Monthly {ym}", cards + table, "monthly")


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


@app.route(PREFIX + "/journal")
@need_auth
def journal():
    trips = sorted(get_trips(), key=lambda t: t["exit_time"], reverse=True)[:150]
    idx = reason_index()
    rows = ""
    for t in trips:
        sym = t["symbol"]
        er, xr = trip_logic(idx, t)
        rows += (f"<tr class=main><td>{t['exit_time'].astimezone(ET):%m-%d %H:%M}</td><td>{sym}</td>"
                 f"<td>{t['qty']:g}</td><td>${t['entry_px']:,.2f}</td><td>${t['exit_px']:,.2f}</td>"
                 f"<td>{_pnl(t['pnl'])}</td><td>{_pct(t['ret'])}</td><td>{t['hold_h']:.1f}h</td></tr>")
        rows += (f"<tr class=detail><td colspan=8>"
                 f"<span class=lbl>entry</span>{_esc(er)} "
                 f"<span class='lbl x'>exit</span>{_esc(xr)}</td></tr>")
    table = (f"<h1>Round-trips <span class=muted style=font-size:13px>· last {len(trips)} · with strategy logic</span></h1>"
             f"<div class=scroll><table><thead><tr><th>Closed</th><th>Sym</th><th>Qty</th><th>Entry</th>"
             f"<th>Exit</th><th>P&L</th><th>Ret</th><th>Held</th></tr></thead><tbody>{rows or '<tr><td colspan=8 class=muted>none</td></tr>'}</tbody></table></div>")
    return page("Journal", table, "journal")


@app.route(PREFIX + "/logs/tail")
@need_auth
def logs_tail():
    import subprocess
    n = min(max(int(request.args.get("n", 400)), 1), 3000)
    src = request.args.get("src", "run")
    try:
        if src == "webhook":
            out = subprocess.run(["journalctl", "-u", "trade-webhook.service", "--no-pager", "-n", str(n)],
                                 capture_output=True, text=True, timeout=12).stdout
        else:
            out = subprocess.run(["tail", "-n", str(n), str(LOG_FILE)],
                                 capture_output=True, text=True, timeout=12).stdout
    except Exception as e:
        out = f"(log read error: {e})"
    return Response(out or "(empty)", mimetype="text/plain")


@app.route(PREFIX + "/logs")
@need_auth
def logs_page():
    body = f"""<h1>Live logs</h1>
    <div class=nav style=margin-bottom:8px>
      <a href="#" id=src-run class="navlink on" onclick="SRC='run';mark();refresh();return false">run.py</a>
      <a href="#" id=src-webhook class=navlink onclick="SRC='webhook';mark();refresh();return false">webhook</a>
      <a href="#" id=pausebtn class=navlink onclick="togglePause(this);return false">⏸ pause</a>
    </div>
    <pre id=log class=logbox>loading…</pre>
    <script>
    var SRC='run', PAUSE=false;
    function mark(){{document.getElementById('src-run').className=(SRC=='run'?'navlink on':'navlink');
      document.getElementById('src-webhook').className=(SRC=='webhook'?'navlink on':'navlink');}}
    function refresh(){{ if(PAUSE) return;
      fetch('{PREFIX}/logs/tail?src='+SRC+'&n=500',{{credentials:'same-origin'}})
       .then(function(r){{return r.text()}}).then(function(t){{
         var el=document.getElementById('log');
         var atBottom = el.scrollTop+el.clientHeight >= el.scrollHeight-60;
         el.textContent=t; if(atBottom) el.scrollTop=el.scrollHeight; }})
       .catch(function(){{}}); }}
    function togglePause(b){{
      if(PAUSE){{PAUSE=false;b.textContent='⏸ pause';return;}}
      var pw=prompt('Password to pause the live logs:');if(!pw)return;
      var fd=new FormData();fd.append('password',pw);
      fetch('{PREFIX}/verify',{{method:'POST',body:fd,credentials:'same-origin'}})
       .then(function(r){{return r.json();}}).then(function(j){{
        if(j.ok){{PAUSE=true;b.textContent='▶ resume';}}
        else{{b.textContent='⏸ wrong pw';setTimeout(function(){{b.textContent='⏸ pause';}},1800);}}
       }}).catch(function(){{}});
    }}
    refresh(); setInterval(refresh, 2500);
    </script>"""
    return page("Logs", body, "logs")


@app.route(PREFIX + "/trade", methods=["POST"])
@need_auth
def trade():
    symbol = (request.form.get("symbol") or "").strip().upper()
    side = (request.form.get("side") or "").strip().lower()
    qty_s = (request.form.get("qty") or "").strip()
    if side not in ("buy", "sell") or not symbol:
        return jsonify({"ok": False, "error": "bad request"}), 400
    # Re-authenticate: every live order requires the password, even within a logged-in session.
    if not DASH_PASSWORD or (request.form.get("password") or "") != DASH_PASSWORD:
        return jsonify({"ok": False, "error": "wrong password"}), 403
    try:
        qty = float(qty_s)
        assert qty > 0
    except Exception:
        return jsonify({"ok": False, "error": "invalid qty"}), 400
    try:
        from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
        tc = _client()
        # Free the symbol first: cancel ALL its open orders so the manual order isn't blocked by a
        # wash trade (an opposing resting order) or insufficient qty (shares tied to a resting sell).
        cancelled = 0
        try:
            for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])):
                tc.cancel_order_by_id(o.id)
                cancelled += 1
            if cancelled:
                time.sleep(0.6)
        except Exception:
            pass
        q = qty if qty != int(qty) else int(qty)          # fractional for equities, whole for options
        req = MarketOrderRequest(symbol=symbol, qty=q,
                                 side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY)
        o = tc.submit_order(req)
        _CACHE.pop("pos", None); _CACHE.pop("acct", None)   # refresh on next page load
        return jsonify({"ok": True, "id": str(o.id), "symbol": symbol, "side": side,
                        "qty": qty_s, "status": str(o.status).split(".")[-1], "cancelled": cancelled})
    except Exception as e:
        msg = str(e)
        try:
            d = json.loads(str(e))
            msg = d.get("message") or d.get("reject_reason") or msg
        except Exception:
            pass
        return jsonify({"ok": False, "error": msg[:220]}), 400


@app.route(PREFIX + "/backtest")
@need_auth
def backtest():
    iframe = (f'<iframe src="{PREFIX}/backtest/raw" loading="eager" '
              f'style="width:100%;height:82vh;border:1px solid #21262d;border-radius:10px;'
              f'background:#0d1117"></iframe>')
    return page("Strategy", iframe, "backtest")


@app.route(PREFIX + "/backtest/raw")
@need_auth
def backtest_raw():
    f = ROOT / "crossover_tearsheet.html"
    if not f.exists():
        return Response("No tearsheet generated yet.", mimetype="text/plain")
    return Response(f.read_text(), mimetype="text/html")


@app.route(PREFIX + "/verify", methods=["POST"])
@need_auth
def verify():
    ok = bool(DASH_PASSWORD) and (request.form.get("password") or "") == DASH_PASSWORD
    return jsonify({"ok": ok})


@app.route(PREFIX + "/health")
def health():
    return "ok"


# ── Open-position mini-charts: Databento intraday candles (additive, fail-safe) ────────────────
def _db_api_key():
    k = os.environ.get("DATABENTO_API") or os.environ.get("DATABENTO_API_KEY")
    if k:
        return k.strip()
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(("DATABENTO_API=", "DATABENTO_API_KEY=")):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _occ_to_databento(sym):
    """Compact OCC (WGS260710C00056000) -> Databento padded raw_symbol (WGS   260710C00056000); None if equity."""
    s = (sym or "").strip().upper()
    if len(s) < 15:
        return None
    tail = s[-15:]
    if not (tail[6] in ("C", "P") and tail[:6].isdigit() and tail[7:].isdigit()):
        return None
    return f"{s[:-15]:<6}{tail}"


def _db_candles(symbol, ttl=300):
    """Recent 1-min OHLC bars [{time,open,high,low,close}] from Databento (equity DBEQ.BASIC, or
    OPRA for an option leg). Cached per symbol. Fail-safe → [] on any error/missing dep."""
    ck = f"cd_{symbol}"
    now = time.time()
    if _CACHE.get(ck) is not None and now - _CACHE.get(ck + "_t", 0) < ttl:
        return _CACHE[ck]
    out = []
    try:
        import databento as db
        c = db.Historical(_db_api_key())
        _now = datetime.now(timezone.utc)
        start = (_now - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M")   # last ~2-3 sessions
        end = _now.strftime("%Y-%m-%dT%H:%M")
        occ = _occ_to_databento(symbol)
        if occ:
            df = c.timeseries.get_range(dataset="OPRA.PILLAR", schema="ohlcv-1m", symbols=occ,
                                        stype_in="raw_symbol", start=start, end=end).to_df()
        else:
            df = c.timeseries.get_range(dataset="DBEQ.BASIC", schema="ohlcv-1m", symbols=symbol,
                                        stype_in="raw_symbol", start=start, end=end).to_df()
        for ts, r in df.iterrows():
            o = float(r["open"])
            if o > 0:
                out.append({"time": int(ts.timestamp()), "open": o, "high": float(r["high"]),
                            "low": float(r["low"]), "close": float(r["close"])})
    except Exception as e:
        import logging
        logging.getLogger("dashboard").warning("db_candles %s: %s", symbol, e)
    _CACHE[ck], _CACHE[ck + "_t"] = out, now
    return out


@app.route(PREFIX + "/candles")
@need_auth
def candles():
    sym = (request.args.get("symbol") or "").strip().upper()
    if not sym:
        return jsonify({"s": "error", "candles": []})
    return jsonify({"s": "ok", "symbol": sym, "candles": _db_candles(sym)})


@app.route(PREFIX + "/positions")
@need_auth
def positions_charts():
    pos = get_positions()
    fresh = positions_html(pos).split("</h2>")[0]  # reuse the freshness banner snippet
    if not pos:
        body = '<h1>Open positions</h1><div class="card"><span class="muted">Flat — no open positions.</span></div>'
        return page("Positions", body, "positions")
    syms, cards = [], []
    for p in pos:
        sym = str(getattr(p, "symbol", "") or "")
        if not sym:
            continue
        syms.append(sym)
        qty = getattr(p, "qty", "")
        mv = float(getattr(p, "market_value", 0) or 0)
        upl = getattr(p, "unrealized_pl", None)
        upl_s = f" · {_pnl(float(upl))}" if upl not in (None, "") else ""
        cards.append(
            f'<div class="card" style="padding:10px">'
            f'<div class=k>{sym}</div>'
            f'<div class=sub>qty {qty} · mv {_money(mv)}{upl_s}</div>'
            f'<div id="ch_{sym}" style="height:200px;margin-top:6px"></div></div>')
    grid = ('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px">'
            + "".join(cards) + "</div>")
    js = (
        '<script src="https://unpkg.com/lightweight-charts@5/dist/lightweight-charts.standalone.production.js"></script>'
        '<script>(function(){'
        f'var P="{PREFIX}";var syms={json.dumps(syms)};'
        'var LWC=window.LightweightCharts;'
        'if(!(LWC&&LWC.createChart&&LWC.CandlestickSeries))return;'
        'syms.forEach(function(sym){'
        ' var el=document.getElementById("ch_"+sym);if(!el)return;'
        ' var chart=LWC.createChart(el,{height:200,width:el.clientWidth,'
        '   layout:{background:{type:"solid",color:"transparent"},textColor:"#8b949e",attributionLogo:false},'
        '   grid:{vertLines:{visible:false},horzLines:{color:"rgba(255,255,255,0.05)"}},'
        '   rightPriceScale:{borderVisible:false},timeScale:{borderVisible:false,timeVisible:true,secondsVisible:false},'
        '   handleScroll:false,handleScale:false});'
        ' var s=chart.addSeries(LWC.CandlestickSeries,{upColor:"#3fb950",downColor:"#f85149",'
        '   borderVisible:false,wickUpColor:"#3fb950",wickDownColor:"#f85149"});'
        ' fetch(P+"/candles?symbol="+encodeURIComponent(sym),{credentials:"same-origin"})'
        '  .then(function(r){return r.json();})'
        '  .then(function(d){if(d.candles&&d.candles.length){s.setData(d.candles);chart.timeScale().fitContent();}'
        '    else{el.innerHTML="<span class=\\"muted\\" style=\\"font-size:12px\\">no intraday data</span>";}})'
        '  .catch(function(){el.innerHTML="<span class=\\"muted\\" style=\\"font-size:12px\\">chart error</span>";});'
        ' new ResizeObserver(function(){chart.resize(el.clientWidth,200);}).observe(el);'
        '});})();</script>')
    body = f"<h1>Open positions <span class=muted style=font-size:13px>· {len(syms)} · intraday 1-min</span></h1>{fresh}</h2>{grid}{js}"
    return page("Positions", body, "positions")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("DASH_PORT", "5001")))
