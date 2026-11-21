#!/usr/bin/env python3
"""Continuous background health monitor for the supervised trade.py service.

Appends a compact JSON health snapshot to logs/monitor_health.log every INTERVAL
seconds and emits a one-line ALERT whenever something notable changes:
  * service down / launchd pid missing
  * crash-loop (launchd pid changed since last tick)
  * scanner live-context stale while the market is open
  * new protective events (loss_removal / kill_switch / order_results / replacement)

Read-only: it inspects on-disk artifacts + `launchctl list`; it makes NO broker
calls and never touches the engine. Safe to run alongside the live service.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

INTERVAL = 180  # seconds between snapshots (market closed -> low cadence is fine)
LABEL = "com.quantumquant.trader925"

ROOT = Path(__file__).resolve().parent
HEALTH_LOG = ROOT / "logs" / "monitor_health.log"
HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)


def _launchd_pid(label: str) -> int | None:
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if label in line:
                tok = line.split()[0]
                return int(tok) if tok.isdigit() else None
    except Exception:
        return None
    return None


def _newest_log_dir() -> Path | None:
    dirs = sorted((d for d in (ROOT / "logs" / "execution").glob("*")
                   if d.is_dir() and re.fullmatch(r"\d{8}", d.name)), reverse=True)
    return dirs[0] if dirs else None


def _event_counts(log_dir: Path) -> tuple[int, dict]:
    ev = list(log_dir.glob("*_execution_events.jsonl"))
    if not ev or not ev[0].exists():
        return 0, {}
    rows = ev[0].read_text(errors="replace").splitlines()
    flagged: dict[str, int] = {}
    for line in rows:
        try:
            e = json.loads(line)["event"]
        except Exception:
            continue
        for key in ("loss_removal", "kill_switch", "order_results", "replacement", "halt"):
            if key in e:
                flagged[key] = flagged.get(key, 0) + 1
    return len(rows), flagged


def _scanner_age() -> float | None:
    lc = ROOT / "market_scan" / "live_context_paper.json"
    return (time.time() - lc.stat().st_mtime) if lc.exists() else None


def _market_open() -> bool | None:
    # best-effort: read the last market_clock_checked event payload
    ld = _newest_log_dir()
    if not ld:
        return None
    ev = list(ld.glob("*_execution_events.jsonl"))
    if not ev or not ev[0].exists():
        return None
    for line in reversed(ev[0].read_text(errors="replace").splitlines()):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "market_clock_checked":
            return bool(r.get("payload", {}).get("is_open"))
    return None


def _emit(obj: dict) -> None:
    with HEALTH_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
    # also to stdout (captured by the launcher) for live tailing
    tag = "ALERT" if obj.get("alerts") else "ok"
    print(f"[{obj['ts']}] {tag} pid={obj.get('pid')} scanner_age={obj.get('scanner_age_s')} "
          f"events={obj.get('events')} {obj.get('alerts') or ''}", flush=True)


def main() -> None:
    last_pid = None
    last_events = None
    last_flagged: dict[str, int] = {}
    while True:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        pid = _launchd_pid(LABEL)
        ld = _newest_log_dir()
        total, flagged = _event_counts(ld) if ld else (0, {})
        age = _scanner_age()
        mopen = _market_open()
        alerts = []
        if pid is None:
            alerts.append("SERVICE_DOWN")
        elif last_pid is not None and pid != last_pid:
            alerts.append(f"RESTARTED({last_pid}->{pid})")
        if mopen and age is not None and age > 120:
            alerts.append(f"SCANNER_STALE_RTH({age:.0f}s)")
        for k, v in flagged.items():
            if v > last_flagged.get(k, 0):
                alerts.append(f"NEW_{k.upper()}(+{v - last_flagged.get(k, 0)})")
        _emit({"ts": ts, "pid": pid, "scanner_age_s": None if age is None else round(age),
               "market_open": mopen, "events": total, "flagged": flagged, "alerts": alerts})
        last_pid = pid if pid is not None else last_pid
        last_events = total
        last_flagged = flagged
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
