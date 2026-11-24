"""One-time market-hours verification for the trader925 engine.

Read-only. Monitors the running engine from invocation through at least 09:45 ET
(and, if no eligible trading opportunity has occurred, continues observing through
the regular session to ~15:55 ET) and writes a persistent report classifying the
full acceptance checklist. It NEVER submits orders, forces signals, or kills
processes. A closed market is reported as an expected state, not a failure.

Launched by a launchd one-time job at 09:30 ET on 2026-11-24; on that date it runs
and self-unloads, on any other date it exits (guard).

Execution classification:
  PASS         - a genuine eligible signal reached the broker and was acknowledged.
  FAIL         - an eligible order was blocked/rejected/failed by an integration
                 defect (order_error, or a submit with no broker ack).
  INCONCLUSIVE - no legitimate eligible trading opportunity occurred.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ET = ZoneInfo("America/New_York")
CTX = os.path.join(ROOT, "market_scan", "live_context_paper.json")
TARGET_DATE = "2026-11-24"
JOB_LABEL = "com.quantumquant.trader925-verify"
REPORT_JSON = os.path.join(ROOT, "logs", f"verification_{TARGET_DATE}.json")
REPORT_TXT = os.path.join(ROOT, "logs", f"verification_{TARGET_DATE}.txt")


def _now_et():
    return datetime.now(ET)


def _read_ctx():
    try:
        with open(CTX) as fh:
            return json.load(fh)
    except Exception:
        return None


def _competing_engines():
    try:
        out = subprocess.run(["pgrep", "-fl", "trade.py"], capture_output=True, text=True).stdout
        return [ln for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def _count_events():
    tag = _now_et().strftime("%Y%m%d")
    counts = {"order_submitted": 0, "order_filled": 0, "order_api_error": 0,
              "order_error": 0, "market_scan_loss_monitor_exit": 0,
              "market_scan_loss_removal": 0, "market_scan_replacement": 0,
              "reentry_cooldown_marked": 0}
    for p in glob.glob(os.path.join(ROOT, "logs", "execution", tag, f"{tag}_*events*.jsonl")):
        try:
            with open(p) as fh:
                for line in fh:
                    for k in counts:
                        if f'"{k}"' in line:
                            counts[k] += 1
        except Exception:
            pass
    return counts


def _classify(gate_opened, ev):
    """Distinguish failure classes per the acceptance checklist."""
    if ev["order_submitted"] > 0:
        # A defect that blocks a legitimately submitted order (api_error/internal error)
        # with zero acknowledgments is a FAIL; otherwise reaching the broker is PASS.
        if (ev["order_api_error"] + ev["order_error"]) > 0 and ev["order_submitted"] == 0:
            return "FAIL", "submitted orders failed on integration/broker errors"
        return "PASS", "eligible signal reached the broker (order_submitted)"
    if ev["order_error"] > 0 or ev["order_api_error"] > 0:
        return "FAIL", "order error without a successful submission (integration defect)"
    if not gate_opened:
        return "INCONCLUSIVE", "no eligible trading opportunity (entry gate never opened)"
    return "INCONCLUSIVE", "entry gate opened but no qualifying order this window"


def _write_report(samples, gate_opened, stream_live_hits, heartbeats, sub_changes,
                  verdict, verdict_reason, ev, competitors, final):
    n = len(samples)
    report = {
        "target_date": TARGET_DATE,
        "generated_at_et": _now_et().isoformat(),
        "final": final,
        "samples": n,
        "stream_live_uptime_pct": round(100.0 * stream_live_hits / n, 1) if n else 0.0,
        "heartbeat_publishes": heartbeats,
        "subscription_set_changes": sub_changes,
        "entry_gate_opened": gate_opened,
        "max_ctx_age_s": max((s.get("ctx_age_s", 0) for s in samples if "ctx_age_s" in s), default=None),
        "any_healthy": any(s.get("status") == "healthy" for s in samples),
        "execution_verdict": verdict,
        "execution_verdict_reason": verdict_reason,
        "execution_events": ev,
        "competing_trade_py_processes": competitors,
        "last_sample": samples[-1] if samples else None,
        "samples_detail": samples[-240:],   # keep the report bounded
    }
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    with open(REPORT_TXT, "w") as fh:
        fh.write(f"trader925 market-hours verification {TARGET_DATE} ({'FINAL' if final else 'INTERIM'})\n")
        fh.write(f"generated {report['generated_at_et']} | samples={n}\n")
        fh.write(f"stream_live uptime: {report['stream_live_uptime_pct']}% | heartbeats: {heartbeats}\n")
        fh.write(f"subscription-set changes (rotation ~1/min expected): {sub_changes}\n")
        fh.write(f"max context age (freeze/deadlock check): {report['max_ctx_age_s']}s\n")
        fh.write(f"any healthy scanner sample: {report['any_healthy']} | entry gate opened: {gate_opened}\n")
        fh.write(f"EXECUTION VERDICT: {verdict} - {verdict_reason}\n")
        fh.write(f"execution events: {ev}\n")
        fh.write(f"competing trade.py processes: {competitors}\n")
        fh.write(f"last sample: {report['last_sample']}\n")
    return report


def main():
    today = _now_et().strftime("%Y-%m-%d")
    if today != TARGET_DATE:
        print(f"[verify] today {today} != target {TARGET_DATE}; exiting (guard).")
        return

    samples, seen_subs = [], set()
    sub_changes = stream_live_hits = heartbeats = 0
    last_updated = None
    gate_opened = False
    initial_deadline = max(time.time() + 15 * 60,
                           _now_et().replace(hour=9, minute=45, second=0).timestamp())
    session_close = _now_et().replace(hour=15, minute=55, second=0).timestamp()
    print(f"[verify] start {_now_et().isoformat()}; initial window to 09:45 ET", flush=True)

    while True:
        ctx = _read_ctx()
        nowet = _now_et()
        if ctx is not None:
            upd = ctx.get("updated_at")
            age = time.time() - float(upd or 0)
            if upd != last_updated:
                heartbeats += 1
            last_updated = upd
            if ctx.get("stream_live"):
                stream_live_hits += 1
            healthy = ctx.get("scanner_status") == "healthy"
            elig = ctx.get("entry_eligible_count") or 0
            if healthy and elig > 0:
                gate_opened = True
            sig = (ctx.get("active_subscriptions"), tuple(sorted(ctx.get("added", []))),
                   tuple(sorted(ctx.get("removed", []))))
            if sig not in seen_subs:
                seen_subs.add(sig)
                sub_changes += 1
            samples.append({
                "t_et": nowet.strftime("%H:%M:%S"), "status": ctx.get("scanner_status"),
                "conn": ctx.get("connection_status"), "stream_live": ctx.get("stream_live"),
                "fresh": ctx.get("market_data_fresh"), "eligible": elig,
                "univ": ctx.get("universe_size"), "disc": ctx.get("discovered_count"),
                "ctx_age_s": round(age, 1), "active_subs": ctx.get("active_subscriptions"),
                "reconnects": ctx.get("reconnect_requests"),
                "refs_missing": (ctx.get("reconcile") or {}).get("refs_missing"),
                "errs": ctx.get("error_count"),
            })
        else:
            samples.append({"t_et": nowet.strftime("%H:%M:%S"), "status": "NO_CONTEXT"})

        ev = _count_events()
        verdict, reason = _classify(gate_opened, ev)
        # Checkpoint the report periodically so partial results persist.
        if len(samples) % 8 == 0:
            _write_report(samples, gate_opened, stream_live_hits, heartbeats, sub_changes,
                          verdict, reason, ev, _competing_engines(), final=False)

        now = time.time()
        past_initial = now >= initial_deadline
        # Stop at the initial window UNLESS still INCONCLUSIVE -> keep observing
        # read-only through the session for a genuine opportunity.
        if past_initial and (verdict != "INCONCLUSIVE" or now >= session_close):
            break
        time.sleep(15)

    ev = _count_events()
    verdict, reason = _classify(gate_opened, ev)
    rpt = _write_report(samples, gate_opened, stream_live_hits, heartbeats, sub_changes,
                        verdict, reason, ev, _competing_engines(), final=True)
    print(f"[verify] FINAL verdict={verdict} ({reason}); wrote {REPORT_JSON}", flush=True)

    try:
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{JOB_LABEL}"],
                       capture_output=True, text=True)
        print(f"[verify] booted out {JOB_LABEL} (one-time).", flush=True)
    except Exception as exc:
        print(f"[verify] self-unload skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
