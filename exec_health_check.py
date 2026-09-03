#!/usr/bin/env python3
"""
exec_health_check.py — always-on droplet self-check: is the run.py paper bot actually
executing trades? Read-only (no secrets printed). Writes a human-readable report to
logs-instance2/exec_health_latest.txt (+ a dated copy). Designed to be run by cron on
the droplet weekday mornings so trade execution is confirmed independent of any laptop.
"""
from __future__ import annotations
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVF = ROOT / ".env.nonl0cal2"
OUT = ROOT / "logs-instance2"


def main():
    from dotenv import dotenv_values
    v = dotenv_values(str(ENVF))
    key = v.get("APCA_API_KEY_ID") or ""
    sec = v.get("APCA_API_SECRET_KEY") or ""
    now = datetime.now(timezone.utc)
    L = ["════ EXEC HEALTH CHECK ════  %s UTC" % now.strftime("%Y-%m-%d %H:%M")]

    # ---- service ----
    try:
        act = subprocess.run(["systemctl", "is-active", "trade-bot-instance2.service"],
                             capture_output=True, text=True).stdout.strip()
        pids = subprocess.run(["pgrep", "-f", "[r]un.py"], capture_output=True, text=True).stdout.split()
        L.append("SERVICE  active=%s  run.py_pids=%s" % (act, ",".join(pids) or "NONE"))
    except Exception as e:
        L.append("SERVICE check err: %r" % e)

    # ---- alpaca (read-only) ----
    traded_today = 0
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        c = TradingClient(key, sec, paper=True)
        a = c.get_account()
        dpnl = float(a.equity) - float(a.last_equity)
        L.append("ACCOUNT  equity=$%s cash=$%s bp=$%s day_pnl=$%.2f status=%s blocked(trade=%s acct=%s)"
                 % (a.equity, a.cash, a.buying_power, dpnl, a.status, a.trading_blocked, a.account_blocked))
        pos = c.get_all_positions()
        exp = sum(abs(float(p.market_value)) for p in pos)
        L.append("POSITIONS  %d open  gross_exposure=$%.0f" % (len(pos), exp))
        for p in sorted(pos, key=lambda x: -abs(float(x.market_value)))[:12]:
            plpc = round(float(p.unrealized_plpc) * 100, 1) if p.unrealized_plpc else 0.0
            L.append("   %-6s qty=%s mv=$%.0f uPL=$%.0f (%s%%)"
                     % (p.symbol, p.qty, float(p.market_value), float(p.unrealized_pl), plpc))
        o = c.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL,
                                                 after=now - timedelta(hours=24), limit=300))
        traded_today = len(o)
        bystat = Counter(x.status.value for x in o)
        filled = sum(1 for x in o if x.status.value == "filled")
        L.append("ORDERS (24h)  total=%d  filled=%d  %s" % (len(o), filled, dict(bystat)))
        coid = Counter((str(x.client_order_id).split("-")[0] if x.client_order_id else "?") for x in o)
        L.append("   by source (coid prefix): %s" % dict(coid))
        for x in sorted(o, key=lambda z: str(z.submitted_at), reverse=True)[:12]:
            L.append("   %s %-4s %-6s qty=%s %-9s @ %s"
                     % (str(x.submitted_at)[:16], x.side.value, x.symbol, x.qty, x.status.value, x.filled_avg_price))
    except Exception as e:
        L.append("ALPACA ERR: %s" % repr(str(e))[:200])

    # ---- recent log lines (exec/blocker markers, secret-filtered) ----
    try:
        logf = OUT / "instance2.out.log"
        if logf.exists():
            tail = subprocess.run(["tail", "-400", str(logf)], capture_output=True, text=True).stdout.splitlines()
            markers = ("BUY ", "SELL ", "tjl_entry", "TJL ENTER", "tjl_trail", "TRAIL ARMED",
                       "order placed", "submitted", "unauthorized", "kill", "blocked",
                       "DAILY_LOSS", "regime", "ENTER ")
            hits = [ln for ln in tail if any(w in ln for w in markers)]
            hits = [ln for ln in hits if not any(b in ln.lower() for b in ("token", "secret", "apca", "password", "key="))]
            L.append("LOG (recent exec/blocker lines: %d)" % len(hits))
            for ln in hits[-10:]:
                L.append("   " + ln[:170])
    except Exception as e:
        L.append("LOG scan err: %r" % e)

    # ---- verdict ----
    L.append("VERDICT  %s" % ("orders present in last 24h — executing" if traded_today
                              else "NO orders in last 24h (weekend/quiet, or signal-starved — see logs)"))
    txt = "\n".join(L)
    try:
        OUT.mkdir(exist_ok=True)
        (OUT / "exec_health_latest.txt").write_text(txt)
        (OUT / ("exec_health_%s.txt" % now.strftime("%Y%m%d"))).write_text(txt)
    except Exception as e:
        txt += "\n(write err: %r)" % e
    print(txt)


if __name__ == "__main__":
    main()
