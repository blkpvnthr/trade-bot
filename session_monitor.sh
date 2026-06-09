#!/bin/zsh
# Run by launchd at 10:00 ET weekdays — captures the live bot's session under the 9-EMA-dip-buy /
# 200-SMA-exit overlay into logs/session_report_YYYYMMDD.txt (read it, or ask Claude to summarize).
cd /Users/blkpvnthr/Desktop/trader || exit 1
mkdir -p logs
D=$(date +%Y%m%d)
REPORT="logs/session_report_${D}.txt"
{
echo "=== Session monitor — $(date '+%Y-%m-%d %H:%M %Z') (9EMA-dip-buy / 200SMA-exit overlay) ==="
ssh -o BatchMode=yes -o ConnectTimeout=20 root@107.170.29.202 'bash -s' <<'REMOTE'
cd /root/trade-bot
T=$(date +%Y-%m-%d)
echo "[service] $(systemctl is-active trade-bot.service) @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "[account] $(grep 'Account cycle=' logs/run.log | tail -1)"
echo "[9EMA DIP BUYS today]"; grep -E 'ORDER buy|notional=' logs/run.log | grep "$T" | tail -12 || echo "  (none)"
echo "[200SMA EXITS today]"; grep -E 'SMA200 EXIT|sma200_exit' logs/run.log | grep "$T" | tail -12 || echo "  (none)"
echo "[trend-gate rejects today] not_9ema_dip=$(grep "$T" logs/run.log | grep -c rejected_not_9ema_dip) below_sma200=$(grep "$T" logs/run.log | grep -c rejected_by_below_sma50)"
echo "[errors today] $(grep "$T" logs/run.log | grep -cE 'ERROR|Traceback')"
venv/bin/python - 2>&1 <<'PY' | tail -30
import os
from datetime import datetime, timezone
for l in open(".env"):
    s=l.strip()
    if "=" in s and not s.startswith("#"):
        k,v=s.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
tc=TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=False)
a=tc.get_account()
print(f"[Alpaca] equity=${float(a.equity):.0f} last_equity=${float(a.last_equity):.0f} day_pnl=${float(a.equity)-float(a.last_equity):+.0f} cash=${float(a.cash):.0f}")
since=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
print("[orders today]")
for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=60, after=since, direction="desc")):
    print(f"  {str(o.submitted_at)[11:19]} {o.symbol} {str(o.side).split('.')[-1]} qty={o.qty} notional={o.notional} {str(o.status).split('.')[-1]} fill={o.filled_avg_price}")
print("[positions]")
for p in tc.get_all_positions():
    print(f"  {p.symbol} qty={p.qty} mv=${float(p.market_value):.0f} unrl={float(p.unrealized_pl):+.0f} ({float(p.unrealized_plpc)*100:+.1f}%)")
PY
REMOTE
} > "$REPORT" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') wrote $REPORT" >> logs/session_monitor.log
