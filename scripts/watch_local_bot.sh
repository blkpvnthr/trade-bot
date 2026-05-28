#!/usr/bin/env bash
# macOS monitoring loop (no `watch` available). Ctrl+C to exit.
# Usage: bash scripts/watch_local_bot.sh [interval_seconds]
INTERVAL="${1:-5}"
TODAY=$(date +%Y%m%d)
cd "$(dirname "$0")/.."

while true; do
  clear
  echo "=== Local Bot Monitor — $(date '+%Y-%m-%d %H:%M:%S ET') (refresh every ${INTERVAL}s, Ctrl+C to exit) ==="
  echo ""
  bash scripts/check_local_multi_instance.sh
  echo ""
  echo "=== Recent Events ==="
  for STRATEGY_ID in tux_mtf_original pivot_points liquidity_sweep; do
    EVENTS="logs/execution/${STRATEGY_ID}/${TODAY}_execution_events.jsonl"
    if [[ -f "$EVENTS" ]]; then
      tail -3 "$EVENTS" | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        print(f'  [$STRATEGY_ID] {d.get(\"event\",\"?\")} | {d.get(\"timestamp_utc\",\"\")[:19]}')
    except Exception:
        pass
" 2>/dev/null
    fi
  done
  echo ""
  echo "=== Alpaca Positions ==="
  python3 - <<'PYEOF' 2>/dev/null || echo "  (credentials not available or alpaca-py not installed)"
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import os

# Load credentials: prefer .env, fall back to first populated .env.bot0N
def _load_creds():
    for env_file in [".env", ".env.bot01", ".env.bot02", ".env.bot03"]:
        p = Path(env_file)
        if not p.exists():
            continue
        vals = dotenv_values(p)
        key = vals.get("APCA_API_KEY_ID", "")
        secret = vals.get("APCA_API_SECRET_KEY", "")
        if key and not key.startswith("REPLACE_ME") and secret and not secret.startswith("REPLACE_ME"):
            load_dotenv(p, override=True)
            return True
    return False

if not _load_creds():
    print("  (no .env file with valid credentials found)")
    raise SystemExit(0)

from alpaca.trading.client import TradingClient
try:
    c = TradingClient(os.getenv("APCA_API_KEY_ID", ""), os.getenv("APCA_API_SECRET_KEY", ""), paper=True)
    pos = c.get_all_positions()
    if pos:
        for p in pos:
            print(f"  {p.symbol}: qty={p.qty} avg={p.avg_entry_price} P&L={float(p.unrealized_pl):.2f}")
    else:
        print("  (no open positions)")
except Exception as exc:
    print(f"  (positions unavailable: {exc})")
PYEOF
  sleep "$INTERVAL"
done
