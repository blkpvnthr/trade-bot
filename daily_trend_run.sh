#!/bin/bash
# Daily driver for the shares-only trend engine + execution audit (PAPER).
# Usage:  daily_trend_run.sh [dry|submit]   (default: dry)
#   dry    -> engine preview only, NO orders (use for the first few days)
#   submit -> places PAPER orders, then runs the execution harness
# Schedule weekdays ~10 min before the close, e.g. cron:
#   50 15 * * 1-5  /Users/blkpvnthr/Desktop/trader/daily_trend_run.sh submit
set -euo pipefail
cd /Users/blkpvnthr/Desktop/trader
PY=/Users/blkpvnthr/Desktop/trader/.venv/bin/python
[ -x "$PY" ] || PY=python
MODE="${1:-dry}"
mkdir -p logs/execution
LOG="logs/execution/daily_$(date +%Y%m%d).log"
{
  echo "================ $(date) mode=$MODE ================"
  "$PY" premarket_levels.py --screener || true   # screener-driven watchlist -> structure levels + gate JSON
  "$PY" volarb_signal.py || true                  # vol-arb IV/RV>1.4 iron-condor signal (monitor-only)
  if [ "$MODE" = "submit" ]; then
    "$PY" live_trend_engine.py --submit
    "$PY" execution_harness.py
  else
    "$PY" live_trend_engine.py
    "$PY" execution_harness.py || true   # harness shows 'not executed' state in dry mode
  fi
  echo "================ done $(date) ================"
} >> "$LOG" 2>&1
echo "wrote $LOG"
