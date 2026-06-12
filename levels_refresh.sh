#!/bin/bash
# Intraday levels refresh (~10:05 ET) — recompute structure levels + pre-trade gate with
# TODAY's opening range / VWAP / POC (vs the 7:00 premarket run's prior-session reference).
# Does NOT trade; only refreshes logs/structure/levels_latest.json.
set -euo pipefail
cd /Users/blkpvnthr/Desktop/trader
PY=/Users/blkpvnthr/Desktop/trader/.venv/bin/python
[ -x "$PY" ] || PY=python
mkdir -p logs/structure
LOG="logs/structure/refresh_$(date +%Y%m%d).log"
{
  echo "==== intraday levels refresh $(date) ===="
  "$PY" premarket_levels.py --screener
  "$PY" volarb_signal.py
  "$PY" execution_harness.py   # post-open fill audit (engine submits ~9:30 open; 10:05 captures fills)
  echo "==== done $(date) ===="
} >> "$LOG" 2>&1
echo "wrote $LOG"
