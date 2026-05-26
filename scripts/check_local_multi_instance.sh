#!/usr/bin/env bash
cd "$(dirname "$0")/.."
TODAY=$(date +%Y%m%d)

for STRATEGY_ID in tux_mtf_original pivot_points liquidity_sweep; do
  PID_FILE=".local_pids/${STRATEGY_ID}.pid"
  EVENTS="logs/execution/${STRATEGY_ID}/${TODAY}_execution_events.jsonl"
  METRICS="logs/execution/${STRATEGY_ID}/metrics.json"

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    STATUS="RUNNING (PID $(cat "$PID_FILE"))"
  else
    STATUS="STOPPED"
  fi

  LAST_EVENT=""
  if [[ -f "$EVENTS" ]]; then
    LAST_EVENT=$(tail -1 "$EVENTS" | python3 -c \
      "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('event','?'),'|',d.get('timestamp_utc','?')[:19])" 2>/dev/null || echo "")
  fi

  EQUITY=""
  if [[ -f "$METRICS" ]]; then
    EQUITY=$(python3 -c \
      "import json; d=json.load(open('$METRICS')); print(f\"equity=\${d.get('equity',0):.0f} skip=\${d.get('skipped_signals_today',0)}\")" 2>/dev/null || echo "")
  fi

  echo "[$STATUS] $STRATEGY_ID | last: $LAST_EVENT | $EQUITY"
done
