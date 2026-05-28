#!/usr/bin/env bash
cd "$(dirname "$0")/.."

for STRATEGY_ID in tux_mtf_original pivot_points liquidity_sweep webhook_order_router; do
  PID_FILE=".local_pids/${STRATEGY_ID}.pid"
  [[ ! -f "$PID_FILE" ]] && continue
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "[stopping] $STRATEGY_ID PID=$PID (SIGTERM — waiting up to 30s)"
    kill -TERM "$PID"
    for _ in $(seq 1 30); do
      sleep 1
      kill -0 "$PID" 2>/dev/null || break
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "[force-kill] $STRATEGY_ID PID=$PID (SIGKILL)"
      kill -KILL "$PID"
    fi
  fi
  rm -f "$PID_FILE"
  echo "[stopped] $STRATEGY_ID"
done
