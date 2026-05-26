#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

INSTANCES=(
  "tux_mtf_original .env.bot01"
  "pivot_points .env.bot02"
  "liquidity_sweep .env.bot03"
)

mkdir -p .local_pids

for entry in "${INSTANCES[@]}"; do
  STRATEGY_ID="${entry%% *}"
  ENV_FILE="${entry##* }"

  if [[ ! -f "$ENV_FILE" ]]; then
    echo "[skip] $STRATEGY_ID — $ENV_FILE not found (copy from ${ENV_FILE}.example and fill credentials)"
    continue
  fi

  PID_FILE=".local_pids/${STRATEGY_ID}.pid"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[already-running] $STRATEGY_ID (PID $(cat "$PID_FILE"))"
    continue
  fi

  LOG_DIR_PATH="logs/execution/$STRATEGY_ID"
  mkdir -p "$LOG_DIR_PATH"

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  nohup python run.py > "$LOG_DIR_PATH/startup.log" 2>&1 &
  echo $! > "$PID_FILE"
  echo "[started] $STRATEGY_ID PID=$! log=$LOG_DIR_PATH/startup.log"
done
