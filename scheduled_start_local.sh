#!/bin/zsh
cd /Users/blkpvnthr/Desktop/trader || exit 1

LOG_DIR="/Users/blkpvnthr/Desktop/trader/logs"
mkdir -p "$LOG_DIR"

exec >> "$LOG_DIR/scheduled_run_$(date +%Y%m%d).log" 2>&1

echo "=== scheduled_start_local $(date) ==="

export APCA_PAPER=true
export ALPACA_PAPER_ONLY=true
export MODE=paper
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"

/Users/blkpvnthr/Desktop/trader/.venv/bin/python /Users/blkpvnthr/Desktop/trader/run.py
