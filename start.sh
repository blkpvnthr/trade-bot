#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; create it and install project requirements first" >&2
  exit 1
fi
mkdir -p logs
export PYTHONUNBUFFERED=1
exec .venv/bin/python -u trade.py "$@"
