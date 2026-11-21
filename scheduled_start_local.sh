#!/bin/zsh
# -----------------------------------------------------------------------------
# scheduled_start_local.sh
#
# Canonical launcher for the continuous AMPS V6 trade.py service, supervised by
# the launchd agent com.quantumquant.trader925.
#
# Runs trade.py fully functional (order submission + enforced exits + order-flow,
# not SHADOW). It defaults to the PAPER account. It deliberately does NOT pass
# --live and does NOT set ALLOW_LIVE_TRADING: going live is the operator's lever.
#
#   To run LIVE instead, change the exec line to:
#       ALLOW_LIVE_TRADING=1 exec .venv/bin/python -u trade.py --enforce-exits --live
#   (LIVE moves real money and is irreversible.)
# -----------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; create it and install project requirements first" >&2
  exit 1
fi

mkdir -p logs
export PYTHONUNBUFFERED=1

# Fully functional, paper account (operator controls live via the comment above).
exec .venv/bin/python -u trade.py --enforce-exits "$@"
