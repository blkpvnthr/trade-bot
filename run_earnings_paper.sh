#!/usr/bin/env bash
# Daily earnings_within_7d paper run: fetch forward calendar -> run paper -> reconcile.
#
# Live trading is impossible here. The Alpaca leg places orders on a PAPER account only, and only
# when dedicated paper 'PK' keys are present; otherwise it fail-closes to the in-process simulator.
#
# Paper broker keys (optional, kept OUT of the live .env): create .env.earnings_paper with
#   EARNINGS_ALPACA_PAPER_KEY_ID=PK...
#   EARNINGS_ALPACA_PAPER_SECRET_KEY=...
# then run:  EARNINGS_PAPER_BROKER=alpaca ./run_earnings_paper.sh
# Without those, it runs the simulator (EARNINGS_PAPER_BROKER defaults to sim).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

# isolated paper keys (never the live APCA_* keys)
[ -f .env.earnings_paper ] && { set -a; source .env.earnings_paper; set +a; }

# 1) fresh forward earnings calendar (ALPHA_VANTAGE_API_KEY read from .env via dotenv)
python3 - <<'PY'
import dotenv; dotenv.load_dotenv('.env')
from datetime import date
from rl_trading.sleeves import earnings_features as EF
from rl_trading.sleeves.initial import EARNINGS_UNIVERSE
print('[calendar]', EF.fetch_av_earnings_calendar(list(EARNINGS_UNIVERSE), date.today()).name)
PY

# 2) paper run (override active; default broker=sim; set EARNINGS_PAPER_BROKER=alpaca for real paper)
# --equity 20000: trade a SMALL slice. The sleeve's conservative caps (per_order $2k, max_notional
# $10k) only stay consistent with the plan's %-sizing at ~$20k equity; at $100k the per-order cap
# would reject every order. This deliberately trades tiny on the $1M paper account ("size small").
SCANNER_ONLY=0 EARNINGS_SLEEVE_PAPER_ENABLED=1 EARNINGS_PAPER_OVERRIDE=1 \
  EARNINGS_PAPER_BROKER="${EARNINGS_PAPER_BROKER:-sim}" \
  python -m rl_trading.sleeves.earnings_within_7d --equity "${EARNINGS_PAPER_EQUITY:-20000}" --json

# 3) reconcile shadow logs vs actual reportedDate (accrues toward READY_FOR_PAPER; never orders)
python -m rl_trading.sleeves.earnings_within_7d --reconcile
