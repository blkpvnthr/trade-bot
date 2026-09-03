#!/usr/bin/env python3
"""
V2.9.8 no-argument runner.
==========================

    python run_engine.py

Runs the ENTIRE engine end-to-end with zero arguments, through the SAME
orchestration + validation path as ``live_trader.py rebalance`` (there is no
alternative order-submission implementation -- #20).

Mode is KEY/ENV-driven and safe-by-default:

  * PK... paper key present            -> PAPER (real paper orders on the paper account)
  * AK... live key + FULL live gate set -> LIVE  (real money; requires MODE=live +
                                            ALPACA_PAPER_ONLY=false + ENABLE_LIVE_TRADING=true)
  * AK... live key WITHOUT the full gates -> BLOCKED (never silently falls back to
                                            paper; never auto-crosses environments)
  * no credentials                      -> DRY-RUN (compute-only plan, 0 orders)

Timing is backtest-faithful: the engine always decides from the most recent
COMPLETED daily session, so this is safe to run at any time or on a schedule
(after-close is ideal). It is idempotent: re-running the same session does not
duplicate orders.
"""
from __future__ import annotations

import os
import sys

# live_trader loads the env file and pins single-threaded BLAS on import.
from live_trader import execute_rebalance, ExecutionConfig
from live_engine import Mode


def resolve_runner_mode(ecfg: ExecutionConfig) -> tuple[Mode, str | None]:
    """Return (mode, block_reason). block_reason non-None means refuse to run."""
    from alpaca_broker import _get_credentials, key_is_paper
    try:
        _get_credentials()
        have_creds = True
    except Exception:
        have_creds = False

    if not have_creds:
        return Mode.DRY_RUN, None

    if key_is_paper():
        return Mode.PAPER, None

    # Live key present -> require the full live gate set; never fall back to paper.
    mode_env = os.environ.get("MODE", "paper").strip().lower()
    paper_only = os.environ.get("ALPACA_PAPER_ONLY", "true").strip().lower() not in ("0", "false", "no", "off")
    if ecfg.live_enabled and mode_env == "live" and not paper_only:
        return Mode.LIVE, None
    return None, ("LIVE TRADING BLOCKED: a live (AK...) key is present but the full "
                  "live gate set is not (need MODE=live + ALPACA_PAPER_ONLY=false + "
                  "ENABLE_LIVE_TRADING=true). Refusing to auto-fall back to paper.")


def main() -> int:
    ecfg = ExecutionConfig.from_env()
    mode, block = resolve_runner_mode(ecfg)
    if block is not None:
        print(block, file=sys.stderr)
        return 2
    print(f"run_engine: resolved mode = {mode.value.upper()} (key/env-driven)")
    return execute_rebalance(mode, dry_run=(mode == Mode.DRY_RUN), ecfg=ecfg)


if __name__ == "__main__":
    raise SystemExit(main())
