# Trade-Bot System Audit — 2026-07-13

Scope note up front: this audit ran in an isolated, ephemeral cloud sandbox with
no `.env` file and no outbound network access to Alpaca (the sandbox's proxy
returns `403 Forbidden` on the Alpaca API tunnel). That means **no live/paper
order was submitted, no real market data was pulled, and no trade-throughput
or fill-quality numbers could be measured this run** — doing so would require
running on the actual deployment host with real credentials. Everything below
is from static code review, dependency/install verification, git history
forensics, and `--diagnostics` dry-runs with placeholder credentials.

## 1. Headline finding: `run.py` does not match the documented architecture

`CLAUDE.md` describes `run.py` as a monolithic orchestrator that sequences
premarket collection → market context → trade planner → intraday dataset →
forecast/RL/options signal layers → execution alpha gate → Alpaca execution,
via a `ProductionOrderExecutor` class.

**That is not what is currently in `run.py`.** The file on this branch (and on
`main`) is a self-contained 1,252-line "ORB Sweep/Retest/Engulf" executor
(`ORBSweepEngulfExecutor`) that:

- Reads one input file: `daily_system_outputs/tradingview_volume_footprint.csv`
  (symbol/timestamp/OHLC/delta rows).
- Trades a single setup — opening-range liquidity sweep, retest, bullish
  engulfing confirmation, 3R bracket exit.
- Has **zero references** to `premarket_data_collector.py`,
  `market_context_builder.py`, `premarket_trade_planner.py`,
  `spx_price_action_layer.py`, `options_booster.py`, `forecast_engine.py`,
  `rl_intraday_structure_trainer.py`, `execution_alpha_gate.py`, or
  `subprocess` at all — confirmed by grep, not just a skim.

All of those subsystem files still exist in the repo (thousands of lines
each, last touched 2026-07-08) but nothing currently imports or calls them.
They are orphaned.

### Root cause (found via `git log -S`)

Commit `85050ba` ("Remove virtual environment from git tracking", 2026-05-28)
is titled as a git-hygiene cleanup, but its diff **also silently replaced the
entire `run.py`** — the previous 1,871-line file defining
`class ProductionOrderExecutor` was overwritten by a new 1,144-line file
defining `class ORBSweepEngulfExecutor`, with no mention of this in the
commit message, no update to `CLAUDE.md`, and no update to the test suite.
Every commit since (`ca707d7`, `e183c60`, `a63b180`, …) has built on top of
the new, disconnected executor — including good work like the recent
duplicate-order-bug fix and `--diagnostics` mode (`a63b180`), which are real
and correctly implemented, just on the wrong base.

**Evidence this is a regression, not a deliberate rename:** the repo's one
test file, `tests/test_live_experience_logging.py`, still targets
`run.ProductionOrderExecutor`, `run.build_default_config`, `run.BrokerPosition`
(as a richer type than today's), and a `ScannerConfig`-based config system —
none of which exist anymore. Running it fails immediately:

```
AttributeError: module 'run' has no attribute 'ProductionOrderExecutor'
```

This test was presumably green before `85050ba` and has been silently broken
in every run since (I did not find CI wired to run it automatically).

**I did not attempt to unilaterally revert or resurrect the old
`ProductionOrderExecutor`** — restoring 1,871 lines of orchestration logic
that hasn't been touched in ~1.5 months, whose relationship to the newer
duplicate-order fix and diagnostics mode is unknown, is a major architectural
call that needs your decision, not a scheduled routine's. The two live
options are:

1. **This was intentional** — the team pivoted to the simpler,
   footprint-CSV-driven ORB strategy and just never updated `CLAUDE.md` or
   deleted the stale test. In that case: update `CLAUDE.md` to describe
   `ORBSweepEngulfExecutor` accurately, delete or rewrite the orphaned test,
   and decide what to do with the now-dead premarket/context/planner/RL/
   options modules (archive vs. wire back in).
2. **This was accidental** — `85050ba` was meant to be a `.gitignore`/venv
   cleanup and something (bad merge, wrong file staged, agent session gone
   wrong) clobbered `run.py` in the same commit. In that case the fix is to
   recover the pre-`85050ba` `run.py` (`git show 85050ba~1:run.py`) and
   reconcile it with the fixes made since.

Either way, this explains most of the rest of this report.

## 2. Why 15–20 trades/day is not currently achievable

The current `run.py`'s only signal source is a manually-exported TradingView
volume-footprint CSV. **No script in the repo generates that file** — I
grepped the entire codebase for writers of
`tradingview_volume_footprint.csv` and found none; `run.py` is the only file
that even references the filename, and only as a reader. Absent a human
manually exporting and dropping that CSV before every session, the executor
will report `no_signal` every cycle. This is a hard ceiling on trade
throughput that has nothing to do with strategy quality, risk sizing, or
execution latency — it's a missing data pipeline. Restoring/reconnecting an
automated feed (either the orphaned premarket/planner pipeline from finding
#1, or a scripted TradingView export) is the prerequisite for any
throughput work.

## 3. Concrete bug fixed this run

`requirements.txt` listed two dependencies under their **import name** rather
than their **PyPI distribution name**, so `pip install -r requirements.txt`
failed outright on a clean environment:

- `python-tradingview-ta>=0.3.12` → no such PyPI package exists. Correct name:
  `tradingview-ta`. (Not actually imported anywhere in the codebase today —
  `tradingview_screener` is the module in active use.)
- `pywt>=1.6.0` → the importable module is `pywt`, but the PyPI package is
  `PyWavelets`. Also not currently imported anywhere.

Both are optional/unused today, but a broken `requirements.txt` blocks
**every** fresh install/deploy, which is a more severe availability risk than
either package's actual feature. Fixed both names, plus the same stale
references in `OPPORTUNITY_FINDER_GUIDE.md`. Verified: full
`pip install -r requirements.txt` now succeeds; `python3 -m py_compile` passes
on every top-level `.py` file and the audited subsystem modules
(`premarket_data_collector.py`, `market_context_builder.py`,
`premarket_trade_planner.py`, `spx_price_action_layer.py`,
`options_booster.py`, `forecast_engine.py`, `execution_alpha_gate.py`,
`build_intraday_training_dataset.py`) with zero syntax errors. Committed as
`b53d588` and pushed to `claude/kind-curie-naqntz`.

## 4. What was validated and is healthy

- `run.py --diagnostics` (added in `a63b180`) works correctly and degrades
  safely in both tested states: no `.env` present, and an `.env` with
  syntactically-valid-but-invalid/unreachable credentials. Neither case
  crashes; both print an actionable status line per subsystem (env,
  footprint file, signal state, kill switch, broker connectivity) and exit
  cleanly. This is good defensive design and matches the "safe fallback
  outputs" principle in `CLAUDE.md`.
- Kill-switch logic (`_kill_switch_allows_execution`) only gates *new* signal
  generation/order submission for the current cycle — it does not cancel
  resting bracket orders or otherwise touch already-open positions. Exits are
  handled by Alpaca-side bracket stop-loss/take-profit legs attached at entry,
  which the kill switch never touches. This is consistent with the "always
  allow exits, never block exits via kill switch" invariant.
- No syntax errors anywhere in the audited surface (all top-level modules +
  the major subsystem files).

## 5. Not validated (env constraints — needs to run on the real host)

- Actual signal-to-order / signal-to-fill conversion rates, execution
  latency, slippage — requires real Alpaca connectivity and real footprint
  data, neither available in this sandbox.
- Premarket collector, market context builder, planner, structure map,
  options booster, RL gates — these all still run standalone per
  `CLAUDE.md`'s "Standalone-runnable scripts" allowance, but none of them are
  reachable from `run.py` today (see #1), so auditing their *output* doesn't
  tell you anything about what the live executor will do with it.
- 15–20 trades/day capacity — blocked on #2 above; not a sizing/risk-control
  question yet, it's a data-pipeline question.

## Recommendation priority

1. Decide intentional-vs-regression on `run.py` (finding #1) — this blocks
   everything else and only you can make that call.
2. Once decided: either wire an automated footprint/premarket feed into the
   current executor, or restore/reconcile the `ProductionOrderExecutor` path
   with the fixes made since `85050ba`.
3. Fix or delete `tests/test_live_experience_logging.py` so the test suite
   reflects whichever executor is canonical.
4. Re-run this audit on the real deployment host (with credentials and
   network access) once #1 is resolved, to get real throughput/fill/slippage
   numbers.
