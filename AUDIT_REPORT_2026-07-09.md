# Trade-Bot System Audit — 2026-07-09

## Headline finding

**`run.py` is not the system documented in `CLAUDE.md`, `RUNTIME_READINESS_REPORT.md`, and `.paul/STATE.md`. It was silently replaced by a merge on 2026-05-28, and the replacement cannot currently produce a single trade.** This is a production-breaking regression, not a documentation drift issue.

## 1. Why a live execution audit could not be run

This sandbox has no `.env` (only `.env.*.example` templates, correctly git-ignored) and no `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` in the shell environment, and none of the project's Python dependencies were pre-installed (only stock Ubuntu/pip tooling). I installed the core runtime deps (`pandas`, `numpy`, `alpaca-py`, `yfinance`, `python-dotenv`, etc.) to enable static/dry-run analysis, but there is no way to authenticate to Alpaca or pull live premarket/market data from this environment. `python run.py --once` was run without credentials and failed exactly where it should:

```
RuntimeError: Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in /home/user/trade-bot/.env
```

That fail-fast behavior is correct. Everything below comes from static code analysis, git history, and cross-referencing the repo's own prior audit docs — not a live run.

## 2. The regression, in detail

- `.paul/STATE.md` (last updated 2026-05-22) describes the *real* running system as of that date: a `STRATEGY_ID`-dispatched orchestrator (`STRATEGY_ID=tux_mtf_original`), with `_startup_safety_checks()`, per-strategy log directories, sell-first execution, options booster, etc. — matching `RUNTIME_READINESS_REPORT.md`'s references to `run.py:7908`, `run.py:2287`, `run.py:10104`, etc.
- Commit `ca707d7` ("Fix executor handling of degraded system outputs", 2026-05-27) grew that orchestrator's `run.py` to **11,882 lines**.
- Commit `87eb6b6` (a parallel/earlier lineage) already carried a much smaller, different `run.py` (**1,144 lines**) — an "Opening-range liquidity sweep + retest + engulf" single-strategy executor.
- Merge commit `5264ee0` ("latest", 2026-05-28) merged `ca707d7` into `87eb6b6`. The merge resolution for `run.py` kept `87eb6b6`'s small file wholesale — **all 11,882 lines of the documented orchestrator were discarded**, including the safety-check startup gate, premarket-pipeline integration, options booster, forecast ensemble, RL gates, and `STRATEGY_ID`/`CONFIG_PATH` dispatch.
- Nothing since has reconciled this. `run.py` on `main` and on this branch today is the 1,144-line ORB-sweep script. `CLAUDE.md`, `RUNTIME_READINESS_REPORT.md`, `.paul/STATE.md`, `docker-compose.yml`, and every `instances/*/.env.example` still describe/assume the old architecture.

## 3. Consequences of the current `run.py` — verified by static analysis

1. **Orphaned signal source.** The only signal input is `daily_system_outputs/tradingview_volume_footprint.csv` (`DEFAULT_FOOTPRINT_FILE`, run.py:74). No script anywhere in this repository writes that file — `webhook.py`/`webhook_order_router.py` write to `daily_system_outputs/webhook_order_router/{signal_queue.jsonl,*_orders.csv}` instead, a different path. `premarket_trade_planner.py`, `market_context_builder.py`, `build_intraday_training_dataset.py`, `options_booster.py`, `rl_intraday_structure_trainer.py` etc. are never invoked from `run.py` (`grep` for `ProductionOrderExecutor|premarket_data_collector|spx_price_action_layer|subprocess` in `run.py` returns nothing). Every cycle, `find_new_signals()` will find no data and `run_cycle()` returns `no_signal`. **The entire premarket pipeline described in `CLAUDE.md` is disconnected from the execution engine.**

2. **Kill switch is misconfigured to permanently block trading on the intended account size.** `run.py:709-747`:
   ```python
   if equity <= self.config.kill_switch_equity_floor:
       blocked = True
       reason = "kill_switch_equity_floor"
   ```
   This trips when equity is **at or below** the floor — i.e. the floor is meant to sit below the account's expected equity, as a drawdown backstop. `instances/live/.env.example` gets this right (`KILL_SWITCH_EQUITY_FLOOR=12000`, with an explicit comment to size it to the real account). But `instances/default`, `instances/paper`, `instances/pivot_engine`, and `instances/rl_breakout` all ship `KILL_SWITCH_EQUITY_FLOOR=95000` — for the ~$13K account this whole system is designed around (per `CLAUDE.md`), `equity <= 95000` is true on every single cycle. The kill switch fires on the first cycle of day one, sets `kill_triggered=true` for the day, and (since equity never exceeds $95K) **re-triggers every subsequent day too**. `run_cycle()` gates on this before even loading signals (run.py:944-950), so new entries are blocked permanently under the shipped defaults. (Exits are not directly affected since this strategy exits via broker-side bracket legs attached at entry — but with the entry path dead, there's nothing to exit either.)

3. **The "multi-strategy fleet" in `docker-compose.yml` is fictional.** `docker-compose.yml` defines 10 services across `STRATEGY_ID` values `tux_mtf_original`, `tux_mtf_long_only`, `pivot_points`, `orb_8am`, `liquidity_sweep`, `flawless_victory` (6 distinct strategies configured in `configs/strategy_experiments.yaml`). `run.py` never reads `STRATEGY_ID` or `CONFIG_PATH` (confirmed via grep — zero matches). Every one of those 10 containers, regardless of label, runs the identical hardcoded ORB-sweep/retest/engulf logic against the same `DEFAULT_UNIVERSE` — which, per finding #1, never receives input data anyway.

Net effect: as currently checked in, this system — deployed exactly as `Dockerfile`'s `ENTRYPOINT ["python", "run.py"]` specifies — cannot place a single trade. Sections 2–7 of the requested audit (execution results, trade throughput, 15–20 trades/day capacity, fill quality, etc.) are moot until this is fixed; there is no execution to audit.

## 4. Recommended remediation (not applied — needs explicit sign-off)

This is a large, high-stakes revert/rebuild decision, so I did not make it unilaterally:

- **Option A — Revert to the orchestrator.** Restore `run.py` from `ca707d7` (`git show ca707d7:run.py`), then replay whatever later work is worth keeping from the current 1,144-line file (the ORB sweep/retest/engulf strategy itself looks reasonable and could be reintroduced as one `STRATEGY_ID` variant inside the restored orchestrator, rather than being the entire system).
- **Option B — Commit to the lightweight executor.** If the ORB-sweep executor is the intentional new direction, then: wire a real producer for `tradingview_volume_footprint.csv` (e.g. from `webhook_order_router.py`, or point `DEFAULT_FOOTPRINT_FILE` at what it actually writes), fix `KILL_SWITCH_EQUITY_FLOOR` in the four wrong `.env.example` files to a value below realistic equity (mirroring `instances/live`'s pattern), and either implement `STRATEGY_ID` dispatch or collapse `docker-compose.yml` to the one strategy that actually exists. Update `CLAUDE.md`, `RUNTIME_READINESS_REPORT.md`, and `.paul/STATE.md` to match reality either way.

Either path is a substantial change to the live-trading entrypoint and account-risk defaults — per `CLAUDE.md`'s own rule ("Do not make live-trading changes without explicit user intent"), I'm flagging this rather than picking a direction and pushing it.

## 5. Smaller, independently-fixable items noticed along the way

- `instances/default`, `instances/paper`, `instances/pivot_engine`, `instances/rl_breakout` `.env.example`: `KILL_SWITCH_EQUITY_FLOOR=95000` should be replaced with an account-appropriate floor per bot, following `instances/live`'s already-correct pattern and comment.
- `MAX_ORDER_DOLLARS` varies wildly across instance examples (1500 / 12000 / 1200 / 1000 / 1000) with no documented rationale tying each to its bot's actual account size — worth a single source of truth (e.g. derive from `MAX_POSITION_WEIGHT * expected_equity` per instance) rather than hand-picked literals.
- `daily_system_outputs/`, `system-data/`, and `logs/` are all empty in this checkout — consistent with no run ever having produced dated artifacts in this branch, and confirming the pipeline outputs referenced throughout `CLAUDE.md` don't exist here to audit.

## What I did *not* do

I did not modify `run.py`, any `instances/*/.env.example`, or `docker-compose.yml`. These findings describe a real regression with account-risk implications; the fix requires a decision (Option A vs B above) that should come from you, not from an unattended audit run.
