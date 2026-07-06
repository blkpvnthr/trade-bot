# Audit: run.py / trade.py Divergence and Trade Funnel Review

**Date:** 2026-07-06
**Scope:** Requested "continuous trading system audit" (execute run.py, monitor, audit trade funnel, optimize throughput).
**Environment constraint:** This session runs in an ephemeral container with no `.env` credentials and none of the required
dependencies installed (`alpaca-py`, `pandas`, `torch`, `stable-baselines3`, `yfinance` are all absent — confirmed via
`python3 run.py`, which fails immediately at `import pandas`). Live execution, multi-day P&L tracking, and the
"15-20 trades/day, 4/5 days > +$500" validation criteria in the task request are **not achievable from this environment**.
This audit is a static code review instead.

## Headline finding: `run.py` silently lost its entire gating pipeline

`CLAUDE.md` and `docs/architecture.md` both describe `run.py` as the sole orchestrator that directly imports and runs
`market_context_builder.py`, `premarket_trade_planner.py`, `execution_alpha_gate.py`, `options_booster.py`,
`opportunity_finder.py`, and the in-process SR-RL PPO loop. **None of that is true of the `run.py` currently in the repo.**

- Current `run.py` (1,144 lines) is a standalone "opening-range liquidity sweep + retest + engulfing confirmation"
  executor that reads a TradingView footprint CSV. It has zero imports of any of the modules above.
- All of that documented pipeline logic actually lives in **`trade.py`** (6,016 lines) — a file not mentioned anywhere
  in `CLAUDE.md` or `docs/architecture.md`. `trade.py` imports `execution_alpha_gate.add_execution_alpha_scores` and
  contains its own market-context/premarket-plan/options-booster/opportunity-finder orchestration.
- `Dockerfile` (`ENTRYPOINT ["python", "run.py"]`) and every file under `instances/*/.env.example`
  (`Run with: INSTANCE=... python run.py`, including `instances/live/`) point at **`run.py`** as the production
  entrypoint — i.e. the file that is actually deployed is the bare ORB-sweep-engulf executor, not the documented
  multi-layer pipeline.

### How this happened (git archaeology)

`run.py` had a real, matching history through commit `e183c60` ("Integrate SR RL PPO policy in-process into run.py",
2026-05-27) — at that point it was 11,617 lines and matched the documented architecture (contained
`_sr_rl_agent_loop`, `opportunity_finder`, `execution_alpha_gate`, etc.). Two commits later:

```
ca707d7 (11,882 lines)  "Fix executor handling of degraded system outputs"
dce00cc                 "add README"
85050ba (1,144 lines)   "Remove virtual environment from git tracking"
```

The commit labeled **"Remove virtual environment from git tracking"** actually replaced `run.py` wholesale
(net -11,655 / +917 lines) with the current ORB-sweep-engulf strategy, alongside genuinely unrelated changes
(`.gitignore` edits, log file additions). The commit message does not reflect what happened to `run.py`. Whether this
was an intentional strategy pivot with a bad commit message, or an accidental overwrite (e.g. a bad merge/checkout
during cleanup), is not recoverable from the diff alone — **this needs the repo owner to confirm intent** before
anyone touches it further.

### Why this matters for the "trade throughput" ask

The task's premise — audit why `run.py`'s gating pipeline isn't producing 15-20 trades/day — doesn't apply to the
file that's actually deployed, because that pipeline isn't in `run.py` anymore. Any threshold-tuning done against
`run.py` today would be tuning kill-switch/position-sizing parameters on an ORB strategy, not the documented
multi-signal system. Conversely, if `trade.py` is what should be deployed, production is currently running the wrong
file entirely.

**Recommendation (not implemented — needs a decision, not a fix):** Confirm with the repo owner which file is meant
to be canonical going forward:
1. If the ORB-sweep-engulf `run.py` is the intended current strategy, `trade.py` and the stale docs describing the
   old pipeline should be archived/updated so the architecture docs match reality.
2. If `trade.py`'s multi-layer pipeline was meant to stay in production, `run.py` should be restored from `e183c60`
   (or reconciled with whatever intentional changes happened after) and `trade.py`'s independent improvements
   (bar caching, rate-limit retries, options flow fixes from `b8282f0`) merged back in.

Either path is a substantial, deliberate change to the production entrypoint and is out of scope for this audit to
decide unilaterally.

## Secondary findings (static review, no code changes made)

- **Dead code confirmed:** `deeplob_orderbook_direction_layer.py` is not imported by `run.py` or `trade.py` (or
  anywhere else) — signals from it are never produced. This matches the deferred issue already tracked in
  `.paul/STATE.md`, but that note undersells scope: it's inert regardless of which file is canonical.
  `sr_rl_agent.py` is also standalone/not invoked by current `run.py` (it *was* wired into the pre-rewrite version at
  `e183c60`; that integration disappeared with the rest of the pipeline).
- **Two scoring implementations risk drift:** `execution_alpha_gate.py:102-136` and `trade.py:3803-4034` implement
  near-duplicate execution-alpha scoring. They currently agree but aren't shared, so they can silently diverge.
- **Options underlying-price fallback is implemented correctly** in `options_booster.py` (row fields → Alpaca latest
  trade → local intraday close → structure map), matching the fallback order `CLAUDE.md` documents as a known risk
  area. No bug found here.
- **No bare `except:` clauses** found anywhere in the audited modules; the `except Exception: pass` blocks in
  `options_booster.py`'s price-fallback chain (lines ~453, 465, 475) are intentional chain-continuation, not silent
  failure.
- **`trade.py`'s gating stack (as documented, for context if it's the intended pipeline):** MTF alignment requires
  3-of-4 bullish timeframes; execution-alpha threshold is 0.70 (blended 0.55 cross-sectional / 0.45 time-series),
  multiplied by 0.70 in risk-off regimes (effective ~0.49 floor); options booster additionally stacks
  confidence≥0.75, direction_score≥0.64, ml_prob≥0.60. This combination is strict enough that, if `trade.py` is ever
  restored to production, it would plausibly explain low trade counts on its own, independent of the file-divergence
  issue above.

## Git hygiene fix already applied

`logs/execution/*.jsonl`, `logs/execution/scanner_state.json`, and `logs/launchd_tradebot.*.log` were tracked in git
despite `logs/` being in `.gitignore` (added before those files were tracked, so the ignore never took effect
retroactively) and despite `CLAUDE.md`'s explicit instruction not to commit execution logs. These were stale,
single-day dry-run artifacts from a different machine (paths reference `/Users/blkpvnthr/Desktop/trader/...`).
Untracked via `git rm --cached` in commit `a381a81` (files remain on disk, just no longer git-tracked). No behavior
change.

## What was not done

Per the task's own stated constraint ("Do not reduce risk standards merely to increase trade count") and the
CLAUDE.md rule against unilateral changes to production orchestration, no gating thresholds were loosened, no
attempt was made to reconcile/merge `run.py` and `trade.py`, and no live execution was attempted (not possible in
this environment regardless). The `run.py`/`trade.py` divergence is the one finding in this audit that needs a human
decision before any further optimization work on the trade funnel is meaningful.
