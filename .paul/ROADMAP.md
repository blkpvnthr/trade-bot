# Roadmap: QuantumQuant

## Long-Term Objective

Build a self-improving quantitative trading system that generates consistent risk-adjusted returns through autonomous research, strategy evaluation, and adaptive execution.

**Optimization objective:** Sharpe ratio, daily P&L consistency, controlled drawdowns, positive expectancy, capital efficiency, execution quality, regime adaptability, low strategy correlation, walk-forward robustness. NOT raw return alone.

**Research target (paper):** ~$400–500/day average P&L while preserving all risk controls. Aspirational benchmark only — no guaranteed returns.

**Safety invariants (never modify):** kill switches, paper-only mode, max exposure limits, sell-order preservation.

---

## Overview

v0.1 establishes a solid production foundation: clean data architecture, reliable feed separation, and consistent training pipelines. v0.2 improves ML/RL model quality and validates the full paper-trading loop end-to-end. v0.3 adds performance analytics and system self-awareness for ongoing iteration.

## Current Milestone

**Production Foundation** (v0.1)
Status: In progress
Phases: 2 of 4 complete

**Next Milestone: Unified Orchestration** (v0.2)
Status: Not started — begins after v0.1 completes
See Phase details below.

## Phases

### v0.1 — Production Foundation

| Phase | Name | Plans | Status | Completed |
|-------|------|-------|--------|-----------|
| 1 | Data Architecture | 3 | Complete | 2026-05-20 |
| 2 | Training Pipeline Consistency | 3 (+02-02b) | ✅ Complete | 2026-07-03 |
| 3 | Forecast & RL Quality | 3 | Planning (03-01 created) | - |
| 4 | Paper Trading Validation | 3 | Not started | - |

### v0.2 — Unified Orchestration

| Phase | Name | Plans | Status | Completed |
|-------|------|-------|--------|-----------|
| 5 | Orchestration Audit | 1 | Not started | - |
| 6 | Walk-Forward & Eval Hooks | 2 | Not started | - |
| 7 | Self-Optimization Integration | 2 | Not started | - |

## Phase Details

### Phase 1: Data Architecture

**Goal:** Alpaca as the single authoritative OHLCV source with explicit feed separation and zero accidental yfinance-for-OHLCV usage.
**Depends on:** Nothing (foundation)
**Research:** Unlikely (Alpaca SDK patterns established)

**Scope:**
- Alpaca-first premarket + intraday bars (SIP→IEX retry)
- Split ALPACA_PREMARKET_FEED / ALPACA_REGULAR_FEED env vars
- ALPACA_FAIL_ON_FEED_ERROR and ALLOW_YFINANCE_MARKET_DATA_FALLBACK controls
- Git cleanup: runtime artifacts removed from tracking, .gitignore hardened
- 3-layer architecture separation documented

**Plans:**
- [x] 01-01: Alpaca-first migration in premarket_trade_planner.py
- [x] 01-02: Feed separation (PREMARKET/REGULAR/FAIL_ON_ERROR/ALLOW_YF_FALLBACK)
- [x] 01-03: Git cleanup + 3-layer architecture docs

### Phase 2: Training Pipeline Consistency

**Goal:** RL and ML training datasets use the same Alpaca feed config as live execution, so training data mirrors what the system sees in production.
**Depends on:** Phase 1 (feed config established)
**Research:** Unlikely (same SDK patterns)

**Scope:**
- build_intraday_training_dataset.py aligned to ALPACA_REGULAR_FEED
- forecast_engine.py consumes only Alpaca-derived training CSVs
- rl_intraday_structure_trainer.py data provenance validated
- Smoke test: full planner → training dataset → model inference cycle

**Plans:**
- [x] 02-01: build_intraday_training_dataset.py feed alignment + provenance diagnostics
- [x] 02-02: forecast_engine.py training data provenance check (+ 02-02b corrective)
- [x] 02-03: rl_intraday_structure_trainer.py canonical I/O + provenance + smoke test (AC-5 live confirmation deferred)

### Phase 3: Forecast & RL Quality

**Goal:** Forecast ensemble and RL policy produce measurable directional accuracy above 55% on held-out intraday data. Model quality directly supports the optimization objective: better-calibrated forecasts → higher Sharpe, lower drawdowns, better regime adaptability.
**Depends on:** Phase 2 (consistent training data)
**Research:** Likely (hyperparameter tuning, feature selection)
**Research topics:** Feature importance, LSTM architecture variants, RL reward shaping

**Scope:**
- Feature engineering review (forecast_engine.py, build_intraday_training_dataset.py)
- RL reward function aligned to composite optimization objective (reward Sharpe/consistency, penalize drawdowns)
- Model evaluation harness with held-out validation and walk-forward windows
- Forecast ensemble metrics logged to system-data/

**Plans:** (reordered 2026-07-03 — reward shaping promoted to 03-01; DISCOVERY ready)
- [ ] 03-01: RL reward shaping — DDR (differential Sortino) replaces naive rolling-Sharpe, selectable + unit-proven (retrain/A-B deferred to 03-03)
- [ ] 03-02: Feature engineering audit + improvements
- [ ] 03-03: Model evaluation harness + accuracy metrics + walk-forward validation (empirical A/B of DDR vs baseline)

### Phase 4: Paper Trading Validation

**Goal:** End-to-end paper trading cycle runs cleanly daily with measurable plan→fill→exit metrics logged automatically. Composite strategy scores trending toward research target with all risk invariants intact.
**Depends on:** Phase 3 (models stable)
**Research:** Unlikely (integration patterns established)

**Scope:**
- Daily execution cycle smoke test (planner → run.py → fills)
- Options overlay integration check (options_booster.py)
- Paper trading metrics summary: composite score, Sharpe, daily P&L consistency, max drawdown, win rate, avg R, fill latency
- Strategy correlation check (capital_allocator.py) — flag correlated failures
- Time-of-day regime gating: session-aware buy sizing and confirmation rules (morning full-size 09:30–11:30, midday reduced 11:30–13:00, afternoon continuation 13:00–15:30, close exit-only 15:30–16:00); exits and stop-losses never blocked

**Plans:**
- [ ] 04-01: Daily execution cycle validation + composite metrics logging
- [ ] 04-02: Options overlay integration + paper trading performance summary report
- [ ] 04-03: Time-of-day regime gating + session discipline layer

---

## v0.2 Phase Details

### Phase 5: Orchestration Audit

**Goal:** Produce a complete gap analysis of `run.py`'s current orchestration coverage — what it calls, how it calls it, and what's missing — before any new orchestration code is written.
**Depends on:** Phase 4 (v0.1 complete, paper trading baseline established)
**Research:** Unlikely (read-only audit of existing code)

**Scope:**
- Inventory every subprocess call in `run.py` (blocking vs. Popen background)
- Inventory every direct import and file-based passive integration
- Identify all `.py` files in project root that are NOT called by `run.py`
- Map the current pipeline DAG: what runs, in what order, with what output
- Identify absent hooks for: walk-forward scripts, evaluate_variants, diagnose_strategy, self-optimization agent
- Produce `docs/orchestration_gap_report.md` — gap analysis only, no new code

**Plans:**
- [ ] 05-01: run.py orchestration coverage audit + DAG + gap report

### Phase 6: Walk-Forward & Eval Hooks

**Goal:** `run.py` triggers walk-forward validation and composite evaluation after each training cycle, with results written to `daily_system_outputs/` and readable by the next execution cycle.
**Depends on:** Phase 5 (gap analysis complete)
**Research:** Unlikely (patterns established)

**Scope:**
- Add safe orchestration interface to `scripts/walkforward_strategy.py` (no circular imports)
- Add safe orchestration interface to `scripts/evaluate_variants.py`
- `run.py` triggers both as background Popen after model training, reads results next cycle
- Preserve standalone invocability of both scripts
- No execution or risk regression

**Plans:**
- [ ] 06-01: Walk-forward hook (run.py → walkforward_strategy.py → composite score output)
- [ ] 06-02: Eval hook (run.py → evaluate_variants.py → ranking + promotion gate)

### Phase 7: Self-Optimization Integration

**Goal:** `run.py` can trigger the self-optimization agent to generate adaptive weight updates based on recent composite scores, with a human-in-the-loop approval gate before any weight changes apply.
**Depends on:** Phase 6 (eval results flowing automatically)
**Research:** Likely (self-optimization safety design)

**Scope:**
- Trigger `scripts/claude_self_optimization_agent.py` on a configurable schedule (not every run)
- Results written to `daily_system_outputs/` as a proposed update — NOT auto-applied
- Human approval checkpoint before weights change in `configs/strategy_experiments.yaml`
- Kill switch: `SELF_OPTIMIZATION_ENABLED=0` disables entirely
- Audit trail: every proposed and accepted change is logged

**Plans:**
- [ ] 07-01: Self-optimization trigger + output format + approval gate design
- [ ] 07-02: Adaptive weight update apply mechanism + audit trail

---
*Roadmap created: 2026-05-20*
*Last updated: 2026-07-03 — Phase 2 (Training Pipeline Consistency) complete; Phase 3 DISCOVERY done (DDR reward shaping)*
