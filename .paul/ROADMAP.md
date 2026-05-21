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
Phases: 1 of 4 complete

## Phases

| Phase | Name | Plans | Status | Completed |
|-------|------|-------|--------|-----------|
| 1 | Data Architecture | 3 | Complete | 2026-05-20 |
| 2 | Training Pipeline Consistency | 3 | Planning | - |
| 3 | Forecast & RL Quality | 3 | Not started | - |
| 4 | Paper Trading Validation | 2 | Not started | - |

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
- [ ] 02-02: forecast_engine.py training data provenance check
- [ ] 02-03: rl_intraday_structure_trainer.py validation + smoke test

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

**Plans:**
- [ ] 03-01: Feature engineering audit + improvements
- [ ] 03-02: RL reward function review + retraining (aligned to optimization objective)
- [ ] 03-03: Model evaluation harness + accuracy metrics + walk-forward validation

### Phase 4: Paper Trading Validation

**Goal:** End-to-end paper trading cycle runs cleanly daily with measurable plan→fill→exit metrics logged automatically. Composite strategy scores trending toward research target with all risk invariants intact.
**Depends on:** Phase 3 (models stable)
**Research:** Unlikely (integration patterns established)

**Scope:**
- Daily execution cycle smoke test (planner → run.py → fills)
- Options overlay integration check (options_booster.py)
- Paper trading metrics summary: composite score, Sharpe, daily P&L consistency, max drawdown, win rate, avg R, fill latency
- Strategy correlation check (capital_allocator.py) — flag correlated failures

**Plans:**
- [ ] 04-01: Daily execution cycle validation + composite metrics logging
- [ ] 04-02: Options overlay integration + paper trading performance summary report

---
*Roadmap created: 2026-05-20*
*Last updated: 2026-05-20*
