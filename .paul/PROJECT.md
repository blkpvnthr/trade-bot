# QuantumQuant

## What This Is

QuantumQuant is an automated intraday trading system running in paper mode. It discovers institutional accumulation candidates via TradingView screener, builds premarket pivot-structure trade plans, executes via Alpaca, and continuously trains RL and ML forecast models on live intraday data. The system produces a daily trade plan, manages entry/exit/sizing logic, and overlays an options booster layer on qualified setups.

## Core Value

An autonomous paper-trading engine that turns institutional accumulation signals into structured, risk-controlled intraday trade plans — so execution is systematic and testable, not discretionary.

## Long-Term Objective

Build a self-improving quantitative trading system capable of generating consistent risk-adjusted returns through autonomous research, strategy evaluation, and adaptive execution.

**Research target (paper trading):** ~$400–500/day average P&L while maintaining strict drawdown controls, capital stability, and avoiding catastrophic losses. This is an aspirational benchmark for paper-trading research — not a guaranteed return or a hard target.

**Optimization objective — NOT raw return alone.** The system optimizes for:
- Sharpe ratio (primary)
- Daily P&L consistency (stable profitable days)
- Controlled drawdowns (Calmar ratio)
- Positive expectancy (profit factor)
- Capital efficiency
- Execution quality (low overtrading)
- Regime adaptability (alpha decay monitoring)
- Low correlation between strategy variants
- Out-of-sample walk-forward robustness

**The system explicitly avoids:**
- Martingale behavior or unlimited averaging down
- Revenge trading patterns
- Unsafe leverage or uncapped exposure
- Uncontrolled overnight risk
- Removing or weakening kill switches
- Any auto-promotion to live trading

## Current State

| Attribute | Value |
|-----------|-------|
| Type | Application |
| Version | v0.1 |
| Status | Beta (paper trading) |
| Last Updated | 2026-07-03 |

## Requirements

### Core Features

- Universe discovery via TradingView institutional screener (top 30 names)
- Premarket pivot-structure trade planning (breakout, reclaim, consolidation setups)
- Intraday execution engine with risk gates and regime filters
- RL intraday structure trainer + ML forecast ensemble
- Options overlay for qualified setups

### Validated (Shipped)

- [x] TradingView screener universe discovery — v0.1
- [x] premarket_trade_planner.py with Alpaca-first OHLCV (SIP→IEX retry) — v0.1
- [x] run.py execution engine with risk/regime gates — v0.1
- [x] Alpaca feed architecture separation (PREMARKET_FEED/REGULAR_FEED/FAIL_ON_FEED_ERROR) — v0.1
- [x] 3-layer architecture separation (source / runtime / Obsidian vault) — v0.1
- [x] Git cleanup: 244 runtime artifacts removed from tracking — v0.1
- [x] RL training pipeline feed consistency (build_intraday_training_dataset.py aligned to ALPACA_REGULAR_FEED) — Phase 2 (02-01)
- [x] forecast_engine.py: consume only Alpaca-derived training datasets — Phase 2 (02-02/02-02b)
- [x] rl_intraday_structure_trainer.py: canonical date-stamped I/O + Alpaca provenance validated — Phase 2 (02-03)
- [x] Static training-pipeline smoke test (10/10, fail-exit gated) — Phase 2 (02-03)

### Active (In Progress)

- [ ] Phase 3: Forecast & RL quality — directional accuracy >55% OOS, reward aligned to composite objective (see DISCOVERY: DDR reward shaping)

### Planned (Next)

- [ ] options_booster.py: integration validation — Phase 4
- [ ] Paper trading metrics dashboard — Phase 4

### Deferred (tracked)

- [ ] AC-5 live confirmation: validate_rl_intraday_structure_model()=True against a freshly-trained canonical model — 02-03 (XS; needs a real trading session, not thin holiday data)

### Out of Scope

- Live / real-money trading — paper only until explicitly approved
- Options execution — overlay signals only, no automated options orders

## Target Users

**Primary:** System operator (solo quant)
- Runs the system daily premarket
- Monitors signals, logs, and diagnostics
- Evaluates model improvements
- Does not intervene in individual trade decisions

## Context

**Technical Context:**
- Primary data: Alpaca StockHistoricalDataClient (SIP/IEX/BOATS feeds)
- Fallback data: yfinance (disabled by default when ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0)
- Universe/TA enrichment: tradingview-screener library (no OHLCV, discovery only)
- ML stack: PyTorch LSTM, sklearn, numpy/pandas
- Runtime artifacts in system-data/ and daily_system_outputs/ — gitignored
- Research vault: ~/Documents/quantumquant/ (Obsidian, Layer 3)

**Business Context:**
- All positions are paper trades (APCA_PAPER=true)
- Risk controls and exit logic are protected — never suppressed

## Constraints

### Technical Constraints

- Paper trading only (APCA_PAPER=true enforced)
- Alpaca free tier: IEX feed only for regular session; BOATS/SIP requires paid subscription
- Planner must finish within PLANNER_TOTAL_TIMEOUT_SECONDS (240s) to avoid collector timeout
- Sell orders must never be suppressed
- Exit logic preserved across all refactors

### Business Constraints

- No live order execution without explicit operator approval
- yfinance acceptable as fallback ONLY when Alpaca is unavailable

## Key Decisions

| Decision | Rationale | Date | Status |
|----------|-----------|------|--------|
| Alpaca as authoritative OHLCV source | yfinance lacks reliable premarket data and throttles | 2026-05-20 | Active |
| Split PREMARKET_FEED vs REGULAR_FEED | Different feed subscriptions cover different sessions | 2026-05-20 | Active |
| TradingView screener: discovery only, no OHLCV | Screener provides ranking signal; Alpaca owns price data | 2026-05-20 | Active |
| 3-layer architecture (source/runtime/Obsidian) | Prevents runtime artifacts polluting git history | 2026-05-20 | Active |
| Align producer to consumers' canonical paths (not vice-versa) | trainer emitted static paths 5+ consumers couldn't find; fixing the producer default keeps consumers untouched | 2026-07-03 | Active |
| `[data-source] training_csv=...` provenance log at every training-data load | subprocess stdout provenance so run.py/consumers can prove feed lineage | 2026-07-03 | Active |

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Premarket planner completion rate | >95% | Unknown | Measuring |
| Alpaca data coverage (symbols with bars) | >90% of universe | Unknown | Measuring |
| Forecast model accuracy (directional) | >55% | Unknown | Training |
| Plan → execution cycle time | <5 min | Unknown | Measuring |
| Composite strategy score (best variant) | >0.35 | Unknown | Evaluating |
| Portfolio Sharpe ratio (30-day rolling) | >1.0 | Unknown | Evaluating |
| Max drawdown (paper equity) | <5% of capital | Unknown | Measuring |
| Daily P&L consistency (profitable days %) | >60% | Unknown | Measuring |
| Walk-forward window win rate | >55% | Unknown | Pending data |
| Strategy correlation (avg pairwise) | <0.4 | Unknown | Measuring |

## Orchestration Architecture

`run.py` is the canonical top-level orchestrator and sole runtime entrypoint. All subsystem scripts are modular and standalone-testable, but production workflows are coordinated through `run.py`.

### Architectural Rules

- `run.py` owns the full session lifecycle: sequencing, safety gates, diagnostics, and runtime state
- Subsystem files may be invoked standalone for development and testing
- No subsystem script should import `run.py` — imports flow one direction only
- New production orchestration hooks must be added to `run.py`; no parallel orchestrators
- Circular imports involving `run.py` are forbidden

### 10 Orchestration Stages

| Stage | Responsibility | Current Owner |
|-------|---------------|---------------|
| 1. Data collection | Universe discovery, premarket bars, market context | `llm_sentiment_universe_selector.py`, `market_context_builder.py`, `premarket_data_collector.py` |
| 2. Feature generation | Intraday feature streams, multi-timeframe context | `build_intraday_training_dataset.py` |
| 3. Training dataset generation | Labeled OHLCV + exit datasets | `build_intraday_training_dataset.py` |
| 4. Model training / retraining | RL timing model, ML forecast ensemble | `rl_intraday_structure_trainer.py`, `forecast_engine.py` |
| 5. Walk-forward validation | OOS Sharpe, parameter stability, win rate | `scripts/walkforward_strategy.py` *(not yet in run.py)* |
| 6. Strategy evaluation | Composite scoring, variant ranking, promotion gating | `scripts/evaluate_variants.py` *(not yet in run.py)* |
| 7. Diagnostics | Provenance checks, self-assessment, planner health | `system_self_awareness.py`, `scripts/diagnose_strategy.py` *(partial)* |
| 8. Self-optimization requests | Adaptive weight updates, patch generation | `scripts/claude_self_optimization_agent.py`, `scripts/adaptive_weight_optimizer.py` *(not yet in run.py)* |
| 9. Paper-trading execution | Order construction, risk gates, Alpaca submission | `run.py` (inline), `execution_alpha_gate.py`, `options_booster.py`, `portfolio_rebalancer.py` |
| 10. Logging / metrics / lifecycle | Journal, equity curve, experience replay, systemd hooks | `run.py` (inline), `logs/execution/` |

### Integration Patterns (current v0.1)

| Pattern | Examples | Notes |
|---------|----------|-------|
| Subprocess blocking | `market_context_builder.py`, `premarket_trade_planner.py`, `forecast_engine.py`, `rl_intraday_structure_trainer.py` | Session-critical; `run.py` waits for exit |
| Subprocess background (Popen) | `build_intraday_training_dataset.py`, `spx_price_action_layer.py` | Non-blocking; results read later from disk |
| Direct import | `execution_alpha_gate.py`, `system_self_awareness.py` | Fast-path; no subprocess overhead |
| File-based passive | `deeplob_orderbook_direction_layer.py` | `run.py` reads signal JSON; script not invoked |
| Inline (embedded) | Evolution Strategy, PSO-LSTM, DWT denoising, scanner | Logic embedded in `run.py`; not yet extracted |

### Known Orchestration Gaps (v0.1 → v0.2 work)

Stages 5–8 (walk-forward validation, strategy evaluation, diagnostics, self-optimization) run as standalone scripts today. `run.py` has no hooks for them. v0.2 will add safe orchestration interfaces — see ROADMAP.

## Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Universe | tradingview-screener | Discovery + TA enrichment only |
| Market data | Alpaca alpaca-py SDK | SIP/IEX/BOATS feeds |
| Execution | Alpaca paper trading API | APCA_PAPER=true |
| ML/RL | PyTorch, sklearn | LSTM forecast + RL structure trainer |
| Data processing | pandas, numpy | OHLCV pipeline |
| Config | python-dotenv | .env file, gitignored |
| Research vault | Obsidian | ~/Documents/quantumquant/ |

## Links

| Resource | URL |
|----------|-----|
| Repository | ~/Desktop/trader |
| Research Vault | ~/Documents/quantumquant |

---
*PROJECT.md — Updated when requirements or context change*
*Last updated: 2026-07-03 after Phase 2 (Training Pipeline Consistency)*
