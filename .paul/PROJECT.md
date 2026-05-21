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
| Last Updated | 2026-05-20 |

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

### Active (In Progress)

- [ ] RL training pipeline feed consistency (build_intraday_training_dataset.py aligned to ALPACA_REGULAR_FEED)
- [ ] End-to-end smoke test with new feed config

### Planned (Next)

- [ ] forecast_engine.py: consume only Alpaca-derived training datasets (no yfinance-sourced CSVs)
- [ ] rl_intraday_structure_trainer.py: validate training data provenance
- [ ] options_booster.py: integration validation
- [ ] Paper trading metrics dashboard

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
*Last updated: 2026-05-20*
