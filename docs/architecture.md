# QuantumQuant — System Architecture

> **Status:** v0.1 Production Foundation  
> **Last updated:** 2026-05-20  
> This document describes the current architecture and planned orchestration evolution.

---

## Design Principles

1. `run.py` is the single top-level runtime orchestrator. All production workflows flow through it.
2. Subsystem scripts are modular and standalone-testable. They never import `run.py`.
3. Risk controls, kill switches, and safety gates live in `run.py` and are never bypassed.
4. Data flows from Alpaca → system-data/ → subsystem scripts → run.py (one direction).
5. Runtime artifacts (CSVs, JSONs, model zips) are ephemeral — never committed to git.

---

## Layer Separation

| Layer | Location | Git? | Description |
|-------|----------|------|-------------|
| **1 — Source** | `~/Desktop/trader/` | Yes | Executable system, all `.py` files, configs |
| **2 — Runtime** | `system-data/`, `daily_system_outputs/`, `logs/` | No | Generated artifacts, dated by YYYYMMDD |
| **3 — Research** | `~/Documents/quantumquant/` (Obsidian) | No | Strategy notes, ADRs, session logs |

---

## Pipeline DAG

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-SESSION  [4:00 AM – 9:30 AM ET]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  └─► premarket_data_collector.py  [subprocess blocking]
        ├─► llm_sentiment_universe_selector.py
        │     └── writes: YYYYMMDD_locked_trading_universe.txt
        │                 YYYYMMDD_latest_llm_sentiment_universe.json
        ├─► market_context_builder.py
        │     └── writes: YYYYMMDD_market_context.json
        └─► premarket_trade_planner.py
              └── writes: YYYYMMDD_premarket_trade_plan.json
                          YYYYMMDD_intraday_structure_map.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET OPEN  [9:30 AM ET]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  └─► spx_price_action_layer.py  [Popen background, always-on]
        └── writes: YYYYMMDD_spx_price_action_signal.json

  └─► build_intraday_training_dataset.py  [Popen background, t+15min]
        └── writes: YYYYMMDD_intraday_bars_1m.csv
                    YYYYMMDD_intraday_training_dataset.csv
                    YYYYMMDD_exit_training_dataset.csv
                    YYYYMMDD_multitimeframe_context.json
                    YYYYMMDD_Feature_stream/

  └─► stream_confirmed_movement_gate_trade_plan_trailing.py  [Popen background, config-gated]
        └── writes: YYYYMMDD_stream_confirmed_rebalance_orders.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST-MARKET  [16:05 – 20:00 ET, optional — disabled by default]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  └─► scripts/post_market_retrain_pipeline.py  [Popen background, opt-in]
        Trigger: POST_MARKET_RETRAIN_ENABLED=true, once per day after 16:05 ET
        Step 1: build_intraday_training_dataset.py → bars + feature + exit datasets
        Step 2: forecast_engine.py → updated ensemble model
        Step 3: rl_intraday_structure_trainer.py → updated PPO model
        └── writes: daily_system_outputs/post_market_retrain_summary.json
                    logs/execution/post_market_retrain_{DATE}.log

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTRADAY LOOP  [9:30 AM – 4:00 PM ET, repeated]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  ├─► rl_intraday_structure_trainer.py  [subprocess blocking, validate/train]
  │     reads: YYYYMMDD_intraday_bars_1m.csv
  │     writes: YYYYMMDD_rl_intraday_structure_model.zip
  │
  ├─► forecast_engine.py  [subprocess blocking, refresh cycle]
  │     reads: YYYYMMDD_intraday_training_dataset.csv
  │     writes: YYYYMMDD_forecast_ensemble_signals.json
  │
  ├─► options_booster.py  [subprocess, config-gated]
  │     writes: YYYYMMDD_latest_option_booster_candidates.csv
  │
  ├─► portfolio_rebalancer.py  [subprocess, config-gated]
  │     writes: YYYYMMDD_daily_walkforward_emulated_rebalance_orders.csv
  │
  ├── execution_alpha_gate.py  [direct import]
  │     add_execution_alpha_scores() — scores candidate orders inline
  │
  ├── system_self_awareness.py  [direct import, optional]
  │     size multiplier + execution mode recommendation
  │
  ├── PASSIVE reads (scripts not invoked by run.py):
  │     deeplob_orderbook_direction_layer.py → reads YYYYMMDD_deeplob_live_signals.json
  │
  └── INLINE layers (code embedded in run.py):
        Evolution Strategy (buy/hold/sell timing)
        PSO-LSTM hyperparameter optimizer
        DWT price series denoising
        Built-in momentum/reversal scanner (ScannerConfig)

  ──► Alpaca TradingClient  [paper execution]
        journal → logs/execution/YYYYMMDD/
        equity curve, experience replay, JSONL events
```

---

## Execution Alpha Threshold

The execution alpha gate uses a per-order score to decide whether a buy is strong enough
to execute. The threshold resolution order is:

1. `EXECUTION_ALPHA_BUY_THRESHOLD` env var (if set, overrides everything)
2. `execution_alpha_min_score` in the strategy's config YAML entry
3. Hardcoded default: `0.70`

For stream-confirmed pivot setups specifically:
- `EXECUTION_ALPHA_BUY_THRESHOLD_STREAM_CONFIRMED_PIVOT` env var (if set)

Both env vars default to empty — existing behavior is fully preserved unless explicitly set.
Each blocked order logs `threshold_source` and `stream_confirmed_pivot` fields to
`execution_alpha_buy_blocked` events for analysis.

---

## Module Responsibilities

| Module | Role | Invoked By | Integration |
|--------|------|-----------|-------------|
| `run.py` | Top-level orchestrator, execution engine | systemd / CLI | — (entrypoint) |
| `premarket_data_collector.py` | Premarket pipeline runner | `run.py` subprocess | Blocking |
| `llm_sentiment_universe_selector.py` | Universe selection + LLM ranking | collector subprocess | Blocking |
| `market_context_builder.py` | Regime + broad market context | collector subprocess | Blocking |
| `premarket_trade_planner.py` | Pivot structure trade planning | collector subprocess | Blocking |
| `build_intraday_training_dataset.py` | 1m bars + feature + label dataset | `run.py` Popen (background) | Non-blocking |
| `forecast_engine.py` | ML forecast ensemble (LSTM, ARIMA, attention) | `run.py` subprocess | Blocking |
| `rl_intraday_structure_trainer.py` | RL PPO timing model | `run.py` subprocess | Blocking |
| `spx_price_action_layer.py` | SPX market structure gate | `run.py` Popen (background) | Non-blocking |
| `options_booster.py` | Long-call overlay candidates | `run.py` subprocess | Config-gated |
| `portfolio_rebalancer.py` | Position rebalancing orders | `run.py` subprocess | Config-gated |
| `execution_alpha_gate.py` | Execution alpha scoring | `run.py` direct import | Always-on |
| `system_self_awareness.py` | System introspection, size multiplier | `run.py` direct import | Optional |
| `deeplob_orderbook_direction_layer.py` | Orderbook direction probability | File-based passive | Not invoked |
| `production_regime_sizing_model.py` | Regime-aware position sizing | `run.py` subprocess | On regime refresh |
| `tradingview_institutional_screener.py` | Universe discovery (TV screener) | Inside planner | Indirect |
| `webhook.py` | TradingView webhook receiver | Standalone (separate process) | Not in run.py |
| `scripts/evaluate_variants.py` | Composite strategy scoring | Standalone | **Gap — not hooked** |
| `scripts/walkforward_strategy.py` | Walk-forward OOS validation | Standalone | **Gap — not hooked** |
| `scripts/diagnose_strategy.py` | Strategy diagnostics report | Standalone | **Gap — partial** |
| `scripts/claude_self_optimization_agent.py` | Self-optimization requests | Standalone | **Gap — not hooked** |
| `scripts/adaptive_weight_optimizer.py` | Adaptive weight updates | Standalone | **Gap — not hooked** |
| `portfolio_optimizer.py` | HRP/allocation optimization | Standalone | **Gap — not hooked** |
| `scanner.py` | Market scanner (legacy) | Standalone | run.py has inline scanner |
| `sr_rl_agent.py` | Support/resistance RL agent | Standalone | Not in run.py |
| `rl_spx_breakout_trainer.py` | SPX RL trainer (variant) | Standalone | Not in run.py |
| `tune_*.py` | Hyperparameter tuning scripts | Standalone | Offline only |
| `visualize_outputs.py` | Output visualization | Standalone | Offline only |

---

## Orchestration Boundaries

### What run.py MUST own (never bypass)

- All Alpaca order submissions
- All risk control evaluations (cash buffer, exposure limits, daily loss gates)
- All kill-switch checks
- Session lifecycle (startup, premarket, intraday loop, shutdown)
- Safety gate logic (SAFE mode triggers, stale plan detection)

### What run.py orchestrates via subprocess

Use subprocess (blocking) for anything that:
- Is session-critical (execution must wait for the result)
- Produces a file that run.py reads in the same cycle
- Must complete before the next stage begins

Use Popen (background) for anything that:
- Can run in parallel without blocking execution
- Produces output that will be read in the next cycle
- Has its own long-running internal loop

### What MUST remain standalone-runnable

These scripts must work when invoked directly, without run.py:

- All training scripts (`rl_intraday_structure_trainer.py`, `forecast_engine.py`, `build_intraday_training_dataset.py`)
- All evaluation scripts (`scripts/evaluate_variants.py`, `scripts/walkforward_strategy.py`)
- All tuning scripts (`tune_*.py`)
- `webhook.py`

**Reason:** Offline training, development, debugging, and research all require direct invocation. Coupling these scripts to run.py internals would break offline workflows.

### What run.py must NEVER do

- Import from scripts it invokes as subprocess (creates circular dependency risk)
- Auto-promote strategy variants to live without explicit human approval
- Suppress or block sell orders for any reason
- Skip the kill switch check
- Submit live (non-paper) orders unless explicitly configured

---

## Allowed Standalone Workflows

These workflows bypass run.py intentionally and are safe:

```bash
# Offline RL training
python rl_intraday_structure_trainer.py

# Offline ML forecast training
python forecast_engine.py

# Build training dataset manually
python build_intraday_training_dataset.py

# Strategy evaluation
python scripts/evaluate_variants.py

# Walk-forward validation
python scripts/walkforward_strategy.py

# Diagnostics
python scripts/diagnose_strategy.py

# Hyperparameter tuning
python tune_rl_intraday.py
python tune_spx_multibar_rl.py

# TradingView webhook (separate process)
python webhook.py

# Visualization
python visualize_outputs.py
```

---

## v0.2 Planned Orchestration Hooks

The following are NOT yet integrated into run.py. v0.2 will add safe hooks:

| Stage | Script | Hook Type | Safety Gate |
|-------|--------|-----------|-------------|
| Walk-forward validation | `scripts/walkforward_strategy.py` | Popen after training | Results readable next cycle only |
| Strategy evaluation | `scripts/evaluate_variants.py` | Popen after WF | Composite score written to daily_system_outputs/ |
| Self-optimization | `scripts/claude_self_optimization_agent.py` | Scheduled (not every run) | Human approval checkpoint before weights apply |
| Portfolio optimization | `portfolio_optimizer.py` | Popen after eval | Allocation targets written to file; not auto-applied |

**Prerequisite:** v0.2 begins only after v0.1 (Production Foundation) is complete and the orchestration gap audit (Phase 5) is done.

---

## Planned: Phase 4 Execution Gates

### Time-of-Day Regime Gate

**Status:** Planned — Phase 4 (paper trading validation). Not yet implemented in `run.py`.

**Purpose:** Translate professional session discipline into autonomous execution rules. The buy-side confirmation threshold and position sizing scale with session context. Exits and stop-losses are never affected.

**Session windows (all times ET, configurable via env vars):**

| Window | Default Range | Behavior |
|--------|--------------|----------|
| Morning full-size | 09:30–11:30 | Full-size qualified entries; normal confirmation; ORB/pivot/momentum setups prioritized |
| Midday reduced | 11:30–13:00 | Reduced buy size; stronger confirmation required; optionally block low-quality new entries; exits always allowed; logs `[time-gate] midday_reduced_mode` |
| Afternoon continuation | 13:00–15:30 | Continuation entries allowed; requires trend + VWAP + regime confirmation; weak mean-reversion chop avoided; logs `[time-gate] afternoon_continuation_mode` |
| Close exit window | 15:30–16:00 | New buys blocked; exits prioritized; optional flatten-before-close; logs `[time-gate] close_exit_window` |

**Environment variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `TIME_OF_DAY_GATING_ENABLED` | `true` | Master switch — set `false` to disable all session gating |
| `MORNING_FULL_SIZE_START` | `0930` | Start of morning full-size window (HHMM, ET) |
| `MORNING_FULL_SIZE_END` | `1130` | End of morning full-size window |
| `MIDDAY_REDUCED_MODE_START` | `1130` | Start of midday reduced window |
| `MIDDAY_REDUCED_MODE_END` | `1300` | End of midday reduced window |
| `AFTERNOON_CONTINUATION_START` | `1300` | Start of afternoon continuation window |
| `AFTERNOON_CONTINUATION_END` | `1530` | End of afternoon continuation window |
| `CLOSE_EXIT_WINDOW_START` | `1530` | Start of close exit-only window |
| `CLOSE_EXIT_WINDOW_END` | `1600` | End of close exit-only window |

**Invariants (never violated by this gate):**

- Exits, stop-losses, and risk-reducing orders are never blocked by the time gate
- Existing kill switches, max-loss limits, and equity floors are not weakened
- Alpaca credentials, paper/live mode flags, and cash buffer rules are unchanged
- Buys blocked by the time gate are logged with a skip reason: `blocked_by=time_gate, session=<window_name>`
- `blocked_by_time_gate` count tracked in execution journal for session performance review

---

## Data Contracts (key files)

| File | Producer | Consumer | Format |
|------|----------|----------|--------|
| `YYYYMMDD_market_context.json` | `market_context_builder.py` | `run.py` | JSON |
| `YYYYMMDD_premarket_trade_plan.json` | `premarket_trade_planner.py` | `run.py` | JSON |
| `YYYYMMDD_intraday_structure_map.json` | `premarket_trade_planner.py` | `run.py`, `rl_intraday_structure_trainer.py` | JSON |
| `YYYYMMDD_intraday_bars_1m.csv` | `build_intraday_training_dataset.py` | `rl_intraday_structure_trainer.py` | CSV |
| `YYYYMMDD_intraday_training_dataset.csv` | `build_intraday_training_dataset.py` | `forecast_engine.py` | CSV |
| `YYYYMMDD_forecast_ensemble_signals.json` | `forecast_engine.py` | `run.py` | JSON |
| `YYYYMMDD_rl_intraday_structure_model.zip` | `rl_intraday_structure_trainer.py` | `run.py` | ZIP |
| `YYYYMMDD_regime_position_plan.csv` | `production_regime_sizing_model.py` | `run.py` | CSV |
| `YYYYMMDD_spx_price_action_signal.json` | `spx_price_action_layer.py` | `run.py` | JSON |
| `YYYYMMDD_deeplob_live_signals.json` | `deeplob_orderbook_direction_layer.py` | `run.py` (passive read) | JSON |

---

*See also:*
- `docs/ARCHITECTURE_LAYERS.md` — Layer 1/2/3 separation
- `.paul/PROJECT.md` — full project context and orchestration stage table
- `.paul/ROADMAP.md` — v0.2 Unified Orchestration milestone
- `CLAUDE.md` — Claude-specific architectural rules
