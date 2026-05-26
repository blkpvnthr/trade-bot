# run.py Orchestration Coverage Audit

> **Type:** Read-only gap analysis. No code was modified.
> **Scope:** All `.py` files in `~/Desktop/trader/` root + `scripts/` directory.
> **Date:** 2026-05-20
> **Method:** grep + targeted reads of run.py; full file NOT loaded.

---

## 1. Integrated Components

### 1a. Subprocess Blocking — run.py waits for exit before continuing

| Script | Trigger function | When called | Config gate |
|--------|-----------------|-------------|-------------|
| `forecast_engine.py` | `refresh_forecast_ensemble_if_available()` | Intraday refresh cycle | `FORECAST_ENSEMBLE_SCRIPT` env var |
| `rl_intraday_structure_trainer.py` | `refresh_rl_intraday_structure_model()` | Intraday refresh cycle | Always attempted |
| `portfolio_rebalancer.py` | Direct subprocess.run call | Intraday cycle (regime step) | `run_production_regime_sizing_model` |
| `llm_sentiment_universe_selector.py` | try-import, fallback subprocess | Pre-session refresh chain | Part of premarket_data_collector |
| `market_context_builder.py` | try-import, fallback subprocess | Pre-session refresh chain | Part of premarket_data_collector |
| `premarket_trade_planner.py` | try-import, fallback subprocess | Pre-session refresh chain | Part of premarket_data_collector |
| `production_regime_sizing_model.py` | `refresh_production_regime_sizing_model()` | Intraday, after collector | `run_production_regime_sizing_model` |

### 1b. Subprocess Background (Popen) — run.py does not wait for exit

| Script | Launch site | When launched | Config gate |
|--------|-------------|--------------|-------------|
| `premarket_data_collector.py` | `run_premarket_data_collector()` | Pre-session 4:00–9:30 AM | `premarket_data_collector_enabled` |
| `spx_price_action_layer.py` | `ensure_spx_price_action_layer_module()` | At/after market open | `SPX_LAYER_ENABLED` + `SPX_LAYER_INTERNAL_SUBPROCESS` |
| `build_intraday_training_dataset.py` | Scheduler Popen | ~15 min after open | Always (scheduler) |
| `options_booster.py` | Direct Popen | Intraday cycle | `OPTIONS_BOOSTER_ENABLED` |
| `sr_rl_agent.py` | SR RL class (train + live loop) | Market open + ongoing | `SR_RL_ENABLED` config flag |
| `stream_confirmed_movement_gate_trade_plan_trailing.py` | `stream_confirmed_movement_script` config | Market open + ongoing | `STREAM_CONFIRMED_MOVEMENT_ENABLED` |

✓ Resolved (02-02b): script path references corrected to `stream_confirmed_movement_gate_trade_plan_trailing.py`. `[subprocess-check]` diagnostic added.

### 1c. Direct Imports — module-level or try/except at startup

| Module | Symbols imported | Always-on? |
|--------|-----------------|------------|
| `execution_alpha_gate` | `add_execution_alpha_scores` | Yes — top-level import (line 89) |
| `system_self_awareness` | `load_self_assessment`, `self_assessment_size_multiplier` | No — try/except optional (lines 92–108) |
| `config.env_loader` | `load_instance_env` | Yes — top-level import (line 87) |

### 1d. Inline Implementations — logic embedded in run.py itself

| Component | Implementation | Notes |
|-----------|---------------|-------|
| PSO-LSTM optimizer | `pso_position_to_lstm_params()` + related functions | Hyperparameter optimization for LSTM |
| Evolution Strategy | `EvolutionStrategyOptimizer`, `EvolutionStrategyAgent`, `EvolutionStrategyModel` | Buy/hold/sell timing model |
| DWT denoising | `_dwt_denoise_series()` | Optional pywt dependency; denoises price series for forecasting |
| Built-in scanner | `ScannerConfig`, inline signal scoring | Replaces legacy `scanner.py` |
| AlpacaBroker engine | `AlpacaBroker` class | Paper/live execution; credentials loaded per-instance |

---

## 2. Standalone and Unhooked Components

### Root-level scripts — NOT invoked by run.py

| Script | Category | Reason not in run.py | Should it be? |
|--------|----------|---------------------|---------------|
| `alpaca_executor.py` | STANDALONE | Manual executor for walkforward emulated orders | No — design: offline batch execution |
| `deeplob_orderbook_direction_layer.py` | PASSIVE | run.py reads `*_deeplob_live_signals.json` but never invokes the script | Gap — signals always stale/missing |
| `features.py` | STANDALONE | Legacy batch feature engineering | No — superseded by build_intraday_training_dataset.py |
| `model.py` | STANDALONE | Legacy model training | No — superseded by forecast_engine.py + rl_intraday_structure_trainer.py |
| `portfolio_optimizer.py` | STANDALONE | HRP/allocation optimization | Partial gap — portfolio_rebalancer.py is hooked; optimizer is not |
| `rl_spx_breakout_trainer.py` | STANDALONE | Offline SPX RL trainer variant | No — offline training only |
| `rl_spx_breakout_trainer_long_only_addlong.py` | STANDALONE | Offline SPX RL trainer variant | No — offline training only |
| `scanner.py` | STANDALONE | Legacy; run.py has inline scanner | No — intentionally superseded |
| `stream_confirmed_movement_gate_trade_plan_trailing.py` | INTEGRATED (Popen background) | ✓ Resolved (02-02b): run.py now references the correct filename; launched at market open, config-gated | None |
| `trade.py` | STANDALONE | Large standalone executor (~12K lines); not imported by run.py | Unknown — may be legacy parallel |
| `tradingview_institutional_screener.py` | INDIRECT | Called inside llm_sentiment_universe_selector.py / premarket_trade_planner.py | By design — correct |
| `tune_rl_intraday.py` | STANDALONE | Offline RL hyperparameter tuning | No — offline only |
| `tune_spx_multibar_rl.py` | STANDALONE | Offline SPX RL tuning | No — offline only |
| `tune_sr_rl_agent.py` | STANDALONE | Offline SR RL tuning | No — offline only |
| `visualize_outputs.py` | STANDALONE | Visualization | No — offline only |
| `walkforward.py` | STANDALONE | Legacy walkforward | No — superseded by scripts/walkforward_strategy.py |
| `webhook.py` | STANDALONE | Separate Flask process | By design — intentionally separate process |

### scripts/ directory — NOT invoked by run.py

| Script | Category | Gap priority |
|--------|----------|--------------|
| `scripts/walkforward_strategy.py` | STANDALONE | **High — v0.2 Phase 6** |
| `scripts/evaluate_variants.py` | STANDALONE | **High — v0.2 Phase 6** |
| `scripts/diagnose_strategy.py` | STANDALONE | Medium — partial coverage via system_self_awareness |
| `scripts/capital_allocator.py` | STANDALONE | Low — analysis tool |
| `scripts/compare_strategy_overlap.py` | STANDALONE | Low — analysis tool |
| `scripts/generate_claude_patch_request.py` | STANDALONE | Low — offline only |
| `scripts/generate_dashboard.py` | STANDALONE | Low — offline only |
| `scripts/promote_variant.py` | STANDALONE | Low — manual gate intentional |
| `scripts/replay_market_day.py` | STANDALONE | Low — dev tool |
| `scripts/smoke_test_training_pipeline.py` | STANDALONE | Low — CI/dev tool |
| `scripts/smoke_test_variants.py` | STANDALONE | Low — CI/dev tool |
| `scripts/strategy_activity_report.py` | STANDALONE | Low — reporting |
| `scripts/strategy_health_monitor.py` | STANDALONE | Low — monitoring |

**Note:** `scripts/claude_self_optimization_agent.py` and `scripts/adaptive_weight_optimizer.py`
are referenced in ROADMAP.md Phase 7 but do **not yet exist** in the project.

---

## 3. Pipeline DAG (current v0.1)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-SESSION  [4:00 AM – 9:30 AM ET]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  └─► premarket_data_collector.py          [Popen + poll stdout; blocking until done]
        ├─► llm_sentiment_universe_selector.py   [try-import, fallback subprocess]
        │     └─► tradingview_institutional_screener.py  [indirect — called inside]
        │     └── writes: *_locked_trading_universe.txt
        │                 *_latest_llm_sentiment_universe.json
        ├─► market_context_builder.py            [try-import, fallback subprocess]
        │     └── writes: *_market_context.json
        └─► premarket_trade_planner.py           [try-import, fallback subprocess]
              └── writes: *_premarket_trade_plan.json
                          *_intraday_structure_map.json

  └─► production_regime_sizing_model.py   [try-import, fallback subprocess; blocking]
        └── writes: *_regime_position_plan.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET OPEN  [9:30 AM ET]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  └─► spx_price_action_layer.py           [Popen background; config-gated]
        └── writes: *_spx_price_action_signal.json

  └─► build_intraday_training_dataset.py  [Popen background; ~t+15min]
        └── writes: *_intraday_bars_1m.csv
                    *_intraday_training_dataset.csv
                    *_exit_training_dataset.csv
                    *_Feature_stream/

  └─► sr_rl_agent.py                      [Popen background; config-gated, train + live loop]
        └── writes: *_sr_rl_live_signals.json

  └─► stream_confirmed_movement_gate.py   [Popen background; config-gated]  ⚠️ FILE MISSING
        └── [expected to write *_stream_confirmed_rebalance_orders.csv]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTRADAY LOOP  [9:30 AM – 4:00 PM ET, repeated]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
run.py
  ├─► rl_intraday_structure_trainer.py    [subprocess blocking; intraday refresh]
  │     reads: *_intraday_bars_1m.csv
  │     writes: *_rl_intraday_structure_model.zip
  │
  ├─► forecast_engine.py                  [subprocess blocking; intraday refresh]
  │     reads: *_intraday_training_dataset.csv
  │     writes: *_forecast_ensemble_signals.json
  │
  ├─► options_booster.py                  [Popen background; config-gated]
  │     writes: *_latest_option_booster_candidates.csv
  │
  ├─► portfolio_rebalancer.py             [subprocess blocking; config-gated]
  │     writes: *_daily_walkforward_emulated_rebalance_orders.csv
  │
  ├── execution_alpha_gate.py             [direct import; always-on]
  │     add_execution_alpha_scores() — scores candidates inline
  │
  ├── system_self_awareness.py            [try/except import; optional]
  │     load_self_assessment() + size_multiplier → applied to buy sizing
  │
  ├── PASSIVE reads (scripts not invoked):
  │     deeplob_orderbook_direction_layer.py → *_deeplob_live_signals.json  [stale risk]
  │     stream_confirmed_movement_gate_trade_plan_trailing.py → *_stream_confirmed_rebalance_orders.csv
  │     sr_rl_agent.py → *_sr_rl_live_signals.json  [also invoked, so hybrid]
  │
  ├── INLINE layers (code embedded in run.py):
  │     Evolution Strategy (EvolutionStrategyAgent) — buy/hold/sell timing
  │     PSO-LSTM (pso_position_to_lstm_params) — hyperparameter optimization
  │     DWT denoising (_dwt_denoise_series) — price series preprocessing
  │     Built-in momentum/reversal scanner (ScannerConfig) — signal scoring
  │     AlpacaBroker — order submission engine
  │
  └──► Alpaca TradingClient  [paper execution]
        journal → logs/execution/YYYYMMDD/
        equity curve, experience replay, JSONL events

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOT IN DAG (standalone / offline only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scripts/walkforward_strategy.py           [offline; v0.2 gap]
scripts/evaluate_variants.py              [offline; v0.2 gap]
scripts/diagnose_strategy.py              [offline; partial gap]
portfolio_optimizer.py                    [offline; v0.2 consideration]
deeplob_orderbook_direction_layer.py      [passive only; never launched]
webhook.py                                [separate process; intentional]
```

---

## 4. Gap Analysis

### Critical Gaps (affect live execution correctness)

| Gap | Impact | File(s) | Priority |
|-----|--------|---------|----------|
| `stream_confirmed_movement_gate.py` references corrected | ✓ Resolved (Plan 02-02b): `STREAM_CONFIRM_SCRIPT_PATH`, `ScannerConfig.stream_confirmed_movement_script` default, and else-fallback string all updated to `stream_confirmed_movement_gate_trade_plan_trailing.py`. `[subprocess-check]` diagnostic added. Note: a silent alt-fallback already existed in `_stream_confirmed_movement_loop` — execution did not hard-fail, but path resolution was unlogged before this fix. | `run.py` (resolved) | None |
| `deeplob_orderbook_direction_layer.py` never invoked | `deeplob_score` always defaults to 0.50 in forecast ensemble; DeepLOB signals are structurally stale | `run.py` line 570 | High — degrades signal quality silently |

### Orchestration Gaps (v0.2 work)

| Gap | Impact | Script | ROADMAP target |
|-----|--------|--------|---------------|
| Walk-forward validation not triggered | No OOS performance check after model training; models may degrade without detection | `scripts/walkforward_strategy.py` | Phase 6 (06-01) |
| Strategy evaluation not triggered | No composite scoring or variant ranking after walk-forward; cannot detect which model variant is best | `scripts/evaluate_variants.py` | Phase 6 (06-02) |
| Self-optimization agent not created | No automated weight adaptation loop | `scripts/claude_self_optimization_agent.py` (doesn't exist yet) | Phase 7 (07-01) |
| Adaptive weight optimizer not created | No mechanism to apply proposed weight changes | `scripts/adaptive_weight_optimizer.py` (doesn't exist yet) | Phase 7 (07-02) |
| `portfolio_optimizer.py` not hooked | HRP allocation optimization not triggered by execution cycle | `portfolio_optimizer.py` | v0.2 consideration |

### Documentation Gaps

| Gap | Notes |
|-----|-------|
| `trade.py` role is unclear | ~12K lines, has AlpacaBroker, not imported by run.py; may be a legacy parallel implementation or predecessor to run.py |
| `tradingview_institutional_screener.py` called indirectly | Called inside planner/selector, not directly from run.py — this is by design, but not explicitly documented |
| PSO-LSTM, Evolution Strategy, DWT are inline | No standalone testability; buried in a 12K+ line file; v0.2 extraction candidates |

---

## 5. Recommended v0.2 Additions

These are documentation-only recommendations. No implementation in this plan.

### Immediate fix (before v0.2)

1. **Rename `stream_confirmed_movement_gate_trade_plan_trailing.py` OR update `STREAM_CONFIRM_SCRIPT_PATH`** in run.py to reference the correct filename. Current state causes silent subprocess failure when `STREAM_CONFIRMED_MOVEMENT_ENABLED=1`.

### v0.2 Phase 5 (Orchestration Audit — already in progress)

2. **Establish baseline** of the exact DAG above as the v0.1 canonical reference. This audit serves as that baseline.

### v0.2 Phase 6 (Walk-Forward & Eval Hooks)

3. **`scripts/walkforward_strategy.py`** — add Popen hook after RL/forecast training completes; results written to `daily_system_outputs/` and readable next cycle. Preserve standalone invocability.

4. **`scripts/evaluate_variants.py`** — add Popen hook after walk-forward; composite score output written to `daily_system_outputs/`. Read next cycle to gate model promotion.

### v0.2 Phase 7 (Self-Optimization)

5. **`scripts/claude_self_optimization_agent.py`** (create) — triggered on configurable schedule; proposes weight updates to `configs/strategy_experiments.yaml`; human approval required before apply.

6. **`scripts/adaptive_weight_optimizer.py`** (create) — apply approved weight proposals with full audit trail.

### Other considerations

7. **`deeplob_orderbook_direction_layer.py`** — evaluate whether to launch it as a background Popen at market open (like sr_rl_agent.py) or remove the passive read from the forecast ensemble to eliminate the silent 0.50 default.

8. **PSO-LSTM / Evolution Strategy / DWT extraction** — consider extracting from run.py into standalone modules for testability. Not a v0.2 blocker but reduces monolith size.

---

*Audit complete. All findings are read-only observations. No files were modified.*
*See also: `docs/architecture.md` — Pipeline DAG and Module Responsibilities table.*
