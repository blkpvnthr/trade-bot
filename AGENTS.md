# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working in this repository.

## Project Overview

This repository contains a regime-aware, multi-layer algorithmic trading system designed for a small Alpaca account, initially targeting roughly a $13K paper account but also supporting larger account bands through dynamic risk controls.

The system combines:

- Alpaca-based market data and execution
- TradingView / yfinance / Alpaca universe discovery
- Premarket trade planning
- Multi-timeframe support and resistance analysis
- Market context and regime classification
- Forecasting models
- Reinforcement learning timing gates
- Options booster logic
- Portfolio rebalancing
- Execution journaling and replay logging

The system should behave like an institutional risk-managed execution engine, not a high-frequency gambling bot.

Primary objectives:

1. Preserve capital.
2. Avoid catastrophic drawdowns.
3. Trade only high-quality opportunities.
4. Capture asymmetric momentum and reversal opportunities.
5. Continuously improve execution quality through logged experience and model feedback.

---

## Orchestration Architecture Rule

`run.py` is the canonical top-level orchestrator and sole runtime entrypoint for the production trading system.

Other Python files are subsystem modules. They may be invoked standalone for development and testing, but production and research workflows must be coordinated through `run.py`.

**`run.py` owns:**
- Session lifecycle (startup, premarket window, intraday loop, shutdown)
- Sequencing of all pipeline stages (data → features → models → signals → execution)
- All safety gates, kill switches, and risk controls
- Diagnostics, journaling, and runtime state

**Architectural invariants:**
- No subsystem file should import `run.py` — imports flow one direction only
- Do not create parallel orchestrators (no second entry points that bypass run.py's safety gates)
- Circular imports involving `run.py` are forbidden
- New production orchestration hooks go into `run.py` only

**Standalone-runnable scripts** (acceptable outside run.py for dev/research):
- Training scripts: `rl_intraday_structure_trainer.py`, `forecast_engine.py`, `build_intraday_training_dataset.py`
- Evaluation scripts: `scripts/evaluate_variants.py`, `scripts/walkforward_strategy.py`
- Tuning scripts: `tune_*.py`, `scripts/diagnose_strategy.py`
- Webhook server: `webhook.py`
- Visualization: `visualize_outputs.py`

These may be invoked directly for offline work but should not bypass `run.py`'s safety gates in any production flow.

---

## Core Philosophy

The trading engine prioritizes survival and disciplined compounding over trade frequency.

Codex should preserve these principles when editing code:

- Always allow exits.
- Never block sell orders because of bearish gating, stale plans, kill switches, or failed planners.
- New entries must be gated by market context, structure, liquidity, confidence, and risk budget.
- Avoid overtrading and re-entry churn.
- Prefer fewer high-conviction trades over broad low-quality coverage.
- Reduce exposure when market context is bearish, choppy, or risk-off.
- Pyramid only confirmed winners.
- Scale out of weak positions.
- Keep logging and diagnostics transparent.

---

## Critical Constraints

These are architectural invariants. Do not break them.

- Always allow exits, even in safe mode or kill-switch states.
- Never suppress sell orders.
- The equity system is long-only by default.
- Use inverse ETFs for bearish exposure instead of shorting equities unless explicitly changed.
- Options booster should use long calls only unless explicitly expanded.
- Premarket plans older than the configured threshold are stale.
- Stale or missing planner data should trigger SAFE mode, not uncontrolled execution.
- Missing market context should reduce risk, not increase it.
- Missing structure maps should block new buys but still allow exits.
- Duplicate open orders must be avoided.
- Existing Alpaca positions must be checked before submitting new orders.
- Cash buffer, max gross exposure, max position weight, and daily loss limits must be enforced.
- All live or paper orders must be journaled.
- File outputs must remain dated under `system-data/YYYYMMDD/` or `logs/execution/YYYYMMDD/`.

---

## Running the System

### Dependencies

Install the main project dependencies:

```bash
pip install -r file.txt
```

Optional packages:

```bash
pip install sb3-contrib
pip install tradingview-screener
pip install rookiepy
```

`tradingview-screener` is used for TradingView scanner/ranking support.

`rookiepy` can optionally load local browser cookies for live TradingView exchange data when the user is logged into TradingView in a supported browser.

---

## Environment

The `.env` file should live at the project root.

Required values:

```env
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret_key
APCA_PAPER=true
OPENAI_API_KEY=your_openai_key
TRADINGVIEW_WEBHOOK_SECRET=your_secret
```

Common optional values:

```env
RUN_DATA_DATE=YYYYMMDD
USE_TRADINGVIEW_SCREENER=1
TRADINGVIEW_USE_BROWSER_COOKIES=0
TRADINGVIEW_REQUIRE_RVOL=1.2
TRADINGVIEW_MAX_PLANNER_SYMBOLS=20

SAFE_MODE_ON_PLANNER_FAILURE=1
MAX_PLAN_AGE_MINUTES=15
BEARISH_MARKET_DISABLE_NEW_LONGS=1

PREMARKET_WAIT_FOR_MARKET_OPEN=0
PREMARKET_DATA_COLLECTOR_ENABLED=1
OPTIONS_BOOSTER_ENABLED=1
RL_SHADOW_MODE=1
```

Important:

- `APCA_PAPER=true` should be used unless the user explicitly intends live trading.
- `PREMARKET_WAIT_FOR_MARKET_OPEN=0` should generally be used inside collector workflows so the planner does not block and timeout.
- `RUN_DATA_DATE` can be used to replay or inspect a previous session.

---

## Manual Pipeline

The legacy manual workflow is:

```bash
python scanner.py
python features.py
python model.py
python walkforward.py
python visualize_outputs.py
python run.py
```

The current preferred workflow is:

```bash
python run.py
```

`run.py` acts as the all-in-one orchestrator.

---

## Production Commands

Systemd deployment examples:

```bash
systemctl start trade-bot
systemctl restart trade-bot
systemctl status trade-bot
journalctl -u trade-bot -f
tail -f /root/trade-bot/logs/run.log
```

TradingView webhook server runs separately:

```bash
python webhook.py
```

If using ngrok:

```bash
ngrok http 5000
```

---

## High-Level Architecture

```text
Universe Discovery
    ↓
Market Context Builder
    ↓
Premarket Data Collector
    ↓
Premarket Trade Planner
    ↓
Intraday Dataset Builder
    ↓
Forecast / ML / RL Signal Layers
    ↓
Execution Alpha Gate
    ↓
Portfolio Rebalancer
    ↓
Alpaca Execution Engine
    ↓
Journal + Equity Curve + Experience Replay
```

---

## Pre-Session Pipeline: 4:00–9:30 AM ET

`premarket_data_collector.py` runs as a subprocess loop. It refreshes the core planning files during the premarket window.

### 1. Universe Selection

Script:

```text
llm_sentiment_universe_selector.py
```

Purpose:

- Reads `data/bullish_YYYY-MM-DD.csv`.
- Uses market data and optional LLM logic to rank/select tradable symbols.
- Writes the locked universe.

Outputs:

```text
YYYYMMDD_locked_trading_universe.txt
system-data/YYYYMMDD/YYYYMMDD_latest_llm_sentiment_universe.json
system-data/YYYYMMDD/YYYYMMDD_latest_llm_candidate_discovery.json
```

### 2. Market Context Builder

Script:

```text
market_context_builder.py
```

Purpose:

- Fetches SPY, QQQ, DIA, VIXY/VXX, TLT, UUP, and sector ETF context.
- Classifies broad market state.
- Detects risk-on/risk-off behavior.
- Writes market context used by execution gates.

Output:

```text
daily_system_outputs/YYYYMMDD/YYYYMMDD_market_context.json
```

Important field examples:

```json
{
  "overall_market_bias": "bearish",
  "trade_permission": "cautious",
  "risk_level": "medium",
  "sector_context_summary": {
    "sector_rotation_bias": "defensive"
  },
  "key_etf_context_summary": {
    "risk_signal_bias": "risk_off"
  }
}
```

### 3. Premarket Trade Planner

Script:

```text
premarket_trade_planner.py
```

Purpose:

- Uses TradingView Screener, yfinance, and local universe files to rank symbols.
- Builds premarket breakout/consolidation plans.
- Maps support/resistance.
- Builds VWAP and ORB context.
- Produces structure gates for `run.py`.

Outputs:

```text
system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan.json
system-data/YYYYMMDD/YYYYMMDD_intraday_structure_map.json
system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan_diagnostics.json
```

Important behavior:

- Should complete quickly.
- Should not wait until market open when called by the collector.
- Should cap the number of symbols before heavy processing.
- Should produce safe fallback outputs even when no valid trades exist.

### 4. Intraday Dataset Builder

Script:

```text
build_intraday_training_dataset.py
```

Purpose:

- Downloads Alpaca 1-minute bars for the selected universe.
- Builds intraday feature streams.
- Builds training and exit datasets.
- Updates multi-timeframe context.

Outputs:

```text
system-data/YYYYMMDD/YYYYMMDD_intraday_bars_1m.csv
system-data/YYYYMMDD/YYYYMMDD_intraday_training_dataset.csv
system-data/YYYYMMDD/YYYYMMDD_exit_training_dataset.csv
system-data/YYYYMMDD/YYYYMMDD_multitimeframe_context.json
system-data/YYYYMMDD/YYYYMMDD_Feature_stream/YYYYMMDD_symbol_state.json
```

### 5. Options Booster

Script:

```text
options_booster.py
```

Purpose:

- Scans option chains for positive-EV call opportunities.
- Long calls only by default.
- Should require strong underlying confirmation.
- Should require an existing open stock position unless explicitly configured otherwise.

Output examples:

```text
system-data/YYYYMMDD/YYYYMMDD_latest_option_booster_summary.json
system-data/YYYYMMDD/YYYYMMDD_daily_walkforward_emulated_rebalance_orders.csv
system-data/YYYYMMDD/YYYYMMDD_daily_walkforward_emulated_exit_orders.csv
```

Known issue to guard against:

- Missing `underlying_price` in options rows.
- Required fallback order:
  1. Row-level underlying price.
  2. Latest stock trade.
  3. Latest stock quote midpoint.
  4. Latest stock bar close.
  5. Intraday dataset close.

---

## Session Execution: run.py

`run.py` is the monolithic production orchestrator.

It performs startup initialization, launches supporting processes, loads signals, applies gates, builds orders, and executes via Alpaca.

Core startup behavior:

1. Load `.env`.
2. Initialize dated paths.
3. Optionally run PSO-LSTM optimization.
4. Refresh forecast ensemble.
5. Train or refresh the Evolution Strategy layer.
6. Start `spx_price_action_layer.py` as a background subprocess.
7. Schedule or launch intraday dataset generation.
8. Start `ProductionOrderExecutor.run()`.

Execution loop responsibilities:

- Run the premarket collector between 4:00 and 9:30 AM ET.
- Load market context.
- Load premarket trade plan.
- Load intraday structure map.
- Validate planner health and freshness.
- Apply SAFE mode when planner or context is stale/missing.
- Load or generate regime sizing plan.
- Build candidate order plan.
- Score candidates through `execution_alpha_gate.py`.
- Apply RL timing gate.
- Apply forecast ensemble gate.
- Apply structure-map gate.
- Sell first.
- Buy second.
- Enforce cash and exposure limits.
- Avoid duplicate orders.
- Journal all actions.

---

## Entry Logic

A new long entry should generally require:

1. Market context is bullish, mixed, or has a high-confidence override.
2. Trade permission is not disabled.
3. Premarket plan is fresh.
4. Symbol appears in the structure map.
5. Symbol is allowed by `long_allowed_symbols` or equivalent structure logic.
6. Liquidity profile is acceptable.
7. Forecast ensemble is supportive.
8. RL timing gate does not reject the entry.
9. Position sizing passes cash buffer and exposure rules.
10. No duplicate open order exists.

Bearish override:

- If `BEARISH_MARKET_DISABLE_NEW_LONGS=1`, new longs should be blocked during bearish regimes unless a strong override condition exists.
- Strong override should require unusually high confidence, strong structure, and risk budget.

---

## Time-of-Day Regime Gating

**Status:** Planned — Phase 4 (paper trading validation). Not yet implemented.

When implemented, the time-of-day gate introduces session-aware buy-side discipline in `run.py`. All gating is configurable via env vars. The gate applies only to new buy entries — never to exits, stop-losses, or risk-reducing orders.

**Session windows (ET, all boundaries configurable):**

| Session | Default | Buy-side rule |
|---------|---------|--------------|
| Morning full-size | 09:30–11:30 | Full size, normal confirmation, ORB/pivot/momentum prioritized |
| Midday reduced | 11:30–13:00 | Reduced size, stronger confirmation, optionally block low-quality entries |
| Afternoon continuation | 13:00–15:30 | Continuation entries allowed, requires trend + VWAP + regime confirmation |
| Close exit window | 15:30–16:00 | New buys blocked, exits prioritized |

**Key env vars:** `TIME_OF_DAY_GATING_ENABLED`, `MORNING_FULL_SIZE_START/END`, `MIDDAY_REDUCED_MODE_START/END`, `AFTERNOON_CONTINUATION_START/END`, `CLOSE_EXIT_WINDOW_START/END`.

**Behavioral invariants Codex must preserve when implementing this feature:**

- Never block exits, stop-losses, or risk-reducing orders via the time gate.
- Never weaken kill switches, equity floors, or max-loss limits.
- Never alter Alpaca credentials or paper/live mode flags.
- All skipped buy signals must be logged: `[time-gate] <session_name>`, with `blocked_by=time_gate` in the skip reason.
- `blocked_by_time_gate` count must be tracked in the execution journal.
- If `TIME_OF_DAY_GATING_ENABLED=false`, the gate is fully bypassed — no behavior change.
- Default to conservative session boundaries; do not widen windows without explicit instruction.

---

## Exit Logic

Exits should always remain available.

Exit conditions may include:

- RL exit signal.
- Structure breakdown.
- Failed VWAP reclaim.
- Support loss.
- Momentum divergence.
- SPY/QQQ weakness.
- Volatility expansion.
- Drawdown threshold.
- Take-profit or trailing-profit rule.
- Regime deterioration.
- Manual or forced risk reduction.

Exits must not be blocked by:

- Bearish market gates.
- Planner failure.
- Stale planner.
- Missing market context.
- Kill-switch state.
- Lack of buy permission.

---

## Safe Mode

SAFE mode is a risk-degraded execution state.

Activate SAFE mode when:

- Premarket plan is missing.
- Premarket plan is stale.
- Structure map is missing.
- Market context is unavailable.
- Market context is bearish/cautious.
- Planner status is not healthy.
- Data validation fails.

SAFE mode behavior:

- Allow exits.
- Suppress or reduce new buys.
- Disable options booster.
- Disable pyramiding.
- Reduce max gross exposure.
- Reduce max order dollars.
- Increase confidence thresholds.
- Limit number of concurrent positions.
- Prefer cash preservation.

---

## Signal Layers

| Layer | Location | Purpose |
|---|---|---|
| Forecast ensemble | inline in `run.py` | ARIMA, LSTM, stacked LSTM, attention LSTM, DeepLOB, RL blend |
| PSO-LSTM optimizer | inline in `run.py` | Tunes LSTM-style hyperparameters |
| DWT preprocessing | inline in `run.py` | Denoises price series for forecasting |
| Evolution Strategy | inline in `run.py` | Buy/hold/sell timing model |
| SPX price action | `spx_price_action_layer.py` | SPY/SPX market structure gate |
| RL intraday timing | `rl_intraday_structure_trainer.py` | PPO timing gate |
| DeepLOB orderbook | `deeplob_orderbook_direction_layer.py` | Orderbook direction probability |
| Stream-confirmed movement | `stream_confirmed_movement_gate_trade_plan_trailing.py` | Trend-confirmed order plan |
| SR RL agent | `sr_rl_agent.py` | Support/resistance RL policy |
| Options booster | `options_booster.py` | High-confidence long-call booster |

---

## Data Contracts

Preserve these contracts unless the user explicitly asks for a breaking change.

### `market_context.json`

Expected fields:

```json
{
  "overall_market_bias": "bullish | mixed | bearish",
  "trade_permission": "normal | cautious | disabled",
  "risk_level": "low | medium | high",
  "summary": "...",
  "event_flags": {},
  "sector_context_summary": {},
  "key_etf_context_summary": {},
  "decision_reasons": []
}
```

### `premarket_trade_plan.json`

Expected fields:

```json
{
  "created_at_utc": "...",
  "status": "ok | safe_mode | error",
  "execution_mode": "NORMAL | SAFE",
  "data_source": "yfinance | tradingview+yfinance",
  "symbol_count": 0,
  "selected_symbol_count": 0,
  "valid_plan_count": 0,
  "valid_symbols": [],
  "long_allowed_symbols": [],
  "daily_trade_plans": [],
  "raw_premarket_pattern_plans": []
}
```

### `intraday_structure_map.json`

Expected fields:

```json
{
  "created_at_utc": "...",
  "status": "ok | safe_mode | error",
  "data_source": "yfinance | tradingview+yfinance",
  "symbol_count": 0,
  "long_allowed_symbols": [],
  "structures": []
}
```

Per-symbol structure object should include:

```json
{
  "symbol": "RDW",
  "current_price": 0.0,
  "support_levels": [],
  "resistance_levels": [],
  "nearest_levels": {},
  "orb": {},
  "vwap": {},
  "liquidity": {},
  "trade_structure": {
    "long_allowed": false,
    "short_or_put_allowed": false,
    "preferred_long_trigger": null,
    "preferred_long_stop": null
  }
}
```

### Rebalance Orders CSV

Expected common columns:

```text
symbol,side,qty,notional,target_weight,confidence_used,reason
```

Optional model columns:

```text
continuation_probability,expected_return,ensemble_score,rl_confidence,alpha_score
```

### Exit Orders CSV

Expected common columns:

```text
symbol,side,qty,reason
```

Sells must remain valid even when buy-side fields are missing.

---

## File and Directory Conventions

Dated system outputs:

```text
system-data/YYYYMMDD/
```

Dated daily outputs:

```text
daily_system_outputs/YYYYMMDD/
```

Dated execution logs:

```text
logs/execution/YYYYMMDD/
```

Important files:

```text
system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan.json
system-data/YYYYMMDD/YYYYMMDD_intraday_structure_map.json
system-data/YYYYMMDD/YYYYMMDD_intraday_training_dataset.csv
system-data/YYYYMMDD/YYYYMMDD_exit_training_dataset.csv
system-data/YYYYMMDD/YYYYMMDD_forecast_ensemble_signals.json
system-data/YYYYMMDD/YYYYMMDD_rl_intraday_structure_model.zip
logs/execution/YYYYMMDD/YYYYMMDD_executor_journal.csv
logs/execution/YYYYMMDD/YYYYMMDD_executor_equity_curve.csv
logs/execution/YYYYMMDD/YYYYMMDD_live_experience_replay.jsonl
```

---

## Risk Controls

Typical small-account defaults:

| Parameter | Typical Default |
|---|---:|
| Minimum cash buffer | `$1,500` |
| Cash buffer percentage | `8%–12%` |
| Max position weight | `12%–16%` |
| Max gross exposure | `72%–88%` |
| Max daily loss | `1.5%–2.0%` |
| Kill-switch equity floor | account-dependent |

Risk behavior:

- Sell before buy.
- Enforce cash buffer.
- Enforce max order dollars.
- Enforce max gross exposure.
- Enforce max position weight.
- Reduce buying in bearish or cautious market context.
- Always allow risk-reducing exits.
- Log all skipped trades with reasons.

---

## Regime States

Common regime states:

```text
TREND_RISK_ON
VOLATILE_RISK_ON
CHOP_NEUTRAL
DEFENSIVE_RISK_OFF
```

Regime effects:

- `TREND_RISK_ON`: allows normal or expanded exposure if other gates confirm.
- `VOLATILE_RISK_ON`: allows selective trades but reduces size.
- `CHOP_NEUTRAL`: reduces trade frequency and requires stronger confirmation.
- `DEFENSIVE_RISK_OFF`: blocks or heavily restricts new longs.

---

## Options Booster Rules

The options booster is a high-risk amplifier and must stay conservative.

Default rules:

- Long calls only.
- Prefer 21–45 DTE.
- Target approximately 30 DTE.
- Delta target generally 0.35–0.70.
- Require sufficient open interest and volume.
- Require tight spreads.
- Require underlying stock momentum.
- Prefer activation only when an underlying stock position is already open.
- Do not use options as the first signal source.
- Disable options in SAFE mode unless explicitly overridden.

Required fallback for missing underlying price:

```text
row.underlying_price
→ row.stock_price
→ latest Alpaca trade
→ latest Alpaca quote midpoint
→ latest Alpaca bar close
→ local intraday close
```

---

## RL Training

Offline training commands:

```bash
python rl_intraday_structure_trainer.py
python rl_spx_breakout_trainer.py
python sr_rl_agent.py
python tune_rl_intraday.py
python tune_spx_multibar_rl.py
python tune_sr_rl_agent.py
```

RL reward design should emphasize:

- Positive next-bar movement after entry.
- Positive multibar continuation.
- Longer clean trend capture over tiny scalps.
- Positive individual position PnL.
- Positive daily account change.
- Lower drawdown.
- Better entry timing near support/resistance.

RL penalties should include:

- Negative next-bar movement after entry.
- Choppy failed trades.
- Re-entry churn.
- Negative equity fluctuation.
- Negative daily change.
- Holding weak positions too long.
- Taking low-confidence entries.
- Overtrading.

RL live mode guidance:

- Keep new or uncalibrated RL models in shadow mode first.
- Do not let RL override hard risk controls.
- RL can block entries.
- RL should not block exits.
- Promote RL to active control only after validation.

---

## Known Failure Modes

### Planner timeout

Common causes:

- Waiting until market open inside collector workflow.
- Too many symbols.
- Slow yfinance requests.
- Blocking subprocess calls.
- Unbounded per-symbol processing.

Preferred fixes:

- Set `PREMARKET_WAIT_FOR_MARKET_OPEN=0`.
- Cap planner symbols.
- Add total and per-symbol timeouts.
- Write fallback safe outputs.
- Skip low-quality symbols early.

### Circular imports

Common around:

```text
TODAY_TAG
market_context_builder.py
```

Preferred fixes:

- Avoid importing large modules just to get date constants.
- Use local date fallback.
- Move shared constants into a lightweight utility module if needed.

### Missing options underlying price

Common symptom:

```text
skip SYMBOL: no underlying price in source rows
```

Preferred fix:

- Add fallback price lookup from Alpaca latest trade, quote, bar, or local intraday data.

### Stale plans

Execution should degrade to SAFE mode when:

- Plan age exceeds `MAX_PLAN_AGE_MINUTES`.
- Plan status is missing or not `ok`.
- Structure map is missing.
- Long-allowed list is empty and market context is bearish/cautious.

### Oversized run.py

`run.py` is large and monolithic.

When editing it:

- Prefer small helper functions.
- Avoid rewriting unrelated sections.
- Preserve startup order.
- Preserve logging.
- Preserve existing data contracts.
- Avoid large style-only diffs.

---

## Current Architecture Status

Stable or mostly stable:

- Premarket collector orchestration.
- Market context builder.
- Premarket planner after timeout fix.
- Intraday dataset generation.
- Execution journaling.
- Alpaca paper execution.
- RL shadow-mode logging.

Active development:

- TradingView screener integration.
- Options booster underlying price fallback.
- Forecast ensemble calibration.
- RL timing calibration.
- Stream-confirmed movement gate.
- SPX price action layer.
- Safe-mode and stale-plan integration.

Known bottlenecks:

- `run.py` is too large.
- Options booster needs better price enrichment.
- Cross-process state management is complex.
- Some models may write stale or missing artifacts.
- Excessive generated CSV/JSON files can pollute git and agent context.

---

## Refactor Preferences

When Codex edits this repository:

- Prefer additive refactors over destructive rewrites.
- Keep backward compatibility with existing output paths.
- Preserve existing CSV/JSON field names.
- Preserve existing logs unless improving them.
- Add explicit skip reasons when blocking trades.
- Keep functions deterministic where possible.
- Avoid broad changes to working modules without need.
- Do not remove safety checks to make trades fire more often.
- Do not make live-trading changes without explicit user intent.
- Avoid hardcoding private credentials or account-specific secrets.
- Use environment variables for tunable thresholds.
- Keep generated artifacts out of git.

---

## Git Hygiene

Recommended `.gitignore` entries:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc

system-data/
daily_system_outputs/
logs/
models/
*.csv
*.jsonl
*.zip
```

Do not commit:

- API keys
- Alpaca account data
- raw execution logs
- generated datasets
- model checkpoints
- large CSV outputs

---

## Codex Working Rules

When using Codex on this repo:

1. Inspect relevant files before editing.
2. Trace data flow before changing behavior.
3. Preserve all risk controls.
4. Preserve all sell/exit behavior.
5. Add tests or dry-run checks where practical.
6. Keep changes focused.
7. Explain any changed execution behavior.
8. Prefer safe defaults.
9. Use paper trading assumptions unless told otherwise.
10. Never assume missing data means bullish.
11. Never turn a failed gate into permission to buy.
12. Never suppress exits.

---

## Good Debugging Commands

Run the executor:

```bash
python run.py
```

Check latest planner output:

```bash
cat system-data/$(date +%Y%m%d)/$(date +%Y%m%d)_premarket_trade_plan.json
```

Check structure map:

```bash
cat system-data/$(date +%Y%m%d)/$(date +%Y%m%d)_intraday_structure_map.json
```

Check market context:

```bash
cat daily_system_outputs/$(date +%Y%m%d)/$(date +%Y%m%d)_market_context.json
```

Tail execution logs:

```bash
tail -f logs/execution/$(date +%Y%m%d)/$(date +%Y%m%d)_executor_journal.csv
```

Check systemd logs:

```bash
journalctl -u trade-bot -f
```

---

## Safe Editing Checklist

Before changing execution behavior, verify:

- Are exits still always allowed?
- Are sell orders unaffected?
- Are stale plans handled safely?
- Are missing files handled safely?
- Are bearish regimes reducing risk?
- Are duplicate orders still prevented?
- Are all skips logged with a reason?
- Are paths still dated correctly?
- Are Alpaca paper/live flags respected?
- Are generated outputs ignored by git?

---

## Summary

This repository is an AI-driven trading execution system. The correct behavior is not simply to generate more trades. The correct behavior is to generate better risk-adjusted decisions, preserve capital, enforce discipline, and continuously improve through structured data, model feedback, and execution logs.
