# QuantumQuant — Always-On Operations Guide

> **Mode:** Paper trading only. `PAPER_ONLY=true` enforced at startup.  
> **Last updated:** 2026-05-22

---

## Local 3-Instance Runtime (macOS)

Run three paper-trading bots locally, each in its own process with a dedicated
Alpaca paper account, log directory, and strategy variant.

### Instance layout

| Bot file | Strategy ID | Character |
|----------|------------|-----------|
| `.env.bot01` | `tux_mtf_original` | Conservative multi-timeframe |
| `.env.bot02` | `pivot_points` | Floor-trader pivot bounce/breakout |
| `.env.bot03` | `liquidity_sweep` | Smart money sweep reversal |

### First-time setup

```bash
cd ~/Desktop/trader

# 1. Copy examples and fill in paper credentials for each bot
cp .env.bot01.example .env.bot01
cp .env.bot02.example .env.bot02
cp .env.bot03.example .env.bot03
#   Edit each file: replace REPLACE_ME with real Alpaca paper key/secret
#   NEVER use live credentials. APCA_PAPER=true and PAPER_ONLY=true are preset.

# 2. Compile check before starting
python -m py_compile run.py strategies/*.py scripts/*.py && echo "All OK"

# 3. Run smoke tests
python scripts/smoke_test_variants.py
python scripts/smoke_test_training_pipeline.py
```

### Daily operations

```bash
# Start all three instances (skips any bot whose .env file is missing)
bash scripts/start_local_multi_instance.sh

# Check status — shows PID, last event, equity, skipped signal count
bash scripts/check_local_multi_instance.sh

# Live dashboard — refreshes every 5s; pass interval as argument
bash scripts/watch_local_bot.sh 5     # Ctrl+C to exit

# Stop all gracefully (SIGTERM, waits 30s before SIGKILL)
bash scripts/stop_local_multi_instance.sh
```

### Logs per instance

| Path | Contents |
|------|---------|
| `logs/execution/{STRATEGY_ID}/startup.log` | stdout from startup |
| `logs/execution/{STRATEGY_ID}/{DATE}_execution_events.jsonl` | all execution events |
| `logs/execution/{STRATEGY_ID}/metrics.json` | equity, exposure, skip count |
| `logs/execution/{STRATEGY_ID}/skipped_signals.jsonl` | signals that failed gating |
| `.local_pids/{STRATEGY_ID}.pid` | PID file (exists only while running) |

---

## Post-Market Retraining

Automatically rebuild the training dataset, forecast ensemble, and RL timing model
after market close. **Disabled by default.**

### Enable (add to `.env.botNN` for one instance only)

```bash
POST_MARKET_RETRAIN_ENABLED=true
POST_MARKET_RETRAIN_START_ET=1605    # launch after 16:05 ET
POST_MARKET_RETRAIN_ONCE_PER_DAY=true
```

### What runs

1. `build_intraday_training_dataset.py` — downloads today's 1m bars, builds feature + exit datasets
2. `forecast_engine.py` — retrains LSTM/ARIMA forecast ensemble
3. `rl_intraday_structure_trainer.py` — retrains PPO timing model on today's bars

### Outputs

| File | Contents |
|------|---------|
| `daily_system_outputs/post_market_retrain_summary.json` | step results + artifact validation |
| `logs/execution/post_market_retrain_{DATE}.log` | full stdout/stderr from all three scripts |
| `system-data/{DATE}/{DATE}_rl_intraday_structure_model.zip` | updated RL model |
| `system-data/{DATE}/{DATE}_rl_intraday_training_metrics.json` | training metrics |

### Manual run

```bash
# Run standalone (today's date)
python scripts/post_market_retrain_pipeline.py

# Run for a specific date
python scripts/post_market_retrain_pipeline.py --date 20260521

# Check results
cat daily_system_outputs/post_market_retrain_summary.json | python3 -m json.tool
```

### Invariants

- Always a fire-and-forget background Popen — never blocks the main execution loop
- Never runs between 09:30–16:04 ET (time guard)
- Once-per-day guard prevents double-launch within a session
- No trades, no API calls, no risk-control interaction
- If the script is missing: writes `post_market_retrain_failed` event and returns silently

---

## RL Model Lifecycle Monitoring

The RL trainer (`rl_intraday_structure_trainer.py`) runs in a blocking subprocess when
triggered by run.py. Failures are now logged to a dedicated file per strategy per day.

### Check status

```bash
# Full diagnostic: process running? log exists? model exists? provenance?
python scripts/check_rl_training_status.py

# For a specific date or strategy:
python scripts/check_rl_training_status.py --date 20260521 --strategy pivot_points
```

Output includes: process PID (if running), log path, artifact existence,
provenance confirmation (`[data-source] training_csv=alpaca_derived`), last error lines,
and the recommended next command.

### Log locations

- `logs/execution/{STRATEGY_ID}/{DATE}_rl_trainer.log` — captured stdout/stderr from run.py
- `/tmp/rl_trainer_run.log` — if run manually with `tee`

### Execution events

| Event | When |
|-------|------|
| `rl_training_started` | Before subprocess launch (includes timeout_seconds) |
| `rl_intraday_model_refreshed` | On success (includes log_path) |
| `rl_intraday_model_refresh_failed` | On failure (includes log_path, model_exists) |
| `rl_model_validation` | After every attempt — shows ok, model_exists, issues |

---

## Execution-Alpha Block Analysis

Identify high-quality setups that the execution-alpha gate blocked, and compute
counterfactual MFE/MAE outcomes using intraday bars.

**This script is analysis-only. It never changes thresholds, orders, or trading state.**

```bash
# Analyze today's blocked setups (default filters)
python scripts/analyze_execution_alpha_blocks.py

# Custom filters
python scripts/analyze_execution_alpha_blocks.py --min-confidence 0.75 --min-rr 1.5

# Specific date or strategy
python scripts/analyze_execution_alpha_blocks.py --date 20260521 --strategy pivot_points
```

Default filters: `stream_confirmed=True`, `pivot_buy_signal=True`,
`trade_confidence >= 0.8`, `risk_reward_ratio >= 2.0`, `rel_volume >= 1.2`.

### Outputs

| File | Contents |
|------|---------|
| `daily_system_outputs/execution_alpha_block_analysis.json` | Full analysis with MFE/MAE |
| `daily_system_outputs/execution_alpha_block_analysis.csv` | Spreadsheet-friendly version |

### Reading the output

- **score_deficit**: how far below threshold the setup scored (`threshold - score`)
- **mfe_Nm**: max favorable excursion at +N minutes (% above entry)
- **mae_Nm**: max adverse excursion at +N minutes (% below entry)
- **target_hit_first**: True if target reached before stop; False if stop hit first; None if same bar or not reached

---

## Execution-Alpha Threshold Configuration

The threshold is configurable but defaults preserve current behavior.

| Variable | Default | Scope |
|----------|---------|-------|
| `EXECUTION_ALPHA_BUY_THRESHOLD` | (empty = use config) | All buys |
| `EXECUTION_ALPHA_BUY_THRESHOLD_STREAM_CONFIRMED_PIVOT` | (empty = same as above) | Stream-confirmed pivot setups only |

**Do not lower thresholds without reviewing block analysis first.**

Each blocked order now logs `threshold_source` (which resolution path was used) and
`stream_confirmed_pivot` (whether the setup qualified for the override) in the
`execution_alpha_buy_blocked` event.

---

## Paper-Only Guardrails (Non-Negotiable)

These are enforced at the code level — no env var or config can bypass them:

- `_startup_safety_checks()` aborts if `PAPER_ONLY != true` AND `ALLOW_LIVE_TRADING != 1`
- `_startup_safety_checks()` aborts if `APCA_PAPER != true` AND `ALLOW_LIVE_TRADING != 1`
- Sell orders and exits are NEVER blocked by execution-alpha, time-gate, or any other filter
- Kill switch checks run every cycle, regardless of market state
- `MAX_DAILY_LOSS_PCT`, `MAX_DAILY_LOSS_DOLLARS`, `KILL_SWITCH_EQUITY_FLOOR` are enforced always

---

*See also:*
- `docs/architecture.md` — full pipeline DAG
- `RUNTIME_READINESS_REPORT.md` — install and deployment checklist
- `FIRST_DEPLOYMENT_CHECKLIST.md` — first-day validation steps
