# QuantumQuant — Architecture Layers

This document defines the three-layer separation that keeps source code, runtime
state, and research memory cleanly isolated from each other.

---

## Layer 1 — Source Code

**Location:** `~/Desktop/trader/` (git-tracked)

**What it is:** The executable system. Everything Claude writes or edits lives here.

**Tracked by Git. Must always be clean.**

### What belongs here

| Path | Purpose |
|---|---|
| `run.py` | Execution engine |
| `premarket_trade_planner.py` | Pivot structure planner (Alpaca-first) |
| `tradingview_institutional_screener.py` | Universe discovery |
| `llm_sentiment_universe_selector.py` | LLM-driven universe ranking |
| `market_context_builder.py` | Regime + market context |
| `forecast_engine.py` | ML forecasting layer |
| `options_booster.py` | Options overlay |
| `rl_intraday_structure_trainer.py` | RL timing model trainer |
| `sr_rl_agent.py` | S/R RL agent |
| `production_regime_sizing_model.py` | Regime-aware position sizing |
| `build_intraday_training_dataset.py` | Training data builder |
| `system_self_awareness.py` | System introspection/diagnostics |
| `model.py` | Walkforward model |
| `alpaca_executor.py` | Alpaca order execution |
| `premarket_data_collector.py` | Premarket pipeline orchestrator |
| `webhook.py` | TradingView webhook receiver |
| `CLAUDE.md` | Claude architectural guidance |
| `README.md` | Project overview |
| `docs/` | Architecture documentation |
| `images/` | Architecture diagrams |
| `.gitignore` | Runtime exclusion rules |

### What never belongs here

- Any file under `system-data/`, `daily_system_outputs/`, `logs/`
- Trade plans, structure maps, signal JSONs, diagnostic JSONs
- Locked universe `.txt` files
- RL model `.zip` / `.pkl` artifacts
- Feature streams (`.jsonl`)
- `.env` credentials
- `__pycache__/`

---

## Layer 2 — Runtime State

**Location:** `~/Desktop/trader/system-data/`, `daily_system_outputs/`, `logs/`

**What it is:** Everything the system generates while running. Ephemeral by design.
Regenerated on every run. Never committed to Git.

**Not tracked by Git. Lives only on disk.**

### What lives here

| Path pattern | Contents |
|---|---|
| `system-data/YYYYMMDD/` | Dated trade plans, structure maps, universe files |
| `daily_system_outputs/YYYYMMDD/` | Collector state, events, market context |
| `logs/execution/` | Execution summaries, account snapshots, kill-switch state |
| `logs/orchestrator/` | Orchestrator run events |
| `*_premarket_trade_plan.json` | Output of `premarket_trade_planner.py` |
| `*_intraday_structure_map.json` | Structure map for `run.py` |
| `*_intraday_feature_stream.jsonl` | Intraday feature event stream |
| `*_locked_trading_universe.txt` | Dated universe snapshot |
| `*_latest_llm_sentiment_universe.json` | LLM-selected universe |
| `*.zip` / `*.pkl` | RL model artifacts |
| `*_signals.json`, `*_diagnostics.json` | Runtime signal snapshots |

### Key rule

> `run.py` reads Layer 2 files as inputs. It never reads from the git index.
> Claude should treat these files as ephemeral: they can be regenerated, they
> should not be reasoned about as if they are persistent source code.

---

## Layer 3 — Research Memory

**Location:** `/Users/blkpvnthr/Documents/quantumquant/` (Obsidian vault)

**What it is:** The strategy brain. Architecture decisions, prompts, experiments,
session logs, and notes that inform how the system is built — but are not
executable code.

**Not in the trading repo's Git. Managed by Obsidian.**

### What lives here

- Strategy design notes and hypothesis logs
- Architecture decision records (ADRs)
- Claude prompts and system instructions
- Experiment results and trade reviews
- Market structure observations
- `CLAUDE.md` — the authoritative project guidance file Claude reads first

### How Claude should use this layer

When Claude Code is invoked in the `~/Desktop/trader/` repo, it should:

1. **Read `/Users/blkpvnthr/Documents/quantumquant/CLAUDE.md` first** before
   starting any architectural task. This file contains current strategy direction,
   data source authority rules, and constraints that override defaults.

2. Treat notes in this vault as **context**, not source code. Do not edit `.py`
   files based on Obsidian notes without confirming intent with the user.

3. Never commit Obsidian files into the `~/Desktop/trader/` Git repo. The vault
   is a separate root — keep it separate.

---

## Decision Table — Where Does It Go?

| Artifact | Layer | Git? |
|---|---|---|
| Python source files | 1 — Source | Yes |
| `CLAUDE.md` (repo root) | 1 — Source | Yes |
| `README.md` | 1 — Source | Yes |
| `docs/` | 1 — Source | Yes |
| `system-data/YYYYMMDD/*` | 2 — Runtime | No |
| `daily_system_outputs/*` | 2 — Runtime | No |
| `logs/*` | 2 — Runtime | No |
| `*_trade_plan.json` | 2 — Runtime | No |
| `*_intraday_structure_map.json` | 2 — Runtime | No |
| `*.jsonl` streams | 2 — Runtime | No |
| `*.zip` / `*.pkl` RL models | 2 — Runtime | No |
| `*_locked_trading_universe.txt` | 2 — Runtime | No |
| `.env` credentials | 2 — Runtime | No (never) |
| Obsidian vault notes | 3 — Memory | No (separate repo) |
| `CLAUDE.md` (Obsidian vault) | 3 — Memory | No (separate root) |

---

## .gitignore Enforcement

The `.gitignore` at the repo root enforces Layer 2 exclusions. Key patterns:

```
system-data/
daily_system_outputs/
logs/
execution_logs/
*.jsonl
*_trade_plan.json
*_diagnostics.json
*_signals.json
*_intraday_structure_map.json
*_locked_trading_universe.txt
*_feature_stream*
*.zip
*.pkl
.env
.env.*
__pycache__/
.venv/
venv/
```

If a new runtime output type is added, update `.gitignore` before the first run
that generates it. The rule is: **if `run.py` or any planner writes it, it goes
in `.gitignore`**.
