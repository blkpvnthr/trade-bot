# Opportunity Finder Implementation Guide

## Summary

The Opportunity Finder system has been successfully implemented at `~/Desktop/trader/`. It's a standalone, adaptive engine that continuously reviews the active trading universe, removes stale/low-quality symbols, and replaces them with stocks most likely to produce bullish momentum continuation and volume footprint imbalance setups.

### Files Created

**Core Modules (7 files)**:
1. `opportunity_logging.py` — JSONL event logger (factory pattern)
2. `opportunity_utils.py` — Path resolution, safe I/O, symbol normalization
3. `opportunity_filters.py` — Stateless filter predicates
4. `opportunity_state.py` — Staleness scoring (9-factor weighted formula)
5. `opportunity_scoring.py` — Opportunity scoring (9-factor composite, setup classification)
6. `opportunity_rotation.py` — Universe rotation algorithm with safety bounds
7. `opportunity_finder.py` — Main orchestrator (importable + standalone)

**File Modified**:
- `requirements.txt` — Added `python-tradingview-ta>=0.3.12`

---

## Installation

### 1. Install Dependencies

```bash
cd ~/Desktop/trader

# Option A: Install only tradingview-ta (python-tradingview-ta already in requirements.txt)
pip install python-tradingview-ta>=0.3.12

# Option B: Reinstall all requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -m py_compile opportunity_*.py
echo "✓ All modules compiled successfully"
```

---

## Usage

### Standalone Execution

```bash
cd ~/Desktop/trader
python opportunity_finder.py
```

Output:
```
[opportunity_finder] complete!
  Final universe size: 25
  Symbols: NVDA, MSFT, AAPL, GOOGL, AMZN, ...
  Elapsed: 12.4s
```

### Importable API

```python
from opportunity_finder import run_opportunity_finder, OpportunityConfig

# Auto-load from environment
result = run_opportunity_finder()

# Or with custom config
config = OpportunityConfig.from_env()
config.min_replacement_score = 80  # Raise threshold
result = run_opportunity_finder(config)

# Access results
print(f"Final universe: {result['final_universe']}")
print(f"Added symbols: {result['added_symbols']}")
print(f"Removed symbols: {result['removed_symbols']}")
```

### Scheduled Execution (example cron)

```bash
# Every 1 hour during market hours (9:30am - 4pm ET)
0 10,11,12,13,14,15 * * 1-5 cd ~/Desktop/trader && python opportunity_finder.py >> logs/opportunity_finder_cron.log 2>&1
```

---

## Configuration

All configuration via environment variables (defaults shown):

```bash
# Enable/disable the finder
OPPORTUNITY_FINDER_ENABLED=true
OPPORTUNITY_FINDER_REPLACE_STALE_NAMES=true

# Candidate quality thresholds
MIN_RVOL=2.0                               # Minimum relative volume
MIN_DOLLAR_VOLUME=20000000                # $20M minimum dollar volume
MIN_PRICE=2                                # Minimum stock price
MAX_PRICE=750                              # Maximum stock price

# Universe sizing
MAX_UNIVERSE_SIZE=25                      # Never exceed 25 symbols
MIN_UNIVERSE_SIZE=8                       # Never drop below 8 symbols

# Staleness evaluation
STALE_SYMBOL_THRESHOLD=65                 # 0-100 scale (65 = stale)

# Replacement candidacy
MIN_REPLACEMENT_SCORE=75                  # Candidates must score ≥75
MAX_REPLACEMENTS_PER_CYCLE=5              # Max 5 swaps per run

# TA Requirements
REQUIRE_TV_TA_BUY_OR_STRONG_BUY=true     # At least 1 of 3 TFs BUY/STRONG_BUY
REQUIRE_MULTI_TIMEFRAME_ALIGNMENT=true    # At least 3 of 4 TFs BUY/STRONG_BUY
```

### Setting via `.env`

Add to root `.env`:
```
OPPORTUNITY_FINDER_ENABLED=true
MIN_REPLACEMENT_SCORE=80
```

---

## Outputs

### Primary Output Files

Generated in `system-data/{DATE}/`:

1. **`{DATE}_opportunity_finder_report.json`**
   - Complete analysis report
   - Universe changes summary
   - Staleness results per symbol
   - Top opportunity candidates
   - Rotation audit trail

2. **`{DATE}_latest_orderflow_candidates.json`**
   - Only the added (accepted) candidates
   - Full OpportunityCandidate schema
   - Setup type classification
   - Multi-timeframe TA alignment
   - Footprint imbalance likelihood

3. **`{DATE}_locked_trading_universe.txt`**
   - Updated trading universe
   - One symbol per line
   - Replaces the existing universe file

4. **`logs/execution/{DATE}/{DATE}_opportunity_finder_events.jsonl`**
   - JSONL event log
   - All staleness evaluations
   - Rotation decisions
   - API errors and diagnostics

### Compatibility Aliases

Also written to `daily_system_outputs/` (flat directory):
- `opportunity_finder_report.json`
- `latest_orderflow_candidates.json`

---

## Example Output JSON

### `opportunity_finder_report.json` (excerpt)

```json
{
  "created_at_utc": "2026-05-23T14:30:00.123456+00:00",
  "data_tag": "20260523",
  "instance_name": "default",
  "elapsed_seconds": 12.4,
  "config": {
    "enabled": true,
    "min_replacement_score": 75,
    "max_replacements": 5,
    ...
  },
  "universe_summary": {
    "original_size": 30,
    "final_size": 27,
    "removed_count": 3,
    "added_count": 3,
    "rotation_count": 3,
    "rotation_blocked_reason": null
  },
  "original_universe": ["AAPL", "NVDA", "MSFT", ...],
  "final_universe": ["NVDA", "MSFT", "GOOGL", "TSLA", ...],
  "removed_symbols": ["AAPL", "META", "NFLX"],
  "added_symbols": ["TSLA", "AMZN", "AVGO"],
  "staleness_results": [
    {
      "symbol": "AAPL",
      "staleness_score": 72.0,
      "is_stale": true,
      "component_scores": {
        "declining_rvol": 0.8,
        "weak_ta_recommendations": 0.7,
        "poor_breakout_probability": 0.6,
        ...
      },
      "reasons": ["RVOL very low", "TA bearish"]
    },
    ...
  ],
  "opportunity_candidates": [
    {
      "symbol": "TSLA",
      "score": 91.2,
      "setup_type": "momentum_continuation",
      "rvol": 4.1,
      "premarket_volume": 2300000,
      "dollar_volume": 325000000.0,
      "trend_strength": 88.5,
      "breakout_probability": 84.3,
      "near_level": "VWAP",
      "ta_recommendation_1m": "STRONG_BUY",
      "ta_recommendation_5m": "STRONG_BUY",
      "ta_recommendation_15m": "BUY",
      "ta_recommendation_1h": "STRONG_BUY",
      "ta_recommendation_1d": "BUY",
      "macd_alignment": true,
      "ma_alignment": true,
      "likely_footprint_imbalance": true
    },
    ...
  ]
}
```

### `latest_orderflow_candidates.json` (excerpt)

```json
{
  "created_at_utc": "2026-05-23T14:30:00.123456+00:00",
  "data_tag": "20260523",
  "instance_name": "default",
  "candidates": [
    {
      "symbol": "TSLA",
      "score": 91.2,
      "setup_type": "momentum_continuation",
      ...
    },
    {
      "symbol": "AMZN",
      "score": 87.5,
      "setup_type": "vwap_reclaim_near_pivot_support",
      ...
    }
  ]
}
```

### Event Log (JSONL excerpt)

```json
{"timestamp_utc": "2026-05-23T14:30:00.123456+00:00", "event": "opportunity_finder_start", "payload": {"data_tag": "20260523", "instance_name": "default", "enabled": true}}
{"timestamp_utc": "2026-05-23T14:30:01.234567+00:00", "event": "staleness_evaluated", "payload": {"symbol": "AAPL", "staleness_score": 72.0, "status": "stale", "reasons": ["RVOL very low", "TA bearish"]}}
{"timestamp_utc": "2026-05-23T14:30:02.345678+00:00", "event": "replacement_candidate_discovered", "payload": {"symbol": "TSLA", "score": 91.2, "setup_type": "momentum_continuation"}}
{"timestamp_utc": "2026-05-23T14:30:03.456789+00:00", "event": "symbol_replaced", "payload": {"removed": "AAPL", "added": "TSLA"}}
...
```

---

## Pipeline Integration

The Opportunity Finder sits in the execution pipeline:

```
tradingview_institutional_screener.py  ← discovers 30 top institutional names
        ↓
locked_trading_universe.txt            ← seeds the finder with current universe
        ↓
opportunity_finder.py                  ← evaluates staleness, rotates universe (NEW)
        ↓
premarket_trade_planner.py             ← builds entry/stop/target plans
        ↓
run.py (execution)                     ← places orders on valid setups
```

The finder **does not** place orders. It only:
1. Evaluates the current universe for quality
2. Discovers stronger candidates
3. Rotates the universe
4. Writes the updated universe file

Downstream systems (`premarket_trade_planner.py`, `run.py`) consume the refreshed universe automatically.

---

## Staleness Formula (9 factors, 0-100 scale)

Each factor is scored 0-1 (1 = most stale), then weighted:

| Factor | Weight | Logic |
|--------|--------|-------|
| **Declining RVOL** | 15% | RVOL < 1x → 1.0; 1-2x → linear; ≥ 2x → 0 |
| **Declining Dollar Volume** | 15% | Below $MIN → 1.0; else linear decay |
| **Weak Momentum** | 12% | momentum_score < 0.3 → 1.0; linear to 0.6 |
| **Weak TA Recs** | 12% | Avg Recommend.All across TFs, inverted |
| **Bearish MTF Alignment** | 10% | MACD bullish, trend flags: low → stale |
| **Poor Breakout Probability** | 10% | From premarket plan, inverted |
| **Not in Valid Symbols** | 10% | Not in premarket plan: 1.0; partial: 0.5 |
| **Below VWAP No Reclaim** | 10% | price < VWAP AND no reclaim → 1.0 |
| **Weak Institutional Score** | 6% | inst_score < 0.50 → 1.0; linear to 0.70 |

**Anti-churn protection**: Symbols in `long_allowed_symbols` are capped below the staleness threshold.

---

## Opportunity Scoring Formula (9 factors, 0-100 scale)

Each factor is scored 0-1, then weighted (sum = 1.0), scaled to 0-100:

| Factor | Weight | Logic |
|--------|--------|-------|
| **RVOL Score** | 20% | clip((rvol - 1) / 4) |
| **Dollar Volume** | 15% | clip(log10(dv / MIN_DV) / 2) |
| **TA Alignment** | 15% | Fraction of 5 TFs BUY/STRONG_BUY |
| **MACD Alignment** | 12% | Fraction of 3 TFs MACD bullish |
| **MTF Rec Score** | 12% | Avg Recommend.All mapped to [0,1] |
| **Price Position** | 10% | Above VWAP, near support, ORB |
| **Breakout Prob** | 8% | From plan or derived |
| **Institutional** | 5% | From discovery or proxy |
| **Liquidity** | 3% | Market cap tier score |

---

## Rotation Safety Guarantees

The rotation algorithm enforces:

- **Never drops below MIN_UNIVERSE_SIZE** (default 8)
- **Never exceeds MAX_UNIVERSE_SIZE** (default 25)
- **Protects `long_allowed_symbols`** from forced removal
- **Limits churn** to MAX_REPLACEMENTS_PER_CYCLE (default 5)
- **Never replaces strong symbols** with weaker candidates
- **Respects market conditions** (halts rotation during halt events)

---

## Events Emitted (JSONL)

The finder logs these events to `logs/execution/{DATE}/{DATE}_opportunity_finder_events.jsonl`:

- `opportunity_finder_start` — initialization
- `staleness_evaluated` — per-symbol staleness computed
- `replacement_candidate_discovered` — candidate found
- `replacement_candidate_accepted` — candidate passes filters
- `replacement_candidate_rejected` — candidate fails filters
- `symbol_replaced` — stale → new swap executed
- `universe_rotation_completed` — rotation finished
- `opportunity_finder_complete` — end of run
- `external_http_error` — API errors (graceful degradation)

---

## Execution Examples

### Run with Default Config

```bash
cd ~/Desktop/trader
python opportunity_finder.py
```

### Run with Custom Config (bash)

```bash
cd ~/Desktop/trader
STALE_SYMBOL_THRESHOLD=60 MIN_REPLACEMENT_SCORE=80 python opportunity_finder.py
```

### Run from Python Script

```python
from opportunity_finder import run_opportunity_finder, OpportunityConfig
import os

# Set env vars
os.environ["MIN_REPLACEMENT_SCORE"] = "80"
os.environ["MAX_REPLACEMENTS_PER_CYCLE"] = "3"

# Run
result = run_opportunity_finder()

# Check results
if result.get("removed_symbols"):
    print(f"Removed (stale): {result['removed_symbols']}")
if result.get("added_symbols"):
    print(f"Added (strong): {result['added_symbols']}")
```

### Run as Part of Pipeline

```bash
#!/bin/bash
set -e

cd ~/Desktop/trader

# Step 1: Discover universe with TradingView screener
echo "[1/3] Discovering universe..."
python tradingview_institutional_screener.py

# Step 2: Rotate universe with Opportunity Finder
echo "[2/3] Rotating universe..."
python opportunity_finder.py

# Step 3: Build premarket plans
echo "[3/3] Building plans..."
python premarket_trade_planner.py

# Step 4: Execute (run.py runs continuously in background)
echo "[done] Universe is ready for execution"
```

---

## Troubleshooting

### No candidates discovered

**Cause**: TradingView screener query failed or returned zero results.

**Solution**:
```bash
# Check event log
tail -f logs/execution/$(date +%Y%m%d)/$(date +%Y%m%d)_opportunity_finder_events.jsonl | grep external_http_error

# Verify TV_FIELDS_FULL and filters in opportunity_finder.py
python -c "from tradingview_screener import Query, col; q = Query(); print('OK')"
```

### Universe not updating

**Cause**: Rotation blocked (disabled, market halt, or floor reached).

**Check**:
- `OPPORTUNITY_FINDER_ENABLED=true` in `.env`
- `market_context.json` → `trade_permission` not `halt`
- `MIN_UNIVERSE_SIZE` not set too high

### Slowness (>30 seconds)

**Cause**: TV screener queries are slow.

**Solution**: Reduce result limit or query less frequently:
```python
# In _fetch_replacement_candidates(), reduce limit from 250 to 100
query = query.set_limit(100)
```

---

## Next Steps

1. **Test locally**: `python opportunity_finder.py`
2. **Monitor first run**: Check `logs/execution/{DATE}/` for events
3. **Integrate with scheduler**: Add to cron or hourly task
4. **Tune thresholds**: Adjust `MIN_REPLACEMENT_SCORE`, `STALE_SYMBOL_THRESHOLD` based on results
5. **Monitor downstream**: Verify `premarket_trade_planner.py` reads new universe correctly

---

## Support

For issues or improvements:
- Check JSONL event log for errors and diagnostic messages
- Review `opportunity_finder_report.json` for staleness details
- Verify config env vars are set correctly
- Ensure `tradingview-screener` and `python-tradingview-ta` are installed

