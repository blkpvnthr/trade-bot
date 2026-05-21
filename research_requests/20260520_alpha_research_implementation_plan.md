# Alpha Research Implementation Plan
**Date:** 2026-05-20  
**Source:** RESEARCH_ALPHA_PORTFOLIO_STRATEGIES.md  
**Purpose:** Convert research findings into concrete, sequenced Claude implementation tasks  
**Mode:** Paper-trading only. All changes are future PRs. No live trading. No broker calls. No credential handling.

---

## How to Use This Plan

Each task below is a self-contained PR scope. Execute them in priority order. For each task:
1. Run `python scripts/smoke_test_variants.py` and `python -m py_compile` before and after
2. Run walk-forward validation if applicable
3. Observe behavior in paper-trading for ≥ 30 days before updating composite score weights

---

## Priority Group A: Zero-Risk Improvements (No Strategy Change, No Execution Change)

---

### A1 — Intraday Seasonality Features

**Research basis:** Heston & Sadka (2010); time-of-day expectancy documented across multiple US equity datasets  
**Research section:** §2.4, §8.1  

**Objective:** Add time-of-day features to training dataset and forecast model. No strategy logic changes.

**Files to modify:**
- `build_intraday_training_dataset.py` → `add_symbol_features()`
- `forecast_engine.py` → `CANDIDATE_FEATURES`

**Exact changes:**
```python
# In add_symbol_features(), after the VWAP block:
et_time = g["timestamp"].dt.tz_convert("America/New_York")
session_open = et_time.dt.floor("D").apply(
    lambda d: pd.Timestamp(d.year, d.month, d.day, 9, 30, tzinfo=et_time.iloc[0].tzinfo)
)
g["session_minute"] = (et_time - session_open).dt.total_seconds() / 60
g["session_minute"] = g["session_minute"].clip(0, 390)
g["session_pct"] = g["session_minute"] / 390.0
g["is_opening_30"] = (g["session_minute"] <= 30).astype(int)
g["is_power_hour"] = ((g["session_minute"] >= 210) & (g["session_minute"] <= 345)).astype(int)
g["is_lunch"] = ((g["session_minute"] >= 120) & (g["session_minute"] <= 210)).astype(int)
```

```python
# In forecast_engine.py CANDIDATE_FEATURES, add:
"session_minute", "session_pct", "is_opening_30", "is_power_hour", "is_lunch",
```

**Validation:**  
- `python -m py_compile build_intraday_training_dataset.py && echo OK`  
- Verify new columns present in output CSV: `python build_intraday_training_dataset.py --dry-run`  
- Retrain LSTM after adding features; compare validation accuracy  

**Acceptance criteria:**  
- All 5 new features present in output CSV  
- LSTM validation accuracy does not decrease > 1%  
- `forecast_engine.py` compiles clean  

**Estimated effort:** S (2–4 hours)  

---

### A2 — VWAP Standard Deviation Bands

**Research basis:** Bollinger Bands applied to VWAP centerline; mean-reversion setup detection  
**Research section:** §8.3  

**Files to modify:**
- `build_intraday_training_dataset.py` → `add_symbol_features()`
- `forecast_engine.py` → `CANDIDATE_FEATURES`

**Exact changes:**
```python
# In add_symbol_features(), after g["dist_vwap"] line:
vwap_std = (g["close"] - g["vwap"]).rolling(20).std()
g["vwap_std"] = vwap_std
g["dist_vwap_std"] = (g["close"] - g["vwap"]) / vwap_std.replace(0, np.nan)
g["vwap_upper_1"] = g["vwap"] + vwap_std
g["vwap_lower_1"] = g["vwap"] - vwap_std
g["vwap_upper_2"] = g["vwap"] + 2 * vwap_std
g["vwap_lower_2"] = g["vwap"] - 2 * vwap_std
g["vwap_reclaim_signal"] = (
    (g["close"] > g["vwap"]) & (g["close"].shift(1) < g["vwap"].shift(1))
).astype(int)
```

**Acceptance criteria:**  
- `dist_vwap_std` replaces raw `dist_vwap` as primary VWAP feature in CANDIDATE_FEATURES (keep both)  
- No NaN rows introduced beyond existing VWAP NaN boundary  

**Estimated effort:** S (1–2 hours)  

---

### A3 — Relative Strength vs SPY Feature

**Research basis:** IBD RS methodology; breakout win-rate improvement with positive RS  
**Research section:** §2.1, §8.2  

**Files to modify:**
- `build_intraday_training_dataset.py` — requires SPY bars alongside symbol bars
- `forecast_engine.py` → `CANDIDATE_FEATURES`

**Approach:**
```python
# In fetch_alpaca_bars(), also fetch SPY 1m bars for the same period
# Store as spy_bars (separate from symbol_bars)
# In add_symbol_features(g, spy_bars):
spy_close = spy_bars.set_index("timestamp")["close"].rename("spy_close")
g = g.set_index("timestamp").join(spy_close, how="left").reset_index()
g["spy_close"] = g["spy_close"].ffill()
g["rs_vs_spy"] = g["close"] / g["spy_close"].replace(0, np.nan)
g["rs_vs_spy_norm"] = g["rs_vs_spy"] / g["rs_vs_spy"].rolling(20).mean()
g["rs_vs_spy_trend"] = (g["rs_vs_spy_norm"] > g["rs_vs_spy_norm"].shift(5)).astype(int)
```

**Risk:** Adds one additional Alpaca API call (SPY bars). Graceful fallback if SPY unavailable.

**Acceptance criteria:**  
- `rs_vs_spy_norm` present in training CSV  
- SPY fetch failure doesn't break dataset build (graceful skip with warning)  

**Estimated effort:** M (4–8 hours)  

---

### A4 — Gap and Volatility Features

**Research basis:** Gap-and-go setup requires gap_pct; volatility regime features for position sizing gate  
**Research section:** §8.5, §8.6  

**Files to modify:**
- `build_intraday_training_dataset.py` → `add_symbol_features()`

```python
# Requires prev_close (last bar of prior session for each symbol)
# Compute per-symbol session join:
g_sorted = g.sort_values("timestamp")
session_dates = g_sorted["timestamp"].dt.tz_convert("America/New_York").dt.date
prev_close = g_sorted.groupby(session_dates)["close"].transform("first").shift(1)
open_price = g_sorted.groupby(session_dates)["open"].transform("first")

g["gap_pct"] = (open_price - prev_close) / prev_close.replace(0, np.nan)
g["vol_expanding"] = (g["atr_14"] > g["atr_14"].shift(5)).astype(int)
g["range_expansion"] = (g["high"] - g["low"]) / g["atr_14"].replace(0, np.nan)
```

**Acceptance criteria:**  
- `gap_pct` correctly reflects overnight gap (positive for gap-up, negative for gap-down)  
- `vol_expanding` is binary integer (0/1)  

**Estimated effort:** M (3–6 hours)  

---

### A5 — Order Flow Proxy Features

**Research basis:** Cont, Kukanov & Stoikov (2014); tick direction as order flow proxy  
**Research section:** §8.4  

```python
# In add_symbol_features():
g["tick_direction"] = np.sign(g["close"] - g["open"])
g["net_tick_5"] = g["tick_direction"].rolling(5).sum()  # -5 to +5 range
g["bull_vol_pct"] = np.where(
    g["close"] > g["open"], g["volume"], 0.0
) / g["volume"].rolling(5).sum().replace(0, np.nan)
g["spread_proxy"] = (g["high"] - g["low"]) / ((g["high"] + g["low"]) / 2).replace(0, np.nan)
g["price_impact"] = (g["close"] - g["open"]).abs() / g["volume"].replace(0, np.nan)
```

**Acceptance criteria:** All 5 features present; `bull_vol_pct` is in [0, 1]  
**Estimated effort:** S (2–3 hours)  

---

### A6 — Extended Evaluator Metrics

**Research basis:** QuantStats, empyrical, pyfolio metric libraries  
**Research section:** §9  

**Files to modify:** `scripts/evaluate_variants.py`

**Add to `compute_metrics()`:**

```python
# Sortino ratio (penalizes only downside deviation)
downside_returns = daily_pnl[daily_pnl < 0]
downside_std = float(downside_returns.std(ddof=1)) if len(downside_returns) > 1 else 1e-6
downside_std = max(downside_std, 1e-6)
sortino = float(mu / downside_std * math.sqrt(252))

# Win/loss ratio
wins = pnl[pnl > 0]
losses = pnl[pnl < 0]
avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 1e-6
win_loss_ratio = avg_win / avg_loss

# Expectancy per dollar risked
expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

# MFE/MAE ratio
mfe_mae_ratio = None
if "mfe" in trades.columns and "mae" in trades.columns:
    valid = trades[["mfe", "mae"]].dropna()
    valid = valid[(valid["mfe"] > 0) & (valid["mae"].abs() > 0)]
    if len(valid) > 3:
        mfe_mae_ratio = float((valid["mfe"] / valid["mae"].abs()).median())

# Consecutive losses
if "date" in trades.columns:
    daily_pnl_series = daily_pnl.values
    max_consec_losses = 0
    current_consec = 0
    for p in daily_pnl_series:
        if p < 0:
            current_consec += 1
            max_consec_losses = max(max_consec_losses, current_consec)
        else:
            current_consec = 0
else:
    max_consec_losses = 0

# Recovery factor
recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
```

**Add to return dict:**
```python
"sortino_ratio":         round(sortino, 3),
"win_loss_ratio":        round(win_loss_ratio, 3),
"expectancy_per_trade":  round(expectancy, 2),
"mfe_mae_ratio":         round(mfe_mae_ratio, 3) if mfe_mae_ratio is not None else None,
"consecutive_loss_max":  max_consec_losses,
"recovery_factor":       round(recovery_factor, 3),
```

**Also add `sortino_ratio` to composite score** (replace or complement `sharpe` component):
- Split weight: `sharpe: 0.20, sortino: 0.10` (same total 0.30 — Sortino penalizes downside, Sharpe penalizes all volatility)

**Acceptance criteria:**
- `python -m py_compile scripts/evaluate_variants.py` clean
- `sortino_ratio` present in output rankings JSON
- `composite_score` changes by < 0.05 for strategies with symmetric return distributions (where Sharpe ≈ Sortino)

**Estimated effort:** M (4–6 hours)  

---

## Priority Group B: Execution Quality (Moderate Risk — Controlled Rollout)

---

### B1 — Time-Window Kill Gates

**Research basis:** Heston et al. (2010) time-of-day expectancy; lunch lull empirically validated  
**Research section:** §5.4  

**Files to modify:** `run.py`

**Approach:**
```python
# Near the top of run.py, define no-entry windows:
NO_ENTRY_WINDOWS_ET = [
    ("11:30", "12:30"),  # lunch lull — low volume, elevated slippage
    ("15:45", "16:01"),  # closing auction — unpredictable price action
]

def _in_no_entry_window(now_et: datetime) -> bool:
    t = now_et.strftime("%H:%M")
    for start, end in NO_ENTRY_WINDOWS_ET:
        if start <= t < end:
            return True
    return False
```

Gate new entries (not exits — exits must always be allowed):
```python
# In the main execution loop, before placing a new order:
if _in_no_entry_window(now_et) and order_side == "buy":  # or short entry
    log_skipped_signal(reason="no_entry_window", ...)
    continue
```

**Safety constraints:**
- NEVER block exits, stop-loss, or sell orders
- Log every suppressed entry as a skipped signal with `reason="no_entry_window"`
- Add `no_entry_window` to the skip-reason analytics in `diagnose_strategy.py`

**Acceptance criteria:**
- Zero entries logged between 11:30–12:30 and 15:45–16:00 ET on any trading day
- Exit/sell orders unaffected
- `skipped_signals.jsonl` entries with `reason="no_entry_window"` visible

**Estimated effort:** S (2–4 hours)  

---

### B2 — Suppress First-Minute Entries

**Research basis:** First-bar bid-ask spreads 2–4× wider; fill quality significantly degraded  
**Research section:** §6.3  

**Files to modify:** `run.py`

```python
SUPPRESS_ENTRIES_BEFORE_ET = "09:31:00"

def _is_too_early(now_et: datetime) -> bool:
    return now_et.strftime("%H:%M:%S") < SUPPRESS_ENTRIES_BEFORE_ET
```

**Estimated effort:** XS (1 hour)  

---

### B3 — Spread-Aware Symbol Filter

**Research basis:** 10 bps spread = 10% of expected profit at $1,500 max order; systematic slippage  
**Research section:** §6.2  

**Files to modify:** `run.py`

```python
MAX_SPREAD_BPS = 10.0  # 10 basis points

def _spread_too_wide(quote) -> bool:
    if quote is None:
        return False  # conservative: allow if data unavailable
    mid = (quote.ask_price + quote.bid_price) / 2.0
    if mid <= 0:
        return False
    spread_bps = (quote.ask_price - quote.bid_price) / mid * 10000
    return spread_bps > MAX_SPREAD_BPS
```

**Use Alpaca `get_stock_latest_quote()` per symbol at signal time (already has quote client).**

**Safety note:** Default to allowing the trade if the quote fetch fails — never block a known profitable exit due to a quote API error.

**Estimated effort:** M (4–6 hours)  

---

### B4 — Breakeven Stop Logic

**Research basis:** Eliminates "winners turned losers" at zero cost to upside capture  
**Research section:** §5.5a  

**Files to modify:** `run.py` (open position management loop)

```python
# In the position tracking loop, for each open long position:
entry_price = position.avg_entry_price
atr_at_entry = position.atr_at_entry  # store at entry time
breakeven_threshold = entry_price + 1.0 * atr_at_entry

if current_price >= breakeven_threshold and position.stop_price < entry_price:
    new_stop = entry_price + 0.01  # 1 cent above entry = breakeven
    position.stop_price = new_stop
    log_event("stop_moved_to_breakeven", strategy_id, symbol, new_stop)
```

**Constraint:** Only move stop UP (never lower an existing stop for longs).

**Acceptance criteria:**
- Zero long positions should exit below their entry price after breakeven stop triggers
- Journal CSV shows `exit_reason="breakeven_stop"` for affected trades
- `winners_giving_back` finding in `diagnose_strategy.py` should decrease by ≥ 20%

**Estimated effort:** M (6–8 hours)  

---

### B5 — ATR Trailing Stop

**Research basis:** Captures extended trend runs while protecting profits  
**Research section:** §5.5b  

**Files to modify:** `run.py`

```python
TRAILING_STOP_ACTIVATION_ATR = 1.5   # activate trailing after 1.5× ATR profit
TRAILING_STOP_DISTANCE_ATR = 1.0     # trail 1.0× ATR below peak close

# In position management loop:
profit_atr = (current_price - entry_price) / atr_at_entry
if profit_atr >= TRAILING_STOP_ACTIVATION_ATR:
    trail_stop = position.highest_close_since_entry - 1.0 * atr_at_entry
    if trail_stop > position.stop_price:  # only ratchet up
        position.stop_price = trail_stop
        log_event("trailing_stop_updated", ...)

# Track highest close since entry:
position.highest_close_since_entry = max(
    position.highest_close_since_entry, current_price
)
```

**Estimated effort:** M (4–6 hours)  
**Dependency:** B4 (breakeven stop) should be implemented first  

---

### B6 — Partial Take-Profit at 60% of Target

**Research basis:** Raises profit factor by increasing win rate; smooths equity curve  
**Research section:** §5.5c, §6.5  

**Files to modify:** `run.py`

```python
PARTIAL_EXIT_PCT = 0.50          # exit this fraction at first target
PARTIAL_EXIT_TARGET_RATIO = 0.6  # first target = 60% of full target distance

# When opening a position, compute two targets:
full_target = signal.target_price
partial_target = entry_price + (full_target - entry_price) * PARTIAL_EXIT_TARGET_RATIO

position.partial_exit_target = partial_target
position.partial_exit_done = False

# In position management loop:
if not position.partial_exit_done and current_price >= position.partial_exit_target:
    shares_to_exit = int(position.shares * PARTIAL_EXIT_PCT)
    if shares_to_exit > 0:
        place_order(symbol, -shares_to_exit, "partial_target")
        position.partial_exit_done = True
        position.shares -= shares_to_exit
        # Move stop to breakeven after partial exit
        position.stop_price = max(position.stop_price, entry_price + 0.01)
```

**Estimated effort:** L (8–12 hours)  
**Dependency:** B4 (breakeven stop should auto-trigger after partial exit)  

---

## Priority Group C: New Strategy Variants

---

### C1 — VWAP Reclaim Strategy

**Research basis:** VWAP as intraday equilibrium attractor; reclaim with volume = institutional buy signal  
**Research section:** §2.2, §3.2  

**Files to create/modify:**
- `strategies/vwap_reversion.py` (new file)
- `strategies/registry.py` (register new strategy)
- `configs/strategy_experiments.yaml` (add variant)

**Strategy spec:**
```python
class VWAPReclaimStrategy(BaseStrategy):
    strategy_id = "vwap_reversion"
    
    DEFAULT_PARAMS = {
        "rvol_min": 1.5,              # volume surge threshold at reclaim
        "min_bars_below_vwap": 2,     # must be below VWAP for N bars
        "atr_stop_mult": 0.5,         # tight stop below reclaim bar low
        "atr_target_mult": 1.5,       # first target above VWAP
        "no_reentry": True,           # only one reclaim trade per symbol per session
        "allow_shorts": False,        # VWAP flush (short) - disable by default
        "session_start_minute": 30,   # don't trade before 10:00 ET
    }
    
    def generate_signals(self, bars, context=None):
        # Compute session VWAP
        vwap_series = self.vwap(bars)
        atr_series = self.atr(bars, 14)
        vol_avg = bars["volume"].rolling(20).mean()
        rvol = bars["volume"] / vol_avg.replace(0, np.nan)
        
        # VWAP reclaim signal:
        # 1. Previous bar(s) below VWAP
        # 2. Current bar closes above VWAP  
        # 3. Volume surge at reclaim
        # 4. Session minute > 30
        below_vwap_count = (bars["close"] < vwap_series).rolling(
            p["min_bars_below_vwap"]
        ).sum()
        
        reclaim = (
            (bars["close"] > vwap_series) &
            (bars["close"].shift(1) < vwap_series.shift(1)) &
            (below_vwap_count.shift(1) >= p["min_bars_below_vwap"]) &
            (rvol >= p["rvol_min"]) &
            (session_minute >= p["session_start_minute"])
        )
        # ... set stop/target per bar
```

**Walk-forward validation required before adding to composite ranking.**

**Acceptance criteria:**
- `smoke_test_variants.py` passes for new variant
- Signal generates correctly on replay data
- Walk-forward OOS Sharpe > 0.3 on 6-month held-out period

**Estimated effort:** L (12–16 hours)  

---

### C2 — Gap-and-Go Strategy

**Research basis:** Post-gap momentum with catalyst; 55–65% continuation rate with volume confirmation  
**Research section:** §3.4  

**Files to create/modify:**
- `strategies/gap_and_go.py` (new file)
- `strategies/registry.py`
- `configs/strategy_experiments.yaml`

**Key parameters:**
```python
DEFAULT_PARAMS = {
    "min_gap_pct": 0.02,        # 2% minimum gap
    "rvol_min": 2.0,            # high volume required
    "atr_stop_mult": 0.75,
    "atr_target_mult": 1.5,
    "no_entry_after_minute": 60, # only first hour
    "require_consolidation": True, # 2-3 bar base before entry
}
```

**Requires:** Gap feature (`gap_pct`) from A4 task above.

**Estimated effort:** L (12–16 hours)  
**Dependency:** A4 (gap feature must exist in training data)  

---

## Priority Group D: Portfolio Intelligence

---

### D1 — Correlation-Aware Capital Allocator

**Research basis:** Lopez de Prado HRP (2016); Ledoit-Wolf covariance (2004)  
**Research section:** §4.2, §4.3  

**Files to modify:** `scripts/capital_allocator.py`

**Approach (Hierarchical Risk Parity — no matrix inversion needed):**
```python
def hierarchical_risk_parity(daily_pnl_df: pd.DataFrame) -> dict[str, float]:
    """
    HRP allocation: cluster strategies by correlation, allocate inversely
    to within-cluster variance. Robust with as few as 20 observations.
    
    Parameters
    ----------
    daily_pnl_df : pd.DataFrame
        Columns = strategy_ids, index = dates, values = daily P&L
    
    Returns
    -------
    dict mapping strategy_id -> allocation weight [0, 1]
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    
    # Covariance with Ledoit-Wolf shrinkage
    cov = daily_pnl_df.cov()
    corr = daily_pnl_df.corr()
    
    # Distance matrix
    dist = np.sqrt((1 - corr.values) / 2)
    np.fill_diagonal(dist, 0)
    dist_condensed = squareform(dist)
    
    # Hierarchical clustering
    link = linkage(dist_condensed, method="ward")
    
    # Quasi-diagonalization: sort by cluster
    order = _get_quasi_diag(link, corr.shape[0])
    sorted_ids = [daily_pnl_df.columns[i] for i in order]
    
    # Recursive bisection allocation
    weights = _recursive_bisect(cov, sorted_ids)
    return weights
```

**Fallback:** If < 20 days of data, use existing Sharpe-weighted allocation unchanged.

**Acceptance criteria:**
- HRP weights sum to 1.0
- `python -m py_compile scripts/capital_allocator.py` clean
- HRP allocation produces lower portfolio variance than Sharpe-weighted on 3-month historical data (validate ex-post)

**Estimated effort:** L (12–16 hours)  
**Risk:** Correlation estimates noisy with < 30 days; HRP may not outperform simple weighting with short history  

---

### D2 — Volatility-Targeted Position Sizing

**Research basis:** Hurst et al. (2012) volatility targeting; reduces drawdowns without reducing expected return  
**Research section:** §4.3  

**Files to modify:** `run.py`, `scripts/capital_allocator.py`

```python
# In capital_allocator.py, add to output:
TARGET_ANNUAL_VOL = 0.10   # 10% target portfolio volatility

recent_portfolio_vol = compute_rolling_portfolio_vol(daily_pnl_df, window=20)
vol_scalar = min(TARGET_ANNUAL_VOL / max(recent_portfolio_vol, 0.005), 1.5)
# vol_scalar: 1.0 = normal, < 1.0 = vol is high (reduce size), > 1.0 = vol is low (allow up to 1.5x)

# Add vol_scalar to capital_allocation.json output
# In run.py, multiply position sizes by vol_scalar when reading allocation
```

**Acceptance criteria:**
- `vol_scalar` present in `capital_allocation.json`
- Position sizes are smaller during high-volatility periods (verify with recent ATR data)

**Estimated effort:** M (6–8 hours)  

---

### D3 — Walk-Forward Integration into Nightly CronJob

**Research basis:** Walk-forward OOS is the only robust performance estimator  
**Research section:** §10.1, §10.2  

**Files to modify:**
- `k8s/cronjobs/nightly_evaluator.yaml`
- `scripts/evaluate_variants.py` (set `walk_forward_oos` weight = 0.10 after data available)

**Nightly sequence:**
```yaml
# In nightly_evaluator.yaml command:
command:
  - python scripts/evaluate_variants.py
  - python scripts/walkforward_strategy.py --strategy-id tux_mtf_original
  - python scripts/walkforward_strategy.py --strategy-id tux_mtf_long_only
  - python scripts/walkforward_strategy.py --strategy-id pivot_points
  - python scripts/walkforward_strategy.py --strategy-id orb_8am
  - python scripts/walkforward_strategy.py --strategy-id liquidity_sweep
  - python scripts/walkforward_strategy.py --strategy-id flawless_victory
  - python scripts/capital_allocator.py
  - python scripts/generate_dashboard.py
```

**After 3 months of WF data:** Set `walk_forward_oos: 0.10` in `configs/strategy_experiments.yaml` composite_score_weights.

**Estimated effort:** S (2–4 hours)  

---

## Priority Group E: ML/RL Improvements

---

### E1 — Sortino-Based Composite Score Update

**Research basis:** Sortino only penalizes downside deviation; more appropriate for asymmetric return distributions  
**Research section:** §9, §7.1  

**Files to modify:** `scripts/evaluate_variants.py`, `configs/strategy_experiments.yaml`

**Change:** Split existing `sharpe` weight (0.30) into `sharpe: 0.20` + `sortino: 0.10`.
Update `compute_composite_score()` to include `sortino_signal = _t(sortino / 1.5)`.

This is safe — adds information without removing existing signals.

**Estimated effort:** S (2–3 hours)  

---

### E2 — RL Reward Function Redesign

**Research basis:** Sharpe-penalized, drawdown-penalized reward aligns RL agent with optimization objective  
**Research section:** §7.5  

**Files to modify:** `rl_intraday_structure_trainer.py`

**New reward function (see §7.5 pseudo-code for full implementation):**
```python
def compute_reward(self, step_pnl, portfolio_std, current_drawdown, action_taken):
    base = step_pnl / (portfolio_std * math.sqrt(252) + 1e-6)
    dd_penalty = -0.5 * (current_drawdown / self.config.max_daily_loss_pct) ** 2
    trade_cost = -0.001 if action_taken != 0 else 0.0  # discourage overtrading
    return float(base + dd_penalty + trade_cost)
```

**Constraint:** New reward function must not change the RL model's action space, observation space, or any risk controls in `run.py`. The RL model only provides timing/sizing signals — execution safety remains in `run.py`.

**Validation:**
- Retrain RL agent with new reward
- Compare walk-forward OOS metrics: new reward should produce lower drawdown, similar or better Sharpe
- Minimum 60-day paper-trading observation before replacing existing model

**Estimated effort:** M (8–12 hours)  
**Risk:** Reward shaping can cause unintended behaviors; validate extensively before deployment  

---

### E3 — Meta-Labeling Secondary Model

**Research basis:** Lopez de Prado (2018) Ch. 3; reduces false signal rate by 20–30%  
**Research section:** §7.3  

**Files to modify:** `forecast_engine.py`, `build_intraday_training_dataset.py`

**Implementation:**
```python
# In build_intraday_training_dataset.py, add meta-label targets:
# For each strategy signal in the training data:
# meta_label_tux_mtf = 1 if (tux_mtf buy signal) AND (future_return_5m > 0) else 0

# In forecast_engine.py, add secondary meta-model:
class MetaLabelModel:
    """
    Lightweight binary classifier: should we trust this signal?
    Input: same CANDIDATE_FEATURES
    Output: meta_label_probability (0–1)
    Filter: only execute signals where meta_label_probability > threshold
    """
    def __init__(self, threshold=0.55):
        self.threshold = threshold
        self.model = None  # RandomForestClassifier or LogisticRegression
    
    def train(self, X: pd.DataFrame, meta_labels: pd.Series):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        base = RandomForestClassifier(n_estimators=100, max_depth=5)
        self.model = CalibratedClassifierCV(base, cv=5)
        self.model.fit(X, meta_labels)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]
```

**Prerequisite:** Minimum 200 labeled trades per strategy before meta-model training.

**Acceptance criteria:**
- Meta-model precision > 0.55 on held-out test set
- Filtering with `meta_label_probability > 0.55` reduces trade count by 15–30% while maintaining > 70% of profit

**Estimated effort:** XL (20–30 hours)  
**Risk:** Requires sufficient trade history; prone to overfitting with small samples; use strict train/test split  

---

### E4 — Probability Calibration for LSTM

**Research basis:** Poorly calibrated probabilities cause incorrect position sizing  
**Research section:** §7.4  

**Files to modify:** `forecast_engine.py`

```python
# After training LSTM, apply isotonic regression calibration:
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression

class CalibratedLSTMWrapper:
    def __init__(self, lstm_model, scaler):
        self.lstm = lstm_model
        self.scaler = scaler
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
    
    def fit_calibration(self, X_cal, y_cal):
        raw_probs = self._predict_raw(X_cal)
        self.calibrator.fit(raw_probs, y_cal)
        self._fitted = True
    
    def predict_proba(self, X):
        raw = self._predict_raw(X)
        if self._fitted:
            return self.calibrator.transform(raw)
        return raw
```

**Estimated effort:** M (8–10 hours)  
**Risk:** Calibration requires a held-out calibration set; don't calibrate on training data  

---

### E5 — Feature Importance Analysis (Monthly)

**Research basis:** Lopez de Prado (2018) Ch. 6–8; MDA is the most robust importance measure  
**Research section:** §7.2  

**Files to create:** `scripts/feature_importance.py`

```python
"""
scripts/feature_importance.py

Compute Mean Decrease Accuracy (MDA) for all CANDIDATE_FEATURES.
Run monthly. Remove features with MDA < 0.5% of total importance.

Usage:
  python scripts/feature_importance.py --config configs/strategy_experiments.yaml
"""
# Load training data from system-data/
# Train RandomForestClassifier on CANDIDATE_FEATURES → future_up_5m
# Compute feature importance via MDA (permutation_importance from sklearn)
# Write: daily_system_outputs/feature_importance_{DATE}.json
# Print ranked features with threshold recommendation
```

**Estimated effort:** M (6–8 hours)  

---

## Validation Framework (All Tasks)

For every task above, before marking as complete:

1. **Compile check:** `python -m py_compile <modified_files>`
2. **Smoke test:** `python scripts/smoke_test_variants.py`
3. **Replay test:** `python scripts/replay_market_day.py --date <recent_date>` — verify signals still generate
4. **No regression:** `evaluate_variants.py` produces same or better composite scores for existing strategies
5. **Walk-forward (for strategy changes):** OOS Sharpe > 0.3 on 6-month held-out period
6. **30-day paper observation (for execution changes):** Observe in paper trading before adjusting weights

---

## Ordering Summary

```
Week 1-2: A1, A2, A5, A6 (features + evaluator metrics — pure data work)
Week 2-3: A3, A4 (SPY RS feature, gap features)
Week 3-4: B1, B2 (time windows, first-minute suppression)
Week 4-6: B3, B4, B5 (spread filter, breakeven, trailing stop)
Week 6-8: B6 (partial take-profit — most complex execution change)
Week 8-10: C1 (VWAP reclaim strategy)
Week 10-12: D3 (walk-forward nightly integration)
Week 12-14: D1 (correlation-aware allocator)
Week 14-16: D2 (volatility targeting)
Month 4-5: C2 (gap-and-go strategy)
Month 5-6: E1, E2 (Sortino weight, RL reward)
Month 6+: E3, E4, E5 (meta-labeling, calibration, feature importance)
```

---

*All proposed changes are paper-trading only. No live trading modifications.*  
*No broker API calls, credential handling, or order placement logic changes outside existing safety gates.*
