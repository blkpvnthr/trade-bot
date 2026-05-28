# GitHub Reference Implementation Plan
**Date:** 2026-05-20  
**Source:** RESEARCH_ALPHA_PORTFOLIO_STRATEGIES.md §14  
**Purpose:** Map open-source repository patterns to concrete QuantumQuant implementation tasks  
**Mode:** Architectural reference only. No code copying. No vendoring without license verification.

---

## Licensing Summary

| Repo | License | Can Use Patterns? | Can Use Code? |
|------|---------|------------------|---------------|
| freqtrade | GPL-3.0 | Yes | No (copyleft) |
| jesse | MIT | Yes | Yes (with attribution) |
| backtesting.py | AGPL-3.0 | Yes | No (strong copyleft) |
| vectorbt | Custom (non-commercial) | Yes | No |
| quantstats | Apache-2.0 | Yes | Yes |
| Riskfolio-Lib | BSD-3-Clause | Yes | Yes |
| PyPortfolioOpt | MIT | Yes | Yes |
| FinRL | MIT | Yes | Yes |
| tensortrade | Apache-2.0 | Yes | Yes |
| empyrical | Apache-2.0 | Yes | Yes |
| pyfolio | Apache-2.0 | Yes | Yes |
| alpaca-py | Apache-2.0 | Yes | Yes |
| ta (bukosabino) | MIT | Yes | Yes |
| ta-lib-python | BSD-2-Clause | Yes | Yes |

> **Rule:** Never copy > 10 lines of GPL or AGPL code. For MIT/Apache/BSD, always add attribution comment if adapting non-trivial logic. For architectural patterns (data flow, class structure), no attribution needed.

---

## Task 1: Adopt quantstats Metric Formulas

**Source repo:** ranaroussi/quantstats (Apache-2.0)  
**Stars:** 7,139 | **Actively maintained:** Yes (updated 2026-05-20)  

**What to borrow conceptually:**  
The `sortino()`, `calmar()`, `omega_ratio()`, `consecutive_losses()`, and `expectancy()` formulas from `quantstats/stats.py`. These are numerically robust implementations handling edge cases (zero vol, insufficient data, NaN series).

**What NOT to copy:**  
The HTML reporting module (`quantstats/reports.py`) — uses matplotlib and is designed for Jupyter notebooks, not production pipelines.

**QuantumQuant files to modify:** `scripts/evaluate_variants.py`

**Specific implementation:**
```python
# Pattern from quantstats/stats.py (re-implemented, not copied):
# sortino: returns / downside_std * sqrt(252), downside = returns[returns < 0]
# omega: sum(positive) / sum(negative), threshold = 0
# expectancy: win_rate * avg_win - loss_rate * avg_loss
```

**Validation test:**
```python
# Test against known series:
pnl = pd.Series([100, -50, 200, -30, 150, -80, 90])
# Verify: sortino > sharpe when losses are small; omega > 1.0; expectancy > 0
```

**Expected performance impact:** Better ranking discrimination between strategies with asymmetric return distributions (liquidity_sweep tends to have asymmetric returns)  
**Risk of overfitting:** None — these are evaluation metrics, not trading signals  
**Licensing concerns:** None (Apache-2.0)  

---

## Task 2: HRP Allocation from Riskfolio-Lib

**Source repo:** dcajasn/Riskfolio-Lib (BSD-3-Clause)  
**Stars:** 4,197 | **Actively maintained:** Yes (updated 2026-05-21)  

**What to borrow conceptually:**  
The **Hierarchical Risk Parity (HRP)** algorithm pattern. Core steps:
1. Compute correlation matrix from returns
2. Convert to distance matrix: `d = sqrt((1 - corr) / 2)`
3. Hierarchical clustering (Ward linkage)
4. Quasi-diagonalization: reorder assets by cluster
5. Recursive bisection: allocate inversely to variance within each cluster

This algorithm does not require matrix inversion (unlike Markowitz) — works with `n_strategies=6` and `n_observations=20+`.

**What NOT to copy:**  
The full `Portfolio` class (heavy dependency: cvxpy, scipy, skfolio). Too complex for the allocator.

**QuantumQuant files to modify:** `scripts/capital_allocator.py`

**Concrete implementation plan:**
```python
# Implement pure-scipy HRP (no cvxpy needed):
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import numpy as np

def hrp_weights(daily_pnl_df: pd.DataFrame) -> dict[str, float]:
    """
    Hierarchical Risk Parity weights.
    Works with as few as 20 daily P&L observations.
    No matrix inversion — numerically stable.
    """
    # Require minimum data
    if len(daily_pnl_df) < 15 or daily_pnl_df.shape[1] < 2:
        return {}  # fallback to existing Sharpe weighting

    corr = daily_pnl_df.corr().clip(-0.999, 0.999)
    cov = daily_pnl_df.cov()
    
    # Distance matrix
    dist = ((1 - corr) / 2) ** 0.5
    np.fill_diagonal(dist.values, 0)
    
    # Hierarchical clustering
    link = linkage(squareform(dist.values), method="ward")
    
    # Sort order (quasi-diagonal)
    sort_ix = _quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    
    # Recursive bisection
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            sub for sub in (
                c[:len(c)//2], c[len(c)//2:]
                for c in clusters
            )
            for sub in sub if len(sub) > 0
        ]
        for left, right in [(clusters[i], clusters[i+1]) for i in range(0, len(clusters)-1, 2)]:
            left_var = _cluster_var(cov, left)
            right_var = _cluster_var(cov, right)
            alloc_factor = 1 - left_var / (left_var + right_var + 1e-10)
            weights[left] *= alloc_factor
            weights[right] *= 1 - alloc_factor
    
    # Normalize
    weights = weights / weights.sum()
    return weights.clip(0.05, 0.40).to_dict()  # apply existing min/max caps


def _cluster_var(cov, cluster_items):
    sub_cov = cov.loc[cluster_items, cluster_items].values
    w = np.ones(len(cluster_items)) / len(cluster_items)  # equal weight within cluster
    return float(w @ sub_cov @ w)


def _quasi_diag(link):
    """Return sort order from hierarchical clustering."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i+1)
        sort_ix = sort_ix.append(df0)
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()
```

**Validation test:**
```python
# With perfectly correlated strategies: HRP should reduce both below Sharpe-weighted
# With uncorrelated strategies: HRP should give equal weight
import numpy as np, pandas as pd
np.random.seed(42)
n = 60
correlated = pd.DataFrame({
    "s1": np.random.normal(10, 50, n),
    "s2": None,  # s2 = 0.9 * s1 + noise
    "s3": np.random.normal(10, 50, n),
})
correlated["s2"] = correlated["s1"] * 0.9 + np.random.normal(0, 15, n)
weights = hrp_weights(correlated)
assert abs(weights["s1"] + weights["s2"] - weights["s3"]) < 0.05  # roughly equal cluster weights
```

**Expected performance impact:** 15–25% reduction in portfolio drawdown during correlated market events  
**Risk of overfitting:** Low — HRP is parameter-free (no tuning)  
**Licensing:** BSD-3-Clause for Riskfolio-Lib. The implementation above is from-scratch based on the Lopez de Prado (2016) paper, not from Riskfolio source code.  

---

## Task 3: FinRL Turbulence Index Pattern

**Source repo:** AI4Finance-Foundation/FinRL (MIT)  
**Stars:** ~10,000 (estimated) | **Actively maintained:** Yes  

**What to borrow conceptually:**  
The **Turbulence Index** — a Mahalanobis distance measure that quantifies how unusual current market conditions are relative to historical observations. When turbulence is elevated, reduce position sizes automatically.

**Formula:**
```
turbulence_t = (r_t - μ)^T · Σ^(-1) · (r_t - μ)
where:
  r_t = return vector of all tracked symbols on day t
  μ = historical mean return vector
  Σ = historical covariance matrix
```

When `turbulence_t > 90th percentile of historical values`: reduce all position sizes by 50%.

**What NOT to copy:**  
The full FinRL environment infrastructure. The turbulence index formula is from Kritzman & Li (2010, "Skulls, Financial Turbulence, and Risk Management") — an academic paper with no code restrictions.

**QuantumQuant files to modify:** `market_context_builder.py` or `run.py`

**Concrete implementation plan:**
```python
def compute_turbulence(current_returns: np.ndarray, 
                       historical_means: np.ndarray,
                       historical_cov: np.ndarray) -> float:
    """
    Mahalanobis distance turbulence index.
    High values indicate unusual market conditions.
    Source: Kritzman & Li (2010), adapted from FinRL architecture pattern.
    """
    try:
        diff = current_returns - historical_means
        inv_cov = np.linalg.pinv(historical_cov)  # pseudo-inverse for stability
        turbulence = float(diff @ inv_cov @ diff)
        return max(0.0, turbulence)
    except Exception:
        return 0.0

# In market_context_builder.py:
# Compute daily returns for SPY, QQQ, IWM, TLT, GLD, VXX
# Compute turbulence vs 252-day history
# Add to market_context output as "turbulence_score" and "turbulence_regime"
```

**Validation test:**
- `turbulence_score` should spike on known crisis dates (2020-03 COVID, 2022-01 Fed rate shock)
- In paper trading: verify position sizes are halved when `turbulence_score > 90th pctile`

**Expected performance impact:** Fewer catastrophic loss events during systemic market stress  
**Risk of overfitting:** Low — turbulence is a regime gate, not a predictive feature; no parameter tuning needed  
**Licensing:** MIT (FinRL). The implementation above is from the Kritzman (2010) paper, not the repo source code.  

---

## Task 4: backtesting.py Trade Lifecycle Pattern

**Source repo:** kernc/backtesting.py (AGPL-3.0)  
**Stars:** 8,388 | **Actively maintained:** Yes  

**What to borrow conceptually:**  
The `Trade` object lifecycle pattern: a trade tracks its own `entry_price`, `entry_bar`, `size`, `sl` (stop-loss), `tp` (take-profit), `_sl_order`, and `_tp_order` as internal state that can be **modified during the trade**. This is the cleanest reference for implementing trailing stops and partial exits in `run.py`.

**What NOT to copy:**  
Any actual code (AGPL-3.0 — strong copyleft).

**QuantumQuant files to modify:** `run.py` (open position management)

**Pattern to adapt:**
```python
# Conceptual pattern from backtesting.py (re-implemented, not copied):
# A Position object should support:
class OpenPosition:
    def __init__(self, symbol, entry_price, shares, stop_price, target_price, atr_at_entry):
        self.symbol = symbol
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = stop_price         # mutable — can be moved up
        self.target_price = target_price     # mutable — can be adjusted
        self.atr_at_entry = atr_at_entry
        self.highest_close = entry_price     # for trailing stop
        self.partial_done = False            # flag for partial exit
        self.entry_bar = 0                   # bar number at entry
        
    def update_trailing(self, current_price: float, trail_atr: float) -> None:
        """Ratchet stop up as price advances."""
        if current_price > self.highest_close:
            self.highest_close = current_price
        trail_level = self.highest_close - trail_atr
        if trail_level > self.stop_price:
            self.stop_price = trail_level
    
    def move_to_breakeven(self) -> None:
        """Move stop to 1 cent above entry."""
        if self.stop_price < self.entry_price:
            self.stop_price = self.entry_price + 0.01
```

**Validation test:**
- After implementing: `winners_giving_back` in `diagnose_strategy.py` should decrease
- `journal.csv` should show `exit_reason in ("trailing_stop", "breakeven_stop", "partial_target")` for relevant trades

**Expected performance impact:** +10–15% profit factor from fewer complete reversals on winners  
**Risk of overfitting:** Low — trailing stops are mechanical, not fitted to historical data  
**Licensing:** AGPL-3.0 for backtesting.py. The `OpenPosition` class above is original code inspired by the design pattern, not copied.  

---

## Task 5: quantstats Rolling Performance Monitoring

**Source repo:** ranaroussi/quantstats (Apache-2.0)  

**What to borrow:** The pattern for computing rolling Sharpe across multiple windows (7d, 14d, 30d) simultaneously to detect strategy degradation. This is how alpha decay should be monitored.

**QuantumQuant files to modify:** `scripts/evaluate_variants.py`, `scripts/strategy_health_monitor.py`

**Pattern:**
```python
# Adapted from quantstats/stats.py rolling_sharpe pattern:
def rolling_sharpe_series(daily_pnl: pd.Series, windows: list[int]) -> dict[str, pd.Series]:
    """Compute rolling Sharpe for multiple windows simultaneously."""
    result = {}
    for window in windows:
        rolling_mean = daily_pnl.rolling(window).mean()
        rolling_std = daily_pnl.rolling(window).std(ddof=1).replace(0, np.nan)
        result[f"sharpe_{window}d"] = (rolling_mean / rolling_std * math.sqrt(252)).fillna(0)
    return result

# In evaluate_variants.py, add:
rolling = rolling_sharpe_series(daily_pnl, windows=[7, 14, 30])
alpha_decay_14d = rolling["sharpe_14d"].iloc[-1] / (rolling["sharpe_30d"].iloc[-1] + 1e-6)
# Flag if alpha_decay_14d < 0.5 (recent 14d Sharpe less than half the 30d Sharpe)
```

**Validation test:** Rolling Sharpe should be stable during good periods and declining during bad periods. Validate on known historical drawdown dates in the paper trading logs.

**Expected performance impact:** Earlier detection of strategy degradation → faster parameter re-optimization  
**Licensing:** Apache-2.0 (clean)  

---

## Task 6: PyPortfolioOpt Covariance Shrinkage

**Source repo:** PyPortfolio/PyPortfolioOpt (MIT)  
**Stars:** 5,730  

**What to borrow conceptually:**  
The Ledoit-Wolf covariance shrinkage pattern from `pypfopt/risk_models.py`. This produces a more accurate covariance estimate than the sample covariance when `n_observations < n_strategies * 10` (which is almost always the case for 6 strategies and < 60 days of data).

**What to use:**  
Can actually use `pypfopt` directly (MIT license, compatible):
```python
# requirements.txt addition:
# PyPortfolioOpt>=1.5.0

from pypfopt.risk_models import CovarianceShrinkage

def shrunk_covariance(daily_pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance — more accurate for small samples."""
    if len(daily_pnl_df) < 10:
        return daily_pnl_df.cov()
    # returns-based shrinkage
    S = CovarianceShrinkage(daily_pnl_df).ledoit_wolf()
    return S
```

**Validation test:**  
- Shrunk covariance should have lower condition number than sample covariance
- `np.linalg.cond(shrunk_cov) < np.linalg.cond(sample_cov)` for small samples

**Expected performance impact:** More stable HRP weights when data is limited  
**Licensing:** MIT — can add to requirements.txt and use directly  

---

## Task 7: TensorTrade Modular Reward Architecture

**Source repo:** tensortrade-org/tensortrade (Apache-2.0)  
**Stars:** 6,276  

**What to borrow conceptually:**  
The modular reward scheme pattern where each reward component is a separate class with a `get_reward()` method, and the total reward is a weighted sum of components. This makes the RL reward function in `rl_intraday_structure_trainer.py` much easier to tune and debug.

**Architecture to adapt:**
```python
# Inspired by tensortrade/env/default/rewards/ pattern:
class RewardScheme:
    def get_reward(self, portfolio_state: dict) -> float:
        raise NotImplementedError

class SharpeReward(RewardScheme):
    def __init__(self, window: int = 20, weight: float = 1.0):
        self.window = window
        self.weight = weight
        self._returns = []
    
    def get_reward(self, state):
        self._returns.append(state["step_pnl"])
        if len(self._returns) < 5:
            return 0.0
        returns = np.array(self._returns[-self.window:])
        mu, sigma = returns.mean(), returns.std() + 1e-6
        return float(mu / sigma * math.sqrt(252)) * self.weight

class DrawdownPenalty(RewardScheme):
    def __init__(self, max_drawdown: float, weight: float = 0.5):
        self.max_drawdown = max_drawdown
        self.weight = weight
    
    def get_reward(self, state):
        dd = state["current_drawdown"]
        return -0.5 * (dd / self.max_drawdown) ** 2 * self.weight

class CompositeReward:
    def __init__(self, schemes: list[RewardScheme]):
        self.schemes = schemes
    
    def get_reward(self, state) -> float:
        return sum(s.get_reward(state) for s in self.schemes)

# Usage in rl_intraday_structure_trainer.py:
reward_fn = CompositeReward([
    SharpeReward(window=20, weight=1.0),
    DrawdownPenalty(max_drawdown=0.02, weight=0.5),
    OvertradingPenalty(max_daily_trades=10, weight=0.3),
])
```

**Validation test:**
- RL agent trained with composite reward should show lower drawdown than agent trained with raw P&L reward
- Compare on 60-day OOS window in paper trading

**Expected performance impact:** RL agent behavior better aligned with composite optimization objective  
**Risk of overfitting:** Reward shaping can cause unexpected behavior; test extensively in paper mode  
**Licensing:** Apache-2.0 — class structure pattern can be freely adapted  

---

## Task 8: empyrical/pyfolio Drawdown Analysis Pattern

**Source repos:** quantopian/empyrical (Apache-2.0), quantopian/pyfolio (Apache-2.0)  

**What to borrow:**  
1. `empyrical.stats.drawdown_series()` pattern — returns underwater equity curve (values always ≤ 0)
2. `pyfolio.timeseries.perf_stats()` output format — ~30 performance metrics in a single DataFrame

**QuantumQuant files to modify:** `scripts/generate_dashboard.py`, `scripts/diagnose_strategy.py`

**Pattern for drawdown series:**
```python
def drawdown_series(cumulative_pnl: pd.Series) -> pd.Series:
    """Underwater equity curve — always ≤ 0, shows depth and duration of drawdowns."""
    peak = cumulative_pnl.cummax()
    return (cumulative_pnl - peak) / (peak.abs() + 1e-6)
```

**Pattern for comprehensive metrics dict:**
```python
# Adapted from pyfolio/timeseries.py perf_stats() (Apache-2.0 — can adapt):
def compute_full_perf_stats(daily_pnl: pd.Series) -> dict:
    """Compute all performance metrics in one call."""
    return {
        "annual_return": (daily_pnl.mean() * 252),
        "annual_volatility": (daily_pnl.std() * math.sqrt(252)),
        "sharpe_ratio": ...,
        "sortino_ratio": ...,
        "calmar_ratio": ...,
        "max_drawdown": ...,
        "drawdown_duration_days": ...,  # longest continuous underwater period
        "win_rate": ...,
        "profit_factor": ...,
        "expectancy": ...,
        "avg_daily_pnl": ...,
    }
```

**Expected performance impact:** Better monitoring → faster detection of strategy problems  
**Licensing:** Apache-2.0 for both repos — pattern adaptation is clean  

---

## Task 9: ta (bukosabino) Money Flow Index

**Source repo:** bukosabino/ta (MIT)  
**Stars:** 5,086  

**What to borrow conceptually:**  
The **Money Flow Index (MFI)** formula — a volume-weighted RSI that serves as an order-flow proxy stronger than RVOL alone.

```
MFI = 100 - 100 / (1 + money_flow_ratio)
where:
  typical_price = (high + low + close) / 3
  money_flow = typical_price × volume
  positive_money_flow = sum(money_flow where typical_price > prev_typical_price, N bars)
  money_flow_ratio = positive_money_flow / negative_money_flow
```

**QuantumQuant files to modify:** `build_intraday_training_dataset.py`

```python
# Adapted from ta/volume.py MFIIndicator (MIT — can adapt freely):
def mfi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI. Better order-flow proxy than RVOL."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    raw_money_flow = typical * bars["volume"]
    pos_flow = raw_money_flow.where(typical > typical.shift(1), 0).rolling(period).sum()
    neg_flow = raw_money_flow.where(typical <= typical.shift(1), 0).rolling(period).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))
```

**Add `mfi_14` to CANDIDATE_FEATURES in `forecast_engine.py`.**

**Validation test:**
- MFI should be 0–100 range
- MFI > 80 = overbought, MFI < 20 = oversold — verify against known high-volume reversal days

**Expected performance impact:** +1–2% improvement in signal quality for VWAP reversion and liquidity sweep strategies  
**Licensing:** MIT — can use formula freely  

---

## Task 10: alpaca-py Snapshot + News API Integration

**Source repo:** alpacahq/alpaca-py (Apache-2.0)  
**Stars:** 1,332  

**What to use directly:**  
1. `StockSnapshotRequest` for multi-symbol snapshots (includes VWAP, daily volume, previous close in one API call)
2. `News` endpoint for catalyst detection

**QuantumQuant files to modify:** `market_context_builder.py`, `premarket_trade_planner.py`

**For gap-and-go catalyst filter:**
```python
# Already in market_context_builder.py as SerpAPI — add Alpaca news as supplemental:
# alpaca-py: from alpaca.data.historical.news import NewsClient
# nc = NewsClient(api_key, api_secret)
# news = nc.get_news(NewsRequest(symbols=["AAPL", "NVDA"], start=yesterday, end=today))
# Filter: symbols with news in ["earnings", "upgrade", "acquisition", "beat"]
```

**For pre-session snapshot (replaces per-symbol yfinance calls):**
```python
# StockSnapshotRequest fetches VWAP, daily_change, volume, prev_close for 30 symbols in one call
# This replaces the current per-symbol loop in premarket_trade_planner.py
```

**Expected performance impact:** Reduced API latency at market open (30 symbols × 1 call vs 30 separate calls)  
**Licensing:** Apache-2.0 — already in use  

---

## What NOT to Borrow from Each Repo

| Repo | Do NOT Use |
|------|-----------|
| freqtrade | Any code (GPL-3.0). Do not use FreqAI code verbatim. |
| backtesting.py | Any code (AGPL-3.0). Pattern reference only. |
| vectorbt | Any code (non-commercial license). Rolling backtest concept is fine to re-implement independently. |
| FinRL | The full environment infrastructure — too complex and crypto-oriented. Turbulence index formula only. |
| Riskfolio-Lib | The `Portfolio` class (heavy cvxpy dependency). HRP algorithm only, re-implemented. |
| pyfolio | Plotting utilities — use statistics functions only, matplotlib not appropriate for production pipeline. |
| empyrical | The `factor_returns` functions — these assume Fama-French factor exposure which QuantumQuant doesn't have. |
| quantstats | HTML report generation — not suitable for automated pipeline; use stat functions only. |

---

## Dependency Impact Summary

Tasks that require new pip dependencies (add to `requirements.txt`):

| Task | New Dependency | Version | License | Size |
|------|---------------|---------|---------|------|
| D1 (HRP allocator) | `scipy` | ≥1.10 | BSD | Already likely present |
| D1 (covariance shrinkage) | `PyPortfolioOpt` | ≥1.5 | MIT | Small |
| E3 (meta-labeling) | `scikit-learn` | ≥1.3 | BSD | Already likely present |
| Task 9 (MFI) | None — pure pandas | — | — | — |
| Task 10 (Alpaca news) | Already in alpaca-py | — | Apache-2.0 | — |

**Do not add:** `ta-lib` (requires C compilation, hard to containerize), `vectorbt` (non-commercial), `cvxpy` unless actually using optimization (large dependency).

---

## Testing Matrix

For each GitHub-inspired implementation, run this validation sequence:

```bash
# 1. Compile check
python -m py_compile <file>

# 2. Unit test for specific function
python3 -c "
from scripts.capital_allocator import hrp_weights
import pandas as pd, numpy as np
np.random.seed(42)
df = pd.DataFrame(np.random.normal(0, 1, (60, 6)), 
                  columns=['s1','s2','s3','s4','s5','s6'])
w = hrp_weights(df)
assert abs(sum(w.values()) - 1.0) < 0.001, 'weights must sum to 1'
assert all(0.05 <= v <= 0.40 for v in w.values()), 'weights outside caps'
print('HRP test PASSED:', w)
"

# 3. Integration test
python scripts/evaluate_variants.py --config configs/strategy_experiments.yaml
python scripts/capital_allocator.py --total-capital 100000
python scripts/generate_dashboard.py

# 4. Smoke test
python scripts/smoke_test_variants.py
```

---

## Implementation Priority Order

```
Immediate (zero-dependency):
  Task 5  (quantstats rolling Sharpe monitoring)
  Task 8  (empyrical drawdown series pattern)
  Task 9  (MFI feature — pure pandas)

Short-term (small additions):
  Task 1  (quantstats metrics in evaluate_variants.py)
  Task 4  (backtesting.py trade lifecycle pattern in run.py)
  Task 6  (PyPortfolioOpt Ledoit-Wolf — add to requirements.txt)

Medium-term (architectural changes):
  Task 2  (HRP allocation)
  Task 3  (FinRL turbulence index)
  Task 10 (Alpaca snapshot + news API)

Long-term (ML changes):
  Task 7  (TensorTrade modular reward — RL rewrite)
```

---

*All implementations are conceptual adaptations. No GPL/AGPL code incorporated.*  
*All changes are paper-trading only. No live trading modifications.*  
*License compatibility verified at time of writing (2026-05-20).*
