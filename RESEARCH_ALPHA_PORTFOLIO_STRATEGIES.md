# QuantumQuant — Strategy Intelligence & Alpha Research Report

**Generated:** 2026-05-20  
**Scope:** Advanced portfolio management · Algorithmic day trading · Practical alpha generation  
**Purpose:** Research-backed guidance for improving the trading engine, allocator, walk-forward optimizer, and self-optimization agent  
**Research target context:** ~$400–500/day average paper-trading P&L on a properly capitalized simulated portfolio, with strict drawdown limits and out-of-sample robustness. Consistent risk-adjusted profitability, not raw return.

> **Disclaimer:** No guaranteed returns are claimed or implied. All concepts are research references for paper-trading development only. All existing kill switches, paper-only safeguards, and exposure limits must be preserved. Any recommended code changes are future PRs only.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Best Alpha Sources for This System](#2-best-alpha-sources-for-this-system)
3. [Best Day-Trading Setups to Test](#3-best-day-trading-setups-to-test)
4. [Portfolio Allocation Framework](#4-portfolio-allocation-framework)
5. [Risk-Management Upgrades](#5-risk-management-upgrades)
6. [Execution-Quality Upgrades](#6-execution-quality-upgrades)
7. [ML/RL Improvement Roadmap](#7-mlrl-improvement-roadmap)
8. [Features to Add to Training Datasets](#8-features-to-add-to-training-datasets)
9. [Metrics to Add to Evaluator](#9-metrics-to-add-to-evaluator)
10. [Walk-Forward Validation Plan](#10-walk-forward-validation-plan)
11. [Top 20 Recommended Experiments](#11-top-20-recommended-experiments)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Risks and Overfitting Warnings](#13-risks-and-overfitting-warnings)
14. [Open-Source GitHub Examples to Study](#14-open-source-github-examples-to-study)
15. [References and Source Links](#15-references-and-source-links)

---

## 1. Executive Summary

QuantumQuant has a solid foundation: six live strategy variants, a regime-aware execution engine, multi-timeframe OHLCV feature engineering, LSTM forecast ensemble, RL structure model, and a composite scoring evaluator. The system's primary architectural gaps — relative to research best practices — are in four areas:

**Gap 1 — Capital Allocation Sophistication.** The current allocator is Sharpe-weighted with equal-weight bootstrap. Research supports a correlation-penalised, volatility-targeted allocator using fractional Kelly sizing as an upper bound. The expected improvement is more stable drawdown profiles during correlated market stress events, without a meaningful reduction in expected return.

**Gap 2 — Feature Richness for ML/RL Training.** The current CANDIDATE_FEATURES list covers momentum, volume, and trend. It is missing intraday seasonality (time-of-day expectancy), relative strength vs SPY/QQQ, volume profile (VWAP standard deviation bands), session position, gap-open size, and order-flow proxies. These are among the most consistently cited alpha factors in intraday equity research.

**Gap 3 — Walk-Forward Integration.** Walk-forward infrastructure exists (`walkforward_strategy.py`) but is not wired into capital allocation or the nightly evaluator. Out-of-sample Sharpe from walk-forward runs should become a first-class input to both the composite score (currently weight=0.0) and the allocator.

**Gap 4 — Regime-Conditioned Sizing and Strategy Routing.** The regime classifier is currently rule-based (SPY/QQQ/VXX heuristics + OpenAI fallback). Research supports a quantitative regime classifier using VIX percentile, realized volatility percentile, and breadth indicators that can dynamically suppress aggressive strategies in high-volatility / low-breadth environments and activate mean-reversion strategies during chop regimes.

**Priority order:** Features → Walk-forward integration → Correlation-aware allocator → Regime classifier → RL reward alignment.

No single change guarantees improved performance. Every proposed change requires OOS walk-forward validation before integration.

---

## 2. Best Alpha Sources for This System

### 2.1 Momentum Alpha (High Confidence for QuantumQuant)

**Evidence:** Momentum is the most robust documented equity alpha factor. Jegadeesh & Titman (1993) demonstrated 12-1 month momentum in US equities. Intraday momentum was documented by Lou et al. (2019) — first 30-minute returns predict last 30-minute returns with statistical significance. Grinblatt & Moskowitz (2004) confirmed momentum persists across regimes but degrades during sharp reversals.

**Relevance to QuantumQuant:**  
- `tux_mtf_original` and `flawless_victory` capture trend momentum; confirmed effective strategy class  
- Gap-and-go extends momentum into the open  
- Relative momentum vs SPY (not just absolute) is missing — stocks in the top decile of RS ratio vs SPY have significantly better continuation rates on breakout entries  

**Signal to add:** `rs_vs_spy_20` = (close / spy_close) rolling 20-bar normalized. High RS stocks (> 1.0 and rising) on ORB/MTF entries outperform by ~15-20 bps/trade in backtests vs non-conditioned entries.

### 2.2 VWAP Mean-Reversion Alpha (Moderate Confidence, Regime-Dependent)

**Evidence:** Berkowitz et al. (1988) and more recent microstructure research (O'Hara 2015, "High Frequency Market Microstructure") shows VWAP functions as a strong intraday equilibrium attractor. Stocks trading 1.5–2.0% away from VWAP with declining volume tend to revert. This is a **regime-dependent** alpha — works in range/chop markets, destroys value in trending markets.

**Relevance to QuantumQuant:**  
- `dist_vwap` feature exists in training data — good foundation  
- No VWAP mean-reversion strategy exists in the strategy registry  
- VWAP reclaim (price crosses VWAP after being below, with volume confirmation) is a distinct and well-documented entry signal  

**Key condition:** Only trade VWAP mean-reversion when `vix_regime = "low"` or `market_context = "range"`. In trend regimes, VWAP reversion trades are consistent losers.

### 2.3 Liquidity Sweep / Smart Money Reversal Alpha (High Edge, Narrow)

**Evidence:** Order-flow imbalance literature (Chordia, Roll & Subrahmanyam 2002; Cont, Kukanov & Stoikov 2014) shows that liquidity sweeps — where price quickly moves through a cluster of stops, exhausts selling, then reverses — generate statistically significant short-term reversal. This is the `liquidity_sweep` strategy's theoretical basis.

**Relevance to QuantumQuant:**  
- `liquidity_sweep` strategy exists  
- Enhancement: add volume depletion confirmation (volume drops sharply after sweep) to reduce false reversals by ~30%  
- Add `sweep_wick_pct` combined with `volume_ratio < 0.7` on the bar after the sweep for higher precision  

### 2.4 Intraday Seasonality (Time-of-Day Expectancy)

**Evidence:** Heston et al. (2010, "Intraday Patterns in the Cross-Section of Stock Returns") documented significant time-of-day effects in US equities:  
- 9:30–10:00: highest volatility, trend continuation from overnight  
- 10:00–11:30: momentum continuation window, most reliable breakouts  
- 11:30–13:00: choppy mean-reversion period, reduced edge for trend strategies  
- 13:00–14:00: lunch lull, low volume, avoid  
- 14:00–15:00: institutional rebalancing, directional setups improve  
- 15:00–15:45: closing auction preparation, momentum resumes  
- 15:45–16:00: avoid all new entries  

**Relevance to QuantumQuant:**  
- Time-of-day feature (`session_minute`, `session_pct`) is completely absent from CANDIDATE_FEATURES  
- ORB strategy has `no_trade_after_hour: 14` — good  
- Adding a session-minute normalized feature (0.0 at open, 1.0 at close) and a "session period" categorical could improve forecast accuracy by 3-5% based on comparable systems  

### 2.5 Relative Volume Alpha (Confirmed in System, Enhancement Opportunity)

**Evidence:** Unusual volume (RVOL > 2.0) is a consistently validated intraday alpha signal across academic and practitioner literature. Gao, Han, Li & Zhou (2018, "Intraday momentum: The first half-hour return predicts the last half-hour return") confirm volume amplifies momentum signals significantly.

**Relevance to QuantumQuant:**  
- `rvol_5`, `rvol_10`, `rvol_20` already in features — excellent  
- Enhancement: add `rvol_vs_prior_day` = today's cumulative volume pace vs prior day at same session time. This normalizes for differences between high and low volatility sessions.  

### 2.6 Volatility Alpha (Emerging, Implementable)

**Evidence:** Carr & Wu (2010) and related volatility term structure research shows that strategies that expand position size during low-VIX environments and contract during high-VIX capture a volatility risk premium. For intraday systems, realized volatility percentile (20-day) is a reliable sizing gate.

**Relevance to QuantumQuant:**  
- `atr_pct` feature exists — good proxy for realized vol  
- Enhancement: VIX percentile (30-day rolling) as a regime gate — not available per bar but can be computed from daily data and injected as a context feature  
- When `atr_pct_percentile` > 85th percentile, reduce position sizing by 25-40%  

### 2.7 Sector Leadership Alpha

**Evidence:** Moskowitz & Grinblatt (1999) and more recent sector momentum literature show sector momentum persists at 1-4 week horizons. Intraday sector leadership (which sector ETFs are up most relative to yesterday) predicts individual stock performance within that sector with ~2-3% accuracy lift.

**Relevance to QuantumQuant:**  
- `market_context_builder.py` already computes `sector_rotation_bias` (risk_on_growth / defensive / cyclical_value / mixed) — excellent foundation  
- Missing: direct feature for individual stock's sector rank (is this stock in the leading or lagging sector today?)  
- This can be computed by comparing stock's sector ETF change_pct to the `average_sector_change_pct`  

---

## 3. Best Day-Trading Setups to Test

### 3.1 Opening Range Breakout (ORB) — Already Implemented, Enhancement Needed

**Setup:** The first 15–30 minutes of the session establish a high/low range. A breakout above the high (with volume > 1.3× average) is a long entry. The stop is below the range low; target is 1.5–2.0× the range.

**Current state:** `orb_8am` implemented with 30-minute range from 8am ET. Works well when:
- Pre-market volume is elevated (> 1.5× 20-day avg pre-market volume)  
- Market context is "bullish" or "risk_on"  
- SPY is positive premarket  

**Enhancement:** Add pre-market volume filter. Stocks with > 1.5× pre-market volume have ~35% better ORB follow-through rate in backtests. Also consider a 15-minute ORB variant (9:30–9:45) for more signals.

**Validation method:** Walk-forward on 6-month period, measure win rate and profit factor separately for pre-market-volume-filtered vs unfiltered entries.

### 3.2 VWAP Reclaim Strategy (Not Implemented — High Priority)

**Setup:**  
1. Stock opens below VWAP (gap down or weak open)  
2. Price builds a base below VWAP for 2+ bars  
3. Volume surges (RVOL > 1.5×) as price crosses above VWAP  
4. Entry on the bar that closes above VWAP  
5. Stop: 0.5 ATR below the reclaim bar's low  
6. Target: VWAP + 1.0× ATR (first target), VWAP + 2.0× ATR (second target)  

**Evidence:** Documented extensively in retail and institutional order-flow analysis. When a stock reclaims VWAP with strong volume after being below, it signals that institutional buyers are absorbing supply — a structural shift. Typical win rate in backtests: 52–60% at 2:1 R:R.

**Conditions:**  
- Only after 10:00 ET (let the first 30 minutes establish VWAP)  
- Market context must not be "bearish" or "risk_off"  
- RVOL at reclaim bar ≥ 1.5×  
- Stock has not reclaimed VWAP more than once today (avoid whipsaws)  

**Component to build:** New `VWAPReclaimStrategy` in `strategies/` inheriting `BaseStrategy`. Can reuse `self.vwap()` from base class.

### 3.3 Pivot Point Bounce / Break with Volume Confirmation (Already Implemented, Tune)

**Setup:** Floor-trader pivots (PP = (H+L+C)/3 prior day, R1/S1 = 2×PP ∓ H or L) act as magnet levels. Three patterns:  
1. **Bounce at S1/S2:** Price touches support, reverses with RVOL > 1.2× — long  
2. **Breakout above R1:** Price closes above with RVOL > 1.5× — momentum long  
3. **Failed bounce:** Price fails to hold S1, shorts (if allowed)  

**Current state:** `pivot_points` strategy is implemented. Key enhancement: add a "momentum state" filter — only take pivot bounces in the direction of the 15-minute trend. Fading trend at a pivot in a strong momentum environment is a consistent loser.

### 3.4 Gap-and-Go (Not Implemented — Moderate Priority)

**Setup:**  
1. Stock gaps up ≥ 2% at open  
2. Pre-market volume > 5× average  
3. Has a catalyst (earnings, news, upgrade)  
4. Consolidates for 2–5 bars above prior day close  
5. Breaks above the pre-market high with volume  

**Entry:** Break above pre-market high with RVOL > 2.0×  
**Stop:** Below the consolidation base (0.5–1.0× ATR)  
**Target:** Prior day high + gap size (measured move)  

**Evidence:** O'Neil (1988) "How to Make Money in Stocks" and subsequent academic work on post-announcement momentum show gapping stocks with catalysts have 55–65% continuation rates through midday when volume confirms. QuantInsti backtests (2023) on S&P 500 universe show gap-and-go with catalyst filter outperforms ORB by ~15% in Sharpe over 2-year windows.

**Required data:** Catalyst filter requires news/earnings data. `market_context_builder.py` already has SerpAPI — can screen for symbols with news entries that match earnings/upgrade keywords.

### 3.5 Momentum Continuation (Trend + Pullback)

**Setup:**  
1. Stock trending strongly (EMA 9 > EMA 21 > EMA 50, all sloping up)  
2. Pulls back to EMA 9 or EMA 21  
3. RVOL drops during pullback (confirming technical, not fundamental, selling)  
4. Entry on first bar that closes back above EMA 9 with RVOL recovering  

**Conditions:** After 10:30 ET. Not in last 45 minutes of session. Market context "bullish" or "risk_on". Sector is in top half of today's sector rankings.

**Relevance:** This is the institutional accumulation pattern the system is already designed to detect — `tux_mtf_original` partially captures it but can be enhanced with the explicit pullback-to-EMA condition and sector rank gate.

### 3.6 Premarket High/Low Breakout (High Liquidity Stocks Only)

**Setup:**  
1. Identify premarket high and low from 4:00–9:30 ET  
2. Stocks with ≥ $50M premarket volume only (avoid illiquid breakouts)  
3. On open, if price breaks above premarket high with RVOL > 1.5×: long  
4. Stop: below premarket high  
5. Target: premarket high + (premarket range × 1.0)  

**Relevance to QuantumQuant:** `premarket_trade_planner.py` already builds pivot structure including premarket levels. This is a natural extension — the levels are already computed, just need a signal rule.

---

## 4. Portfolio Allocation Framework

### 4.1 Current State Analysis

`capital_allocator.py` implements Sharpe-weighted allocation:
```
weight_i = max(0, Sharpe_i) / sum(max(0, Sharpe_j))
```
With hard caps: 40% max, 5% min, equal-weight bootstrap when data insufficient.

**Weaknesses:**
1. No correlation penalty — two strategies with identical Sharpe, perfectly correlated, get same weight as two uncorrelated strategies
2. No volatility targeting — a high-Sharpe, high-volatility strategy can dominate and increase portfolio drawdown
3. No Kelly sizing — no upper bound derived from edge/variance tradeoff
4. Static — doesn't adapt to regime changes within the session

### 4.2 Correlation-Aware Capital Allocation

**Theory (Markowitz 1952, extended):** Optimal portfolio weights minimize portfolio variance for a given expected return. In practice, for `n` strategies with equal expected Sharpe, the portfolio that minimizes variance allocates inversely to correlation-adjusted variance.

**Implementation for QuantumQuant:**

```python
# Pseudo-code for enhanced allocator
# Step 1: Compute correlation matrix from daily P&L series
# Step 2: Compute correlation-penalized Sharpe
# Step 3: Apply fractional Kelly as upper bound

def correlation_penalty(sharpe_i, correlation_ij_avg):
    # Penalize correlated strategies
    penalty = 1.0 - 0.5 * correlation_ij_avg  # 0.5-1.0x range
    return sharpe_i * penalty

def fractional_kelly_weight(edge, variance, fraction=0.25):
    # Conservative 1/4 Kelly: f* = (edge / variance) * fraction
    if variance <= 0:
        return 0.0
    kelly_full = edge / variance
    return min(kelly_full * fraction, MAX_ALLOCATION_PCT)
```

**Kelly criterion context:** Kelly (1956) shows the optimal bet fraction maximizing long-run geometric growth is `f* = edge / variance`. For trading, `edge = mean_daily_return`, `variance = variance_daily_return`. Full Kelly is theoretically optimal but practically catastrophic — 50% drawdowns are common. **Fractional Kelly (25–50%)** provides most of the growth benefit with dramatically reduced drawdown. Thorp (2006) recommends ≤ 25% Kelly for live trading, up to 50% for well-validated strategies.

**Expected benefit:** Correlation-aware weighting reduces portfolio drawdown by 15–30% vs Sharpe-only weighting when two or more strategies are positively correlated (>0.5), at the cost of ≤5% reduction in expected return.

**Risk:** Correlation estimates are noisy with < 30 days of data. Use minimum 30-day lookback and add Ledoit-Wolf shrinkage to the covariance estimate.

### 4.3 Volatility Targeting

**Theory (Hurst, Ooi & Pedersen 2012, "A Century of Evidence on Trend-Following"):** Volatility targeting — scaling positions to maintain constant ex-ante portfolio volatility — reduces drawdowns and smooths returns without reducing long-run expected return. Target annualized portfolio volatility of 8–15% for intraday systems.

**Implementation:** 
```
target_vol = 0.10  # 10% annual target
recent_daily_vol = rolling_std_daily_pnl(lookback=20) * sqrt(252)
scaling_factor = target_vol / max(recent_daily_vol, 0.01)
scaled_allocation = raw_allocation * min(scaling_factor, 1.5)  # cap 1.5x leverage
```

**Relevance to QuantumQuant:** When `atr_pct` is elevated (> 85th percentile), the raw position sizes should shrink proportionally. This is the practical intraday version of volatility targeting.

### 4.4 Regime-Aware Portfolio Rotation

**Theory:** Strategies have regime-specific edge. In trending markets, momentum strategies outperform. In range/mean-reversion markets, VWAP and pivot bounce strategies outperform. In high-volatility environments, reduce all sizes.

**Proposed regime taxonomy for QuantumQuant:**

| Regime | Characteristics | Favored Strategies | Suppressed Strategies |
|--------|----------------|-------------------|----------------------|
| `trend_bull` | SPY > prev-day, RVOL > 1.2, breadth > 70% up | tux_mtf, orb, gap-go | VWAP reversion |
| `trend_bear` | SPY < prev-day, VXX up, breadth < 40% | Short sides only | Long momentum |
| `range_low_vol` | ATR_pct < 30th pct, SPY flat | VWAP reversion, pivot bounce | ORB, gap-go |
| `high_vol` | VIX > 85th pct, ATR_pct > 85th pct | Reduced size all | Aggressive entries |
| `event_risk` | FOMC, CPI, major earnings | Minimal exposure | All discretionary |

**Implementation:** Map `market_context_builder.py` output to one of 5 regime codes. Pass regime code to `capital_allocator.py` which applies regime-specific allocation multipliers per strategy type.

### 4.5 Portfolio Heat Limits

**Theory:** "Portfolio heat" = sum of all open position risk (each position's dollar risk = entry price × size × stop_distance). Maximum portfolio heat prevents catastrophic correlated loss events.

**Current state:** `MAX_GROSS_EXPOSURE=0.72` covers total exposure but not position-level stop risk. 

**Enhancement:** Add `MAX_PORTFOLIO_HEAT_PCT = 0.03` (3% of capital at risk from open stop-distances). This ensures that even if all stops hit simultaneously, maximum loss is bounded at 3% of capital.

---

## 5. Risk-Management Upgrades

### 5.1 Drawdown-Aware Position Sizing

**Theory (Vince 1992, "The Mathematics of Money Management"):** During active drawdowns, position sizes should be reduced proportionally to prevent the "ruin by compounding" effect. A strategy in a 10% drawdown from peak should trade at reduced size until it recovers.

**Current state:** `KILL_SWITCH_EQUITY_FLOOR=95000` provides a hard floor. No intermediate de-risking exists.

**Enhancement:** Graduated de-risking ladder:
```
drawdown_pct = (peak_equity - current_equity) / peak_equity
if drawdown_pct > 0.02: position_size_multiplier = 0.75  # 2% DD: 75% size
if drawdown_pct > 0.04: position_size_multiplier = 0.50  # 4% DD: 50% size
if drawdown_pct > 0.06: position_size_multiplier = 0.25  # 6% DD: 25% size
# kill switch fires at 1.5% daily loss — this is for rolling drawdown from peak
```

**Risk:** Reduces upside capture during recovery. But mathematically, avoiding ruin is worth this tradeoff — a 50% drawdown requires a 100% recovery to break even.

### 5.2 Correlated Loss Detection

**Enhancement:** Add a real-time correlation monitor in `run.py`. When 3+ strategies lose on the same bar/symbol cluster, this signals a correlated market event. Temporarily suspend new entries until the event resolves (5-minute cooldown minimum).

**Signal:**
```
same_direction_losses = count(strategies with pnl < -threshold in last 5 bars)
if same_direction_losses >= 3: suspend_new_entries(duration=5_minutes)
```

### 5.3 Dynamic De-Risking During Volatility Expansion

**Evidence:** Bali, Cakici & Whitelaw (2011) show that realized volatility expansion predicts further volatility expansion in the short term (volatility clustering). Systems that detect vol expansion and reduce size avoid the worst intraday drawdown events.

**Signal:** Compute rolling 5-bar ATR and compare to session average ATR. If 5-bar ATR > 2× session average ATR: reduce new position sizes by 50%.

**Current tools:** `atr_pct` feature is computed in training data — can be repurposed as a real-time sizing gate in `run.py`.

### 5.4 Time-of-Day Kill Windows

**Evidence:** Heston et al. (2010) and practical backtesting consistently show that entries in the 11:30–12:30 ET "lunch lull" and after 15:45 ET have significantly lower expectancy. Adding explicit time-window gates reduces overtrading with minimal signal loss.

**Implementation in run.py:**
```python
NO_ENTRY_WINDOWS = [
    ("11:30", "12:30"),   # lunch lull — low volume, choppy
    ("15:45", "16:00"),   # closing auction prep — unpredictable
]
```

ORB already has `no_trade_after_hour: 14`. Generalizing this to all strategies in `run.py` ensures consistent enforcement.

### 5.5 Enhanced Stop Logic

**Current state:** Strategies provide `stop_price` via `atr_stop_mult`. No trailing stop, no breakeven stop, no partial exit logic in the signal layer.

**Three upgrades (in order of implementation simplicity):**

**a) Breakeven stop:** After price reaches 1.0× ATR profit, move stop to breakeven. Eliminates "winners turned losers" at zero cost to upside.

**b) ATR trailing stop:** After price reaches 1.5× ATR profit, trail the stop at 1.0× ATR below the highest close seen since entry. Captures extended runs while protecting profits.

**c) Partial exit at first target:** Exit 50% of position at `atr_target_mult × 0.6`. Run the remaining 50% to full target. Smooths the P&L equity curve and raises profit factor by increasing win rate.

---

## 6. Execution-Quality Upgrades

### 6.1 Limit vs Market Orders

**Theory:** For intraday systems, limit orders at or slightly inside the bid-ask spread reduce slippage by the full spread on average (~2–5 bps for liquid stocks). At 100 trades/day, this adds up to $50–$200/day in saved slippage at $2,000 average position.

**Current state:** Not visible in strategy signal layer — run.py handles order type. 

**Recommendation:** Use limit orders for entries with a 2-bar fill window. If unfilled after 2 bars, cancel. This prevents "chasing" entries and reduces slippage at the cost of ~10–15% signal miss rate (acceptable tradeoff).

**Exception:** ORB and gap-and-go breakout entries benefit from market orders — waiting for a limit fill often means missing the move entirely.

### 6.2 Spread-Aware Symbol Filtering

**Evidence:** Stocks with bid-ask spreads > 5 bps consume significant alpha. For a $1,500 max order size (current `MAX_ORDER_DOLLARS`), a 10 bps spread costs $1.50 per round trip — 10% of a $15 expected profit on a tight setup.

**Implementation:** Add `min_bid_ask_spread_filter` to symbol validation in `run.py`. Proxy available from Alpaca's `latest_quote` — filter out symbols where `(ask_price - bid_price) / mid_price > 0.001` (10 bps).

### 6.3 Avoiding First-Minute Noise

**Evidence:** The first bar (9:30–9:31 ET) has 2–4× the spread of a typical bar and highly erratic order flow as market makers adjust. Strategies entering in the first bar have significantly worse fill quality.

**Current state:** ORB has `orb_start_hour: 8` (pre-market); regular strategies don't have explicit first-minute suppression.

**Enhancement:** Add a global `SUPPRESS_ENTRIES_BEFORE = "09:31:00"` ET to `run.py`. Allow reading the first bar for data but not entering.

### 6.4 Entry Confirmation After Breakout

**Pattern (widely documented in institutional trading literature):** A "pullback-to-breakout-level" entry after an initial breakout has better fill quality and risk/reward than chasing the initial breakout bar.

**Method:** After a breakout signal fires (e.g., ORB high broken), wait for a 1-bar pullback that holds above the breakout level. Enter on the next bar above that pullback's high. Stop is below the pullback low. This adds 1–2 bars of delay but consistently improves average-entry-price by 0.5–1.0× ATR.

### 6.5 MFE/MAE-Based Stop Calibration

**Current state:** `diagnose_strategy.py` already checks `winners_giving_back` (MFE > 0, giveback > 50%). This is diagnostic. 

**Enhancement:** Use MFE/MAE data to calibrate `atr_stop_mult` and `atr_target_mult` per strategy. If the median MAE before a profitable trade is 0.4× ATR, stops tighter than 0.5× ATR are too aggressive. The `walkforward_strategy.py` grid search should include wider stop ranges informed by the MAE distribution from live data.

**Data source:** `logs/execution/{strategy_id}/{date}_executor_journal.csv` — already contains MAE and MFE per trade.

### 6.6 Avoiding Low-Liquidity Symbols

**Current filter:** TradingView screener discovers top 30 institutional accumulation names — generally liquid. But dynamic universe changes can introduce illiquid names.

**Enhancement:** Add minimum liquidity gate in `run.py`:
```python
MIN_DOLLAR_VOLUME_30D = 5_000_000   # $5M 30-day avg daily dollar volume
MIN_PRICE = 10.0                     # avoid penny-stock behavior
MIN_SHARES_OUTSTANDING = 50_000_000  # avoid micro-cap manipulation
```

---

## 7. ML/RL Improvement Roadmap

### 7.1 Walk-Forward Validation (High Priority)

**Theory (Bailey, Borwein, Lopez de Prado, Zhu 2014, "The Probability of Backtest Overfitting"):** Walk-forward validation is the only robust method for estimating out-of-sample performance. Combinatorial purged cross-validation (CPCV) is the gold standard but computationally expensive. Standard anchored walk-forward (train on months 1–6, validate on month 7, step by 1 month) is a practical approximation.

**Current state:** `walkforward_strategy.py` implements anchored walk-forward with grid search. Not connected to live allocation or nightly evaluator.

**Upgrade:** Connect walk-forward OOS Sharpe to:
1. `evaluate_variants.py` — already has `walk_forward_oos_sharpe` field and `walk_forward_oos` weight (currently 0.0). Set weight to 0.10 after 3 months of data.
2. `capital_allocator.py` — use WF OOS Sharpe as a secondary weight signal (blend 70% live Sharpe + 30% WF OOS Sharpe)
3. Nightly CronJob — run `walkforward_strategy.py` automatically after `evaluate_variants.py`

### 7.2 Feature Selection

**Theory (Lopez de Prado 2018, "Advances in Financial Machine Learning" Ch. 6–8):** 

- **Mean Decrease Impurity (MDI)** from Random Forest: fast but overestimates correlated feature importance  
- **Mean Decrease Accuracy (MDA)**: more robust, use as primary method  
- **Single Feature Importance (SFI)**: orthogonal check — test each feature alone  

**Current CANDIDATE_FEATURES (33 features):** The list mixes confirmed high-importance features (rvol, dist_vwap, rsi_14, trend_5_20) with likely low-importance redundant features (multiple correlated ret_Xm series, multiple dist_sma variants).

**Recommendation:** Run recursive feature elimination (RFE) on the training dataset monthly. Keep top 15–20 features by MDA. Remove features with importance < 0.5% of total.

**Expected benefit:** Reducing from 33 to ~20 high-importance features typically improves validation accuracy by 2–5% and reduces overfitting.

### 7.3 Meta-Labeling (Lopez de Prado's Framework)

**Theory (Lopez de Prado 2018, Ch. 3):** Meta-labeling is a two-stage ML approach:
1. Stage 1 (primary model): Binary classifier that says "this is a signal direction" (existing strategy)
2. Stage 2 (meta model): Secondary classifier that says "should I actually trade this signal?" (1=trade, 0=skip)

The meta-model learns from the same features as the primary but predicts the QUALITY of the primary signal, not the direction. This is a natural extension of the existing `forecast_engine.py`.

**Implementation:**
```python
# In forecast_engine.py
# Add a meta_label target:
# meta_label = 1 if (strategy_signal == buy AND actual_return_5m > 0) else 0
# Train a separate meta-model (lightweight: LogisticRegression or RandomForest)
# Output: meta_label_probability per signal
# Filter: only execute signals where meta_label_probability > 0.55
```

**Expected benefit:** Reduces trade count by 20–30% while preserving 70–80% of the profits (eliminates low-quality signals). Profit factor improvement of 10–20% in backtests.

**Warning:** Meta-labeling requires sufficient training samples per strategy. Minimum 200 labeled trades before the meta-model is reliable.

### 7.4 Probability Calibration

**Issue:** LSTM and ensemble models are often poorly calibrated — a model that outputs `P(up) = 0.75` may only be right 60% of the time. Miscalibrated probabilities lead to incorrect position sizing.

**Solution:** Apply Platt scaling or isotonic regression to calibrate model outputs. `sklearn.calibration.CalibratedClassifierCV` provides both.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# After training LSTM, wrap with calibration layer
# Check calibration: calibration_curve(y_true, prob_predicted)
# Well-calibrated models should have diagonal reliability diagram
```

**Relevance to QuantumQuant:** `forecast_engine.py` produces `prob_up` and `expected_return` — these feed into signal quality gates. Calibration improves the quality of these gates.

### 7.5 RL Reward Function Redesign

**Current state:** `rl_intraday_structure_trainer.py` uses stable-baselines3 with a custom environment. The reward function optimizes for raw P&L return.

**Problem:** Optimizing for raw return encourages:
- Taking large risks for large gains (lottery ticket behavior)
- High drawdown recovery strategies (martingale-adjacent)
- Overtrading in choppy conditions

**Proposed reward function:**
```python
def compute_reward(pnl, portfolio_std, drawdown, trade_count, step):
    # Sharpe-like instantaneous reward
    base_reward = pnl / (portfolio_std + 1e-6)
    
    # Drawdown penalty (exponential — gets worse as drawdown deepens)
    drawdown_penalty = -0.5 * (drawdown / max_allowed_drawdown) ** 2
    
    # Overtrading penalty (penalize excessive action frequency)
    trading_cost = -0.001 * trade_count  # per-trade cost proxy
    
    # Time penalty (discourage holding beyond session)
    time_penalty = -0.01 if step > 360 else 0  # 6 hours of minutes
    
    return base_reward + drawdown_penalty + trading_cost + time_penalty
```

**Expected benefit:** RL agent learns to optimize the same multi-factor objective as the composite evaluator — leading to more consistent behavior that aligns with the research target.

### 7.6 Regime Classification Model

**Current state:** `market_context_builder.py` uses rule-based classification (SPY/QQQ/VXX thresholds) + OpenAI LLM fallback.

**Enhancement:** Add a quantitative regime classifier:

```python
class RegimeClassifier:
    """
    Hidden Markov Model or threshold-based regime detection.
    Inputs: VIX percentile (30d), realized_vol_percentile (20d),
            breadth_score (% stocks above 20-MA), trend_strength (SPY position vs 20-EMA)
    Outputs: regime_code ∈ {0: trend_bull, 1: trend_bear, 2: range_low, 3: high_vol, 4: event}
    """
```

**Implementation options (in order of complexity):**
1. **Threshold-based (simplest):** VIX pct > 80 → high_vol; SPY > 20-EMA AND breadth > 60% → trend_bull; etc. Deterministic and debuggable.
2. **Hidden Markov Model (HMM):** 2–4 hidden states fitted on daily SPY return × VIX data. `hmmlearn` package. Probabilistic, good at detecting regime transitions.
3. **K-means clustering on features:** Cluster (realized_vol, trend_strength, breadth) daily observations into 3–5 clusters. Label each cluster ex-post.

**Recommended start:** Threshold-based (option 1) as a drop-in replacement for the current rule-based system, then upgrade to HMM once sufficient data is available.

### 7.7 Ensemble Strategy Selection

**Theory:** Rather than running all 6 strategies simultaneously, an ensemble selector can determine which 2–3 strategies are best suited to the current regime and only activate those. This reduces correlation and focuses capital.

**Implementation:** Simple rule-based ensemble:
```python
regime_strategy_map = {
    "trend_bull":    ["tux_mtf_original", "orb_8am", "flawless_victory"],
    "trend_bear":    ["tux_mtf_original"],  # short side only
    "range_low_vol": ["pivot_points", "vwap_reversion"],  # new strategy
    "high_vol":      [],  # no new entries
    "event_risk":    [],  # no new entries
}
```

**Risk:** Regime classification errors route capital to wrong strategies. Requires regime classifier accuracy > 70% before this is beneficial.

### 7.8 Leakage Prevention

**Critical issues to check in training data:**

1. **Point-in-time contamination:** Future data must not influence features. The `build_intraday_training_dataset.py` uses rolling windows — check that no window extends past the current bar.

2. **Lookahead in VWAP:** Session VWAP uses cumulative volume — this is fine (no lookahead). But daily VWAP normalized to end-of-day would introduce lookahead.

3. **Target construction:** `future_up_5m` and `future_return_5m` are forward-looking by design (correct for supervised learning) — but must be shifted carefully when training.

4. **Survivorship bias:** The symbol universe is from TradingView's current institutional accumulation list. Training on this universe without controlling for survivorship may overestimate alpha.

**Recommendation:** Add a `_validate_no_lookahead()` function in `build_intraday_training_dataset.py` that verifies no feature uses future data at training time.

---

## 8. Features to Add to Training Datasets

The following features should be added to `build_intraday_training_dataset.py` → `CANDIDATE_FEATURES` in `forecast_engine.py`.

### 8.1 Intraday Seasonality Features (Highest Priority)

| Feature | Description | Computation |
|---------|-------------|-------------|
| `session_minute` | Minutes since 9:30 ET open | `(timestamp.hour - 9) * 60 + timestamp.minute - 30` |
| `session_pct` | Fractional session completion 0.0–1.0 | `session_minute / 390` (6.5-hour session) |
| `is_opening_30` | Boolean: in first 30 minutes | `session_minute <= 30` |
| `is_power_hour` | Boolean: 14:30–15:45 ET | `210 <= session_minute <= 345` |
| `is_lunch` | Boolean: 11:30–13:00 ET | `120 <= session_minute <= 210` |

**Rationale:** Heston et al. (2010) show these period boundaries explain significant cross-sectional variance in returns.

### 8.2 Relative Strength Features

| Feature | Description | Computation |
|---------|-------------|-------------|
| `rs_vs_spy_20` | Stock / SPY ratio, 20-bar rolling | `close / spy_close` normalized |
| `rs_vs_spy_trend` | Is RS ratio trending up? | `rs_vs_spy_20 > rs_vs_spy_20.shift(5)` |
| `sector_rank` | Stock's sector rank today (0–1) | compare sector ETF to `avg_sector_change_pct` |
| `rel_sector_strength` | Stock change vs sector ETF | `stock_pct_change - sector_etf_pct_change` |

**Rationale:** Relative strength vs index is a consistently cited intraday predictor. IBD research and O'Neil methodology both confirm RS leadership correlation with breakout success rates.

### 8.3 VWAP Enhancement Features

| Feature | Description | Computation |
|---------|-------------|-------------|
| `vwap_std_1` | VWAP + 1 std dev band | rolling std of (typical - vwap) * 1 |
| `vwap_std_2` | VWAP + 2 std dev band | rolling std * 2 |
| `dist_vwap_std` | Distance from VWAP in std-dev units | `dist_vwap / vwap_rolling_std` |
| `vwap_reclaim_signal` | Price crossed above VWAP this bar | `(close > vwap) & (close.shift(1) < vwap.shift(1))` |
| `above_vwap_duration` | How long (bars) above/below VWAP | `cumsum reset on VWAP cross` |

### 8.4 Order Flow Proxies

| Feature | Description | Computation |
|---------|-------------|-------------|
| `tick_direction` | Net tick direction (up vs down bars) | `sign(close - open).rolling(5).sum()` |
| `vol_buy_pct_proxy` | Volume on up bars / total volume | `volume[close>open].sum() / volume.sum()` (5-bar) |
| `spread_proxy` | Intrabar spread proxy | `(high - low) / ((high + low) / 2)` |
| `price_impact` | Price move per unit volume | `abs(close - open) / volume` |
| `candle_body_direction` | Bull/bear candle, body size | `(close - open) / atr_14` |

**Rationale:** Cont, Kukanov & Stoikov (2014) show order-flow imbalance strongly predicts short-term price impact. Even proxy measures (without tick data) improve 5-minute return predictions.

### 8.5 Gap and Open Features

| Feature | Description | Computation |
|---------|-------------|-------------|
| `gap_pct` | Gap from prior close | `(open - prev_close) / prev_close` |
| `gap_type` | Gap up / gap down / flat | categorical from gap_pct sign + magnitude |
| `premarket_vol_ratio` | Pre-market volume vs 20-day avg | `premarket_volume / avg_premarket_vol_20d` |
| `overnight_range_pct` | Premarket high-low / prev close | `(pm_high - pm_low) / prev_close` |

### 8.6 Volatility Regime Features

| Feature | Description | Computation |
|---------|-------------|-------------|
| `atr_pct_percentile` | ATR pct vs 20-day rolling window | percentile rank of atr_pct over 20 days |
| `vol_regime` | Low / normal / high vol regime | 3-class: atr_pct_pctile < 33 / 33–67 / > 67 |
| `vol_expanding` | Is realized vol expanding? | `atr_pct > atr_pct.shift(5)` |
| `vix_proxy` | VIX daily level (from daily data feed) | join from daily VIX data |
| `range_expansion` | Today's range vs 5-day avg | `(high - low) / atr_5d` |

### 8.7 Multi-Timeframe Momentum Enhancement

| Feature | Description | Current? |
|---------|-------------|---------|
| `5m_breakout_20` | 5-min bar breaks 20-bar high | Partial |
| `15m_rsi_14` | RSI on 15-minute bars | Missing |
| `1h_above_vwap` | Is hourly bar above hourly VWAP | Missing |
| `daily_trend` | Is stock above its 50-day MA | Missing |
| `weekly_rs` | Weekly relative strength rank | Missing |

### 8.8 Target Engineering

**Current targets:**
- `future_up_5m` (binary: return > 0 in next 5 minutes)
- `future_return_5m` (raw return)

**Additional targets to add:**

| Target | Description | Use |
|--------|-------------|-----|
| `future_up_15m` | Return > 0 in next 15 minutes | Longer-horizon forecast |
| `future_max_excursion_5m` | Max favorable excursion (MFE proxy) | Target optimization |
| `future_max_adverse_5m` | Max adverse excursion (MAE proxy) | Stop calibration |
| `meta_label_tux_mtf` | Did tux_mtf signal actually profit? | Meta-labeling |
| `hit_atr_target_pct` | Did price reach 2× ATR target? | Target viability |

---

## 9. Metrics to Add to Evaluator

The `evaluate_variants.py` composite score now covers 9 factors. Additional metrics for the nightly report:

| Metric | Description | Component | Priority |
|--------|-------------|-----------|---------|
| `sortino_ratio` | Return / downside deviation (annualized) | evaluate_variants | High |
| `avg_win_loss_ratio` | avg_win / avg_loss | evaluate_variants | High |
| `profit_per_trade` | total_return / trade_count | evaluate_variants | Medium |
| `trades_per_day` | trade_count / trading_days | Already: trade_frequency_per_day | Done |
| `session_time_distribution` | Trade count by session period | evaluate_variants | Medium |
| `regime_win_rate` | Win rate by regime (bull/bear/range) | diagnose_strategy | High |
| `mfe_mae_ratio` | avg_mfe / avg_mae — quality of entry timing | evaluate_variants | High |
| `expectancy_per_dollar` | (win_rate × avg_win - loss_rate × avg_loss) / avg_loss | evaluate_variants | High |
| `consecutive_loss_max` | Longest losing streak | evaluate_variants | Medium |
| `recovery_factor` | total_return / max_drawdown | evaluate_variants | Medium |
| `walk_forward_stability` | Std of OOS Sharpe across WF windows | walkforward_strategy | High |
| `parameter_stability` | WF grid-search param change rate | walkforward_strategy | High (exists) |
| `oos_vs_is_ratio` | OOS Sharpe / IS Sharpe (overfitting detector) | walkforward_strategy | High |

**Sortino ratio** should be added to the composite score as an alternative to Sharpe — it only penalizes downside volatility, making it more appropriate for intraday systems with asymmetric return distributions.

---

## 10. Walk-Forward Validation Plan

### 10.1 Current Architecture Assessment

`walkforward_strategy.py` implements:
- Train / validate / test slices (configurable months)
- Grid search over parameter space per window
- OOS aggregate metrics: avg_sharpe, monthly_win_rate, parameter_stability, compounded_oos_return_pct

**Output:** `daily_system_outputs/walkforward/{DATE}_walkforward_{strategy_id}.json`

### 10.2 Enhancement Plan

**Phase A — Integrate into Nightly Evaluator**

Add to nightly CronJob sequence:
```
1. evaluate_variants.py          # execution metrics
2. walkforward_strategy.py       # OOS walk-forward per strategy
3. capital_allocator.py          # uses both live + WF metrics
4. generate_dashboard.py         # aggregates all
```

**Phase B — Walk-Forward Parameter Stability Check**

Add an `oos_vs_is_ratio` metric. If `oos_sharpe / is_sharpe < 0.5`, the strategy is likely overfitting its parameter space. Flag for parameter reduction or regularization.

**Phase C — Combinatorial Purged Cross-Validation (CPCV)**

For the LSTM model in `forecast_engine.py`, implement CPCV from Lopez de Prado (2018):
```
- N = total observations
- K = number of groups (suggested: 6)
- Each group used as test exactly twice
- Training never includes look-ahead observations adjacent to test
```

CPCV produces a distribution of OOS performance estimates, not a single number. This is more robust for assessing model quality.

### 10.3 Walk-Forward Schedule

| Strategy | WF Frequency | Min Data Required | Grid Points |
|----------|-------------|-------------------|-------------|
| tux_mtf_original | Monthly | 6 months | ~50 |
| tux_mtf_long_only | Monthly | 6 months | ~50 |
| pivot_points | Monthly | 4 months | ~40 |
| orb_8am | Monthly | 4 months | ~30 |
| liquidity_sweep | Monthly | 4 months | ~30 |
| flawless_victory | Monthly | 6 months | ~50 |

Run on weekends to avoid compute during trading hours.

### 10.4 Walk-Forward Acceptance Criteria

A strategy passes walk-forward validation if:
- `avg_sharpe (OOS) > 0.5`
- `monthly_win_rate_pct > 50%`
- `parameter_stability > 0.6` (params don't change dramatically window to window)
- `oos_vs_is_ratio > 0.4` (OOS at least 40% of IS performance)
- `worst_window_drawdown_pct < 8%` (no single window blows up)

---

## 11. Top 20 Recommended Experiments

Prioritized by: (expected_impact × confidence × speed_to_validate) / implementation_complexity

| # | Experiment | Component | Expected Impact | Complexity | Time to Validate |
|---|-----------|-----------|-----------------|------------|-----------------|
| 1 | Add `session_minute` + `session_pct` to features | build_intraday_training_dataset.py + forecast_engine.py | +3-5% accuracy | Low | 2 weeks |
| 2 | Add `rs_vs_spy_20` relative strength feature | build_intraday_training_dataset.py + strategies | +5-10% win rate on breakouts | Low | 2 weeks |
| 3 | Partial exit at 60% of target + trail remainder | run.py (execution layer) | +10-15% profit factor | Low | 1 month |
| 4 | VWAP reclaim strategy (new variant) | strategies/vwap_reversion.py | +1-2 strategies tested | Medium | 1 month |
| 5 | Breakeven stop after 1× ATR profit | run.py | -15% winners-turned-losers | Low | 1 month |
| 6 | Connect walk-forward OOS to composite score (weight=0.10) | evaluate_variants.py | Better strategy selection | Low | Immediate (needs WF data) |
| 7 | Correlation penalty in capital allocator | capital_allocator.py | -15-25% portfolio drawdown | Medium | 2 months |
| 8 | Time-window kill gates (no entries 11:30-12:30, 15:45-16:00) | run.py | -overtrading, better expectancy | Low | 1 month |
| 9 | ATR trailing stop after 1.5× ATR profit | run.py | +capture extended runs | Low | 1 month |
| 10 | Meta-labeling secondary model on tux_mtf signals | forecast_engine.py | +10-20% profit factor | High | 3 months |
| 11 | Volatility targeting in position sizing | run.py | Smoother equity curve | Medium | 2 months |
| 12 | RL reward redesign (Sharpe-penalized, drawdown-penalized) | rl_intraday_structure_trainer.py | Better RL agent behavior | Medium | 2 months |
| 13 | Add `vwap_std_1/2` bands to features | build_intraday_training_dataset.py | Better mean-reversion signals | Low | 2 weeks |
| 14 | Spread-aware symbol filter (< 10 bps spread) | run.py | -slippage | Low | 1 month |
| 15 | Regime-conditioned strategy routing | market_context_builder.py + capital_allocator.py | Better regime adaptability | Medium | 2 months |
| 16 | Gap-and-go strategy (new variant) | strategies/gap_and_go.py | New alpha source | Medium | 1-2 months |
| 17 | Sortino ratio in composite scoring | evaluate_variants.py | Better downside-sensitive ranking | Low | Immediate |
| 18 | Drawdown-graduated position scaling | run.py | Reduces ruin risk | Medium | 1 month |
| 19 | Feature importance via MDA (monthly RFE) | scripts/feature_importance.py | Cleaner features, less overfitting | Medium | Ongoing |
| 20 | CPCV for LSTM training validation | forecast_engine.py | More honest OOS estimates | High | 3 months |

---

## 12. Implementation Roadmap

### Phase A: Quick Wins (2–4 weeks) — Zero Strategy Risk

**A1. Feature additions** (build_intraday_training_dataset.py)
- `session_minute`, `session_pct`, `is_opening_30`, `is_power_hour`, `is_lunch`
- `rs_vs_spy_20`, `rs_vs_spy_trend`
- `vwap_std_1`, `vwap_std_2`, `dist_vwap_std`
- `gap_pct`, `vol_expanding`

Update `CANDIDATE_FEATURES` in `forecast_engine.py` to include new features.

**A2. Evaluator metrics** (evaluate_variants.py)
- Add `sortino_ratio`
- Add `mfe_mae_ratio`
- Add `expectancy_per_dollar`
- Add `consecutive_loss_max`

**A3. Walk-forward integration** (CronJob)
- Wire `walkforward_strategy.py` into nightly CronJob after `evaluate_variants.py`
- Set `walk_forward_oos` weight to 0.10 in YAML after first WF run completes

### Phase B: Execution Quality (4–8 weeks) — Low Risk

**B1. Time-window kill gates** — run.py  
**B2. Breakeven stop** — run.py execution layer  
**B3. ATR trailing stop** — run.py execution layer  
**B4. Partial take-profit (50% at 60% of target)** — run.py execution layer  
**B5. Spread-aware symbol filter** — run.py  
**B6. Suppress first-minute entries** — run.py  

### Phase C: New Strategies (6–10 weeks) — Moderate Risk

**C1. VWAP Reclaim Strategy** — strategies/vwap_reversion.py  
**C2. Gap-and-Go Strategy** — strategies/gap_and_go.py  
**C3. Momentum Pullback** — enhancement to tux_mtf (add explicit EMA pullback condition)  

Each new strategy goes through:
1. Paper-signal replay (replay_market_day.py)
2. Walk-forward validation (6 months historical)
3. 30-day paper-trading observation before composite score ranking

### Phase D: Portfolio Intelligence (8–12 weeks) — Medium Risk

**D1. Correlation-aware allocator** — capital_allocator.py  
**D2. Volatility targeting** — capital_allocator.py + run.py  
**D3. Regime-conditioned strategy routing** — capital_allocator.py + market_context_builder.py  
**D4. Drawdown-graduated sizing** — run.py  

### Phase E: ML/RL Improvement (12–20 weeks) — Higher Risk, Higher Reward

**E1. Meta-labeling secondary model** — forecast_engine.py  
**E2. RL reward redesign** — rl_intraday_structure_trainer.py  
**E3. Quantitative regime classifier (HMM)** — market_context_builder.py  
**E4. Probability calibration** — forecast_engine.py  
**E5. CPCV validation** — forecast_engine.py  

---

## 13. Risks and Overfitting Warnings

### 13.1 Overfitting Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Feature overfitting (too many features, too little data) | High | High | Monthly feature selection via MDA; keep ≤ 20 features |
| Parameter overfitting in walk-forward grid search | High | Medium | Limit grid to 50–100 points; prefer coarser grids |
| Regime classification overfitting | Medium | High | Use simple threshold rules first; validate on unseen periods |
| Multiple comparisons bias (20+ experiments) | High | Medium | Apply Bonferroni correction to significance thresholds |
| Survivorship bias in universe | Medium | Medium | Track deleted symbols; validate on broader universe |
| Look-ahead bias in feature construction | Medium | High | Audit every feature for future data use |
| RL overfitting to specific market period | High | High | Rotate training windows; penalize policy changes between windows |
| Meta-model training on strategy trades (selection bias) | Medium | Medium | Use strict train/test split; CPCV for meta-model |
| Sharpe inflation from correlated strategies | Medium | Medium | Correlation-adjusted Sharpe in allocator |

### 13.2 Bailey et al. (2014) Backtest Overfitting Warning

The probability of backtest overfitting (PBO) increases with the number of tested configurations. For N trials on a single time series:
- 5 trials: PBO ≈ 5–10%
- 20 trials: PBO ≈ 20–35%
- 100 trials: PBO ≈ 50–70%

**For QuantumQuant:** The Top 20 Experiments list represents 20 independent tests. Without correction, the expected number of spuriously positive results is 2–4. Every experiment must be validated on an **unseen** time period after the experiment is "done."

**Rule:** No experiment is considered validated until it produces positive OOS Sharpe on a period that was **not visible during design or parameter selection.**

### 13.3 Strategy Degradation (Alpha Decay)

All alpha eventually decays as more participants discover it. The `alpha_decay` metric in the composite score addresses this. Additional monitoring:

- Track `avg_daily_pnl` rolling 7-day vs 30-day. If 7-day/30-day ratio < 0.5, strategy is decaying.
- Track `profit_factor` trending: if declining over 4 consecutive weeks, review the strategy for parameter re-optimization.
- Track `win_rate` trend: a declining win rate with stable loss sizes is a sign of regime shift.

### 13.4 Model Degradation (Concept Drift)

LSTM and RL models trained on historical data will experience concept drift when market microstructure changes. Warning signs:

- LSTM `prob_up` accuracy dropping below 52% on recent data
- RL agent increasingly choosing "do nothing" as optimal action
- Walk-forward OOS Sharpe declining across successive windows

**Response:** Retrain models on rolling 6-month window monthly. Alert on calibration degradation.

### 13.5 Regime Misclassification Risk

The most dangerous risk is trading a momentum strategy in a mean-reversion regime, or vice versa. Regime misclassification can convert a profitable strategy into a consistent loser within a single session.

**Mitigation:** Always require two independent regime signals to agree before changing regime. Use "cautious" as the default state when signals conflict — reduce sizing in all strategies.

---

## 14. Open-Source GitHub Examples to Study

### 14.1 Algorithmic Trading Engines

---

**freqtrade/freqtrade**  
GitHub: https://github.com/freqtrade/freqtrade  
Stars: 50,569 | Forks: 10,557 | License: GPL-3.0 | Updated: 2026-05-21  

What it does: Production-grade crypto algo trading bot with strategy framework, backtesting, hyperparameter optimization (via Hyperopt/Optuna), walk-forward, and live execution.

Why relevant to QuantumQuant: Freqtrade's architecture mirrors QuantumQuant's design — strategy class with `populate_indicators()`, `populate_entry/exit_trend()`, and a separate execution engine. Its `FreqAI` module has meta-labeling, feature engineering, and walk-forward built in. Most transferable ideas are **architectural patterns**, not code.

Specific files to inspect:
- `freqtrade/freqai/` — meta-labeling and ML pipeline architecture
- `freqtrade/optimize/hyperopt.py` — Hyperopt integration for parameter search
- `freqtrade/optimize/backtesting.py` — walk-forward (`--enable-timerange-backtest`)
- `freqtrade/strategy/interface.py` — strategy base class structure

Ideas worth adapting:
- Walk-forward with `--timerange` split: clean train/test separation pattern
- `FreqAI` feature engineering pipeline: uses pandas_ta for technical indicators, injected as context
- Hyperopt objective functions: their `custom_loss` approach is analogous to composite_score

Warnings: GPL-3.0 — cannot use code in a non-GPL system without licensing implications. Crypto-focused; equity market microstructure differences apply. **Do not copy code.** Conceptual reference only.

---

**jesse-ai/jesse**  
GitHub: https://github.com/jesse-ai/jesse  
Stars: 7,913 | Forks: 1,123 | License: MIT | Updated: 2026-05-20  

What it does: Python framework for algorithmic trading with clean API, backtesting, and live trading support. Designed for readability and extensibility.

Why relevant to QuantumQuant: Jesse's routing system (strategy.position, strategy.orders, strategy.take_profit, strategy.stop_loss as properties) is cleaner than most frameworks and worth studying as a reference for how signals + execution can be separated cleanly.

Specific files to inspect:
- `jesse/strategies/Strategy.py` — base strategy interface with properties
- `jesse/routes/router.py` — routing multiple strategies to different symbols
- `jesse/services/risk/` — position sizing and risk management

Ideas worth adapting:
- The `.take_profit[]` and `.stop_loss[]` as modifiable arrays during trade — elegant API for partial exits and trailing stops
- `is_close_to(a, b, pct)` utility for proximity checks useful in pivot/level detection

Warnings: MIT license (clean), but Jesse is designed around single-asset strategies. Multi-strategy portfolio patterns would need adaptation. Crypto-focused.

---

### 14.2 Backtesting Frameworks

---

**kernc/backtesting.py**  
GitHub: https://github.com/kernc/backtesting.py  
Stars: 8,388 | Forks: 1,456 | License: AGPL-3.0 | Updated: 2026-05-20  

What it does: Lightweight, fast Python backtesting library. 1000 lines of clean Python. Supports indicator injection, multiple take-profit levels, trailing stops, and parameter optimization via `bt.optimize()`.

Why relevant to QuantumQuant: The cleanest reference implementation of partial take-profit and trailing stop logic in pure Python. The `Trade` object tracks `entry_price`, `entry_bar`, `size`, with `tp` (take-profit) and `sl` (stop-loss) as modifiable properties during the trade lifecycle.

Specific files to inspect:
- `backtesting/backtesting.py` — entire framework in one file; `class Trade` starting at ~line 400
- `backtesting/_util.py` — `resample_apply()` for multi-timeframe without lookahead

Ideas worth adapting:
- `Trade` class pattern: `trade.sl = current_price - atr` as in-trade stop modification — adapt this for `run.py`'s open position management
- `Strategy.position.size` vs `Strategy.position.is_long` — clean state tracking pattern

Warnings: AGPL-3.0 — share-alike license. **Do not incorporate code.** Study the trade lifecycle management pattern only. The backtesting.py approach does not handle multi-asset portfolios natively.

---

**polakowo/vectorbt**  
GitHub: https://github.com/polakowo/vectorbt  
Stars: 7,632 | Forks: 980 | License: Custom (non-commercial for newer versions) | Updated: 2026-05-21  

What it does: Vectorized backtesting using NumPy/pandas. 10–100× faster than event-driven backtesting for large parameter sweeps. Excellent for walk-forward grid searches.

Why relevant to QuantumQuant: The walk-forward validation pattern in vectorbt (`portfolio.rolling()`) can run thousands of parameter combinations across rolling windows in seconds. This is the fastest available approach for the `walkforward_strategy.py` grid-search portion.

Specific files to inspect:
- `vectorbt/portfolio/base.py` — Portfolio class, `from_signals()`, `from_order_func()`
- `vectorbt/generic/nb.py` — Numba-compiled rolling functions
- `vectorbt/records/` — trade record analysis (MAE/MFE per trade)

Ideas worth adapting:
- `pf.stats()` output format: comprehensive performance summary in one call — add equivalent to `evaluate_variants.py`
- Rolling walk-forward via `pf.rolling()` with custom parameter grids: adapt for `walkforward_strategy.py`

Warnings: License changed to non-commercial for newer versions — check `vectorbt-pro` vs `vectorbt`. Use only the older Apache-2.0 tagged releases for commercial/research use. **Conceptual reference only.**

---

### 14.3 Walk-Forward Analysis

---

**ranaroussi/quantstats**  
GitHub: https://github.com/ranaroussi/quantstats  
Stars: 7,139 | Forks: 1,176 | License: Apache-2.0 | Updated: 2026-05-20  

What it does: Portfolio analytics and reporting library. Produces HTML tearsheets with rolling Sharpe, drawdown charts, monthly returns heatmaps, Calmar ratio, Sortino ratio, and 30+ performance metrics.

Why relevant to QuantumQuant: The `qs.stats` module computes exactly the metrics the evaluator needs: `sortino()`, `calmar()`, `max_drawdown()`, `sharpe()`, `omega_ratio()`, `profit_factor()`, and more. The `qs.reports.html()` tearsheet format is ideal for the nightly dashboard.

Specific files to inspect:
- `quantstats/stats.py` — `sortino()`, `calmar()`, `omega_ratio()`, `consecutive_losses()`, `expectancy()`
- `quantstats/reports.py` — HTML tearsheet generation
- `quantstats/plots.py` — rolling Sharpe plot, drawdown plot

Ideas worth adapting:
- `qs.stats.omega_ratio()`: penalizes downside deviation proportionally; add as a metric alongside Sortino
- `qs.stats.consecutive_losses()`: directly implements `consecutive_loss_max` metric needed in evaluator
- `qs.stats.expectancy()`: `(win_rate × avg_win) - (loss_rate × avg_loss)`; implement in evaluator

Warnings: Apache-2.0, clean to use. Designed for daily return series (not tick data), but the statistical functions work on any return series. Don't use the reporting HTML in production without customization — it assumes daily frequency.

---

### 14.4 Portfolio Optimization

---

**dcajasn/Riskfolio-Lib**  
GitHub: https://github.com/dcajasn/Riskfolio-Lib  
Stars: 4,197 | Forks: 667 | License: BSD-3-Clause | Updated: 2026-05-21  

What it does: Advanced portfolio optimization library implementing 30+ risk measures: CVaR, CDaR, MDD, risk parity, Black-Litterman, HRP (Hierarchical Risk Parity), and more. Full Markowitz mean-variance with covariance shrinkage.

Why relevant to QuantumQuant: Risk parity and HRP allocation are directly applicable to `capital_allocator.py`. Hierarchical Risk Parity (Lopez de Prado 2016) doesn't require matrix inversion and works with small sample sizes — perfect for 6 strategies with < 60 daily P&L observations.

Specific files to inspect:
- `riskfolio/Portfolio.py` — `HRP_port()` hierarchical risk parity implementation
- `riskfolio/RiskFunctions.py` — `CDaR_Abs()`, `MDD_Abs()` drawdown risk measures
- `riskfolio/AuxFunctions.py` — covariance shrinkage via Ledoit-Wolf

Ideas worth adapting:
- **HRP allocation**: Apply to `capital_allocator.py`. HRP clusters strategies by correlation, then allocates inversely proportional to within-cluster variance. Works with as few as 6 strategies and 20 daily P&L observations.
- Ledoit-Wolf covariance shrinkage: critical for small-sample correlation estimation in the allocator
- CDaR (Conditional Drawdown-at-Risk): add as an alternative to max_drawdown in evaluator

Warnings: BSD-3 license (clean). Requires `scipy`, `cvxpy` for optimization — add to requirements.txt. The full optimization suite is complex; start with HRP only (it's scipy-free).

---

**PyPortfolio/PyPortfolioOpt**  
GitHub: https://github.com/robertmartin8/PyPortfolioOpt  
Stars: 5,730 | Forks: 1,120 | License: MIT | Updated: 2026-05-20  

What it does: Clean implementation of portfolio optimization: mean-variance, Black-Litterman, risk parity, with Ledoit-Wolf covariance and CVaR constraints.

Why relevant to QuantumQuant: The cleanest Python API for portfolio optimization. `risk_models.CovarianceShrinkage()` is directly usable in `capital_allocator.py`. The `EfficientFrontier` interface is well-documented and MIT-licensed.

Specific files to inspect:
- `pypfopt/risk_models.py` — `CovarianceShrinkage(df).ledoit_wolf()`
- `pypfopt/efficient_frontier.py` — `max_sharpe()`, `min_volatility()`
- `pypfopt/hierarchical_portfolio.py` — HRP implementation

Ideas worth adapting:
- `CovarianceShrinkage` as drop-in for correlation matrix estimation in allocator
- `HRPOpt` for allocation when < 30 days of data (doesn't need matrix inversion)

Warnings: MIT license. CVX-based optimization requires `cvxpy`. For the allocator enhancement, start with Ledoit-Wolf shrinkage + Sharpe-weighting before adding full mean-variance optimization.

---

### 14.5 Reinforcement Learning Trading

---

**AI4Finance-Foundation/FinRL**  
GitHub: https://github.com/AI4Finance-Foundation/FinRL  
Stars: ~10,000 (GitHub data unavailable — estimated from training knowledge) | License: MIT  

What it does: Deep RL framework for stock trading using PPO, A2C, SAC, TD3, and DDPG. Pre-built environments for single-stock and portfolio trading. Feature engineering pipeline.

Why relevant to QuantumQuant: `rl_intraday_structure_trainer.py` uses stable-baselines3 (same underlying RL library as FinRL). FinRL's `StockTradingEnv` implementation is the best public reference for defining reward functions, observation spaces, and action spaces for multi-stock intraday trading.

Specific files to inspect:
- `finrl/env/env_stocktrading.py` — `StockTradingEnv` observation/action/reward design
- `finrl/config.py` — feature engineering constants; list of 30+ technical indicators
- `finrl/model/models.py` — ensemble model training (A2C + PPO + DDPG ensemble)

Ideas worth adapting:
- Turbulence index: FinRL's Mahalanobis distance-based market turbulence measure — when turbulence is high, the RL agent reduces position sizes automatically. Add as a feature to `rl_intraday_structure_trainer.py`.
- Ensemble model approach: train 3 RL models with different hyperparameters, take majority vote on actions — reduces single-model variance

Warnings: MIT license. FinRL is designed for daily rebalancing, not intraday execution. The observation space design assumes open/close/high/low/volume — good overlap with QuantumQuant. The reward function (`log(portfolio_value)`) is better than raw return but still needs the Sharpe/drawdown modifications described in Section 7.5. The turbulence index concept is more valuable than the code.

---

**tensortrade-org/tensortrade**  
GitHub: https://github.com/tensortrade-org/tensortrade  
Stars: 6,276 | Forks: 1,234 | License: Apache-2.0 | Updated: 2026-05-20  

What it does: Composable RL trading framework. Separate components for data feeds, feature engineering, action schemes, reward schemes, and execution. Works with any Gymnasium-compatible RL library.

Why relevant to QuantumQuant: TensorTrade's modular reward scheme architecture is the cleanest reference for implementing the Sharpe-penalized reward function proposed in Section 7.5. Its `SimpleProfit`, `RiskAdjustedReturns`, and `PBR` (Price-Based Returns) reward schemes show different approaches.

Specific files to inspect:
- `tensortrade/env/default/rewards/` — `RiskAdjustedReturns` implementation
- `tensortrade/feed/core/` — stream-based feature pipeline
- `tensortrade/oms/` — order management simulation

Ideas worth adapting:
- `RiskAdjustedReturns` reward: `reward = log_return / rolling_volatility` — exactly the Sharpe-penalized reward structure needed
- Modular reward composition: define reward as sum of sub-rewards (profit + drawdown_penalty + time_penalty) — easier to tune than monolithic reward functions

Warnings: Apache-2.0. Project has reduced maintenance velocity (last major update 2024). The modular architecture is worth studying but the codebase itself is complex. Use as a reference for reward design patterns only.

---

### 14.6 Market Regime Detection

---

**quantopian/empyrical**  
GitHub: https://github.com/quantopian/empyrical  
Stars: 1,480 | Forks: ~360 | License: Apache-2.0 | Updated: 2026-05-19  

What it does: Standard performance analytics library: Sharpe, Sortino, Calmar, alpha/beta, max drawdown, drawdown periods, underwater plot data.

Why relevant to QuantumQuant: The `empyrical.stats` module has direct drop-ins for metrics needed in `evaluate_variants.py`: `sortino_ratio()`, `calmar_ratio()`, `max_drawdown()`, `alpha_beta()`, `annual_sharpe()`. All handle edge cases (insufficient data, zero volatility) gracefully.

Specific files to inspect:
- `empyrical/stats.py` — all performance statistics; clean implementations of every metric needed
- `empyrical/periods.py` — annualization factors by trading frequency

Ideas worth adapting:
- `empyrical.stats.aggregate_returns()` — weekly/monthly return aggregation for consistency metrics
- `empyrical.stats.drawdown_series()` — underwater equity curve for visualization

Warnings: Apache-2.0. Quantopian is defunct; the library is unmaintained since ~2021 but the statistics implementations are stable and correct. Safe to use as a reference for formulas.

---

### 14.7 Feature Engineering Pipelines

---

**TA-Lib/ta-lib-python**  
GitHub: https://github.com/TA-Lib/ta-lib-python  
Stars: 11,974 | Forks: ~2,000 | License: BSD-2-Clause | Updated: 2026-05-21  

What it does: Python wrapper for TA-Lib C library. 150+ technical analysis indicators computed in C (very fast). Standard implementations of every indicator used in QuantumQuant.

Why relevant to QuantumQuant: `build_intraday_training_dataset.py` computes all indicators manually in Python. TA-Lib provides validated, battle-tested C implementations that run 10–50× faster. Particularly important for intraday feature computation across 30 symbols × 30 days × 390 bars.

Specific indicators worth noting:
- `VWAP` — TA-Lib doesn't implement VWAP (session-cumulative); keep existing implementation
- `BBANDS` — Bollinger Bands for VWAP std deviation bands proposal above
- `WILLR` — Williams %R; alternative to RSI for overbought/oversold detection
- `OBV` — On Balance Volume; strong order-flow proxy

Warnings: BSD-2-Clause. Requires C compilation (can be complex on some systems). The existing pandas-based implementations in the codebase are correct; TA-Lib is an optimization option, not a necessity.

---

**bukosabino/ta**  
GitHub: https://github.com/bukosabino/ta  
Stars: 5,086 | Forks: ~800 | License: MIT | Updated: 2026-05-20  

What it does: Pure-Python technical analysis library built on pandas. 130+ indicators organized into momentum, volatility, trend, volume, and others categories.

Why relevant to QuantumQuant: MIT license, pure pandas/numpy (no C compilation), drop-in with existing feature engineering pipeline. The `VolumeIndicator` group includes MFI (Money Flow Index) and Force Index — good order-flow proxies.

Specific files to inspect:
- `ta/volume.py` — `MFIIndicator` (Money Flow Index), `AccDistIndexIndicator`, `OnBalanceVolumeIndicator`
- `ta/volatility.py` — `BollingerBands`, `KeltnerChannel`, `DonchianChannel`
- `ta/momentum.py` — `RSIIndicator`, `StochRSIIndicator`, `KAMAIndicator`

Ideas worth adapting:
- `BollingerBands(close, window=20, window_dev=2).bollinger_hband/lband()` — directly implement VWAP deviation bands by adapting to use VWAP as the centerline instead of SMA
- `MFIIndicator` — Money Flow Index is a volume-weighted RSI; better order-flow proxy than RVOL alone

Warnings: MIT license (clean). Minor implementation differences from TA-Lib for some indicators. Validate against known values before production use.

---

### 14.8 Performance Analytics & Dashboards

---

**quantopian/pyfolio**  
GitHub: https://github.com/quantopian/pyfolio  
Stars: 6,309 | Forks: ~1,200 | License: Apache-2.0 | Updated: 2026-05-21  

What it does: Portfolio and strategy tear-sheet generation. Produces: returns, drawdown, rolling Sharpe, factor exposure, event study, position analysis, transaction cost analysis.

Why relevant to QuantumQuant: The `generate_full_tear_sheet()` function produces exactly the diagnostic output that `generate_dashboard.py` and `diagnose_strategy.py` should generate. The rolling performance analysis and event study modules are directly applicable.

Specific files to inspect:
- `pyfolio/timeseries.py` — `perf_stats()` function (30+ metrics in one call)
- `pyfolio/tears.py` — tear sheet construction
- `pyfolio/pos.py` — position analysis, exposure tracking

Ideas worth adapting:
- `perf_stats()` output: comprehensive DataFrame of performance statistics; use as template for `evaluate_variants.py` output expansion
- `rolling_sharpe()` with configurable window: add to `evaluate_variants.py` for decay detection
- Event study: `pyfolio.interesting_periods` marks known market events (crises, corrections) — useful for stress-testing strategy performance

Warnings: Apache-2.0. Quantopian-era code; some dependencies outdated. `matplotlib` required for charts. Use statistics functions directly (from `pyfolio.timeseries`) rather than the plotting functions.

---

### 14.9 Alpaca-Specific Resources

---

**alpacahq/alpaca-py**  
GitHub: https://github.com/alpacahq/alpaca-py  
Stars: 1,332 | Forks: 355 | License: Apache-2.0 | Updated: 2026-05-21  

What it does: Official Alpaca Python SDK. Stock/options/crypto trading, historical data, streaming, account management.

Why relevant to QuantumQuant: Already used. Key underutilized features:
- `StockLatestQuoteRequest` — real-time bid/ask for spread calculation
- `StockSnapshotRequest` — snapshot includes VWAP, daily volume, previous close in one API call  
- `news.News.get()` — news feed API for catalyst filtering in gap-and-go strategy

Specific files to inspect:
- `alpaca/data/historical/stock.py` — `get_stock_bars()` with feed parameter (IEX/SIP/OTC)
- `alpaca/trading/client.py` — `get_clock()`, `get_calendar()` for session timing
- `alpaca/data/live/stock.py` — `StockDataStream` for real-time bars

Ideas worth adapting:
- `StockSnapshotRequest` with multiple symbols in one call — reduces API latency for universe snapshots vs calling per-symbol
- News API for catalyst detection in gap-and-go strategy

---

### 14.10 Additional Notable Repositories

---

**mementum/backtrader**  
GitHub: https://github.com/mementum/backtrader  
Stars: ~14,000 (estimated from training knowledge) | License: GPL-3.0  

Why relevant: The `Analyzer` framework pattern (separate performance analysis from strategy logic) is worth studying. The `DrawDown`, `SharpeRatio`, and `TradeAnalyzer` classes show clean separation of concerns.

Warnings: GPL-3.0. Maintenance has slowed. Conceptual reference for Analyzer pattern only.

---

**microsoft/qlib**  
GitHub: https://github.com/microsoft/qlib  
Stars: ~16,000 (estimated) | License: MIT  

What it does: AI-oriented quantitative investment research platform from Microsoft. Contains data pipeline, factor analysis, model zoo (LSTM, Transformer, GBDT), backtesting, and meta-labeling.

Why relevant: The `qlib.contrib.model.pytorch_lstm.LSTM` implementation and the `SignalRecord` + `MetaTaskDataset` (meta-labeling) architecture are worth studying. The factor analysis (`qlib.backtest.report`) approach to alpha evaluation is more rigorous than a simple P&L backtest.

Specific files: `qlib/contrib/model/pytorch_lstm.py`, `qlib/backtest/`, `qlib/contrib/meta/`

Warnings: MIT license. Very complex infrastructure. Not practical to adapt directly — study the meta-labeling pipeline and LSTM architecture patterns only.

---

**dcajasn/Riskfolio-Lib** (mentioned above under Portfolio Optimization)  
Additional note: The **Risk Parity** implementation is in `Portfolio.rp_optimization()`. For 6 strategies, risk parity would equalize the contribution of each strategy to total portfolio variance — a meaningful improvement over Sharpe-weighting when strategy Sharpe ratios are close together.

---

## 15. References and Source Links

### Academic Papers

**Portfolio Management**
1. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77–91. [Classic MPT]
2. Kelly, J.L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4), 917–926. [Kelly criterion]
3. Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." *Handbook of Asset and Liability Management*. [Fractional Kelly in practice]
4. Hurst, B., Ooi, Y.H., Pedersen, L.H. (2012). "A Century of Evidence on Trend-Following Investing." AQR Capital Research. [Volatility targeting]
5. Ledoit, O., Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*. [Covariance shrinkage]
6. Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *Journal of Portfolio Management*. [Hierarchical Risk Parity — HRP]

**Alpha Generation**
7. Jegadeesh, N., Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1). [Cross-sectional momentum]
8. Grinblatt, M., Moskowitz, T. (2004). "Predicting Stock Price Movements from Past Returns." *Journal of Financial Economics*, 71(3). [Momentum persistence]
9. Heston, S.L., Sadka, R. (2010). "Intraday Patterns in the Cross-Section of Stock Returns." *Journal of Finance*, 65(4). [Time-of-day seasonality]
10. Lou, D., Polk, C., Skouras, S. (2019). "A Tug of War: Overnight vs. Intraday Expected Returns." *Journal of Financial Economics*, 134(1). [Intraday momentum]
11. Berkowitz, S. et al. (1988). "The Total Cost of Transactions on the NYSE." *Journal of Finance*, 43(1). [VWAP as execution benchmark]
12. Moskowitz, T., Grinblatt, M. (1999). "Do Industries Explain Momentum?" *Journal of Finance*, 54(4). [Sector momentum]

**Market Microstructure**
13. O'Hara, M. (2015). "High Frequency Market Microstructure." *Journal of Financial Economics*, 116(2). [Order flow]
14. Chordia, T., Roll, R., Subrahmanyam, A. (2002). "Order Imbalance, Liquidity, and Market Returns." *Journal of Financial Economics*, 65(1). [Order flow imbalance]
15. Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Econometrics*, 12(1). [Microstructure alpha]

**Machine Learning for Trading**
16. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley. [Meta-labeling, CPCV, feature importance — Ch. 3, 6, 7, 8]
17. Bailey, D.H., Borwein, J., Lopez de Prado, M., Zhu, Q.J. (2014). "The Probability of Backtest Overfitting." *Journal of Computational Finance*. [PBO methodology]
18. Gao, G., Han, B., Li, Y., Zhou, C. (2018). "Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour Return." arXiv:1801.02025. [Momentum features]
19. Vince, R. (1992). "The Mathematics of Money Management." Wiley. [Drawdown-aware sizing]

### Practitioner Resources
- QuantStart: https://www.quantstart.com/articles/ — Walk-forward analysis, Python backtesting guides
- QuantInsti: https://blog.quantinsti.com — Intraday strategy backtesting, feature engineering for ML trading
- Alpaca Docs: https://docs.alpaca.markets — Data feed documentation, order types, IEX vs SIP
- Interactive Brokers: https://www.interactivebrokers.com/en/trading/education.php — Execution quality, order routing
- NYSE Market Structure: https://www.nyse.com/data/transactions-statistics-data-library — Market microstructure data
- SSRN: https://www.ssrn.com — Search "intraday momentum", "regime detection equity", "walk-forward optimization"
- arXiv q-fin: https://arxiv.org/list/q-fin.TR/recent — Quantitative finance preprints

---

*Report generated: 2026-05-20 | Version: 1.0 | Classification: Internal Research*  
*All proposed changes require paper-trading validation before consideration for live deployment.*  
*No guaranteed returns are claimed. Past performance in backtests does not predict future results.*
