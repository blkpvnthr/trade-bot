"""Shared, pure performance-metrics utilities for research and reporting.

This module exists because the same bug -- using CAGR (a geometric,
annualized return) as the Sharpe numerator instead of the mean of daily
excess returns -- was independently re-implemented in walkforward.py (twice)
and backtest.py. All three computed:

    sharpe = annualized_return / annualized_vol         # WRONG

instead of the standard definition:

    sharpe = mean(daily_excess_return) / std(daily_excess_return) * sqrt(252)

The difference matters: the wrong formula conflates a geometric-return
numerator with an arithmetic-return-based denominator, and silently drops
the risk-free rate in two of the three call sites. This module is the single
place that formula lives from now on; call sites should import it rather
than recompute it inline.

Every function here is pure (no I/O) and operates on plain pandas/numpy
inputs so it is trivially unit-testable with synthetic equity curves of
known expected Sharpe/Sortino/Calmar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Daily-return-based ratios (the correct pattern)
# =============================================================================

def _clean_daily_returns(daily_returns) -> np.ndarray:
    return np.asarray(pd.Series(daily_returns, dtype=float).fillna(0.0))


def daily_risk_free_rate(risk_free_rate_annual: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Convert an annualized risk-free rate to a per-period rate.

    Configurable / date-appropriate: callers should pass the rate prevailing
    over the period being measured (e.g. the contemporaneous T-bill yield),
    not a single hardcoded constant for all history.
    """
    return (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0


def sharpe_ratio(
    daily_returns,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio from daily excess returns.

    sharpe = mean(daily_return - daily_rf) / std(daily_return - daily_rf) * sqrt(periods_per_year)

    Returns 0.0 for fewer than 2 observations or zero volatility (explicit
    handling, not a silent NaN/inf).
    """
    arr = _clean_daily_returns(daily_returns)
    if len(arr) < 2:
        return 0.0
    excess = arr - daily_risk_free_rate(risk_free_rate_annual, periods_per_year)
    std = excess.std(ddof=1)
    # Floating-point noise on a theoretically-constant series can leave std a
    # tiny nonzero value (e.g. 1e-19) instead of exactly 0, which would blow
    # the ratio up to an enormous, meaningless number. 1e-12 is far below any
    # real daily-return volatility, so this only catches that noise floor.
    if np.isnan(std) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    daily_returns,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio: excess return over downside deviation only."""
    arr = _clean_daily_returns(daily_returns)
    if len(arr) < 2:
        return 0.0
    excess = arr - daily_risk_free_rate(risk_free_rate_annual, periods_per_year)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std(ddof=1)
    if np.isnan(downside_std) or downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(periods_per_year))


def cagr(total_return: float, n_periods: int, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate. Reported separately from Sharpe -- never
    use this as a Sharpe numerator (see module docstring)."""
    if n_periods <= 0:
        return 0.0
    return float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)


def equity_curve_from_daily_returns(daily_returns, starting_equity: float = 1.0) -> np.ndarray:
    arr = _clean_daily_returns(daily_returns)
    return starting_equity * np.cumprod(1.0 + arr)


def max_drawdown(equity_curve) -> float:
    eq = np.asarray(pd.Series(equity_curve, dtype=float))
    if len(eq) == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    running_max = np.where(running_max == 0, np.nan, running_max)
    dd = eq / running_max - 1.0
    dd = dd[~np.isnan(dd)]
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """CAGR / |max drawdown|. 0.0 when there is no drawdown (undefined, not infinite)."""
    if max_dd == 0 or np.isnan(max_dd):
        return 0.0
    return float(cagr_value / abs(max_dd))


# =============================================================================
# Trade / exposure / turnover reporting
# =============================================================================

def win_rate(trade_pnls) -> float:
    arr = _clean_daily_returns(trade_pnls)
    if len(arr) == 0:
        return 0.0
    return float((arr > 0).mean())


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    """Per-period L1 turnover from a (time x symbol) weight matrix:
    sum_i |w_i,t - w_i,t-1|. First period's turnover is the absolute sum of
    its weights (entering from flat)."""
    if weights.empty:
        return pd.Series(dtype=float)
    diffs = weights.fillna(0.0).diff()
    diffs.iloc[0] = weights.fillna(0.0).iloc[0]
    return diffs.abs().sum(axis=1)


def annualized_turnover(per_period_turnover, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    arr = _clean_daily_returns(per_period_turnover)
    if len(arr) == 0:
        return 0.0
    return float(arr.mean() * periods_per_year)


# =============================================================================
# Uncertainty: standard error, bootstrap CI, deflated Sharpe
# =============================================================================

def sharpe_standard_error(sharpe: float, n_obs: int, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """First-order asymptotic standard error of an annualized Sharpe estimate
    under iid-normal per-period returns (Lo, 2002), with n_obs the number of
    underlying (non-annualized) return observations. NaN for n_obs < 2 --
    there is no meaningful uncertainty estimate from fewer than 2 points, and
    that should be visible, not hidden behind a 0.0."""
    if n_obs < 2:
        return float("nan")
    per_period_sharpe = sharpe / math.sqrt(periods_per_year)
    return float(math.sqrt((1.0 + 0.5 * per_period_sharpe ** 2) / n_obs) * math.sqrt(periods_per_year))


def effective_sample_size_ar1(daily_returns) -> float:
    """Autocorrelation-adjusted effective sample size for daily returns.

    Overlapping multi-day holding periods and persistent exposure both
    induce positive autocorrelation in a daily return series, which makes
    the raw observation count overstate how much independent information
    the sample actually contains. Using the standard AR(1) approximation:

        n_eff = n * (1 - rho) / (1 + rho)

    where rho is the lag-1 autocorrelation of daily returns. Returns n
    unchanged (no adjustment) if rho cannot be estimated or is <= 0 (i.e.
    never *increases* the apparent sample size).
    """
    arr = _clean_daily_returns(daily_returns)
    n = len(arr)
    if n < 3:
        return float(n)
    rho = np.corrcoef(arr[:-1], arr[1:])[0, 1]
    if np.isnan(rho) or rho <= 0:
        return float(n)
    rho = min(rho, 0.999)
    return float(n * (1.0 - rho) / (1.0 + rho))


def bootstrap_sharpe_ci(
    daily_returns,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float]:
    """IID bootstrap confidence interval for the annualized Sharpe ratio.

    A wide interval on a short sample is the correct and expected behavior,
    not a bug -- it is the point of reporting uncertainty at all.
    """
    arr = _clean_daily_returns(daily_returns)
    if len(arr) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    n = len(arr)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled = rng.choice(arr, size=n, replace=True)
        samples[i] = sharpe_ratio(resampled, risk_free_rate_annual, periods_per_year)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Probability that `observed_sharpe` reflects genuine skill rather than
    the best of `n_trials` independently-tried variants (Bailey & Lopez de
    Prado, 2014). Returns a probability in [0, 1]. Values that aren't well
    above the 0.90-0.95 range after accounting for how many variants were
    tried should not be treated as a validated edge.

    `n_trials` must be supplied honestly -- it is the number of strategy
    variants / parameter settings / thresholds actually evaluated before
    landing on this one, including ones that were discarded.
    """
    from scipy.stats import norm

    if n_obs < 2 or n_trials < 1:
        return float("nan")

    per_period_sr = observed_sharpe / math.sqrt(periods_per_year)

    if n_trials == 1:
        sr0_per_period = 0.0
    else:
        var_sr = (1.0 - skew * per_period_sr + (excess_kurtosis / 4.0) * per_period_sr ** 2) / max(n_obs - 1, 1)
        var_sr = max(var_sr, 1e-12)
        euler_mascheroni = 0.5772156649
        sr0_per_period = math.sqrt(var_sr) * (
            (1.0 - euler_mascheroni) * norm.ppf(1.0 - 1.0 / n_trials)
            + euler_mascheroni * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )

    denom = math.sqrt(max(1.0 - skew * per_period_sr + (excess_kurtosis / 4.0) * per_period_sr ** 2, 1e-12))
    z = (per_period_sr - sr0_per_period) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


# =============================================================================
# Cost sensitivity
# =============================================================================

def apply_turnover_cost(daily_returns, per_period_turnover, cost_bps: float) -> np.ndarray:
    """Net daily returns after deducting a flat cost (in bps of traded
    notional) proportional to that period's turnover. Used to generate
    baseline / adverse / severe cost-sensitivity scenarios from the same
    gross-return series."""
    gross = _clean_daily_returns(daily_returns)
    turnover = _clean_daily_returns(per_period_turnover)
    if len(turnover) != len(gross):
        raise ValueError("daily_returns and per_period_turnover must be the same length")
    return gross - turnover * (cost_bps / 10_000.0)


# =============================================================================
# Convenience summary
# =============================================================================

@dataclass(frozen=True)
class PerformanceSummary:
    n_obs: int
    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    sharpe_se: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    sortino: float
    max_drawdown: float
    calmar: float
    trade_count: int
    win_rate: float
    avg_exposure: float
    annual_turnover: float


def summarize_performance(
    daily_returns,
    trade_pnls=None,
    exposure=None,
    per_period_turnover=None,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    bootstrap: bool = True,
    random_state: int | None = None,
) -> PerformanceSummary:
    arr = _clean_daily_returns(daily_returns)
    n_obs = len(arr)
    total_return = float(np.prod(1.0 + arr) - 1.0) if n_obs else 0.0
    annual_vol = float(arr.std(ddof=1) * math.sqrt(periods_per_year)) if n_obs > 1 else 0.0
    sharpe = sharpe_ratio(arr, risk_free_rate_annual, periods_per_year)
    sortino = sortino_ratio(arr, risk_free_rate_annual, periods_per_year)
    cagr_value = cagr(total_return, n_obs, periods_per_year)
    eq = equity_curve_from_daily_returns(arr)
    max_dd = max_drawdown(eq)
    calmar = calmar_ratio(cagr_value, max_dd)
    sharpe_se = sharpe_standard_error(sharpe, n_obs, periods_per_year)

    if bootstrap and n_obs >= 10:
        ci_low, ci_high = bootstrap_sharpe_ci(
            arr, risk_free_rate_annual, periods_per_year, random_state=random_state
        )
    else:
        ci_low, ci_high = (float("nan"), float("nan"))

    trade_count = int(len(trade_pnls)) if trade_pnls is not None else 0
    wr = win_rate(trade_pnls) if trade_pnls is not None and len(trade_pnls) else 0.0
    avg_exposure = float(_clean_daily_returns(exposure).mean()) if exposure is not None and len(exposure) else 0.0
    ann_turnover = annualized_turnover(per_period_turnover, periods_per_year) if per_period_turnover is not None else 0.0

    return PerformanceSummary(
        n_obs=n_obs,
        total_return=total_return,
        cagr=cagr_value,
        annual_vol=annual_vol,
        sharpe=sharpe,
        sharpe_se=sharpe_se,
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        trade_count=trade_count,
        win_rate=wr,
        avg_exposure=avg_exposure,
        annual_turnover=ann_turnover,
    )
