#!/usr/bin/env python3
"""
alpha_discovery/analytics_feature_builder.py — Advanced Analytics features.

Turns the normalized Alpha Vantage Advanced Analytics result dicts (produced by
data_sources.alpha_vantage_analytics_client) into the §2 fixed-window and §3
sliding-window feature families, plus the §5 hidden-alpha candidates and
interactions, used by universe discovery, risk management, and the (gated) RL
options state.

Every feature is a bounded / centered numeric value. Missing inputs degrade to
neutral values (0.0 / None) — never an exception. Raw API payloads never enter
here; only the normalized result dicts do.

Feature family name: ``advanced_analytics``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

_FAMILY = "advanced_analytics"


# =============================================================================
# Neutral schema
# =============================================================================

_FIXED_KEYS = (
    "fixed_return_mean", "fixed_return_median", "fixed_cumulative_return",
    "fixed_return_stddev", "fixed_return_variance", "fixed_max_drawdown",
    "fixed_min_return", "fixed_max_return", "fixed_autocorrelation_lag1",
    "fixed_autocorrelation_lag2", "fixed_correlation_to_spy",
    "fixed_correlation_to_qqq", "fixed_covariance_to_spy", "fixed_covariance_to_sector",
)

_ROLLING_KEYS = (
    "rolling_return_mean", "rolling_return_median", "rolling_cumulative_return",
    "rolling_stddev", "rolling_variance", "rolling_correlation_to_spy",
    "rolling_correlation_to_qqq", "rolling_correlation_to_sector",
    "rolling_covariance_to_spy", "rolling_autocorrelation", "rolling_volatility_slope",
    "rolling_correlation_slope", "rolling_risk_regime_score", "rolling_relationship_shift_score",
)

_HIDDEN_KEYS = (
    "hidden_autocorrelation_momentum_score", "hidden_autocorrelation_mean_reversion_score",
    "hidden_correlation_break_score", "hidden_correlation_reversion_score",
    "hidden_volatility_expansion_score", "hidden_volatility_contraction_score",
    "hidden_risk_adjusted_momentum_score", "hidden_drawdown_recovery_score",
    "hidden_relationship_shift_score", "hidden_market_independence_score",
)


def neutral_analytics_features() -> dict[str, Any]:
    out: dict[str, Any] = {k: None for k in _FIXED_KEYS}
    out.update({k: None for k in _ROLLING_KEYS})
    out.update({k: 0.0 for k in _HIDDEN_KEYS})
    out["analytics_available"] = 0
    return out


# =============================================================================
# Helpers
# =============================================================================

def _clamp(v: Optional[float], lo: float = -1.0, hi: float = 1.0) -> float:
    if v is None:
        return 0.0
    return max(lo, min(hi, float(v)))


def _scalar(calcs: dict[str, Any], calc: str, symbol: str) -> Optional[float]:
    """Read a scalar calculation value for ``symbol``.

    For sliding-window summaries (dict with "latest") returns the latest value.
    """
    node = (calcs or {}).get(calc)
    if not isinstance(node, dict):
        return None
    val = node.get(symbol.upper())
    if isinstance(val, dict):  # sliding-window per-symbol summary
        return _to_f(val.get("latest"))
    return _to_f(val)


def _scalar_slope(calcs: dict[str, Any], calc: str, symbol: str) -> Optional[float]:
    """Read the rolling slope of a sliding-window calculation summary, if present."""
    node = (calcs or {}).get(calc)
    if not isinstance(node, dict):
        return None
    val = node.get(symbol.upper())
    if isinstance(val, dict):
        return _to_f(val.get("slope"))
    return None


def _to_f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN/inf guard
        return None
    return f


def relationship_value(calcs: dict[str, Any], calc: str, symbol: str, reference: str) -> Optional[float]:
    """Read symbol↔reference value from a CORRELATION / COVARIANCE matrix calc.

    ``calc`` is "CORRELATION" or "COVARIANCE". Returns matrix[i][j] where i is the
    symbol's row and j the reference's column, or None when either is absent.
    """
    node = (calcs or {}).get(calc)
    if not isinstance(node, dict):
        return None
    index = node.get("index")
    matrix = node.get("matrix")
    if not isinstance(index, list) or not isinstance(matrix, list):
        return None
    idx = {str(s).upper(): i for i, s in enumerate(index)}
    i, j = idx.get(symbol.upper()), idx.get(reference.upper())
    if i is None or j is None or i >= len(matrix):
        return None
    row = matrix[i]
    if not isinstance(row, list) or j >= len(row):
        return None
    return _to_f(row[j])


# =============================================================================
# §2 Fixed-window features
# =============================================================================

def build_fixed_window_features(
    symbol: str,
    result: Optional[dict[str, Any]],
    spy: str = "SPY",
    qqq: str = "QQQ",
    sector: Optional[str] = None,
) -> dict[str, Any]:
    """Build the ``fixed_*`` feature block for ``symbol`` from a fixed-window result."""
    out: dict[str, Any] = {k: None for k in _FIXED_KEYS}
    calcs = (result or {}).get("calculations") if isinstance(result, dict) else None
    if not calcs:
        return out
    sym = symbol.upper()
    out["fixed_return_mean"] = _scalar(calcs, "MEAN", sym)
    out["fixed_return_median"] = _scalar(calcs, "MEDIAN", sym)
    out["fixed_cumulative_return"] = _scalar(calcs, "CUMULATIVE_RETURN", sym)
    out["fixed_return_stddev"] = _scalar(calcs, "STDDEV", sym)
    out["fixed_return_variance"] = _scalar(calcs, "VARIANCE", sym)
    out["fixed_max_drawdown"] = _scalar(calcs, "MAX_DRAWDOWN", sym)
    out["fixed_min_return"] = _scalar(calcs, "MIN", sym)
    out["fixed_max_return"] = _scalar(calcs, "MAX", sym)
    out["fixed_autocorrelation_lag1"] = _scalar(calcs, "AUTOCORRELATION", sym)
    # AV exposes lag-2 under the same AUTOCORRELATION block keyed per-lag only when
    # requested; when unavailable it stays neutral.
    out["fixed_autocorrelation_lag2"] = _scalar(calcs, "AUTOCORRELATION_LAG2", sym)
    out["fixed_correlation_to_spy"] = relationship_value(calcs, "CORRELATION", sym, spy)
    out["fixed_correlation_to_qqq"] = relationship_value(calcs, "CORRELATION", sym, qqq)
    out["fixed_covariance_to_spy"] = relationship_value(calcs, "COVARIANCE", sym, spy)
    if sector:
        out["fixed_covariance_to_sector"] = relationship_value(calcs, "COVARIANCE", sym, sector)
    return out


# =============================================================================
# §3 Sliding-window features
# =============================================================================

def build_sliding_window_features(
    symbol: str,
    result: Optional[dict[str, Any]],
    spy: str = "SPY",
    qqq: str = "QQQ",
    sector: Optional[str] = None,
) -> dict[str, Any]:
    """Build the ``rolling_*`` feature block for ``symbol`` from a sliding result."""
    out: dict[str, Any] = {k: None for k in _ROLLING_KEYS}
    calcs = (result or {}).get("calculations") if isinstance(result, dict) else None
    if not calcs:
        return out
    sym = symbol.upper()
    out["rolling_return_mean"] = _scalar(calcs, "MEAN", sym)
    out["rolling_return_median"] = _scalar(calcs, "MEDIAN", sym)
    out["rolling_cumulative_return"] = _scalar(calcs, "CUMULATIVE_RETURN", sym)
    out["rolling_stddev"] = _scalar(calcs, "STDDEV", sym)
    out["rolling_variance"] = _scalar(calcs, "VARIANCE", sym)
    out["rolling_correlation_to_spy"] = relationship_value(calcs, "CORRELATION", sym, spy)
    out["rolling_correlation_to_qqq"] = relationship_value(calcs, "CORRELATION", sym, qqq)
    if sector:
        out["rolling_correlation_to_sector"] = relationship_value(calcs, "CORRELATION", sym, sector)
    out["rolling_covariance_to_spy"] = relationship_value(calcs, "COVARIANCE", sym, spy)
    out["rolling_autocorrelation"] = _scalar(calcs, "AUTOCORRELATION", sym)

    # Slope-derived regime reads (from the sliding-window per-symbol series slope).
    vol_slope = _scalar_slope(calcs, "STDDEV", sym)
    corr_slope = _scalar_slope(calcs, "CORRELATION", sym)
    out["rolling_volatility_slope"] = vol_slope
    out["rolling_correlation_slope"] = corr_slope

    # rolling_risk_regime_score [0,1]: higher = riskier (vol expanding + crowding).
    crowd = abs(out["rolling_correlation_to_spy"]) if out["rolling_correlation_to_spy"] is not None else 0.0
    vol_exp = _clamp((vol_slope or 0.0) * 50.0, 0.0, 1.0)
    out["rolling_risk_regime_score"] = round(0.5 * vol_exp + 0.5 * crowd, 4)

    # rolling_relationship_shift_score [0,1]: magnitude of correlation change.
    out["rolling_relationship_shift_score"] = round(_clamp(abs(corr_slope or 0.0) * 50.0, 0.0, 1.0), 4)
    return out


# =============================================================================
# §5 Hidden-alpha candidates
# =============================================================================

def build_hidden_alpha_features(fixed: dict[str, Any], rolling: dict[str, Any]) -> dict[str, Any]:
    """Derive bounded hidden-alpha candidate scores from fixed + rolling blocks."""
    out: dict[str, Any] = {k: 0.0 for k in _HIDDEN_KEYS}

    ac1 = _to_f(fixed.get("fixed_autocorrelation_lag1")) or 0.0
    cum = _to_f(fixed.get("fixed_cumulative_return"))
    sd = _to_f(fixed.get("fixed_return_stddev"))
    mdd = _to_f(fixed.get("fixed_max_drawdown"))
    corr_spy = _to_f(fixed.get("fixed_correlation_to_spy"))

    corr_slope = _to_f(rolling.get("rolling_correlation_slope")) or 0.0
    vol_slope = _to_f(rolling.get("rolling_volatility_slope")) or 0.0
    shift = _to_f(rolling.get("rolling_relationship_shift_score")) or 0.0

    # Autocorrelation regime: positive -> momentum persistence, negative -> mean reversion.
    out["hidden_autocorrelation_momentum_score"] = round(max(0.0, _clamp(ac1)), 4)
    out["hidden_autocorrelation_mean_reversion_score"] = round(max(0.0, _clamp(-ac1)), 4)

    # Correlation break vs reversion (sign of rolling correlation slope).
    out["hidden_correlation_break_score"] = round(_clamp(-corr_slope * 50.0, 0.0, 1.0), 4)
    out["hidden_correlation_reversion_score"] = round(_clamp(corr_slope * 50.0, 0.0, 1.0), 4)

    # Volatility expansion / contraction (sign of rolling stddev slope).
    out["hidden_volatility_expansion_score"] = round(_clamp(vol_slope * 50.0, 0.0, 1.0), 4)
    out["hidden_volatility_contraction_score"] = round(_clamp(-vol_slope * 50.0, 0.0, 1.0), 4)

    # Risk-adjusted momentum: cumulative return per unit of volatility (Sharpe-like).
    if cum is not None and sd not in (None, 0.0):
        out["hidden_risk_adjusted_momentum_score"] = round(_clamp(cum / sd, -1.0, 1.0), 4)

    # Drawdown recovery: positive cumulative return despite a deep drawdown.
    if cum is not None and mdd is not None and mdd < 0:
        out["hidden_drawdown_recovery_score"] = round(_clamp((cum) / (abs(mdd) + 1e-6), 0.0, 1.0), 4)

    # Relationship shift passthrough + market independence.
    out["hidden_relationship_shift_score"] = round(_clamp(shift, 0.0, 1.0), 4)
    if corr_spy is not None:
        out["hidden_market_independence_score"] = round(_clamp(1.0 - abs(corr_spy), 0.0, 1.0), 4)
    return out


# =============================================================================
# §5 Interactions (analytics × existing event signals)
# =============================================================================

def build_analytics_interactions(features: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Cross analytics features with existing per-event signals (§5 interactions).

    ``event`` is the live per-symbol dict carrying upstream signals (price
    momentum, RSI, IV, relative strength, sector strength, news sentiment, volume
    acceleration, reversal pressure). Each interaction is the product of a bounded
    analytics read and a bounded event read; missing inputs yield 0.0.
    """
    def ev(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = _to_f(event.get(k))
            if v is not None:
                return v
        return default

    f = lambda k: _to_f(features.get(k)) or 0.0  # noqa: E731

    ac1 = f("fixed_autocorrelation_lag1")
    out = {
        "ix_autocorr_x_price_momentum": round(ac1 * ev("price_momentum", "realtime_quote_momentum"), 4),
        "ix_autocorr_x_rsi": round(ac1 * _clamp((ev("rsi", default=50.0) - 50.0) / 50.0), 4),
        "ix_rolling_stddev_x_iv": round(f("rolling_stddev") * ev("iv_rank", "implied_volatility", "iv"), 4),
        "ix_corr_spy_x_rel_strength": round(f("rolling_correlation_to_spy") * ev("relative_strength", "rel_strength"), 4),
        "ix_maxdd_x_reversal": round(f("fixed_max_drawdown") * ev("reversal_pressure"), 4),
        "ix_cumret_x_news_sentiment": round(f("fixed_cumulative_return") * ev("news_sentiment", "sentiment_mean_60m"), 4),
        "ix_corr_shift_x_sector_strength": round(f("rolling_relationship_shift_score") * ev("sector_strength", "sector_regime"), 4),
        "ix_vol_slope_x_volume_accel": round(f("rolling_volatility_slope") * ev("volume_acceleration"), 4),
    }
    return out


# =============================================================================
# Top-level builder
# =============================================================================

def build_analytics_features(
    symbol: str,
    fixed_result: Optional[dict[str, Any]] = None,
    sliding_result: Optional[dict[str, Any]] = None,
    spy: str = "SPY",
    qqq: str = "QQQ",
    sector: Optional[str] = None,
    event: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the full ``advanced_analytics`` feature block for ``symbol``.

    Combines the fixed-window (§2), sliding-window (§3), and hidden-alpha (§5)
    blocks, and — when ``event`` is supplied — the §5 interactions. Always returns
    the complete schema; absent inputs map to neutral values.
    """
    feats = neutral_analytics_features()
    fixed = build_fixed_window_features(symbol, fixed_result, spy, qqq, sector)
    rolling = build_sliding_window_features(symbol, sliding_result, spy, qqq, sector)
    hidden = build_hidden_alpha_features(fixed, rolling)
    feats.update(fixed)
    feats.update(rolling)
    feats.update(hidden)

    fixed_avail = isinstance(fixed_result, dict) and fixed_result.get("available")
    sliding_avail = isinstance(sliding_result, dict) and sliding_result.get("available")
    feats["analytics_available"] = 1 if (fixed_avail or sliding_avail) else 0

    if event is not None:
        feats.update(build_analytics_interactions(feats, event))
    return feats
