#!/usr/bin/env python3
"""
portfolio_optimizer.py

High-conviction portfolio optimizer for the daily continuation system.

Purpose:
    - Convert ranked model candidates into target portfolio weights.
    - Favor max return + max Sharpe setups.
    - Penalize volatility, reversal risk, weak confidence, and concentration.
    - Keep cash healthy.
    - Cap max position size.
    - Produce target dollar allocations for Alpaca execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OptimizerConfig:
    # This is overwritten by optimize_portfolio(equity=...). Keep a sane default
    # for offline tests, but never rely on it for live sizing.
    starting_equity: float = 100_000.0
    cash_buffer_pct: float = 0.08
    max_gross_exposure: float = 0.92
    max_position_weight: float = 0.15
    min_position_weight: float = 0.02
    max_positions: int = 10
    min_trade_dollars: float = 100.0
    dynamic_account_scaling: bool = True

    confidence_power: float = 1.75
    return_weight: float = 0.30
    sharpe_weight: float = 0.30
    confidence_weight: float = 0.25
    momentum_weight: float = 0.10
    trend_weight: float = 0.05

    volatility_penalty: float = 0.35
    reversal_penalty: float = 0.50
    bear_penalty: float = 0.40


def scale_optimizer_config_for_equity(
    equity: float,
    config: OptimizerConfig | None = None,
) -> OptimizerConfig:
    """
    Scale optimizer constraints from the actual broker account size.

    This prevents a $13.5K account from inheriting $100K-style position sizes.
    The caps are deliberately tiered: smaller accounts use fewer names, smaller
    order floors, and lower gross exposure; larger accounts can diversify more.
    """
    cfg = config or OptimizerConfig(starting_equity=equity)
    equity = max(float(equity or 0.0), 1.0)
    cfg.starting_equity = equity

    if not cfg.dynamic_account_scaling:
        return cfg

    # Trade floor: high enough to avoid dust, low enough not to block small accounts.
    cfg.min_trade_dollars = max(50.0, min(250.0, equity * 0.0075))

    if equity < 15_000:
        cfg.cash_buffer_pct = max(cfg.cash_buffer_pct, 0.12)
        cfg.max_gross_exposure = min(cfg.max_gross_exposure, 0.65)
        cfg.max_position_weight = min(cfg.max_position_weight, 0.15)
        cfg.min_position_weight = max(cfg.min_position_weight, 0.035)
        cfg.max_positions = min(cfg.max_positions, 4)
    elif equity < 25_000:
        cfg.cash_buffer_pct = max(cfg.cash_buffer_pct, 0.10)
        cfg.max_gross_exposure = min(cfg.max_gross_exposure, 0.75)
        cfg.max_position_weight = min(cfg.max_position_weight, 0.18)
        cfg.min_position_weight = max(cfg.min_position_weight, 0.030)
        cfg.max_positions = min(cfg.max_positions, 5)
    elif equity < 50_000:
        cfg.cash_buffer_pct = max(cfg.cash_buffer_pct, 0.09)
        cfg.max_gross_exposure = min(cfg.max_gross_exposure, 0.85)
        cfg.max_position_weight = min(cfg.max_position_weight, 0.20)
        cfg.min_position_weight = max(cfg.min_position_weight, 0.025)
        cfg.max_positions = min(cfg.max_positions, 7)
    else:
        cfg.cash_buffer_pct = max(cfg.cash_buffer_pct, 0.08)
        cfg.max_gross_exposure = min(cfg.max_gross_exposure, 0.92)
        cfg.max_position_weight = min(cfg.max_position_weight, 0.25)
        cfg.max_positions = min(cfg.max_positions, 10)

    return cfg


REQUIRED_COLUMNS = [
    "symbol",
    "price",
    "trade_confidence",
    "continuation_prob",
    "model_score",
    "expected_return_5d",
    "expected_sharpe_score",
    "trend_score",
    "momentum_score",
    "volatility20",
    "confluence_reversal_risk",
    "bear",
]


def _normalize_positive(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.fillna(0.0)
    min_v = float(s.min())
    max_v = float(s.max())

    if max_v <= min_v:
        return pd.Series(0.5, index=s.index)

    return (s - min_v) / (max_v - min_v)


def _validate_candidates(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"optimizer candidates missing columns: {missing}")


def build_optimizer_scores(
    candidates: pd.DataFrame,
    config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    config = config or OptimizerConfig()
    _validate_candidates(candidates)

    out = candidates.copy()

    for col in REQUIRED_COLUMNS:
        if col != "symbol":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["symbol", "price"]).copy()
    out = out[out["price"] > 0].copy()

    if out.empty:
        return out

    out["norm_expected_return"] = _normalize_positive(out["expected_return_5d"])
    out["norm_expected_sharpe"] = _normalize_positive(out["expected_sharpe_score"])
    out["norm_confidence"] = out["trade_confidence"].clip(0.0, 1.0)
    out["norm_momentum"] = out["momentum_score"].clip(0.0, 1.0)
    out["norm_trend"] = out["trend_score"].clip(0.0, 1.0)
    out["norm_volatility"] = _normalize_positive(out["volatility20"])

    raw_score = (
        config.return_weight * out["norm_expected_return"]
        + config.sharpe_weight * out["norm_expected_sharpe"]
        + config.confidence_weight * out["norm_confidence"]
        + config.momentum_weight * out["norm_momentum"]
        + config.trend_weight * out["norm_trend"]
    )

    penalty = (
        config.volatility_penalty * out["norm_volatility"]
        + config.reversal_penalty * out["confluence_reversal_risk"].fillna(0.0)
        + config.bear_penalty * out["bear"].fillna(0.0)
    )

    out["optimizer_score"] = (raw_score - penalty).clip(lower=0.0)

    out["optimizer_score"] = (
        out["optimizer_score"]
        * out["continuation_prob"].clip(0.0, 1.0)
        * out["model_score"].clip(0.0, 1.0)
    )

    out["allocation_score"] = (
        out["optimizer_score"] * out["trade_confidence"].clip(0.0, 1.0).pow(config.confidence_power)
    )

    out = out.sort_values("allocation_score", ascending=False).reset_index(drop=True)

    return out


def optimize_weights(
    candidates: pd.DataFrame,
    config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    config = config or OptimizerConfig()

    scored = build_optimizer_scores(candidates, config=config)

    if scored.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "price",
                "target_weight",
                "target_dollars",
                "target_shares",
                "optimizer_score",
                "allocation_score",
            ]
        )

    scored = scored.head(config.max_positions).copy()
    scored = scored[scored["allocation_score"] > 0].copy()

    if scored.empty:
        return pd.DataFrame()

    max_investable_weight = min(
        config.max_gross_exposure,
        max(0.0, 1.0 - config.cash_buffer_pct),
    )

    score_sum = scored["allocation_score"].sum()

    if score_sum <= 0:
        scored["raw_weight"] = 1.0 / len(scored)
    else:
        scored["raw_weight"] = scored["allocation_score"] / score_sum

    scored["target_weight"] = scored["raw_weight"] * max_investable_weight
    scored["target_weight"] = scored["target_weight"].clip(upper=config.max_position_weight)

    total_weight = scored["target_weight"].sum()

    if total_weight > max_investable_weight and total_weight > 0:
        scored["target_weight"] *= max_investable_weight / total_weight

    scored = scored[scored["target_weight"] >= config.min_position_weight].copy()

    if scored.empty:
        return pd.DataFrame()

    total_weight = scored["target_weight"].sum()

    if total_weight > max_investable_weight and total_weight > 0:
        scored["target_weight"] *= max_investable_weight / total_weight

    scored["target_dollars"] = scored["target_weight"] * config.starting_equity
    scored = scored[scored["target_dollars"] >= config.min_trade_dollars].copy()

    if scored.empty:
        return pd.DataFrame()

    scored["target_shares"] = np.floor(scored["target_dollars"] / scored["price"]).astype(int)
    scored["target_dollars"] = scored["target_shares"] * scored["price"]
    scored["target_weight"] = scored["target_dollars"] / config.starting_equity

    scored = scored[scored["target_shares"] > 0].copy()

    scored["rank"] = np.arange(1, len(scored) + 1)

    cols = [
        "rank",
        "symbol",
        "price",
        "target_weight",
        "target_dollars",
        "target_shares",
        "optimizer_score",
        "allocation_score",
        "trade_confidence",
        "continuation_prob",
        "model_score",
        "expected_return_5d",
        "expected_sharpe_score",
        "trend_score",
        "momentum_score",
        "volatility20",
    ]

    available_cols = [c for c in cols if c in scored.columns]

    return scored[available_cols].sort_values("target_weight", ascending=False).reset_index(drop=True)


def apply_existing_position_adjustments(
    targets: pd.DataFrame,
    current_positions: pd.DataFrame,
    equity: float,
    min_trade_dollars: float = 100.0,
) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()

    out = targets.copy()

    if current_positions is None or current_positions.empty:
        out["current_qty"] = 0.0
        out["current_market_value"] = 0.0
    else:
        pos = current_positions.copy()
        pos["symbol"] = pos["symbol"].astype(str).str.upper()
        pos["qty"] = pd.to_numeric(pos["qty"], errors="coerce").fillna(0.0)
        pos["market_value"] = pd.to_numeric(pos["market_value"], errors="coerce").fillna(0.0)

        out = out.merge(
            pos[["symbol", "qty", "market_value"]],
            on="symbol",
            how="left",
        )

        out = out.rename(
            columns={
                "qty": "current_qty",
                "market_value": "current_market_value",
            }
        )

        out["current_qty"] = out["current_qty"].fillna(0.0)
        out["current_market_value"] = out["current_market_value"].fillna(0.0)

    out["current_weight"] = out["current_market_value"] / max(float(equity), 1.0)
    out["delta_dollars"] = out["target_dollars"] - out["current_market_value"]
    out["delta_shares"] = np.floor(out["delta_dollars"] / out["price"]).astype(int)

    out["side"] = np.where(out["delta_dollars"] > 0, "buy", "sell")
    out["abs_delta_dollars"] = out["delta_dollars"].abs()

    out = out[out["abs_delta_dollars"] >= min_trade_dollars].copy()

    return out.reset_index(drop=True)


def build_exit_orders_for_removed_symbols(
    current_positions: pd.DataFrame,
    target_symbols: list[str],
    min_position_dollars: float = 25.0,
) -> pd.DataFrame:
    if current_positions is None or current_positions.empty:
        return pd.DataFrame()

    target_set = {str(s).upper() for s in target_symbols}

    pos = current_positions.copy()
    pos["symbol"] = pos["symbol"].astype(str).str.upper()
    pos["qty"] = pd.to_numeric(pos["qty"], errors="coerce").fillna(0.0)
    pos["market_value"] = pd.to_numeric(pos["market_value"], errors="coerce").fillna(0.0)

    exits = pos[
        (~pos["symbol"].isin(target_set))
        & (pos["qty"] > 0)
        & (pos["market_value"].abs() >= min_position_dollars)
    ].copy()

    if exits.empty:
        return pd.DataFrame()

    exits["side"] = "sell"
    exits["target_weight"] = 0.0
    exits["target_dollars"] = 0.0
    exits["target_shares"] = 0
    exits["delta_shares"] = -exits["qty"].astype(int)
    exits["delta_dollars"] = -exits["market_value"]
    exits["reason"] = "removed_from_high_conviction_universe"

    return exits.reset_index(drop=True)


def summarize_portfolio_targets(targets: pd.DataFrame, equity: float) -> dict[str, float]:
    if targets.empty:
        return {
            "position_count": 0,
            "gross_exposure": 0.0,
            "cash_buffer": 1.0,
            "target_dollars": 0.0,
        }

    gross_exposure = float(targets["target_weight"].sum())
    target_dollars = float(targets["target_dollars"].sum())

    return {
        "position_count": float(len(targets)),
        "gross_exposure": gross_exposure,
        "cash_buffer": max(0.0, 1.0 - gross_exposure),
        "target_dollars": target_dollars,
        "equity": float(equity),
    }


def optimize_portfolio(
    candidates: pd.DataFrame,
    equity: float,
    current_positions: pd.DataFrame | None = None,
    config: OptimizerConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    config = scale_optimizer_config_for_equity(equity, config=config)

    targets = optimize_weights(candidates, config=config)

    rebalance_orders = apply_existing_position_adjustments(
        targets=targets,
        current_positions=current_positions if current_positions is not None else pd.DataFrame(),
        equity=equity,
        min_trade_dollars=config.min_trade_dollars,
    )

    exits = build_exit_orders_for_removed_symbols(
        current_positions=current_positions if current_positions is not None else pd.DataFrame(),
        target_symbols=targets["symbol"].tolist() if not targets.empty else [],
    )

    summary = summarize_portfolio_targets(targets, equity)

    return targets, rebalance_orders, exits, summary


def print_portfolio_summary(summary: dict[str, float]) -> None:
    print("===== PORTFOLIO OPTIMIZER SUMMARY =====")
    print(f"positions       : {summary.get('position_count', 0):.0f}")
    print(f"gross_exposure  : {summary.get('gross_exposure', 0.0):.2%}")
    print(f"cash_buffer     : {summary.get('cash_buffer', 0.0):.2%}")
    print(f"target_dollars  : ${summary.get('target_dollars', 0.0):,.2f}")
    print(f"equity          : ${summary.get('equity', 0.0):,.2f}")


# =============================================================================
# Objective 2: covariance-aware sizing (ADDITIVE).
#
# This is an alternate, opt-in sizing path. It does NOT touch
# optimize_weights()/optimize_portfolio() (the live default, score-weighted
# path), which portfolio_rebalancer.py depends on. The covariance-aware path
# adds:
#   * single_expected_score()     -- de-duplicated (averaged, not multiplied)
#                                    risk-adjusted expected score per name.
#   * build_shrinkage_covariance() -- Ledoit-Wolf shrinkage + PSD validation.
#   * solve_covariance_aware_weights() -- convex mean-variance solve with
#                                    position/gross/sector/leveraged caps and a
#                                    turnover penalty.
#   * fallback_inverse_vol_weights() -- conservative deterministic fallback
#                                    used when there is no usable covariance.
#   * optimize_weights_covariance_aware() -- end-to-end driver that picks the
#                                    shrinkage path when history is available
#                                    and the inverse-vol fallback otherwise.
# =============================================================================


@dataclass
class CovarianceAwareConfig:
    """Tunables for the covariance-aware sizing path.

    Kept separate from OptimizerConfig so the existing (live-default) path is
    completely unaffected. Position-weight / gross-exposure / cash-buffer caps
    are still read from OptimizerConfig; the knobs here govern the risk model
    and the additional group/leverage/turnover constraints.
    """

    # Minimum rows of return history required before trusting a sample
    # covariance (below this -> no covariance -> inverse-vol fallback).
    min_lookback_days: int = 20
    # Mean-variance trade-off: higher penalizes portfolio variance more.
    risk_aversion: float = 1.0
    # L1 penalty on |w - prev_w|: discourages churn / unnecessary turnover.
    turnover_penalty: float = 0.001
    # Max gross weight allowed in any single exposure group (sector / index).
    exposure_group_cap: float = 0.25
    # Max combined gross weight in leveraged / inverse ETFs (hard risk cap).
    leveraged_inverse_cap: float = 0.15


def single_expected_score(
    candidates: pd.DataFrame,
    config: OptimizerConfig | None = None,
) -> pd.Series:
    """Risk-adjusted expected score per candidate, with correlated confidence
    channels DE-DUPLICATED.

    The legacy path (build_optimizer_scores) multiplies several strongly
    correlated probability/confidence signals together
    (optimizer_score * continuation_prob * model_score, then
    * trade_confidence**power) -- four multiplicative passes that compound a
    single underlying "is this a good setup" belief into a near-quartic
    collapse. Here those correlated signals are AVERAGED into one confidence
    proxy that enters the score additively (weighted by confidence_weight), so
    scaling all confidence inputs by k scales the result by roughly k, not
    k**3/k**4.

    Returns a non-negative pd.Series aligned to ``candidates.index``.
    """
    config = config or OptimizerConfig()
    _validate_candidates(candidates)

    out = candidates.copy()
    for col in REQUIRED_COLUMNS:
        if col != "symbol":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if out.empty:
        return pd.Series(dtype=float, index=out.index)

    norm_expected_return = _normalize_positive(out["expected_return_5d"])
    norm_expected_sharpe = _normalize_positive(out["expected_sharpe_score"])
    norm_momentum = out["momentum_score"].fillna(0.0).clip(0.0, 1.0)
    norm_trend = out["trend_score"].fillna(0.0).clip(0.0, 1.0)
    norm_volatility = _normalize_positive(out["volatility20"])

    # ---- de-duplication: ONE averaged confidence channel, not a product ----
    confidence_proxy = pd.concat(
        [
            out["continuation_prob"].fillna(0.0).clip(0.0, 1.0),
            out["model_score"].fillna(0.0).clip(0.0, 1.0),
            out["trade_confidence"].fillna(0.0).clip(0.0, 1.0),
        ],
        axis=1,
    ).mean(axis=1)

    raw_score = (
        config.return_weight * norm_expected_return
        + config.sharpe_weight * norm_expected_sharpe
        + config.confidence_weight * confidence_proxy
        + config.momentum_weight * norm_momentum
        + config.trend_weight * norm_trend
    )

    penalty = (
        config.volatility_penalty * norm_volatility
        + config.reversal_penalty * out["confluence_reversal_risk"].fillna(0.0)
        + config.bear_penalty * out["bear"].fillna(0.0)
    )

    score = (raw_score - penalty).clip(lower=0.0)
    score.name = "expected_score"
    return score


def build_shrinkage_covariance(
    returns_history: pd.DataFrame | None,
    min_lookback_days: int = 20,
) -> tuple[np.ndarray, list[str]] | None:
    """Ledoit-Wolf shrinkage covariance of daily returns, with PSD validation.

    Returns ``(cov, order)`` where ``cov`` is an ``(n, n)`` numpy covariance
    matrix and ``order`` is the list of symbols matching ``cov``'s row/column
    order. Returns ``None`` (caller falls back to inverse-vol) when there is
    not enough usable history:

      * fewer than ``min_lookback_days`` clean observations, or
      * fewer than 2 symbols (a covariance needs at least a pair).
    """
    if returns_history is None:
        return None

    hist = returns_history.copy()
    if not isinstance(hist, pd.DataFrame) or hist.shape[1] == 0:
        return None

    # Numeric, drop all-NaN columns, then drop any row with a missing value so
    # every pair is estimated over the same aligned sample.
    hist = hist.apply(pd.to_numeric, errors="coerce")
    hist = hist.replace([np.inf, -np.inf], np.nan)
    hist = hist.dropna(axis=1, how="all")
    hist = hist.dropna(axis=0, how="any")

    if hist.shape[1] < 2:
        return None
    if hist.shape[0] < int(min_lookback_days):
        return None

    order = [str(c) for c in hist.columns]

    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(hist.values)
        cov = np.asarray(lw.covariance_, dtype=float)
    except Exception:
        # Deterministic manual fallback: sample covariance shrunk toward a
        # scaled-identity target (same spirit as Ledoit-Wolf).
        x = hist.values
        sample = np.cov(x, rowvar=False)
        mu = np.trace(sample) / sample.shape[0]
        target = mu * np.eye(sample.shape[0])
        shrink = 0.1
        cov = (1.0 - shrink) * sample + shrink * target

    # ---- PSD validation: symmetrize and clip any tiny negative eigenvalues --
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if (eigvals < 0).any():
        eigvals = np.clip(eigvals, 0.0, None)
        cov = (eigvecs * eigvals) @ eigvecs.T
        cov = 0.5 * (cov + cov.T)

    return cov, order


def fallback_inverse_vol_weights(
    candidates: pd.DataFrame,
    config: OptimizerConfig | None = None,
    max_investable_weight: float = 0.5,
) -> pd.DataFrame:
    """Conservative deterministic fallback: inverse-volatility weights.

    Used when no trustworthy covariance is available. Lower-volatility names
    receive proportionally more weight. Hard risk constraints are still
    honored here -- per-position cap (``config.max_position_weight``) and the
    gross-exposure cap (``max_investable_weight``) -- so the fallback can never
    size up past the limits the covariance path respects.

    Expects an ``expected_score`` column (names with non-positive score are
    excluded when at least one positive-score name exists). Returns the input
    frame with a ``target_weight`` column added.
    """
    config = config or OptimizerConfig()
    out = candidates.copy()

    if out.empty:
        out["target_weight"] = pd.Series(dtype=float)
        return out

    vol = pd.to_numeric(out.get("volatility20"), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid = vol[(vol > 0) & np.isfinite(vol)]
    if not valid.empty:
        fill_vol = float(valid.median())
        vol = vol.where((vol > 0) & np.isfinite(vol), fill_vol)
    else:
        # No usable volatility anywhere -> equal risk assumption.
        vol = pd.Series(1.0, index=out.index)

    inv_vol = 1.0 / vol

    # Gate on positive expected_score when that information is present.
    if "expected_score" in out.columns:
        score = pd.to_numeric(out["expected_score"], errors="coerce").fillna(0.0)
        if (score > 0).any():
            inv_vol = inv_vol.where(score > 0, 0.0)

    total = float(inv_vol.sum())
    if total <= 0 or not np.isfinite(total):
        raw_weight = pd.Series(1.0 / len(out), index=out.index)
    else:
        raw_weight = inv_vol / total

    target = raw_weight * float(max_investable_weight)
    target = target.clip(upper=config.max_position_weight)

    # Clipping only ever reduces, so the gross cap is preserved by construction.
    gross = float(target.sum())
    if gross > max_investable_weight and gross > 0:
        target = target * (max_investable_weight / gross)

    out["target_weight"] = target.fillna(0.0)
    return out


def _is_leveraged_or_inverse(symbol: str) -> bool:
    """True if a symbol is a leveraged or inverse ETF per the universe map."""
    try:
        import universe as uni

        info = uni.classify_symbol(symbol)
        itype = getattr(info, "instrument_type", None)
        if itype in (uni.InstrumentType.LEVERAGED_ETF, uni.InstrumentType.INVERSE_ETF):
            return True
        return abs(float(getattr(info, "leverage", 1.0) or 1.0)) > 1.0
    except Exception:
        return False


def _apply_group_and_leverage_caps(
    weights: pd.Series,
    exposure_groups: dict[str, str],
    leveraged: set[str],
    exposure_group_cap: float,
    leveraged_inverse_cap: float,
    max_investable_weight: float,
) -> pd.Series:
    """Deterministically clamp a weight vector to the group/leverage/gross caps
    by proportionally scaling down any violating group. Used so the inverse-vol
    fallback cannot bypass the hard risk constraints the solver enforces."""
    w = weights.copy().astype(float)

    # Exposure-group caps.
    groups: dict[str, list] = {}
    for sym in w.index:
        groups.setdefault(exposure_groups.get(sym, sym), []).append(sym)
    for members in groups.values():
        grp_sum = float(w.loc[members].sum())
        if grp_sum > exposure_group_cap > 0:
            w.loc[members] = w.loc[members] * (exposure_group_cap / grp_sum)

    # Leveraged / inverse combined cap.
    lev_members = [s for s in w.index if s in leveraged]
    if lev_members:
        lev_sum = float(w.loc[lev_members].sum())
        if lev_sum > leveraged_inverse_cap > 0:
            w.loc[lev_members] = w.loc[lev_members] * (leveraged_inverse_cap / lev_sum)

    # Gross cap.
    gross = float(w.sum())
    if gross > max_investable_weight and gross > 0:
        w = w * (max_investable_weight / gross)

    return w


def solve_covariance_aware_weights(
    expected_scores: pd.Series,
    cov: np.ndarray,
    symbols: list[str],
    prev_weights: pd.Series,
    exposure_groups: dict[str, str],
    leveraged: set[str],
    config: OptimizerConfig,
    cov_config: CovarianceAwareConfig,
    max_investable_weight: float,
) -> pd.Series | None:
    """Convex mean-variance weight solve subject to hard risk constraints.

    Maximizes::

        score . w  -  risk_aversion * w' Cov w  -  turnover_penalty * |w - prev|

    subject to (all hard constraints):
        * 0 <= w_i <= config.max_position_weight          (per-position cap)
        * sum(w) <= max_investable_weight                 (gross-exposure cap)
        * sum_{i in group g} w_i <= exposure_group_cap    (sector/index cap)
        * sum_{i in leveraged} w_i <= leveraged_inverse_cap (leverage cap)

    Returns a pd.Series of weights indexed by ``symbols``, or ``None`` if the
    solver fails to produce a usable solution.
    """
    n = len(symbols)
    if n == 0:
        return pd.Series(dtype=float)

    scores_vec = (
        pd.to_numeric(expected_scores.reindex(symbols), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    prev_vec = (
        pd.to_numeric(prev_weights.reindex(symbols), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)

    try:
        import cvxpy as cp

        w = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(
            scores_vec @ w
            - float(cov_config.risk_aversion) * cp.quad_form(w, cp.psd_wrap(cov))
            - float(cov_config.turnover_penalty) * cp.norm1(w - prev_vec)
        )

        constraints = [
            w <= float(config.max_position_weight),
            cp.sum(w) <= float(max_investable_weight),
        ]

        groups: dict[str, list[int]] = {}
        for i, sym in enumerate(symbols):
            groups.setdefault(exposure_groups.get(sym, sym), []).append(i)
        for idx in groups.values():
            constraints.append(cp.sum(w[idx]) <= float(cov_config.exposure_group_cap))

        lev_idx = [i for i, sym in enumerate(symbols) if sym in leveraged]
        if lev_idx:
            constraints.append(
                cp.sum(w[lev_idx]) <= float(cov_config.leveraged_inverse_cap)
            )

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if w.value is None or prob.status not in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        ):
            return None

        vals = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
        return pd.Series(vals, index=symbols)
    except Exception:
        return None


def optimize_weights_covariance_aware(
    candidates: pd.DataFrame,
    returns_history: pd.DataFrame | None = None,
    config: OptimizerConfig | None = None,
    cov_config: CovarianceAwareConfig | None = None,
    prev_weights: pd.Series | None = None,
    equity: float | None = None,
) -> pd.DataFrame:
    """End-to-end covariance-aware sizing.

    Picks the covariance-aware solve when usable return history is available
    (>= ``cov_config.min_lookback_days`` clean rows across >= 2 candidate
    symbols); otherwise uses the conservative inverse-vol fallback. Both paths
    honor the same hard risk caps. The chosen path is recorded per row in a
    ``sizing_method`` column (``"COVARIANCE_AWARE_SHRINKAGE"`` or
    ``"FALLBACK_INVERSE_VOL"``).

    This is purely additive and never used by the live-default
    optimize_weights()/optimize_portfolio() path.
    """
    config = config or OptimizerConfig()
    cov_config = cov_config or CovarianceAwareConfig()

    empty = pd.DataFrame(
        columns=[
            "symbol",
            "price",
            "target_weight",
            "target_dollars",
            "target_shares",
            "expected_score",
            "sizing_method",
        ]
    )

    if candidates is None or candidates.empty:
        return empty

    _validate_candidates(candidates)

    if equity is not None:
        config = scale_optimizer_config_for_equity(equity, config=config)
    equity_val = float(config.starting_equity)

    work = candidates.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work.dropna(subset=["symbol", "price"])
    work = work[work["price"] > 0].copy()
    if work.empty:
        return empty

    work["expected_score"] = single_expected_score(work, config).to_numpy()
    work = work.reset_index(drop=True)

    symbols = [str(s) for s in work["symbol"].tolist()]
    max_investable_weight = min(
        config.max_gross_exposure,
        max(0.0, 1.0 - config.cash_buffer_pct),
    )

    exposure_groups: dict[str, str] = {}
    leveraged: set[str] = set()
    for sym in symbols:
        try:
            import universe as uni

            exposure_groups[sym] = uni.classify_symbol(sym).exposure_group
        except Exception:
            exposure_groups[sym] = sym
        if _is_leveraged_or_inverse(sym):
            leveraged.add(sym)

    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=symbols)

    # ---- choose path --------------------------------------------------------
    cov_result = None
    if returns_history is not None:
        hist_cols = [c for c in returns_history.columns if str(c) in set(symbols)]
        if hist_cols:
            cov_result = build_shrinkage_covariance(
                returns_history[hist_cols], min_lookback_days=cov_config.min_lookback_days
            )

    weights: pd.Series | None = None
    sizing_method = "FALLBACK_INVERSE_VOL"

    if cov_result is not None:
        cov, order = cov_result
        solve_symbols = [s for s in symbols if s in set(order)]
        if len(solve_symbols) >= 2:
            idx = [order.index(s) for s in solve_symbols]
            cov_aligned = cov[np.ix_(idx, idx)]
            scores_series = work.set_index("symbol")["expected_score"].reindex(
                solve_symbols
            )
            solved = solve_covariance_aware_weights(
                scores_series,
                cov_aligned,
                solve_symbols,
                prev_weights.reindex(solve_symbols).fillna(0.0),
                {s: exposure_groups[s] for s in solve_symbols},
                {s for s in solve_symbols if s in leveraged},
                config,
                cov_config,
                max_investable_weight,
            )
            if solved is not None:
                weights = solved.reindex(symbols).fillna(0.0)
                sizing_method = "COVARIANCE_AWARE_SHRINKAGE"

    if weights is None:
        fb = fallback_inverse_vol_weights(
            work, config, max_investable_weight=max_investable_weight
        )
        fb_weights = pd.Series(fb["target_weight"].to_numpy(), index=symbols)
        # Fallback must NOT bypass the hard group/leverage/gross caps.
        weights = _apply_group_and_leverage_caps(
            fb_weights,
            exposure_groups,
            leveraged,
            cov_config.exposure_group_cap,
            cov_config.leveraged_inverse_cap,
            max_investable_weight,
        )
        sizing_method = "FALLBACK_INVERSE_VOL"

    work = work.set_index("symbol")
    work["target_weight"] = weights.reindex(work.index).fillna(0.0)
    work = work.reset_index()

    # ---- convert weights -> dollars -> whole shares -------------------------
    work["target_dollars"] = work["target_weight"] * equity_val
    work["target_shares"] = np.floor(work["target_dollars"] / work["price"]).astype(int)
    work["target_dollars"] = work["target_shares"] * work["price"]
    work["target_weight"] = work["target_dollars"] / max(equity_val, 1.0)

    work = work[work["target_shares"] > 0].copy()
    work = work[work["target_dollars"] >= config.min_trade_dollars].copy()
    work = work[work["target_weight"] >= config.min_position_weight].copy()

    if work.empty:
        return empty

    work["sizing_method"] = sizing_method
    work = work.sort_values("target_weight", ascending=False).reset_index(drop=True)
    work["rank"] = np.arange(1, len(work) + 1)

    cols = [
        "rank",
        "symbol",
        "price",
        "target_weight",
        "target_dollars",
        "target_shares",
        "expected_score",
        "sizing_method",
    ]
    available = [c for c in cols if c in work.columns]
    return work[available]