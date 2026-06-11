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

    # --- Mean-variance (scipy) optimizer knobs --------------------------------
    # Used only by the max_return / max_sharpe / min_variance modes in the CLI.
    # These are additive defaults; the legacy heuristic allocator above ignores
    # them, so the rebalancer's behaviour is unchanged.
    risk_free_rate: float = 0.045          # annualized, used for Sharpe + cash sleeve
    max_portfolio_vol: float = 0.25        # annualized vol ceiling (--max-vol)
    min_sharpe: float | None = None        # optional Sharpe floor (--min-sharpe)
    max_etf_weight: float = 0.40           # aggregate ETF exposure cap (fraction of equity)
    mvo_return_lookback_days: int = 180    # daily-bar lookback for the covariance matrix


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


# ===========================================================================
# Mean-variance optimizer (scipy.optimize.minimize)
#
# A genuine covariance-aware optimizer used by the CLI modes:
#   max_return   maximize expected annual return s.t. a vol ceiling
#   max_sharpe   maximize the Sharpe ratio (used as the safe fallback)
#   min_variance maximize -risk (lowest-vol feasible portfolio)
#
# All functions below are pure (no I/O): they take expected-return and
# covariance arrays and return weights/metrics, so they are unit-testable
# without a broker connection. Data loading lives in the __main__ CLI block.
# ===========================================================================

TRADING_DAYS = 252

# Static ETF universe for the aggregate ETF cap. Augmented at runtime by an
# "is_etf" column on the candidate frame if one is present.
KNOWN_ETFS = {
    # broad index / cap-weighted
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "RSP", "MDY", "VEA", "VWO",
    "EFA", "EEM", "ACWI", "QQQM",
    # sector SPDRs + variants
    "XLK", "XLC", "XLY", "XLP", "XLV", "XLF", "XLI", "XLE", "XLB", "XLU", "XLRE",
    "SMH", "SOXX", "XBI", "IBB", "XHB", "XRT", "KRE", "XME", "XOP", "GDX", "GDXJ",
    # bonds / rates / dollar
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIP", "UUP",
    # volatility / leveraged / inverse
    "VXX", "UVXY", "SVXY", "TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS",
    "TNA", "TZA", "UPRO", "SPXU", "UDOW", "SDOW",
    # commodities / themes
    "GLD", "SLV", "USO", "UNG", "ARKK", "ARKG", "XME",
}


def is_etf_symbol(symbol: str, extra_etfs: set[str] | None = None) -> bool:
    sym = str(symbol or "").upper().strip()
    if extra_etfs and sym in extra_etfs:
        return True
    return sym in KNOWN_ETFS


def annualize_expected_return(r_5d: float) -> float:
    """
    Convert a ~5-trading-day expected-return estimate into an annualized figure.

    We scale linearly (r * 252/5) rather than compounding, because these are
    forward *alpha estimates*, not realized paths -- compounding a noisy 5-day
    edge 50x produces absurd numbers. The result is clipped to a sane band so a
    single optimistic name cannot dominate the max-return objective.
    """
    try:
        r = float(r_5d)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(r):
        return 0.0
    annual = r * (TRADING_DAYS / 5.0)
    return float(np.clip(annual, -0.75, 2.0))


def condition_covariance(cov: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    """
    Shrink a sample covariance matrix toward its diagonal and nudge it to be
    positive-definite. Keeps SLSQP's vol constraint well-behaved when the bar
    history is short or names are highly collinear.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return cov
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - shrinkage) * cov + shrinkage * diag
    # Symmetrize and add a tiny ridge for numerical PD-ness.
    shrunk = 0.5 * (shrunk + shrunk.T)
    eps = 1e-8 * (np.trace(shrunk) / max(n, 1) + 1e-12)
    shrunk = shrunk + eps * np.eye(n)
    return shrunk


def portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    risk_free_rate: float = 0.045,
    cash_weight: float = 0.0,
) -> dict[str, float]:
    """
    Annualized return / vol / Sharpe for a risky-weight vector.

    `weights` need not sum to 1; the leftover (`cash_weight`) is modeled as cash
    earning the risk-free rate with zero vol. Sharpe is excess return over rf.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    risky_return = float(w @ mu)
    total_return = risky_return + cash_weight * risk_free_rate
    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (total_return - risk_free_rate) / vol if vol > 1e-9 else 0.0

    return {
        "expected_return": total_return,
        "risky_return": risky_return,
        "volatility": vol,
        "sharpe": sharpe,
    }


def mean_variance_weights(
    symbols: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    mode: str = "max_return",
    cash_buffer: float = 0.08,
    max_position: float = 0.15,
    etf_mask: np.ndarray | None = None,
    etf_cap: float = 0.40,
    max_vol: float = 0.25,
    min_sharpe: float | None = None,
    risk_free_rate: float = 0.045,
) -> dict[str, object]:
    """
    Solve a long-only mean-variance program with scipy SLSQP.

    Constraints:
      * long-only, no shorts        -> bounds [0, max_position] per name
      * weights sum to 1 - cash     -> equality constraint
      * single-position cap         -> upper bound max_position
      * aggregate ETF cap           -> sum(w[etf]) <= etf_cap
      * portfolio vol ceiling       -> sqrt(w' Σ w) <= max_vol
      * optional Sharpe floor       -> (ret - rf)/vol >= min_sharpe

    Objective (scipy minimizes, so we negate):
      max_return   -> minimize -(w · mu)
      max_sharpe   -> minimize -Sharpe
      min_variance -> minimize  w' Σ w

    Returns a dict with weights, success flag, and a human-readable message.
    """
    from scipy.optimize import minimize

    n = len(symbols)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if n == 0:
        return {"success": False, "weights": np.zeros(0), "message": "no symbols", "mode": mode}

    if etf_mask is None:
        etf_mask = np.zeros(n, dtype=bool)
    etf_mask = np.asarray(etf_mask, dtype=bool)

    invested = max(0.0, 1.0 - float(cash_buffer))

    # Feasibility floor: the position cap must physically allow reaching the
    # invested budget; otherwise the equality constraint is unsatisfiable.
    capacity = max_position * n
    if capacity < invested:
        invested = capacity * 0.999

    bounds = [(0.0, max_position)] * n

    constraints: list[dict] = [
        # weights sum to the invested budget
        {"type": "eq", "fun": lambda w: float(np.sum(w) - invested)},
        # portfolio vol ceiling
        {"type": "ineq", "fun": lambda w: float(max_vol - np.sqrt(max(w @ cov @ w, 0.0)))},
    ]

    if etf_mask.any():
        constraints.append(
            {"type": "ineq", "fun": lambda w: float(etf_cap - np.sum(w[etf_mask]))}
        )

    if min_sharpe is not None:
        def _sharpe_excess(w: np.ndarray) -> float:
            vol = np.sqrt(max(w @ cov @ w, 0.0))
            if vol < 1e-9:
                return -1.0
            ret = w @ mu + (1.0 - invested) * risk_free_rate
            return float((ret - risk_free_rate) / vol - min_sharpe)

        constraints.append({"type": "ineq", "fun": _sharpe_excess})

    def neg_return(w: np.ndarray) -> float:
        return -float(w @ mu)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ cov @ w, 0.0))
        if vol < 1e-9:
            return 1e6
        ret = w @ mu + (1.0 - invested) * risk_free_rate
        return -float((ret - risk_free_rate) / vol)

    def variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    objective = {
        "max_return": neg_return,
        "max_sharpe": neg_sharpe,
        "min_variance": variance,
    }.get(mode, neg_return)

    # Seed near an equal-weight portfolio that respects the position cap.
    x0 = np.full(n, invested / n)
    x0 = np.clip(x0, 0.0, max_position)
    if x0.sum() > 0:
        x0 *= invested / x0.sum()

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    w = np.clip(np.asarray(result.x, dtype=float), 0.0, max_position)

    # Validate the solution actually honors the hard risk constraints; SLSQP can
    # report success while sitting just outside a non-linear constraint.
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    tol = 1e-4
    feasible = (
        bool(result.success)
        and abs(float(w.sum()) - invested) < 1e-3
        and vol <= max_vol + tol
    )
    if etf_mask.any():
        feasible = feasible and (float(w[etf_mask].sum()) <= etf_cap + tol)
    if min_sharpe is not None and feasible:
        m = portfolio_metrics(w, mu, cov, risk_free_rate=risk_free_rate,
                              cash_weight=1.0 - invested)
        feasible = feasible and (m["sharpe"] >= min_sharpe - 1e-3)

    return {
        "success": feasible,
        "weights": w,
        "message": str(getattr(result, "message", "")),
        "mode": mode,
        "invested": invested,
    }


def print_portfolio_summary(summary: dict[str, float]) -> None:
    print("===== PORTFOLIO OPTIMIZER SUMMARY =====")
    print(f"positions       : {summary.get('position_count', 0):.0f}")
    print(f"gross_exposure  : {summary.get('gross_exposure', 0.0):.2%}")
    print(f"cash_buffer     : {summary.get('cash_buffer', 0.0):.2%}")
    print(f"target_dollars  : ${summary.get('target_dollars', 0.0):,.2f}")
    print(f"equity          : ${summary.get('equity', 0.0):,.2f}")


# ===========================================================================
# CLI: python portfolio_optimizer.py --mode max_return
#
# Wires the mean-variance optimizer to the live candidate/positions plumbing.
# Paper-only by design (positions are read from the paper account). Nothing is
# executed against the broker here -- it prints metrics + the rebalance trades.
# ===========================================================================


def _fetch_daily_returns(symbols: list[str], lookback_days: int) -> "pd.DataFrame":
    """Daily pct-return matrix (columns=symbols) from Alpaca daily bars.

    Symbols with no/insufficient history are silently dropped; the caller falls
    back to per-name volatility for those. Uses the IEX feed (free on paper).
    """
    import os
    from datetime import datetime, timedelta

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    client = StockHistoricalDataClient(key, secret)

    syms = sorted({str(s).upper().strip() for s in symbols if str(s).strip()})
    if not syms:
        return pd.DataFrame()

    start = datetime.now() - timedelta(days=int(lookback_days) + 10)

    frames: dict[str, pd.Series] = {}
    # Request in chunks so one bad symbol doesn't poison the whole batch.
    for i in range(0, len(syms), 50):
        chunk = syms[i : i + 50]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
        except Exception as exc:  # noqa: BLE001
            print(f"[bars] chunk fetch failed ({chunk[:3]}...): {exc}")
            continue

        df = getattr(bars, "df", None)
        if df is None or df.empty:
            continue

        for sym in chunk:
            try:
                closes = df.loc[sym]["close"].astype(float)
            except Exception:
                continue
            if len(closes) >= 20:
                frames[sym] = closes.pct_change().dropna()

    if not frames:
        return pd.DataFrame()

    out = pd.DataFrame(frames).dropna(how="all")
    return out.dropna()


def _load_account_state(
    use_broker: bool,
    equity_override: float | None,
) -> tuple[float, "pd.DataFrame"]:
    """Resolve equity + current positions without forcing a broker call.

    Default source is the bot's dashboard snapshot (the same no-creds source the
    dashboard reads), which avoids re-hitting Alpaca auth (lockout risk). Falls
    back to the broker only with --broker, and to an empty book if neither
    source is available.
    """
    import json
    from pathlib import Path

    if equity_override is not None and not use_broker:
        # Caller pinned equity and wants no position context.
        return max(float(equity_override), 1.0), pd.DataFrame()

    if use_broker:
        from portfolio_rebalancer import get_account_and_positions
        account, positions = get_account_and_positions()
        equity = max(float(getattr(account, "equity", 0) or 0.0), 1.0)
        return equity, positions

    snap_path = Path(__file__).resolve().parent / "system-data" / "dashboard_snapshot.json"
    try:
        snap = json.loads(snap_path.read_text())
        equity = max(float(snap.get("account", {}).get("equity", 0) or 0.0), 1.0)
        pos_rows = []
        for p in snap.get("positions", []):
            pos_rows.append({
                "symbol": str(p.get("symbol", "")).upper(),
                "qty": float(p.get("qty", 0) or 0.0),
                "market_value": float(p.get("market_value", 0) or 0.0),
                "current_price": float(p.get("current_price", 0) or 0.0),
            })
        positions = pd.DataFrame(pos_rows)
        print(f"[account] source=dashboard_snapshot  updated={snap.get('updated_at_utc')}")
        if equity_override is not None:
            equity = max(float(equity_override), 1.0)
        return equity, positions
    except Exception as exc:  # noqa: BLE001
        print(f"[account] snapshot unavailable ({exc}); "
              f"using equity={equity_override or 100_000.0} with empty book")
        return max(float(equity_override or 100_000.0), 1.0), pd.DataFrame()


def _run_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Mean-variance portfolio optimizer (paper). "
        "max_return maximizes expected annual return under a vol ceiling."
    )
    parser.add_argument(
        "--mode",
        default="max_return",
        choices=["max_return", "max_sharpe", "min_variance"],
        help="optimization objective (default: max_return)",
    )
    parser.add_argument("--max-vol", type=float, default=0.25,
                        help="max annualized portfolio volatility (default 0.25)")
    parser.add_argument("--min-sharpe", type=float, default=None,
                        help="optional minimum Sharpe constraint (e.g. 0.50)")
    parser.add_argument("--etf-cap", type=float, default=0.40,
                        help="aggregate ETF exposure cap, fraction of equity (default 0.40)")
    parser.add_argument("--cash-buffer", type=float, default=None,
                        help="override cash buffer fraction (default: account-scaled)")
    parser.add_argument("--max-position", type=float, default=None,
                        help="override single-position cap (default: account-scaled)")
    parser.add_argument("--rf", type=float, default=0.045,
                        help="annual risk-free rate for Sharpe (default 0.045)")
    parser.add_argument("--lookback", type=int, default=180,
                        help="daily-bar lookback days for covariance (default 180)")
    parser.add_argument("--pool", type=int, default=25,
                        help="max candidate names fed to the optimizer (default 25)")
    parser.add_argument("--save", action="store_true",
                        help="also write targets/rebalance CSVs to system-data/<today>/")
    parser.add_argument("--broker", action="store_true",
                        help="fetch account/positions from Alpaca instead of the bot snapshot")
    parser.add_argument("--equity", type=float, default=None,
                        help="override account equity (e.g. for what-if sizing)")
    args = parser.parse_args()

    # Reuse the rebalancer's candidate loaders (env + CSV discovery).
    from portfolio_rebalancer import (
        ensure_optimizer_columns,
        load_candidates,
        norm,
    )

    print(f"===== PORTFOLIO OPTIMIZER (mode={args.mode}) =====")

    raw = load_candidates()
    candidates = ensure_optimizer_columns(raw)

    equity, positions = _load_account_state(use_broker=args.broker, equity_override=args.equity)

    cfg = scale_optimizer_config_for_equity(equity)
    cash_buffer = args.cash_buffer if args.cash_buffer is not None else cfg.cash_buffer_pct
    max_position = args.max_position if args.max_position is not None else cfg.max_position_weight

    print(f"[account] equity=${equity:,.2f}  open_positions={len(positions)}")
    print(f"[caps] cash_buffer={cash_buffer:.0%}  max_position={max_position:.0%}  "
          f"etf_cap={args.etf_cap:.0%}  max_vol={args.max_vol:.0%}"
          + (f"  min_sharpe={args.min_sharpe:.2f}" if args.min_sharpe is not None else ""))

    # Rank candidates with the existing heuristic, take a focused pool.
    scored = build_optimizer_scores(candidates, cfg)
    if scored.empty:
        print("[abort] no scorable candidates")
        return 1
    pool_df = scored.head(max(args.pool, cfg.max_positions)).copy()

    pool_syms = [norm(s) for s in pool_df["symbol"].tolist()]
    held_syms = [norm(s) for s in positions["symbol"].tolist()] if not positions.empty else []
    universe_request = sorted(set(pool_syms) | set(held_syms))

    print(f"[data] fetching {args.lookback}d daily bars for {len(universe_request)} symbols...")
    returns = _fetch_daily_returns(universe_request, args.lookback)
    if returns.empty or returns.shape[1] < 2:
        print("[abort] insufficient price history to build a covariance matrix")
        return 1

    # Investable universe = pool names that have usable history.
    universe = [s for s in pool_syms if s in returns.columns]
    if len(universe) < 2:
        print("[abort] fewer than 2 investable names with history")
        return 1

    price_map = dict(zip(pool_df["symbol"].map(norm), pool_df["price"].astype(float)))
    er5d_map = dict(zip(pool_df["symbol"].map(norm),
                        pool_df["expected_return_5d"].astype(float)))

    hist_mean_annual = returns.mean() * TRADING_DAYS

    def _mu_for(sym: str) -> float:
        er = er5d_map.get(sym, 0.0)
        if er and np.isfinite(er):
            return annualize_expected_return(er)
        return float(hist_mean_annual.get(sym, 0.0))

    mu = np.array([_mu_for(s) for s in universe], dtype=float)
    cov = condition_covariance(returns[universe].cov().values * TRADING_DAYS)
    etf_mask = np.array([is_etf_symbol(s) for s in universe], dtype=bool)

    # --- optimize with escalating fallback ladder --------------------------
    def _solve(mode: str, min_sharpe):
        return mean_variance_weights(
            universe, mu, cov,
            mode=mode,
            cash_buffer=cash_buffer,
            max_position=max_position,
            etf_mask=etf_mask,
            etf_cap=args.etf_cap,
            max_vol=args.max_vol,
            min_sharpe=min_sharpe,
            risk_free_rate=args.rf,
        )

    res = _solve(args.mode, args.min_sharpe)
    used_mode = args.mode

    if not res["success"] and args.mode == "max_return":
        print("\n" + "!" * 64)
        print(f"WARNING: max_return optimization FAILED ({res['message']}).")
        print("         Falling back to max_sharpe.")
        print("!" * 64 + "\n")
        res = _solve("max_sharpe", args.min_sharpe)
        used_mode = "max_sharpe"

    if not res["success"] and args.min_sharpe is not None:
        print(f"WARNING: optimization infeasible with min_sharpe={args.min_sharpe:.2f}; "
              "relaxing the Sharpe floor.")
        res = _solve("max_sharpe", None)
        used_mode = "max_sharpe"

    if not res["success"]:
        print("WARNING: vol/ETF constraints infeasible; falling back to min_variance "
              "(lowest-risk feasible portfolio).")
        res = _solve("min_variance", None)
        used_mode = "min_variance"

    if not res["success"]:
        print("[abort] optimizer could not find any feasible portfolio")
        return 1

    weights = np.asarray(res["weights"], dtype=float)
    invested = float(res["invested"])
    cash_w = 1.0 - invested

    opt = portfolio_metrics(weights, mu, cov, risk_free_rate=args.rf, cash_weight=cash_w)

    # Current-portfolio return estimate (held names mapped onto mu / hist mean).
    cur_return = cash_w_cur = 0.0
    if not positions.empty:
        held_w = 0.0
        for _, row in positions.iterrows():
            sym = norm(row["symbol"])
            w_i = float(row.get("market_value", 0.0)) / equity
            held_w += w_i
            if sym in universe:
                mu_i = mu[universe.index(sym)]
            else:
                mu_i = float(hist_mean_annual.get(sym, 0.0))
            cur_return += w_i * mu_i
        cash_w_cur = max(0.0, 1.0 - held_w)
        cur_return += cash_w_cur * args.rf
    else:
        cur_return = args.rf  # all cash

    # ---------------- report ----------------
    print("\n===== PORTFOLIO METRICS =====")
    print(f"objective used        : {used_mode}"
          + ("  (FALLBACK)" if used_mode != args.mode else ""))
    print(f"expected annual return: {opt['expected_return']:.2%}")
    print(f"annual volatility     : {opt['volatility']:.2%}   (cap {args.max_vol:.0%})")
    print(f"sharpe ratio          : {opt['sharpe']:.3f}"
          + (f"   (floor {args.min_sharpe:.2f})" if args.min_sharpe is not None else ""))
    print(f"current return est.   : {cur_return:.2%}")
    print(f"optimized return est. : {opt['expected_return']:.2%}")
    print(f"invested / cash       : {invested:.0%} / {cash_w:.0%}")

    # Optimized weights table.
    order = np.argsort(-weights)
    rows = []
    for idx in order:
        w_i = weights[idx]
        if w_i < 1e-4:
            continue
        sym = universe[idx]
        rows.append({
            "symbol": sym,
            "weight": w_i,
            "dollars": w_i * equity,
            "exp_ret": mu[idx],
            "etf": "ETF" if etf_mask[idx] else "",
        })
    weights_df = pd.DataFrame(rows)

    print("\n===== OPTIMIZED WEIGHTS =====")
    if weights_df.empty:
        print("(no funded positions)")
    else:
        print(f"{'symbol':<8}{'weight':>9}{'dollars':>14}{'exp_ret':>10}  type")
        for _, r in weights_df.iterrows():
            print(f"{r['symbol']:<8}{r['weight']:>8.2%}{('$'+format(r['dollars'],',.0f')):>14}"
                  f"{r['exp_ret']:>9.1%}  {r['etf']}")
        etf_total = float(weights_df.loc[weights_df['etf'] == 'ETF', 'weight'].sum())
        print(f"{'TOTAL':<8}{weights_df['weight'].sum():>8.2%}"
              f"{('$'+format(weights_df['dollars'].sum(),',.0f')):>14}"
              f"{'':>9}  ETF={etf_total:.1%}")

    # Rebalance trades vs current positions.
    funded = [i for i in range(len(universe)) if weights[i] >= 1e-4]
    targets = pd.DataFrame({
        "symbol": [universe[i] for i in funded],
        "price": [price_map.get(universe[i], 0.0) for i in funded],
        "target_weight": [weights[i] for i in funded],
    })
    targets["target_dollars"] = targets["target_weight"] * equity
    targets = targets[targets["price"] > 0].copy()
    targets["target_shares"] = np.floor(targets["target_dollars"] / targets["price"]).astype(int)

    rebal = apply_existing_position_adjustments(
        targets=targets,
        current_positions=positions,
        equity=equity,
        min_trade_dollars=cfg.min_trade_dollars,
    )
    exits = build_exit_orders_for_removed_symbols(
        current_positions=positions,
        target_symbols=targets["symbol"].tolist(),
    )

    print("\n===== REBALANCE TRADES =====")
    if rebal.empty and exits.empty:
        print("(already aligned -- no trades)")
    else:
        for _, r in rebal.iterrows():
            print(f"  {r['side'].upper():<4} {r['symbol']:<8} "
                  f"{abs(int(r['delta_shares']))} sh  (${abs(float(r['delta_dollars'])):,.0f})")
        for _, r in exits.iterrows():
            print(f"  SELL {r['symbol']:<8} {abs(int(r['delta_shares']))} sh  "
                  f"(${abs(float(r['delta_dollars'])):,.0f})  [exit: not in target set]")

    if args.save:
        from pathlib import Path
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        tag = _dt.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        out_dir = Path(__file__).resolve().parent / "system-data" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        tpath = out_dir / f"{tag}_mvo_{used_mode}_targets.csv"
        targets.to_csv(tpath, index=False)
        print(f"\n[saved] {tpath}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_cli())