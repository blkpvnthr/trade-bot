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