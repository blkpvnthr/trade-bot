from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def rank_pct(series: pd.Series) -> pd.Series:
    return series.astype(float).rank(pct=True).fillna(0.0)


def add_cross_sectional_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks symbols against each other.
    This answers: is this stock stronger than the rest of today's universe?
    """
    if df.empty:
        return df

    out = df.copy()

    for col in [
        "ret_1d", "ret_3d", "ret_5d",
        "momentum_score", "breakout_score",
        "trade_confidence", "setup_score",
        "liquidity_score", "model_probability",
    ]:
        if col not in out.columns:
            out[col] = 0.0

    out["cs_return_rank"] = (
        0.20 * rank_pct(out["ret_1d"]) +
        0.25 * rank_pct(out["ret_3d"]) +
        0.25 * rank_pct(out["ret_5d"]) +
        0.30 * rank_pct(out["momentum_score"])
    )

    out["cs_quality_rank"] = (
        0.30 * rank_pct(out["breakout_score"]) +
        0.25 * rank_pct(out["trade_confidence"]) +
        0.25 * rank_pct(out["setup_score"]) +
        0.20 * rank_pct(out["liquidity_score"])
    )

    out["cross_sectional_score"] = (
        0.55 * out["cs_return_rank"] +
        0.45 * out["cs_quality_rank"]
    ).clip(0, 1)

    return out


def time_series_confirmation(row: pd.Series) -> float:
    """
    Confirms the asset is moving correctly by itself.
    This answers: is this stock's own trend persistent right now?
    """
    close = _f(row.get("close", row.get("price")))
    vwap = _f(row.get("vwap", row.get("running_vwap", close)), close)
    rsi = _f(row.get("rsi14", row.get("rsi_14")), 50)
    ret_1d = _f(row.get("ret_1d"))
    ret_3d = _f(row.get("ret_3d"))
    ret_5d = _f(row.get("ret_5d"))
    momentum = _f(row.get("momentum_score"))
    breakout = _f(row.get("breakout_score"))
    model_prob = _f(row.get("model_probability", row.get("ml_next_bar_probability")), 0.5)

    score = 0.0

    if close > 0 and vwap > 0 and close > vwap:
        score += 0.18
    if ret_1d > 0:
        score += 0.12
    if ret_3d > 0:
        score += 0.12
    if ret_5d > 0:
        score += 0.10
    if momentum >= 0.60:
        score += 0.16
    if breakout >= 0.55:
        score += 0.14
    if model_prob >= 0.58:
        score += 0.14
    if 52 <= rsi <= 75:
        score += 0.14

    return float(np.clip(score, 0, 1))


def add_execution_alpha_scores(df: pd.DataFrame, market_context: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Final blended execution score:
    cross-sectional strength + time-series trend confirmation + regime adjustment.
    """
    if df.empty:
        return df

    out = add_cross_sectional_scores(df)
    out["time_series_score"] = out.apply(time_series_confirmation, axis=1)

    market_context = market_context or {}
    bias = str(market_context.get("overall_market_bias", market_context.get("market_bias", ""))).lower()
    permission = str(market_context.get("trade_permission", "")).lower()

    risk_off = bias in {"bearish", "risk_off", "defensive"} or permission == "halt_new_buys"
    trend_on = bias in {"bullish", "risk_on", "trend_risk_on"}

    # In trend markets, favor time-series continuation.
    # In choppy markets, require better relative strength.
    if trend_on:
        cs_w, ts_w = 0.40, 0.60
    else:
        cs_w, ts_w = 0.55, 0.45

    out["execution_alpha_score"] = (
        cs_w * out["cross_sectional_score"] +
        ts_w * out["time_series_score"]
    ).clip(0, 1)

    if risk_off:
        out["execution_alpha_score"] *= 0.70

    out["execution_alpha_pass"] = out["execution_alpha_score"] >= 0.70

    return out