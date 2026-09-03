#!/usr/bin/env python3
"""
buffett_portfolio_optimizer.py — Efficient-Frontier optimizer for the Buffett layer.

Consumes Black-Litterman posterior returns + covariance (buffett_black_litterman) and
produces TARGET WEIGHTS via a blend of four objectives (min-vol, max-Sharpe, max
quadratic utility, efficient-return) with L2 regularization (diversification) and a
turnover penalty (holding discipline). Then a walk-forward harness compares six
optimizer configurations out-of-sample and selects the best by an evidence score, so
Black-Litterman has to EARN its place.

Target weights answer "how much to own". Composite DCA answers "when to buy". RiskManager
decides "whether allowed". This module is advisory: it never places orders, and every
layer falls back safely so no optimizer failure can stop the Buffett/Alpha layers or run.py.
"""
from __future__ import annotations

import json
import math
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import buffett_black_litterman as bbl

logger = logging.getLogger(__name__)

try:
    from pypfopt import EfficientFrontier, objective_functions
    _PYPFOPT = True
except Exception:  # pragma: no cover
    _PYPFOPT = False

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

DIVERSIFICATION_REPORT = "buffett_diversification_report.json"
OPTIMIZER_COMPARISON = "buffett_optimizer_comparison.json"
ACTIVE_OPTIMIZER = "active_buffett_optimizer.json"
OPTIMIZER_PERFORMANCE = "optimizer_performance_report.json"

OBJECTIVES = ("min_volatility", "max_sharpe", "max_quadratic_utility", "efficient_return")
BLEND_WEIGHTS = {"min_volatility": 0.20, "max_sharpe": 0.30,
                 "max_quadratic_utility": 0.25, "efficient_return": 0.25}
CONFIGS = ("equal_weight", "conviction_allocation", "ef_historical",
           "bl_ef", "bl_ef_l2", "bl_ef_l2_turnover")


def _research_dir(date_tag: str | None = None) -> Path:
    d = DAILY_OUTPUTS / (date_tag or DATA_TAG)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


@dataclass
class OptimizerConfig:
    enabled: bool = os.getenv("ENABLE_BUFFETT_OPTIMIZER", "true").lower() in {"1", "true"}
    risk_aversion: float = float(os.getenv("BUFFETT_OPT_RISK_AVERSION", "1.0"))
    target_return: float = float(os.getenv("BUFFETT_OPT_TARGET_RETURN", "0.12"))
    use_l2: bool = os.getenv("BUFFETT_OPT_USE_L2_REGULARIZATION", "true").lower() in {"1", "true"}
    l2_gamma: float = float(os.getenv("BUFFETT_OPT_L2_GAMMA", "0.10"))
    turnover_penalty: float = float(os.getenv("BUFFETT_OPT_TRANSACTION_COST_PENALTY", "0.001"))
    risk_free_rate: float = float(os.getenv("BUFFETT_BL_RISK_FREE", os.getenv("OPTIONS_RISK_FREE_RATE", "0.043")))
    max_weight: float = float(os.getenv("MAX_SINGLE_STOCK_WEIGHT", "0.10"))
    # Walk-forward
    wf_train: int = int(os.getenv("BUFFETT_OPT_WF_TRAIN", "63"))     # ~3 months daily
    wf_test: int = int(os.getenv("BUFFETT_OPT_WF_TEST", "21"))       # ~1 month
    wf_cost_bps: float = float(os.getenv("BUFFETT_OPT_WF_COST_BPS", "5"))
    wf_slippage_bps: float = float(os.getenv("BUFFETT_OPT_WF_SLIPPAGE_BPS", "2"))
    periods_per_year: int = int(os.getenv("BUFFETT_OPT_PPY", "252"))
    default_active: str = os.getenv("BUFFETT_OPT_DEFAULT", "bl_ef_l2_turnover")


# ── single-objective optimization ────────────────────────────────────────────────
def _feasible_max_weight(n: int, cap: float) -> float:
    return float(min(1.0, max(cap, 2.0 / n))) if n > 0 else 1.0


def _equal_weights(symbols) -> dict:
    n = len(symbols)
    return {s: 1.0 / n for s in symbols} if n else {}


def _normalize(weights: dict) -> dict:
    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    tot = sum(w.values())
    if tot <= 0:
        return _equal_weights(list(weights.keys()))
    return {k: v / tot for k, v in w.items()}


def optimize_single(mu: pd.Series, S: pd.DataFrame, objective: str, cfg: OptimizerConfig,
                    current_weights: dict | None = None, use_l2: bool | None = None,
                    use_turnover: bool = False) -> dict | None:
    """
    Solve one efficient-frontier objective with optional L2 + turnover penalties.
    Returns a weight dict (sums to 1) or None on failure. max_sharpe omits added convex
    objectives (PyPortfolioOpt forbids them on the Sharpe transformation).
    """
    use_l2 = cfg.use_l2 if use_l2 is None else use_l2
    symbols = list(mu.index)
    n = len(symbols)
    if n == 0:
        return None
    # Guard against a non-finite covariance/return matrix (forces fallback).
    try:
        if not np.all(np.isfinite(S.values)) or not np.all(np.isfinite(mu.values)):
            return None
    except Exception:
        return None
    if not _PYPFOPT:
        return _scipy_optimize(mu, S, objective, cfg)
    max_w = _feasible_max_weight(n, cfg.max_weight)
    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, max_w))
        if objective != "max_sharpe":   # added objectives are incompatible with max_sharpe
            if use_l2:
                ef.add_objective(objective_functions.L2_reg, gamma=cfg.l2_gamma)
            if use_turnover and current_weights is not None and cfg.turnover_penalty > 0:
                w_prev = np.array([float(current_weights.get(a, 0.0)) for a in symbols])
                ef.add_objective(objective_functions.transaction_cost, w_prev=w_prev,
                                 k=cfg.turnover_penalty)
        if objective == "min_volatility":
            ef.min_volatility()
        elif objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=cfg.risk_free_rate)
        elif objective == "max_quadratic_utility":
            ef.max_quadratic_utility(risk_aversion=max(cfg.risk_aversion, 1e-6))
        elif objective == "efficient_return":
            lo, hi = float(mu.min()), float(mu.max())
            target = _clamp(cfg.target_return, lo + 1e-4, hi - 1e-4) if hi > lo else lo
            ef.efficient_return(target_return=target, market_neutral=False)
        else:
            return None
        return dict(ef.clean_weights())
    except Exception as exc:
        logger.debug("optimize_single(%s) failed: %s", objective, exc)
        return None


def _scipy_optimize(mu: pd.Series, S: pd.DataFrame, objective: str, cfg: OptimizerConfig) -> dict | None:
    """Lightweight scipy fallback when PyPortfolioOpt is unavailable."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return _normalize({s: 1.0 for s in mu.index})
    symbols = list(mu.index)
    n = len(symbols)
    cov = S.values
    mu_v = mu.values
    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(mu_v))):
        return None
    max_w = _feasible_max_weight(n, cfg.max_weight)
    bounds = [(0.0, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        ret = w @ mu_v
        vol = math.sqrt(max(w @ cov @ w, 1e-12))
        return -(ret - cfg.risk_free_rate) / vol

    def variance(w):
        return w @ cov @ w + (cfg.l2_gamma * (w @ w) if cfg.use_l2 else 0.0)

    def neg_util(w):
        return -(w @ mu_v - 0.5 * cfg.risk_aversion * (w @ cov @ w))

    fn = {"min_volatility": variance, "max_sharpe": neg_sharpe,
          "max_quadratic_utility": neg_util, "efficient_return": variance}.get(objective, variance)
    try:
        res = minimize(fn, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
        if not res.success:
            return _normalize({s: 1.0 for s in symbols})
        return _normalize({s: float(w) for s, w in zip(symbols, res.x)})
    except Exception:
        return _normalize({s: 1.0 for s in symbols})


def blended_optimization(mu: pd.Series, S: pd.DataFrame, cfg: OptimizerConfig,
                         current_weights: dict | None = None, use_l2: bool | None = None,
                         use_turnover: bool = False) -> dict:
    """
    Run all four objectives and blend 0.20/0.30/0.25/0.25, renormalizing the blend over
    whichever objectives succeed. Falls back to equal weight if every objective fails.
    Returns {weights, per_objective, used_objectives, fallback}.
    """
    symbols = list(mu.index)
    per_obj, used = {}, []
    for obj in OBJECTIVES:
        w = optimize_single(mu, S, obj, cfg, current_weights, use_l2=use_l2,
                            use_turnover=use_turnover)
        if w:
            per_obj[obj] = w
            used.append(obj)
    if not used:
        return {"weights": _equal_weights(symbols), "per_objective": {}, "used_objectives": [],
                "fallback": True}
    total_bw = sum(BLEND_WEIGHTS[o] for o in used)
    blended = {s: 0.0 for s in symbols}
    for obj in used:
        bw = BLEND_WEIGHTS[obj] / total_bw
        for s in symbols:
            blended[s] += bw * float(per_obj[obj].get(s, 0.0))
    return {"weights": _normalize(blended), "per_objective": per_obj,
            "used_objectives": used, "fallback": False}


# ── diversification analytics ──────────────────────────────────────────────────────
def diversification_metrics(weights: dict, prev_weights: dict | None = None,
                            sectors: dict | None = None) -> dict:
    w = {k: float(v) for k, v in (weights or {}).items() if float(v) > 1e-6}
    vals = np.array(list(w.values())) if w else np.array([0.0])
    hhi = float(np.sum(vals ** 2)) if w else 0.0
    enh = (1.0 / hhi) if hhi > 0 else 0.0
    turnover = 0.0
    if prev_weights is not None:
        keys = set(w) | set(prev_weights)
        turnover = 0.5 * sum(abs(w.get(k, 0.0) - float(prev_weights.get(k, 0.0))) for k in keys)
    sector_conc = {}
    if sectors:
        agg = {}
        for k, v in w.items():
            agg[sectors.get(k, "unknown")] = agg.get(sectors.get(k, "unknown"), 0.0) + v
        sector_conc = {s: round(v, 4) for s, v in agg.items()}
    return {
        "holdings_count": len(w),
        "largest_position": round(float(max(vals)), 4) if w else 0.0,
        "hhi_concentration": round(hhi, 4),
        "effective_number_of_holdings": round(enh, 2),
        "turnover": round(turnover, 4),
        "sector_concentration": sector_conc,
    }


# ── portfolio metrics + walk-forward ────────────────────────────────────────────────
def _portfolio_metrics(returns: np.ndarray, cfg: OptimizerConfig) -> dict:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
                "max_drawdown": 0.0, "volatility": 0.0, "total_return": 0.0}
    ppy = cfg.periods_per_year
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1]) - 1.0
    cagr = (equity[-1] ** (ppy / len(r)) - 1.0) if equity[-1] > 0 else -1.0
    vol = float(np.std(r)) * math.sqrt(ppy)
    mean_ann = float(np.mean(r)) * ppy
    sharpe = (mean_ann - cfg.risk_free_rate) / vol if vol > 1e-12 else 0.0
    downside = r[r < 0]
    dvol = float(np.std(downside)) * math.sqrt(ppy) if len(downside) else 0.0
    sortino = (mean_ann - cfg.risk_free_rate) / dvol if dvol > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) else 0.0
    calmar = (cagr / abs(max_dd)) if max_dd < -1e-9 else 0.0
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
            "calmar": round(calmar, 3), "max_drawdown": round(max_dd, 4),
            "volatility": round(vol, 4), "total_return": round(total, 4)}


def optimizer_score(metrics: dict, turnover: float = 0.0, hhi: float = 0.0) -> float:
    """Evidence ranking score (higher = better)."""
    return round(
        0.30 * metrics.get("sharpe", 0.0)
        + 0.20 * metrics.get("sortino", 0.0)
        + 0.15 * metrics.get("calmar", 0.0)
        + 0.15 * metrics.get("cagr", 0.0)
        - 0.10 * abs(metrics.get("max_drawdown", 0.0))
        - 0.05 * turnover
        - 0.05 * hhi, 4)


def _backtest_bl_posterior(in_sample: pd.DataFrame, cfg: OptimizerConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward BL proxy: shrink in-sample mean returns toward the equilibrium prior."""
    symbols = list(in_sample.columns)
    S = in_sample.cov() * cfg.periods_per_year
    mu_hist = in_sample.mean() * cfg.periods_per_year
    blc = bbl.BLConfig(risk_aversion=cfg.risk_aversion, risk_free_rate=cfg.risk_free_rate)
    pi = bbl.market_implied_prior(symbols, S, None, blc)
    P = np.eye(len(symbols))
    Q = mu_hist.values
    conf = np.full(len(symbols), 0.5)
    omega = bbl.build_omega(P, conf, S, blc.tau)
    post_ret, post_cov = bbl._closed_form_bl(pi, S, P, Q, omega, blc.tau)
    return post_ret, post_cov


def _weights_for_config(name: str, in_sample: pd.DataFrame, cfg: OptimizerConfig,
                        conviction_weights: dict | None, prev_weights: dict | None) -> dict:
    symbols = list(in_sample.columns)
    if name == "equal_weight":
        return _equal_weights(symbols)
    if name == "conviction_allocation":
        return _normalize(conviction_weights) if conviction_weights else _equal_weights(symbols)
    S = in_sample.cov() * cfg.periods_per_year
    mu_hist = in_sample.mean() * cfg.periods_per_year
    if name == "ef_historical":
        return blended_optimization(mu_hist, S, cfg, prev_weights, use_l2=False)["weights"]
    post_ret, post_cov = _backtest_bl_posterior(in_sample, cfg)
    if name == "bl_ef":
        return blended_optimization(post_ret, post_cov, cfg, prev_weights, use_l2=False)["weights"]
    if name == "bl_ef_l2":
        return blended_optimization(post_ret, post_cov, cfg, prev_weights, use_l2=True)["weights"]
    if name == "bl_ef_l2_turnover":
        return blended_optimization(post_ret, post_cov, cfg, prev_weights, use_l2=True, use_turnover=True)["weights"]
    return _equal_weights(symbols)


def walk_forward_compare(returns_df: pd.DataFrame, cfg: OptimizerConfig | None = None,
                         conviction_weights: dict | None = None, sectors: dict | None = None,
                         start_equity: float = 100_000.0, date_tag: str | None = None,
                         write: bool = True) -> dict:
    """
    Out-of-sample walk-forward comparison of the six optimizer configurations over a
    returns matrix (rows = periods, cols = assets), net of transaction costs + slippage.
    Ranks by optimizer_score, selects the active optimizer, and writes the comparison +
    active-optimizer + optimizer-performance reports.
    """
    cfg = cfg or OptimizerConfig()
    results: dict[str, dict] = {}
    cost = (cfg.wf_cost_bps + cfg.wf_slippage_bps) / 10000.0
    n_rows = len(returns_df)
    usable = n_rows >= (cfg.wf_train + cfg.wf_test)

    for name in CONFIGS:
        oos_returns: list[float] = []
        turnovers: list[float] = []
        hhis: list[float] = []
        prev_w: dict | None = None
        if usable:
            start = cfg.wf_train
            while start + cfg.wf_test <= n_rows:
                in_sample = returns_df.iloc[start - cfg.wf_train:start]
                test = returns_df.iloc[start:start + cfg.wf_test]
                try:
                    w = _weights_for_config(name, in_sample, cfg, conviction_weights, prev_w)
                except Exception:
                    w = _equal_weights(list(returns_df.columns))
                wv = np.array([w.get(c, 0.0) for c in returns_df.columns])
                turn = 0.5 * np.sum(np.abs(wv - np.array([(prev_w or {}).get(c, 0.0)
                                                          for c in returns_df.columns]))) if prev_w else 0.0
                turnovers.append(float(turn))
                hhis.append(float(np.sum(wv ** 2)))
                period_rets = test.values @ wv
                if len(period_rets):
                    period_rets = period_rets.astype(float).copy()
                    period_rets[0] -= turn * cost     # rebalance cost on first test period
                oos_returns.extend(period_rets.tolist())
                prev_w = w
                start += cfg.wf_test
        metrics = _portfolio_metrics(np.array(oos_returns), cfg)
        avg_turn = float(np.mean(turnovers)) if turnovers else 0.0
        avg_hhi = float(np.mean(hhis)) if hhis else 0.0
        score = optimizer_score(metrics, avg_turn, avg_hhi)
        results[name] = {**metrics, "turnover": round(avg_turn, 4),
                         "hhi_concentration": round(avg_hhi, 4),
                         "effective_number_of_holdings": round(1.0 / avg_hhi, 2) if avg_hhi > 0 else 0.0,
                         "optimizer_score": score, "oos_periods": len(oos_returns)}

    ranked = sorted(results.items(), key=lambda kv: kv[1]["optimizer_score"], reverse=True)
    active = ranked[0][0] if (ranked and usable) else cfg.default_active
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "date_tag": date_tag or DATA_TAG,
        "start_equity": start_equity, "usable": usable, "configs": results,
        "ranking": [{"config": k, "optimizer_score": v["optimizer_score"]} for k, v in ranked],
        "active_buffett_optimizer": active,
        "score_formula": "0.30*Sharpe+0.20*Sortino+0.15*Calmar+0.15*CAGR-0.10*|MaxDD|-0.05*Turnover-0.05*HHI",
    }
    active_doc = {"generated_at": comparison["generated_at"], "date_tag": comparison["date_tag"],
                  "active_buffett_optimizer": active,
                  "selected_by": "walk_forward_score" if usable else "default_insufficient_data",
                  "score": results.get(active, {}).get("optimizer_score", 0.0)}
    feedback = [{
        "optimizer_name": name, "realized_return": m.get("total_return", 0.0),
        "realized_sharpe": m.get("sharpe", 0.0), "realized_sortino": m.get("sortino", 0.0),
        "realized_drawdown": m.get("max_drawdown", 0.0), "realized_turnover": m.get("turnover", 0.0),
        "realized_concentration": m.get("hhi_concentration", 0.0),
        "optimizer_score": m.get("optimizer_score", 0.0), "is_active": (name == active),
    } for name, m in results.items()]
    if write:
        _write(OPTIMIZER_COMPARISON, comparison, date_tag)
        _write(ACTIVE_OPTIMIZER, active_doc, date_tag)
        write_optimizer_performance(feedback, date_tag)
    comparison["_active_doc"] = active_doc
    comparison["_feedback"] = feedback
    return comparison


# ── orchestrator (live target weights) ──────────────────────────────────────────────
def run_optimizer(signals: list[dict], prices: pd.DataFrame | None = None,
                  market_caps: dict | None = None, vol_by_symbol: dict | None = None,
                  current_weights: dict | None = None, conviction_weights: dict | None = None,
                  returns_df: pd.DataFrame | None = None, sectors: dict | None = None,
                  cfg: OptimizerConfig | None = None, bl_cfg: bbl.BLConfig | None = None,
                  date_tag: str | None = None, write: bool = True, market_state: dict | None = None) -> dict:
    """
    Live optimizer pass: Black-Litterman posterior → blended efficient-frontier target
    weights → diversification analytics. Optionally runs the walk-forward comparison when
    a returns matrix is supplied. Fully fail-safe — returns target weights or an empty
    dict so the caller falls back to conviction allocation.
    """
    cfg = cfg or OptimizerConfig()
    out = {"date_tag": date_tag or DATA_TAG, "generated_at": datetime.now(timezone.utc).isoformat(),
           "target_weights": {}, "fallback": False, "active_optimizer": cfg.default_active}
    try:
        bl = bbl.run_black_litterman(signals, prices=prices, market_caps=market_caps,
                                     vol_by_symbol=vol_by_symbol, cfg=bl_cfg, date_tag=date_tag,
                                     write=write, market_state=market_state)
        mu = bl.get("_posterior_returns_series")
        S = bl.get("_posterior_covariance_df")
        if mu is None or S is None or len(mu) < 2:
            out["fallback"] = True
            out["reason"] = "insufficient_bl_output"
            return out
        blended = blended_optimization(mu, S, cfg, current_weights,
                                       use_l2=cfg.use_l2, use_turnover=True)
        weights = blended["weights"]
        out["target_weights"] = {k: round(float(v), 6) for k, v in weights.items()}
        out["bl_fallback"] = bl.get("fallback", False)
        out["used_objectives"] = blended["used_objectives"]
        out["optimizer_fallback"] = blended["fallback"]
        div = diversification_metrics(weights, current_weights, sectors)
        out["diversification"] = div
        if write:
            _write(DIVERSIFICATION_REPORT, {"generated_at": out["generated_at"],
                                            "date_tag": out["date_tag"], **div,
                                            "weights": out["target_weights"]}, date_tag)
        if returns_df is not None and not returns_df.empty:
            cmp = walk_forward_compare(returns_df, cfg, conviction_weights, sectors,
                                       date_tag=date_tag, write=write)
            out["active_optimizer"] = cmp["active_buffett_optimizer"]
            out["walk_forward_ran"] = True
    except Exception as exc:
        logger.error("run_optimizer failed (Buffett continues on conviction): %s", exc)
        out["fallback"] = True
        out["reason"] = str(exc)[:200]
    return out


def write_optimizer_performance(records: list[dict], date_tag: str | None = None) -> None:
    """Research-Lab feedback: realized optimizer performance for evidence-driven selection."""
    try:
        _write(OPTIMIZER_PERFORMANCE, {
            "generated_at": datetime.now(timezone.utc).isoformat(), "date_tag": date_tag or DATA_TAG,
            "count": len(records), "records": records,
            "tracked": ["optimizer_name", "target_weights", "realized_return", "realized_sharpe",
                        "realized_drawdown", "realized_turnover", "realized_concentration"],
        }, date_tag)
    except Exception as exc:
        logger.debug("write_optimizer_performance failed: %s", exc)


def _write(name: str, payload: dict, date_tag: str | None = None) -> None:
    try:
        clean = {k: v for k, v in payload.items() if not k.startswith("_")}
        (_research_dir(date_tag) / name).write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("write %s failed: %s", name, exc)


def run_optimizer_safe(*args, **kwargs):
    try:
        return run_optimizer(*args, **kwargs)
    except Exception as exc:
        logger.error("run_optimizer_safe swallowed: %s", exc)
        return {"target_weights": {}, "fallback": True, "reason": str(exc)[:200]}
