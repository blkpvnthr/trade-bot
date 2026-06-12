"""Evaluate challengers against PREREGISTERED promotion criteria. A challenger is PROMOTEd only if
it clears every bar; clearly-worse challengers are REJECTed; the rest CONTINUE under observation.

Significance: bootstrap the challenger−champion Sharpe gap by resampling the shared candidate pool
and re-running BOTH filter+weight pipelines on each resample (shared resample = fair comparison).
Per-regime robustness: every year with enough trades must be non-catastrophic (catches 2022-type
fragility). Burn-in: not enough observed days yet → CONTINUE, never promote prematurely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Dict, List, Optional

import numpy as np

from .events import EventLog
from .fills import FillModel
from .portfolios import PortfolioConfig, daily_series, port_stats, run_portfolio


@dataclass
class PromotionCriteria:
    min_trades: int = 30
    min_sharpe_improvement: float = 0.0      # challenger Sharpe must exceed champion by at least this
    max_p_value: float = 0.10
    min_year_trades: int = 8                 # years with >= this many trades must be non-catastrophic
    max_year_loss: float = -0.01             # per-year mean net return floor
    burn_in_days: int = 60                   # need at least this many observed days
    max_drawdown: float = -0.40
    train_end: str = ""                      # if set, also require improvement on the OOS holdout (date >= train_end)


def _as_date(d):
    return d if hasattr(d, "year") else _date.fromisoformat(str(d))


def _per_year_ok(res, crit: PromotionCriteria) -> bool:
    by_year: Dict[int, List[float]] = {}
    for t, nv in zip(res.taken, res.net):
        y = t["date"].year if hasattr(t["date"], "year") else int(str(t["date"])[:4])
        by_year.setdefault(y, []).append(nv)
    for y, vals in by_year.items():
        if len(vals) >= crit.min_year_trades and float(np.mean(vals)) < crit.max_year_loss:
            return False
    return True


def _sharpe(cfg: PortfolioConfig, trades, fill: FillModel, return_key: str, metric: str) -> float:
    w = cfg.weight_fn(trades) if trades else np.array([])
    rt = fill.round_trip_cost()
    if metric == "capacity":
        perf, _ = capacity_stats(trades, w, cfg.max_conc, cfg.per_name, cfg.cap,
                                 cfg.deployable, return_key, rt)
        return perf["sharpe"]
    _, ser = daily_series(trades, w, return_key, rt)
    return float(ser.mean() / ser.std()) if len(ser) > 1 and ser.std() > 0 else 0.0


def _bootstrap_sharpe_gap_p(champ_cfg: PortfolioConfig, chal_cfg: PortfolioConfig,
                            pool: List[Dict[str, Any]], fill: FillModel,
                            return_key: str, metric: str = "capacity",
                            B: int = 250, seed: int = 7) -> float:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pool))
    neg = 0
    for _ in range(B):
        s = rng.choice(idx, size=len(pool), replace=True)
        rs = [pool[k] for k in s]
        ct = [r for r in rs if champ_cfg.filter_fn(r)]
        ht = [r for r in rs if chal_cfg.filter_fn(r)]
        if len(ct) < 5 or len(ht) < 5:
            neg += 1
            continue
        if (_sharpe(chal_cfg, ht, fill, return_key, metric)
                - _sharpe(champ_cfg, ct, fill, return_key, metric)) <= 0:
            neg += 1
    return neg / B


def _pick(res, metric: str) -> Dict[str, Any]:
    return res.cap_stats if metric == "capacity" else res.stats


def evaluate(champ_cfg: PortfolioConfig, chal_cfg: PortfolioConfig,
             pool: List[Dict[str, Any]], fill: FillModel, crit: PromotionCriteria,
             experiment: str, return_key: str = "gross", metric: str = "capacity",
             allow_promote: bool = True,
             log: Optional[EventLog] = None, log_champion: bool = False) -> Dict[str, Any]:
    champ = run_portfolio(champ_cfg, pool, fill, experiment, log=(log if log_champion else None),
                          return_key=return_key, log_skips=False)
    chal = run_portfolio(chal_cfg, pool, fill, experiment, log=log,
                         return_key=return_key, log_skips=True)
    cs, hs = _pick(champ, metric), _pick(chal, metric)

    d_sharpe = hs["sharpe"] - cs["sharpe"]
    checks = {
        "min_trades": hs["n"] >= crit.min_trades,
        "burn_in": hs["ndays"] >= crit.burn_in_days,
        "sharpe_improvement": d_sharpe >= crit.min_sharpe_improvement,
        "drawdown": hs["mdd"] >= crit.max_drawdown,
        "per_year_nonneg": _per_year_ok(chal, crit),
    }
    oos = {}
    if crit.train_end:
        te = _date.fromisoformat(crit.train_end)
        test_pool = [r for r in pool if _as_date(r["date"]) >= te]
        c_oos = _pick(run_portfolio(champ_cfg, test_pool, fill, experiment, return_key=return_key, log_skips=False), metric)
        h_oos = _pick(run_portfolio(chal_cfg, test_pool, fill, experiment, return_key=return_key, log_skips=False), metric)
        d_oos = h_oos["sharpe"] - c_oos["sharpe"]
        checks["oos_sharpe_improvement"] = d_oos >= crit.min_sharpe_improvement
        oos = {"champ_oos_sharpe": c_oos["sharpe"], "chal_oos_sharpe": h_oos["sharpe"],
               "d_oos_sharpe": d_oos, "chal_oos_n": h_oos["n"]}

    p_val = float("nan")
    if checks["sharpe_improvement"] and checks["min_trades"]:
        p_val = _bootstrap_sharpe_gap_p(champ_cfg, chal_cfg, pool, fill, return_key, metric)
        checks["significance"] = (p_val <= crit.max_p_value)
    else:
        checks["significance"] = False

    if all(checks.values()):
        verdict = "PROMOTE"
    elif (not checks["sharpe_improvement"]) or (not checks["drawdown"]) \
            or (not checks["per_year_nonneg"]) or (not checks.get("oos_sharpe_improvement", True)):
        verdict = "REJECT"
    else:
        verdict = "CONTINUE"

    # directive: no promotion while the champion baseline is unreconciled → cap at CONTINUE
    if verdict == "PROMOTE" and not allow_promote:
        verdict = "CONTINUE"

    return {"champion": champ, "challenger": chal, "d_sharpe": d_sharpe, "p_value": p_val,
            "checks": checks, "verdict": verdict, "oos": oos, "metric": metric,
            "promotion_locked": (not allow_promote)}
