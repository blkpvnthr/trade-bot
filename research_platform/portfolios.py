"""Champion/challenger shadow portfolios. A portfolio = filter_fn + weight_fn over a SHARED
candidate pool. Modeling universe/gate/sizing changes all as filter+weight means every kind of
change is compared the same way. Returns are net of the shared FillModel cost.

A candidate is a dict with at least: {sym, date, gross, hold, ...features}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .events import DecisionEvent, EventLog
from .fills import FillModel


@dataclass
class PortfolioConfig:
    portfolio_id: str
    role: str                                   # 'champion' | 'challenger'
    filter_fn: Callable[[Dict[str, Any]], bool]
    weight_fn: Callable[[List[Dict[str, Any]]], np.ndarray]
    description: str = ""
    # capacity-sim params (cash-aware, fixed-fractional) — the CORRECT metric for selection/gate
    max_conc: int = 8
    per_name: float = 0.06
    cap: float = 0.10                           # single-name cap
    deployable: float = 0.55                    # 1 - cash buffer


@dataclass
class PortfolioResult:
    portfolio_id: str
    role: str
    taken: List[Dict[str, Any]]
    weights: np.ndarray
    net: List[float]                            # per-trade net returns
    series_dates: List[Any]
    series: np.ndarray                          # weighted-average daily return series (sizing metric)
    stats: Dict[str, float]                     # weighted-average stats (always-invested; sizing only)
    cap_stats: Dict[str, float] = None          # cash-aware capacity sim (selection/gate metric)
    geometry: Dict[str, float] = None           # portfolio geometry (concurrency, cash%, HHI, ...)
    description: str = ""


def daily_series(trades: List[Dict[str, Any]], weights: np.ndarray, return_key: str = "gross",
                 rt_cost: float = 0.0):
    """Weighted-average daily return series (scale-invariant in weights → same-gross)."""
    num: Dict[Any, float] = {}
    den: Dict[Any, float] = {}
    for r, w in zip(trades, weights):
        hold = int(r.get("hold", 1)) or 1
        net = r[return_key] - rt_cost
        gday = (1 + net) ** (1 / hold) - 1
        for d in pd.bdate_range(r["date"], periods=hold):
            dd = d.date()
            num[dd] = num.get(dd, 0.0) + w * gday
            den[dd] = den.get(dd, 0.0) + w
    dates = sorted(num)
    ser = np.array([num[d] / den[d] for d in dates]) if dates else np.array([])
    return dates, ser


def port_stats(series: np.ndarray, net: List[float]) -> Dict[str, float]:
    n = len(net)
    if len(series) < 5:
        return {"n": n, "sharpe": 0.0, "cagr": 0.0, "mdd": 0.0, "ndays": len(series),
                "mean_net": float(np.mean(net)) if net else 0.0}
    sh = float(series.mean() / series.std() * np.sqrt(252)) if series.std() > 0 else 0.0
    eq = np.cumprod(1 + series)
    cagr = float(eq[-1] ** (252.0 / len(series)) - 1)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return {"n": n, "sharpe": sh, "cagr": cagr, "mdd": mdd, "ndays": len(series),
            "mean_net": float(np.mean(net))}


def capacity_stats(trades: List[Dict[str, Any]], weights: np.ndarray, max_conc: int = 8,
                   per_name: float = 0.06, cap: float = 0.10, deployable: float = 0.55,
                   return_key: str = "gross", rt_cost: float = 0.0):
    """Cash-aware, capacity-constrained fixed-fractional sim → CORRECT metric when concurrency
    differs across portfolios. A thin book sits in CASH (low vol), not 100% in one name.
    Returns (perf_stats, geometry). Geometry exposes whether bad metrics are thin-book artifacts.
    """
    zero_perf = {"n": len(trades), "sharpe": 0.0, "cagr": 0.0, "mdd": 0.0, "ndays": 0}
    zero_geo = {"n_admitted": 0, "n_rejected": len(trades), "n_names": 0, "avg_concurrent": 0.0,
                "cash_pct": 1.0, "utilization": 0.0, "idle_frac": 1.0, "hhi": 0.0, "max_weight": 0.0}
    if not trades:
        return zero_perf, zero_geo
    w = np.asarray(weights, float)
    wn = w / (w.mean() if w.mean() else 1.0)
    order = sorted(range(len(trades)), key=lambda k: trades[k]["date"])
    active = []          # (exit_date, alloc) for admission control
    admitted = []        # (idx, alloc)
    for k in order:
        t = trades[k]
        d = t["date"]
        hold = int(t.get("hold", 1)) or 1
        active = [(xd, a) for (xd, a) in active if xd > d]
        alloc = float(min(per_name * wn[k], cap))
        if len(active) >= max_conc or sum(a for _, a in active) + alloc > deployable + 1e-9:
            continue                                   # capacity / budget rejected
        exit_d = (pd.Timestamp(d) + pd.tseries.offsets.BDay(hold)).date()
        active.append((exit_d, alloc))
        admitted.append((k, alloc))
    if not admitted:
        return zero_perf, zero_geo
    day_ret, day_alloc, day_open, day_hhinum = {}, {}, {}, {}
    for k, alloc in admitted:
        t = trades[k]
        hold = int(t.get("hold", 1)) or 1
        dayret = (1 + (t[return_key] - rt_cost)) ** (1 / hold) - 1
        for bd in pd.bdate_range(t["date"], periods=hold):
            dd = bd.date()
            day_ret[dd] = day_ret.get(dd, 0.0) + alloc * dayret
            day_alloc[dd] = day_alloc.get(dd, 0.0) + alloc
            day_open[dd] = day_open.get(dd, 0) + 1
            day_hhinum[dd] = day_hhinum.get(dd, 0.0) + alloc * alloc
    start = min(trades[k]["date"] for k, _ in admitted)
    end = max(pd.Timestamp(trades[k]["date"]).date() for k, _ in admitted)
    full = [b.date() for b in pd.bdate_range(start, pd.Timestamp(end) + pd.tseries.offsets.BDay(7))]
    fr = np.array([day_ret.get(d, 0.0) for d in full])
    eq = np.cumprod(1 + fr)
    sh = float(fr.mean() / fr.std() * np.sqrt(252)) if fr.std() > 0 else 0.0
    cagr = float(eq[-1] ** (252.0 / len(fr)) - 1) if len(fr) else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    opens = [day_open.get(d, 0) for d in full]
    allocs = [day_alloc.get(d, 0.0) for d in full]
    hhis = [day_hhinum[d] / (day_alloc[d] ** 2) for d in day_alloc if day_alloc[d] > 0]
    perf = {"n": len(trades), "sharpe": sh, "cagr": cagr, "mdd": mdd, "ndays": len(full)}
    geo = {"n_admitted": len(admitted), "n_rejected": len(trades) - len(admitted),
           "n_names": len({trades[k]["sym"] for k, _ in admitted}),
           "avg_concurrent": float(np.mean(opens)), "cash_pct": 1.0 - float(np.mean(allocs)),
           "utilization": float(np.mean(allocs)) / deployable, "idle_frac": sum(1 for c in opens if c == 0) / len(full),
           "hhi": float(np.mean(hhis)) if hhis else 0.0, "max_weight": float(max(a for _, a in admitted))}
    return perf, geo


def run_portfolio(cfg: PortfolioConfig, candidates: List[Dict[str, Any]], fill: FillModel,
                  experiment: str, log: EventLog = None, return_key: str = "gross",
                  log_skips: bool = True) -> PortfolioResult:
    taken = [r for r in candidates if cfg.filter_fn(r)]
    skipped = [r for r in candidates if not cfg.filter_fn(r)]
    weights = cfg.weight_fn(taken) if taken else np.array([])
    rt = fill.round_trip_cost()
    net = [r[return_key] - rt for r in taken]
    dates, ser = daily_series(taken, weights, return_key, rt)
    stats = port_stats(ser, net)
    cap_perf, geo = capacity_stats(taken, weights, cfg.max_conc, cfg.per_name, cfg.cap,
                                   cfg.deployable, return_key, rt)

    if log is not None:
        tag = "realized" if cfg.role == "champion" else "modeled"
        evs = []
        for r, w, nv in zip(taken, weights, net):
            fr = fill.fill("buy", float(r.get("ref_price", 100.0)))
            evs.append(DecisionEvent(
                experiment=experiment, portfolio=cfg.portfolio_id, ts=str(r["date"]),
                symbol=str(r.get("sym", "?")), action="enter",
                features={k: r[k] for k in r if k not in ("gross", "ret")},
                decision={"weight": float(w), "reason": cfg.description},
                fill={"tag": tag, **fr}, outcome={"net_return": float(nv)}))
        if log_skips:
            for r in skipped:
                evs.append(DecisionEvent(
                    experiment=experiment, portfolio=cfg.portfolio_id, ts=str(r["date"]),
                    symbol=str(r.get("sym", "?")), action="skip",
                    features={k: r[k] for k in r if k not in ("gross", "ret")},
                    decision={"weight": 0.0, "reason": "filtered"},
                    fill={"tag": "n/a"}, outcome={}))
        log.append_many(evs)

    return PortfolioResult(cfg.portfolio_id, cfg.role, taken, weights, net, dates, ser, stats,
                           cap_stats=cap_perf, geometry=geo, description=cfg.description)


# ---- reusable weight functions ----
def flat_weights(trades: List[Dict[str, Any]]) -> np.ndarray:
    return np.ones(len(trades))


def zscore_weights(key: str, mu: float, sd: float, lam: float = 0.5) -> Callable:
    def f(trades):
        return np.array([np.exp(lam * np.clip((t[key] - mu) / (sd or 1.0), -2, 2)) for t in trades])
    return f


def bucket_weights(score_keys: List[tuple], lam_table=None) -> Callable:
    """score_keys = [(key, mu, sd), ...]; composite z → quintile → A+8/A6/B4/C2 weights."""
    table = lam_table or {4: 8.0, 3: 6.0, 2: 4.0, 1: 2.0, 0: 2.0}

    def f(trades):
        if not trades:
            return np.array([])
        z = np.zeros(len(trades))
        for key, mu, sd in score_keys:
            z += np.clip((np.array([t[key] for t in trades], float) - mu) / (sd or 1.0), -2, 2)
        try:
            q = pd.qcut(pd.Series(z).rank(method="first"), 5, labels=False).values
            return np.array([table[int(x)] for x in q])
        except Exception:
            return np.ones(len(trades))
    return f
