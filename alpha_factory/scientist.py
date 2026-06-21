"""Scientist Agent - proves whether an alpha candidate is real.

Wraps rl_trading.event_study (forward panels + block-bootstrap CI + rank-IC + random-shift placebo)
and adds: time-fold OOS IC stability ("walk-forward"), a net-of-cost long/short sim on 1-DAY
forward returns (the short-vol method-fix: net P&L must use fut_ret_1, not the holding-horizon
return), and Benjamini-Hochberg FDR across the batch. Verdict is preregistered, not eyeballed.

A candidate is a cross-sectional signal panel: DataFrame [dates x symbols] of strengths. Higher
strength = more bullish. Entry is the first bar STRICTLY AFTER the signal date (leak-free).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from rl_trading.event_study import _ords, build_fwd_panels, event_forward, rank_ic, stats
from .schema import (
    ACCEPTED, NEEDS_MORE_DATA, REJECTED, AlphaRecord, prediction_hash, prediction_vector,
)


@dataclass
class Criteria:
    """Preregistered acceptance bar. Change a field -> a different (logged) experiment."""
    min_dates: int = 120              # need this many signal dates or it's NEEDS_MORE_DATA
    min_obs: int = 400                # need this many (date,symbol) observations
    min_oos_ic: float = 0.01          # mean daily rank-IC across time folds
    max_placebo_ic: float = 0.03      # random-shift control must be ~0 (finite-sample tolerance)
    min_net_sharpe: float = 0.5       # after cost, 1-day LS portfolio, annualized
    max_corr: float = 0.75            # crowding cap vs already-accepted alphas
    fdr_alpha: float = 0.10           # Benjamini-Hochberg level across the batch
    cost_bps: float = 5.0             # round-trip cost per unit turnover
    horizons: Tuple[int, ...] = (1, 5, 10)
    primary_h: int = 1                # daily alpha: signal_t predicts t->t+1; matches the cost sim
    folds: int = 4
    # promotion gate (allocator export) - stricter than acceptance, evaluated at REALISTIC cost
    promote_min_sharpe: float = 1.0       # net Sharpe at promote_cost_bps
    promote_cost_bps: float = 10.0        # the cost an allocator decision must clear
    promote_max_turnover: float = 1.0     # >1.0/day = full book churns daily = pathological
    promote_min_year_frac: float = 0.60   # fraction of calendar years with positive net return
    promote_min_wf_frac: float = 0.60     # fraction of walk-forward folds with positive IC


# --- core sims ------------------------------------------------------------

def _daily_ic(signals: pd.DataFrame, fwd1: pd.DataFrame) -> pd.Series:
    """Per-date Spearman(signal_t, return_{t->t+1}). Leak-free: signal known at close t."""
    out = {}
    for d in signals.index:
        s, f = signals.loc[d], fwd1.loc[d]
        m = s.notna() & f.notna()
        if m.sum() >= 5 and s[m].std() > 0 and f[m].std() > 0:
            out[d] = sps.spearmanr(s[m], f[m]).statistic
    return pd.Series(out).dropna()


def _ls_series(signals: pd.DataFrame, fwd1: pd.DataFrame):
    """Cross-sectional long/short, gross=1/day, held 1 day. Returns (pnl, turnover) series.

    Split out so one sim can be priced at ANY cost (the cost sweep is just arithmetic on these).
    """
    rank = signals.rank(axis=1, pct=True) - 0.5
    gross = rank.abs().sum(axis=1).replace(0, np.nan)
    w = rank.div(gross, axis=0).fillna(0.0)
    pnl = (w * fwd1).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    return pnl, turnover


def _cost_stats(pnl: pd.Series, turnover: pd.Series, cost_bps: float) -> Dict[str, float]:
    net = pnl - turnover * cost_bps * 1e-4
    valid = pnl.dropna()
    if len(valid) < 20 or net.std() == 0:
        return {"gross_sharpe": 0.0, "net_sharpe": 0.0, "turnover": 0.0,
                "breakeven_bps": 0.0, "net_mean": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0}
    eq = net.cumsum()
    dd = float((eq - eq.cummax()).min())
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    return {
        "gross_sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)),
        "net_sharpe": float(net.mean() / net.std() * np.sqrt(252)),
        "turnover": float(turnover.mean()),
        "breakeven_bps": float(pnl.mean() / turnover.mean() * 1e4) if turnover.mean() > 0 else 0.0,
        "net_mean": float(net.mean()),
        "max_drawdown": dd,
        "profit_factor": float(pos / neg) if neg > 0 else float("inf"),
    }


def cost_curve(signals: pd.DataFrame, fwd1: pd.DataFrame, costs: Sequence[float]) -> Dict[float, float]:
    """{cost_bps: net_sharpe} from a single LS sim - the cost-sensitivity row for one alpha."""
    pnl, to = _ls_series(signals, fwd1)
    return {float(c): _cost_stats(pnl, to, c)["net_sharpe"] for c in costs}


def _promotion(pnl: pd.Series, turnover: pd.Series, fold_ic: List[float], crit: Criteria) -> Dict:
    """Allocator-export gate: net Sharpe >= bar at realistic cost, sane turnover, survives
    walk-forward, positive in >= min_year_frac of calendar years. ALL must hold to promote."""
    st = _cost_stats(pnl, turnover, crit.promote_cost_bps)
    net = pnl - turnover * crit.promote_cost_bps * 1e-4
    yr = net.dropna().groupby(net.dropna().index.year).mean()
    year_pos_frac = float((yr > 0).mean()) if len(yr) else 0.0
    wf_frac = float(np.mean([f > 0 for f in fold_ic])) if fold_ic else 0.0
    checks = {
        "net_sharpe>=1": st["net_sharpe"] >= crit.promote_min_sharpe,
        "turnover_ok": st["turnover"] <= crit.promote_max_turnover,
        "walk_forward": wf_frac >= crit.promote_min_wf_frac,
        "yearly_60pct": year_pos_frac >= crit.promote_min_year_frac,
    }
    return {"promotable": all(checks.values()), "cost_bps": crit.promote_cost_bps,
            "net_sharpe": st["net_sharpe"], "turnover": st["turnover"],
            "wf_frac": wf_frac, "year_pos_frac": year_pos_frac, "checks": checks}


def _score(oos_ic: float, oos_cv: float, cost: Dict[str, float], n_days: int,
           corr: float) -> Tuple[float, Dict[str, float]]:
    """Transparent composite (user weights). Each component clipped to [0,1]."""
    clip = lambda x: float(max(0.0, min(1.0, x)))
    pf = cost["profit_factor"]
    comp = {
        "profit_factor": clip((pf - 1.0) / 2.0) if np.isfinite(pf) else 1.0,
        "expectancy": clip(cost["net_mean"] / 0.001),          # 0.1%/day -> 1.0
        "wf_stability": clip(1.0 - oos_cv),                    # low CV of fold ICs -> 1.0
        "drawdown_safety": clip(1.0 + cost["max_drawdown"] / 0.20),  # dd of -20% -> 0
        "trade_count": clip(n_days / 250.0),
        "independence": clip(1.0 - corr),
    }
    score = (0.25 * comp["profit_factor"] + 0.20 * comp["expectancy"]
             + 0.20 * comp["wf_stability"] + 0.15 * comp["drawdown_safety"]
             + 0.10 * comp["trade_count"] + 0.10 * comp["independence"])
    return float(score), comp


def evaluate(signals: pd.DataFrame, closes: pd.DataFrame, crit: Criteria,
             corr_to_accepted: float = 0.0) -> Dict:
    """Run the full diagnostic. Returns metrics incl a raw p_value (FDR is applied at batch level)."""
    horizons = tuple(sorted(set(crit.horizons) | {1, crit.primary_h}))
    fwd = build_fwd_panels(closes, horizons)
    fwd1 = fwd[1]

    # events = signal firings. A cross-sectional alpha computed at close t is entered AT t (it
    # predicts t->t+1). event_forward enters strictly-after event_ord, so event_ord = ord(t)-1.
    s = signals.stack(future_stack=True).dropna()
    ev = pd.DataFrame({"symbol": [i[1] for i in s.index], "strength": s.to_numpy()})
    ev["event_ord"] = _ords(pd.DatetimeIndex([i[0] for i in s.index])) - 1
    ef = event_forward(ev, closes, fwd, horizons, attribution_cols=["strength"])
    efh = ef[ef["H"] == crit.primary_h] if len(ef) else ef

    n_obs = int(len(efh))
    n_dates = int(signals.notna().any(axis=1).sum())
    pooled_ic = rank_ic(efh, "strength", "exc_univ") if n_obs else float("nan")
    sig = stats(efh, "exc_univ") if n_obs else {"n": 0, "tstat": 0.0,
                                                "block_lo": float("nan"), "block_hi": float("nan")}

    # placebo: random-shift the event dates (strength kept) -> IC must collapse to ~0
    if n_obs:
        rng = np.random.default_rng(0)
        shift = rng.integers(40, 160, len(ev)) * rng.choice([-1, 1], len(ev))
        pl = ev.copy()
        pl["event_ord"] = ev["event_ord"].to_numpy() + shift
        placebo_ef = event_forward(pl, closes, fwd, horizons, attribution_cols=["strength"])
        placebo_ic = rank_ic(placebo_ef[placebo_ef["H"] == crit.primary_h], "strength", "exc_univ") \
            if len(placebo_ef) else float("nan")
    else:
        placebo_ic = float("nan")

    # walk-forward = OOS IC stability across contiguous time folds
    ic_daily = _daily_ic(signals, fwd1)
    folds = np.array_split(ic_daily.to_numpy(), crit.folds) if len(ic_daily) >= crit.folds else []
    fold_ic = [float(np.mean(f)) for f in folds if len(f)]
    oos_ic = float(np.mean(fold_ic)) if fold_ic else float("nan")
    oos_std = float(np.std(fold_ic)) if len(fold_ic) > 1 else 0.0
    oos_cv = abs(oos_std / oos_ic) if oos_ic not in (0.0,) and np.isfinite(oos_ic) else 1.0

    # significance of the daily-IC series. A cross-sectional alpha predicts RELATIVE ranking, not
    # up/down, so significance is on the IC - NOT on mean excess return (which is ~0 for a balanced
    # long/short). t-test feeds FDR; bootstrap CI of the IC mean is the "excludes 0" gate.
    if len(ic_daily) > 2 and ic_daily.std() > 0:
        t = ic_daily.mean() / (ic_daily.std() / np.sqrt(len(ic_daily)))
        p_value = float(2 * sps.norm.sf(abs(t)))
        rng = np.random.default_rng(0)
        x = ic_daily.to_numpy()
        bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(2000)])
        ic_lo, ic_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    else:
        p_value, ic_lo, ic_hi = 1.0, float("nan"), float("nan")

    pnl_s, to_s = _ls_series(signals, fwd1)
    cost = _cost_stats(pnl_s, to_s, crit.cost_bps)
    cost["cost_bps"] = crit.cost_bps
    cost["net_sharpe_15"] = _cost_stats(pnl_s, to_s, 15.0)["net_sharpe"]   # fragility probe
    cost["net_sharpe_20"] = _cost_stats(pnl_s, to_s, 20.0)["net_sharpe"]
    promotion = _promotion(pnl_s, to_s, fold_ic, crit)
    score, comp = _score(oos_ic if np.isfinite(oos_ic) else 0.0, oos_cv, cost, n_dates, corr_to_accepted)

    return {
        "n_dates": n_dates, "n_obs": n_obs,
        "pooled_rank_ic": pooled_ic, "tstat": sig.get("tstat", 0.0),
        "ic_ci_lo": ic_lo, "ic_ci_hi": ic_hi,
        "placebo_ic": placebo_ic,
        "oos_ic_mean": oos_ic, "oos_ic_std": oos_std, "fold_ic": fold_ic,
        "p_value": p_value,
        "cost": cost, "correlation_to_accepted": corr_to_accepted,
        "promotion": promotion,
        "score": score, "score_components": comp,
    }


# --- FDR + verdict --------------------------------------------------------

def benjamini_hochberg(pvals: Sequence[float], alpha: float) -> List[bool]:
    """Return per-p significance under BH at level alpha (controls false discovery rate)."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    thresh = [(i + 1) / m * alpha for i in range(m)]
    passed = [pvals[order[i]] <= thresh[i] for i in range(m)]
    k = max([i for i, ok in enumerate(passed) if ok], default=-1)  # largest passing rank
    sig = [False] * m
    for i in range(k + 1):
        sig[order[i]] = True
    return sig


def _block_excludes_zero(lo, hi) -> bool:
    return np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)


def decide(m: Dict, crit: Criteria, fdr_significant: bool) -> Tuple[str, str]:
    if m["n_dates"] < crit.min_dates or m["n_obs"] < crit.min_obs:
        return NEEDS_MORE_DATA, f"only {m['n_dates']} dates / {m['n_obs']} obs"
    if m["correlation_to_accepted"] > crit.max_corr:
        return REJECTED, f"crowded: corr {m['correlation_to_accepted']:.2f} > {crit.max_corr}"
    fails = []
    if not (m["oos_ic_mean"] >= crit.min_oos_ic):
        fails.append(f"oos_ic {m['oos_ic_mean']:.3f} < {crit.min_oos_ic}")
    if not _block_excludes_zero(m["ic_ci_lo"], m["ic_ci_hi"]):
        fails.append("IC bootstrap CI includes 0")
    if abs(m["placebo_ic"]) > crit.max_placebo_ic:
        fails.append(f"placebo IC {m['placebo_ic']:.3f} too large (leak)")
    if m["cost"]["net_sharpe"] < crit.min_net_sharpe:
        fails.append(f"net Sharpe {m['cost']['net_sharpe']:.2f} < {crit.min_net_sharpe}")
    if not fdr_significant:
        fails.append("fails Benjamini-Hochberg FDR")
    if fails:
        return REJECTED, "; ".join(fails)
    return ACCEPTED, ""


def run_batch(candidates: List, closes: pd.DataFrame, crit: Criteria,
              registry, now: str = "") -> List[AlphaRecord]:
    """Evaluate a batch, apply FDR across it, write verdicts to the registry.

    Each candidate: object/dict with .name/.source/.formula/.symbols/.timeframe/.created_at and a
    .signals DataFrame [dates x symbols].
    """
    metrics, preds, vecs = [], [], []
    for c in candidates:
        sig = _get(c, "signals")
        vec = prediction_vector(sig)
        corr = registry.correlation_to_accepted(vec)
        metrics.append(evaluate(sig, closes, crit, corr))
        vecs.append(vec)
        preds.append(sig)

    sig_flags = benjamini_hochberg([m["p_value"] for m in metrics], crit.fdr_alpha)

    records = []
    for c, m, vec, sigp, fdr_ok in zip(candidates, metrics, vecs, preds, sig_flags):
        decision, reason = decide(m, crit, fdr_ok)
        # fail-safe: an accept at the selected cost that dies by 15-20bps is a cost mirage
        fragile = decision == ACCEPTED and (m["cost"].get("net_sharpe_15", 9.0) < 0.5
                                            or m["cost"].get("net_sharpe_20", 9.0) < 0.5)
        tags = [_get(c, "source", "")] + (["FRAGILE_COST_MIRAGE"] if fragile else [])
        rec = AlphaRecord(
            alpha_id=prediction_hash(vec),
            strategy_name=_get(c, "name"),
            source=_get(c, "source", "unknown"),
            formula=_get(c, "formula", ""),
            symbols=list(_get(c, "symbols", list(sigp.columns))),
            timeframe=_get(c, "timeframe", "1D"),
            created_at=_get(c, "created_at", now),
            evaluated_at=now,
            prediction_hash=prediction_hash(vec),
            in_sample_metrics={"pooled_rank_ic": m["pooled_rank_ic"], "tstat": m["tstat"]},
            out_of_sample_metrics={"oos_ic_mean": m["oos_ic_mean"], "oos_ic_std": m["oos_ic_std"],
                                   "placebo_ic": m["placebo_ic"], "p_value": m["p_value"],
                                   "fdr_significant": fdr_ok},
            walk_forward_metrics={"fold_ic": m["fold_ic"], "ic_ci": [m["ic_ci_lo"], m["ic_ci_hi"]]},
            cost_model=m["cost"],
            correlation_to_accepted=m["correlation_to_accepted"],
            promotion=m["promotion"],
            score=m["score"], score_components=m["score_components"],
            decision=decision, rejection_reason=reason, tags=tags,
        )
        added = registry.add(rec, pred=vec)
        notes = []
        if not added:
            notes.append("duplicate prediction - not stored")
        if "FRAGILE_COST_MIRAGE" in tags:
            notes.append(f"FRAGILE: net Sharpe {m['cost']['net_sharpe_15']:.2f}@15bps / "
                         f"{m['cost']['net_sharpe_20']:.2f}@20bps - not promotable")
        rec.notes = "; ".join(notes)
        records.append(rec)
    return records


def _get(c, k, default=None):
    return getattr(c, k, None) if hasattr(c, k) else (c.get(k, default) if isinstance(c, dict) else default)
