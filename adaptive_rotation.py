#!/usr/bin/env python3
"""
V2.9.8 adaptive opportunity-gated rotation engine (pure, shared).
=================================================================

The UNIFIED rotation overlay. Real-data sweeps proved a fixed % loss stop produces
turnover, not alpha. So a deterioration signal here does NOT auto-sell -- it OPENS a
HOLD-vs-ROTATE evaluation. Capital moves only when a candidate (or CASH) is
materially superior after costs, turnover, portfolio risk, and hysteresis.

Decision (per held "source" asset s that deteriorated):

  source iff  damage[s] <= regime_threshold[state]   (vol-normalized adverse move)
              OR  exhaustion hazard confirmed
  candidate c in (flat/eligible/non-cooldown risky assets) + CASH
  ROTATE s->c iff  BOTH:
     Gate 1 (rank-units, cross-sectional opportunity):
        net_edge = candidate_opportunity[c] - hold_score[s]
                   - turnover_penalty[s] - concentration_penalty[c]
        net_edge > minimum_rotation_edge
        AND candidate_opportunity[c] >= entry_score_threshold   (quality floor)
        AND hold_score[s] <= hold_score_floor                   (hysteresis)
     Gate 2 (excess-return units, transaction-cost gate, size-independent):
        forecast_improvement > 2*transaction_cost + cost_safety_margin
        (risky: view[c]-view[s];  cash: -view[s])
  else HOLD (this is the anti-churn behavior).

Scores are cross-sectional percentile ranks (the codebase's native currency -- the
rolling IC is Spearman rank IC), computed point-in-time from the same backward-only
signals as the other overlays (reused exhaustion_panel/reversal_panel + multi-horizon
momentum + the allocator's own excess-return `view`). No look-ahead. Deterministic:
all ordering by (metric, ticker); greedy assignment; edges rounded to 1e-9.

CASH is a first-class candidate. Because `view` is an EXCESS return, cash's forecast
is identically 0 excess -- no risk-free rate is threaded into the overlay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strat_allocator import AllocatorConfig, enforce_constraints, annualized_vols
from momentum_exhaustion import exhaustion_panel, reversal_panel

# Cash's "reversal" improves as the risk regime deteriorates (capital preservation).
CASH_R0 = {"CRISIS": 0.90, "DEFENSIVE": 0.75, "BOTTOMING": 0.50,
           "RECOVERING": 0.30, "BULL": 0.15}
_DAMAGE_HORIZON = 20.0          # position_return is a multi-day cumulative return
TRADING_DAYS = 252.0


@dataclass
class AdaptiveRotationResult:
    triggered: bool
    weights: np.ndarray
    events: list[dict] = field(default_factory=list)     # executed rotations
    skipped: list[dict] = field(default_factory=list)    # evaluated-but-HELD
    exited: list[str] = field(default_factory=list)
    rebalance_reason: str | None = None


def adaptive_signal_panels(prices: pd.DataFrame, cfg: AllocatorConfig,
                           view: pd.DataFrame | None):
    """Precomputable point-in-time panels shared by backtest + live (index=prices.index)."""
    assets = list(cfg.risk_assets)
    px = prices[assets].astype(float)
    exh, _ = exhaustion_panel(prices, cfg, view)
    rev, _ = reversal_panel(prices, cfg, view)
    mom20 = px / px.shift(20) - 1.0
    accel = (px / px.shift(5) - 1.0) - mom20          # mom5 - mom20
    return exh[assets], rev[assets], mom20, accel


def _pctrank(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank in (0,1] over the mask&finite population; NaN else."""
    vals = np.asarray(vals, float)
    idx = np.where(mask & np.isfinite(vals))[0]
    out = np.full(len(vals), np.nan)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[idx[0]] = 0.5
        return out
    r = pd.Series(vals[idx]).rank(pct=True).to_numpy()   # deterministic, avg ties
    out[idx] = r
    return out


def _diversification_benefit(c: int, base_weights: np.ndarray, cov_adj: np.ndarray,
                             held: np.ndarray, cfg: AllocatorConfig, eps: float) -> float:
    """min(w * (1-book_weighted_corr)/2, cap) from cov_adj -- bounded so it can't dominate."""
    H = [j for j in range(len(base_weights)) if held[j] and base_weights[j] > eps and j != c]
    d = np.sqrt(np.maximum(np.diag(cov_adj), 1e-16))
    if not H:
        div_raw = 1.0
    else:
        wsum = sum(base_weights[j] for j in H)
        if wsum <= eps:
            div_raw = 1.0
        else:
            wbar = sum(base_weights[j] * (cov_adj[c, j] / (d[c] * d[j])) for j in H) / wsum
            div_raw = float(np.clip((1.0 - wbar) / 2.0, 0.0, 1.0))
    return float(min(cfg.adaptive_diversification_weight * div_raw,
                     cfg.adaptive_diversification_cap))


def apply_adaptive_rotation(*,
                            base_weights: np.ndarray,
                            position_returns: np.ndarray,   # (n,) P/avg_entry-1; NaN if flat
                            days_held: np.ndarray,          # (n,) sessions since first entry; NaN if flat
                            view_row: np.ndarray,           # (n,) 20d EXCESS return (raw view)
                            sigma_ann: np.ndarray,          # (n,) annualized_vols(cov_adj)
                            mom20_row: np.ndarray,          # (n,)
                            accel_row: np.ndarray,          # (n,) mom5-mom20
                            reversal_score: np.ndarray,     # (n,) reversal_panel row
                            exhaustion_recent: np.ndarray,  # (k,n) last-k exhaustion scores
                            cov_adj: np.ndarray,            # (n,n)
                            state: str, severe: bool,
                            cfg: AllocatorConfig,
                            growth_ceiling: float, vol_budget: float, leverage_cap: float,
                            cooldown_mask: np.ndarray | None = None,
                            eligible_mask: np.ndarray | None = None,
                            hazard_prob: np.ndarray | None = None,       # ML hook (None -> deterministic)
                            reversal_prob: np.ndarray | None = None      # ML hook (None -> deterministic)
                            ) -> AdaptiveRotationResult:
    assets = list(cfg.risk_assets)
    n = len(assets)
    w = np.asarray(base_weights, float).copy()
    pr = np.asarray(position_returns, float)
    dh = np.asarray(days_held, float)
    view = np.asarray(view_row, float)
    sigma = np.asarray(sigma_ann, float)
    rev = np.asarray(reversal_score, float)
    eps = float(cfg.loss_replacement_hold_eps)
    if cooldown_mask is None:
        cooldown_mask = np.zeros(n, bool)
    if eligible_mask is None:
        eligible_mask = np.ones(n, bool)
    cooldown_mask = np.asarray(cooldown_mask, bool)
    eligible_mask = np.asarray(eligible_mask, bool)

    held = np.isfinite(pr)
    if not held.any():
        return AdaptiveRotationResult(False, w, [], [], [], None)

    # --- vol-normalized damage (20-day vol scale; NOT 1-day) ----------------
    vol_h = sigma * math.sqrt(_DAMAGE_HORIZON / TRADING_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        damage = np.where((vol_h > 1e-9) & held, pr / vol_h, np.nan)

    # --- exhaustion-hazard confirmation (ML hook -> deterministic fallback) --
    kconf = max(1, int(cfg.adaptive_hazard_confirmations))
    recent = np.asarray(exhaustion_recent, float)[-kconf:]
    if reversal_prob is not None:
        rev = np.where(np.isfinite(reversal_prob), np.asarray(reversal_prob, float), rev)
    if hazard_prob is not None:
        hazard_confirmed = np.isfinite(hazard_prob) & (np.asarray(hazard_prob, float)
                                                       >= cfg.momentum_exhaustion_threshold)
    else:
        hazard_confirmed = np.all(recent >= cfg.momentum_exhaustion_threshold - 1e-12, axis=0)

    # --- cross-sectional ranks over the eligible-finite population ----------
    rankable = eligible_mask & np.isfinite(view)
    with np.errstate(invalid="ignore", divide="ignore"):
        riskadj = np.where(sigma > 1e-9, mom20_row / sigma, np.nan)
    forecast_rank = _pctrank(view, rankable)
    accel_rank = _pctrank(accel_row, rankable)
    riskadj_rank = _pctrank(riskadj, rankable)
    reversal_rank = _pctrank(rev, rankable)
    _stack = np.vstack([forecast_rank, accel_rank, riskadj_rank])
    _cnt = np.sum(np.isfinite(_stack), axis=0)
    base_score = np.where(_cnt > 0, np.nansum(_stack, axis=0) / np.maximum(_cnt, 1), np.nan)

    # --- sources: held assets that deteriorated (damage OR hazard) ----------
    reg_thr = float(cfg.adaptive_damage_thresholds.get(str(state).upper(), -1.25))
    real_damage = held & np.isfinite(damage) & (damage <= reg_thr)
    sources = [j for j in range(n) if held[j] and (real_damage[j] or hazard_confirmed[j])]
    if not sources:
        return AdaptiveRotationResult(False, w, [], [], [], None)

    # --- opportunity scoring: reversal is SYMMETRIC (credited to the incumbent AND
    # candidates alike) so a deteriorated-but-REVERSING holding is not unfairly churned
    # out of its own bounce; diversification is the ONLY candidate-only term -- the
    # genuine switching benefit of a name not yet in the book.
    rev_bonus = cfg.adaptive_reversal_bonus
    hold_value = base_score + rev_bonus * np.where(np.isfinite(reversal_rank), reversal_rank, 0.0)

    def cand_opp(c):
        return float(hold_value[c] + _diversification_benefit(c, w, cov_adj, held, cfg, eps))

    risky_cands = [c for c in range(n)
                   if eligible_mask[c] and not cooldown_mask[c] and not held[c]
                   and np.isfinite(base_score[c])]
    cand_opp_c = {c: cand_opp(c) for c in risky_cands}

    # CASH candidate on the same scale (view is EXCESS -> cash forecast = 0 excess)
    finite_view = view[rankable]
    # cash ranks above only the assets with NEGATIVE excess forecast (neutral views
    # do not favour cash) -> cash strengthens as views turn broadly bearish.
    forecast_rank_cash = float(np.mean(finite_view < 0.0)) if finite_view.size else 0.0
    rev_rank_cash = max(CASH_R0.get(str(state).upper(), 0.5), 0.90 if severe else 0.0)
    cand_opp_cash = (forecast_rank_cash + rev_bonus * rev_rank_cash
                     + cfg.adaptive_diversification_cap)

    def turnover_penalty(s):
        d = dh[s]
        if not np.isfinite(d):
            d = 0.0
        return cfg.adaptive_turnover_penalty * float(
            np.clip((cfg.adaptive_min_hold_days - d) / max(cfg.adaptive_min_hold_days, 1), 0.0, 1.0))

    def concentration_penalty(c):
        cap = float(cfg.asset_max_weights[assets[c]])
        return cfg.adaptive_concentration_penalty * float(np.clip(w[c] / max(cap, 1e-9), 0.0, 1.0))

    cost_gate = 2.0 * cfg.transaction_cost + cfg.adaptive_cost_safety_margin      # round trip (risky)
    cash_cost_gate = cfg.transaction_cost + cfg.adaptive_cost_safety_margin       # single sell leg (cash)
    min_edge = cfg.adaptive_minimum_rotation_edge

    # candidate list per source: risky + CASH, with (opp, view_improvement, net_edge)
    def evaluate(s):
        hs = hold_value[s] if np.isfinite(hold_value[s]) else 0.0
        tp = turnover_penalty(s)
        opts = []
        for c in risky_cands:
            opp = cand_opp_c[c]
            ne = round(opp - hs - tp - concentration_penalty(c), 9)
            fi = round(float(view[c] - view[s]), 12) if np.isfinite(view[c]) and np.isfinite(view[s]) else -np.inf
            opts.append(("risky", c, assets[c], opp, ne, fi))
        # cash (reusable)
        ne_cash = round(cand_opp_cash - hs - tp, 9)     # no concentration penalty for cash
        fi_cash = round(float(-view[s]), 12) if np.isfinite(view[s]) else -np.inf
        opts.append(("cash", -1, "CASH", cand_opp_cash, ne_cash, fi_cash))
        return hs, tp, opts

    # --- deterministic greedy over the rotation matrix ----------------------
    # order sources by their best available net_edge (desc), then ticker (asc)
    src_best = {}
    src_eval = {}
    for s in sources:
        hs, tp, opts = evaluate(s)
        src_eval[s] = (hs, tp, opts)
        src_best[s] = max(o[4] for o in opts)
    ordered = sorted(sources, key=lambda s: (-src_best[s], assets[s]))

    used_risky: set[int] = set()
    events, skipped, exited = [], [], []
    for s in ordered:
        hs, tp, opts = src_eval[s]
        base_ev = {"asset": assets[s], "position_return": _rnd(pr[s]),
                   "damage": _rnd(damage[s]), "hold_score": _rnd(hs),
                   "regime": str(state), "days_held": (None if not np.isfinite(dh[s]) else int(dh[s])),
                   "real_damage": bool(real_damage[s]), "hazard_confirmed": bool(hazard_confirmed[s]),
                   "type": "adaptive_rotation"}

        # SOURCE-level gates (block the whole source; recorded with the funnel stage).
        if round(hs, 9) > cfg.adaptive_hold_score_floor:
            skipped.append({**base_ev, "action": "hold", "replacement_asset": None,
                            "stage": "hysteresis", "reason": "hold_still_strong_hysteresis"})
            continue
        bypass = bool(real_damage[s])
        if (not bypass) and np.isfinite(dh[s]) and dh[s] < cfg.adaptive_min_hold_days:
            skipped.append({**base_ev, "action": "hold", "replacement_asset": None,
                            "stage": "min_hold", "reason": "min_hold_days_not_met"})
            continue

        # CANDIDATE-level gates: first option passing quality AND edge AND cost wins.
        opts_sorted = sorted(opts, key=lambda o: (-o[4], o[2]))
        stage, chosen, edge_blocked = "no_candidate", None, False
        for kind, c, tick, opp, ne, fi in opts_sorted:
            if kind == "risky" and c in used_risky:
                continue
            gc = cost_gate if kind == "risky" else cash_cost_gate   # cash = single sell leg
            stage = _adv(stage, "candidate")
            if opp < cfg.adaptive_entry_score_threshold:
                continue
            stage = _adv(stage, "quality")
            if ne <= min_edge:
                if fi > gc:                         # would pass cost -> min_edge is the binding gate
                    edge_blocked = True
                continue
            stage = _adv(stage, "edge")
            if fi <= gc:
                continue
            stage = _adv(stage, "cost")
            chosen = (kind, c, tick, opp, ne, fi)
            break

        if chosen is None:
            reason = {"no_candidate": "no_eligible_candidate",
                      "candidate": "candidate_below_quality_floor",
                      "quality": ("insufficient_net_rotation_edge" if edge_blocked else "cost_gated"),
                      "edge": "cost_gated"}.get(stage, "insufficient_net_rotation_edge")
            skipped.append({**base_ev, "action": "hold", "replacement_asset": None,
                            "stage": stage, "reason": reason, "edge_blocked": bool(edge_blocked)})
            continue
        kind, c, tick, opp, ne, fi = chosen
        freed = float(w[s])
        w[s] = 0.0
        exited.append(assets[s])
        if kind == "risky":
            used_risky.add(c)
            w[c] += freed
        events.append({**base_ev, "exited_asset": assets[s], "stage": "executed",
                       "action": ("rotate" if kind == "risky" else "rotate_to_cash"),
                       "replacement_asset": (None if kind == "cash" else tick),
                       "candidate_opportunity": _rnd(opp),
                       "gross_rotation_edge": _rnd(opp - hs),
                       "net_rotation_edge": _rnd(ne),
                       "minimum_required_edge": float(min_edge),
                       "forecast_improvement": _rnd(fi),
                       "bullish_reversal_score": _rnd(rev[c]) if kind == "risky" else None,
                       "reason": "superior_opportunity_after_costs"})

    if not exited:
        return AdaptiveRotationResult(False, w, [], skipped, [], None)
    w = enforce_constraints(w, cfg, growth_ceiling, np.asarray(cov_adj, float),
                            vol_budget, leverage_cap)
    return AdaptiveRotationResult(True, w, events, skipped, exited, "adaptive_rotation")


_STAGES = ["no_candidate", "candidate", "quality", "edge", "cost", "executed"]


def _adv(cur: str, new: str) -> str:
    """Return the later of two funnel stages (monotone progress)."""
    return new if _STAGES.index(new) > _STAGES.index(cur) else cur


def _rnd(x, nd=6):
    try:
        return round(float(x), nd) if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def demo() -> None:
    """Self-check: opportunity gate holds a lone mediocre swap; rotates on a clear edge."""
    cfg = AllocatorConfig(adaptive_rotation_enabled=True)
    n = len(cfg.risk_assets)
    a = list(cfg.risk_assets)
    cov = np.eye(n) * 0.02
    sigma = annualized_vols(cov)
    base = np.zeros(n)
    i_spy, i_iwm, i_qqq = a.index("SPY"), a.index("IWM"), a.index("QQQ")
    base[i_spy] = 0.30

    vol_h = sigma[i_spy] * math.sqrt(20 / 252)

    def call(*, pr_spy, spy_weak=True, dh_spy=50):
        pr = np.full(n, np.nan); pr[i_spy] = pr_spy
        dh = np.full(n, np.nan); dh[i_spy] = dh_spy
        view = np.zeros(n); mom20 = np.zeros(n); accel = np.zeros(n)
        if spy_weak:                         # SPY clearly weakest, IWM clearly strongest
            view[i_spy], mom20[i_spy], accel[i_spy] = -0.02, -0.05, -0.03
            view[i_iwm], mom20[i_iwm], accel[i_iwm] = 0.03, 0.06, 0.04
        recent = np.zeros((cfg.adaptive_hazard_confirmations, n))
        return apply_adaptive_rotation(
            base_weights=base, position_returns=pr, days_held=dh, view_row=view,
            sigma_ann=sigma, mom20_row=mom20, accel_row=accel, reversal_score=np.zeros(n),
            exhaustion_recent=recent, cov_adj=cov, state="DEFENSIVE", severe=False, cfg=cfg,
            growth_ceiling=1.4, vol_budget=0.15, leverage_cap=1.4)

    # SPY damaged (-2 vol-units) and weakest; IWM clearly superior -> ROTATE SPY->IWM
    r = call(pr_spy=-2.0 * vol_h)
    assert r.triggered and r.exited == ["SPY"], (r.triggered, r.exited, r.skipped)
    assert r.events[0]["replacement_asset"] == "IWM", r.events

    # Same damage but every asset equally mediocre -> hysteresis HOLD (anti-churn)
    r2 = call(pr_spy=-2.0 * vol_h, spy_weak=False)
    assert not r2.triggered and r2.skipped, "must HOLD when nothing is superior"

    # Not damaged and not hazard -> not even a source -> no-op
    r4 = call(pr_spy=-0.2 * vol_h)           # damage ~ -0.2, above the DEFENSIVE threshold
    assert not r4.triggered

    # Genuinely damaged but the position is only 1 day old -> real damage BYPASSES min_hold
    r5 = call(pr_spy=-2.0 * vol_h, dh_spy=1)
    assert r5.triggered, "real damage must bypass min_hold"
    print("adaptive_rotation demo: OK")


if __name__ == "__main__":
    demo()
