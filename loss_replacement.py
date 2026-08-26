#!/usr/bin/env python3
"""
V2.9.8 loss-replacement / momentum-rotation overlay (pure, shared).
===================================================================

ONE implementation of the rotation rule, shared by the backtest (``strat.run_backtest``)
and the live decision (``strat_decision.compute_target_portfolio``), so historical
and live behaviour are identical.

Rule (all thresholds/config live on ``AllocatorConfig``, never hard-coded here):

* An actively HELD position whose return vs its cost basis is at or below
  ``loss_replacement_threshold`` (default -0.5%) is a replacement candidate to EXIT.
* Its target allocation is rotated into the strongest ELIGIBLE positive-momentum
  asset that is (a) not currently held, (b) not itself being exited, (c) not in the
  post-exit cooldown, (d) has a finite positive momentum, (e) passes any external
  eligibility mask. Momentum is the point-in-time simple return P_t / P_{t-N} - 1.
* Multiple losers are processed worst-first (most negative return), ties by ticker
  ascending; candidates are ranked momentum descending, ties by ticker ascending;
  each candidate is assigned to at most one exit (no double-assignment).
* The swapped book is ALWAYS re-run through the strategy's ``enforce_constraints``
  with the SAME covariance / vol budget / growth ceiling / leverage cap that
  produced the base target, so no risk limit can be violated by the rotation. If a
  replacement would exceed a cap the excess is clipped (routes to cash); if there
  is no eligible candidate the freed weight routes to cash.

The trigger uses the ACTUAL position cost basis, not a one-day price move. In the
backtest, cost basis is tracked deterministically with a weighted average across
fills (``CostBasisTracker``) using information available at each decision close. In
live/paper, the engine supplies the broker-reported average entry price.

Determinism: no dict/set iteration order affects the result; all ordering is by
(metric, ticker). Point-in-time: ranking uses only ``momentum`` and
``position_returns`` computed from data at/through the decision close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strat_allocator import AllocatorConfig, enforce_constraints


@dataclass
class LossReplacementResult:
    triggered: bool
    weights: np.ndarray
    events: list[dict] = field(default_factory=list)
    rebalance_reason: str | None = None
    exited: list[str] = field(default_factory=list)


def _event(exited: str, pr: float, thr: float,
           replacement: str | None, mom: float | None, reason: str) -> dict:
    return {
        "type": "loss_replacement",
        "exited_asset": exited,
        "position_return": round(float(pr), 6),
        "threshold": float(thr),
        "replacement_asset": replacement,
        "replacement_momentum": (round(float(mom), 6) if mom is not None else None),
        "reason": reason,
    }


def apply_loss_replacement(*,
                           base_weights: np.ndarray,
                           position_returns: np.ndarray,
                           momentum: np.ndarray,
                           cov_adj: np.ndarray,
                           cfg: AllocatorConfig,
                           growth_ceiling: float,
                           vol_budget: float,
                           leverage_cap: float,
                           cooldown_mask: np.ndarray | None = None,
                           eligible_mask: np.ndarray | None = None) -> LossReplacementResult:
    """
    Rotate breached-loss holdings into positive-momentum replacements, then repair.

    base_weights     : the allocator's target weights for the risk assets (post
                       decide_target_weights), in ``cfg.risk_assets`` order.
    position_returns : per asset current/avg_entry - 1; NaN where NOT currently held.
    momentum         : per asset point-in-time momentum (P_t/P_{t-N}-1); NaN if N/A.
    cov_adj / growth_ceiling / vol_budget / leverage_cap : the SAME risk inputs the
                       base decision used (from its diag), so the repaired book is
                       constrained identically.
    cooldown_mask    : bool per asset; True => barred as a replacement (recent exit).
    eligible_mask    : bool per asset; False => ineligible (data/tradability/etc).

    Returns a LossReplacementResult. When no held asset breaches the threshold the
    result is (triggered=False, base_weights unchanged) -- a guaranteed no-op.
    """
    assets = list(cfg.risk_assets)
    n = len(assets)
    w = np.asarray(base_weights, float).astype(float).copy()
    pr = np.asarray(position_returns, float)
    mom = np.asarray(momentum, float)
    thr = float(cfg.loss_replacement_threshold)
    eps = float(cfg.loss_replacement_hold_eps)
    if cooldown_mask is None:
        cooldown_mask = np.zeros(n, bool)
    if eligible_mask is None:
        eligible_mask = np.ones(n, bool)
    cooldown_mask = np.asarray(cooldown_mask, bool)
    eligible_mask = np.asarray(eligible_mask, bool)

    held = np.isfinite(pr)
    breached = held & (pr <= thr + 1e-12)   # inclusive threshold
    breached_idx = [j for j in range(n) if breached[j]]
    if not breached_idx:
        return LossReplacementResult(False, w, [], None, [])

    def is_candidate(j: int) -> bool:
        return (bool(eligible_mask[j]) and not bool(cooldown_mask[j])
                and not held[j]                     # must be currently flat
                and not breached[j]
                and np.isfinite(mom[j]) and float(mom[j]) > 0.0)

    # candidates ranked momentum DESC, ticker ASC (deterministic tie-break)
    pool = sorted((j for j in range(n) if is_candidate(j)),
                  key=lambda j: (-float(mom[j]), assets[j]))
    # exits worst-first (most-negative return), ticker ASC
    breached_idx.sort(key=lambda j: (float(pr[j]), assets[j]))

    events: list[dict] = []
    exited: list[str] = []
    used: set[int] = set()
    for ex in breached_idx:
        freed = float(w[ex])
        w[ex] = 0.0                       # exit the position in the target book
        exited.append(assets[ex])
        repl = next((j for j in pool if j not in used), None)
        if repl is None or freed <= eps:
            # nothing to rotate into (no candidate) OR nothing to redeploy
            reason = ("no_positive_momentum_candidate" if repl is None
                      else "position_loss_threshold_breached")
            events.append(_event(assets[ex], pr[ex], thr, None, None, reason))
            continue
        used.add(repl)
        w[repl] += freed
        events.append(_event(assets[ex], pr[ex], thr, assets[repl], mom[repl],
                             "position_loss_threshold_breached"))

    # Re-run the strategy's own constraint machinery on the swapped book so the
    # rotation cannot violate per-asset caps, growth ceiling, leverage, or vol.
    w = enforce_constraints(w, cfg, growth_ceiling, np.asarray(cov_adj, float),
                            vol_budget, leverage_cap)
    return LossReplacementResult(True, w, events, "loss_replacement", exited)


class CostBasisTracker:
    """
    Deterministic weighted-average cost basis for the BACKTEST.

    Tracks a synthetic share count and average entry price per asset so the
    historical loss trigger uses the same quantity the live broker reports
    (avg_entry_price), not a one-day return. Prices in / equity in are the
    decision-close values available at each step -- never future information.
    """

    def __init__(self, assets, *, min_shares: float = 1e-12):
        self.assets = list(assets)
        self.n = len(self.assets)
        self.min_shares = float(min_shares)
        self.shares = np.zeros(self.n)
        self.cost = np.full(self.n, np.nan)   # avg entry price; NaN when flat
        self.entry_session = np.full(self.n, -1, dtype=int)  # first-entry idx; -1 = flat

    def position_returns(self, price_row: np.ndarray) -> np.ndarray:
        p = np.asarray(price_row, float)
        out = np.full(self.n, np.nan)
        live = np.isfinite(self.cost) & (self.shares > self.min_shares) & (p > 0)
        out[live] = p[live] / self.cost[live] - 1.0
        return out

    def days_held(self, session_i: int) -> np.ndarray:
        """Sessions since first entry (NaN where flat). Scale-ins do NOT reset it."""
        out = np.full(self.n, np.nan)
        held = self.entry_session >= 0
        out[held] = float(session_i) - self.entry_session[held]
        return out

    def update(self, target_weights: np.ndarray, price_row: np.ndarray,
               equity: float, session: int | None = None) -> None:
        """
        Move from current shares to the target book at ``price_row``. If ``session``
        is given, stamp the first-entry session on a flat->held transition (a
        scale-in does NOT reset it, matching a broker avg_entry_price) and clear it
        on a full exit -- the basis for days_held.
        """
        p = np.asarray(price_row, float)
        tw = np.asarray(target_weights, float)
        eq = float(equity)
        tgt_shares = np.where(p > 0, tw * eq / np.where(p > 0, p, 1.0), 0.0)
        for j in range(self.n):
            d = float(tgt_shares[j] - self.shares[j])
            if d > self.min_shares:                     # buy -> weighted-avg cost
                prev_sh = float(self.shares[j])
                if prev_sh <= self.min_shares or not np.isfinite(self.cost[j]):
                    self.cost[j] = float(p[j])
                    if session is not None:             # flat -> held: stamp entry
                        self.entry_session[j] = int(session)
                else:
                    self.cost[j] = (prev_sh * self.cost[j] + d * float(p[j])) / (prev_sh + d)
                self.shares[j] = prev_sh + d
            elif d < -self.min_shares:                  # sell -> reduce, cost held
                self.shares[j] = float(self.shares[j]) + d
                if self.shares[j] <= self.min_shares:
                    self.shares[j] = 0.0
                    self.cost[j] = np.nan
                    self.entry_session[j] = -1          # full exit: clear entry


def demo() -> None:
    """Self-check: trigger boundary, candidate ranking, no-candidate, cost basis."""
    cfg = AllocatorConfig(loss_replacement_enabled=True)
    n = len(cfg.risk_assets)
    a = list(cfg.risk_assets)
    cov = np.eye(n) * 0.02
    base = np.zeros(n)
    # hold SPY at index 0 with a big weight; flats elsewhere
    i_spy, i_qqq, i_iwm = a.index("SPY"), a.index("QQQ"), a.index("IWM")
    base[i_spy] = 0.30

    # --- trigger boundary: -0.49% does NOT trip, -0.50% DOES ---
    pr = np.full(n, np.nan); pr[i_spy] = -0.0049
    mom = np.full(n, np.nan); mom[i_qqq] = 0.02
    r = apply_loss_replacement(base_weights=base, position_returns=pr, momentum=mom,
                               cov_adj=cov, cfg=cfg, growth_ceiling=1.4,
                               vol_budget=0.15, leverage_cap=1.4)
    assert r.triggered is False, "-0.49% must not trigger"
    pr[i_spy] = -0.0050
    r = apply_loss_replacement(base_weights=base, position_returns=pr, momentum=mom,
                               cov_adj=cov, cfg=cfg, growth_ceiling=1.4,
                               vol_budget=0.15, leverage_cap=1.4)
    assert r.triggered is True, "-0.50% must trigger (inclusive)"
    assert r.exited == ["SPY"]
    assert r.events[0]["replacement_asset"] == "QQQ"
    assert r.weights[i_spy] == 0.0 and r.weights[i_qqq] > 0.0

    # --- candidate ranking: strongest positive momentum wins; negative excluded ---
    pr = np.full(n, np.nan); pr[i_spy] = -0.007
    mom = np.full(n, np.nan)
    mom[i_qqq] = 0.02; mom[i_iwm] = 0.04; mom[a.index("EFA")] = -0.01
    r = apply_loss_replacement(base_weights=base, position_returns=pr, momentum=mom,
                               cov_adj=cov, cfg=cfg, growth_ceiling=1.4,
                               vol_budget=0.15, leverage_cap=1.4)
    assert r.events[0]["replacement_asset"] == "IWM", "highest momentum must win"

    # --- no eligible candidate: exit anyway, route to cash ---
    mom = np.full(n, np.nan); mom[i_qqq] = -0.02
    r = apply_loss_replacement(base_weights=base, position_returns=pr, momentum=mom,
                               cov_adj=cov, cfg=cfg, growth_ceiling=1.4,
                               vol_budget=0.15, leverage_cap=1.4)
    assert r.triggered and r.events[0]["replacement_asset"] is None
    assert r.weights[i_spy] == 0.0 and float(r.weights.sum()) < float(base.sum()) + 1e-12

    # --- cost basis: two fills weighted-average; sell resets on flat ---
    t = CostBasisTracker(a)
    t.update(_w(n, i_spy, 0.5), _p(n, i_spy, 100.0), 1000.0)   # buy 5 @ 100
    t.update(_w(n, i_spy, 1.02), _p(n, i_spy, 102.0), 1000.0)  # buy ~5 more @ 102
    assert 100.5 < float(t.cost[i_spy]) < 101.5, t.cost[i_spy]
    t.update(_w(n, i_spy, 0.0), _p(n, i_spy, 102.0), 1000.0)   # sell all -> flat
    assert not np.isfinite(t.cost[i_spy])
    print("loss_replacement demo: OK")


def _w(n, i, v):
    w = np.zeros(n); w[i] = v; return w


def _p(n, i, v):
    p = np.full(n, 100.0); p[i] = v; return p


if __name__ == "__main__":
    demo()
