"""Trading-horizon selection (V6 spec section 15).

For an opportunity evaluated across supported horizons, pick the horizon with the best RISK-ADJUSTED
expected NET profitability -- and only if that horizon has its own approved evidence. If no horizon
is economically favorable or evidence-backed, return NO_TRADE. Larger nominal swing targets never
win by size alone; the score penalizes capital occupation, holding time, and overnight gap risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

NO_TRADE = "NO_TRADE"


@dataclass
class HorizonEval:
    horizon: str
    expected_gross_return_bps: float
    expected_execution_cost_bps: float
    expected_downside_bps: float             # positive magnitude of expected adverse move
    target_before_stop_prob: float
    expected_holding_days: float
    overnight_gap_risk_bps: float = 0.0
    evidence_ok: bool = False                # does THIS horizon have approved evidence?

    @property
    def expected_net_return_bps(self) -> float:
        return self.expected_gross_return_bps - self.expected_execution_cost_bps


def _score(e: HorizonEval) -> float:
    """Risk-adjusted net return per unit of risk and time, penalized for overnight gap risk.
    Only meaningful when expected_net > 0; non-positive net yields a non-positive score."""
    net = e.expected_net_return_bps
    denom_risk = max(e.expected_downside_bps, 1.0)
    time_pen = max(e.expected_holding_days, 0.25)          # per-day efficiency; scalps ~ fraction
    gap_pen = 1.0 + max(e.overnight_gap_risk_bps, 0.0) / 100.0
    return (net / denom_risk) / (time_pen ** 0.5) / gap_pen


def select(evals: list[HorizonEval], min_net_bps: float = 0.0) -> tuple[str, Optional[HorizonEval], dict]:
    """Return (chosen_horizon, chosen_eval_or_None, per_horizon_scores). NO_TRADE if nothing
    qualifies (no evidence, or no positive-net horizon)."""
    scores = {}
    best = None
    best_score = float("-inf")
    for e in evals:
        s = _score(e) if (e.evidence_ok and e.expected_net_return_bps > min_net_bps) else float("-inf")
        scores[e.horizon] = round(s, 4) if s != float("-inf") else None
        if s > best_score:
            best_score = s
            best = e
    if best is None or best_score == float("-inf"):
        return NO_TRADE, None, scores
    return best.horizon, best, scores
