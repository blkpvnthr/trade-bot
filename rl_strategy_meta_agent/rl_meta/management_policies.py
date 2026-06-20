"""Mid-trade management policies driven by the engineered p_severe_giveback score.

Each policy takes a trade's signed return path `r` (return-from-entry at each
bar; r[-1] is the strategy's own exit) and a per-bar giveback-probability array
`p` (p[k] = P(severe giveback) known AT bar k), and returns the realized return
under that policy. Pure simulation -- research only, no broker, no orders.

  baseline           : ride to the strategy's original exit (r[-1]).
  full_exit          : at the first bar p>=thr, close the whole position (r[k]).
  partial_exit       : at the first bar p>=thr, sell `frac`; ride the rest to r[-1].
  tighten_stop       : ride until the first bar p>=thr, then a TIGHT trailing stop.
  de_risk            : at the first bar p>=thr, trim `frac` and trail the rest
                       loosely (no new adds -- single-entry trades here, so the
                       meaningful part is the trim + protective trail).
"""
from __future__ import annotations

import numpy as np

EPS = 0.001          # 0.1% return tolerance for "better/worse" comparisons
SEVERE = 0.01        # >=1% surrendered from the path peak to the final = severe


def _first_trigger(p, thr: float):
    hits = np.where(np.asarray(p, float) >= thr)[0]
    return int(hits[0]) if len(hits) else None


def _trail_from(r, k: int, trail: float) -> float:
    """Tight trailing stop applied from bar k onward; r[-1] if never hit."""
    peak = float(r[k])
    for j in range(k + 1, len(r)):
        peak = max(peak, float(r[j - 1]))
        if peak - float(r[j]) >= trail:
            return float(r[j])
    return float(r[-1])


def baseline(r, p=None) -> float:
    return float(r[-1])


def full_exit(r, p, thr: float = 0.65) -> float:
    k = _first_trigger(p, thr)
    return float(r[k]) if k is not None else float(r[-1])


def partial_exit(r, p, thr: float = 0.60, frac: float = 0.5) -> float:
    k = _first_trigger(p, thr)
    if k is None:
        return float(r[-1])
    return frac * float(r[k]) + (1 - frac) * float(r[-1])


def tighten_stop(r, p, thr: float = 0.55, trail: float = 0.005) -> float:
    k = _first_trigger(p, thr)
    if k is None:
        return float(r[-1])
    return _trail_from(r, k, trail)


def de_risk(r, p, thr: float = 0.60, frac: float = 0.5, trail: float = 0.01) -> float:
    k = _first_trigger(p, thr)
    if k is None:
        return float(r[-1])
    return frac * float(r[k]) + (1 - frac) * _trail_from(r, k, trail)


# the experiment's five policies, by name
POLICIES = {
    "baseline": lambda r, p: baseline(r, p),
    "full_exit_p065": lambda r, p: full_exit(r, p, 0.65),
    "partial_50_p060": lambda r, p: partial_exit(r, p, 0.60, 0.5),
    "tighten_stop_p055": lambda r, p: tighten_stop(r, p, 0.55, 0.005),
    "de_risk_p060": lambda r, p: de_risk(r, p, 0.60, 0.5, 0.01),
}


def severe_giveback_happened(r, severe: float = SEVERE) -> bool:
    """Did the trade surrender >= `severe` from its path peak to the final bar?"""
    return (float(np.max(r)) - float(r[-1])) >= severe


def action_labels(r, k: int, trail: float = 0.005, partial_trail: float = 0.01) -> dict:
    """Counterfactual ACTION labels at decision bar k -- "which action wins?",
    not "will a giveback occur?". Compares each action's realized return to
    holding to the strategy's exit (r[-1]).

      exit_now_better    : exiting fully now beats holding (r[k] > r[-1]).
      tighten_stop_better: a tight trailing stop from k beats holding.
      take_partial_better: sell half now + trail the rest beats holding 100%.

    NB: a naive 'sell half, ride the rest to the SAME exit' collapses to
    exit_now_better algebraically, so the partial here TRAILS its remainder.
    """
    rk, final = float(r[k]), float(r[-1])
    tight = _trail_from(r, k, trail)
    partial = 0.5 * rk + 0.5 * _trail_from(r, k, partial_trail)
    return {"exit_now_better": int(rk > final),
            "tighten_stop_better": int(tight > final),
            "take_partial_better": int(partial > final)}


def giveback_recovery(r, k: int, severe: float = SEVERE) -> dict:
    """Diagnostic at bar k: did a (future) severe giveback also make a NEW HIGH
    above the decision-time peak before the strategy's exit? If yes, the giveback
    flag is tagging a recoverable pullback, not terminal deterioration."""
    run_peak = float(np.max(r[:k + 1]))
    fut = r[k + 1:]
    fut_max = float(np.max(fut)) if len(fut) else float(r[k])
    peak_from_here = max(float(r[k]), fut_max)
    return {"severe_giveback": int(peak_from_here - float(r[-1]) >= severe),
            "new_high_after": int(fut_max > run_peak),
            "exit_now_better": int(float(r[k]) > float(r[-1]))}


def _demo():
    # rise to +5% then give back to -1% (a severe giveback); trigger fires at k=4
    r = np.array([0.0, 0.01, 0.03, 0.04, 0.05, 0.02, -0.01])
    p = np.array([0.0, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90])   # crosses 0.65 at k=4
    assert baseline(r, p) == -0.01
    assert full_exit(r, p, 0.65) == 0.05                       # locks the peak
    assert abs(partial_exit(r, p, 0.60, 0.5) - (0.5 * 0.05 + 0.5 * -0.01)) < 1e-9
    assert tighten_stop(r, p, 0.55, 0.005) > baseline(r, p)    # trail beats riding down
    assert severe_giveback_happened(r)
    # no trigger -> every policy == baseline
    flat = np.array([0.0, 0.01, 0.02, 0.03])
    pq = np.zeros(4)
    assert all(abs(fn(flat, pq) - flat[-1]) < 1e-9 for fn in POLICIES.values())
    # action labels: at k=4 (peak +5%, final -1%) exiting now clearly beats holding
    al = action_labels(r, 4)
    assert al["exit_now_better"] == 1 and al["tighten_stop_better"] == 1
    gr = giveback_recovery(r, 2)            # peak ahead is 0.05 > peak-so-far 0.03
    assert gr["severe_giveback"] == 1 and gr["new_high_after"] == 1
    print("management_policies._demo OK", {k: round(fn(r, p), 4) for k, fn in POLICIES.items()})


if __name__ == "__main__":
    _demo()
