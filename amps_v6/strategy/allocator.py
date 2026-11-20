"""Event-driven allocator (V6-Integrated spec sections 10-11; paper 1 Eq. 3, paper 2 Eq. 5).

Converts alpha-confirmed candidates into TARGET WEIGHTS, and runs ONLY on reallocation events (the
pipeline calls this when state.diff_events() is non-empty) -- never per bar. The allocator proposes;
the independent risk gateway may still reduce/reject.

CORRECTNESS NOTE (spec section 10): both papers write the step update as min(w*-w, kappa, w_bar-w)
(paper 1 Eq. 4, paper 2 Eq. 6). A bare min() mishandles REDUCTIONS: when w* < w the requested delta
is negative and min(...) clips it toward the caps incorrectly, so a position can fail to shrink. We
use the signed clip() form the spec mandates, which is correct for both increases and reductions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp_weight(current_weight: float, target_weight: float, step_cap: float,
                 minimum_weight: float, maximum_weight: float) -> tuple[float, list[str]]:
    """The signed, reduction-safe weight clamp (spec section 10). Returns (approved_weight, reasons).

    requested_delta = target - current
    bounded_delta   = clip(requested_delta, -step_cap, +step_cap)   # symmetric: caps up AND down
    approved_weight = clip(current + bounded_delta, min, max)
    """
    reasons: list[str] = []
    requested_delta = target_weight - current_weight
    bounded_delta = _clip(requested_delta, -step_cap, step_cap)
    if bounded_delta != requested_delta:
        reasons.append(f"step_cap({step_cap}) {'up' if requested_delta > 0 else 'down'}")
    approved = _clip(current_weight + bounded_delta, minimum_weight, maximum_weight)
    if approved != current_weight + bounded_delta:
        reasons.append(f"bounds[{minimum_weight},{maximum_weight}]")
    return approved, reasons


@dataclass
class AllocConfig:
    step_cap: float = 0.10              # max weight change per reallocation (daily build limit)
    min_weight: float = 0.0            # long-only at the allocation layer (no shorting here)
    max_weight: float = 0.20           # per-symbol cap
    max_gross: float = 1.0             # total gross-weight cap
    base_target: float = 0.10          # base target weight for a confirmed candidate
    # adaptive exposure cap E_t scales base_target by regime*alpha confidence (paper 2 Eq. 5)
    min_exposure_scale: float = 0.25
    max_exposure_scale: float = 1.0


@dataclass
class Candidate:
    symbol: str
    direction: str                     # "long" | "inverse"
    alpha_confirmed: bool
    alpha_confidence: float            # [0,1]
    regime_confidence: float           # [0,1]
    volatility: float = 0.0            # higher vol -> smaller target (optional)
    current_weight: float = 0.0


@dataclass
class Allocation:
    symbol: str
    target_weight: float
    approved_weight: float
    clamp_reasons: list
    reason: str


def adaptive_exposure_scale(regime_confidence: float, alpha_confidence: float,
                            cfg: AllocConfig) -> float:
    """E_t: scale the base target by combined confidence (paper 2: 'adaptive exposure cap determined
    by regime and eligibility state'). Bounded to [min_exposure_scale, max_exposure_scale]."""
    combined = max(0.0, min(1.0, regime_confidence)) * max(0.0, min(1.0, alpha_confidence))
    return _clip(cfg.min_exposure_scale + (cfg.max_exposure_scale - cfg.min_exposure_scale) * combined,
                 cfg.min_exposure_scale, cfg.max_exposure_scale)


def allocate(candidates: list[Candidate], cfg: Optional[AllocConfig] = None) -> list[Allocation]:
    """Produce approved target weights for ALPHA-CONFIRMED candidates only. Non-confirmed candidates
    get target 0 (i.e., they are allocated to flat; existing positions reduce toward 0 via the signed
    clamp). Enforces per-symbol cap, step cap, and a gross cap across the book."""
    cfg = cfg or AllocConfig()
    allocs: list[Allocation] = []
    # 1) raw targets
    raw: list[tuple[Candidate, float]] = []
    for c in candidates:
        if c.alpha_confirmed:
            scale = adaptive_exposure_scale(c.regime_confidence, c.alpha_confidence, cfg)
            vol_adj = 1.0 / (1.0 + max(0.0, c.volatility) * 10.0)   # mild vol de-scaling
            target = cfg.base_target * scale * vol_adj
        else:
            target = 0.0   # not confirmed -> target flat (reduce existing toward 0, never add)
        raw.append((c, min(target, cfg.max_weight)))
    # 2) gross cap: if sum of (positive) targets exceeds max_gross, scale them down proportionally
    pos_sum = sum(t for _, t in raw if t > 0)
    gross_scale = (cfg.max_gross / pos_sum) if pos_sum > cfg.max_gross else 1.0
    # 3) clamp each (signed, reduction-safe)
    for c, t in raw:
        t_scaled = t * gross_scale if t > 0 else t
        approved, reasons = clamp_weight(c.current_weight, t_scaled, cfg.step_cap,
                                         cfg.min_weight, cfg.max_weight)
        if gross_scale < 1.0 and t > 0:
            reasons.append(f"gross_cap_scale({gross_scale:.3f})")
        allocs.append(Allocation(c.symbol, round(t_scaled, 4), round(approved, 4), reasons,
                                 "alpha_confirmed" if c.alpha_confirmed else "not_confirmed_reduce"))
    return allocs
