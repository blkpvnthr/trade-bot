"""Market regime engine (V6-Integrated spec section 6; paper 2 "Opening-Session Market Regime Engine").

Classifies BULL / BEAR / TRANSITION / CRISIS / INDETERMINATE from liquid proxy instruments and runs
a lock state machine (UNLOCKED -> PRELIMINARY -> FAST_LOCK / CONFIRMED_LOCK, with DETERIORATION /
REVERSAL / unlock). The regime GATES eligibility (which universe may be entered); it never creates a
trade by itself.

Provenance: paper 2 defines only BULL/BEAR/TRANSITION + fast-lock vs time-lock + a boolean lock. The
CRISIS and INDETERMINATE regimes and the preliminary/confirmed/deterioration/reversal/unlock machine
are spec section 6 extensions beyond the paper (documented, not claimed as the paper's). Proxy set and
s_t = f(breadth, opening-range, VWAP alignment, volatility) are from the paper (Eq. 2).

Deterministic + pure. Reuses amps_v4_regime.directional_score for per-proxy scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import amps_v4_regime as V4

# regimes
BULL = "BULL"
BEAR = "BEAR"
TRANSITION = "TRANSITION"
CRISIS = "CRISIS"
INDETERMINATE = "INDETERMINATE"

# lock states
UNLOCKED = "UNLOCKED"
PRELIMINARY = "PRELIMINARY"
FAST_LOCK = "FAST_LOCK"
CONFIRMED_LOCK = "CONFIRMED_LOCK"

# paper 2 proxy set (regime detection only; never traded as signals)
PROXIES = ("SPY", "QQQ", "DIA", "IWM", "VIXY", "TQQQ", "SQQQ")
BREADTH_PROXIES = ("SPY", "QQQ", "DIA", "IWM")


@dataclass
class RegimeConfig:
    bull_thresh: float = 25.0           # composite score >= -> bull candidate
    bear_thresh: float = 25.0           # composite score <= -(this) -> bear candidate
    crisis_vol: float = 0.006           # per-bar realized vol on proxies -> CRISIS
    fast_lock_confidence: float = 0.75  # confidence to lock before the full opening window
    confirm_confidence: float = 0.55    # confidence to confirm-lock at/after window
    min_confirm_ticks: int = 2          # ticks a candidate must persist to confirm
    deteriorate_confidence: float = 0.35  # below this while locked -> DETERIORATION
    min_proxies: int = 3                # fewer scored proxies than this -> INDETERMINATE


@dataclass
class RegimeAssessment:
    regime: str
    composite_score: float
    confidence: float
    breadth: float
    volatility: float
    reason: str


def classify(proxy_rows: dict[str, dict], cfg: Optional[RegimeConfig] = None) -> RegimeAssessment:
    """Point-in-time regime from proxy feature rows (each row carries pdi/mdi/adx/vwap_dist/
    short_mom/realized_vol, as produced by the feature engine). Pure; no locking here."""
    cfg = cfg or RegimeConfig()
    vcfg = V4.RegimeConfig()
    scores = {}
    for p in PROXIES:
        row = proxy_rows.get(p)
        if not row:
            continue
        s = V4.directional_score(row, vcfg)
        if s == s:  # not NaN
            scores[p] = s
    breadth_scores = [scores[p] for p in BREADTH_PROXIES if p in scores]
    if len(scores) < cfg.min_proxies or not breadth_scores:
        return RegimeAssessment(INDETERMINATE, 0.0, 0.0, 0.0, 0.0, "insufficient_proxy_data")
    composite = sum(breadth_scores) / len(breadth_scores)
    breadth = sum(1 for s in breadth_scores if s > 0) / len(breadth_scores)  # fraction bullish
    vol = max((proxy_rows.get(p, {}).get("realized_vol", 0.0) or 0.0) for p in scores)
    # agreement-based confidence: how aligned the breadth proxies are
    signs = [1 if s > 0 else -1 for s in breadth_scores]
    agree = abs(sum(signs)) / len(signs)
    confidence = round(agree, 3)
    # CRISIS: elevated volatility dominates (defensive posture regardless of direction)
    if vol > cfg.crisis_vol:
        return RegimeAssessment(CRISIS, round(composite, 2), confidence, round(breadth, 3),
                                round(vol, 5), "volatility_above_crisis")
    if composite >= cfg.bull_thresh and confidence >= 0.5:
        reg = BULL
    elif composite <= -cfg.bear_thresh and confidence >= 0.5:
        reg = BEAR
    else:
        reg = TRANSITION
    return RegimeAssessment(reg, round(composite, 2), confidence, round(breadth, 3),
                            round(vol, 5), f"composite={composite:.1f},conf={confidence}")


def permissions(regime: str, hedge_mode: bool = False) -> dict:
    """Trading permissions per regime (spec section 6). Regime controls ELIGIBILITY only."""
    if regime == BULL:
        return {"new_entries_allowed": True, "long_universe": True,
                "inverse_universe": bool(hedge_mode), "reduce_only": False}
    if regime == BEAR:
        return {"new_entries_allowed": True, "long_universe": False,
                "inverse_universe": True, "reduce_only": False}
    if regime == TRANSITION:
        return {"new_entries_allowed": False, "long_universe": False,
                "inverse_universe": False, "reduce_only": False}
    if regime == CRISIS:
        return {"new_entries_allowed": False, "long_universe": False,
                "inverse_universe": False, "reduce_only": True}
    # INDETERMINATE
    return {"new_entries_allowed": False, "long_universe": False,
            "inverse_universe": False, "reduce_only": False}


@dataclass
class RegimeState:
    """Lock state machine. `in_opening_window=True` means we are still inside the opening window, so
    a high-confidence read may FAST_LOCK; after the window closes a persistent candidate CONFIRMS."""
    lock_status: str = UNLOCKED
    regime: str = INDETERMINATE
    lock_reason: str = ""
    candidate: str = INDETERMINATE
    candidate_ticks: int = 0
    last_confidence: float = 0.0

    def update(self, assessment: RegimeAssessment, in_opening_window: bool,
               cfg: Optional[RegimeConfig] = None) -> "RegimeState":
        cfg = cfg or RegimeConfig()
        self.last_confidence = assessment.confidence
        reg = assessment.regime
        conf = assessment.confidence

        # CRISIS always takes over immediately (defensive) and confirms.
        if reg == CRISIS:
            self.regime, self.lock_status, self.candidate = CRISIS, CONFIRMED_LOCK, CRISIS
            self.candidate_ticks = 0
            self.lock_reason = "crisis_volatility"
            return self

        # While locked: detect deterioration / reversal.
        if self.lock_status in (FAST_LOCK, CONFIRMED_LOCK):
            if reg != self.regime and reg in (BULL, BEAR) and conf >= cfg.confirm_confidence:
                # reversal to the opposite actionable regime -> drop to PRELIMINARY on the new one
                self.lock_status, self.candidate, self.candidate_ticks = PRELIMINARY, reg, 1
                self.lock_reason = f"reversal_from_{self.regime}"
                return self
            if conf < cfg.deteriorate_confidence or reg == TRANSITION:
                self.lock_status = UNLOCKED
                self.lock_reason = "deterioration"
                self.candidate, self.candidate_ticks = reg, 1
                return self
            # still aligned -> hold the lock
            self.lock_reason = "holding"
            return self

        # Not locked: accumulate a candidate.
        if reg == self.candidate:
            self.candidate_ticks += 1
        else:
            self.candidate, self.candidate_ticks = reg, 1

        if reg in (BULL, BEAR):
            if in_opening_window and conf >= cfg.fast_lock_confidence:
                self.regime, self.lock_status = reg, FAST_LOCK
                self.lock_reason = "fast_lock_high_confidence"
                return self
            if (not in_opening_window and conf >= cfg.confirm_confidence
                    and self.candidate_ticks >= cfg.min_confirm_ticks):
                self.regime, self.lock_status = reg, CONFIRMED_LOCK
                self.lock_reason = "confirmed_after_window"
                return self
            self.lock_status, self.regime = PRELIMINARY, reg
            self.lock_reason = "accumulating_confidence"
            return self
        # TRANSITION / INDETERMINATE -> stay unlocked, no new entries
        self.lock_status, self.regime = UNLOCKED, reg
        self.lock_reason = "non_actionable_regime"
        return self

    @property
    def is_locked(self) -> bool:
        return self.lock_status in (FAST_LOCK, CONFIRMED_LOCK)
