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

# lock states (lifecycle)
UNLOCKED = "UNLOCKED"
PRELIMINARY = "PRELIMINARY"
FAST_LOCK = "FAST_LOCK"
CONFIRMED_LOCK = "CONFIRMED_LOCK"
DETERIORATING = "DETERIORATING"   # locked read is weakening but not yet reversed
INVALIDATED = "INVALIDATED"       # regime evidence invalidated (e.g. coverage lost)

# per-proxy directional states
PROXY_BULLISH = "BULLISH"
PROXY_BEARISH = "BEARISH"
PROXY_NEUTRAL = "NEUTRAL"
PROXY_UNAVAILABLE = "UNAVAILABLE"

# paper 2 proxy set (regime detection only; never traded as signals)
PROXIES = ("SPY", "QQQ", "DIA", "IWM", "VIXY", "TQQQ", "SQQQ")
# Cross-market index proxies that form the breadth/aggregate regime read.
# SPY=S&P 500, QQQ=Nasdaq-100, DIA=Dow Jones, IWM=Russell 2000. Each is a
# market-CONTEXT instrument: observing it never makes it an order candidate
# (that requires explicit presence in universe.txt + a strategy signal + the
# entry-alpha gate + allocator + risk approval).
BREADTH_PROXIES = ("SPY", "QQQ", "DIA", "IWM")

# Market-context role for each proxy (diagnostics only; separate from any
# trading role a symbol may also have if it is in universe.txt).
PROXY_ROLES = {
    "SPY": "sp500_proxy",
    "QQQ": "nasdaq100_proxy",
    "DIA": "dow_proxy",
    "IWM": "russell2000_proxy",
    "VIXY": "volatility_proxy",
    "TQQQ": "nasdaq100_leveraged_proxy",
    "SQQQ": "nasdaq100_inverse_proxy",
}


@dataclass
class RegimeConfig:
    bull_thresh: float = 25.0           # composite score >= -> bull candidate
    bear_thresh: float = 25.0           # composite score <= -(this) -> bear candidate
    crisis_vol: float = 0.006           # per-bar realized vol on proxies -> CRISIS
    fast_lock_confidence: float = 0.75  # confidence to lock before the full opening window
    confirm_confidence: float = 0.55    # confidence to confirm-lock at/after window
    min_confirm_ticks: int = 2          # ticks a candidate must persist to confirm
    deteriorate_confidence: float = 0.35  # below this while locked -> DETERIORATION
    min_proxies: int = 3                # fewer scored breadth proxies than this -> INDETERMINATE
    proxy_state_thresh: float = 5.0     # |per-proxy score| >= -> BULLISH/BEARISH (else NEUTRAL)


@dataclass
class ProxyState:
    """Per-proxy market-context read (one of SPY/QQQ/DIA/IWM/...).

    This is the reusable classifier output the regime engine aggregates. It is
    CONTEXT only: a ProxyState never authorizes an order. ``data_status`` is
    "OK" or "UNAVAILABLE" -- an unavailable proxy (stale/missing/malformed/
    future-dated/duplicated data) is excluded from the aggregate and lowers
    coverage; it is never silently treated as NEUTRAL, and no other proxy is
    substituted for it.
    """
    symbol: str
    role: str
    data_status: str                    # OK | UNAVAILABLE
    state: str                          # BULLISH | BEARISH | NEUTRAL | UNAVAILABLE
    score: float                        # directional score in [-100, 100]; 0.0 if unavailable
    confidence: float                   # |score|/100 clipped to [0,1]; 0.0 if unavailable
    available: bool                     # True when data_status == OK and score is finite
    trend_state: str = "unknown"
    vwap_alignment: str = "unknown"
    opening_range_state: str = "unknown"
    sma_alignment: str = "unknown"
    daily_state: str = "unknown"
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    return_1m: float = 0.0
    return_5m: float = 0.0
    return_15m: float = 0.0
    return_30m: float = 0.0
    return_60m: float = 0.0
    timestamp: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "role": self.role,
            "data_status": self.data_status,
            "state": self.state,
            "trend_state": self.trend_state,
            "vwap_alignment": self.vwap_alignment,
            "opening_range_state": self.opening_range_state,
            "sma_alignment": self.sma_alignment,
            "adx": self.adx,
            "di_plus": self.di_plus,
            "di_minus": self.di_minus,
            "return_1m": self.return_1m,
            "return_5m": self.return_5m,
            "return_15m": self.return_15m,
            "return_30m": self.return_30m,
            "return_60m": self.return_60m,
            "daily_state": self.daily_state,
            "score": self.score,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


@dataclass
class RegimeAssessment:
    regime: str
    composite_score: float
    confidence: float
    breadth: float
    volatility: float
    reason: str
    # New (defaulted so existing positional construction keeps working):
    proxy_states: dict = field(default_factory=dict)   # {symbol: ProxyState}
    available_proxies: int = 0
    required_proxies: int = 0
    conflicting_proxies: int = 0
    coverage_ok: bool = True

    def diagnostics(self) -> dict:
        """Aggregate regime diagnostics with each proxy's contribution."""
        return {
            "regime": self.regime,
            "composite_score": self.composite_score,
            "aggregate_confidence": self.confidence,
            "breadth": self.breadth,
            "volatility": self.volatility,
            "available_proxy_count": self.available_proxies,
            "required_proxy_count": self.required_proxies,
            "conflicting_proxy_count": self.conflicting_proxies,
            "coverage_ok": self.coverage_ok,
            "reason": self.reason,
            "proxies": {sym: ps.to_dict() for sym, ps in self.proxy_states.items()},
        }


def _finite(x) -> bool:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return xf == xf and xf not in (float("inf"), float("-inf"))


def classify_proxy_state(
    symbol: str,
    features: Optional[dict],
    cfg: Optional[RegimeConfig] = None,
) -> ProxyState:
    """Reusable per-proxy market-context classifier (SPY, QQQ, DIA, IWM, ...).

    Returns a :class:`ProxyState`. CONTEXT ONLY: observing a proxy and
    classifying its state never makes it eligible for an order.

    Data health: features that are missing/empty, explicitly flagged
    (``data_status != OK`` or stale/future_dated/duplicated/invalid/malformed),
    or that cannot yield a finite directional score are marked UNAVAILABLE --
    never silently NEUTRAL -- with the reason recorded. An UNAVAILABLE proxy is
    excluded from the aggregate and lowers coverage; no other proxy is
    substituted for it.
    """
    cfg = cfg or RegimeConfig()
    role = PROXY_ROLES.get(symbol, "context_proxy")

    def _unavailable(reason: str) -> ProxyState:
        return ProxyState(symbol=symbol, role=role, data_status="UNAVAILABLE",
                          state=PROXY_UNAVAILABLE, score=0.0, confidence=0.0,
                          available=False, reason=reason,
                          timestamp=(features or {}).get("timestamp") if features else None)

    if not features:
        return _unavailable("no_data")

    declared = str(features.get("data_status", "")).upper()
    if declared and declared != "OK":
        return _unavailable(f"data_status={declared.lower()}")
    for flag, why in (("stale", "stale"), ("future_dated", "future_dated"),
                      ("duplicated", "duplicated"), ("invalid", "invalid"),
                      ("malformed", "malformed")):
        if features.get(flag):
            return _unavailable(why)

    vcfg = V4.RegimeConfig()
    score = V4.directional_score(features, vcfg)
    if not _finite(score):
        return _unavailable("non_finite_score")

    score = float(score)
    if score >= cfg.proxy_state_thresh:
        state, trend = PROXY_BULLISH, "up"
    elif score <= -cfg.proxy_state_thresh:
        state, trend = PROXY_BEARISH, "down"
    else:
        state, trend = PROXY_NEUTRAL, "flat"

    vwap_dist = float(features.get("vwap_dist", 0.0) or 0.0)
    vwap_align = "above" if vwap_dist > 0 else ("below" if vwap_dist < 0 else "at")

    return ProxyState(
        symbol=symbol, role=role, data_status="OK", state=state,
        score=round(score, 2), confidence=round(min(1.0, abs(score) / 100.0), 3),
        available=True, trend_state=trend, vwap_alignment=vwap_align,
        opening_range_state=str(features.get("opening_range_state", "unknown")),
        sma_alignment=str(features.get("sma_alignment", "unknown")),
        daily_state=str(features.get("daily_state", "unknown")),
        adx=float(features.get("adx", 0.0) or 0.0),
        di_plus=float(features.get("pdi", 0.0) or 0.0),
        di_minus=float(features.get("mdi", 0.0) or 0.0),
        return_1m=float(features.get("return_1m", features.get("short_mom", 0.0)) or 0.0),
        return_5m=float(features.get("return_5m", 0.0) or 0.0),
        return_15m=float(features.get("return_15m", 0.0) or 0.0),
        return_30m=float(features.get("return_30m", 0.0) or 0.0),
        return_60m=float(features.get("return_60m", 0.0) or 0.0),
        timestamp=features.get("timestamp"),
        reason=f"score={score:.1f}",
    )


def classify(proxy_rows: dict[str, dict], cfg: Optional[RegimeConfig] = None) -> RegimeAssessment:
    """Point-in-time regime from proxy feature rows (each row carries pdi/mdi/adx/vwap_dist/
    short_mom/realized_vol, as produced by the feature engine). Pure; no locking here.

    DIA is a first-class breadth proxy (the Dow component) alongside SPY/QQQ/IWM.
    The aggregate confidence is reduced by cross-index DISAGREEMENT and by missing
    proxy COVERAGE (e.g. an UNAVAILABLE DIA), and ``coverage_ok`` gates CONFIRMED_LOCK
    downstream. No single proxy (DIA included) determines the regime on its own, and
    no proxy ever creates an order -- the regime only gates eligibility.
    """
    cfg = cfg or RegimeConfig()

    proxy_states: dict[str, ProxyState] = {
        p: classify_proxy_state(p, proxy_rows.get(p), cfg) for p in BREADTH_PROXIES
    }
    available = {p: ps for p, ps in proxy_states.items() if ps.available}
    required = cfg.min_proxies
    breadth_scores = [ps.score for ps in available.values()]

    # Volatility across all scored proxies (incl. VIXY/TQQQ/SQQQ) drives CRISIS.
    vol = 0.0
    for p in PROXIES:
        row = proxy_rows.get(p)
        if row:
            vol = max(vol, float(row.get("realized_vol", 0.0) or 0.0))

    coverage_ok = len(available) >= required

    if not breadth_scores or not coverage_ok:
        return RegimeAssessment(
            INDETERMINATE, 0.0, 0.0, 0.0, round(vol, 5), "insufficient_proxy_coverage",
            proxy_states=proxy_states, available_proxies=len(available),
            required_proxies=required, conflicting_proxies=0, coverage_ok=False,
        )

    composite = sum(breadth_scores) / len(breadth_scores)
    breadth = sum(1 for s in breadth_scores if s > 0) / len(breadth_scores)  # fraction bullish

    signs = [1 if s > 0 else -1 for s in breadth_scores]
    agree = abs(sum(signs)) / len(signs)              # 1.0 = unanimous, 0.0 = evenly split
    coverage_fraction = len(available) / len(BREADTH_PROXIES)
    confidence = round(agree * coverage_fraction, 3)   # disagreement AND missing coverage both lower it

    n_bull = sum(1 for s in signs if s > 0)
    conflicting = min(n_bull, len(signs) - n_bull)     # size of the dissenting minority

    reason = (f"composite={composite:.1f},conf={confidence},"
              f"cov={len(available)}/{len(BREADTH_PROXIES)},conflicting={conflicting}")

    # CRISIS: elevated volatility dominates (defensive posture regardless of direction)
    if vol > cfg.crisis_vol:
        return RegimeAssessment(
            CRISIS, round(composite, 2), confidence, round(breadth, 3), round(vol, 5),
            "volatility_above_crisis;" + reason,
            proxy_states=proxy_states, available_proxies=len(available),
            required_proxies=required, conflicting_proxies=conflicting, coverage_ok=coverage_ok,
        )
    if composite >= cfg.bull_thresh and confidence >= 0.5:
        reg = BULL
    elif composite <= -cfg.bear_thresh and confidence >= 0.5:
        reg = BEAR
    else:
        reg = TRANSITION
    return RegimeAssessment(
        reg, round(composite, 2), confidence, round(breadth, 3), round(vol, 5), reason,
        proxy_states=proxy_states, available_proxies=len(available),
        required_proxies=required, conflicting_proxies=conflicting, coverage_ok=coverage_ok,
    )


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

        coverage_ok = getattr(assessment, "coverage_ok", True)

        # While locked: detect coverage loss (INVALIDATED) / deterioration / reversal.
        if self.lock_status in (FAST_LOCK, CONFIRMED_LOCK):
            if not coverage_ok or reg == INDETERMINATE:
                # Proxy coverage lost (e.g. DIA went UNAVAILABLE): invalidate the
                # regime evidence rather than silently holding or calling it neutral.
                # Protective exits/reduce-only remain available upstream; this only
                # blocks NEW entries by dropping out of a locked state.
                self.lock_status = INVALIDATED
                self.lock_reason = "proxy_coverage_lost"
                self.candidate, self.candidate_ticks = reg, 1
                return self
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
            if not coverage_ok:
                # Insufficient proxy coverage (e.g. DIA UNAVAILABLE) must never
                # promote to a lock; hold at PRELIMINARY until coverage returns.
                self.lock_status, self.regime = PRELIMINARY, reg
                self.lock_reason = "insufficient_proxy_coverage"
                return self
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
