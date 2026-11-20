#!/usr/bin/env python3
"""
buy_opportunity_engine.py -- order-flow-driven BUY opportunity + entry timing.

The inverse of profit_exit_engine.py. It detects the transition from seller dominance to
buyer dominance -- seller exhaustion, aggressive-buying accumulation, positive CVD
divergence, bullish market-structure reversals, VWAP reclamation, breakout confirmation,
favorable pullbacks, and improving execution conditions -- and times LONG entries.

SAFETY / SCOPE (identical discipline to the exit engine)
    * This module NEVER places an order. It returns recommendations; the EXISTING scanner
      decides which symbols are tradeable and the EXISTING funnel submits them. In
      enforcement mode the coordinator only CONFIRMS/holds back the scanner's own buy
      candidates -- it never invents new symbols the strategy rejected.
    * SHADOW by default -> no effect on orders.
    * It never overrides the kill switch, portfolio caps, risk limits, pyramiding rules,
      direction gates, or scanner eligibility. A high score cannot authorize an order on
      its own: BUY_READY+ requires scanner agreement + independent confirmations +
      persistence + cooldown, and it defers to any pending risk exit.
    * Missing/stale order flow is NEVER treated as bullish confirmation.

Buy-opportunity score (experimental, per spec):
    buy_score = 100 * ( 0.30*buy_pressure + 0.20*cvd_improvement + 0.15*seller_exhaustion
                      + 0.20*bullish_structure + 0.15*entry_quality )
All components normalized to [0,1].
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Decision labels
NO_BUY = "NO_BUY"
WATCH = "WATCH"
BUY_SETUP = "BUY_SETUP"
BUY_READY = "BUY_READY"
STRONG_BUY = "STRONG_BUY_CONFIRMATION"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

_DECISION_RANK = {INSUFFICIENT_DATA: -1, NO_BUY: 0, WATCH: 1, BUY_SETUP: 2,
                  BUY_READY: 3, STRONG_BUY: 4}

# Entry setup types
SETUP_REVERSAL = "REVERSAL"       # seller-exhaustion reversal
SETUP_PULLBACK = "PULLBACK"       # bullish pullback continuation
SETUP_BREAKOUT = "BREAKOUT"       # breakout confirmation
SETUP_NONE = "NONE"


# ======================================================================
# Config
# ======================================================================
@dataclass
class BuyOpportunityConfig:
    score_watch: float = 40.0
    score_setup: float = 60.0
    score_ready: float = 75.0
    score_strong: float = 90.0
    min_confirmations: int = 3
    max_spread_bps: float = 15.0
    reentry_cooldown_s: float = 300.0
    require_existing_entry_signal: bool = True
    require_valid_order_flow: bool = True
    persistence_cycles: int = 2
    cooldown_s: float = 120.0
    over_extension_pct: float = 0.03   # > this far above the short MA => chasing; penalize


# ======================================================================
# Per-symbol persistent state (restart-safe)
# ======================================================================
@dataclass
class BuyState:
    symbol: str
    last_band: str = NO_BUY
    band_streak: int = 0
    last_action_ts: float = 0.0
    last_score: float = 0.0


# ======================================================================
# Recommendation record
# ======================================================================
@dataclass
class BuyRecommendation:
    symbol: str
    ts: str
    decision: str = INSUFFICIENT_DATA
    buy_score: float = 0.0
    setup: str = SETUP_NONE
    components: dict[str, float] = field(default_factory=dict)
    confirmations: list[str] = field(default_factory=list)
    confirmation_count: int = 0
    pressure: str = "NEUTRAL"
    cvd_trend: float = 0.0
    seller_exhaustion: float = 0.0
    buy_vol_60s: float = 0.0
    sell_vol_60s: float = 0.0
    spread_bps: Optional[float] = None
    scanner_eligible: bool = False
    data_status: str = INSUFFICIENT_DATA
    shadow: bool = True
    enforce: bool = False     # would this confirm a scanner buy (if not shadow)?
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================
# Pure scoring / classification (unit-tested)
# ======================================================================
def buy_score(buy_pressure: float, cvd_improvement: float, seller_exhaustion: float,
              bullish_structure: float, entry_quality: float) -> float:
    """Experimental buy-opportunity score. Inputs clamped to [0,1]; output [0,100]."""
    def c(x: float) -> float:
        return max(0.0, min(1.0, float(x)))
    return 100.0 * (
        0.30 * c(buy_pressure)
        + 0.20 * c(cvd_improvement)
        + 0.15 * c(seller_exhaustion)
        + 0.20 * c(bullish_structure)
        + 0.15 * c(entry_quality)
    )


def bullish_structure_score(higher_low: bool, reclaimed_vwap: bool, reclaimed_sma50: bool,
                            di_plus_over_minus: bool, adx_rising: bool,
                            failed_breakdown_reversal: bool, breakout_confirmed: bool) -> float:
    """[0,1] blend of bullish market-structure flags (weights sum to 1)."""
    return (0.18 * bool(higher_low) + 0.18 * bool(reclaimed_vwap) + 0.15 * bool(reclaimed_sma50)
            + 0.15 * bool(di_plus_over_minus) + 0.10 * bool(adx_rising)
            + 0.12 * bool(failed_breakdown_reversal) + 0.12 * bool(breakout_confirmed))


def entry_quality_score(spread_bps: Optional[float], max_spread_bps: float, liquidity: str,
                        over_extended: bool) -> float:
    """[0,1]: good when spread is tight, liquidity healthy, and price is NOT over-extended.
    Missing spread -> 0 entirely (unknown execution data is never a bullish freebie)."""
    if spread_bps is None:
        return 0.0
    if spread_bps <= max_spread_bps * 0.5:
        spread_q = 1.0
    elif spread_bps >= max_spread_bps:
        spread_q = 0.0
    else:
        spread_q = 1.0 - (spread_bps - max_spread_bps * 0.5) / (max_spread_bps * 0.5)
    liq_q = {"tight": 1.0, "normal": 0.6, "wide": 0.1}.get(liquidity, 0.3)
    q = 0.5 * spread_q + 0.5 * liq_q
    if over_extended:
        q *= 0.4          # chasing an extended price is a poor entry
    return max(0.0, min(1.0, q))


def decision_from_buy_score(score: float, cfg: BuyOpportunityConfig) -> str:
    if score >= cfg.score_strong:
        return STRONG_BUY
    if score >= cfg.score_ready:
        return BUY_READY
    if score >= cfg.score_setup:
        return BUY_SETUP
    if score >= cfg.score_watch:
        return WATCH
    return NO_BUY


def classify_entry_setup(cvd_divergence: float, seller_exhaustion: float, structure: dict) -> str:
    """Classify the dominant setup. Breakout/pullback/reversal are checked in that order of
    specificity; NONE when no bullish setup is present."""
    if structure.get("breakout_confirmed"):
        return SETUP_BREAKOUT
    if structure.get("above_sma50") and structure.get("pulled_back") and structure.get("reclaimed_vwap"):
        return SETUP_PULLBACK
    if (cvd_divergence > 0.05 or seller_exhaustion >= 0.4) and (
            structure.get("higher_low") or structure.get("reclaimed_vwap")
            or structure.get("failed_breakdown_reversal")):
        return SETUP_REVERSAL
    return SETUP_NONE


def count_buy_confirmations(cfg: BuyOpportunityConfig, buy_pressure: float, cvd_improvement: float,
                            seller_exhaustion: float, cvd_divergence: float, executed_imbalance: float,
                            delta_10s: float, spread_bps: Optional[float], structure: dict
                            ) -> tuple[int, list[str]]:
    """Independent bullish confirmations (the 10 seller-exhaustion / buyer-control combos).
    Each is a distinct observable; BUY_READY+ needs >= cfg.min_confirmations."""
    conf: list[str] = []
    if cvd_divergence > 0.05:
        conf.append("bullish_cvd_divergence")           # price low, CVD higher low
    if seller_exhaustion >= 0.40:
        conf.append("selling_declining")                # aggressive selling fading
    if executed_imbalance >= -0.10:
        conf.append("imbalance_neutralizing")           # negative imbalance -> neutral
    if buy_pressure >= 0.50 or delta_10s > 0:
        conf.append("buy_volume_increasing")            # aggressive buying building
    if structure.get("price_stops_lower_low"):
        conf.append("no_new_lower_low")                 # price stops making lower lows
    if structure.get("reclaimed_vwap"):
        conf.append("vwap_or_support_reclaim")
    if structure.get("di_plus_over_minus") and structure.get("adx_rising"):
        conf.append("di_cross_adx_rising")              # DI+>DI- with strengthening ADX
    if structure.get("reclaimed_sma50"):
        conf.append("holds_sma50")
    if structure.get("failed_breakdown_reversal"):
        conf.append("failed_breakdown_reversal")
    if spread_bps is not None and spread_bps <= cfg.max_spread_bps:
        conf.append("liquidity_improved")
    return len(conf), conf


# ======================================================================
# Stateful engine
# ======================================================================
class BuyOpportunityEngine:
    """Evaluates one symbol per call with restart-safe per-symbol persistence (score-band
    streak + action cooldown)."""

    def __init__(self, config: Optional[BuyOpportunityConfig] = None, state_path: Optional[Path] = None):
        self.cfg = config or BuyOpportunityConfig()
        self.state_path = Path(state_path) if state_path else None
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.state_path and self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text()) or {}
            except Exception:
                self._state = {}

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, default=str))
        except Exception as exc:  # pragma: no cover
            logger.warning("buy_opportunity state save failed: %s", exc)

    def _get_state(self, symbol: str) -> BuyState:
        d = self._state.get(symbol)
        if d is None:
            return BuyState(symbol=symbol)
        st = BuyState(**{k: d.get(k) for k in BuyState.__dataclass_fields__ if k in d})
        st.symbol = symbol
        return st

    def prune(self, live_symbols: Sequence[str]) -> None:
        live = {s.upper() for s in live_symbols}
        self._state = {s: v for s, v in self._state.items() if s in live}
        self._save()

    def evaluate(self, symbol: str, snapshot: Any, structure: dict, scanner_eligible: bool,
                 now: Optional[float] = None, shadow: bool = True) -> BuyRecommendation:
        symbol = symbol.upper()
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        rec = BuyRecommendation(symbol=symbol,
                                ts=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                                scanner_eligible=scanner_eligible, shadow=shadow)
        rec.data_status = getattr(snapshot, "status", INSUFFICIENT_DATA)

        ok = bool(getattr(snapshot, "ok", False))
        buy_pressure = float(getattr(snapshot, "buy_pressure", 0.0) or 0.0)
        cvd_improvement = float(getattr(snapshot, "cvd_improvement", 0.0) or 0.0)
        seller_exh = float(getattr(snapshot, "seller_exhaustion", 0.0) or 0.0)
        cvd_div = float(getattr(snapshot, "cvd_divergence", 0.0) or 0.0)
        executed_imb = float(getattr(snapshot, "executed_imbalance", 0.0) or 0.0)
        delta_10s = float(getattr(snapshot, "delta_10s", 0.0) or 0.0)
        spread_bps = getattr(snapshot, "spread_bps", None)
        liquidity = str(getattr(snapshot, "liquidity", "unknown"))
        rec.pressure = str(getattr(snapshot, "pressure", "NEUTRAL"))
        rec.cvd_trend = float(getattr(snapshot, "cvd_trend", 0.0) or 0.0)
        rec.seller_exhaustion = seller_exh
        rec.buy_vol_60s = float(getattr(snapshot, "buy_vol_60s", 0.0) or 0.0)
        rec.sell_vol_60s = float(getattr(snapshot, "sell_vol_60s", 0.0) or 0.0)
        rec.spread_bps = spread_bps

        struct_score = bullish_structure_score(
            bool(structure.get("higher_low")), bool(structure.get("reclaimed_vwap")),
            bool(structure.get("reclaimed_sma50")), bool(structure.get("di_plus_over_minus")),
            bool(structure.get("adx_rising")), bool(structure.get("failed_breakdown_reversal")),
            bool(structure.get("breakout_confirmed")))
        eq = entry_quality_score(spread_bps, self.cfg.max_spread_bps, liquidity,
                                 bool(structure.get("over_extended")))
        score = buy_score(buy_pressure, cvd_improvement, seller_exh, struct_score, eq)
        rec.buy_score = round(score, 2)
        rec.components = {
            "buy_pressure": round(buy_pressure, 4), "cvd_improvement": round(cvd_improvement, 4),
            "seller_exhaustion": round(seller_exh, 4), "bullish_structure": round(struct_score, 4),
            "entry_quality": round(eq, 4),
        }
        rec.setup = classify_entry_setup(cvd_div, seller_exh, structure)
        n_conf, conf = count_buy_confirmations(self.cfg, buy_pressure, cvd_improvement, seller_exh,
                                               cvd_div, executed_imb, delta_10s, spread_bps, structure)
        rec.confirmations = conf
        rec.confirmation_count = n_conf

        st = self._get_state(symbol)
        if self.cfg.require_valid_order_flow and not ok:
            rec.decision = INSUFFICIENT_DATA
            rec.reason = f"order_flow={rec.data_status}; no entry on unreliable data"
            st.band_streak = 0
            self._commit(st, NO_BUY, score, acted=False, now=now)
            return rec

        candidate = decision_from_buy_score(score, self.cfg)

        if candidate == st.last_band:
            st.band_streak = int(st.band_streak) + 1
        else:
            st.band_streak = 1
        persisted = st.band_streak >= self.cfg.persistence_cycles
        in_cooldown = (now - float(st.last_action_ts or 0.0)) < self.cfg.cooldown_s

        decision = candidate
        acted = False
        bits = [f"score={score:.1f} band={candidate} streak={st.band_streak} setup={rec.setup}"]

        # BUY_READY / STRONG require independent confirmations + persistence + scanner
        # agreement + acceptable spread + no over-extension. A high score alone never acts.
        if _DECISION_RANK[candidate] >= _DECISION_RANK[BUY_READY]:
            spread_ok = (spread_bps is not None and spread_bps <= self.cfg.max_spread_bps)
            not_extended = not bool(structure.get("over_extended"))
            eligible_ok = scanner_eligible or (not self.cfg.require_existing_entry_signal)
            gates = {
                "confirmations": n_conf >= self.cfg.min_confirmations,
                "persisted": persisted, "not_cooldown": not in_cooldown,
                "scanner_eligible": eligible_ok, "spread_ok": spread_ok,
                "not_over_extended": not_extended, "has_setup": rec.setup != SETUP_NONE,
            }
            if all(gates.values()):
                acted = True
                bits.append(f"CONFIRMED ({n_conf} confirmations: {','.join(conf)})")
            else:
                failed = [k for k, v in gates.items() if not v]
                decision = BUY_SETUP      # de-escalate to a watch-grade setup, do not authorize
                bits.append(f"ready score but gated -> setup (failed: {','.join(failed)})")

        rec.decision = decision
        rec.enforce = acted and _DECISION_RANK[decision] >= _DECISION_RANK[BUY_READY]
        rec.reason = "; ".join(bits)
        self._commit(st, candidate, score, acted=acted, now=now)
        return rec

    def _commit(self, st: BuyState, band: str, score: float, acted: bool, now: float) -> None:
        """Persist the SAME state object (keeps the band_streak accumulated in evaluate)."""
        st.last_band = band
        st.last_score = score
        if acted:
            st.last_action_ts = now
        self._state[st.symbol] = asdict(st)
        self._save()


# ======================================================================
# Centralized coordinator (shadow by default; confirms scanner buys only)
# ======================================================================
class BuyOpportunityCoordinator:
    """Evaluates monitored symbols and returns (a) recommendations (always logged) and
    (b) the set of scanner buy-candidates that are order-flow CONFIRMED for entry -- used
    (only when NOT shadow) to time/gate the scanner's existing buys. Enforces coordination
    with the exit side and all mandatory gates."""

    def __init__(self, engine: BuyOpportunityEngine):
        self.engine = engine

    def evaluate(self, monitored: dict[str, Any], structure_by_symbol: dict[str, dict],
                 scanner_candidates: set, scanner_eligible: set, held: set,
                 pending_exit: set, reentry_block: set, now: Optional[float] = None,
                 shadow: bool = True) -> tuple[list[BuyRecommendation], set]:
        recs: list[BuyRecommendation] = []
        confirmed: set = set()
        cand = {s.upper() for s in scanner_candidates}
        elig = {s.upper() for s in scanner_eligible}
        held_u = {s.upper() for s in held}
        pend = {s.upper() for s in pending_exit}
        reentry = {s.upper() for s in reentry_block}
        for sym, snap in monitored.items():
            s = sym.upper()
            rec = self.engine.evaluate(s, snap, structure_by_symbol.get(s, {}),
                                       scanner_eligible=(s in elig), now=now, shadow=shadow)
            # ---- coordination rules (never override risk; defer to the exit side) ----
            block = ""
            if s in pend:
                block = "pending_risk_or_full_exit"       # no contradictory BUY vs exit
            elif s in held_u:
                block = "already_held_no_pyramid"         # preserve pyramiding restrictions
            elif s in reentry:
                block = "reentry_cooldown"                 # just exited; needs fresh setup
            elif self.engine.cfg.require_existing_entry_signal and s not in cand:
                block = "not_scanner_candidate"            # scanner did not flag -> monitor only
            if block:
                rec.enforce = False
                rec.reason = (rec.reason + f"; blocked:{block}").strip("; ")
            recs.append(rec)
            if (not shadow) and rec.enforce and s in cand and not block:
                confirmed.add(s)
        return recs, confirmed
