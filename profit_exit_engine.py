#!/usr/bin/env python3
"""
profit_exit_engine.py -- order-flow-driven profit-taking and exit-timing.

PURPOSE
    Decide WHEN to protect, scale out of, or fully exit an open long position, using
    order-flow exhaustion/reversal signals (order_flow_engine.py) plus price/market
    structure and profit giveback. Produces interpretable recommendations:
        HOLD_STRONG | HOLD_CAUTION | PROTECT_PROFIT | PARTIAL_TAKE_PROFIT |
        FULL_EXIT | RISK_EXIT | INSUFFICIENT_DATA

SAFETY / SCOPE
    * This module NEVER places, modifies, or cancels a broker order. It returns
      recommendations and (when enforcement is on) plain exit-row dicts that the
      EXISTING execution funnel (run_once_from_frames) submits as market sells --
      so there is no competing broker path.
    * In SHADOW mode the coordinator returns NO enforcement rows at all.
    * It never supersedes or delays mandatory risk exits / the kill switch (those run
      downstream, unchanged). It defers to any symbol the protective layer is already
      exiting this cycle.
    * It does not liquidate solely because a score crosses a threshold: full exits
      require independent reversal confirmations, signal persistence, and a cooldown.

The exit-risk formula (experimental, per spec):
    exit_score = 100 * ( 0.30*sell_pressure + 0.20*cvd_deterioration
                       + 0.15*buyer_exhaustion + 0.20*structure_breakdown
                       + 0.15*profit_giveback )
All five inputs are normalized to [0,1] and individually interpretable.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from costs import TransactionCostConfig, BASELINE_COSTS, total_cost_bps
except Exception:  # pragma: no cover - costs.py always present, but stay import-safe
    @dataclass
    class TransactionCostConfig:  # type: ignore
        spread_bps: float = 10.0
        slippage_bps: float = 5.0
        commission_bps: float = 1.0
        impact_coefficient: float = 10.0

        def round_trip_cost_bps(self) -> float:
            return self.spread_bps + self.slippage_bps + self.commission_bps

    BASELINE_COSTS = TransactionCostConfig()

    def total_cost_bps(order_notional: float, adv: float, cfg: TransactionCostConfig) -> float:  # type: ignore
        return cfg.round_trip_cost_bps()


# Decision labels
HOLD_STRONG = "HOLD_STRONG"
HOLD_CAUTION = "HOLD_CAUTION"
PROTECT_PROFIT = "PROTECT_PROFIT"
PARTIAL_TAKE_PROFIT = "PARTIAL_TAKE_PROFIT"
FULL_EXIT = "FULL_EXIT"
RISK_EXIT = "RISK_EXIT"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Ordering for "most urgent wins" comparisons.
_DECISION_RANK = {
    INSUFFICIENT_DATA: -1, HOLD_STRONG: 0, HOLD_CAUTION: 1, PROTECT_PROFIT: 2,
    PARTIAL_TAKE_PROFIT: 3, FULL_EXIT: 4, RISK_EXIT: 5,
}


# ======================================================================
# Config
# ======================================================================
@dataclass
class ProfitExitConfig:
    score_protect: float = 60.0
    score_partial: float = 75.0
    score_full: float = 90.0
    partial_fraction: float = 0.50          # 25-50% configurable
    min_confirmations: int = 3
    # Adaptive ATR trailing-stop multipliers by order-flow pressure.
    atr_mult_strong_buy: float = 2.5
    atr_mult_neutral: float = 2.0
    atr_mult_selling: float = 1.25
    # Noise control.
    persistence_cycles: int = 2             # candidate band must persist N cycles before acting
    cooldown_s: float = 120.0               # min seconds between enforced actions per symbol
    # Profit gates.
    giveback_peak_trigger_pct: float = 0.03  # peak must exceed this before giveback is counted
    min_net_profit_dollars: float = 0.0      # partial/full-for-profit require net > this
    # Cost model for net-profit checks.
    cost_config: TransactionCostConfig = field(default_factory=lambda: BASELINE_COSTS)
    adv_default: float = 1.0e7


# ======================================================================
# Per-position persistent state (restart-safe)
# ======================================================================
@dataclass
class PositionExitState:
    symbol: str
    entry_price: float = 0.0
    peak_price: float = 0.0
    stop_level: float = 0.0                  # monotonic trailing stop (only tightens)
    last_band: str = HOLD_STRONG             # last score-band decision (for persistence)
    band_streak: int = 0                     # consecutive cycles in last_band
    last_action_ts: float = 0.0              # epoch of last enforced partial/full
    last_score: float = 0.0


# ======================================================================
# Recommendation record
# ======================================================================
@dataclass
class ExitRecommendation:
    symbol: str
    ts: str
    decision: str = INSUFFICIENT_DATA
    exit_score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    confirmations: list[str] = field(default_factory=list)
    confirmation_count: int = 0
    proposed_stop: float = 0.0
    stop_breached: bool = False
    fraction: float = 0.0                    # of qty to sell if acted on
    qty: float = 0.0
    net_profitable: bool = False
    net_pnl_est: float = 0.0
    gross_pnl: float = 0.0
    pressure: str = "NEUTRAL"
    cvd_divergence: float = 0.0
    buyer_exhaustion: float = 0.0
    unrealized_plpc: float = 0.0
    peak_plpc: float = 0.0
    data_status: str = INSUFFICIENT_DATA
    shadow: bool = True
    enforce: bool = False                    # would this produce a broker order (if not shadow)?
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================
# Pure scoring / stop / cost functions (unit-tested)
# ======================================================================
def exit_score(sell_pressure: float, cvd_deterioration: float, buyer_exhaustion: float,
               structure_breakdown: float, profit_giveback: float) -> float:
    """The experimental exit-risk formula. Inputs clamped to [0,1]; output [0,100]."""
    def c(x: float) -> float:
        return max(0.0, min(1.0, float(x)))
    return 100.0 * (
        0.30 * c(sell_pressure)
        + 0.20 * c(cvd_deterioration)
        + 0.15 * c(buyer_exhaustion)
        + 0.20 * c(structure_breakdown)
        + 0.15 * c(profit_giveback)
    )


def structure_breakdown_score(lower_high: bool, lower_low: bool, lost_vwap: bool,
                              failed_breakout: bool, below_fast_slow: bool) -> float:
    """[0,1] blend of bearish market-structure flags."""
    return (0.25 * bool(lower_high) + 0.25 * bool(lower_low) + 0.25 * bool(lost_vwap)
            + 0.15 * bool(failed_breakout) + 0.10 * bool(below_fast_slow))


def profit_giveback_score(entry: float, current: float, peak: float,
                          peak_trigger_pct: float = 0.03) -> float:
    """[0,1]: fraction of peak unrealized profit that has been given back.

    0 when at/above peak or when peak never exceeded the trigger; approaches 1 as the
    position retraces its entire gain back toward the entry.
    """
    if entry <= 0 or peak <= entry:
        return 0.0
    peak_gain = (peak - entry) / entry
    if peak_gain < peak_trigger_pct:
        return 0.0
    peak_profit = peak - entry
    give = peak - current
    if peak_profit <= 0:
        return 0.0
    return max(0.0, min(1.0, give / peak_profit))


def decision_from_score(score: float, cfg: ProfitExitConfig) -> str:
    if score >= cfg.score_full:
        return FULL_EXIT
    if score >= cfg.score_partial:
        return PARTIAL_TAKE_PROFIT
    if score >= cfg.score_protect:
        return PROTECT_PROFIT
    if score >= 40.0:
        return HOLD_CAUTION
    return HOLD_STRONG


def atr_multiplier_for_pressure(pressure: str, cfg: ProfitExitConfig) -> float:
    p = (pressure or "").upper()
    if p in ("STRONG_BUY", "BUY"):
        return cfg.atr_mult_strong_buy
    if p in ("SELL", "STRONG_SELL"):
        return cfg.atr_mult_selling
    return cfg.atr_mult_neutral


def adaptive_trailing_stop(peak_price: float, atr: float, multiplier: float,
                           existing_stop: float) -> float:
    """candidate_stop = peak - multiplier*ATR; new_stop = max(existing, candidate).
    Protective stops only tighten, never loosen."""
    if peak_price <= 0 or atr <= 0 or multiplier <= 0:
        return existing_stop
    candidate = peak_price - multiplier * atr
    return max(existing_stop, candidate)


def net_profit_estimate(qty: float, avg_entry: float, exit_px: float, adv: float,
                        cfg: TransactionCostConfig, spread_bps_override: Optional[float] = None
                        ) -> tuple[float, float, float]:
    """Return (net_pnl, gross_pnl, exit_cost_dollars) for selling `qty` at `exit_px`.

    gross_pnl = (exit_px - avg_entry)*qty ; exit cost uses the one-way cost model
    (spread+slippage+commission+impact), with a live spread override when available.
    """
    if qty <= 0 or exit_px <= 0:
        return 0.0, 0.0, 0.0
    gross = (exit_px - avg_entry) * qty
    notional = exit_px * qty
    cost_cfg = cfg
    if spread_bps_override is not None and spread_bps_override >= 0:
        cost_cfg = TransactionCostConfig(
            spread_bps=float(spread_bps_override),
            slippage_bps=cfg.slippage_bps, commission_bps=cfg.commission_bps,
            impact_coefficient=cfg.impact_coefficient,
        )
    cost_bps = total_cost_bps(notional, adv if adv > 0 else 1.0, cost_cfg)
    cost_dollars = notional * cost_bps / 1.0e4
    return gross - cost_dollars, gross, cost_dollars


def count_confirmations(score: float, cfg: ProfitExitConfig, sell_pressure: float,
                        cvd_deterioration: float, buyer_exhaustion: float,
                        structure_breakdown: float, cvd_divergence: float,
                        pressure: str, stop_breached: bool) -> tuple[int, list[str]]:
    """Independent reversal confirmations. Each is a distinct observable; a full exit
    requires >= cfg.min_confirmations of these, so no single noisy signal can liquidate."""
    conf: list[str] = []
    if sell_pressure >= 0.60:
        conf.append("sell_pressure")
    if cvd_deterioration >= 0.50:
        conf.append("cvd_deterioration")
    if buyer_exhaustion >= 0.50 or cvd_divergence <= -0.10:
        conf.append("buyer_exhaustion_or_divergence")
    if structure_breakdown >= 0.50:
        conf.append("structure_breakdown")
    if (pressure or "").upper() in ("SELL", "STRONG_SELL"):
        conf.append("pressure_selling")
    if stop_breached:
        conf.append("trailing_stop_breached")
    return len(conf), conf


# ======================================================================
# Stateful engine
# ======================================================================
class ProfitExitEngine:
    """Evaluates one position per call, maintaining restart-safe per-symbol state
    (peak price, trailing stop, score-band persistence, action cooldown)."""

    def __init__(self, config: Optional[ProfitExitConfig] = None, state_path: Optional[Path] = None):
        self.cfg = config or ProfitExitConfig()
        self.state_path = Path(state_path) if state_path else None
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    # ---- state persistence ----
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
            logger.warning("profit_exit state save failed: %s", exc)

    def _get_state(self, symbol: str, entry: float, current: float) -> PositionExitState:
        d = self._state.get(symbol)
        if d is None:
            st = PositionExitState(symbol=symbol, entry_price=entry,
                                   peak_price=max(entry, current), stop_level=0.0)
        else:
            st = PositionExitState(**{k: d.get(k) for k in PositionExitState.__dataclass_fields__ if k in d})
            st.symbol = symbol
        return st

    def prune(self, live_symbols: Sequence[str]) -> None:
        """Drop state for symbols no longer held (restart-safe housekeeping)."""
        live = {s.upper() for s in live_symbols}
        self._state = {s: v for s, v in self._state.items() if s in live}
        self._save()

    # ---- evaluation ----
    def evaluate(self, symbol: str, qty: float, avg_entry: float, current_price: float,
                 snapshot: Any, structure: dict[str, Any], atr: float, adv: float,
                 now: Optional[float] = None, shadow: bool = True) -> ExitRecommendation:
        symbol = symbol.upper()
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        rec = ExitRecommendation(symbol=symbol,
                                 ts=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                                 qty=qty, shadow=shadow)
        rec.data_status = getattr(snapshot, "status", INSUFFICIENT_DATA)

        st = self._get_state(symbol, avg_entry, current_price)
        st.entry_price = avg_entry or st.entry_price
        st.peak_price = max(st.peak_price or avg_entry, current_price, avg_entry)

        rec.unrealized_plpc = ((current_price - avg_entry) / avg_entry) if avg_entry > 0 else 0.0
        rec.peak_plpc = ((st.peak_price - avg_entry) / avg_entry) if avg_entry > 0 else 0.0

        # Order-flow signals (fall back to neutral if the snapshot is unusable).
        ok = bool(getattr(snapshot, "ok", False))
        sell_pressure = float(getattr(snapshot, "sell_pressure", 0.0) or 0.0)
        cvd_det = float(getattr(snapshot, "cvd_deterioration", 0.0) or 0.0)
        buyer_exh = float(getattr(snapshot, "buyer_exhaustion", 0.0) or 0.0)
        cvd_div = float(getattr(snapshot, "cvd_divergence", 0.0) or 0.0)
        pressure = str(getattr(snapshot, "pressure", "NEUTRAL"))
        spread_bps = getattr(snapshot, "spread_bps", None)
        rec.pressure = pressure
        rec.cvd_divergence = cvd_div
        rec.buyer_exhaustion = buyer_exh

        struct_score = structure_breakdown_score(
            bool(structure.get("lower_high")), bool(structure.get("lower_low")),
            bool(structure.get("lost_vwap")), bool(structure.get("failed_breakout")),
            bool(structure.get("below_fast_slow")))
        giveback = profit_giveback_score(avg_entry, current_price, st.peak_price,
                                         self.cfg.giveback_peak_trigger_pct)

        # Adaptive trailing stop (monotonic). Tracked even in shadow mode.
        mult = atr_multiplier_for_pressure(pressure, self.cfg)
        st.stop_level = adaptive_trailing_stop(st.peak_price, atr, mult, st.stop_level)
        rec.proposed_stop = st.stop_level
        stop_breached = bool(st.stop_level > 0 and current_price > 0 and current_price <= st.stop_level)
        rec.stop_breached = stop_breached

        score = exit_score(sell_pressure, cvd_det, buyer_exh, struct_score, giveback)
        rec.exit_score = round(score, 2)
        rec.components = {
            "sell_pressure": round(sell_pressure, 4), "cvd_deterioration": round(cvd_det, 4),
            "buyer_exhaustion": round(buyer_exh, 4), "structure_breakdown": round(struct_score, 4),
            "profit_giveback": round(giveback, 4),
        }

        # Net-profit estimate (for profit-taking gates).
        net_pnl, gross, _cost = net_profit_estimate(
            qty, avg_entry, current_price, adv or self.cfg.adv_default,
            self.cfg.cost_config, spread_bps_override=spread_bps)
        rec.net_pnl_est = round(net_pnl, 2)
        rec.gross_pnl = round(gross, 2)
        rec.net_profitable = net_pnl > self.cfg.min_net_profit_dollars

        n_conf, conf = count_confirmations(
            score, self.cfg, sell_pressure, cvd_det, buyer_exh, struct_score,
            cvd_div, pressure, stop_breached)
        rec.confirmations = conf
        rec.confirmation_count = n_conf

        if not ok:
            # STALE or INSUFFICIENT_DATA: never act on bad data.
            rec.decision = INSUFFICIENT_DATA
            rec.reason = f"order_flow={rec.data_status}; no action on unreliable data"
            self._commit_state(st, HOLD_STRONG, score, acted=False, now=now)
            return rec

        candidate = decision_from_score(score, self.cfg)

        # Persistence: the candidate band must hold for N cycles before a destructive act.
        band = candidate
        if band == st.last_band:
            st.band_streak = int(st.band_streak) + 1
        else:
            st.band_streak = 1
        persisted = st.band_streak >= self.cfg.persistence_cycles
        in_cooldown = (now - float(st.last_action_ts or 0.0)) < self.cfg.cooldown_s

        decision = candidate
        reason_bits: list[str] = [f"score={score:.1f} band={candidate} streak={st.band_streak}"]
        acted = False

        if candidate == FULL_EXIT:
            # Require independent reversal confirmation + persistence; else de-escalate.
            if n_conf >= self.cfg.min_confirmations and persisted and not in_cooldown:
                decision = FULL_EXIT
                acted = True
                reason_bits.append(f"confirmed exit ({n_conf} confirmations: {','.join(conf)})")
            else:
                decision = PROTECT_PROFIT
                reason_bits.append(
                    f"full-exit score but gated (confirmations={n_conf}/{self.cfg.min_confirmations}, "
                    f"persisted={persisted}, cooldown={in_cooldown}) -> protect")
        elif candidate == PARTIAL_TAKE_PROFIT:
            if rec.net_profitable and persisted and not in_cooldown and n_conf >= max(2, self.cfg.min_confirmations - 1):
                decision = PARTIAL_TAKE_PROFIT
                acted = True
                reason_bits.append(f"partial scale-out, net_pnl={net_pnl:.2f}, confirmations={n_conf}")
            elif not rec.net_profitable:
                decision = PROTECT_PROFIT
                reason_bits.append("partial score but not net profitable -> protect")
            else:
                decision = PROTECT_PROFIT
                reason_bits.append(f"partial score but gated (persisted={persisted}, cooldown={in_cooldown}) -> protect")

        # Stop breach is an independent trigger: escalate protect -> full exit when the
        # trailing stop is broken AND order flow confirms (never on the stop alone).
        if stop_breached and _DECISION_RANK.get(decision, 0) < _DECISION_RANK[FULL_EXIT]:
            if n_conf >= max(2, self.cfg.min_confirmations - 1) and not in_cooldown:
                decision = FULL_EXIT
                acted = True
                reason_bits.append("trailing-stop breach confirmed by order flow -> full exit")
            elif _DECISION_RANK.get(decision, 0) < _DECISION_RANK[PROTECT_PROFIT]:
                decision = PROTECT_PROFIT
                reason_bits.append("trailing-stop breach (unconfirmed) -> protect")

        rec.decision = decision
        rec.enforce = decision in (FULL_EXIT, PARTIAL_TAKE_PROFIT, RISK_EXIT)
        rec.fraction = (self.cfg.partial_fraction if decision == PARTIAL_TAKE_PROFIT
                        else (1.0 if decision in (FULL_EXIT, RISK_EXIT) else 0.0))
        rec.reason = "; ".join(reason_bits)

        self._commit_state(st, candidate, score, acted=acted, now=now)
        return rec

    def _commit_state(self, st: PositionExitState, band: str, score: float,
                      acted: bool, now: float) -> None:
        st.last_band = band
        st.last_score = score
        if acted:
            st.last_action_ts = now
        self._state[st.symbol] = asdict(st)
        self._save()


# ======================================================================
# Centralized coordinator (shadow by default; reuses the existing exec funnel)
# ======================================================================
# Exit-row columns expected by trade.py's run_once_from_frames / _build_order_plan.
EXIT_ROW_COLUMNS = ["symbol", "side", "delta_shares", "delta_dollars", "price", "reason",
                    "target_weight", "current_weight", "delta_weight", "trade_confidence"]


class ProfitExitCoordinator:
    """Turns per-position recommendations into (a) a recommendation list that is always
    logged and (b) enforcement exit-rows that the EXISTING funnel submits -- but only
    when NOT in shadow mode and only for symbols the protective layer is not already
    exiting (priority: risk/kill > protective stops > full > partial > protect > hold)."""

    def __init__(self, engine: ProfitExitEngine):
        self.engine = engine

    def evaluate_positions(
        self,
        positions: Sequence[Any],
        protective_symbols: set[str],
        flow_snapshots: dict[str, Any],
        structure_by_symbol: dict[str, dict[str, Any]],
        atr_by_symbol: dict[str, float],
        adv_by_symbol: dict[str, float],
        now: Optional[float] = None,
        shadow: bool = True,
        fractional_ok: bool = True,
    ) -> tuple[list[ExitRecommendation], list[dict[str, Any]]]:
        recs: list[ExitRecommendation] = []
        rows: list[dict[str, Any]] = []
        protective = {s.upper() for s in protective_symbols}
        for pos in positions:
            symbol = str(getattr(pos, "symbol", "") or "").upper()
            qty = float(getattr(pos, "qty", 0.0) or 0.0)
            if not symbol or qty <= 0:
                continue
            snap = flow_snapshots.get(symbol)
            rec = self.engine.evaluate(
                symbol=symbol, qty=qty,
                avg_entry=float(getattr(pos, "avg_entry_price", 0.0) or 0.0),
                current_price=float(getattr(pos, "current_price", 0.0) or 0.0),
                snapshot=snap,
                structure=structure_by_symbol.get(symbol, {}),
                atr=float(atr_by_symbol.get(symbol, 0.0) or 0.0),
                adv=float(adv_by_symbol.get(symbol, 0.0) or 0.0),
                now=now, shadow=shadow,
            )
            # Priority: defer to the protective layer already exiting this name.
            if symbol in protective:
                rec.enforce = False
                rec.reason = (rec.reason + "; deferred to existing protective exit").strip("; ")
            recs.append(rec)

            if shadow or not rec.enforce or symbol in protective:
                continue
            # Build an enforcement exit-row for the existing funnel.
            price = float(getattr(pos, "current_price", 0.0) or 0.0)
            mv = float(getattr(pos, "market_value", 0.0) or (price * qty))
            if rec.decision == PARTIAL_TAKE_PROFIT:
                shares = qty * rec.fraction
                if not fractional_ok:
                    shares = float(int(shares))
                if shares <= 0:
                    rec.reason += "; partial qty rounds to 0 -> skipped"
                    continue
                reason = "profit_exit_partial"
            else:  # FULL_EXIT / RISK_EXIT
                shares = qty
                reason = "profit_exit_full" if rec.decision == FULL_EXIT else "profit_exit_risk"
            frac_of_pos = shares / qty if qty > 0 else 0.0
            rows.append({
                "symbol": symbol, "side": "sell",
                "delta_shares": float(shares), "delta_dollars": float(mv * frac_of_pos),
                "price": price, "reason": reason,
                "target_weight": 0.0, "current_weight": 0.0,
                "delta_weight": 0.0, "trade_confidence": 1.0,
            })
        return recs, rows
