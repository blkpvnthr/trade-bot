"""Paper-only order execution with hard risk controls.

This layer NEVER calls a broker. It is a deterministic forward-validation recorder: it accepts or
rejects intended orders against risk caps + the sleeve gate, simulates the resulting book, and
appends every decision (accepted OR blocked, with reasons) to an audit log. Live is impossible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .core import RiskLimits, Sleeve


@dataclass(frozen=True)
class Order:
    sleeve: str
    symbol: str
    side: str          # 'buy' | 'sell'
    notional: float    # absolute dollar notional, > 0
    ts: str            # caller-supplied PIT/event timestamp (keeps behavior deterministic)
    live: bool = False  # if a caller ever sets True -> HARD BLOCK


@dataclass
class Decision:
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    order: Optional[Order] = None


class PaperExecutor:
    """Deterministic paper executor. One instance ~ one paper account.

    Behavior depends only on its inputs and accumulated state (no wall clock, no RNG), so a given
    sequence of submit()/record_pnl() calls always yields the same decisions and audit log.
    """

    # Class-level invariant: this executor has no live capability whatsoever.
    LIVE_SUPPORTED = False

    def __init__(self, equity: float, risk_limits: RiskLimits, audit_path: str | Path):
        if equity <= 0:
            raise ValueError("equity must be > 0")
        self.equity = float(equity)
        self.limits = risk_limits
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.killed = False
        self.kill_reason = ""
        self.day_pnl = 0.0
        # symbol -> signed notional currently open (buy positive, sell negative)
        self.positions: Dict[str, float] = {}
        self.audit_records: List[dict] = []

    # ---- risk state helpers -------------------------------------------------
    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    def open_count(self) -> int:
        return sum(1 for v in self.positions.values() if abs(v) > 1e-9)

    def trip_kill_switch(self, reason: str = "manual") -> None:
        self.killed = True
        self.kill_reason = reason

    def reset_day(self) -> None:
        self.day_pnl = 0.0

    def record_pnl(self, amount: float) -> None:
        """Update day PnL; auto-trip the kill switch if the daily loss cap is breached."""
        self.day_pnl += float(amount)
        if self.day_pnl <= -abs(self.limits.max_daily_loss):
            self.trip_kill_switch("daily_loss_limit")

    # ---- the gate-and-risk check (pure given state) -------------------------
    def _check(self, sleeve: Sleeve, order: Order) -> List[str]:
        reasons: List[str] = []

        # 1. Live is ALWAYS blocked — first and unconditional.
        if order.live or self.LIVE_SUPPORTED or sleeve.live_enabled:
            reasons.append("live_blocked")
            return reasons  # never evaluate anything else for a live request

        # 2. Sleeve must be validated AND paper-enabled.
        if not sleeve.can_trade():
            reasons.append("sleeve_not_tradeable:" + (sleeve.why_not_tradeable() or "blocked"))

        # 3. Kill switch.
        if self.killed:
            reasons.append("kill_switch:" + (self.kill_reason or "tripped"))

        # 4. Daily loss already breached (defensive; record_pnl also trips the kill switch).
        if self.day_pnl <= -abs(self.limits.max_daily_loss):
            reasons.append("daily_loss_limit")

        # 5. Order sanity + per-order cap.
        if order.notional <= 0:
            reasons.append("non_positive_notional")
        if order.notional > self.limits.per_order_notional:
            reasons.append("per_order_notional_exceeded")

        # 6. Per-symbol position cap (projected absolute position vs equity).
        signed = order.notional if order.side == "buy" else -order.notional
        projected = abs(self.positions.get(order.symbol, 0.0) + signed)
        if projected > self.limits.max_position_pct * self.equity:
            reasons.append("max_position_pct_exceeded")

        # 7. Total deployable notional cap.
        if self.gross_exposure() + order.notional > self.limits.max_notional:
            reasons.append("max_notional_exceeded")

        # 8. Gross exposure cap (fraction of equity).
        if self.gross_exposure() + order.notional > self.limits.max_gross_exposure * self.equity:
            reasons.append("max_gross_exposure_exceeded")

        # 9. Max concurrent positions (only if this opens a NEW symbol).
        is_new = abs(self.positions.get(order.symbol, 0.0)) <= 1e-9
        if is_new and self.open_count() >= self.limits.max_concurrent:
            reasons.append("max_concurrent_exceeded")

        return reasons

    # ---- public API ---------------------------------------------------------
    def submit(self, sleeve: Sleeve, order: Order) -> Decision:
        reasons = self._check(sleeve, order)
        accepted = len(reasons) == 0
        if accepted:
            signed = order.notional if order.side == "buy" else -order.notional
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + signed
        self._audit(order, accepted, reasons)
        return Decision(accepted=accepted, reasons=reasons, order=order)

    def _audit(self, order: Order, accepted: bool, reasons: List[str]) -> None:
        rec = {
            "ts": order.ts,
            "sleeve": order.sleeve,
            "symbol": order.symbol,
            "side": order.side,
            "notional": order.notional,
            "live_requested": bool(order.live),
            "accepted": accepted,
            "reasons": reasons,
            "gross_after": round(self.gross_exposure(), 6),
            "open_after": self.open_count(),
            "day_pnl": round(self.day_pnl, 6),
            "killed": self.killed,
            "mode": "paper",  # never anything else
        }
        self.audit_records.append(rec)
        with self.audit_path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
