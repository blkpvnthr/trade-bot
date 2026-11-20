"""Persistent order state machine (V6-Integrated spec section 14).

Covers the full lifecycle the spec enumerates, with explicit legal transitions, unique client-order-id
idempotency, and an ambiguity guard: an order in SUBMITTING whose outcome is unknown goes to
RECONCILIATION_REQUIRED and MUST NOT be resubmitted until reconciled against the broker.

Pure/in-memory + JSON-serializable (restart recovery). No broker calls here -- the execution gateway
drives the broker and advances states through this machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSM_PATH = PROJECT_ROOT / "logs" / "execution" / "amps_v6_order_states.json"


class OrderState(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PENDING_NEW = "PENDING_NEW"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELED = "CANCELED"
    PENDING_REPLACE = "PENDING_REPLACE"
    REPLACED = "REPLACED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    DONE_FOR_DAY = "DONE_FOR_DAY"
    SUSPENDED = "SUSPENDED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    TRADE_CORRECTED = "TRADE_CORRECTED"
    TRADE_BUSTED = "TRADE_BUSTED"


S = OrderState
# Legal forward transitions. Terminal states have empty sets. RECONCILIATION_REQUIRED is reachable
# from any pre-terminal state (ambiguity) and can resolve to the real broker state.
TRANSITIONS: dict[OrderState, set[OrderState]] = {
    S.CREATED: {S.VALIDATED, S.REJECTED},
    S.VALIDATED: {S.RISK_APPROVED, S.REJECTED},
    S.RISK_APPROVED: {S.SUBMITTING, S.REJECTED},
    S.SUBMITTING: {S.SUBMITTED, S.REJECTED, S.RECONCILIATION_REQUIRED},
    S.SUBMITTED: {S.ACCEPTED, S.PENDING_NEW, S.REJECTED, S.RECONCILIATION_REQUIRED},
    S.ACCEPTED: {S.PENDING_NEW, S.NEW, S.REJECTED},
    S.PENDING_NEW: {S.NEW, S.REJECTED, S.CANCELED},
    S.NEW: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.PENDING_REPLACE, S.EXPIRED,
            S.DONE_FOR_DAY, S.REJECTED, S.SUSPENDED},
    S.PARTIALLY_FILLED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.PENDING_REPLACE,
                         S.CANCELED, S.EXPIRED, S.DONE_FOR_DAY, S.TRADE_CORRECTED, S.TRADE_BUSTED},
    S.FILLED: {S.TRADE_CORRECTED, S.TRADE_BUSTED},
    S.PENDING_CANCEL: {S.CANCELED, S.PARTIALLY_FILLED, S.FILLED, S.RECONCILIATION_REQUIRED},
    S.PENDING_REPLACE: {S.REPLACED, S.PARTIALLY_FILLED, S.FILLED, S.REJECTED, S.RECONCILIATION_REQUIRED},
    S.REPLACED: {S.NEW, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED},
    S.CANCELED: set(),
    S.EXPIRED: set(),
    S.REJECTED: set(),
    S.DONE_FOR_DAY: {S.RECONCILIATION_REQUIRED},
    S.SUSPENDED: {S.NEW, S.CANCELED, S.RECONCILIATION_REQUIRED},
    S.RECONCILIATION_REQUIRED: {S.NEW, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.REJECTED,
                                S.EXPIRED, S.REPLACED},
    S.TRADE_CORRECTED: set(),
    S.TRADE_BUSTED: set(),
}

TERMINAL = {s for s, nxt in TRANSITIONS.items() if not nxt}


@dataclass
class OrderRecord:
    client_order_id: str
    symbol: str
    side: str
    qty: float
    state: str = OrderState.CREATED.value
    broker_order_id: Optional[str] = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    history: list = field(default_factory=list)   # [(from, to), ...]

    def as_dict(self) -> dict:
        return asdict(self)


class OrderStateMachine:
    def __init__(self, records: Optional[dict] = None):
        self._rec: dict[str, OrderRecord] = records or {}

    def create(self, client_order_id: str, symbol: str, side: str, qty: float) -> OrderRecord:
        if client_order_id in self._rec:
            # idempotency: never create a duplicate for the same client_order_id
            return self._rec[client_order_id]
        rec = OrderRecord(client_order_id, symbol, side, float(qty))
        self._rec[client_order_id] = rec
        return rec

    def get(self, cid: str) -> Optional[OrderRecord]:
        return self._rec.get(cid)

    def can_advance(self, cid: str, to: OrderState) -> bool:
        rec = self._rec.get(cid)
        if rec is None:
            return False
        return to in TRANSITIONS.get(OrderState(rec.state), set())

    def advance(self, cid: str, to: OrderState, **kw) -> bool:
        rec = self._rec.get(cid)
        if rec is None or not self.can_advance(cid, to):
            return False
        rec.history.append((rec.state, to.value))
        rec.state = to.value
        if "broker_order_id" in kw and kw["broker_order_id"]:
            rec.broker_order_id = kw["broker_order_id"]
        if "filled_qty" in kw:
            rec.filled_qty = float(kw["filled_qty"])
        if "avg_fill_price" in kw:
            rec.avg_fill_price = float(kw["avg_fill_price"])
        return True

    def needs_reconciliation(self) -> list[str]:
        return [c for c, r in self._rec.items() if r.state == OrderState.RECONCILIATION_REQUIRED.value]

    def resubmittable(self, cid: str) -> bool:
        """An order may only be (re)submitted from RISK_APPROVED and when NOT ambiguous. An order in
        SUBMITTING/RECONCILIATION_REQUIRED must never be blindly resubmitted (spec section 14)."""
        rec = self._rec.get(cid)
        return bool(rec and rec.state == OrderState.RISK_APPROVED.value)

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or OSM_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({c: r.as_dict() for c, r in self._rec.items()}, indent=2))
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "OrderStateMachine":
        p = path or OSM_PATH
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls({c: OrderRecord(**r) for c, r in raw.items()})
