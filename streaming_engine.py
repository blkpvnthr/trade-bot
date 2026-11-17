from __future__ import annotations
from enum import Enum
from collections import defaultdict, deque


class OrderState(Enum):
    SIGNAL_CONFIRMED = "SIGNAL_CONFIRMED"
    RISK_APPROVED = "RISK_APPROVED"
    ORDER_SUBMITTING = "ORDER_SUBMITTING"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    BROKER_ACKNOWLEDGED = "BROKER_ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


_ALLOWED = {
    None: {OrderState.SIGNAL_CONFIRMED},
    OrderState.SIGNAL_CONFIRMED: {OrderState.RISK_APPROVED, OrderState.REJECTED},
    OrderState.RISK_APPROVED: {OrderState.ORDER_SUBMITTING, OrderState.REJECTED},
    OrderState.ORDER_SUBMITTING: {OrderState.ORDER_SUBMITTED, OrderState.REJECTED},
    OrderState.ORDER_SUBMITTED: {OrderState.BROKER_ACKNOWLEDGED, OrderState.REJECTED, OrderState.CANCELED},
    OrderState.BROKER_ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED},
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
}


class OrderStateMachine:
    def __init__(self) -> None:
        self._state: dict[str, OrderState] = {}

    def state(self, cid: str) -> OrderState | None:
        return self._state.get(cid)

    def advance(self, cid: str, to_state: OrderState) -> bool:
        cur = self._state.get(cid)
        if to_state in _ALLOWED.get(cur, set()):
            self._state[cid] = to_state
            return True
        return False


def percentile(values, q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


class LatencyTracker:
    def __init__(self, maxlen: int = 1000) -> None:
        self._samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def record(self, stage: str, value: float) -> None:
        self._samples[stage].append(float(value))

    def summary(self) -> dict:
        out = {}
        for stage, xs in self._samples.items():
            out[stage] = {
                "n": len(xs),
                "p50": percentile(xs, 50),
                "p95": percentile(xs, 95),
                "p99": percentile(xs, 99),
            }
        return out
