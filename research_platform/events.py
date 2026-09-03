"""Append-only decision ledger. Every decision is an immutable event with a PIT feature snapshot
and an explicit realized-vs-modeled fill tag (champion=realized, challenger=modeled counterfactual).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator


def make_event_id(experiment: str, portfolio: str, symbol: str, ts: str, action: str) -> str:
    blob = f"{experiment}|{portfolio}|{symbol}|{ts}|{action}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class DecisionEvent:
    experiment: str                 # spec_hash linking to the registry
    portfolio: str                  # which config made/considered this
    ts: str                         # decision time (point-in-time)
    symbol: str
    action: str                     # enter | exit | skip | hold
    features: Dict[str, Any]        # PIT snapshot — the inputs to the decision
    decision: Dict[str, Any]        # weight, conviction, reason
    fill: Dict[str, Any]            # {tag: realized|modeled, price, slippage, commission}
    outcome: Dict[str, Any]         # realized/modeled net return when known
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = make_event_id(self.experiment, self.portfolio, self.symbol,
                                          self.ts, self.action)


class EventLog:
    """JSONL append-only log."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, ev: DecisionEvent) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(ev), default=str) + "\n")

    def append_many(self, evs) -> None:
        with self.path.open("a") as f:
            for ev in evs:
                f.write(json.dumps(asdict(ev), default=str) + "\n")

    def read(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)

    def count(self) -> int:
        return sum(1 for _ in self.read())
