"""Canonical position manager (V6-Integrated spec section 15; paper 2 position manager).

The single source of truth for live position state. Every component reads positions from here. After
each (partial or full) fill the manager updates quantity/cost basis, recomputes exposure, and the
pipeline then (re)establishes proportionate protection and audits the transition.

Pure/in-memory + JSON-serializable. No broker calls; the gateway feeds fills in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_entry: float = 0.0
    filled_cost: float = 0.0
    realized_pnl: float = 0.0
    session_open_price: float = 0.0
    high_since_entry: float = 0.0
    low_since_entry: float = 0.0
    regime_at_entry: str = ""
    entry_alpha_evidence_id: Optional[str] = None
    protective_order_state: str = "NONE"       # NONE|PENDING|ACTIVE|MISSING
    last_fill_price: float = 0.0
    last_fill_qty: float = 0.0
    pending_qty: float = 0.0
    effective_leverage: float = 1.0            # signed daily leverage of the instrument

    # --- derived helpers (paper 2) ---
    def market_value(self, price: float) -> float:
        return self.qty * price

    def effective_exposure(self, price: float) -> float:
        return self.qty * price * self.effective_leverage

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_entry) * self.qty if self.qty else 0.0

    def return_pct(self, price: float) -> float:
        return (price / self.avg_entry - 1.0) if self.avg_entry else 0.0

    def drawdown_from_high(self, price: float) -> float:
        peak = max(self.high_since_entry, price) if self.high_since_entry else price
        return (price / peak - 1.0) if peak else 0.0

    def run_up_from_entry(self) -> float:
        return (self.high_since_entry / self.avg_entry - 1.0) if self.avg_entry and self.high_since_entry else 0.0

    def adverse_from_entry(self) -> float:
        return (self.low_since_entry / self.avg_entry - 1.0) if self.avg_entry and self.low_since_entry else 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class PositionManager:
    def __init__(self, positions: Optional[dict] = None):
        self._pos: dict[str, Position] = positions or {}

    def get(self, symbol: str) -> Optional[Position]:
        return self._pos.get(symbol)

    def all(self) -> list[Position]:
        return [p for p in self._pos.values() if abs(p.qty) > 1e-12]

    def apply_fill(self, symbol: str, fill_qty: float, fill_price: float, side: str,
                   regime: str = "", evidence_id: Optional[str] = None,
                   effective_leverage: float = 1.0) -> Position:
        """Apply a (partial) fill to the canonical position. Handles buys (add / average) and sells
        (reduce / realize). Returns the updated Position."""
        p = self._pos.get(symbol)
        if p is None:
            p = Position(symbol=symbol, regime_at_entry=regime, entry_alpha_evidence_id=evidence_id,
                         effective_leverage=effective_leverage)
            self._pos[symbol] = p
        signed = fill_qty if side.lower() == "buy" else -fill_qty
        if side.lower() == "buy":
            new_qty = p.qty + fill_qty
            # weighted-average entry on increases
            p.avg_entry = ((p.avg_entry * p.qty) + (fill_price * fill_qty)) / new_qty if new_qty else fill_price
            p.filled_cost += fill_price * fill_qty
            p.qty = new_qty
            if evidence_id and not p.entry_alpha_evidence_id:
                p.entry_alpha_evidence_id = evidence_id
            if regime and not p.regime_at_entry:
                p.regime_at_entry = regime
        else:
            # reduce; realize P&L on the reduced quantity against avg_entry
            reduce_qty = min(fill_qty, p.qty) if p.qty > 0 else fill_qty
            p.realized_pnl += (fill_price - p.avg_entry) * reduce_qty
            p.qty -= reduce_qty
            if p.qty <= 1e-12:
                p.qty = 0.0
        p.last_fill_price = fill_price
        p.last_fill_qty = signed
        p.high_since_entry = max(p.high_since_entry or fill_price, fill_price)
        p.low_since_entry = min(p.low_since_entry or fill_price, fill_price)
        return p

    def mark(self, symbol: str, price: float) -> None:
        """Update high/low watermarks from a live price (for trailing/drawdown helpers)."""
        p = self._pos.get(symbol)
        if p is None or p.qty == 0:
            return
        p.high_since_entry = max(p.high_since_entry or price, price)
        p.low_since_entry = min(p.low_since_entry or price, price) if p.low_since_entry else price

    def gross_exposure(self, prices: dict) -> float:
        return sum(abs(p.market_value(prices.get(p.symbol, p.avg_entry))) for p in self.all())

    def effective_exposure(self, prices: dict) -> float:
        return sum(abs(p.effective_exposure(prices.get(p.symbol, p.avg_entry))) for p in self.all())

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or (Path(__file__).resolve().parents[2] / "logs" / "execution" / "amps_v6_positions.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({s: v.as_dict() for s, v in self._pos.items()}, indent=2))
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PositionManager":
        p = path or (Path(__file__).resolve().parents[2] / "logs" / "execution" / "amps_v6_positions.json")
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls({s: Position(**v) for s, v in raw.items()})
