"""Post-merge corrective regression tests (Phase 8).

Covers the scenarios not already pinned by the scanner suite: partial sell
fills, concurrent order attempts, and insufficient buying power. All use mocked
brokers — NO real orders are ever submitted.

(Other required scenarios are already covered: loss-removal boundaries + protective
exit priority + 5s monitoring + duplicate prevention + kill-switch + stale data +
scanner disconnect in tests/test_market_scan_trade_integration.py and
tests/test_market_scan_corrective.py; covariance constraints in
tests/test_portfolio_optimizer_covariance.py; model leakage in
tests/test_model_leakage.py; dry-run mutation prevention in tests/test_dry_run_mode.py.)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import trade


def _pos(sym, plpc, qty=10.0):
    return SimpleNamespace(symbol=sym, unrealized_plpc=plpc, qty=float(qty),
                           current_price=50.0, avg_entry_price=50.0)


class _MonBroker:
    def __init__(self, positions, open_orders=(), is_open=True):
        self._pos = list(positions)
        self.open_orders = set(open_orders)
        self._is_open = is_open
        self.sold = []

    def get_clock_info(self):
        return {"is_open": self._is_open}

    def get_all_open_positions(self):
        return list(self._pos)

    def has_open_order_for_symbol(self, s):
        return s in self.open_orders

    def submit_market_sell(self, symbol, qty, cid, tif):
        self.sold.append((symbol, qty))
        self.open_orders.add(symbol)   # order now in flight
        return SimpleNamespace(status="accepted")


def _mon_exec(broker, enable_kill=False):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = broker
    e.config = SimpleNamespace(order_tag_prefix="qq", time_in_force="day",
                               enable_kill_switch=enable_kill)
    e._pending_replacements = set()
    e._pending_replacements_lock = threading.RLock()
    return e


def _patch(monkeypatch):
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(trade, "run_id", lambda: "rid")
    monkeypatch.delenv("MARKET_SCAN_LOSS_REMOVAL", raising=False)


# ---- partial sell fills ----
def test_partial_fill_in_flight_is_not_double_sold(monkeypatch):
    # A protective sell is already in flight (partial fill pending) -> the monitor
    # must NOT submit a duplicate.
    _patch(monkeypatch)
    b = _MonBroker([_pos("A", -0.02, qty=10)], open_orders=["A"])
    assert _mon_exec(b)._loss_removal_monitor_tick() == 0
    assert b.sold == []


def test_residual_after_partial_fill_is_exited(monkeypatch):
    # The in-flight order completed (no open order) leaving a smaller residual
    # still below -0.5% -> the monitor exits exactly the residual quantity.
    _patch(monkeypatch)
    b = _MonBroker([_pos("A", -0.02, qty=3.0)], open_orders=[])
    _mon_exec(b)._loss_removal_monitor_tick()
    assert b.sold == [("A", 3.0)]


# ---- concurrent order attempts ----
def test_concurrent_attempts_blocked_by_open_order_guard(monkeypatch):
    # Two back-to-back monitor passes (modeling the fast monitor racing the cycle):
    # the first submits and marks an open order; the second sees it and skips.
    _patch(monkeypatch)
    b = _MonBroker([_pos("A", -0.02, qty=10)])
    e = _mon_exec(b)
    e._loss_removal_monitor_tick()   # submits A, open order now set
    e._loss_removal_monitor_tick()   # open order present -> no duplicate
    assert b.sold == [("A", 10.0)]   # exactly once


# ---- insufficient buying power ----
def test_insufficient_buying_power_holds_cash(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = SimpleNamespace(get_all_open_positions=lambda: [],
                               list_open_orders=lambda: [])
    e.config = SimpleNamespace(
        paper=True, scanner=SimpleNamespace(base_target_weight=0.03,
                                            enable_fractional_orders=True))
    e.scanner = SimpleNamespace(
        confirm_buy_direction=lambda syms: pd.DataFrame(
            {"symbol": list(syms), "direction_confirmed": [True] * len(list(syms))}))
    e._pending_replacements = {"LOSER"}
    e._pending_replacements_lock = threading.RLock()
    e._read_scanner_replacement_candidates = lambda: [{"symbol": "NEW", "price": 50.0}]
    # Zero equity -> target dollars 0 -> no replacement BUY emitted (hold cash).
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=0.0)
    assert out.empty
    assert "LOSER" not in e._pending_replacements   # slot attempted once, cleared
