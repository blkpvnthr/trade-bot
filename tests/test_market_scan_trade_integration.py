"""Integration tests for the trade.py hot-reload seam.

Verifies the dynamic ENTRY universe wiring without constructing a real broker/
engine (object.__new__ on the scanner). Monitoring/exits are broker-driven and
not exercised here (they are universe-independent by construction).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import market_scan
import trade
from market_scan.health import STATUS_DISABLED, STATUS_HEALTHY, DynamicUniverseResult


def _scanner_stub(paper=True):
    # Mirror the REAL MomentumSetupScanner attribute layout: self.config is a
    # ScannerConfig (NO `paper`); paper/live lives on self.executor_config
    # (ExecutorConfig). A stub that puts `paper` on config would mask the bug
    # where _dynamic_entry_universe mistakenly read self.config.paper.
    s = object.__new__(trade.MomentumSetupScanner)
    s.config = SimpleNamespace(
        universe=None, use_llm_sentiment_universe=False,
        use_sp500_fallback=False, broad_market_scan=False, max_universe_symbols=500,
    )
    s.executor_config = SimpleNamespace(paper=paper)
    s.llm_metadata_by_symbol = {}
    return s


def test_disabled_blocks_entries(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_ENABLED", "0")
    assert _scanner_stub()._dynamic_entry_universe() == []


def test_exception_fails_closed_blocks_entries(monkeypatch):
    def boom(**k):
        raise RuntimeError("scanner module broke")
    monkeypatch.setattr(market_scan, "load_entry_universe", boom)
    assert _scanner_stub()._dynamic_entry_universe() == []


def test_unhealthy_blocks_entries(monkeypatch):
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: DynamicUniverseResult([], "unhealthy", "stale"))
    assert _scanner_stub()._dynamic_entry_universe() == []


def test_healthy_returns_ranked_symbols(monkeypatch):
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: DynamicUniverseResult(["NVDA", "AMD"], STATUS_HEALTHY, "ok"))
    assert _scanner_stub()._dynamic_entry_universe() == ["NVDA", "AMD"]


def test_universe_else_branch_routes_through_dynamic(monkeypatch):
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: DynamicUniverseResult(["NVDA", "TSLA"], STATUS_HEALTHY, "ok"))
    out = _scanner_stub()._universe()
    assert out == ["NVDA", "TSLA"]  # passes through is_simple_alpaca_symbol/normalize


def test_explicit_universe_override_still_wins(monkeypatch):
    # An operator --universe override must bypass the scanner entirely.
    called = {"n": 0}
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: called.__setitem__("n", called["n"] + 1) or
                        DynamicUniverseResult(["X"], STATUS_HEALTHY, "ok"))
    s = _scanner_stub()
    s.config.universe = ["SPY", "QQQ"]
    out = s._universe()
    assert out == ["SPY", "QQQ"]
    assert called["n"] == 0  # scanner path not consulted


def test_live_mode_uses_live_context(monkeypatch):
    seen = {}
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: seen.update(k) or DynamicUniverseResult([], STATUS_DISABLED, "x"))
    _scanner_stub(paper=False)._dynamic_entry_universe()
    assert seen.get("mode") == "live"


# ---- -0.5% monitor-based loss removal (exit funnel) ----
import pandas as pd  # noqa: E402


def _pos(sym, plpc, qty=10, price=50.0):
    return SimpleNamespace(symbol=sym, unrealized_plpc=plpc, qty=qty,
                           current_price=price, avg_entry_price=price)


def _exec_with_positions(positions):
    import threading as _t
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = SimpleNamespace(get_all_open_positions=lambda: positions)
    e._pending_replacements = set()
    e._pending_replacements_lock = _t.RLock()
    return e


def test_loss_removal_triggers_at_and_below_threshold(monkeypatch):
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.delenv("MARKET_SCAN_LOSS_REMOVAL", raising=False)
    e = _exec_with_positions([_pos("A", -0.006), _pos("B", -0.004),
                              _pos("C", -0.005), _pos("D", 0.03)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert set(out["symbol"]) == {"A", "C"}         # <= -0.005 only
    assert set(out["side"]) == {"sell"}


def test_loss_removal_un_overridable_full_exit(monkeypatch):
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _exec_with_positions([_pos("A", -0.01, qty=25)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert int(out.iloc[0]["delta_shares"]) == 25   # FULL position, not scaled


def test_loss_removal_dedups_against_existing_exit(monkeypatch):
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _exec_with_positions([_pos("A", -0.02)])
    existing = pd.DataFrame([{"symbol": "A", "side": "sell", "delta_shares": 5}])
    out = e._augment_exit_orders_with_loss_removal(existing)
    assert list(out["symbol"]).count("A") == 1      # not double-added


def test_loss_removal_marks_no_reentry(monkeypatch):
    marked = []
    monkeypatch.setattr(trade, "mark_no_reentry_symbol",
                        lambda s, reason="": marked.append(s))
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _exec_with_positions([_pos("A", -0.02)])
    e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert "A" in marked                            # no immediate re-entry


def test_loss_removal_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_LOSS_REMOVAL", "0")
    e = _exec_with_positions([_pos("A", -0.02)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert out.empty


def test_loss_removal_fully_exits_fractional_position(monkeypatch):
    # Regression: int() truncation would skip a sub-1-share loser entirely.
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _exec_with_positions([_pos("A", -0.02, qty=0.7)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert set(out["symbol"]) == {"A"}                       # NOT skipped
    assert abs(float(out.iloc[0]["delta_shares"]) - 0.7) < 1e-9  # FULL fractional qty


def test_loss_removal_positive_threshold_typo_does_not_liquidate(monkeypatch):
    # Regression: a positive MARKET_SCAN_LOSS_THRESHOLD must NOT liquidate winners.
    monkeypatch.setenv("MARKET_SCAN_LOSS_THRESHOLD", "0.5")
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _exec_with_positions([_pos("WIN", 0.03), _pos("SMALL", -0.002), _pos("REAL", -0.02)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert set(out["symbol"]) == {"REAL"}       # only the true -0.5%+ loser


# ---- #7 gated replacement path (verify Section-9 gates reach the order path) ----
def _repl_exec(positions=(), pending=(), paper=True, confirm=None):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = SimpleNamespace(
        get_all_open_positions=lambda: [
            SimpleNamespace(symbol=s, qty=1, market_value=50.0, current_price=50.0)
            for s in positions],
        list_open_orders=lambda: [SimpleNamespace(symbol=s) for s in pending])
    e.config = SimpleNamespace(
        paper=paper,
        scanner=SimpleNamespace(base_target_weight=0.03, enable_fractional_orders=True))
    # The engine's directional gate (DI/VWAP/structure) used for replacements.
    # Default confirms everything; pass confirm=set(...) to restrict.
    def _cbd(syms):
        syms = list(syms)
        ok = set(syms) if confirm is None else set(confirm)
        return pd.DataFrame({"symbol": syms,
                             "direction_confirmed": [s in ok for s in syms]})
    e.scanner = SimpleNamespace(confirm_buy_direction=_cbd)
    import threading as _t
    e._pending_replacements = set()
    e._pending_replacements_lock = _t.RLock()
    return e


def test_replacement_requires_confirmed_close(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _repl_exec(positions=["LOSER"])          # LOSER still OPEN (not flat)
    e._pending_replacements = {"LOSER"}
    e._read_scanner_replacement_candidates = lambda: [{"symbol": "NEW", "price": 50.0, "gates": {}}]
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert out.empty                              # no replacement until close confirmed
    assert "LOSER" in e._pending_replacements


def test_replacement_emitted_after_confirmed_close(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _repl_exec(positions=[])                  # LOSER now FLAT
    e._pending_replacements = {"LOSER"}
    e._read_scanner_replacement_candidates = lambda: [{"symbol": "NEW", "price": 50.0, "gates": {}}]
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert list(out["symbol"]) == ["NEW"] and list(out["side"]) == ["buy"]
    assert "LOSER" not in e._pending_replacements  # slot consumed (one attempt)


def test_replacement_holds_cash_when_nothing_gate_eligible(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _repl_exec(positions=[])
    e._pending_replacements = {"LOSER"}
    e._read_scanner_replacement_candidates = lambda: []   # none pass the gates
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert out.empty                              # cash, never forced
    assert "LOSER" not in e._pending_replacements


def test_replacement_excludes_held_pending_and_no_reentry(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: {"NR"})
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _repl_exec(positions=["HELD"], pending=["PEND"])
    e._pending_replacements = {"LOSER"}
    e._read_scanner_replacement_candidates = lambda: [
        {"symbol": "HELD", "price": 50.0}, {"symbol": "PEND", "price": 50.0},
        {"symbol": "NR", "price": 50.0}, {"symbol": "GOOD", "price": 50.0}]
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert list(out["symbol"]) == ["GOOD"]        # skips held/pending/no-reentry


def test_replacement_requires_engine_direction_confirmation(monkeypatch):
    # DI/BB/VWAP are enforced via the engine's confirm_buy_direction: a candidate
    # the engine does NOT confirm is skipped (cash), even if the scanner published it.
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    e = _repl_exec(positions=[], confirm=set())  # engine confirms NOTHING
    e._pending_replacements = {"LOSER"}
    e._read_scanner_replacement_candidates = lambda: [{"symbol": "NEW", "price": 50.0}]
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert out.empty                               # unconfirmed -> cash
    # and when the engine DOES confirm it, the replacement is emitted:
    e2 = _repl_exec(positions=[], confirm={"NEW"})
    e2._pending_replacements = {"LOSER"}
    e2._read_scanner_replacement_candidates = lambda: [{"symbol": "NEW", "price": 50.0}]
    out2 = e2._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert list(out2["symbol"]) == ["NEW"]


def test_read_replacement_candidates_blocked_when_scanner_unhealthy(monkeypatch):
    from market_scan.health import DynamicUniverseResult
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: DynamicUniverseResult([], "unhealthy", "stale"))
    e = _repl_exec()
    assert e._read_scanner_replacement_candidates() == []  # gates unavailable -> cash


def test_read_replacement_candidates_blocked_when_replace_disabled(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_REPLACE_ENABLED", "0")
    e = _repl_exec()
    assert e._read_scanner_replacement_candidates() == []


# ---- fast dedicated -0.5% loss-removal monitor ----
import threading as _threading  # noqa: E402


class _MonBroker:
    def __init__(self, positions=(), open_order_syms=(), is_open=True):
        self._pos = list(positions)
        self._oo = set(open_order_syms)
        self._is_open = is_open
        self.sold = []

    def get_clock_info(self):
        return {"is_open": self._is_open}

    def get_all_open_positions(self):
        return list(self._pos)

    def has_open_order_for_symbol(self, s):
        return s in self._oo

    def submit_market_sell(self, symbol, qty, cid, tif):
        self.sold.append((symbol, qty))
        return SimpleNamespace(status="accepted")


def _mon_exec(broker, enable_kill=False):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = broker
    e.config = SimpleNamespace(order_tag_prefix="qq", time_in_force="day",
                               enable_kill_switch=enable_kill)
    e._pending_replacements = set()
    e._pending_replacements_lock = _threading.RLock()
    return e


def _mon_patch(monkeypatch):
    monkeypatch.delenv("MARKET_SCAN_LOSS_REMOVAL", raising=False)
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(trade, "run_id", lambda: "rid")


def test_fast_monitor_submits_protective_exit(monkeypatch):
    _mon_patch(monkeypatch)
    b = _MonBroker(positions=[_pos("A", -0.02, qty=3), _pos("B", -0.001, qty=2)])
    e = _mon_exec(b)
    n = e._loss_removal_monitor_tick()
    assert n == 1 and b.sold == [("A", 3.0)]      # only the <= -0.5% name
    assert "A" in e._pending_replacements          # registered for replacement


def test_fast_monitor_fractional_full_exit(monkeypatch):
    _mon_patch(monkeypatch)
    b = _MonBroker(positions=[_pos("A", -0.02, qty=0.7)])
    e = _mon_exec(b)
    e._loss_removal_monitor_tick()
    assert b.sold == [("A", 0.7)]                   # full fractional qty


def test_fast_monitor_skips_when_market_closed(monkeypatch):
    _mon_patch(monkeypatch)
    b = _MonBroker(positions=[_pos("A", -0.02)], is_open=False)
    assert _mon_exec(b)._loss_removal_monitor_tick() == 0 and b.sold == []


def test_fast_monitor_dedups_when_open_order_exists(monkeypatch):
    _mon_patch(monkeypatch)
    b = _MonBroker(positions=[_pos("A", -0.02)], open_order_syms=["A"])
    assert _mon_exec(b)._loss_removal_monitor_tick() == 0 and b.sold == []


def test_fast_monitor_disabled_by_env(monkeypatch):
    _mon_patch(monkeypatch)
    monkeypatch.setenv("MARKET_SCAN_LOSS_REMOVAL", "0")
    b = _MonBroker(positions=[_pos("A", -0.02)])
    assert _mon_exec(b)._loss_removal_monitor_tick() == 0 and b.sold == []


def test_fast_monitor_honors_triggered_kill_switch(monkeypatch):
    _mon_patch(monkeypatch)
    b = _MonBroker(positions=[_pos("A", -0.02)])
    e = _mon_exec(b, enable_kill=True)
    e._read_kill_state = lambda: {"kill_triggered": True}
    assert e._loss_removal_monitor_tick() == 0 and b.sold == []  # halt wins


# ---- #8 kill-switch conflict resolution (explicit, not silent) ----
def test_positions_below_loss_threshold_identified_for_suppression_logging():
    e = _exec_with_positions([])
    positions = [_pos("A", -0.02), _pos("B", -0.004), _pos("C", -0.006), _pos("D", 0.01)]
    below = set(e._positions_below_loss_threshold(positions, -0.005))
    assert below == {"A", "C"}   # the ones the halt is suppressing


def test_loss_removal_threshold_helper_clamps_and_toggles(monkeypatch):
    e = _exec_with_positions([])
    monkeypatch.delenv("MARKET_SCAN_LOSS_REMOVAL", raising=False)
    monkeypatch.setenv("MARKET_SCAN_LOSS_THRESHOLD", "0.5")  # typo
    enabled, thr = e._loss_removal_enabled_threshold()
    assert enabled is True and thr == -0.005               # clamped
    monkeypatch.setenv("MARKET_SCAN_LOSS_REMOVAL", "0")
    assert e._loss_removal_enabled_threshold()[0] is False


def test_loss_removal_independent_of_scanner_health(monkeypatch):
    # loss exit must fire even if the scanner module/universe is broken
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("scanner down")))
    e = _exec_with_positions([_pos("A", -0.02)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert set(out["symbol"]) == {"A"}              # exit fires regardless
