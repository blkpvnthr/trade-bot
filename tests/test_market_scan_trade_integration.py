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
    s = object.__new__(trade.MomentumSetupScanner)
    s.config = SimpleNamespace(
        paper=paper, universe=None, use_llm_sentiment_universe=False,
        use_sp500_fallback=False, broad_market_scan=False, max_universe_symbols=500,
    )
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
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = SimpleNamespace(get_all_open_positions=lambda: positions)
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


def test_loss_removal_independent_of_scanner_health(monkeypatch):
    # loss exit must fire even if the scanner module/universe is broken
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(market_scan, "load_entry_universe",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("scanner down")))
    e = _exec_with_positions([_pos("A", -0.02)])
    out = e._augment_exit_orders_with_loss_removal(pd.DataFrame())
    assert set(out["symbol"]) == {"A"}              # exit fires regardless
