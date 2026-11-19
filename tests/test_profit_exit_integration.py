"""Integration tests for the profit-exit overlay inside trade.py's ProductionOrderExecutor.

Verifies the safety-critical properties: SHADOW mode submits/produces nothing, enforcement
produces exit ROWS only (never a direct broker call), the overlay defers to the existing
protective layer, and the kill switch / existing exit path remain untouched.
"""
import pandas as pd
import pytest

import trade
import profit_exit_engine as pe


class _Pos:
    def __init__(self, symbol, qty, avg_entry, current):
        self.symbol = symbol; self.qty = qty
        self.avg_entry_price = avg_entry; self.current_price = current
        self.market_value = qty * current
        self.unrealized_pl = (current - avg_entry) * qty
        self.unrealized_plpc = (current / avg_entry - 1.0) if avg_entry else 0.0


class _Broker:
    """Broker stub with NO order-submission methods -- if the overlay tried to submit an
    order directly (a competing broker path), the test would AttributeError."""
    def __init__(self, positions):
        self._positions = positions

    def get_all_open_positions(self):
        return self._positions

    def get_account_snapshot(self):
        return trade.AccountSnapshot(buying_power=1e5, equity=1e5, cash=1e5, portfolio_value=1e5)


class _FakeFlow:
    """Order-flow engine stub returning a strong-sell snapshot for every symbol."""
    def __init__(self, snap):
        self._snap = snap

    def poll_rest(self, symbols, lookback_s=300):
        return {s.upper(): self._snap for s in symbols}


class _Scanner:
    """Scanner stub exposing only what the overlay reuses."""
    data = None

    def _download_intraday_bars(self, symbols, minutes):
        rows = []
        for s in symbols:
            for i in range(20):
                rows.append({"symbol": s, "timestamp": pd.Timestamp("2026-11-18 15:00", tz="UTC"),
                             "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.2,
                             "volume": 10000, "atr14": 0.1})
        return pd.DataFrame(rows)

    def _intraday_direction_metrics(self, symbol, frame, minutes, avg_entry_price):
        return {f"{minutes}m_lower_highs": True, f"{minutes}m_lower_lows": True,
                f"{minutes}m_above_vwap": False, f"{minutes}m_failed_reclaim": True,
                f"{minutes}m_below_fast_sma": True, f"{minutes}m_below_slow_sma": True}


def _make_executor(positions, shadow, snap):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.scanner.profit_exit_enabled = True
    e.config.scanner.profit_exit_shadow_mode = shadow
    e.config.scanner.order_flow_enabled = True
    e.config.scanner.enable_fractional_orders = True
    e.scanner = _Scanner()
    e.broker = _Broker(positions)
    # Pre-inject engines so _ensure_profit_exit_engines short-circuits and we control order flow.
    e._order_flow_engine = _FakeFlow(snap)
    e._profit_exit_engine = pe.ProfitExitEngine(
        pe.ProfitExitConfig(persistence_cycles=1, min_confirmations=3, cooldown_s=0.0),
        state_path=None)
    e._profit_exit_coordinator = pe.ProfitExitCoordinator(e._profit_exit_engine)
    return e


def _strong_sell_snap():
    class S:
        ok = True; status = "OK"; sell_pressure = 0.9; cvd_deterioration = 0.8
        buyer_exhaustion = 0.8; cvd_divergence = -0.4; pressure = "STRONG_SELL"; spread_bps = 10.0
    return S()


def test_shadow_mode_produces_no_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    e = _make_executor([_Pos("AAA", 100, 10.0, 10.3)], shadow=True, snap=_strong_sell_snap())
    rows = e._run_profit_exit_overlay(equity=1e5, protective_symbols=set())
    assert rows.empty  # SHADOW: no broker-bound rows regardless of decision


def test_enforce_mode_produces_exit_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    e = _make_executor([_Pos("AAA", 100, 10.0, 10.3)], shadow=False, snap=_strong_sell_snap())
    rows = e._run_profit_exit_overlay(equity=1e5, protective_symbols=set())
    assert not rows.empty
    assert set(rows["side"]) == {"sell"}
    assert rows.iloc[0]["reason"].startswith("profit_exit_")


def test_enforce_defers_to_protective_layer(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    e = _make_executor([_Pos("AAA", 100, 10.0, 10.3)], shadow=False, snap=_strong_sell_snap())
    # AAA already being exited by the protective layer -> overlay must add nothing for it.
    rows = e._run_profit_exit_overlay(equity=1e5, protective_symbols={"AAA"})
    assert rows.empty


def test_overlay_no_positions_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    e = _make_executor([], shadow=False, snap=_strong_sell_snap())
    rows = e._run_profit_exit_overlay(equity=1e5, protective_symbols=set())
    assert rows.empty


def test_overlay_insufficient_flow_no_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    class _Stale:
        ok = False; status = "INSUFFICIENT_DATA"; sell_pressure = 0.0; cvd_deterioration = 0.0
        buyer_exhaustion = 0.0; cvd_divergence = 0.0; pressure = "NEUTRAL"; spread_bps = 10.0
    e = _make_executor([_Pos("AAA", 100, 10.0, 10.3)], shadow=False, snap=_Stale())
    rows = e._run_profit_exit_overlay(equity=1e5, protective_symbols=set())
    assert rows.empty  # never act on unreliable order-flow data
