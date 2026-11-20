"""Integration tests for the buy-opportunity overlay inside trade.py + the DMI/ADX and
bullish-structure helpers. Verifies shadow isolation, enforcement gating of the scanner's
own candidates, coordination with pending exits, and that no new symbols are ever added."""
import numpy as np
import pandas as pd

import trade
import buy_opportunity_engine as bo


# ------------------------------------------------ structure helpers
def _uptrend_frame(n=60, start=100.0, step=0.1):
    rows = []
    ts = pd.Timestamp("2026-11-18 14:30", tz="UTC")
    px = start
    for i in range(n):
        o = px
        px = px + step
        rows.append({"symbol": "AAA", "timestamp": ts + pd.Timedelta(minutes=5 * i),
                     "open": o, "high": px + 0.05, "low": o - 0.05, "close": px, "volume": 10000})
    return pd.DataFrame(rows)


def test_dmi_adx_detects_uptrend():
    dip, dim, adx, rising = trade.dmi_adx(_uptrend_frame(), 14)
    assert dip > dim and dip > 0          # DI+ dominant in a clean uptrend
    assert adx > 0


def test_dmi_adx_insufficient_bars_neutral():
    small = _uptrend_frame(n=5)
    assert trade.dmi_adx(small, 14) == (0.0, 0.0, 0.0, False)


def test_compute_buy_structure_uptrend():
    frame = _uptrend_frame()
    scn = object.__new__(trade.MomentumSetupScanner)
    m = scn._intraday_direction_metrics("AAA", frame, 5, 0.0)
    st = trade.compute_buy_structure(frame, m)
    assert st["above_sma50"] and st["di_plus_over_minus"]
    # empty frame -> all False (never bullish by default)
    empty = trade.compute_buy_structure(pd.DataFrame(), {})
    assert not any(empty.values())


# ------------------------------------------------ overlay
class _Pos:
    def __init__(self, symbol, qty, avg_entry, current):
        self.symbol = symbol; self.qty = qty
        self.avg_entry_price = avg_entry; self.current_price = current
        self.market_value = qty * current


class _Broker:
    def __init__(self, positions):
        self._positions = positions

    def get_all_open_positions(self):
        return self._positions

    def get_account_snapshot(self):
        return trade.AccountSnapshot(buying_power=1e5, equity=1e5, cash=1e5, portfolio_value=1e5)

    def list_open_orders(self):
        return []


class _FakeFlow:
    def __init__(self, snap):
        self._snap = snap

    def poll_universe(self, priority_groups, lookback_s=300, max_symbols=None):
        syms = []
        for g in priority_groups:
            for s in g:
                if s and s not in syms:
                    syms.append(s)
        return {s: self._snap for s in syms[: (max_symbols or 40)]}


class _Scanner:
    data = None

    def _universe(self):
        return ["AAA", "BBB", "CCC"]

    def _download_intraday_bars(self, symbols, minutes):
        return _uptrend_frame()

    def _intraday_direction_metrics(self, symbol, frame, minutes, avg_entry_price):
        return object.__new__(trade.MomentumSetupScanner)._intraday_direction_metrics(
            symbol, frame, minutes, avg_entry_price)


def _strong_buy_snap():
    class S:
        ok = True; status = "OK"; buy_pressure = 0.95; cvd_improvement = 0.9; seller_exhaustion = 0.8
        cvd_divergence = 0.4; executed_imbalance = 0.3; delta_10s = 60.0; spread_bps = 5.0
        liquidity = "tight"; pressure = "STRONG_BUY"; cvd_trend = 0.9
        buy_vol_60s = 2000.0; sell_vol_60s = 200.0
    return S()


def _make_executor(shadow, positions=None):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.scanner.buy_opportunity_enabled = True
    e.config.scanner.buy_opportunity_shadow_mode = shadow
    e.config.scanner.order_flow_enabled = True
    e.config.scanner.buy_persistence_cycles = 1
    e.scanner = _Scanner()
    e.broker = _Broker(positions or [])
    e._order_flow_engine = _FakeFlow(_strong_buy_snap())
    e._buy_opportunity_engine = bo.BuyOpportunityEngine(
        bo.BuyOpportunityConfig(persistence_cycles=1, min_confirmations=3), state_path=None)
    e._buy_opportunity_coordinator = bo.BuyOpportunityCoordinator(e._buy_opportunity_engine)
    return e


def _rebalance(symbols):
    return pd.DataFrame([{"symbol": s, "side": "buy", "delta_shares": 10, "price": 100.0} for s in symbols])


def test_shadow_mode_leaves_rebalance_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    monkeypatch.setattr(trade, "BUY_REENTRY_STATE_PATH", tmp_path / "re.json")
    e = _make_executor(shadow=True)
    rb = _rebalance(["AAA", "BBB"])
    out = e._run_buy_opportunity_overlay(rb, pd.DataFrame(), equity=1e5)
    assert list(out["symbol"]) == ["AAA", "BBB"]   # SHADOW: no change to orders


def test_enforce_gates_to_confirmed_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    monkeypatch.setattr(trade, "BUY_REENTRY_STATE_PATH", tmp_path / "re.json")
    e = _make_executor(shadow=False)
    rb = _rebalance(["AAA", "BBB"])
    out = e._run_buy_opportunity_overlay(rb, pd.DataFrame(), equity=1e5)
    # confirmed candidates kept; never adds symbols beyond the scanner's own candidates
    assert set(out["symbol"]).issubset({"AAA", "BBB"})
    assert not out.empty


def test_enforce_defers_to_pending_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    monkeypatch.setattr(trade, "BUY_REENTRY_STATE_PATH", tmp_path / "re.json")
    e = _make_executor(shadow=False)
    rb = _rebalance(["AAA"])
    exits = pd.DataFrame([{"symbol": "AAA", "side": "sell", "reason": "profit_exit_full"}])
    out = e._run_buy_opportunity_overlay(rb, exits, equity=1e5)
    assert "AAA" not in set(out["symbol"])   # no BUY while a FULL_EXIT is pending for AAA


def test_enforce_blocks_held_symbol_no_pyramid(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    monkeypatch.setattr(trade, "BUY_REENTRY_STATE_PATH", tmp_path / "re.json")
    e = _make_executor(shadow=False, positions=[_Pos("AAA", 10, 100.0, 101.0)])
    rb = _rebalance(["AAA", "BBB"])
    out = e._run_buy_opportunity_overlay(rb, pd.DataFrame(), equity=1e5)
    assert "AAA" not in set(out["symbol"])   # already held -> pyramiding restriction
