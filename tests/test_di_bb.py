"""Tests for the DI/Bollinger technical signal + trade.py overlay.

ENTRY: DI+ > DI- AND close > BB-middle(20). EXIT: DI- > DI+ AND close < BB-middle(20),
on a completed candle. Verifies the rule, insufficient-data handling, shadow isolation,
and enforcement gating (entries filtered, exits added) through the existing funnel."""
import numpy as np
import pandas as pd

import trade


# ------------------------------------------------ pure signals
def test_entry_exit_truth_table():
    assert trade.di_bb_entry_signal(25, 15, 101, 100) is True
    assert trade.di_bb_entry_signal(15, 25, 101, 100) is False   # DI- dominant
    assert trade.di_bb_entry_signal(25, 15, 99, 100) is False    # below BB mid
    assert trade.di_bb_exit_signal(15, 25, 99, 100) is True
    assert trade.di_bb_exit_signal(25, 15, 99, 100) is False     # DI+ dominant
    assert trade.di_bb_exit_signal(15, 25, 101, 100) is False    # above BB mid


def test_bollinger_middle():
    c = pd.Series([float(i) for i in range(1, 21)])  # 1..20
    assert trade.bollinger_middle(c, 20) == 10.5
    assert trade.bollinger_middle(pd.Series([1.0, 2.0]), 20) != trade.bollinger_middle(pd.Series([1.0, 2.0]), 20) \
        or np.isnan(trade.bollinger_middle(pd.Series([1.0, 2.0]), 20))  # too few -> NaN


def _frame(closes):
    ts = pd.Timestamp("2026-11-18 14:30", tz="UTC")
    rows = []
    for i, c in enumerate(closes):
        rows.append({"symbol": "AAA", "timestamp": ts + pd.Timedelta(minutes=5 * i),
                     "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1000})
    return pd.DataFrame(rows)


def test_signal_for_frame_uptrend_entry():
    frame = _frame([100 + i * 0.2 for i in range(40)])  # steady uptrend
    sig = trade.di_bb_signal_for_frame(frame, 14, 20)
    assert sig and sig["plus_di"] > sig["minus_di"] and sig["close"] > sig["bb_mid"]
    assert sig["entry"] is True and sig["exit"] is False


def test_signal_for_frame_downtrend_exit():
    frame = _frame([100 - i * 0.2 for i in range(40)])  # steady downtrend
    sig = trade.di_bb_signal_for_frame(frame, 14, 20)
    assert sig and sig["minus_di"] > sig["plus_di"] and sig["close"] < sig["bb_mid"]
    assert sig["exit"] is True and sig["entry"] is False


def test_signal_for_frame_insufficient_bars():
    assert trade.di_bb_signal_for_frame(_frame([100, 101, 102]), 14, 20) == {}


# ------------------------------------------------ overlay
class _Pos:
    def __init__(self, symbol, qty, current):
        self.symbol = symbol; self.qty = qty
        self.current_price = current; self.market_value = qty * current
        self.avg_entry_price = current


class _Broker:
    def __init__(self, positions):
        self._positions = positions

    def get_all_open_positions(self):
        return self._positions


class _Scanner:
    def __init__(self, sig_by_symbol):
        self._sigs = sig_by_symbol

    def _download_intraday_bars(self, symbols, minutes):
        rows = []
        ts = pd.Timestamp("2026-11-18 14:30", tz="UTC")
        for s in symbols:
            closes = self._sigs.get(s, [100.0] * 40)
            for i, c in enumerate(closes):
                rows.append({"symbol": s, "timestamp": ts + pd.Timedelta(minutes=5 * i),
                             "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1000})
        return pd.DataFrame(rows)


def _up(n=40):
    return [100 + i * 0.2 for i in range(n)]


def _down(n=40):
    return [100 - i * 0.2 for i in range(n)]


def _make_exec(shadow, positions, sigs):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.scanner.di_bb_enabled = True
    e.config.scanner.di_bb_shadow_mode = shadow
    e.scanner = _Scanner(sigs)
    e.broker = _Broker(positions)
    return e


def _rebalance(symbols):
    return pd.DataFrame([{"symbol": s, "side": "buy", "delta_shares": 10, "price": 100.0} for s in symbols])


def test_overlay_shadow_leaves_orders_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    e = _make_exec(True, positions=[], sigs={"AAA": _up(), "BBB": _down()})
    rb = _rebalance(["AAA", "BBB"])
    out_rb, out_ex = e._run_di_bb_overlay(rb, pd.DataFrame(), equity=1e5)
    assert list(out_rb["symbol"]) == ["AAA", "BBB"] and out_ex is None  # SHADOW: no change


def test_overlay_enforce_gates_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    # AAA shows a DI/BB entry (uptrend); BBB does not (downtrend)
    e = _make_exec(False, positions=[], sigs={"AAA": _up(), "BBB": _down()})
    rb = _rebalance(["AAA", "BBB"])
    out_rb, out_ex = e._run_di_bb_overlay(rb, pd.DataFrame(), equity=1e5)
    assert list(out_rb["symbol"]) == ["AAA"]  # only the DI/BB-entry name kept


def test_overlay_enforce_adds_exit_for_held_exit_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    # hold CCC which shows a DI/BB EXIT (downtrend)
    e = _make_exec(False, positions=[_Pos("CCC", 10, 95.0)], sigs={"CCC": _down()})
    out_rb, out_ex = e._run_di_bb_overlay(pd.DataFrame(), pd.DataFrame(), equity=1e5)
    assert out_ex is not None and list(out_ex["symbol"]) == ["CCC"]
    assert out_ex.iloc[0]["reason"] == "di_bb_exit" and out_ex.iloc[0]["side"] == "sell"


def test_overlay_enforce_defers_to_pending_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(trade, "OUT_DIR", tmp_path)
    e = _make_exec(False, positions=[_Pos("CCC", 10, 95.0)], sigs={"CCC": _down()})
    exits = pd.DataFrame([{"symbol": "CCC", "side": "sell", "reason": "scanner_stop_loss"}])
    out_rb, out_ex = e._run_di_bb_overlay(pd.DataFrame(), exits, equity=1e5)
    assert out_ex is None  # already exiting CCC -> no duplicate DI/BB exit
