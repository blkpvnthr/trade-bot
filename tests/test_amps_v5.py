"""Tests for amps_v5_investigation.py -- parameterized exit sim (bps + ATR targets/stops,
conservative intrabar), exit attribution, and metrics determinism."""
import numpy as np
import pandas as pd
from datetime import time

import amps_v5_investigation as V5


def _frame(seq):
    """seq: list of (open,high,low,close). All RTH mid-session, 5-min spaced. ATR col filled."""
    ts = pd.Timestamp("2026-11-18 15:00", tz="UTC")  # 10:00 ET, mid-session
    rows = []
    for i, (o, h, l, c) in enumerate(seq):
        rows.append({"open": o, "high": h, "low": l, "close": c,
                     "et_time": time(10, 0), "date": pd.Timestamp("2026-11-18").date(),
                     "atr": 1.0})
    return pd.DataFrame(rows)


def test_bps_target_hit():
    # entry at bar1 open=100; target +30bps=100.3 reached at bar2 high
    d = _frame([(100, 100, 100, 100), (100, 100.0, 100.0, 100.0), (100, 100.5, 99.99, 100.4)])
    mask = np.array([True, False, False])
    tr = V5.simulate_exits(d, mask, "X", cost_bps=0.0, target_mode="bps", target_val=30,
                           stop_mode="bps", stop_val=10, max_hold_bars=5)
    assert len(tr) == 1 and tr[0]["reason"] == "TARGET"
    assert tr[0]["gross_bps"] > 29  # ~ +30bps minus slippage already in rt(=0 here) -> target price


def test_bps_stop_hit_conservative():
    # price gaps down through the stop on bar2 -> fills at the (lower) open, conservatively
    d = _frame([(100, 100, 100, 100), (99.8, 99.85, 99.7, 99.75), (99.8, 99.9, 99.7, 99.8)])
    mask = np.array([True, False, False])
    tr = V5.simulate_exits(d, mask, "X", 0.0, "bps", 30, "bps", 10, max_hold_bars=5)
    assert len(tr) == 1 and tr[0]["reason"] == "STOP"
    assert tr[0]["net_bps"] < 0


def test_atr_target_and_stop():
    # ATR=1.0; entry 100; target 2xATR=102, stop 0.5xATR=99.5
    up = _frame([(100, 100, 100, 100), (100, 101, 100, 100.5), (100, 102.1, 100, 102.0)])
    tr = V5.simulate_exits(up, np.array([True, False, False]), "X", 0.0, "atr", 2.0, "atr", 0.5, 5)
    assert tr and tr[0]["reason"] == "TARGET" and tr[0]["gross_bps"] > 190  # ~+2% = 200bps
    down = _frame([(100, 100, 100, 100), (100, 100.2, 99.4, 99.5), (100, 100, 99, 99.3)])
    tr2 = V5.simulate_exits(down, np.array([True, False, False]), "X", 0.0, "atr", 2.0, "atr", 0.5, 5)
    assert tr2 and tr2[0]["reason"] == "STOP"


def test_max_hold_exit():
    d = _frame([(100, 100, 100, 100)] + [(100, 100.05, 99.95, 100.0)] * 5)
    tr = V5.simulate_exits(d, np.array([True] + [False] * 5), "X", 0.0, "bps", 500, "bps", 500, max_hold_bars=2)
    assert tr and tr[0]["reason"] == "MAX_HOLD"


def test_exit_attribution_and_metrics():
    trades = [
        {"symbol": "X", "session": 1, "net_bps": 20, "gross_bps": 23, "reason": "TARGET", "hold_min": 10, "mfe_bps": 25, "mae_bps": -3},
        {"symbol": "X", "session": 1, "net_bps": -15, "gross_bps": -12, "reason": "STOP", "hold_min": 5, "mfe_bps": 2, "mae_bps": -15},
        {"symbol": "X", "session": 2, "net_bps": -5, "gross_bps": -2, "reason": "MAX_HOLD", "hold_min": 30, "mfe_bps": 10, "mae_bps": -8},
    ]
    att = V5.exit_attribution(trades)
    assert att["TARGET"]["trades"] == 1 and att["TARGET"]["win_rate"] == 1.0
    assert att["STOP"]["net_exp_bps"] == -15
    m = V5.trade_metrics(trades)
    assert m["trades"] == 3 and 0 <= m["win_rate"] <= 1
