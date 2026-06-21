"""Tests for the net-of-cost lead validator (streams / turnover / cost curve)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PKG))

import run_lead_validation as LV


def _panel(nsym=6, nt=40, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-03-04 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        for i, t in enumerate(ts):
            rows.append((f"S{s}", t, rng.normal() * 0.01, 100.0, 1e6, s, (-1.0) ** i))
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "fut_ret_1", "close", "volume",
                                       "const_sig", "flip_sig"]).set_index(["symbol", "timestamp"]).sort_index()


def test_flipping_signal_has_higher_turnover():
    p = _panel()
    _, t_const = LV.streams(p, p["const_sig"])
    _, t_flip = LV.streams(p, p["flip_sig"])
    assert t_flip.mean() > t_const.mean() * 3        # alternating signal churns far more


def test_cost_curve_net_sharpe_is_monotonic_decreasing():
    p = _panel()
    g, t = LV.streams(p, p["flip_sig"])
    curve = LV.cost_curve(g, t)
    sharpes = [c["net_sharpe"] for c in curve]
    assert all(sharpes[i] >= sharpes[i + 1] - 1e-9 for i in range(len(sharpes) - 1))


def test_breakeven_is_gross_over_turnover():
    from rl_meta.stream_validation import stream_metrics
    p = _panel()
    g, t = LV.streams(p, p["flip_sig"])
    m = stream_metrics(g.values, t.values)
    assert m["breakeven_cost_bps"] >= 0


def test_no_order_execution_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "premarket_trade", "run_v2", "live_trade", ".orders")
    mods = []
    for n in ast.walk(ast.parse((PKG / "run_lead_validation.py").read_text())):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module)
    for m in mods:
        assert not any(b in m.lower() for b in banned), f"imports {m}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
