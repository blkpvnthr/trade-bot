"""Tests for options/IV feature computation + leak-free alignment. Network-free."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))

import options_features as OF


def _chain(date="2024-03-04"):
    rows = []
    for exp, dte in [("2024-03-11", 7), ("2024-04-03", 30)]:
        for typ, dl in [("call", 0.5), ("put", -0.5), ("call", 0.25), ("put", -0.25)]:
            iv = 0.40 if dte == 7 else 0.30          # backwardation: IV7 > IV30
            iv += 0.05 if (typ == "put" and dl == -0.25) else 0.0   # put skew
            rows.append({"type": typ, "delta": dl, "implied_volatility": iv,
                         "open_interest": 100 if typ == "call" else 150,
                         "expiration": exp, "date": date})
    return rows


def test_daily_features_are_sane():
    f = OF._daily_features(_chain())
    assert abs(f["iv7"] - 0.40) < 1e-6 and abs(f["iv30"] - 0.30) < 1e-6
    assert f["term_slope"] < 0                       # IV30 - IV7 < 0 (backwardation)
    assert abs(f["pc_oi_ratio"] - (150 * 4) / (100 * 4 + 1.0)) < 0.05
    assert f["skew_25d"] > 0                          # 25d put IV > 25d call IV


def test_alignment_is_leak_free():
    idx = pd.MultiIndex.from_product(
        [["X"], pd.date_range("2024-03-04 10:00", periods=4, freq="D")], names=["symbol", "timestamp"])
    feats = {"X": pd.DataFrame({"iv_rank": [0.1, 0.9, 0.2, 0.3]},
                               index=pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]))}
    v = OF.align(idx, feats, ["iv_rank"])["iv_rank"].to_numpy()
    assert v[0] == 0.0                                # first bar: no prior chain
    assert v[1] == 0.1                                # 03-05 bar sees 03-04 chain (not 03-05)
    assert v[2] == 0.9                                # 03-06 bar sees 03-05 chain


def test_no_order_execution_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "premarket_trade", "run_v2", "live_trade", ".orders")
    mods = []
    for n in ast.walk(ast.parse((PKG / "options_features.py").read_text())):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module)
    for m in mods:
        assert not any(b in m.lower() for b in banned), f"options_features imports {m}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
