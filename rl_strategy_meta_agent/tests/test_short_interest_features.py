"""Tests for short-interest/short-volume leak-free alignment. Network-free (synthetic)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))

import short_interest_features as SI


def _idx(n=8):
    return pd.MultiIndex.from_product(
        [["X"], pd.date_range("2024-03-04", periods=n, freq="D")], names=["symbol", "timestamp"])


def test_short_volume_alignment_is_leak_free():
    idx = _idx()
    sv = {"X": pd.DataFrame({"date": pd.to_datetime(["2024-03-05", "2024-03-06"]),
                            "sv_ratio": [0.55, 0.40]})}
    si = {"X": pd.DataFrame(columns=["settlement_date", "short_interest", "days_to_cover"])}
    out = SI.align(idx, sv, si)
    bars = idx.get_level_values(1)
    # the 03-05 short-volume must NOT reach the 03-05 bar (same session), only 03-06 onward
    assert out["sv_ratio"][bars == pd.Timestamp("2024-03-05")].iloc[0] == 0.0
    assert out["sv_ratio"][bars == pd.Timestamp("2024-03-06")].iloc[0] == 0.55


def test_short_interest_respects_publication_lag():
    # a settlement on 03-04 is NOT public until 03-04 + SI_PUB_LAG_DAYS
    idx = pd.MultiIndex.from_product(
        [["X"], pd.date_range("2024-03-04", periods=30, freq="D")], names=["symbol", "timestamp"])
    sv = {"X": pd.DataFrame(columns=["date", "sv_ratio"])}
    si = {"X": pd.DataFrame({"settlement_date": pd.to_datetime(["2024-03-04"]),
                            "short_interest": [1e6], "days_to_cover": [2.5]})}
    out = SI.align(idx, sv, si)
    bars = idx.get_level_values(1)
    avail = pd.Timestamp("2024-03-04") + pd.Timedelta(days=SI.SI_PUB_LAG_DAYS)
    assert out["si_days_to_cover"][bars <= avail].max() == 0.0          # not yet published
    assert out["si_days_to_cover"][bars > avail].max() == 2.5          # visible after lag


def test_sv_features_compute():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30, freq="D"),
                       "sv_ratio": np.linspace(0.2, 0.5, 30)})
    f = SI._sv_features(df)
    assert "sv_ratio_z20" in f and "abnormal_sv" in f
    assert np.isfinite(f["sv_ratio_chg_5d"].iloc[-1])


def test_no_order_execution_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "premarket_trade", "run_v2", "live_trade", ".orders")
    mods = []
    for n in ast.walk(ast.parse((PKG / "short_interest_features.py").read_text())):
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
