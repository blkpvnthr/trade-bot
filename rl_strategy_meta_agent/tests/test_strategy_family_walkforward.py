"""Tests for walk-forward strategy-family selection: whitelist + no-leak folds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import strategy_family_walkforward as W


def _trades(n=400):
    ts = pd.date_range("2024-01-01", periods=n, freq="6h", tz="UTC")
    rng = np.random.default_rng(0)
    fam = np.where(np.arange(n) % 2 == 0, "good", "bad")
    ret = np.where(fam == "good", rng.normal(0.02, 0.01, n), rng.normal(-0.03, 0.01, n))
    reg = np.where(np.arange(n) % 3 == 0, "bull", "chop")
    return pd.DataFrame({"exit_time": ts, "strategy": fam, "regime": reg, "trade_return": ret})


def test_whitelist_keeps_good_drops_bad():
    wl = W.whitelist_families(_trades())
    assert "good" in wl and "bad" not in wl


def test_whitelist_drops_small_families():
    df = _trades(400)
    df.loc[df["strategy"] == "good", "strategy"] = "good"
    tiny = df[df["strategy"] == "good"].head(W.MIN_FAM - 1)        # below MIN_FAM
    assert "good" not in W.whitelist_families(tiny)               # unprovable -> excluded


def test_folds_are_chronological_and_leak_free():
    df = _trades(400).sort_values("exit_time").reset_index(drop=True)
    fl = W.folds(df, n=4, initial_frac=0.4)
    assert len(fl) == 4
    for lo, hi, train, test in fl:
        assert (train["exit_time"] < lo).all()                   # train strictly before test
        assert (test["exit_time"] >= lo).all()
        assert lo <= hi


def test_select_policies_filter_expected():
    df = _trades(200)
    df["strategy"] = np.where(np.arange(200) % 2 == 0, "martingale", "linear_regression")
    sel, _ = W.select("exclude_global_bad", df, df)
    assert (sel["strategy"] != "martingale").all()
    sel2, _ = W.select("trade_all", df, df)
    assert len(sel2) == len(df)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
