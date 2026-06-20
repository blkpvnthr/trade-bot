"""Unit tests for the surviving-family validation helpers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import surviving_family_validation as V


def _df(n=40):
    return pd.DataFrame({
        "strategy_id": ["linear_regression"] * n,
        "return_pct": np.linspace(-0.02, 0.03, n),
        "mae_pct": np.full(n, 0.01), "mfe_pct": np.full(n, 0.02),
        "exit_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "realized_vol_20": np.linspace(0.1, 0.5, n),
        "return_20_bars": np.linspace(-0.05, 0.05, n),
        "sma50_slope": np.linspace(-0.01, 0.01, n)})


def test_fam_metrics_fields_and_pf():
    m = V.fam_metrics(_df())
    assert m["n"] == 40 and "expectancy" in m and "profit_factor" in m
    assert m["median_mfe_pct"] == 0.02 and m["median_mae_pct"] == 0.01
    assert m["mfe_mae_ratio"] == 2.0


def test_add_regimes_creates_three_axes():
    s = V.add_regimes(_df())
    for col in ("vol_regime", "trend_strength", "direction"):
        assert col in s and s[col].nunique() == 2


def test_pf_and_maxdd():
    r = np.array([0.02, -0.01, 0.03, -0.02])
    assert V._pf(r) > 0
    assert V._maxdd(r, np.arange(4)) >= 0.0


def test_empty_family_metrics():
    assert V.fam_metrics(_df().iloc[0:0])["n"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
