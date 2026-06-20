"""Tests for strategy-stream builders: retired guard, leak-free, finite streams."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import strategy_streams as ss


def _panel(nsym=6, nt=120):
    ts = pd.date_range("2024-03-04 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        rng = np.random.default_rng(s)
        for i, t in enumerate(ts):
            rows.append((f"S{s}", t, rng.normal(0, 0.01), rng.normal(0, 0.01),
                         rng.uniform(0.5, 3), rng.normal(0, 0.005)))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "ret", "or_high_dist",
                                    "rel_volume", "fut_ret_1"]).set_index(["symbol", "timestamp"]).sort_index()
    return p


def test_retired_guard_raises():
    ss.validate_no_retired(["alpha_mined_ensemble", "linear_regression_baseline"])  # ok
    for bad in ("sniper_strategy", "martingale_daily", "x_pin_bar_magic_v1_y"):
        try:
            ss.validate_no_retired([bad]); assert False, bad
        except ValueError:
            pass


def test_long_short_stream_daily_finite():
    p = _panel()
    sig = p["ret"]                                      # any per-bar signal
    daily = ss.long_short_stream(p, sig)
    assert len(daily) >= 1 and np.isfinite(daily.values).all()


def test_exposure_stream_finite():
    p = _panel()
    daily = ss.exposure_stream(p, p["ret"])
    assert len(daily) >= 1 and np.isfinite(daily.values).all()


def test_long_short_is_leak_free():
    # the daily stream up to day D must not change if FUTURE bars change
    p = _panel()
    sig = ss._cs_z(p["ret"])
    d1 = ss.long_short_stream(p, sig)
    k = p.index.get_level_values(1)[60]
    bad = p.copy()
    bad.loc[bad.index.get_level_values(1) > k, "fut_ret_1"] *= 9
    sig2 = ss._cs_z(bad["ret"])
    d2 = ss.long_short_stream(bad, sig2)
    early = d1.index <= pd.Timestamp(k).tz_convert("America/New_York").normalize()
    common = d1.index[early].intersection(d2.index)
    # exclude the boundary day (its later bars were corrupted)
    bday = pd.Timestamp(k).tz_convert("America/New_York").normalize()
    common = common[common < bday]
    assert np.allclose(d1.loc[common].values, d2.loc[common].values, atol=1e-9)


def test_orb_relvol_triggers():
    p = _panel(nt=60)
    # force a clear breakout+volume bar for S0
    p = p.copy()
    p.loc[("S0", p.loc["S0"].index[10]), ["or_high_dist", "rel_volume", "ret"]] = [0.01, 2.0, 0.01]
    daily = ss.orb_relvol_stream(p, relvol_min=1.5)
    assert len(daily) >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
