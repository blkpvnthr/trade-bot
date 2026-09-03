"""Leakage-prevention tests for the market-context / pattern / regime layer.

The whole layer is only valid if a value at bar k uses no information from bar
k+1 onward. These tests corrupt the future and assert the present is unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import market_context as mc


def _bars(n=80, start="2024-03-04 13:30"):
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    rng = np.linspace(100, 120, n)
    return pd.DataFrame({"timestamp": idx, "open": rng, "high": rng + 0.5,
                         "low": rng - 0.5, "close": rng, "volume": np.full(n, 1000.0)})


def test_context_features_are_point_in_time():
    b = _bars()
    ctx = mc.build_context_frame(b)
    k = 40
    bad = b.copy()
    bad.loc[k + 1:, ["high", "low", "close", "volume"]] *= 7.0  # wreck the future
    ctx2 = mc.build_context_frame(bad)
    for col in mc.CTX_NUMERIC:
        a, c = ctx[col].iloc[k], ctx2[col].iloc[k]
        assert (pd.isna(a) and pd.isna(c)) or abs(a - c) < 1e-9, f"future leaked into {col}@k"
    assert ctx["ctx_own_regime"].iloc[k] == ctx2["ctx_own_regime"].iloc[k]


def test_reference_trend_uses_no_future_bars():
    b = _bars()
    spy = _bars()  # same grid
    ctx = mc.build_context_frame(b, spy=spy)
    k = 50
    spy_bad = spy.copy()
    spy_bad.loc[k + 1:, "close"] *= 5.0
    ctx2 = mc.build_context_frame(b, spy=spy_bad)
    a, c = ctx["ctx_spy_trend"].iloc[k], ctx2["ctx_spy_trend"].iloc[k]
    assert (pd.isna(a) and pd.isna(c)) or abs(a - c) < 1e-9, "future SPY bar leaked into trend@k"


def test_pattern_only_reads_bars_up_to_entry():
    b = _bars()
    ctx = mc.build_context_frame(b)
    e = 45
    p1 = mc.classify_pattern(b["high"].to_numpy(), b["low"].to_numpy(),
                             b["close"].to_numpy(), ctx, e, is_long=True)
    bad = b.copy()
    bad.loc[e + 1:, ["high", "low", "close"]] *= 9.0
    ctx_bad = mc.build_context_frame(bad)
    p2 = mc.classify_pattern(bad["high"].to_numpy(), bad["low"].to_numpy(),
                             bad["close"].to_numpy(), ctx_bad, e, is_long=True)
    assert p1 == p2, "pattern label depends on bars after entry"
    assert p1 in mc.PATTERNS


def test_opening_range_is_nan_until_it_forms():
    b = _bars()
    ctx = mc.build_context_frame(b)
    # first OR_BARS bars of session 1 cannot know the opening range yet
    assert pd.isna(ctx["ctx_or_dist_high"].iloc[0])
    assert pd.notna(ctx["ctx_or_dist_high"].iloc[mc.OR_BARS + 1])


def test_prior_day_high_is_absent_on_the_first_session():
    b = _bars()
    ctx = mc.build_context_frame(b)
    assert pd.isna(ctx["ctx_pdh_dist"].iloc[0]), "no prior day exists for the first session"


def test_ref_symbols_and_sector_map():
    assert "SPY" in mc.REF_SYMBOLS and "QQQ" in mc.REF_SYMBOLS
    assert set(mc.SECTOR_ETF.values()) <= set(mc.REF_SYMBOLS)


def test_no_broker_or_order_path_in_module():
    src = Path(mc.__file__).read_text().lower()
    for banned in ("tradingclient", "submit_order", "base_url", "place_order"):
        assert banned not in src, f"market_context must stay research-only ({banned})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
