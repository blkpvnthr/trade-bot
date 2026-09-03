"""Leakage + correctness tests for the chart-state sequence layer.

A trade's candle window must be a function of bars[:e+1] only, never of later
bars or any outcome/exit field. Features must be finite and scale-free.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import sequences as S


def _bars(n=320, start="2024-03-04 13:30"):
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    base = np.linspace(100, 130, n) + np.sin(np.arange(n) / 5)
    return pd.DataFrame({"timestamp": idx, "open": base, "high": base + 0.5,
                         "low": base - 0.5, "close": base + 0.1,
                         "volume": np.linspace(1000, 2000, n)})


def test_window_shape_and_features():
    f = S.candle_feature_frame(_bars())
    w, nv = S.extract_window(f, 300, 240)
    assert w.shape == (240, S.N_FEATURES)
    assert nv == 240 and np.isfinite(w).all()


def test_no_lookahead_future_bars_do_not_change_the_window():
    b = _bars()
    e, W = 280, 200
    w1, _ = S.extract_window(S.candle_feature_frame(b), e, W)
    bad = b.copy()
    bad.loc[e + 1:, ["open", "high", "low", "close", "volume"]] *= 5.0
    w2, _ = S.extract_window(S.candle_feature_frame(bad), e, W)
    assert np.allclose(w1, w2, atol=1e-6), "future bars leaked into the window"


def test_entry_candle_is_the_reference_close():
    f = S.candle_feature_frame(_bars())
    w, _ = S.extract_window(f, 300, 240)
    for name in ("c_rel",):
        assert abs(w[-1, S.SEQ_FEATURES.index(name)]) < 1e-6  # close/close - 1 == 0


def test_short_history_is_left_padded_with_zeros():
    f = S.candle_feature_frame(_bars(n=80))
    w, nv = S.extract_window(f, 40, 240)        # only 41 candles available
    assert w.shape == (240, S.N_FEATURES)
    assert nv == 41
    assert np.allclose(w[:240 - 41], 0.0)       # older slots padded
    assert not np.allclose(w[-1], 0.0)          # the real entry candle is not zero


def test_invalid_index_returns_none():
    f = S.candle_feature_frame(_bars())
    assert S.extract_window(f, 10_000, 240)[0] is None
    assert S.extract_window(f, -1, 240)[0] is None


def test_features_are_finite_after_extract():
    f = S.candle_feature_frame(_bars())
    for e in (60, 120, 200, 300):
        w, _ = S.extract_window(f, e, 128)
        assert np.isfinite(w).all(), f"non-finite feature at e={e}"


def _ohlcv(o, h, l, c, v, start="2024-03-04 13:30"):
    idx = pd.date_range(start, periods=len(c), freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "open": o, "high": h, "low": l,
                         "close": c, "volume": v})


def test_volume_behavior_features():
    n = 40
    px = np.linspace(100, 108, n)                       # rising price
    vol = np.concatenate([np.full(28, 3000.0), np.linspace(3000, 400, 12)])  # fading
    f = S.candle_feature_frame(_ohlcv(px, px + 0.4, px - 0.4, px + 0.2, vol))
    last = f.iloc[-1]
    assert last["up_volume_flag"] == 1.0 and last["down_volume_flag"] == 0.0   # green
    assert last["volume_acceleration"] < 0                                     # fading
    assert last["rising_price_fading_volume"] == 1.0                           # giveback risk
    assert last["volume_dryup_flag"] == 1.0
    assert last["volume_on_red_candle"] == 0.0                                 # not red


def test_breakout_climax_exhaustion_fires():
    n = 35
    px = np.full(n, 100.0); px[-1] = 102.0              # last bar breaks out
    hi = px + 0.3; hi[-1] = 103.5                        # big upper wick on the breakout
    lo = px - 0.3; lo[-1] = 101.8
    cl = px.copy(); cl[-1] = 102.0                       # close below the high -> upper wick
    op = px.copy(); op[-1] = 101.9
    vol = np.full(n, 1000.0); vol[-1] = 80000.0          # volume climax
    f = S.candle_feature_frame(_ohlcv(op, hi, lo, cl, vol))
    last = f.iloc[-1]
    assert last["volume_climax_flag"] == 1.0
    assert last["breakout_with_volume_expansion"] == 1.0
    assert last["breakout_climax_exhaustion"] == 1.0


def test_no_outcome_or_broker_path_in_module():
    src = Path(S.__file__).read_text().lower()
    for tok in ("outcome", "mfe", "mae", "pnl", "exit_price"):
        for access in (f'"{tok}"', f"'{tok}'", f".{tok}"):
            assert access not in src, f"sequences must not read the {tok} column"
    for banned in ("submit_order", "base_url", "tradingclient"):
        assert banned not in src


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
