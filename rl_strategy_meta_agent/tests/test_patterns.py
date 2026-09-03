"""Tests for chart/candle pattern recognition: correctness + no lookahead.

Patterns must be a function of bars[:e+1] only and never of outcome/exit data.
Candle math is checked against hand calculations; each detector has a crafted
positive case; insufficient history returns 'unknown', not a fake label.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import patterns as P


def _bars(specs, start="2024-03-04 13:30"):
    """specs = list of (o,h,l,c); one 15Min session, constant volume."""
    idx = pd.date_range(start, periods=len(specs), freq="15min", tz="UTC")
    a = np.array(specs, float)
    return pd.DataFrame({"timestamp": idx, "open": a[:, 0], "high": a[:, 1],
                         "low": a[:, 2], "close": a[:, 3], "volume": np.full(len(a), 1000.0)})


def _filler(n, lvl=100.0):
    return [(lvl, lvl + 0.3, lvl - 0.3, lvl) for _ in range(n)]


def test_no_lookahead_changing_future_candles_does_not_change_labels():
    b = _bars(_filler(24) + [(100, 104, 99, 103)])
    e = len(b) - 1
    r1 = P.detect_patterns(b, e)
    bad = b.copy()
    # there are no bars after e here, so extend with wild bars and re-point e
    ext = pd.concat([b, _bars([(200, 300, 100, 250)] * 5,
                              start="2024-03-05 13:30")], ignore_index=True)
    r2 = P.detect_patterns(ext, e)
    for k in P.ALL_COLS:
        a, c = r1[k], r2[k]
        assert a == c or (isinstance(a, float) and np.isnan(a) and np.isnan(c)), f"lookahead in {k}"


def test_candle_body_and_wick_math():
    b = _bars(_filler(23) + [(100, 104, 99, 103)])          # rng 5, body 3, close high
    r = P.detect_patterns(b, len(b) - 1)
    assert abs(r["candle_body_pct"] - 0.6) < 1e-9
    assert abs(r["upper_wick_pct"] - 0.2) < 1e-9
    assert abs(r["lower_wick_pct"] - 0.2) < 1e-9
    assert abs(r["close_location"] - 0.8) < 1e-9


def test_doji_detection():
    b = _bars(_filler(23) + [(100, 101, 99, 100.05)])       # body 0.05 of range 2
    r = P.detect_patterns(b, len(b) - 1)
    assert r["doji_flag"] == 1.0


def test_pin_bar_detection():
    b = _bars(_filler(23) + [(100.4, 100.6, 99.0, 100.5)])  # long lower wick, tiny body
    r = P.detect_patterns(b, len(b) - 1)
    assert r["pin_bar_flag"] == 1.0


def test_engulfing_detection():
    b = _bars(_filler(22) + [(101, 101.1, 99.9, 100.0),     # bearish prev
                             (99.9, 101.6, 99.8, 101.1)])    # bullish engulf
    r = P.detect_patterns(b, len(b) - 1)
    assert r["engulfing_flag"] == 1.0


def test_higher_low_sequence_detection():
    seq = [(99.0, 99.5, 98.8, 99.2), (99.2, 99.7, 99.0, 99.4),
           (99.4, 99.9, 99.2, 99.6), (99.6, 100.1, 99.4, 99.8)]   # lows strictly rising
    b = _bars(_filler(22) + seq)
    r = P.detect_patterns(b, len(b) - 1)
    assert r["higher_low_sequence"] >= 3


def test_failed_breakout_detection():
    # flat resistance ~100.3, a poke to 103 in the last 3 bars, close back below
    b = _bars(_filler(20) + [(100, 103, 100, 100.2),        # poke above resistance
                             (100.1, 100.4, 99.9, 100.1),
                             (100.0, 100.2, 99.8, 99.9)])    # closes back under
    r = P.detect_patterns(b, len(b) - 1)
    assert r["failed_breakout_flag"] == 1.0


def test_vwap_reclaim_detection():
    # day 1 = history; day 2 resets VWAP: dip below then reclaim on the entry bar
    d1 = _bars(_filler(22), start="2024-03-04 13:30")
    d2 = _bars([(100, 100.2, 99.9, 100.0), (98, 98.2, 97.8, 98.0),
                (98, 98.2, 97.8, 98.0), (98, 98.2, 97.8, 98.0),
                (102.8, 103.2, 102.6, 103.0)], start="2024-03-05 13:30")
    b = pd.concat([d1, d2], ignore_index=True)
    r = P.detect_patterns(b, len(b) - 1)
    assert r["vwap_reclaim_flag"] == 1.0


def test_range_compression_detection():
    wide = [(100, 101, 99, 100) for _ in range(20)]         # range 2
    tight = [(100, 100.15, 99.95, 100.05) for _ in range(5)]  # range 0.2
    b = _bars(wide + tight)
    r = P.detect_patterns(b, len(b) - 1)
    assert r["range_compression_flag"] == 1.0


def test_primary_pattern_is_the_strongest_present():
    b = _bars(_filler(22) + [(100.02, 100.05, 99.98, 100.0),  # tiny bearish prev body
                             (99.9, 101.6, 99.85, 101.5)])     # large bull engulf -> strength 1.0
    r = P.detect_patterns(b, len(b) - 1)
    assert r["primary_pattern"] == "engulfing"
    assert abs(r["primary_pattern_strength"] - 1.0) < 1e-9


def test_missing_bars_return_unknown_not_fake_labels():
    short = _bars(_filler(10))
    r = P.detect_patterns(short, len(short) - 1)             # e < MIN_HISTORY
    assert r["primary_pattern"] == "unknown"
    assert np.isnan(r["pattern_count"])
    b = _bars(_filler(24))
    assert P.detect_patterns(b, 999)["primary_pattern"] == "unknown"  # e past end


def test_no_outcome_or_exit_fields_referenced():
    # prose may mention them; what must never appear is a CODE access of a result
    # column (e.g. ["mfe"], .pnl) or a broker symbol.
    src = Path(P.__file__).read_text().lower()
    for tok in ("mfe", "mae", "pnl", "outcome", "exit_price", "profit_capture",
                "return_pct", "normalized_pnl"):
        for access in (f'"{tok}"', f"'{tok}'", f".{tok}"):
            assert access not in src, f"patterns must not read the {tok} column"
    for banned in ("submit_order", "base_url", "tradingclient"):
        assert banned not in src, f"patterns must stay research-only ({banned})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
