"""Chart-state sequences for sequence models (CNN / LSTM / Transformer).

The point of this module: stop compressing a chart into a handful of summary
numbers and instead hand the model the raw candle sequence -- so it can learn
structure (flags, squeezes, failed breakouts, accumulation) on its own.

For a trade entered at bar e, extract_window returns the last `window` candles
with timestamp <= entry_time, each as a feature vector (OHLC relative to the
entry close + per-candle indicators). Everything is point-in-time: every
indicator is a backward EWMA / rolling / diff, and the VWAP/PDH/PDL distances
come from market_context.build_context_frame, which is PIT and unit-tested. No
post-entry bar and no outcome/exit field is ever read.

  # ponytail: features are scale-free by construction (returns, ratios,
  #           distances) so no fitted scaler is needed -- one less leak surface.

Research-only: no broker, no orders, no live path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rl_meta.market_context import build_context_frame

# per-candle feature order (F = 42). Price/indicator + candle-structure +
# level-distance + a rich VOLUME-state block + price*volume interactions -- the
# things a trader reads off each bar, with volume promoted to a first-class input.
SEQ_FEATURES = [
    # price / indicators
    "ret", "o_rel", "h_rel", "l_rel", "c_rel", "rel_volume", "rsi", "atr_pct",
    "macd_hist", "ema9_dist", "ema21_dist", "ema50_dist", "ema200_dist",
    "vwap_dist", "pdh_dist", "pdl_dist",
    # candle structure / level distances
    "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "body_to_range_ratio",
    "or_high_dist", "or_low_dist",
    # volume state (rel_volume above == relative_volume_20)
    "relative_volume_50", "volume_zscore_20", "volume_zscore_50",
    "volume_acceleration", "volume_slope_5", "volume_slope_10", "dollar_volume",
    "dollar_volume_zscore_20", "up_volume_flag", "down_volume_flag",
    "volume_on_green_candle", "volume_on_red_candle", "volume_climax_flag",
    "volume_dryup_flag",
    # price * volume interactions (volume behavior, not just volume)
    "breakout_with_volume_expansion", "pullback_on_declining_volume",
    "green_candle_high_volume", "red_candle_high_volume",
    "upper_wick_high_volume", "lower_wick_high_volume",
    "volume_near_vwap", "volume_near_pdh", "volume_near_pdl",
    "breakout_climax_exhaustion", "rising_price_fading_volume",
]
N_FEATURES = len(SEQ_FEATURES)
_SCALEFREE = SEQ_FEATURES[5:]            # everything except ret + the 4 *_rel
MIN_VALID = 60                           # need at least this many real candles


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def candle_feature_frame(bars: pd.DataFrame, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-bar leak-free features (raw OHLC kept for window-relative scaling)."""
    b = bars.reset_index(drop=True)
    o, h, l, c, v = (b["open"].astype(float), b["high"].astype(float),
                     b["low"].astype(float), b["close"].astype(float),
                     b["volume"].astype(float))
    ema = {p: c.ewm(span=p, adjust=False).mean() for p in (9, 21, 50, 200)}
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    if ctx is None:
        ctx = build_context_frame(b)     # PIT vwap / pdh / pdl / OR distances
    ctx = ctx.reset_index(drop=True)
    rng = (h - l).replace(0, np.nan)
    body = c - o
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng

    # --- volume state (all backward rolling / shift -> point-in-time) ---
    vma20 = v.rolling(20, min_periods=5).mean()
    vma50 = v.rolling(50, min_periods=10).mean()
    vsd20 = v.rolling(20, min_periods=5).std()
    vsd50 = v.rolling(50, min_periods=10).std()
    relv20 = v / vma20
    relv50 = v / vma50
    vaccel = relv20.diff()                                  # rising>0 / fading<0
    vslope5 = (v - v.shift(5)) / vma20
    vslope10 = (v - v.shift(10)) / vma20
    dv = v * c
    dvma20 = dv.rolling(20, min_periods=5).mean()
    dvsd20 = dv.rolling(20, min_periods=5).std()
    green = (c > o)
    red = (c < o)
    climax = (((v - vma20) / vsd20) > 2.0)                  # volume blow-off
    dryup = (relv20 < 0.5)                                  # volume drying up
    rmax = h.rolling(20, min_periods=5).max().shift(1)      # prior structure only
    breakout = c > rmax
    near_vwap = ctx["ctx_vwap_dist"].abs() < 0.005
    near_pdh = ctx["ctx_pdh_dist"].abs() < 0.005
    near_pdl = ctx["ctx_pdl_dist"].abs() < 0.005
    rising = c.pct_change(3) > 0
    z0 = lambda s: s.where(s.notna(), 0.0)

    f = pd.DataFrame({
        "open": o, "high": h, "low": l, "close": c,
        "ret": np.log(c / prev_c),
        "rel_volume": relv20,
        "rsi": _rsi(c) / 100 - 0.5,
        "atr_pct": atr / c,
        "macd_hist": hist / c,
        "ema9_dist": (c - ema[9]) / ema[9],
        "ema21_dist": (c - ema[21]) / ema[21],
        "ema50_dist": (c - ema[50]) / ema[50],
        "ema200_dist": (c - ema[200]) / ema[200],
        "vwap_dist": ctx["ctx_vwap_dist"],
        "pdh_dist": ctx["ctx_pdh_dist"],
        "pdl_dist": ctx["ctx_pdl_dist"],
        "candle_body_pct": body.abs() / rng,
        "upper_wick_pct": upper_wick,
        "lower_wick_pct": lower_wick,
        "body_to_range_ratio": body / rng,                 # signed body fraction
        "or_high_dist": ctx["ctx_or_dist_high"],
        "or_low_dist": ctx["ctx_or_dist_low"],
        # volume state
        "relative_volume_50": relv50,
        "volume_zscore_20": (v - vma20) / vsd20,
        "volume_zscore_50": (v - vma50) / vsd50,
        "volume_acceleration": vaccel,
        "volume_slope_5": vslope5,
        "volume_slope_10": vslope10,
        "dollar_volume": dv / dvma20,                      # scale-free (vs 20-bar avg)
        "dollar_volume_zscore_20": (dv - dvma20) / dvsd20,
        "up_volume_flag": green.astype(float),
        "down_volume_flag": red.astype(float),
        "volume_on_green_candle": relv20.where(green, 0.0),
        "volume_on_red_candle": relv20.where(red, 0.0),
        "volume_climax_flag": climax.astype(float),
        "volume_dryup_flag": dryup.astype(float),
        # price * volume interactions (volume BEHAVIOR)
        "breakout_with_volume_expansion": (breakout & (relv20 > 1.5)).astype(float),
        "pullback_on_declining_volume": ((c.pct_change(3) < 0) & (vslope5 < 0)).astype(float),
        "green_candle_high_volume": (green & (relv20 > 1.5)).astype(float),
        "red_candle_high_volume": (red & (relv20 > 1.5)).astype(float),
        "upper_wick_high_volume": z0((upper_wick > 0.4) & (relv20 > 1.5)).astype(float),
        "lower_wick_high_volume": z0((lower_wick > 0.4) & (relv20 > 1.5)).astype(float),
        "volume_near_vwap": z0(near_vwap).astype(float) * z0(relv20),
        "volume_near_pdh": z0(near_pdh).astype(float) * z0(relv20),
        "volume_near_pdl": z0(near_pdl).astype(float) * z0(relv20),
        # breakout + climax + upper wick = exhaustion risk
        "breakout_climax_exhaustion": z0(breakout & climax & (upper_wick > 0.3)).astype(float),
        # price rising while volume fades = giveback risk
        "rising_price_fading_volume": z0(rising & (vaccel < 0)).astype(float),
    })
    return f


def extract_window(frame: pd.DataFrame, e: int, window: int) -> tuple:
    """[window, N_FEATURES] float32 for the trade at bar e. Left-pads with zeros
    if fewer than `window` candles precede entry. Returns (array, n_valid)."""
    if e < 0 or e >= len(frame):
        return None, 0
    lo = max(0, e - window + 1)
    sl = frame.iloc[lo:e + 1]
    ec = float(sl["close"].iloc[-1])
    if ec <= 0:
        return None, 0
    cols = {
        "ret": sl["ret"].to_numpy(),
        "o_rel": sl["open"].to_numpy() / ec - 1,
        "h_rel": sl["high"].to_numpy() / ec - 1,
        "l_rel": sl["low"].to_numpy() / ec - 1,
        "c_rel": sl["close"].to_numpy() / ec - 1,
    }
    for k in _SCALEFREE:
        cols[k] = sl[k].to_numpy()
    arr = np.column_stack([cols[k] for k in SEQ_FEATURES]).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n_valid = len(arr)
    if n_valid < window:                 # left-pad older history with zeros
        arr = np.vstack([np.zeros((window - n_valid, N_FEATURES), np.float32), arr])
    return arr[-window:], n_valid


def _demo():
    n = 320
    idx = pd.date_range("2024-03-04 13:30", periods=n, freq="15min", tz="UTC")
    rng = np.linspace(100, 130, n) + np.sin(np.arange(n) / 5)
    b = pd.DataFrame({"timestamp": idx, "open": rng, "high": rng + 0.5,
                      "low": rng - 0.5, "close": rng + 0.1,
                      "volume": np.linspace(1000, 2000, n)})
    f = candle_feature_frame(b)
    e, W = 300, 240
    w1, nv = extract_window(f, e, W)
    assert w1.shape == (W, N_FEATURES) and nv == W
    # no-lookahead: wreck bars after e, the window at e must be identical
    bad = b.copy(); bad.loc[e + 1:, ["open", "high", "low", "close", "volume"]] *= 4
    w2, _ = extract_window(candle_feature_frame(bad), e, W)
    assert np.allclose(w1, w2, atol=1e-6), "future bars leaked into the window"
    # the entry candle's c_rel is 0 by construction
    assert abs(w1[-1, SEQ_FEATURES.index("c_rel")]) < 1e-6
    print("sequences._demo OK  window", w1.shape, "n_valid", nv)


if __name__ == "__main__":
    _demo()
