"""M4 multi-timeframe momentum gate - pure, testable primitives.

No I/O, no look-ahead (callers pass only COMPLETED candles). Trend/momentum math
is deliberately simple and deterministic so it can be unit-tested against fixtures
and reused by both the streaming engine and any backtest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> float:
    """Close-weighted VWAP (repo convention). Falls back to last close if no volume."""
    if df is None or df.empty:
        return 0.0
    close = df["close"].astype(float)
    vol = df["volume"].astype(float).replace(0, np.nan)
    if vol.notna().any():
        return float((close * vol).sum() / max(float(vol.sum()), 1e-9))
    return float(close.iloc[-1])


def resample_ohlcv(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 5m OHLCV to a higher timeframe (e.g. '30min','60min'). Expects a
    tz-aware 'timestamp' column. Returns a frame with the same columns."""
    if df_5m is None or df_5m.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    d = df_5m.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = d.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
    return out.reset_index()


def classify_trend(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50,
                   slope_lookback: int = 5, swing_lookback: int = 10) -> str:
    """BULLISH/BEARISH/NEUTRAL from EMA alignment + EMA slope + swing structure."""
    if df is None or len(df) < ema_slow + slope_lookback:
        return NEUTRAL
    close = df["close"].astype(float).reset_index(drop=True)
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    fast = ema(close, ema_fast)
    slow = ema(close, ema_slow)
    c, f, s = float(close.iloc[-1]), float(fast.iloc[-1]), float(slow.iloc[-1])
    f_prev = float(fast.iloc[-1 - slope_lookback])

    aligned_up = c > f > s
    aligned_dn = c < f < s
    slope_up = f > f_prev
    slope_dn = f < f_prev

    n = min(swing_lookback, len(close) // 2)
    if n >= 2:
        recent_hi, prior_hi = float(high.iloc[-n:].max()), float(high.iloc[-2 * n:-n].max())
        recent_lo, prior_lo = float(low.iloc[-n:].min()), float(low.iloc[-2 * n:-n].min())
        structure_up = recent_hi > prior_hi and recent_lo > prior_lo
        structure_dn = recent_hi < prior_hi and recent_lo < prior_lo
    else:
        structure_up = structure_dn = False

    if aligned_up and slope_up and structure_up:
        return BULLISH
    if aligned_dn and slope_dn and structure_dn:
        return BEARISH
    return NEUTRAL


def ltf_momentum_bullish(df_5m_today: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21,
                         ret_lookback: int = 6, min_rel_vol: float = 1.0) -> bool:
    """5m momentum is bullish: above VWAP, EMA9>EMA21, rising, and healthy rel-volume."""
    if df_5m_today is None or len(df_5m_today) < max(ema_slow, ret_lookback + 1):
        return False
    close = df_5m_today["close"].astype(float).reset_index(drop=True)
    vol = df_5m_today["volume"].astype(float).reset_index(drop=True)
    last = float(close.iloc[-1])
    vwap = session_vwap(df_5m_today)
    ef = float(ema(close, ema_fast).iloc[-1])
    es = float(ema(close, ema_slow).iloc[-1])
    rising = last > float(close.iloc[-1 - ret_lookback])
    avg_vol = float(vol.tail(20).mean()) if len(vol) else 0.0
    rel_vol = (float(vol.iloc[-1]) / avg_vol) if avg_vol > 0 else 1.0
    return last > vwap and ef > es and rising and rel_vol >= min_rel_vol


def momentum_lost(df_5m_today: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21) -> bool:
    """Reversal signal: 5m momentum lost - price < VWAP OR EMA9 < EMA21."""
    if df_5m_today is None or len(df_5m_today) < ema_slow:
        return False
    close = df_5m_today["close"].astype(float).reset_index(drop=True)
    last = float(close.iloc[-1])
    vwap = session_vwap(df_5m_today)
    ef = float(ema(close, ema_fast).iloc[-1])
    es = float(ema(close, ema_slow).iloc[-1])
    return last < vwap or ef < es


def one_min_trigger(bars_1m, lookback: int = 3) -> bool:
    """Micro-breakout: last 1m close > prior close AND >= max high of the prior `lookback`."""
    bars = list(bars_1m) if bars_1m is not None else []
    if len(bars) < lookback + 1:
        return False
    last = bars[-1]
    prior = bars[-1 - lookback:-1]
    last_close = float(last.get("close", 0.0))
    prev_close = float(bars[-2].get("close", 0.0))
    prior_high = max(float(b.get("high", 0.0)) for b in prior)
    return last_close > prev_close and last_close >= prior_high


def mtf_entry_ok(htf30: str, htf60: str, mom5m_ok: bool, trig1m_ok: bool,
                 require_both_htf: bool = False) -> bool:
    """Long entry confirmation across timeframes."""
    htfs = {htf30, htf60}
    if require_both_htf:
        htf_ok = htf30 == BULLISH and htf60 == BULLISH
    else:
        htf_ok = BULLISH in htfs and BEARISH not in htfs
    return bool(htf_ok and mom5m_ok and trig1m_ok)


def pullback_add_signal(last_price: float, avg_entry: float, peak_price: float,
                        mom5m_bullish: bool, dip_lo: float = 0.003, dip_hi: float = 0.010) -> bool:
    """Pyramid into a winner on a minor pullback that is still trend-intact."""
    if last_price <= 0 or avg_entry <= 0 or peak_price <= 0:
        return False
    if last_price <= avg_entry:
        return False  # only add to a position in profit
    dip = (peak_price - last_price) / peak_price
    return mom5m_bullish and (dip_lo <= dip <= dip_hi)
