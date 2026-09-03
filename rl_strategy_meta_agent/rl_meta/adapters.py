"""PineScript-pattern -> Python strategy adapters.

We do not execute PineScript. Instead we recognise common patterns from the
parsed metadata and map them to a deterministic Python implementation that
runs on real OHLCV bars. Each adapter returns aligned boolean entry/exit
arrays (long-only); the shared execution engine in signals.py turns those into
TradeRecords with next-bar-open fills (no look-ahead).

Adapters are deliberately simple, documented approximations of the canonical
setup -- not bit-exact PineScript reproductions. Unrecognised scripts get no
adapter and are marked parsed_but_not_executable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import StrategyMeta, Status

SUPPORTED = ("sma_crossover", "ema_crossover", "rsi_mean_reversion",
             "vwap_reclaim", "pdh_pdl_break_retest", "tjl_trend_join")


# -- input helpers ---------------------------------------------------------
def _int_input(meta: StrategyMeta, key: str, default: int) -> int:
    try:
        return int(float(meta.inputs.get(key)))
    except (TypeError, ValueError):
        return default


def _cross_up(a: pd.Series, b: pd.Series) -> np.ndarray:
    prev = (a.shift(1) <= b.shift(1))
    now = (a > b)
    return (prev & now).fillna(False).to_numpy()


def _cross_dn(a: pd.Series, b: pd.Series) -> np.ndarray:
    prev = (a.shift(1) >= b.shift(1))
    now = (a < b)
    return (prev & now).fillna(False).to_numpy()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _prior_session_levels(df: pd.DataFrame) -> tuple:
    """Per-bar prior-calendar-day high/low (works for daily and intraday)."""
    day = df["timestamp"].dt.date
    daily = df.groupby(day).agg(dh=("high", "max"), dl=("low", "min"))
    prior = daily.shift(1)
    ph = day.map(prior["dh"]).to_numpy(dtype=float)
    pl = day.map(prior["dl"]).to_numpy(dtype=float)
    return ph, pl


# -- adapters: (df, meta) -> (entries, exits) boolean arrays ----------------
def sma_crossover(df: pd.DataFrame, meta: StrategyMeta) -> tuple:
    fast = _int_input(meta, "fast", 10)
    slow = _int_input(meta, "slow", 30)
    if fast >= slow:
        fast, slow = 10, 30
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    return _cross_up(f, s), _cross_dn(f, s)


def ema_crossover(df: pd.DataFrame, meta: StrategyMeta) -> tuple:
    fast = _int_input(meta, "fast", 12)
    slow = _int_input(meta, "slow", 26)
    if fast >= slow:
        fast, slow = 12, 26
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    return _cross_up(f, s), _cross_dn(f, s)


def rsi_mean_reversion(df: pd.DataFrame, meta: StrategyMeta) -> tuple:
    n = _int_input(meta, "length", 14)
    lower = _int_input(meta, "lower", 30)
    exit_level = _int_input(meta, "upper", 55)
    rsi = _rsi(df["close"], n)
    # Enter when RSI reclaims the oversold line; exit when it reaches midline.
    entries = ((rsi.shift(1) < lower) & (rsi >= lower)).fillna(False).to_numpy()
    exits = (rsi >= exit_level).fillna(False).to_numpy()
    return entries, exits


def vwap_reclaim(df: pd.DataFrame, meta: StrategyMeta) -> tuple:
    w = _int_input(meta, "length", 20)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).rolling(w).sum()
    vv = df["volume"].rolling(w).sum().replace(0.0, np.nan)
    vwap = pv / vv
    # ponytail: rolling VWAP over `w` bars stands in for session-anchored VWAP;
    # swap to a session-anchored cumulative for true intraday reclaim.
    return _cross_up(df["close"], vwap), _cross_dn(df["close"], vwap)


def pdh_pdl_break_retest(df: pd.DataFrame, meta: StrategyMeta) -> tuple:
    ph, pl = _prior_session_levels(df)
    close = df["close"].to_numpy(dtype=float)
    prev_close = df["close"].shift(1).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        broke_up = (prev_close <= ph) & (close > ph)   # break above prior high
        fail_dn = close < pl                            # close below prior low
    entries = np.nan_to_num(broke_up, nan=0.0).astype(bool)
    exits = np.nan_to_num(fail_dn, nan=0.0).astype(bool)
    return entries, exits


def tjl_trend_join(df: pd.DataFrame, meta: StrategyMeta) -> tuple:
    sma200 = df["close"].rolling(200).mean()
    sma50 = df["close"].rolling(50).mean()
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    uptrend = (df["close"] > sma200) & (sma50 > sma200)
    # Join the trend on a pullback that reclaims the fast EMA.
    reclaim = (df["close"].shift(1) < ema20.shift(1)) & (df["close"] >= ema20)
    entries = (uptrend & reclaim).fillna(False).to_numpy()
    exits = (df["close"] < sma50).fillna(False).to_numpy()
    return entries, exits


ADAPTERS = {
    "sma_crossover": sma_crossover,
    "ema_crossover": ema_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
    "vwap_reclaim": vwap_reclaim,
    "pdh_pdl_break_retest": pdh_pdl_break_retest,
    "tjl_trend_join": tjl_trend_join,
}


def detect_adapter(meta: StrategyMeta) -> tuple:
    """Map parsed metadata to an adapter name. Returns (name|None, reason)."""
    ind = set(meta.indicators)
    text = " ".join([meta.strategy_name or ""] + meta.entry_logic
                    + meta.exit_logic).lower()
    name = (meta.strategy_name or "").lower()

    if "tjl" in name or "trend join" in name or "trend join" in text:
        return "tjl_trend_join", ""
    if "pdh" in text or "pdl" in text or "previous day" in text or \
            ("retest" in text and "break" in text):
        return "pdh_pdl_break_retest", ""
    if "rsi" in ind:
        return "rsi_mean_reversion", ""
    if "vwap" in ind:
        return "vwap_reclaim", ""
    if "ema" in ind and ("crossover" in ind or "crossunder" in ind or "cross" in text):
        return "ema_crossover", ""
    if "sma" in ind and ("crossover" in ind or "crossunder" in ind or "cross" in text):
        return "sma_crossover", ""
    if "ema" in ind:
        return "ema_crossover", ""
    if "sma" in ind:
        return "sma_crossover", ""
    return None, "no supported adapter for this PineScript pattern"


def annotate_executability(meta: StrategyMeta) -> StrategyMeta:
    """Set executable / adapter_type / unsupported_reason / status / family."""
    from .correlation import family_for
    # An indicator()/study (e.g. a regime-probability indicator) is not a
    # tradeable strategy even if it uses moving averages -> never adapter-screen it.
    if not meta.declares_strategy:
        meta.executable = False
        meta.adapter_type = None
        meta.unsupported_reason = "indicator/study, not a tradeable strategy (no entries/exits)"
        meta.status = Status.NOT_EXECUTABLE.value
        meta.strategy_family = family_for(meta.strategy_name, None, meta.family_override)
        return meta
    name, reason = detect_adapter(meta)
    if name is None:
        meta.executable = False
        meta.adapter_type = None
        meta.unsupported_reason = reason
        meta.status = Status.NOT_EXECUTABLE.value
    else:
        meta.executable = True
        meta.adapter_type = name
        meta.unsupported_reason = None
        if meta.status == Status.NOT_EXECUTABLE.value:
            meta.status = Status.PENDING.value
    meta.strategy_family = family_for(meta.strategy_name, meta.adapter_type,
                                      meta.family_override)
    return meta


def _demo():
    from .registry import load_strategies, PINE_DIR
    metas = load_strategies(PINE_DIR)
    sample = next(x for x in metas if x.strategy_id == "sample_strategy")
    m = annotate_executability(sample)
    assert m.executable and m.adapter_type == "sma_crossover", m.adapter_type
    junk = StrategyMeta("j", "Mystery Oscillator", "h", "2023-01-01", "j.txt",
                        indicators=["adx"])
    annotate_executability(junk)
    assert not junk.executable and junk.status == Status.NOT_EXECUTABLE.value
    print("adapters._demo OK", m.adapter_type)


if __name__ == "__main__":
    _demo()
