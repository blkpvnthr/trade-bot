"""Multi-timeframe feature engine (V6 spec section 9).

Timeframes that are load-bearing for the decisive diagnostic:
  * 5-minute: reuse amps_v2_backtest.compute_features (cached, PIT-safe). Within-session, so
    UNADJUSTED split jumps (overnight) do not contaminate it -- this is why the 5m V6.1 analysis
    cleanly reproduces V5.
  * Daily (SPLIT/DIV ADJUSTED): the multi-day swing question REQUIRES corporate-action-adjusted
    prices. Leveraged/inverse ETFs reverse-split constantly (SOXS 10x, SQQQ 5x, NVD 24x) and bull
    ETFs forward-split (TQQQ 2x, SOXL 15x); on UNADJUSTED daily data these overnight jumps produce
    spurious +/- hundreds-of-percent "returns" that destroy any multi-day statistic. We therefore
    fetch Alpaca daily bars with adjustment=ALL and cache them. We NEVER approximate a multi-day
    leveraged-ETF return by multiplying an underlying return by stated leverage (spec section 24).

No look-ahead: daily features at row t use only sessions completed on/before t; forward excursions
start at the NEXT session's open. Order-flow/microstructure features need genuine ticks and are
absent/flagged, never fabricated.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_CACHE = PROJECT_ROOT / "data" / "cache" / "amps_v6_daily"          # resampled (unadjusted)
DAILY_ADJ_CACHE = PROJECT_ROOT / "data" / "cache" / "amps_v6_daily_adj"  # split/div adjusted

from amps_v2_backtest import compute_features as _five_min_features, _rma


def five_min_features(symbol: str) -> pd.DataFrame:
    """Cached 5-minute decision-time features (delegates to the validated V2 builder)."""
    return _five_min_features(symbol)


def _dmi_adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n)
    pdi = 100 * _rma(pdm, n) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, n) / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _rma(dx, n)
    return atr, pdi, mdi, adx


def _attach_daily_features(d: pd.DataFrame) -> pd.DataFrame:
    """Attach daily DMI/ADX/Bollinger/ATR/SMA/momentum features to a daily OHLCV frame.
    Input columns required: date, open, high, low, close, volume. PIT (shifts only backward)."""
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) < 60:
        return pd.DataFrame()
    h, l, c, v = d["high"], d["low"], d["close"], d["volume"]
    d["dollar_volume"] = c * v
    atr, pdi, mdi, adx = _dmi_adx(h, l, c, 14)
    d["atr"] = atr
    d["pdi"], d["mdi"], d["adx"] = pdi, mdi, adx
    d["di_sep"] = pdi - mdi
    d["di_sep_prev"] = d["di_sep"].shift(1)
    d["di_cross_up"] = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
    d["adx_prev"] = adx.shift(1)
    d["bb_mid"] = c.rolling(20).mean()
    d["bb_std"] = c.rolling(20).std()
    d["bb_dist"] = (c - d["bb_mid"]) / d["bb_mid"]
    d["bb_dist_prev"] = d["bb_dist"].shift(1)
    d["bb_slope"] = d["bb_mid"] - d["bb_mid"].shift(3)
    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma_slope"] = d["sma20"] - d["sma20"].shift(3)
    d["short_mom"] = c - c.shift(2)
    d["trend_pos"] = (c - d["sma20"]) / d["sma20"]
    d["vwap_dist"] = d["trend_pos"]      # daily analogue for directional_score reuse
    d["vwap_slope"] = d["sma_slope"]     # daily analogue for directional_score reuse
    d["realized_vol"] = c.pct_change(fill_method=None).rolling(10).std()
    d["rvol"] = v / v.rolling(20).mean().replace(0, np.nan)
    return d


def daily_frame(symbol: str) -> pd.DataFrame:
    """Resample real 5m RTH bars -> daily OHLCV + daily features. UNADJUSTED for splits -- use only
    for intraday-derived diagnostics, NOT multi-day swing (use daily_frame_adj for that)."""
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = DAILY_CACHE / f"{symbol}_1D_v1.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    fm = five_min_features(symbol)
    if fm.empty:
        return pd.DataFrame()
    g = fm.groupby("date")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()}).reset_index()
    d = _attach_daily_features(d)
    if d.empty:
        return d
    try:
        d.to_parquet(cache)
    except Exception:
        pass
    return d


# ---------------------------------------------------------------- split/div-adjusted daily (swing)
def _fetch_adjusted_daily(symbols: list[str], lookback_days: int = 1900) -> dict[str, pd.DataFrame]:
    """Fetch Alpaca daily bars with adjustment=ALL for `symbols`. Returns {sym: OHLCV df}.
    Network call -- used only to (re)build the adjusted cache; empty dict on failure (disclosed)."""
    try:
        from features import load_alpaca_client
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed, Adjustment
    except Exception as e:
        print(f"[mtf] adjusted-daily fetch unavailable (import): {e}")
        return {}
    try:
        client = load_alpaca_client()
    except Exception as e:
        print(f"[mtf] adjusted-daily fetch unavailable (creds): {e}")
        return {}
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    out: dict[str, pd.DataFrame] = {}
    CH = 40
    for i in range(0, len(symbols), CH):
        chunk = symbols[i:i + CH]
        try:
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start,
                                   end=end, feed=DataFeed.IEX, adjustment=Adjustment.ALL)
            df = client.get_stock_bars(req).df.reset_index()
        except Exception as e:
            print(f"[mtf] adjusted-daily chunk {i//CH} failed: {e}")
            continue
        if df.empty:
            continue
        df["symbol"] = df["symbol"].astype(str)
        for sym, g in df.groupby("symbol"):
            gg = g[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            gg["date"] = pd.to_datetime(gg["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
            gg = gg.drop(columns=["timestamp"])
            for c in ("open", "high", "low", "close", "volume"):
                gg[c] = pd.to_numeric(gg[c], errors="coerce")
            out[sym] = gg.dropna().reset_index(drop=True)
    return out


def daily_frame_adj(symbol: str, allow_fetch: bool = True) -> pd.DataFrame:
    """Split/dividend-ADJUSTED daily OHLCV + daily features (cached). The correct source for
    multi-day swing excursion analysis. Returns empty frame if unavailable (disclosed, not faked)."""
    DAILY_ADJ_CACHE.mkdir(parents=True, exist_ok=True)
    cache = DAILY_ADJ_CACHE / f"{symbol}_1D_adj_v1.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    if not allow_fetch:
        return pd.DataFrame()
    raw = _fetch_adjusted_daily([symbol])
    d = raw.get(symbol)
    if d is None or d.empty:
        return pd.DataFrame()
    d = _attach_daily_features(d)
    if d.empty:
        return d
    try:
        d.to_parquet(cache)
    except Exception:
        pass
    return d


def prefetch_adjusted_daily(symbols: list[str]) -> dict:
    """Batch-build the adjusted-daily cache for `symbols` in one set of chunked calls.
    Returns a coverage dict {fetched, cached, missing}."""
    DAILY_ADJ_CACHE.mkdir(parents=True, exist_ok=True)
    need = []
    for s in symbols:
        if not (DAILY_ADJ_CACHE / f"{s}_1D_adj_v1.parquet").exists():
            need.append(s)
    fetched = []
    if need:
        raw = _fetch_adjusted_daily(need)
        for sym, d in raw.items():
            dd = _attach_daily_features(d)
            if dd.empty:
                continue
            try:
                dd.to_parquet(DAILY_ADJ_CACHE / f"{sym}_1D_adj_v1.parquet")
                fetched.append(sym)
            except Exception:
                pass
    cached = [s for s in symbols if (DAILY_ADJ_CACHE / f"{s}_1D_adj_v1.parquet").exists()]
    missing = [s for s in symbols if s not in cached]
    return {"requested": len(symbols), "newly_fetched": sorted(fetched),
            "cached_available": sorted(cached), "missing": sorted(missing)}


def avg_dollar_volume(symbol: str, lookback_days: int = 30, adjusted: bool = True) -> float:
    """Trailing average daily dollar volume (0.0 if unavailable)."""
    d = daily_frame_adj(symbol, allow_fetch=False) if adjusted else daily_frame(symbol)
    if d.empty or "dollar_volume" not in d:
        return 0.0
    tail = d["dollar_volume"].tail(lookback_days)
    return float(tail.mean()) if len(tail) else 0.0
