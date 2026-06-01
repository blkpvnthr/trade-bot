#!/usr/bin/env python3
"""
rl_alpaca_bar_enrichment.py — Bar-context enrichment for RL replay events.
==========================================================================

Fills the "seriesless" gap: many replay events (especially alpha) carry only a symbol +
timestamp + outcome and therefore had neutral multi-timeframe / catalyst features. This
module reconstructs the bar context and computes the trend / volume / volatility / regime
features the RL agents consume:

    mtf_trend_fast | mtf_trend_mid | mtf_trend_slow | mtf_alignment
    rel_volume | vwap_distance | atr_pct | catalyst_strength
    spy_regime | sector_regime

Two data paths, both OFFLINE w.r.t. trading:
  * intraday features (5m/15m/1h trend, rel-vol, VWAP, ATR, catalyst) are computed from a
    LOCAL 1-minute bar frame when one is supplied (no network), or
  * daily / SPY / sector-ETF regime context is pulled from Alpaca HISTORICAL bars
    (read-only `alpaca.data.historical` via rl_alpaca_backfill.fetch_bars) and CACHED under
    data/cache/alpaca_bars/ so repeated runs never re-hit the API.

SAFETY: read-only market data only. NEVER imports AlpacaGateway / alpaca.trading, never
submits an order. Defaults to the IEX feed; SIP is never used by default. Missing/partial
data degrades to NEUTRAL feature values (0.0) with a warning — never fabricated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import rl_common as rc

logger = logging.getLogger(__name__)

CACHE_DIR = rc.PROJECT_ROOT / "data" / "cache" / "alpaca_bars"

# The enrichment fields, with their neutral defaults (used on missing/partial data).
# The first 10 are the original trend/volume/volatility/regime block; the ga_* fields are the
# "Generating Alpha" hybrid technical block (EMA50/200 ratio, MACD histogram, RSI, Bollinger
# %b, per-stock SMA20 regime switch) computed from daily closes. ga_sentiment_st is filled
# separately by the offline sentiment loader (rl_strategy_overlay), not from bars.
ENRICH_DEFAULTS: Dict[str, float] = {
    "mtf_trend_fast": 0.0, "mtf_trend_mid": 0.0, "mtf_trend_slow": 0.0, "mtf_alignment": 0.0,
    "rel_volume": 1.0, "vwap_distance": 0.0, "atr_pct": 0.0, "catalyst_strength": 0.0,
    "spy_regime": 0.0, "sector_regime": 0.0,
    "ga_ema_ratio": 0.0, "ga_macd_hist": 0.0, "ga_rsi": 50.0, "ga_bollinger_pctb": 0.5,
    "ga_regime_switch": 0.0,
    # Opening Range entry assessment (Bysshe handbook) — computed from the 09:30–10:00 ET slice.
    "or_position": 0.0, "or_breakout": 0.0, "or_volume_confirm": 0.0, "or_level_quality": 0.0,
}


def _opening_range_slice(df, or_minutes: int = 30):
    """Return the (highs, lows, closes, volumes) of the 09:30–(09:30+or_minutes) ET window from
    an intraday frame with a DatetimeIndex, plus the day's baseline mean volume. Returns None if
    the frame has no usable session-anchored index. Never raises (read-only, defensive)."""
    try:
        import pandas as pd
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex) or len(df) == 0:
            return None
        et = idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("UTC").tz_convert("America/New_York")
        minutes = et.hour * 60 + et.minute
        in_or = (minutes >= 9 * 60 + 30) & (minutes < 9 * 60 + 30 + or_minutes)
        win = df[in_or]
        if len(win) < 2:
            return None
        baseline = float(df["volume"].mean()) if "volume" in df else 0.0
        return ([float(x) for x in win["high"]], [float(x) for x in win["low"]],
                [float(x) for x in win["close"]], [float(x) for x in win["volume"]], baseline)
    except Exception:
        return None

# Minimal symbol→sector-ETF map (extensible). Unknown symbols → sector_regime neutral.
SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK", "ACMR": "XLK",
    "ADEA": "XLK", "AKAM": "XLK", "AMAT": "XLK", "INTC": "XLK", "MU": "XLK", "QCOM": "XLK",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY", "DECK": "XLY",
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF", "GS": "XLF", "BNS": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "MRK": "XLV", "PENN": "XLY",
    "XOM": "XLE", "CVX": "XLE", "PLUG": "XLE", "FCEL": "XLE", "IPWR": "XLI", "AIRJ": "XLI",
    "KO": "XLP", "PG": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "SPY": "SPY", "QQQ": "XLK",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF.values()))


# ── small numeric helpers (pure) ─────────────────────────────────────────────────────
def _sma(xs: List[float], w: int) -> Optional[float]:
    return rc._sma_last(xs, w)


def _trend(closes: List[float], window: int) -> float:
    """Signed price-vs-SMA trend over `window` (0.0 if not computable)."""
    cs = [float(c) for c in closes if c is not None]
    if len(cs) < 2 or cs[-1] <= 0:
        return 0.0
    sma = _sma(cs, min(window, len(cs)))
    return round((cs[-1] - sma) / sma, 6) if (sma and sma > 0) else 0.0


def _atr_pct(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """ATR(period) as a fraction of the last close (0.0 if not computable)."""
    try:
        h, l, c = [list(map(float, x)) for x in (highs, lows, closes)]
        n = len(c)
        if n < 2:
            return 0.0
        trs = []
        for i in range(1, n):
            trs.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        seg = trs[-period:] if len(trs) >= period else trs
        atr = sum(seg) / len(seg) if seg else 0.0
        return round(atr / c[-1], 6) if c[-1] > 0 else 0.0
    except Exception:
        return 0.0


def _vwap_distance(highs, lows, closes, volumes) -> float:
    """(last_close − session VWAP) / VWAP using typical price × volume (0.0 if not computable)."""
    try:
        h, l, c, v = [list(map(float, x)) for x in (highs, lows, closes, volumes)]
        num = sum(((hi + lo + cl) / 3.0) * vol for hi, lo, cl, vol in zip(h, l, c, v))
        den = sum(v)
        if den <= 0:
            return 0.0
        vwap = num / den
        return round((c[-1] - vwap) / vwap, 6) if vwap > 0 else 0.0
    except Exception:
        return 0.0


def _rel_volume(volumes: List[float], window: int = 20, recent: int = 3) -> float:
    """Relative volume = mean(recent bars) / mean(window). 1.0 = average (neutral)."""
    try:
        v = [float(x) for x in volumes if x is not None]
        if len(v) < 2:
            return 1.0
        base = v[-window:] if len(v) >= window else v
        mean_base = sum(base) / len(base)
        rec = v[-recent:] if len(v) >= recent else v
        mean_rec = sum(rec) / len(rec)
        return round(mean_rec / mean_base, 4) if mean_base > 0 else 1.0
    except Exception:
        return 1.0


def _catalyst_from_rvol(rel_volume: float) -> float:
    """Map relative volume → catalyst strength in [0, 1] (rvol 1→0, 3→1) — thesis §3.9."""
    return round(max(0.0, min(1.0, (float(rel_volume) - 1.0) / 2.0)), 4)


def regime_score(closes: List[float]) -> float:
    """Daily regime in {-1, -0.5, 0, 0.5, 1} from price vs the 50/200-day SMAs."""
    cs = [float(c) for c in closes if c is not None]
    if len(cs) < 2 or cs[-1] <= 0:
        return 0.0
    price = cs[-1]
    s = 0.0
    sma50 = _sma(cs, min(50, len(cs)))
    sma200 = _sma(cs, min(200, len(cs)))
    if sma50 and sma50 > 0:
        s += 0.5 if price > sma50 else -0.5
    if sma200 and sma200 > 0:
        s += 0.5 if price > sma200 else -0.5
    return round(s, 4)


# ── pure feature assembly (unit-testable on synthetic frames) ────────────────────────
def bar_features(frames: Dict[str, Any], spy_daily=None, sector_daily=None) -> Dict[str, float]:
    """
    Assemble the 10 enrichment features from bar frames. `frames` maps timeframe →
    DataFrame[open,high,low,close,volume] (any subset of {"5m","15m","1h","daily"}); each
    missing/empty frame leaves its component at the neutral default. `spy_daily` /
    `sector_daily` are daily DataFrames (or close-lists) for the regime scores. Never raises.
    """
    out = dict(ENRICH_DEFAULTS)

    def closes(df):
        try:
            return [float(x) for x in df["close"]]
        except Exception:
            try:
                return [float(x) for x in df]      # already a close-list
            except Exception:
                return []

    f5 = frames.get("5m"); f15 = frames.get("15m"); f1h = frames.get("1h"); fd = frames.get("daily")
    tf_trends: List[float] = []
    if f5 is not None and len(f5):
        out["mtf_trend_fast"] = _trend(closes(f5), 20)
        tf_trends.append(out["mtf_trend_fast"])
        try:
            out["rel_volume"] = _rel_volume(list(f5["volume"]))
            out["vwap_distance"] = _vwap_distance(f5["high"], f5["low"], f5["close"], f5["volume"])
            out["atr_pct"] = _atr_pct(f5["high"], f5["low"], f5["close"])
        except Exception:
            pass
    if f15 is not None and len(f15):
        tf_trends.append(_trend(closes(f15), 20))
    if f1h is not None and len(f1h):
        out["mtf_trend_mid"] = _trend(closes(f1h), 20)
        tf_trends.append(out["mtf_trend_mid"])
    if fd is not None and len(fd):
        dcs = closes(fd)
        out["mtf_trend_slow"] = _trend(dcs, 50)
        tf_trends.append(out["mtf_trend_slow"])
        # "Generating Alpha" hybrid technical block, computed from daily closes (paper §C).
        out["ga_ema_ratio"] = rc.ema50_200_ratio(dcs)
        out["ga_macd_hist"] = rc.macd_histogram(dcs)
        out["ga_rsi"] = rc.rsi(dcs)
        out["ga_bollinger_pctb"] = rc.bollinger_pctb(dcs)
        out["ga_regime_switch"] = rc.regime_switch(dcs)

    signs = [1.0 if t > 0 else (-1.0 if t < 0 else 0.0) for t in tf_trends if t != 0.0]
    if signs:
        out["mtf_alignment"] = round(sum(signs) / len(signs), 4)
    out["catalyst_strength"] = _catalyst_from_rvol(out["rel_volume"])
    # Opening Range entry assessment (Bysshe handbook): compute from the session-anchored
    # intraday frame's first-30-min slice. Prefers 5m, falls back to 15m. Neutral if not anchored.
    or_src = f5 if (f5 is not None and len(f5)) else (f15 if (f15 is not None and len(f15)) else None)
    orw = _opening_range_slice(or_src) if or_src is not None else None
    if orw is not None:
        hs, ls, cs, vs, baseline = orw
        # Current price = latest close of the FULL frame, so or_position/or_breakout measure
        # where price is NOW vs the OR levels (H/L/volume/quality come from the OR window).
        try:
            last_price = float(or_src["close"].iloc[-1])
        except Exception:
            last_price = cs[-1] if cs else None
        atr_abs = out["atr_pct"] * (last_price or 0.0) if out["atr_pct"] else 0.0
        orf = rc.opening_range_features(hs, ls, cs, vs, price=last_price,
                                        baseline_volume=baseline, atr_abs=atr_abs)
        out.update(orf)
    if spy_daily is not None and len(spy_daily):
        out["spy_regime"] = regime_score(closes(spy_daily))
    if sector_daily is not None and len(sector_daily):
        out["sector_regime"] = regime_score(closes(sector_daily))
    return out


# ── Alpaca bar cache (read-only, IEX default; never SIP by default) ──────────────────
class BarCache:
    """Cached read-only historical bars. Cache files: data/cache/alpaca_bars/{sym}_{tf}.parquet.

    `fetch_fn(symbols, start, end, timeframe)` defaults to the read-only
    rl_alpaca_backfill.fetch_bars; tests inject a fake to exercise hit/miss without network.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, fetch_fn: Optional[Callable] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if fetch_fn is None:
            from rl_alpaca_backfill import fetch_bars as _fb
            fetch_fn = _fb
        self.fetch_fn = fetch_fn
        self.hits = 0
        self.misses = 0

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = str(symbol).replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}_{timeframe}.parquet"

    def _load(self, symbol: str, timeframe: str):
        p = self._path(symbol, timeframe)
        if not p.exists():
            return None
        try:
            import pandas as pd
            return pd.read_parquet(p)
        except Exception:
            return None

    def _save(self, symbol: str, timeframe: str, df) -> None:
        try:
            df.to_parquet(self._path(symbol, timeframe))
        except Exception as exc:
            logger.debug("bar cache save failed for %s/%s: %s", symbol, timeframe, exc)

    def get(self, symbol: str, timeframe: str, start, end):
        """Return cached/fetched bars for [start, end]; empty DataFrame on failure. Read-only."""
        import pandas as pd
        cached = self._load(symbol, timeframe)
        if cached is not None and not cached.empty:
            try:
                idx = cached.index
                if idx.min() <= start and idx.max() >= end:
                    self.hits += 1
                    return cached.loc[start:end]
            except Exception:
                pass
        # miss → read-only fetch, merge into cache
        self.misses += 1
        try:
            fetched = (self.fetch_fn([symbol], start, end, timeframe=timeframe) or {}).get(symbol)
        except Exception as exc:
            logger.warning("bar fetch failed for %s/%s (%s) — neutral", symbol, timeframe, exc)
            fetched = None
        if fetched is None or len(fetched) == 0:
            if cached is not None:
                try:
                    return cached.loc[start:end]
                except Exception:
                    pass
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        merged = fetched if cached is None else pd.concat([cached, fetched])
        try:
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass
        self._save(symbol, timeframe, merged)
        try:
            return merged.loc[start:end]
        except Exception:
            return merged


# ── orchestration over the daily/regime context (used by the backfill ingestion) ─────
class RegimeContext:
    """Prefetches & caches daily bars for symbols + SPY + sector ETFs, then scores regimes.

    One batched read-only Alpaca call per group (not per symbol). Graceful: if Alpaca is
    unavailable, every regime score falls back to neutral 0.0.
    """

    def __init__(self, cache: Optional[BarCache] = None, lookback_days: int = 420):
        self.cache = cache or BarCache()
        self.lookback_days = lookback_days
        self._daily: Dict[str, Any] = {}       # symbol → daily DataFrame
        self._end = datetime.now(timezone.utc) - timedelta(minutes=20)
        self._start = self._end - timedelta(days=lookback_days)

    def prefetch(self, symbols: List[str]) -> None:
        wanted = sorted(set([str(s).upper() for s in symbols]) | {"SPY"} | set(SECTOR_ETFS))
        for s in wanted:
            try:
                self._daily[s] = self.cache.get(s, "day", self._start, self._end)
            except Exception:
                self._daily[s] = None

    def _closes_upto(self, symbol: str, ts) -> List[float]:
        df = self._daily.get(str(symbol).upper())
        if df is None or len(df) == 0:
            return []
        try:
            import pandas as pd
            t = pd.Timestamp(ts)
            if t.tz is None:
                t = t.tz_localize("UTC")
            sub = df[df.index <= t]
            return [float(x) for x in (sub["close"] if len(sub) else df["close"])]
        except Exception:
            try:
                return [float(x) for x in df["close"]]
            except Exception:
                return []

    def regimes(self, symbol: str, ts) -> Dict[str, float]:
        spy = self._closes_upto("SPY", ts)
        etf = SECTOR_ETF.get(str(symbol).upper())
        sec = self._closes_upto(etf, ts) if etf else []
        slow = self._closes_upto(symbol, ts)
        # "Generating Alpha" per-stock daily technical block (paper §C/§F): computed on the
        # symbol's own daily closes up to the event timestamp (no look-ahead). Neutral on thin.
        return {"spy_regime": regime_score(spy), "sector_regime": regime_score(sec),
                "daily_trend_slow": _trend(slow, 50),
                "ga_ema_ratio": rc.ema50_200_ratio(slow), "ga_macd_hist": rc.macd_histogram(slow),
                "ga_rsi": rc.rsi(slow), "ga_bollinger_pctb": rc.bollinger_pctb(slow),
                "ga_regime_switch": rc.regime_switch(slow)}


def enrich_events(events: List[dict], regime_ctx: Optional[RegimeContext] = None,
                  add_regime: bool = True) -> Dict[str, int]:
    """
    Backward-compatibly add regime fields (spy_regime / sector_regime, and mtf_trend_slow when
    still neutral) to alpha events that carry a real symbol + timestamp. Intraday MTF / rel-vol
    / VWAP / ATR / catalyst are expected to already be on the event (computed at ingest from
    local 1m bars). Returns {enriched, neutral} counts. Never raises; safe on missing Alpaca.
    """
    counts = {"enriched": 0, "neutral": 0, "skipped_non_symbol": 0}
    alpha = [e for e in events if e.get("event_type") in ("alpha_entry", "alpha_exit")]
    if not alpha or not add_regime:
        return counts
    syms = []
    for e in alpha:
        s = str(e.get("symbol") or "")
        if s and not s.isdigit() and s.upper() != "REPLAY":
            syms.append(s)
    if not syms:
        counts["skipped_non_symbol"] = len(alpha)
        return counts
    ctx = regime_ctx or RegimeContext()
    try:
        ctx.prefetch(syms)
    except Exception as exc:
        logger.warning("regime prefetch failed (%s) — regimes stay neutral", exc)
    for e in alpha:
        s = str(e.get("symbol") or "")
        if not s or s.isdigit() or s.upper() == "REPLAY":
            counts["skipped_non_symbol"] += 1
            continue
        try:
            r = ctx.regimes(s, e.get("timestamp"))
            e.setdefault("spy_regime", 0.0)
            e.setdefault("sector_regime", 0.0)
            e["spy_regime"] = r["spy_regime"]
            e["sector_regime"] = r["sector_regime"]
            if abs(rc.to_f(e.get("mtf_trend_slow"))) < 1e-9 and r["daily_trend_slow"] != 0.0:
                e["mtf_trend_slow"] = r["daily_trend_slow"]
            # Fill GA daily-technical fields when the event still carries the neutral default
            # (intraday ingest leaves them neutral; the per-stock daily series fills them here).
            for k, neutral in (("ga_ema_ratio", 0.0), ("ga_macd_hist", 0.0), ("ga_rsi", 50.0),
                               ("ga_bollinger_pctb", 0.5), ("ga_regime_switch", 0.0)):
                if abs(rc.to_f(e.get(k), neutral) - neutral) < 1e-9 and abs(r[k] - neutral) > 1e-9:
                    e[k] = r[k]
            if abs(r["spy_regime"]) > 1e-9 or abs(r["sector_regime"]) > 1e-9:
                counts["enriched"] += 1
            else:
                counts["neutral"] += 1
        except Exception:
            counts["neutral"] += 1
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bc = BarCache()
    print(f"BarCache ready at {bc.cache_dir} (hits={bc.hits} misses={bc.misses})")
    print(f"Sector ETFs tracked: {SECTOR_ETFS}")
