"""
streaming_indicator_engine.py — Real-Time Indicator Infrastructure Layer
========================================================================

Maintains per-symbol technical indicator state, updated candle-by-candle, so the
Alpha Call Engine, Buffett Accumulation Layer, Portfolio Research Lab, Adaptive
Capital Allocator, Probability Model and Regime Specialist models can all read
ONE shared real-time indicator state instead of each recomputing from raw bars.

This is INDICATOR INFRASTRUCTURE, not a trading strategy.
  - No streaming indicator may directly generate an order.
  - Alpha decides tactical trades. Buffett decides long-term accumulation.
  - Research Lab measures performance. RiskManager decides permission.

Design
------
  * Optional dependency `streaming_indicators` (PyPI: streaming-indicators).
    When installed, the supported incremental indicators (RSI/SMA/EMA/VWAP/ATR/
    SuperTrend) are maintained as true streaming objects (source="streaming").
  * When the package is unavailable — or for indicators it does not implement
    (MACD/Stochastic/Williams %R/CCI/MFI/Aroon) — the engine computes the value
    from an internal rolling candle buffer (source="static_fallback").
  * Every failure is caught: the engine NEVER raises into the trading loop. On
    any error it marks fallback_used=True, uses the static value, and continues.
  * Stale state (older than STREAMING_INDICATOR_STALE_SEC) is flagged so callers
    can fall back to their own bar-based calculations rather than trust it blindly.

Candle format (dict)::

    {"timestamp": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}

`update_many` also accepts a pandas DataFrame per symbol (the shape produced by
run.fetch_1m_bars) and normalises it internally.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Shared atomic JSON writer (single implementation in opportunity_utils); guarded so
# this module stays import-safe standalone. Falls back to an atomic tmp+rename write.
try:
    from opportunity_utils import write_json_safe as _write_json_safe
except Exception:
    def _write_json_safe(path, data, log_fn=None) -> bool:
        import json as _json, os as _os
        from pathlib import Path as _P
        p = _P(path); tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(_json.dumps(data, indent=2, default=str), encoding="utf-8")
        _os.replace(tmp, p)
        return True


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

# How old (seconds) per-symbol state may be before it is considered stale and
# callers should fall back to bar-based calculations rather than trust it.
STALE_SEC = float(os.getenv("STREAMING_INDICATOR_STALE_SEC", "180"))
# Rolling candle buffer length — must cover the longest lookback (SMA-200).
BUFFER_LEN = int(os.getenv("STREAMING_INDICATOR_BUFFER", "260"))

SOURCE_STREAMING = "streaming"
SOURCE_STATIC = "static_fallback"

# Stable indicator key order (also defines indicators_tracked in the health report).
INDICATOR_KEYS: List[str] = [
    "rsi_14", "sma_20", "sma_50", "sma_200", "ema_20", "vwap",
    "macd", "macd_signal", "macd_hist", "atr_14",
    "stoch_k", "stoch_d", "williams_r", "cci", "mfi",
    "aroon_up", "aroon_down", "aroon_osc", "supertrend", "supertrend_dir",
    "dmi_plus", "dmi_minus", "adx",
]

# Optional streaming-indicators package — probed once at import (never fatal).
try:  # pragma: no cover - exercised via STREAMING_PACKAGE_AVAILABLE
    import streaming_indicators as _si  # type: ignore
    STREAMING_PACKAGE_AVAILABLE = True
except Exception:  # ImportError or any load failure
    _si = None
    STREAMING_PACKAGE_AVAILABLE = False
    logger.warning(
        "streaming_indicators package unavailable — using static_fallback "
        "indicator calculations (install `streaming-indicators` to enable)."
    )


# ───────────────────────────── static indicator math ─────────────────────────────
# Pure-Python implementations used when the streaming package is missing or does
# not implement a given indicator. Operate on plain lists; return None until warm.

def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    """EMA aligned to `values` (None during warm-up). Seeds on the SMA of period 1."""
    n = len(values)
    if n < period:
        return [None] * n
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    out: List[Optional[float]] = [None] * (period - 1) + [ema]
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
        out.append(ema)
    return out


def _ema(values: List[float], period: int) -> Optional[float]:
    s = _ema_series(values, period)
    return s[-1] if s else None


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) — any may be None during warm-up."""
    if len(closes) < slow:
        return None, None, None
    fast_s = _ema_series(closes, fast)
    slow_s = _ema_series(closes, slow)
    macd_line = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(fast_s, slow_s)
    ]
    valid = [m for m in macd_line if m is not None]
    macd_val = macd_line[-1]
    sig_val = _ema(valid, signal) if len(valid) >= signal else None
    hist = (macd_val - sig_val) if (macd_val is not None and sig_val is not None) else None
    return macd_val, sig_val, hist


def _true_ranges(highs, lows, closes) -> List[float]:
    trs: List[float] = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return trs


def _atr(highs, lows, closes, period: int = 14) -> Optional[float]:
    if len(closes) <= period:
        return None
    trs = _true_ranges(highs, lows, closes)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _stochastic(highs, lows, closes, k_period: int = 14, d_period: int = 3):
    n = len(closes)
    if n < k_period:
        return None, None
    ks: List[float] = []
    for i in range(k_period - 1, n):
        hh = max(highs[i - k_period + 1:i + 1])
        ll = min(lows[i - k_period + 1:i + 1])
        rng = hh - ll
        ks.append(100.0 * (closes[i] - ll) / rng if rng > 0 else 50.0)
    k_val = ks[-1]
    d_val = sum(ks[-d_period:]) / min(d_period, len(ks))
    return k_val, d_val


def _williams_r(highs, lows, closes, period: int = 14) -> Optional[float]:
    if len(closes) < period:
        return None
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    rng = hh - ll
    if rng <= 0:
        return -50.0
    return -100.0 * (hh - closes[-1]) / rng


def _cci(highs, lows, closes, period: int = 20) -> Optional[float]:
    tps = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes))]
    if len(tps) < period:
        return None
    window = tps[-period:]
    sma = sum(window) / period
    mad = sum(abs(x - sma) for x in window) / period
    if mad == 0:
        return 0.0
    return (tps[-1] - sma) / (0.015 * mad)


def _mfi(highs, lows, closes, volumes, period: int = 14) -> Optional[float]:
    n = len(closes)
    if n <= period:
        return None
    tps = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
    pos = neg = 0.0
    for i in range(n - period, n):
        rmf = tps[i] * volumes[i]
        if tps[i] > tps[i - 1]:
            pos += rmf
        elif tps[i] < tps[i - 1]:
            neg += rmf
    if neg == 0:
        return 100.0 if pos > 0 else 50.0
    mfr = pos / neg
    return 100.0 - 100.0 / (1.0 + mfr)


def _dmi(highs, lows, closes, period: int = 14):
    """Wilder DMI/ADX → (plus_di, minus_di, adx). None until warm. Pure; never raises."""
    n = len(closes)
    if n <= period + 1:
        return None, None, None
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    if len(trs) < period:
        return None, None, None
    # Wilder smoothing
    atr = sum(trs[:period]); sp = sum(plus_dm[:period]); sm = sum(minus_dm[:period])
    dxs = []
    for i in range(period, len(trs)):
        atr = atr - atr / period + trs[i]
        sp = sp - sp / period + plus_dm[i]
        sm = sm - sm / period + minus_dm[i]
        if atr <= 0:
            continue
        pdi = 100.0 * sp / atr
        mdi = 100.0 * sm / atr
        denom = pdi + mdi
        dxs.append(100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0)
    if not dxs:
        return None, None, None
    pdi = 100.0 * sp / atr if atr > 0 else 0.0
    mdi = 100.0 * sm / atr if atr > 0 else 0.0
    adx = sum(dxs[-period:]) / min(period, len(dxs))
    return round(pdi, 4), round(mdi, 4), round(adx, 4)


def _aroon(highs, lows, period: int = 25):
    """Returns (aroon_up, aroon_down, aroon_oscillator)."""
    if len(highs) < period + 1:
        return None, None, None
    win_h = highs[-(period + 1):]
    win_l = lows[-(period + 1):]
    since_high = (len(win_h) - 1) - max(range(len(win_h)), key=lambda i: win_h[i])
    since_low = (len(win_l) - 1) - min(range(len(win_l)), key=lambda i: win_l[i])
    up = 100.0 * (period - since_high) / period
    down = 100.0 * (period - since_low) / period
    return up, down, up - down


def _supertrend(highs, lows, closes, period: int = 10, mult: float = 3.0):
    """Returns (band_value, direction) where direction is 1 (up) or -1 (down)."""
    n = len(closes)
    if n < period + 1:
        return None, None
    trs = [0.0] + _true_ranges(highs, lows, closes)  # index-aligned with closes
    atr: List[Optional[float]] = [None] * n
    atr[period] = sum(trs[1:period + 1]) / period
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    direction: List[Optional[int]] = [None] * n
    st: List[Optional[float]] = [None] * n
    for i in range(period, n):
        hl2 = (highs[i] + lows[i]) / 2.0
        basic_u = hl2 + mult * atr[i]
        basic_l = hl2 - mult * atr[i]
        if i == period:
            upper[i], lower[i], direction[i] = basic_u, basic_l, 1
            st[i] = basic_l
            continue
        upper[i] = basic_u if (basic_u < upper[i - 1] or closes[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = basic_l if (basic_l > lower[i - 1] or closes[i - 1] < lower[i - 1]) else lower[i - 1]
        if closes[i] > upper[i - 1]:
            direction[i] = 1
        elif closes[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        st[i] = lower[i] if direction[i] == 1 else upper[i]
    return st[-1], direction[-1]


def _vwap(highs, lows, closes, volumes) -> Optional[float]:
    num = 0.0
    den = 0.0
    for i in range(len(closes)):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        num += tp * volumes[i]
        den += volumes[i]
    if den <= 0:
        return closes[-1] if closes else None
    return num / den


# ───────────────────────────── per-symbol state ─────────────────────────────

class _SymbolState:
    """Rolling buffer + (optional) streaming objects + latest computed values."""

    __slots__ = (
        "symbol", "buffer", "last_ts", "indicators", "indicator_sources",
        "updated_at", "updated_ts", "bars_seen", "_stream",
    )

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=BUFFER_LEN)
        self.last_ts: Any = None
        self.indicators: Dict[str, Any] = {}
        self.indicator_sources: Dict[str, str] = {}
        self.updated_at: Optional[str] = None
        self.updated_ts: float = 0.0
        self.bars_seen: int = 0
        self._stream: Dict[str, Any] = {}
        if STREAMING_PACKAGE_AVAILABLE:
            self._init_stream()

    def _init_stream(self) -> None:
        """Instantiate streaming objects for the indicators the package implements."""
        def make(name: str, *args):
            cls = getattr(_si, name, None)
            if cls is None:
                return None
            try:
                return cls(*args)
            except Exception:
                return None
        self._stream = {
            "rsi_14": make("RSI", 14),
            "sma_20": make("SMA", 20),
            "sma_50": make("SMA", 50),
            "sma_200": make("SMA", 200),
            "ema_20": make("EMA", 20),
            "vwap": make("VWAP"),
            "atr_14": make("ATR", 14),
            "supertrend": make("SuperTrend", 10, 3.0),
        }

    def _feed_stream(self, candle: Dict[str, float]) -> None:
        """Feed ONE candle to each live streaming object (best-effort; never raises)."""
        if not self._stream:
            return
        close = candle["close"]
        for key in ("rsi_14", "sma_20", "sma_50", "sma_200", "ema_20"):
            obj = self._stream.get(key)
            if obj is not None:
                try:
                    obj.update(close)
                except Exception:
                    self._stream[key] = None  # disable on first failure → static fallback
        for key in ("vwap", "atr_14", "supertrend"):
            obj = self._stream.get(key)
            if obj is not None:
                try:
                    obj.update(candle)
                except Exception:
                    self._stream[key] = None


# ───────────────────────────── the engine ─────────────────────────────

class StreamingIndicatorEngine:
    """Maintains per-symbol streaming technical indicators updated candle-by-candle.

    Public API:
        update_symbol(symbol, candle)
        update_many(bars_by_symbol)   # values may be candle-lists OR pandas DataFrames
        get(symbol)                   # current per-symbol state (fresh age/staleness)
        snapshot()                    # {symbol: state} for all tracked symbols
        write_report(out_dir=None)    # snapshot + health JSON to daily outputs
    """

    def __init__(self, stale_sec: float = STALE_SEC):
        self.stale_sec = float(stale_sec)
        self.state_by_symbol: Dict[str, _SymbolState] = {}
        self.update_errors: int = 0
        self.streaming_package_available: bool = STREAMING_PACKAGE_AVAILABLE
        # Phase 4 feed provenance + update accounting (advisory diagnostics only).
        self.alpaca_feed_used: bool = False
        self.alpaca_feed_type: Optional[str] = None
        self.indicator_updates: int = 0
        self.stale_updates: int = 0
        self.fallback_updates: int = 0

    def set_feed(self, feed_type: Optional[str], used: bool = True) -> None:
        """Record which Alpaca feed supplied the bars (e.g. 'iex'). Advisory only."""
        self.alpaca_feed_type = feed_type
        self.alpaca_feed_used = bool(used and feed_type not in (None, "fallback"))

    # ── ingestion ──────────────────────────────────────────────────────────
    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _now_ts() -> float:
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _normalise(bars: Any) -> List[Dict[str, float]]:
        """Coerce a DataFrame / list-of-dicts / single dict into a list of candles."""
        if bars is None:
            return []
        # pandas DataFrame (duck-typed so pandas stays an optional import here)
        if hasattr(bars, "iterrows") and hasattr(bars, "columns"):
            out: List[Dict[str, float]] = []
            for ts, row in bars.iterrows():
                try:
                    out.append({
                        "timestamp": ts,
                        "open": _f(row.get("open")),
                        "high": _f(row.get("high")),
                        "low": _f(row.get("low")),
                        "close": _f(row.get("close")),
                        "volume": _f(row.get("volume")) or 0.0,
                    })
                except Exception:
                    continue
            return out
        if isinstance(bars, dict):
            return [bars]
        if isinstance(bars, (list, tuple)):
            return [c for c in bars if isinstance(c, dict)]
        return []

    def _clean_candle(self, candle: Dict[str, Any]) -> Optional[Dict[str, float]]:
        c = {
            "timestamp": candle.get("timestamp"),
            "open": _f(candle.get("open")),
            "high": _f(candle.get("high")),
            "low": _f(candle.get("low")),
            "close": _f(candle.get("close")),
            "volume": _f(candle.get("volume")) or 0.0,
        }
        # Reject fabricated/empty candles — never invent indicators from missing data.
        if c["close"] is None:
            return None
        if c["high"] is None:
            c["high"] = c["close"]
        if c["low"] is None:
            c["low"] = c["close"]
        if c["open"] is None:
            c["open"] = c["close"]
        return c

    def _is_new(self, st: _SymbolState, ts: Any) -> bool:
        if ts is None or st.last_ts is None:
            return True
        try:
            return ts > st.last_ts
        except Exception:
            return True  # uncomparable types → treat as new

    def update_symbol(self, symbol: str, candle: Dict[str, Any]) -> None:
        """Update a single symbol with one candle dict. Fail-safe (never raises)."""
        try:
            self._ingest(symbol, [candle])
        except Exception as exc:
            self.update_errors += 1
            logger.debug("update_symbol(%s) failed: %s", symbol, exc)

    def update_many(self, bars_by_symbol: Dict[str, Any]) -> None:
        """Update many symbols. Each value may be candles or a pandas DataFrame."""
        if not bars_by_symbol:
            return
        for symbol, bars in bars_by_symbol.items():
            try:
                self._ingest(symbol, self._normalise(bars))
            except Exception as exc:
                self.update_errors += 1
                logger.debug("update_many(%s) failed: %s", symbol, exc)

    def _ingest(self, symbol: str, candles: List[Dict[str, Any]]) -> None:
        if not symbol:
            return
        st = self.state_by_symbol.get(symbol)
        created = st is None
        if st is None:
            st = _SymbolState(symbol)
        new_candles: List[Dict[str, float]] = []
        for raw in candles:
            c = self._clean_candle(raw)
            if c is None:
                continue
            if not self._is_new(st, c["timestamp"]):
                continue
            st.buffer.append(c)
            st.last_ts = c["timestamp"]
            st.bars_seen += 1
            new_candles.append(c)
        if not new_candles:
            # No usable candles — never register a fabricated/empty symbol state.
            return
        if created:
            self.state_by_symbol[symbol] = st
        # Feed streaming objects per new candle (keeps their incremental state correct),
        # then compute the latest values ONCE from the final buffer (no duplicate work).
        for c in new_candles:
            st._feed_stream(c)
        self._recompute(st)
        st.updated_at = self._now_iso()
        st.updated_ts = self._now_ts()
        # Update accounting (advisory): one update per recompute; flag static-fallback use.
        self.indicator_updates += 1
        if any(v == SOURCE_STATIC for v in st.indicator_sources.values()):
            self.fallback_updates += 1

    # ── computation ──────────────────────────────────────────────────────────
    @staticmethod
    def _stream_value(st: _SymbolState, key: str) -> Optional[float]:
        """Read the current value off a streaming object (None if absent/unusable)."""
        obj = st._stream.get(key) if st._stream else None
        if obj is None:
            return None
        for attr in ("value", "vwap", "rsi"):
            if hasattr(obj, attr):
                return _f(getattr(obj, attr))
        return None

    def _recompute(self, st: _SymbolState) -> None:
        buf = list(st.buffer)
        closes = [c["close"] for c in buf]
        highs = [c["high"] for c in buf]
        lows = [c["low"] for c in buf]
        vols = [c["volume"] for c in buf]

        ind: Dict[str, Any] = {}
        src: Dict[str, str] = {}

        def assign(key: str, stream_key: Optional[str], static_val):
            """Prefer a usable streaming value; else fall back to the static value."""
            val = self._stream_value(st, stream_key) if (stream_key and STREAMING_PACKAGE_AVAILABLE) else None
            if val is not None:
                ind[key] = val
                src[key] = SOURCE_STREAMING
            else:
                ind[key] = static_val
                src[key] = SOURCE_STATIC

        assign("rsi_14", "rsi_14", _rsi(closes, 14))
        assign("sma_20", "sma_20", _sma(closes, 20))
        assign("sma_50", "sma_50", _sma(closes, 50))
        assign("sma_200", "sma_200", _sma(closes, 200))
        assign("ema_20", "ema_20", _ema(closes, 20))
        assign("vwap", "vwap", _vwap(highs, lows, closes, vols))
        assign("atr_14", "atr_14", _atr(highs, lows, closes, 14))

        # SuperTrend (streaming returns (dir, band)); static returns (band, dir).
        st_band = st_dir = None
        used_stream_st = False
        if STREAMING_PACKAGE_AVAILABLE and st._stream.get("supertrend") is not None:
            obj = st._stream["supertrend"]
            try:
                st_dir = _f(getattr(obj, "super_trend", None))
                st_band = _f(getattr(obj, "final_band", None))
                used_stream_st = st_band is not None or st_dir is not None
            except Exception:
                used_stream_st = False
        if not used_stream_st:
            st_band, st_dir = _supertrend(highs, lows, closes, 10, 3.0)
        ind["supertrend"] = st_band
        ind["supertrend_dir"] = st_dir
        src["supertrend"] = SOURCE_STREAMING if used_stream_st else SOURCE_STATIC
        src["supertrend_dir"] = src["supertrend"]

        # Indicators the package does not implement → always static_fallback.
        macd_val, macd_sig, macd_hist = _macd(closes)
        ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd_val, macd_sig, macd_hist
        k_val, d_val = _stochastic(highs, lows, closes)
        ind["stoch_k"], ind["stoch_d"] = k_val, d_val
        ind["williams_r"] = _williams_r(highs, lows, closes)
        ind["cci"] = _cci(highs, lows, closes)
        ind["mfi"] = _mfi(highs, lows, closes, vols)
        a_up, a_down, a_osc = _aroon(highs, lows)
        ind["aroon_up"], ind["aroon_down"], ind["aroon_osc"] = a_up, a_down, a_osc
        dmi_p, dmi_m, adx = _dmi(highs, lows, closes)
        ind["dmi_plus"], ind["dmi_minus"], ind["adx"] = dmi_p, dmi_m, adx
        for k in ("macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
                  "williams_r", "cci", "mfi", "aroon_up", "aroon_down", "aroon_osc",
                  "dmi_plus", "dmi_minus", "adx"):
            src[k] = SOURCE_STATIC

        # Round floats for compact, JSON-friendly state.
        st.indicators = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ind.items()}
        st.indicator_sources = src

    # ── access ──────────────────────────────────────────────────────────────
    def _state_view(self, st: _SymbolState, now_ts: Optional[float] = None) -> Dict[str, Any]:
        now_ts = self._now_ts() if now_ts is None else now_ts
        age = (now_ts - st.updated_ts) if st.updated_ts else None
        stale = bool(age is not None and age > self.stale_sec)
        sources = set(st.indicator_sources.values())
        if not sources:
            overall_source = SOURCE_STATIC
        elif sources == {SOURCE_STREAMING}:
            overall_source = SOURCE_STREAMING
        elif SOURCE_STREAMING in sources:
            overall_source = "mixed"
        else:
            overall_source = SOURCE_STATIC
        fallback_used = SOURCE_STATIC in sources or not self.streaming_package_available
        return {
            "symbol": st.symbol,
            "indicators": dict(st.indicators),
            "indicator_sources": dict(st.indicator_sources),
            "source": overall_source,
            "fallback_used": fallback_used,
            "streaming_indicator_available": self.streaming_package_available,
            "updated_at": st.updated_at,
            "indicator_age_seconds": (round(age, 3) if age is not None else None),
            "stale": stale,
            "bars_seen": st.bars_seen,
        }

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        st = self.state_by_symbol.get(symbol)
        if st is None:
            return None
        return self._state_view(st)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Per-symbol current state keyed by symbol (consumed by the engines)."""
        now_ts = self._now_ts()
        return {sym: self._state_view(st, now_ts) for sym, st in self.state_by_symbol.items()}

    def health(self) -> Dict[str, Any]:
        snap = self.snapshot()
        stale = [s for s, v in snap.items() if v.get("stale")]
        fallback = sum(1 for v in snap.values() if v.get("fallback_used"))
        return {
            "generated_at_utc": self._now_iso(),
            "date_tag": DATA_TAG,
            "symbols_tracked": len(snap),
            "indicators_tracked": len(INDICATOR_KEYS),
            "indicator_keys": list(INDICATOR_KEYS),
            "streaming_package_available": self.streaming_package_available,
            "stale_symbols": stale,
            "stale_count": len(stale),
            "update_errors": self.update_errors,
            "fallback_count": fallback,
            "alpaca_feed_used": self.alpaca_feed_used,
            "alpaca_feed_type": self.alpaca_feed_type,
            "indicator_updates": self.indicator_updates,
            "stale_updates": len(stale),
            "fallback_updates": self.fallback_updates,
        }

    # ── output ────────────────────────────────────────────────────────────────
    def write_report(self, out_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Write snapshot + health JSON. Returns the written paths. Fail-safe."""
        import json
        if out_dir is None:
            out_dir = DAILY_OUTPUTS / DATA_TAG
        out_dir = Path(out_dir)
        paths: Dict[str, Path] = {}
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            snap = self.snapshot()
            snap_payload = {
                "generated_at_utc": self._now_iso(),
                "date_tag": DATA_TAG,
                "streaming_package_available": self.streaming_package_available,
                "symbols": snap,
            }
            snap_path = out_dir / "streaming_indicator_snapshot.json"
            _write_json_safe(snap_path, snap_payload)
            paths["snapshot"] = snap_path

            health_path = out_dir / "streaming_indicator_health.json"
            _write_json_safe(health_path, self.health())
            paths["health"] = health_path
        except Exception as exc:
            self.update_errors += 1
            logger.debug("write_report failed: %s", exc)
        return paths


# Module-level shared singleton (run.py and the engines read the same state).
ENGINE = StreamingIndicatorEngine()


def get_engine() -> StreamingIndicatorEngine:
    return ENGINE
