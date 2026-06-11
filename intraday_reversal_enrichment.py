#!/usr/bin/env python3
"""Intraday-bar enrichment for the reversal engine.

A screener export gives the UNIVERSE (price/volume/chg/ratings) but no reclaim
TRIGGERS — those need intraday bars. This stage takes screener-derived ReversalInput
rows, attaches per-symbol 1m/5m bars (+ daily history), DERIVES the reversal signals
(VWAP reclaim, ORB reclaim, liquidity sweep/reclaim, RSI rising / cross-40, above 5m
EMA20, ATR, PDH, recovering), and writes those fields back onto each ReversalInput so
the engine can score real setups.

Strictly separated:
  * ``derive_signals`` / ``enrich_input`` / ``enrich_universe`` are PURE — they take
    bars (or an injected bar provider) and never touch a broker or place orders.
  * the Alpaca bar provider (``_alpaca_bar_provider``) is the only impure part and
    uses READ-ONLY market data (no trading client, no orders).

Gated OFF by default; ``maybe_enrich`` is a no-op unless INTRADAY_REVERSAL_ENRICH_BARS.

Env:
  INTRADAY_REVERSAL_ENRICH_BARS=false
  INTRADAY_REVERSAL_BAR_TIMEFRAME=1Min
  INTRADAY_REVERSAL_ORB_MINUTES=15
  INTRADAY_REVERSAL_MAX_ENRICH_SYMBOLS=50
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Optional

from intraday_reversal_engine import (
    ReversalInput, previous_day_high, vwap_series, rsi_series, ema_series, atr_from_daily,
    _safe_len, _finite, _env_bool, _env_int, rel_vol_5m_of,
)

logger = logging.getLogger("intraday_reversal_enrichment")

# fields written back onto ReversalInput from derive_signals()
_BOOL_FIELDS = ("vwap_reclaim", "rsi_rising", "rsi_cross_40", "close_gt_open",
                "sweep_reclaim", "orb_reclaim", "above_ema20_5m", "recovering")
_FLOAT_FIELDS = ("atr", "extended_atr", "near_low_atr")

# A bar provider maps a symbol to (intraday_bars_df, daily_history_df) or None.
BarProvider = Callable[[str], Optional[tuple]]


def _bars_per_min(timeframe: str) -> float:
    m = re.match(r"\s*(\d+)", str(timeframe or "1"))
    minutes = int(m.group(1)) if m else 1
    return 1.0 / max(1, minutes)


# ---------------------------------------------------------------------------
# PURE signal derivation
# ---------------------------------------------------------------------------
def derive_signals(bars, daily_history=None, orb_minutes: int = 15,
                   bars_per_min: float = 1.0) -> dict[str, Any]:
    """Compute the reversal signals from intraday OHLCV bars (+ optional daily history
    for ATR/PDH). Pure; tolerant of short series (returns only what it can compute)."""
    out: dict[str, Any] = {}
    n = _safe_len(bars)
    if n < 3:
        return out
    try:
        close = bars["close"]; high = bars["high"]; low = bars["low"]; openp = bars["open"]
    except Exception:
        return out
    last_c = float(close.iloc[-1])

    # ATR: daily TR if available, else intraday mean range, else 1% of price.
    atr = atr_from_daily(daily_history) if daily_history is not None else None
    if not atr or atr <= 0:
        try:
            atr = float((high - low).tail(14).mean())
        except Exception:
            atr = None
    out["atr"] = float(atr) if (atr and atr > 0) else max(0.01, last_c * 0.01)

    # VWAP reclaim: above VWAP now after being below it recently.
    vw = vwap_series(bars)
    if vw is not None:
        try:
            last_vw = float(vw.iloc[-1])
            out["vwap_reclaim"] = bool(last_c > last_vw and (close.tail(6) < vw.tail(6)).any())
            out["extended_atr"] = (last_c - last_vw) / out["atr"] if out["atr"] > 0 else 0.0
        except Exception:
            pass

    # RSI rising over last 5 bars / cross above 40.
    rsi = rsi_series(close)
    if rsi is not None:
        r = rsi.dropna()
        if len(r) >= 5:
            out["rsi_rising"] = bool(float(r.iloc[-1]) > float(r.iloc[-5]))
        if len(r) >= 2:
            out["rsi_cross_40"] = bool(float(r.iloc[-2]) < 40.0 <= float(r.iloc[-1]))

    # confirmation bar close > open.
    try:
        out["close_gt_open"] = bool(last_c > float(openp.iloc[-1]))
    except Exception:
        pass

    # above 5m EMA20.
    ema = ema_series(close, 20)
    if ema is not None:
        try:
            out["above_ema20_5m"] = bool(last_c >= float(ema.iloc[-1]))
        except Exception:
            pass

    # ORB reclaim: opening-range (orb_minutes) low broken then reclaimed.
    n_or = max(2, int(round(orb_minutes * bars_per_min)))
    if n > n_or + 2:
        try:
            or_low = float(low.iloc[:n_or].min())
            after = bars.iloc[n_or:]
            out["orb_reclaim"] = bool((after["low"] < or_low).any() and last_c > or_low)
        except Exception:
            pass

    # liquidity sweep then reclaim: recent bar takes out a prior swing low, price reclaims it.
    if n >= 8:
        try:
            prior_low = float(low.iloc[:-3].min())
            out["sweep_reclaim"] = bool((low.tail(3) < prior_low).any() and last_c > prior_low)
        except Exception:
            pass

    # recovering: off the lows and rising (red or green).
    try:
        day_low = float(low.min())
        off_lows = (last_c - day_low) > 0.5 * out["atr"]
        rising = last_c > float(close.iloc[max(0, n - 4)])
        out["recovering"] = bool(off_lows and rising)
        out["near_low_atr"] = (last_c - day_low) / out["atr"] if out["atr"] > 0 else None
    except Exception:
        pass

    # last-5-minute volume.
    try:
        out["vol_5m"] = float(bars["volume"].tail(max(1, int(round(5 * bars_per_min)))).sum())
    except Exception:
        pass

    # PDH from daily history (prior completed session high).
    if daily_history is not None:
        try:
            out["prev_day_high"] = float(daily_history["high"].iloc[-1])
        except Exception:
            pass
    return out


def enrich_input(inp: ReversalInput, bars, daily_history=None, orb_minutes: int = 15,
                 bars_per_min: float = 1.0) -> ReversalInput:
    """Derive signals from bars and write them back onto ``inp`` (in place). Explicit
    fields already on ``inp`` (e.g. from an enriched export) are NOT overwritten for
    vol_5m / prev_day_high; derived signals fill the reversal triggers."""
    sig = derive_signals(bars, daily_history, orb_minutes, bars_per_min)
    for f in _BOOL_FIELDS:
        if f in sig:
            setattr(inp, f, sig[f])
    for f in _FLOAT_FIELDS:
        if f in sig:
            setattr(inp, f, sig[f])
    if inp.vol_5m is None and "vol_5m" in sig:
        inp.vol_5m = sig["vol_5m"]
    if inp.prev_day_high is None and "prev_day_high" in sig:
        inp.prev_day_high = sig["prev_day_high"]
    return inp


def _vol5_key(i: ReversalInput) -> float:
    return -_finite(rel_vol_5m_of(i))   # 5-minute RELATIVE volume desc


def enrich_universe(rows: list[ReversalInput], bar_provider: BarProvider,
                    max_symbols: int = 50, orb_minutes: int = 15,
                    bars_per_min: float = 1.0) -> list[ReversalInput]:
    """Attach bars (via the injected provider) to each row and derive signals. Pure given
    a pure provider — no orders, no broker. Returns rows sorted by 5-min RELATIVE volume desc.
    Rows beyond ``max_symbols`` are passed through un-enriched (logged)."""
    enriched = 0
    for inp in rows[:max(0, max_symbols)]:
        try:
            data = bar_provider(inp.symbol)
        except Exception as exc:
            logger.debug("bar provider failed for %s: %s", inp.symbol, exc)
            data = None
        if data:
            bars, daily = (data if isinstance(data, (tuple, list)) and len(data) == 2
                           else (data, None))
            enrich_input(inp, bars, daily, orb_minutes, bars_per_min)
            enriched += 1
    if len(rows) > max_symbols:
        logger.info("enrichment: capped at %d/%d symbols (INTRADAY_REVERSAL_MAX_ENRICH_SYMBOLS)",
                    max_symbols, len(rows))
    logger.info("enrichment: enriched %d/%d rows from bars", enriched, len(rows))
    rows.sort(key=_vol5_key)
    return rows


# ---------------------------------------------------------------------------
# IMPURE: Alpaca read-only bar provider (no trading client, no orders)
# ---------------------------------------------------------------------------
def _alpaca_bar_provider(timeframe: str = "1Min") -> BarProvider:
    """Build a symbol -> (intraday_bars, daily_history) provider from Alpaca READ-ONLY
    market data. Never imports the trading client or places orders. Returns a provider
    that yields None on any failure (enrichment then leaves that row un-enriched)."""
    from datetime import datetime, timedelta, timezone
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key = os.getenv("APCA_API_KEY_ID", ""); secret = os.getenv("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("APCA creds missing for read-only bar enrichment")
    client = StockHistoricalDataClient(key, secret)
    feed = os.getenv("ALPACA_DATA_FEED", "sip").lower()
    minutes = int(re.match(r"\s*(\d+)", timeframe).group(1)) if re.match(r"\s*(\d+)", timeframe) else 1
    intraday_tf = TimeFrame(minutes, TimeFrameUnit.Minute)
    now = datetime.now(timezone.utc)

    def provider(symbol: str):
        try:
            intraday = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=intraday_tf,
                start=now - timedelta(hours=12), feed=feed)).df
            daily = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                start=now - timedelta(days=20), feed=feed)).df
            # multi-index (symbol, ts) -> single symbol frame
            if hasattr(intraday, "index") and intraday.index.nlevels > 1:
                intraday = intraday.xs(symbol, level=0)
            if hasattr(daily, "index") and daily.index.nlevels > 1:
                daily = daily.xs(symbol, level=0)
            if intraday is None or len(intraday) < 3:
                return None
            return intraday, daily
        except Exception as exc:
            logger.debug("alpaca bar fetch failed for %s: %s", symbol, exc)
            return None
    return provider


# ---------------------------------------------------------------------------
# gated entry point used by the engine's collect_reversal_opportunities
# ---------------------------------------------------------------------------
def maybe_enrich(rows: list[ReversalInput],
                 bar_provider: Optional[BarProvider] = None) -> list[ReversalInput]:
    """No-op unless INTRADAY_REVERSAL_ENRICH_BARS is set. Then enriches via the injected
    provider (tests) or the Alpaca read-only provider, honoring the timeframe / ORB /
    max-symbols env flags. Fail-safe: returns rows unchanged on any provider error."""
    if not rows or not _env_bool("INTRADAY_REVERSAL_ENRICH_BARS", False):
        return rows
    timeframe = os.getenv("INTRADAY_REVERSAL_BAR_TIMEFRAME", "1Min")
    orb_minutes = _env_int("INTRADAY_REVERSAL_ORB_MINUTES", 15)
    max_symbols = _env_int("INTRADAY_REVERSAL_MAX_ENRICH_SYMBOLS", 50)
    bpm = _bars_per_min(timeframe)
    if bar_provider is None:
        try:
            bar_provider = _alpaca_bar_provider(timeframe)
        except Exception as exc:
            logger.warning("enrichment: bar provider unavailable (%s) — rows un-enriched", exc)
            return rows
    return enrich_universe(rows, bar_provider, max_symbols, orb_minutes, bpm)
