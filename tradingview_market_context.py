#!/usr/bin/env python3
"""
tradingview_market_context.py — Futures / Index Market Context Engine (enhancement layer).

Produces a market REGIME and an options-exposure THROTTLE from index / futures / volatility
signals, so run.py can gate options sizing + selection. This is an ENHANCEMENT LAYER, never a
rewrite: on any failure it falls back to NEUTRAL / 0.50 and the existing system is unaffected.

────────────────────────────────────────────────────────────────────────────────────────────
DATA SOURCES (honest, layered, fail-safe)
  1. PRIMARY  — Alpaca ETF proxies: SPY→ES/SPX, QQQ→NQ/NDX, IWM→RTY/RUT, DIA→YM, VIXY→VIX.
                Reliable, real-time during RTH, OPRA/IEX-sourced. The regime computed from these
                is ~identical to the index/futures regime (>0.99 intraday correlation).
  2. ENHANCE  — tradingview-screener (when TRADINGVIEW_FUTURES_FEED_ENABLED + reachable): the
                actual continuous futures (ES1!/NQ1!/RTY1!/YM1!) + the VIX futures TERM STRUCTURE
                (VX1 vs VX2). Adds OVERNIGHT / extended-hours awareness + backwardation detection,
                which ETFs cannot provide. Best-effort; any failure → use the Alpaca core.
  3. FALLBACK — the existing Alpha Vantage market context → regime=NEUTRAL, throttle=0.50.

NOTE: TradingView's purchased OPRA/CME/Cboe feeds are PLATFORM entitlements (charts/screeners/
alerts) with no official quote API. OPRA OPTION CHAINS are read programmatically from Alpaca
(which is OPRA-sourced) + AV realtime — this module exposes `opra_contract_ok()` for the long-call
scanner. We NEVER use these feeds to sell naked calls (long calls / defined-risk / hedges only).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except Exception:
        return default


# ── config flags (see .env.example) ────────────────────────────────────────────────────────
TRADINGVIEW_CONTEXT_ENABLED = _env_bool("TRADINGVIEW_CONTEXT_ENABLED", True)
TRADINGVIEW_FUTURES_FEED_ENABLED = _env_bool("TRADINGVIEW_FUTURES_FEED_ENABLED", True)
TRADINGVIEW_OPTIONS_FEED_ENABLED = _env_bool("TRADINGVIEW_OPTIONS_FEED_ENABLED", True)
REGIME_THROTTLE_ENABLED = _env_bool("REGIME_THROTTLE_ENABLED", True)
CONTEXT_TTL_SEC = _env_float("REGIME_CONTEXT_TTL_SEC", 300.0)   # cache; protects the data feed
STALE_QUOTE_MAX_SEC = _env_float("OPRA_STALE_QUOTE_MAX_SEC", 90.0)

REGIMES = ("STRONG_RISK_ON", "RISK_ON", "NEUTRAL", "RISK_OFF", "PANIC")
# req #2 — exposure throttle per regime
THROTTLE: Dict[str, float] = {
    "STRONG_RISK_ON": 1.00,
    "RISK_ON": 0.75,
    "NEUTRAL": 0.50,
    "RISK_OFF": 0.25,
    "PANIC": 0.00,
}
_TREND_NUM = {"strong_up": 2.0, "up": 1.0, "flat": 0.0, "down": -1.0, "strong_down": -2.0}

# Alpaca ETF proxies for the index/futures complex
_PROXY = {"ES": "SPY", "NQ": "QQQ", "RTY": "IWM", "YM": "DIA"}
# TradingView continuous-futures symbols (best-effort enhancement)
_TV_FUTURES = {"ES": "CME_MINI:ES1!", "NQ": "CME_MINI:NQ1!", "RTY": "CME_MINI:RTY1!",
               "YM": "CBOT_MINI:YM1!", "VX1": "CBOE_FUT:VX1!", "VX2": "CBOE_FUT:VX2!"}


@dataclass
class MarketContext:
    regime: str = "NEUTRAL"
    throttle: float = 0.50
    es_trend: str = "flat"
    nq_trend: str = "flat"
    rty_trend: str = "flat"
    ym_trend: str = "flat"
    vix_state: str = "calm"                # calm | elevated | rising | stress
    vix_vx_backwardation: bool = False
    breadth: float = 0.0                   # 0..1 fraction of {ES,NQ,RTY,YM} trending up
    nq_es_rs: float = 0.0                  # NQ minus ES relative strength (risk appetite)
    block_new_long_calls: bool = False     # req #4 — VIX/VX risk gate
    max_positions_factor: float = 1.0      # req #4 — reduce open positions under stress
    blocked_reason: str = ""
    source: str = "fallback"               # tradingview | alpaca_proxy | alpha_vantage | fallback | disabled
    detail: Dict[str, Any] = field(default_factory=dict)

    def log_line(self) -> str:
        # req #5 — one compact MARKET_CONTEXT line every cycle
        s = (f"MARKET_CONTEXT regime={self.regime} ES={self.es_trend} NQ={self.nq_trend} "
             f"RTY={self.rty_trend} VIX={self.vix_state} throttle={self.throttle:.2f} src={self.source}")
        if self.blocked_reason:
            s += f" blocked={self.blocked_reason}"
        return s

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ════════════════════════════════════════════════════════════════════════════════════════════
# Pure classification logic (unit-tested without any live data)
# ════════════════════════════════════════════════════════════════════════════════════════════
def vix_state(vix_change_pct: float, vx_backwardation: bool, vix_level: float = 0.0) -> str:
    """Classify volatility stress. vix_change_pct is fractional (0.05 = +5%)."""
    if vx_backwardation and vix_change_pct > 0.03:
        return "stress"                         # backwardation + VIX rising = acute fear
    if vix_change_pct >= 0.08:
        return "stress"                         # VIX spiking
    if vix_change_pct >= 0.04:
        return "rising"
    if vix_level >= 25.0 or vix_change_pct >= 0.02:
        return "elevated"
    return "calm"


def classify(m: Dict[str, Any]) -> MarketContext:
    """Map a metrics dict → MarketContext. Pure + deterministic (the core unit-tested logic)."""
    es = _TREND_NUM.get(m.get("es_trend", "flat"), 0.0)
    nq = _TREND_NUM.get(m.get("nq_trend", "flat"), 0.0)
    rty = _TREND_NUM.get(m.get("rty_trend", "flat"), 0.0)
    ym = _TREND_NUM.get(m.get("ym_trend", "flat"), 0.0)
    breadth = float(m.get("breadth", 0.5))
    rs = float(m.get("nq_es_rs", 0.0))
    vstate = str(m.get("vix_state", "calm"))
    backw = bool(m.get("vix_vx_backwardation", False))

    base = (es + nq + rty + ym) / 4.0                       # avg trend, -2..+2
    rs_term = max(-1.0, min(1.0, rs * 100.0)) * 0.5         # NQ vs ES risk appetite
    rty_conf = 0.5 if rty > 0 else (-0.5 if rty < 0 else 0.0)   # small-cap confirmation
    vix_pen = {"calm": 0.3, "elevated": -0.5, "rising": -1.2, "stress": -3.0}.get(vstate, 0.0)
    score = base + rs_term + rty_conf + vix_pen

    if vstate == "stress":
        regime = "PANIC"
    elif score >= 1.8 and vstate == "calm" and breadth >= 0.75:
        regime = "STRONG_RISK_ON"
    elif score >= 0.7:
        regime = "RISK_ON"
    elif score <= -1.2:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"

    block = vstate in ("stress", "rising") or backw
    reason = ""
    if regime == "PANIC":
        reason = "panic:vix_stress" + ("+vx_backwardation" if backw else "")
    elif block:
        reason = "vix_gate:" + ("vx_backwardation" if backw else f"vix_{vstate}")

    return MarketContext(
        regime=regime,
        throttle=THROTTLE.get(regime, 0.5),
        es_trend=m.get("es_trend", "flat"), nq_trend=m.get("nq_trend", "flat"),
        rty_trend=m.get("rty_trend", "flat"), ym_trend=m.get("ym_trend", "flat"),
        vix_state=vstate, vix_vx_backwardation=backw, breadth=breadth, nq_es_rs=rs,
        block_new_long_calls=block,
        max_positions_factor=(0.5 if regime in ("RISK_OFF", "PANIC") or block else 1.0),
        blocked_reason=reason, source=str(m.get("source", "alpaca_proxy")),
        detail={"score": round(score, 2)},
    )


def opra_contract_ok(c: Dict[str, Any], regime_aligned: bool = True) -> Tuple[bool, str]:
    """
    OPRA-aware LONG-CALL contract filter (req #3). `c` keys: volume, open_interest, bid, ask,
    delta, dte, quote_age_sec (optional). Long calls only — never used to sell naked calls.
    Pass criteria: real-time volume > OI, spread<=10%, delta>=0.47, premium $0.05-$2.50,
    DTE 21-60, underlying aligned with the futures regime; reject stale/zero-bid quotes.
    """
    bid = float(c.get("bid", 0) or 0.0)
    ask = float(c.get("ask", 0) or 0.0)
    if bid <= 0.0:
        return False, "zero_bid"
    age = c.get("quote_age_sec")
    if age is not None and float(age) > STALE_QUOTE_MAX_SEC:
        return False, f"stale_quote:{float(age):.0f}s"
    mid = (bid + ask) / 2.0 if ask > 0 else bid
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
    delta = float(c.get("delta", 0) or 0.0)
    dte = int(c.get("dte", 0) or 0)
    vol = float(c.get("volume", 0) or 0.0)
    oi = float(c.get("open_interest", 0) or 0.0)
    if spread_pct > 0.10:
        return False, f"spread_{spread_pct:.0%}>10%"
    if delta < 0.47:
        return False, f"delta_{delta:.2f}<0.47"
    if not (0.05 <= mid <= 2.50):
        return False, f"premium_{mid:.2f}_out_of_band"
    if not (21 <= dte <= 60):
        return False, f"dte_{dte}_out_of_band"
    if vol <= oi:
        return False, f"vol_{vol:.0f}<=oi_{oi:.0f}"
    if not regime_aligned:
        return False, "regime_misaligned"
    return True, "ok"


# ════════════════════════════════════════════════════════════════════════════════════════════
# Data providers (best-effort, fail-safe — never raise)
# ════════════════════════════════════════════════════════════════════════════════════════════
_DC = None  # lazy Alpaca StockHistoricalDataClient


def _trend(closes) -> str:
    """up/down/flat (+ strong_*) from a close series, via EMA position + slope."""
    try:
        if closes is None or len(closes) < 5:
            return "flat"
        ema = closes.ewm(span=min(9, len(closes)), adjust=False).mean()
        last = float(closes.iloc[-1]); e = float(ema.iloc[-1])
        n = min(5, len(ema) - 1)
        slope = float(ema.iloc[-1] - ema.iloc[-1 - n])
        rel = (last - e) / e if e else 0.0
        if last > e and slope > 0:
            return "strong_up" if rel > 0.004 else "up"
        if last < e and slope < 0:
            return "strong_down" if rel < -0.004 else "down"
        return "flat"
    except Exception:
        return "flat"


def _alpaca_intraday(symbols, lookback_days=2):
    """{symbol: closes_5m_series}. Fail-safe (returns {} on any error)."""
    out = {}
    try:
        global _DC
        from datetime import datetime, timedelta, timezone
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
        if not key or not sec:
            return {}
        if _DC is None:
            _DC = StockHistoricalDataClient(key, sec)
        end = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=end - timedelta(days=lookback_days), end=end, feed=DataFeed.IEX)
        df = _DC.get_stock_bars(req).df
        if df is None or df.empty:
            return {}
        for s in symbols:
            try:
                sub = df.xs(s, level="symbol") if "symbol" in df.index.names else df
                out[s] = sub["close"].astype(float)
            except Exception:
                continue
    except Exception as e:
        logger.debug("alpaca intraday fetch failed: %s", e)
    return out


def _alpaca_proxy_metrics() -> Optional[Dict[str, Any]]:
    """PRIMARY source: per-proxy trend/VWAP from Alpaca ETFs. None if unavailable."""
    syms = list(_PROXY.values()) + ["VIXY"]
    bars = _alpaca_intraday(syms)
    if not bars or _PROXY["ES"] not in bars:
        return None
    trends, vwap_dist = {}, {}
    for fut, etf in _PROXY.items():
        c = bars.get(etf)
        trends[fut] = _trend(c)
        try:
            vwap_dist[fut] = (float(c.iloc[-1]) / float(c.tail(78).mean()) - 1.0) if c is not None and len(c) else 0.0
        except Exception:
            vwap_dist[fut] = 0.0
    # VIX proxy via VIXY change over the session
    vix_chg = 0.0
    try:
        v = bars.get("VIXY")
        if v is not None and len(v) > 1:
            vix_chg = float(v.iloc[-1]) / float(v.iloc[0]) - 1.0
    except Exception:
        vix_chg = 0.0
    up = sum(1 for t in trends.values() if t in ("up", "strong_up"))
    return {
        "es_trend": trends["ES"], "nq_trend": trends["NQ"], "rty_trend": trends["RTY"], "ym_trend": trends["YM"],
        "breadth": up / 4.0,
        "nq_es_rs": vwap_dist.get("NQ", 0.0) - vwap_dist.get("ES", 0.0),
        "vix_change_pct": vix_chg, "vix_vx_backwardation": False, "vix_level": 0.0,
        "source": "alpaca_proxy",
    }


def _tradingview_metrics() -> Optional[Dict[str, Any]]:
    """
    ENHANCEMENT: real continuous futures + VIX term structure via tradingview-screener.
    Best-effort — any failure returns None and the caller uses the Alpaca core. Adds
    overnight/extended-hours regime + VX1>VX2 backwardation (the genuine futures-feed value).
    """
    if not TRADINGVIEW_FUTURES_FEED_ENABLED:
        return None
    try:
        from tradingview_screener import Query  # repo dependency
        tickers = list(_TV_FUTURES.values())
        _, df = (Query().select("name", "close", "change").set_tickers(*tickers).get_scanner_data())
        if df is None or df.empty:
            return None
        by = {str(r.get("name") or r.get("ticker")): r for _, r in df.iterrows()}

        def chg(tvsym):
            r = by.get(tvsym) or by.get(tvsym.split(":")[-1])
            try:
                return float(r["change"]) / 100.0 if r is not None else None
            except Exception:
                return None

        out: Dict[str, Any] = {"source": "tradingview"}
        m = {"ES": "es_trend", "NQ": "nq_trend", "RTY": "rty_trend", "YM": "ym_trend"}
        ups = 0; have = 0
        for fut, key in m.items():
            c = chg(_TV_FUTURES[fut])
            if c is None:
                continue
            have += 1
            out[key] = "strong_up" if c > 0.004 else "up" if c > 0 else "strong_down" if c < -0.004 else "down" if c < 0 else "flat"
            if c > 0:
                ups += 1
        if have == 0:
            return None
        out["breadth"] = ups / have
        # VIX futures term structure (the futures-feed-only signal)
        vx1, vx2 = None, None
        for sym, r in by.items():
            try:
                if "VX1" in sym:
                    vx1 = float(r["close"])
                elif "VX2" in sym:
                    vx2 = float(r["close"])
            except Exception:
                continue
        if vx1 is not None and vx2 is not None:
            out["vix_vx_backwardation"] = vx1 > vx2
        return out
    except Exception as e:
        logger.debug("tradingview futures metrics unavailable (best-effort): %s", e)
        return None


# ════════════════════════════════════════════════════════════════════════════════════════════
# Public entry — cached, fail-safe
# ════════════════════════════════════════════════════════════════════════════════════════════
_CACHE: Dict[str, Any] = {"ctx": None, "t": 0.0}


def get_market_context(av_context: Optional[Dict[str, Any]] = None, force: bool = False) -> MarketContext:
    """
    Build the market context. Never raises. Layered: Alpaca core (+ TV enhancement) → AV fallback
    → NEUTRAL/0.50. `av_context` is the existing build_market_context() dict (req #7 fallback).
    """
    if not TRADINGVIEW_CONTEXT_ENABLED:
        return MarketContext(regime="NEUTRAL", throttle=1.0, source="disabled",
                             blocked_reason="tradingview_context_disabled")
    now = time.time()
    if not force and _CACHE["ctx"] is not None and (now - _CACHE["t"]) < CONTEXT_TTL_SEC:
        return _CACHE["ctx"]
    try:
        metrics = _alpaca_proxy_metrics()                       # PRIMARY (reliable)
        if metrics is None:
            ctx = _fallback_context(av_context, "alpaca_proxy_unavailable")
        else:
            tv = _tradingview_metrics()                         # ENHANCEMENT (best-effort)
            if tv:
                for k in ("es_trend", "nq_trend", "rty_trend", "ym_trend", "breadth"):
                    if k in tv:
                        metrics[k] = tv[k]
                if tv.get("vix_vx_backwardation"):
                    metrics["vix_vx_backwardation"] = True
                metrics["source"] = "tradingview"
            metrics["vix_state"] = vix_state(
                float(metrics.get("vix_change_pct", 0.0)),
                bool(metrics.get("vix_vx_backwardation", False)),
                float(metrics.get("vix_level", 0.0)))
            ctx = classify(metrics)
    except Exception as e:
        logger.warning("market context build failed, falling back: %s", e)
        ctx = _fallback_context(av_context, f"exception:{str(e)[:60]}")
    _CACHE["ctx"], _CACHE["t"] = ctx, now
    return ctx


def _fallback_context(av_context: Optional[Dict[str, Any]], why: str) -> MarketContext:
    """req #7 — TradingView/Alpaca unavailable → existing AV context, NEUTRAL, throttle 0.50."""
    src = "fallback"
    block = False
    if av_context:
        src = "alpha_vantage"
        bias = str(av_context.get("overall_market_bias") or "").lower()
        if "bear" in bias or "risk_off" in bias:
            block = True
    logger.info("MARKET_CONTEXT fallback (%s) → regime=NEUTRAL throttle=0.50 src=%s", why, src)
    return MarketContext(regime="NEUTRAL", throttle=0.50, source=src,
                         block_new_long_calls=block, blocked_reason=f"fallback:{why}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctx = get_market_context(force=True)
    print(ctx.log_line())
    print(ctx.as_dict())
