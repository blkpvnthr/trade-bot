#!/usr/bin/env python3
"""
alpha_discovery/live_feature_transformer.py — single entry point that turns
realtime Alpha Vantage data into the merged feature dict the live system uses.

This is the integration seam for the existing system files (alpha_call_engine,
portfolio_research_lab, options_booster, market_state_builder, ...): call
``build_realtime_feature_bundle(symbol)`` to get one flat dict of all realtime
price / technical / options / news + hidden-alpha features, then merge it into
the per-symbol event dict via ``transform_event(event)``.

Safety:
    * Entirely advisory. Every fetch is guarded; any failure yields neutral
      features for that family and the rest continue. Never raises, never blocks
      execution (per the fallback requirement).
    * Reads the master enable flag ALPHA_VANTAGE_REALTIME_FEATURES (default on);
      when off, returns neutral features so callers are a no-op.
    * Raw API payloads / text never leave this layer — only normalized features.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)

from alpha_discovery.price_feature_builder import build_price_features, neutral_price_features
from alpha_discovery.technical_feature_builder import build_technical_features, neutral_technical_features
from alpha_discovery.options_feature_builder import build_options_features, neutral_options_features
from alpha_discovery.news_feature_builder import build_news_features, neutral_news_features
from alpha_discovery.analytics_feature_builder import build_analytics_features, neutral_analytics_features

_DEFAULT_INDICATORS = ("SMA", "EMA", "RSI", "MACD", "VWAP", "BBANDS")

# Default analytics calculations for the per-symbol feature block. Kept compact
# (relationship + risk reads) to limit API cost; the batch universe scorer uses a
# wider set. Correlation/covariance require the benchmarks in the same request.
_DEFAULT_ANALYTICS_CALCS = (
    "MEAN", "MEDIAN", "STDDEV", "VARIANCE", "CUMULATIVE_RETURN", "MAX_DRAWDOWN",
    "MIN", "MAX", "AUTOCORRELATION(lag=1)", "CORRELATION", "COVARIANCE",
)
_ANALYTICS_BENCHMARKS = ("SPY", "QQQ")


def _flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def neutral_bundle() -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(neutral_price_features())
    out.update(neutral_technical_features())
    out.update(neutral_options_features())
    out.update(neutral_news_features())
    out.update(neutral_analytics_features())
    out["realtime_bundle_available"] = 0
    return out


def build_realtime_feature_bundle(
    symbol: str,
    av_client: Optional[Any] = None,
    news_client: Optional[Any] = None,
    analytics_client: Optional[Any] = None,
    price_context: Optional[dict[str, Any]] = None,
    is_premarket: bool = False,
    interval: Optional[str] = None,
    indicators: tuple[str, ...] = _DEFAULT_INDICATORS,
    as_of: Optional[date] = None,
) -> dict[str, Any]:
    """Fetch realtime data for ``symbol`` and build the full feature bundle.

    Clients are injectable for tests; when omitted the process-wide defaults are
    lazily created. Returns a neutral bundle when the master flag is off or on
    any failure.
    """
    if not _flag("ALPHA_VANTAGE_REALTIME_FEATURES", True):
        return neutral_bundle()

    symbol = (symbol or "").upper().strip()
    if not symbol:
        return neutral_bundle()

    try:
        if av_client is None:
            from data_sources.alpha_vantage_client import get_default_av_client
            av_client = get_default_av_client()
    except Exception:  # pragma: no cover - import guard
        av_client = None

    bundle: dict[str, Any] = {}
    any_data = False

    # --- price ---
    try:
        bars = av_client.get_intraday(symbol, interval) if av_client else []
        quote = av_client.get_global_quote(symbol) if av_client else {}
        pfeat = build_price_features(bars, quote)
        bundle.update(pfeat)
        any_data = any_data or bool(pfeat.get("realtime_price_available"))
        last_price = pfeat.get("realtime_last_price")
    except Exception as exc:
        logger.warning("realtime price features failed for %s: %s", symbol, exc)
        bundle.update(neutral_price_features())
        last_price = None

    # --- technical ---
    try:
        ind_data: dict[str, list[dict[str, Any]]] = {}
        if av_client:
            for ind in indicators:
                try:
                    ind_data[ind] = av_client.get_technical_indicator(symbol, ind, interval)
                except Exception:
                    ind_data[ind] = []
        tfeat = build_technical_features(ind_data, last_price)
        bundle.update(tfeat)
        any_data = any_data or bool(tfeat.get("realtime_technical_available"))
    except Exception as exc:
        logger.warning("realtime technical features failed for %s: %s", symbol, exc)
        bundle.update(neutral_technical_features())

    # --- options ---
    try:
        chain = av_client.get_realtime_options(symbol) if av_client else []
        pcr = av_client.get_put_call_ratio(symbol) if av_client else {}
        voi = av_client.get_volume_oi_ratio(symbol) if av_client else {}
        ofeat = build_options_features(chain, pcr, voi, spot=last_price, as_of=as_of)
        bundle.update(ofeat)
        any_data = any_data or bool(ofeat.get("realtime_options_available"))
    except Exception as exc:
        logger.warning("realtime options features failed for %s: %s", symbol, exc)
        bundle.update(neutral_options_features())

    # --- news ---
    try:
        if news_client is None and _flag("NEWS_SENTIMENT_ENABLED", True):
            from data_sources.news_sentiment_client import get_default_news_client
            news_client = get_default_news_client()
        news_rows = news_client.get_symbol_news(symbol, limit=50) if news_client else []
        nfeat = build_news_features(news_rows, price_context=price_context, is_premarket=is_premarket)
        bundle.update(nfeat)
        any_data = any_data or bool(nfeat.get("news_available"))
    except Exception as exc:
        logger.warning("realtime news features failed for %s: %s", symbol, exc)
        bundle.update(neutral_news_features())

    # --- advanced analytics (§2/§3/§5) ---
    # Schema always present; real values populated only when explicitly enabled
    # (the analytics endpoints are batch-heavy — the universe scorer is the
    # efficient path; this per-symbol block is opt-in).
    try:
        if _flag("ALPHA_VANTAGE_ANALYTICS_FEATURES", False):
            afeat = _build_analytics_block(symbol, analytics_client, event=price_context)
            bundle.update(afeat)
            any_data = any_data or bool(afeat.get("analytics_available"))
        else:
            bundle.update(neutral_analytics_features())
    except Exception as exc:
        logger.warning("advanced analytics features failed for %s: %s", symbol, exc)
        bundle.update(neutral_analytics_features())

    bundle["realtime_bundle_available"] = 1 if any_data else 0
    return bundle


def _build_analytics_block(
    symbol: str,
    analytics_client: Optional[Any] = None,
    event: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Fetch fixed + sliding analytics (symbol + benchmarks) and build features.

    Advisory-safe: returns neutral analytics features on any failure.
    """
    try:
        if analytics_client is None:
            from data_sources.alpha_vantage_analytics_client import get_default_analytics_client
            analytics_client = get_default_analytics_client()
    except Exception:  # pragma: no cover - import guard
        return neutral_analytics_features()
    if analytics_client is None:
        return neutral_analytics_features()

    syms = [symbol] + [b for b in _ANALYTICS_BENCHMARKS if b != symbol.upper()]
    try:
        fixed = analytics_client.get_fixed_window(
            syms, list(_DEFAULT_ANALYTICS_CALCS), range_="60day", interval="DAILY")
    except Exception:
        fixed = None
    try:
        sliding = analytics_client.get_sliding_window(
            syms, list(_DEFAULT_ANALYTICS_CALCS), range_="3month", interval="5min", window_size=20)
    except Exception:
        sliding = None
    return build_analytics_features(symbol, fixed, sliding, event=event)


def transform_event(
    event: dict[str, Any],
    symbol: Optional[str] = None,
    av_client: Optional[Any] = None,
    news_client: Optional[Any] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Merge a realtime feature bundle into a per-symbol event dict (in place).

    Existing event keys win on collision (we never overwrite upstream signals).
    Returns the same event for chaining. Safe no-op on failure.
    """
    try:
        sym = symbol or event.get("symbol") or event.get("ticker") or ""
        price_context = {
            "price_momentum": event.get("price_momentum", event.get("realtime_quote_momentum", 0.0)),
            "volume_acceleration": event.get("volume_acceleration", 0.0),
            "relative_volume": event.get("rel_volume", 1.0),
            "vwap_distance": event.get("vwap_distance", 0.0),
            "volatility": event.get("atr_pct", 0.0),
            "iv_change": event.get("iv_change", 0.0),
            "options_flow": event.get("options_flow", 0.0),
            "index_strength": event.get("spy_regime", 0.0),
        }
        bundle = build_realtime_feature_bundle(
            sym, av_client=av_client, news_client=news_client,
            price_context=price_context,
            is_premarket=bool(event.get("is_premarket", False)),
            **kwargs,
        )
        for k, v in bundle.items():
            event.setdefault(k, v)
    except Exception as exc:  # pragma: no cover - must never break the event path
        logger.warning("transform_event failed: %s", exc)
    return event
