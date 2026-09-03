#!/usr/bin/env python3
"""
alpha_discovery/news_feature_builder.py — windowed news, catalyst and
news-price interaction features from normalized NEWS_SENTIMENT rows.

Pure functions: given a list of normalized news rows (from
data_sources.news_sentiment_client) and an ``as_of`` timestamp, produce a flat
feature dict. Missing data -> neutral features (0.0 counts, 0.0 sentiment),
never an exception. News failures must never interrupt execution.

Feature families produced
-------------------------
Counts / recency:
    news_count_15m, news_count_30m, news_count_60m, news_count_24h
    minutes_since_latest_news
Sentiment windows (relevance-weighted ticker sentiment):
    sentiment_mean_15m/30m/60m/24h
    sentiment_change_30m, sentiment_change_60m
    sentiment_acceleration, sentiment_volatility
    news_density_score, news_relevance_weighted_sentiment
Catalyst detection:
    has_recent_news, has_premarket_catalyst, has_earnings_news, has_merger_news,
    has_macro_news, has_monetary_policy_news, has_fiscal_policy_news,
    has_sector_news, catalyst_strength_score
News-price interactions (require a price_context dict):
    news_sentiment_x_price_momentum, news_sentiment_x_volume_acceleration,
    news_density_x_relative_volume, news_relevance_x_vwap_distance,
    news_sentiment_x_volatility, earnings_news_x_iv_change,
    merger_news_x_options_flow, macro_news_x_index_strength
Hidden alpha (bounded composites):
    hidden_news_continuation_score, hidden_news_fade_score,
    hidden_sentiment_momentum_score, hidden_sentiment_reversal_score,
    hidden_catalyst_strength_score, hidden_news_density_score,
    hidden_news_volume_confirmation_score, hidden_news_volatility_expansion_score,
    hidden_news_price_dislocation_score, hidden_topic_momentum_score
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Topic groups for catalyst flags.
_EARNINGS_TOPICS = {"earnings"}
_MERGER_TOPICS = {"mergers_and_acquisitions", "ipo"}
_MACRO_TOPICS = {"economy_macro"}
_MONETARY_TOPICS = {"economy_monetary"}
_FISCAL_TOPICS = {"economy_fiscal"}
_SECTOR_TOPICS = {"technology", "life_sciences", "manufacturing", "real_estate",
                  "retail_wholesale", "energy_transportation", "finance", "blockchain"}


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _to_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _sentiment_of(row: dict[str, Any]) -> Optional[float]:
    s = row.get("ticker_sentiment_score")
    if s is None:
        s = row.get("overall_sentiment_score")
    try:
        return float(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _relevance_of(row: dict[str, Any]) -> float:
    r = row.get("relevance_score")
    try:
        return max(0.0, min(1.0, float(r))) if r is not None else 0.5
    except (TypeError, ValueError):
        return 0.5


def neutral_news_features() -> dict[str, Any]:
    """All-neutral feature dict — returned when no news is available."""
    feats: dict[str, Any] = {
        "news_count_15m": 0, "news_count_30m": 0, "news_count_60m": 0, "news_count_24h": 0,
        "minutes_since_latest_news": 1440.0,
        "sentiment_mean_15m": 0.0, "sentiment_mean_30m": 0.0,
        "sentiment_mean_60m": 0.0, "sentiment_mean_24h": 0.0,
        "sentiment_change_30m": 0.0, "sentiment_change_60m": 0.0,
        "sentiment_acceleration": 0.0, "sentiment_volatility": 0.0,
        "news_density_score": 0.0, "news_relevance_weighted_sentiment": 0.0,
        "has_recent_news": 0, "has_premarket_catalyst": 0, "has_earnings_news": 0,
        "has_merger_news": 0, "has_macro_news": 0, "has_monetary_policy_news": 0,
        "has_fiscal_policy_news": 0, "has_sector_news": 0, "catalyst_strength_score": 0.0,
        "news_available": 0,
    }
    feats.update(_neutral_interactions())
    feats.update(_neutral_hidden())
    return feats


def _neutral_interactions() -> dict[str, float]:
    return {
        "news_sentiment_x_price_momentum": 0.0,
        "news_sentiment_x_volume_acceleration": 0.0,
        "news_density_x_relative_volume": 0.0,
        "news_relevance_x_vwap_distance": 0.0,
        "news_sentiment_x_volatility": 0.0,
        "earnings_news_x_iv_change": 0.0,
        "merger_news_x_options_flow": 0.0,
        "macro_news_x_index_strength": 0.0,
    }


def _neutral_hidden() -> dict[str, float]:
    return {
        "hidden_news_continuation_score": 0.0,
        "hidden_news_fade_score": 0.0,
        "hidden_sentiment_momentum_score": 0.0,
        "hidden_sentiment_reversal_score": 0.0,
        "hidden_catalyst_strength_score": 0.0,
        "hidden_news_density_score": 0.0,
        "hidden_news_volume_confirmation_score": 0.0,
        "hidden_news_volatility_expansion_score": 0.0,
        "hidden_news_price_dislocation_score": 0.0,
        "hidden_topic_momentum_score": 0.0,
    }


def _window_rows(rows: list[dict[str, Any]], as_of: datetime, minutes: float) -> list[dict[str, Any]]:
    cutoff = as_of - timedelta(minutes=minutes)
    out = []
    for r in rows:
        dt = _to_dt(r.get("published_at"))
        if dt is not None and cutoff <= dt <= as_of:
            out.append(r)
    return out


def _weighted_sentiment(rows: list[dict[str, Any]]) -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        s = _sentiment_of(r)
        if s is None:
            continue
        w = _relevance_of(r)
        num += s * w
        den += w
    return (num / den) if den > 0 else 0.0


def build_news_features(
    news_rows: list[dict[str, Any]],
    as_of: Optional[datetime] = None,
    price_context: Optional[dict[str, Any]] = None,
    is_premarket: bool = False,
) -> dict[str, Any]:
    """Build the full news feature dict.

    Parameters
    ----------
    news_rows : list of normalized news dicts (published_at ISO, sentiment, ...)
    as_of     : reference time (UTC). Defaults to the latest news time, else now.
    price_context : optional dict with keys: price_momentum, volume_acceleration,
                    relative_volume, vwap_distance, volatility, iv_change,
                    options_flow, index_strength.
    is_premarket : whether the current session is premarket (for catalyst flag).
    """
    rows = [r for r in (news_rows or []) if _to_dt(r.get("published_at")) is not None]
    if not rows:
        return neutral_news_features()

    times = [_to_dt(r["published_at"]) for r in rows]
    latest = max(times)
    if as_of is None:
        as_of = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)

    w15 = _window_rows(rows, as_of, 15)
    w30 = _window_rows(rows, as_of, 30)
    w60 = _window_rows(rows, as_of, 60)
    w24h = _window_rows(rows, as_of, 24 * 60)
    # Prior windows for change/acceleration (30-60m ago, 60-120m ago).
    prev30 = [r for r in _window_rows(rows, as_of, 60) if r not in w30]
    prev60 = [r for r in _window_rows(rows, as_of, 120) if r not in w60]

    minutes_since = max(0.0, (as_of - latest).total_seconds() / 60.0)

    sent_15 = _weighted_sentiment(w15)
    sent_30 = _weighted_sentiment(w30)
    sent_60 = _weighted_sentiment(w60)
    sent_24h = _weighted_sentiment(w24h)
    sent_prev30 = _weighted_sentiment(prev30)
    sent_prev60 = _weighted_sentiment(prev60)

    change_30 = sent_30 - sent_prev30
    change_60 = sent_60 - sent_prev60
    acceleration = change_30 - change_60

    # Sentiment volatility (std of per-article sentiment over 24h).
    sents_24h = [s for s in (_sentiment_of(r) for r in w24h) if s is not None]
    if len(sents_24h) >= 2:
        mean = sum(sents_24h) / len(sents_24h)
        var = sum((s - mean) ** 2 for s in sents_24h) / len(sents_24h)
        sentiment_volatility = var ** 0.5
    else:
        sentiment_volatility = 0.0

    # Density: articles/hour over the last 60m, squashed to [0,1].
    density = min(1.0, len(w60) / 6.0)  # 6 articles/hour -> saturated
    relevance_weighted = _weighted_sentiment(w24h)

    # Catalyst flags from topics across the recent (24h) window.
    topics_seen: set[str] = set()
    for r in w24h:
        for t in (r.get("topics") or []):
            topics_seen.add(str(t).lower())
    has_earnings = int(bool(topics_seen & _EARNINGS_TOPICS))
    has_merger = int(bool(topics_seen & _MERGER_TOPICS))
    has_macro = int(bool(topics_seen & _MACRO_TOPICS))
    has_monetary = int(bool(topics_seen & _MONETARY_TOPICS))
    has_fiscal = int(bool(topics_seen & _FISCAL_TOPICS))
    has_sector = int(bool(topics_seen & _SECTOR_TOPICS))
    has_recent = int(len(w60) > 0)
    has_premarket_catalyst = int(is_premarket and len(w60) > 0)

    # Catalyst strength: weighted blend of recency, density, topic salience, |sentiment|.
    catalyst_strength = _clip(
        0.35 * (1.0 if has_recent else 0.0)
        + 0.25 * density
        + 0.20 * min(1.0, (has_earnings + has_merger + has_macro + has_monetary + has_fiscal) / 2.0)
        + 0.20 * min(1.0, abs(sent_60)),
        0.0, 1.0,
    )

    feats: dict[str, Any] = {
        "news_count_15m": len(w15), "news_count_30m": len(w30),
        "news_count_60m": len(w60), "news_count_24h": len(w24h),
        "minutes_since_latest_news": round(minutes_since, 2),
        "sentiment_mean_15m": round(sent_15, 4), "sentiment_mean_30m": round(sent_30, 4),
        "sentiment_mean_60m": round(sent_60, 4), "sentiment_mean_24h": round(sent_24h, 4),
        "sentiment_change_30m": round(change_30, 4), "sentiment_change_60m": round(change_60, 4),
        "sentiment_acceleration": round(acceleration, 4),
        "sentiment_volatility": round(sentiment_volatility, 4),
        "news_density_score": round(density, 4),
        "news_relevance_weighted_sentiment": round(relevance_weighted, 4),
        "has_recent_news": has_recent, "has_premarket_catalyst": has_premarket_catalyst,
        "has_earnings_news": has_earnings, "has_merger_news": has_merger,
        "has_macro_news": has_macro, "has_monetary_policy_news": has_monetary,
        "has_fiscal_policy_news": has_fiscal, "has_sector_news": has_sector,
        "catalyst_strength_score": round(catalyst_strength, 4),
        "news_available": 1,
    }

    feats.update(build_news_price_interactions(feats, price_context))
    feats.update(build_hidden_news_features(feats))
    return feats


def build_news_price_interactions(news_feats: dict[str, Any],
                                  price_context: Optional[dict[str, Any]]) -> dict[str, float]:
    """Cross news features with price/options context. Neutral when context absent."""
    if not price_context:
        return _neutral_interactions()

    def g(k: str, default: float = 0.0) -> float:
        try:
            return float(price_context.get(k, default))
        except (TypeError, ValueError):
            return default

    sent60 = float(news_feats.get("sentiment_mean_60m", 0.0))
    relevance = float(news_feats.get("news_relevance_weighted_sentiment", 0.0))
    density = float(news_feats.get("news_density_score", 0.0))

    return {
        "news_sentiment_x_price_momentum": round(sent60 * g("price_momentum"), 5),
        "news_sentiment_x_volume_acceleration": round(sent60 * g("volume_acceleration"), 5),
        "news_density_x_relative_volume": round(density * g("relative_volume", 1.0), 5),
        "news_relevance_x_vwap_distance": round(relevance * g("vwap_distance"), 5),
        "news_sentiment_x_volatility": round(sent60 * g("volatility"), 5),
        "earnings_news_x_iv_change": round(news_feats.get("has_earnings_news", 0) * g("iv_change"), 5),
        "merger_news_x_options_flow": round(news_feats.get("has_merger_news", 0) * g("options_flow"), 5),
        "macro_news_x_index_strength": round(news_feats.get("has_macro_news", 0) * g("index_strength"), 5),
    }


def build_hidden_news_features(news_feats: dict[str, Any]) -> dict[str, float]:
    """Bounded composite 'hidden alpha' scores derived from news features."""
    sent60 = float(news_feats.get("sentiment_mean_60m", 0.0))
    sent24 = float(news_feats.get("sentiment_mean_24h", 0.0))
    change30 = float(news_feats.get("sentiment_change_30m", 0.0))
    accel = float(news_feats.get("sentiment_acceleration", 0.0))
    density = float(news_feats.get("news_density_score", 0.0))
    vol = float(news_feats.get("sentiment_volatility", 0.0))
    catalyst = float(news_feats.get("catalyst_strength_score", 0.0))

    continuation = _clip(sent60 * (0.5 + 0.5 * density) + 0.5 * change30)
    fade = _clip(-sent60 * vol + 0.5 * (sent24 - sent60))
    momentum = _clip(change30 + accel)
    reversal = _clip(-accel if abs(sent60) > 0.3 else 0.0)
    density_score = _clip(density)
    vol_confirm = _clip(sent60 * density)
    vol_expansion = _clip(vol * (1.0 + density))
    dislocation = _clip(abs(sent60) * density - 0.5 * abs(change30), 0.0, 1.0)
    topic_momentum = _clip(catalyst * (0.5 + 0.5 * (sent60 > 0) - 0.5 * (sent60 < 0)))

    return {
        "hidden_news_continuation_score": round(continuation, 4),
        "hidden_news_fade_score": round(fade, 4),
        "hidden_sentiment_momentum_score": round(momentum, 4),
        "hidden_sentiment_reversal_score": round(reversal, 4),
        "hidden_catalyst_strength_score": round(_clip(catalyst, 0.0, 1.0), 4),
        "hidden_news_density_score": round(density_score, 4),
        "hidden_news_volume_confirmation_score": round(vol_confirm, 4),
        "hidden_news_volatility_expansion_score": round(vol_expansion, 4),
        "hidden_news_price_dislocation_score": round(dislocation, 4),
        "hidden_topic_momentum_score": round(topic_momentum, 4),
    }
