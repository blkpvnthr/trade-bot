#!/usr/bin/env python3
"""
enrich_events_with_news.py — inject PIT-safe Alpaca-news features into RL research events.

The RL datasets are built from research events that currently carry NO news columns, so
rc._news_vals() extracts all-zeros — enabling RL_NEWS_FEATURES=1 would train on zeros. This
module backfills real Alpaca news (FinBERT-scored) and, for each event, computes the 10
rl_common.NEWS_FEATURES values from the news that existed STRICTLY AT OR BEFORE the event
timestamp (no lookahead), writing them into the event dict.

Cost: each symbol's news is backfilled ONCE (cached in feature_store/news/<SYM>.jsonl), then every
event for that symbol is served from the local cache — one API sweep per symbol, not per event.

PIT-safety: the per-event news window is [event_ts - WINDOW_HOURS, event_ts], published_at ≤ event_ts.
The 3 compressed_news_embedding_* features fit a PCA on the representative news text across events —
a representation (not the label), so it does not leak realized_R; the leakage audit still gates it.

Used by rl_training_dataset_builder.build_all() (flag-gated on RL_NEWS_FEATURES=1). Standalone:
    python enrich_events_with_news.py        # report news coverage over the current event set
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger("enrich_events_with_news")

WINDOW_HOURS = int(os.getenv("RL_NEWS_WINDOW_HOURS", "24"))
BACKFILL_BUFFER_DAYS = 3

# The 10 RL-consumed news features (rl_common.NEWS_FEATURES). 7 come from build_news_features,
# 3 (compressed_news_embedding_*) from the embedding engine.
_DIRECT_KEYS = ["sentiment_mean_60m", "sentiment_change_30m", "sentiment_acceleration",
                "news_density_score", "catalyst_strength_score",
                "hidden_news_continuation_score", "hidden_sentiment_momentum_score"]
_EMB_KEYS = ["compressed_news_embedding_1", "compressed_news_embedding_2", "compressed_news_embedding_3"]


def _parse_ts(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def enrich_events(events: list[dict], window_hours: int = WINDOW_HOURS) -> dict[str, Any]:
    """Mutate `events` in place, adding the 10 NEWS_FEATURES per event from PIT-safe Alpaca news.
    Returns a stats dict. Fail-safe: on any error an event keeps neutral (absent → 0.0 downstream)."""
    from data_sources.alpaca_news_client import get_default_alpaca_news_client
    from alpha_discovery.news_feature_builder import build_news_features
    try:
        from alpha_discovery.news_embedding_engine import NewsEmbeddingEngine
    except Exception:
        NewsEmbeddingEngine = None  # type: ignore

    client = get_default_alpaca_news_client()
    win = timedelta(hours=window_hours)

    # event timestamps + symbols
    parsed = [(_parse_ts(e.get("timestamp")), str(e.get("symbol") or "").upper()) for e in events]
    syms = sorted({s for ts, s in parsed if ts and s})
    if not syms:
        return {"events": len(events), "symbols": 0, "enriched": 0, "with_news": 0}

    # one backfill sweep per symbol, covering [min_event_ts - window, now]
    ts_all = [ts for ts, _ in parsed if ts]
    span_days = (datetime.now(timezone.utc) - min(ts_all)).days + window_hours // 24 + BACKFILL_BUFFER_DAYS
    hist: dict[str, list[dict]] = {}
    for s in syms:
        try:
            client.backfill_symbol(s, days=max(span_days, 7), persist=True)
            rows = client.load_symbol_history(s)
            for r in rows:
                r["_pub"] = _parse_ts(r.get("published_at"))
            hist[s] = [r for r in rows if r.get("_pub")]
        except Exception as exc:
            logger.warning("backfill failed for %s: %s", s, str(exc)[:120])
            hist[s] = []

    # per-event direct features + collect a representative news row for the embedding fit
    enriched = with_news = 0
    rep_rows: list[Optional[dict]] = [None] * len(events)
    for i, (e, (ts, sym)) in enumerate(zip(events, parsed)):
        if not (ts and sym):
            continue
        rows = hist.get(sym, [])
        window = [r for r in rows if (ts - win) <= r["_pub"] <= ts]   # PIT: published_at ≤ event_ts
        try:
            nf = build_news_features(window, as_of=ts)
        except Exception:
            continue
        for k in _DIRECT_KEYS:
            if k in nf:
                e[k] = nf[k]
        enriched += 1
        if window:
            with_news += 1
            rep_rows[i] = max(window, key=lambda r: r["_pub"])   # most-recent in-window news

    # 3 compressed embeddings: fit one PCA on all representative rows, transform per event
    if NewsEmbeddingEngine is not None:
        idx = [i for i, r in enumerate(rep_rows) if r is not None]
        if idx:
            try:
                eng = NewsEmbeddingEngine().fit([rep_rows[i] for i in idx])
                embs = eng.transform([rep_rows[i] for i in idx])
                for i, emb in zip(idx, embs):
                    for k in _EMB_KEYS:
                        if k in emb:
                            events[i][k] = emb[k]
            except Exception as exc:
                logger.warning("embedding fit/transform failed: %s", str(exc)[:120])

    return {"events": len(events), "symbols": len(syms), "enriched": enriched, "with_news": with_news}


if __name__ == "__main__":
    import json
    import rl_common as rc
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evs = rc.load_research_events() + rc.load_backfill_events()
    stats = enrich_events(evs)
    print(json.dumps(stats, indent=2))
    # coverage sample
    nz = [e for e in evs if any(abs(float(e.get(k, 0) or 0)) > 1e-9 for k in _DIRECT_KEYS)]
    print(f"events with non-zero news features: {len(nz)}/{len(evs)}")
    if nz:
        s = nz[0]
        print("sample:", {k: round(float(s.get(k, 0) or 0), 4) for k in _DIRECT_KEYS + _EMB_KEYS})
