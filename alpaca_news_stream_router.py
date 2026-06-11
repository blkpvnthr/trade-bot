#!/usr/bin/env python3
"""
alpaca_news_stream_router.py — the ONE real-time Alpaca news connection for the system.

Mirrors alpaca_stream_router.py (StockDataStream) but for news: owns a single
alpaca.data.live.news.NewsDataStream, FinBERT-scores each article, and appends normalized rows
to feature_store/news/<SYMBOL>.jsonl — the same store the historical AlpacaNewsClient reads.
So the live path gets fresh, sentiment-scored news in real time while on-demand historical fetch
still works if this feeder isn't running.

Run as a service (like trade-bot-instance2):  python alpaca_news_stream_router.py
Creds: APCA_API_KEY_ID / APCA_API_SECRET_KEY. Advisory-safe: the handler NEVER raises.

News is far lower-volume than bars, so there is no 30-symbol rotation — we subscribe the whole
locked universe (NEWS_STREAM_SYMBOLS=* subscribes ALL news, a firehose; off by default).
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from data_sources import finbert_scorer
from data_sources.alpaca_news_client import normalize_alpaca_news_item, DEFAULT_FEATURE_STORE

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TAG = datetime.now(ET).strftime("%Y%m%d")

FEATURE_STORE = DEFAULT_FEATURE_STORE
FEATURE_STORE.mkdir(parents=True, exist_ok=True)

_UNIVERSE_DATED = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"
_UNIVERSE_UNDATED = PROJECT_ROOT / "locked_trading_universe.txt"
RELOAD_SECONDS = int(os.getenv("NEWS_STREAM_RELOAD_SECONDS", "60"))
# "*" subscribes to ALL news (firehose). Default empty → subscribe the locked universe only.
NEWS_STREAM_SYMBOLS = os.getenv("NEWS_STREAM_SYMBOLS", "").strip()

_STOP = False
_SEEN: dict[str, set] = {}   # symbol -> set of dedupe keys already persisted


def _normalize(x: Any) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def load_universe() -> list[str]:
    for path in (_UNIVERSE_DATED, _UNIVERSE_UNDATED):
        if path.exists():
            try:
                return [s for s in (_normalize(x) for x in path.read_text().splitlines()) if s]
            except Exception:
                pass
    return []


def _seen_for(symbol: str) -> set:
    if symbol not in _SEEN:
        ids: set = set()
        p = FEATURE_STORE / f"{symbol}.jsonl"
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        ids.add(r.get("url") or r.get("published_at"))
            except Exception:
                pass
        _SEEN[symbol] = ids
    return _SEEN[symbol]


def append_news_row(symbol: str, row: dict[str, Any]) -> None:
    """Append one normalized row to <SYMBOL>.jsonl, deduped on url/published_at. Never raises."""
    try:
        seen = _seen_for(symbol)
        key = row.get("url") or row.get("published_at")
        if key in seen:
            return
        seen.add(key)
        with (FEATURE_STORE / f"{symbol}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"[news-stream] persist error {symbol}: {exc}", flush=True)


async def news_handler(news: Any) -> None:
    """Score once per article, persist a row per mentioned symbol. NEVER raises."""
    try:
        headline = str(getattr(news, "headline", "") or "")
        summary = str(getattr(news, "summary", "") or "")
        sent = finbert_scorer.score_text((headline + ". " + summary).strip())
        nsyms = [_normalize(s) for s in (getattr(news, "symbols", None) or [])]
        for sym in nsyms:
            if not sym:
                continue
            append_news_row(sym, normalize_alpaca_news_item(news, sym, sent))
        print(f"[news-stream] {headline[:70]} | {nsyms} | {sent.get('label')}", flush=True)
    except Exception as exc:  # pragma: no cover - handler must never crash the stream
        print(f"[news-stream] handler error: {exc}", flush=True)


async def subscription_manager(stream: Any) -> None:
    """Subscribe the locked universe (or all news when NEWS_STREAM_SYMBOLS=*); resubscribe deltas."""
    subscribed: set[str] = set()
    if NEWS_STREAM_SYMBOLS == "*":
        stream.subscribe_news(news_handler, "*")
        print("[news-stream] subscribed ALL news (*)", flush=True)
        return
    while not _STOP:
        target = set(load_universe())
        add = target - subscribed
        drop = subscribed - target
        if drop:
            try:
                stream.unsubscribe_news(*sorted(drop))
            except Exception as exc:
                print(f"[news-stream] unsubscribe error: {exc}", flush=True)
            subscribed.difference_update(drop)
        if add:
            stream.subscribe_news(news_handler, *sorted(add))
            subscribed.update(add)
            print(f"[news-stream] subscribed {len(subscribed)} symbols (+{len(add)})", flush=True)
        await asyncio.sleep(RELOAD_SECONDS)


async def run_stream() -> None:
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        print("[news-stream] APCA_API_KEY_ID / APCA_API_SECRET_KEY not set", flush=True)
        sys.exit(1)
    from alpaca.data.live.news import NewsDataStream
    stream = NewsDataStream(api_key=key, secret_key=secret)
    manager = asyncio.create_task(subscription_manager(stream))
    try:
        await stream._run_forever()
    finally:
        manager.cancel()


def _handle_stop(signum: int, frame: Any) -> None:
    global _STOP
    _STOP = True
    print("[news-stream] shutdown requested", flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    print(f"[news-stream] starting | finbert={finbert_scorer.is_available()} store={FEATURE_STORE}", flush=True)
    try:
        asyncio.run(run_stream())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[news-stream] fatal: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
