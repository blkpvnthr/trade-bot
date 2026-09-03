#!/usr/bin/env python3
"""
alpaca_stream_router.py

Single-source Alpaca WebSocket stream router.

Owns the ONE StockDataStream connection for the whole system and writes
real-time 1-minute bars to the shared CSV consumed by both
stream_confirmed_movement_gate_trade_plan_trailing.py and sr_rl_agent.py.

Neither of those modules should open their own Alpaca data connections when
this router is running. Set STREAM_ROUTER_MODE=1 in their env to engage
shared-cache mode.

Run:
    python alpaca_stream_router.py

Key env vars:
    APCA_API_KEY_ID                        Alpaca key (required)
    APCA_API_SECRET_KEY                    Alpaca secret (required)
    ALPACA_DATA_FEED                       "iex" (default) or "sip"
    STREAM_ROUTER_RELOAD_SECONDS           universe refresh interval (default 30)
    STREAM_ROUTER_SHARED_BAR_RETENTION_DAYS days of bars to keep in CSV (default 7)
    RUN_DATA_DATE                          YYYYMMDD date override
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

ET = ZoneInfo("America/New_York")
_TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", _TODAY_TAG))

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared bar cache — the single file all downstream modules read from.
SHARED_BARS_1M_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_bars_1m.csv"

# Universe files written by the premarket pipeline / opportunity finder.
_UNIVERSE_DATED = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"
_UNIVERSE_UNDATED = PROJECT_ROOT / "locked_trading_universe.txt"

RELOAD_SECONDS = int(os.getenv("STREAM_ROUTER_RELOAD_SECONDS", "30"))
# Alpaca caps a single data stream at 30 symbols. When the universe exceeds the cap we ROTATE a
# sliding window so every symbol gets streamed in turn (each window dwells STREAM_ROTATE_SECONDS,
# >= 60s so each symbol captures at least one full 1-min bar before rotating out).
STREAM_MAX_SYMBOLS = int(os.getenv("STREAM_MAX_SYMBOLS", "30"))
STREAM_ROTATE_SECONDS = int(os.getenv("STREAM_ROTATE_SECONDS", "60"))
RETENTION_DAYS = int(os.getenv("STREAM_ROUTER_SHARED_BAR_RETENTION_DAYS", "7"))
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip").lower()

_STOP = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize(x: Any) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def get_feed() -> DataFeed:
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def load_universe() -> list[str]:
    for path in (_UNIVERSE_DATED, _UNIVERSE_UNDATED):
        if path.exists():
            try:
                syms = [_normalize(s) for s in path.read_text().splitlines() if s.strip()]
                return [s for s in syms if s]
            except Exception:
                pass
    return []


def persist_bar(row: dict[str, Any]) -> None:
    """Append one bar dict to SHARED_BARS_1M_PATH, deduplicating on (symbol, timestamp)."""
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(row):
        return
    try:
        new = pd.DataFrame([row])
        new["symbol"] = new["symbol"].map(_normalize)
        new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True, errors="coerce")
        new = new.dropna(subset=["symbol", "timestamp", "close"])
        if new.empty:
            return

        if SHARED_BARS_1M_PATH.exists():
            old = pd.read_csv(SHARED_BARS_1M_PATH)
            old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
            old["symbol"] = old["symbol"].map(_normalize)
            combined = pd.concat([old, new], ignore_index=True)
        else:
            combined = new

        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        combined = (
            combined
            .drop_duplicates(["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"])
        )
        combined = combined[combined["timestamp"] >= cutoff]
        combined[list(required)].to_csv(SHARED_BARS_1M_PATH, index=False)
    except Exception as exc:
        print(f"[stream-router] bar persist error: {exc}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Async stream loop
# ─────────────────────────────────────────────────────────────────────────────


async def bar_handler(bar: Any) -> None:
    symbol = _normalize(getattr(bar, "symbol", ""))
    if not symbol:
        return
    persist_bar({
        "symbol": symbol,
        "timestamp": getattr(bar, "timestamp", datetime.now(timezone.utc)),
        "open": float(getattr(bar, "open", 0.0)),
        "high": float(getattr(bar, "high", 0.0)),
        "low": float(getattr(bar, "low", 0.0)),
        "close": float(getattr(bar, "close", 0.0)),
        "volume": float(getattr(bar, "volume", 0.0)),
    })


async def subscription_manager(stream: StockDataStream) -> None:
    """Keep the live subscription in sync with the universe, honoring Alpaca's symbol cap.

    universe <= cap : subscribe all (steady state, no churn).
    universe  > cap : rotate a sliding window of `cap` symbols so EVERY symbol gets streamed in
                      turn — subscribe the window's new names, unsubscribe those rotating out,
                      advance the window each cycle. Full coverage every ceil(n/cap) cycles."""
    subscribed: set[str] = set()
    offset = 0
    while not _STOP:
        universe = list(dict.fromkeys(load_universe()))   # ordered, de-duped
        n = len(universe)
        if n == 0:
            await asyncio.sleep(RELOAD_SECONDS)
            continue
        cap = max(1, STREAM_MAX_SYMBOLS)
        rotating = n > cap
        if rotating:
            window = [universe[(offset + i) % n] for i in range(cap)]
            target = set(window)
            offset = (offset + cap) % n
        else:
            target = set(universe)

        add = sorted(target - subscribed)
        drop = sorted(subscribed - target)
        if drop:
            try:
                stream.unsubscribe_bars(*drop)
            except Exception as exc:
                print(f"[stream-router] unsubscribe error (continuing): {exc}", flush=True)
            subscribed.difference_update(drop)
        if add:
            stream.subscribe_bars(bar_handler, *add)
            subscribed.update(add)

        if rotating:
            print(f"[stream-router] rotating window {len(subscribed)}/{n} cap={cap} "
                  f"+{add} -{drop}", flush=True)
            await asyncio.sleep(max(RELOAD_SECONDS, STREAM_ROTATE_SECONDS))
        else:
            if add or drop:
                print(f"[stream-router] subscribed={len(subscribed)}/{n} +{add} -{drop}", flush=True)
            await asyncio.sleep(RELOAD_SECONDS)


async def run_stream() -> None:
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        print("[stream-router] APCA_API_KEY_ID / APCA_API_SECRET_KEY not set", flush=True)
        sys.exit(1)

    stream = StockDataStream(api_key=key, secret_key=secret, feed=get_feed())
    manager = asyncio.create_task(subscription_manager(stream))
    try:
        await stream._run_forever()
    finally:
        manager.cancel()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _handle_stop(signum: int, frame: Any) -> None:
    global _STOP
    _STOP = True
    print("[stream-router] shutdown requested", flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print(f"[stream-router] starting | feed={ALPACA_DATA_FEED} data_tag={DATA_TAG}", flush=True)
    print(f"[stream-router] shared_bars={SHARED_BARS_1M_PATH}", flush=True)
    print(f"[stream-router] universe_reload_seconds={RELOAD_SECONDS}", flush=True)

    try:
        asyncio.run(run_stream())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[stream-router] fatal: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
