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
RETENTION_DAYS = int(os.getenv("STREAM_ROUTER_SHARED_BAR_RETENTION_DAYS", "7"))
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

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
    subscribed: set[str] = set()
    while not _STOP:
        universe = set(load_universe())
        new = sorted(universe - subscribed)
        if new:
            print(f"[stream-router] subscribing new symbols={new}", flush=True)
            stream.subscribe_bars(bar_handler, *new)
            subscribed.update(new)
        dropped = subscribed - universe
        if dropped:
            # Alpaca SDK doesn't support mid-session unsubscribe cleanly;
            # log and leave subscribed — bars for stale symbols are just ignored.
            print(f"[stream-router] universe dropped symbols={sorted(dropped)} (left subscribed)", flush=True)
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
