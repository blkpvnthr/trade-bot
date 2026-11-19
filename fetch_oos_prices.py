#!/usr/bin/env python3
"""
fetch_oos_prices.py — fetch daily bars for the OUT-OF-SAMPLE insider universe.

Why a separate fetcher
    The in-sample cache was built with end=2026-06-07. OOS signals fall in 2026 Q2+, and
    their 21d/63d horizons exit in Jul-Sep 2026 -- BEYOND the cached bars -- so those trades
    would be silently dropped. This fetcher pulls full history through today for every
    OOS-eligible ticker (and SPY), OVERWRITING the parquet so the exit bars exist. The result
    is a superset of the old file (same old bars + new bars appended), safe for both studies.

Usage
    python fetch_oos_prices.py --events data/cache/oos_insider_events.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

BARS = Path("data/cache/alpaca_bars")
BARS.mkdir(parents=True, exist_ok=True)


def eligible_tickers(events_path: str, min_value: float, window_days: int) -> list[str]:
    ev = json.load(open(events_path))
    elig = []
    for sym, evs in ev.items():
        rows = sorted((date.fromisoformat(d), v) for d, v in evs)
        for d, _ in rows:
            if sum(v for dd, v in rows if d - timedelta(days=window_days) <= dd <= d) > min_value:
                elig.append(sym)
                break
    # clean tickers only (alpha, 1-5 chars) — same filter as fetch_pit_prices
    return [s for s in elig if s.isalpha() and 1 <= len(s) <= 5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/cache/oos_insider_events.json")
    ap.add_argument("--min-value", type=float, default=1_000_000.0)
    ap.add_argument("--window-days", type=int, default=2)
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    args = ap.parse_args()

    client = StockHistoricalDataClient(key, sec)
    feed = DataFeed.IEX if args.feed == "iex" else DataFeed.SIP

    elig = eligible_tickers(args.events, args.min_value, args.window_days)
    todo = sorted(set(elig) | {"SPY"})  # always (re)fetch SPY to extend benchmark through OOS exits
    print(f"OOS eligible(clean)={len(elig)}  to_fetch(incl SPY)={len(todo)}  feed={args.feed}")

    start = datetime(2020, 6, 1)
    end = datetime.now() - timedelta(days=1)  # through most recent complete session
    fetched = empty = 0
    B = 100
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                   start=start, end=end, feed=feed)
            df = client.get_stock_bars(req).df
        except Exception as e:
            print(f"  batch {i//B}: ERR {str(e)[:70]}")
            continue
        if df is None or df.empty:
            empty += len(batch)
            continue
        got = set(df.index.get_level_values(0))
        for s in batch:
            if s not in got:
                empty += 1
                continue
            d = df.xs(s, level=0)[["open", "high", "low", "close", "volume"]].copy()
            d.index.name = "timestamp"
            if len(d) < 30:
                empty += 1
                continue
            d.to_parquet(BARS / f"{s}_day.parquet")
            fetched += 1
        print(f"  batch {i//B+1}/{(len(todo)+B-1)//B}: fetched so far={fetched} empty={empty}")

    print(f"DONE fetched={fetched} empty/unavailable={empty}")
    have = {os.path.basename(p).rsplit("_day", 1)[0] for p in glob.glob(str(BARS / "*_day.parquet"))}
    priced = [s for s in elig if s in have]
    cov = {"eligible": len(elig), "priced": len(priced),
           "drop_rate": round(1 - len(priced) / len(elig), 3) if elig else None}
    json.dump(cov, open("data/cache/oos_coverage.json", "w"))
    print(f"COVERAGE: eligible={len(elig)} priced={len(priced)} "
          f"unpriceable={len(elig)-len(priced)} drop_rate={cov['drop_rate']}")


if __name__ == "__main__":
    main()
