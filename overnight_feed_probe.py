#!/usr/bin/env python3
"""overnight_feed_probe.py — how far back does Alpaca extended-hours / overnight minute history go?

The overnight evaluator wants a realistic gap-risk / overnight-drawdown distribution. The daily
alpaca_bars cache only has regular-session OHLC (close→next-open proxy). True overnight risk needs
extended-hours minute bars (premarket 04:00-09:30, after-hours 16:00-20:00, overnight 20:00-04:00 ET).

This probes single trading days across years and reports, per session, how many minute bars exist —
so we KNOW the usable backtest depth before relying on it (per the explicit caveat).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

client = StockHistoricalDataClient(key, sec)
ET = ZoneInfo("America/New_York")
SYM = "AAPL"


def session(ts_utc) -> str:
    t = ts_utc.astimezone(ET)
    hm = t.hour * 60 + t.minute
    if 9 * 60 + 30 <= hm < 16 * 60:
        return "regular"
    if 4 * 60 <= hm < 9 * 60 + 30:
        return "premarket"
    if 16 * 60 <= hm < 20 * 60:
        return "afterhours"
    return "overnight"          # 20:00-04:00 ET (Blue Ocean)


def probe(day: str, feed: DataFeed):
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) - timedelta(hours=8)
    end = start + timedelta(days=1, hours=12)
    try:
        df = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=[SYM], timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start, end=end, feed=feed)).df
    except Exception as e:
        return f"ERROR {str(e)[:70]}"
    if df is None or df.empty:
        return "no data"
    df = df.reset_index()
    counts = {"premarket": 0, "regular": 0, "afterhours": 0, "overnight": 0}
    for ts in df["timestamp"]:
        counts[session(ts.to_pydatetime().astimezone(timezone.utc))] += 1
    ext = counts["premarket"] + counts["afterhours"] + counts["overnight"]
    return (f"reg={counts['regular']:>3}  pre={counts['premarket']:>3}  "
            f"aft={counts['afterhours']:>3}  ovn={counts['overnight']:>3}  | extended={ext}")


def main():
    print(f"ALPACA_DATA_FEED={os.environ.get('ALPACA_DATA_FEED')}  probe symbol={SYM}\n")
    days = ["2016-06-15", "2018-06-15", "2020-06-15", "2022-06-15",
            "2024-06-17", "2025-06-16", "2026-06-08"]
    for feed in (DataFeed.SIP, DataFeed.IEX):
        print(f"=== feed={feed.value} ===")
        for d in days:
            print(f"  {d}: {probe(d, feed)}")
        print()


if __name__ == "__main__":
    main()
