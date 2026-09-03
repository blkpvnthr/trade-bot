#!/usr/bin/env python3
"""Fetch historical OHLCV bars from Alpaca into the local CSV cache.

RESEARCH-ONLY: reads the Alpaca Data API only. No orders, no trading API.

    python scripts/fetch_alpaca_bars.py --symbols NVDA GOOGL ASML TSM KLAC FIGS TGTX YOU \
        --timeframe 1Day --start 2018-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.data import (load_bars, read_universe, cache_path, TIMEFRAMES,
                          DataError)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="tickers (default: config/universe.json)")
    ap.add_argument("--timeframe", default="1Day", choices=sorted(TIMEFRAMES))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download, ignore cache")
    args = ap.parse_args()

    symbols = args.symbols or read_universe()
    print(f"Fetching {len(symbols)} symbols @ {args.timeframe} from {args.start} "
          f"to {args.end or 'today'}\n")

    ok = 0
    for sym in symbols:
        try:
            df = load_bars(sym, args.timeframe, args.start, args.end,
                           refresh=args.refresh)
        except DataError as e:
            print(f"  {sym:<6} ERROR: {e}")
            continue
        path = cache_path(sym, args.timeframe, args.start, args.end or _today())
        print(f"  {sym:<6} {len(df):>6} bars -> {path.name}")
        ok += 1
    print(f"\n{ok}/{len(symbols)} symbols cached under data/bars/")


def _today():
    from datetime import date
    return date.today().isoformat()


if __name__ == "__main__":
    main()
