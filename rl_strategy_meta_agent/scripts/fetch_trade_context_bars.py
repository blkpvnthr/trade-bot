#!/usr/bin/env python3
"""Fetch 15Min OHLCV context bars for every symbol in the trade-outcome dataset.

For each (symbol, timeframe): need >= lookback-days before the earliest entry
through the latest exit. Fetches only missing ranges, merges with cache (no
duplicate rows), validates coverage, and writes a coverage report. Data pull
only -- no orders, no live trading. Credentials come from the environment /
a local .env (APCA_API_KEY_ID / APCA_API_SECRET_KEY).

    python scripts/fetch_trade_context_bars.py \
        --dataset data/trade_outcomes/trade_outcome_dataset.parquet \
        --timeframe 15Min --lookback-days 30 --bars-dir data/bars
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.bar_coverage import fetch_range, fetch_with_coverage, coverage_ok
from rl_meta.data import DataError


def _load_dataset(path: Path) -> pd.DataFrame:
    cols = ["symbol", "timeframe", "entry_time", "exit_time"]
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--timeframe", default="15Min")
    ap.add_argument("--lookback-days", type=int, default=30)
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None, help="limit to these symbols")
    args = ap.parse_args()

    ds = _load_dataset(Path(args.dataset))
    ds = ds[ds["timeframe"] == args.timeframe]
    grp = ds.groupby("symbol").agg(earliest=("entry_time", "min"),
                                   latest=("exit_time", "max")).reset_index()
    if args.symbols:
        grp = grp[grp["symbol"].isin(args.symbols)]

    bars_dir = Path(args.bars_dir)
    stats = {"fetched": 0, "cached": 0, "failed": 0, "covered": 0}
    report = []
    print(f"Fetching {args.timeframe} context bars for {len(grp)} symbols "
          f"(lookback {args.lookback_days}d)...\n")
    for _, r in grp.iterrows():
        sym = r["symbol"]
        req_start, req_end = fetch_range(r["earliest"], r["latest"], args.lookback_days)
        rec = {"symbol": sym, "timeframe": args.timeframe,
               "required_start": req_start, "required_end": req_end,
               "cached_start": None, "cached_end": None, "rows": 0,
               "coverage_ok": False, "missing_reason": None}
        try:
            df, status = fetch_with_coverage(sym, args.timeframe, req_start, req_end,
                                             bars_dir, refresh=args.refresh)
            stats[status] = stats.get(status, 0) + 1
            if not df.empty:
                rec["cached_start"] = df["timestamp"].min().date().isoformat()
                rec["cached_end"] = df["timestamp"].max().date().isoformat()
                rec["rows"] = len(df)
            ok, reason = coverage_ok(df, r["earliest"], r["latest"])
            rec["coverage_ok"], rec["missing_reason"] = ok, (None if ok else reason)
            stats["covered"] += int(ok)
            print(f"  {sym:<6} {status:<8} rows={len(df):>6} "
                  f"{'OK' if ok else 'MISS: ' + str(reason)}")
        except DataError as e:
            stats["failed"] += 1
            rec["missing_reason"] = str(e)
            print(f"  {sym:<6} FAILED   {e}")
        except Exception as e:                         # network / data error
            stats["failed"] += 1
            rec["missing_reason"] = f"{type(e).__name__}: {e}"
            print(f"  {sym:<6} ERROR    {type(e).__name__}: {str(e)[:80]}")
        report.append(rec)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "bar_coverage_report.json").write_text(json.dumps(report, indent=2))
    n = len(grp)
    print("\n=== Coverage summary ===")
    print(f"Symbols required: {n}")
    print(f"Fetched:          {stats['fetched']}")
    print(f"From cache:       {stats['cached']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Full coverage:    {stats['covered']}/{n} "
          f"({stats['covered']/n*100:.0f}%)" if n else "")
    print(f"\ncoverage report -> {out / 'bar_coverage_report.json'}")
    if stats["covered"]:
        print("Next: rebuild the dataset to populate pre-entry features:\n"
              "  python scripts/build_trade_outcome_dataset.py --timeframe 15Min")


if __name__ == "__main__":
    main()
