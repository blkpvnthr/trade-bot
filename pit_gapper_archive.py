#!/usr/bin/env python3
"""
Point-in-time premarket-gapper ARCHIVE collector.  DATA COLLECTION ONLY.

Run once per trading morning during premarket. Snapshots that day's TradingView
premarket-gapper scan to a dated CSV so future studies use only the symbols that
actually appeared on that day's scan - killing the survivorship/selection bias of
backtesting today's scanner universe backward.

    data/pit_gappers/YYYY-MM-DD_premarket_gappers.csv

Idempotent: re-running appends a new snapshot only if its scan_timestamp_et is new;
existing snapshots are never overwritten. --once skips entirely if today already
has a file (safe for launchd double-fires / catch-up-on-wake).

Places no orders. Connects to no broker.

Schedule (launchd, ~weekday 09:00 ET premarket):
    python pit_gapper_archive.py --once
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import dotenv
from dotenv import load_dotenv

load_dotenv(dotenv.find_dotenv())

ET = ZoneInfo("America/New_York")

# exact CSV schema (column order is the archive contract - keep stable)
FIELDS = [
    "scan_timestamp_et", "scan_date", "symbol", "exchange", "premarket_gap",
    "premarket_close", "premarket_volume", "close", "volume", "market_cap_basic",
    "description", "type", "typespecs", "active_symbol", "is_primary",
]


def scan(gap_threshold: float, limit: int) -> pd.DataFrame:
    """Run the TradingView premarket-gapper scan and normalise to FIELDS."""
    from tradingview_screener import Query, col

    _, df = (
        Query()
        .select(
            "name", "exchange", "premarket_gap", "premarket_close", "premarket_volume",
            "close", "volume", "market_cap_basic", "description", "type", "typespecs",
            "active_symbol", "is_primary",
        )
        .where(
            col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]),
            col("is_primary") == True,  # noqa: E712
            col("typespecs").has("common"),
            col("typespecs").has_none_of("preferred"),
            col("type") == "stock",
            col("premarket_gap").not_empty(),
            col("active_symbol") == True,  # noqa: E712
        )
        .order_by("premarket_gap", ascending=False, nulls_first=False)
        .limit(max(limit, 1))
        .set_markets("america")
        .set_property("symbols", {"query": {"types": ["stock", "fund", "dr", "structured"]}})
        .set_property("preset", "pre_market_gappers")
        .get_scanner_data()
    )

    df = df.copy()
    df["premarket_gap"] = pd.to_numeric(df["premarket_gap"], errors="coerce")
    df = df[df["premarket_gap"] >= gap_threshold].copy()
    df = df.sort_values("premarket_gap", ascending=False).head(limit)
    df["symbol"] = df["name"].astype(str).str.replace(r"^[A-Z]+:", "", regex=True)
    return df


def build_snapshot(df: pd.DataFrame, scan_ts: str, scan_date: str) -> pd.DataFrame:
    df = df.copy()
    df["scan_timestamp_et"] = scan_ts
    df["scan_date"] = scan_date
    for c in FIELDS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[FIELDS]


def main():
    p = argparse.ArgumentParser(description="Point-in-time premarket-gapper archive (data only)")
    p.add_argument("--gap-threshold", type=float, default=5.0, help="percent, default 5")
    p.add_argument("--limit", type=int, default=100, help="keep top N by premarket_gap")
    p.add_argument("--out-dir", default="data/pit_gappers")
    p.add_argument("--once", action="store_true",
                   help="skip if today already has a file (idempotent once-per-day)")
    p.add_argument("--dry-run", action="store_true", help="scan and print, write nothing")
    a = p.parse_args()

    now = datetime.now(ET)
    scan_ts = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    scan_date = now.strftime("%Y-%m-%d")
    out_path = os.path.join(a.out_dir, f"{scan_date}_premarket_gappers.csv")

    exists = os.path.exists(out_path)
    if a.once and exists and not a.dry_run:
        print(f"--once: {out_path} already exists; skipping (idempotent).")
        return

    df = scan(a.gap_threshold, a.limit)
    snap = build_snapshot(df, scan_ts, scan_date)

    print("=== PIT premarket-gapper snapshot ===")
    print(f"Scan time (ET): {scan_ts}")
    print(f"Symbols (gap >= {a.gap_threshold}%, top {a.limit}): {len(snap)}")
    top = snap.sort_values("premarket_gap", ascending=False).head(10)
    print("Top 10 gappers:")
    for _, r in top.iterrows():
        print(f"  {r['symbol']:<8} gap={float(r['premarket_gap']):.1f}%  "
              f"pmvol={r['premarket_volume']}")
    print(f"Output path: {out_path}")

    if a.dry_run:
        print("--dry-run: nothing written.")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    if not exists:
        snap.to_csv(out_path, index=False)
        print(f"Wrote new archive ({len(snap)} rows).")
        return

    # idempotent append: only if this scan_timestamp_et is not already present
    prior = pd.read_csv(out_path)
    if "scan_timestamp_et" in prior.columns and scan_ts in set(prior["scan_timestamp_et"].astype(str)):
        print("scan_timestamp_et already archived; nothing to do.")
        return
    snap.to_csv(out_path, mode="a", header=False, index=False)
    print(f"Appended snapshot ({len(snap)} rows) - file now has "
          f"{prior['scan_timestamp_et'].nunique() + 1 if 'scan_timestamp_et' in prior else 2} snapshots.")


if __name__ == "__main__":
    main()
