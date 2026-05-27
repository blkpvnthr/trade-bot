#!/usr/bin/env python3
"""
trade_execution_journaler.py

Watches:
    /root/trade-bot/logs/run.log

Extracts executed trade activity and appends structured records to:
    /root/trade-bot/logs/execution/executed_trade_journal.csv

Features:
- tails run.log in real time
- journals BUY/SELL executions
- captures timestamps, symbols, qty, side, price
- deduplicates repeated lines
- survives log rotation
- flushes immediately
- lightweight daemon-friendly process

Run:
    python trade_execution_journaler.py

Optional background:
    nohup python trade_execution_journaler.py > journaler.out 2>&1 &
"""

from __future__ import annotations

import csv
import os
import re
import time
from datetime import datetime, UTC
from pathlib import Path

RUN_LOG = Path("/root/trade-bot/logs/run.log")
OUTPUT_CSV = Path("/root/trade-bot/logs/execution/executed_trade_journal.csv")

POLL_INTERVAL = 0.5

# Adjust/add patterns based on your actual execution log format.
TRADE_PATTERNS = [
    re.compile(
        r"(?P<side>BUY|SELL).*?(?P<symbol>[A-Z]{1,10}).*?(qty|shares)=?(?P<qty>[0-9.]+).*?(price|filled_avg_price)=?(?P<price>[0-9.]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"submitted (?P<side>buy|sell).*?(?P<symbol>[A-Z]{1,10}).*?(?P<qty>[0-9.]+).*?\$?(?P<price>[0-9.]+)",
        re.IGNORECASE,
    ),
]

CSV_HEADERS = [
    "timestamp_utc",
    "side",
    "symbol",
    "qty",
    "price",
    "raw_line",
]

_seen_lines: set[str] = set()


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_csv_exists() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def append_trade(
    side: str,
    symbol: str,
    qty: str,
    price: str,
    raw_line: str,
) -> None:
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                now_utc_iso(),
                side.upper(),
                symbol.upper(),
                qty,
                price,
                raw_line.strip(),
            ]
        )
        f.flush()

    print(
        f"[journaled] {side.upper()} {symbol.upper()} qty={qty} price={price}"
    )


def parse_trade(line: str) -> bool:
    clean = line.strip()

    if not clean:
        return False

    if clean in _seen_lines:
        return False

    for pattern in TRADE_PATTERNS:
        match = pattern.search(clean)

        if match:
            _seen_lines.add(clean)

            data = match.groupdict()

            append_trade(
                side=data.get("side", ""),
                symbol=data.get("symbol", ""),
                qty=data.get("qty", ""),
                price=data.get("price", ""),
                raw_line=clean,
            )
            return True

    return False


def follow(file_path: Path):
    """
    Tail -F behavior:
    - follows appended lines
    - survives truncation/rotation
    """

    while True:
        try:
            with open(file_path, "r") as f:
                f.seek(0, os.SEEK_END)

                inode = os.fstat(f.fileno()).st_ino

                while True:
                    line = f.readline()

                    if line:
                        yield line
                        continue

                    try:
                        current_inode = os.stat(file_path).st_ino

                        # log rotated/recreated
                        if current_inode != inode:
                            break

                    except FileNotFoundError:
                        break

                    time.sleep(POLL_INTERVAL)

        except FileNotFoundError:
            print(f"[waiting] log file missing: {file_path}")

        except Exception as exc:
            print(f"[journaler-error] {exc}")

        time.sleep(2)


def main() -> None:
    ensure_csv_exists()

    print(f"[journaler] watching: {RUN_LOG}")
    print(f"[journaler] output:   {OUTPUT_CSV}")

    for line in follow(RUN_LOG):
        try:
            parse_trade(line)
        except Exception as exc:
            print(f"[parse-error] {exc}")


if __name__ == "__main__":
    main()
