"""Walk-forward split boundaries and trade partitioning.

Default split (inclusive):
    train       : earliest data .. 2023-12-31
    validation  : 2024-01-01 .. 2024-12-31
    out-of-sample (oos) : 2025-01-01 .. latest available

Trades are assigned to a split by EXIT date (same convention as
rolling_metrics), so a trade is scored in the period it closes. Indicators are
computed on the full continuous series, so no warm-up gap leaks across a split
boundary; partitioning by date is leak-free because no signal at bar i uses
data after bar i.
"""
from __future__ import annotations

from datetime import date

TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
OOS_START = "2025-01-01"

SPLITS = ("train", "validation", "oos")


def split_of(iso_date: str) -> str:
    d = iso_date if isinstance(iso_date, str) else iso_date.isoformat()
    if d <= TRAIN_END:
        return "train"
    if d <= VAL_END:
        return "validation"
    return "oos"


def split_trades(trades: list) -> dict:
    """Partition TradeRecords into {split: [trades]} by exit date."""
    buckets = {s: [] for s in SPLITS}
    for t in trades:
        buckets[split_of(t.exit_date)].append(t)
    return buckets


def split_bar_ranges(start: str, end: str) -> dict:
    """Date ranges per split, clipped to [start, end]. For data-coverage notes."""
    end = end or date.today().isoformat()
    return {
        "train": (start, min(TRAIN_END, end)),
        "validation": (max(VAL_START, start), min(VAL_END, end)),
        "oos": (max(OOS_START, start), end),
    }


def _demo():
    from .types import TradeRecord
    mk = lambda ed: TradeRecord("2020-01-01", ed, 1, 1, 0, 0.0, 1)
    trades = [mk("2022-06-01"), mk("2024-03-01"), mk("2025-02-01"), mk("2025-09-01")]
    b = split_trades(trades)
    assert len(b["train"]) == 1 and len(b["validation"]) == 1 and len(b["oos"]) == 2
    assert split_of("2023-12-31") == "train"
    assert split_of("2024-01-01") == "validation"
    assert split_of("2025-01-01") == "oos"
    print("splits._demo OK")


if __name__ == "__main__":
    _demo()
