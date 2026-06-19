"""TradingView Strategy Tester "List of Trades" CSV importer.

This is the EVALUATION SOURCE OF TRUTH. When a strategy has imported TradingView
trades, the system evaluates those exact entries/exits instead of the Python
adapter approximation (adapters become fallback-only).

TradingView exports two rows per trade (an Entry row and an Exit row sharing a
Trade #). Column names vary across TradingView versions / locales, so headers
are matched fuzzily. Numbers may carry currency symbols, thousands separators,
percent signs, or parenthesised negatives.

A CSV maps to one (strategy_id, symbol, timeframe). The version_hash is resolved
from the registry so an export is pinned to the exact PineScript version that
produced it -- a stale export (old hash) is ignored, not silently trusted.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from .registry import DATA_DIR
from .signals import CAPITAL_PER_TRADE
from .types import TradeRecord

EXPORTS_DIR = DATA_DIR / "tradingview_exports"


def _num(s):
    """Parse a TradingView numeric cell. Returns float or None."""
    s = str(s).strip()
    if s in ("", "-", "—", "n/a", "N/A", "—"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = s.replace("−", "-").replace(",", "").replace("%", "").replace("$", "")
    s = re.sub(r"[A-Za-z]", "", s).strip()      # drop currency codes (USD, etc.)
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _col(headers: dict, must, must_not=()):
    for orig, low in headers.items():
        if all(m in low for m in must) and not any(x in low for x in must_not):
            return orig
    return None


def _parse_dt(s):
    try:
        return pd.to_datetime(str(s).strip(), utc=True)
    except (ValueError, TypeError):
        return pd.NaT


def parse_tv_csv(path) -> list:
    """Parse a TradingView List-of-Trades CSV into TradeRecords (closed trades).

    Open trades (an entry with no matching exit) are skipped.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []

    headers = {h: (h or "").strip().lower() for h in rows[0].keys()}
    c_type = _col(headers, ["type"])
    c_num = _col(headers, ["trade"]) or _col(headers, ["#"])
    c_dt = _col(headers, ["date"]) or _col(headers, ["time"])
    c_price = _col(headers, ["price"])
    c_pnl = (_col(headers, ["profit"], ["%", "cum"])
             or _col(headers, ["p&l"], ["%", "cum"])
             or _col(headers, ["pnl"], ["%", "cum"]))
    c_pnl_pct = (_col(headers, ["profit", "%"], ["cum"])
                 or _col(headers, ["p&l", "%"], ["cum"])
                 or _col(headers, ["pnl", "%"], ["cum"]))
    if c_type is None or c_dt is None or c_price is None:
        raise ValueError(f"{path.name}: not a recognisable TradingView "
                         "List-of-Trades CSV (missing Type/Date/Price columns)")

    # Group the entry+exit rows of each trade.
    groups = {}
    for i, r in enumerate(rows):
        key = (r.get(c_num) or "").strip() if c_num else None
        groups.setdefault(key if key else f"_seq{i // 2}", []).append(r)

    trades = []
    for grp in groups.values():
        entry = next((r for r in grp if "entry" in (r.get(c_type) or "").lower()), None)
        exit_ = next((r for r in grp if "exit" in (r.get(c_type) or "").lower()), None)
        if entry is None or exit_ is None:
            continue
        e_dt, x_dt = _parse_dt(entry[c_dt]), _parse_dt(exit_[c_dt])
        e_px, x_px = _num(entry[c_price]), _num(exit_[c_price])
        if pd.isna(e_dt) or pd.isna(x_dt) or e_px is None or x_px is None:
            continue
        if x_dt < e_dt:                              # rows could be reversed
            e_dt, x_dt, e_px, x_px = x_dt, e_dt, x_px, e_px

        is_short = "short" in (entry.get(c_type) or "").lower()
        pnl_pct = _num(exit_.get(c_pnl_pct)) if c_pnl_pct else None
        if pnl_pct is not None:
            ret = pnl_pct / 100.0
        else:
            ret = (x_px - e_px) / e_px
            if is_short:
                ret = -ret
        pnl = _num(exit_.get(c_pnl)) if c_pnl else None
        if pnl is None:
            pnl = ret * CAPITAL_PER_TRADE

        trades.append(TradeRecord(
            entry_date=e_dt.date().isoformat(),
            exit_date=x_dt.date().isoformat(),
            entry_price=round(float(e_px), 4),
            exit_price=round(float(x_px), 4),
            pnl=round(float(pnl), 2),
            return_pct=float(ret),
            hold_days=round((x_dt - e_dt).total_seconds() / 86400.0, 3),
        ))
    trades.sort(key=lambda t: (t.exit_date, t.entry_date))
    return trades


def parse_filename(name: str):
    """Convention: '{strategy_id}__{symbol}__{timeframe}.csv'. Returns tuple|None."""
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) == 3 and all(parts):
        return parts[0], parts[1], parts[2]
    return None


def _demo():
    import io
    sample = (
        "Trade #,Type,Date/Time,Signal,Price USD,Net P&L USD,Net P&L %\n"
        "1,Entry Long,2023-02-01,buy,100,,\n"
        "1,Exit Long,2023-02-10,sell,110,1000,10\n"
        "2,Entry Long,2023-03-01,buy,110,,\n"
        "2,Exit Long,2023-03-05,sell,104.5,-500,-5\n"
    )
    p = DATA_DIR / "_tv_demo.csv"
    p.write_text(sample)
    try:
        trades = parse_tv_csv(p)
    finally:
        p.unlink()
    assert len(trades) == 2, len(trades)
    assert trades[0].entry_date == "2023-02-01" and trades[0].pnl == 1000.0
    assert abs(trades[0].return_pct - 0.10) < 1e-9
    assert trades[1].pnl == -500.0
    assert parse_filename("sample_strategy__NVDA__1Day.csv") == \
        ("sample_strategy", "NVDA", "1Day")
    print("tv_import._demo OK", len(trades), "trades")


if __name__ == "__main__":
    _demo()
