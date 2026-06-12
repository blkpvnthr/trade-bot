#!/usr/bin/env python3
"""
build_pit_insider_universe.py — POINT-IN-TIME insider-buy universe from SEC Form 345 data sets.

Why
    The earlier small/mid-cap backtest selected names from TODAY's SEC cache, which
    bakes in survivorship bias (delisted losers are absent). This rebuilds the universe
    the honest way: SEC's quarterly "Insider Transactions Data Sets" (Forms 3/4/5) list
    EVERY Form 4 filed in each quarter — so the set of names with insider buying in,
    say, 2022Q1 is exactly what it was then, including companies later delisted.

What it does
    For each quarter in the range:
      1. Download + cache  {YYYY}q{Q}_form345.zip  (SEC structured-data sets).
      2. Join SUBMISSION (accession → ticker, filing_date) to NONDERIV_TRANS.
      3. Keep open-market common-stock PURCHASES (TRANS_CODE=='P', acquired, +shares/+price),
         using the same security-title rules as the production sec_form4_feed.
      4. Emit per-ticker buy events [(transaction_date, value)] to
         data/cache/pit_insider_events.json.

    The forward-return backtest (backtest_insider_copier.py --source pit) then prices
    whatever tickers it can; tickers with no price history (mostly delisted) are reported
    as dropped so the RESIDUAL survivorship in the return measurement is visible.

Usage
    python build_pit_insider_universe.py --start 2021q1 --end 2024q4
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import ssl
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
ZIP_DIR = PROJECT_ROOT / "data" / "sec_form345"
OUT_PATH = PROJECT_ROOT / "data" / "cache" / "pit_insider_events.json"
DATASET_URL = ("https://www.sec.gov/files/structureddata/data/"
               "insider-transactions-data-sets/{yq}_form345.zip")
UA = {"User-Agent": "QuantumQuant research contact@quantumquant.trade",
      "Accept-Encoding": "gzip, deflate"}

_CTX = ssl.create_default_context()
try:
    _CTX.load_default_certs()
except Exception:
    pass


def _quarters(start: str, end: str) -> List[str]:
    sy, sq = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    out = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        out.append(f"%dq%d" % (y, q))
        q += 1
        if q > 4:
            q = 1; y += 1
    return out


def _download(yq: str) -> bytes | None:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    cache = ZIP_DIR / f"{yq}_form345.zip"
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_bytes()
    url = DATASET_URL.format(yq=yq)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
            data = r.read()
        cache.write_bytes(data)
        time.sleep(0.25)  # SEC politeness
        return data
    except Exception as e:
        print(f"  {yq}: download FAILED — {type(e).__name__}: {str(e)[:80]}")
        return None


def _is_common_purchase(title: str, code: str, ad: str, shares: float, price: float) -> bool:
    if code != "P":
        return False
    if ad not in ("A", ""):
        return False
    if shares <= 0 or price <= 0:
        return False
    t = (title or "").lower()
    if any(k in t for k in ("option", "warrant", "right", "convert", "rsu", "restricted", "unit")):
        return False
    if t and not any(k in t for k in ("common", "ordinary", "class", "share", "stock")):
        return False
    return True


def _f(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


def _parse_trans_date(s: str):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except Exception:
            continue
    return None


def extract_quarter(data: bytes, events: Dict[str, list]) -> Tuple[int, int]:
    """Append (date, value) buy events per ticker. Returns (rows_scanned, buys_kept)."""
    z = zipfile.ZipFile(io.BytesIO(data))
    # accession → ticker
    acc_ticker: Dict[str, str] = {}
    with z.open("SUBMISSION.tsv") as f:
        rd = csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t")
        for row in rd:
            if str(row.get("DOCUMENT_TYPE", "")).strip() not in ("4", "4/A"):
                continue
            tic = str(row.get("ISSUERTRADINGSYMBOL", "") or "").upper().strip()
            if tic and tic not in ("NONE", "N/A", "NA", "-"):
                acc_ticker[row["ACCESSION_NUMBER"]] = tic
    scanned = kept = 0
    with z.open("NONDERIV_TRANS.tsv") as f:
        rd = csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t")
        for row in rd:
            scanned += 1
            tic = acc_ticker.get(row.get("ACCESSION_NUMBER", ""))
            if not tic:
                continue
            shares = _f(row.get("TRANS_SHARES"))
            price = _f(row.get("TRANS_PRICEPERSHARE"))
            if not _is_common_purchase(row.get("SECURITY_TITLE", ""),
                                       str(row.get("TRANS_CODE", "")).upper().strip(),
                                       str(row.get("TRANS_ACQUIRED_DISP_CD", "")).upper().strip(),
                                       shares, price):
                continue
            d = _parse_trans_date(row.get("TRANS_DATE"))
            if d is None:
                continue
            events[tic].append([d.isoformat(), round(shares * price, 2)])
            kept += 1
    return scanned, kept


def _eligible_count(events: Dict[str, list], min_value: float, window_days: int) -> int:
    n = 0
    for sym, evs in events.items():
        rows = sorted((date.fromisoformat(d), v) for d, v in evs)
        for d, _ in rows:
            wv = sum(v for dd, v in rows if d - timedelta(days=window_days) <= dd <= d)
            if wv > min_value:
                n += 1
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021q1")
    ap.add_argument("--end", default="2024q4")
    ap.add_argument("--min-value", type=float, default=1_000_000.0)
    ap.add_argument("--window-days", type=int, default=2)
    args = ap.parse_args()

    quarters = _quarters(args.start, args.end)
    print(f"Quarters: {quarters[0]} … {quarters[-1]} ({len(quarters)} total)")
    events: Dict[str, list] = defaultdict(list)
    total_buys = 0
    for yq in quarters:
        data = _download(yq)
        if not data:
            continue
        scanned, kept = extract_quarter(data, events)
        total_buys += kept
        print(f"  {yq}: scanned {scanned:>7} non-deriv rows, kept {kept:>5} open-market buys")

    events = {k: v for k, v in events.items() if v}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(events))
    elig = _eligible_count(events, args.min_value, args.window_days)
    print(f"\nDistinct tickers with open-market buys: {len(events)}")
    print(f"Total open-market buy events: {total_buys}")
    print(f"Tickers with a >${args.min_value:,.0f} {args.window_days}-day cluster (copier-eligible): {elig}")
    print(f"Written: {OUT_PATH}")


if __name__ == "__main__":
    main()
