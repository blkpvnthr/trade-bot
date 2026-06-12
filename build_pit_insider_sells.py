#!/usr/bin/env python3
"""
build_pit_insider_sells.py — PIT-clean Form-4 SELLS dataset, extracted from the SAME cached SEC
Form-345 zips as the buys (data/sec_form345/). Mirrors build_pit_insider_universe.py but keeps
open-market common-stock SALES (TRANS_CODE=='S', disposed 'D', +shares/+price). Emits per-ticker
[(date, value)] sell events → data/cache/pit_insider_sells.json. No survivorship: every issuer in
the filings is kept; price-coverage is handled downstream.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
ZIP_DIR = PROJECT_ROOT / "data" / "sec_form345"
OUT_PATH = PROJECT_ROOT / "data" / "cache" / "pit_insider_sells.json"


def _f(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


def _parse_date(s: str):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except Exception:
            continue
    return None


def _is_common_sale(title: str, code: str, ad: str, shares: float, price: float) -> bool:
    if code != "S":                       # open-market SALE
        return False
    if ad not in ("D", ""):               # disposed
        return False
    if shares <= 0 or price <= 0:
        return False
    t = (title or "").lower()
    if any(k in t for k in ("option", "warrant", "right", "convert", "rsu", "restricted", "unit")):
        return False
    if t and not any(k in t for k in ("common", "ordinary", "class", "share", "stock")):
        return False
    return True


def extract_quarter(data: bytes, events: Dict[str, list]) -> Tuple[int, int]:
    z = zipfile.ZipFile(io.BytesIO(data))
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
            shares = _f(row.get("TRANS_SHARES")); price = _f(row.get("TRANS_PRICEPERSHARE"))
            if not _is_common_sale(row.get("SECURITY_TITLE", ""),
                                   str(row.get("TRANS_CODE", "")).upper().strip(),
                                   str(row.get("TRANS_ACQUIRED_DISP_CD", "")).upper().strip(),
                                   shares, price):
                continue
            d = _parse_date(row.get("TRANS_DATE"))
            if d is None:
                continue
            events[tic].append([d.isoformat(), round(shares * price, 2)])
            kept += 1
    return scanned, kept


def main():
    zips = sorted(ZIP_DIR.glob("*_form345.zip"))
    events: Dict[str, list] = defaultdict(list)
    tot_s = tot_k = 0
    for zp in zips:
        s, k = extract_quarter(zp.read_bytes(), events)
        tot_s += s; tot_k += k
        print(f"  {zp.name}: scanned {s:,} → sells kept {k:,}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(events))
    print(f"\n{len(zips)} quarters | {tot_s:,} rows scanned | {tot_k:,} sell events | "
          f"{len(events):,} tickers → {OUT_PATH}")


if __name__ == "__main__":
    main()
