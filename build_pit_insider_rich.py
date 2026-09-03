#!/usr/bin/env python3
"""
build_pit_insider_rich.py — RICH PIT Form-4 BUYS with insider identity + role, from the cached
Form-345 zips. The aggregated pit_insider_events.json dropped owner identity/role; this keeps it so
we can test edge AMPLIFIERS (cluster breadth = # distinct insiders, C-suite role, repeat buyer).

Per open-market common-stock PURCHASE (TRANS_CODE='P', acquired) emits:
  [date, value, owner_cik, is_officer, is_director, is_10pct, title]
→ data/cache/pit_insider_rich.json  {ticker: [ [date,value,cik,off,dir,ten,title], ... ]}
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
ZIP_DIR = PROJECT_ROOT / "data" / "sec_form345"
OUT = PROJECT_ROOT / "data" / "cache" / "pit_insider_rich.json"


def _f(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


def _date(s):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except Exception:
            continue
    return None


def _is_buy(title, code, ad, sh, px):
    if code != "P" or ad not in ("A", "") or sh <= 0 or px <= 0:
        return False
    t = (title or "").lower()
    if any(k in t for k in ("option", "warrant", "right", "convert", "rsu", "restricted", "unit")):
        return False
    return (not t) or any(k in t for k in ("common", "ordinary", "class", "share", "stock"))


def extract(data: bytes, events: Dict[str, list]) -> int:
    z = zipfile.ZipFile(io.BytesIO(data))
    acc_tic, acc_owner = {}, {}
    with z.open("SUBMISSION.tsv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t"):
            if str(r.get("DOCUMENT_TYPE", "")).strip() in ("4", "4/A"):
                t = str(r.get("ISSUERTRADINGSYMBOL", "") or "").upper().strip()
                if t and t not in ("NONE", "N/A", "NA", "-"):
                    acc_tic[r["ACCESSION_NUMBER"]] = t
    with z.open("REPORTINGOWNER.tsv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t"):
            acc = r.get("ACCESSION_NUMBER")
            if acc and acc not in acc_owner:        # first owner per accession
                rel = str(r.get("RPTOWNER_RELATIONSHIP", "") or "").lower()
                acc_owner[acc] = (str(r.get("RPTOWNERCIK", "") or ""),
                                  int("officer" in rel), int("director" in rel),
                                  int("10%" in rel or "ten percent" in rel),
                                  str(r.get("RPTOWNER_TITLE", "") or "")[:40])
    kept = 0
    with z.open("NONDERIV_TRANS.tsv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, "latin-1"), delimiter="\t"):
            acc = r.get("ACCESSION_NUMBER", "")
            tic = acc_tic.get(acc)
            if not tic:
                continue
            sh, px = _f(r.get("TRANS_SHARES")), _f(r.get("TRANS_PRICEPERSHARE"))
            if not _is_buy(r.get("SECURITY_TITLE", ""), str(r.get("TRANS_CODE", "")).upper().strip(),
                           str(r.get("TRANS_ACQUIRED_DISP_CD", "")).upper().strip(), sh, px):
                continue
            d = _date(r.get("TRANS_DATE"))
            if d is None:
                continue
            o = acc_owner.get(acc, ("", 0, 0, 0, ""))
            events[tic].append([d.isoformat(), round(sh * px, 2), o[0], o[1], o[2], o[3], o[4]])
            kept += 1
    return kept


def main():
    events: Dict[str, list] = defaultdict(list)
    tot = 0
    for zp in sorted(ZIP_DIR.glob("*_form345.zip")):
        k = extract(zp.read_bytes(), events)
        tot += k
        print(f"  {zp.name}: buys kept {k:,}")
    OUT.write_text(json.dumps(events))
    print(f"\n{tot:,} rich buy events, {len(events):,} tickers → {OUT}")


if __name__ == "__main__":
    main()
