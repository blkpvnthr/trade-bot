#!/usr/bin/env python3
"""
build_pit_13d.py — PIT-clean NEW Schedule 13D (activist >5% stake) event dataset from EDGAR.

Activist 13D filings are a documented information edge (Brav-Jiang-Partnoy-Thomas: ~6-7% abnormal
return around the filing) and structurally independent of insider buys. New SC 13D only (not /A
amendments, not passive 13G). Filing date = PIT-correct (public when filed).

Pipeline (aggressively cached → re-runs are cheap):
  1. master.idx per quarter (2021Q1-2024Q4) → rows where Form Type == "SC 13D".
  2. subject ticker: master.idx CIK → ticker (company_tickers.json) when it's the subject (~38%);
     else fetch the filing SEC-header (Range 0-6KB) → SUBJECT-COMPANY CIK → ticker.
  3. emit {ticker: [filing_date, ...]} → data/cache/pit_13d_events.json.
SEC-polite: 8 req/s, UA header, all responses cached to data/cache/edgar/.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE = PROJECT_ROOT / "data" / "cache" / "edgar"
OUT = PROJECT_ROOT / "data" / "cache" / "pit_13d_events.json"
UA = {"User-Agent": "QuantumQuant research contact@quantumquant.trade"}
_CTX = ssl.create_default_context()
try:
    _CTX.load_default_certs()
except Exception:
    pass
_last = 0.0


def _get(url: str, rng: Optional[int] = None, cache_key: Optional[str] = None) -> Optional[bytes]:
    global _last
    if cache_key:
        cf = CACHE / cache_key
        if cf.exists():
            return cf.read_bytes() or None
    wait = 0.13 - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes=0-{rng}"
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30, context=_CTX).read()
        _last = time.time()
    except Exception:
        _last = time.time()
        return None
    if cache_key:
        (CACHE / cache_key).parent.mkdir(parents=True, exist_ok=True)
        try:
            (CACHE / cache_key).write_bytes(data)
        except Exception:
            pass
    return data


def _quarters():
    return [f"{y}/QTR{q}" for y in range(2021, 2025) for q in range(1, 5)]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ct = json.loads(_get("https://www.sec.gov/files/company_tickers.json",
                         cache_key="company_tickers.json").decode())
    cik2tic = {int(v["cik_str"]): v["ticker"].upper() for v in ct.values()}
    print(f"CIK→ticker map: {len(cik2tic):,}")

    filings = []                      # (subject_idx_cik, name, date, filename)
    for yq in _quarters():
        raw = _get(f"https://www.sec.gov/Archives/edgar/full-index/{yq}/master.idx",
                   cache_key=f"master_{yq.replace('/', '_')}.idx")
        if not raw:
            print(f"  {yq}: master.idx FAILED"); continue
        rows = [l.split("|") for l in raw.decode("latin-1", "replace").splitlines() if "|SC 13D|" in l]
        for r in rows:
            if len(r) >= 5:
                filings.append((int(r[0]), r[1], r[3], r[4]))
        print(f"  {yq}: {len(rows)} new SC 13D")

    # dedup by accession (filename)
    seen, uniq = set(), []
    for f in filings:
        if f[3] not in seen:
            seen.add(f[3]); uniq.append(f)
    print(f"\n{len(uniq):,} unique SC 13D filings. Resolving subject tickers...")

    events: Dict[str, list] = defaultdict(list)
    direct = fetched = nomap = 0
    for k, (cik, name, dfiled, fn) in enumerate(uniq):
        if k % 500 == 0:
            print(f"  ...{k}/{len(uniq)} ({len(events)} tickers so far)", flush=True)
            if events:                       # incremental save → robust to interruption
                OUT.write_text(json.dumps({t: sorted(set(d)) for t, d in events.items()}))
        tic = cik2tic.get(cik)            # master.idx CIK is the subject (~38%)
        if not tic:
            acc = fn.rsplit("/", 1)[-1].replace(".txt", "")
            hdr = _get(f"https://www.sec.gov/Archives/{fn}", rng=6000, cache_key=f"hdr/{acc}.txt")
            fetched += 1
            if hdr:
                blk = hdr.decode("latin-1", "replace")
                sub = blk.split("SUBJECT COMPANY")[1] if "SUBJECT COMPANY" in blk else ""
                m = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", sub)
                if m:
                    tic = cik2tic.get(int(m.group(1)))
        else:
            direct += 1
        if tic:
            events[tic].append(dfiled[:10])
        else:
            nomap += 1
    OUT.write_text(json.dumps({t: sorted(set(d)) for t, d in events.items()}))
    n_ev = sum(len(v) for v in events.values())
    print(f"\nresolved: {direct} direct + {fetched} header-fetched | unmapped {nomap}")
    print(f"{len(events):,} subject tickers, {n_ev:,} 13D events → {OUT}")


if __name__ == "__main__":
    main()
