#!/usr/bin/env python3
"""
trump_insider_tracker.py — copy-trade the Trump family's *disclosed* trades.
============================================================================

Scope (set by what is actually disclosed on SEC EDGAR — verified 2026-06):
  * Donald Trump Jr   CIK 0002016181  — active Form 4 filer (DJT, PEW, PSQH …)
  * Donald J Trump    CIK 0000947033  — sparse (DJT founder/trust)
  * Ivanka Trump      CIK 0001406847  — dormant (last 2009)
  * Eric Trump / Jared Kushner        — file NOTHING (private cos; not insiders,
                                        not in Congress) → nothing to copy.

Reality: their recent Form 4s are board-comp GRANTS (code A) and GIFTS (G), not
open-market BUYS (code P). So this module does two things:

  1. WATCHLIST  — maintain a "Trump-affiliated" basket from every issuer these
     CIKs file on (DJT, GrabAGun/PEW, PublicSquare/PSQH, + any new one). A NEW
     ticker = a fresh Trump board seat/grant, a known catalyst for these names.
  2. COPY       — emit a BUY signal on the rare genuine open-market PURCHASE
     (code P) by any tracked CIK — the literal "copy their trade".

Network-polite (SEC User-Agent + rate limit), fail-safe (never raises), and
reuses the Form 4 parser from sec_form4_feed. Pure data layer — order placement
lives in run.py (run_trump_insider_tracker), gated + risk-checked like the rest.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import sec_form4_feed as _s4

# CIK (zero-padded to 10) -> display name. Eric Trump / Jared Kushner intentionally
# absent — they have no SEC Form 4 filings to copy.
TRACKED_CIKS: Dict[str, str] = {
    "0002016181": "Donald Trump Jr",
    "0000947033": "Donald J Trump",
    "0001406847": "Ivanka Trump",
}

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "QuantumQuant Research contact@quantumquant.trade")
RATE_LIMIT_SEC = float(os.getenv("SEC_FORM4_RATE_LIMIT_SECONDS", "0.25"))
MAX_FILINGS_PER_CIK = int(os.getenv("TRUMP_TRACKER_MAX_FILINGS_PER_CIK", "12"))

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/index.json"
_FILE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{name}"

_CTX = ssl.create_default_context()
_LAST_REQ = [0.0]


def _get(url: str) -> str | None:
    """Polite GET with SEC User-Agent + rate limit. Returns text or None (never raises)."""
    try:
        wait = RATE_LIMIT_SEC - (time.time() - _LAST_REQ[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT,
                                                   "Accept-Encoding": "gzip, deflate"})
        with urllib.request.urlopen(req, timeout=20, context=_CTX) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", "replace")
    except Exception:
        return None
    finally:
        _LAST_REQ[0] = time.time()


def _recent_form4_accessions(cik10: str, lookback_days: int) -> List[Dict[str, str]]:
    """[{accession, filing_date}] of Form 4 filings for this CIK within the lookback window."""
    txt = _get(_SUBMISSIONS.format(cik=cik10))
    if not txt:
        return []
    try:
        recent = json.loads(txt).get("filings", {}).get("recent", {})
    except Exception:
        return []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    out: List[Dict[str, str]] = []
    for i in range(min(len(forms), len(dates), len(accs))):
        if forms[i] in ("4", "4/A") and dates[i] >= cutoff:
            out.append({"accession": accs[i], "filing_date": dates[i]})
        if len(out) >= MAX_FILINGS_PER_CIK:
            break
    return out


def _fetch_parsed_form4(cik10: str, accession: str) -> dict | None:
    """Fetch + parse a single Form 4 (returns parse_form4_xml output, or None)."""
    cik_int = str(int(cik10))
    acc = accession.replace("-", "")
    idx = _get(_INDEX.format(cik_int=cik_int, acc=acc))
    if not idx:
        return None
    try:
        items = json.loads(idx).get("directory", {}).get("item", [])
    except Exception:
        return None
    xml_name = next((it.get("name") for it in items
                     if str(it.get("name", "")).lower().endswith(".xml")), None)
    if not xml_name:
        return None
    xml = _get(_FILE.format(cik_int=cik_int, acc=acc, name=xml_name))
    if not xml:
        return None
    try:
        return _s4.parse_form4_xml(xml)
    except Exception:
        return None


def discover(lookback_days: int = 30, seen_accessions: set[str] | None = None) -> dict:
    """
    Pull the tracked CIKs' recent Form 4s.

    Returns {
      "basket":           [tickers across all tracked filings in the window],
      "open_market_buys": [{symbol, owner, owner_cik, shares, price, value, date, accession}],
      "filings":          [{owner, owner_cik, symbol, date, accession, codes, is_new}],
      "new_tickers":      [tickers seen for the first time vs the prior basket — caller diffs],
      "errors":           [str],
    }
    """
    seen = seen_accessions or set()
    basket: set[str] = set()
    buys: List[dict] = []
    filings: List[dict] = []
    errors: List[str] = []

    for cik10, name in TRACKED_CIKS.items():
        try:
            for f in _recent_form4_accessions(cik10, lookback_days):
                acc = f["accession"]
                parsed = _fetch_parsed_form4(cik10, acc)
                if not parsed:
                    continue
                issuer = parsed.get("issuer", {}) or {}
                ticker = str(issuer.get("ticker") or "").upper().strip()
                txns = parsed.get("transactions", []) or []
                codes = sorted({str(t.get("code", "")).upper() for t in txns if t.get("code")})
                if ticker:
                    basket.add(ticker)
                filings.append({
                    "owner": name, "owner_cik": cik10, "symbol": ticker,
                    "date": f["filing_date"], "accession": acc, "codes": codes,
                    "is_new": acc not in seen,
                })
                for t in txns:
                    if _s4.is_open_market_purchase(t) and ticker:
                        buys.append({
                            "symbol": ticker, "owner": name, "owner_cik": cik10,
                            "shares": float(t.get("shares") or 0.0),
                            "price": float(t.get("price") or 0.0),
                            "value": _s4.transaction_value(t),
                            "date": t.get("transaction_date") or f["filing_date"],
                            "accession": acc,
                        })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {str(exc)[:120]}")

    return {
        "basket": sorted(basket),
        "open_market_buys": sorted(buys, key=lambda b: b.get("date", ""), reverse=True),
        "filings": sorted(filings, key=lambda x: x.get("date", ""), reverse=True),
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ── operator CLI: print the current basket + recent filings + any P-buys ────────
if __name__ == "__main__":
    import sys
    lb = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    res = discover(lookback_days=lb)
    print(f"Trump insider tracker — last {lb}d (tracked: {', '.join(TRACKED_CIKS.values())})")
    print(f"\nBASKET ({len(res['basket'])}): {', '.join(res['basket']) or '(none)'}")
    print(f"\nFILINGS ({len(res['filings'])}):")
    for f in res["filings"]:
        print(f"  {f['date']}  {f['owner']:18s} {f['symbol'] or '?':6s} codes={f['codes']}")
    buys = res["open_market_buys"]
    print(f"\nOPEN-MARKET BUYS to copy ({len(buys)}):")
    for b in buys:
        print(f"  {b['date']}  {b['owner']:18s} {b['symbol']} {b['shares']:g}@${b['price']:g} = ${b['value']:,.0f}")
    if not buys:
        print("  (none — these insiders receive grants/gifts, not open-market buys)")
    if res["errors"]:
        print("\nERRORS:", res["errors"])
