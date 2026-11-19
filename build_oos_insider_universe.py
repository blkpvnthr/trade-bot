#!/usr/bin/env python3
"""
build_oos_insider_universe.py — TRUE out-of-sample insider-buy universe via the SEC daily index.

Why
    The headline insider edge was built from SEC structured Form 345 data sets through
    2026q1 (forms FILED Jan-Mar 2026). SEC has not yet published the 2026q2/q3 structured
    data sets (HTTP 404), so the clean post-study window is unreachable by the bulk route.
    This crawler reconstructs the SAME event schema the bulk builder produces, but from the
    EDGAR *daily index* for filings dated on/after the OOS start date -- so every event is
    strictly out-of-sample: filed after the quarter the study was built on.

How
    For each weekday in [--start, --end]:
      1. Fetch the daily master index  (edgar/daily-index/<yr>/QTR<q>/master.<YYYYMMDD>.idx).
      2. Keep Form 4 / 4/A rows.
      3. For each, fetch the full submission .txt (ONE request; it embeds the ownership XML),
         extract every <XML> block, parse with the PRODUCTION parser (sec_form4_feed), and
         keep open-market common-stock PURCHASES (code 'P') via the production classifier.
      4. Append [transaction_date, value] per ticker, matching build_pit_insider_universe.py
         exactly so backtest_insider_copier.py runs unchanged.

No-lookahead
    * Events are keyed on the transaction date (same as the in-sample builder); the backtest
      adds the SEC Form 4 filing lag (2 business days) and enters at the next open.
    * The real filing_date is captured in a sidecar (oos_insider_filingdates.json) so a
      lookahead audit can confirm filing_date <= entry_date for every trade.
    * Because the crawl is keyed by filing date, every included event became public in-window.

Politeness
    Disk-cached (immutable historical data), rate-limited (default 8 req/s < SEC's 10 req/s),
    descriptive User-Agent required. Re-runs hit cache and make no network calls.

Usage
    python build_oos_insider_universe.py --start 2026-04-01 --end 2026-05-16
    python build_oos_insider_universe.py --start 2026-04-01 --end 2026-06-30 --rps 8
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sec_form4_feed as sf  # production pure parsers (parse_form4_xml, extract_open_market_purchases)

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "sec_cache_oos"
OUT_EVENTS = PROJECT_ROOT / "data" / "cache" / "oos_insider_events.json"
OUT_FILINGDATES = PROJECT_ROOT / "data" / "cache" / "oos_insider_filingdates.json"

DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index/{yr}/QTR{q}/master.{ymd}.idx"
ARCHIVE_BASE = "https://www.sec.gov/Archives/"
USER_AGENT = "QuantumQuant research contact@quantumquant.trade"
XML_BLOCK = re.compile(r"<XML>\s*(.*?)\s*</XML>", re.DOTALL | re.IGNORECASE)


class RateLimiter:
    """Shared token-bucket; guarantees >= min_interval between request starts across threads."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 0.1)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            sleep = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.min_interval
        if sleep > 0:
            time.sleep(sleep)


def _get(url: str, limiter: RateLimiter, *, retries: int = 3) -> Optional[str]:
    """Rate-limited GET with small backoff. Returns text or None (404/empty treated as None)."""
    last_exc = None
    for attempt in range(retries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            # urllib does not auto-decompress; detect gzip by magic bytes.
            if raw[:2] == b"\x1f\x8b":
                import gzip
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 - network fail-safe
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
    if last_exc is not None:
        print(f"    GET failed ({type(last_exc).__name__}) {url[:90]}")
    return None


def _cache_read(name: str) -> Optional[str]:
    p = CACHE_DIR / name
    if p.exists() and p.stat().st_size > 0:
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def _cache_write(name: str, text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (CACHE_DIR / name).write_text(text, encoding="utf-8")
    except Exception:
        pass


def index_rows_for_day(d: date, limiter: RateLimiter) -> List[Dict[str, str]]:
    """Return Form 4 rows for one day's master index (cached). [] if no index (weekend/holiday)."""
    ymd = d.strftime("%Y%m%d")
    q = (d.month - 1) // 3 + 1
    cache_name = f"idx_{ymd}.idx"
    txt = _cache_read(cache_name)
    if txt is None:
        txt = _get(DAILY_INDEX.format(yr=d.year, q=q, ymd=ymd), limiter)
        if txt is None:
            _cache_write(cache_name, "")  # negative cache: no index this day
            return []
        _cache_write(cache_name, txt)
    rows: List[Dict[str, str]] = []
    for line in txt.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form, filed, filename = (p.strip() for p in parts)
        if form not in ("4", "4/A"):
            continue
        rows.append({"cik": cik, "company": company, "filing_date": filed, "filename": filename})
    return rows


def _accession_from(filename: str) -> str:
    m = re.search(r"(\d{10}-\d{2}-\d{6})", filename)
    return m.group(1) if m else filename.replace("/", "_")


def fetch_and_parse_filing(row: Dict[str, str], limiter: RateLimiter) -> List[dict]:
    """Fetch one Form 4 full submission (.txt, cached) and return open-market-purchase rows."""
    acc = _accession_from(row["filename"])
    cache_name = f"f4_{acc}.txt"
    txt = _cache_read(cache_name)
    if txt is None:
        txt = _get(ARCHIVE_BASE + row["filename"], limiter)
        _cache_write(cache_name, txt or "")
    if not txt:
        return []
    for block in XML_BLOCK.findall(txt):
        parsed = sf.parse_form4_xml(block)
        if not parsed or not parsed.get("transactions"):
            continue
        rows = sf.extract_open_market_purchases(
            parsed, accession=acc, filing_url=ARCHIVE_BASE + row["filename"],
            filing_date=row["filing_date"])
        if rows:
            return rows
    return []


def _weekdays(start: date, end: date) -> List[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _eligible_count(events: Dict[str, list], min_value: float, window_days: int) -> int:
    n = 0
    for _sym, evs in events.items():
        rows = sorted((date.fromisoformat(dd), v) for dd, v in evs)
        for dd, _ in rows:
            if sum(v for d2, v in rows if dd - timedelta(days=window_days) <= d2 <= dd) > min_value:
                n += 1
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="OOS start filing date YYYY-MM-DD (>= 2026-04-01 for clean OOS)")
    ap.add_argument("--end", required=True, help="OOS end filing date YYYY-MM-DD")
    ap.add_argument("--rps", type=float, default=8.0, help="max requests/sec (SEC fair-access < 10)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-value", type=float, default=1_000_000.0)
    ap.add_argument("--window-days", type=int, default=2)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    limiter = RateLimiter(args.rps)
    days = _weekdays(start, end)
    print(f"OOS crawl: filings {start} .. {end}  ({len(days)} weekdays)  rps={args.rps} workers={args.workers}")
    print(f"cache: {CACHE_DIR}")

    events: Dict[str, list] = defaultdict(list)
    # filing_dates[ticker][transaction_date] = latest filing date seen (for lookahead audit)
    filing_dates: Dict[str, Dict[str, str]] = defaultdict(dict)
    total_filings = total_buys = 0

    def _checkpoint() -> None:
        OUT_EVENTS.parent.mkdir(parents=True, exist_ok=True)
        OUT_EVENTS.write_text(json.dumps({k: sorted(v) for k, v in events.items() if v}))
        OUT_FILINGDATES.write_text(json.dumps(filing_dates))

    for d in days:
        ymd = d.strftime("%Y%m%d")
        day_cache = CACHE_DIR / f"day_{ymd}.json"
        n_rows = 0
        if day_cache.exists():
            # Resume-fast: reuse the already-parsed open-market-buy rows for this day.
            try:
                payload = json.loads(day_cache.read_text())
                day_rows = payload.get("buys", [])
                n_rows = int(payload.get("n_filings", 0))
            except Exception:
                day_rows = []
        else:
            rows = index_rows_for_day(d, limiter)
            # dedupe accession within-day (master.idx lists each doc of a filing separately)
            seen_day: set[str] = set()
            uniq = []
            for r in rows:
                acc = _accession_from(r["filename"])
                if acc in seen_day:
                    continue
                seen_day.add(acc)
                uniq.append(r)
            n_rows = len(uniq)
            day_rows = []
            if uniq:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(fetch_and_parse_filing, r, limiter): r for r in uniq}
                    for fut in as_completed(futs):
                        try:
                            purchases = fut.result()
                        except Exception:
                            purchases = []
                        for p in purchases:
                            tic = str(p.get("symbol") or "").upper().strip()
                            tdate = str(p.get("transaction_date") or "")[:10]
                            val = float(p.get("transaction_value") or 0.0)
                            fdate = str(p.get("filing_date") or "")[:10]
                            if not tic or not tdate or val <= 0:
                                continue
                            try:
                                date.fromisoformat(tdate)
                            except Exception:
                                continue
                            day_rows.append({"sym": tic, "td": tdate, "v": round(val, 2), "fd": fdate})
            day_cache.write_text(json.dumps({"n_filings": n_rows, "buys": day_rows}))

        for p in day_rows:
            tic, tdate, val, fdate = p["sym"], p["td"], float(p["v"]), p.get("fd", "")
            events[tic].append([tdate, val])
            prev = filing_dates[tic].get(tdate, "")
            if fdate and (not prev or fdate > prev):
                filing_dates[tic][tdate] = fdate
        total_filings += n_rows
        total_buys += len(day_rows)
        _checkpoint()  # durable after every day so a kill never loses progress
        print(f"  {d} Q{(d.month-1)//3+1}: form4={n_rows:>4} buys_kept={len(day_rows):>3} "
              f"| cum_filings={total_filings} cum_buys={total_buys} tickers={len(events)}", flush=True)

    events = {k: sorted(v) for k, v in events.items() if v}
    OUT_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_EVENTS.write_text(json.dumps(events))
    OUT_FILINGDATES.write_text(json.dumps(filing_dates))
    elig = _eligible_count(events, args.min_value, args.window_days)
    print("")
    print(f"Distinct tickers with open-market buys: {len(events)}")
    print(f"Total open-market buy events: {total_buys}")
    print(f"Tickers with a >${args.min_value:,.0f} {args.window_days}-day cluster (copier-eligible): {elig}")
    print(f"Written: {OUT_EVENTS}")
    print(f"Written: {OUT_FILINGDATES}")


if __name__ == "__main__":
    main()
