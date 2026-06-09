#!/usr/bin/env python3
"""
sec_form4_feed.py — SEC EDGAR Form 4 insider-buy DISCOVERY system.

Purpose
    SEC Form 4 open-market insider purchases are a CANDIDATE-DISCOVERY mechanism,
    NOT a buy signal. This module discovers tickers with material open-market insider
    buying and hands them to the Buffett Accumulation Layer's review universe. It can
    never generate an order, and it never bypasses the BAL score, valuation, conviction,
    Composite DCA, or RiskManager — those gates run downstream as usual.

Pipeline
    SEC EDGAR Form 4 (daily index + ownership XML)
      → open-market-purchase extraction (transactionCode == "P")
      → aggregate by symbol
      → discovery-quality insider_buy_score
      → daily_system_outputs/sec_form4_insider_buys.json
      → daily_system_outputs/insider_buy_candidates.json   (consumed by the Buffett screener / layer)

Design
    * Pure functions (parse_form4_xml, extract/aggregate/score, build_candidates) are
      network-free and unit-tested offline.
    * The network layer (SECForm4Feed) adds a custom User-Agent, rate limiting, disk
      caching, bounded work, and total fail-safety — any failure logs an error, writes
      a diagnostic output, and returns empty so the rest of run.py is unaffected.

SEC politeness: data.sec.gov / www.sec.gov require a descriptive User-Agent and a
modest request rate (≈10 req/s). Configure SEC_USER_AGENT with a real contact.
"""
from __future__ import annotations

import json
import logging
import os
import re
import ssl
import time
import hashlib
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ET_TZ = ZoneInfo("America/New_York")

# SSL context: use system CA bundle, fall back to unverified if cert store is incomplete.
def _make_ssl_ctx() -> ssl.SSLContext:
    try:
        ctx = ssl.create_default_context()
        ctx.load_default_certs()
        return ctx
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

_SSL_CTX = _make_ssl_ctx()
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET_TZ).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)

SEC_FORM4_INSIDER_BUYS_PATH = DAILY_OUTPUTS / "sec_form4_insider_buys.json"
INSIDER_BUY_CANDIDATES_PATH = DAILY_OUTPUTS / "insider_buy_candidates.json"

SOURCE_TAG = "SEC_EDGAR_FORM4"
DISCOVERY_SOURCE = "SEC_FORM4_DISCOVERY"

# Transaction codes (SEC Form 4). Only "P" (open-market purchase) discovers candidates.
OPEN_MARKET_PURCHASE = "P"
NON_DISCOVERY_CODES = {
    "S": "sale", "A": "grant_award", "M": "option_exercise", "F": "tax_withholding",
    "G": "gift", "C": "conversion", "X": "option_exercise", "D": "disposition",
    "W": "acquisition_disposition_by_will",
}


# ── configuration ───────────────────────────────────────────────────────────────
@dataclass
class SECForm4Config:
    enabled: bool = os.getenv("ENABLE_SEC_FORM4_FEED", "true").lower() in {"1", "true"}
    user_agent: str = os.getenv("SEC_USER_AGENT", "QuantumQuant/1.0 contact@example.com")
    lookback_days: int = int(os.getenv("SEC_FORM4_LOOKBACK_DAYS", "5"))  # reduced 30→5 to limit network requests per cycle
    min_transaction_value: float = float(os.getenv("SEC_FORM4_MIN_TRANSACTION_VALUE", "1000000"))  # lowered 100M→1M (2026-06-03); $100M was filtering out all candidates
    cache_dir: str = os.getenv("SEC_FORM4_CACHE_DIR", "data/sec_cache")
    rate_limit_seconds: float = float(os.getenv("SEC_FORM4_RATE_LIMIT_SECONDS", "0.2"))
    # Bounds so a single run can never hammer SEC or hang the cycle.
    max_filings: int = int(os.getenv("SEC_FORM4_MAX_FILINGS", "200"))  # reduced 4000→200; 4000×0.2s=800s blocking
    max_index_days: int = int(os.getenv("SEC_FORM4_MAX_INDEX_DAYS", "5"))   # reduced 31→5
    request_timeout: int = int(os.getenv("SEC_FORM4_REQUEST_TIMEOUT", "15"))
    index_cache_ttl_seconds: int = int(os.getenv("SEC_FORM4_INDEX_CACHE_TTL", "21600"))  # 6h
    # How long (seconds) before the output file is considered stale and a new network run is needed.
    # Set to 4 hours: insider data doesn't need refreshing every cycle (it's a discovery input, not a live signal).
    output_cache_ttl_seconds: int = int(os.getenv("SEC_FORM4_OUTPUT_CACHE_TTL", "14400"))  # 4h


# ── XML parsing (pure, network-free) ─────────────────────────────────────────────
def _v(elem, path: str, default: str = "") -> str:
    """Text of <path>/<value> (or <path>) under elem; '' when absent."""
    if elem is None:
        return default
    node = elem.find(path)
    if node is None:
        return default
    val = node.find("value")
    text = (val.text if val is not None else node.text)
    return (text or "").strip() if text is not None else default


def _f(text: str) -> float:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _classify_role(title: str, is_director: bool, is_officer: bool, is_ten_pct: bool) -> str:
    t = (title or "").lower()
    if any(k in t for k in ("chief executive", "ceo")):
        return "CEO"
    if any(k in t for k in ("chief financial", "cfo")):
        return "CFO"
    if "chairman" in t or "chair" in t:
        return "Chairman"
    if any(k in t for k in ("president", "chief operating", "coo", "chief")):
        return "Officer"
    if is_director:
        return "Director"
    if is_officer:
        return "Officer"
    if is_ten_pct:
        return "10% Owner"
    return title.strip() or ("Insider" if (is_director or is_officer or is_ten_pct) else "")


def parse_form4_xml(xml_text: str) -> dict:
    """
    Parse a Form 4 ownership XML document into a structured dict. Network-free and
    fail-safe: returns {} on any parse error. Handles documents with or without an
    XML namespace.

    Returns:
        {issuer:{cik,name,ticker}, owner:{name,title,role,is_director,is_officer,is_ten_percent},
         transactions:[{security_title, transaction_date, code, shares, price, value,
                        acquired_disposed, direct_indirect, table}],
         non_derivative_count, derivative_count}
    """
    if not xml_text or not xml_text.strip():
        return {}
    try:
        # Strip a default namespace so plain tag paths work.
        cleaned = re.sub(r'\sxmlns="[^"]+"', "", xml_text, count=1)
        root = ET.fromstring(cleaned)
    except Exception as exc:
        logger.debug("parse_form4_xml failed: %s", exc)
        return {}

    issuer_el = root.find("issuer")
    owner_el = root.find("reportingOwner")
    rel_el = owner_el.find("reportingOwnerRelationship") if owner_el is not None else None
    is_director = _v(rel_el, "isDirector") in ("1", "true", "Y")
    is_officer = _v(rel_el, "isOfficer") in ("1", "true", "Y")
    is_ten_pct = _v(rel_el, "isTenPercentOwner") in ("1", "true", "Y")
    title = _v(rel_el, "officerTitle")

    owner_name = ""
    if owner_el is not None:
        oid = owner_el.find("reportingOwnerId")
        owner_name = _v(oid, "rptOwnerName") if oid is not None else ""

    issuer = {
        "cik": _v(issuer_el, "issuerCik"),
        "name": _v(issuer_el, "issuerName"),
        "ticker": _v(issuer_el, "issuerTradingSymbol").upper(),
    }
    owner = {
        "name": owner_name, "title": title,
        "role": _classify_role(title, is_director, is_officer, is_ten_pct),
        "is_director": is_director, "is_officer": is_officer, "is_ten_percent": is_ten_pct,
    }

    transactions: list[dict] = []

    def _read_table(table_tag: str, txn_tag: str) -> int:
        count = 0
        table = root.find(table_tag)
        if table is None:
            return 0
        for txn in table.findall(txn_tag):
            count += 1
            coding = txn.find("transactionCoding")
            code = _v(coding, "transactionCode") if coding is not None else ""
            amounts = txn.find("transactionAmounts")
            shares = _f(_v(amounts, "transactionShares")) if amounts is not None else 0.0
            price = _f(_v(amounts, "transactionPricePerShare")) if amounts is not None else 0.0
            ad = _v(amounts, "transactionAcquiredDisposedCode") if amounts is not None else ""
            nature = txn.find("ownershipNature")
            di = _v(nature, "directOrIndirectOwnership") if nature is not None else ""
            transactions.append({
                "security_title": _v(txn, "securityTitle"),
                "transaction_date": _v(txn, "transactionDate"),
                "code": code.upper().strip(),
                "shares": shares, "price": price, "value": round(shares * price, 2),
                "acquired_disposed": ad.upper().strip(),
                "direct_indirect": di.upper().strip(),
                "table": table_tag,
            })
        return count

    non_deriv = _read_table("nonDerivativeTable", "nonDerivativeTransaction")
    deriv = _read_table("derivativeTable", "derivativeTransaction")

    return {
        "issuer": issuer, "owner": owner, "transactions": transactions,
        "non_derivative_count": non_deriv, "derivative_count": deriv,
    }


def transaction_value(txn: dict) -> float:
    """shares × price (uses precomputed value when present)."""
    v = float(txn.get("value") or 0.0)
    if v:
        return v
    return float(txn.get("shares") or 0.0) * float(txn.get("price") or 0.0)


def is_open_market_purchase(txn: dict) -> bool:
    """
    True only for a valid open-market common-stock purchase:
      code == "P", non-derivative table, acquired (A or blank), positive shares & price,
      common-stock security.
    """
    if str(txn.get("code", "")).upper() != OPEN_MARKET_PURCHASE:
        return False
    if txn.get("table") != "nonDerivativeTable":
        return False
    if str(txn.get("acquired_disposed", "")).upper() not in ("A", ""):
        return False
    if float(txn.get("shares") or 0.0) <= 0 or float(txn.get("price") or 0.0) <= 0:
        return False
    title = str(txn.get("security_title", "")).lower()
    if title and not any(k in title for k in ("common", "ordinary", "class", "share", "stock")):
        # Reject obvious derivatives / units that slipped into the non-derivative table.
        if any(k in title for k in ("option", "warrant", "right", "convert", "rsu", "restricted")):
            return False
    return True


def extract_open_market_purchases(parsed: dict, *, accession: str = "", filing_url: str = "",
                                  filing_date: str = "") -> list[dict]:
    """Flatten a parsed Form 4 into open-market-purchase rows (one per qualifying txn)."""
    if not parsed:
        return []
    issuer = parsed.get("issuer", {})
    owner = parsed.get("owner", {})
    ticker = str(issuer.get("ticker") or "").upper()
    rows: list[dict] = []
    for txn in parsed.get("transactions", []):
        if not is_open_market_purchase(txn):
            continue
        if not ticker:
            continue
        rows.append({
            "symbol": ticker,
            "company": issuer.get("name", ""),
            "issuer_cik": issuer.get("cik", ""),
            "owner_name": owner.get("name", ""),
            "owner_title": owner.get("title", ""),
            "owner_role": owner.get("role", ""),
            "is_director": owner.get("is_director", False),
            "is_officer": owner.get("is_officer", False),
            "transaction_date": txn.get("transaction_date", ""),
            "filing_date": filing_date,
            "transaction_code": txn.get("code", ""),
            "shares": txn.get("shares", 0.0),
            "price_per_share": txn.get("price", 0.0),
            "transaction_value": transaction_value(txn),
            "direct_indirect": txn.get("direct_indirect", ""),
            "accession_number": accession,
            "sec_filing_url": filing_url,
            "derivative_heavy": parsed.get("derivative_count", 0) > parsed.get("non_derivative_count", 0),
        })
    return rows


# ── aggregation + scoring (pure) ─────────────────────────────────────────────────
def _parse_date(s: str):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def aggregate_by_symbol(purchases: list[dict]) -> dict[str, dict]:
    """Aggregate open-market-purchase rows into per-symbol candidate records."""
    agg: dict[str, dict] = {}
    for p in purchases:
        sym = p["symbol"]
        a = agg.setdefault(sym, {
            "symbol": sym, "company": p.get("company", ""),
            "total_open_market_purchase_value": 0.0, "transaction_count": 0,
            "insiders": {}, "filings": set(), "sec_filing_urls": set(),
            "largest_purchase_value": 0.0, "latest_transaction_date": "",
            "latest_filing_date": "", "roles": set(), "any_direct": False,
            "derivative_heavy": False,
        })
        val = float(p.get("transaction_value") or 0.0)
        a["total_open_market_purchase_value"] += val
        a["transaction_count"] += 1
        a["largest_purchase_value"] = max(a["largest_purchase_value"], val)
        owner = p.get("owner_name", "") or "unknown"
        a["insiders"].setdefault(owner, {"name": owner, "role": p.get("owner_role", ""),
                                         "title": p.get("owner_title", ""), "buys": 0, "value": 0.0})
        a["insiders"][owner]["buys"] += 1
        a["insiders"][owner]["value"] += val
        if p.get("owner_role"):
            a["roles"].add(p["owner_role"])
        if p.get("accession_number"):
            a["filings"].add(p["accession_number"])
        if p.get("sec_filing_url"):
            a["sec_filing_urls"].add(p["sec_filing_url"])
        if str(p.get("direct_indirect", "")).upper() == "D":
            a["any_direct"] = True
        if p.get("derivative_heavy"):
            a["derivative_heavy"] = True
        td = _parse_date(p.get("transaction_date", ""))
        if td and (not a["latest_transaction_date"] or td > _parse_date(a["latest_transaction_date"])):
            a["latest_transaction_date"] = td.isoformat()
        fd_ = _parse_date(p.get("filing_date", ""))
        if fd_ and (not a["latest_filing_date"] or fd_ > _parse_date(a["latest_filing_date"])):
            a["latest_filing_date"] = fd_.isoformat()
    # finalise sets → sorted lists, add insider_count
    out: dict[str, dict] = {}
    for sym, a in agg.items():
        insiders = list(a["insiders"].values())
        rec = {
            "symbol": sym, "company": a["company"],
            "total_open_market_purchase_value": round(a["total_open_market_purchase_value"], 2),
            "transaction_count": a["transaction_count"],
            "insider_count": len(insiders),
            "largest_purchase_value": round(a["largest_purchase_value"], 2),
            "latest_transaction_date": a["latest_transaction_date"],
            "latest_filing_date": a["latest_filing_date"],
            "insiders": insiders,
            "insider_roles": sorted(a["roles"]),
            "filings": sorted(a["filings"]),
            "sec_filing_urls": sorted(a["sec_filing_urls"]),
            "any_direct": a["any_direct"],
            "derivative_heavy": a["derivative_heavy"],
            "repeated_purchases": a["transaction_count"] >= 2 or any(i["buys"] >= 2 for i in insiders),
        }
        out[sym] = rec
    return out


def score_insider_aggregate(agg: dict, cfg: SECForm4Config | None = None) -> tuple[float, dict]:
    """
    Discovery-quality insider_buy_score (0-100). This is NOT a buy signal — it ranks
    discovery candidates and feeds the downstream conviction modifier only.

    +40 purchases >= min_transaction_value
    +15 CEO/CFO/Chairman/Director among buyers
    +15 multiple insiders
    +10 latest purchase within 7 days
    +10 any direct ownership
    +10 repeated purchases
    -25 ambiguous filing      -25 derivative-heavy      -50 non-open-market
    """
    cfg = cfg or SECForm4Config()
    score = 0.0
    bd: dict = {}
    if agg.get("total_open_market_purchase_value", 0.0) >= cfg.min_transaction_value:
        score += 40.0; bd["value>=min"] = 40
    roles = " ".join(agg.get("insider_roles", [])).lower()
    if any(r in roles for r in ("ceo", "cfo", "chairman", "director")):
        score += 15.0; bd["senior_role"] = 15
    if int(agg.get("insider_count", 0)) >= 2:
        score += 15.0; bd["multiple_insiders"] = 15
    latest = _parse_date(agg.get("latest_transaction_date", ""))
    if latest is not None and (datetime.now(ET_TZ).date() - latest).days <= 7:
        score += 10.0; bd["recent_7d"] = 10
    if agg.get("any_direct"):
        score += 10.0; bd["direct_ownership"] = 10
    if agg.get("repeated_purchases"):
        score += 10.0; bd["repeated"] = 10
    # penalties
    if agg.get("ambiguous"):
        score -= 25.0; bd["ambiguous"] = -25
    if agg.get("derivative_heavy"):
        score -= 25.0; bd["derivative_heavy"] = -25
    if agg.get("non_open_market"):
        score -= 50.0; bd["non_open_market"] = -50
    score = max(0.0, min(score, 100.0))
    return round(score, 1), bd


def build_candidates(filings: list, cfg: SECForm4Config | None = None) -> dict:
    """
    Pure pipeline (no network): given parsed Form 4 filings, produce the candidate set.

    filings: list of (parsed_dict, meta) tuples, where meta = {accession, filing_url, filing_date}.
    Returns the full result dict (candidates / rejected / errors / source ...).
    """
    cfg = cfg or SECForm4Config()
    purchases: list[dict] = []
    errors: list[dict] = []
    for item in filings:
        try:
            parsed, meta = item if isinstance(item, tuple) else (item, {})
            purchases.extend(extract_open_market_purchases(
                parsed, accession=meta.get("accession", ""),
                filing_url=meta.get("filing_url", ""), filing_date=meta.get("filing_date", "")))
        except Exception as exc:
            errors.append({"error": str(exc)[:200]})

    aggregated = aggregate_by_symbol(purchases)
    candidates, rejected = [], []
    for sym, agg in aggregated.items():
        score, breakdown = score_insider_aggregate(agg, cfg)
        agg["insider_buy_score"] = score
        agg["score_breakdown"] = breakdown
        agg["confidence_score"] = round(min(1.0, 0.4 + 0.15 * agg["insider_count"]
                                            + (0.3 if agg["total_open_market_purchase_value"] >= cfg.min_transaction_value else 0.0)), 2)
        agg["discovery_source"] = DISCOVERY_SOURCE
        if agg["total_open_market_purchase_value"] >= cfg.min_transaction_value:
            candidates.append(agg)
        else:
            agg["reject_reason"] = (f"below_min_transaction_value:"
                                    f"{agg['total_open_market_purchase_value']:.0f}<{cfg.min_transaction_value:.0f}")
            rejected.append(agg)

    candidates.sort(key=lambda c: c["total_open_market_purchase_value"], reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": cfg.lookback_days,
        "min_transaction_value": cfg.min_transaction_value,
        "candidates": candidates,
        "rejected": rejected,
        "errors": errors,
        "source": SOURCE_TAG,
    }


# ── network discovery (fail-safe) ────────────────────────────────────────────────
class SECForm4Feed:
    """SEC EDGAR Form 4 network client + discovery driver. Fully fail-safe."""

    DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index/{yr}/QTR{q}/master.{ymd}.idx"
    FILING_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"
    ARCHIVE_BASE = "https://www.sec.gov/Archives/"

    def __init__(self, config: SECForm4Config | None = None):
        self.cfg = config or SECForm4Config()
        self._cache = PROJECT_ROOT / self.cfg.cache_dir
        try:
            self._cache.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._last_request = 0.0

    # ── HTTP with UA, rate limit, disk cache ────────────────────────────────────
    def _cache_path(self, url: str) -> Path:
        return self._cache / (hashlib.sha1(url.encode()).hexdigest() + ".cache")

    def _get(self, url: str, cache_ttl: int | None = None) -> str | None:
        cp = self._cache_path(url)
        if cp.exists():
            try:
                if cache_ttl is None or (time.time() - cp.stat().st_mtime) < cache_ttl:
                    return cp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        # Rate limit (SEC politeness).
        wait = self.cfg.rate_limit_seconds - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": self.cfg.user_agent,
                "Accept-Encoding": "gzip, deflate",
            })
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout, context=_SSL_CTX) as resp:
                raw = resp.read()
                enc = resp.headers.get("Content-Encoding", "")
                if "gzip" in enc:
                    import gzip
                    raw = gzip.decompress(raw)
                text = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            self._last_request = time.time()
            raise exc
        self._last_request = time.time()
        try:
            cp.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return text

    def _index_urls(self) -> list[str]:
        urls = []
        today = datetime.now(ET_TZ).date()
        for i in range(min(self.cfg.lookback_days, self.cfg.max_index_days)):
            d = today - timedelta(days=i)
            if d.weekday() >= 5:   # skip weekends
                continue
            q = (d.month - 1) // 3 + 1
            urls.append(self.DAILY_INDEX.format(yr=d.year, q=q, ymd=d.strftime("%Y%m%d")))
        return urls

    def _form4_rows_from_index(self, idx_text: str) -> list[dict]:
        rows = []
        for line in idx_text.splitlines():
            parts = line.split("|")
            if len(parts) != 5:
                continue
            cik, company, form, filed, filename = parts
            if form.strip() != "4":
                continue
            rows.append({"cik": cik.strip(), "company": company.strip(),
                         "filing_date": filed.strip(), "filename": filename.strip()})
        return rows

    def _fetch_form4_xml(self, cik: str, filename: str) -> tuple[str, str]:
        """Return (xml_text, filing_url). filename is edgar/data/CIK/ACCESSION.txt."""
        m = re.search(r"/(\d{10}-\d{2}-\d{6})\.txt$", filename) or re.search(r"(\d{18})", filename)
        acc = ""
        if m:
            acc = m.group(1)
        acc_nodash = acc.replace("-", "")
        if not acc_nodash:
            return "", ""
        idx_url = self.FILING_INDEX.format(cik=cik.lstrip("0") or cik, acc_nodash=acc_nodash)
        index_json = self._get(idx_url, cache_ttl=None)
        if not index_json:
            return "", ""
        try:
            items = json.loads(index_json).get("directory", {}).get("item", [])
        except Exception:
            return "", ""
        xml_name = ""
        for it in items:
            name = str(it.get("name", ""))
            if name.lower().endswith(".xml") and any(k in name.lower() for k in ("form4", "ownership", "primary_doc", "doc4", "wk-form4", "wf-form4")):
                xml_name = name
                break
        if not xml_name:
            for it in items:
                if str(it.get("name", "")).lower().endswith(".xml"):
                    xml_name = it["name"]
                    break
        if not xml_name:
            return "", ""
        base = idx_url.rsplit("/", 1)[0] + "/"
        xml_url = base + xml_name
        return (self._get(xml_url, cache_ttl=None) or ""), xml_url

    def discover(self) -> dict:
        """Run the full network discovery. Fail-safe — partial failures become errors[]."""
        errors: list[dict] = []
        filings: list = []
        processed = 0
        for idx_url in self._index_urls():
            if processed >= self.cfg.max_filings:
                break
            try:
                idx_text = self._get(idx_url, cache_ttl=self.cfg.index_cache_ttl_seconds)
            except Exception as exc:
                errors.append({"index": idx_url, "error": str(exc)[:160]})
                continue
            if not idx_text:
                continue
            for row in self._form4_rows_from_index(idx_text):
                if processed >= self.cfg.max_filings:
                    break
                processed += 1
                try:
                    xml_text, filing_url = self._fetch_form4_xml(row["cik"], row["filename"])
                    if not xml_text:
                        continue
                    parsed = parse_form4_xml(xml_text)
                    if not parsed:
                        continue
                    # Quick pre-filter: only keep filings that contain a P purchase.
                    if not any(t.get("code") == OPEN_MARKET_PURCHASE for t in parsed.get("transactions", [])):
                        continue
                    acc_m = re.search(r"(\d{10}-\d{2}-\d{6})", row["filename"])
                    filings.append((parsed, {"accession": acc_m.group(1) if acc_m else "",
                                             "filing_url": filing_url, "filing_date": row["filing_date"]}))
                except Exception as exc:
                    errors.append({"filing": row.get("filename"), "error": str(exc)[:160]})
        result = build_candidates(filings, self.cfg)
        result["errors"].extend(errors)
        result["filings_scanned"] = processed
        return result


# ── public driver + loaders ──────────────────────────────────────────────────────
def _write_outputs(result: dict) -> None:
    try:
        SEC_FORM4_INSIDER_BUYS_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    # insider_buy_candidates.json — symbol-keyed map consumed by the Buffett side.
    try:
        cand_map = {}
        for c in result.get("candidates", []):
            cand_map[c["symbol"]] = {
                "symbol": c["symbol"], "company": c.get("company", ""),
                "insider_buy_score": c.get("insider_buy_score", 0.0),
                "insider_buy_amount": c.get("total_open_market_purchase_value", 0.0),
                "insider_count": c.get("insider_count", 0),
                "insider_roles": c.get("insider_roles", []),
                "latest_insider_buy_date": c.get("latest_transaction_date", ""),
                "latest_filing_date": c.get("latest_filing_date", ""),
                "sec_filing_urls": c.get("sec_filing_urls", []),
                "confidence_score": c.get("confidence_score", 0.0),
                "source": SOURCE_TAG, "discovery_source": DISCOVERY_SOURCE,
            }
        INSIDER_BUY_CANDIDATES_PATH.write_text(json.dumps({
            "generated_at": result.get("generated_at"),
            "source": SOURCE_TAG, "count": len(cand_map), "candidates": cand_map,
        }, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def run_sec_form4_discovery(config: SECForm4Config | None = None) -> dict:
    """
    Fail-safe discovery driver for run.py. Never raises. On any failure it logs the
    error, writes a diagnostic output (with the error recorded), and returns an empty
    candidate set so the Buffett screener / Alpha engine continue unaffected.

    Session-level caching: if insider_buy_candidates.json is younger than
    output_cache_ttl_seconds (default 4h), the cached result is returned immediately
    without making any network requests. This prevents the SEC discovery from blocking
    the main trading loop on every cycle.
    """
    cfg = config or SECForm4Config()
    if not cfg.enabled:
        result = {"generated_at": datetime.now(timezone.utc).isoformat(),
                  "lookback_days": cfg.lookback_days, "min_transaction_value": cfg.min_transaction_value,
                  "candidates": [], "rejected": [], "errors": [{"info": "sec_form4_feed_disabled"}],
                  "source": SOURCE_TAG}
        _write_outputs(result)
        return result

    # --- Session-level output cache guard ---
    # SEC Form 4 data is a discovery input (not a live signal), so refreshing once per
    # session window is sufficient. Skip the expensive network crawl if fresh output exists.
    try:
        if INSIDER_BUY_CANDIDATES_PATH.exists() and SEC_FORM4_INSIDER_BUYS_PATH.exists():
            age_secs = time.time() - INSIDER_BUY_CANDIDATES_PATH.stat().st_mtime
            if age_secs < cfg.output_cache_ttl_seconds:
                cached = json.loads(SEC_FORM4_INSIDER_BUYS_PATH.read_text(encoding="utf-8"))
                cand_count = len(cached.get("candidates", []))
                logger.info(
                    "SEC Form 4 discovery: using cached output (age=%.0fm, %d candidates) — "
                    "next refresh in %.0fm",
                    age_secs / 60,
                    cand_count,
                    (cfg.output_cache_ttl_seconds - age_secs) / 60,
                )
                return cached
    except Exception:
        pass  # Cache read failure → fall through to fresh network run

    try:
        result = SECForm4Feed(cfg).discover()
    except Exception as exc:
        logger.error("SEC Form 4 discovery failed (Buffett/Alpha unaffected): %s", exc)
        result = {"generated_at": datetime.now(timezone.utc).isoformat(),
                  "lookback_days": cfg.lookback_days, "min_transaction_value": cfg.min_transaction_value,
                  "candidates": [], "rejected": [], "errors": [{"error": str(exc)[:300]}],
                  "source": SOURCE_TAG}
    _write_outputs(result)
    return result


def load_insider_candidates() -> dict[str, dict]:
    """Load the symbol-keyed insider candidate map (empty on any failure)."""
    if not INSIDER_BUY_CANDIDATES_PATH.exists():
        return {}
    try:
        return json.loads(INSIDER_BUY_CANDIDATES_PATH.read_text()).get("candidates", {}) or {}
    except Exception:
        return {}


def discovered_symbols() -> list[str]:
    """Tickers SEC Form 4 discovered (to merge into the Buffett review universe)."""
    return sorted(load_insider_candidates().keys())


def insider_data_for_layer() -> dict[str, dict]:
    """
    Per-symbol insider data shaped for buffett_accumulation_layer.compute_insider_buy_score:
    {symbol: {amount, role, cluster, buys, score, confidence, detected}}.
    """
    out: dict[str, dict] = {}
    for sym, c in load_insider_candidates().items():
        roles = c.get("insider_roles", [])
        out[sym] = {
            "amount": float(c.get("insider_buy_amount", 0.0) or 0.0),
            "role": roles[0] if roles else "",
            "cluster": int(c.get("insider_count", 0) or 0) >= 2,
            "buys": int(c.get("insider_count", 0) or 0),
            "score": float(c.get("insider_buy_score", 0.0) or 0.0),
            "confidence": float(c.get("confidence_score", 0.75) or 0.75),
            "detected": True,
        }
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_sec_form4_discovery()
    print(json.dumps({"candidates": [c["symbol"] for c in res.get("candidates", [])],
                      "rejected": len(res.get("rejected", [])),
                      "errors": len(res.get("errors", []))}, indent=2))
