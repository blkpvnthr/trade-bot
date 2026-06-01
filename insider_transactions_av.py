#!/usr/bin/env python3
"""
insider_transactions_av.py — Alpha Vantage INSIDER_TRANSACTIONS advisory signal.

Purpose
    Fetch recent insider buying/selling for a symbol and score the trend as an
    advisory Buffett-style quality/conviction signal. This module never triggers
    a buy, never overrides valuation/quality/liquidity/risk gates, and never
    raises from public APIs. Failures return a neutral result.

Auth
    Reads ALPHA_VANTAGE_API_KEY from environment or from a project .env file.
    The key is never printed or logged.

Cache
    data/cache/alpha_vantage/insider_transactions/{SYMBOL}_{YYYYMMDD}.json

Public API
    get_insider_trend(symbol, from_days=None, from_date=None, config=None) -> dict
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York") if ZoneInfo else None


def _load_project_env() -> None:
    """Load .env from common project locations without failing if dotenv is absent."""
    if load_dotenv is None:
        return

    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        Path.cwd() / ".env",
    ]

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            load_dotenv(path, override=False)


_load_project_env()


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.replace("_", "").strip())
    except Exception:
        logger.warning("AV_INSIDER: invalid int env %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw.replace("_", "").strip())
    except Exception:
        logger.warning("AV_INSIDER: invalid float env %s=%r; using %s", name, raw, default)
        return float(default)


def _today_et() -> date:
    if ET is not None:
        return datetime.now(ET).date()
    return date.today()


def _now_et_iso() -> str:
    if ET is not None:
        return datetime.now(ET).isoformat()
    return datetime.now().isoformat()


@dataclass
class AVInsiderConfig:
    api_key: str = field(
        default_factory=lambda: os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    )
    enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_AV_INSIDER", True))
    from_days: int = field(default_factory=lambda: _env_int("AV_INSIDER_FROM_DAYS", 180))
    cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "AV_INSIDER_CACHE_DIR",
            "data/cache/alpha_vantage/insider_transactions",
        ).strip()
    )
    cache_ttl_days: int = field(default_factory=lambda: _env_int("AV_INSIDER_CACHE_TTL_DAYS", 7))
    rate_limit_seconds: float = field(
        default_factory=lambda: _env_float("AV_INSIDER_RATE_LIMIT_SECONDS", 15.0)
    )
    request_timeout: int = field(default_factory=lambda: _env_int("AV_INSIDER_REQUEST_TIMEOUT", 20))
    scale_usd: float = field(default_factory=lambda: _env_float("AV_INSIDER_SCALE_USD", 1_000_000.0))
    sell_discount: float = field(default_factory=lambda: _env_float("AV_INSIDER_SELL_DISCOUNT", 0.70))

    def cache_path(self, symbol: str) -> Path:
        safe_symbol = "".join(ch for ch in symbol.upper().strip() if ch.isalnum() or ch in {".", "-"})
        today = _today_et().strftime("%Y%m%d")
        base = PROJECT_ROOT / self.cache_dir
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{safe_symbol}_{today}.json"


_SENIOR_ROLES = frozenset({"CEO", "CFO", "Chairman", "Founder"})
_MID_ROLES = frozenset({"Director", "Officer", "President", "COO", "EVP", "SVP"})


def _classify_role(title: str) -> str:
    text = (title or "").lower()
    if any(k in text for k in ("chief executive", "ceo")):
        return "CEO"
    if any(k in text for k in ("chief financial", "cfo")):
        return "CFO"
    if any(k in text for k in ("chairman", "chair")):
        return "Chairman"
    if "founder" in text:
        return "Founder"
    if any(k in text for k in ("president", "chief operating", "coo")):
        return "President"
    if "evp" in text or "executive vice" in text:
        return "EVP"
    if "svp" in text or "senior vice" in text:
        return "SVP"
    if "chief" in text or "officer" in text:
        return "Officer"
    if "director" in text:
        return "Director"
    if text:
        return "Insider"
    return "Unknown"


def _role_weight(role: str) -> float:
    if role in _SENIOR_ROLES:
        return 1.5
    if role in _MID_ROLES:
        return 1.0
    if role == "Insider":
        return 0.6
    return 0.4


def _classify_transaction(acquisition_or_disposal: str, security_type: str, price: float) -> str:
    """
    Returns:
        open_market_buy
        open_market_sell
        routine_sell
        non_qualifying
    """
    aod = (acquisition_or_disposal or "").upper().strip()
    sec = (security_type or "").lower().strip()

    is_common = any(k in sec for k in ("common stock", "common share", "ordinary share"))
    is_option_like = any(k in sec for k in ("option", "right to buy", "warrant"))
    is_comp = any(k in sec for k in ("rsu", "restricted", "phantom", "award", "unit", "promised fee"))

    if is_common and not is_comp:
        if aod == "A" and price > 0:
            return "open_market_buy"
        if aod == "D":
            return "open_market_sell"

    if aod == "D" and is_option_like:
        return "routine_sell"

    return "non_qualifying"


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(str(value).replace(",", "").replace("_", "").strip())
    except Exception:
        return 0.0


def _safe_date(value: Any) -> Optional[date]:
    try:
        text = str(value or "")[:10]
        if not text:
            return None
        return date.fromisoformat(text)
    except Exception:
        return None


def normalize_transaction(raw: Dict[str, Any]) -> Dict[str, Any]:
    aod = str(
        raw.get("acquisition_or_disposal")
        or raw.get("acquisitionOrDisposal")
        or ""
    ).upper().strip()

    security_type = str(
        raw.get("security_type")
        or raw.get("securityType")
        or raw.get("transaction_type")
        or raw.get("transactionType")
        or ""
    ).strip()

    shares = _safe_float(raw.get("shares") or raw.get("number_of_shares") or raw.get("numberOfShares"))
    price = _safe_float(raw.get("share_price") or raw.get("sharePrice") or raw.get("price"))

    value = _safe_float(raw.get("value"))
    if value <= 0 and shares > 0 and price > 0:
        value = shares * price

    title = str(raw.get("executive_title") or raw.get("executiveTitle") or "").strip()
    role = _classify_role(title)
    txn_class = _classify_transaction(aod, security_type, price)

    return {
        "transaction_date": str(raw.get("transaction_date") or raw.get("transactionDate") or ""),
        "executive_name": str(raw.get("executive") or raw.get("executiveName") or "").strip(),
        "executive_title": title,
        "executive_role": role,
        "security_type": security_type,
        "acquisition_or_disposal": aod,
        "transaction_class": txn_class,
        "shares": shares,
        "price": price,
        "value": round(value, 2),
        "symbol": str(raw.get("ticker") or raw.get("symbol") or "").upper().strip(),
        "is_buy": txn_class == "open_market_buy",
        "is_open_market": txn_class in {"open_market_buy", "open_market_sell"},
        "role_weight": _role_weight(role),
    }


_last_request_ts = 0.0


def _api_url(symbol: str, api_key: str) -> str:
    query = urllib.parse.urlencode(
        {
            "function": "INSIDER_TRANSACTIONS",
            "symbol": symbol.upper().strip(),
            "apikey": api_key,
        }
    )
    return f"https://www.alphavantage.co/query?{query}"


def _fetch_from_api(symbol: str, config: AVInsiderConfig) -> List[Dict[str, Any]]:
    """Fetch raw rows. Returns [] on any failure."""
    global _last_request_ts

    if not config.api_key:
        logger.warning("AV_INSIDER: ALPHA_VANTAGE_API_KEY not set; returning empty")
        return []

    wait = config.rate_limit_seconds - (time.time() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)

    try:
        req = urllib.request.Request(
            _api_url(symbol, config.api_key),
            headers={"User-Agent": "QuantumQuant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=config.request_timeout) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        logger.warning("AV_INSIDER: HTTP %s for %s", exc.code, symbol)
        return []
    except Exception as exc:
        logger.warning("AV_INSIDER: request failed for %s: %s", symbol, exc)
        return []
    finally:
        _last_request_ts = time.time()

    try:
        payload = json.loads(raw_text)
    except Exception:
        logger.warning("AV_INSIDER: JSON parse failed for %s", symbol)
        return []

    info_message = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
    if info_message:
        logger.warning("AV_INSIDER: Alpha Vantage message for %s: %s", symbol, str(info_message)[:160])
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    return [row for row in data if isinstance(row, dict)]


def _filter_and_normalize(raw_rows: List[Dict[str, Any]], from_date: date) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in raw_rows or []:
        norm = normalize_transaction(row)
        txn_date = _safe_date(norm.get("transaction_date"))
        if txn_date is None or txn_date < from_date:
            continue
        output.append(norm)
    return output


def fetch_insider_transactions(
    symbol: str,
    from_date: Optional[date | str] = None,
    config: Optional[AVInsiderConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or AVInsiderConfig()
    if not cfg.enabled:
        return []

    if from_date is None:
        cutoff = _today_et() - timedelta(days=cfg.from_days)
    elif isinstance(from_date, str):
        cutoff = _safe_date(from_date) or (_today_et() - timedelta(days=cfg.from_days))
    else:
        cutoff = from_date

    cache_path = cfg.cache_path(symbol)

    if cache_path.exists():
        try:
            modified = date.fromtimestamp(cache_path.stat().st_mtime)
            age_days = (_today_et() - modified).days
            if age_days < cfg.cache_ttl_days:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                rows = cached.get("data") if isinstance(cached, dict) else []
                if isinstance(rows, list):
                    return _filter_and_normalize(rows, cutoff)
        except Exception:
            logger.debug("AV_INSIDER: cache read failed for %s", symbol, exc_info=True)

    raw_rows = _fetch_from_api(symbol, cfg)

    try:
        cache_path.write_text(
            json.dumps(
                {
                    "fetched_at": _now_et_iso(),
                    "symbol": symbol.upper().strip(),
                    "data": raw_rows,
                },
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("AV_INSIDER: cache write failed for %s", symbol, exc_info=True)

    return _filter_and_normalize(raw_rows, cutoff)


def compute_insider_trend_score(
    transactions: List[Dict[str, Any]],
    from_days: int = 180,
    config: Optional[AVInsiderConfig] = None,
) -> Dict[str, Any]:
    cfg = config or AVInsiderConfig()

    buys = [t for t in transactions if t.get("transaction_class") == "open_market_buy"]
    open_sells = [t for t in transactions if t.get("transaction_class") == "open_market_sell"]
    routine_sells = [t for t in transactions if t.get("transaction_class") == "routine_sell"]

    buy_value = sum(_safe_float(t.get("value")) for t in buys)
    sell_value = sum(_safe_float(t.get("value")) for t in open_sells)
    routine_sell_value = sum(_safe_float(t.get("value")) for t in routine_sells)

    net_value = buy_value - sell_value
    net_shares = sum(_safe_float(t.get("shares")) for t in buys) - sum(
        _safe_float(t.get("shares")) for t in open_sells
    )

    buy_count = len(buys)
    sell_count = len(open_sells)

    if sell_value > 0:
        buy_sell_ratio = buy_value / sell_value
    elif buy_value > 0:
        buy_sell_ratio = 9999.0
    else:
        buy_sell_ratio = 1.0

    weighted_buy = sum(_safe_float(t.get("value")) * _safe_float(t.get("role_weight")) for t in buys)
    weighted_sell = sum(_safe_float(t.get("value")) * _safe_float(t.get("role_weight")) for t in open_sells)
    weighted_routine = sum(
        _safe_float(t.get("value")) * _safe_float(t.get("role_weight")) * cfg.sell_discount
        for t in routine_sells
    )

    net_weighted = weighted_buy - weighted_sell - weighted_routine
    trend_score = math.tanh(net_weighted / max(1.0, cfg.scale_usd))

    buyer_names = {str(t.get("executive_name") or "").strip() for t in buys if t.get("executive_name")}
    cluster_count = len(buyer_names)
    if trend_score > 0 and cluster_count >= 3:
        trend_score *= 1.25
    elif trend_score > 0 and cluster_count >= 2:
        trend_score *= 1.10

    trend_score = round(max(-1.0, min(1.0, trend_score)), 4)

    if trend_score >= 0.70:
        modifier = 10.0
    elif trend_score >= 0.40:
        modifier = 6.0
    elif trend_score >= 0.15:
        modifier = 3.0
    elif trend_score >= -0.10:
        modifier = 0.0
    elif trend_score >= -0.35:
        modifier = -3.0
    elif trend_score >= -0.60:
        modifier = -6.0
    else:
        modifier = -10.0

    flags: List[str] = []
    senior_roles = {"CEO", "CFO", "Chairman", "Founder"}

    if not transactions:
        flags.append("no_recent_insider_data")
    if sell_value > 0 and buy_value == 0:
        flags.append("no_open_market_buys")
    if sell_value >= 3 * max(buy_value, 1.0):
        flags.append("heavy_insider_selling")
    if any(t.get("executive_role") in senior_roles for t in open_sells) and not any(
        t.get("executive_role") in senior_roles for t in buys
    ):
        flags.append("senior_exec_selling_no_buying")
    if routine_sells and not open_sells and not buys:
        flags.append("only_routine_option_exercise_sells")
    if buy_count == 1 and not any(t.get("executive_role") in senior_roles for t in buys):
        flags.append("single_minor_insider_buy")

    if not transactions:
        summary = f"No insider transactions in past {from_days} days. Neutral."
    elif buy_count > 0 and sell_count == 0:
        roles = ", ".join(sorted({str(t.get("executive_role")) for t in buys if t.get("executive_role")}))
        summary = f"{buy_count} open-market buy(s) by {roles} totalling ${buy_value:,.0f}. No discretionary selling."
    elif buy_count > 0 and sell_count > 0:
        summary = f"{buy_count} open-market buy(s) (${buy_value:,.0f}) vs {sell_count} discretionary sell(s) (${sell_value:,.0f}). Mixed activity."
    elif sell_count > 0:
        summary = f"No open-market buying. {sell_count} discretionary sell(s) totalling ${sell_value:,.0f}."
    else:
        summary = f"Only routine/planned transactions ({len(routine_sells)} exercise sales). Neutral."

    return {
        "insider_net_shares": round(net_shares, 0),
        "insider_net_value": round(net_value, 2),
        "insider_buy_value": round(buy_value, 2),
        "insider_sell_value": round(sell_value, 2),
        "insider_routine_sell_value": round(routine_sell_value, 2),
        "insider_buy_count": buy_count,
        "insider_sell_count": sell_count,
        "insider_buy_sell_ratio": round(buy_sell_ratio, 4),
        "insider_trend_score": trend_score,
        "insider_confidence_modifier": round(modifier, 2),
        "insider_warning_flags": flags,
        "insider_summary_text": summary,
    }


_NEUTRAL: Dict[str, Any] = {
    "insider_net_shares": 0.0,
    "insider_net_value": 0.0,
    "insider_buy_value": 0.0,
    "insider_sell_value": 0.0,
    "insider_routine_sell_value": 0.0,
    "insider_buy_count": 0,
    "insider_sell_count": 0,
    "insider_buy_sell_ratio": 1.0,
    "insider_trend_score": 0.0,
    "insider_confidence_modifier": 0.0,
    "insider_warning_flags": ["no_recent_insider_data"],
    "insider_summary_text": "No data available. Neutral.",
    "insider_from_date": "",
    "fetch_error": "",
}


def get_insider_trend(
    symbol: str,
    from_days: Optional[int] = None,
    from_date: Optional[date | str] = None,
    config: Optional[AVInsiderConfig] = None,
) -> Dict[str, Any]:
    cfg = config or AVInsiderConfig()
    effective_days = int(from_days if from_days is not None else cfg.from_days)

    if from_date is None:
        cutoff = _today_et() - timedelta(days=effective_days)
    elif isinstance(from_date, str):
        cutoff = _safe_date(from_date) or (_today_et() - timedelta(days=effective_days))
    else:
        cutoff = from_date

    result = dict(_NEUTRAL)
    result["insider_from_date"] = cutoff.isoformat()

    if not cfg.enabled:
        result["fetch_error"] = "disabled"
        return result

    if not cfg.api_key:
        result["fetch_error"] = "no_api_key"
        return result

    try:
        transactions = fetch_insider_transactions(symbol, from_date=cutoff, config=cfg)
        scored = compute_insider_trend_score(transactions, from_days=effective_days, config=cfg)
        result.update(scored)
        result["insider_from_date"] = cutoff.isoformat()
        result["fetch_error"] = ""
        return result
    except Exception as exc:
        logger.warning("AV_INSIDER: get_insider_trend failed for %s: %s", symbol, exc)
        result["fetch_error"] = str(exc)[:120]
        return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    output = get_insider_trend(ticker)
    print(json.dumps(output, indent=2, default=str))
