#!/usr/bin/env python3
"""
data_sources/alpha_vantage_client.py — Unified Alpha Vantage realtime client.

Single entry point for all Alpha Vantage equity + options market data the
trading system consumes, with first-class support for the `entitlement=realtime`
parameter (Alpha Vantage's realtime US market data entitlement).

Supported endpoints
-------------------
Equity / technical (entitlement-aware):
    TIME_SERIES_INTRADAY        -> get_intraday(symbol, interval)
    GLOBAL_QUOTE                -> get_global_quote(symbol)
    TIME_SERIES_DAILY_ADJUSTED  -> get_daily_adjusted(symbol)
    SMA / EMA / RSI / MACD / VWAP / BBANDS -> get_technical_indicator(...)

Options (realtime):
    REALTIME_OPTIONS                       -> get_realtime_options(symbol, require_greeks=True)
    REALTIME_PUT_CALL_RATIO                -> get_put_call_ratio(symbol)
    REALTIME_VOLUME_OPEN_INTEREST_RATIO    -> get_volume_oi_ratio(symbol)

Design contract (matches options_sentiment_av.py / insider_transactions_av.py):
    * API key read ONLY from the environment (ALPHA_VANTAGE_API_KEY, with the
      ALPHA_VANTAGE_API alias accepted for parity with existing modules).
      NEVER hardcoded, NEVER logged.
    * Advisory-safe: any failure (no key, timeout, throttle, bad payload)
      degrades to an empty / neutral result and is logged — never raised — so
      live trading is never interrupted.
    * Every request is logged (params WITH the apikey stripped) to
      logs/alpha_vantage_realtime_requests.csv and a rolling status snapshot is
      written to reports/data_sources/alpha_vantage_realtime_status.json.

Configuration (env)
-------------------
    ALPHA_VANTAGE_API_KEY            (required for live calls; else neutral)
    ALPHA_VANTAGE_ENTITLEMENT        "realtime" (default) | "delayed" | ""
    ALPHA_VANTAGE_ENABLED            "1"/"true" (default true)
    ALPHA_VANTAGE_INTRADAY_INTERVAL  "5min" (default)
    ALPHA_VANTAGE_TIMEOUT_SECONDS    12.0
    ALPHA_VANTAGE_MAX_RETRIES        2
    ALPHA_VANTAGE_RETRY_BACKOFF_SECONDS 1.5
    ALPHA_VANTAGE_MIN_REQUEST_INTERVAL_SECONDS 0.0  (politeness throttle)
    ALPHA_VANTAGE_CACHE_SECONDS      30  (realtime: short TTL)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:
    from config.env_loader import load_instance_env

    load_instance_env()
except Exception:  # pragma: no cover - env loader optional in isolated tests
    pass

try:
    import requests
except Exception:  # pragma: no cover - requests is a hard dep, but stay graceful
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AV_BASE_URL = "https://www.alphavantage.co/query"

DEFAULT_REQUEST_LOG = PROJECT_ROOT / "logs" / "alpha_vantage_realtime_requests.csv"
DEFAULT_STATUS_REPORT = PROJECT_ROOT / "reports" / "data_sources" / "alpha_vantage_realtime_status.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "system-data" / "av_realtime"

VALID_INTRADAY_INTERVALS = ("1min", "5min", "15min", "30min", "60min")

# Endpoints for which the realtime entitlement parameter is meaningful.
_ENTITLEMENT_FUNCTIONS = frozenset({
    "TIME_SERIES_INTRADAY",
    "GLOBAL_QUOTE",
    "TIME_SERIES_DAILY_ADJUSTED",
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3",
    "RSI", "MACD", "MACDEXT", "STOCH", "STOCHF", "STOCHRSI", "WILLR",
    "ADX", "ADXR", "APO", "PPO", "MOM", "BOP", "CCI", "CMO", "ROC", "ROCR",
    "AROON", "AROONOSC", "MFI", "TRIX", "ULTOSC", "DX", "MINUS_DI", "PLUS_DI",
    "MINUS_DM", "PLUS_DM", "BBANDS", "MIDPOINT", "MIDPRICE", "SAR", "TRANGE",
    "ATR", "NATR", "AD", "ADOSC", "OBV", "HT_TRENDLINE", "VWAP",
})

# Alpha Vantage notice keys that indicate throttle / info / error payloads.
_NOTICE_KEYS = ("Error Message", "Note", "Information")


# =============================================================================
# Env helpers
# =============================================================================

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return default


def _api_key() -> str:
    """API key — environment only. Never hardcoded."""
    return (os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API") or "").strip()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_float(value: Any) -> Optional[float]:
    """Lenient float parse. Returns None for missing / NaN / non-numeric values."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value in ("", "None", "nan", "NaN", "-", "N/A"):
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def to_int(value: Any) -> Optional[int]:
    f = to_float(value)
    return int(f) if f is not None else None


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of params with the apikey removed — for safe logging."""
    return {k: v for k, v in params.items() if k.lower() != "apikey"}


# =============================================================================
# Config
# =============================================================================

@dataclass
class AlphaVantageConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("ALPHA_VANTAGE_ENABLED", True))
    # "realtime" appends entitlement=realtime to supported equity requests.
    entitlement: str = field(default_factory=lambda: (os.getenv("ALPHA_VANTAGE_ENTITLEMENT", "realtime") or "").strip().lower())

    default_interval: str = field(default_factory=lambda: (os.getenv("ALPHA_VANTAGE_INTRADAY_INTERVAL", "5min") or "5min").strip())

    timeout_seconds: float = field(default_factory=lambda: _env_float("ALPHA_VANTAGE_TIMEOUT_SECONDS", 12.0))
    max_retries: int = field(default_factory=lambda: _env_int("ALPHA_VANTAGE_MAX_RETRIES", 2))
    retry_backoff_seconds: float = field(default_factory=lambda: _env_float("ALPHA_VANTAGE_RETRY_BACKOFF_SECONDS", 1.5))
    min_request_interval_seconds: float = field(default_factory=lambda: _env_float("ALPHA_VANTAGE_MIN_REQUEST_INTERVAL_SECONDS", 0.0))
    cache_seconds: int = field(default_factory=lambda: _env_int("ALPHA_VANTAGE_CACHE_SECONDS", 30))

    @property
    def entitlement_enabled(self) -> bool:
        return self.entitlement in {"realtime", "delayed"} and bool(self.entitlement)


# =============================================================================
# Client
# =============================================================================

class AlphaVantageClient:
    """Unified, advisory-safe Alpha Vantage realtime client.

    All public ``get_*`` methods always return a value (list / dict). On any
    failure they return an empty list or a neutral dict and log the reason; they
    never raise, so callers can use the result unconditionally.
    """

    def __init__(
        self,
        config: Optional[AlphaVantageConfig] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        request_log_path: Optional[Path] = None,
        status_report_path: Optional[Path] = None,
        http_get: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config or AlphaVantageConfig()
        # api_key explicitly passed only for tests; production reads env.
        self.api_key = api_key if api_key is not None else _api_key()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.request_log_path = Path(request_log_path) if request_log_path is not None else DEFAULT_REQUEST_LOG
        self.status_report_path = Path(status_report_path) if status_report_path is not None else DEFAULT_STATUS_REPORT
        self._http_get = http_get  # injectable for tests; defaults to requests.get
        self._mem_cache: dict[str, tuple[float, Any]] = {}
        self._last_request_ts: float = 0.0
        self.api_call_count = 0
        self.fallback_count = 0
        for d in (self.cache_dir, self.request_log_path.parent, self.status_report_path.parent):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - non-fatal
                pass

    # ---- entitlement / params ---------------------------------------------

    def _apply_entitlement(self, params: dict[str, str]) -> dict[str, str]:
        """Append entitlement=realtime when enabled and the function supports it."""
        fn = params.get("function", "")
        if self.config.entitlement_enabled and fn in _ENTITLEMENT_FUNCTIONS:
            params = dict(params)
            params["entitlement"] = self.config.entitlement
        return params

    # ---- logging / status --------------------------------------------------

    def _log_request(
        self,
        function: str,
        symbol: str,
        params: dict[str, Any],
        status: str,
        latency_ms: float,
        rows: int,
        fallback_used: bool,
        error: str = "",
    ) -> None:
        """Append one request record to the CSV log. API key is NEVER written."""
        safe_params = _redact(params)
        row = {
            "timestamp_utc": now_utc().isoformat(),
            "function": function,
            "symbol": symbol,
            "params": json.dumps(safe_params, sort_keys=True),
            "entitlement": safe_params.get("entitlement", ""),
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "rows": rows,
            "fallback_used": int(bool(fallback_used)),
            "error": (error or "")[:240],
        }
        try:
            new_file = not self.request_log_path.exists()
            with self.request_log_path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
        except Exception:  # pragma: no cover - logging must never break trading
            pass

    def write_status_report(self) -> dict[str, Any]:
        """Write + return a rolling status snapshot for monitoring."""
        status = {
            "generated_at_utc": now_utc().isoformat(),
            "enabled": self.config.enabled,
            "entitlement": self.config.entitlement,
            "entitlement_active": self.config.entitlement_enabled,
            "api_key_present": bool(self.api_key),
            "default_interval": self.config.default_interval,
            "api_call_count": self.api_call_count,
            "fallback_count": self.fallback_count,
            "base_url": AV_BASE_URL,
        }
        try:
            self.status_report_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        except Exception:  # pragma: no cover - non-fatal
            pass
        return status

    # ---- caching -----------------------------------------------------------

    def _cache_get(self, key: str) -> Any:
        ttl = max(self.config.cache_seconds, 0)
        cached = self._mem_cache.get(key)
        if cached is not None:
            ts, payload = cached
            if (time.time() - ts) <= ttl:
                return payload
        return None

    def _cache_put(self, key: str, payload: Any) -> None:
        self._mem_cache[key] = (time.time(), payload)

    # ---- low-level fetch ---------------------------------------------------

    def _do_http_get(self, params: dict[str, str]) -> dict[str, Any]:
        if self._http_get is not None:
            return self._http_get(AV_BASE_URL, params=params, timeout=self.config.timeout_seconds)
        if requests is None:
            raise RuntimeError("requests library unavailable")
        resp = requests.get(AV_BASE_URL, params=params, timeout=self.config.timeout_seconds)
        resp.raise_for_status()
        if "datatype" in params and params["datatype"] == "csv":
            return {"_csv_text": resp.text}
        return resp.json()

    def _throttle(self) -> None:
        gap = self.config.min_request_interval_seconds - (time.time() - self._last_request_ts)
        if gap > 0:
            try:
                time.sleep(gap)
            except Exception:
                pass

    @staticmethod
    def _notice(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in _NOTICE_KEYS:
            if key in payload:
                return f"{key}: {str(payload[key])[:160]}"
        return None

    def fetch(self, params: dict[str, str], *, cache: bool = True) -> Optional[dict[str, Any]]:
        """Fetch a raw payload with entitlement, retry, throttle, cache, logging.

        Returns the parsed JSON dict, or None on hard failure (logged).
        """
        function = params.get("function", "UNKNOWN")
        symbol = params.get("symbol", "")
        params = self._apply_entitlement(params)
        cache_key = json.dumps(_redact(params), sort_keys=True)

        if cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._log_request(function, symbol, params, "cache_hit", 0.0, _payload_rows(cached), False)
                return cached

        if not self.config.enabled:
            self._log_request(function, symbol, params, "disabled", 0.0, 0, True, "client disabled")
            self.fallback_count += 1
            return None
        if not self.api_key:
            self._log_request(function, symbol, params, "no_api_key", 0.0, 0, True, "ALPHA_VANTAGE_API_KEY unset")
            self.fallback_count += 1
            logger.debug("AV %s/%s skipped: no API key", function, symbol)
            return None

        # apikey injected here only — it is stripped from every log line.
        request_params = dict(params)
        request_params["apikey"] = self.api_key

        last_err: Optional[str] = None
        for attempt in range(max(self.config.max_retries, 0) + 1):
            self._throttle()
            t0 = time.time()
            try:
                self.api_call_count += 1
                payload = self._do_http_get(request_params)
                self._last_request_ts = time.time()
                latency_ms = (self._last_request_ts - t0) * 1000.0
                notice = self._notice(payload)
                if notice is not None:
                    last_err = notice
                    self._log_request(function, symbol, params, "notice", latency_ms, 0, False, notice)
                    logger.warning("AV %s %s notice: %s", function, symbol, notice)
                    break  # throttle/info notices won't clear within a retry window
                if cache:
                    self._cache_put(cache_key, payload)
                self._log_request(function, symbol, params, "ok", latency_ms, _payload_rows(payload), False)
                return payload
            except Exception as exc:
                self._last_request_ts = time.time()
                latency_ms = (self._last_request_ts - t0) * 1000.0
                last_err = str(exc)
                self._log_request(function, symbol, params, "error", latency_ms, 0, False, last_err)
                logger.warning(
                    "AV %s %s fetch failed (attempt %d/%d): %s",
                    function, symbol, attempt + 1, self.config.max_retries + 1, exc,
                )
                if attempt < self.config.max_retries:
                    try:
                        time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
                    except Exception:
                        pass

        self.fallback_count += 1
        if last_err:
            logger.error("AV %s %s unavailable -> fallback: %s", function, symbol, last_err)
        return None

    # =========================================================================
    # Public: equity / price endpoints
    # =========================================================================

    def get_intraday(self, symbol: str, interval: Optional[str] = None) -> list[dict[str, Any]]:
        """Realtime intraday OHLCV bars normalized to the standard schema.

        Returns a list (newest-last) of dicts: symbol, timestamp, open, high,
        low, close, volume, source, entitlement. Empty list on failure.
        """
        symbol = (symbol or "").upper().strip()
        interval = (interval or self.config.default_interval or "5min").strip()
        if interval not in VALID_INTRADAY_INTERVALS:
            logger.warning("AV intraday: invalid interval %r, defaulting to 5min", interval)
            interval = "5min"
        payload = self.fetch({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "compact",
            "datatype": "json",
        })
        if not payload:
            return []
        return normalize_time_series(payload, symbol, self.config.entitlement, source="alpha_vantage_intraday")

    def get_global_quote(self, symbol: str) -> dict[str, Any]:
        """Realtime global quote normalized to a single OHLCV-style dict."""
        symbol = (symbol or "").upper().strip()
        payload = self.fetch({"function": "GLOBAL_QUOTE", "symbol": symbol, "datatype": "json"})
        if not payload:
            return {}
        return normalize_global_quote(payload, symbol, self.config.entitlement)

    def get_daily_adjusted(self, symbol: str) -> list[dict[str, Any]]:
        """Realtime adjusted daily OHLCV bars normalized to the standard schema."""
        symbol = (symbol or "").upper().strip()
        payload = self.fetch({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "compact",
            "datatype": "json",
        })
        if not payload:
            return []
        return normalize_time_series(payload, symbol, self.config.entitlement, source="alpha_vantage_daily_adjusted")

    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: Optional[str] = None,
        time_period: int = 14,
        series_type: str = "close",
        extra: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        """Realtime technical indicator normalized to feature rows.

        ``indicator``: SMA, EMA, RSI, MACD, VWAP, BBANDS, ... Returns a list of
        dicts: {symbol, timestamp, <indicator columns...>, source, entitlement}.
        """
        symbol = (symbol or "").upper().strip()
        indicator = (indicator or "").upper().strip()
        interval = (interval or self.config.default_interval or "5min").strip()
        params: dict[str, str] = {
            "function": indicator,
            "symbol": symbol,
            "interval": interval,
            "datatype": "json",
        }
        # VWAP / OBV / AD take no time_period or series_type.
        if indicator not in {"VWAP", "OBV", "AD", "TRANGE", "HT_TRENDLINE", "MACD", "STOCH", "BOP", "SAR"}:
            params["time_period"] = str(int(time_period))
        if indicator not in {"VWAP", "OBV", "AD", "ATR", "NATR", "ADX", "AROON", "MFI",
                             "WILLR", "CCI", "DX", "MINUS_DI", "PLUS_DI", "SAR", "TRANGE"}:
            params["series_type"] = series_type
        if extra:
            params.update(extra)
        payload = self.fetch(params)
        if not payload:
            return []
        return normalize_technical_indicator(payload, symbol, indicator, self.config.entitlement)

    # =========================================================================
    # Public: options endpoints
    # =========================================================================

    def get_realtime_options(
        self,
        symbol: str,
        require_greeks: bool = True,
        contract: Optional[str] = None,
        expiration: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Realtime full option chain (calls + puts) normalized to one list.

        Each row: contract_id, symbol, expiration, strike, type, bid, ask, mark,
        last, volume, open_interest, implied_volatility, delta, gamma, theta,
        vega, rho, source, entitlement. Empty list on failure.
        """
        symbol = (symbol or "").upper().strip()
        params: dict[str, str] = {"function": "REALTIME_OPTIONS", "symbol": symbol, "datatype": "json"}
        if require_greeks:
            params["require_greeks"] = "true"
        if contract:
            params["contract"] = str(contract)
        # NB: REALTIME_OPTIONS is itself a realtime endpoint and does not take the
        # entitlement param; it is excluded from _ENTITLEMENT_FUNCTIONS.
        payload = self.fetch(params)
        if not payload:
            return []
        chain = normalize_options_chain(payload, symbol, self.config.entitlement)
        if expiration:
            chain = [c for c in chain if c.get("expiration") == expiration]
        return chain

    def get_put_call_ratio(self, symbol: str) -> dict[str, Any]:
        """Realtime put-call ratio (chain aggregate + per-expiration)."""
        symbol = (symbol or "").upper().strip()
        payload = self.fetch({"function": "REALTIME_PUT_CALL_RATIO", "symbol": symbol})
        if not payload:
            return {"symbol": symbol, "available": False, "put_call_ratio_chain": None,
                    "put_call_ratio_expiration": {}}
        chain, by_exp = _parse_put_call_ratio(payload)
        return {
            "symbol": symbol,
            "available": chain is not None or bool(by_exp),
            "put_call_ratio_chain": chain,
            "put_call_ratio_expiration": by_exp,
            "source": "alpha_vantage",
            "entitlement": self.config.entitlement,
        }

    def get_volume_oi_ratio(self, symbol: str) -> dict[str, Any]:
        """Realtime volume/open-interest ratio (chain aggregate + per-side)."""
        symbol = (symbol or "").upper().strip()
        payload = self.fetch({"function": "REALTIME_VOLUME_OPEN_INTEREST_RATIO", "symbol": symbol})
        if not payload:
            return {"symbol": symbol, "available": False, "volume_oi_ratio_chain": None,
                    "volume_oi_ratio_call": None, "volume_oi_ratio_put": None}
        chain, call_v, put_v = _parse_volume_oi_ratio(payload)
        return {
            "symbol": symbol,
            "available": chain is not None,
            "volume_oi_ratio_chain": chain,
            "volume_oi_ratio_call": call_v,
            "volume_oi_ratio_put": put_v,
            "source": "alpha_vantage",
            "entitlement": self.config.entitlement,
        }

    # =========================================================================
    # Public: insider transactions (long-TTL; delegates to insider_transactions_av)
    # =========================================================================

    def get_insider_transactions(
        self,
        symbol: str,
        from_days: int = 180,
        from_date: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Normalized SEC Form 4 insider transactions for *symbol*.

        Delegates to ``insider_transactions_av.fetch_insider_transactions`` which
        maintains a 7-day disk cache (separate from the 30-second in-memory cache
        used for price/options data). Fully advisory-safe: returns [] on any failure.

        Each row carries the fields produced by ``insider_transactions_av.normalize_transaction``:
            transaction_date, executive_name, executive_title, executive_role,
            security_type, acquisition_or_disposal, transaction_class,
            shares, price, value, symbol, is_buy, is_open_market, role_weight
        """
        symbol = (symbol or "").upper().strip()
        t0 = time.time()
        try:
            from insider_transactions_av import AVInsiderConfig, fetch_insider_transactions
            from datetime import date, timedelta

            cfg = AVInsiderConfig()
            # Prefer the key explicitly supplied to this client instance.
            if self.api_key:
                cfg.api_key = self.api_key
            if from_date is None:
                from_date = date.today() - timedelta(days=from_days)
            txns = fetch_insider_transactions(symbol, from_date=from_date, config=cfg)
            latency_ms = (time.time() - t0) * 1000.0
            self._log_request(
                "INSIDER_TRANSACTIONS", symbol,
                {"function": "INSIDER_TRANSACTIONS", "symbol": symbol},
                "ok", latency_ms, len(txns), False,
            )
            return txns
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000.0
            self._log_request(
                "INSIDER_TRANSACTIONS", symbol,
                {"function": "INSIDER_TRANSACTIONS", "symbol": symbol},
                "error", latency_ms, 0, True, str(exc)[:160],
            )
            self.fallback_count += 1
            logger.warning("AV INSIDER_TRANSACTIONS %s failed: %s", symbol, exc)
            return []


# =============================================================================
# Normalizers (pure functions; safe to unit-test in isolation)
# =============================================================================

def _payload_rows(payload: Any) -> int:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, list):
                return len(v)
            if "Time Series" in k and isinstance(v, dict):
                return len(v)
    return 0


def _std_bar(symbol: str, ts: str, ohlcv: dict[str, Any], entitlement: str, source: str) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "timestamp": ts,
        "open": to_float(ohlcv.get("1. open") or ohlcv.get("open")),
        "high": to_float(ohlcv.get("2. high") or ohlcv.get("high")),
        "low": to_float(ohlcv.get("3. low") or ohlcv.get("low")),
        "close": to_float(ohlcv.get("4. close") or ohlcv.get("5. adjusted close") or ohlcv.get("close")),
        "volume": to_float(ohlcv.get("6. volume") or ohlcv.get("5. volume") or ohlcv.get("volume")),
        "source": source,
        "entitlement": entitlement,
    }


def normalize_time_series(payload: dict[str, Any], symbol: str, entitlement: str,
                          source: str = "alpha_vantage") -> list[dict[str, Any]]:
    """Normalize TIME_SERIES_INTRADAY / DAILY_ADJUSTED payloads to OHLCV rows
    (sorted oldest-first)."""
    if not isinstance(payload, dict):
        return []
    series_key = next((k for k in payload if "Time Series" in k), None)
    if series_key is None:
        return []
    series = payload.get(series_key) or {}
    rows: list[dict[str, Any]] = []
    for ts, ohlcv in series.items():
        if isinstance(ohlcv, dict):
            rows.append(_std_bar(symbol, ts, ohlcv, entitlement, source))
    rows.sort(key=lambda r: r.get("timestamp") or "")
    return rows


def normalize_global_quote(payload: dict[str, Any], symbol: str, entitlement: str) -> dict[str, Any]:
    """Normalize GLOBAL_QUOTE to a single OHLCV-style dict."""
    quote = (payload or {}).get("Global Quote") or (payload or {}).get("Global Quote - DATA DELAYED BY 15 MINUTES") or {}
    if not isinstance(quote, dict) or not quote:
        return {}
    return {
        "symbol": (quote.get("01. symbol") or symbol).upper(),
        "timestamp": quote.get("07. latest trading day") or "",
        "open": to_float(quote.get("02. open")),
        "high": to_float(quote.get("03. high")),
        "low": to_float(quote.get("04. low")),
        "close": to_float(quote.get("05. price")),
        "price": to_float(quote.get("05. price")),
        "volume": to_float(quote.get("06. volume")),
        "previous_close": to_float(quote.get("08. previous close")),
        "change": to_float(quote.get("09. change")),
        "change_percent": to_float((quote.get("10. change percent") or "").replace("%", "")),
        "source": "alpha_vantage_global_quote",
        "entitlement": entitlement,
    }


def normalize_technical_indicator(payload: dict[str, Any], symbol: str, indicator: str,
                                  entitlement: str) -> list[dict[str, Any]]:
    """Normalize a technical-indicator payload to feature rows (oldest-first)."""
    if not isinstance(payload, dict):
        return []
    tech_key = next((k for k in payload if k.startswith("Technical Analysis")), None)
    if tech_key is None:
        return []
    series = payload.get(tech_key) or {}
    rows: list[dict[str, Any]] = []
    prefix = indicator.lower()
    for ts, values in series.items():
        if not isinstance(values, dict):
            continue
        row: dict[str, Any] = {"symbol": symbol.upper(), "timestamp": ts,
                               "source": f"alpha_vantage_{prefix}", "entitlement": entitlement}
        for name, val in values.items():
            col = f"{prefix}_{name}".lower().replace(" ", "_")
            row[col] = to_float(val)
        rows.append(row)
    rows.sort(key=lambda r: r.get("timestamp") or "")
    return rows


def normalize_options_chain(payload: dict[str, Any], symbol: str, entitlement: str) -> list[dict[str, Any]]:
    """Normalize a REALTIME_OPTIONS payload to a unified calls+puts list.

    Tolerant of field-name variants and missing Greeks. Each row carries a
    computed ``mark`` (mid) when bid/ask are present.
    """
    if not isinstance(payload, dict):
        return []
    records = payload.get("data") or payload.get("options") or []
    if not isinstance(records, list):
        return []
    out: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        bid = to_float(rec.get("bid"))
        ask = to_float(rec.get("ask"))
        mark = to_float(rec.get("mark"))
        if mark is None and bid is not None and ask is not None:
            mark = (bid + ask) / 2.0
        ctype = str(rec.get("type") or rec.get("option_type") or "").lower().strip()
        if ctype in ("c", "call"):
            ctype = "call"
        elif ctype in ("p", "put"):
            ctype = "put"
        out.append({
            "contract_id": rec.get("contractID") or rec.get("contract_id") or rec.get("contractId") or "",
            "symbol": (rec.get("symbol") or symbol).upper(),
            "expiration": rec.get("expiration") or rec.get("expiration_date") or "",
            "strike": to_float(rec.get("strike")),
            "type": ctype,
            "bid": bid,
            "ask": ask,
            "mark": mark,
            "last": to_float(rec.get("last") or rec.get("last_price")),
            "volume": to_int(rec.get("volume")),
            "open_interest": to_int(rec.get("open_interest") or rec.get("openInterest")),
            "implied_volatility": to_float(rec.get("implied_volatility") or rec.get("impliedVolatility") or rec.get("iv")),
            "delta": to_float(rec.get("delta")),
            "gamma": to_float(rec.get("gamma")),
            "theta": to_float(rec.get("theta")),
            "vega": to_float(rec.get("vega")),
            "rho": to_float(rec.get("rho")),
            "source": "alpha_vantage_realtime_options",
            "entitlement": entitlement,
        })
    return out


def _parse_put_call_ratio(payload: dict[str, Any]) -> tuple[Optional[float], dict[str, float]]:
    """Extract (chain_ratio, {expiration: ratio}) from a PCR payload."""
    chain = to_float(payload.get("put_call_ratio_full_chain") or payload.get("put_call_ratio"))
    by_exp: dict[str, float] = {}
    records = payload.get("put_call_ratio_by_expiration") or []
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            exp = rec.get("date") or rec.get("expiration")
            val = to_float(rec.get("value", rec.get("put_call_ratio")))
            if exp and val is not None:
                by_exp[str(exp)] = val
    if chain is None and by_exp:
        chain = sum(by_exp.values()) / len(by_exp)
    return chain, by_exp


def _parse_volume_oi_ratio(payload: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Aggregate per-contract volume/OI ratios into (chain_mean, call_mean, put_mean)."""
    records = payload.get("data") or []
    all_vals: list[float] = []
    call_vals: list[float] = []
    put_vals: list[float] = []
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            val = to_float(rec.get("volume_open_interest_ratio") or rec.get("volume_oi_ratio"))
            if val is None:
                continue
            all_vals.append(val)
            ctype = str(rec.get("type") or "").lower()
            if ctype.startswith("c"):
                call_vals.append(val)
            elif ctype.startswith("p"):
                put_vals.append(val)
    chain = (sum(all_vals) / len(all_vals)) if all_vals else None
    call_m = (sum(call_vals) / len(call_vals)) if call_vals else None
    put_m = (sum(put_vals) / len(put_vals)) if put_vals else None
    return chain, call_m, put_m


# =============================================================================
# Module-level singleton convenience
# =============================================================================

_DEFAULT_CLIENT: Optional[AlphaVantageClient] = None


def get_default_av_client() -> AlphaVantageClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = AlphaVantageClient()
    return _DEFAULT_CLIENT


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["IBM"]
    client = AlphaVantageClient()
    print(f"API key present: {bool(client.api_key)} | entitlement: {client.config.entitlement} "
          f"| enabled: {client.config.enabled}")
    for s in syms:
        quote = client.get_global_quote(s)
        bars = client.get_intraday(s)
        opts = client.get_realtime_options(s)
        print(json.dumps({
            "symbol": s,
            "quote": quote,
            "intraday_bars": len(bars),
            "latest_bar": bars[-1] if bars else None,
            "option_contracts": len(opts),
        }, indent=2, default=str))
    client.write_status_report()
