#!/usr/bin/env python3
"""
data_sources/alpha_vantage_analytics_client.py — Alpha Vantage Advanced Analytics.

Single entry point for the Alpha Vantage *Advanced Analytics* endpoints, which
return precomputed statistical features (means, drawdowns, autocorrelation,
correlation / covariance matrices, histograms, ...) over windows of returns:

    ANALYTICS_FIXED_WINDOW   -> get_fixed_window(symbols, calculations, range, interval, ...)
    ANALYTICS_SLIDING_WINDOW -> get_sliding_window(symbols, calculations, range, interval, window_size, ...)

These feed the §2/§3 statistical feature families consumed by universe
discovery, regime detection, hidden-alpha discovery, risk management, and the
(gated) RL options state.

Design contract (matches data_sources/alpha_vantage_client.py):
    * API key read ONLY from the environment (ALPHA_VANTAGE_API_KEY, with the
      ALPHA_VANTAGE_API alias). NEVER hardcoded, NEVER logged.
    * Advisory-safe: any failure (no key, timeout, throttle, bad payload)
      degrades to a local-OHLCV computation where possible, else a neutral
      result — it is logged, never raised, so live trading is never interrupted.
    * Every request is logged (params WITH the apikey stripped) to
      logs/alpha_vantage_analytics_requests.csv with the §9 schema, and a rolling
      status snapshot is written to
      reports/data_sources/alpha_vantage_analytics_status.json.

Request parameters (Advanced Analytics):
    SYMBOLS       comma-joined list of tickers
    RANGE         a single range token OR an explicit [start, end] pair
                  (repeated RANGE params), e.g. "3month", "full", "2023-07",
                  "2023-07-01", or ["2023-07-01", "2023-08-31"]
    INTERVAL      1min|5min|15min|30min|60min|DAILY|WEEKLY|MONTHLY
    CALCULATIONS  comma-joined list of calculation tokens, e.g.
                  ["MEAN", "STDDEV(annualized=True)", "CORRELATION(method=KENDALL)"]
    WINDOW_SIZE   sliding-window only; integer >= 10
    OHLC          optional: open|high|low|close (default close)

Configuration (env)
-------------------
    ALPHA_VANTAGE_API_KEY              (required for live calls; else fallback)
    ALPHA_VANTAGE_ANALYTICS_ENABLED   "1"/"true" (default true)
    ALPHA_VANTAGE_ANALYTICS_TIMEOUT_SECONDS    15.0
    ALPHA_VANTAGE_ANALYTICS_MAX_RETRIES        2
    ALPHA_VANTAGE_ANALYTICS_RETRY_BACKOFF_SECONDS 1.5
    ALPHA_VANTAGE_ANALYTICS_MIN_REQUEST_INTERVAL_SECONDS 0.0
    ALPHA_VANTAGE_ANALYTICS_CACHE_SECONDS      900   (analytics: longer TTL)
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
from typing import Any, Callable, Optional, Sequence, Union

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

DEFAULT_REQUEST_LOG = PROJECT_ROOT / "logs" / "alpha_vantage_analytics_requests.csv"
DEFAULT_STATUS_REPORT = PROJECT_ROOT / "reports" / "data_sources" / "alpha_vantage_analytics_status.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "system-data" / "av_analytics"

FIXED_WINDOW_FUNCTION = "ANALYTICS_FIXED_WINDOW"
SLIDING_WINDOW_FUNCTION = "ANALYTICS_SLIDING_WINDOW"

VALID_INTERVALS = ("1min", "5min", "15min", "30min", "60min", "DAILY", "WEEKLY", "MONTHLY")

# Per Alpha Vantage, sliding WINDOW_SIZE must be at least 10.
MIN_WINDOW_SIZE = 10

# Calculations whose payload is a {symbol: matrix-row} relationship object rather
# than a {symbol: scalar} mapping. Matched by the token BEFORE any "(" arguments.
_MATRIX_CALCS = frozenset({"CORRELATION", "COVARIANCE"})

# Calculations the ANALYTICS_SLIDING_WINDOW endpoint does NOT support. Verified
# against the live API: including ANY of these makes Alpha Vantage return an empty
# RETURNS_CALCULATIONS for the WHOLE request, so they are dropped (with a logged
# note) from sliding-window requests rather than silently breaking the response.
SLIDING_UNSUPPORTED_CALCS = frozenset({"MAX_DRAWDOWN", "MIN", "MAX"})

# Alpha Vantage notice keys that indicate throttle / info / error payloads.
_NOTICE_KEYS = ("Error Message", "Note", "Information")

TRADING_DAYS_PER_YEAR = 252.0


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


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of params with the apikey removed — for safe logging."""
    return {k: v for k, v in params.items() if k.lower() != "apikey"}


def _calc_base(token: str) -> str:
    """The calculation name with any ``(args)`` and whitespace stripped.

    e.g. "STDDEV(annualized=True)" -> "STDDEV", "CORRELATION(method=KENDALL)" -> "CORRELATION".
    """
    return str(token or "").split("(", 1)[0].strip().upper()


# =============================================================================
# Config
# =============================================================================

@dataclass
class AnalyticsConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("ALPHA_VANTAGE_ANALYTICS_ENABLED", True))
    timeout_seconds: float = field(default_factory=lambda: _env_float("ALPHA_VANTAGE_ANALYTICS_TIMEOUT_SECONDS", 15.0))
    max_retries: int = field(default_factory=lambda: _env_int("ALPHA_VANTAGE_ANALYTICS_MAX_RETRIES", 2))
    retry_backoff_seconds: float = field(default_factory=lambda: _env_float("ALPHA_VANTAGE_ANALYTICS_RETRY_BACKOFF_SECONDS", 1.5))
    min_request_interval_seconds: float = field(default_factory=lambda: _env_float("ALPHA_VANTAGE_ANALYTICS_MIN_REQUEST_INTERVAL_SECONDS", 0.0))
    cache_seconds: int = field(default_factory=lambda: _env_int("ALPHA_VANTAGE_ANALYTICS_CACHE_SECONDS", 900))


# =============================================================================
# Request building (pure; unit-testable in isolation)
# =============================================================================

def _normalize_symbols(symbols: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(symbols, str):
        raw = symbols.split(",")
    else:
        raw = list(symbols or [])
    out: list[str] = []
    for s in raw:
        s = str(s).upper().strip()
        if s and s not in out:
            out.append(s)
    return out


def _normalize_calculations(calculations: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(calculations, str):
        raw = calculations.split(",")
    else:
        raw = list(calculations or [])
    out: list[str] = []
    for c in raw:
        c = str(c).strip()
        if c and c not in out:
            out.append(c)
    return out


def build_analytics_params(
    function: str,
    symbols: Union[str, Sequence[str]],
    calculations: Union[str, Sequence[str]],
    range_: Union[str, Sequence[str]],
    interval: str,
    window_size: Optional[int] = None,
    ohlc: Optional[str] = None,
) -> dict[str, Any]:
    """Build the Advanced Analytics request parameter dict (no apikey).

    ``range_`` may be a single token (e.g. "3month") or a two-element [start, end]
    sequence, which is encoded as a *repeated* RANGE parameter (a list value the
    requests library serializes as RANGE=...&RANGE=...).

    Raises ValueError on an unsupported interval or an out-of-range WINDOW_SIZE so
    callers fail fast in tests; the client wraps these and degrades gracefully.
    """
    interval = str(interval or "").strip()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"unsupported INTERVAL {interval!r}; expected one of {VALID_INTERVALS}")

    syms = _normalize_symbols(symbols)
    if not syms:
        raise ValueError("SYMBOLS must contain at least one ticker")
    calcs = _normalize_calculations(calculations)
    if not calcs:
        raise ValueError("CALCULATIONS must contain at least one calculation")

    params: dict[str, Any] = {
        "function": function,
        "SYMBOLS": ",".join(syms),
        "INTERVAL": interval,
        "CALCULATIONS": ",".join(calcs),
    }

    # RANGE: single token, or repeated start/end pair.
    if isinstance(range_, str):
        params["RANGE"] = range_.strip()
    else:
        rng = [str(r).strip() for r in (range_ or []) if str(r).strip()]
        if not rng:
            raise ValueError("RANGE must be a token or a [start, end] pair")
        params["RANGE"] = rng if len(rng) > 1 else rng[0]

    if function == SLIDING_WINDOW_FUNCTION:
        if window_size is None:
            raise ValueError("WINDOW_SIZE is required for ANALYTICS_SLIDING_WINDOW")
        ws = int(window_size)
        if ws < MIN_WINDOW_SIZE:
            raise ValueError(f"WINDOW_SIZE must be >= {MIN_WINDOW_SIZE}, got {ws}")
        params["WINDOW_SIZE"] = str(ws)

    if ohlc:
        ohlc = str(ohlc).lower().strip()
        if ohlc in ("open", "high", "low", "close"):
            params["OHLC"] = ohlc
    return params


# =============================================================================
# Client
# =============================================================================

class AlphaVantageAnalyticsClient:
    """Advisory-safe Alpha Vantage Advanced Analytics client.

    Both public ``get_*_window`` methods always return a normalized result dict
    (never raise). On any failure they fall back to a local OHLCV computation
    (when ``local_series`` is provided) or to a neutral result, and log the
    reason.
    """

    def __init__(
        self,
        config: Optional[AnalyticsConfig] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        request_log_path: Optional[Path] = None,
        status_report_path: Optional[Path] = None,
        http_get: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config or AnalyticsConfig()
        self.api_key = api_key if api_key is not None else _api_key()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.request_log_path = Path(request_log_path) if request_log_path is not None else DEFAULT_REQUEST_LOG
        self.status_report_path = Path(status_report_path) if status_report_path is not None else DEFAULT_STATUS_REPORT
        self._http_get = http_get
        self._mem_cache: dict[str, tuple[float, Any]] = {}
        self._last_request_ts: float = 0.0
        self.api_call_count = 0
        self.fallback_count = 0
        for d in (self.cache_dir, self.request_log_path.parent, self.status_report_path.parent):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - non-fatal
                pass

    # ---- logging / status --------------------------------------------------

    def _log_request(
        self,
        params: dict[str, Any],
        status: str,
        latency_ms: float,
        rows: int,
        fallback_used: bool,
        error: str = "",
    ) -> None:
        """Append one request record to the §9 CSV log. API key is NEVER written."""
        safe = _redact(params)
        rng = safe.get("RANGE", "")
        if isinstance(rng, list):
            rng = "|".join(str(x) for x in rng)
        row = {
            "timestamp_utc": now_utc().isoformat(),
            "endpoint": safe.get("function", ""),
            "symbols": safe.get("SYMBOLS", ""),
            "range": rng,
            "interval": safe.get("INTERVAL", ""),
            "window_size": safe.get("WINDOW_SIZE", ""),
            "calculations": safe.get("CALCULATIONS", ""),
            "ohlc": safe.get("OHLC", ""),
            "status": status,
            "latency": round(latency_ms, 1),
            "rows_returned": rows,
            "fallback_used": int(bool(fallback_used)),
            "error_message": (error or "")[:240],
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
        status = {
            "generated_at_utc": now_utc().isoformat(),
            "enabled": self.config.enabled,
            "api_key_present": bool(self.api_key),
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

    def _do_http_get(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._http_get is not None:
            return self._http_get(AV_BASE_URL, params=params, timeout=self.config.timeout_seconds)
        if requests is None:
            raise RuntimeError("requests library unavailable")
        resp = requests.get(AV_BASE_URL, params=params, timeout=self.config.timeout_seconds)
        resp.raise_for_status()
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

    def _cache_key(self, params: dict[str, Any]) -> str:
        return json.dumps(_redact(params), sort_keys=True, default=str)

    def fetch(self, params: dict[str, Any], *, cache: bool = True) -> Optional[dict[str, Any]]:
        """Fetch a raw analytics payload with retry, throttle, cache, logging.

        Returns the parsed JSON dict, or None on hard failure (logged).
        """
        cache_key = self._cache_key(params)

        if cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._log_request(params, "cache_hit", 0.0, _payload_rows(cached), False)
                return cached

        if not self.config.enabled:
            self._log_request(params, "disabled", 0.0, 0, True, "analytics client disabled")
            self.fallback_count += 1
            return None
        if not self.api_key:
            self._log_request(params, "no_api_key", 0.0, 0, True, "ALPHA_VANTAGE_API_KEY unset")
            self.fallback_count += 1
            logger.debug("AV analytics %s skipped: no API key", params.get("function"))
            return None

        request_params = dict(params)
        request_params["apikey"] = self.api_key  # injected here only; stripped from every log line

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
                    self._log_request(params, "notice", latency_ms, 0, False, notice)
                    logger.warning("AV analytics %s notice: %s", params.get("function"), notice)
                    break  # throttle/info notices won't clear within a retry window
                if cache:
                    self._cache_put(cache_key, payload)
                self._log_request(params, "ok", latency_ms, _payload_rows(payload), False)
                return payload
            except Exception as exc:
                self._last_request_ts = time.time()
                latency_ms = (self._last_request_ts - t0) * 1000.0
                last_err = str(exc)
                self._log_request(params, "error", latency_ms, 0, False, last_err)
                logger.warning(
                    "AV analytics %s fetch failed (attempt %d/%d): %s",
                    params.get("function"), attempt + 1, self.config.max_retries + 1, exc,
                )
                if attempt < self.config.max_retries:
                    try:
                        time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
                    except Exception:
                        pass

        self.fallback_count += 1
        if last_err:
            logger.error("AV analytics %s unavailable -> fallback: %s", params.get("function"), last_err)
        return None

    # =========================================================================
    # Public endpoints
    # =========================================================================

    def get_fixed_window(
        self,
        symbols: Union[str, Sequence[str]],
        calculations: Union[str, Sequence[str]],
        range_: Union[str, Sequence[str]] = "60day",
        interval: str = "DAILY",
        ohlc: Optional[str] = None,
        local_series: Optional[dict[str, Sequence[float]]] = None,
    ) -> dict[str, Any]:
        """ANALYTICS_FIXED_WINDOW — current-period statistical features.

        Returns a normalized result dict (see ``normalize_analytics_result``).
        On API failure, computes equivalent metrics locally from ``local_series``
        (per-symbol close-price sequences) when provided, else returns neutral.
        """
        return self._run(
            FIXED_WINDOW_FUNCTION, symbols, calculations, range_, interval,
            window_size=None, ohlc=ohlc, local_series=local_series,
        )

    def get_sliding_window(
        self,
        symbols: Union[str, Sequence[str]],
        calculations: Union[str, Sequence[str]],
        range_: Union[str, Sequence[str]] = "3month",
        interval: str = "5min",
        window_size: int = 20,
        ohlc: Optional[str] = None,
        local_series: Optional[dict[str, Sequence[float]]] = None,
    ) -> dict[str, Any]:
        """ANALYTICS_SLIDING_WINDOW — rolling statistical features."""
        return self._run(
            SLIDING_WINDOW_FUNCTION, symbols, calculations, range_, interval,
            window_size=window_size, ohlc=ohlc, local_series=local_series,
        )

    # ---- shared driver -----------------------------------------------------

    def _run(
        self,
        function: str,
        symbols: Union[str, Sequence[str]],
        calculations: Union[str, Sequence[str]],
        range_: Union[str, Sequence[str]],
        interval: str,
        window_size: Optional[int],
        ohlc: Optional[str],
        local_series: Optional[dict[str, Sequence[float]]],
    ) -> dict[str, Any]:
        syms = _normalize_symbols(symbols)
        calcs = _normalize_calculations(calculations)
        if function == SLIDING_WINDOW_FUNCTION:
            kept = [c for c in calcs if _calc_base(c) not in SLIDING_UNSUPPORTED_CALCS]
            dropped = [c for c in calcs if _calc_base(c) in SLIDING_UNSUPPORTED_CALCS]
            if dropped:
                logger.info("AV analytics sliding window: dropping unsupported calcs %s", dropped)
                calcs = kept
                calculations = kept
        try:
            params = build_analytics_params(function, symbols, calculations, range_, interval, window_size, ohlc)
        except ValueError as exc:
            logger.warning("AV analytics bad request (%s): %s", function, exc)
            self._log_request(
                {"function": function, "SYMBOLS": ",".join(syms), "INTERVAL": interval,
                 "CALCULATIONS": ",".join(calcs)},
                "bad_request", 0.0, 0, True, str(exc),
            )
            return _local_or_neutral(function, syms, calcs, local_series)

        payload = self.fetch(params)
        if payload is None:
            return _local_or_neutral(function, syms, calcs, local_series)
        return normalize_analytics_result(payload, function, syms)


# =============================================================================
# Local-math fallback (pure functions; §10)
# =============================================================================

def _returns(prices: Sequence[float]) -> list[float]:
    out: list[float] = []
    prev: Optional[float] = None
    for p in prices:
        f = to_float(p)
        if f is None or f <= 0:
            prev = f if (f is not None and f > 0) else prev
            continue
        if prev is not None and prev > 0:
            out.append(f / prev - 1.0)
        prev = f
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _variance(xs: Sequence[float], annualized: bool = False) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var * TRADING_DAYS_PER_YEAR if annualized else var


def _stddev(xs: Sequence[float], annualized: bool = False) -> Optional[float]:
    var = _variance(xs, annualized=False)
    if var is None:
        return None
    sd = math.sqrt(max(var, 0.0))
    return sd * math.sqrt(TRADING_DAYS_PER_YEAR) if annualized else sd


def _cumulative_return(rets: Sequence[float]) -> Optional[float]:
    rets = [r for r in rets if r is not None]
    if not rets:
        return None
    acc = 1.0
    for r in rets:
        acc *= (1.0 + r)
    return acc - 1.0


def _max_drawdown(prices: Sequence[float]) -> Optional[float]:
    vals = [to_float(p) for p in prices if to_float(p) is not None]
    if len(vals) < 2:
        return None
    peak = vals[0]
    mdd = 0.0
    for v in vals:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def _autocorrelation(xs: Sequence[float], lag: int = 1) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n <= lag + 1:
        return None
    m = sum(xs) / n
    denom = sum((x - m) ** 2 for x in xs)
    if denom <= 0:
        return None
    num = sum((xs[i] - m) * (xs[i - lag] - m) for i in range(lag, n))
    return num / denom


def _correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = list(a[:n]), list(b[:n])
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def _covariance(a: Sequence[float], b: Sequence[float], annualized: bool = False) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = list(a[:n]), list(b[:n])
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    return cov * TRADING_DAYS_PER_YEAR if annualized else cov


def compute_local_analytics(
    symbols: Sequence[str],
    calculations: Sequence[str],
    series: dict[str, Sequence[float]],
) -> dict[str, Any]:
    """Compute fixed-window-equivalent analytics locally from close-price series.

    ``series`` maps SYMBOL -> sequence of close prices (oldest-first). Only the
    scalar + relationship calculations that can be derived from prices are
    produced; anything unsupported is simply omitted (callers treat missing as
    neutral). Never raises.
    """
    rets = {s: _returns(series.get(s, [])) for s in symbols}
    calc_bases = {_calc_base(c) for c in calculations}
    out: dict[str, Any] = {}

    def scalar(name: str, fn: Callable[[str], Optional[float]]) -> None:
        if name in calc_bases:
            vals = {s: fn(s) for s in symbols}
            out[name] = {s: v for s, v in vals.items() if v is not None}

    scalar("MEAN", lambda s: _mean(rets[s]))
    scalar("MEDIAN", lambda s: _median(rets[s]))
    scalar("STDDEV", lambda s: _stddev(rets[s]))
    scalar("VARIANCE", lambda s: _variance(rets[s]))
    scalar("CUMULATIVE_RETURN", lambda s: _cumulative_return(rets[s]))
    scalar("MAX_DRAWDOWN", lambda s: _max_drawdown(series.get(s, [])))
    scalar("MIN", lambda s: (min(rets[s]) if rets[s] else None))
    scalar("MAX", lambda s: (max(rets[s]) if rets[s] else None))
    scalar("AUTOCORRELATION", lambda s: _autocorrelation(rets[s], 1))

    if "CORRELATION" in calc_bases or "COVARIANCE" in calc_bases:
        idx = list(symbols)
        if "CORRELATION" in calc_bases:
            out["CORRELATION"] = {
                "index": idx,
                "matrix": [[_safe01(_correlation(rets[a], rets[b]), 1.0 if a == b else 0.0)
                            for b in idx] for a in idx],
            }
        if "COVARIANCE" in calc_bases:
            out["COVARIANCE"] = {
                "index": idx,
                "matrix": [[(_covariance(rets[a], rets[b]) or 0.0) for b in idx] for a in idx],
            }
    return out


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _safe01(v: Optional[float], default: float) -> float:
    return default if v is None else v


def _local_or_neutral(
    function: str,
    symbols: Sequence[str],
    calculations: Sequence[str],
    local_series: Optional[dict[str, Sequence[float]]],
) -> dict[str, Any]:
    """§10 fallback: local computation when price series are available, else neutral."""
    if local_series:
        try:
            calcs = compute_local_analytics(symbols, calculations, local_series)
            if calcs:
                return {
                    "endpoint": function,
                    "symbols": list(symbols),
                    "available": True,
                    "calculations": calcs,
                    "meta": {},
                    "fallback_used": True,
                    "source": "local_fallback",
                }
        except Exception as exc:  # pragma: no cover - fallback must never raise
            logger.warning("local analytics fallback failed: %s", exc)
    return neutral_analytics_result(function, symbols)


def neutral_analytics_result(function: str, symbols: Sequence[str]) -> dict[str, Any]:
    return {
        "endpoint": function,
        "symbols": list(symbols),
        "available": False,
        "calculations": {},
        "meta": {},
        "fallback_used": True,
        "source": "neutral",
    }


# =============================================================================
# Normalizers (pure functions; safe to unit-test in isolation)
# =============================================================================

def _payload_rows(payload: Any) -> int:
    node = _returns_calculations(payload)
    return len(node) if isinstance(node, dict) else 0


def _returns_calculations(payload: Any) -> dict[str, Any]:
    """Locate the RETURNS_CALCULATIONS block inside an analytics payload.

    Tolerant of the documented ``payload.RETURNS_CALCULATIONS`` nesting as well
    as a flattened top-level block.
    """
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("payload")
    if isinstance(inner, dict):
        rc = inner.get("RETURNS_CALCULATIONS")
        if isinstance(rc, dict):
            return rc
        return inner
    rc = payload.get("RETURNS_CALCULATIONS")
    if isinstance(rc, dict):
        return rc
    return {}


def parse_matrix_calc(node: Any) -> Optional[dict[str, Any]]:
    """Parse a CORRELATION / COVARIANCE relationship object into a uniform shape.

    Accepts the Alpha Vantage form ``{"index": [...], "correlation": [[...]]}`` (or
    "covariance"), and our own ``{"index": [...], "matrix": [[...]]}``. Returns
    ``{"index": [...symbols...], "matrix": [[float, ...], ...]}`` (a full N×N
    symmetric matrix) or None.

    Alpha Vantage returns a **lower-triangular** matrix (row i has i+1 entries,
    the last being the diagonal). We symmetrize into a full square so downstream
    lookups work regardless of symbol order; full square inputs pass through
    unchanged (mirroring is idempotent for symmetric matrices).
    """
    if not isinstance(node, dict):
        return None
    index = node.get("index") or node.get("symbols")
    matrix = None
    for k in ("matrix", "correlation", "covariance", "values"):
        if isinstance(node.get(k), list):
            matrix = node.get(k)
            break
    if not isinstance(index, list) or not isinstance(matrix, list):
        return None
    n = len(index)
    sym: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or i >= n:
            return None
        for j, val in enumerate(row):
            if j >= n:
                continue
            f = to_float(val)
            if f is None:
                continue
            sym[i][j] = f
            sym[j][i] = f  # mirror lower<->upper triangle
    return {"index": [str(s).upper() for s in index], "matrix": sym}


def _unwrap_running(node: Any) -> Any:
    """Unwrap the sliding-window ``{"RUNNING_<X>": {...}}`` envelope.

    Sliding-window responses nest each calculation under a ``RUNNING_<NAME>`` key
    (e.g. RUNNING_MEAN, RUNNING_CORRELATION), often alongside sibling metadata such
    as ``WINDOW_START``; fixed-window responses do not. Returns the inner
    ``RUNNING_<NAME>`` node when present, else ``node`` unchanged.
    """
    if isinstance(node, dict):
        for key, val in node.items():
            if isinstance(key, str) and key.upper().startswith("RUNNING_"):
                return val
    return node


def parse_running_matrix(node: Any) -> Optional[dict[str, Any]]:
    """Parse a sliding-window pairwise relationship (``{"A;B": {ts: val}}``).

    Builds a symmetric latest-value matrix plus a parallel ``slope_matrix`` (the
    per-pair rolling slope, last-minus-first over points), with ``index`` derived
    from the symbols seen across all pairs. Returns the uniform matrix shape or
    None when the node is not a pairwise running structure.
    """
    if not isinstance(node, dict) or not node:
        return None
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    symbols: list[str] = []
    for key, series in node.items():
        if not isinstance(key, str) or ";" not in key or not isinstance(series, dict):
            return None
        a, b = (p.strip().upper() for p in key.split(";", 1))
        for s in (a, b):
            if s not in symbols:
                symbols.append(s)
        ordered = [to_float(series[k]) for k in sorted(series.keys())]
        ordered = [v for v in ordered if v is not None]
        pairs[(a, b)] = {
            "latest": ordered[-1] if ordered else None,
            "slope": ((ordered[-1] - ordered[0]) / max(len(ordered) - 1, 1)) if len(ordered) >= 2 else 0.0,
        }
    if not pairs:
        return None
    symbols.sort()
    n = len(symbols)
    idx = {s: i for i, s in enumerate(symbols)}
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    slope_matrix = [[0.0] * n for _ in range(n)]
    for (a, b), v in pairs.items():
        i, j = idx[a], idx[b]
        latest = v["latest"] if v["latest"] is not None else 0.0
        matrix[i][j] = matrix[j][i] = latest
        slope_matrix[i][j] = slope_matrix[j][i] = v["slope"]
    return {"index": symbols, "matrix": matrix, "slope_matrix": slope_matrix}


def parse_scalar_calc(node: Any) -> dict[str, Any]:
    """Parse a {symbol: value} calculation block.

    Values may be scalars (fixed window) or per-symbol time series / arrays
    (sliding window). For series, returns a {"series": {...}, "latest": float,
    "slope": float, "mean": float} summary so downstream feature builders get a
    single normalized read while retaining the raw series.
    """
    out: dict[str, Any] = {}
    if not isinstance(node, dict):
        return out
    for sym, val in node.items():
        key = str(sym).upper()
        f = to_float(val)
        if f is not None:
            out[key] = f
        elif isinstance(val, list):
            out[key] = _summarize_series([to_float(x) for x in val])
        elif isinstance(val, dict):
            # time-indexed mapping -> ordered series by key
            ordered = [to_float(val[k]) for k in sorted(val.keys())]
            out[key] = _summarize_series(ordered)
    return out


def _summarize_series(raw: Sequence[Optional[float]]) -> dict[str, Any]:
    vals = [v for v in raw if v is not None]
    if not vals:
        return {"series": [], "latest": None, "mean": None, "slope": 0.0, "first": None}
    slope = (vals[-1] - vals[0]) / max(len(vals) - 1, 1)
    return {
        "series": vals,
        "latest": vals[-1],
        "first": vals[0],
        "mean": sum(vals) / len(vals),
        "slope": slope,
    }


def parse_histogram(node: Any) -> dict[str, Any]:
    """Parse a HISTOGRAM calculation into {symbol: {"bins": [...], "counts": [...]}}."""
    out: dict[str, Any] = {}
    if not isinstance(node, dict):
        return out
    for sym, val in node.items():
        if isinstance(val, dict):
            bins = val.get("bins") or val.get("bin_edges")
            counts = val.get("counts") or val.get("frequency")
            out[str(sym).upper()] = {
                "bins": [to_float(b) for b in bins] if isinstance(bins, list) else [],
                "counts": [to_float(c) for c in counts] if isinstance(counts, list) else [],
            }
    return out


def normalize_analytics_result(payload: dict[str, Any], function: str, symbols: Sequence[str]) -> dict[str, Any]:
    """Normalize a raw Advanced Analytics payload into a uniform result dict.

    Result shape::

        {
          "endpoint": "ANALYTICS_FIXED_WINDOW",
          "symbols": [...],
          "available": bool,
          "calculations": {
              "MEAN": {symbol: float, ...},
              "CORRELATION": {"index": [...], "matrix": [[...]]},
              "HISTOGRAM": {symbol: {"bins": [...], "counts": [...]}},
              ...
          },
          "meta": {...},
          "fallback_used": False,
          "source": "alpha_vantage_analytics",
        }
    """
    rc = _returns_calculations(payload)
    meta = payload.get("meta_data") if isinstance(payload, dict) else {}
    calculations: dict[str, Any] = {}
    for raw_name, node in (rc or {}).items():
        base = _calc_base(raw_name)
        node = _unwrap_running(node)  # sliding-window: strip the RUNNING_<X> envelope
        if base in _MATRIX_CALCS:
            # fixed-window matrix (has "index"), else sliding pairwise running matrix
            parsed = parse_matrix_calc(node) or parse_running_matrix(node)
            if parsed is not None:
                calculations[base] = parsed
        elif base == "HISTOGRAM":
            calculations[base] = parse_histogram(node)
        else:
            calculations[base] = parse_scalar_calc(node)
    return {
        "endpoint": function,
        "symbols": [str(s).upper() for s in symbols],
        "available": bool(calculations),
        "calculations": calculations,
        "meta": meta if isinstance(meta, dict) else {},
        "fallback_used": False,
        "source": "alpha_vantage_analytics",
    }


# =============================================================================
# Module-level singleton convenience
# =============================================================================

_DEFAULT_CLIENT: Optional[AlphaVantageAnalyticsClient] = None


def get_default_analytics_client() -> AlphaVantageAnalyticsClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = AlphaVantageAnalyticsClient()
    return _DEFAULT_CLIENT


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["AAPL", "MSFT", "SPY"]
    client = AlphaVantageAnalyticsClient()
    print(f"API key present: {bool(client.api_key)} | enabled: {client.config.enabled}")
    res = client.get_fixed_window(
        syms,
        ["MEAN", "STDDEV(annualized=True)", "CUMULATIVE_RETURN", "MAX_DRAWDOWN",
         "AUTOCORRELATION(lag=1)", "CORRELATION(method=PEARSON)"],
        range_="60day", interval="DAILY",
    )
    print(json.dumps(res, indent=2, default=str)[:2000])
    client.write_status_report()
