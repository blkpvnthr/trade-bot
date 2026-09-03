"""
data_sources/alpha_vantage_premarket.py — Alpha Vantage premarket RVOL adapter.
================================================================================

Computes premarket relative volume from Alpha Vantage extended-hours intraday bars, with a
HARD provider-consistency guarantee: the current-session volume (numerator) and the trailing
average (denominator) BOTH come from Alpha Vantage — they are never mixed with Alpaca/IEX.

Rules (safety-first; never fabricate):
  * Alpha Vantage intraday timestamps are US/Eastern — parsed AS Eastern, never UTC.
  * Premarket slice is 04:00–09:30 ET.
  * Requires ≥ MIN_SESSIONS (5) PRIOR premarket sessions; otherwise rvol_available=False.
  * Any missing data / throttle / malformed response / entitlement failure → rvol_available=False
    with a reason; the rvol value stays None. Never raises.

Return shape (always a dict; `premarket_rvol` is None unless rvol_available is True):
  {symbol, current_premarket_volume, avg_premarket_volume, premarket_rvol, sessions_used,
   historical_sessions, rvol_provider="alpha_vantage", rvol_available, rvol_reason}
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

logger = logging.getLogger(__name__)

PROVIDER = "alpha_vantage"
MIN_SESSIONS = 5
PM_START_MIN = 4 * 60          # 04:00 ET
PM_END_MIN = 9 * 60 + 30       # 09:30 ET
DEFAULT_WINDOW = 20            # trailing premarket sessions to average
# Time-of-day matching (default ON): compare today-up-to-now against prior sessions truncated to
# the SAME clock time, so mid-premarket RVOL isn't a partial numerator over a full-session avg.
TIME_MATCHED_DEFAULT = os.getenv("RVOL_TIME_MATCHED", "true").strip().lower() in {"1", "true", "yes", "on"}
# Settle guard: a freshly-opened session has little accumulated volume, so cumulative RVOL is
# noisy/spiky for the first few minutes — hold RVOL UNAVAILABLE until this many minutes have
# elapsed since the 04:00 open (rather than emit a churny value). 0 disables the guard.
RVOL_WARMUP_MIN = int(os.getenv("RVOL_WARMUP_MIN", "10"))
# Optional short trailing rolling window (minutes) instead of cumulative-from-open — steadier
# mid-session. 0 = cumulative (default).
RVOL_ROLLING_MIN = int(os.getenv("RVOL_ROLLING_MIN", "0"))


def _base_result(symbol: str) -> Dict[str, Any]:
    return {"symbol": str(symbol or "").upper(),
            "current_premarket_volume": None, "avg_premarket_volume": None,
            "premarket_rvol": None, "sessions_used": 0, "historical_sessions": 0,
            "rvol_provider": PROVIDER, "rvol_available": False, "rvol_reason": "",
            "rvol_mode": None, "as_of_et": None}


def _parse_et(ts: Any) -> Optional[datetime]:
    """Parse an Alpha Vantage intraday timestamp AS US/Eastern (its native tz). Naive strings
    are localised to ET; tz-aware inputs are converted to ET. Returns None on failure."""
    try:
        s = str(ts).strip().replace("T", " ")
        dt = datetime.fromisoformat(s) if any(c in s for c in "-/") else None
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_ET) if _ET else dt
        return dt.astimezone(_ET) if _ET else dt
    except Exception:
        return None


def _is_premarket(dt: datetime) -> bool:
    m = dt.hour * 60 + dt.minute
    return PM_START_MIN <= m < PM_END_MIN


def _premarket_volume_by_session(bars: List[Dict[str, Any]]) -> Dict[Any, float]:
    """Sum premarket (04:00–09:30 ET) volume per ET session date."""
    by_day: Dict[Any, float] = {}
    for b in bars or []:
        ts = b.get("timestamp") if isinstance(b, dict) else None
        vol = b.get("volume") if isinstance(b, dict) else None
        if ts is None or vol is None:
            continue
        dt = _parse_et(ts)
        if dt is None or not _is_premarket(dt):
            continue
        try:
            by_day[dt.date()] = by_day.get(dt.date(), 0.0) + float(vol)
        except Exception:
            continue
    return by_day


def _premarket_bars_by_session(bars: List[Dict[str, Any]]) -> Dict[Any, List]:
    """Premarket bars per ET session date as (minute_of_day, volume) pairs — enables a
    time-of-day cutoff so today-to-now is compared against prior sessions to the same time."""
    by_day: Dict[Any, List] = {}
    for b in bars or []:
        ts = b.get("timestamp") if isinstance(b, dict) else None
        vol = b.get("volume") if isinstance(b, dict) else None
        if ts is None or vol is None:
            continue
        dt = _parse_et(ts)
        if dt is None or not _is_premarket(dt):
            continue
        try:
            by_day.setdefault(dt.date(), []).append((dt.hour * 60 + dt.minute, float(vol)))
        except Exception:
            continue
    return by_day


def _session_volume(bar_list: List, cutoff_min: Optional[int], start_min: Optional[int] = None) -> float:
    """Sum a session's premarket volume in (start_min, cutoff_min]. start_min=None → from the
    04:00 open (cumulative); a non-None start_min gives a trailing rolling window."""
    return sum(v for (m, v) in bar_list
               if (cutoff_min is None or m <= cutoff_min)
               and (start_min is None or m > start_min))


def compute_premarket_rvol(symbol: str, client: Any = None, interval: str = "5min",
                           min_sessions: int = MIN_SESSIONS, window: int = DEFAULT_WINDOW,
                           bars: Optional[List[Dict[str, Any]]] = None,
                           time_matched: Optional[bool] = None) -> Dict[str, Any]:
    """Compute AV premarket RVOL for `symbol`. `bars` may be injected (tests); otherwise the
    Alpha Vantage client fetches extended-hours full-history intraday.

    time_matched (default RVOL_TIME_MATCHED, on): compare today's premarket-to-now against each
    prior session truncated to the SAME clock time (apples-to-apples mid-premarket). When off,
    uses the whole 04:00–09:30 window on both sides. Never raises."""
    if time_matched is None:
        time_matched = TIME_MATCHED_DEFAULT
    result = _base_result(symbol)
    result["rvol_mode"] = "time_matched" if time_matched else "full_session"
    sym = result["symbol"]
    if not sym:
        result["rvol_reason"] = "no_symbol"
        return result
    try:
        if bars is None:
            if client is None:
                from data_sources.alpha_vantage_client import AlphaVantageClient
                client = AlphaVantageClient()
            bars = client.get_intraday_extended(sym, interval=interval,
                                                outputsize="full", extended_hours=True)
        if not bars:
            result["rvol_reason"] = "no_data_or_entitlement_unavailable"
            return result

        sess = _premarket_bars_by_session(bars)     # date → [(minute_of_day, volume)]
        if not sess:
            result["rvol_reason"] = "no_premarket_bars"
            return result

        days = sorted(sess)                          # ascending ET session dates
        current_day = days[-1]
        # Time-of-day cutoff = latest premarket bar minute observed TODAY → compare prior
        # sessions only up to that same clock time (None = full 04:00–09:30 window).
        cutoff = max(m for m, _ in sess[current_day]) if time_matched else None
        rolling = RVOL_ROLLING_MIN if (time_matched and RVOL_ROLLING_MIN > 0) else 0
        lo = (cutoff - rolling) if (time_matched and rolling > 0 and cutoff is not None) else None
        if cutoff is not None:
            result["as_of_et"] = f"{cutoff // 60:02d}:{cutoff % 60:02d}"
            result["rvol_mode"] = f"time_matched_rolling{rolling}m" if rolling else "time_matched"

        # Settle guard: don't emit a noisy RVOL in the first minutes after the 04:00 open.
        if time_matched and cutoff is not None:
            elapsed = cutoff - PM_START_MIN
            need = max(RVOL_WARMUP_MIN, rolling)
            if elapsed < need:
                result["current_premarket_volume"] = round(_session_volume(sess[current_day], cutoff, lo), 2)
                result["rvol_reason"] = f"warmup_settling(elapsed={elapsed}m<{need}m)"
                return result   # rvol_available stays False — settle before ranking on it

        current = _session_volume(sess[current_day], cutoff, lo)
        result["current_premarket_volume"] = round(current, 2)

        prior = [d for d in days if d < current_day]
        result["historical_sessions"] = len(prior)
        if len(prior) < int(min_sessions):
            result["rvol_reason"] = f"insufficient_premarket_sessions({len(prior)}<{int(min_sessions)})"
            return result

        used = prior[-int(window):] if window else prior
        prior_vols = [_session_volume(sess[d], cutoff, lo) for d in used]
        avg = (sum(prior_vols) / len(prior_vols)) if prior_vols else 0.0
        if avg <= 0:
            # e.g. time-matched cutoff is so early that prior sessions had no bars yet
            result["rvol_reason"] = "zero_average_volume_at_cutoff" if time_matched else "zero_average_volume"
            return result

        # NOTE: current (numerator) and avg (denominator) both derive from `sess`, i.e. the SAME
        # Alpha Vantage extended-hours feed, and both use the SAME cutoff — provider AND
        # time-of-day consistency are guaranteed here.
        result["avg_premarket_volume"] = round(avg, 2)
        result["sessions_used"] = len(used)
        result["premarket_rvol"] = round(current / avg, 4)
        result["rvol_available"] = True
        result["rvol_reason"] = "ok"
        return result
    except Exception as exc:                        # throttle / malformed / network → fail safe
        logger.debug("AV premarket RVOL failed for %s: %s", sym, exc)
        result["rvol_reason"] = f"error:{type(exc).__name__}"
        result["rvol_available"] = False
        result["premarket_rvol"] = None
        return result


def compute_many(symbols: List[str], client: Any = None, interval: str = "5min",
                 min_sessions: int = MIN_SESSIONS, window: int = DEFAULT_WINDOW,
                 max_symbols: int = 25, time_matched: Optional[bool] = None) -> Dict[str, Dict[str, Any]]:
    """Compute AV premarket RVOL for several symbols (shared client; capped). Fail-safe."""
    out: Dict[str, Dict[str, Any]] = {}
    syms = [str(s).upper() for s in (symbols or []) if s][:max_symbols]
    if syms and client is None:
        try:
            from data_sources.alpha_vantage_client import AlphaVantageClient
            client = AlphaVantageClient()
        except Exception:
            client = None
    for s in syms:
        out[s] = compute_premarket_rvol(s, client=client, interval=interval,
                                        min_sessions=min_sessions, window=window,
                                        time_matched=time_matched)
    return out
