#!/usr/bin/env python3
"""
alpaca_feed_manager.py — Alpaca live-feed detection + overnight metrics (read-only).
====================================================================================

Phase 4 data-quality layer. Detects which Alpaca market-data feeds are available under
the current subscription and gracefully downgrades, in priority order:

    Premarket / overnight :  overnight (v1beta1/overnight) → iex → (yliveticker fallback)
    Regular trading hours :  iex (v2/iex)
    Optional / advisory   :  boats (if available)

FEED POLICY: delayed-SIP is never used. Real-time SIP is OFF by default and is enabled only
when entitled, via ALPACA_DATA_FEED=sip (or ALPACA_SIP_ENABLED=true) — e.g. an Alpaca Algo
Trader Plus subscription. With the flag off, behavior is IEX/overnight-only exactly as before
(`v2/sip` / `v2/delayed_sip` excluded by construction). This module is READ-ONLY market data —
it never imports AlpacaGateway / alpaca.trading, never submits an order, and never
changes execution authority. All probes are bounded + guarded; on missing creds or any
error it downgrades safely (everything "unavailable") and never raises into the loop.

Outputs: daily_system_outputs/YYYYMMDD/alpaca_feed_diagnostic.json
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import rl_common as _rc  # reused only for atomic write_json + data_tag (no RL behavior)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

# Real-time SIP is opt-in and entitlement-gated (Algo Trader Plus). delayed_sip stays off.
SIP_ENABLED = (
    os.getenv("ALPACA_DATA_FEED", "iex").strip().lower() == "sip"
    or os.getenv("ALPACA_SIP_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
)
if SIP_ENABLED:
    # SIP entitled: prefer the real-time consolidated feed, keep IEX as the downgrade.
    SUPPORTED_FEEDS = ["overnight", "sip", "iex", "boats"]
    FORBIDDEN_FEEDS = ["delayed_sip", "v2/delayed_sip"]
    PREMARKET_PRIORITY = ["overnight", "sip", "iex"]
    RTH_PRIORITY = ["sip", "iex"]
else:
    # Default / unentitled: SIP and delayed_sip are intentionally absent.
    SUPPORTED_FEEDS = ["overnight", "iex", "boats"]
    FORBIDDEN_FEEDS = ["sip", "delayed_sip", "v2/sip", "v2/delayed_sip"]
    PREMARKET_PRIORITY = ["overnight", "iex"]
    RTH_PRIORITY = ["iex"]
PROBE_TIMEOUT_SEC = float(os.getenv("ALPACA_PROBE_TIMEOUT_SEC", "20"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(x: Any, d: float = 0.0) -> float:
    try:
        if x is None:
            return d
        v = float(x)
        return d if v != v else v
    except Exception:
        return d


def _creds():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass
    key = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
           or os.environ.get("ALPACA_API_KEY_ID") or "")
    sec = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
           or os.environ.get("ALPACA_API_SECRET_KEY") or "")
    return key, sec


# ── overnight metrics (pure; testable without network) ──────────────────────────────
def compute_overnight_metrics(df: Any, prev_close: Optional[float] = None,
                              avg_overnight_volume: Optional[float] = None) -> Dict[str, Any]:
    """Overnight-session metrics from a bars DataFrame. Pure; never raises."""
    try:
        if df is None or not hasattr(df, "__len__") or len(df) == 0:
            return {"overnight_available": False}
        highs = [float(x) for x in df["high"]]
        lows = [float(x) for x in df["low"]]
        closes = [float(x) for x in df["close"]]
        opens = [float(x) for x in df["open"]]
        vols = [float(x) for x in df["volume"]]
        vol = sum(vols)
        num = sum(((highs[i] + lows[i] + closes[i]) / 3.0) * vols[i] for i in range(len(closes)))
        vwap = (num / vol) if vol > 0 else closes[-1]
        hi, lo = max(highs), min(lows)
        first, last = opens[0], closes[-1]
        range_pct = (hi - lo) / first if first else 0.0
        if last > first * 1.001:
            trend = "up"
        elif last < first * 0.999:
            trend = "down"
        else:
            trend = "flat"
        rvol = (vol / avg_overnight_volume) if (avg_overnight_volume and avg_overnight_volume > 0) else None
        # session strength: signed magnitude of the overnight move, 0..1
        move = (last - first) / first if first else 0.0
        session_strength = max(-1.0, min(1.0, move * 20.0))
        gap = ((first - prev_close) / prev_close) if (prev_close and prev_close > 0) else None
        return {
            "overnight_available": True,
            "overnight_volume": round(vol, 2),
            "overnight_vwap": round(vwap, 4),
            "overnight_range_pct": round(range_pct, 6),
            "overnight_rvol": (round(rvol, 4) if rvol is not None else None),
            "overnight_trend": trend,
            "overnight_session_strength": round(session_strength, 4),
            "overnight_gap_pct": (round(gap * 100.0, 4) if gap is not None else None),
            "overnight_bars": len(closes),
        }
    except Exception as exc:
        logger.debug("compute_overnight_metrics failed: %s", exc)
        return {"overnight_available": False, "error": str(exc)[:120]}


class AlpacaFeedManager:
    """Detects available Alpaca feeds (read-only) and exposes the active feed per session."""

    def __init__(self):
        self.detected: Dict[str, Dict[str, Any]] = {}
        self.failover_count = 0
        self.reconnect_count = 0
        self.last_message_at: Optional[str] = None
        self._detected_once = False

    # ── read-only historical fetch (shared by detection + overnight metrics) ──────
    def fetch_bars(self, symbols: List[str], start, end, timeframe: str = "day",
                   feed: str = "iex") -> Dict[str, Any]:
        if feed in FORBIDDEN_FEEDS:
            logger.warning("refusing forbidden feed %s", feed)
            return {}
        key, sec = _creds()
        if not key or not sec:
            return {}
        try:
            import socket
            socket.setdefaulttimeout(PROBE_TIMEOUT_SEC)
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed
            client = StockHistoricalDataClient(key, sec)
            tf = TimeFrame.Day if timeframe == "day" else TimeFrame(int(timeframe.replace("min", "") or 5),
                                                                    TimeFrameUnit.Minute)
            # SIP when entitled + requested; otherwise IEX. Overnight metrics derive from
            # extended-hours bars on whichever real-time feed is active.
            dfeed = DataFeed.SIP if (feed == "sip" and SIP_ENABLED) else DataFeed.IEX
            req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf,
                                   start=start, end=end, feed=dfeed)
            df = client.get_stock_bars(req).df
            out: Dict[str, Any] = {}
            if df is None or df.empty:
                return out
            df = df.reset_index()
            for sym, grp in df.groupby("symbol"):
                out[str(sym)] = grp.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
            return out
        except Exception as exc:
            logger.warning("Alpaca read-only fetch failed (feed=%s): %s", feed, exc)
            return {}

    # ── feed detection (bounded, read-only, cached) ───────────────────────────────
    def detect_feeds(self, force: bool = False, probe_symbol: str = "AAPL") -> Dict[str, Dict[str, Any]]:
        if self._detected_once and not force:
            return self.detected
        key, sec = _creds()
        auth = bool(key and sec)
        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        for feed in SUPPORTED_FEEDS:
            entry = {"feed_type": feed, "feed_source": "alpaca",
                     "feed_available": False, "feed_auth_success": auth,
                     "feed_downgrade_reason": None, "feed_latency_ms": None,
                     "feed_last_message": None}
            if not auth:
                entry["feed_downgrade_reason"] = "no_credentials"
                self.detected[feed] = entry
                continue
            if feed == "boats":
                # BOATS is optional; not part of the standard subscription — advisory only.
                entry["feed_downgrade_reason"] = "not_in_subscription_or_advisory_only"
                self.detected[feed] = entry
                continue
            # Probe iex (and overnight, which derives from extended-hours IEX bars).
            t0 = time.time()
            if feed == "overnight":
                start = end - timedelta(days=5)
            else:
                start = end - timedelta(days=5)
            probe_feed = "sip" if (feed == "sip" and SIP_ENABLED) else "iex"
            bars = self.fetch_bars([probe_symbol], start, end, timeframe="day", feed=probe_feed)
            latency = round((time.time() - t0) * 1000.0, 1)
            ok = bool(bars)
            entry.update({"feed_available": ok, "feed_latency_ms": latency,
                          "feed_last_message": _now_iso() if ok else None})
            if not ok:
                entry["feed_downgrade_reason"] = "probe_returned_no_data"
            self.detected[feed] = entry
        self._detected_once = True
        self.last_message_at = _now_iso()
        return self.detected

    def active_feed(self, session: str = "rth") -> str:
        det = self.detect_feeds()
        priority = PREMARKET_PRIORITY if session in ("premarket", "overnight") else RTH_PRIORITY
        for feed in priority:
            if det.get(feed, {}).get("feed_available"):
                return feed
        # graceful downgrade: nothing available → caller uses yfinance/cached fallback
        self.failover_count += 1
        return "fallback"

    # ── overnight metrics (read-only; guarded) ─────────────────────────────────────
    def fetch_overnight_metrics(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compute overnight metrics per symbol from extended-hours IEX bars (best-effort)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=2)
        bars = self.fetch_bars(symbols, start, end, timeframe="5min",
                               feed=("sip" if SIP_ENABLED else "iex"))
        out: Dict[str, Dict[str, Any]] = {}
        for sym, df in bars.items():
            try:
                # overnight window = bars outside 13:30-20:00 UTC (≈ RTH); keep extended hours
                idx = df.index
                et_hours = [(d.tz_convert("America/New_York") if d.tzinfo else d).hour for d in idx]
                mins = [(d.tz_convert("America/New_York") if d.tzinfo else d) for d in idx]
                mask = []
                for d in idx:
                    dd = d.tz_convert("America/New_York") if d.tzinfo else d
                    m = dd.hour * 60 + dd.minute
                    mask.append(not (9 * 60 + 30 <= m < 16 * 60))   # exclude RTH
                on = df[mask]
                out[sym] = compute_overnight_metrics(on)
            except Exception:
                out[sym] = compute_overnight_metrics(df)
        return out

    # ── diagnostics ─────────────────────────────────────────────────────────────────
    def diagnostic(self, session: str = "rth") -> Dict[str, Any]:
        det = self.detect_feeds()
        active = self.active_feed(session)
        return {
            "generated_at_utc": _now_iso(),
            "date_tag": _rc.data_tag(),
            "session": session,
            "active_feed": active,
            "feeds": det,
            "feed_failover_count": self.failover_count,
            "feed_reconnect_count": self.reconnect_count,
            "feed_last_message": self.last_message_at,
            "sip_used": active == "sip", "delayed_sip_used": False,
            "supported_feeds": SUPPORTED_FEEDS,
        }

    def write_diagnostic(self, out_dir: Path = None, session: str = "rth") -> Optional[Path]:
        target = Path(out_dir) if out_dir else (DAILY_OUTPUTS / _rc.data_tag())
        try:
            p = target / "alpaca_feed_diagnostic.json"
            _rc.write_json(p, self.diagnostic(session))
            return p
        except Exception as exc:
            logger.debug("write_diagnostic failed: %s", exc)
            return None


_MANAGER: Optional[AlpacaFeedManager] = None


def get_manager() -> AlpacaFeedManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AlpacaFeedManager()
    return _MANAGER


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = get_manager()
    diag = m.diagnostic("rth")
    m.write_diagnostic()
    print(f"Alpaca feed detection (read-only, sip_enabled={SIP_ENABLED}):")
    for feed, e in diag["feeds"].items():
        print(f"  {feed}: available={e['feed_available']} auth={e['feed_auth_success']} "
              f"latency_ms={e['feed_latency_ms']} reason={e['feed_downgrade_reason']}")
    print("  active(rth):", m.active_feed("rth"), "| active(premarket):", m.active_feed("premarket"))
