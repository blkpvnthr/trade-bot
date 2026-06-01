"""
premarket_live_feed.py — Live Premarket Quote Awareness (yliveticker)
=====================================================================

Maintains real-time premarket quote state for candidate symbols using the
`yliveticker` Yahoo Finance live websocket stream, replacing delayed/static
yfinance polling for PREMARKET quote awareness only.

Scope (replaces yfinance ONLY for):
  * premarket live price collection
  * premarket quote snapshots
  * premarket volume / change monitoring
  * premarket high / low tracking

Does NOT replace: historical OHLC backfills, backtesting data, daily bars,
fundamentals, SEC data, or Alpaca execution data.

This is awareness infrastructure, NOT a strategy:
  - Premarket data NEVER directly creates an order.
  - Alpha decides tactical trades; Buffett decides long-term ownership.
  - Research Lab measures whether premarket signals improve expectancy.
  - RiskManager decides whether orders are permitted.

Safety / fallback:
  * Optional dependency `yliveticker`. If unavailable, the feed marks
    feed_status="fallback" and run.py uses the existing yfinance/Alpaca
    premarket logic — it NEVER stops run.py, Alpha or Buffett.
  * The blocking YLiveTicker websocket runs in a daemon thread.
  * Stale quotes (older than max_stale_seconds) are flagged so callers fall
    back to the latest Alpaca/yfinance price rather than trust them.

yliveticker API note: ``YLiveTicker(on_ticker, ticker_names, reconnect=5,
ping_interval=15, ping_timeout=10)`` BLOCKS (internal reconnect) and the only
callback is ``on_ticker(ws, msg)`` (no on_open/on_error/on_close). ``ws`` is
captured from the first tick so the stream can be closed on stop().
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Shared atomic JSON writer (single implementation in opportunity_utils); guarded so
# this module stays import-safe standalone. Falls back to an atomic tmp+rename write.
try:
    from opportunity_utils import write_json_safe as _write_json_safe
except Exception:
    def _write_json_safe(path, data, log_fn=None) -> bool:
        import json as _json, os as _os
        from pathlib import Path as _P
        p = _P(path); tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(_json.dumps(data, indent=2, default=str), encoding="utf-8")
        _os.replace(tmp, p)
        return True


PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_LIVE = "yliveticker"
SOURCE_FALLBACK = "yfinance"

# Per-symbol quote fields exposed in every snapshot entry.
QUOTE_FIELDS: List[str] = [
    "symbol", "price", "previous_close", "premarket_change", "premarket_change_pct",
    "premarket_volume", "premarket_high", "premarket_low", "bid", "ask",
    "bid_ask_spread", "updated_at", "source", "stale", "error",
    "premarket_rvol", "avg_premarket_volume_20d", "rvol_source", "rvol_available",
    # Alpaca overnight-session metrics (advisory: ranking / market-state / research only).
    "overnight_volume", "overnight_vwap", "overnight_range_pct", "overnight_rvol",
    "overnight_trend", "overnight_session_strength",
]

# Optional yliveticker package — probed once at import (never fatal).
try:  # pragma: no cover - exercised via YLIVETICKER_AVAILABLE in tests
    import yliveticker as _ylt  # type: ignore
    YLIVETICKER_AVAILABLE = True
except Exception:
    _ylt = None
    YLIVETICKER_AVAILABLE = False
    logger.warning(
        "yliveticker package unavailable — premarket live feed will fall back to "
        "yfinance/Alpaca premarket logic (install `yliveticker` to enable streaming)."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _f(x: Any) -> Optional[float]:
    """Coerce to float; None on failure or NaN. yliveticker sends sparse fields."""
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _i(x: Any) -> Optional[int]:
    """Coerce to int (yliveticker int64 fields arrive as strings)."""
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def compute_premarket_rank_score(quote: Dict[str, Any],
                                 market_state: Optional[dict] = None,
                                 sector_strength: Optional[float] = None) -> float:
    """
    Advisory premarket ranking score for Alpha candidate prioritization ONLY:

        premarket_rank_score = gap_score + premarket_rvol_score
                             + sector_strength_score + risk_on_score

    It MAY improve/reorder ranking; it may NEVER create a trade by itself (the caller
    still routes every entry through setup quality, liquidity, and RiskManager). A stale
    quote contributes 0. Returns 0.0 on missing data — never raises.
    """
    try:
        q = quote or {}
        if q.get("stale"):
            return 0.0
        # gap component (0-40): reward magnitude of the premarket gap
        gap = abs(_f(q.get("premarket_change_pct")) or 0.0)
        gap_score = min(40.0, gap * 8.0)
        # premarket RVOL component (0-35): only when RVOL is genuinely available
        rvol_score = 0.0
        if q.get("rvol_available") and _f(q.get("premarket_rvol")) is not None:
            rvol = _f(q.get("premarket_rvol")) or 0.0
            rvol_score = min(35.0, max(0.0, (rvol - 1.0) * 17.5))   # 1x→0, 3x→35
        # sector strength component (0-15)
        ss = _f(sector_strength)
        sector_score = min(15.0, max(0.0, ss * 15.0)) if ss is not None else 0.0
        # risk-on regime component (0-10)
        risk_on_score = 0.0
        ms = market_state or {}
        if ms.get("risk_on") or ms.get("premarket_risk_state") == "risk_on":
            risk_on_score = 10.0
        elif ms.get("risk_off") or ms.get("premarket_risk_state") == "risk_off":
            risk_on_score = -10.0
        return round(gap_score + rvol_score + sector_score + risk_on_score, 3)
    except Exception:
        return 0.0


class PremarketLiveFeed:
    """Real-time premarket quote state for candidate symbols.

    Thread-safety: on_message runs in the websocket thread; all state mutations
    and reads are guarded by an internal lock.
    """

    def __init__(self, symbols: List[str], output_dir: Any = None,
                 max_stale_seconds: float = 30.0):
        seen: set = set()
        self.symbols: List[str] = []
        for s in (symbols or []):
            su = str(s or "").upper().strip()
            if su and su not in seen:
                seen.add(su)
                self.symbols.append(su)
        self.output_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "daily_system_outputs")
        self.max_stale_seconds = float(max_stale_seconds)

        self.state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.shutdown_requested: bool = False   # checked in the stream loop + parser
        self._ws: Any = None
        self._reconnect_delay = 5.0

        # Health counters
        self.feed_status: str = "init"   # init|running|fallback|failed|stopped|no_symbols
        self.symbols_updated: set = set()
        self.last_message_at: Optional[str] = None
        self.reconnect_count: int = 0
        self.error_count: int = 0
        self.error: Optional[str] = None
        # Fallback is a CLEARABLE state, not a one-way latch: it activates when the live
        # stream is unavailable/stale and CLEARS as soon as a fresh live tick arrives.
        self.fallback_active: bool = False
        self.fallback_reason: Optional[str] = None
        self.fallback_last_used: Optional[str] = None
        self.fallback_source: Optional[str] = None

    @property
    def fallback_used(self) -> bool:
        """Back-compat alias — True only while fallback is CURRENTLY active."""
        return self.fallback_active

    # ── state helpers ────────────────────────────────────────────────────────
    def _new_state(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol, "price": None, "previous_close": None,
            "premarket_change": None, "premarket_change_pct": None,
            "premarket_volume": None, "premarket_high": None, "premarket_low": None,
            "bid": None, "ask": None, "bid_ask_spread": None,
            "updated_at": None, "source": SOURCE_LIVE, "stale": False, "error": None,
            "market_hours": None, "_updated_ts": 0.0,
            # Premarket relative volume (computed when 20d history is supplied).
            "premarket_rvol": None, "avg_premarket_volume_20d": None,
            "rvol_source": None, "rvol_available": False,
            # Alpaca overnight-session metrics (advisory only; never creates a trade).
            "overnight_volume": None, "overnight_vwap": None, "overnight_range_pct": None,
            "overnight_rvol": None, "overnight_trend": None, "overnight_session_strength": None,
        }

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the live stream in a daemon thread. Fail-safe — never raises."""
        if not self.symbols:
            self.feed_status = "no_symbols"
            logger.debug("PremarketLiveFeed: no symbols to subscribe")
            return
        if not YLIVETICKER_AVAILABLE:
            self._mark_fallback("yliveticker_unavailable")
            return
        try:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_stream, name="premarket-live-feed",
                                            daemon=True)
            self._thread.start()
            self.feed_status = "running"
            logger.info("PremarketLiveFeed started: %d symbols", len(self.symbols))
        except Exception as e:
            self.error = str(e)[:300]
            self.error_count += 1
            self._mark_fallback(f"start_failed:{e}")

    def stop(self) -> None:
        """Stop the stream and join the thread (best-effort). Does NOT rely solely on
        closing the websocket: a shutdown flag freezes the parser and breaks the reconnect
        loop even if no tick ever arrived (so self._ws was never captured)."""
        self.shutdown_requested = True
        self._stop_event.set()
        try:
            if self._ws is not None:
                self._ws.close()   # best-effort; only available once a tick arrived
        except Exception:
            pass
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=3.0)
        except Exception:
            pass
        if self.feed_status in ("running", "init"):
            self.feed_status = "stopped"

    def _mark_fallback(self, reason: str) -> None:
        self.feed_status = "fallback"
        self.fallback_active = True
        self.fallback_reason = reason
        self.fallback_last_used = _now_iso()
        self.fallback_source = SOURCE_FALLBACK
        logger.warning("PremarketLiveFeed falling back to %s (%s)", SOURCE_FALLBACK, reason)

    def _run_stream(self) -> None:
        """Blocking websocket loop. yliveticker reconnects internally. Daemon thread."""
        while not self._stop_event.is_set() and not self.shutdown_requested:
            try:
                # Real yliveticker API: YLiveTicker(on_ticker, ticker_names, reconnect,
                # ping_interval, ping_timeout). It BLOCKS with internal reconnect; ws is
                # captured from the first tick. on_message is frozen once shutdown is set.
                _ylt.YLiveTicker(
                    on_ticker=self._on_ticker, ticker_names=list(self.symbols),
                    reconnect=int(self._reconnect_delay),
                )
            except Exception as e:
                self.error = str(e)[:300]
                self.error_count += 1
                self.feed_status = "failed"
                logger.warning("PremarketLiveFeed stream error: %s", e)
            if self._stop_event.is_set() or self.shutdown_requested:
                break
            # YLiveTicker returned (socket closed) — back off, then reconnect.
            self.reconnect_count += 1
            self._stop_event.wait(self._reconnect_delay)

    def _on_ticker(self, ws: Any, msg: Dict[str, Any]) -> None:
        # Capture ws so stop() can close it; if shutting down, close immediately and drop.
        self._ws = ws
        if self._stop_event.is_set() or self.shutdown_requested:
            try:
                ws.close()
            except Exception:
                pass
            return
        if self.feed_status not in ("running",):
            self.feed_status = "running"
        self.on_message(msg)

    # ── message parsing ──────────────────────────────────────────────────────
    def on_message(self, message: Dict[str, Any]) -> None:
        """Parse one yliveticker tick dict and update the symbol's quote state."""
        try:
            if self._stop_event.is_set() or self.shutdown_requested:
                return   # frozen after stop() — never mutate state post-shutdown
            if not isinstance(message, dict):
                return
            sym = (message.get("id") or message.get("symbol") or "").upper()
            if not sym:
                return
            with self._lock:
                st = self.state.get(sym) or self._new_state(sym)

                price = _f(message.get("price"))
                if price is not None:
                    st["price"] = price
                    st["premarket_high"] = price if st["premarket_high"] is None \
                        else max(st["premarket_high"], price)
                    st["premarket_low"] = price if st["premarket_low"] is None \
                        else min(st["premarket_low"], price)

                # Incorporate any exchange-reported session high/low too.
                dh = _f(message.get("dayHigh"))
                if dh is not None:
                    st["premarket_high"] = dh if st["premarket_high"] is None \
                        else max(st["premarket_high"], dh)
                dl = _f(message.get("dayLow"))
                if dl is not None:
                    st["premarket_low"] = dl if st["premarket_low"] is None \
                        else min(st["premarket_low"], dl)

                pc = _f(message.get("previousClose"))
                if pc is not None:
                    st["previous_close"] = pc
                vol = _i(message.get("dayVolume"))
                if vol is not None:
                    st["premarket_volume"] = vol

                # Prefer change derived from price vs previous_close; else the tick's
                # own change/changePercent fields.
                if st["price"] is not None and st["previous_close"]:
                    st["premarket_change"] = round(st["price"] - st["previous_close"], 6)
                    st["premarket_change_pct"] = round(
                        st["premarket_change"] / st["previous_close"] * 100.0, 6)
                else:
                    ch = _f(message.get("change"))
                    if ch is not None:
                        st["premarket_change"] = ch
                    chp = _f(message.get("changePercent"))
                    if chp is not None:
                        st["premarket_change_pct"] = chp

                bid = _f(message.get("bid"))
                if bid is not None:
                    st["bid"] = bid
                ask = _f(message.get("ask"))
                if ask is not None:
                    st["ask"] = ask
                if st["bid"] is not None and st["ask"] is not None:
                    st["bid_ask_spread"] = round(st["ask"] - st["bid"], 6)

                st["market_hours"] = message.get("marketHours")
                st["updated_at"] = _now_iso()
                st["_updated_ts"] = _now_ts()
                st["source"] = SOURCE_LIVE
                st["stale"] = False
                st["error"] = None

                self.state[sym] = st
                self.symbols_updated.add(sym)
                self.last_message_at = st["updated_at"]
                # A fresh LIVE tick clears fallback — live data is authoritative again.
                self.fallback_active = False
        except Exception as e:
            self.error = str(e)[:300]
            self.error_count += 1
            logger.debug("PremarketLiveFeed.on_message failed: %s", e)

    def apply_fallback_quotes(self, quotes_by_symbol: Dict[str, Dict[str, Any]],
                              source: str = SOURCE_FALLBACK) -> None:
        """
        Inject premarket quotes obtained from the existing yfinance/Alpaca logic
        (run.py owns that fetch) so the snapshot stays the single source of truth
        when the live stream is unavailable or a symbol is stale. Fail-safe.
        """
        if not quotes_by_symbol:
            return
        with self._lock:
            for sym, q in quotes_by_symbol.items():
                symu = str(sym or "").upper()
                if not symu or not isinstance(q, dict):
                    continue
                st = self.state.get(symu) or self._new_state(symu)
                price = _f(q.get("price") or q.get("premarket_price"))
                if price is not None:
                    st["price"] = price
                    st["premarket_high"] = price if st["premarket_high"] is None \
                        else max(st["premarket_high"], price)
                    st["premarket_low"] = price if st["premarket_low"] is None \
                        else min(st["premarket_low"], price)
                if _f(q.get("previous_close")) is not None:
                    st["previous_close"] = _f(q.get("previous_close"))
                if _f(q.get("premarket_change")) is not None:
                    st["premarket_change"] = _f(q.get("premarket_change"))
                if _f(q.get("premarket_change_pct")) is not None:
                    st["premarket_change_pct"] = _f(q.get("premarket_change_pct"))
                if _i(q.get("premarket_volume")) is not None:
                    st["premarket_volume"] = _i(q.get("premarket_volume"))
                if _f(q.get("bid")) is not None:
                    st["bid"] = _f(q.get("bid"))
                if _f(q.get("ask")) is not None:
                    st["ask"] = _f(q.get("ask"))
                if st["bid"] is not None and st["ask"] is not None:
                    st["bid_ask_spread"] = round(st["ask"] - st["bid"], 6)
                st["updated_at"] = _now_iso()
                st["_updated_ts"] = _now_ts()
                st["source"] = source
                st["stale"] = False
                st["error"] = None
                self.state[symu] = st
            self.fallback_active = True
            self.fallback_reason = "applied_fallback_quotes"
            self.fallback_last_used = _now_iso()
            self.fallback_source = source

    # ── premarket relative volume ──────────────────────────────────────────────
    def set_avg_premarket_volume(self, symbol: str, avg_volume: Any,
                                 source: str = "premarket_20d") -> None:
        """Inject the trailing-average premarket volume for RVOL (run.py supplies history).

        `source` documents provenance ("premarket_20d" for true premarket history, or
        "opening_hour_avg" when only opening-hour volume is available as a proxy).
        If avg_volume is missing/invalid, RVOL stays unavailable — never fabricated.
        """
        avg = _f(avg_volume)
        if avg is None or avg <= 0:
            return
        symu = str(symbol or "").upper()
        if not symu:
            return
        with self._lock:
            st = self.state.get(symu) or self._new_state(symu)
            st["avg_premarket_volume_20d"] = avg
            st["rvol_source"] = source
            self.state[symu] = st

    def set_avg_premarket_volumes(self, avg_by_symbol: Dict[str, Any],
                                  source: str = "premarket_20d") -> None:
        for sym, avg in (avg_by_symbol or {}).items():
            self.set_avg_premarket_volume(sym, avg, source)

    def set_overnight_metrics(self, symbol: str, metrics: Dict[str, Any]) -> None:
        """Inject Alpaca overnight-session metrics for a symbol (advisory; never a trade)."""
        if not metrics or not metrics.get("overnight_available", True):
            return
        symu = str(symbol or "").upper()
        if not symu:
            return
        with self._lock:
            st = self.state.get(symu) or self._new_state(symu)
            for k in ("overnight_volume", "overnight_vwap", "overnight_range_pct",
                      "overnight_rvol", "overnight_trend", "overnight_session_strength"):
                if metrics.get(k) is not None:
                    st[k] = metrics.get(k)
            self.state[symu] = st

    def set_overnight_metrics_many(self, metrics_by_symbol: Dict[str, Dict[str, Any]]) -> None:
        for sym, m in (metrics_by_symbol or {}).items():
            self.set_overnight_metrics(sym, m)

    def write_overnight_report(self, out_dir: Any = None) -> Optional[Path]:
        """Write overnight_feed_report.json (per-symbol overnight metrics). Fail-safe."""
        target = Path(out_dir) if out_dir else self.output_dir
        try:
            target.mkdir(parents=True, exist_ok=True)
            snap = self.get_snapshot()
            rows = {s: {k: q.get(k) for k in ("overnight_volume", "overnight_vwap",
                        "overnight_range_pct", "overnight_rvol", "overnight_trend",
                        "overnight_session_strength")} for s, q in snap["quotes"].items()}
            available = sum(1 for r in rows.values() if r.get("overnight_volume") is not None)
            payload = {"generated_at": _now_iso(), "date_tag": os.getenv(
                           "RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d")),
                       "overnight_available_count": available, "symbols": rows,
                       "advisory_only": True}
            import json as _json
            p = target / "overnight_feed_report.json"
            p.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return p
        except Exception as exc:
            logger.debug("write_overnight_report failed: %s", exc)
            return None

    def write_rvol_report(self, out_dir: Any = None) -> Optional[Path]:
        """Write premarket_rvol_report.json (per-symbol RVOL + availability). Fail-safe."""
        target = Path(out_dir) if out_dir else self.output_dir
        try:
            target.mkdir(parents=True, exist_ok=True)
            snap = self.get_snapshot()
            rows = {sym: {"premarket_rvol": q.get("premarket_rvol"),
                          "premarket_volume": q.get("premarket_volume"),
                          "avg_premarket_volume_20d": q.get("avg_premarket_volume_20d"),
                          "rvol_source": q.get("rvol_source"),
                          "rvol_available": q.get("rvol_available")}
                    for sym, q in snap["quotes"].items()}
            payload = {
                "generated_at": _now_iso(),
                "date_tag": os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d")),
                "rvol_available_count": sum(1 for r in rows.values() if r["rvol_available"]),
                "symbols": rows,
            }
            p = target / "premarket_rvol_report.json"
            _write_json_safe(p, payload)
            return p
        except Exception as exc:
            logger.debug("write_rvol_report failed: %s", exc)
            return None

    # ── access ───────────────────────────────────────────────────────────────
    def _view(self, st: Dict[str, Any], now_ts: Optional[float] = None) -> Dict[str, Any]:
        now_ts = _now_ts() if now_ts is None else now_ts
        age = (now_ts - st["_updated_ts"]) if st.get("_updated_ts") else None
        stale = bool(age is not None and age > self.max_stale_seconds) or st["updated_at"] is None
        out = {k: st.get(k) for k in QUOTE_FIELDS}
        out["stale"] = stale
        out["market_hours"] = st.get("market_hours")
        out["age_seconds"] = (round(age, 3) if age is not None else None)
        # Premarket relative volume — only when a true average AND actual premarket prints
        # exist. premarket_volume of 0/None means "no prints yet", NOT valid low volume, so
        # RVOL stays unavailable (never fabricated). vol > 0 distinguishes the two.
        avg = st.get("avg_premarket_volume_20d")
        vol = st.get("premarket_volume")
        if avg and avg > 0 and vol is not None and vol > 0:
            out["premarket_rvol"] = round(vol / avg, 4)
            out["rvol_available"] = True
        else:
            out["premarket_rvol"] = None
            out["rvol_available"] = False
        out["avg_premarket_volume_20d"] = avg
        out["rvol_source"] = st.get("rvol_source")
        return out

    def get_symbol_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        symu = str(symbol or "").upper()
        with self._lock:
            st = self.state.get(symu)
            if st is None:
                return None
            return self._view(st)

    def get_snapshot(self) -> Dict[str, Any]:
        """Full premarket snapshot: per-symbol quotes + stale list + meta."""
        now_ts = _now_ts()
        with self._lock:
            quotes = {sym: self._view(st, now_ts) for sym, st in self.state.items()}
        stale_symbols = [s for s, q in quotes.items() if q.get("stale")]
        source = SOURCE_LIVE
        if quotes:
            srcs = {q.get("source") for q in quotes.values()}
            source = SOURCE_LIVE if SOURCE_LIVE in srcs else (next(iter(srcs)) if srcs else SOURCE_LIVE)
        elif self.fallback_used:
            source = self.fallback_source or SOURCE_FALLBACK
        return {
            "generated_at": _now_iso(),
            "source": source,
            "symbols": list(self.symbols),
            "quotes": quotes,
            "stale_symbols": stale_symbols,
            "fallback_used": self.fallback_used,
        }

    def health(self) -> Dict[str, Any]:
        snap = self.get_snapshot()
        return {
            "generated_at": _now_iso(),
            "feed_status": self.feed_status,
            "symbols_subscribed": len(self.symbols),
            "symbols_updated": len(self.symbols_updated),
            "last_message_at": self.last_message_at,
            "reconnect_count": self.reconnect_count,
            "error_count": self.error_count,
            "fallback_source": self.fallback_source,
            "fallback_used": self.fallback_used,
            "fallback_active": self.fallback_active,
            "fallback_reason": self.fallback_reason,
            "fallback_last_used": self.fallback_last_used,
            "shutdown_requested": self.shutdown_requested,
            "stale_count": len(snap["stale_symbols"]),
            "error": self.error,
            "yliveticker_available": YLIVETICKER_AVAILABLE,
        }

    # ── output ───────────────────────────────────────────────────────────────
    def write_snapshot(self, out_dir: Any = None) -> Dict[str, Path]:
        """Write premarket_live_snapshot.json + premarket_live_health.json. Fail-safe."""
        target = Path(out_dir) if out_dir else self.output_dir
        paths: Dict[str, Path] = {}
        try:
            target.mkdir(parents=True, exist_ok=True)
            snap_path = target / "premarket_live_snapshot.json"
            _write_json_safe(snap_path, self.get_snapshot())
            paths["snapshot"] = snap_path
            health_path = target / "premarket_live_health.json"
            _write_json_safe(health_path, self.health())
            paths["health"] = health_path
        except Exception as e:
            self.error = str(e)[:300]
            self.error_count += 1
            logger.debug("PremarketLiveFeed.write_snapshot failed: %s", e)
        return paths
