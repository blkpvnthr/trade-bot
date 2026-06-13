"""
tjl_engine.py — run.py-facing execution wrapper for the Trend Join Long strategy.

Separation of concerns:
  * tjl_signal.py  = PURE decision logic (no I/O, fully unit-tested).
  * tjl_engine.py  = bar fetching + order routing. NO trade decision is made here
    that isn't returned by tjl_signal. The LLM is never in this path.

Safety design (every constraint from the task is enforced here):
  * Self-gates on TJL_ENABLED — returns 0 immediately when disabled, so run.py can
    call it unconditionally with zero effect when off.
  * Orders route ONLY through gateway.submit_market_order(), inheriting every existing
    risk gate (kill-switch, equity floor, PDT, daily-loss, exposure caps, ownership
    guard, self-assessment throttle, size-to-fit). Entries pass allow_momentum=True so
    the dip-buy 9-EMA gate doesn't silently kill the crossover entry; the regime gate
    and below-200SMA block still apply.
  * Higher-timeframe bars are filtered to COMPLETED bars only (bar_start + tf <= now),
    matching the Pine request.security(..., _expr[1], lookahead_on) non-repaint contract.
  * Exits are scoped to TJL-opened positions only (tjl_open_positions.json) so this
    engine never touches positions owned by other engines or the user.
  * Fail-safe: every public entry/exit wrapped so it never raises into the trade loop.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import tjl_signal as T

logger = logging.getLogger("tjl.engine")
ET = ZoneInfo("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parent
_STATE_PATH = PROJECT_ROOT / "tjl_open_positions.json"

ALPACA_KEY = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or "")
ALPACA_SECRET = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or "")

# Per-order dollar budget (operational, not a secret). Risk caps still trim this.
TJL_ORDER_DOLLARS = float(os.environ.get("TJL_ORDER_DOLLARS", "200") or 200)
# How many entry symbols to scan per cycle (bounds the HTF fetch cost).
TJL_MAX_SCAN = int(os.environ.get("TJL_MAX_SCAN", "25") or 25)
_HTF_FETCH_TIMEOUT = 20  # seconds, per HTF batch


def _in_test() -> bool:
    import sys as _sys
    return ("PYTEST_CURRENT_TEST" in os.environ
            or "unittest" in _sys.modules
            or any("pytest" in a or "unittest" in a for a in _sys.argv[:2] if a))


# ──────────────────────────────────────────────────────────────────────────────
# TJL open-position state (scopes exits to TJL-opened names only)
# ──────────────────────────────────────────────────────────────────────────────

def _state_load() -> Dict[str, Dict[str, Any]]:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception:
        logger.debug("tjl state load failed", exc_info=True)
    return {}


def _state_save(state: Dict[str, Dict[str, Any]]) -> None:
    if _in_test():
        return
    try:
        _STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        logger.debug("tjl state save failed", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# Higher-timeframe bar fetch (COMPLETED bars only — no lookahead)
# ──────────────────────────────────────────────────────────────────────────────

def _data_client():
    if not (ALPACA_KEY and ALPACA_SECRET):
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    except Exception:
        logger.debug("tjl: alpaca data client unavailable", exc_info=True)
        return None


def fetch_htf_ohlc(symbol: str, tf_min: int, count: int,
                   now_utc: Optional[datetime] = None
                   ) -> Tuple[List[float], List[float], List[float]]:
    """Return (highs, lows, closes) of the last `count` COMPLETED `tf_min`-minute bars,
    oldest→newest. A bar starting at T is complete only when now >= T + tf_min, which
    is what makes the trend/momentum filters non-repainting. Empty on any failure
    (callers treat empty as 'filter not satisfied' → no entry; fail-safe)."""
    client = _data_client()
    if client is None or tf_min <= 0 or count <= 0:
        return [], [], []
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        import concurrent.futures as _cf

        # Alpaca: Minute unit allows only 1-59; 60m+ must use the Hour unit.
        if tf_min >= 60 and tf_min % 60 == 0:
            tf = TimeFrame(tf_min // 60, TimeFrameUnit.Hour)
        else:
            tf = TimeFrame(tf_min, TimeFrameUnit.Minute)
        # Calendar lookback must cover `count` bars of ~390 RTH min/day (+weekend/holiday buffer),
        # NOT count*tf wall-clock minutes — else hourly trend bars come back far too short.
        _tdays = max(1.0, (count * tf_min) / 390.0)
        start = now_utc - timedelta(days=int(_tdays * 1.6) + 3)
        req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=tf,
                               start=start, end=now_utc, feed=DataFeed.IEX)
        pool = _cf.ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(lambda: client.get_stock_bars(req).df)
        try:
            df = fut.result(timeout=_HTF_FETCH_TIMEOUT)
        finally:
            pool.shutdown(wait=False)
        if df is None or df.empty:
            return [], [], []
        df = df.reset_index()
        cutoff = now_utc - timedelta(minutes=tf_min)   # bar_start <= cutoff => completed
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts <= cutoff:                            # COMPLETED bar only
                highs.append(float(row["high"]))
                lows.append(float(row["low"]))
                closes.append(float(row["close"]))
        return highs[-count:], lows[-count:], closes[-count:]
    except Exception as e:
        logger.warning("tjl: HTF fetch failed for %s @%dm: %s", symbol, tf_min, e)
        return [], [], []


# ──────────────────────────────────────────────────────────────────────────────
# Entry / exit passes (called from run_two_engine_cycle)
# ──────────────────────────────────────────────────────────────────────────────

def _closes_and_ts(df) -> Tuple[List[float], Optional[datetime]]:
    """Extract close series + last bar timestamp (ET) from a run.py 1m bar DataFrame."""
    try:
        closes = [float(x) for x in df["close"].tolist()]
        ts = df.index[-1]
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return closes, ts.astimezone(ET)
    except Exception:
        return [], None


def run_tjl_entries(gateway, risk, symbols: List[str], bars_1m: Dict[str, Any],
                    market_context: dict, equity_positions: Dict[str, Any]) -> int:
    """TJL long-only entries. Deterministic: tjl_signal decides, gateway executes.
    Returns count of BUYs placed. Never raises."""
    cfg = T.TJLConfig.from_env()
    if not cfg.enabled:
        return 0
    placed = 0
    try:
        held = set((equity_positions or {}).keys())
        state = _state_load()
        scanned = 0
        for sym in symbols:
            if scanned >= TJL_MAX_SCAN:
                break
            df = (bars_1m or {}).get(sym)
            if df is None or len(df) < 2:
                continue
            scanned += 1
            entry_closes, now_ts = _closes_and_ts(df)
            if now_ts is None:
                continue
            in_position = sym in held or sym in state

            # Cheap gates first; only pay for HTF bars if those pass (perf + no wasted I/O).
            if in_position:
                logger.info("TJL %s blocked: %s", sym, T.ALREADY_IN_POSITION)
                continue
            if not T.is_rth(now_ts):
                logger.debug("TJL %s blocked: %s", sym, T.OUTSIDE_RTH)
                continue

            t_hi, t_lo, t_cl = fetch_htf_ohlc(sym, cfg.trend_tf_min, cfg.trend_sma_len + 10)
            m_hi, m_lo, m_cl = fetch_htf_ohlc(
                sym, cfg.momo_tf_min, cfg.stoch_len + cfg.smooth_k + cfg.smooth_d + 10)

            sig = T.evaluate_entry(cfg, t_cl, m_hi, m_lo, m_cl, entry_closes,
                                   now_ts, in_position, log=True)
            if not sig.enter:
                continue

            px = entry_closes[-1]
            qty = max(1, int(TJL_ORDER_DOLLARS / px)) if px > 0 else 0
            if qty <= 0:
                continue
            reason = f"tjl_entry:emaX&trend&momo:px={px:.2f}"
            resp = gateway.submit_market_order(sym, "buy", qty, reason=reason,
                                               reference_price=px, allow_momentum=True)
            if resp:
                placed += 1
                state[sym] = {"avg_entry": px, "qty": qty,
                              "opened_at": now_ts.isoformat()}
                _state_save(state)
                _log_exec(symbol=sym, side="buy", qty=qty, price=px,
                          reason=reason, engine="tjl")
                logger.info("TJL ENTER %s x%d @ ~%.2f", sym, qty, px)
        return placed
    except Exception as e:
        logger.error("TJL entries failed (cycle unaffected): %s", e, exc_info=True)
        return placed


def run_tjl_exits(gateway, equity_positions: Dict[str, Any]) -> int:
    """Exit TJL-opened positions on stop-loss, take-profit, or end-of-session flatten.
    Scoped to names in tjl_open_positions.json so other engines/user positions are never
    touched. Exits are NEVER regime-gated. Returns count of SELLs. Never raises."""
    cfg = T.TJLConfig.from_env()
    if not cfg.enabled:
        return 0
    sold = 0
    try:
        state = _state_load()
        if not state:
            return 0
        now_et = datetime.now(ET)
        after_close = (now_et.weekday() < 5 and (now_et.hour * 60 + now_et.minute) >= 16 * 60)
        for sym in list(state.keys()):
            pos = (equity_positions or {}).get(sym)
            if pos is None:
                # position already gone (closed elsewhere) — drop from state
                state.pop(sym, None)
                _state_save(state)
                continue
            try:
                qty = abs(int(float(getattr(pos, "qty", 0) or 0)))
                avg = float(getattr(pos, "avg_entry_price", 0) or 0) or float(state[sym].get("avg_entry", 0))
                price = gateway._resolve_reference_price(sym)
            except Exception:
                continue
            if qty <= 0 or price <= 0 or avg <= 0:
                continue

            reason = ""
            if T.hit_stop(price, avg, cfg.stop_pct):
                reason = f"tjl_stop:px={price:.2f}<entry={avg:.2f}-{cfg.stop_pct*100:.1f}%"
            elif T.hit_take_profit(price, avg, cfg.tp_pct):
                reason = f"tjl_tp:px={price:.2f}>entry={avg:.2f}+{cfg.tp_pct*100:.1f}%"
            elif cfg.eod_flat and after_close:
                reason = f"tjl_eod_flat:px={price:.2f}"
            if not reason:
                continue

            resp = gateway.submit_market_order(sym, "sell", qty, reason=reason)
            if resp:
                sold += 1
                state.pop(sym, None)
                _state_save(state)
                _log_exec(symbol=sym, side="sell", qty=qty, price=price,
                          reason=reason, engine="tjl")
                logger.info("TJL EXIT %s x%d (%s)", sym, qty, reason)
        return sold
    except Exception as e:
        logger.error("TJL exits failed (cycle unaffected): %s", e, exc_info=True)
        return sold


def _log_exec(**kwargs) -> None:
    """Best-effort executed-order log via run.py's logger if importable (no hard dep)."""
    if _in_test():
        return
    try:
        import run  # type: ignore
        run.log_executed_order(**kwargs)
    except Exception:
        logger.debug("tjl: log_executed_order unavailable", exc_info=True)
