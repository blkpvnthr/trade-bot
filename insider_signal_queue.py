#!/usr/bin/env python3
"""
insider_signal_queue.py — defer insider-copy signals blocked by transient, resets-daily caps.
=============================================================================================

A qualified insider-copy signal (passed value / freshness / vol / liquidity) can still be
blocked at the order gate by a TRANSIENT, resets-daily constraint:

    * daily-loss limit breached  (rejected_by_daily_loss)   — clears at the next session
    * per-order dollar cap        (rejected_by_order_size)   — can clear as equity/sizing shift

Rather than drop those signals (the backtest takes the entry on the filing day), queue them and
REPLAY next session — but ONLY while still fresh: a hard ≤ INSIDER_COPY_MAX_AGE_DAYS gate so a
stale signal is never fired (the edge depends on entering near the filing, not days later).

NOT queued: structural rejections (regime, kill-switch, vol/liquidity, already-held, max-open) —
those aren't transient and queuing them would either re-fire the same block or trade stale.

Pure JSON file + stdlib; fail-safe (never raises into the trading loop).
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
QUEUE_PATH = Path(os.getenv("INSIDER_SIGNAL_QUEUE_PATH",
                            str(PROJECT_ROOT / "logs" / "insider_signal_queue.json")))

# Only these (transient, resets-daily) blocks are queued.
TRANSIENT_REASONS = ("daily_loss", "order_size")


def is_transient_block(reason: str) -> bool:
    r = str(reason or "").lower()
    return any(t in r for t in TRANSIENT_REASONS)


def _load() -> List[Dict[str, Any]]:
    try:
        if QUEUE_PATH.exists():
            data = json.loads(QUEUE_PATH.read_text() or "[]")
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _save(q: List[Dict[str, Any]]) -> None:
    try:
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = QUEUE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(q, indent=2, default=str))
        os.replace(tmp, QUEUE_PATH)
    except Exception:
        pass


def _parse_date(s: Any):
    try:
        if isinstance(s, (date, datetime)):
            return s.date() if isinstance(s, datetime) else s
        return datetime.fromisoformat(str(s)[:10]).date()
    except Exception:
        return None


def enqueue(signal: Dict[str, Any]) -> None:
    """Add a transient-blocked signal (deduped by symbol; keeps the most recent metadata)."""
    try:
        sym = str(signal.get("symbol", "")).upper()
        if not sym:
            return
        q = [s for s in _load() if str(s.get("symbol", "")).upper() != sym]
        rec = dict(signal)
        rec["symbol"] = sym
        rec.setdefault("queued_utc", datetime.now(timezone.utc).isoformat())
        q.append(rec)
        _save(q)
    except Exception:
        pass


def mark_executed(symbol: str) -> None:
    """Remove a symbol from the queue (call after a successful copy)."""
    try:
        sym = str(symbol).upper()
        q = [s for s in _load() if str(s.get("symbol", "")).upper() != sym]
        _save(q)
    except Exception:
        pass


def replay_fresh(max_age_days: int) -> List[Dict[str, Any]]:
    """
    Return queued signals still within the ≤ max_age_days freshness window as candidate dicts
    (symbol, value, last [date], company, breadth, repeated). STALE entries are DROPPED from the
    queue (logged via the returned 'dropped' side file is omitted — caller logs counts). Never
    fires a stale signal.
    """
    out: List[Dict[str, Any]] = []
    keep: List[Dict[str, Any]] = []
    try:
        today = datetime.now(timezone.utc).date()
        for s in _load():
            last = _parse_date(s.get("last"))
            if last is None:
                continue  # un-dateable → drop (can't prove freshness)
            if (today - last).days <= int(max_age_days):
                keep.append(s)
                out.append({
                    "symbol": str(s.get("symbol", "")).upper(),
                    "value": float(s.get("value") or 0.0),
                    "last": last,
                    "company": str(s.get("company") or ""),
                    "breadth": int(s.get("breadth") or 1),
                    "repeated": bool(s.get("repeated")),
                    "from_queue": True,
                })
            # else: stale → not kept (dropped)
        _save(keep)
    except Exception:
        return []
    return out


def snapshot() -> List[Dict[str, Any]]:
    return _load()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        _save([])
        print("queue cleared.")
    else:
        q = _load()
        print(f"insider signal queue ({QUEUE_PATH}): {len(q)} pending")
        for s in q:
            print(f"  {s.get('symbol'):8s} value=${float(s.get('value') or 0):,.0f} "
                  f"last={s.get('last')} block={s.get('block_reason')} queued={s.get('queued_utc')}")
