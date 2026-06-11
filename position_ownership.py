#!/usr/bin/env python3
"""
position_ownership.py — bot position-ownership ledger + manual-order guard.
============================================================================

Purpose
-------
run.py shares ONE live Alpaca account with the user's MANUAL orders (Alpaca
app/web) and the TradingView webhook (trade-webhook.service). run.py's exit
paths (sma200_exit, option-safety, BAL/insider exits, kill-switch flatten)
historically acted on EVERY position in the account — so they kept overriding
positions the user opened by hand.

Directive (2026-06-10): **NOTHING should override manual orders.**

This module is the enforcement layer. It maintains a small persistent ledger of
exactly what THIS bot opened, and exposes a guard the order chokepoint
(AlpacaGateway.submit_market_order) consults before any SELL:

  * The bot may only SELL a symbol it has recorded as bot-opened, and only up to
    the quantity it opened (the rest is the user's — never touched).
  * Anything not in the ledger is, by definition, NOT the bot's → SELL is blocked.

Design choices
--------------
* **Safe by default / fail-closed.** Empty ledger ⇒ the bot owns nothing ⇒ it
  will not sell any pre-existing position until it opens new ones itself. On any
  error the guard denies the sell (never wrongly overrides a manual order).
* **Local, API-free.** The ledger is a JSON file, so the guard works even if the
  Alpaca order API is flaky. Keyed by EXACT symbol (OCC for options, ticker for
  equities) so SOXS shares and SOXS calls are tracked independently.
* **client_order_id tagging.** Bot orders are stamped with BOT_COID_PREFIX so
  the bot's fills are identifiable in Alpaca history for audits/reconciliation.

This module never raises into the trading loop — every public call is guarded.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent
LEDGER_PATH = Path(
    os.getenv("BOT_OWNED_LEDGER_PATH", str(PROJECT_ROOT / "system-data" / "bot_owned_ledger.json"))
)

# Every bot-submitted order's client_order_id starts with this, so bot fills are
# distinguishable from manual (Alpaca app/web) and webhook fills in order history.
BOT_COID_PREFIX = os.getenv("BOT_COID_PREFIX", "qqbot")

# Quantities below this are treated as fully closed (fractional-share dust).
_EPS = 1e-4

_LOCK = threading.RLock()
_CACHE: Dict[str, float] | None = None


def _norm(symbol: str) -> str:
    return str(symbol or "").upper().strip()


def make_bot_coid(engine: str = "") -> str:
    """A bot-tagged client_order_id, e.g. 'qqbot-insider-3f9a1c'. <=48 chars (Alpaca limit)."""
    eng = "".join(ch for ch in str(engine).lower() if ch.isalnum())[:12]
    coid = f"{BOT_COID_PREFIX}-{eng}-{uuid.uuid4().hex[:8]}" if eng else f"{BOT_COID_PREFIX}-{uuid.uuid4().hex[:10]}"
    return coid[:48]


def is_bot_coid(coid: str) -> bool:
    return str(coid or "").startswith(BOT_COID_PREFIX)


def _load() -> Dict[str, float]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data: Dict[str, float] = {}
    try:
        if LEDGER_PATH.exists():
            raw = json.loads(LEDGER_PATH.read_text() or "{}")
            owned = raw.get("owned", raw) if isinstance(raw, dict) else {}
            for k, v in owned.items():
                try:
                    data[_norm(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    except Exception:
        data = {}
    _CACHE = data
    return data


def _save(data: Dict[str, float]) -> None:
    global _CACHE
    _CACHE = data
    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"owned": {k: round(v, 6) for k, v in data.items() if v > _EPS}},
                                  indent=2, sort_keys=True))
        os.replace(tmp, LEDGER_PATH)
    except Exception:
        pass


def bot_owned_qty(symbol: str) -> float:
    """Quantity of `symbol` this bot opened and still holds (0.0 if none / not bot-owned)."""
    try:
        with _LOCK:
            return max(0.0, float(_load().get(_norm(symbol), 0.0)))
    except Exception:
        return 0.0


def is_bot_owned(symbol: str) -> bool:
    return bot_owned_qty(symbol) > _EPS


def record_bot_open(symbol: str, qty: float) -> None:
    """Record that the bot opened/added `qty` of `symbol` (call after a successful BUY)."""
    try:
        q = float(qty)
        if q <= 0:
            return
        with _LOCK:
            data = dict(_load())
            data[_norm(symbol)] = data.get(_norm(symbol), 0.0) + q
            _save(data)
    except Exception:
        pass


def record_bot_close(symbol: str, qty: float) -> None:
    """Record that the bot closed `qty` of `symbol` (call after a successful SELL)."""
    try:
        q = float(qty)
        if q <= 0:
            return
        with _LOCK:
            data = dict(_load())
            key = _norm(symbol)
            remaining = data.get(key, 0.0) - q
            if remaining > _EPS:
                data[key] = remaining
            else:
                data.pop(key, None)
            _save(data)
    except Exception:
        pass


def sell_allowance(symbol: str, requested_qty: float) -> float:
    """
    How many shares/contracts of `symbol` the bot is permitted to SELL.

    Returns min(requested, bot-owned). 0.0 means BLOCK — the position was opened
    manually or by the webhook and must not be overridden.
    """
    try:
        owned = bot_owned_qty(symbol)
        if owned <= _EPS:
            return 0.0
        return min(float(requested_qty), owned)
    except Exception:
        return 0.0


def seed(symbol: str, qty: float) -> None:
    """Manually claim an existing position as bot-owned (operator use; default is to claim nothing)."""
    record_bot_open(symbol, qty)


def snapshot() -> Dict[str, float]:
    with _LOCK:
        return dict(_load())


# ── operator CLI: inspect / seed / clear the ledger ─────────────────────────────
if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args or args[0] in ("show", "list"):
        snap = snapshot()
        print(f"bot-owned ledger ({LEDGER_PATH}):")
        if not snap:
            print("  (empty — bot owns nothing; it will not sell any existing position)")
        for s, q in sorted(snap.items()):
            print(f"  {s:24s} {q:g}")
    elif args[0] == "seed":
        # python position_ownership.py seed AAPL:10 SOXS260612C00006500:0  (qty>0 to claim)
        for spec in args[1:]:
            sym, _, q = spec.partition(":")
            seed(sym, float(q or 0))
            print(f"seeded {sym} += {q}")
        print("ledger:", snapshot())
    elif args[0] == "clear":
        with _LOCK:
            _save({})
        print("ledger cleared — bot owns nothing.")
    else:
        print("usage: position_ownership.py [show|seed SYM:QTY ...|clear]")
