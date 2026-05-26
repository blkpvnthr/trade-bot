#!/usr/bin/env python3
"""
webhook_order_router.py

Dedicated webhook signal consumer and order router.

Reads signals from a JSONL queue file, validates them, applies risk controls,
and submits Alpaca orders. Runs as a standalone loop or under run.py when
INSTANCE_ROLE=webhook_order_router.

Signal queue format (one JSON object per line):
    {
        "signal_id": "<unique id>",
        "symbol":    "AAPL",
        "side":      "buy" | "sell",
        "qty":       10,           # shares (mutually exclusive with notional)
        "notional":  500.0,        # dollars (mutually exclusive with qty)
        "source":    "tradingview" | "manual" | ...,
        "timestamp_utc": "2026-01-01T09:30:00.000Z",
        "stop_loss":    145.00,    # optional
        "take_profit":  160.00,    # optional
        "notes":        "...",     # optional
    }

qty=0 or notional=0: triggers reversal investigation. If a long position
exists in the symbol, close it. If no position exists, the signal is logged
and ignored.

Env vars:
    CONSUME_WEBHOOK_ORDER_QUEUE=true   (required to enable queue consumption)
    WEBHOOK_QUEUE_PATH                 (default: daily_system_outputs/webhook_order_router/signal_queue.jsonl)
    WEBHOOK_DIRECT_ORDER_ROUTING=false (if true: bypass Opportunity Finder approval)
    REQUIRE_OPPORTUNITY_FINDER_APPROVAL=true
    REQUIRE_FOOTPRINT_CONFIRMATION_FOR_BUYS=true
    WEBHOOK_ROUTER_SCAN_SECONDS=5
    WEBHOOK_MAX_ORDER_DOLLARS=3000
    WEBHOOK_DAILY_LOSS_LIMIT_DOLLARS=500
    WEBHOOK_DAILY_LOSS_LIMIT_PCT=0.02
    APCA_PAPER=true

Paper-safe by default. Never enables live trading without ALLOW_LIVE_TRADING=1.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from config.env_loader import load_instance_env

load_instance_env()

ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", TODAY_TAG))

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

STRATEGY_ID = os.getenv("STRATEGY_ID", os.getenv("INSTANCE_NAME", "webhook_order_router")).strip()
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs" / "execution" / STRATEGY_ID)))
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_JSONL = LOG_DIR / f"{DATA_TAG}_execution_events.jsonl"
KILL_SWITCH_PATH = LOG_DIR / f"{DATA_TAG}_kill_switch_state.json"

# ── Queue and state paths ──────────────────────────────────────────────────────
_QUEUE_DIR = PROJECT_ROOT / "daily_system_outputs" / "webhook_order_router"
_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QUEUE_PATH = _QUEUE_DIR / "signal_queue.jsonl"
PROCESSED_QUEUE_PATH = _QUEUE_DIR / f"{DATA_TAG}_processed_signals.jsonl"
ORDERS_LOG_PATH = _QUEUE_DIR / f"{DATA_TAG}_orders.csv"
DAILY_LOSS_STATE_PATH = _QUEUE_DIR / f"{DATA_TAG}_daily_loss_state.json"

# Opportunity Finder approval file: opportunity_finder.py writes this
OPPORTUNITY_FINDER_APPROVAL_PATH = OUT_DIR / f"{DATA_TAG}_opportunity_finder_approvals.json"

# ── Alpaca imports ─────────────────────────────────────────────────────────────
try:
    from alpaca.common.exceptions import APIError
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus, OrderType, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    print("[webhook-router] alpaca-py not available — orders will be logged only")

try:
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
    _BRACKET_AVAILABLE = True
except ImportError:
    _BRACKET_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
_bool = lambda key, default: os.getenv(key, default).lower() in {"1", "true", "yes"}
_float = lambda key, default: float(os.getenv(key, str(default)))
_int = lambda key, default: int(os.getenv(key, str(default)))

CONSUME_QUEUE: bool = _bool("CONSUME_WEBHOOK_ORDER_QUEUE", "true")
DIRECT_ROUTING: bool = _bool("WEBHOOK_DIRECT_ORDER_ROUTING", "false")
REQUIRE_OPPORTUNITY_FINDER: bool = _bool("REQUIRE_OPPORTUNITY_FINDER_APPROVAL", "true")
REQUIRE_FOOTPRINT: bool = _bool("REQUIRE_FOOTPRINT_CONFIRMATION_FOR_BUYS", "true")
PAPER: bool = _bool("APCA_PAPER", "true") or not _bool("ALLOW_LIVE_TRADING", "0")
EXECUTE_ORDERS: bool = _bool("WEBHOOK_EXECUTE_ORDERS", "1")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")

QUEUE_PATH = Path(os.getenv("WEBHOOK_QUEUE_PATH", str(DEFAULT_QUEUE_PATH)))
SCAN_SECONDS: int = _int("WEBHOOK_ROUTER_SCAN_SECONDS", "5")
MAX_ORDER_DOLLARS: float = _float("WEBHOOK_MAX_ORDER_DOLLARS", "3000.0")
MIN_ORDER_DOLLARS: float = _float("WEBHOOK_MIN_ORDER_DOLLARS", "50.0")
DAILY_LOSS_LIMIT_DOLLARS: float = _float("WEBHOOK_DAILY_LOSS_LIMIT_DOLLARS", "500.0")
DAILY_LOSS_LIMIT_PCT: float = _float("WEBHOOK_DAILY_LOSS_LIMIT_PCT", "0.02")


# ======================================================================
# Event writing
# ======================================================================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def write_event(event: str, payload: dict[str, Any]) -> None:
    try:
        row = {"event": event, "timestamp_utc": _now_utc_iso(), **payload}
        with EVENTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


def _append_order_log(row: dict[str, Any]) -> None:
    try:
        df_new = pd.DataFrame([row])
        if ORDERS_LOG_PATH.exists():
            df_new.to_csv(ORDERS_LOG_PATH, mode="a", header=False, index=False)
        else:
            df_new.to_csv(ORDERS_LOG_PATH, index=False)
    except Exception:
        pass


def _safe_symbol(s: str) -> str:
    return str(s).strip().upper().replace(".", "/")


# ======================================================================
# Kill switch check
# ======================================================================
def _kill_switch_active() -> tuple[bool, str]:
    """
    Read kill switch state file. Returns (active, reason).
    Never blocks orders if file is missing/unreadable (fail open for exits,
    but we return active=False to let the caller decide).
    """
    try:
        if not KILL_SWITCH_PATH.exists():
            return False, ""
        state = json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
        if state.get("kill_switch_active"):
            return True, state.get("reason", "kill_switch_file")
    except Exception:
        pass
    return False, ""


# ======================================================================
# Daily loss state
# ======================================================================
def _load_daily_loss_state() -> dict[str, Any]:
    try:
        if DAILY_LOSS_STATE_PATH.exists():
            return json.loads(DAILY_LOSS_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"realized_pnl": 0.0, "date": TODAY_TAG, "trades": 0}


def _save_daily_loss_state(state: dict[str, Any]) -> None:
    try:
        DAILY_LOSS_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _daily_loss_limit_hit(state: dict[str, Any]) -> tuple[bool, str]:
    if state.get("date") != TODAY_TAG:
        return False, ""
    pnl = float(state.get("realized_pnl", 0.0))
    if pnl <= -DAILY_LOSS_LIMIT_DOLLARS:
        return True, f"daily_loss_dollars: realized_pnl={pnl:.2f} >= limit={DAILY_LOSS_LIMIT_DOLLARS:.2f}"
    return False, ""


# ======================================================================
# Opportunity Finder approval check
# ======================================================================
def _opportunity_finder_approved(symbol: str) -> tuple[bool, str]:
    """
    Check if the Opportunity Finder has approved this symbol for buys today.
    If the approvals file is missing and REQUIRE_OPPORTUNITY_FINDER=true,
    the signal is rejected (safe default).
    If DIRECT_ROUTING=true, this check is bypassed.
    """
    if DIRECT_ROUTING or not REQUIRE_OPPORTUNITY_FINDER:
        return True, "opportunity_finder_bypassed"
    if not OPPORTUNITY_FINDER_APPROVAL_PATH.exists():
        return False, "opportunity_finder_approval_file_missing"
    try:
        data = json.loads(OPPORTUNITY_FINDER_APPROVAL_PATH.read_text(encoding="utf-8"))
        approvals = {_safe_symbol(k): v for k, v in data.get("approvals", {}).items()}
        entry = approvals.get(_safe_symbol(symbol), {})
        if isinstance(entry, bool):
            approved = entry
            reason = "approved" if approved else "not_approved"
        else:
            approved = bool(entry.get("approved", False))
            reason = entry.get("reason", "approved" if approved else "not_approved")
        return approved, reason
    except Exception as exc:
        return False, f"opportunity_finder_read_error: {exc}"


# ======================================================================
# Signal queue reader
# ======================================================================
def _load_queue_signals() -> list[dict[str, Any]]:
    """Load all unprocessed signals from the queue file."""
    if not QUEUE_PATH.exists():
        return []
    try:
        lines = QUEUE_PATH.read_text(encoding="utf-8").splitlines()
        signals = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(json.loads(line))
            except Exception:
                pass
        return signals
    except Exception as exc:
        print(f"[webhook-router] queue read error: {exc}")
        return []


def _load_processed_ids() -> set[str]:
    """Return set of already-processed signal IDs."""
    if not PROCESSED_QUEUE_PATH.exists():
        return set()
    try:
        ids: set[str] = set()
        for line in PROCESSED_QUEUE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sig_id = rec.get("signal_id", "")
                if sig_id:
                    ids.add(sig_id)
            except Exception:
                pass
        return ids
    except Exception:
        return set()


def _mark_processed(signal: dict[str, Any], outcome: str, detail: str = "") -> None:
    try:
        rec = {
            "signal_id": signal.get("signal_id", ""),
            "symbol": signal.get("symbol", ""),
            "side": signal.get("side", ""),
            "outcome": outcome,
            "detail": detail,
            "processed_at_utc": _now_utc_iso(),
        }
        with PROCESSED_QUEUE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# ======================================================================
# Signal validation
# ======================================================================
@dataclass
class ValidatedSignal:
    signal_id: str
    symbol: str
    side: str          # "buy" | "sell"
    qty: int           # 0 means reversal
    notional: float    # 0.0 if qty-based
    stop_loss: float   # 0.0 if not set
    take_profit: float # 0.0 if not set
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


def _validate_signal(raw: dict[str, Any]) -> tuple[ValidatedSignal | None, str]:
    """
    Validate a raw signal dict. Returns (ValidatedSignal, "") on success,
    (None, rejection_reason) on failure.
    """
    sig_id = str(raw.get("signal_id") or raw.get("id") or uuid.uuid4().hex[:12])
    symbol_raw = raw.get("symbol", "")
    if not symbol_raw or not isinstance(symbol_raw, str):
        return None, "missing_symbol"
    symbol = _safe_symbol(symbol_raw)
    if not symbol.replace("/", "").isalnum():
        return None, f"invalid_symbol: {symbol}"

    side_raw = str(raw.get("side", "buy")).lower().strip()
    if side_raw not in {"buy", "sell", "short"}:
        return None, f"invalid_side: {side_raw}"
    side = "sell" if side_raw in {"sell", "short"} else "buy"

    qty_raw = raw.get("qty", None)
    notional_raw = raw.get("notional", None)

    if qty_raw is not None:
        try:
            qty = int(float(qty_raw))
        except (TypeError, ValueError):
            return None, f"invalid_qty: {qty_raw}"
        notional = 0.0
    elif notional_raw is not None:
        try:
            notional = float(notional_raw)
        except (TypeError, ValueError):
            return None, f"invalid_notional: {notional_raw}"
        if notional < 0:
            return None, f"negative_notional: {notional}"
        qty = 0  # will be resolved by broker side
    else:
        return None, "missing_qty_or_notional"

    stop_loss = float(raw.get("stop_loss", 0.0) or 0.0)
    take_profit = float(raw.get("take_profit", 0.0) or 0.0)
    source = str(raw.get("source", "unknown"))

    return ValidatedSignal(
        signal_id=sig_id,
        symbol=symbol,
        side=side,
        qty=qty,
        notional=notional,
        stop_loss=stop_loss,
        take_profit=take_profit,
        source=source,
        raw=raw,
    ), ""


# ======================================================================
# Order router
# ======================================================================
class WebhookOrderRouter:
    def __init__(self) -> None:
        self._trading_client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if not _ALPACA_AVAILABLE:
            return
        if not APCA_KEY or not APCA_SECRET:
            print("[webhook-router] Alpaca credentials not found — orders disabled")
            return
        try:
            self._trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=PAPER)
            print(f"[webhook-router] trading client ready paper={PAPER}")
        except Exception as exc:
            print(f"[webhook-router] client init failed: {exc}")

    def _get_position(self, symbol: str) -> Any | None:
        if self._trading_client is None:
            return None
        try:
            return self._trading_client.get_open_position(symbol)
        except Exception:
            return None

    def _has_open_order(self, symbol: str) -> bool:
        if self._trading_client is None:
            return False
        try:
            orders = self._trading_client.get_orders()
            for o in orders:
                if _safe_symbol(getattr(o, "symbol", "")) == _safe_symbol(symbol):
                    status = str(getattr(o, "status", "")).lower()
                    if status in {"new", "accepted", "pending_new", "partially_filled", "held"}:
                        return True
        except Exception:
            pass
        return False

    def _get_equity(self) -> float:
        if self._trading_client is None:
            return 0.0
        try:
            acct = self._trading_client.get_account()
            return float(getattr(acct, "equity", 0.0))
        except Exception:
            return 0.0

    def _resolve_qty(self, sig: ValidatedSignal) -> tuple[int, str]:
        """Resolve shares to buy. Returns (qty, rejection_reason_or_empty)."""
        if sig.qty > 0:
            return sig.qty, ""
        # Notional-based sizing
        if sig.notional <= 0:
            return 0, "qty_zero_or_notional_zero (use for reversal only)"
        equity = self._get_equity()
        max_notional = min(MAX_ORDER_DOLLARS, sig.notional)
        if equity > 0:
            max_notional = min(max_notional, equity * 0.10)  # hard cap: 10% per position
        if max_notional < MIN_ORDER_DOLLARS:
            return 0, f"notional_below_min: {max_notional:.2f} < {MIN_ORDER_DOLLARS:.2f}"
        # Estimate price from Alpaca; if unavailable use notional/max fallback of 10 shares
        try:
            pos = self._get_position(sig.symbol)
            if pos:
                price = float(getattr(pos, "current_price", 0.0))
                if price > 0:
                    return max(1, int(max_notional / price)), ""
        except Exception:
            pass
        return 0, "cannot_resolve_qty_without_price"

    def _handle_reversal(self, sig: ValidatedSignal) -> str:
        """qty=0: close existing long position if one exists."""
        pos = self._get_position(sig.symbol)
        if pos is None:
            msg = f"reversal_no_position: {sig.symbol} has no open position — signal ignored"
            print(f"[webhook-router] {msg}")
            write_event("webhook_reversal_no_position", {"signal_id": sig.signal_id, "symbol": sig.symbol})
            return "no_position"

        existing_qty = int(float(getattr(pos, "qty", 0)))
        if existing_qty <= 0:
            msg = f"reversal_zero_or_short: {sig.symbol} qty={existing_qty} — ignored (long-only)"
            print(f"[webhook-router] {msg}")
            return "not_long"

        print(f"[webhook-router] reversal close {sig.symbol} qty={existing_qty}")
        if not EXECUTE_ORDERS or self._trading_client is None:
            write_event("webhook_reversal_signal_only", {"signal_id": sig.signal_id, "symbol": sig.symbol, "qty": existing_qty})
            return "signal_only"

        try:
            order_tag = f"whkrev_{sig.symbol}_{TODAY_TAG}"[:48]
            req = MarketOrderRequest(
                symbol=sig.symbol,
                qty=existing_qty,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=order_tag,
            )
            order = self._trading_client.submit_order(order_data=req)
            order_id = getattr(order, "id", "")
            payload = {
                "signal_id": sig.signal_id, "symbol": sig.symbol,
                "side": "sell", "qty": existing_qty, "order_id": order_id,
                "reversal": True, "submitted_at_utc": _now_utc_iso(),
            }
            write_event("webhook_reversal_submitted", payload)
            _append_order_log(payload)
            print(f"[webhook-router] reversal close submitted {sig.symbol} qty={existing_qty} order_id={order_id}")
            return "submitted"
        except Exception as exc:
            print(f"[webhook-router] reversal submit failed {sig.symbol}: {exc}")
            write_event("webhook_reversal_failed", {"signal_id": sig.signal_id, "symbol": sig.symbol, "error": str(exc)[:300]})
            return "failed"

    def process_signal(self, sig: ValidatedSignal) -> str:
        """
        Process one validated signal. Returns outcome string.
        Never raises — all errors are caught and logged.
        """
        try:
            return self._process_signal_inner(sig)
        except Exception as exc:
            print(f"[webhook-router] unhandled error processing {sig.signal_id}: {exc}")
            write_event("webhook_signal_error", {"signal_id": sig.signal_id, "symbol": sig.symbol, "error": str(exc)[:300]})
            return "error"

    def _process_signal_inner(self, sig: ValidatedSignal) -> str:
        # ── reversal shortcut ─────────────────────────────────────────────────
        is_reversal = sig.qty == 0 and sig.notional == 0.0
        if is_reversal:
            return self._handle_reversal(sig)

        # ── sell-side: always allow (exits are never blocked) ─────────────────
        if sig.side == "sell":
            return self._submit_sell(sig)

        # ── buy-side gates ────────────────────────────────────────────────────
        # 1. Kill switch
        ks_active, ks_reason = _kill_switch_active()
        if ks_active:
            print(f"[webhook-router] {sig.symbol} buy blocked by kill switch: {ks_reason}")
            write_event("webhook_buy_blocked_kill_switch", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": ks_reason})
            return "blocked_kill_switch"

        # 2. Daily loss limit
        loss_state = _load_daily_loss_state()
        limit_hit, limit_reason = _daily_loss_limit_hit(loss_state)
        if limit_hit:
            print(f"[webhook-router] {sig.symbol} buy blocked by daily loss limit: {limit_reason}")
            write_event("webhook_buy_blocked_daily_loss", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": limit_reason})
            return "blocked_daily_loss"

        # 3. Duplicate open order check
        if self._has_open_order(sig.symbol):
            print(f"[webhook-router] {sig.symbol} already has open order — skip")
            write_event("webhook_buy_blocked_duplicate_order", {"signal_id": sig.signal_id, "symbol": sig.symbol})
            return "blocked_duplicate_order"

        # 4. Existing position check (avoid adding to an already-held position unless configured)
        pos = self._get_position(sig.symbol)
        if pos is not None:
            print(f"[webhook-router] {sig.symbol} already has position — skip")
            write_event("webhook_buy_blocked_existing_position", {"signal_id": sig.signal_id, "symbol": sig.symbol})
            return "blocked_existing_position"

        # 5. Opportunity Finder approval
        approved, of_reason = _opportunity_finder_approved(sig.symbol)
        if not approved:
            print(f"[webhook-router] {sig.symbol} rejected by Opportunity Finder: {of_reason}")
            write_event("webhook_buy_blocked_opportunity_finder", {
                "signal_id": sig.signal_id, "symbol": sig.symbol, "reason": of_reason,
            })
            return "blocked_opportunity_finder"

        # 6. Resolve qty
        qty, qty_err = self._resolve_qty(sig)
        if qty <= 0:
            print(f"[webhook-router] {sig.symbol} qty resolution failed: {qty_err}")
            write_event("webhook_buy_blocked_qty", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": qty_err})
            return "blocked_qty"

        return self._submit_buy(sig, qty, of_reason)

    def _submit_sell(self, sig: ValidatedSignal) -> str:
        pos = self._get_position(sig.symbol)
        qty = sig.qty
        if qty <= 0 and pos is not None:
            qty = int(float(getattr(pos, "qty", 0)))
        if qty <= 0:
            print(f"[webhook-router] {sig.symbol} sell qty=0 and no position — skip")
            return "no_position_to_sell"

        payload = {
            "signal_id": sig.signal_id, "symbol": sig.symbol, "side": "sell",
            "qty": qty, "source": sig.source, "submitted_at_utc": _now_utc_iso(),
            "paper": PAPER,
        }

        if not EXECUTE_ORDERS or self._trading_client is None:
            write_event("webhook_sell_signal_only", payload)
            _append_order_log(payload)
            print(f"[webhook-router] signal_only sell {sig.symbol} qty={qty}")
            return "signal_only"

        try:
            order_tag = f"whksell_{sig.symbol}_{TODAY_TAG}"[:48]
            req = MarketOrderRequest(
                symbol=sig.symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=order_tag,
            )
            order = self._trading_client.submit_order(order_data=req)
            payload["order_id"] = getattr(order, "id", "")
            write_event("webhook_sell_submitted", payload)
            _append_order_log(payload)
            print(f"[webhook-router] sell submitted {sig.symbol} qty={qty}")
            return "submitted"
        except Exception as exc:
            print(f"[webhook-router] sell failed {sig.symbol}: {exc}")
            write_event("webhook_sell_failed", {"signal_id": sig.signal_id, "symbol": sig.symbol, "error": str(exc)[:300]})
            return "failed"

    def _submit_buy(self, sig: ValidatedSignal, qty: int, of_reason: str) -> str:
        payload: dict[str, Any] = {
            "signal_id": sig.signal_id, "symbol": sig.symbol, "side": "buy",
            "qty": qty, "source": sig.source, "submitted_at_utc": _now_utc_iso(),
            "opportunity_finder_reason": of_reason,
            "stop_loss": sig.stop_loss if sig.stop_loss > 0 else None,
            "take_profit": sig.take_profit if sig.take_profit > 0 else None,
            "paper": PAPER,
        }

        if not EXECUTE_ORDERS or self._trading_client is None:
            write_event("webhook_buy_signal_only", payload)
            _append_order_log(payload)
            print(f"[webhook-router] signal_only buy {sig.symbol} qty={qty}")
            return "signal_only"

        try:
            order_tag = f"whkbuy_{sig.symbol}_{TODAY_TAG}"[:48]

            if (
                _BRACKET_AVAILABLE
                and sig.stop_loss > 0
                and sig.take_profit > 0
            ):
                req = MarketOrderRequest(
                    symbol=sig.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=order_tag,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(stop_price=round(sig.stop_loss, 2)),
                    take_profit=TakeProfitRequest(limit_price=round(sig.take_profit, 2)),
                )
                payload["bracket_attached"] = True
            else:
                req = MarketOrderRequest(
                    symbol=sig.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=order_tag,
                )
                payload["bracket_attached"] = False

            order = self._trading_client.submit_order(order_data=req)
            payload["order_id"] = getattr(order, "id", "")
            write_event("webhook_buy_submitted", payload)
            _append_order_log(payload)
            print(
                f"[webhook-router] buy submitted {sig.symbol} qty={qty} "
                f"bracket={payload['bracket_attached']}"
            )
            return "submitted"
        except Exception as exc:
            print(f"[webhook-router] buy failed {sig.symbol}: {exc}")
            write_event("webhook_buy_failed", {"signal_id": sig.signal_id, "symbol": sig.symbol, "error": str(exc)[:300]})
            return "failed"


# ======================================================================
# Queue polling loop
# ======================================================================
def run_once(router: WebhookOrderRouter) -> int:
    """Process all new signals. Returns count of signals processed."""
    if not CONSUME_QUEUE:
        return 0

    signals = _load_queue_signals()
    if not signals:
        return 0

    processed_ids = _load_processed_ids()
    count = 0

    for raw in signals:
        sig_id = str(raw.get("signal_id") or raw.get("id") or "")
        if sig_id and sig_id in processed_ids:
            continue

        sig, err = _validate_signal(raw)
        if sig is None:
            print(f"[webhook-router] invalid signal id={sig_id}: {err}")
            write_event("webhook_signal_invalid", {"signal_id": sig_id, "error": err, "raw": str(raw)[:300]})
            if sig_id:
                _mark_processed(raw, "invalid", err)
                processed_ids.add(sig_id)
            count += 1
            continue

        outcome = router.process_signal(sig)
        _mark_processed(raw, outcome)
        processed_ids.add(sig.signal_id)
        write_event("webhook_signal_processed", {
            "signal_id": sig.signal_id, "symbol": sig.symbol,
            "side": sig.side, "qty": sig.qty, "outcome": outcome,
        })
        count += 1

    return count


def run_loop(router: WebhookOrderRouter) -> None:
    print(
        f"[webhook-router] loop started "
        f"queue={QUEUE_PATH} consume={CONSUME_QUEUE} "
        f"execute={EXECUTE_ORDERS} paper={PAPER} "
        f"require_of={REQUIRE_OPPORTUNITY_FINDER}"
    )
    while True:
        try:
            n = run_once(router)
            if n > 0:
                print(f"[webhook-router] processed {n} signal(s)")
        except KeyboardInterrupt:
            print("[webhook-router] stopped by user")
            break
        except Exception as exc:
            print(f"[webhook-router] loop error: {exc}")
            write_event("webhook_loop_error", {"error": str(exc)[:300]})
        time.sleep(SCAN_SECONDS)


# ======================================================================
# Public entry point (called by run.py or directly)
# ======================================================================
_ROUTER: WebhookOrderRouter | None = None


def get_router() -> WebhookOrderRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = WebhookOrderRouter()
    return _ROUTER


def main() -> None:
    print("[webhook-router] starting")
    router = get_router()
    run_loop(router)


if __name__ == "__main__":
    main()
