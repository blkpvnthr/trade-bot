#!/usr/bin/env python3
"""
webhook_order_router.py

Dedicated TradingView/webhook signal consumer and Alpaca order router.

Accepts queue records from webhook.py in either format:

    {
        "symbol": "AAPL",
        "action": "buy" | "sell",
        "qty": 1,
        "price": "205.12"
    }

or normalized:

    {
        "signal_id": "...",
        "symbol": "AAPL",
        "side": "buy" | "sell",
        "qty": 1,
        "notional": 500,
        "stop_loss": 145.0,
        "take_profit": 160.0
    }

Important:
- TradingView strategy.position_size may be negative for exits.
- Alpaca order qty must be positive.
- Exits are never blocked by buy-side approval gates.
- Paper mode is the default unless ALLOW_LIVE_TRADING=1 and APCA_PAPER=false.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

QUEUE_DIR = PROJECT_ROOT / "daily_system_outputs" / "webhook_order_router"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QUEUE_PATH = QUEUE_DIR / "signal_queue.jsonl"
PROCESSED_QUEUE_PATH = QUEUE_DIR / f"{DATA_TAG}_processed_signals.jsonl"
ORDERS_LOG_PATH = QUEUE_DIR / f"{DATA_TAG}_orders.csv"
DAILY_LOSS_STATE_PATH = QUEUE_DIR / f"{DATA_TAG}_daily_loss_state.json"
OPPORTUNITY_FINDER_APPROVAL_PATH = OUT_DIR / f"{DATA_TAG}_opportunity_finder_approvals.json"

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[webhook-router] alpaca-py not available; orders will be logged only")


def _bool(key: str, default: str) -> bool:
    return os.getenv(key, default).lower() in {"1", "true", "yes"}


def _float(key: str, default: float | str) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int | str) -> int:
    return int(os.getenv(key, str(default)))


CONSUME_QUEUE = _bool("CONSUME_WEBHOOK_ORDER_QUEUE", "true")
DIRECT_ROUTING = _bool("WEBHOOK_DIRECT_ORDER_ROUTING", "true")
REQUIRE_OPPORTUNITY_FINDER = _bool("REQUIRE_OPPORTUNITY_FINDER_APPROVAL", "false")
PAPER = _bool("APCA_PAPER", "true") or not _bool("ALLOW_LIVE_TRADING", "0")
EXECUTE_ORDERS = _bool("WEBHOOK_EXECUTE_ORDERS", "true")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")

QUEUE_PATH = Path(os.getenv("WEBHOOK_QUEUE_PATH", str(DEFAULT_QUEUE_PATH)))
SCAN_SECONDS = _int("WEBHOOK_ROUTER_SCAN_SECONDS", "5")
MAX_ORDER_DOLLARS = _float("WEBHOOK_MAX_ORDER_DOLLARS", "3000.0")
MIN_ORDER_DOLLARS = _float("WEBHOOK_MIN_ORDER_DOLLARS", "50.0")
DAILY_LOSS_LIMIT_DOLLARS = _float("WEBHOOK_DAILY_LOSS_LIMIT_DOLLARS", "500.0")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def write_event(event: str, payload: dict[str, Any]) -> None:
    try:
        row = {"event": event, "timestamp_utc": _now_utc_iso(), **payload}
        with EVENTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
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


def _safe_symbol(symbol: Any) -> str:
    return (
        str(symbol or "")
        .strip()
        .upper()
        .replace("NASDAQ:", "")
        .replace("NYSE:", "")
        .replace("AMEX:", "")
        .replace("ARCA:", "")
        .replace(".", "/")
    )


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_side(raw: Any) -> str:
    value = str(raw or "").lower().strip()
    if value in {"buy", "long", "entry", "open_long"}:
        return "buy"
    if value in {"sell", "short", "close", "exit", "close_long"}:
        return "sell"
    raise ValueError(f"invalid_side_or_action: {raw}")


def _kill_switch_active() -> tuple[bool, str]:
    try:
        if not KILL_SWITCH_PATH.exists():
            return False, ""
        state = json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
        if state.get("kill_switch_active"):
            return True, state.get("reason", "kill_switch_file")
    except Exception:
        pass
    return False, ""


def _load_daily_loss_state() -> dict[str, Any]:
    try:
        if DAILY_LOSS_STATE_PATH.exists():
            return json.loads(DAILY_LOSS_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"realized_pnl": 0.0, "date": TODAY_TAG, "trades": 0}


def _daily_loss_limit_hit(state: dict[str, Any]) -> tuple[bool, str]:
    if state.get("date") != TODAY_TAG:
        return False, ""
    pnl = float(state.get("realized_pnl", 0.0))
    if pnl <= -DAILY_LOSS_LIMIT_DOLLARS:
        return True, f"daily_loss_dollars: realized_pnl={pnl:.2f} limit={DAILY_LOSS_LIMIT_DOLLARS:.2f}"
    return False, ""


def _opportunity_finder_approved(symbol: str) -> tuple[bool, str]:
    if DIRECT_ROUTING or not REQUIRE_OPPORTUNITY_FINDER:
        return True, "direct_routing"
    if not OPPORTUNITY_FINDER_APPROVAL_PATH.exists():
        return False, "opportunity_finder_approval_file_missing"
    try:
        data = json.loads(OPPORTUNITY_FINDER_APPROVAL_PATH.read_text(encoding="utf-8"))
        approvals = {_safe_symbol(k): v for k, v in data.get("approvals", {}).items()}
        entry = approvals.get(_safe_symbol(symbol), {})
        if isinstance(entry, bool):
            return entry, "approved" if entry else "not_approved"
        approved = bool(entry.get("approved", False))
        return approved, entry.get("reason", "approved" if approved else "not_approved")
    except Exception as exc:
        return False, f"opportunity_finder_read_error: {exc}"


def _load_queue_signals() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return []
    signals: list[dict[str, Any]] = []
    try:
        for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    signals.append(parsed)
            except Exception:
                write_event("webhook_queue_bad_json", {"line": line[:300]})
        return signals
    except Exception as exc:
        print(f"[webhook-router] queue read error: {exc}")
        return []


def _load_processed_ids() -> set[str]:
    if not PROCESSED_QUEUE_PATH.exists():
        return set()
    ids: set[str] = set()
    try:
        for line in PROCESSED_QUEUE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                sig_id = rec.get("signal_id", "")
                if sig_id:
                    ids.add(str(sig_id))
            except Exception:
                pass
    except Exception:
        pass
    return ids


def _mark_processed(signal: dict[str, Any], outcome: str, detail: str = "") -> None:
    try:
        rec = {
            "signal_id": signal.get("signal_id") or signal.get("id") or "",
            "symbol": signal.get("symbol", ""),
            "side": signal.get("side") or signal.get("action") or "",
            "outcome": outcome,
            "detail": detail,
            "processed_at_utc": _now_utc_iso(),
        }
        with PROCESSED_QUEUE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:
        pass


@dataclass
class ValidatedSignal:
    signal_id: str
    symbol: str
    side: str
    qty: int
    notional: float
    price: float
    stop_loss: float
    take_profit: float
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


def _validate_signal(raw: dict[str, Any]) -> tuple[ValidatedSignal | None, str]:
    sig_id = str(raw.get("signal_id") or raw.get("id") or f"tv_{uuid.uuid4().hex[:16]}")

    symbol = _safe_symbol(raw.get("symbol", ""))
    if not symbol:
        return None, "missing_symbol"
    if not symbol.replace("/", "").isalnum():
        return None, f"invalid_symbol: {symbol}"

    try:
        side = _normalize_side(raw.get("side") or raw.get("action"))
    except ValueError as exc:
        return None, str(exc)

    qty_raw = raw.get("qty", None)
    notional_raw = raw.get("notional", None)

    if qty_raw is not None:
        qty_float = _parse_float(qty_raw, None)
        if qty_float is None:
            return None, f"invalid_qty: {qty_raw}"
        qty = int(abs(qty_float))
        notional = 0.0
    elif notional_raw is not None:
        notional = _parse_float(notional_raw, None)
        if notional is None or notional < 0:
            return None, f"invalid_notional: {notional_raw}"
        qty = 0
    else:
        return None, "missing_qty_or_notional"

    price = float(_parse_float(raw.get("price"), 0.0) or 0.0)
    stop_loss = float(_parse_float(raw.get("stop_loss"), 0.0) or 0.0)
    take_profit = float(_parse_float(raw.get("take_profit"), 0.0) or 0.0)
    source = str(raw.get("source", "tradingview"))

    return (
        ValidatedSignal(
            signal_id=sig_id,
            symbol=symbol,
            side=side,
            qty=qty,
            notional=notional,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            source=source,
            raw=raw,
        ),
        "",
    )


class WebhookOrderRouter:
    def __init__(self) -> None:
        self._trading_client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if not ALPACA_AVAILABLE:
            return
        if not APCA_KEY or not APCA_SECRET:
            print("[webhook-router] Alpaca credentials not found; orders disabled")
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
            for order in self._trading_client.get_orders():
                if _safe_symbol(getattr(order, "symbol", "")) != _safe_symbol(symbol):
                    continue
                status = str(getattr(order, "status", "")).lower()
                if status in {"new", "accepted", "pending_new", "partially_filled", "held"}:
                    return True
        except Exception:
            pass
        return False

    def _get_equity(self) -> float:
        if self._trading_client is None:
            return 0.0
        try:
            return float(getattr(self._trading_client.get_account(), "equity", 0.0))
        except Exception:
            return 0.0

    def _resolve_qty(self, sig: ValidatedSignal) -> tuple[int, str]:
        if sig.qty > 0:
            return sig.qty, ""

        if sig.side == "sell":
            pos = self._get_position(sig.symbol)
            if pos is not None:
                return max(0, int(abs(float(getattr(pos, "qty", 0))))), ""
            return 0, "sell_qty_zero_no_position"

        if sig.notional <= 0:
            return 0, "buy_qty_zero_or_notional_zero"

        max_notional = min(MAX_ORDER_DOLLARS, sig.notional)
        equity = self._get_equity()
        if equity > 0:
            max_notional = min(max_notional, equity * 0.10)

        if max_notional < MIN_ORDER_DOLLARS:
            return 0, f"notional_below_min: {max_notional:.2f} < {MIN_ORDER_DOLLARS:.2f}"

        if sig.price > 0:
            return max(1, int(max_notional / sig.price)), ""

        return 0, "cannot_resolve_notional_without_price"

    def process_signal(self, sig: ValidatedSignal) -> str:
        try:
            return self._process_signal_inner(sig)
        except Exception as exc:
            print(f"[webhook-router] unhandled error processing {sig.signal_id}: {exc}")
            write_event("webhook_signal_error", {"signal_id": sig.signal_id, "symbol": sig.symbol, "error": str(exc)[:300]})
            return "error"

    def _process_signal_inner(self, sig: ValidatedSignal) -> str:
        if sig.side == "sell":
            return self._submit_sell(sig)

        ks_active, ks_reason = _kill_switch_active()
        if ks_active:
            print(f"[webhook-router] {sig.symbol} buy blocked by kill switch: {ks_reason}")
            write_event("webhook_buy_blocked_kill_switch", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": ks_reason})
            return "blocked_kill_switch"

        limit_hit, limit_reason = _daily_loss_limit_hit(_load_daily_loss_state())
        if limit_hit:
            print(f"[webhook-router] {sig.symbol} buy blocked by daily loss limit: {limit_reason}")
            write_event("webhook_buy_blocked_daily_loss", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": limit_reason})
            return "blocked_daily_loss"

        if self._has_open_order(sig.symbol):
            print(f"[webhook-router] {sig.symbol} already has an open order; skip")
            write_event("webhook_buy_blocked_duplicate_order", {"signal_id": sig.signal_id, "symbol": sig.symbol})
            return "blocked_duplicate_order"

        if self._get_position(sig.symbol) is not None:
            print(f"[webhook-router] {sig.symbol} already has a position; skip")
            write_event("webhook_buy_blocked_existing_position", {"signal_id": sig.signal_id, "symbol": sig.symbol})
            return "blocked_existing_position"

        approved, approval_reason = _opportunity_finder_approved(sig.symbol)
        if not approved:
            print(f"[webhook-router] {sig.symbol} rejected by Opportunity Finder: {approval_reason}")
            write_event("webhook_buy_blocked_opportunity_finder", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": approval_reason})
            return "blocked_opportunity_finder"

        qty, qty_error = self._resolve_qty(sig)
        if qty <= 0:
            print(f"[webhook-router] {sig.symbol} qty resolution failed: {qty_error}")
            write_event("webhook_buy_blocked_qty", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": qty_error})
            return "blocked_qty"

        return self._submit_buy(sig, qty, approval_reason)

    def _submit_sell(self, sig: ValidatedSignal) -> str:
        qty, qty_error = self._resolve_qty(sig)
        if qty <= 0:
            print(f"[webhook-router] {sig.symbol} no position/qty to sell: {qty_error}")
            write_event("webhook_sell_no_position", {"signal_id": sig.signal_id, "symbol": sig.symbol, "reason": qty_error})
            return "no_position_to_sell"

        payload = {
            "signal_id": sig.signal_id,
            "symbol": sig.symbol,
            "side": "sell",
            "qty": qty,
            "source": sig.source,
            "submitted_at_utc": _now_utc_iso(),
            "paper": PAPER,
        }

        if not EXECUTE_ORDERS or self._trading_client is None:
            write_event("webhook_sell_signal_only", payload)
            _append_order_log(payload)
            print(f"[webhook-router] signal_only sell {sig.symbol} qty={qty}")
            return "signal_only"

        try:
            req = MarketOrderRequest(
                symbol=sig.symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"whksell_{sig.symbol}_{uuid.uuid4().hex[:8]}"[:48],
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

    def _submit_buy(self, sig: ValidatedSignal, qty: int, approval_reason: str) -> str:
        payload: dict[str, Any] = {
            "signal_id": sig.signal_id,
            "symbol": sig.symbol,
            "side": "buy",
            "qty": qty,
            "source": sig.source,
            "submitted_at_utc": _now_utc_iso(),
            "approval_reason": approval_reason,
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
            kwargs: dict[str, Any] = {
                "symbol": sig.symbol,
                "qty": qty,
                "side": OrderSide.BUY,
                "type": OrderType.MARKET,
                "time_in_force": TimeInForce.DAY,
                "client_order_id": f"whkbuy_{sig.symbol}_{uuid.uuid4().hex[:8]}"[:48],
            }

            if sig.stop_loss > 0 and sig.take_profit > 0:
                kwargs.update(
                    {
                        "order_class": OrderClass.BRACKET,
                        "stop_loss": StopLossRequest(stop_price=round(sig.stop_loss, 2)),
                        "take_profit": TakeProfitRequest(limit_price=round(sig.take_profit, 2)),
                    }
                )
                payload["bracket_attached"] = True
            else:
                payload["bracket_attached"] = False

            req = MarketOrderRequest(**kwargs)
            order = self._trading_client.submit_order(order_data=req)
            payload["order_id"] = getattr(order, "id", "")
            write_event("webhook_buy_submitted", payload)
            _append_order_log(payload)
            print(f"[webhook-router] buy submitted {sig.symbol} qty={qty} bracket={payload['bracket_attached']}")
            return "submitted"
        except Exception as exc:
            print(f"[webhook-router] buy failed {sig.symbol}: {exc}")
            write_event("webhook_buy_failed", {"signal_id": sig.signal_id, "symbol": sig.symbol, "error": str(exc)[:300]})
            return "failed"


def run_once(router: WebhookOrderRouter) -> int:
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
            _mark_processed(raw, "invalid", err)
            if sig_id:
                processed_ids.add(sig_id)
            count += 1
            continue

        outcome = router.process_signal(sig)
        raw.setdefault("signal_id", sig.signal_id)
        _mark_processed(raw, outcome)
        processed_ids.add(sig.signal_id)
        write_event(
            "webhook_signal_processed",
            {"signal_id": sig.signal_id, "symbol": sig.symbol, "side": sig.side, "qty": sig.qty, "outcome": outcome},
        )
        count += 1

    return count


def run_loop(router: WebhookOrderRouter) -> None:
    print(
        f"[webhook-router] loop started queue={QUEUE_PATH} consume={CONSUME_QUEUE} "
        f"execute={EXECUTE_ORDERS} paper={PAPER} direct_routing={DIRECT_ROUTING} "
        f"require_of={REQUIRE_OPPORTUNITY_FINDER}"
    )
    while True:
        try:
            processed = run_once(router)
            if processed:
                print(f"[webhook-router] processed {processed} signal(s)")
        except KeyboardInterrupt:
            print("[webhook-router] stopped by user")
            break
        except Exception as exc:
            print(f"[webhook-router] loop error: {exc}")
            write_event("webhook_loop_error", {"error": str(exc)[:300]})
        time.sleep(SCAN_SECONDS)


_ROUTER: WebhookOrderRouter | None = None


def get_router() -> WebhookOrderRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = WebhookOrderRouter()
    return _ROUTER


def main() -> None:
    print("[webhook-router] starting")
    run_loop(get_router())


if __name__ == "__main__":
    main()
