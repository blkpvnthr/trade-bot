from __future__ import annotations

from flask import Flask, request, jsonify
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.env_loader import load_instance_env

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

load_instance_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tradingview-webhook")

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")
ALPACA_PAPER = os.getenv("APCA_PAPER", "true").lower() in {"true", "1", "yes"}

TRADINGVIEW_WEBHOOK_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
REQUIRE_WEBHOOK_SECRET = os.getenv("REQUIRE_WEBHOOK_SECRET", "false").lower() in {"1", "true", "yes"}

WEBHOOK_MODE = os.getenv("WEBHOOK_MODE", "direct").lower().strip()  # direct | queue | both
WEBHOOK_EXECUTE_ORDERS = os.getenv("WEBHOOK_EXECUTE_ORDERS", "true").lower() in {"1", "true", "yes"}
WEBHOOK_QUEUE_PATH = Path(
    os.getenv(
        "WEBHOOK_QUEUE_PATH",
        str(PROJECT_ROOT / "daily_system_outputs" / "webhook_order_router" / "signal_queue.jsonl"),
    )
)

MONITOR_POSITIONS = os.getenv("WEBHOOK_MONITOR_POSITIONS", "false").lower() in {"1", "true", "yes"}
MONITOR_INTERVAL_SECONDS = float(os.getenv("MONITOR_INTERVAL_SECONDS", "10"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-1.5"))
MIN_CASH_FLOOR = float(os.getenv("MIN_CASH_FLOOR", "0"))

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError("Missing Alpaca API credentials: APCA_API_KEY_ID/APCA_API_SECRET_KEY")

client = TradingClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET, paper=ALPACA_PAPER)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(symbol: Any) -> str:
    return (
        str(symbol or "")
        .upper()
        .replace("NASDAQ:", "")
        .replace("NYSE:", "")
        .replace("AMEX:", "")
        .replace("ARCA:", "")
        .replace(".", "/")
        .strip()
    )


def normalize_side(action_raw: Any) -> OrderSide:
    action = str(action_raw or "").lower().strip()
    if action in {"buy", "long", "entry", "open_long"}:
        return OrderSide.BUY
    if action in {"sell", "short", "close", "exit", "close_long"}:
        return OrderSide.SELL
    raise ValueError(f"Unsupported side/action: {action_raw}")


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def request_secret(data: dict[str, Any]) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return (
        str(data.get("secret") or "")
        or str(request.headers.get("X-Webhook-Secret") or "")
        or str(request.headers.get("X-TradingView-Secret") or "")
    ).strip()


def validate_optional_secret(data: dict[str, Any]) -> None:
    if not REQUIRE_WEBHOOK_SECRET:
        return
    if not TRADINGVIEW_WEBHOOK_SECRET:
        raise ValueError("REQUIRE_WEBHOOK_SECRET=true but TRADINGVIEW_WEBHOOK_SECRET is missing")
    if request_secret(data) != TRADINGVIEW_WEBHOOK_SECRET:
        raise PermissionError("unauthorized")


def get_account_cash() -> float:
    return float(client.get_account().cash)


def resolve_qty(data: dict[str, Any], side: OrderSide) -> float:
    qty_raw = parse_float(data.get("qty"), default=1.0)

    if qty_raw is None:
        qty_raw = 0.0

    qty = abs(float(qty_raw))

    if qty <= 0:
        if side == OrderSide.SELL:
            symbol = normalize_symbol(data.get("symbol"))
            try:
                position = client.get_open_position(symbol)
                position_qty = abs(float(position.qty))
            except Exception:
                logger.warning("Ignoring SELL webhook because no open position exists | symbol=%s", symbol)
                return 0.0

            if position_qty <= 0:
                logger.warning("Ignoring SELL webhook with empty position | symbol=%s", symbol)
                return 0.0

            return float(position_qty)

        return 1.0

    return float(qty)

def validate_cash_for_buy(symbol: str, qty: float, price: float | None) -> tuple[float, float, float]:
    if price is None or price <= 0:
        raise ValueError('Missing/invalid price for BUY cash check. Send "price": "{{close}}".')

    cash = get_account_cash()
    estimated_notional = qty * price
    projected_cash = cash - estimated_notional

    if projected_cash < MIN_CASH_FLOOR:
        raise ValueError(
            f"Cash guard rejected BUY: symbol={symbol}, cash={cash:.2f}, "
            f"required={estimated_notional:.2f}, projected_cash={projected_cash:.2f}, "
            f"min_cash_floor={MIN_CASH_FLOOR:.2f}"
        )

    return cash, estimated_notional, projected_cash


def submit_market_order(symbol: str, side: OrderSide, qty: float):
    order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    return client.submit_order(order_data=order)


def append_to_queue(signal: dict[str, Any]) -> None:
    WEBHOOK_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WEBHOOK_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(signal, separators=(",", ":")) + "\n")


def normalized_signal_from_payload(data: dict[str, Any]) -> tuple[dict[str, Any], OrderSide, float, float | None]:
    raw_symbol = data.get("symbol")
    if not raw_symbol:
        raise ValueError("Missing symbol field")

    side_value = data.get("side") or data.get("action")
    if not side_value:
        raise ValueError("Missing side/action field")

    symbol = normalize_symbol(raw_symbol)
    side = normalize_side(side_value)
    qty = resolve_qty(data, side)
    price = parse_float(data.get("price"), default=None)

    signal = {
        "signal_id": str(data.get("signal_id") or data.get("id") or f"tv_{uuid.uuid4().hex[:16]}"),
        "symbol": symbol,
        "side": side.value,
        "action": side.value,
        "qty": qty,
        "price": price,
        "source": str(data.get("source") or "tradingview"),
        "timestamp_utc": utc_now_iso(),
        "raw_action": str(side_value),
        "raw_qty": data.get("qty"),
    }

    for optional_key in ("stop_loss", "take_profit", "strategy", "notes"):
        if optional_key in data:
            signal[optional_key] = data.get(optional_key)

    return signal, side, qty, price


def monitor_open_positions() -> None:
    logger.info(
        "Position monitor started | interval=%ss take_profit=%s%% stop_loss=%s%% min_cash_floor=%s",
        MONITOR_INTERVAL_SECONDS,
        TAKE_PROFIT_PCT,
        STOP_LOSS_PCT,
        MIN_CASH_FLOOR,
    )

    while True:
        try:
            account_cash = get_account_cash()
            if account_cash < MIN_CASH_FLOOR:
                logger.critical("Cash floor breached | cash=%.2f min_cash_floor=%.2f", account_cash, MIN_CASH_FLOOR)

            for position in client.get_all_positions():
                symbol = position.symbol
                qty = abs(float(position.qty))
                unrealized_plpc = float(position.unrealized_plpc) * 100

                logger.info(
                    "Monitor | symbol=%s qty=%s unrealized_plpc=%.2f%% cash=%.2f",
                    symbol,
                    qty,
                    unrealized_plpc,
                    account_cash,
                )

                if qty <= 0:
                    continue

                reason = None
                if unrealized_plpc >= TAKE_PROFIT_PCT:
                    reason = f"take_profit_{TAKE_PROFIT_PCT}%"
                elif unrealized_plpc <= STOP_LOSS_PCT:
                    reason = f"stop_loss_{STOP_LOSS_PCT}%"

                if reason:
                    submitted = submit_market_order(symbol=symbol, side=OrderSide.SELL, qty=qty)
                    logger.warning(
                        "Monitor exit submitted | symbol=%s qty=%s reason=%s order_id=%s",
                        symbol,
                        qty,
                        reason,
                        submitted.id,
                    )

        except Exception as exc:
            logger.exception("Position monitor failed: %s", exc)

        time.sleep(MONITOR_INTERVAL_SECONDS)


@app.get("/")
def health():
    try:
        cash = get_account_cash()
    except Exception:
        cash = None

    return jsonify(
        {
            "ok": True,
            "service": "tradingview-webhook",
            "paper": ALPACA_PAPER,
            "mode": WEBHOOK_MODE,
            "execute_orders": WEBHOOK_EXECUTE_ORDERS,
            "queue_path": str(WEBHOOK_QUEUE_PATH),
            "cash": cash,
            "secret_required": REQUIRE_WEBHOOK_SECRET,
            "monitor_positions": MONITOR_POSITIONS,
            "timestamp": utc_now_iso(),
        }
    )


@app.post("/webhook")
def webhook():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "Invalid JSON payload"}), 400

        logger.info("Incoming webhook: %s", data)
        validate_optional_secret(data)

        signal, side, qty, price = normalized_signal_from_payload(data)
        
        qty = resolve_qty(data, side)

        if qty <= 0:
            return jsonify({
                "ok": True,
                "ignored": True,
                "reason": "no_position_to_close"
            }), 200

        queued = False
        submitted = None

        if WEBHOOK_MODE in {"queue", "both"}:
            append_to_queue(signal)
            queued = True
            logger.info("Queued TradingView signal | %s", signal)

        if WEBHOOK_MODE in {"direct", "both"}:
            if side == OrderSide.BUY:
                cash, estimated_notional, projected_cash = validate_cash_for_buy(signal["symbol"], qty, price)
                logger.info(
                    "Cash check passed | symbol=%s qty=%s price=%.4f cash=%.2f estimated_notional=%.2f projected_cash=%.2f",
                    signal["symbol"],
                    qty,
                    price,
                    cash,
                    estimated_notional,
                    projected_cash,
                )

            if WEBHOOK_EXECUTE_ORDERS:
                submitted = submit_market_order(symbol=signal["symbol"], side=side, qty=qty)
                logger.info(
                    "Order submitted | symbol=%s side=%s qty=%s order_id=%s",
                    signal["symbol"],
                    side.value,
                    qty,
                    submitted.id,
                )
            else:
                logger.warning("WEBHOOK_EXECUTE_ORDERS=false; signal accepted but no order submitted")

        return jsonify(
            {
                "ok": True,
                "mode": WEBHOOK_MODE,
                "queued": queued,
                "submitted": submitted is not None,
                "symbol": signal["symbol"],
                "side": side.value,
                "qty": qty,
                "price": price,
                "order_id": str(submitted.id) if submitted is not None else None,
                "signal_id": signal["signal_id"],
            }
        )

    except PermissionError as exc:
        logger.warning("Unauthorized webhook attempt")
        return jsonify({"ok": False, "error": str(exc)}), 401
    except Exception as exc:
        logger.exception("Webhook processing failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    if MONITOR_POSITIONS:
        threading.Thread(target=monitor_open_positions, daemon=True).start()

    logger.info("Starting TradingView webhook server on port 5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
