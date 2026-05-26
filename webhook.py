from flask import Flask, request, jsonify
import os
import logging
import threading
import time
from datetime import datetime, timezone

from config.env_loader import load_instance_env

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_instance_env()

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("tradingview-webhook")

# =============================================================================
# Config
# =============================================================================

MONITOR_INTERVAL_SECONDS = float(os.getenv("MONITOR_INTERVAL_SECONDS", "10"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-1.5"))
MIN_CASH_FLOOR = float(os.getenv("MIN_CASH_FLOOR", "0"))

# =============================================================================
# Flask App
# =============================================================================

app = Flask(__name__)

# =============================================================================
# Environment
# =============================================================================

SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET")
ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")
ALPACA_PAPER = os.getenv("APCA_PAPER", "true").lower() in ["true", "1", "yes"]

if not SECRET:
    raise RuntimeError("Missing TRADINGVIEW_WEBHOOK_SECRET")

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError("Missing Alpaca API credentials")

# =============================================================================
# Alpaca Client
# =============================================================================

client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=ALPACA_PAPER,
)

# =============================================================================
# Helpers
# =============================================================================

def normalize_symbol(symbol: str) -> str:
    return (
        str(symbol)
        .upper()
        .replace("NASDAQ:", "")
        .replace("NYSE:", "")
        .replace("AMEX:", "")
        .replace("ARCA:", "")
        .strip()
    )


def normalize_side(side_raw: str) -> OrderSide:
    side_raw = str(side_raw).lower().strip()

    if side_raw in ["buy", "long"]:
        return OrderSide.BUY

    if side_raw in ["sell", "short", "close", "exit"]:
        return OrderSide.SELL

    raise ValueError(f"Unsupported side/action: {side_raw}")


def parse_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_account_cash() -> float:
    account = client.get_account()
    return float(account.cash)


def validate_cash_for_buy(symbol: str, qty: float, price: float):
    if price is None or price <= 0:
        raise ValueError(
            "Missing or invalid price field for cash check. "
            "Send price from TradingView using \"price\": \"{{close}}\"."
        )

    cash = get_account_cash()
    estimated_notional = qty * price
    projected_cash = cash - estimated_notional

    if projected_cash < MIN_CASH_FLOOR:
        raise ValueError(
            f"Cash guard rejected BUY: symbol={symbol}, "
            f"cash={cash:.2f}, required={estimated_notional:.2f}, "
            f"projected_cash={projected_cash:.2f}, "
            f"min_cash_floor={MIN_CASH_FLOOR:.2f}"
        )

    return cash, estimated_notional, projected_cash


def submit_market_order(symbol: str, side: OrderSide, qty: float):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    return client.submit_order(order)


# =============================================================================
# Position Monitor
# =============================================================================

def monitor_open_positions():
    logger.info(
        f"Position monitor started | "
        f"interval={MONITOR_INTERVAL_SECONDS}s "
        f"take_profit={TAKE_PROFIT_PCT}% "
        f"stop_loss={STOP_LOSS_PCT}% "
        f"min_cash_floor={MIN_CASH_FLOOR}"
    )

    while True:
        try:
            account_cash = get_account_cash()

            if account_cash < MIN_CASH_FLOOR:
                logger.critical(
                    f"Cash floor breached | cash={account_cash:.2f} "
                    f"min_cash_floor={MIN_CASH_FLOOR:.2f}"
                )

            positions = client.get_all_positions()

            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                unrealized_plpc = float(position.unrealized_plpc) * 100

                logger.info(
                    f"Monitor | symbol={symbol} qty={qty} "
                    f"unrealized_plpc={unrealized_plpc:.2f}% "
                    f"cash={account_cash:.2f}"
                )

                if qty <= 0:
                    continue

                should_exit = False
                reason = None

                if unrealized_plpc >= TAKE_PROFIT_PCT:
                    should_exit = True
                    reason = f"take_profit_{TAKE_PROFIT_PCT}%"

                elif unrealized_plpc <= STOP_LOSS_PCT:
                    should_exit = True
                    reason = f"stop_loss_{STOP_LOSS_PCT}%"

                if not should_exit:
                    continue

                logger.warning(
                    f"Exit triggered | symbol={symbol} qty={qty} reason={reason}"
                )

                submitted = submit_market_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=abs(qty),
                )

                logger.warning(
                    f"Exit order submitted | symbol={symbol} qty={abs(qty)} "
                    f"reason={reason} order_id={submitted.id}"
                )

        except Exception as e:
            logger.exception(f"Position monitor failed: {e}")

        time.sleep(MONITOR_INTERVAL_SECONDS)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/")
def health():
    try:
        cash = get_account_cash()
    except Exception:
        cash = None

    return jsonify({
        "ok": True,
        "service": "tradingview-webhook",
        "paper": ALPACA_PAPER,
        "cash": cash,
        "monitor": {
            "interval_seconds": MONITOR_INTERVAL_SECONDS,
            "take_profit_pct": TAKE_PROFIT_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "min_cash_floor": MIN_CASH_FLOOR,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# =============================================================================
# Webhook Endpoint
# =============================================================================

@app.post("/webhook")
def webhook():
    try:
        data = request.get_json(force=True)

        if not isinstance(data, dict):
            return jsonify({
                "ok": False,
                "error": "Invalid JSON payload",
            }), 400

        logger.info(f"Incoming webhook: {data}")

        if data.get("secret") != SECRET:
            logger.warning("Unauthorized webhook attempt")
            return jsonify({
                "ok": False,
                "error": "unauthorized",
            }), 401

        raw_symbol = data.get("symbol")

        if not raw_symbol:
            return jsonify({
                "ok": False,
                "error": "Missing symbol field",
                "received": data,
            }), 400

        symbol = normalize_symbol(raw_symbol)

        side_value = data.get("side") or data.get("action")

        if not side_value:
            return jsonify({
                "ok": False,
                "error": "Missing side/action field",
                "received": data,
            }), 400

        side = normalize_side(side_value)

        qty = parse_float(data.get("qty"), default=1)

        if qty is None:
            return jsonify({
                "ok": False,
                "error": "Invalid qty",
                "received_qty": data.get("qty"),
            }), 400

        if qty <= 0:
            return jsonify({
                "ok": False,
                "error": "qty must be > 0",
                "received_qty": qty,
            }), 400

        price = parse_float(data.get("price"), default=None)

        if side == OrderSide.BUY:
            cash, estimated_notional, projected_cash = validate_cash_for_buy(
                symbol=symbol,
                qty=qty,
                price=price,
            )

            logger.info(
                f"Cash check passed | symbol={symbol} qty={qty} "
                f"price={price:.4f} cash={cash:.2f} "
                f"estimated_notional={estimated_notional:.2f} "
                f"projected_cash={projected_cash:.2f}"
            )

        submitted = submit_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
        )

        logger.info(
            f"Order submitted | symbol={symbol} side={side.value} "
            f"qty={qty} order_id={submitted.id}"
        )

        return jsonify({
            "ok": True,
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "order_id": str(submitted.id),
        })

    except Exception as e:
        logger.exception("Webhook processing failed")
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    monitor_thread = threading.Thread(
        target=monitor_open_positions,
        daemon=True,
    )
    monitor_thread.start()

    logger.info("Starting TradingView webhook server on port 5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )