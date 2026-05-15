from flask import Flask, request, jsonify
import os
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("tradingview-webhook")

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
    paper=True,  # Change to False for live trading
)

# =============================================================================
# Helpers
# =============================================================================

def normalize_symbol(symbol: str) -> str:
    return (
        symbol.upper()
        .replace("NASDAQ:", "")
        .replace("NYSE:", "")
        .replace("AMEX:", "")
        .replace("ARCA:", "")
        .strip()
    )


def normalize_side(side_raw: str) -> OrderSide:
    side_raw = side_raw.lower().strip()

    if side_raw in ["buy", "long"]:
        return OrderSide.BUY

    if side_raw in ["sell", "short", "close"]:
        return OrderSide.SELL

    raise ValueError(f"Unsupported side: {side_raw}")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/")
def health():
    return jsonify({
        "ok": True,
        "service": "tradingview-webhook",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# =============================================================================
# Webhook Endpoint
# =============================================================================

@app.post("/webhook")
def webhook():
    try:
        data = request.get_json(force=True)

        logger.info(f"Incoming webhook: {data}")

        # ---------------------------------------------------------------------
        # Validate Secret
        # ---------------------------------------------------------------------

        if data.get("secret") != SECRET:
            logger.warning("Unauthorized webhook attempt")

            return jsonify({
                "ok": False,
                "error": "unauthorized",
            }), 401

        # ---------------------------------------------------------------------
        # Parse Fields
        # ---------------------------------------------------------------------

        symbol = normalize_symbol(data["symbol"])

        side_value = data.get("side") or data.get("action")

        if not side_value:
            return jsonify({
                "ok": False,
                "error": "Missing side/action field",
                "received": data,
            }), 400

        side = normalize_side(side_value)

        qty = float(data.get("qty", 1))

        # ---------------------------------------------------------------------
        # Submit Alpaca Order
        # ---------------------------------------------------------------------

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        submitted = client.submit_order(order)

        logger.info(
            f"Order submitted | "
            f"symbol={symbol} "
            f"side={side.value} "
            f"qty={qty} "
            f"order_id={submitted.id}"
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
    logger.info("Starting TradingView webhook server on port 5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )