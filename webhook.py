from flask import Flask, request, jsonify
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET")

client = TradingClient(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    paper=True,
)

@app.post("/webhook")
def webhook():
    data = request.json

    if data.get("secret") != SECRET:
        return jsonify({"error": "unauthorized"}), 401

    symbol = data["symbol"].replace("NASDAQ:", "").replace("NYSE:", "")
    side_raw = data["side"].lower()

    side = OrderSide.BUY if side_raw == "buy" else OrderSide.SELL

    order = MarketOrderRequest(
        symbol=symbol,
        qty=float(data.get("qty", 1)),
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    submitted = client.submit_order(order)

    return jsonify({"ok": True, "order_id": str(submitted.id)})