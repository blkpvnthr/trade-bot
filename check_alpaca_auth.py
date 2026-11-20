import os

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

load_dotenv(override=True)

key = os.getenv("APCA_API_KEY_ID")
secret = os.getenv("APCA_API_SECRET_KEY")

if not key or not secret:
    raise SystemExit(
        "[ERROR] Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in .env"
    )

print("===== ALPACA AUTHENTICATION CHECK =====")

for name, paper in [("PAPER", True), ("LIVE", False)]:
    try:
        client = TradingClient(
            api_key=key,
            secret_key=secret,
            paper=paper,
        )

        account = client.get_account()

        print(f"[SUCCESS] {name}: Authenticated")
        print(f"          Account status: {account.status}")

    except APIError as exc:
        print(f"[FAILED] {name}: {exc}")

print("===== CHECK COMPLETE =====")
