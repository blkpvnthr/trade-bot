"""alpaca_paper_broker - real Alpaca PAPER-account order placement for the earnings sleeve.

HARD SAFETY (defense in depth; this module physically cannot reach a live account):
  - Reads ONLY dedicated env vars EARNINGS_ALPACA_PAPER_KEY_ID / EARNINGS_ALPACA_PAPER_SECRET_KEY.
    It NEVER reads APCA_API_KEY_ID / APCA_API_SECRET_KEY (which in this repo are LIVE 'AK' keys).
  - Refuses any key that is not a paper key (must start with 'PK'); an 'AK' (live) key is rejected.
  - Refuses if the key equals the live APCA_API_KEY_ID (paranoia guard against copy-paste).
  - Forces paper=True -> the alpaca SDK routes to paper-api.alpaca.markets. There is no code path
    that constructs a live client.
  - Fail-closed: missing/invalid keys -> ready=False -> places NO orders (the caller falls back to
    the in-process simulator).

This is the broker LEG only. Risk limits, the sleeve gate, fail-closed, the kill switch, and order
acceptance are decided UPSTREAM by PaperExecutor; this just places an already-accepted paper BUY.
"""
from __future__ import annotations

import os
from typing import Optional


class AlpacaPaperBroker:
    """Places real orders on an Alpaca PAPER account. Live is structurally impossible here."""

    LIVE_SUPPORTED = False
    PAPER_KEY_PREFIX = "PK"           # paper accounts use PK...; live accounts use AK...
    LIVE_KEY_PREFIX = "AK"

    def __init__(self) -> None:
        self.client = None
        self.ready = False
        self.reason = ""
        key = os.getenv("EARNINGS_ALPACA_PAPER_KEY_ID")
        secret = os.getenv("EARNINGS_ALPACA_PAPER_SECRET_KEY")
        live_key = os.getenv("APCA_API_KEY_ID")

        if not key or not secret:
            self.reason = ("no_paper_keys (set EARNINGS_ALPACA_PAPER_KEY_ID + "
                           "EARNINGS_ALPACA_PAPER_SECRET_KEY to PAPER 'PK' keys)")
            return
        if key.startswith(self.LIVE_KEY_PREFIX) or not key.startswith(self.PAPER_KEY_PREFIX):
            self.reason = f"refusing non-paper key (must start with '{self.PAPER_KEY_PREFIX}'; live blocked)"
            return
        if live_key and key == live_key:
            self.reason = "refusing key identical to the live APCA_API_KEY_ID"
            return
        try:
            from alpaca.trading.client import TradingClient
            self.client = TradingClient(key, secret, paper=True)  # paper endpoint forced
            self.ready = True
            self.reason = "ready (paper)"
        except Exception as e:  # pragma: no cover - network/SDK dependent
            self.reason = f"client_init_failed:{type(e).__name__}:{e}"

    def submit_paper_buy(self, symbol: str, notional: float, client_order_id: Optional[str] = None) -> dict:
        """Place a notional market BUY on the paper account. No-op (placed=False) if not ready."""
        if not self.ready or self.client is None:
            return {"placed": False, "reason": self.reason, "symbol": symbol}
        if notional <= 0:
            return {"placed": False, "reason": "non_positive_notional", "symbol": symbol}
        try:  # pragma: no cover - requires live network + paper keys
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = MarketOrderRequest(
                symbol=symbol, notional=round(float(notional), 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
            order = self.client.submit_order(req)
            return {"placed": True, "order_id": str(getattr(order, "id", "")),
                    "symbol": symbol, "notional": round(float(notional), 2), "mode": "paper"}
        except Exception as e:  # pragma: no cover
            return {"placed": False, "reason": f"submit_failed:{type(e).__name__}:{e}", "symbol": symbol}
