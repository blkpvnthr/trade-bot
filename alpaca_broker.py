#!/usr/bin/env python3
"""
Alpaca brokerage adapter for the V2.9.8 live engine (account / positions / orders).
===================================================================================

Hard isolation rules (never relaxed):

* ``paper`` is passed EXPLICITLY and used verbatim for ``TradingClient(paper=...)``.
  The environment is NEVER inferred from an endpoint and there is NO silent retry
  against the other environment (#19).
* A PK... key may only be used with paper=True; an AK... (live) key may only be
  used with paper=False. A mismatch raises before any network call.
* For live orders the caller may configure ``EXPECTED_ALPACA_ACCOUNT_ID``; the
  retrieved account id is compared and a mismatch BLOCKS live trading (#18). The
  full account number is never logged -- only a masked suffix.

Only simple market orders (qty or notional) are submitted. No shorts, options,
or synthetic leverage.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest, ClosePositionRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

log = logging.getLogger("v298.broker")

# Terminal order states: no further fills will occur. Terminal != "requested
# quantity achieved" -- the engine reconciles filled vs desired separately.
TERMINAL_ORDER_STATES = {"filled", "canceled", "cancelled", "expired", "rejected",
                         "done_for_day", "stopped", "suspended"}


class BrokerError(RuntimeError):
    pass


class AccountMismatchError(BrokerError):
    pass


def _get_credentials() -> tuple[str, str]:
    key = (os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
           or os.getenv("ALPACA_KEY"))
    sec = (os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
           or os.getenv("ALPACA_SECRET"))
    if not key or not sec:
        raise BrokerError("Missing Alpaca credentials (APCA_API_KEY_ID / "
                          "APCA_API_SECRET_KEY).")
    return key, sec


def key_is_paper(key: str | None = None) -> bool:
    """Alpaca paper keys start with 'PK'; live keys start with 'AK'."""
    key = key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or ""
    return key.upper().startswith("PK")


def _mask(value: str | None) -> str:
    if not value:
        return "????"
    return f"...{str(value)[-4:]}"


@dataclass
class AccountSnapshot:
    account_id: str
    account_suffix: str
    status: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    long_market_value: float
    maintenance_margin: float
    last_equity: float
    pattern_day_trader: bool
    daytrade_count: int
    trading_blocked: bool
    account_blocked: bool
    currency: str

    def blocked_reason(self) -> str | None:
        if self.trading_blocked:
            return "trading_blocked"
        if self.account_blocked:
            return "account_blocked"
        if str(self.status).upper() not in ("ACTIVE",):
            return f"account_status={self.status}"
        return None


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class AlpacaBroker:
    def __init__(self, paper: bool):
        key, sec = _get_credentials()
        is_pk = key_is_paper(key)
        if paper and not is_pk:
            raise BrokerError("paper=True but credential is not a PK... paper key "
                              "-- refusing to avoid crossing environments (#19).")
        if (not paper) and is_pk:
            raise BrokerError("paper=False (LIVE) but credential is a PK... paper "
                              "key -- a paper key cannot place live orders.")
        self.paper = bool(paper)
        # Explicit environment; never inferred, never auto-fallback.
        self._client = TradingClient(key, sec, paper=self.paper)
        log.info("v298 broker connected: env=%s", "PAPER" if self.paper else "LIVE")

    # ---- account ---------------------------------------------------------
    def account_snapshot(self) -> AccountSnapshot:
        a = self._client.get_account()
        _status = getattr(a, "status", "")
        return AccountSnapshot(
            account_id=str(getattr(a, "id", "")),
            account_suffix=_mask(getattr(a, "account_number", None)),
            status=str(getattr(_status, "value", _status)),   # enum -> "ACTIVE"
            equity=_f(getattr(a, "equity", 0.0)),
            cash=_f(getattr(a, "cash", 0.0)),
            buying_power=_f(getattr(a, "buying_power", 0.0)),
            portfolio_value=_f(getattr(a, "portfolio_value", 0.0)),
            long_market_value=_f(getattr(a, "long_market_value", 0.0)),
            maintenance_margin=_f(getattr(a, "maintenance_margin", 0.0)),
            last_equity=_f(getattr(a, "last_equity", 0.0)),
            pattern_day_trader=bool(getattr(a, "pattern_day_trader", False)),
            daytrade_count=int(_f(getattr(a, "daytrade_count", 0))),
            trading_blocked=bool(getattr(a, "trading_blocked", False)),
            account_blocked=bool(getattr(a, "account_blocked", False)),
            currency=str(getattr(a, "currency", "USD")),
        )

    def assert_expected_account(self, snapshot: AccountSnapshot,
                                expected_id: str | None) -> None:
        """Block if the connected account id does not match the configured one (#18)."""
        if expected_id and snapshot.account_id != str(expected_id).strip():
            raise AccountMismatchError(
                f"connected account {_mask(snapshot.account_id)} != expected "
                f"{_mask(expected_id)} -- LIVE TRADING BLOCKED: ACCOUNT MISMATCH")

    # ---- positions -------------------------------------------------------
    def positions(self) -> dict[str, dict]:
        """Map symbol -> {qty, market_value, avg_entry_price, current_price}."""
        out: dict[str, dict] = {}
        for p in self._client.get_all_positions():
            sym = str(getattr(p, "symbol", "")).upper()
            out[sym] = {
                "qty": _f(getattr(p, "qty", 0.0)),
                "market_value": _f(getattr(p, "market_value", 0.0)),
                "avg_entry_price": _f(getattr(p, "avg_entry_price", 0.0)),
                "current_price": _f(getattr(p, "current_price", 0.0)),
            }
        return out

    def positions_df(self) -> pd.DataFrame:
        pos = self.positions()
        if not pos:
            return pd.DataFrame(columns=["symbol", "qty", "market_value",
                                         "avg_entry_price", "current_price"])
        rows = [{"symbol": s, **v} for s, v in pos.items()]
        return pd.DataFrame(rows).set_index("symbol")

    # ---- open orders -----------------------------------------------------
    def open_orders(self) -> list[dict]:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        return [self._order_dict(o) for o in self._client.get_orders(req)]

    def recent_orders(self, symbols: list[str] | None = None, limit: int = 200) -> list[dict]:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit,
                               symbols=list(symbols) if symbols else None)
        return [self._order_dict(o) for o in self._client.get_orders(req)]

    # ---- tradability -----------------------------------------------------
    def asset_caps(self, symbol: str) -> dict:
        a = self._client.get_asset(str(symbol).upper())
        return {"tradable": bool(getattr(a, "tradable", False)),
                "fractionable": bool(getattr(a, "fractionable", False)),
                "status": str(getattr(a, "status", ""))}

    # ---- order submission (market only) ----------------------------------
    def submit_market_order(self, symbol: str, side: str, *,
                            qty: float | None = None,
                            notional: float | None = None,
                            client_order_id: str) -> dict:
        if (qty is None) == (notional is None):
            raise BrokerError("submit_market_order needs exactly one of qty/notional")
        order_side = OrderSide.BUY if str(side).lower().startswith("b") else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=str(symbol).upper(),
            side=order_side,
            type="market",
            time_in_force=TimeInForce.DAY,
            qty=(float(qty) if qty is not None else None),
            notional=(round(float(notional), 2) if notional is not None else None),
            client_order_id=client_order_id,
        )
        o = self._client.submit_order(order_data=req)
        return self._order_dict(o)

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        try:
            o = self._client.get_order_by_client_id(client_order_id)
        except Exception:
            return None
        return self._order_dict(o) if o is not None else None

    def get_order(self, broker_order_id: str) -> dict:
        return self._order_dict(self._client.get_order_by_id(broker_order_id))

    def poll_fill(self, broker_order_id: str, *, attempts: int = 30,
                  sleep_sec: float = 2.0) -> dict:
        """Poll until the order reaches a terminal state or attempts run out."""
        last = self.get_order(broker_order_id)
        for _ in range(max(1, attempts)):
            if str(last["status"]).lower() in TERMINAL_ORDER_STATES:
                return last
            time.sleep(sleep_sec)
            last = self.get_order(broker_order_id)
        return last

    # ---- cancel / flatten (explicit operator actions) --------------------
    def cancel_all_open_orders(self) -> int:
        resp = self._client.cancel_orders()
        return len(list(resp)) if resp is not None else 0

    def cancel_order(self, broker_order_id: str) -> None:
        self._client.cancel_order_by_id(broker_order_id)

    def close_position(self, symbol: str, *, qty: float | None = None,
                       percentage: float | None = None) -> dict:
        req = None
        if qty is not None:
            req = ClosePositionRequest(qty=str(qty))
        elif percentage is not None:
            req = ClosePositionRequest(percentage=str(percentage))
        o = self._client.close_position(str(symbol).upper(), close_options=req)
        return self._order_dict(o)

    def close_all_positions(self, cancel_orders: bool = True) -> list[dict]:
        resp = self._client.close_all_positions(cancel_orders=cancel_orders)
        out = []
        for item in (resp or []):
            body = getattr(item, "body", None)
            if body is not None:
                out.append(self._order_dict(body))
        return out

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _order_dict(o) -> dict:
        return {
            "broker_order_id": str(getattr(o, "id", "")),
            "client_order_id": str(getattr(o, "client_order_id", "")),
            "symbol": str(getattr(o, "symbol", "")).upper(),
            "side": str(getattr(o, "side", "")).lower().replace("orderside.", ""),
            "status": str(getattr(o, "status", "")).lower().replace("orderstatus.", ""),
            "filled_qty": _f(getattr(o, "filled_qty", 0.0)),
            "filled_avg_price": _f(getattr(o, "filled_avg_price", 0.0)) or None,
            "qty": _f(getattr(o, "qty", 0.0)) or None,
            "notional": _f(getattr(o, "notional", 0.0)) or None,
        }


def is_order_terminal(status: str) -> bool:
    return str(status).lower() in TERMINAL_ORDER_STATES
