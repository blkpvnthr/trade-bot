"""
In-memory fakes for the V2.9.8 live engine tests (no network, deterministic).

FakeBroker simulates Alpaca account/positions/orders with configurable fill
behavior (full / partial / reject) and a crash injection, updating cash and
positions on each fill so exposure/drift are realistic.
"""
from __future__ import annotations

import pandas as pd

from alpaca_broker import AccountSnapshot, AccountMismatchError

_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "stopped"}


class FakeData:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def get_completed_daily_bars(self, symbols, need, **kw):
        return self.prices[list(symbols)].copy()

    def get_latest_prices(self, symbols):
        last = self.prices.iloc[-1]
        return {s: float(last[s]) for s in symbols}

    def get_next_trading_session(self, after=None):
        base = pd.Timestamp(after) if after is not None else self.prices.index[-1]
        return (base + pd.tseries.offsets.BDay(1)).normalize()

    def get_market_clock(self):
        return {"is_open": False, "timestamp": self.prices.index[-1],
                "timestamp_et": self.prices.index[-1],
                "next_open": self.prices.index[-1], "next_close": self.prices.index[-1]}


class FakeBroker:
    def __init__(self, prices, *, equity=100_000.0, fill_mode="full",
                 partial_fraction=0.4, reject_symbols=(), buying_power=None,
                 last_equity=None, account_id="acct-123", tradable=True,
                 fractionable=True, trading_blocked=False, account_blocked=False,
                 status="ACTIVE", crash_after=None, avg_entry=None):
        self.px = {s: float(prices.iloc[-1][s]) for s in prices.columns if s != "CASH_RATE_3M"}
        self.cash = float(equity)
        self.positions_qty: dict[str, float] = {}
        # Optional per-symbol avg entry price (else current price -> 0% position P&L).
        self.avg_entry: dict[str, float] = dict(avg_entry or {})
        self.fill_mode = fill_mode
        self.partial_fraction = partial_fraction
        self.reject_symbols = set(reject_symbols)
        self._bp = buying_power
        self._last_equity = last_equity
        self.account_id = account_id
        self.tradable = tradable
        self.fractionable = fractionable
        self.trading_blocked = trading_blocked
        self.account_blocked = account_blocked
        self.status = status
        self.crash_after = crash_after
        self.orders: dict[str, dict] = {}
        self.by_broker: dict[str, str] = {}
        self._next_id = 1
        self.submit_calls = 0
        self._symbol_submits: dict[str, int] = {}

    # test helper
    def seed_position(self, symbol, qty):
        self.positions_qty[symbol] = self.positions_qty.get(symbol, 0.0) + float(qty)
        self.cash -= float(qty) * self.px[symbol]

    def _equity(self):
        return self.cash + sum(q * self.px[s] for s, q in self.positions_qty.items())

    def _lmv(self):
        return sum(max(0.0, q) * self.px[s] for s, q in self.positions_qty.items())

    def account_snapshot(self):
        eq = self._equity()
        return AccountSnapshot(
            account_id=self.account_id, account_suffix="...123", status=self.status,
            equity=eq, cash=self.cash,
            buying_power=(self._bp if self._bp is not None else 4 * eq),
            portfolio_value=eq, long_market_value=self._lmv(), maintenance_margin=0.0,
            last_equity=(self._last_equity if self._last_equity is not None else eq),
            pattern_day_trader=False, daytrade_count=0,
            trading_blocked=self.trading_blocked, account_blocked=self.account_blocked,
            currency="USD")

    def assert_expected_account(self, snap, expected_id):
        if expected_id and snap.account_id != str(expected_id):
            raise AccountMismatchError("account mismatch")

    def positions(self):
        return {s: {"qty": q, "market_value": q * self.px[s],
                    "avg_entry_price": float(self.avg_entry.get(s, self.px[s])),
                    "current_price": self.px[s]}
                for s, q in self.positions_qty.items() if abs(q) > 1e-12}

    def asset_caps(self, sym):
        return {"tradable": self.tradable, "fractionable": self.fractionable, "status": "active"}

    def submit_market_order(self, symbol, side, *, qty=None, notional=None, client_order_id):
        self.submit_calls += 1
        if self.crash_after is not None and self.submit_calls > self.crash_after:
            raise RuntimeError("simulated crash during submit")
        price = self.px[symbol]
        req = float(qty) if qty is not None else float(notional) / price
        n_sym = self._symbol_submits.get(symbol, 0)
        self._symbol_submits[symbol] = n_sym + 1

        if symbol in self.reject_symbols:
            filled, stat = 0.0, "rejected"
        elif self.fill_mode == "partial" and n_sym == 0 and "-s" not in client_order_id:
            filled, stat = req * self.partial_fraction, "done_for_day"  # terminal partial
        else:
            filled, stat = req, "filled"

        signed = filled if side.startswith("b") else -filled
        self.positions_qty[symbol] = self.positions_qty.get(symbol, 0.0) + signed
        self.cash -= signed * price
        bid = f"b{self._next_id}"
        self._next_id += 1
        o = {"broker_order_id": bid, "client_order_id": client_order_id,
             "symbol": symbol.upper(), "side": ("buy" if side.startswith("b") else "sell"),
             "status": stat, "filled_qty": filled,
             "filled_avg_price": (price if filled else None),
             "qty": qty, "notional": notional}
        self.orders[client_order_id] = o
        self.by_broker[bid] = client_order_id
        return o

    def get_order_by_client_id(self, coid):
        return self.orders.get(coid)

    def get_order(self, bid):
        return self.orders[self.by_broker[bid]]

    def poll_fill(self, bid, **kw):
        return self.orders[self.by_broker[bid]]

    def open_orders(self):
        return [o for o in self.orders.values() if o["status"] not in _TERMINAL]

    def cancel_all_open_orders(self):
        n = 0
        for o in self.orders.values():
            if o["status"] not in _TERMINAL:
                o["status"] = "canceled"
                n += 1
        return n

    def close_all_positions(self, cancel_orders=True):
        out = []
        for s, q in list(self.positions_qty.items()):
            if abs(q) > 1e-12:
                self.cash += q * self.px[s]
                self.positions_qty[s] = 0.0
                out.append({"symbol": s, "status": "filled"})
        return out
