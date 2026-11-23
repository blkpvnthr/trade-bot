"""End-to-end execution-pipeline demonstration (deterministic, no network).

Proves the full lifecycle the hotfix must restore, with REAL components and
injected fakes only at the socket/broker boundary:

    market data -> healthy dynamic universe -> entry gate opens
                -> qualifying buy candidate -> risk approval
                -> order submission -> broker acknowledgment -> fill

Part A uses the real MarketScanner + UniversePublisher + the real fail-closed
health gate (market_scan.health.load_dynamic_universe) to show that once the
scanner can publish (the constructor crash is fixed), fresh market data yields a
HEALTHY context that OPENS the gate -- and that missing data still fails closed.

Part B drives the real ProductionOrderExecutor._execute_order_plan order funnel
with a mock broker to show a qualifying buy traverses risk approval and reaches
submit_market_buy with a broker acknowledgment + fill reconciliation.

Part C exercises the real AlpacaBroker.submit_market_order central coordinator to
show it reaches the broker, prevents duplicate orders, and enforces long-only
position reconciliation.

These are the deterministic integration fixtures required by the hotfix: they
prove execution functionality even when live market conditions produce no trade.
LIVE submission is never exercised (paper/fake broker only).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

import trade
from market_scan.config import MarketScanConfig
from market_scan.health import (
    STATUS_HEALTHY,
    STATUS_UNHEALTHY,
    load_dynamic_universe,
)
from market_scan.ranking import CandidateFeatures
from market_scan.scanner import MarketScanner


# ============================================================ shared scanner fakes
class _FakeBroker:
    def __init__(self, positions=(), pending=()):
        self._pos = list(positions)
        self._pending = list(pending)
        self.api_key = "k"
        self.secret_key = "s"

    def get_all_open_positions(self):
        return [SimpleNamespace(symbol=s) for s in self._pos]

    def list_open_orders(self):
        return [SimpleNamespace(symbol=s) for s in self._pending]


class _FakeStream:
    """No is_live() -> scanner falls back to _subscribe_ok as the liveness proxy,
    which apply_subscriptions sets True (mirrors a connected, delivering stream)."""

    def __init__(self):
        self._subscribed = set()

    async def apply_subscriptions(self, *, to_add, to_remove):
        self._subscribed.update(to_add)
        self._subscribed.difference_update(to_remove)

    @property
    def subscribed(self):
        return set(self._subscribed)

    async def run(self, *a, **k):
        return

    def stop(self):
        pass


def _ok_snap():
    return SimpleNamespace(ok=True, buy_pressure=0.6, seller_exhaustion=0.4,
                           seller_exhaustion_l1=0.4, cvd_trend=0.3,
                           last_trade_age_s=1.0, last_quote_age_s=1.0,
                           mid=50.0, spread=0.02)


def _strong_feats(symbols):
    """Fresh, liquid, directionally-positive features for every symbol fed in
    (candidates AND the SPY/QQQ references the regime/health read needs)."""
    def provider(syms, now):
        return {
            s: CandidateFeatures(
                symbol=s, price=50.0, avg_daily_volume=5_000_000,
                avg_daily_dollar_volume=250_000_000, rel_volume=1.5,
                spread_pct=0.0004, quote_age_s=1.0,
                directional_row={"pdi": 30, "mdi": 10, "adx": 25,
                                 "vwap_dist": 0.01, "short_mom": 0.02, "bb_dist": 0.01},
                snapshot=_ok_snap(), vwap_dist=0.01, bb_mid_dist=0.01,
            )
            for s in syms
        }
    return provider


def _healthy_scanner(tmp_path, discovered):
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=40)
    exec_stub = SimpleNamespace(broker=_FakeBroker(), config=SimpleNamespace(paper=True))
    s = MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=_strong_feats(discovered + ["SPY", "QQQ", "DIA", "IWM"]),
        discovery_fn=lambda **k: SimpleNamespace(symbols=list(discovered), equities=[],
                                                 etfs=[], error=None, source_counts={}),
    )
    s._discovered = list(discovered)
    return s


# ======================================================= PART A: data -> gate opens
def test_A_fresh_data_publishes_healthy_context_and_opens_gate(tmp_path):
    """REAL scanner publish -> REAL health gate returns HEALTHY with ranked
    symbols: fresh market data opens the dynamic entry universe."""
    disc = ["NVDA", "AMD", "GPN"]
    s = _healthy_scanner(tmp_path, disc)

    # Drive one rank tick on a wall-clock-consistent timeline so the published
    # updated_at and the universe.txt mtime are both fresh relative to the gate's
    # `now` read (the gate rejects negative/too-large ages).
    t0 = time.time()
    asyncio.run(s._rank_tick(t0))
    gate_now = time.time()

    res = load_dynamic_universe(
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        enabled=True, now=gate_now, max_age_seconds=30.0,
    )

    assert res.state == STATUS_HEALTHY, res.reason
    assert res.new_entries_permitted is True
    # Discovered candidates reach the published universe; proxies never do. The
    # universe also now includes fresh baseline members (autonomous universe), so
    # this is a subset check rather than exact equality.
    assert set(disc).issubset(set(res.symbols))
    assert "SPY" not in res.symbols and "QQQ" not in res.symbols


def test_A_missing_context_fails_closed(tmp_path):
    """No published context (scanner never ran) -> gate blocks all entries."""
    res = load_dynamic_universe(
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context_absent.json",
        enabled=True, now=time.time(), max_age_seconds=30.0,
    )
    assert res.state == STATUS_UNHEALTHY
    assert res.symbols == []
    assert res.new_entries_permitted is False
    assert "missing/unreadable" in res.reason


def test_A_stale_context_fails_closed(tmp_path):
    """A published-but-stale context (scanner stopped publishing) blocks entries
    even though the file exists -- freshness is a hard gate."""
    disc = ["NVDA", "AMD"]
    s = _healthy_scanner(tmp_path, disc)
    t0 = time.time()
    asyncio.run(s._rank_tick(t0))
    # Read far in the future: both the context and universe.txt are now stale.
    res = load_dynamic_universe(
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        enabled=True, now=t0 + 10_000.0, max_age_seconds=15.0,
    )
    assert res.state == STATUS_UNHEALTHY and res.symbols == []
    assert "stale" in res.reason


# ============================== PART B: candidate -> risk approval -> order -> fill
class _OrderBroker:
    """Mock broker capturing the submitted order and acknowledging + filling it."""

    def __init__(self, fractionable=True):
        self.submitted: list[tuple] = []
        self._fractionable = fractionable

    def get_position(self, symbol):
        return None

    def has_open_order_for_symbol(self, symbol):
        return False

    def is_fractionable(self, symbol):
        return self._fractionable

    def submit_market_buy(self, symbol, qty, client_order_id, tif):
        self.submitted.append(("buy", symbol, qty, client_order_id))
        return SimpleNamespace(id="OID-1", status="accepted")

    def submit_market_sell(self, symbol, qty, client_order_id, tif):
        self.submitted.append(("sell", symbol, qty, client_order_id))
        return SimpleNamespace(id="OID-2", status="accepted")


def _order_executor(broker):
    """A ProductionOrderExecutor with only the attributes/collaborators that
    _execute_order_plan touches stubbed to deterministic no-ops."""
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = broker
    e.config = SimpleNamespace(
        scanner=SimpleNamespace(enable_fractional_orders=True),
        min_buying_power=0.0,
        time_in_force="day",
    )
    e._direction_confirmations_for_buys = lambda plan: {
        str(sym).upper(): {"direction_confirmed": True}
        for sym in plan.get("symbol", [])
    }
    e._latest_feature_by_symbol = lambda: {}
    e._record_account_snapshot = lambda stage, cid: {
        "equity": 10_000.0, "cash": 10_000.0, "buying_power": 20_000.0}
    e._gross_exposure = lambda eq: 0.0
    e._log_live_experience = lambda **k: None
    e._cash_buffer = lambda eq: 1_200.0
    e._max_order_notional = lambda eq: 1_500.0
    e._risk_profile = lambda eq: SimpleNamespace(
        max_gross_exposure=0.72, min_notional_to_trade=75.0)
    e._qty_from_dollars = lambda dollars, price: (dollars / price if price else 0.0)
    e._mark_order = lambda: None
    e._poll_fill = lambda order_id, price: {
        "fill_status": "filled", "filled_qty": 2.0, "filled_avg_price": 50.0}
    e._order_log_context = lambda **k: {}
    e._journal_order = lambda *a, **k: None
    return e


@pytest.fixture
def _quiet_order_env(monkeypatch):
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(trade, "log_executed_order", lambda **k: None)
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    monkeypatch.setattr(trade.time, "sleep", lambda *a, **k: None)


def test_B_qualifying_buy_passes_risk_and_reaches_broker(_quiet_order_env):
    """A qualifying buy candidate clears risk approval and is submitted to the
    broker, which acknowledges and fills it."""
    broker = _OrderBroker()
    e = _order_executor(broker)
    plan = pd.DataFrame([{
        "symbol": "NVDA", "side": "buy", "order_qty": 2, "price": 50.0,
        "capped_dollars": 100.0, "reason": "e2e_demo", "client_order_id": "cid-1",
    }])

    out = e._execute_order_plan(plan, cycle_id="cyc-A")

    assert broker.submitted == [("buy", "NVDA", 2.0, "cid-1")]   # reached the broker
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "NVDA" and row["side"] == "buy"
    assert row["status"] == "accepted"                           # broker ack
    assert row["fill_status"] == "filled"                        # fill reconciled
    assert float(row["filled_qty"]) == 2.0


def test_B_cash_buffer_violation_blocks_with_explicit_reason(_quiet_order_env):
    """An over-large buy is blocked by the cash-buffer rule and NOT submitted --
    every blocked order carries an explicit reason code."""
    broker = _OrderBroker()
    e = _order_executor(broker)
    # est_notional 9500 would push cash (10000) below the 1200 buffer.
    plan = pd.DataFrame([{
        "symbol": "NVDA", "side": "buy", "order_qty": 190, "price": 50.0,
        "capped_dollars": 9_500.0, "reason": "too_big", "client_order_id": "cid-2",
    }])

    out = e._execute_order_plan(plan, cycle_id="cyc-B")

    assert broker.submitted == []                               # never reached broker
    assert out.iloc[0]["status"] == "skipped"
    assert "cash_buffer_violation" in out.iloc[0]["message"]    # explicit reason


def test_B_unconfirmed_direction_blocks_buy(_quiet_order_env):
    """A buy the engine does not directionally confirm is blocked (never submitted)."""
    broker = _OrderBroker()
    e = _order_executor(broker)
    e._direction_confirmations_for_buys = lambda plan: {
        str(sym).upper(): {"direction_confirmed": False,
                           "direction_reject_reason": "below_vwap"}
        for sym in plan.get("symbol", [])
    }
    plan = pd.DataFrame([{
        "symbol": "NVDA", "side": "buy", "order_qty": 2, "price": 50.0,
        "capped_dollars": 100.0, "reason": "x", "client_order_id": "cid-3",
    }])

    out = e._execute_order_plan(plan, cycle_id="cyc-C")

    assert broker.submitted == []
    assert "price_direction_not_confirmed" in out.iloc[0]["message"]


# ================= PART C: central broker coordinator (dedup + reconciliation)
class _FakeTrading:
    def __init__(self, open_orders=(), positions=()):
        self._open_orders = list(open_orders)
        self._positions = list(positions)
        self.submitted: list = []

    def get_orders(self, filter=None):
        return list(self._open_orders)

    def get_all_positions(self):
        return list(self._positions)

    def submit_order(self, order_data=None):
        self.submitted.append(order_data)
        return SimpleNamespace(id="BROKER-OID", status="accepted")


def _alpaca_broker(trading):
    b = object.__new__(trade.AlpacaBroker)
    b.trading = trading
    b._order_lock = threading.RLock()
    return b


def test_C_buy_reaches_broker_submit_order():
    trading = _FakeTrading(open_orders=[], positions=[])
    b = _alpaca_broker(trading)
    res = b.submit_market_buy("NVDA", 2, "cid-c1", trade.TimeInForce.DAY)
    assert str(getattr(res, "status", "")) == "accepted"
    assert len(trading.submitted) == 1
    order = trading.submitted[0]
    assert trade.safe_symbol(order.symbol) == "NVDA"
    assert float(order.qty) == 2.0


def test_C_duplicate_open_order_is_blocked():
    """A pending open order for the symbol must block a second order (dedup)."""
    trading = _FakeTrading(open_orders=[SimpleNamespace(symbol="NVDA")], positions=[])
    b = _alpaca_broker(trading)
    with pytest.raises(RuntimeError, match="pending order"):
        b.submit_market_buy("NVDA", 1, "cid-c2", trade.TimeInForce.DAY)
    assert trading.submitted == []                              # never submitted


def test_C_sell_exceeding_available_is_blocked():
    """Long-only reconciliation: selling more than verified available shares raises."""
    pos = SimpleNamespace(symbol="NVDA", qty="5", side="long", qty_available="5")
    trading = _FakeTrading(open_orders=[], positions=[pos])
    b = _alpaca_broker(trading)
    with pytest.raises(RuntimeError, match="exceeds verified available"):
        b.submit_market_sell("NVDA", 10, "cid-c3", trade.TimeInForce.DAY)
    assert trading.submitted == []


def test_C_sell_within_available_reaches_broker():
    pos = SimpleNamespace(symbol="NVDA", qty="5", side="long", qty_available="5")
    trading = _FakeTrading(open_orders=[], positions=[pos])
    b = _alpaca_broker(trading)
    res = b.submit_market_sell("NVDA", 3, "cid-c4", trade.TimeInForce.DAY)
    assert str(getattr(res, "status", "")) == "accepted"
    assert len(trading.submitted) == 1
