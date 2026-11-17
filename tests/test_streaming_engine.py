import streaming_engine as se
import trade

def _st(**kw):
    d = dict(symbol="QQQ", today_open=660.0, last_price=661.0, last_event_ts=100.0, held=False, in_flight=False, done_today=False)
    d.update(kw); return se.SymbolState(**d)

def test_state_machine_legal_and_illegal():
    m = se.OrderStateMachine()
    cid = "c1"
    assert m.advance(cid, se.OrderState.SIGNAL_CONFIRMED) is True
    assert m.advance(cid, se.OrderState.RISK_APPROVED) is True
    assert m.advance(cid, se.OrderState.ORDER_SUBMITTING) is True
    assert m.advance(cid, se.OrderState.ORDER_SUBMITTED) is True
    assert m.advance(cid, se.OrderState.BROKER_ACKNOWLEDGED) is True
    assert m.advance(cid, se.OrderState.FILLED) is True
    # illegal: cannot jump back to SUBMITTING from FILLED
    assert m.advance(cid, se.OrderState.ORDER_SUBMITTING) is False
    assert m.state(cid) == se.OrderState.FILLED

def test_state_machine_reject_path():
    m = se.OrderStateMachine()
    m.advance("c2", se.OrderState.SIGNAL_CONFIRMED)
    m.advance("c2", se.OrderState.RISK_APPROVED)
    assert m.advance("c2", se.OrderState.REJECTED) is True

def test_percentile_and_summary():
    assert se.percentile([1,2,3,4,5], 50) == 3
    assert se.percentile([], 50) == 0.0
    t = se.LatencyTracker()
    for v in [10,20,30,40,50]:
        t.record("e2e_ms", v)
    s = t.summary()["e2e_ms"]
    assert s["n"] == 5 and s["p50"] == 30

def test_is_stale():
    assert se.is_stale(100.0, 106.0, 5.0) is True
    assert se.is_stale(100.0, 104.0, 5.0) is False

def test_decide_entry_green_and_blocks():
    cfg = trade.ScannerConfig(); ecfg = trade.ExecutorConfig()
    row = se.decide_streaming_entry(_st(), 824.74, set(), cfg, ecfg)
    assert row is not None and row["symbol"] == "QQQ" and row["side"] == "buy" and 0 < row["delta_shares"] < 1
    assert se.decide_streaming_entry(_st(last_price=659.0), 824.74, set(), cfg, ecfg) is None   # red vs open
    assert se.decide_streaming_entry(_st(held=True), 824.74, set(), cfg, ecfg) is None
    assert se.decide_streaming_entry(_st(in_flight=True), 824.74, set(), cfg, ecfg) is None
    assert se.decide_streaming_entry(_st(done_today=True), 824.74, set(), cfg, ecfg) is None
    assert se.decide_streaming_entry(_st(), 824.74, {"QQQ"}, cfg, ecfg) is None   # no-reentry


# ---------------------------------------------------------------------------
# Task 4: StreamingEngine event wiring (async, fakes - no network)
# ---------------------------------------------------------------------------
import asyncio
import types


class _FakeBroker:
    def __init__(self):
        self.buys = []
        self.sells = []

    def submit_market_buy(self, sym, qty, cid, tif):
        self.buys.append((sym, qty, cid))
        return types.SimpleNamespace(id="o1", status="accepted")

    def submit_market_sell(self, sym, qty, cid, tif):
        self.sells.append((sym, qty, cid))
        return types.SimpleNamespace(id="o2", status="accepted")

    def is_fractionable(self, sym):
        return True


class _FakeExec:
    def _kill_switch_allows_execution(self, eq):
        return (True, "", 0.0)

    def _cash_buffer(self, eq):
        return 100.0

    def _max_order_notional(self, eq):
        return 10_000.0


def _engine(monkeypatch):
    monkeypatch.setattr(trade, "within_eod_flatten_window", lambda *a, **k: False)
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    e = object.__new__(se.StreamingEngine)
    e.config = trade.ExecutorConfig()
    e.exec = _FakeExec()
    e.broker = _FakeBroker()
    e.states = {"QQQ": se.SymbolState(symbol="QQQ", today_open=660.0, last_price=660.0, last_event_ts=0.0)}
    e.sm = se.OrderStateMachine()
    e.latency = se.LatencyTracker()
    e.cid_symbol = {}
    e.cid_side = {}
    e.cid_ts = {}
    e.cid_committed = {}
    e._committed_cash = 0.0
    e._positions = []
    e._gross_before = 0.0
    e._kill_ok = True
    e._kill_reason = ""
    e.inflight_timeout_s = 20.0
    e._acct = types.SimpleNamespace(equity=824.74, cash=824.74, buying_power=824.74)
    e._no_reentry = set()
    e._close_dt = None
    return e


def test_streaming_entry_and_dedupe(monkeypatch):
    async def go():
        e = _engine(monkeypatch)
        e._loop = asyncio.get_running_loop()
        await e.on_trade(types.SimpleNamespace(symbol="QQQ", price=661.0))  # green vs open 660
        assert len(e.broker.buys) == 1
        await e.on_trade(types.SimpleNamespace(symbol="QQQ", price=662.0))  # in_flight -> no dupe
        assert len(e.broker.buys) == 1
    asyncio.run(go())


def test_streaming_exit_on_red(monkeypatch):
    async def go():
        e = _engine(monkeypatch)
        e._loop = asyncio.get_running_loop()
        st = e.states["QQQ"]
        st.held, st.qty, st.avg_entry, st.in_flight = True, 0.1, 661.0, False
        await e.on_trade(types.SimpleNamespace(symbol="QQQ", price=659.0))  # red vs open -> exit
        assert len(e.broker.sells) == 1 and abs(e.broker.sells[0][1] - 0.1) < 1e-9
    asyncio.run(go())


def test_order_update_fill_sets_held(monkeypatch):
    async def go():
        e = _engine(monkeypatch)
        e._loop = asyncio.get_running_loop()
        cid = "c1"
        e.cid_symbol[cid] = "QQQ"
        e.cid_side[cid] = "buy"  # side recorded at submit (not parsed from order obj)
        e.states["QQQ"].in_flight = True
        order = types.SimpleNamespace(client_order_id=cid, side="buy", status="filled")
        await e.on_order_update(types.SimpleNamespace(order=order, event="fill"))
        st = e.states["QQQ"]
        assert st.held and st.done_today and not st.in_flight
    asyncio.run(go())


def test_make_cid_is_unique_same_symbol_side(monkeypatch):
    e = _engine(monkeypatch)
    a = e._make_cid("QQQ", "buy")
    b = e._make_cid("QQQ", "buy")
    assert a != b and len(a) <= 48 and "QQQ" in a


def test_order_update_sell_fill_classified_by_recorded_side(monkeypatch):
    async def go():
        e = _engine(monkeypatch)
        e._loop = asyncio.get_running_loop()
        cid = "c2"
        e.cid_symbol[cid] = "QQQ"
        e.cid_side[cid] = "sell"  # even though a real order.side would stringify oddly
        st = e.states["QQQ"]
        st.held, st.qty, st.in_flight = True, 0.1, True
        # order.side intentionally an enum-like object to prove we do NOT parse it
        order = types.SimpleNamespace(client_order_id=cid, side=types.SimpleNamespace(value="buy"), status="filled")
        await e.on_order_update(types.SimpleNamespace(order=order, event="fill"))
        assert st.held is False and st.qty == 0.0 and st.done_today is True
    asyncio.run(go())


def test_reaper_then_late_buy_fill_not_misclassified(monkeypatch):
    # Regression: reaper releases a cid but must KEEP cid_side, so a late buy
    # terminal update is classified as a buy (held), not defaulted to an exit.
    async def go():
        e = _engine(monkeypatch)
        e._loop = asyncio.get_running_loop()
        cid = "c3"
        e.cid_symbol[cid] = "QQQ"
        e.cid_side[cid] = "buy"
        e.cid_committed[cid] = 74.0
        e._committed_cash = 74.0
        e.cid_ts[cid] = 0.0
        st = e.states["QQQ"]
        st.in_flight = True
        e._release_cid(cid)          # reaper-style release
        st.in_flight = False
        assert cid in e.cid_side      # side retained for a late terminal
        assert e._committed_cash == 0.0
        order = types.SimpleNamespace(client_order_id=cid, side=types.SimpleNamespace(value="buy"), status="filled")
        await e.on_order_update(types.SimpleNamespace(order=order, event="fill"))
        assert st.held is True and st.done_today is True   # BUY, not exit
        assert st.qty == 0.0 and not st.in_flight


def test_gross_exposure_gate_blocks(monkeypatch):
    e = _engine(monkeypatch)
    e._gross_before = 0.72  # at MICRO max_gross_exposure
    ok, why = e._risk_ok({"price": 50.0, "delta_shares": 1.0}, 824.74)
    assert ok is False and why == "exceeds_gross_exposure"
