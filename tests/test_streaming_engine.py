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
