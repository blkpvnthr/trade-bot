import streaming_engine as se

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
