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
