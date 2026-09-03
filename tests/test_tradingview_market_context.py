"""Unit tests for tradingview_market_context — regime classification, panic gate, fallback,
OPRA contract filter, throttle mapping. No live data required (logic is pure + mocked)."""
import tradingview_market_context as tmc


def base(**kw):
    d = dict(es_trend="flat", nq_trend="flat", rty_trend="flat", ym_trend="flat",
             breadth=0.5, nq_es_rs=0.0, vix_state="calm", vix_vx_backwardation=False)
    d.update(kw)
    return d


# ── req #9: throttle mapping ───────────────────────────────────────────────────────────────
def test_throttle_mapping():
    assert tmc.THROTTLE == {"STRONG_RISK_ON": 1.0, "RISK_ON": 0.75, "NEUTRAL": 0.5,
                            "RISK_OFF": 0.25, "PANIC": 0.0}


# ── req #9: regime classification ──────────────────────────────────────────────────────────
def test_classify_strong_risk_on():
    ctx = tmc.classify(base(es_trend="strong_up", nq_trend="strong_up", rty_trend="strong_up",
                            ym_trend="strong_up", breadth=1.0, nq_es_rs=0.005))
    assert ctx.regime == "STRONG_RISK_ON"
    assert ctx.throttle == 1.0
    assert not ctx.block_new_long_calls


def test_classify_risk_on():
    ctx = tmc.classify(base(es_trend="up", nq_trend="up", breadth=0.5))
    assert ctx.regime == "RISK_ON"
    assert ctx.throttle == 0.75


def test_classify_neutral():
    ctx = tmc.classify(base())            # all flat, calm
    assert ctx.regime == "NEUTRAL"
    assert ctx.throttle == 0.50


def test_classify_risk_off():
    ctx = tmc.classify(base(es_trend="strong_down", nq_trend="strong_down",
                            rty_trend="down", ym_trend="down"))
    assert ctx.regime == "RISK_OFF"
    assert ctx.throttle == 0.25


# ── req #9: panic gate ─────────────────────────────────────────────────────────────────────
def test_classify_panic_blocks_long_calls():
    ctx = tmc.classify(base(vix_state="stress"))
    assert ctx.regime == "PANIC"
    assert ctx.throttle == 0.0
    assert ctx.block_new_long_calls
    assert ctx.max_positions_factor < 1.0
    assert "panic" in ctx.blocked_reason


def test_vix_state_levels():
    assert tmc.vix_state(0.0, False) == "calm"
    assert tmc.vix_state(0.025, False) == "elevated"
    assert tmc.vix_state(0.05, False) == "rising"
    assert tmc.vix_state(0.10, False) == "stress"
    assert tmc.vix_state(0.04, True) == "stress"      # backwardation + VIX rising


def test_panic_gate_vx_backwardation():
    st = tmc.vix_state(0.04, True)                     # VX1>VX2 + VIX rising
    ctx = tmc.classify(base(vix_state=st, vix_vx_backwardation=True))
    assert ctx.regime == "PANIC"
    assert ctx.block_new_long_calls
    assert "backwardation" in ctx.blocked_reason


def test_rising_vix_blocks_without_panic():
    ctx = tmc.classify(base(es_trend="up", nq_trend="up", vix_state="rising"))
    assert ctx.block_new_long_calls                    # req #4: rising VIX blocks new long calls
    assert ctx.regime != "PANIC"                       # but not full panic


# ── req #9: fallback behavior ──────────────────────────────────────────────────────────────
def test_fallback_disabled(monkeypatch):
    monkeypatch.setattr(tmc, "TRADINGVIEW_CONTEXT_ENABLED", False)
    ctx = tmc.get_market_context(force=True)
    assert ctx.regime == "NEUTRAL"
    assert ctx.throttle == 1.0                         # disabled => no throttling
    assert ctx.source == "disabled"


def test_fallback_no_data(monkeypatch):
    monkeypatch.setattr(tmc, "TRADINGVIEW_CONTEXT_ENABLED", True)
    monkeypatch.setattr(tmc, "_alpaca_proxy_metrics", lambda: None)
    ctx = tmc.get_market_context(force=True, av_context={"overall_market_bias": "neutral"})
    assert ctx.regime == "NEUTRAL"
    assert ctx.throttle == 0.50                        # req #7: data unavailable => 0.50
    assert ctx.source in ("alpha_vantage", "fallback")


def test_fallback_bear_blocks(monkeypatch):
    monkeypatch.setattr(tmc, "TRADINGVIEW_CONTEXT_ENABLED", True)
    monkeypatch.setattr(tmc, "_alpaca_proxy_metrics", lambda: None)
    ctx = tmc.get_market_context(force=True, av_context={"overall_market_bias": "bearish"})
    assert ctx.block_new_long_calls


def test_fallback_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("feed down")
    monkeypatch.setattr(tmc, "TRADINGVIEW_CONTEXT_ENABLED", True)
    monkeypatch.setattr(tmc, "_alpaca_proxy_metrics", boom)
    ctx = tmc.get_market_context(force=True)            # must not raise
    assert ctx.regime == "NEUTRAL" and ctx.throttle == 0.50


# ── req #9: OPRA contract filter (long calls only — never naked) ───────────────────────────
GOOD = dict(volume=500, open_interest=100, bid=1.00, ask=1.05, delta=0.55, dte=35, quote_age_sec=10)


def test_opra_filter_pass():
    ok, reason = tmc.opra_contract_ok(GOOD, regime_aligned=True)
    assert ok, reason


def test_opra_filter_rejects():
    assert not tmc.opra_contract_ok({**GOOD, "bid": 0})[0]                       # zero bid
    assert not tmc.opra_contract_ok({**GOOD, "ask": 1.30})[0]                    # spread >10%
    assert not tmc.opra_contract_ok({**GOOD, "delta": 0.40})[0]                  # delta <0.47
    assert not tmc.opra_contract_ok({**GOOD, "bid": 3.00, "ask": 3.05})[0]       # premium >2.50
    assert not tmc.opra_contract_ok({**GOOD, "dte": 10})[0]                      # DTE <21
    assert not tmc.opra_contract_ok({**GOOD, "dte": 75})[0]                      # DTE >60
    assert not tmc.opra_contract_ok({**GOOD, "volume": 50, "open_interest": 100})[0]  # vol<=OI
    assert not tmc.opra_contract_ok({**GOOD, "quote_age_sec": 300})[0]           # stale quote
    assert not tmc.opra_contract_ok(GOOD, regime_aligned=False)[0]              # regime misaligned
