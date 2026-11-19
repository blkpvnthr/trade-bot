"""Tests for order_flow_engine.py -- trade classification, CVD, divergence, pressure,
staleness, duplicate/out-of-order protection, bounded memory, fail-safe REST."""
import order_flow_engine as of


# --------------------------------------------------------------- trade classification
def test_classify_trade_quote_rule():
    assert of.classify_trade(100.10, 100.00, 100.10) == 1    # lifted the offer
    assert of.classify_trade(100.00, 100.00, 100.10) == -1   # hit the bid
    assert of.classify_trade(100.06, 100.00, 100.10) == 1    # above midpoint
    assert of.classify_trade(100.04, 100.00, 100.10) == -1   # below midpoint


def test_classify_trade_midpoint_uses_tick_fallback():
    # exactly at mid -> use last tick direction
    assert of.classify_trade(100.05, 100.00, 100.10, last_tick_sign=1) == 1
    assert of.classify_trade(100.05, 100.00, 100.10, last_tick_sign=-1) == -1
    assert of.classify_trade(100.05, 100.00, 100.10, last_tick_sign=0) == 0


def test_classify_trade_no_quote_and_bad_price():
    assert of.classify_trade(50.0, None, None, last_tick_sign=1) == 1   # tick fallback
    assert of.classify_trade(50.0, None, None, last_tick_sign=0) == 0
    assert of.classify_trade(0.0, 1.0, 2.0) == 0                        # invalid price


# --------------------------------------------------------------- metric primitives
def test_signed_delta_and_volumes_windowed():
    ev = [(0.0, 1, 100), (5.0, -1, 50), (9.0, 1, 200), (50.0, -1, 999)]  # last outside 10s
    assert of.signed_delta(ev, now=10.0, window_s=10) == 100 - 50 + 200
    buy, sell = of.buy_sell_volumes(ev, now=10.0, window_s=10)
    assert buy == 300 and sell == 50


def test_imbalance_ratio_quote_spread():
    assert of.executed_imbalance(300, 100) == 0.5
    assert of.executed_imbalance(0, 0) == 0.0
    assert of.buy_sell_ratio(300, 100) == 3.0
    assert of.buy_sell_ratio(100, 0) == float("inf")
    assert of.quote_imbalance(200, 600) == -0.5
    mid, spread, bps = of.spread_metrics(100.0, 100.10)
    assert mid == 100.05 and round(spread, 2) == 0.10 and bps > 0
    assert of.spread_metrics(100.1, 100.0) == (None, None, None)  # crossed/invalid


# --------------------------------------------------------------- CVD divergence
def test_cvd_divergence_bearish_and_bullish():
    # price rising, CVD falling -> bearish (negative)
    price_up = [(0.0, 100.0), (30.0, 101.0), (60.0, 102.0)]
    cvd_down = [(0.0, 0.0), (30.0, -500.0), (60.0, -1000.0)]
    assert of.cvd_divergence(price_up, cvd_down, now=60.0, lookback_s=120) < 0
    # price falling, CVD rising -> bullish (positive)
    price_dn = [(0.0, 102.0), (30.0, 101.0), (60.0, 100.0)]
    cvd_up = [(0.0, 0.0), (30.0, 500.0), (60.0, 1000.0)]
    assert of.cvd_divergence(price_dn, cvd_up, now=60.0, lookback_s=120) > 0
    # aligned -> ~0
    assert of.cvd_divergence(price_up, cvd_up, now=60.0, lookback_s=120) == 0.0
    # insufficient -> 0
    assert of.cvd_divergence([(0.0, 1.0)], [(0.0, 1.0)], now=1.0, lookback_s=120) == 0.0


def test_classify_pressure_and_normalized_signals():
    assert of.classify_pressure(0.9, 500, 0.5)[0] == "STRONG_BUY"
    assert of.classify_pressure(-0.9, -500, -0.5)[0] == "STRONG_SELL"
    assert of.classify_pressure(0.0, 0, 0.0)[0] == "NEUTRAL"
    # sell pressure in [0,1], higher when more selling
    assert of.norm_sell_pressure(-1.0, -1.0, -100, buy_vol=0, sell_vol=100) > 0.8
    assert of.norm_sell_pressure(1.0, 1.0, 100, buy_vol=100, sell_vol=0) < 0.2
    # cvd deterioration rises with giveback from peak
    assert of.norm_cvd_deterioration(cvd=0.0, cvd_peak=1000.0, delta_30s=-50, delta_300s=200) > 0.5
    assert of.norm_cvd_deterioration(cvd=1000.0, cvd_peak=1000.0, delta_30s=50, delta_300s=200) == 0.0


# --------------------------------------------------------------- accumulator
def _feed_sells(sof, t0, n=12, px=100.0):
    sof.on_quote(t0, bid=100.0, ask=100.1, bid_size=100, ask_size=400)
    for i in range(n):
        sof.on_trade(t0 + i, price=px, size=100)  # at bid -> aggressive sells


def test_snapshot_detects_selling():
    sof = of.SymbolOrderFlow("AAA", of.OrderFlowConfig(min_trades_for_signal=4, stale_trade_s=1e9))
    t0 = 1_000_000.0
    _feed_sells(sof, t0, n=12)
    s = sof.snapshot(now=t0 + 12)
    assert s.ok and s.status == "OK"
    assert s.pressure in ("SELL", "STRONG_SELL")
    assert s.sell_pressure > 0.5
    assert s.cvd < 0 and s.delta_60s < 0
    assert s.data_source == "IEX" and "IEX" in s.caveat


def test_insufficient_data():
    sof = of.SymbolOrderFlow("AAA", of.OrderFlowConfig(min_trades_for_signal=8, stale_trade_s=1e9))
    t0 = 1_000_000.0
    sof.on_quote(t0, 100.0, 100.1, 100, 100)
    for i in range(3):
        sof.on_trade(t0 + i, 100.05, 100)
    s = sof.snapshot(now=t0 + 3)
    assert s.status == "INSUFFICIENT_DATA" and not s.ok


def test_stale_data_detection():
    sof = of.SymbolOrderFlow("AAA", of.OrderFlowConfig(min_trades_for_signal=4, stale_trade_s=5.0))
    t0 = 1_000_000.0
    _feed_sells(sof, t0, n=6)
    s = sof.snapshot(now=t0 + 100)  # last trade 94s ago -> stale
    assert s.status == "STALE" and not s.ok and s.last_trade_age_s > 5.0


def test_duplicate_protection():
    sof = of.SymbolOrderFlow("AAA", of.OrderFlowConfig(min_trades_for_signal=1, stale_trade_s=1e9))
    t0 = 1_000_000.0
    sof.on_quote(t0, 100.0, 100.1, 100, 100)
    sof.on_trade(t0 + 1, 100.0, 100)
    sof.on_trade(t0 + 1, 100.0, 100)  # exact duplicate -> ignored
    assert len(sof._trades) == 1


def test_out_of_order_rejected():
    sof = of.SymbolOrderFlow("AAA", of.OrderFlowConfig(min_trades_for_signal=1, stale_trade_s=1e9,
                                                       out_of_order_tolerance_s=2.0))
    t0 = 1_000_000.0
    sof.on_quote(t0, 100.0, 100.1, 100, 100)
    sof.on_trade(t0 + 10, 100.0, 100)
    sof.on_trade(t0 + 1, 100.0, 50)  # 9s before last, beyond tolerance -> rejected
    assert len(sof._trades) == 1


def test_bounded_memory_trim():
    cfg = of.OrderFlowConfig(min_trades_for_signal=1, stale_trade_s=1e9, retain_s=60)
    sof = of.SymbolOrderFlow("AAA", cfg)
    t0 = 1_000_000.0
    sof.on_quote(t0, 100.0, 100.1, 100, 100)
    for i in range(0, 200, 10):
        sof.on_trade(t0 + i, 100.0, 100)
    # events older than retain_s (60) before the latest (t0+190) are trimmed
    assert all(ts >= (t0 + 190) - 60 for ts, _s, _sz in sof._trades)


# --------------------------------------------------------------- engine + fail-safe REST
def test_engine_ingest_and_snapshot():
    eng = of.OrderFlowEngine(of.OrderFlowConfig(min_trades_for_signal=2, stale_trade_s=1e9))
    t0 = 1_000_000.0
    eng.ingest_quote("AAA", t0, 100.0, 100.1, 100, 100)
    eng.ingest_trade("AAA", t0 + 1, 100.1, 100)   # buy
    eng.ingest_trade("AAA", t0 + 2, 100.1, 100)   # buy
    s = eng.snapshot("AAA", now=t0 + 2)
    assert s.ok and s.buy_vol_60s == 200 and s.sell_pressure < 0.5


def test_poll_rest_failsafe_without_client():
    eng = of.OrderFlowEngine(data_client=None)
    out = eng.poll_rest(["AAA", "BBB"])  # must not raise
    assert set(out.keys()) == {"AAA", "BBB"}
    assert all(snap.status == "INSUFFICIENT_DATA" for snap in out.values())
