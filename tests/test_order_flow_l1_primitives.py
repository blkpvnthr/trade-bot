"""Tests for the L1 microstructure primitives added to order_flow_engine.

These are honest, bounded [0,1] heuristic proxies (microprice, absorption,
bid-replenishment, volume-climax) that extend the existing CVD-based exhaustion
signals for the scanner's seller-exhaustion confirmation. L2/iceberg are
UNAVAILABLE on IEX. Tests pin contracts (bounds, direction, zero-data safety)
and backward compatibility — NOT predictive power.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import order_flow_engine as ofe


# ---- microprice ----
def test_microprice_leans_toward_pressure_side():
    # heavy resting bid (demand) -> microprice above the mid (upward pressure)
    m = ofe.microprice(bid=10.0, ask=10.10, bid_size=900, ask_size=100)
    assert m is not None and m > 10.05
    # heavy resting ask (supply) -> below the mid
    m2 = ofe.microprice(bid=10.0, ask=10.10, bid_size=100, ask_size=900)
    assert m2 is not None and m2 < 10.05


def test_microprice_invalid_or_zero_size():
    assert ofe.microprice(0, 10, 1, 1) is None
    assert ofe.microprice(10, 9, 1, 1) is None  # crossed
    assert ofe.microprice(10.0, 10.10, 0, 0) == 10.05  # falls back to mid


def test_microprice_tilt_bounds_and_sign():
    assert ofe.microprice_tilt(10.06, 10.05, 0.10) > 0
    assert ofe.microprice_tilt(10.04, 10.05, 0.10) < 0
    assert -1.0 <= ofe.microprice_tilt(99.0, 10.05, 0.10) <= 1.0
    assert ofe.microprice_tilt(None, 10.05, 0.10) == 0.0
    assert ofe.microprice_tilt(10.05, 10.05, 0.0) == 0.0


# ---- absorption (net selling soaked up without price decline) ----
def test_absorption_high_when_selling_but_price_holds():
    s = ofe.absorption_score(buy_vol=100, sell_vol=900, price_change_pct=0.0)
    assert s > 0.5
    # price fell hard -> no absorption
    s2 = ofe.absorption_score(buy_vol=100, sell_vol=900, price_change_pct=-0.02)
    assert s2 < s
    # net buying -> not absorption of selling
    assert ofe.absorption_score(buy_vol=900, sell_vol=100, price_change_pct=0.0) == 0.0
    assert 0.0 <= s <= 1.0


# ---- bid replenishment ----
def test_bid_replenishment_rewards_refilling_bid():
    # bid size keeps recovering after dips -> high replenishment
    refilling = [(1, 500), (2, 200), (3, 600), (4, 300), (5, 700)]
    assert ofe.replenishment_score(refilling, now=5, window_s=10) > 0.5
    # bid size only bleeds down -> no replenishment
    bleeding = [(1, 700), (2, 500), (3, 300), (4, 100)]
    assert ofe.replenishment_score(bleeding, now=4, window_s=10) == 0.0
    assert ofe.replenishment_score([(1, 500)], now=1, window_s=10) == 0.0


# ---- volume climax ----
def test_volume_climax_detects_spike():
    # 10s window carrying ~5x its proportional share of 300s baseline
    s = ofe.volume_climax_score(short_vol=5000, baseline_vol=30000,
                                short_s=10, baseline_s=300, full_ratio=3.0)
    assert s == 1.0  # 5x expected >= full_ratio
    # normal pace -> no climax
    assert ofe.volume_climax_score(1000, 30000, 10, 300, 3.0) == 0.0
    assert ofe.volume_climax_score(0, 0, 10, 300) == 0.0


def test_seller_exhaustion_l1_composite_bounds():
    s = ofe.seller_exhaustion_l1(absorption=1.0, bid_replenishment=1.0, microprice_tilt=1.0)
    assert s == 1.0
    assert ofe.seller_exhaustion_l1(0, 0, -1) == 0.0
    assert 0.0 <= ofe.seller_exhaustion_l1(0.5, 0.2, 0.1) <= 1.0


# ---- snapshot integration (backward compatible) ----
def _feed_absorption(sf, base_ts):
    # strong resting bid, sellers hitting it, price holding flat
    for i in range(40):
        ts = base_ts + i * 0.2
        sf.on_quote(ts, bid=10.00, ask=10.02, bid_size=5000, ask_size=200)
        sf.on_trade(ts, price=10.00, size=300)  # trades at bid = aggressive sells


def test_snapshot_populates_l1_fields_without_breaking_existing():
    sf = ofe.SymbolOrderFlow("TEST")
    base = 1_000_000.0
    _feed_absorption(sf, base)
    snap = sf.snapshot(now=base + 40 * 0.2)
    # existing contract intact
    assert snap.symbol == "TEST"
    assert hasattr(snap, "seller_exhaustion")
    assert snap.status in ("OK", "STALE", "INSUFFICIENT_DATA")
    # new L1 fields present and bounded
    assert snap.microprice is not None
    assert -1.0 <= snap.microprice_tilt <= 1.0
    assert 0.0 <= snap.absorption <= 1.0
    assert 0.0 <= snap.bid_replenishment <= 1.0
    assert 0.0 <= snap.volume_climax <= 1.0
    assert 0.0 <= snap.seller_exhaustion_l1 <= 1.0
    assert snap.volume_climax_side in (-1, 0, 1)
    # sellers hitting a replenishing bid with flat price -> some absorption signal
    assert snap.absorption > 0.0


def test_prune_evicts_unkept_symbols_bounds_memory():
    e = ofe.OrderFlowEngine()
    for s in ("A", "B", "C"):
        e.ingest_trade(s, 1_000_000.0, 10.0, 5.0)
    assert set(e._symbols) == {"A", "B", "C"}
    evicted = e.prune({"B"})
    assert evicted == 2
    assert set(e._symbols) == {"B"}


def test_snapshot_as_dict_roundtrips_new_fields():
    sf = ofe.SymbolOrderFlow("ZZ")
    d = sf.snapshot(now=1_000_000.0).as_dict()
    for k in ("microprice", "microprice_tilt", "absorption", "bid_replenishment",
              "volume_climax", "seller_exhaustion_l1"):
        assert k in d
