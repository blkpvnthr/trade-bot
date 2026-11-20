"""Tests for amps_v4_regime.py -- instrument registry, directional score, regime classification,
hysteresis, underlying alignment, inverse-ETF directional eligibility, divergence overrides."""
import amps_v4_regime as R


def _cfg(**over):
    import dataclasses
    c = R.RegimeConfig()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _bull():
    return {"pdi": 30, "mdi": 12, "adx": 28, "vwap_dist": 0.002, "vwap_slope": 0.01,
            "bb_dist": 0.001, "short_mom": 0.5, "realized_vol": 0.001}


def _bear():
    return {"pdi": 12, "mdi": 30, "adx": 28, "vwap_dist": -0.002, "vwap_slope": -0.01,
            "bb_dist": -0.001, "short_mom": -0.5, "realized_vol": 0.001}


# ----------------------------------------------------- registry
def test_registry_complete_and_consistent():
    assert len(R.TRADE_UNIVERSE) == 20
    for sym, m in R.REGISTRY.items():
        assert m["dir"] in ("bull", "bear") and m["ref"] and m["pair"]
    # each underlying has exactly one bull and one bear ETF
    for ref in {m["ref"] for m in R.REGISTRY.values()}:
        dirs = sorted(m["dir"] for m in R.REGISTRY.values() if m["ref"] == ref)
        assert dirs == ["bear", "bull"], (ref, dirs)
    assert "SPY" not in R.TRADE_UNIVERSE and "NVDA" not in R.TRADE_UNIVERSE  # refs never traded


# ----------------------------------------------------- directional score
def test_directional_score_sign_and_bounds():
    cfg = _cfg()
    sb = R.directional_score(_bull(), cfg); sr = R.directional_score(_bear(), cfg)
    assert sb > 50 and sr < -50 and -100 <= sb <= 100 and -100 <= sr <= 100
    # ADX scales strength, does not set direction: bearish DI with high ADX stays negative
    strong_bear = {**_bear(), "adx": 45}
    assert R.directional_score(strong_bear, cfg) < 0
    # missing DI -> NaN
    assert R.directional_score({"pdi": None, "mdi": 1}, cfg) != R.directional_score({"pdi": None, "mdi": 1}, cfg)


# ----------------------------------------------------- regime classification
def test_classify_regime_bands():
    cfg = _cfg()
    assert R.classify_regime(70, 0.001, cfg) == R.CONFIRMED_BULLISH
    assert R.classify_regime(40, 0.001, cfg) == R.EMERGING_BULLISH
    assert R.classify_regime(0, 0.001, cfg) == R.SIDEWAYS
    assert R.classify_regime(-40, 0.001, cfg) == R.EMERGING_BEARISH
    assert R.classify_regime(-70, 0.001, cfg) == R.CONFIRMED_BEARISH
    assert R.classify_regime(70, 0.01, cfg) == R.HIGH_VOLATILITY  # high vol overrides
    assert R.classify_regime(float("nan"), 0.001, cfg) == R.UNCERTAIN


def test_market_regime_cross_index():
    cfg = _cfg()
    lab, sc, conf = R.market_regime({"SPY": _bull(), "QQQ": _bull(), "DIA": _bull(), "IWM": _bull()}, cfg)
    assert lab == R.CONFIRMED_BULLISH and conf == 1.0
    lab2, _, conf2 = R.market_regime({"SPY": _bull(), "QQQ": _bear(), "DIA": _bull(), "IWM": _bear()}, cfg)
    assert conf2 < 1.0  # disagreement lowers confidence


# ----------------------------------------------------- hysteresis
def test_regime_hysteresis():
    cfg = _cfg(min_confirm_bars=2)
    st = R.RegimeState(active_label=R.SIDEWAYS)
    assert st.update(R.CONFIRMED_BULLISH, cfg) == R.SIDEWAYS   # 1st candidate, not active
    assert st.update(R.CONFIRMED_BULLISH, cfg) == R.CONFIRMED_BULLISH  # confirmed -> active
    # a single conflicting bar does not flip it
    assert st.update(R.CONFIRMED_BEARISH, cfg) == R.CONFIRMED_BULLISH


# ----------------------------------------------------- eligibility
def test_underlying_alignment():
    cfg = _cfg()
    assert R.underlying_aligned(60, "bull", cfg) and not R.underlying_aligned(60, "bear", cfg)
    assert R.underlying_aligned(-60, "bear", cfg) and not R.underlying_aligned(-60, "bull", cfg)
    assert not R.underlying_aligned(10, "bull", cfg)  # below threshold


def test_inverse_etf_eligibility():
    cfg = _cfg()
    sb = R.directional_score(_bull(), cfg); sr = R.directional_score(_bear(), cfg)
    # SQQQ (bear/QQQ): long only when QQQ bearish + bearish regime
    ok, why = R.eligible_for_long("SQQQ", True, sr, R.CONFIRMED_BEARISH, cfg)
    assert ok and why == "regime_aligned_bear"
    ok2, _ = R.eligible_for_long("SQQQ", True, sb, R.CONFIRMED_BULLISH, cfg)
    assert not ok2  # QQQ bullish -> no SQQQ long
    # TQQQ (bull/QQQ): long when QQQ bullish + bullish regime
    ok3, _ = R.eligible_for_long("TQQQ", True, sb, R.CONFIRMED_BULLISH, cfg)
    assert ok3
    # ETF own gates fail -> never eligible regardless of regime
    ok4, why4 = R.eligible_for_long("TQQQ", False, sb, R.CONFIRMED_BULLISH, cfg)
    assert not ok4 and why4 == "etf_amps_gates_fail"


def test_divergence_override():
    cfg = _cfg()
    # NVDD long when NVDA strongly bearish even though broad regime is bullish
    ok, why = R.eligible_for_long("NVDD", True, -70, R.CONFIRMED_BULLISH, cfg)
    assert ok and why == "divergence_override"
    # weak underlying divergence (below divergence_thresh) does NOT override
    ok2, _ = R.eligible_for_long("NVDD", True, -35, R.CONFIRMED_BULLISH, cfg)
    assert not ok2


def test_uncertain_and_high_vol_block_entries():
    cfg = _cfg()
    assert not R.eligible_for_long("TQQQ", True, 80, R.UNCERTAIN, cfg)[0]
    assert not R.eligible_for_long("TQQQ", True, 80, R.HIGH_VOLATILITY, cfg)[0]
