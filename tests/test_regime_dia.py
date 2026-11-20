"""Tests for DIA as a first-class Dow market-context proxy in the AMPS V6
canonical regime engine (amps_v6/regime/regime_engine.py).

Proves:
  - DIA is requested by the regime-data path (a breadth proxy with the dow role)
  - DIA uses the shared classify_proxy_state interface (same as SPY/QQQ/IWM)
  - DIA contributes to aggregate regime scoring
  - DIA disagreement lowers aggregate confidence (and is recorded)
  - a single strong/weak DIA reading does NOT determine the regime
  - stale / missing / malformed DIA becomes UNAVAILABLE (never silently NEUTRAL)
  - missing DIA reduces proxy coverage rather than becoming neutral
  - minimum proxy coverage blocks CONFIRMED_LOCK (and invalidates a held lock)
  - DIA cannot independently authorize an entry (regime gates eligibility only)

Pure/offline. No broker, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amps_v6.regime import regime_engine as RE


def _proxy(pdi, mdi, adx=25, vwap_dist=0.001, short_mom=0.1, rv=0.001):
    return {"pdi": pdi, "mdi": mdi, "adx": adx, "vwap_dist": vwap_dist,
            "vwap_slope": vwap_dist, "bb_dist": vwap_dist, "short_mom": short_mom,
            "realized_vol": rv}


def _bull():
    return _proxy(30, 5, vwap_dist=0.01, short_mom=1)


def _bear():
    return _proxy(5, 30, vwap_dist=-0.01, short_mom=-1)


# ---------------------------------------------------------------- DIA is a proxy
def test_dia_is_requested_as_dow_breadth_proxy():
    assert "DIA" in RE.BREADTH_PROXIES
    assert RE.PROXY_ROLES["DIA"] == "dow_proxy"
    # classify always builds a DIA state (i.e. requests/observes it)
    rows = {p: _bull() for p in RE.BREADTH_PROXIES}
    a = RE.classify(rows)
    assert "DIA" in a.proxy_states


def test_dia_uses_shared_proxy_classifier():
    ps = RE.classify_proxy_state("DIA", _bull())
    assert isinstance(ps, RE.ProxyState)
    assert ps.symbol == "DIA" and ps.role == "dow_proxy"
    assert ps.state == RE.PROXY_BULLISH and ps.available
    # The exact same interface classifies the other index proxies.
    for p in ("SPY", "QQQ", "IWM"):
        other = RE.classify_proxy_state(p, _bull())
        assert isinstance(other, RE.ProxyState) and other.state == RE.PROXY_BULLISH


def test_dia_contributes_to_aggregate_scoring():
    rows = {p: _bull() for p in RE.BREADTH_PROXIES}
    a = RE.classify(rows)
    assert a.available_proxies == 4
    assert a.proxy_states["DIA"].available
    # diagnostics carry DIA's per-proxy contribution
    diag = a.diagnostics()
    assert "DIA" in diag["proxies"]
    assert diag["proxies"]["DIA"]["role"] == "dow_proxy"


# ---------------------------------------------------------------- disagreement
def test_dia_disagreement_lowers_confidence_and_is_recorded():
    agree = {p: _bull() for p in RE.BREADTH_PROXIES}
    conf_agree = RE.classify(agree).confidence

    disagree = dict(agree)
    disagree["DIA"] = _bear()
    res = RE.classify(disagree)

    assert res.confidence < conf_agree
    assert res.conflicting_proxies >= 1
    assert "conflicting" in res.reason


def test_single_dia_reading_does_not_determine_regime():
    # SPY/QQQ/IWM clearly bullish; DIA strongly bearish. DIA alone must not flip
    # the aggregate -- the majority still governs, and disagreement is recorded.
    rows = {p: _bull() for p in ("SPY", "QQQ", "IWM")}
    rows["DIA"] = _proxy(1, 40, vwap_dist=-0.02, short_mom=-2)
    res = RE.classify(rows)
    assert res.conflicting_proxies == 1
    assert res.regime in (RE.BULL, RE.TRANSITION)   # not BEAR on DIA's say-so


# ---------------------------------------------------------------- data health
def test_missing_or_empty_dia_is_unavailable():
    assert RE.classify_proxy_state("DIA", None).state == RE.PROXY_UNAVAILABLE
    assert RE.classify_proxy_state("DIA", {}).state == RE.PROXY_UNAVAILABLE


def test_stale_malformed_future_dia_is_unavailable_not_neutral():
    for flag in ("stale", "future_dated", "duplicated", "invalid", "malformed"):
        row = _bull()
        row[flag] = True
        ps = RE.classify_proxy_state("DIA", row)
        assert ps.state == RE.PROXY_UNAVAILABLE
        assert ps.data_status == "UNAVAILABLE"
        assert ps.reason  # a reason is recorded
    # a row without pdi/mdi cannot score -> UNAVAILABLE, never NEUTRAL
    ps = RE.classify_proxy_state("DIA", {"adx": 25, "vwap_dist": 0.0})
    assert ps.state == RE.PROXY_UNAVAILABLE


def test_declared_data_status_override_marks_unavailable():
    row = _bull()
    row["data_status"] = "STALE"
    assert RE.classify_proxy_state("DIA", row).state == RE.PROXY_UNAVAILABLE


def test_missing_dia_reduces_coverage_not_neutral():
    # Only SPY + QQQ available (DIA + IWM absent) -> coverage 2 < 3 -> INDETERMINATE.
    rows = {"SPY": _bull(), "QQQ": _bull()}
    res = RE.classify(rows)
    assert res.available_proxies == 2
    assert res.coverage_ok is False
    assert res.regime == RE.INDETERMINATE
    # DIA is explicitly UNAVAILABLE in the diagnostics, NOT silently neutral.
    assert res.proxy_states["DIA"].state == RE.PROXY_UNAVAILABLE
    assert res.proxy_states["DIA"].state != RE.PROXY_NEUTRAL


# ---------------------------------------------------------------- coverage gating
def test_min_coverage_blocks_confirmed_and_fast_lock():
    cfg = RE.RegimeConfig()
    st = RE.RegimeState()
    low_cov = RE.RegimeAssessment(RE.BULL, 40.0, 0.9, 1.0, 0.001, "x", coverage_ok=False)

    st.update(low_cov, in_opening_window=True, cfg=cfg)
    assert st.lock_status == RE.PRELIMINARY and not st.is_locked

    st.update(low_cov, in_opening_window=False, cfg=cfg)
    st.update(low_cov, in_opening_window=False, cfg=cfg)
    assert st.lock_status == RE.PRELIMINARY
    assert st.lock_reason == "insufficient_proxy_coverage"


def test_coverage_restored_allows_confirmed_lock():
    cfg = RE.RegimeConfig()
    st = RE.RegimeState()
    good = RE.RegimeAssessment(RE.BULL, 30.0, 0.6, 0.75, 0.001, "x", coverage_ok=True)
    st.update(good, in_opening_window=False, cfg=cfg)  # tick 1 -> PRELIMINARY
    st.update(good, in_opening_window=False, cfg=cfg)  # tick 2 -> CONFIRMED
    assert st.lock_status == RE.CONFIRMED_LOCK


def test_coverage_loss_while_locked_invalidates():
    cfg = RE.RegimeConfig()
    st = RE.RegimeState(lock_status=RE.CONFIRMED_LOCK, regime=RE.BULL, candidate=RE.BULL)
    lost = RE.RegimeAssessment(RE.INDETERMINATE, 0.0, 0.0, 0.0, 0.001, "lost", coverage_ok=False)
    st.update(lost, in_opening_window=False, cfg=cfg)
    assert st.lock_status == RE.INVALIDATED
    assert st.lock_reason == "proxy_coverage_lost"
    assert not st.is_locked


# ---------------------------------------------------------------- no order authority
def test_dia_cannot_independently_authorize_an_order():
    # The regime output only grants ELIGIBILITY; it carries no order instruction,
    # and permissions are identical regardless of any single DIA reading.
    perm = RE.permissions(RE.BULL)
    assert perm["long_universe"] is True
    assert "order" not in perm and "submit" not in perm
    # INDETERMINATE (e.g. DIA-driven coverage loss) forbids new entries entirely.
    assert RE.permissions(RE.INDETERMINATE)["new_entries_allowed"] is False
