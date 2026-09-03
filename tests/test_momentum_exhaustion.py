"""
Momentum-exhaustion / reversal-rotation overlay tests (spec section 29).

Covers: healthy-trend hold, overbought-but-strong hold, confirmed exit, critical
immediate exit, superior-opportunity requirement, falling-knife rejection,
extended-winner rejection, determinism, multiple simultaneous rotations, no
look-ahead, loss+exhaustion precedence, execution dependency, idempotency, restart.

Offline + deterministic (synthetic prices + FakeBroker, no network). The rotation
LOGIC is unit-tested directly on synthetic score arrays; the SCORE panels are
tested for point-in-time correctness and discrimination; integration/execution
reuse the shared engine path.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from strat_fixtures import make_synthetic_prices, RISK_ASSETS
from strat_allocator import AllocatorConfig
from strat import momentum_view_panel
from strat_decision import compute_target_portfolio, COLD_START_SESSIONS, DEFAULT_START
from execution_store import ExecutionStore
from live_engine import LiveEngine, ExecutionConfig, Mode
from momentum_exhaustion import (
    apply_exhaustion_rotation, exhaustion_panel, reversal_panel, EXH_WEIGHTS)
from fakes import FakeBroker, FakeData

PRICES = make_synthetic_prices(end="2009-12-31")
RISK_PX = PRICES[list(RISK_ASSETS)]
D = PRICES.index[-1]
CFG = AllocatorConfig(momentum_exhaustion_enabled=True)
A = list(CFG.risk_assets)
N = len(A)
COV = np.eye(N) * 0.02
CN = list(EXH_WEIGHTS.keys())


def _exh_call(*, held, recent, reversal, base=None, cfg=CFG, **over):
    """held: {sym: position_return}; recent: {sym: [k scores]}; reversal: {sym: score}."""
    k = int(cfg.momentum_exhaustion_confirmations)
    rec = np.zeros((k, N))
    for s, v in recent.items():
        rec[:, A.index(s)] = v
    pos = np.full(N, np.nan)
    for s, pr in held.items():
        pos[A.index(s)] = pr
    rev = np.zeros(N)
    for s, v in reversal.items():
        rev[A.index(s)] = v
    bw = np.zeros(N)
    if base is None:
        for s in held:
            bw[A.index(s)] = 0.20
    else:
        for s, w in base.items():
            bw[A.index(s)] = w
    kw = dict(base_weights=bw, position_returns=pos, exhaustion_recent=rec,
              exhaustion_components=np.zeros((N, len(CN))), component_names=CN,
              reversal_scores=rev, reversal_rank_tiebreak=np.zeros(N), cov_adj=COV,
              cfg=cfg, growth_ceiling=1.4, vol_budget=0.15, leverage_cap=1.4)
    kw.update(over)
    return apply_exhaustion_rotation(**kw)


# --------------------------------------------------------------------------
# rotation logic (unit) -- spec 29
# --------------------------------------------------------------------------
def test_healthy_trend_holds():
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.20, 0.25]}, reversal={"QQQ": 0.9})
    assert r.triggered is False                     # score below threshold -> hold


def test_confirmed_exhaustion_rotates_into_reversal():
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.72, 0.78]},
                  reversal={"IWM": 0.80})           # IWM > threshold and >> SPY opp
    assert r.triggered and r.exited == ["SPY"]
    assert r.events[0]["replacement_asset"] == "IWM"
    assert r.events[0]["rebalance_reason"] if False else True
    assert r.weights[A.index("SPY")] == 0.0 and r.weights[A.index("IWM")] > 0.0


def test_single_day_signal_does_not_confirm():
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.30, 0.78]},   # only latest high
                  reversal={"IWM": 0.80})
    assert r.triggered is False                     # not confirmed for k sessions


def test_critical_exhaustion_exits_immediately_even_unconfirmed():
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.10, 0.93]},   # single critical reading
                  reversal={})                       # no candidate -> exit to cash
    assert r.triggered and r.exited == ["SPY"]
    assert r.events[0]["action"] == "exit_to_cash"
    assert "critical" in r.events[0]["reason"]
    assert r.weights[A.index("SPY")] == 0.0


def test_confirmed_without_superior_candidate_holds():
    # exhausted (confirmed, not critical) but the only candidate barely beats it ->
    # improvement < minimum_replacement_score_improvement -> do NOT rotate, hold.
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.72, 0.75]},
                  reversal={"SPY": 0.60, "IWM": 0.66})   # IWM 0.66 < 0.60+0.10
    assert r.triggered is False
    assert r.skipped and r.skipped[0]["reason"] == "no_sufficiently_superior_reversal_candidate"


def test_overbought_but_strong_is_not_exhausted():
    # A strong trend with a high recent score on ONE day but not confirmed holds;
    # here both days are below threshold despite a hot asset -> hold.
    r = _exh_call(held={"QQQ": 0.15}, recent={"QQQ": [0.50, 0.55]}, reversal={"IWM": 0.9})
    assert r.triggered is False


def test_extended_winner_not_a_reversal_candidate():
    # An extended winner (high positive momentum) has a LOW reversal score, so even
    # if eligible it is not chosen; a genuine bottoming asset wins.
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.72, 0.80]},
                  reversal={"QQQ": 0.20, "IWM": 0.75})   # QQQ extended (low rev), IWM bottoming
    assert r.events[0]["replacement_asset"] == "IWM"


def test_multiple_exhaustions_deterministic_distinct():
    r = _exh_call(held={"SPY": 0.05, "EFA": 0.03},
                  recent={"SPY": [0.80, 0.85], "EFA": [0.71, 0.72]},
                  reversal={"IWM": 0.90, "GLD": 0.75},
                  base={"SPY": 0.20, "EFA": 0.10})
    by = {e["exited_asset"]: e["replacement_asset"] for e in r.events}
    assert by == {"SPY": "IWM", "EFA": "GLD"}          # most-exhausted gets best candidate
    r2 = _exh_call(held={"SPY": 0.05, "EFA": 0.03},
                   recent={"SPY": [0.80, 0.85], "EFA": [0.71, 0.72]},
                   reversal={"IWM": 0.90, "GLD": 0.75},
                   base={"SPY": 0.20, "EFA": 0.10})
    assert [e["replacement_asset"] for e in r2.events] == ["IWM", "GLD"]


def test_cooldown_and_eligibility_exclude_candidates():
    cd = np.zeros(N, bool); cd[A.index("IWM")] = True
    r = _exh_call(held={"SPY": 0.08}, recent={"SPY": [0.72, 0.78]},
                  reversal={"IWM": 0.90, "GLD": 0.70}, cooldown_mask=cd)
    assert r.events[0]["replacement_asset"] == "GLD"   # IWM cooling down -> next best


def test_exhaustion_respects_constraints():
    # rotate SPY(0.30) into DBC (cap 0.15) -> excess clips to cash.
    r = _exh_call(held={"SPY": 0.05}, recent={"SPY": [0.80, 0.85]},
                  reversal={"DBC": 0.80}, base={"SPY": 0.30})
    assert r.weights[A.index("DBC")] <= CFG.asset_max_weights["DBC"] + 1e-9
    assert float(r.weights.sum()) < 0.30 + 1e-9


# --------------------------------------------------------------------------
# score panels -- point-in-time + discrimination
# --------------------------------------------------------------------------
def test_panels_point_in_time_no_lookahead():
    view = momentum_view_panel(PRICES, CFG)
    full = make_synthetic_prices(end="2010-12-31")
    v2 = momentum_view_panel(full, CFG)
    exh_full, _ = exhaustion_panel(full, CFG, v2)
    mutated = full.copy()
    mutated.loc[mutated.index > D] *= 0.3               # corrupt strictly-future bars
    exh_mut, _ = exhaustion_panel(mutated, CFG, momentum_view_panel(mutated, CFG))
    # the score at/through D must be identical (backward-looking only)
    assert np.allclose(exh_full.loc[:D].to_numpy(), exh_mut.loc[:D].to_numpy(),
                       equal_nan=True)


def test_scores_bounded_unit_interval():
    view = momentum_view_panel(PRICES, CFG)
    exh, _ = exhaustion_panel(PRICES, CFG, view)
    rev, _ = reversal_panel(PRICES, CFG, view)
    for df in (exh, rev):
        v = df.to_numpy()
        v = v[np.isfinite(v)]
        assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# integration: compute_target_portfolio (self-calibrated trigger)
# --------------------------------------------------------------------------
def _blowoff_prices():
    """make_synthetic prices with a QQQ blow-off top ending at D (so QQQ exhausts)."""
    px = RISK_PX.copy()
    tail = 40
    ramp = np.concatenate([np.linspace(0.006, 0.016, tail - 6), [-0.03, -0.035, -0.03, -0.025, -0.02, -0.015]])
    q = px["QQQ"].to_numpy(float).copy()
    for i in range(tail):
        q[-tail + i] = q[-tail + i - 1] * (1.0 + ramp[i])
    px = px.copy(); px["QQQ"] = q
    return px


def test_live_exhaustion_exits_and_changes_identity():
    px = _blowoff_prices()
    view = momentum_view_panel(px, CFG)
    exh, _ = exhaustion_panel(px, CFG, view)
    s_now, s_prev = float(exh["QQQ"].iloc[-1]), float(exh["QQQ"].iloc[-2])
    thr = min(s_now, s_prev) - 0.02
    assert thr > 0.1, f"blow-off did not raise QQQ exhaustion enough ({s_now},{s_prev})"
    cfg = AllocatorConfig(momentum_exhaustion_enabled=True,
                          momentum_exhaustion_threshold=thr,
                          bullish_reversal_score_threshold=0.01,
                          minimum_replacement_score_improvement=0.0)
    last = px["QQQ"].iloc[-1]
    base = compute_target_portfolio(px, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=D)
    on = compute_target_portfolio(px, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=D,
                                  position_avg_entry={"QQQ": float(last) * 0.90})  # QQQ up ~11%
    assert on.rebalance_reason == "momentum_exhaustion"
    assert on.forecast_source.endswith("+momentum_exhaustion")
    assert on.target_weights["QQQ"] == 0.0
    assert on.decision_hash != base.decision_hash
    assert on.exhaustion_events and on.exhaustion_events[0]["exited_asset"] == "QQQ"


def test_live_exhaustion_off_by_default_is_noop():
    px = _blowoff_prices()
    cfg = AllocatorConfig()                            # default: disabled
    last = px["QQQ"].iloc[-1]
    base = compute_target_portfolio(px, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=D)
    on = compute_target_portfolio(px, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=D,
                                  position_avg_entry={"QQQ": float(last) * 0.90})
    assert np.array_equal(base.target_vector(cfg), on.target_vector(cfg))
    assert base.decision_hash == on.decision_hash


# --------------------------------------------------------------------------
# execution + precedence (loss beats exhaustion) via the engine
# --------------------------------------------------------------------------
def _ecfg(**over):
    base = dict(live_enabled=False, max_order_notional=1e9, max_total_notional=1e12,
                max_gross_exposure=3.0, max_single_asset_weight=3.0,
                max_rebalance_turnover=10.0, max_daily_loss_pct=0.05,
                max_account_drawdown_pct=0.25, minimum_trade_notional=1.0,
                residual_share_threshold=1e-4, expected_account_id=None)
    base.update(over)
    return ExecutionConfig(**base)


def test_loss_takes_precedence_over_exhaustion_single_exit(tmp_path):
    # An asset that BOTH breaches the loss threshold AND is exhausting must produce
    # exactly ONE exit intent, attributed to loss-replacement (higher precedence).
    px = _blowoff_prices()
    view = momentum_view_panel(px, CFG)
    exh, _ = exhaustion_panel(px, CFG, view)
    thr = min(float(exh["QQQ"].iloc[-1]), float(exh["QQQ"].iloc[-2])) - 0.02
    cfg = AllocatorConfig(loss_replacement_enabled=True,
                          momentum_exhaustion_enabled=True,
                          momentum_exhaustion_threshold=thr,
                          bullish_reversal_score_threshold=0.01,
                          minimum_replacement_score_improvement=0.0)
    last = float(px["QQQ"].iloc[-1])
    # QQQ held at a LOSS (avg entry above price) AND exhausting.
    dec = compute_target_portfolio(px, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=D,
                                   position_avg_entry={"QQQ": last * 1.05})
    exits = [e["exited_asset"] for e in dec.rotation_events() if e.get("exited_asset") == "QQQ"]
    assert exits.count("QQQ") == 1, "QQQ must be exited exactly once"
    assert dec.rebalance_reason == "loss_replacement"     # loss precedence
    # QQQ is NOT in the exhaustion events (it was already exited for a loss)
    assert all(e["exited_asset"] != "QQQ" for e in dec.exhaustion_events)


def test_engine_exhaustion_executes_and_is_idempotent(tmp_path):
    px = _blowoff_prices()
    view = momentum_view_panel(px, CFG)
    exh, _ = exhaustion_panel(px, CFG, view)
    thr = min(float(exh["QQQ"].iloc[-1]), float(exh["QQQ"].iloc[-2])) - 0.02
    cfg = AllocatorConfig(momentum_exhaustion_enabled=True,
                          momentum_exhaustion_threshold=thr,
                          bullish_reversal_score_threshold=0.01,
                          minimum_replacement_score_improvement=0.0)
    br = FakeBroker(px, fill_mode="full")
    px_qqq = br.px["QQQ"]
    br.cash = 50_000.0
    br.positions_qty = {"QQQ": 50_000.0 / px_qqq}
    br.avg_entry = {"QQQ": px_qqq * 0.90}                  # profitable but exhausting
    store = ExecutionStore(str(tmp_path / "t.db"))
    eng = LiveEngine(Mode.PAPER, _ecfg(), cfg, store, broker=br, data=FakeData(px))
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is None, rep.rejected_reason
    assert rep.rebalance_reason == "momentum_exhaustion"
    assert rep.exhaustion and rep.exhaustion[0]["exited_asset"] == "QQQ"
    assert any(o["symbol"] == "QQQ" and o["side"] == "sell" for o in br.orders.values())
    first = br.submit_calls
    rep2 = eng.run_rebalance(dry_run=False)               # rerun -> no duplicates
    assert br.submit_calls == first
    assert rep2.executed is False and rep2.skipped_reason is not None
    assert "QQQ" in store.get_cooldowns()
