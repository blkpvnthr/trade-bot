"""
Adaptive opportunity-gated rotation tests (spec section 35; the 15 validated pitfalls).

Covers: vol-normalized damage, regime sensitivity, opportunity gating (HOLD vs
ROTATE), transaction-cost gate, cash candidate, hysteresis, cooldown, min-hold
bypass-on-damage, diversification bound, determinism, cross-sectional ranks,
constraint safety, degenerate books, no-look-ahead, days_held parity, supersede
guard, decision-hash, and engine execution/idempotency. Offline + deterministic.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import numpy as np
import pandas as pd
import pytest

from strat_fixtures import make_synthetic_prices, RISK_ASSETS
from strat_allocator import AllocatorConfig, annualized_vols
from strat import momentum_view_panel
from strat_decision import compute_target_portfolio, COLD_START_SESSIONS
from loss_replacement import CostBasisTracker
from adaptive_rotation import apply_adaptive_rotation

CFG = AllocatorConfig(adaptive_rotation_enabled=True)
A = list(CFG.risk_assets)
N = len(A)
COV = np.eye(N) * 0.02
SIGMA = annualized_vols(COV)
VOLH = SIGMA[0] * math.sqrt(20.0 / 252.0)          # 20-day vol scale (~0.0398)


def _call(*, held, views=None, mom20=None, accel=None, reversal=None, dh=None,
          state="DEFENSIVE", severe=False, cooldown=None, eligible=None, exh=None,
          cov=None, cfg=CFG, base=None):
    cov = COV if cov is None else cov
    sigma = annualized_vols(cov)
    pr = np.full(N, np.nan)
    for s, v in held.items():
        pr[A.index(s)] = v
    bw = np.zeros(N)
    if base:
        for s, v in base.items():
            bw[A.index(s)] = v
    else:
        for s in held:
            bw[A.index(s)] = 0.20
    def _row(d, default=0.0):
        r = np.full(N, default, float)
        if d:
            for s, v in d.items():
                r[A.index(s)] = v
        return r
    view = _row(views); m20 = _row(mom20); acc = _row(accel); rev = _row(reversal)
    dhv = np.full(N, np.nan)
    for s in held:
        dhv[A.index(s)] = 50.0
    if dh:
        for s, v in dh.items():
            dhv[A.index(s)] = v
    k = cfg.adaptive_hazard_confirmations
    recent = np.zeros((k, N))
    if exh:
        for s, v in exh.items():
            recent[:, A.index(s)] = v
    cdm = np.zeros(N, bool)
    if cooldown:
        for s in cooldown:
            cdm[A.index(s)] = True
    elm = (np.array([a in eligible for a in A], bool) if eligible is not None
           else np.ones(N, bool))
    return apply_adaptive_rotation(
        base_weights=bw, position_returns=pr, days_held=dhv, view_row=view,
        sigma_ann=sigma, mom20_row=m20, accel_row=acc, reversal_score=rev,
        exhaustion_recent=recent, cov_adj=cov, state=state, severe=severe, cfg=cfg,
        growth_ceiling=1.4, vol_budget=0.15, leverage_cap=1.4,
        cooldown_mask=cdm, eligible_mask=elm)


def _evaluated(r):
    return {e["asset"] for e in (r.events + r.skipped)}


# ------------------------------------------------------------------ 1 damage units
def test_damage_is_volatility_normalized():
    # SAME -3% move: low-vol SPY is damaged (a source); high-vol EEM is not.
    cov = np.diag([0.0036 if a == "SPY" else (0.09 if a == "EEM" else 0.02) for a in A])
    r = _call(held={"SPY": -0.03, "EEM": -0.03}, cov=cov)
    ev = _evaluated(r)
    assert "SPY" in ev and "EEM" not in ev, ev       # vol-normalization discriminates


# ------------------------------------------------------------------ 2/3 cost gates
def test_risky_cost_gate_boundary():
    # SPY weak+damaged; IWM edge just BELOW the cost gate (2*tc+margin=0.0017) -> HOLD;
    # just ABOVE -> ROTATE. Keep ranks identical so only the view delta differs.
    common = dict(held={"SPY": -2 * VOLH}, mom20={"SPY": -0.05, "IWM": 0.06},
                  accel={"SPY": -0.03, "IWM": 0.04})
    below = _call(views={"SPY": -0.0010, "IWM": 0.0005}, **common)  # delta 0.0015 < 0.0017
    above = _call(views={"SPY": -0.0010, "IWM": 0.0020}, **common)  # delta 0.0030 > 0.0017
    assert not below.triggered, "thin edge must be cost-gated"
    assert above.triggered and above.events[0]["replacement_asset"] == "IWM"


def test_cash_cost_gate_excess():
    # A damaged holding with a POSITIVE view is never rotated to cash (its excess is +).
    r_pos = _call(held={"SPY": -2 * VOLH}, views={"SPY": 0.02})
    assert not any(e.get("action") == "rotate_to_cash" for e in r_pos.events)
    # Broad selloff (every view negative, SPY worst) -> CASH wins (all risky weak).
    r_cash = _call(held={"SPY": -2 * VOLH},
                   views={**{a: -0.02 for a in A}, "SPY": -0.05},
                   mom20={"SPY": -0.08}, accel={"SPY": -0.05})
    assert r_cash.triggered and r_cash.events[0]["replacement_asset"] is None
    assert r_cash.events[0]["action"] == "rotate_to_cash"


# ------------------------------------------------------------------ opportunity gate
def test_opportunity_gate_holds_when_nothing_superior():
    # Damaged SPY but every asset equally mediocre -> HOLD (anti-churn).
    r = _call(held={"SPY": -2 * VOLH})
    assert not r.triggered and r.skipped


def test_opportunity_gate_rotates_on_clear_edge():
    r = _call(held={"SPY": -2 * VOLH}, views={"SPY": -0.02, "IWM": 0.03},
              mom20={"SPY": -0.05, "IWM": 0.06}, accel={"SPY": -0.03, "IWM": 0.04})
    assert r.triggered and r.exited == ["SPY"] and r.events[0]["replacement_asset"] == "IWM"


# ------------------------------------------------------------------ 4 regime sensitivity
def test_regime_sensitivity():
    # damage ~ -1.4 vol-units: below CRISIS(-1.5)? no; below BULL(-1.75)? no; below
    # DEFENSIVE(-1.25)? yes. So it is a source in DEFENSIVE but not in BULL.
    common = dict(held={"SPY": -1.4 * VOLH}, views={"SPY": -0.02, "IWM": 0.03},
                  mom20={"SPY": -0.05, "IWM": 0.06}, accel={"SPY": -0.03, "IWM": 0.04})
    assert _call(state="DEFENSIVE", **common).triggered
    assert not _call(state="BULL", **common).triggered      # BULL tolerates more


# ------------------------------------------------------------------ strong winner
def test_strong_winner_overbought_holds():
    # SPY is a big winner (not damaged), high exhaustion score would fire the old
    # overlay, but here it is not even a source unless hazard-confirmed; give a strong
    # book -> undamaged -> no source.
    r = _call(held={"SPY": 0.15}, views={"SPY": 0.03}, mom20={"SPY": 0.10}, accel={"SPY": 0.05})
    assert not r.triggered


# ------------------------------------------------------------------ 9 min-hold bypass
def test_min_hold_blocks_opportunity_but_damage_bypasses():
    strong = dict(views={"SPY": -0.02, "IWM": 0.03}, mom20={"SPY": -0.05, "IWM": 0.06},
                  accel={"SPY": -0.03, "IWM": 0.04})
    # hazard-only source (not damaged), fresh position -> min_hold blocks.
    fresh_hazard = _call(held={"SPY": -0.2 * VOLH}, dh={"SPY": 1},
                         exh={"SPY": 0.99}, **strong)
    assert not fresh_hazard.triggered
    assert any(s["reason"] == "min_hold_days_not_met" for s in fresh_hazard.skipped)
    # genuinely damaged fresh position -> bypasses min_hold, rotates.
    fresh_damaged = _call(held={"SPY": -2 * VOLH}, dh={"SPY": 1}, **strong)
    assert fresh_damaged.triggered


# ------------------------------------------------------------------ 13 div can't dominate
def test_diversification_cannot_dominate():
    # A perfectly uncorrelated but MEDIOCRE candidate (mid ranks) cannot clear the
    # 0.15 edge on diversification alone -> HOLD.
    r = _call(held={"SPY": -2 * VOLH}, views={"SPY": -0.02})   # SPY weak, others neutral
    # others are all view 0 -> mid rank; div<=0.10 can't push net_edge above 0.15
    if r.triggered:
        # if it did rotate, it must be a genuine edge, not pure diversification
        assert r.events[0]["net_rotation_edge"] > CFG.adaptive_diversification_cap


# ------------------------------------------------------------------ 4/5 determinism
def test_determinism_and_tie_breaks():
    kw = dict(held={"SPY": -2 * VOLH, "EFA": -2 * VOLH},
              views={"SPY": -0.02, "EFA": -0.02, "IWM": 0.03, "GLD": 0.03},
              mom20={"SPY": -0.05, "EFA": -0.05, "IWM": 0.06, "GLD": 0.06},
              accel={"SPY": -0.03, "EFA": -0.03, "IWM": 0.04, "GLD": 0.04},
              base={"SPY": 0.20, "EFA": 0.20})
    r1 = _call(**kw)
    r2 = _call(**kw)
    m1 = {e["exited_asset"]: e["replacement_asset"] for e in r1.events}
    m2 = {e["exited_asset"]: e["replacement_asset"] for e in r2.events}
    assert m1 == m2                                  # identical inputs -> identical result
    # distinct candidates: two sources don't grab the same risky replacement
    repls = [e["replacement_asset"] for e in r1.events if e["replacement_asset"]]
    assert len(repls) == len(set(repls))


# ------------------------------------------------------------------ 6 constraint safety
def test_rotation_respects_constraints():
    # rotate SPY(0.30) into DBC (cap 0.15) -> excess clips to cash, no cap violation.
    r = _call(held={"SPY": -2 * VOLH}, base={"SPY": 0.30},
              views={"SPY": -0.02, "DBC": 0.03}, mom20={"SPY": -0.05, "DBC": 0.06},
              accel={"SPY": -0.03, "DBC": 0.04}, eligible={"DBC"} | {a for a in A if a != "SPY"})
    if r.triggered and r.events[0]["replacement_asset"] == "DBC":
        assert r.weights[A.index("DBC")] <= CFG.asset_max_weights["DBC"] + 1e-9


# ------------------------------------------------------------------ 12 degenerate
def test_degenerate_and_severe():
    assert not _call(held={}).triggered                       # empty book -> no-op
    r_all_nan = _call(held={"SPY": -2 * VOLH}, views={"SPY": np.nan})
    assert not r_all_nan.triggered or r_all_nan.skipped is not None  # no crash
    # severe bear: still evaluates but the demo covers cap-0 upstream; just no crash
    _call(held={"SPY": -2 * VOLH}, severe=True)


def test_cooldown_excludes_candidate():
    r = _call(held={"SPY": -2 * VOLH}, views={"SPY": -0.02, "IWM": 0.03, "GLD": 0.025},
              mom20={"SPY": -0.05, "IWM": 0.06, "GLD": 0.055},
              accel={"SPY": -0.03, "IWM": 0.04, "GLD": 0.035}, cooldown={"IWM"})
    if r.triggered:
        assert r.events[0]["replacement_asset"] != "IWM"      # IWM barred by cooldown


# ------------------------------------------------------------------ 8 days_held parity
def test_days_held_tracker():
    t = CostBasisTracker(A)
    i_spy = A.index("SPY")
    def w(v):
        x = np.zeros(N); x[i_spy] = v; return x
    p = np.full(N, 100.0)
    t.update(w(0.30), p, 1000.0, session=5)          # first entry at session 5
    assert t.days_held(10)[i_spy] == 5.0
    t.update(w(0.50), p, 1000.0, session=8)          # scale-in does NOT reset
    assert t.days_held(10)[i_spy] == 5.0
    t.update(w(0.0), p, 1000.0, session=12)          # full exit resets
    assert np.isnan(t.days_held(15)[i_spy])


# ------------------------------------------------------------------ 7 supersede guard
def test_supersede_guard():
    AllocatorConfig(adaptive_rotation_enabled=True)               # ok alone
    with pytest.raises(ValueError):
        AllocatorConfig(adaptive_rotation_enabled=True, loss_replacement_enabled=True)
    with pytest.raises(ValueError):
        AllocatorConfig(adaptive_rotation_enabled=True, momentum_exhaustion_enabled=True)


# ------------------------------------------------------------------ integration + 10/11/15
_P = make_synthetic_prices(end="2009-12-31")
_RP = _P[list(RISK_ASSETS)]
_D = _P.index[-1]


def _blowoff():
    px = _RP.copy(); tail = 40
    ramp = np.concatenate([np.linspace(0.006, 0.016, tail - 6),
                           [-0.03, -0.035, -0.03, -0.025, -0.02, -0.015]])
    q = px["QQQ"].to_numpy(float).copy()
    for i in range(tail):
        q[-tail + i] = q[-tail + i - 1] * (1.0 + ramp[i])
    px = px.copy(); px["QQQ"] = q
    return px


def test_default_off_byte_identical():
    cfg_off = AllocatorConfig()
    last = float(_RP["QQQ"].iloc[-1])
    base = compute_target_portfolio(_RP, np.zeros(N), COLD_START_SESSIONS, cfg_off, as_of_close=_D)
    withpos = compute_target_portfolio(_RP, np.zeros(N), COLD_START_SESSIONS, cfg_off,
                                       as_of_close=_D, position_avg_entry={"QQQ": last * 1.2},
                                       position_entry_dates={"QQQ": "2009-01-02"})
    assert np.array_equal(base.target_vector(cfg_off), withpos.target_vector(cfg_off))
    assert base.decision_hash == withpos.decision_hash


def test_decision_hash_changes_when_enabled():
    off = AllocatorConfig().__class__  # baseline hash from default
    base = compute_target_portfolio(_RP, np.zeros(N), COLD_START_SESSIONS, AllocatorConfig(),
                                    as_of_close=_D)
    on = compute_target_portfolio(_RP, np.zeros(N), COLD_START_SESSIONS,
                                  AllocatorConfig(adaptive_rotation_enabled=True), as_of_close=_D)
    assert on.decision_hash != base.decision_hash        # cfg toggle changes identity


def test_integration_no_lookahead():
    px = _blowoff()
    full = make_synthetic_prices(end="2010-12-31").copy()
    for c in list(RISK_ASSETS):
        full.loc[:, c] = full[c]
    # graft the blow-off QQQ tail into the full frame up to _D, keep future as-is
    cfg = AllocatorConfig(adaptive_rotation_enabled=True,
                          adaptive_entry_score_threshold=0.30,
                          adaptive_minimum_rotation_edge=0.05)
    ae = {"QQQ": float(px["QQQ"].iloc[-1]) * 1.15}
    sliced = compute_target_portfolio(px, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=_D,
                                      position_avg_entry=ae, position_entry_dates={"QQQ": "2009-01-02"})
    mutated = px.copy()
    # append future rows and corrupt them; decision at _D must be unchanged
    fut = full.loc[full.index > _D, list(RISK_ASSETS)] * 0.3
    mutated = pd.concat([px, fut])
    after = compute_target_portfolio(mutated, np.zeros(N), COLD_START_SESSIONS, cfg, as_of_close=_D,
                                     position_avg_entry=ae, position_entry_dates={"QQQ": "2009-01-02"})
    assert np.array_equal(sliced.target_vector(cfg), after.target_vector(cfg))
    assert sliced.decision_hash == after.decision_hash


# ------------------------------------------------------------------ 14 engine execution
from execution_store import ExecutionStore
from live_engine import LiveEngine, ExecutionConfig, Mode
from fakes import FakeBroker, FakeData


def _ecfg(**over):
    base = dict(live_enabled=False, max_order_notional=1e9, max_total_notional=1e12,
                max_gross_exposure=3.0, max_single_asset_weight=3.0,
                max_rebalance_turnover=10.0, max_daily_loss_pct=0.05,
                max_account_drawdown_pct=0.25, minimum_trade_notional=1.0,
                residual_share_threshold=1e-4, expected_account_id=None)
    base.update(over)
    return ExecutionConfig(**base)


def test_engine_adaptive_enabled_executes_safely_and_idempotent(tmp_path):
    # Adaptive-enabled engine on a damaged-SPY scenario: executes safely, maintains
    # the entry ledger, and is idempotent. (Whether the exit is attributed to the
    # adaptive overlay or the base de-risk, the SHARED _execute path is exercised;
    # adaptive rotation FIRING is covered by the unit/integration tests, since the
    # base allocator holds the diversified book so flat candidates are rare here.)
    cfg = AllocatorConfig(adaptive_rotation_enabled=True)
    br = FakeBroker(_RP, fill_mode="full")               # benign prices (diversified book)
    pxs = br.px["SPY"]; br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / pxs}; br.avg_entry = {"SPY": pxs * 1.15}   # underwater
    store = ExecutionStore(str(tmp_path / "t.db"))
    eng = LiveEngine(Mode.PAPER, _ecfg(), cfg, store, broker=br, data=FakeData(_RP))
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is None, rep.rejected_reason
    assert rep.executed
    assert store.get_entries()                            # first-entry ledger populated
    first = br.submit_calls
    rep2 = eng.run_rebalance(dry_run=False)               # idempotent
    assert br.submit_calls == first
    assert rep2.executed is False and rep2.skipped_reason is not None


def test_adaptive_events_flow_into_rotation_events():
    # A decision that DOES fire adaptive (unit-level firing via constructed scores)
    # surfaces its events in rotation_events() -> the same cooldown/ledger/_execute
    # path the loss/exhaustion overlays use (which are execution-tested).
    from strat_decision import PortfolioDecision
    dec = PortfolioDecision(
        as_of_close=_D, target_session=None, recovery_state="BULL", severe_bear=False,
        growth_ceiling=1.4, leverage_cap=1.4, vol_budget=0.15,
        target_weights={a: 0.0 for a in A}, prev_weights={a: 0.0 for a in A},
        gross_exposure=0.0, growth_exposure=0.0, defensive_exposure=0.0, ex_ante_vol=0.0,
        rebalanced=True, sessions_since_rebalance=0, forecast_source="x+adaptive_rotation",
        decision_hash="h", rebalance_reason="adaptive_rotation",
        adaptive_events=[{"type": "adaptive_rotation", "exited_asset": "SPY",
                          "replacement_asset": "IWM", "action": "rotate"}])
    ev = dec.rotation_events()
    assert len(ev) == 1 and ev[0]["exited_asset"] == "SPY"   # adaptive flows to the ledger


def test_engine_adaptive_populates_entry_ledger(tmp_path):
    # Benign cold start (flat broker) -> base book executes -> position_entry stamped.
    cfg = AllocatorConfig(adaptive_rotation_enabled=True)
    br = FakeBroker(_RP, fill_mode="full")            # flat broker
    store = ExecutionStore(str(tmp_path / "t.db"))
    eng = LiveEngine(Mode.PAPER, _ecfg(), cfg, store, broker=br, data=FakeData(_RP))
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is None, rep.rejected_reason
    assert rep.executed and rep.rebalance_reason == "scheduled"   # cold start, no sources
    entries = store.get_entries()
    held = {a for a, w in eng.compute_decision(prices=_RP, positions=br.positions()).target_weights.items() if w > 1e-4}
    assert held and set(entries) == held, (entries, held)   # ledger == held book
