"""
Loss-replacement / momentum-rotation overlay tests (spec section 12).

Covers: trigger boundary, candidate selection + ineligibility, no-candidate cash
fallback, already-held exclusion, cooldown, cost-basis (not one-day return),
point-in-time (no lookahead), portfolio-constraint enforcement, sell->buy
dependency, idempotency, and restart recovery -- WITHOUT weakening any existing
safety test. Offline + deterministic (synthetic prices + FakeBroker, no network).
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
import pytest

from strat_fixtures import make_synthetic_prices, RISK_ASSETS
from strat_allocator import AllocatorConfig
from strat_decision import compute_target_portfolio, COLD_START_SESSIONS, DEFAULT_START
from execution_store import ExecutionStore
from live_engine import LiveEngine, ExecutionConfig, Mode
from loss_replacement import apply_loss_replacement, CostBasisTracker, LossReplacementResult
from fakes import FakeBroker, FakeData

PRICES = make_synthetic_prices(end="2009-12-31")
RISK_PX = PRICES[list(RISK_ASSETS)]
D = PRICES.index[-1]
CFG_ON = AllocatorConfig(loss_replacement_enabled=True)
A = list(CFG_ON.risk_assets)
N = len(A)
COV = np.eye(N) * 0.02


def _pr(**kw):
    v = np.full(N, np.nan)
    for k, x in kw.items():
        v[A.index(k)] = x
    return v


def _mom(**kw):
    return _pr(**kw)


def _base(**kw):
    w = np.zeros(N)
    for k, x in kw.items():
        w[A.index(k)] = x
    return w


def _call(base, pos_ret, mom, **over):
    kw = dict(base_weights=base, position_returns=pos_ret, momentum=mom, cov_adj=COV,
              cfg=CFG_ON, growth_ceiling=1.4, vol_budget=0.15, leverage_cap=1.4)
    kw.update(over)
    return apply_loss_replacement(**kw)


# --------------------------------------------------------------------------
# 1) Trigger boundary: -0.49% no, -0.50% yes (inclusive), -0.51% yes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ret,expected", [(-0.0049, False), (-0.0050, True), (-0.0051, True)])
def test_trigger_boundary(ret, expected):
    r = _call(_base(SPY=0.30), _pr(SPY=ret), _mom(QQQ=0.02))
    assert r.triggered is expected, f"pr={ret} -> triggered {r.triggered}, want {expected}"


# --------------------------------------------------------------------------
# 2) Candidate selection: A exits, highest-momentum C wins, negative D ineligible
# --------------------------------------------------------------------------
def test_candidate_selection_ranks_by_momentum():
    r = _call(_base(SPY=0.30), _pr(SPY=-0.007),
              _mom(QQQ=0.02, IWM=0.04, EFA=-0.01))   # QQQ +2, IWM +4, EFA -1
    assert r.exited == ["SPY"]
    ev = r.events[0]
    assert ev["replacement_asset"] == "IWM"          # strongest positive momentum
    assert ev["reason"] == "position_loss_threshold_breached"
    assert r.weights[A.index("SPY")] == 0.0
    assert r.weights[A.index("IWM")] > 0.0
    assert r.weights[A.index("EFA")] == 0.0           # negative momentum never chosen


# --------------------------------------------------------------------------
# 3) No eligible replacement -> exit anyway, allocation to cash
# --------------------------------------------------------------------------
def test_no_candidate_routes_to_cash():
    r = _call(_base(SPY=0.30), _pr(SPY=-0.02), _mom(QQQ=-0.02, IWM=-0.05))
    assert r.triggered is True
    assert r.events[0]["replacement_asset"] is None
    assert r.events[0]["reason"] == "no_positive_momentum_candidate"
    assert r.weights[A.index("SPY")] == 0.0
    assert float(r.weights.sum()) < 0.30 + 1e-12       # freed weight went to cash


# --------------------------------------------------------------------------
# 4) Already-held assets are excluded as replacements (no accidental concentration)
# --------------------------------------------------------------------------
def test_already_held_excluded_as_replacement():
    # QQQ is currently HELD (finite position_return) AND has the best momentum, but
    # it must NOT be chosen -- only a flat asset can be a replacement.
    r = _call(_base(SPY=0.30, QQQ=0.20), _pr(SPY=-0.02, QQQ=0.05),
              _mom(QQQ=0.09, IWM=0.03))
    assert r.exited == ["SPY"]
    assert r.events[0]["replacement_asset"] == "IWM"   # QQQ held -> skipped despite +9%
    assert r.weights[A.index("QQQ")] == 0.20           # QQQ untouched (not chosen)


# --------------------------------------------------------------------------
# 5) Cooldown: a recently rotated-out ticker cannot immediately rotate back in
# --------------------------------------------------------------------------
def test_cooldown_blocks_reentry():
    cd = np.zeros(N, bool)
    cd[A.index("IWM")] = True                          # IWM cooling down
    r = _call(_base(SPY=0.30), _pr(SPY=-0.02),
              _mom(IWM=0.04, QQQ=0.02), cooldown_mask=cd)
    assert r.events[0]["replacement_asset"] == "QQQ"   # IWM skipped, next-best QQQ


def test_eligible_mask_excludes():
    el = np.ones(N, bool)
    el[A.index("IWM")] = False                         # IWM ineligible (e.g. untradable)
    r = _call(_base(SPY=0.30), _pr(SPY=-0.02),
              _mom(IWM=0.04, QQQ=0.02), eligible_mask=el)
    assert r.events[0]["replacement_asset"] == "QQQ"


# --------------------------------------------------------------------------
# 6) Multiple simultaneous losers: deterministic, distinct replacements
# --------------------------------------------------------------------------
def test_multiple_losers_deterministic_distinct():
    # SPY (-0.9%) and EFA (-0.6%) both breach; IWM (+5%) and QQQ (+3%) are candidates.
    r = _call(_base(SPY=0.20, EFA=0.10), _pr(SPY=-0.009, EFA=-0.006),
              _mom(IWM=0.05, QQQ=0.03))
    # worst loser (SPY) gets best candidate (IWM); next loser (EFA) gets QQQ.
    by_exit = {e["exited_asset"]: e["replacement_asset"] for e in r.events}
    assert by_exit == {"SPY": "IWM", "EFA": "QQQ"}
    # deterministic: identical call yields identical assignment
    r2 = _call(_base(SPY=0.20, EFA=0.10), _pr(SPY=-0.009, EFA=-0.006),
               _mom(IWM=0.05, QQQ=0.03))
    assert [e["replacement_asset"] for e in r2.events] == ["IWM", "QQQ"]


# --------------------------------------------------------------------------
# 7) Cost basis is weighted-average across fills, NOT a one-day return
# --------------------------------------------------------------------------
def test_cost_basis_weighted_average():
    t = CostBasisTracker(A)
    i = A.index("SPY")
    def w(v):
        x = np.zeros(N); x[i] = v; return x
    def p(v):
        x = np.full(N, 100.0); x[i] = v; return x
    t.update(w(0.5), p(100.0), 1000.0)                 # buy 5 @ 100
    t.update(w(1.02), p(102.0), 1000.0)                # buy ~5 more @ 102 -> avg ~101
    assert 100.5 < float(t.cost[i]) < 101.5
    # A -0.51% ONE-DAY move at price 102 is not a trigger; the position (avg 101)
    # is actually UP, so position_return must be positive here.
    pr = t.position_returns(p(102.0))
    assert pr[i] > 0.0
    # Only a price below cost basis by the threshold trips it.
    trip = t.position_returns(p(101.0 * (1 - 0.006)))
    assert trip[i] <= -0.005


def test_cost_basis_resets_on_full_exit():
    t = CostBasisTracker(A)
    i = A.index("SPY")
    x = np.zeros(N); x[i] = 0.5
    t.update(x, np.full(N, 100.0), 1000.0)
    assert np.isfinite(t.cost[i])
    t.update(np.zeros(N), np.full(N, 100.0), 1000.0)   # sell all
    assert not np.isfinite(t.cost[i])
    assert np.all(np.isnan(t.position_returns(np.full(N, 100.0))))


# --------------------------------------------------------------------------
# 8) Portfolio constraints hold after a rotation (enforce_constraints re-run)
# --------------------------------------------------------------------------
def test_replacement_respects_per_asset_cap():
    # Exit SPY(0.30) into DBC, whose cap is 0.15 -> the excess must clip to cash.
    r = _call(_base(SPY=0.30), _pr(SPY=-0.02), _mom(DBC=0.05))
    assert r.events[0]["replacement_asset"] == "DBC"
    assert r.weights[A.index("DBC")] <= CFG_ON.asset_max_weights["DBC"] + 1e-9
    assert float(r.weights.sum()) < 0.30 + 1e-9        # clipped remainder -> cash


def test_replacement_respects_vol_budget():
    # A tiny vol budget forces the whole book to scale down after the swap.
    r = _call(_base(SPY=0.30), _pr(SPY=-0.02), _mom(QQQ=0.05),
              vol_budget=0.01, cov_adj=np.eye(N) * 0.04)
    ex_ante = float(np.sqrt(r.weights @ (np.eye(N) * 0.04) @ r.weights))
    assert ex_ante <= 0.01 + 1e-9


# --------------------------------------------------------------------------
# 9) compute_target_portfolio: overlay changes identity + is point-in-time
# --------------------------------------------------------------------------
def _decide(avg_entry=None, cooldown=None, prices=RISK_PX, cfg=CFG_ON, as_of=D):
    return compute_target_portfolio(
        prices, prev_weights=np.zeros(N), sessions_since_rebalance=COLD_START_SESSIONS,
        cfg=cfg, as_of_close=as_of, start=DEFAULT_START,
        position_avg_entry=avg_entry, cooldown_state=cooldown)


def test_live_overlay_fires_and_changes_hash():
    base = _decide()                                   # no positions -> no overlay
    last = RISK_PX.loc[D]
    ae = {"SPY": float(last["SPY"]) * 1.03}            # SPY -3% vs cost -> breach
    on = _decide(avg_entry=ae)
    assert on.rebalance_reason == "loss_replacement"
    assert on.forecast_source.endswith("+loss_replacement")
    assert on.target_weights["SPY"] == 0.0             # SPY exited
    assert on.decision_hash != base.decision_hash
    assert on.loss_replacement_events and on.loss_replacement_events[0]["exited_asset"] == "SPY"
    # constraints intact on the swapped live book
    w = on.target_vector(CFG_ON)
    assert np.all(w <= CFG_ON.caps + 1e-6) and np.all(w >= -1e-9)
    assert float(w.sum()) <= on.leverage_cap + 1e-6


def test_live_overlay_no_lookahead():
    last = RISK_PX.loc[D]
    ae = {"SPY": float(last["SPY"]) * 1.03}
    full = make_synthetic_prices(end="2010-12-31")[list(RISK_ASSETS)]
    on_slice = _decide(avg_entry=ae, prices=full.loc[:D])
    on_full = _decide(avg_entry=ae, prices=full)        # future bars present but ignored
    assert np.array_equal(on_slice.target_vector(CFG_ON), on_full.target_vector(CFG_ON))
    assert on_slice.decision_hash == on_full.decision_hash
    # mutating strictly-after-D bars must not change the as_of=D decision
    m = full.copy()
    m.loc[m.index > D] *= 0.5
    on_mut = _decide(avg_entry=ae, prices=m)
    assert np.array_equal(on_slice.target_vector(CFG_ON), on_mut.target_vector(CFG_ON))


def test_live_overlay_off_by_default_is_noop():
    cfg_off = AllocatorConfig()                        # default: disabled
    last = RISK_PX.loc[D]
    ae = {"SPY": float(last["SPY"]) * 1.10}            # would breach if enabled
    base = compute_target_portfolio(RISK_PX, np.zeros(N), COLD_START_SESSIONS, cfg_off,
                                    as_of_close=D)
    with_pos = compute_target_portfolio(RISK_PX, np.zeros(N), COLD_START_SESSIONS, cfg_off,
                                        as_of_close=D, position_avg_entry=ae)
    assert np.array_equal(base.target_vector(cfg_off), with_pos.target_vector(cfg_off))
    assert base.decision_hash == with_pos.decision_hash
    assert with_pos.rebalance_reason == "scheduled"


# --------------------------------------------------------------------------
# 10) Engine: overlay executes, and a rejected EXIT stops the dependent BUY
# --------------------------------------------------------------------------
def _ecfg(**over):
    base = dict(live_enabled=False, max_order_notional=1e9, max_total_notional=1e12,
                max_gross_exposure=3.0, max_single_asset_weight=3.0,
                max_rebalance_turnover=10.0, max_daily_loss_pct=0.05,
                max_account_drawdown_pct=0.25, minimum_trade_notional=1.0,
                residual_share_threshold=1e-4, expected_account_id=None)
    base.update(over)
    return ExecutionConfig(**base)


def _engine(tmp_path, broker, mode=Mode.PAPER, ecfg=None, store=None):
    store = store or ExecutionStore(str(tmp_path / "t.db"))
    return LiveEngine(mode, ecfg or _ecfg(), CFG_ON, store,
                      broker=broker, data=FakeData(PRICES))


def test_engine_loss_replacement_executes(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}         # ~50% SPY
    br.avg_entry = {"SPY": px_spy * 1.05}                 # bought 5% higher -> breach
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is None, rep.rejected_reason
    assert rep.rebalance_reason == "loss_replacement"
    assert rep.loss_replacement and rep.loss_replacement[0]["exited_asset"] == "SPY"
    assert rep.executed is True
    # SPY was rotated out -> a SPY sell was submitted
    assert any(o["symbol"] == "SPY" and o["side"] == "sell" for o in br.orders.values())


def test_engine_rejected_exit_stops_replacement_buy(tmp_path):
    # SPY breaches and the overlay rotates it out, but the SPY SELL is rejected ->
    # the dependent replacement BUY must NOT execute (strengthens the sell->buy dep).
    br = FakeBroker(PRICES, fill_mode="full", reject_symbols=["SPY"])
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}
    br.avg_entry = {"SPY": px_spy * 1.05}
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rebalance_reason == "loss_replacement"
    assert rep.rejected_reason is not None
    assert "sell legs did not complete" in rep.rejected_reason
    assert not any(o["side"] == "buy" and o["status"] == "filled" for o in br.orders.values())


def test_engine_loss_replacement_idempotent(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}
    br.avg_entry = {"SPY": px_spy * 1.05}
    store = ExecutionStore(str(tmp_path / "t.db"))
    eng = _engine(tmp_path, br, store=store)
    eng.run_rebalance(dry_run=False)
    first = br.submit_calls
    assert first > 0
    rep2 = eng.run_rebalance(dry_run=False)               # rerun same session
    assert br.submit_calls == first, "loss-replacement rerun duplicated orders"
    assert rep2.executed is False and rep2.skipped_reason is not None


def test_engine_loss_replacement_sets_cooldown(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}
    br.avg_entry = {"SPY": px_spy * 1.05}
    store = ExecutionStore(str(tmp_path / "t.db"))
    eng = _engine(tmp_path, br, store=store)
    eng.run_rebalance(dry_run=False)
    assert "SPY" in store.get_cooldowns(), "exited asset must be put on cooldown"


def test_engine_restart_recovers_loss_replacement_without_duplicates(tmp_path):
    class HardCrashBroker(FakeBroker):
        def submit_market_order(self, *a, **k):
            if self.crash_after is not None and self.submit_calls >= self.crash_after:
                raise SystemExit("simulated hard crash")
            return super().submit_market_order(*a, **k)

    dbp = str(tmp_path / "t.db")
    br = HardCrashBroker(PRICES, fill_mode="full", crash_after=1)
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}
    br.avg_entry = {"SPY": px_spy * 1.05}
    store = ExecutionStore(dbp)
    eng = LiveEngine(Mode.PAPER, _ecfg(), CFG_ON, store, broker=br, data=FakeData(PRICES))
    with pytest.raises(SystemExit):
        eng.run_rebalance(dry_run=False)
    assert store.active_run() is not None                 # lock held after crash

    br.crash_after = None
    store2 = ExecutionStore(dbp)
    eng2 = LiveEngine(Mode.PAPER, _ecfg(), CFG_ON, store2, broker=br, data=FakeData(PRICES))
    rep = eng2.run_rebalance(dry_run=False)
    assert store2.active_run() is None                    # recovered + lock released
    coids = [o["client_order_id"] for o in br.orders.values()]
    assert len(coids) == len(set(coids)), "duplicate order legs after restart"


# --------------------------------------------------------------------------
# Safety-fix regressions found by adversarial review (recovery sell->buy, etc.)
# --------------------------------------------------------------------------
def test_recovery_rejected_sell_stops_dependent_buy(tmp_path):
    """A SELL that rejects DURING restart recovery must fail-closed before buys."""
    class HardCrashBroker(FakeBroker):
        def submit_market_order(self, *a, **k):
            if self.crash_after is not None and self.submit_calls >= self.crash_after:
                raise SystemExit("crash before the SPY sell is submitted")
            return super().submit_market_order(*a, **k)

    dbp = str(tmp_path / "t.db")
    br = HardCrashBroker(PRICES, fill_mode="full", crash_after=0)   # crash on 1st submit
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}
    br.avg_entry = {"SPY": px_spy * 1.05}                           # SPY breaches -> exit
    store = ExecutionStore(dbp)
    eng = LiveEngine(Mode.PAPER, _ecfg(), CFG_ON, store, broker=br, data=FakeData(PRICES))
    with pytest.raises(SystemExit):
        eng.run_rebalance(dry_run=False)                           # intents written, sell not filled
    assert store.active_run() is not None

    # Restart: the SPY sell now REJECTS. Recovery must NOT drive the replacement buys.
    br.crash_after = None
    br.reject_symbols = {"SPY"}
    store2 = ExecutionStore(dbp)
    eng2 = LiveEngine(Mode.PAPER, _ecfg(), CFG_ON, store2, broker=br, data=FakeData(PRICES))
    rep = eng2.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert "did not complete" in rep.rejected_reason
    assert not any(o["side"] == "buy" and o["status"] == "filled" for o in br.orders.values())


def test_nonterminal_sell_blocks_dependent_buys(tmp_path):
    """A sell stuck in a NON-terminal state after poll must still block buys (#2)."""
    class StuckSellBroker(FakeBroker):
        def submit_market_order(self, symbol, side, *, qty=None, notional=None, client_order_id):
            if symbol == "SPY" and side.startswith("s"):
                self.submit_calls += 1
                bid = f"b{self._next_id}"; self._next_id += 1
                o = {"broker_order_id": bid, "client_order_id": client_order_id,
                     "symbol": "SPY", "side": "sell", "status": "accepted",
                     "filled_qty": 0.0, "filled_avg_price": None,
                     "qty": qty, "notional": notional}
                self.orders[client_order_id] = o
                self.by_broker[bid] = client_order_id
                return o
            return super().submit_market_order(symbol, side, qty=qty, notional=notional,
                                               client_order_id=client_order_id)

    br = StuckSellBroker(PRICES, fill_mode="full")
    px_spy = br.px["SPY"]
    br.cash = 50_000.0
    br.positions_qty = {"SPY": 50_000.0 / px_spy}
    br.avg_entry = {"SPY": px_spy * 1.05}
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None and "did not complete" in rep.rejected_reason
    assert not any(o["side"] == "buy" and o["status"] == "filled" for o in br.orders.values())


def test_recovery_cooldown_only_from_rotation_events(tmp_path):
    """Recovery cools down ONLY the persisted overlay exits, not every sell leg (#5)."""
    from live_engine import RebalanceReport
    dbp = str(tmp_path / "t.db")
    store = ExecutionStore(dbp)
    br = FakeBroker(PRICES, fill_mode="full")
    br.positions_qty = {"SPY": 10.0, "DBC": 10.0}          # SPY overlay-exit, DBC a plain trim
    eng = LiveEngine(Mode.PAPER, _ecfg(), CFG_ON, store, broker=br, data=FakeData(PRICES))

    dhash = "recover-cooldown-hash"
    store.record_decision({
        "decision_hash": dhash, "as_of_close": "2009-12-31", "target_session": None,
        "forecast_source": "momentum_proxy+loss_replacement", "recovery_state": "BULL",
        "severe_bear": False, "leverage_cap": 1.4, "growth_ceiling": 1.0,
        "vol_budget": 0.15, "ex_ante_vol": 0.1, "gross": 0.5, "growth": 0.3,
        "defensive": 0.2, "target_weights": {a: 0.0 for a in A}, "mode": "paper",
        "rebalance_reason": "loss_replacement",
        "rotation_events": [{"type": "loss_replacement", "exited_asset": "SPY",
                             "position_return": -0.02, "threshold": -0.005,
                             "replacement_asset": None, "replacement_momentum": None,
                             "reason": "no_positive_momentum_candidate"}]})
    rid, _, _ = store.begin_or_resume_run(dhash, "paper")
    store.record_order_intent("v298-x-SPY-S-a", rid, dhash, "SPY", "sell", "market", 10.0, None)
    store.record_order_intent("v298-x-DBC-S-a", rid, dhash, "DBC", "sell", "market", 10.0, None)

    status = eng._recover_incomplete_run(RebalanceReport(mode="paper"))
    assert status == "completed"
    assert set(store.get_cooldowns().keys()) == {"SPY"}, \
        "recovery must cool down only overlay exits, not normal trims"


def test_recovery_persist_failure_fails_closed(tmp_path):
    """If persisting the recovered book fails, recovery must NOT mark the run
    COMPLETE and must leave the lock held (no COMPLETE-but-stale-state window)."""
    from live_engine import RebalanceReport
    dbp = str(tmp_path / "t.db")
    store = ExecutionStore(dbp)
    br = FakeBroker(PRICES, fill_mode="full")
    br.positions_qty = {"SPY": 10.0}
    eng = LiveEngine(Mode.PAPER, _ecfg(), CFG_ON, store, broker=br, data=FakeData(PRICES))

    dhash = "persist-fail-hash"
    store.record_decision({
        "decision_hash": dhash, "as_of_close": "2009-12-31", "target_session": None,
        "forecast_source": "x", "recovery_state": "BULL", "severe_bear": False,
        "leverage_cap": 1.4, "growth_ceiling": 1.0, "vol_budget": 0.15, "ex_ante_vol": 0.1,
        "gross": 0.0, "growth": 0.0, "defensive": 0.0,
        "target_weights": {a: 0.0 for a in A}, "mode": "paper",
        "rebalance_reason": "loss_replacement", "rotation_events": []})
    rid, _, _ = store.begin_or_resume_run(dhash, "paper")
    store.record_order_intent("v298-x-SPY-S-a", rid, dhash, "SPY", "sell", "market", 10.0, None)

    def boom(*a, **k):
        raise RuntimeError("simulated db-locked during persist")
    store.set_strategy_state = boom                     # break persistence

    with pytest.raises(RuntimeError):
        eng._recover_incomplete_run(RebalanceReport(mode="paper"))
    assert store.active_run() is not None               # lock still held for retry
    assert store.get_run(rid)["state"] != "COMPLETE"    # not falsely completed
