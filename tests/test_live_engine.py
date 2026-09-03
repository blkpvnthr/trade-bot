"""
Live-engine safety tests: dry-run, idempotency, restart recovery, partial fills,
broker rejection, constraint + operational gates, live 3-gate protection,
account-id protection, and kill-switch classification.

Offline + deterministic (synthetic prices + FakeBroker; no network).
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

from strat_fixtures import make_synthetic_prices
from strat_allocator import AllocatorConfig
from execution_store import ExecutionStore
import live_engine
from live_engine import LiveEngine, ExecutionConfig, Mode, GateRejection
from fakes import FakeBroker, FakeData

# Short fixture -> the decision at the last row is a post-recovery rebalance.
PRICES = make_synthetic_prices(end="2009-12-31")
ASSETS = list(AllocatorConfig().risk_assets)


def _ecfg(**over):
    base = dict(live_enabled=False, max_order_notional=1e9, max_total_notional=1e12,
                max_gross_exposure=3.0, max_single_asset_weight=3.0,
                max_rebalance_turnover=10.0, max_daily_loss_pct=0.05,
                max_account_drawdown_pct=0.25, minimum_trade_notional=1.0,
                residual_share_threshold=1e-4, expected_account_id=None)
    base.update(over)
    return ExecutionConfig(**base)


def _engine(tmp_path, broker, data=None, mode=Mode.PAPER, ecfg=None):
    store = ExecutionStore(str(tmp_path / "t.db"))
    return LiveEngine(mode, ecfg or _ecfg(), AllocatorConfig(), store,
                      broker=broker, data=data or FakeData(PRICES))


def test_dry_run_submits_zero_orders(tmp_path):
    br = FakeBroker(PRICES)
    eng = _engine(tmp_path, br, mode=Mode.DRY_RUN)
    rep = eng.run_rebalance(dry_run=True)
    assert rep.rejected_reason is None, rep.rejected_reason
    assert br.submit_calls == 0
    assert len(rep.plans) > 0            # a real plan was produced
    assert rep.rebalanced is True


def test_paper_rebalance_executes_and_reconciles(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is None, rep.rejected_reason
    assert rep.executed is True
    assert br.submit_calls > 0
    # final book close to target
    worst = max(abs(v) for v in rep.final_drift.values())
    assert worst < 0.02, f"drift too large: {worst}"


def test_idempotent_rerun_no_duplicate_orders(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    store = ExecutionStore(str(tmp_path / "t.db"))
    eng = LiveEngine(Mode.PAPER, _ecfg(), AllocatorConfig(), store, broker=br, data=FakeData(PRICES))
    eng.run_rebalance(dry_run=False)
    first = br.submit_calls
    assert first > 0
    # Re-run the same session: after rebalancing, last_rebalance == today so the
    # session is no longer `due` -> the strategy holds -> ZERO new orders. That is
    # the idempotency guarantee (no duplicates on re-run).
    rep2 = eng.run_rebalance(dry_run=False)
    assert br.submit_calls == first, "re-run duplicated orders"
    assert rep2.rejected_reason is None
    assert rep2.executed is False and rep2.skipped_reason is not None

    # The run-state machine's COMPLETE short-circuit (crash-after-COMPLETE case) is
    # covered directly at the store level (execution_store.demo) and by the restart
    # test below.


def test_restart_mid_rebalance_recovers_without_duplicates(tmp_path):
    # Hard crash after 3 submits (SystemExit is not caught -> lock stays held).
    class HardCrashBroker(FakeBroker):
        def submit_market_order(self, *a, **k):
            if self.crash_after is not None and self.submit_calls >= self.crash_after:
                raise SystemExit("simulated hard crash")
            return super().submit_market_order(*a, **k)

    dbp = str(tmp_path / "t.db")
    br = HardCrashBroker(PRICES, fill_mode="full", crash_after=3)
    store = ExecutionStore(dbp)
    eng = LiveEngine(Mode.PAPER, _ecfg(), AllocatorConfig(), store, broker=br, data=FakeData(PRICES))
    with pytest.raises(SystemExit):
        eng.run_rebalance(dry_run=False)
    submitted_before = br.submit_calls
    assert store.active_run() is not None   # lock held after hard crash

    # Restart: new engine, healthy broker sharing the SAME positions/orders + DB.
    br.crash_after = None
    store2 = ExecutionStore(dbp)
    eng2 = LiveEngine(Mode.PAPER, _ecfg(), AllocatorConfig(), store2, broker=br, data=FakeData(PRICES))
    rep = eng2.run_rebalance(dry_run=False)
    assert store2.active_run() is None      # lock released, run completed
    # every client_order_id was submitted at most once (no duplicate legs)
    coids = [o["client_order_id"] for o in br.orders.values()]
    assert len(coids) == len(set(coids))
    # book reached target after recovery
    worst = max(abs(v) for v in eng2._drift(eng2.compute_decision(prices=PRICES),
                                            br.positions(), br.account_snapshot().equity).values())
    assert worst < 0.03, f"recovery left drift {worst}"


def test_partial_fill_reconciliation(tmp_path):
    br = FakeBroker(PRICES, fill_mode="partial", partial_fraction=0.4)
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is None, rep.rejected_reason
    # residual slices were used -> some -s1 coids exist
    assert any("-s1" in c for c in br.orders), "no residual slice submitted"
    worst = max(abs(v) for v in rep.final_drift.values())
    assert worst < 0.03, f"partial-fill drift {worst}"


def test_broker_rejection_stops_dependent_buys(tmp_path):
    # DBC has the lowest prior Sharpe / a 0.15 cap, so its target is small. Seed a
    # ~50% DBC book -> a large DBC SELL is required -> reject it -> fail closed.
    br = FakeBroker(PRICES, fill_mode="full", reject_symbols=["DBC"])
    px_dbc = br.px["DBC"]
    br.cash = 50_000.0
    br.positions_qty = {"DBC": 50_000.0 / px_dbc}   # DBC ~50% of a 100k book
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert "sell legs did not complete" in rep.rejected_reason
    # no buy leg was executed after the failed sell
    assert not any(o["side"] == "buy" and o["status"] == "filled" for o in br.orders.values())


def test_operational_gate_rejects_oversized_order(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    eng = _engine(tmp_path, br, ecfg=_ecfg(max_order_notional=10.0))
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert "LIVE_MAX_ORDER_NOTIONAL" in rep.rejected_reason
    assert br.submit_calls == 0          # rejected before any order


def test_operational_gate_rejects_gross_over_limit(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    eng = _engine(tmp_path, br, ecfg=_ecfg(max_gross_exposure=0.10))
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert "LIVE_MAX_GROSS_EXPOSURE" in rep.rejected_reason
    assert br.submit_calls == 0


def test_account_blocked_rejects(tmp_path):
    br = FakeBroker(PRICES, trading_blocked=True)
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert "not tradable" in rep.rejected_reason
    assert br.submit_calls == 0


def test_account_id_mismatch_blocks_live(tmp_path):
    br = FakeBroker(PRICES, account_id="real-account")
    eng = _engine(tmp_path, br, mode=Mode.LIVE, ecfg=_ecfg(expected_account_id="expected-other"))
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert "MISMATCH" in rep.rejected_reason.upper()
    assert br.submit_calls == 0


def test_kill_switch_daily_loss_blocks_risk_increasing(tmp_path):
    # equity 20% below prior close -> daily loss kill; cold-start buys are risk-increasing
    br = FakeBroker(PRICES, equity=80_000.0, last_equity=100_000.0)
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert br.submit_calls == 0
    assert rep.skipped_reason is not None and "KILL_SWITCH_ACTIVE" in rep.skipped_reason
    assert "daily_loss" in rep.kill_switch


def test_kill_switch_file_halts_all_orders(tmp_path, monkeypatch):
    ks = tmp_path / ".kill_switch"
    ks.write_text("halt")
    monkeypatch.setattr(live_engine, "KILL_SWITCH_PATH", str(ks))
    br = FakeBroker(PRICES)
    eng = _engine(tmp_path, br)
    rep = eng.run_rebalance(dry_run=False)
    assert br.submit_calls == 0
    assert "file" in rep.kill_switch
    assert rep.skipped_reason is not None and "KILL_SWITCH_ACTIVE" in rep.skipped_reason


def test_nan_weights_rejected_by_strategy_gate(tmp_path):
    br = FakeBroker(PRICES, fill_mode="full")
    eng = _engine(tmp_path, br)
    # force a NaN into the realized book via a broken executable price
    orig = eng._executable_prices
    eng._executable_prices = lambda syms, pos: {**orig(syms, pos), "SPY": float("nan")}
    rep = eng.run_rebalance(dry_run=False)
    assert rep.rejected_reason is not None
    assert br.submit_calls == 0


# ---- live 3-gate CLI protection (no engine needed) ----
def test_live_mode_blocked_without_all_three_gates(monkeypatch):
    from live_trader import resolve_mode, LiveBlocked

    class A:
        dry_run = False
        paper = False
        live = True
        confirm_live = None
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    monkeypatch.setenv("ALPACA_PAPER_ONLY", "true")
    ec = ExecutionConfig.from_env()
    with pytest.raises(LiveBlocked):
        resolve_mode(A(), ec)                 # missing confirm + gates

    # all three present -> LIVE resolves
    class B(A):
        confirm_live = "LIVE"
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ALPACA_PAPER_ONLY", "false")
    monkeypatch.setenv("MODE", "live")
    ec2 = ExecutionConfig.from_env()
    assert resolve_mode(B(), ec2) == Mode.LIVE


def test_paper_and_dry_run_modes_resolve(monkeypatch):
    from live_trader import resolve_mode

    class P:
        dry_run = False
        paper = True
        live = False
        confirm_live = None
    assert resolve_mode(P(), ExecutionConfig.from_env()) == Mode.PAPER

    class D(P):
        paper = False
        dry_run = True
    assert resolve_mode(D(), ExecutionConfig.from_env()) == Mode.DRY_RUN
