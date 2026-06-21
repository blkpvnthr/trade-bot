"""Tests for the earnings_within_7d paper sleeve + shadow promotion gate. Network-free."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))

import paper_sleeve as PS
import shadow as SH

_DQ_OK = {"earnings_age_days": 1, "missing_symbols_ratio": 0.0, "feed_available": True}


def _sig(n_in=3, n=20):
    return {f"S{i}": (1.0 if i < n_in else 0.0) for i in range(n)}


# --- self-contained: no RL allocator / execution / broker / live path ---
def test_no_live_or_allocator_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "place_order", "run_v2", "live_trade", ".orders", "portfolio_optimizer",
              "rl_trading.allocator", "alpaca.broker")
    for f in ("paper_sleeve.py", "run_paper_sleeve.py", "shadow.py"):
        mods = []
        for n in ast.walk(ast.parse((PKG / f).read_text())):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            assert not any(b in m.lower() for b in banned), f"{f} imports {m}"


# --- market-neutral, exact validated object (gross 100%, net 0, +50/-50) ---
def test_market_neutral_gross_100():
    w, d = PS.target_weights(_sig(), _DQ_OK)
    assert abs(d["gross"] - 1.0) < 1e-9 and abs(d["net"]) < 1e-9
    assert abs(sum(v for v in w.values() if v > 0) - 0.5) < 1e-9
    assert abs(sum(v for v in w.values() if v < 0) + 0.5) < 1e-9
    assert d["n_long"] == 3 and d["n_short"] == 17


def test_no_capital_scaling_applied_to_weights():
    # the 10% capital fraction is metadata for the review gate, NOT applied to the object
    w, _ = PS.target_weights(_sig(), _DQ_OK)
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-9
    assert PS.SLEEVE["safety"]["capital_fraction_at_review"] == 0.10


# --- fail closed on any data doubt ---
def test_fail_closed_on_stale_missing_or_feed_down():
    for bad in ({"earnings_age_days": 5, "missing_symbols_ratio": 0.0, "feed_available": True},
                {"earnings_age_days": 1, "missing_symbols_ratio": 0.10, "feed_available": True},
                {"earnings_age_days": 1, "missing_symbols_ratio": 0.0, "feed_available": False},
                {"earnings_age_days": None, "missing_symbols_ratio": 0.0, "feed_available": True}):
        w, d = PS.target_weights(_sig(), bad)
        assert d["blocked"] and w == {}, bad
    w, d = PS.target_weights(_sig(), _DQ_OK)
    assert not d["blocked"] and w                                   # clean -> trades


# --- frozen signal + preserved metadata ---
def test_signal_is_frozen():
    assert PS.FROZEN_SIGNAL == "earnings_within_7d" and PS.SLEEVE["frozen"] is True
    m = PS.SLEEVE["validated_metrics"]
    assert m["net_sharpe_5bps"] == 1.11 and m["breakeven_bps"] == 56.0 and m["vol_corr"] == 0.03
    c = PS.SLEEVE["construction"]
    assert c["gross_target"] == 1.0 and c["net_target"] == 0.0 and c["market_neutral"] is True


def test_registry_export_includes_earnings_within_7d(tmp_path):
    p = tmp_path / "reg.jsonl"
    PS.export_registry(p, as_of="2026-06-18")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    r = rows[0]
    assert r["alpha_id"] == "earnings_within_7d" and r["decision"] == "ACCEPT"
    assert r["status"] == "PAPER_PROMOTED" and r["promotion_requires_shadow_period"] is True
    assert r["construction"]["gross_target"] == 1.0


# --- shadow promotion gate ---
def _shadow_df(days, ic=0.02, spread=0.001, pnl=0.0005, outages=0, turnover=0.24):
    rows = []
    for i in range(days):
        blk = i < outages
        rows.append({"date": f"d{i}", "blocked": blk, "daily_ic": np.nan if blk else ic,
                     "rank_spread": np.nan if blk else spread, "daily_pnl": np.nan if blk else pnl,
                     "turnover": np.nan if blk else turnover, "data_outage": blk})
    return pd.DataFrame(rows)


def test_shadow_gate_ready_on_directional_evidence_not_ic_equality():
    # paper IC far from research IC in MAGNITUDE but same sign + positive spread/Sharpe + clean -> READY
    rep = SH.compare_to_research(_shadow_df(70, ic=0.005), research_ic=0.02, research_turnover=0.24)
    assert rep["verdict"] == "READY_FOR_CAPITAL_REVIEW" and SH.capital_review_ready(rep)
    assert rep["ic_sign_consistent"] and rep["rank_spread_positive"] and rep["sharpe_positive"]


def test_shadow_gate_blocks_on_operational_failures():
    rep = SH.compare_to_research(_shadow_df(70, outages=5), research_ic=0.02, research_turnover=0.24)
    assert not rep["promotion_ready"] and rep["operational_failures"] == 5


def test_standalone_runner_frozen_weights_parity():
    # the TCC-exempt stdlib runner must reproduce the canonical frozen object EXACTLY (no drift)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sds", PKG / "shadow_daily_standalone.py")
    STD = importlib.util.module_from_spec(spec); spec.loader.exec_module(STD)
    for n_in in (2, 3, 7):
        sig = {f"S{i}": (1.0 if i < n_in else 0.0) for i in range(20)}
        ws, ds = STD.frozen_weights(sig, _DQ_OK)
        wp, dp = PS.target_weights(sig, _DQ_OK)
        assert set(ws) == set(wp) and all(abs(ws[k] - wp[k]) < 1e-9 for k in ws)
        assert ds["gross"] == dp["gross"] and ds["n_long"] == dp["n_long"] and ds["net"] == dp["net"]
    for bad in ({"earnings_age_days": 5, "missing_symbols_ratio": 0.0, "feed_available": True},
                {"earnings_age_days": 1, "missing_symbols_ratio": 0.10, "feed_available": True},
                {"earnings_age_days": 1, "missing_symbols_ratio": 0.0, "feed_available": False}):
        assert STD.frozen_weights({"A": 1.0, "B": 0.0}, bad)[1]["blocked"] is True
        assert PS.target_weights({"A": 1.0, "B": 0.0}, bad)[1]["blocked"] is True


def test_shadow_gate_blocks_on_underrun_or_wrong_direction():
    assert not SH.compare_to_research(_shadow_df(30), 0.02, 0.24)["promotion_ready"]              # too short
    flip = SH.compare_to_research(_shadow_df(70, ic=-0.02, spread=-0.001, pnl=-0.0005), 0.02, 0.24)
    assert not flip["promotion_ready"] and flip["verdict"] == "SHADOW_FAIL"                       # wrong sign/neg


if __name__ == "__main__":
    import inspect
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            (fn(Path("/tmp")) if "tmp_path" in inspect.signature(fn).parameters else fn())
            print("ok", name)
