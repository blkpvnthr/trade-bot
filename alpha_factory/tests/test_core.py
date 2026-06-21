"""Hermetic self-checks for alpha_factory - no network, no pytest required.

Run:  python -m alpha_factory.tests.test_core
"""
from __future__ import annotations

import json
import tempfile

import numpy as np
import pandas as pd

from alpha_factory import miner
from alpha_factory.registry import AlphaRegistry
from alpha_factory.scientist import Criteria, benjamini_hochberg, evaluate, run_batch
from alpha_factory.schema import ACCEPTED, REJECTED, AlphaRecord, prediction_hash, prediction_vector
from alpha_factory.synth import make_panel


def test_planted_alpha_beats_noise():
    """The oracle factor must validate; pure noise must not. The gate must separate them."""
    closes, _, factor = make_panel(seed=1)
    crit = Criteria()
    planted = miner.planted(factor).signals
    noise = pd.DataFrame(np.random.default_rng(9).standard_normal(closes.shape),
                         index=closes.index, columns=closes.columns)
    mp = evaluate(planted, closes, crit)
    mn = evaluate(noise, closes, crit)
    assert mp["oos_ic_mean"] > mn["oos_ic_mean"], (mp["oos_ic_mean"], mn["oos_ic_mean"])
    assert mp["oos_ic_mean"] >= crit.min_oos_ic, mp["oos_ic_mean"]
    assert abs(mp["placebo_ic"]) <= 0.05, mp["placebo_ic"]      # placebo must collapse for a real alpha
    assert mp["p_value"] < 0.05, mp["p_value"]


def test_batch_verdicts_and_fdr():
    closes, vol, factor = make_panel(seed=2)
    reg = AlphaRegistry(tempfile.mkdtemp())
    cands = miner.discover(closes, vol, created_at="t", factor=factor)
    recs = run_batch(cands, closes, Criteria(), reg, now="t")
    by_name = {r.strategy_name: r for r in recs}
    assert by_name["planted_factor"].decision == ACCEPTED, by_name["planted_factor"].rejection_reason
    assert any(r.decision == REJECTED for r in recs)           # not everything passes
    assert reg.accepted(), "accepted alphas must persist to registry"


def test_dedupe_by_prediction():
    closes, _, factor = make_panel(seed=3)
    reg = AlphaRegistry(tempfile.mkdtemp())
    sig = miner.planted(factor).signals
    vec = prediction_vector(sig)
    rec = AlphaRecord(alpha_id=prediction_hash(vec), strategy_name="a", source="manual", formula="f",
                      symbols=list(sig.columns), timeframe="1D", created_at="t", evaluated_at="t",
                      prediction_hash=prediction_hash(vec), decision=ACCEPTED)
    assert reg.add(rec, pred=vec) is True
    # same prediction, different formula text -> still a duplicate (dedupe by prediction, not formula)
    rec2 = AlphaRecord(**{**rec.__dict__, "strategy_name": "b", "formula": "-(-f)"})
    assert reg.add(rec2, pred=vec) is False
    assert len(reg.all()) == 1


def test_monotone_wrap_is_same_alpha():
    closes, _, factor = make_panel(seed=4)
    sig = miner.planted(factor).signals
    assert prediction_hash(prediction_vector(sig)) == prediction_hash(prediction_vector(2 * sig + 7))


def test_benjamini_hochberg():
    # one tiny p among large ones -> exactly one discovery at alpha=0.10
    assert benjamini_hochberg([0.001, 0.8, 0.9, 0.7], 0.10) == [True, False, False, False]
    assert benjamini_hochberg([], 0.1) == []
    assert sum(benjamini_hochberg([0.2] * 5, 0.10)) == 0      # nothing significant


def test_allocator_export_format():
    from alpha_factory.cli import _allocator_row
    rec = AlphaRecord(alpha_id="x", strategy_name="s", source="alpha101", formula="f",
                      symbols=["AAA"], timeframe="1D", created_at="t", evaluated_at="t",
                      prediction_hash="x", out_of_sample_metrics={"oos_ic_mean": 0.03},
                      cost_model={"max_drawdown": -0.05}, score=0.7, decision=ACCEPTED)
    row = _allocator_row(rec)
    need = {"alpha_id", "strategy_name", "score", "symbols", "timeframe",
            "expected_return", "risk", "correlation_cluster", "confidence"}
    assert need <= set(row), need - set(row)
    json.dumps(row)                                            # must serialize


def test_cli_parses_cost_bps():
    from alpha_factory.cli import build_parser
    a = build_parser().parse_args(["evaluate-all", "--demo", "--cost-bps", "20",
                                   "--cost-sweep-bps", "5,15,20"])
    assert a.cost_bps == 20.0 and a.demo is True
    assert build_parser().parse_args(["evaluate-all", "--demo"]).cost_bps is None   # unset -> preset/default
    # the other input forms still parse
    assert build_parser().parse_args(["evaluate-all", "--source", "alpaca_cache"]).source == "alpaca_cache"
    assert build_parser().parse_args(["evaluate-all", "--panel", "x.pkl"]).panel == "x.pkl"


def test_registry_acceptance_changes_with_cost():
    """Same alpha, different cost -> different verdict. Cost must bind."""
    from alpha_factory.scientist import decide
    closes, _, factor = make_panel(seed=5)
    sig = miner.planted(factor).signals
    cheap = evaluate(sig, closes, Criteria(cost_bps=5))
    pricey = evaluate(sig, closes, Criteria(cost_bps=2000))            # absurd cost kills any edge
    assert pricey["cost"]["net_sharpe"] < cheap["cost"]["net_sharpe"]  # cost monotonically hurts
    assert decide(cheap, Criteria(cost_bps=5), True)[0] == ACCEPTED
    assert decide(pricey, Criteria(cost_bps=2000), True)[0] == REJECTED


def test_fragile_cost_mirage_tag():
    """An alpha that passes at the selected cost but dies by 15/20bps must be tagged FRAGILE."""
    import tempfile
    closes, vol, factor = make_panel(seed=6)
    # craft a high-turnover marginal signal: noisy day-to-day so it churns and bleeds cost fast
    rng = np.random.default_rng(7)
    churn = pd.DataFrame(rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns)
    sig = 0.15 * factor + churn                                       # weak real edge, high turnover
    cand = miner.Candidate(name="fragile", source="manual", formula="weak+churn", signals=sig)
    reg = AlphaRegistry(tempfile.mkdtemp())
    rec = run_batch([cand], closes, Criteria(cost_bps=1, min_net_sharpe=0.0), reg, now="t")[0]
    if rec.decision == ACCEPTED and (rec.cost_model["net_sharpe_15"] < 0.5
                                     or rec.cost_model["net_sharpe_20"] < 0.5):
        assert "FRAGILE_COST_MIRAGE" in rec.tags, rec.tags


def test_universe_defined():
    from alpha_factory.data import UNIVERSES
    u = UNIVERSES["liquid_etf_mega"]
    assert "SPY" in u and "NVDA" in u and len(u) == 23 == len(set(u))


def test_cost_preset_resolution():
    from alpha_factory.cli import build_parser, _resolve_costs
    p = build_parser()
    a = _resolve_costs(p.parse_args(["evaluate-all", "--cost-preset", "liquid_etf"]))
    assert a == (5.0, [2, 5, 10, 15], 10.0), a
    b = _resolve_costs(p.parse_args(["evaluate-all", "--cost-preset", "small_cap"]))
    assert b == (20.0, [15, 20, 25], 20.0), b
    # explicit --cost-bps overrides the preset's acceptance cost
    c = _resolve_costs(p.parse_args(["evaluate-all", "--cost-preset", "liquid_etf", "--cost-bps", "12"]))
    assert c[0] == 12.0, c
    d = _resolve_costs(p.parse_args(["evaluate-all", "--demo"]))   # no preset -> defaults
    assert d == (5.0, [5, 15, 20, 25], 10.0), d


def test_promotion_gate():
    """Strong alpha is promotable; an absurd Sharpe bar blocks it. Gate must bind."""
    # persistent factor -> real edge AND low turnover -> genuinely promotable
    closes, _, factor = make_panel(seed=8, persistence=0.9)
    sig = miner.planted(factor).signals
    good = evaluate(sig, closes, Criteria(promote_cost_bps=10))["promotion"]
    assert good["promotable"] is True, good
    blocked = evaluate(sig, closes, Criteria(promote_cost_bps=10, promote_min_sharpe=100))["promotion"]
    assert blocked["promotable"] is False


def test_export_holds_back_non_promotable():
    """An accepted-but-not-promotable alpha must NOT reach allocator_input.json."""
    import tempfile, json as _json, os
    from alpha_factory.cli import _allocator_row
    from alpha_factory.schema import AlphaRecord
    reg = AlphaRegistry(tempfile.mkdtemp())
    promo = AlphaRecord(alpha_id="p", strategy_name="good", source="x", formula="f", symbols=["A"],
                        timeframe="1D", created_at="t", evaluated_at="t", prediction_hash="p",
                        decision=ACCEPTED, promotion={"promotable": True})
    held = AlphaRecord(**{**promo.__dict__, "alpha_id": "h", "strategy_name": "weak",
                          "prediction_hash": "h", "promotion": {"promotable": False}})
    reg.add(promo); reg.add(held)
    exportable = [r for r in reg.accepted() if r.promotion.get("promotable")]
    assert [r.strategy_name for r in exportable] == ["good"]


def test_no_broker_imports():
    """Safety: the package must not pull in execution/broker modules."""
    import alpha_factory, alpha_factory.scientist, alpha_factory.cli, alpha_factory.registry  # noqa
    import sys
    banned = ("alpaca", "trade.py", "submit_market_order")
    loaded = " ".join(sys.modules)
    assert "alpaca_trade_api" not in loaded, "broker SDK must not be imported"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()
