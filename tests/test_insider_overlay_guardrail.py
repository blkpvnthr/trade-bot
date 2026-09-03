"""Guardrail: an insider-book overlay must NOT be promoted on headline metrics alone.

Locks the Track-1 conclusion (docs/insider_track1_closed.md): to promote a sizing/exit
overlay it must improve net return AND hold Sharpe AND preserve the top-decile winners AND
beat a same-frequency random early-exit placebo. This test pins the rule so a future tweak
that only chases Sharpe/return cannot quietly pass.

Run: python tests/test_insider_overlay_guardrail.py   (no pytest dependency)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest_insider_giveback import overlay_promotable

BASELINE = {"total_return": 0.336, "sharpe_active": 2.569}


def _best(total, sharpe, retention, cut):
    return {"total_return": total, "sharpe_active": sharpe,
            "winner_retention": {"mean_retention": retention, "winners_cut": cut}}


def test_headline_good_but_tail_amputated_fails():
    # the actual give-back result: better return+Sharpe, but cuts winners and loses to placebo
    ok, checks = overlay_promotable(_best(0.3772, 2.849, 0.62, 6), BASELINE, placebo_q90=0.615)
    assert not ok
    assert checks["net_improves"] and checks["sharpe_holds"]      # headline passes...
    assert not checks["tail_preserved"] and not checks["beats_placebo"]  # ...guards fail


def test_headline_good_but_loses_to_placebo_fails():
    ok, checks = overlay_promotable(_best(0.40, 2.9, 0.95, 1), BASELINE, placebo_q90=0.50)
    assert not ok and not checks["beats_placebo"]


def test_full_pass_requires_all_four():
    ok, checks = overlay_promotable(_best(0.40, 2.9, 0.95, 1), BASELINE, placebo_q90=0.30)
    assert ok and all(checks.values())


def test_no_promotion_without_net_improvement():
    ok, _ = overlay_promotable(_best(0.30, 3.5, 0.99, 0), BASELINE, placebo_q90=0.10)
    assert not ok                                                # great Sharpe, but return < baseline


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
    print("guardrail OK")
