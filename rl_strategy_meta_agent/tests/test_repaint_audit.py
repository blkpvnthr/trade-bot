"""Tests for the repaint/lookahead audit (static scan + stats implausibility)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import repaint_audit as R


def test_pre_v3_security_signal_is_invalid():
    src = ("//@version=2\nch1 = security(tickerid, '720', open)\n"
           "ch2 = security(tickerid, '720', close)\n"
           "longCondition = crossover(ch2, ch1)\nstrategy.entry('.', strategy.long)")
    s = R.scan_pine_source(src)
    assert s["signal_uses_repainting_security"]
    assert s["static_verdict"] == "invalid"


def test_explicit_lookahead_on_is_invalid():
    src = "//@version=5\nx = request.security(syminfo.tickerid,'D',close, lookahead=barmerge.lookahead_on)"
    assert R.scan_pine_source(src)["static_verdict"] == "invalid"


def test_clean_modern_source_is_safe():
    src = "//@version=5\nlongCondition = ta.crossover(close, ta.sma(close, 20))\n"
    assert R.scan_pine_source(src)["static_verdict"] == "safe"


def test_implausible_trades_flagged():
    perfect = pd.DataFrame({"return_pct": [0.02] * 95 + [-0.01] * 5,
                            "mae_pct": [0.0] * 90 + [0.03] * 10})
    st = R.signal_implausibility(perfect)
    assert st["stats_verdict"] == "invalid"
    assert any("adverse excursion" in f for f in st["flags"])


def test_plausible_trades_safe():
    ok = pd.DataFrame({"return_pct": [0.01, -0.02, 0.03, -0.01, 0.015, -0.025, 0.005, -0.03],
                       "mae_pct": [0.02, 0.05, 0.01, 0.04, 0.03, 0.06, 0.02, 0.05]})
    assert R.signal_implausibility(ok)["stats_verdict"] == "safe"


def test_combine_escalates_to_invalid():
    static = {"static_verdict": "suspect", "signal_uses_repainting_security": True}
    stats = {"stats_verdict": "suspect"}
    assert R.combine(static, stats) == "invalid"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
