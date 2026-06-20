"""Tests for the accepted-alpha registry gates + reasons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import alpha_registry as R


def _c(formula, ic, oos_ic, p, sig, turn, leak=True):
    return {"formula": formula, "ic": ic, "oos_ic": oos_ic, "p_value": p,
            "fdr_significant": sig, "turnover": turn, "leak_ok": leak}


def _corr(formulas):
    return pd.DataFrame(np.eye(len(formulas)), index=formulas, columns=formulas)


def test_clean_alpha_accepted():
    acc, rej = R.promote([_c("a", 0.3, 0.28, 0.0, True, 0.3)], _corr(["a"]))
    assert len(acc) == 1 and rej == [] and acc[0]["corr_to_existing"] == 0.0


def test_each_gate_rejects_with_reason():
    cands = [_c("sign", 0.2, -0.05, 0.0, True, 0.3),
             _c("decay", 0.25, 0.02, 0.0, True, 0.3),
             _c("fdr", 0.22, 0.20, 0.3, False, 0.3),
             _c("turn", 0.21, 0.20, 0.0, True, 99.0),
             _c("leak", 0.21, 0.20, 0.0, True, 0.3, leak=False)]
    acc, rej = R.promote(cands, _corr([c["formula"] for c in cands]))
    assert acc == []
    reasons = {r["formula"]: r["reasons"][0].split("(")[0] for r in rej}
    assert reasons == {"sign": "oos_sign_flip", "decay": "ic_decay_excessive",
                       "fdr": "fdr_not_significant", "turn": "turnover_above_cap",
                       "leak": "leakage_violation"}


def test_correlation_gate_drops_redundant():
    cands = [_c("a", 0.3, 0.28, 0.0, True, 0.3), _c("b", 0.25, 0.24, 0.0, True, 0.3)]
    corr = _corr(["a", "b"]); corr.loc["a", "b"] = corr.loc["b", "a"] = 0.95
    acc, rej = R.promote(cands, corr, {"corr_cap": 0.7})
    assert [a["formula"] for a in acc] == ["a"]
    assert rej[0]["formula"] == "b" and "correlated_redundant" in rej[0]["reasons"][0]


def test_oos_missing_is_rejected():
    acc, rej = R.promote([_c("a", 0.3, None, 0.0, True, 0.3)], _corr(["a"]))
    assert acc == [] and "oos_ic_missing" in rej[0]["reasons"]


def test_is_pit_safe():
    assert R.is_pit_safe(("ts_corr", ("in", "close"), ("in", "volume"), 20))
    assert not R.is_pit_safe(("ts_delay", ("in", "x"), -2))      # future shift
    assert not R.is_pit_safe(("cs_rank", ("ts_mean", ("in", "x"), 0)))  # zero window


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
