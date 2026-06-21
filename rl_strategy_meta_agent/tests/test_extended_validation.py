"""Tests for the extended-validation decision logic + placebo. Network-free."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PKG))

import run_extended_validation as EV


def _rec(**kw):
    base = dict(net_sharpe_5bps=1.0, breakeven_bps=50.0, vol_corr=0.05, net_bootstrap_p=0.01,
                placebo_p=0.01, holdout_net_sharpe=0.3, wf_positive_year_frac=0.7)
    base.update(kw); return base


def test_accept_requires_all_gates():
    assert EV.decide(_rec(), fdr_ok=True) == "ACCEPT"
    assert EV.decide(_rec(), fdr_ok=False) != "ACCEPT"            # FDR is necessary


def test_reject_on_cost_and_volatility():
    assert EV.decide(_rec(breakeven_bps=3.0), fdr_ok=True) == "REJECT"      # costs kill
    assert EV.decide(_rec(net_sharpe_5bps=-0.5), fdr_ok=True) == "REJECT"
    assert EV.decide(_rec(vol_corr=0.8), fdr_ok=True) == "REJECT"           # vol in disguise


def test_quarantine_when_underpowered():
    # positive net + placebo-ish but bootstrap/FDR fail -> QUARANTINE not ACCEPT/REJECT
    assert EV.decide(_rec(net_bootstrap_p=0.10, placebo_p=0.05), fdr_ok=False) == "QUARANTINE"


def test_placebo_high_for_noise_signal():
    # a signal independent of returns -> real Sharpe sits inside the placebo null (p not tiny)
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-01-01", periods=120, freq="B")
    rows = []
    for s in range(12):
        for t in ts:
            rows.append((f"S{s}", t, rng.normal(), rng.normal() * 0.01))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "noise", "fut_ret_1"]).set_index(
        ["symbol", "timestamp"]).sort_index()
    mask = np.ones(len(p), bool)
    pp = EV._placebo_p(p, p["noise"], mask, real_gross_sharpe=0.0, n=60)
    assert pp > 0.2                                              # noise is not distinguishable


def test_no_live_trading_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "premarket_trade", "run_v2", "live_trade", ".orders")
    for f in ("run_extended_validation.py", "daily_bars.py"):
        mods = []
        for n in ast.walk(ast.parse((PKG / f).read_text())):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            assert not any(b in m.lower() for b in banned), f"{f} imports {m}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
