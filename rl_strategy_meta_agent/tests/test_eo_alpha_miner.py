"""Tests for the EO-backed symbolic alpha miner. C++/EO-dependent tests skip
cleanly when the binary is not built (CI without EO)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "experiments" / "eo_alpha_miner"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from rl_meta.alphas import eval_alpha, formula_str
import backend as BK
import features as FEAT
import python_ga as PGA


def _panel(nsym=10, nt=60, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        for t in ts:
            f0, f1 = rng.normal(size=2)
            rows.append((f"S{s}", t, f0, f1, f0 - 0.5 * f1 + 0.3 * rng.normal()))
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "f0", "f1", "target"]).set_index(
        ["symbol", "timestamp"]).sort_index()


# --- expression evaluation -------------------------------------------------
def test_sexp_parses_and_evaluates():
    node = BK.parse_sexp("add(f0,mul(f1,2))", ["f0", "f1"])
    assert node == ("add", ("in", "f0"), ("mul", ("in", "f1"), ("const", 2.0)))
    p = _panel()
    out = eval_alpha(node, p)                       # f0 + 2*f1
    expected = p["f0"] + 2 * p["f1"]
    assert np.allclose(out.values, expected.values, atol=1e-9)


def test_safe_div_is_nan_safe():
    node = BK.parse_sexp("safe_div(f0,f1)", ["f0", "f1"])
    p = _panel()
    p = p.copy(); p["f1"] = 0.0                      # divide by zero everywhere
    assert eval_alpha(node, p).notna().all()


# --- fitness calculation ---------------------------------------------------
def test_fitness_recovers_known_signal_sign():
    p = _panel()
    good = PGA._fitness(("in", "f0"), p, p["target"])     # f0 is positively related to target
    noise = PGA._fitness(("const", 0.0), p, p["target"])  # constant -> ~0 IC
    assert good > noise


# --- deterministic search --------------------------------------------------
def test_python_ga_deterministic():
    p = _panel()
    cfg = {"seed": 3, "population": 60, "generations": 8}
    a = PGA.mine(p, p["target"], ["f0", "f1"], cfg)
    b = PGA.mine(p, p["target"], ["f0", "f1"], cfg)
    assert [formula_str(n) for n in a] == [formula_str(n) for n in b]


# --- leak-free features ----------------------------------------------------
def test_features_have_no_future_in_inputs():
    p, feats, tgt = FEAT.build_features(_ohlcv())
    # last bar per symbol has no future return label
    last = p[tgt].groupby(level=0).apply(lambda s: s.iloc[-1])
    assert last.isna().all()
    assert "target" not in feats


def _ohlcv():
    rng = np.random.default_rng(1)
    rows = []
    for s in ("A", "B", "C"):
        px = 100.0
        for t in pd.date_range("2024-01-02 14:30", periods=50, freq="15min", tz="UTC"):
            px *= 1 + rng.normal(0, 0.01)
            rows.append((s, t, px, px * 1.01, px * 0.99, px, rng.integers(1e4, 1e5)))
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
                        ).set_index(["symbol", "timestamp"])


# --- research-only: no trading/order/live imports --------------------------
def test_no_live_trading_imports():
    banned = ("alpaca", "broker", "execution", "order", "dashboard", "premarket",
              "tjl_engine", "alpaca_feed_manager", "run_v2", "live")
    for f in PKG.glob("*.py"):
        tree = ast.parse(f.read_text())
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            top = m.split(".")[0]
            assert not any(b in m for b in banned), f"{f.name} imports live module {m}"
            assert top != "run", f"{f.name} imports run.py"


# --- EO C++ backend (skip if not built) ------------------------------------
eo = pytest.mark.skipif(not BK.available(), reason="EO binary not built")


@eo
def test_eo_backend_runs_and_is_deterministic(tmp_path):
    p, feats, tgt = FEAT.build_features(_ohlcv())
    csv = str(tmp_path / "feat.csv")
    FEAT.export_for_eo(p, feats, tgt, csv)
    cfg = {"population": 100, "generations": 10, "seed": 42, "top_k": 10}
    a = BK.run(csv, str(tmp_path / "o1.txt"), cfg, feats)
    b = BK.run(csv, str(tmp_path / "o2.txt"), cfg, feats)
    assert a and [formula_str(n) for n in a] == [formula_str(n) for n in b]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and not name.endswith("deterministic"):
            try:
                fn()
                print("ok", name)
            except TypeError:
                print("skip (needs fixture)", name)
