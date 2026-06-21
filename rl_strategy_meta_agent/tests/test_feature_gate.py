"""Tests for the orthogonal-feature gate. Research-only; runs without EO/vendors."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "experiments" / "orthogonal_features"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

import feature_gate as G


def _panel(nsym=20, nt=200, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        for t in ts:
            vol = abs(rng.normal())
            orth = rng.normal()
            rows.append((f"S{s}", t, vol, orth, rng.normal(), 0.3 * orth + 0.02 * rng.normal()))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "atr_pct", "_orth", "_noise", "fut_ret_1"]
                     ).set_index(["symbol", "timestamp"]).sort_index()
    p["rel_volume"] = p["atr_pct"]; p["volume_acceleration"] = p["atr_pct"]
    return p


def test_gate_separates_signal_noise_and_volatility():
    p = _panel()
    rng = np.random.default_rng(1)
    cand = pd.DataFrame({"vol_proxy": p["atr_pct"] + 0.01 * rng.normal(size=len(p)),
                         "orthogonal_signal": p["_orth"],
                         "pure_noise": p["_noise"]}, index=p.index)
    rep = G.study(p, cand, accepted_nodes=None)
    v = {r["feature"]: r["verdict"] for r in rep["features"]}
    assert v["orthogonal_signal"] == "ORTHOGONAL_INFORMATIVE", v
    assert v["pure_noise"] == "UNINFORMATIVE", v
    assert v["vol_proxy"] in ("REDUNDANT_VOLATILITY", "UNINFORMATIVE"), v


def test_redundant_volatility_flagged_when_corr_high():
    p = _panel()
    cand = pd.DataFrame({"twin": p["atr_pct"] * 1.0}, index=p.index)   # identical to vol cluster
    rep = G.study(p, cand, accepted_nodes=None)
    # a perfect volatility twin must never be called orthogonal-informative
    assert rep["features"][0]["verdict"] != "ORTHOGONAL_INFORMATIVE"


def test_deterministic():
    p = _panel()
    cand = pd.DataFrame({"orthogonal_signal": p["_orth"]}, index=p.index)
    a = G.study(p, cand, accepted_nodes=None)["features"][0]
    b = G.study(p, cand, accepted_nodes=None)["features"][0]
    assert a == b


def test_no_live_trading_imports():
    # ban broker/order/execution/live modules. Read-only data/news (the Alpaca *data*
    # API via data_sources.alpaca_news_client) is research-allowed, not a trading path.
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "dashboard", "premarket_trade", "tjl_engine",
              "alpaca_feed_manager", "run_v2", "live_trade", ".orders")
    for f in PKG.glob("*.py"):
        tree = ast.parse(f.read_text())
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            assert not any(b in m.lower() for b in banned), f"{f.name} imports live module {m}"
            assert m.split(".")[0] != "run", f"{f.name} imports run.py"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
