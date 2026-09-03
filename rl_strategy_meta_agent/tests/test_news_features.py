"""Tests for leak-free news-feature alignment. Network-free (synthetic news)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))

import news_features as NF


def _idx(syms=("A", "B"), n=12):
    return pd.MultiIndex.from_product(
        [list(syms), pd.date_range("2024-01-02 10:00", periods=n, freq="h")],
        names=["symbol", "timestamp"])


def test_alignment_is_leak_free():
    idx = _idx(("A",))
    T = pd.Timestamp("2024-01-02 15:00")
    news = {"A": pd.DataFrame({"time": [T], "score": [0.7], "relevance": [1.0],
                              "pos": [1.0], "neg": [0.0], "neu": [0.0]})}
    out = NF.align(idx, news, {"A": np.array([], dtype="datetime64[ns]")}, {"window_h": 6.0})
    bars = idx.get_level_values(1)
    # news at T must be invisible to every bar <= T, visible to bars in (T, T+window]
    assert out["news_volume_count"][bars <= T].max() == 0.0
    assert out["news_volume_count"][(bars > T) & (bars <= T + pd.Timedelta(hours=6))].min() >= 1.0
    assert out["news_sentiment_mean"][bars <= T].abs().max() == 0.0


def test_window_decays_out_old_news():
    idx = _idx(("A",), n=24)
    T = pd.Timestamp("2024-01-02 11:00")
    news = {"A": pd.DataFrame({"time": [T], "score": [0.5], "relevance": [1.0],
                              "pos": [1.0], "neg": [0.0], "neu": [0.0]})}
    out = NF.align(idx, news, {"A": np.array([], dtype="datetime64[ns]")}, {"window_h": 3.0})
    bars = idx.get_level_values(1)
    far = bars > T + pd.Timedelta(hours=3)
    assert out["news_volume_count"][far].max() == 0.0      # rolls out of the 3h window


def test_earnings_within_flag():
    idx = _idx(("A",), n=10)
    ed = {"A": np.array([np.datetime64("2024-01-02 18:00")], dtype="datetime64[ns]")}
    out = NF.align(idx, {"A": NF.load_news.__wrapped__ if False else
                         pd.DataFrame(columns=["time", "score", "relevance", "pos", "neg", "neu"])},
                   ed, {"window_h": 6.0})
    bars = idx.get_level_values(1)
    days_to = (pd.Timestamp("2024-01-02 18:00") - bars) / pd.Timedelta(days=1)
    within7 = out["earnings_within_7d"].to_numpy()
    # all bars before the report and within 7 days -> flagged
    assert within7[(days_to >= 0) & (days_to <= 7)].min() == 1.0


BANNED_IMPORTS = ("alpaca.trading", "trading_client", "executor", "broker",
                  "submit_order", "execution", "premarket_trade", "run_v2",
                  "live_trade", "dashboard", ".orders")


def _imported_modules(path):
    mods = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module)
    return mods


def test_no_order_execution_imports():
    # news/finbert/av_catalyst are READ-ONLY data (Alpaca *data* API, not trading);
    # scan IMPORTS only and ban broker/order/execution/live modules.
    for f in PKG.glob("*.py"):
        for m in _imported_modules(f):
            assert not any(b in m.lower() for b in BANNED_IMPORTS), f"{f.name} imports {m}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
