"""Tests for the alpha grammar: operators correct, leak-free, IC sane, mutate valid."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import alphas as A


def _panel(nsym=6, nt=60):
    ts = pd.date_range("2024-03-04 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        rng = np.random.default_rng(s)
        c = np.cumsum(rng.normal(0, 1, nt)) + 100
        for i, t in enumerate(ts):
            rows.append((f"S{s}", t, c[i], c[i] + 0.3, c[i] - 0.3, c[i] + 0.1, 1000.0 + i + s))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol")["close"].shift(1))
    return p.set_index(["symbol", "timestamp"]).sort_index()


def test_ts_mean_matches_manual_rolling():
    p = _panel()
    v = A.eval_alpha(("ts_mean", ("in", "close"), 5), p)
    manual = p["close"].groupby(level=0).transform(lambda x: x.rolling(5, min_periods=2).mean())
    assert np.allclose(v.dropna().values, manual.dropna().values, atol=1e-9)


def test_cs_demean_centers_each_timestamp():
    p = _panel()
    v = A.eval_alpha(("cs_demean", ("in", "close")), p)
    assert np.allclose(v.groupby(level=1).sum().dropna().values, 0.0, atol=1e-6)


def test_cs_rank_orders_within_timestamp():
    p = _panel()
    v = A.eval_alpha(("cs_rank", ("in", "close")), p)
    t0 = p.index.get_level_values(1)[0]
    at_t0 = v.xs(t0, level=1)
    raw = p["close"].xs(t0, level=1)
    # cs_rank must be monotone in the raw input across the cross-section
    assert (at_t0.rank().values == raw.rank().values).all()


def test_no_lookahead_future_bars_do_not_change_past_alpha():
    p = _panel()
    node = ("cs_zscore", ("ts_delta", ("in", "close"), 3))
    v = A.eval_alpha(node, p)
    k = p.index.get_level_values(1)[30]
    bad = p.copy()
    bad.loc[bad.index.get_level_values(1) > k, ["open", "high", "low", "close", "volume"]] *= 7
    v2 = A.eval_alpha(node, bad)
    early = v.index.get_level_values(1) <= k
    assert np.allclose(v[early].dropna().values, v2[early].dropna().values, atol=1e-9)


def test_ic_of_alpha_against_itself_is_one():
    p = _panel()
    v = A.eval_alpha(("ts_delta", ("in", "close"), 1), p)
    ic, icir, n = A.spearman_ic(v, v)
    assert ic > 0.99 and n > 20


def test_safe_div_no_inf():
    p = _panel()
    v = A.eval_alpha(("div", ("in", "ret"), ("in", "ret")), p)
    assert np.isfinite(v.dropna()).all()


def test_random_and_mutate_eval():
    p = _panel()
    rng = np.random.default_rng(3)
    for _ in range(20):
        node = A.random_alpha(rng, ["ret", "close", "volume"])
        v = A.eval_alpha(node, p)
        assert v.shape[0] == len(p)
        m = A.mutate(node, rng, ["ret", "close", "volume"])
        assert A.eval_alpha(m, p).shape[0] == len(p)
        assert isinstance(A.formula_str(node), str) and len(A.alpha_id(node)) == 12


def test_no_broker_path():
    src = Path(A.__file__).read_text().lower()
    for banned in ("submit_order", "base_url", "tradingclient", "strategy.entry"):
        assert banned not in src


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
