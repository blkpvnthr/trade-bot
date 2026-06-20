"""AlphaEvolve-style formulaic-alpha grammar, evaluation, and IC scoring.

An alpha is a small formula tree over a (symbol, timestamp) MultiIndex panel of
base features. Time-series operators are strictly backward (rolling / shift /
ewm) so they are point-in-time; cross-section operators use only the
contemporaneous cross-section at one timestamp. Neither can see the future, so
any alpha here is leak-free by construction; only the LABELS look forward.

Tree node forms (plain tuples -> JSON-serialisable, mutable, hashable as str):
  ("in", name)
  ("ts_mean"|... , child, window)             # backward time-series, per symbol
  ("ts_corr", child_a, child_b, window)
  ("cs_rank"|"cs_demean"|"cs_zscore"|"winsorize", child)   # per timestamp
  ("add"|"sub"|"mul"|"div", child_a, child_b)

Research-only: no broker, no orders.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]
BASE_INPUTS = [
    "ret", "open", "high", "low", "close", "volume", "rsi", "atr_pct", "macd_hist",
    "rel_volume", "volume_acceleration", "vwap_dist", "pdh_dist", "pdl_dist",
    "or_high_dist", "or_low_dist", "upper_wick_pct", "lower_wick_pct",
    "candle_body_pct", "body_to_range_ratio", "spy_trend", "qqq_trend", "mkt_regime",
]


def _gsym(s):
    return s.groupby(level=0, group_keys=False)


def _gtime(s):
    return s.groupby(level=1, group_keys=False)


def _roll(s, w, fn):
    p = max(2, w // 2)
    return _gsym(s).transform(lambda x: getattr(x.rolling(w, min_periods=p), fn)())


def _pctrank(a):
    return float((a[-1] >= a).mean())


def _argpos(fn):
    return lambda s, w: _gsym(s).transform(
        lambda x: x.rolling(w, min_periods=w).apply(lambda a: float(fn(a)), raw=True))


def _decay_linear(s, w):
    wts = np.arange(1, w + 1, dtype=float); wts /= wts.sum()
    return _gsym(s).transform(
        lambda x: x.rolling(w, min_periods=w).apply(lambda a: float(np.dot(a, wts)), raw=True))


TS_UNARY = {
    "ts_mean": lambda s, w: _roll(s, w, "mean"),
    "ts_std": lambda s, w: _roll(s, w, "std"),
    "ts_sum": lambda s, w: _roll(s, w, "sum"),
    "ts_min": lambda s, w: _roll(s, w, "min"),
    "ts_max": lambda s, w: _roll(s, w, "max"),
    "ts_zscore": lambda s, w: (s - _roll(s, w, "mean")) / (_roll(s, w, "std") + 1e-9),
    "ts_delta": lambda s, w: s - _gsym(s).shift(w),
    "ts_ema": lambda s, w: _gsym(s).transform(lambda x: x.ewm(span=w, min_periods=max(2, w // 2)).mean()),
    "ts_rank": lambda s, w: _gsym(s).transform(
        lambda x: x.rolling(w, min_periods=max(2, w // 2)).apply(_pctrank, raw=True)),
    # -- 101 Formulaic Alphas operators --
    "ts_delay": lambda s, w: _gsym(s).shift(w),                       # delay(x, d)
    "decay_linear": _decay_linear,
    "ts_product": lambda s, w: _gsym(s).transform(
        lambda x: x.rolling(w, min_periods=w).apply(lambda a: float(np.prod(a)), raw=True)),
    "ts_argmax": _argpos(np.argmax),
    "ts_argmin": _argpos(np.argmin),
}
CS_UNARY = {
    "cs_rank": lambda s: _gtime(s).rank(pct=True) - 0.5,
    "rank": lambda s: _gtime(s).rank(pct=True),                       # 101-Alphas rank (0..1)
    "cs_demean": lambda s: s - _gtime(s).transform("mean"),
    "indneutralize": lambda s: s - _gtime(s).transform("mean"),       # no industry data -> demean
    "cs_zscore": lambda s: (s - _gtime(s).transform("mean")) / (_gtime(s).transform("std") + 1e-9),
    "scale": lambda s: s / (_gtime(s.abs()).transform("sum") + 1e-12),  # scale(x, a=1)
    "winsorize": lambda s: _winsorize(s),
    "log": lambda s: np.sign(s) * np.log1p(s.abs()),
    "abs": lambda s: s.abs(),
    "sign": lambda s: np.sign(s),
}
BIN = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b.replace(0, np.nan),
    "lt": lambda a, b: (a < b).astype(float), "gt": lambda a, b: (a > b).astype(float),
}


def _winsorize(s):
    m = _gtime(s).transform("mean"); sd = _gtime(s).transform("std")
    return s.clip(m - 3 * sd, m + 3 * sd)


def _ts_corr(a, b, w):
    ma, mb = _roll(a, w, "mean"), _roll(b, w, "mean")
    cov = _roll(a * b, w, "mean") - ma * mb
    va = (_roll(a * a, w, "mean") - ma * ma).clip(lower=0)
    vb = (_roll(b * b, w, "mean") - mb * mb).clip(lower=0)
    return cov / (np.sqrt(va * vb) + 1e-9)


def _ts_cov(a, b, w):                                                 # covariance(x, y, d)
    return _roll(a * b, w, "mean") - _roll(a, w, "mean") * _roll(b, w, "mean")


def evaluate(node, panel):
    op = node[0]
    if op == "in":
        return panel[node[1]]
    if op == "const":
        return pd.Series(float(node[1]), index=panel.index)
    if op in TS_UNARY:
        return TS_UNARY[op](evaluate(node[1], panel), node[2])
    if op == "ts_corr":
        return _ts_corr(evaluate(node[1], panel), evaluate(node[2], panel), node[3])
    if op == "covariance":
        return _ts_cov(evaluate(node[1], panel), evaluate(node[2], panel), node[3])
    if op == "signedpower":
        s = evaluate(node[1], panel); return np.sign(s) * (s.abs() ** float(node[2]))
    if op == "where":
        c = evaluate(node[1], panel)
        return evaluate(node[2], panel).where(c > 0, evaluate(node[3], panel))
    if op in CS_UNARY:
        return CS_UNARY[op](evaluate(node[1], panel))
    if op in BIN:
        return BIN[op](evaluate(node[1], panel), evaluate(node[2], panel))
    raise ValueError(f"unknown op {op}")


def eval_alpha(node, panel) -> pd.Series:
    return evaluate(node, panel).reindex(panel.index).replace([np.inf, -np.inf], np.nan)


# -- generation / mutation -------------------------------------------------
def random_alpha(rng, inputs=BASE_INPUTS, depth=0, max_depth=3):
    if depth == 0 and rng.random() < 0.6:                # alphas are usually cross-sectional
        return (str(rng.choice(list(CS_UNARY))), random_alpha(rng, inputs, 1, max_depth))
    if depth >= max_depth or rng.random() < 0.35:
        return ("in", str(rng.choice(inputs)))
    r = rng.random()
    if r < 0.35:
        return (str(rng.choice(list(TS_UNARY))), random_alpha(rng, inputs, depth + 1, max_depth),
                int(rng.choice(WINDOWS)))
    if r < 0.50:
        return ("ts_corr", random_alpha(rng, inputs, depth + 1, max_depth),
                random_alpha(rng, inputs, depth + 1, max_depth), int(rng.choice(WINDOWS)))
    if r < 0.72:
        return (str(rng.choice(list(CS_UNARY))), random_alpha(rng, inputs, depth + 1, max_depth))
    return (str(rng.choice(list(BIN))), random_alpha(rng, inputs, depth + 1, max_depth),
            random_alpha(rng, inputs, depth + 1, max_depth))


def _local_mutation(node, rng, inputs):
    op = node[0]
    if op == "in":
        return ("in", str(rng.choice(inputs)))
    if (op in TS_UNARY or op == "ts_corr") and rng.random() < 0.6:
        new = list(node); new[-1] = int(rng.choice(WINDOWS)); return tuple(new)
    if rng.random() < 0.5:                                # wrap in a cross-section op
        return (str(rng.choice(list(CS_UNARY))), node)
    return node


def mutate(node, rng, inputs=BASE_INPUTS):
    if rng.random() < 0.35 or len(node) == 2 and node[0] == "in":
        return _local_mutation(node, rng, inputs)
    kids = [i for i in range(1, len(node)) if isinstance(node[i], tuple)]
    if not kids:
        return _local_mutation(node, rng, inputs)
    i = int(rng.choice(kids))
    new = list(node); new[i] = mutate(node[i], rng, inputs); return tuple(new)


def formula_str(node) -> str:
    op = node[0]
    if op == "in":
        return node[1]
    if op == "const":
        return str(node[1])
    args = ",".join(formula_str(a) if isinstance(a, tuple) else str(a) for a in node[1:])
    return f"{op}({args})"


def alpha_id(node) -> str:
    return hashlib.md5(formula_str(node).encode()).hexdigest()[:12]


def category(node) -> str:
    f = formula_str(node)
    for tag in ("volume", "vwap", "pdh", "pdl", "or_high", "or_low", "wick", "body"):
        if tag in f:
            return {"or_high": "opening_range", "or_low": "opening_range",
                    "wick": "candle", "body": "candle", "pdh": "level", "pdl": "level"}.get(tag, tag)
    return "price"


# -- IC scoring ------------------------------------------------------------
def spearman_ic(alpha: pd.Series, target: pd.Series, min_xs: int = 5):
    """Mean per-timestamp rank correlation (IC) and IC_IR. Vectorised."""
    df = pd.DataFrame({"a": alpha, "t": target}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 100:
        return 0.0, 0.0, 0
    tl = df.index.get_level_values(1)
    ra = df["a"].groupby(tl).rank(); rt = df["t"].groupby(tl).rank()
    ca = ra - ra.groupby(tl).transform("mean"); ct = rt - rt.groupby(tl).transform("mean")
    num = (ca * ct).groupby(tl).sum()
    den = np.sqrt((ca * ca).groupby(tl).sum() * (ct * ct).groupby(tl).sum())
    ic_t = (num / den.replace(0, np.nan))
    sizes = df["a"].groupby(tl).size()
    ic_t = ic_t[sizes >= min_xs].dropna()
    if len(ic_t) < 20:
        return 0.0, 0.0, len(ic_t)
    return round(float(ic_t.mean()), 5), round(float(ic_t.mean() / (ic_t.std() + 1e-9)), 4), len(ic_t)


def ic_series(alpha: pd.Series, target: pd.Series, min_xs: int = 5) -> np.ndarray:
    """Per-timestamp IC array (for multiple-testing p-values). Same math as
    spearman_ic but returns the time series instead of its mean."""
    df = pd.DataFrame({"a": alpha, "t": target}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 100:
        return np.array([])
    tl = df.index.get_level_values(1)
    ra = df["a"].groupby(tl).rank(); rt = df["t"].groupby(tl).rank()
    ca = ra - ra.groupby(tl).transform("mean"); ct = rt - rt.groupby(tl).transform("mean")
    num = (ca * ct).groupby(tl).sum()
    den = np.sqrt((ca * ca).groupby(tl).sum() * (ct * ct).groupby(tl).sum())
    ic_t = (num / den.replace(0, np.nan))
    sizes = df["a"].groupby(tl).size()
    return ic_t[sizes >= min_xs].dropna().to_numpy()


def _demo():
    ts = pd.date_range("2024-03-04 14:30", periods=40, freq="15min", tz="UTC")
    rows = []
    for sym in ("AAA", "BBB", "CCC"):
        base = np.cumsum(np.random.default_rng(hash(sym) % 99).normal(0, 1, 40)) + 100
        for i, t in enumerate(ts):
            rows.append((sym, t, base[i], base[i] + 0.3, base[i] - 0.3, base[i] + 0.1, 1000 + i))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol")["close"].shift(1))
    panel = p.set_index(["symbol", "timestamp"]).sort_index()
    a = ("cs_rank", ("ts_mean", ("in", "ret"), 5))
    v = eval_alpha(a, panel)
    assert v.shape[0] == len(panel)
    # leak-free: corrupting the future leaves an earlier bar's alpha unchanged
    k = panel.index.get_level_values(1)[20]
    bad = panel.copy()
    bad.loc[bad.index.get_level_values(1) > k, "close"] *= 5
    v2 = eval_alpha(a, bad)
    early = v.index.get_level_values(1) <= k
    assert np.allclose(v[early].dropna().values,
                       v2[early].reindex(v.index)[early].dropna().values, atol=1e-9), "future leaked"
    ic, icir, n = spearman_ic(v, eval_alpha(("in", "ret"), panel))
    rng = np.random.default_rng(0)
    m = mutate(a, rng)
    assert eval_alpha(m, panel).shape[0] == len(panel)
    print(f"alphas._demo OK  '{formula_str(a)}' -> '{formula_str(m)}' | ic test {ic} over {n} ts")


if __name__ == "__main__":
    _demo()
