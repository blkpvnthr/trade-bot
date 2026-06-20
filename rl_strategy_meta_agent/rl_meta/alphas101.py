"""A faithful subset of Kakushadze (2015) "101 Formulaic Alphas", translated to
the rl_meta.alphas AST so they are evaluable, IC-scorable, mutatable, and usable
as domain-expert SEEDS for the evolution (the AlphaEvolve paper shows seeding
from a real alpha beats random init).

Operator name map (paper -> AST): rank->rank, delta->ts_delta, correlation->
ts_corr, covariance->covariance, delay->ts_delay, sum->ts_sum, stddev->ts_std,
Ts_Rank->ts_rank, ts_min/ts_max as-is, signedpower/scale/decay_linear/ts_argmax
as-is, log/abs/sign elementwise, "?:" -> where(lt(...)).

Caveats (documented, not hidden): indneutralize() has no industry data here so it
degrades to cross-sectional demean; `cap` alphas are excluded; `vwap` is the
session VWAP recovered from market_context; daily windows are reused as intraday
bar counts. So these are faithful STRUCTURES, not the production daily alphas.
Research-only.
"""
from __future__ import annotations


def IN(n):
    return ("in", n)


def C(v):
    return ("const", float(v))


def adv(d):
    return ("ts_mean", ("mul", IN("close"), IN("volume")), d)


_A = {
    "alpha001": ("sub", ("rank", ("ts_argmax", ("signedpower", ("where",
        ("lt", IN("ret"), C(0)), ("ts_std", IN("ret"), 20), IN("close")), 2.0), 5)), C(0.5)),
    "alpha002": ("mul", C(-1), ("ts_corr", ("rank", ("ts_delta", ("log", IN("volume")), 2)),
        ("rank", ("div", ("sub", IN("close"), IN("open")), IN("open"))), 6)),
    "alpha003": ("mul", C(-1), ("ts_corr", ("rank", IN("open")), ("rank", IN("volume")), 10)),
    "alpha004": ("mul", C(-1), ("ts_rank", ("rank", IN("low")), 9)),
    "alpha005": ("mul", ("rank", ("sub", IN("open"), ("div", ("ts_sum", IN("vwap"), 10), C(10)))),
        ("mul", C(-1), ("abs", ("rank", ("sub", IN("close"), IN("vwap")))))),
    "alpha006": ("mul", C(-1), ("ts_corr", IN("open"), IN("volume"), 10)),
    "alpha007": ("where", ("lt", adv(20), IN("volume")),
        ("mul", ("mul", C(-1), ("ts_rank", ("abs", ("ts_delta", IN("close"), 7)), 60)),
            ("sign", ("ts_delta", IN("close"), 7))), C(-1)),
    "alpha008": ("mul", C(-1), ("rank", ("sub",
        ("mul", ("ts_sum", IN("open"), 5), ("ts_sum", IN("ret"), 5)),
        ("ts_delay", ("mul", ("ts_sum", IN("open"), 5), ("ts_sum", IN("ret"), 5)), 10)))),
    "alpha011": ("mul", ("add", ("rank", ("ts_max", ("sub", IN("vwap"), IN("close")), 3)),
        ("rank", ("ts_min", ("sub", IN("vwap"), IN("close")), 3))), ("rank", ("ts_delta", IN("volume"), 3))),
    "alpha012": ("mul", ("sign", ("ts_delta", IN("volume"), 1)), ("mul", C(-1), ("ts_delta", IN("close"), 1))),
    "alpha013": ("mul", C(-1), ("rank", ("covariance", ("rank", IN("close")), ("rank", IN("volume")), 5))),
    "alpha014": ("mul", ("mul", C(-1), ("rank", ("ts_delta", IN("ret"), 3))), ("ts_corr", IN("open"), IN("volume"), 10)),
    "alpha015": ("mul", C(-1), ("ts_sum", ("rank", ("ts_corr", ("rank", IN("high")), ("rank", IN("volume")), 3)), 3)),
    "alpha016": ("mul", C(-1), ("rank", ("covariance", ("rank", IN("high")), ("rank", IN("volume")), 5))),
    "alpha017": ("mul", ("mul", ("mul", C(-1), ("rank", ("ts_rank", IN("close"), 10))),
        ("rank", ("ts_delta", ("ts_delta", IN("close"), 1), 1))),
        ("rank", ("ts_rank", ("div", IN("volume"), adv(20)), 5))),
    "alpha018": ("mul", C(-1), ("rank", ("add", ("add",
        ("ts_std", ("abs", ("sub", IN("close"), IN("open"))), 5), ("sub", IN("close"), IN("open"))),
        ("ts_corr", IN("close"), IN("open"), 10)))),
    "alpha019": ("mul", ("mul", C(-1), ("sign", ("add",
        ("sub", IN("close"), ("ts_delay", IN("close"), 7)), ("ts_delta", IN("close"), 7)))),
        ("add", C(1), ("rank", ("add", C(1), ("ts_sum", IN("ret"), 250))))),
    "alpha020": ("mul", ("mul", ("mul", C(-1), ("rank", ("sub", IN("open"), ("ts_delay", IN("high"), 1)))),
        ("rank", ("sub", IN("open"), ("ts_delay", IN("close"), 1)))),
        ("rank", ("sub", IN("open"), ("ts_delay", IN("low"), 1)))),
    "alpha022": ("mul", C(-1), ("mul", ("ts_delta", ("ts_corr", IN("high"), IN("volume"), 5), 5),
        ("rank", ("ts_std", IN("close"), 20)))),
    "alpha101": ("div", ("sub", IN("close"), IN("open")),
        ("add", ("sub", IN("high"), IN("low")), C(0.001))),
}

# inputs these alphas reference (the panel must provide them)
REQUIRED_INPUTS = ["open", "high", "low", "close", "volume", "vwap", "ret"]


def library() -> list:
    """[(name, ast_node), ...]."""
    return list(_A.items())


def _demo():
    import numpy as np
    import pandas as pd
    from rl_meta.alphas import eval_alpha, formula_str
    ts = pd.date_range("2024-03-04 14:30", periods=300, freq="15min", tz="UTC")
    rows = []
    for s in range(8):
        rng = np.random.default_rng(s)
        c = np.cumsum(rng.normal(0, 1, 300)) + 100
        for i, t in enumerate(ts):
            rows.append((f"S{s}", t, c[i], c[i] + 0.4, c[i] - 0.4, c[i] + 0.1, 1000.0 + i, c[i] + 0.05))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low",
                                    "close", "volume", "vwap"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol")["close"].shift(1))
    panel = p.set_index(["symbol", "timestamp"]).sort_index()
    ok = 0
    for name, node in library():
        v = eval_alpha(node, panel)
        assert v.shape[0] == len(panel), name
        assert v.notna().any(), f"{name} all-NaN"
        ok += 1
    print(f"alphas101._demo OK  {ok} alphas evaluate | e.g. alpha101 = {formula_str(_A['alpha101'])}")


if __name__ == "__main__":
    _demo()
