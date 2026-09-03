"""Stream-level validation sprint: does a return stream survive COSTS and OUT-OF-TIME?

The allocator only helps if the underlying streams carry a real, cost-adjusted,
out-of-sample edge. For each stream we compute gross/net return, turnover, Sharpe,
Sortino, max drawdown, a breakeven-cost (capacity) proxy, and train/OOS IC decay,
then FAIL promotion unless every gate holds. Pure stats, research-only.
"""
from __future__ import annotations

import numpy as np

ANN = 252
DEFAULT_CFG = {
    "cost_bps": 10.0,
    "min_net_sharpe": 0.5,
    "oos_ic_retain": 0.4,
}


def _sharpe(x):
    return float(x.mean() / (x.std() + 1e-12) * np.sqrt(ANN)) if len(x) > 5 else 0.0


def _sortino(x):
    d = x[x < 0]
    dd = d.std() if len(d) > 1 else x.std()
    return float(x.mean() / (dd + 1e-12) * np.sqrt(ANN)) if len(x) > 5 else 0.0


def _maxdd(ret):
    curve = np.cumprod(1 + np.asarray(ret))
    peak = np.maximum.accumulate(curve)
    return round(float(np.max((peak - curve) / peak)), 4) if len(curve) else 0.0


def stream_metrics(ret, turn, cost_bps=10.0):
    """ret, turn: daily gross-return and daily-turnover arrays (aligned)."""
    ret = np.asarray(ret, float); turn = np.asarray(turn, float)
    cost = cost_bps / 1e4
    net = ret - cost * turn
    gm, tm = float(ret.mean()), float(turn.mean())
    return {
        "n_days": int(len(ret)),
        "gross_mean_daily": round(gm, 6), "net_mean_daily": round(float(net.mean()), 6),
        "gross_cum_return": round(float(np.prod(1 + ret) - 1), 4),
        "net_cum_return": round(float(np.prod(1 + net) - 1), 4),
        "mean_turnover": round(tm, 4),
        "net_sharpe": round(_sharpe(net), 3), "net_sortino": round(_sortino(net), 3),
        "net_max_drawdown": _maxdd(net),
        # capacity proxy: the round-trip cost (bps) at which net expectancy hits 0
        "breakeven_cost_bps": round(gm / (tm + 1e-12) * 1e4, 2) if tm > 0 else float("inf"),
    }


def validate_stream(ret, turn, train_ic=None, oos_ic=None, oos_ret=None, oos_turn=None,
                    cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    m = stream_metrics(ret, turn, cfg["cost_bps"])
    if oos_ret is not None and len(oos_ret) > 5:
        m["oos"] = stream_metrics(oos_ret, oos_turn, cfg["cost_bps"])
    m["train_ic"] = round(float(train_ic), 5) if train_ic is not None else None
    m["oos_ic"] = round(float(oos_ic), 5) if oos_ic is not None else None

    reasons = []
    if m["net_mean_daily"] <= 0:
        reasons.append(f"cost_adjusted_expectancy<=0(net {m['net_mean_daily']:+.5f}/day)")
    if m["net_sharpe"] < cfg["min_net_sharpe"]:
        reasons.append(f"net_sharpe {m['net_sharpe']}<{cfg['min_net_sharpe']}")
    if m["breakeven_cost_bps"] < cfg["cost_bps"]:
        reasons.append(f"breakeven {m['breakeven_cost_bps']}bps<{cfg['cost_bps']} (turnover eats edge)")
    if "oos" in m and m["oos"]["net_mean_daily"] <= 0:
        reasons.append("oos_cost_adjusted_expectancy<=0")
    if train_ic is not None and oos_ic is not None:
        if np.sign(train_ic) != np.sign(oos_ic):
            reasons.append(f"ic_sign_flip(train {train_ic:+.3f},oos {oos_ic:+.3f})")
        elif abs(oos_ic) < cfg["oos_ic_retain"] * abs(train_ic):
            reasons.append(f"ic_decay_excessive(oos {oos_ic:+.3f})")
    m["promoted"] = not reasons
    m["reasons"] = reasons
    return m


def _demo():
    rng = np.random.default_rng(0)
    # a real edge that survives costs: positive mean, modest turnover
    good = validate_stream(rng.normal(0.001, 0.01, 200), np.full(200, 0.5),
                           train_ic=0.05, oos_ic=0.04)
    assert good["promoted"], good["reasons"]
    # over-trader: small gross edge, huge turnover -> net negative, fails
    bad = validate_stream(rng.normal(0.0005, 0.01, 200), np.full(200, 10.0))
    assert not bad["promoted"] and any("breakeven" in r or "expectancy" in r for r in bad["reasons"])
    # ic decay
    dec = validate_stream(rng.normal(0.001, 0.01, 200), np.full(200, 0.5),
                          train_ic=0.10, oos_ic=0.01)
    assert not dec["promoted"] and any("ic_decay" in r for r in dec["reasons"])
    print("stream_validation._demo OK  good:", good["net_sharpe"],
          "| bad reasons:", bad["reasons"][:1])


if __name__ == "__main__":
    _demo()
