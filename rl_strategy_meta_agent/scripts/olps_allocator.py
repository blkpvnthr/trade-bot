#!/usr/bin/env python3
"""Online portfolio selection allocator layer over strategy-family "assets".

Alpha mining finds edges; this layer decides how much capital each strategy gets
over time. Each strategy family is treated as an asset/expert: input is a time-
indexed matrix of strategy returns, output is a time-indexed matrix of simplex
weights. Compares the OLPS taxonomy (benchmarks / Follow-the-Winner / Follow-the-
Loser / pattern-matching / meta-learning) on cumulative wealth, CAGR, Sharpe, max
drawdown, turnover -- gross AND net of transaction costs -- with a walk-forward
split and a per-regime breakdown.

NOTE: demonstrated on the per-strategy-family daily return streams available in
the dataset (the new-stack streams -- alpha_mined_ensemble, giveback_exit_model,
ORB_relvol -- plug into the same generic interface once produced). Research-only:
no broker / order / allocator-execution / live path.

    python scripts/olps_allocator.py --cost-bps 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import olps
from rl_meta.bar_coverage import cached_bars
from rl_meta.strategy_allowlist import ALLOWED_FAMILIES, BLOCKED_FAMILIES

ALLOCATORS = list(olps.ALGOS_RELATIVE) + ["olmar", "rmr", "aggregating"]


def strategy_returns(dataset, min_days=50, include_retired=False):
    """Per-strategy-family daily return streams. By default keeps only the VALIDATED
    families (allowlist) -- the retired/invalid ones (martingale, pin_bar, sniper)
    are artifact-dominated and would make the allocator chase a repaint."""
    df = pd.read_parquet(dataset, columns=["strategy_id", "exit_time", "return_pct"])
    df["date"] = pd.to_datetime(df["exit_time"], utc=True).dt.normalize()
    daily = (df.groupby(["date", "strategy_id"])["return_pct"].mean()
             .unstack().sort_index().fillna(0.0))
    fams = [c for c in daily.columns if (daily[c] != 0).sum() >= min_days]
    if not include_retired:
        fams = [c for c in fams if c in ALLOWED_FAMILIES]   # validated assets only
    return daily[fams]


def run_allocator(name, X, P, experts):
    if name in ("olmar", "rmr"):
        return {"olmar": olps.olmar, "rmr": olps.rmr}[name](P)
    if name == "aggregating":
        return olps.aggregating(X, experts)
    return olps.ALGOS_RELATIVE[name](X)


def _metrics(curve, turn):
    lr = np.diff(np.log(np.concatenate([[1.0], np.maximum(curve, 1e-9)])))
    sharpe = lr.mean() / (lr.std() + 1e-12) * np.sqrt(252)
    cagr = curve[-1] ** (252 / max(len(curve), 1)) - 1
    peak = np.maximum.accumulate(curve)
    return {"final_wealth": round(float(curve[-1]), 3), "cagr": round(float(cagr), 3),
            "sharpe": round(float(sharpe), 2),
            "max_drawdown": round(float(((peak - curve) / peak).max()), 3),
            "mean_turnover": round(float(turn.mean()), 3)}


def _spy_regime(dates, bars_dir):
    spy = cached_bars("SPY", "15Min", bars_dir)
    if spy.empty:
        return None
    s = spy.set_index(pd.to_datetime(spy["timestamp"], utc=True))["close"].resample("1D").last().dropna()
    mom = s.pct_change(20)
    reg = mom.reindex(dates, method="ffill")
    return np.where(reg > 0.01, "bull", np.where(reg < -0.01, "bear", "chop"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="transaction cost, basis points")
    ap.add_argument("--oos-frac", type=float, default=0.4)
    ap.add_argument("--include-retired", action="store_true",
                    help="include martingale/pin_bar/sniper (artifact-dominated; cautionary only)")
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()
    cost = args.cost_bps / 1e4

    R = strategy_returns(args.dataset, include_retired=args.include_retired)
    if R.shape[1] < 2:
        print(f"only {R.shape[1]} validated family stream(s); need >=2 for allocation. "
              "Add new-stack return streams or pass --include-retired for the cautionary baseline.")
        return
    dates = R.index
    Rm = R.to_numpy(float)
    X = 1.0 + Rm
    P = np.vstack([np.ones(X.shape[1]), np.cumprod(X, axis=0)])     # synthetic prices
    print(f"assets ({X.shape[1]}): {list(R.columns)} | {X.shape[0]} days "
          f"{dates.min().date()}..{dates.max().date()}")

    experts = [olps.eg(X), olps.pamr(X), olps.olmar(P), olps.corn(X), olps.crp(X)]
    cut = int(len(X) * (1 - args.oos_frac))
    regime = _spy_regime(dates, Path(args.bars_dir))

    rows = {}
    for name in ALLOCATORS:
        B = run_allocator(name, X, P, experts)
        gross, turn = olps.wealth(X, B)
        net, _ = olps.wealth(X, B, cost=cost)
        oos_net, oos_turn = olps.wealth(X[cut:], B[cut:], cost=cost)
        m = _metrics(net, turn)
        m["gross_final_wealth"] = round(float(gross[-1]), 3)
        m["net_final_wealth"] = round(float(net[-1]), 3)
        m["oos_net_wealth"] = round(float(oos_net[-1]), 3)
        m["oos_sharpe"] = _metrics(oos_net, oos_turn)["sharpe"]
        if regime is not None:
            port = (B * X).sum(1) - 1.0
            m["by_regime_mean_ret"] = {r: round(float(port[regime == r].mean()), 5)
                                       for r in ("bull", "bear", "chop") if (regime == r).any()}
        rows[name] = m

    rep = {"assets": list(R.columns), "days": int(len(X)), "cost_bps": args.cost_bps,
           "oos_frac": args.oos_frac, "allocators": rows}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).T.to_csv(out / "olps_allocator.csv")
    (out / "olps_allocator_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nartifacts -> {out}/ (olps_allocator_report.json, olps_allocator.csv)")


def _print(rep):
    print(f"\n=== OLPS allocators over {len(rep['assets'])} strategy assets "
          f"({rep['days']} days, cost {rep['cost_bps']}bps) ===")
    h = f"{'allocator':<14}{'grossW':>8}{'netW':>8}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'turn':>7}{'oosShrp':>8}"
    print(h); print("-" * len(h))
    for name, m in sorted(rep["allocators"].items(), key=lambda kv: -kv[1]["net_final_wealth"]):
        print(f"{name:<14}{m['gross_final_wealth']:>8.2f}{m['net_final_wealth']:>8.2f}"
              f"{m['cagr']*100:>6.1f}%{m['sharpe']:>7.2f}{m['max_drawdown']:>7.2f}"
              f"{m['mean_turnover']:>7.2f}{m['oos_sharpe']:>8.2f}")
    has_reg = any("by_regime_mean_ret" in m for m in rep["allocators"].values())
    if has_reg:
        print("\nbest allocator by regime (mean daily return):")
        for r in ("bull", "bear", "chop"):
            best = max(rep["allocators"].items(),
                       key=lambda kv: kv[1].get("by_regime_mean_ret", {}).get(r, -9))
            v = best[1].get("by_regime_mean_ret", {}).get(r)
            if v is not None:
                print(f"  {r:<6} -> {best[0]} ({v*100:+.3f}%/day)")
    base = rep["allocators"]["uniform_bah"]["net_final_wealth"]
    best = max(rep["allocators"].items(), key=lambda kv: kv[1]["net_final_wealth"])
    print(f"\nbuy&hold net {base:.2f} | best net '{best[0]}' {best[1]['net_final_wealth']:.2f}")


if __name__ == "__main__":
    main()
