#!/usr/bin/env python3
"""Independent validation of the only surviving families: linear_regression + mean_reversion.

No model, no bars -- a stability analysis of realized trade returns straight from
the dataset. Answers: do these two families have robust, regime-survivable
expectancy, or is it symbol-fitting / time-decay / a single-regime artifact?

  1 walk-forward by SYMBOL  GroupKFold on symbols -> test on unseen symbols
  2 walk-forward by TIME    chronological folds + explicit old vs new split
  3 trade-level             expectancy, PF, win, median MAE, median MFE
  4 regime decomposition    direction (up/down), trend-strength (trending/choppy),
                            volatility (high/low) -- does linear_regression only
                            work in trends? does mean_reversion only work in chop?

Research-only: reads the dataset; touches no broker/order/risk path.

    python scripts/surviving_family_validation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAMILIES = ["linear_regression", "mean_reversion"]


def _pf(r):
    up, dn = r[r > 0].sum(), -r[r < 0].sum()
    return round(float(up / dn), 3) if dn > 0 else float("inf")


def _maxdd(r, et):
    eq = np.cumsum(r[np.argsort(et)])
    return round(float(np.max(np.maximum.accumulate(eq) - eq)), 4) if len(eq) else 0.0


def fam_metrics(d):
    r = d["return_pct"].to_numpy(float)
    if len(r) == 0:
        return {"n": 0}
    mae = np.abs(d["mae_pct"].to_numpy(float)); mfe = d["mfe_pct"].to_numpy(float)
    return {"n": int(len(r)),
            "expectancy": round(float(r.mean()), 6),
            "median_return": round(float(np.median(r)), 6),
            "win_rate": round(float((r > 0).mean()), 3),
            "profit_factor": _pf(r),
            "max_drawdown": _maxdd(r, d["exit_time"].to_numpy()),
            "median_mae_pct": round(float(np.median(mae)), 5),
            "median_mfe_pct": round(float(np.median(mfe)), 5),
            "mfe_mae_ratio": round(float(np.median(mfe) / np.median(mae)), 3)
            if np.median(mae) > 0 else None}


def add_regimes(s):
    s = s.copy()
    s["vol_regime"] = np.where(s["realized_vol_20"] > s["realized_vol_20"].median(),
                               "high_vol", "low_vol")
    s["trend_strength"] = np.where(s["return_20_bars"].abs() > s["return_20_bars"].abs().median(),
                                   "trending", "choppy")
    s["direction"] = np.where(s["sma50_slope"] > 0, "up", "down")
    return s


def symbol_walkforward(s, k=4):
    from sklearn.model_selection import GroupKFold
    out = {f: [] for f in FAMILIES}
    gk = GroupKFold(n_splits=k)
    for _, te in gk.split(s, groups=s["symbol"]):
        test = s.iloc[te]
        for f in FAMILIES:
            m = fam_metrics(test[test["strategy_id"] == f])
            if m["n"]:
                out[f].append(m["expectancy"])
    res = {}
    for f in FAMILIES:
        e = np.array(out[f])
        res[f] = {"fold_expectancy": [round(x, 6) for x in e.tolist()],
                  "mean": round(float(e.mean()), 6), "std": round(float(e.std()), 6),
                  "positive_folds": int((e > 0).sum()), "folds": len(e)}
    return res


def time_walkforward(s, k=4):
    t = s["exit_time"]
    qs = np.linspace(0, 1, k + 1)
    bounds = [t.quantile(q) for q in qs]
    out = {f: [] for f in FAMILIES}; windows = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        up = (t <= hi) if i == k - 1 else (t < hi)
        fold = s[(t >= lo) & up]
        windows.append(f"{lo.date()}..{hi.date()}")
        for f in FAMILIES:
            m = fam_metrics(fold[fold["strategy_id"] == f])
            out[f].append(m.get("expectancy") if m["n"] else None)
    # explicit old (first 60%) vs new (last 40%)
    cut = t.quantile(0.6)
    oldnew = {}
    for f in FAMILIES:
        d = s[s["strategy_id"] == f]
        oldnew[f] = {"old": fam_metrics(d[d["exit_time"] < cut])["expectancy"],
                     "new": fam_metrics(d[d["exit_time"] >= cut])["expectancy"]}
    return {"windows": windows,
            "fold_expectancy": {f: out[f] for f in FAMILIES},
            "old_vs_new": oldnew, "cut_date": str(cut.date())}


def regime_decomp(s):
    out = {}
    for f in FAMILIES:
        d = s[s["strategy_id"] == f]
        out[f] = {}
        for axis in ["direction", "trend_strength", "vol_regime"]:
            out[f][axis] = {str(k): {"expectancy": fam_metrics(g)["expectancy"],
                                     "profit_factor": fam_metrics(g)["profit_factor"],
                                     "win_rate": fam_metrics(g)["win_rate"], "n": int(len(g))}
                            for k, g in d.groupby(axis)}
    return out


def _verdict(f, sym, tim, reg):
    sy = sym[f]; tn = tim["old_vs_new"][f]
    sym_robust = sy["positive_folds"] >= max(3, sy["folds"] - 1)
    time_robust = (tn["old"] or 0) > 0 and (tn["new"] or 0) > 0
    # regime dependence: any axis where one bucket > 0 and another <= 0
    regime_dep = []
    for axis, buckets in reg[f].items():
        es = [b["expectancy"] for b in buckets.values()]
        if es and max(es) > 0 and min(es) <= 0:
            regime_dep.append(axis)
    if sym_robust and time_robust and not regime_dep:
        v = "robust"
    elif not time_robust:
        v = "time-decayed"
    elif not sym_robust:
        v = "symbol-fit-risk"
    else:
        v = "regime-dependent"
    return {"verdict": v, "symbol_robust": sym_robust, "time_robust": time_robust,
            "regime_dependent_axes": regime_dep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    s = add_regimes(df[df["strategy_id"].isin(FAMILIES)].copy())
    print(f"surviving-family trades: {len(s):,} | " +
          " ".join(f"{f}={int((s.strategy_id==f).sum()):,}" for f in FAMILIES))

    overall = {f: fam_metrics(s[s["strategy_id"] == f]) for f in FAMILIES}
    sym = symbol_walkforward(s)
    tim = time_walkforward(s)
    reg = regime_decomp(s)
    verdicts = {f: _verdict(f, sym, tim, reg) for f in FAMILIES}

    rep = {"trades": int(len(s)), "trade_level": overall, "symbol_walkforward": sym,
           "time_walkforward": tim, "regime_decomposition": reg, "verdicts": verdicts}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in FAMILIES:
        for axis, buckets in reg[f].items():
            for b, m in buckets.items():
                rows.append({"family": f, "regime_axis": axis, "bucket": b, **m})
    pd.DataFrame(rows).to_csv(out / "surviving_family_regime.csv", index=False)
    (out / "surviving_family_validation_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _write_md(rep, out / "surviving_family_validation.md")
    _print(rep)
    print(f"\nartifacts -> {out}/ (report.json, regime.csv, validation.md)")


def _print(rep):
    print("\n=== trade-level (all data) ===")
    for f, m in rep["trade_level"].items():
        print(f"  {f:<18} n={m['n']:>6} exp {m['expectancy']*100:+.3f}% PF {m['profit_factor']} "
              f"win {m['win_rate']} MAE {m['median_mae_pct']*100:.2f}% MFE {m['median_mfe_pct']*100:.2f}% "
              f"DD {m['max_drawdown']:.3f}")
    print("\n=== symbol walk-forward (expectancy on held-out symbols) ===")
    for f, d in rep["symbol_walkforward"].items():
        print(f"  {f:<18} folds {d['fold_expectancy']} | +folds {d['positive_folds']}/{d['folds']}")
    print("\n=== time walk-forward (old vs new expectancy) ===")
    for f, d in rep["time_walkforward"]["old_vs_new"].items():
        o, n = d["old"], d["new"]
        print(f"  {f:<18} old {o*100:+.3f}% -> new {n*100:+.3f}%  "
              f"({'survives' if (o or 0)>0 and (n or 0)>0 else 'DECAYS'})")
    print("\n=== regime decomposition (expectancy by bucket) ===")
    for f, axes in rep["regime_decomposition"].items():
        print(f"  {f}:")
        for axis, buckets in axes.items():
            cells = "  ".join(f"{b}:{m['expectancy']*100:+.3f}%(PF{m['profit_factor']})"
                              for b, m in buckets.items())
            print(f"    {axis:<15} {cells}")
    print("\n=== verdicts ===")
    for f, v in rep["verdicts"].items():
        dep = (" regime-dep on " + ",".join(v["regime_dependent_axes"])) if v["regime_dependent_axes"] else ""
        print(f"  {f:<18} {v['verdict'].upper()}  (symbol_robust={v['symbol_robust']}, "
              f"time_robust={v['time_robust']}{dep})")


def _write_md(rep, path):
    lines = ["# Surviving-family validation: linear_regression + mean_reversion", "",
             "No model, no bars -- stability of realized trade returns.", "",
             "| family | n | expectancy | PF | win | med MAE | med MFE | maxDD | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in FAMILIES:
        m = rep["trade_level"][f]; v = rep["verdicts"][f]["verdict"]
        lines.append(f"| {f} | {m['n']} | {m['expectancy']*100:+.3f}% | {m['profit_factor']} | "
                     f"{m['win_rate']} | {m['median_mae_pct']*100:.2f}% | {m['median_mfe_pct']*100:.2f}% "
                     f"| {m['max_drawdown']:.3f} | **{v}** |")
    lines += ["", "## Old vs new (time survival)"]
    for f, d in rep["time_walkforward"]["old_vs_new"].items():
        lines.append(f"- {f}: old {d['old']*100:+.3f}% -> new {d['new']*100:+.3f}%")
    lines += ["", "## Regime decomposition (expectancy by bucket)"]
    for f, axes in rep["regime_decomposition"].items():
        lines.append(f"### {f}")
        for axis, buckets in axes.items():
            lines.append(f"- {axis}: " + ", ".join(
                f"{b} {m['expectancy']*100:+.3f}% (PF {m['profit_factor']}, n={m['n']})"
                for b, m in buckets.items()))
    lines += ["", "## Read", "These are the only families surviving walk-forward + repaint "
              "audit. Expectancy is small; this validates they are CLEAN, not that they are "
              "strong. A regime-dependent verdict means the allocator should condition on that "
              "axis. Research-only; not a paper-deployment decision."]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
