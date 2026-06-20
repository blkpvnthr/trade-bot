#!/usr/bin/env python3
"""Robustness + threshold validation for the severe_giveback management overlay.

Confirms whether the risk-quality improvement is real and stable (not threshold
luck) before promoting anything. Scores each held-out trade's path once with the
engineered model, then runs in-memory:

  1 threshold sweep   full_exit / partial_exit / tighten_stop across thresholds
  2 walk-forward      train earlier dates -> test later dates (symbol holdout kept)
  3 symbol contrib    per-symbol delta vs baseline; is it 1-2 names or broad?
  4 recovery cost     avoided-giveback return vs missed-recovery return
  5 capital realism   fixed-fractional equity curve: portfolio return / DD / PF
  6 promotion gates   DD >=20% better, PF within 5%, total within 10%, >1 symbol

Outputs giveback_policy_threshold_sweep.{json,csv} + a markdown recommendation.
Research-only: no broker/order/risk path.

    python scripts/giveback_policy_validation.py --sample-trades 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.management_policies import (full_exit, partial_exit, tighten_stop,
                                         severe_giveback_happened, EPS)
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from giveback_policy_backtest import train_model, collect_paths, _load_refs

SWEEP = {
    "full_exit": ([0.55, 0.60, 0.65, 0.70, 0.75], lambda thr: lambda r, p: full_exit(r, p, thr)),
    "partial_exit": ([0.55, 0.60, 0.65, 0.70], lambda thr: lambda r, p: partial_exit(r, p, thr, 0.5)),
    "tighten_stop": ([0.50, 0.55, 0.60, 0.65], lambda thr: lambda r, p: tighten_stop(r, p, thr, 0.005)),
}


def _arrays(paths):
    r = [np.asarray(d["r"], float) for d in paths]
    p = [np.asarray(d["p"], float) for d in paths]
    base = np.array([x[-1] for x in r])
    severe = np.array([severe_giveback_happened(x) for x in r])
    et = np.array([d["exit_time"] for d in paths])
    sym = np.array([d["symbol"] for d in paths])
    return r, p, base, severe, et, sym


def _apply(r, p, fn):
    return np.array([fn(r[i], p[i]) for i in range(len(r))])


def _maxdd(x, et):
    eq = np.cumsum(x[np.argsort(et)])
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def _pf(x):
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    return float(up / dn) if dn > 0 else float("inf")


def _metrics(x, base, severe, et):
    return {"total_return": round(float(x.sum()), 4),
            "avg_trade_return": round(float(x.mean()), 5),
            "median_trade_return": round(float(np.median(x)), 5),
            "profit_factor": round(_pf(x), 3),
            "win_rate": round(float((x > 0).mean()), 3),
            "max_drawdown": round(_maxdd(x, et), 4),
            "avoided_severe_givebacks": int(((severe) & (x > base + EPS)).sum()),
            "missed_profitable_recoveries": int((base > x + EPS).sum())}


def _gates(m, b):
    dd_imp = (b["max_drawdown"] - m["max_drawdown"]) / b["max_drawdown"] if b["max_drawdown"] else 0
    return {"dd_improve_pct": round(dd_imp, 3),
            "dd_better_20pct": dd_imp >= 0.20,
            "pf_within_5pct": m["profit_factor"] >= b["profit_factor"] * 0.95,
            "total_within_10pct": m["total_return"] >= b["total_return"] * 0.90}


def _equity(x, et, frac=0.10, e0=100_000.0):
    """Fixed-fractional sequential book (ponytail: positions don't overlap here;
    max-open-positions not modeled -- trades are booked in exit order)."""
    eq = e0
    curve = [e0]
    for ret in x[np.argsort(et)]:
        eq *= (1 + frac * ret)
        curve.append(eq)
    curve = np.array(curve)
    peak = np.maximum.accumulate(curve)
    pnl = np.diff(curve)
    return {"total_return_pct": round(float(curve[-1] / e0 - 1) * 100, 2),
            "max_drawdown_pct": round(float(np.max((peak - curve) / peak)) * 100, 2),
            "profit_factor": round(_pf(pnl), 3),
            "final_equity": round(float(curve[-1]), 0)}


def _build(df, pre_cols, bars_dir, refs, train_mask, test_mask):
    model, feat = train_model(df[train_mask], pre_cols, bars_dir, refs)
    paths = collect_paths(df[test_mask], pre_cols, feat, model, bars_dir, refs)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--sample-trades", type=int, default=8000)
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()
    bars_dir = Path(args.bars_dir)

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    if args.sample_trades and len(df) > args.sample_trades:
        df = df.sample(args.sample_trades, random_state=0).reset_index(drop=True)

    start = df["entry_time"].min().date().isoformat()
    end = (df["exit_time"].max() + pd.Timedelta(days=1)).date().isoformat()
    refs = _load_refs(start, end, bars_dir)

    from sklearn.model_selection import GroupShuffleSplit
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0)
                  .split(df, groups=df["symbol"]))
    train_syms = set(df["symbol"].iloc[gi]); test_syms = set(df["symbol"].iloc[ge])
    in_train = df["symbol"].isin(train_syms); in_test = df["symbol"].isin(test_syms)
    print(f"symbol holdout: train {len(train_syms)} / test {len(test_syms)} symbols")

    print("training + scoring held-out paths (symbol holdout)...")
    paths = _build(df, pre_cols, bars_dir, refs, in_train, in_test)
    r, p, base, severe, et, sym = _arrays(paths)
    base_m = _metrics(base, base, severe, et)
    print(f"held-out trades {len(r)} | severe-gb rate {severe.mean():.2f} | "
          f"baseline total {base_m['total_return']:.3f} DD {base_m['max_drawdown']:.3f} "
          f"PF {base_m['profit_factor']}")

    # 1) threshold sweep ----------------------------------------------------
    rows = [{"policy": "baseline", "threshold": None, **base_m,
             "dd_improve_pct": 0.0, "promote": False}]
    for fam, (thrs, mk) in SWEEP.items():
        for thr in thrs:
            x = _apply(r, p, mk(thr))
            m = _metrics(x, base, severe, et)
            g = _gates(m, base_m)
            promote = (g["dd_better_20pct"] and g["pf_within_5pct"]
                       and g["total_within_10pct"]
                       and _n_symbols_improved(x, base, sym) >= 2)
            rows.append({"policy": fam, "threshold": thr, **m,
                         "dd_improve_pct": g["dd_improve_pct"], "promote": promote})

    sweep_df = pd.DataFrame(rows)
    rep = {"held_out_trades": len(r), "severe_giveback_rate": round(float(severe.mean()), 3),
           "baseline": base_m, "sweep": rows}

    # recommended = passing policy with best PF, then lowest DD; else best risk-adj
    passing = [x for x in rows if x["promote"]]
    pool = passing or [x for x in rows if x["policy"] != "baseline"]
    rec = sorted(pool, key=lambda d: (-d["profit_factor"], d["max_drawdown"]))[0]
    rec_fn = _fn_for(rec)
    rec_x = _apply(r, p, rec_fn)
    rep["recommended"] = {"policy": rec["policy"], "threshold": rec["threshold"],
                          "promoted": rec["promote"]}

    # 3) symbol contribution -------------------------------------------------
    rep["symbol_contribution"] = _symbol_contrib(rec_x, base, sym)
    # 4) recovery cost -------------------------------------------------------
    rep["recovery_cost"] = _recovery_cost(rec_x, base)
    # 5) capital realism -----------------------------------------------------
    rep["capital_realism"] = {"baseline": _equity(base, et),
                              "recommended": _equity(rec_x, et)}
    # 2) walk-forward --------------------------------------------------------
    rep["walk_forward"] = _walk_forward(df, pre_cols, bars_dir, refs,
                                        train_syms, test_syms, rec_fn)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out / "giveback_policy_threshold_sweep.csv", index=False)
    (out / "giveback_policy_threshold_sweep.json").write_text(json.dumps(rep, indent=2, default=str))
    _write_md(rep, out / "giveback_policy_recommendation.md")
    _print(rep)
    print(f"\nartifacts -> {out}/ (threshold_sweep.json/.csv, recommendation.md)")


def _fn_for(row):
    thr = row["threshold"]
    return {"full_exit": lambda r, p: full_exit(r, p, thr),
            "partial_exit": lambda r, p: partial_exit(r, p, thr, 0.5),
            "tighten_stop": lambda r, p: tighten_stop(r, p, thr, 0.005)}[row["policy"]]


def _n_symbols_improved(x, base, sym):
    return sum(1 for s in set(sym)
               if x[sym == s].sum() > base[sym == s].sum() + EPS)


def _symbol_contrib(x, base, sym):
    deltas = {s: round(float(x[sym == s].sum() - base[sym == s].sum()), 4)
              for s in sorted(set(sym))}
    deltas = dict(sorted(deltas.items(), key=lambda kv: kv[1]))
    pos = [d for d in deltas.values() if d > EPS]
    neg = [d for d in deltas.values() if d < -EPS]
    total = sum(deltas.values())
    top2 = sum(sorted(deltas.values(), reverse=True)[:2])
    return {"per_symbol_delta": deltas,
            "symbols_improved": len(pos), "symbols_worse": len(neg),
            "total_delta": round(total, 4),
            "top2_share_of_positive": round(top2 / sum(pos), 3) if pos else None,
            "broad_based": len(pos) >= 3 and (top2 / sum(pos) < 0.8 if pos else False)}


def _recovery_cost(x, base, e0=100_000.0, frac=0.10):
    d = x - base
    avoided = float(d[d > 0].sum())          # gains from exiting before givebacks
    missed = float(-d[d < 0].sum())          # upside given up on recoveries
    dollars = e0 * frac
    return {"avoided_giveback_return": round(avoided, 4),
            "missed_recovery_return": round(missed, 4),
            "net_return": round(avoided - missed, 4),
            "avoided_dollars": round(avoided * dollars, 0),
            "missed_dollars": round(missed * dollars, 0),
            "net_dollars": round((avoided - missed) * dollars, 0),
            "n_avoided": int((d > EPS).sum()), "n_missed": int((d < -EPS).sum())}


def _walk_forward(df, pre_cols, bars_dir, refs, train_syms, test_syms, rec_fn):
    cutoff = df["exit_time"].quantile(0.6)
    tr = df["symbol"].isin(train_syms) & (df["exit_time"] < cutoff)
    te = df["symbol"].isin(test_syms) & (df["exit_time"] >= cutoff)
    if tr.sum() < 200 or te.sum() < 50:
        return {"note": "insufficient data for the temporal split"}
    paths = _build(df.reset_index(drop=True), pre_cols, bars_dir, refs, tr.values, te.values)
    r, p, base, severe, et, _ = _arrays(paths)
    bm = _metrics(base, base, severe, et)
    rx = _apply(r, p, rec_fn)
    rm = _metrics(rx, base, severe, et)
    g = _gates(rm, bm)
    return {"cutoff": str(cutoff.date()), "test_trades": len(r),
            "baseline": {"total": bm["total_return"], "pf": bm["profit_factor"],
                         "max_drawdown": bm["max_drawdown"]},
            "recommended": {"total": rm["total_return"], "pf": rm["profit_factor"],
                            "max_drawdown": rm["max_drawdown"]},
            "dd_improve_pct": g["dd_improve_pct"],
            "improvement_persists": g["dd_better_20pct"] and g["pf_within_5pct"]}


def _print(rep):
    print(f"\n=== threshold sweep (held-out symbols, {rep['held_out_trades']} trades) ===")
    h = f"{'policy':<14}{'thr':>5}{'total':>8}{'PF':>7}{'win':>6}{'maxDD':>8}{'dd%':>7}{'avoid':>7}{'miss':>6}{'promo':>6}"
    print(h); print("-" * len(h))
    for x in rep["sweep"]:
        t = "" if x["threshold"] is None else f"{x['threshold']:.2f}"
        print(f"{x['policy']:<14}{t:>5}{x['total_return']:>8.3f}{x['profit_factor']:>7}"
              f"{x['win_rate']:>6.2f}{x['max_drawdown']:>8.3f}{x['dd_improve_pct']*100:>6.0f}%"
              f"{x['avoided_severe_givebacks']:>7}{x['missed_profitable_recoveries']:>6}"
              f"{'Y' if x['promote'] else '-':>6}")
    rec = rep["recommended"]
    print(f"\nrecommended: {rec['policy']} @ {rec['threshold']} "
          f"(promoted={rec['promoted']})")
    wf = rep["walk_forward"]
    if "improvement_persists" in wf:
        print(f"walk-forward (test>{wf['cutoff']}): DD {wf['baseline']['max_drawdown']:.3f}"
              f"->{wf['recommended']['max_drawdown']:.3f}  persists={wf['improvement_persists']}")
    sc = rep["symbol_contribution"]
    print(f"symbol contribution: {sc['symbols_improved']} improved / {sc['symbols_worse']} worse"
          f"  broad_based={sc['broad_based']}")
    rc = rep["recovery_cost"]
    print(f"recovery cost: avoided ${rc['avoided_dollars']:,.0f} vs missed "
          f"${rc['missed_dollars']:,.0f}  net ${rc['net_dollars']:,.0f}")
    cr = rep["capital_realism"]
    print(f"capital (10% fixed-frac): baseline {cr['baseline']['total_return_pct']}% "
          f"DD {cr['baseline']['max_drawdown_pct']}%  ->  recommended "
          f"{cr['recommended']['total_return_pct']}% DD {cr['recommended']['max_drawdown_pct']}%")


def _write_md(rep, path):
    rec = rep["recommended"]; wf = rep["walk_forward"]; sc = rep["symbol_contribution"]
    rc = rep["recovery_cost"]; cr = rep["capital_realism"]
    promo = [x for x in rep["sweep"] if x["promote"]]
    verdict = ("PROMOTE" if (promo and wf.get("improvement_persists"))
               else "DO NOT PROMOTE (yet)")
    lines = [
        "# Severe-giveback management overlay - validation",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"Held-out trades: **{rep['held_out_trades']}**  |  severe-giveback rate "
        f"**{rep['severe_giveback_rate']:.2f}**",
        "",
        f"Baseline: total {rep['baseline']['total_return']:.3f}, PF "
        f"{rep['baseline']['profit_factor']}, maxDD {rep['baseline']['max_drawdown']:.3f}",
        "",
        "## Recommended policy",
        f"**{rec['policy']} @ {rec['threshold']}** (promotion gates "
        f"{'PASSED' if rec['promoted'] else 'NOT fully passed'}).",
        "",
        "| check | result |",
        "|---|---|",
        f"| policies passing all gates | {len(promo)} |",
        f"| walk-forward DD improvement persists | "
        f"{wf.get('improvement_persists', 'n/a')} |",
        f"| symbols improved / worse | {sc['symbols_improved']} / {sc['symbols_worse']} |",
        f"| broad-based (not 1-2 names) | {sc['broad_based']} |",
        f"| recovery cost net | ${rc['net_dollars']:,.0f} "
        f"(avoided ${rc['avoided_dollars']:,.0f} - missed ${rc['missed_dollars']:,.0f}) |",
        f"| capital DD: baseline -> recommended | "
        f"{cr['baseline']['max_drawdown_pct']}% -> {cr['recommended']['max_drawdown_pct']}% |",
        f"| capital total: baseline -> recommended | "
        f"{cr['baseline']['total_return_pct']}% -> {cr['recommended']['total_return_pct']}% |",
        "",
        "## Read",
        "The overlay is a **risk-quality** overlay: it trades a small amount of total "
        "return for lower drawdown and steadier outcomes. Promote only if the gates "
        "above hold AND the walk-forward improvement persists. High giveback risk is "
        "probabilistic, so partial/tighten policies (which keep some upside) are "
        "preferred over full exit. Research-only; no live execution.",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
