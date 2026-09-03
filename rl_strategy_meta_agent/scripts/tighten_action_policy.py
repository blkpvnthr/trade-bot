#!/usr/bin/env python3
"""Direct-action policy: tighten the stop when p_tighten_stop_better is high.

Trains a model DIRECTLY on the action label tighten_stop_better (AUC ~0.738 in
the action-label experiment), NOT on p_severe_giveback. Then simulates a single
policy family -- tighten the stop once p_tighten_stop_better >= threshold -- and
runs it through the full validation gauntlet (held-out symbols, threshold sweep,
walk-forward, symbol contribution, recovery cost, fixed-fractional equity,
promotion gates).

If this passes, the system finally has a robust, paper-only management overlay.
If it fails, the conclusion strengthens: the system predicts deterioration but
cannot yet convert it into a durable management policy.

Research-only: no broker/order/risk/RL/sequence path.

    python scripts/tighten_action_policy.py --sample-trades 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.management_policies import tighten_stop
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from giveback_policy_backtest import train_model, collect_paths, _load_refs
from giveback_policy_validation import (_arrays, _apply, _metrics, _gates, _equity,
                                        _recovery_cost, _symbol_contrib,
                                        _n_symbols_improved)

TARGET = "tighten_stop_better"
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
TRAIL = 0.005


def _policy(thr):
    return lambda r, p: tighten_stop(r, p, thr, TRAIL)


def build_paths(df, pre_cols, bars_dir, refs, train_mask, test_mask):
    model, feat = train_model(df[train_mask], pre_cols, bars_dir, refs, target=TARGET)
    return collect_paths(df[test_mask], pre_cols, feat, model, bars_dir, refs)


def _walk_forward(df, pre_cols, bars_dir, refs, in_train, in_test, rec_fn):
    cutoff = df["exit_time"].quantile(0.6)
    tr = in_train & (df["exit_time"] < cutoff)
    te = in_test & (df["exit_time"] >= cutoff)
    if tr.sum() < 200 or te.sum() < 50:
        return {"note": "insufficient data for temporal split"}
    paths = build_paths(df.reset_index(drop=True), pre_cols, bars_dir, refs,
                        tr.values, te.values)
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
            "improvement_persists": g["dd_better_20pct"] and g["pf_within_5pct"]
            and g["total_within_10pct"]}


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

    print(f"training direct-action model on '{TARGET}' + scoring held-out paths...")
    paths = build_paths(df, pre_cols, bars_dir, refs, in_train.values, in_test.values)
    r, p, base, severe, et, sym = _arrays(paths)
    base_m = _metrics(base, base, severe, et)
    print(f"held-out trades {len(r)} | baseline total {base_m['total_return']:.3f} "
          f"DD {base_m['max_drawdown']:.3f} PF {base_m['profit_factor']}")

    # threshold sweep + per-trade returns ----------------------------------
    rows = [{"policy": "baseline", "threshold": None, **base_m,
             "dd_improve_pct": 0.0, "promote": False}]
    trades = pd.DataFrame({"symbol": sym, "exit_time": et, "baseline": base})
    for thr in THRESHOLDS:
        x = _apply(r, p, _policy(thr))
        trades[f"tighten_p{int(thr*100)}"] = x
        m = _metrics(x, base, severe, et)
        g = _gates(m, base_m)
        promote = (g["dd_better_20pct"] and g["pf_within_5pct"] and g["total_within_10pct"]
                   and _n_symbols_improved(x, base, sym) >= 2)
        rows.append({"policy": "tighten_stop", "threshold": thr, **m,
                     "dd_improve_pct": g["dd_improve_pct"], "promote": promote})

    passing = [x for x in rows if x["promote"]]
    pool = passing or [x for x in rows if x["threshold"] is not None]
    rec = sorted(pool, key=lambda d: (-d["profit_factor"], d["max_drawdown"]))[0]
    rec_fn = _policy(rec["threshold"])
    rec_x = _apply(r, p, rec_fn)

    wf = _walk_forward(df, pre_cols, bars_dir, refs, in_train, in_test, rec_fn)
    rep = {"target": TARGET, "held_out_trades": len(r),
           "severe_giveback_rate": round(float(severe.mean()), 3),
           "baseline": base_m, "sweep": rows,
           "recommended": {"policy": "tighten_stop", "threshold": rec["threshold"],
                           "promoted": rec["promote"]},
           "symbol_contribution": _symbol_contrib(rec_x, base, sym),
           "recovery_cost": _recovery_cost(rec_x, base),
           "capital_realism": {"baseline": _equity(base, et),
                               "recommended": _equity(rec_x, et)},
           "walk_forward": wf,
           "promoted": bool(rec["promote"] and wf.get("improvement_persists"))}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tighten_action_policy_sweep.csv", index=False)
    trades.to_csv(out / "tighten_action_policy_trades.csv", index=False)
    (out / "tighten_action_policy_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _write_md(rep, out / "tighten_action_policy_verdict.md")
    _print(rep)
    print(f"\nartifacts -> {out}/ (report.json, sweep.csv, trades.csv, verdict.md)")


def _print(rep):
    print(f"\n=== tighten-stop direct-action sweep (held-out, {rep['held_out_trades']} trades) ===")
    h = f"{'thr':>6}{'total':>8}{'PF':>7}{'win':>6}{'maxDD':>8}{'dd%':>7}{'avoid':>7}{'miss':>6}{'promo':>6}"
    print(h); print("-" * len(h))
    for x in rep["sweep"]:
        t = "base" if x["threshold"] is None else f"{x['threshold']:.2f}"
        print(f"{t:>6}{x['total_return']:>8.3f}{x['profit_factor']:>7}{x['win_rate']:>6.2f}"
              f"{x['max_drawdown']:>8.3f}{x['dd_improve_pct']*100:>6.0f}%"
              f"{x['avoided_severe_givebacks']:>7}{x['missed_profitable_recoveries']:>6}"
              f"{'Y' if x['promote'] else '-':>6}")
    rec = rep["recommended"]; wf = rep["walk_forward"]; sc = rep["symbol_contribution"]
    rc = rep["recovery_cost"]; cr = rep["capital_realism"]
    print(f"\nrecommended tighten_stop @ {rec['threshold']} | PROMOTED={rep['promoted']}")
    if "improvement_persists" in wf:
        print(f"walk-forward (test>{wf['cutoff']}): DD {wf['baseline']['max_drawdown']:.3f}"
              f"->{wf['recommended']['max_drawdown']:.3f} persists={wf['improvement_persists']}")
    print(f"symbols improved/worse: {sc['symbols_improved']}/{sc['symbols_worse']} "
          f"broad_based={sc['broad_based']}")
    print(f"recovery: avoided ${rc['avoided_dollars']:,.0f} vs missed ${rc['missed_dollars']:,.0f}"
          f" net ${rc['net_dollars']:,.0f}")
    print(f"capital (10% fixed-frac): baseline {cr['baseline']['total_return_pct']}% "
          f"DD {cr['baseline']['max_drawdown_pct']}% -> recommended "
          f"{cr['recommended']['total_return_pct']}% DD {cr['recommended']['max_drawdown_pct']}%")


def _write_md(rep, path):
    rec = rep["recommended"]; wf = rep["walk_forward"]; sc = rep["symbol_contribution"]
    rc = rep["recovery_cost"]; cr = rep["capital_realism"]; b = rep["baseline"]
    verdict = "PROMOTE (paper-only overlay)" if rep["promoted"] else "DO NOT PROMOTE (yet)"
    lines = [
        "# Direct-action tighten-stop overlay - verdict",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"Model trained DIRECTLY on `{rep['target']}` (not p_severe_giveback). "
        f"Held-out trades: **{rep['held_out_trades']}**.",
        "",
        f"Baseline: total {b['total_return']:.3f}, PF {b['profit_factor']}, "
        f"maxDD {b['max_drawdown']:.3f}",
        f"Recommended: tighten_stop @ {rec['threshold']} (gates passed: {rec['promoted']})",
        "",
        "| check | result |",
        "|---|---|",
        f"| any threshold passes all gates | {any(x['promote'] for x in rep['sweep'])} |",
        f"| walk-forward improvement persists | {wf.get('improvement_persists', 'n/a')} |",
        f"| symbols improved / worse | {sc['symbols_improved']} / {sc['symbols_worse']} |",
        f"| broad-based (not 1-2 names) | {sc['broad_based']} |",
        f"| recovery net | ${rc['net_dollars']:,.0f} "
        f"(avoided ${rc['avoided_dollars']:,.0f} - missed ${rc['missed_dollars']:,.0f}) |",
        f"| capital DD baseline -> recommended | "
        f"{cr['baseline']['max_drawdown_pct']}% -> {cr['recommended']['max_drawdown_pct']}% |",
        f"| capital total baseline -> recommended | "
        f"{cr['baseline']['total_return_pct']}% -> {cr['recommended']['total_return_pct']}% |",
        "",
        "## Read",
        "Promotion requires maxDD >=20% better, total within 10%, PF within 5%, "
        "broad multi-symbol improvement, AND persistence in walk-forward. A pass means "
        "a paper-only overlay candidate; a fail strengthens the conclusion that the "
        "system predicts deterioration but cannot yet convert it into a durable policy. "
        "Research-only; no live execution.",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
