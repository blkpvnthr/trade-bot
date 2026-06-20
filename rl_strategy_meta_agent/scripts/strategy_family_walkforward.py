#!/usr/bin/env python3
"""Walk-forward validation of strategy-family selection (a whitelist, not an ML model).

Question: is "drop the bad strategy families" durable over TIME, or just an
in-sample family ranking? Purely temporal walk-forward: for each chronological
fold, rank families ONLY on trades that closed before the test window, then trade
only the whitelisted families on the next unseen period. No future leakage.

Policies:
  1 trade_all                      every trade
  2 exclude_global_bad             drop martingale + pin_bar (fixed, a-priori)
  3 rolling_family_whitelist       per fold: keep families with positive return,
                                   PF > 1.2, and enough trades (ranked on the past)
  4 rolling_family_regime_whitelist same, on family x regime combos
  (diagnostic) exclude_martingale_only

Aggregate out-of-sample equity (selected test trades across all folds, time-
ordered) is the headline; per-fold pass-count is the robustness check.

Research-only: no broker/order/risk/RL/model-architecture path.

    python scripts/strategy_family_walkforward.py --sample-trades 8000 --folds 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_meta.trade_features import PRE_ENTRY_FEATURES
from strategy_attribution import build, _trade_metrics, _pf
from giveback_policy_backtest import _load_refs

GLOBAL_BAD = {"martingale", "pin_bar_magic_v1"}
PF_MIN = 1.2
MIN_FAM = 20
MIN_COMBO = 15


def load_trades(args, bars_dir):
    cache = Path(args.out) / "_wf_trades.parquet"
    if cache.exists() and not args.rebuild:
        return pd.read_parquet(cache)
    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    if args.sample_trades and len(df) > args.sample_trades:
        df = df.sample(args.sample_trades, random_state=0)
    start = df["entry_time"].min().date().isoformat()
    end = (df["exit_time"].max() + pd.Timedelta(days=1)).date().isoformat()
    refs = _load_refs(start, end, bars_dir)
    print("rebuilding per-trade table (strategy_attribution.build)...")
    _, trades, _ = build(df, pre_cols, bars_dir, refs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(cache, index=False)
    return trades


def whitelist_families(train):
    wl = []
    for fam, sub in train.groupby("strategy"):
        ret = sub["trade_return"].to_numpy()
        if len(sub) >= MIN_FAM and ret.sum() > 0 and _pf(ret) > PF_MIN:
            wl.append(str(fam))
    return wl


def whitelist_combos(train):
    good = set()
    for key, sub in train.groupby(["strategy", "regime"]):
        ret = sub["trade_return"].to_numpy()
        if len(sub) >= MIN_COMBO and ret.sum() > 0 and _pf(ret) > PF_MIN:
            good.add(key)
    return good


def select(policy, train, test):
    """Return (selected_test_df, label_of_whats_kept)."""
    if policy == "trade_all":
        return test, "all"
    if policy == "exclude_global_bad":
        return test[~test["strategy"].isin(GLOBAL_BAD)], "all - {martingale,pin_bar}"
    if policy == "exclude_martingale_only":
        return test[test["strategy"] != "martingale"], "all - martingale"
    if policy == "rolling_family_whitelist":
        wl = whitelist_families(train)
        return test[test["strategy"].isin(wl)], ",".join(sorted(wl)) or "(none)"
    if policy == "rolling_family_regime_whitelist":
        good = whitelist_combos(train)
        mask = test.apply(lambda x: (x["strategy"], x["regime"]) in good, axis=1)
        return test[mask], f"{len(good)} combos"
    raise ValueError(policy)


POLICIES = ["trade_all", "exclude_global_bad", "rolling_family_whitelist",
            "rolling_family_regime_whitelist", "exclude_martingale_only"]


def folds(trades, n, initial_frac):
    t = trades["exit_time"]
    qs = np.linspace(initial_frac, 1.0, n + 1)
    bounds = [t.quantile(q) for q in qs]
    out = []
    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        train = trades[trades["exit_time"] < lo]
        upper = (trades["exit_time"] <= hi) if i == n - 1 else (trades["exit_time"] < hi)
        test = trades[(trades["exit_time"] >= lo) & upper]
        out.append((lo, hi, train, test))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--sample-trades", type=int, default=8000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--initial-frac", type=float, default=0.4)
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    trades = load_trades(args, Path(args.bars_dir)).sort_values("exit_time").reset_index(drop=True)
    print(f"trades {len(trades):,} | {trades['exit_time'].min().date()} .. "
          f"{trades['exit_time'].max().date()} | families {trades['strategy'].nunique()}")

    fold_list = folds(trades, args.folds, args.initial_frac)
    rows, oos = [], {p: [] for p in POLICIES}      # oos[p] = list of selected test frames
    whitelists = {p: [] for p in POLICIES}
    for fi, (lo, hi, train, test) in enumerate(fold_list, 1):
        for p in POLICIES:
            sel, label = select(p, train, test)
            whitelists[p].append(label)
            oos[p].append(sel)
            m = _trade_metrics(sel["trade_return"].to_numpy(), sel["exit_time"].to_numpy())
            rows.append({"fold": fi, "policy": p, "kept": label,
                         "train_end": str(lo.date()), "test_end": str(hi.date()), **m})

    # aggregate out-of-sample (selected test trades across all folds, time-ordered)
    agg = {}
    for p in POLICIES:
        d = pd.concat(oos[p]).sort_values("exit_time")
        agg[p] = _trade_metrics(d["trade_return"].to_numpy(), d["exit_time"].to_numpy())
    base = agg["trade_all"]

    def gates(m):
        ddimp = (base["max_drawdown"] - m["max_drawdown"]) / base["max_drawdown"] if base["max_drawdown"] else 0
        return {"dd_improve_pct": round(ddimp, 3), "dd_better_20pct": ddimp >= 0.20,
                "pf_within_5pct": m["profit_factor"] >= base["profit_factor"] * 0.95,
                "total_within_10pct": m["total_return"] >= base["total_return"] * 0.90}

    # per-fold pass count vs trade_all on that fold
    base_fold = {r["fold"]: r for r in rows if r["policy"] == "trade_all"}
    folds_passed = {}
    for p in POLICIES:
        c = 0
        for r in [r for r in rows if r["policy"] == p]:
            b = base_fold[r["fold"]]
            if not r["trades"] or not b["max_drawdown"]:
                continue
            ddi = (b["max_drawdown"] - r["max_drawdown"]) / b["max_drawdown"]
            if ddi >= 0.20 and r["profit_factor"] >= b["profit_factor"] * 0.95 \
                    and r["total_return"] >= b["total_return"] * 0.90:
                c += 1
        folds_passed[p] = c

    policies_rep = {}
    for p in POLICIES:
        g = gates(agg[p])
        stable = len(set(whitelists[p])) <= 2          # whitelist barely changes across folds
        promote = (p != "trade_all" and g["dd_better_20pct"] and g["pf_within_5pct"]
                   and g["total_within_10pct"] and folds_passed[p] >= 3 and stable)
        policies_rep[p] = {"aggregate": agg[p], **g, "folds_passed": folds_passed[p],
                           "whitelist_per_fold": whitelists[p], "whitelist_stable": stable,
                           "promote": promote}

    # contributions on the trade_all out-of-sample set
    allset = pd.concat(oos["trade_all"]).sort_values("exit_time")
    fam_contrib = {str(k): round(float(v), 4) for k, v in
                   allset.groupby("strategy")["trade_return"].sum().sort_values().items()}
    rec = _recommend(policies_rep)
    sym_contrib = _symbol_contrib(allset, oos[rec]) if rec else {}

    rep = {"trades": int(len(trades)), "folds": args.folds, "families": int(trades["strategy"].nunique()),
           "baseline_oos": base, "policies": policies_rep,
           "recommended": rec, "family_contribution_trade_all_oos": fam_contrib,
           "symbol_contribution_recommended": sym_contrib}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "strategy_family_walkforward.csv", index=False)
    (out / "strategy_family_walkforward_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _write_md(rep, out / "strategy_family_walkforward_verdict.md")
    _print(rep)
    print(f"\nartifacts -> {out}/ (report.json, .csv, verdict.md)")


def _recommend(pr):
    # prefer the SIMPLEST passing policy
    for p in ["exclude_global_bad", "exclude_martingale_only",
              "rolling_family_whitelist", "rolling_family_regime_whitelist"]:
        if pr[p]["promote"]:
            return p
    return None


def _symbol_contrib(allset, sel_frames):
    sel = pd.concat(sel_frames)
    base = allset.groupby("symbol")["trade_return"].sum()
    keep = sel.groupby("symbol")["trade_return"].sum()
    d = (keep - base).reindex(base.index).fillna(-base).sort_values()
    return {str(k): round(float(v), 4) for k, v in d.items() if abs(v) > 1e-6}


def _print(rep):
    print(f"\n=== walk-forward strategy-family selection ({rep['folds']} folds) ===")
    b = rep["baseline_oos"]
    print(f"baseline OOS: trades {b['trades']} total {b['total_return']:.3f} "
          f"PF {b['profit_factor']} win {b['win_rate']} maxDD {b['max_drawdown']:.3f}\n")
    h = f"{'policy':<32}{'trades':>7}{'total':>8}{'PF':>7}{'win':>6}{'maxDD':>8}{'dd%':>6}{'folds':>7}{'promo':>6}"
    print(h); print("-" * len(h))
    for p, d in rep["policies"].items():
        a = d["aggregate"]
        print(f"{p:<32}{a['trades']:>7}{a['total_return']:>8.3f}{a['profit_factor']:>7}"
              f"{a['win_rate']:>6.2f}{a['max_drawdown']:>8.3f}{d['dd_improve_pct']*100:>5.0f}%"
              f"{d['folds_passed']:>5}/{rep['folds']}{'  Y' if d['promote'] else '  -':>6}")
    print(f"\nfamily contribution (trade_all OOS total return by family):")
    for k, v in rep["family_contribution_trade_all_oos"].items():
        print(f"  {k:<22}{v:>+8.3f}")
    rec = rep["recommended"]
    print(f"\nRECOMMENDED: {rec if rec else 'NONE -> do not promote'}")
    if rec:
        print(f"  whitelist per fold: {rep['policies'][rec]['whitelist_per_fold']}")


def _write_md(rep, path):
    rec = rep["recommended"]
    verdict = (f"PROMOTE (paper-only): {rec}" if rec else "DO NOT PROMOTE (yet)")
    b = rep["baseline_oos"]
    lines = [f"# Strategy-family selection - walk-forward verdict", "",
             f"## Verdict: **{verdict}**", "",
             f"Purely temporal walk-forward, {rep['folds']} folds, "
             f"whitelist ranked only on past closed trades (no leakage).", "",
             f"Baseline (trade_all) OOS: total {b['total_return']:.3f}, "
             f"PF {b['profit_factor']}, maxDD {b['max_drawdown']:.3f}, win {b['win_rate']}.", "",
             "| policy | total | PF | maxDD | dd% | folds | promote |",
             "|---|---|---|---|---|---|---|"]
    for p, d in rep["policies"].items():
        a = d["aggregate"]
        lines.append(f"| {p} | {a['total_return']:.3f} | {a['profit_factor']} | "
                     f"{a['max_drawdown']:.3f} | {d['dd_improve_pct']*100:.0f}% | "
                     f"{d['folds_passed']}/{rep['folds']} | {'YES' if d['promote'] else 'no'} |")
    lines += ["", "## Family contribution (trade_all OOS total by family)",
              "| family | OOS total return |", "|---|---|"]
    for k, v in rep["family_contribution_trade_all_oos"].items():
        lines.append(f"| {k} | {v:+.3f} |")
    if rec:
        lines += ["", "## Recommended paper-only action",
                  f"`{rec}` -- whitelist per fold: {rep['policies'][rec]['whitelist_per_fold']}.",
                  "If this is `exclude_global_bad`, the promotion is trivially simple and safe: "
                  "**disable martingale and pin_bar; trade the validated families only. No ML "
                  "overlay.** Confirm how much of the gain is just dropping the structurally bad "
                  "families (see family contribution above)."]
    else:
        lines += ["", "## Read", "No policy cleared the gates across >=3 folds; family "
                  "selection is not durable enough to promote yet."]
    lines += ["", "Research-only; no live execution."]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
