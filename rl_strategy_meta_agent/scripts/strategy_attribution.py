#!/usr/bin/env python3
"""Strategy-family attribution + trade-selection study.

Two questions, one model:

  WHERE does the severe_giveback signal originate? Slice the held-out AUC of the
  engineered model by strategy family, symbol, and market regime, and read each
  family's feature importance. If the predictive power concentrates in a few
  families, the edge may be about WHICH trades to take, not how to manage them.

  Is the signal better as a TRADE-SELECTION FILTER than a management overlay?
  Simulate, on held-out symbols:
    1 trade all (baseline)
    2 trade only the bottom-25% giveback-risk entries (pre-entry risk model)
    3 trade only the top-expectancy strategy families (ranked on train)
    4 trade only family/regime combos with positive train expectancy

Research-only: no broker/order/risk/RL/sequence path.

    python scripts/strategy_attribution.py --sample-trades 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.exit_quality import (running_features, decision_targets,
                                  sample_decision_ks, RUN_FEATURES)
from rl_meta.market_context import CTX_NUMERIC
from rl_meta.management_policies import severe_giveback_happened
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from rl_meta.trade_quality import make_models, proba
from rl_meta.bar_coverage import cached_bars
from giveback_policy_backtest import _load_refs, _frames, _path_and_idx


def build(df, pre_cols, bars_dir, refs):
    feat = pre_cols + list(RUN_FEATURES) + list(CTX_NUMERIC)
    dec, trades = [], []
    cols = pre_cols + ["strategy_id", "entry_time", "exit_time", "entry_price", "direction"]
    for sym, g in df.groupby("symbol"):
        bars = cached_bars(sym, "15Min", bars_dir)
        if bars.empty:
            continue
        bars, ts_ns, ctx = _frames(bars, refs, sym)
        close = bars["close"].to_numpy(float)
        for t in g[cols].itertuples(index=False):
            lo, r = _path_and_idx(t, ts_ns, close)
            if r is None:
                continue
            pre = [getattr(t, c) for c in pre_cols]
            ent = ctx.iloc[lo]
            trades.append({**{c: getattr(t, c) for c in pre_cols},
                           "strategy": t.strategy_id, "symbol": sym,
                           "regime": ent["ctx_mkt_regime"],
                           "trade_severe": int(severe_giveback_happened(r)),
                           "trade_return": float(r[-1]), "exit_time": t.exit_time})
            for k in sample_decision_ks(len(r) - 1):
                run = running_features(r, k)
                crow = ctx.iloc[lo + k]
                row = dict(zip(feat, pre + [run[f] for f in RUN_FEATURES] +
                               [crow[c] for c in CTX_NUMERIC]))
                row.update({"strategy": t.strategy_id, "symbol": sym,
                            "regime": crow["ctx_mkt_regime"],
                            "severe_giveback": int(decision_targets(r, k)["severe_giveback"])})
                dec.append(row)
    return pd.DataFrame(dec), pd.DataFrame(trades), feat


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 40:
        return None
    return round(float(roc_auc_score(y, p)), 3)


def _pf(x):
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    return round(float(up / dn), 3) if dn > 0 else float("inf")


def _trade_metrics(ret, et):
    if len(ret) == 0:
        return {"trades": 0}
    order = np.argsort(et)
    eq = np.cumsum(ret[order])
    return {"trades": int(len(ret)), "total_return": round(float(ret.sum()), 4),
            "avg_return": round(float(ret.mean()), 5),
            "median_return": round(float(np.median(ret)), 5),
            "profit_factor": _pf(ret), "win_rate": round(float((ret > 0).mean()), 3),
            "max_drawdown": round(float(np.max(np.maximum.accumulate(eq) - eq)), 4)}


def attribution(dec, feat, train_syms, test_syms):
    tr = dec[dec["symbol"].isin(train_syms)]; te = dec[dec["symbol"].isin(test_syms)]
    model = make_models().get("xgboost") or make_models()["random_forest"]
    model.fit(tr[feat], tr["severe_giveback"])
    te = te.copy(); te["p"] = proba(model, te[feat])
    overall = _auc(te["severe_giveback"], te["p"])

    def by(col):
        out = {}
        for key, sub in te.groupby(col):
            out[str(key)] = {"n": int(len(sub)),
                             "severe_rate": round(float(sub["severe_giveback"].mean()), 3),
                             "auc": _auc(sub["severe_giveback"], sub["p"])}
        return dict(sorted(out.items(), key=lambda kv: -(kv[1]["auc"] or 0)))

    # feature importance per family (train rows of that family)
    fam_imp = {}
    for s, sub in tr.groupby("strategy"):
        if len(sub) < 100 or sub["severe_giveback"].nunique() < 2:
            continue
        m = make_models().get("xgboost") or make_models()["random_forest"]
        m.fit(sub[feat], sub["severe_giveback"])
        imp = getattr(m, "feature_importances_", None)
        if imp is not None:
            fam_imp[str(s)] = dict(sorted(zip(feat, imp.tolist()), key=lambda x: -x[1])[:8])
    return {"overall_auc": overall, "auc_by_strategy": by("strategy"),
            "auc_by_symbol": by("symbol"), "auc_by_regime": by("regime"),
            "feature_importance_by_strategy": fam_imp}


def selection(trades, pre_cols, train_syms, test_syms):
    tr = trades[trades["symbol"].isin(train_syms)].copy()
    te = trades[trades["symbol"].isin(test_syms)].copy()
    et = te["exit_time"].to_numpy(); ret = te["trade_return"].to_numpy()
    sims = {"1_trade_all": _trade_metrics(ret, et)}

    # 2) pre-entry giveback-risk model -> keep the lowest-risk 25%
    rm = make_models().get("xgboost") or make_models()["random_forest"]
    rm.fit(tr[pre_cols], tr["trade_severe"])
    thr = np.quantile(proba(rm, tr[pre_cols]), 0.25)          # train-derived cut (no lookahead)
    te["risk"] = proba(rm, te[pre_cols])
    keep = te["risk"] <= thr
    sims["2_bottom25_giveback_risk"] = {**_trade_metrics(ret[keep.values], et[keep.values]),
                                        "kept_pct": round(float(keep.mean()), 3)}

    # 3) top-expectancy strategy families (ranked on train mean return)
    fam_exp = tr.groupby("strategy")["trade_return"].mean()
    top_fams = list(fam_exp[fam_exp > 0].index)
    m3 = te["strategy"].isin(top_fams)
    sims["3_top_strategy_families"] = {**_trade_metrics(ret[m3.values], et[m3.values]),
                                       "families": top_fams,
                                       "family_train_expectancy": {k: round(float(v), 5)
                                                                   for k, v in fam_exp.items()}}

    # 4) family/regime combos with positive train expectancy
    combo = tr.groupby(["strategy", "regime"])["trade_return"].mean()
    good = set(combo[combo > 0].index)
    m4 = te.apply(lambda x: (x["strategy"], x["regime"]) in good, axis=1)
    sims["4_positive_family_regime"] = {**_trade_metrics(ret[m4.values], et[m4.values]),
                                        "n_combos_kept": len(good)}
    return sims


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
        df = df.sample(args.sample_trades, random_state=0)
    start = df["entry_time"].min().date().isoformat()
    end = (df["exit_time"].max() + pd.Timedelta(days=1)).date().isoformat()
    refs = _load_refs(start, end, bars_dir)

    from sklearn.model_selection import GroupShuffleSplit
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(df, groups=df["symbol"]))
    train_syms = set(df["symbol"].iloc[gi]); test_syms = set(df["symbol"].iloc[ge])

    print("building decision + trade rows...")
    dec, trades, feat = build(df, pre_cols, bars_dir, refs)
    print(f"decision rows {len(dec):,} | trades {len(trades):,}")

    attr = attribution(dec, feat, train_syms, test_syms)
    sims = selection(trades, pre_cols, train_syms, test_syms)
    rep = {"train_symbols": len(train_syms), "test_symbols": len(test_syms),
           "decision_rows": int(len(dec)), "trades": int(len(trades)),
           "trade_count_by_strategy": {k: int(v) for k, v in
                                       trades["strategy"].value_counts().items()},
           "attribution": attr, "selection": sims}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "strategy_attribution_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nreport -> {out / 'strategy_attribution_report.json'}")


def _print(rep):
    a = rep["attribution"]
    print(f"\n=== giveback AUC attribution (overall held-out {a['overall_auc']}) ===")
    print("by strategy family:")
    for s, d in a["auc_by_strategy"].items():
        print(f"  {s:<22} AUC {d['auc']}  (n={d['n']:,}, severe {d['severe_rate']})")
    print("by market regime:")
    for s, d in a["auc_by_regime"].items():
        print(f"  {s:<10} AUC {d['auc']}  (n={d['n']:,})")
    print("\n=== trade SELECTION on held-out symbols ===")
    h = f"{'selection':<28}{'trades':>7}{'total':>8}{'avg':>8}{'PF':>7}{'win':>6}{'maxDD':>8}"
    print(h); print("-" * len(h))
    base = rep["selection"]["1_trade_all"]
    for name, m in rep["selection"].items():
        if not m.get("trades"):
            print(f"{name:<28}{'(empty)':>7}"); continue
        print(f"{name:<28}{m['trades']:>7}{m['total_return']:>8.3f}{m['avg_return']*100:>7.2f}%"
              f"{m['profit_factor']:>7}{m['win_rate']:>6.2f}{m['max_drawdown']:>8.3f}")
    print("\n=== Read ===")
    best = max((m for m in rep["selection"].values() if m.get("trades")),
               key=lambda m: m["avg_return"])
    bn = [k for k, v in rep["selection"].items() if v is best][0]
    print(f"baseline avg/trade {base['avg_return']*100:.2f}% -> best selection '{bn}' "
          f"{best['avg_return']*100:.2f}%/trade "
          f"({'+' if best['avg_return'] >= base['avg_return'] else ''}"
          f"{(best['avg_return']-base['avg_return'])*100:.2f} pp); "
          "selection improves per-trade expectancy" if best["avg_return"] > base["avg_return"]
          else "selection does NOT beat trading everything")


if __name__ == "__main__":
    main()
