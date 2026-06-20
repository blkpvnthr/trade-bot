#!/usr/bin/env python3
"""AlphaEvolve-style alpha mining for intraday severe-giveback prediction.

Implements the paper's pipeline on the formulaic-alpha class:
  1 alpha database (rl_meta.alpha_db)
  2 grammar / operators (rl_meta.alphas)
  3 generate thousands of candidate alphas
  4 IC fitness vs FUTURE labels -- primary target: per-bar severe_giveback
  5 weak-correlation pruning (feature redundancy, since these feed an ensemble)
  6 regularized evolution (tournament select -> mutate -> add -> evict oldest)
  7 mine directly against P(giveback), the project's strongest signal
  8 LightGBM ensemble on the top independent alphas vs raw inputs (honest check)

Leak-free: alphas use only backward/contemporaneous data; only labels look
forward. The ensemble uses a group-by-symbol split. Research-only: no broker,
allocator, execution, or live-trading path is touched.

    python scripts/alpha_mining.py --max-symbols 25 --months 8 --evo-steps 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.alphas import (BASE_INPUTS, random_alpha, mutate, eval_alpha,
                            formula_str, alpha_id, category, spearman_ic, ic_series)
from rl_meta import alpha_db, alphas101, multiple_testing as mt, alpha_registry
from rl_meta.sequences import candle_feature_frame
from rl_meta.market_context import build_context_frame, SECTOR_ETF
from rl_meta.bar_coverage import cached_bars
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from giveback_policy_backtest import _load_refs

LABELS = ["fut_ret_1", "fut_ret_4", "fut_mfe", "fut_giveback", "severe_giveback"]
H = 8                       # forward horizon (bars) for the giveback label
TARGET = "severe_giveback"


def build_panel(symbols, bars_dir, refs, months):
    frames = []
    for sym in symbols:
        bars = cached_bars(sym, "15Min", bars_dir)
        if bars.empty:
            continue
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        if months:
            cut = bars["timestamp"].max() - pd.Timedelta(days=30 * months)
            bars = bars[bars["timestamp"] >= cut].reset_index(drop=True)
        if len(bars) < 300:
            continue
        ctx = build_context_frame(bars, refs.get("SPY"), refs.get("QQQ"),
                                  refs.get(SECTOR_ETF.get(sym)))
        f = candle_feature_frame(bars, ctx)
        f["volume"] = bars["volume"].to_numpy(float)
        f["spy_trend"] = ctx["ctx_spy_trend"].values
        f["qqq_trend"] = ctx["ctx_qqq_trend"].values
        f["mkt_regime"] = ctx["ctx_mkt_regime"].map({"bull": 1, "chop": 0, "bear": -1}).fillna(0).values
        f["vwap"] = (bars["close"].to_numpy(float) / (1 + ctx["ctx_vwap_dist"].fillna(0).values))
        c = bars["close"].to_numpy(float)
        fr = np.vstack([np.concatenate([c[h:] / c[:-h] - 1, [np.nan] * h]) for h in range(1, H + 1)])
        peak = np.nanmax(np.vstack([np.zeros(len(c)), fr]), axis=0)
        final = fr[H - 1]
        gb = peak - final
        f["fut_ret_1"], f["fut_ret_4"], f["fut_mfe"] = fr[0], fr[3], peak
        f["fut_giveback"] = gb
        f["severe_giveback"] = (gb >= 0.01).astype(float)
        f["timestamp"], f["symbol"] = bars["timestamp"].values, sym
        frames.append(f[BASE_INPUTS + ["vwap"] + LABELS + ["timestamp", "symbol"]])
    panel = pd.concat(frames).dropna(subset=["severe_giveback"])
    return panel.set_index(["symbol", "timestamp"]).sort_index()


def _fingerprint(v):
    """Paper's pruning (Sec 4.2): identify alphas by their PREDICTION, not formula.
    Per-timestamp ranks -> monotone/cross-section-wrapped variants collapse to one,
    so rank-IC degeneracy (cs_rank(atr_pct) == atr_pct) cannot flood the search."""
    rk = v.replace([np.inf, -np.inf], np.nan).groupby(level=1).rank(pct=True).dropna()
    if len(rk) < 200:
        return None
    step = max(1, len(rk) // 2500)
    return hash(tuple(np.round(rk.values[::step], 2)))


def mine(train, target, rng, pop_n, evo_steps, tour, seeds=None):
    """Regularized evolution; returns every distinct (by prediction) evaluated alpha.
    `seeds` (e.g. the 101 alphas) are injected into the initial population."""
    seen_f, seen_fp, evald, pop = set(), set(), [], []

    def consider(node, source):
        f = formula_str(node)
        if f in seen_f:
            return None
        seen_f.add(f)
        try:
            v = eval_alpha(node, train)
        except Exception:
            return None
        fp = _fingerprint(v)
        if fp is None or fp in seen_fp:        # value-equivalent -> not a new alpha
            return None
        seen_fp.add(fp)
        ic, icir, n = spearman_ic(v, target)
        if n < 30 or not np.isfinite(ic):
            return None
        rec = {"node": node, "formula": f, "ic": ic, "ic_ir": icir, "source": source}
        evald.append(rec)
        return rec

    for node in (seeds or []):                       # domain-expert 101 alpha seeds
        r = consider(node, "seed_101")
        if r:
            pop.append(r)
    tries = 0
    while len(pop) < pop_n and tries < pop_n * 6:
        tries += 1
        r = consider(random_alpha(rng, BASE_INPUTS), "random")
        if r:
            pop.append(r)
    for step in range(evo_steps):
        sel = [pop[i] for i in rng.choice(len(pop), min(tour, len(pop)), replace=False)]
        parent = max(sel, key=lambda r: abs(r["ic"]))
        r = consider(mutate(parent["node"], rng, BASE_INPUTS), "mutation")
        if r is None:                       # duplicate/invalid -> inject fresh blood
            r = consider(random_alpha(rng, BASE_INPUTS), "random")
        if r:
            pop.append(r); pop.pop(0)       # evict the oldest (aging)
        if (step + 1) % 100 == 0:
            print(f"  evo {step + 1}/{evo_steps} | distinct {len(evald)} | "
                  f"best|IC| {max(abs(x['ic']) for x in evald):.4f}")
    return evald


def build_candidates(evald, train, oos, tgt_tr, tgt_oos, panel, topk, q, n_tests):
    """Compute every registry-gate metric for the top-K alphas: block wild-bootstrap
    p-value -> Benjamini-Hochberg significance (full search burden), OOS IC, turnover,
    leakage check. Returns (candidates, abs-corr matrix, fdr_diagnostics)."""
    top = sorted(evald, key=lambda r: -abs(r["ic"]))[:topk]
    vals, cands = {}, []
    for r in top:
        v = eval_alpha(r["node"], train)
        vals[r["formula"]] = v
        p = round(mt.wild_bootstrap_pvalue(ic_series(v, tgt_tr), 1000, block=H), 4)
        oic, _, _ = spearman_ic(eval_alpha(r["node"], oos), tgt_oos)
        to = _turnover(eval_alpha(r["node"], panel))
        cands.append({"node": r["node"], "formula": r["formula"], "ic": r["ic"],
                      "ic_ir": r["ic_ir"], "oos_ic": oic, "p_value": p, "turnover": to,
                      "leak_ok": alpha_registry.is_pit_safe(r["node"]),
                      "source": r["source"], "category": category(r["node"])})
    fdr = mt.fdr_control([c["p_value"] for c in cands], q=q, n_tests=n_tests)
    fdr["n_tested"], fdr["search_burden"] = len(cands), n_tests
    for c, d in zip(cands, fdr["discoveries"]):
        c["fdr_significant"] = bool(d)
    M = pd.DataFrame(vals)
    good = {c for c in M.columns if M[c].notna().mean() > 0.3 and M[c].std(skipna=True) > 1e-9}
    Msub = M[list(good)].sample(min(40000, len(M)), random_state=0).rank()
    corr = Msub.corr().abs()
    return cands, corr, fdr


def _reason_counts(rejected):
    from collections import Counter
    c = Counter(r["reasons"][0].split("(")[0] for r in rejected if r["reasons"])
    if c:
        print("  rejection reasons:", ", ".join(f"{k}={n}" for k, n in c.most_common()))


def _turnover(v):
    """Per the Performance-vs-Turnover paper: fraction of the (z-scored) book
    traded per period. Reported as a diagnostic -- NOT a selection criterion,
    since turnover has no significant effect on performance."""
    v = v.replace([np.inf, -np.inf], np.nan)
    tl = v.index.get_level_values(1)
    w = v.groupby(tl).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    num = w.groupby(level=0).diff().abs().groupby(level=1).sum()
    den = w.abs().groupby(level=1).sum() + 1e-12
    t = float((num / den).mean())
    return round(t, 3) if np.isfinite(t) and t < 10 else None   # guard degenerate alphas


def benchmark_101(train, oos, tgt_tr, tgt_oos):
    """IC of each 101 alpha vs severe_giveback (train + oos) and vs forward return."""
    rows = []
    for name, node in alphas101.library():
        try:
            ic, icir, _ = spearman_ic(eval_alpha(node, train), tgt_tr)
            oic, _, _ = spearman_ic(eval_alpha(node, oos), tgt_oos)
            ric, _, _ = spearman_ic(eval_alpha(node, train), train["fut_ret_4"])
        except Exception:
            continue
        rows.append({"name": name, "ic": ic, "ic_ir": icir, "oos_ic": oic, "ret_ic": ric,
                     "formula": formula_str(node)})
    return sorted(rows, key=lambda r: -abs(r["ic"]))


def _sharpe(v, fut1):
    df = pd.DataFrame({"w": v, "r": fut1}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 100:
        return 0.0
    tl = df.index.get_level_values(1)
    z = df["w"].groupby(tl).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    pnl = (z * df["r"]).groupby(tl).sum() / (z.abs().groupby(tl).sum() + 1e-9)
    return round(float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252 * 26)), 3) if pnl.std() else 0.0


def ensemble(panel, kept, out):
    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    y = panel[TARGET].astype(int).to_numpy()
    g = panel.index.get_level_values(0)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(panel, groups=g))

    def auc(X):
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        return round(float(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1])), 4)

    # lightgbm rejects special chars in feature names -> use a0,a1,... not formulas
    Xa = pd.DataFrame({f"a{i}": eval_alpha(r["node"], panel) for i, r in enumerate(kept)},
                      index=panel.index)
    raw = auc(panel[BASE_INPUTS])
    mined = auc(Xa)
    both = auc(pd.concat([panel[BASE_INPUTS], Xa], axis=1))
    return {"raw_inputs_auc": raw, "mined_alphas_auc": mined, "raw_plus_alphas_auc": both,
            "n_alpha_features": len(kept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--max-symbols", type=int, default=25)
    ap.add_argument("--months", type=int, default=8)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--evo-steps", type=int, default=500)
    ap.add_argument("--tournament", type=int, default=10)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--corr-cut", type=float, default=0.7)
    ap.add_argument("--max-accept", type=int, default=60)
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--fdr-q", type=float, default=0.10, help="false-discovery-rate level")
    ap.add_argument("--oos-retain", type=float, default=0.4, help="min |OOS IC| / |train IC|")
    ap.add_argument("--turnover-cap", type=float, default=2.0, help="max per-bar turnover")
    ap.add_argument("--seed-101", action="store_true", help="seed evolution with the 101 alphas")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bars_dir = Path(args.bars_dir)

    df = pd.read_parquet(args.dataset, columns=["symbol", "strategy_id"])
    symbols = list(df["symbol"].value_counts().index[:args.max_symbols])
    cache = out / f"_alpha_panel_{args.max_symbols}_{args.months}_v2.parquet"
    if cache.exists() and not args.rebuild:
        panel = pd.read_parquet(cache)
        print(f"loaded panel {cache.name}: {panel.shape}")
    else:
        st = pd.read_parquet(args.dataset, columns=["entry_time", "exit_time"])
        start = pd.to_datetime(st["entry_time"]).min().date().isoformat()
        end = (pd.to_datetime(st["exit_time"]).max() + pd.Timedelta(days=1)).date().isoformat()
        refs = _load_refs(start, end, bars_dir)
        print(f"building panel: {len(symbols)} symbols, last {args.months} months...")
        panel = build_panel(symbols, bars_dir, refs, args.months)
        panel.to_parquet(cache)
    print(f"panel: {panel.shape[0]:,} rows x {len(BASE_INPUTS)} inputs | "
          f"symbols {panel.index.get_level_values(0).nunique()} | "
          f"severe-giveback rate {panel[TARGET].mean():.3f}")

    ts = panel.index.get_level_values(1)
    cut = pd.Series(ts).quantile(0.7)
    train = panel[ts < cut]; oos = panel[ts >= cut]
    tgt_tr, tgt_oos = train[TARGET], oos[TARGET]

    bench = None
    seeds = None
    if args.seed_101:
        bench = benchmark_101(train, oos, tgt_tr, tgt_oos)
        seeds = [node for _, node in alphas101.library()]
        best = max(bench, key=lambda b: abs(b["ic"]))
        print(f"101-alpha benchmark vs severe_giveback: best {best['name']} "
              f"IC {best['ic']} (oos {best['oos_ic']}) | median |IC| "
              f"{np.median([abs(b['ic']) for b in bench]):.3f}")

    rng = np.random.default_rng(0)
    print(f"\nmining (pop {args.pop}, evo {args.evo_steps}, tournament {args.tournament}, "
          f"seed_101={args.seed_101})...")
    evald = mine(train, tgt_tr, rng, args.pop, args.evo_steps, args.tournament, seeds)
    print(f"distinct alphas evaluated: {len(evald):,}")

    cands, corr, fdr = build_candidates(evald, train, oos, tgt_tr, tgt_oos, panel,
                                        args.topk, args.fdr_q, len(evald))
    cfg = {"oos_min_retain": args.oos_retain, "corr_cap": args.corr_cut,
           "turnover_cap": args.turnover_cap}
    accepted, rejected = alpha_registry.promote(cands, corr, cfg)
    accepted = accepted[:args.max_accept]
    kept = accepted                                      # the accepted registry IS the ensemble set
    print(f"FDR q={args.fdr_q}: {fdr['n_discoveries']} of {fdr['n_tested']} significant -> "
          f"REGISTRY accepted {len(accepted)} / rejected {len(rejected)} "
          f"(of {len(cands)} candidates)")

    # accepted_alphas.csv / rejected_alphas.csv with exact reasons
    acol = ["formula", "category", "source", "ic", "oos_ic", "ic_ir", "p_value",
            "fdr_significant", "turnover", "corr_to_existing"]
    rcol = ["formula", "category", "ic", "oos_ic", "p_value", "fdr_significant",
            "turnover", "reasons"]
    pd.DataFrame([{k: a.get(k) for k in acol} for a in accepted]).to_csv(
        out / "accepted_alphas.csv", index=False)
    pd.DataFrame([{**{k: r.get(k) for k in rcol[:-1]}, "reasons": "; ".join(r["reasons"])}
                  for r in rejected]).to_csv(out / "rejected_alphas.csv", index=False)
    _reason_counts(rejected)

    # persist DB (accepted + rejected with status)
    records = []
    for a in accepted:
        v_all = eval_alpha(a["node"], panel)
        rec = alpha_db.record(alpha_id(a["node"]), a["formula"], a["category"], a["ic"],
                              a["ic_ir"], _sharpe(v_all, panel["fut_ret_1"]),
                              a.get("corr_to_existing"),
                              float(v_all.dropna().groupby(level=1).size().median()),
                              a["oos_ic"], a["source"], "accepted")
        rec["p_value"] = a["p_value"]; rec["turnover"] = a["turnover"]
        records.append(rec)
    for r in rejected:
        rec = alpha_db.record(alpha_id(r["node"]), r["formula"], r["category"], r["ic"],
                              r["ic_ir"], None, None, None, r["oos_ic"], r["source"], "rejected")
        rec["p_value"] = r["p_value"]; rec["turnover"] = r["turnover"]
        rec["rejection_reasons"] = r["reasons"]
        records.append(rec)
    alpha_db.save(records, out / "alpha_db.json")

    print("\ntraining LightGBM ensemble (mined alphas vs raw inputs)...")
    ens = ensemble(panel, kept, out)

    rep = {"panel_rows": int(panel.shape[0]), "symbols": int(panel.index.get_level_values(0).nunique()),
           "severe_giveback_rate": round(float(panel[TARGET].mean()), 3),
           "distinct_alphas": len(evald), "accepted_alphas": len(kept),
           "rejected_alphas": len(rejected),
           "best_ic": round(max(abs(x["ic"]) for x in evald), 5),
           "top_accepted": [{"formula": r["formula"], "ic": r["ic"], "ic_ir": r["ic_ir"],
                             "corr_to_existing": r.get("corr_to_existing")}
                            for r in sorted(kept, key=lambda r: -abs(r["ic"]))[:15]],
           "alpha101_benchmark": bench, "ensemble": ens,
           "n_seed_101_accepted": sum(1 for r in kept if r["source"] == "seed_101"),
           "fdr": {k: v for k, v in fdr.items() if k != "discoveries"},
           "median_turnover": round(float(np.median([rec["turnover"] for rec in records
                                                     if rec["status"] == "accepted" and rec.get("turnover") is not None])), 3)
           if kept else None}
    pd.DataFrame(records).to_csv(out / "alpha_mining_alphas.csv", index=False)
    (out / "alpha_mining_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nartifacts -> {out}/ (alpha_db.json, alpha_mining_report.json, alphas.csv)")


def _print(rep):
    print(f"\n=== alpha mining (severe_giveback, {rep['panel_rows']:,} bars, "
          f"{rep['symbols']} symbols) ===")
    print(f"distinct alphas {rep['distinct_alphas']:,} | accepted {rep['accepted_alphas']} | "
          f"best |IC| {rep['best_ic']}")
    f = rep["fdr"]
    print(f"FDR (q={f['q']}): {f['n_discoveries']} significant of {f['n_tested']} tested "
          f"(burden {f['search_burden']}), ~{f['expected_false_discoveries']} expected false; "
          f"median turnover {rep['median_turnover']}")
    print("\ntop accepted alphas (IC vs severe_giveback):")
    for r in rep["top_accepted"]:
        print(f"  IC {r['ic']:+.4f} (IR {r['ic_ir']}) corr {r['corr_to_existing']}  {r['formula']}")
    if rep.get("alpha101_benchmark"):
        b = rep["alpha101_benchmark"]
        print(f"\n=== 101-alpha benchmark vs severe_giveback (top 8 by |IC|) ===")
        print(f"  ({rep.get('n_seed_101_accepted', 0)} of the 101 seeds survived into the "
              "accepted independent library)")
        for r in b[:8]:
            print(f"  IC {r['ic']:+.4f} (oos {r['oos_ic']:+.4f}, ret-IC {r['ret_ic']:+.4f})  {r['name']}")
    e = rep["ensemble"]
    print(f"\n=== ensemble AUC (held-out symbols) ===")
    print(f"  raw {len(BASE_INPUTS)} inputs        {e['raw_inputs_auc']}")
    print(f"  {e['n_alpha_features']} mined alphas       {e['mined_alphas_auc']}")
    print(f"  raw + mined            {e['raw_plus_alphas_auc']}")
    d = e["mined_alphas_auc"] - e["raw_inputs_auc"]
    print(f"\n-> mined alphas {'BEAT' if d > 0.005 else 'do NOT beat'} raw inputs "
          f"({d:+.4f}); combining {'helps' if e['raw_plus_alphas_auc'] > max(e['raw_inputs_auc'], e['mined_alphas_auc']) + 0.002 else 'no extra'}.")


if __name__ == "__main__":
    main()
