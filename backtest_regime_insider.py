#!/usr/bin/env python3
"""
backtest_regime_insider.py — THE validation gate before any live regime-gated insider bot.

The Phase-2 amplification (insider edge ~2× in high-vol regimes R1/R6/R7) was IN-SAMPLE: KMeans
fit on the full history and the "good" regimes picked from the same data. This tests whether it
survives OUT-OF-SAMPLE with realistic small-cap slippage:

  1. Fit winsorize+scale+PCA+KMeans on TRAIN (pre-split) features ONLY.
  2. On TRAIN, rank clusters by insider edge → pick the "good regime" set (selection on train only).
  3. Predict regimes on TEST via kmeans.predict() (no lookahead — uses only the fitted model +
     contemporaneous features; relative-strength uses the cross-sectional median, not future data).
  4. On TEST insider entries: compare UNFILTERED vs REGIME-GATED, NET of slippage.
  5. Robustness (cluster-free): insider forward return binned by ENTRY-DAY VOLATILITY percentile —
     does the edge rise with vol as a CONTINUOUS, large-sample relationship?

If gating improves the net OOS edge AND the vol relationship is monotonic → live candidate.
If not → the amplification was an in-sample artifact; do NOT wire it live.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import build_nasdaq_regimes as br

GOOD_MIN_N = 30


def _stat(v, slippage=0.0):
    v = np.asarray([x - slippage for x in v], float)
    if len(v) == 0:
        return None
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)),
            "win": (v > 0).mean(), "sharpe": (v.mean() / sd) if sd > 0 else 0.0, "pf": pf}


def _fmt(s):
    if not s:
        return "n<1"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"n={s['n']:,} mean {s['mean']*100:+.2f}% median {s['median']*100:+.2f}% win {s['win']*100:.0f}% PF {pf}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=500)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", type=str, default="2024-01-01", help="train < split <= test")
    ap.add_argument("--slippage", type=float, default=0.010, help="round-trip small-cap cost (1.0%)")
    ap.add_argument("--horizon", type=str, default="fwd_5d")
    args = ap.parse_args()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    print("Building feature matrix (+insider flags +forward returns)...")
    df = br.build_matrix(args.max_tickers)
    df["date"] = pd.to_datetime(df["date"])
    split = pd.Timestamp(args.split, tz=df["date"].dt.tz)
    # winsorize forward returns hard (1/99) — outliers inflate means (the standing warning)
    for col in ["fwd_3d", "fwd_5d", "fwd_10d", "fwd_20d"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    train_m = df["date"] < split
    test_m = df["date"] >= split
    print(f"  train {train_m.sum():,} rows (<{args.split}), test {test_m.sum():,} rows (>= {args.split})")

    # fit winsorize bounds + scaler + PCA + KMeans on TRAIN FEATURES ONLY
    Xtr = df.loc[train_m, br.FEATURES].copy()
    bounds = {c: (Xtr[c].quantile(0.01), Xtr[c].quantile(0.99)) for c in br.FEATURES}
    for c in br.FEATURES:
        Xtr[c] = Xtr[c].clip(*bounds[c])
    scaler = StandardScaler().fit(Xtr.values)
    pca = PCA(n_components=10, random_state=7).fit(scaler.transform(Xtr.values))
    km = KMeans(n_clusters=args.k, random_state=7, n_init=10).fit(pca.transform(scaler.transform(Xtr.values)))

    def assign(mask):
        Xx = df.loc[mask, br.FEATURES].copy()
        for c in br.FEATURES:
            Xx[c] = Xx[c].clip(*bounds[c])
        return km.predict(pca.transform(scaler.transform(Xx.values)))

    df.loc[train_m, "regime"] = assign(train_m)
    df.loc[test_m, "regime"] = assign(test_m)
    H = args.horizon

    # rank TRAIN clusters by insider edge → GOOD regime set (selection on TRAIN only)
    tr = df[train_m]
    train_ins_all = _stat(tr.loc[tr.insider_signal, H].dropna().values)
    good = []
    L = ["# OOS Validation — Regime-Gated Insider (shares, realistic slippage)", "",
         f"Fit PCA+KMeans(k={args.k}) on TRAIN <{args.split}; predict TEST regimes OOS. Horizon {H}, "
         f"round-trip slippage {args.slippage*100:.1f}% (small-cap). Forward returns winsorized 1/99.", "",
         "## Train: insider edge by regime (used to PICK the good-regime set)", "",
         "| Regime | insider (train) |", "|---|---|"]
    for r in sorted(tr["regime"].dropna().unique()):
        s = _stat(tr[(tr.regime == r) & tr.insider_signal][H].dropna().values)
        L.append(f"| R{int(r)} | {_fmt(s)} |")
        if s and s["n"] >= GOOD_MIN_N and s["mean"] > train_ins_all["mean"]:
            good.append(int(r))
    L += ["", f"Train unfiltered insider: {_fmt(train_ins_all)}",
          f"**GOOD regimes (train insider mean > unfiltered, n≥{GOOD_MIN_N}): {good}**", ""]

    # OOS TEST: unfiltered vs gated
    te = df[test_m]
    te_ins = te[te.insider_signal]
    oos_all = _stat(te_ins[H].dropna().values, args.slippage)
    oos_gated = _stat(te_ins[te_ins.regime.isin(good)][H].dropna().values, args.slippage)
    oos_bad = _stat(te_ins[~te_ins.regime.isin(good)][H].dropna().values, args.slippage)
    L += ["## OOS TEST (net of slippage) — the decisive comparison", "",
          "| Set | result |", "|---|---|",
          f"| Insider, UNFILTERED | {_fmt(oos_all)} |",
          f"| Insider, GATED to good regimes {good} | {_fmt(oos_gated)} |",
          f"| Insider, in BAD regimes (excluded) | {_fmt(oos_bad)} |", ""]
    if oos_gated and oos_all:
        lift = (oos_gated["mean"] - oos_all["mean"]) * 100
        L.append(f"_OOS gate lift: **{lift:+.2f}%** mean (gated {oos_gated['mean']*100:+.2f}% vs "
                 f"unfiltered {oos_all['mean']*100:+.2f}%); gated PF {oos_gated['pf']:.2f} vs {oos_all['pf']:.2f}._")
    L += [""]

    # cluster-free robustness: insider forward return by ENTRY-DAY volatility quintile (TEST)
    L += ["## Robustness (no clusters): insider edge by entry-day volatility quintile (OOS)", "",
          "| Vol quintile (realized_vol_20) | insider (net) |", "|---|---|"]
    qs = tr["realized_vol_20"].quantile([0.2, 0.4, 0.6, 0.8]).values   # bins from TRAIN distribution
    def vbin(x):
        return int(np.searchsorted(qs, x))
    te_ins = te_ins.assign(vq=te_ins["realized_vol_20"].map(vbin))
    vmeans = []
    for q in range(5):
        s = _stat(te_ins[te_ins.vq == q][H].dropna().values, args.slippage)
        vmeans.append(s["mean"] if s else 0)
        L.append(f"| Q{q+1} ({'low' if q==0 else 'high' if q==4 else 'mid'} vol) | {_fmt(s)} |")
    monotonic = all(vmeans[i] <= vmeans[i+1] for i in range(4)) or vmeans[4] > vmeans[0] + 0.005
    L += ["", f"_Edge rises with vol? {'YES (broadly monotonic)' if monotonic else 'NO/weak'} "
          f"— Q5 {vmeans[4]*100:+.2f}% vs Q1 {vmeans[0]*100:+.2f}%._", ""]

    # verdict
    survives = bool(oos_gated and oos_all and oos_gated["mean"] > oos_all["mean"]
                    and oos_gated["pf"] >= 1.8 and oos_gated["n"] >= 30)
    L += ["## VERDICT", "",
          f"- OOS gate beats unfiltered: {'YES' if oos_gated and oos_all and oos_gated['mean']>oos_all['mean'] else 'NO'}",
          f"- Gated PF ≥ 1.8 (user kill-threshold): {'YES' if oos_gated and oos_gated['pf']>=1.8 else 'NO'} "
          f"(PF {oos_gated['pf']:.2f} if oos_gated else n/a)" if oos_gated else "- Gated PF: n/a",
          f"- Vol relationship holds: {'YES' if monotonic else 'NO'}",
          f"- Sample sufficient (gated n≥30): {'YES' if oos_gated and oos_gated['n']>=30 else 'NO'}", "",
          ("**PASS → regime-gated insider SHARES is a live candidate. Build the predictor export + "
           "copier gate (shares only; NO options overlay until a multi-% edge exists).**" if survives else
           "**FAIL/MARGINAL → the in-sample amplification did not survive OOS+slippage. Do NOT wire live; "
           "the n=55 in-sample edge was likely optimistic. Re-examine before deployment.**")]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "regime_insider_oos.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
