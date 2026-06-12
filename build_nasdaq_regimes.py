#!/usr/bin/env python3
"""
build_nasdaq_regimes.py — UNSUPERVISED NASDAQ regime/factor discovery.

Find naturally occurring market regimes WITHOUT using future returns during fitting. Only AFTER
regimes are discovered do we attach forward returns and ask whether any regime/transition has edge.

Discipline (enforced structurally): the feature matrix X holds ONLY the 19 contemporaneous features.
Winsorize → standardize → PCA → UMAP → HDBSCAN/KMeans all see X only. forward_{5,10,20}d are
computed and joined AFTER cluster labels exist. No supervised labels, no strategy training, no
optimization — pure factor/regime discovery.

Outputs: backtest_outputs/unsupervised_nasdaq_regimes.{md,csv} + 3 PNGs (explained variance,
cluster forward returns, 2D UMAP scatter).
"""
from __future__ import annotations

import argparse
import glob
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_ablation_studies as ab

warnings.filterwarnings("ignore")
OUT = base.PROJECT_ROOT / "backtest_outputs"

FEATURES = ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "realized_vol_10",
            "realized_vol_20", "ATR_pct", "RSI_14", "ROC_10", "MACD_hist", "volume_zscore_20",
            "relative_volume_20", "distance_to_SMA20", "distance_to_SMA50", "distance_to_SMA200",
            "drawdown_from_52w_high", "relative_strength_vs_QQQ_20", "relative_strength_vs_QQQ_60"]


def _rsi(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _features(bars: pd.DataFrame, qqq_ret20: pd.Series, qqq_ret60: pd.Series) -> pd.DataFrame:
    c, h, l, v = bars["close"], bars["high"], bars["low"], bars["volume"]
    r1 = c.pct_change()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f = pd.DataFrame(index=bars.index)
    f["ret_1d"] = r1
    f["ret_5d"] = c.pct_change(5)
    f["ret_10d"] = c.pct_change(10)
    f["ret_20d"] = c.pct_change(20)
    f["ret_60d"] = c.pct_change(60)
    f["realized_vol_10"] = r1.rolling(10).std()
    f["realized_vol_20"] = r1.rolling(20).std()
    f["ATR_pct"] = tr.rolling(14).mean() / c
    f["RSI_14"] = _rsi(c)
    f["ROC_10"] = (c / c.shift(10) - 1) * 100
    f["MACD_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    f["volume_zscore_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["relative_volume_20"] = v / v.rolling(20).mean()
    f["distance_to_SMA20"] = c / c.rolling(20).mean() - 1
    f["distance_to_SMA50"] = c / c.rolling(50).mean() - 1
    f["distance_to_SMA200"] = c / c.rolling(200).mean() - 1
    f["drawdown_from_52w_high"] = c / h.rolling(252).max() - 1
    f["relative_strength_vs_QQQ_20"] = c.pct_change(20) - qqq_ret20.reindex(bars.index)
    f["relative_strength_vs_QQQ_60"] = c.pct_change(60) - qqq_ret60.reindex(bars.index)
    # forward returns — computed here but kept SEPARATE; NEVER enter X
    f["fwd_3d"] = c.shift(-3) / c - 1
    f["fwd_5d"] = c.shift(-5) / c - 1
    f["fwd_10d"] = c.shift(-10) / c - 1
    f["fwd_20d"] = c.shift(-20) / c - 1
    return f


def build_matrix(max_tickers: int, min_bars: int = 500) -> pd.DataFrame:
    pit = base.load_pit_buy_events()                       # for the insider-signal flag (Phase 2)
    allfp = {fp.rsplit("/", 1)[-1].rsplit("_day", 1)[0]: fp
             for fp in sorted(glob.glob(str(base.BARS_DIR / "*_day.parquet")))}
    # universe: PRIORITISE insider PIT names (so Phase 2 has insider coverage), then stride-fill
    insider_syms = [s for s in sorted(pit.keys()) if s in allfp and s != "QQQ"]
    others = [s for s in allfp if s not in set(insider_syms) and s != "QQQ"]
    others = others[:: max(1, len(others) // (max_tickers))]
    order = (insider_syms + others)[: max_tickers * 3]
    # PASS 1: load bars, select tickers with enough history
    selected = []
    for sym in order:
        if len(selected) >= max_tickers:
            break
        try:
            b = pd.read_parquet(allfp[sym])
        except Exception:
            continue
        if b is not None and len(b) >= min_bars:
            selected.append((sym, b))
    # full-coverage MARKET PROXY from the universe's own cross-sectional median daily return
    # (the cached QQQ file is incomplete — only recent bars — so it can't be the benchmark).
    rets = pd.DataFrame({sym: b["close"].pct_change() for sym, b in selected})
    mkt_idx = (1 + rets.median(axis=1).fillna(0.0)).cumprod()
    mkt_ret20 = mkt_idx.pct_change(20); mkt_ret60 = mkt_idx.pct_change(60)
    rows = []
    used = 0
    for sym, b in selected:
        f = _features(b, mkt_ret20, mkt_ret60)
        # SIGNAL FLAGS (momentum, insider) — added like forward returns, NEVER enter X
        td = list(b.index)
        cd = {d: float(b.loc[d, "close"]) for d in td}
        hd = {d: float(b.loc[d, "high"]) for d in td}
        vd = {d: float(b.loc[d, "volume"]) for d in td}
        mflag = np.zeros(len(b), bool); iflag = np.zeros(len(b), bool)
        for i in ab._momentum_entry_indices(td, cd, hd, vd):
            mflag[i] = True
        if sym in pit:
            td_dates = [d.date() if hasattr(d, "date") else d for d in td]   # insider logic wants dates
            for i in ab._insider_entry_indices(td_dates, sym, pit):
                iflag[i] = True
        f["momentum_signal"] = mflag
        f["insider_signal"] = iflag
        f.insert(0, "ticker", sym)
        f = f.reset_index().rename(columns={f.index.name or "index": "date"})
        f.columns = ["date", "ticker"] + list(f.columns[2:])
        rows.append(f)
        used += 1
    df = pd.concat(rows, ignore_index=True)
    # drop rows with any missing FEATURE (forward cols may be NaN at tails — kept)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"  feature matrix: {len(df):,} rows × {len(FEATURES)} features from {used} tickers")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=500)
    ap.add_argument("--k", type=int, default=8, help="KMeans regimes (fixed, no optimization)")
    ap.add_argument("--sub", type=int, default=40000, help="subsample for UMAP/HDBSCAN")
    args = ap.parse_args()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import umap, hdbscan

    print("1) Building feature matrix (no forward returns in X)...")
    df = build_matrix(args.max_tickers)

    print("2) Winsorize (1/99 pct) + 3) standardize...")
    X = df[FEATURES].copy()
    for col in FEATURES:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(lo, hi)
    Xs = StandardScaler().fit_transform(X.values)

    print("4) PCA...")
    pca = PCA(n_components=len(FEATURES), random_state=7).fit(Xs)
    Xp = pca.transform(Xs)
    evr = pca.explained_variance_ratio_

    print("6) UMAP (subsample) + 7) HDBSCAN (subsample) + KMeans (full)...")
    rng = np.random.RandomState(7)
    sub = rng.choice(len(Xp), size=min(args.sub, len(Xp)), replace=False)
    top = Xp[:, :10]                                  # cluster on the leading 10 PCs
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=7).fit_transform(top[sub])
    hdb = hdbscan.HDBSCAN(min_cluster_size=max(50, len(sub) // 100), min_samples=10).fit(top[sub])
    n_hdb = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
    noise = float((hdb.labels_ == -1).mean())
    km = KMeans(n_clusters=args.k, random_state=7, n_init=10).fit(top)
    df["cluster"] = km.labels_
    df["pc1"], df["pc2"], df["pc3"] = Xp[:, 0], Xp[:, 1], Xp[:, 2]

    print("9) Attaching forward returns (post-clustering) + analysis...")
    # winsorize forward returns (1/99) so cluster MEANS aren't dominated by small-cap tail outliers
    # (a few names that 5x in 10 days). Features were already winsorized; this is only for the stats.
    for col in ["fwd_3d", "fwd_5d", "fwd_10d", "fwd_20d"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
    _write_outputs(df, pca, evr, emb, sub, km, n_hdb, noise, args)
    print("DONE")


def _stat(s: pd.Series):
    v = s.dropna().values
    if len(v) == 0:
        return None
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)),
            "win": (v > 0).mean(), "sharpe": (v.mean() / sd) if sd > 0 else 0.0, "pf": pf}


def _row(label, s):
    if not s:
        return f"| {label} | 0 |—|—|—|—|—|"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {label} | {s['n']:,} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | "
            f"{s['win']*100:.0f}% | {s['sharpe']:+.3f} | {pf} |")


def _write_outputs(df, pca, evr, emb, sub, km, n_hdb, noise, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = args.k
    L = ["# Unsupervised NASDAQ Regime Discovery", "",
         f"{len(df):,} stock-days, {df['ticker'].nunique()} tickers, {len(FEATURES)} contemporaneous "
         f"features. NO forward returns in fitting (winsorize→standardize→PCA→UMAP→HDBSCAN/KMeans see "
         f"features only). Forward returns attached AFTER clustering. KMeans k={K} on top-10 PCs.", ""]

    # ── PCA interpretation ──
    L += ["## 1. PCA — latent factors", "",
          f"Top-10 PCs explain **{evr[:10].sum()*100:.0f}%** of variance "
          f"(PC1 {evr[0]*100:.0f}%, PC2 {evr[1]*100:.0f}%, PC3 {evr[2]*100:.0f}%). Top loadings:", ""]
    comp = pd.DataFrame(pca.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
    LABELS = {0: "PC1", 1: "PC2", 2: "PC3"}
    for j in range(3):
        top_load = comp[LABELS[j]].abs().sort_values(ascending=False).head(5)
        loads = ", ".join(f"{k} ({comp[LABELS[j]][k]:+.2f})" for k in top_load.index)
        L.append(f"- **{LABELS[j]}** ({evr[j]*100:.0f}%): {loads}")
    L += [""]

    # ── Cluster forward-return stats ──
    L += ["## 2-3. Cluster forward-return stats", "",
          "| Regime | n | Mean | Median | Win% | Sharpe | PF | (fwd 10d) |", "|---|---|---|---|---|---|---|---|"]
    cl_means = {}
    for cyc in sorted(df["cluster"].unique()):
        s = _stat(df.loc[df["cluster"] == cyc, "fwd_10d"])
        cl_means[cyc] = s["mean"] if s else 0
        L.append(_row(f"Regime {cyc}", s) + " |")
    L += ["", "_Also computed fwd_5d / fwd_20d (in CSV). Judge edge by Sharpe + PF + n._", ""]
    # overall baseline
    base_s = _stat(df["fwd_10d"])
    L += [f"Baseline (all stock-days): mean {base_s['mean']*100:+.2f}%, median {base_s['median']*100:+.2f}%, "
          f"Sharpe {base_s['sharpe']:+.3f}, PF {base_s['pf']:.2f}", ""]

    # ── Persistence ──
    df_sorted = df.sort_values(["ticker", "date"])
    nxt = df_sorted.groupby("ticker")["cluster"].shift(-1)
    same = (df_sorted["cluster"] == nxt)
    p_same = float(same[nxt.notna()].mean())
    # avg run length per cluster
    runs = []
    for _, g in df_sorted.groupby("ticker"):
        cl = g["cluster"].values
        rl = 1
        for a in range(1, len(cl)):
            if cl[a] == cl[a - 1]:
                rl += 1
            else:
                runs.append(rl); rl = 1
        runs.append(rl)
    L += ["## 4. Regime persistence", "",
          f"- P(same regime next day) = **{p_same*100:.0f}%** (random would be ~{100/K:.0f}%)",
          f"- Average consecutive days in a regime = **{np.mean(runs):.1f}**", ""]

    # ── Transitions ──
    prev = df_sorted.groupby("ticker")["cluster"].shift(1)
    df_sorted = df_sorted.assign(prev_cluster=prev)
    trans = df_sorted[df_sorted["prev_cluster"].notna() & (df_sorted["prev_cluster"] != df_sorted["cluster"])]
    tr_stats = []
    for (p, c), g in trans.groupby(["prev_cluster", "cluster"]):
        s = _stat(g["fwd_10d"])
        if s and s["n"] >= 50:
            tr_stats.append((s["mean"], int(p), int(c), s))
    tr_stats.sort(reverse=True)
    L += ["## 5. Regime transitions (fwd 10d, n≥50)", "",
          "| Transition | n | Mean | Median | Win% | Sharpe | PF |", "|---|---|---|---|---|---|---|"]
    for m, p, c, s in tr_stats[:6]:
        L.append(_row(f"R{p} → R{c}", s))
    if len(tr_stats) > 6:
        L.append("| … | | | | | | |")
        for m, p, c, s in tr_stats[-3:]:
            L.append(_row(f"R{p} → R{c}", s))
    L += [""]

    # ── PHASE 2: existing signals INSIDE each regime (the real payoff) ──
    L += ["## PHASE 2 — signal performance by regime (regime as a FILTER, not a strategy)", "",
          "Momentum-shares & insider-shares forward returns within each regime vs the regime's "
          "unconditional baseline. **Benchmark to beat = insider-shares: +2.71%/PF2.35 @3d, "
          "+3.02%/PF2.24 @5d.** A regime that AMPLIFIES a signal's edge is the prize.", ""]
    def _sig(mask_col, cyc, fwd):
        sub = df[(df["cluster"] == cyc) & df[mask_col]]
        return _stat(sub[fwd])
    for fwd, lbl in (("fwd_3d", "3-day"), ("fwd_5d", "5-day")):
        L += [f"### {lbl} forward returns by regime", "",
              "| Regime | clust n | Base mean | **Mom** n / mean / edge / PF | **Insider** n / mean / edge / PF |",
              "|---|---|---|---|---|"]
        for cyc in sorted(df["cluster"].unique()):
            base = _stat(df.loc[df["cluster"] == cyc, fwd])
            m = _sig("momentum_signal", cyc, fwd); ins = _sig("insider_signal", cyc, fwd)
            def cell(s):
                if not s or s["n"] < 20:
                    return f"{s['n'] if s else 0}⚠ / — / — / —"
                pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
                return f"{s['n']:,} / {s['mean']*100:+.2f}% / {(s['mean']-base['mean'])*100:+.2f}% / {pf}"
            L.append(f"| R{cyc} | {base['n']:,} | {base['mean']*100:+.2f}% | {cell(m)} | {cell(ins)} |")
        L += [""]
    # best amplification (n≥30) for each signal at 5d
    def _best_amp(mask_col):
        best = None
        for cyc in sorted(df["cluster"].unique()):
            base = _stat(df.loc[df["cluster"] == cyc, "fwd_5d"])
            s = _sig(mask_col, cyc, "fwd_5d")
            if s and s["n"] >= 30:
                edge = s["mean"] - base["mean"]
                if best is None or s["mean"] > best[1]:
                    best = (cyc, s["mean"], edge, s["pf"], s["n"])
        return best
    bm, bi = _best_amp("momentum_signal"), _best_amp("insider_signal")
    L += ["**Best regime-conditioned signal (5d, n≥30):**",
          (f"- Momentum: R{bm[0]} → {bm[1]*100:+.2f}% (edge {bm[2]*100:+.2f}% vs regime base), PF {bm[3]:.2f}, n={bm[4]:,}"
           if bm else "- Momentum: no regime with n≥30"),
          (f"- Insider: R{bi[0]} → {bi[1]*100:+.2f}% (edge {bi[2]*100:+.2f}% vs regime base), PF {bi[3]:.2f}, n={bi[4]:,}"
           if bi else "- Insider: no regime with n≥30"), ""]

    # ── plots ──
    try:
        plt.figure(figsize=(6, 3)); plt.bar(range(1, 11), evr[:10] * 100)
        plt.plot(range(1, 11), np.cumsum(evr[:10]) * 100, "r-o", ms=3)
        plt.xlabel("PC"); plt.ylabel("% variance"); plt.title("PCA explained variance"); plt.tight_layout()
        plt.savefig(OUT / "regimes_pca_variance.png", dpi=90); plt.close()
        order = sorted(cl_means, key=lambda k: cl_means[k])
        plt.figure(figsize=(6, 3)); plt.bar([str(k) for k in order], [cl_means[k] * 100 for k in order])
        plt.xlabel("regime"); plt.ylabel("mean fwd 10d %"); plt.title("Regime forward returns"); plt.tight_layout()
        plt.savefig(OUT / "regimes_forward_returns.png", dpi=90); plt.close()
        plt.figure(figsize=(5, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=km.labels_[sub], s=2, cmap="tab10", alpha=0.4)
        plt.title("UMAP (subsample) colored by KMeans regime"); plt.tight_layout()
        plt.savefig(OUT / "regimes_umap_scatter.png", dpi=90); plt.close()
        plots = "regimes_pca_variance.png, regimes_forward_returns.png, regimes_umap_scatter.png"
    except Exception as e:
        plots = f"(plot error: {str(e)[:60]})"

    # ── HDBSCAN + verdict ──
    L += ["## HDBSCAN (natural cluster count)", "",
          f"On a {len(sub):,}-row subsample HDBSCAN found **{n_hdb} dense clusters** with "
          f"**{noise*100:.0f}% noise** — {'supports' if 4 <= n_hdb <= 14 else 'differs from'} the k={K} "
          f"KMeans choice. (KMeans used for full assignment as it labels every point + is predict-able.)", ""]

    best_cl = max(cl_means, key=lambda k: cl_means[k]); worst_cl = min(cl_means, key=lambda k: cl_means[k])
    spread = (cl_means[best_cl] - cl_means[worst_cl]) * 100
    best_tr = tr_stats[0] if tr_stats else None
    L += ["## Final answers", "",
          f"1. **Did PCA find latent factors?** Yes — top-10 PCs capture {evr[:10].sum()*100:.0f}%; "
          f"PC1≈{evr[0]*100:.0f}% is a dominant axis. See loadings above for the factor meanings.",
          f"2. **Persistent regimes?** P(stay)={p_same*100:.0f}% vs ~{100/K:.0f}% random, avg "
          f"{np.mean(runs):.1f} days/regime → {'YES, regimes persist' if p_same > 1.5/K else 'weak persistence'}.",
          f"3. **Any regime with forward edge?** Best regime {best_cl} mean {cl_means[best_cl]*100:+.2f}% vs "
          f"worst {worst_cl} {cl_means[worst_cl]*100:+.2f}% (spread {spread:+.2f}%/10d). "
          f"{'A meaningful cross-section' if abs(spread) > 1.5 else 'Modest dispersion'}.",
          f"4. **Any transition with edge?** " + (f"Best R{best_tr[1]}→R{best_tr[2]}: "
          f"{best_tr[3]['mean']*100:+.2f}%/10d, PF {best_tr[3]['pf']:.2f}, n={best_tr[3]['n']:,}." if best_tr else "n/a"),
          f"5. **Stable enough for a supervised study?** " +
          ("YES — persistent regimes + a forward-return cross-section + sensible PCA factors justify a "
           "follow-up supervised regime-conditioned study." if (p_same > 1.5 / K and abs(spread) > 1.5)
           else "MARGINAL — persistence and/or forward dispersion are weak; treat as exploratory."),
          "", f"_Plots: {plots}. Constraints honored: no forward returns in fitting, no supervised labels, "
          "no strategy training, no optimization._"]

    (OUT / "unsupervised_nasdaq_regimes.md").write_text("\n".join(L))
    cols = ["date", "ticker", "cluster", "pc1", "pc2", "pc3", "fwd_3d", "fwd_5d", "fwd_10d",
            "fwd_20d", "momentum_signal", "insider_signal"]
    df[cols].to_csv(OUT / "unsupervised_nasdaq_regimes.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
