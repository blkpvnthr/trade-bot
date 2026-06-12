# Unsupervised NASDAQ Regime Discovery

489,346 stock-days, 500 tickers, 19 contemporaneous features. NO forward returns in fitting (winsorize→standardize→PCA→UMAP→HDBSCAN/KMeans see features only). Forward returns attached AFTER clustering. KMeans k=8 on top-10 PCs.

## 1. PCA — latent factors

Top-10 PCs explain **96%** of variance (PC1 39%, PC2 16%, PC3 12%). Top loadings:

- **PC1** (39%): distance_to_SMA20 (+0.34), distance_to_SMA50 (+0.34), ret_20d (+0.32), relative_strength_vs_QQQ_20 (+0.31), RSI_14 (+0.31)
- **PC2** (16%): realized_vol_20 (+0.53), ATR_pct (+0.53), realized_vol_10 (+0.52), drawdown_from_52w_high (-0.36), relative_strength_vs_QQQ_60 (+0.12)
- **PC3** (12%): relative_strength_vs_QQQ_60 (+0.44), ret_60d (+0.44), distance_to_SMA200 (+0.41), ret_5d (-0.28), MACD_hist (-0.27)

## 2-3. Cluster forward-return stats

| Regime | n | Mean | Median | Win% | Sharpe | PF | (fwd 10d) |
|---|---|---|---|---|---|---|---|
| Regime 0 | 156,632 | -0.02% | -0.01% | 49% | -0.002 | 0.99 | |
| Regime 1 | 54,317 | -0.14% | -1.21% | 45% | -0.009 | 0.98 | |
| Regime 2 | 7,611 | -0.24% | -1.91% | 44% | -0.012 | 0.97 | |
| Regime 3 | 41,052 | +0.07% | -0.13% | 49% | +0.006 | 1.02 | |
| Regime 4 | 95,460 | +0.00% | -0.05% | 49% | +0.000 | 1.00 | |
| Regime 5 | 85,683 | +0.20% | -0.27% | 48% | +0.015 | 1.04 | |
| Regime 6 | 19,382 | -0.06% | -1.19% | 46% | -0.003 | 0.99 | |
| Regime 7 | 24,411 | +2.05% | +0.00% | 48% | +0.103 | 1.31 | |

_Also computed fwd_5d / fwd_20d (in CSV). Judge edge by Sharpe + PF + n._

Baseline (all stock-days): mean +0.12%, median -0.17%, Sharpe +0.010, PF 1.03

## 4. Regime persistence

- P(same regime next day) = **71%** (random would be ~12%)
- Average consecutive days in a regime = **3.4**

## 5. Regime transitions (fwd 10d, n≥50)

| Transition | n | Mean | Median | Win% | Sharpe | PF |
|---|---|---|---|---|---|---|
| R2 → R7 | 82 | +14.43% | +8.90% | 59% | +0.506 | 3.62 |
| R6 → R7 | 369 | +13.81% | +6.59% | 55% | +0.545 | 4.46 |
| R1 → R7 | 1,173 | +2.92% | +0.00% | 48% | +0.136 | 1.42 |
| R0 → R6 | 149 | +1.47% | +0.34% | 52% | +0.086 | 1.27 |
| R4 → R5 | 247 | +1.42% | -0.29% | 48% | +0.085 | 1.25 |
| R4 → R7 | 61 | +1.20% | -2.28% | 39% | +0.060 | 1.18 |
| … | | | | | | |
| R5 → R6 | 155 | -2.50% | -4.33% | 39% | -0.115 | 0.75 |
| R7 → R2 | 93 | -6.22% | -9.50% | 31% | -0.284 | 0.49 |
| R7 → R6 | 335 | -10.03% | -13.23% | 27% | -0.434 | 0.35 |

## PHASE 2 — signal performance by regime (regime as a FILTER, not a strategy)

Momentum-shares & insider-shares forward returns within each regime vs the regime's unconditional baseline. **Benchmark to beat = insider-shares: +2.71%/PF2.35 @3d, +3.02%/PF2.24 @5d.** A regime that AMPLIFIES a signal's edge is the prize.

### 3-day forward returns by regime

| Regime | clust n | Base mean | **Mom** n / mean / edge / PF | **Insider** n / mean / edge / PF |
|---|---|---|---|---|
| R0 | 157,770 | -0.02% | 16⚠ / — / — / — | 191 / +0.97% / +0.99% / 1.93 |
| R1 | 54,676 | -0.16% | 166 / -1.72% / -1.56% / 0.61 | 83 / +4.35% / +4.51% / 3.57 |
| R2 | 7,689 | -0.35% | 142 / -0.69% / -0.34% / 0.88 | 19⚠ / — / — / — |
| R3 | 41,311 | +0.03% | 1,947 / -0.11% / -0.15% / 0.94 | 98 / +2.10% / +2.07% / 2.33 |
| R4 | 96,217 | -0.02% | 2,360 / -0.07% / -0.05% / 0.96 | 98 / +1.95% / +1.96% / 3.26 |
| R5 | 86,116 | +0.07% | 0⚠ / — / — / — | 141 / +2.88% / +2.82% / 3.03 |
| R6 | 19,610 | -0.39% | 1,807 / -0.39% / -0.00% / 0.91 | 55 / +4.51% / +4.90% / 3.06 |
| R7 | 24,518 | +0.83% | 0⚠ / — / — / — | 99 / +3.77% / +2.94% / 2.63 |

### 5-day forward returns by regime

| Regime | clust n | Base mean | **Mom** n / mean / edge / PF | **Insider** n / mean / edge / PF |
|---|---|---|---|---|
| R0 | 157,464 | -0.02% | 16⚠ / — / — / — | 191 / +1.33% / +1.34% / 2.09 |
| R1 | 54,585 | -0.18% | 166 / -1.83% / -1.64% / 0.66 | 83 / +3.26% / +3.44% / 2.65 |
| R2 | 7,669 | -0.32% | 142 / -0.59% / -0.27% / 0.92 | 19⚠ / — / — / — |
| R3 | 41,228 | +0.00% | 1,947 / -0.04% / -0.05% / 0.98 | 98 / +2.19% / +2.19% / 2.23 |
| R4 | 95,990 | -0.01% | 2,360 / -0.05% / -0.04% / 0.98 | 98 / +2.37% / +2.38% / 2.86 |
| R5 | 85,996 | +0.12% | 0⚠ / — / — / — | 141 / +2.91% / +2.78% / 2.54 |
| R6 | 19,531 | -0.45% | 1,807 / -0.95% / -0.50% / 0.83 | 55 / +4.45% / +4.89% / 2.53 |
| R7 | 24,485 | +1.20% | 0⚠ / — / — / — | 99 / +3.56% / +2.35% / 1.94 |

**Best regime-conditioned signal (5d, n≥30):**
- Momentum: R3 → -0.04% (edge -0.05% vs regime base), PF 0.98, n=1,947
- Insider: R6 → +4.45% (edge +4.89% vs regime base), PF 2.53, n=55

## HDBSCAN (natural cluster count)

On a 40,000-row subsample HDBSCAN found **0 dense clusters** with **100% noise** — differs from the k=8 KMeans choice. (KMeans used for full assignment as it labels every point + is predict-able.)

## Final answers

1. **Did PCA find latent factors?** Yes — top-10 PCs capture 96%; PC1≈39% is a dominant axis. See loadings above for the factor meanings.
2. **Persistent regimes?** P(stay)=71% vs ~12% random, avg 3.4 days/regime → YES, regimes persist.
3. **Any regime with forward edge?** Best regime 7 mean +2.05% vs worst 2 -0.24% (spread +2.30%/10d). A meaningful cross-section.
4. **Any transition with edge?** Best R2→R7: +14.43%/10d, PF 3.62, n=82.
5. **Stable enough for a supervised study?** YES — persistent regimes + a forward-return cross-section + sensible PCA factors justify a follow-up supervised regime-conditioned study.

_Plots: regimes_pca_variance.png, regimes_forward_returns.png, regimes_umap_scatter.png. Constraints honored: no forward returns in fitting, no supervised labels, no strategy training, no optimization._