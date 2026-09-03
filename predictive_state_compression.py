#!/usr/bin/env python3
"""
predictive_state_compression.py — Quantum-inspired predictive state compression.
================================================================================

Implements the *computational-mechanics* dimensionality reduction of Elliott, Yang, Binder,
Garner, Thompson & Gu, "Extreme dimensionality reduction with quantum modelling" (Phys. Rev. X
adjacent; arXiv:1909.02817, 2020) for this trading system's feature space.

The paper's thesis (faithfully adapted):
  * A stochastic process's PAST contains a few *predictive features* relevant to the FUTURE,
    buried in much irrelevant history. The minimal classical model is the ε-MACHINE, whose
    CAUSAL STATES are equivalence classes of pasts with IDENTICAL conditional future
    distributions:  f(x⃖) = f(x⃖')  ⟺  P(X⃗|x⃖) = P(X⃗|x⃖').  They are the minimal sufficient
    statistic of the past for the future — and "closely mirror the belief states of RL".
  * Statistical complexity  C_μ = −Σ P(s) log₂ P(s)   (Shannon entropy of the causal states).
  * Quantum causal states encode each state as a (generally NON-ORTHOGONAL) pure state |ψ_s⟩;
    quantum memory  C_q = −Tr[ρ log₂ ρ]  (von Neumann entropy of ρ = Σ P(s)|ψ_s⟩⟨ψ_s|), with
    the provable advantage  C_q ≤ C_μ  — overlapping states "enable efficient isolation of
    predictive features". We use the canonical amplitude encoding |ψ_s⟩ = Σ_k √P(k|s)|k⟩,
    whose overlap ⟨ψ_s|ψ_s'⟩ = Σ_k √(P(k|s)P(k|s')) is the Bhattacharyya fidelity of the two
    conditional future distributions — non-orthogonal exactly when futures overlap.

In trading terms: thousands of raw features collapse to a handful of latent CAUSAL MARKET
STATES (momentum continuation/exhaustion, volatility expansion/contraction, accumulation/
distribution, breakout failure, regime transition, …) that carry the predictive information
about FUTURE RETURNS while discarding historical noise.

OBJECTIVE:  maximize I(latent_state ; future_return),  minimize dim(latent) and irrelevant
info. We measure I on a held-out (OOS) slice; every map is fit IN-SAMPLE only, so there is no
look-ahead. A representation is APPROVED only if it preserves OOS predictive power, compresses
dimension, achieves quantum compression (C_q < C_μ), and is leakage-free.

SAFETY: OFFLINE, read-only. No Alpaca, no orders. Never auto-wires embeddings into live; it
writes a report + an embeddings file + an approval recommendation for review (the approved
embeddings are consumed by the RL only when RL_PSC_FEATURES=1 is explicitly set).

Deps: numpy (required); pandas + scikit-learn used when present (graceful numpy fallbacks for
PCA via SVD, k-NN belief states, KMeans, and histogram MI); torch autoencoder optional.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rl_common as rc

logger = logging.getLogger("predictive_state_compression")

# ── feature families (per the brief), mapped to the alpha_entry_dataset columns ──────
FEATURE_GROUPS: Dict[str, List[str]] = {
    "price_technical": ["setup_score", "mtf_trend_fast", "mtf_trend_mid", "mtf_trend_slow",
                        "mtf_alignment", "ga_ema_ratio", "ga_macd_hist", "ga_rsi", "ga_bollinger_pctb"],
    "volume":          ["rel_volume", "catalyst_strength", "or_volume_confirm"],
    "volatility":      ["atr_pct", "vol_regime", "iv_rank"],
    "news_sentiment":  ["ga_sentiment_st", "sentiment_mean_60m", "sentiment_change_30m",
                        "sentiment_acceleration", "news_density_score", "catalyst_strength_score",
                        "hidden_news_continuation_score", "hidden_sentiment_momentum_score"],
    "realtime_av":     ["realtime_quote_momentum", "realtime_intraday_vwap_distance",
                        "realtime_intraday_volume_acceleration", "realtime_technical_alignment_score",
                        "realtime_option_iv_skew", "realtime_option_delta_exposure",
                        "realtime_option_gamma_exposure", "realtime_put_call_sentiment_score",
                        "realtime_volume_oi_unusual_activity_score", "realtime_options_liquidity_score"],
    "insider":         ["insider_trend_score", "insider_confidence_modifier"],
    "options":         ["black_scholes_score", "oops_score", "optionlab_score",
                        "probability_of_profit", "expected_R", "oops_cert_equiv_return",
                        "oops_prob_total_loss"],
    "options_sentiment": ["put_call_pressure_factor", "call_pressure_factor", "put_pressure_factor",
                          "volume_oi_activity_factor", "unusual_options_activity_factor",
                          "gamma_pressure_factor", "delta_pressure_factor", "iv_skew_factor",
                          "iv_term_structure_factor", "option_sentiment_factor",
                          "latent_options_factor_1", "latent_options_factor_2", "latent_options_factor_3",
                          "latent_options_sentiment_state", "latent_gamma_state"],
    "market_context":  ["market_state_confidence", "spy_regime", "ga_regime_switch"],
    "sector_relative": ["sector_regime"],
    "opening_range":   ["or_position", "or_breakout", "or_level_quality"],
    "premarket_time":  ["premarket_gap_pct", "premarket_rank_score", "time_of_day"],
}
TARGET = os.getenv("PSC_TARGET", "realized_R")        # the FUTURE return (already attached)
N_BINS = int(os.getenv("PSC_N_BINS", "4"))            # future-return bins (causal-state futures)
N_STATES = int(os.getenv("PSC_N_STATES", "8"))        # number of causal market states to seek
N_COMPONENTS = int(os.getenv("PSC_N_COMPONENTS", "4"))  # latent dim for the linear compressors
TRAIN_FRAC = float(os.getenv("PSC_TRAIN_FRAC", "0.7"))
RETAIN_TAU = float(os.getenv("PSC_RETAIN_TAU", "0.80"))  # OOS pred-info retention to approve
SEED = 0


def _sk():
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


# ── data loading (X = ex-ante features, y = FUTURE return) ───────────────────────────
def load_xy(dataset: str = "alpha_entry_dataset") -> Tuple[np.ndarray, np.ndarray, List[str], Any]:
    """Return (X, y, feature_names, df) time-ordered. X are ex-ante features present in the
    dataset; y is the future-return target. Empty arrays if the dataset is missing."""
    rows = rc.load_dataset(dataset)
    if not rows:
        return np.empty((0, 0)), np.empty((0,)), [], None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if "date_tag" in df:
            sort_keys = [k for k in ("date_tag", "timestamp", "symbol") if k in df]
            df = df.sort_values(sort_keys).reset_index(drop=True)
    except Exception:
        df = None
    names = [f for f in rc.ALPHA_ENTRY_FEATURES
             if (df is not None and f in df.columns) or any(f in r for r in rows[:5])]
    if df is not None:
        X = df[names].astype(float).fillna(0.0).to_numpy()
        y = df[TARGET].astype(float).fillna(0.0).to_numpy() if TARGET in df else np.zeros(len(df))
    else:  # pure-python fallback
        X = np.array([[rc.to_f(r.get(f)) for f in names] for r in rows], dtype=float)
        y = np.array([rc.to_f(r.get(TARGET)) for r in rows], dtype=float)
    return X, y, names, df


def _standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0); sd = train.std(axis=0); sd[sd < 1e-9] = 1.0
    out = [(train - mu) / sd] + [(o - mu) / sd for o in others]
    return (out[0] if len(out) == 1 else tuple(out)), mu, sd


def _bin_future(y_train: np.ndarray, *ys: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Discretize the future return into N_BINS quantile bins (edges from TRAIN only)."""
    qs = np.linspace(0, 1, N_BINS + 1)[1:-1]
    edges = np.unique(np.quantile(y_train, qs)) if len(y_train) else np.array([0.0])
    binned = [np.digitize(yy, edges) for yy in (y_train, *ys)]
    return binned, edges


# ── mutual information (predictive-information measurement) ───────────────────────────
def _mi_continuous(Z: np.ndarray, y: np.ndarray) -> float:
    """Σ per-component MI(Z_j ; y) in nats. sklearn when present; histogram fallback."""
    if Z.size == 0:
        return 0.0
    if _sk():
        try:
            from sklearn.feature_selection import mutual_info_regression
            return float(mutual_info_regression(Z, y, random_state=SEED).sum())
        except Exception:
            pass
    return float(sum(_hist_mi(Z[:, j], y) for j in range(Z.shape[1])))


def _hist_mi(a: np.ndarray, b: np.ndarray, bins: int = 16) -> float:
    """Histogram estimate of I(a;b) in nats. Never raises."""
    try:
        c, _, _ = np.histogram2d(a, b, bins=bins)
        pab = c / max(c.sum(), 1e-12)
        pa = pab.sum(1, keepdims=True); pb = pab.sum(0, keepdims=True)
        nz = pab > 0
        return float((pab[nz] * np.log(pab[nz] / (pa @ pb)[nz])).sum())
    except Exception:
        return 0.0


def _discrete_mi(states: np.ndarray, ybin: np.ndarray) -> float:
    """I(state ; future_bin) in BITS from a contingency table."""
    try:
        su = np.unique(states); bu = np.unique(ybin)
        n = len(states)
        if n == 0:
            return 0.0
        mi = 0.0
        for s in su:
            ps = (states == s).mean()
            for b in bu:
                pb = (ybin == b).mean()
                psb = ((states == s) & (ybin == b)).mean()
                if psb > 0 and ps > 0 and pb > 0:
                    mi += psb * math.log2(psb / (ps * pb))
        return float(mi)
    except Exception:
        return 0.0


# ── compressors (each returns Z_train, Z_oos, info dict) ──────────────────────────────
def compress_pca(Xtr, Xte, k=N_COMPONENTS, kind="pca") -> Optional[Tuple]:
    if not _sk():
        # numpy SVD fallback
        U, S, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
        comp = Vt[:k]
        return (Xtr - Xtr.mean(0)) @ comp.T, (Xte - Xtr.mean(0)) @ comp.T, {"impl": "numpy_svd"}
    try:
        from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA
        kmax = min(k, Xtr.shape[1])
        if kind == "sparse_pca":
            m = SparsePCA(n_components=kmax, random_state=SEED, max_iter=50)
        elif kind == "incremental_pca":
            m = IncrementalPCA(n_components=kmax)
        else:
            m = PCA(n_components=kmax, random_state=SEED)
        Ztr = m.fit_transform(Xtr); Zte = m.transform(Xte)
        ev = getattr(m, "explained_variance_ratio_", None)
        return Ztr, Zte, {"impl": kind, "explained_variance": None if ev is None else float(np.sum(ev))}
    except Exception as exc:
        logger.debug("%s failed: %s", kind, exc)
        return None


def compress_pls(Xtr, ytr, Xte, k=N_COMPONENTS) -> Optional[Tuple]:
    """Supervised predictive components — maximizes covariance with the FUTURE return (a
    practical predictive-information / information-bottleneck surrogate). Fit on TRAIN only."""
    if not _sk():
        return None
    try:
        from sklearn.cross_decomposition import PLSRegression
        m = PLSRegression(n_components=min(k, Xtr.shape[1]))
        m.fit(Xtr, ytr)
        return m.transform(Xtr), m.transform(Xte), {"impl": "pls"}
    except Exception as exc:
        logger.debug("pls failed: %s", exc)
        return None


def compress_autoencoder(Xtr, Xte, k=N_COMPONENTS) -> Optional[Tuple]:
    """Compact torch autoencoder bottleneck (optional). Fit on TRAIN; encode both. Deterministic."""
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        d = Xtr.shape[1]
        enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, k))
        dec = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, d))
        ae = nn.Sequential(enc, dec)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
        xt = torch.tensor(Xtr, dtype=torch.float32)
        for _ in range(120):
            opt.zero_grad(); out = ae(xt); loss = ((out - xt) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            Ztr = enc(xt).numpy()
            Zte = enc(torch.tensor(Xte, dtype=torch.float32)).numpy()
        return Ztr, Zte, {"impl": "autoencoder", "final_recon_mse": float(loss.item())}
    except Exception as exc:
        logger.debug("autoencoder skipped/failed: %s", exc)
        return None


# ── the centerpiece: causal-state discovery + quantum compression ────────────────────
def belief_states(Xtr, ybin_tr, Xte, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Conditional future distribution P(future_bin | x) per row — the 'belief state' / minimal
    predictive statistic. Fit a classifier on TRAIN, predict_proba on TRAIN+OOS (no leakage).
    Falls back to k-NN empirical future distributions when sklearn is absent."""
    classes = np.arange(n_bins + 1)  # digitize → 0..len(edges)
    if _sk():
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                                 max_depth=4, random_state=SEED)
            clf.fit(Xtr, ybin_tr)
            def proba(X):
                p = clf.predict_proba(X)
                full = np.zeros((len(X), len(classes)))
                for j, c in enumerate(clf.classes_):
                    full[:, int(c)] = p[:, j]
                return full
            return proba(Xtr), proba(Xte)
        except Exception as exc:
            logger.debug("HGB belief-state failed (%s) — k-NN fallback", exc)
    # k-NN empirical conditional future distribution (pure fallback)
    from numpy.linalg import norm  # noqa
    k = 50
    def knn_proba(Q):
        out = np.zeros((len(Q), len(classes)))
        for i in range(len(Q)):
            d = np.sum((Xtr - Q[i]) ** 2, axis=1)
            idx = np.argpartition(d, min(k, len(d) - 1))[:k]
            for b in ybin_tr[idx]:
                out[i, int(b)] += 1.0
            out[i] /= max(out[i].sum(), 1e-9)
        return out
    return knn_proba(Xtr), knn_proba(Xte)


def quantum_complexity(state_probs: np.ndarray, future_dists: np.ndarray) -> Dict[str, float]:
    """Given P(s) and the conditional future distribution P(·|s) per causal state, compute the
    classical statistical complexity C_μ and the quantum memory C_q via the amplitude-encoded,
    generally non-orthogonal states |ψ_s⟩ = Σ_k √P(k|s)|k⟩ (Elliott et al.). Returns C_μ, C_q,
    D_μ, D_q and the compression ratio. C_q ≤ C_μ by construction."""
    ps = np.clip(state_probs, 1e-12, None); ps = ps / ps.sum()
    C_mu = float(-(ps * np.log2(ps)).sum())
    amps = np.sqrt(np.clip(future_dists, 0.0, None))            # |ψ_s⟩ amplitudes (real, ≥0)
    amps = amps / np.clip(np.linalg.norm(amps, axis=1, keepdims=True), 1e-12, None)
    rho = np.zeros((amps.shape[1], amps.shape[1]))
    for p, a in zip(ps, amps):
        rho += p * np.outer(a, a)                               # ρ = Σ P(s)|ψ_s⟩⟨ψ_s|
    rho = (rho + rho.T) / 2.0
    eig = np.clip(np.linalg.eigvalsh(rho), 1e-12, None); eig = eig / eig.sum()
    C_q = float(-(eig * np.log2(eig)).sum())
    D_mu = float(math.log2(max(1, len(ps))))
    D_q = float(math.log2(max(1, int((eig > 1e-6).sum()))))
    return {"C_mu_statistical_complexity": round(C_mu, 4),
            "C_q_quantum_memory": round(C_q, 4),
            "D_mu_topological": round(D_mu, 4), "D_q_quantum_topological": round(D_q, 4),
            "quantum_compression_ratio": round(C_q / C_mu, 4) if C_mu > 1e-9 else 1.0,
            "memory_saving_bits": round(C_mu - C_q, 4)}


# ── state naming (interpretive, from feature + future signatures) ────────────────────
def name_state(feat_centroid: Dict[str, float], future_mean_R: float, future_std_R: float) -> str:
    """Heuristic label for a discovered causal state from its feature centroid + future return.
    Advisory only — the numeric state id is what the RL consumes."""
    g = lambda k: feat_centroid.get(k, 0.0)
    trend = g("mtf_alignment") + g("ga_regime_switch") + g("ga_ema_ratio") * 5
    vol = g("atr_pct") * 50 + (1 if g("vol_regime") > 0.6 else 0)
    volume = g("rel_volume") + g("catalyst_strength") + g("or_volume_confirm")
    rsi = g("ga_rsi"); breakout = g("or_breakout"); sent = g("ga_sentiment_st")
    pos = future_mean_R > 0.1; neg = future_mean_R < -0.1
    if breakout > 0.2 and neg:
        return "breakout_failure"
    if trend > 0.3 and pos:
        return "momentum_continuation"
    if trend > 0.3 and (neg or rsi > 0.4):
        return "momentum_exhaustion"
    if volume > 0.6 and pos:
        return "institutional_accumulation"
    if volume > 0.6 and neg:
        return "distribution"
    if sent > 0.2 and pos:
        return "news_continuation"
    if sent > 0.2 and neg:
        return "news_fade"
    if vol > 1.0 and future_std_R > 1.6:
        return "volatility_expansion"
    if vol < 0.3 and future_std_R < 1.2:
        return "volatility_contraction"
    if future_std_R > 1.8:
        return "regime_transition"
    return "neutral_state"


def discover_causal_states(Xtr, ytr, ybin_tr, Xte, yte, ybin_te, names: List[str]) -> Dict[str, Any]:
    """Belief states → KMeans causal states → C_μ/C_q + named states + OOS predictive info.
    All maps fit on TRAIN; clustering predicted on OOS; I(state;future) measured OOS (no leak)."""
    bp_tr, bp_te = belief_states(Xtr, ybin_tr, Xte, N_BINS)     # P(future_bin | x)
    # cluster the belief-state simplex into causal states (fit on TRAIN)
    if _sk():
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(N_STATES, len(np.unique(bp_tr, axis=0))), n_init=5, random_state=SEED)
        s_tr = km.fit_predict(bp_tr); s_te = km.predict(bp_te)
        centroids = km.cluster_centers_
        # state_confidence = closeness to assigned centroid (1 = on centroid)
        d_te = np.linalg.norm(bp_te - centroids[s_te], axis=1)
        conf_te = 1.0 / (1.0 + d_te)
    else:
        s_tr, centroids = _kmeans_numpy(bp_tr, min(N_STATES, 8))
        s_te = np.array([int(np.argmin(np.linalg.norm(centroids - r, axis=1))) for r in bp_te])
        conf_te = np.ones(len(s_te)) * 0.5
    used = np.unique(s_tr)
    # steady-state P(s) + conditional future distribution P(future_bin | s) (from TRAIN)
    n_bins_eff = bp_tr.shape[1]
    ps = np.array([(s_tr == s).mean() for s in used])
    fdist = np.array([bp_tr[s_tr == s].mean(axis=0) if (s_tr == s).any()
                      else np.ones(n_bins_eff) / n_bins_eff for s in used])
    q = quantum_complexity(ps, fdist)
    # named states from feature centroids + future-R signature (TRAIN)
    states_named = []
    for j, s in enumerate(used):
        mask = s_tr == s
        cen = {names[i]: float(Xtr[mask, i].mean()) for i in range(len(names))} if mask.any() else {}
        fmean = float(ytr[mask].mean()) if mask.any() else 0.0
        fstd = float(ytr[mask].std()) if mask.any() else 0.0
        states_named.append({"state_id": int(s), "label": name_state(cen, fmean, fstd),
                             "probability": round(float(ps[j]), 4),
                             "future_mean_R": round(fmean, 4), "future_std_R": round(fstd, 4)})
    # OOS predictive information of the discrete causal-state partition
    mi_state_oos = _discrete_mi(s_te, ybin_te)                  # bits
    # future_return_embedding per OOS row = expected future R implied by belief state
    bin_centers = _bin_centers(ytr, ybin_tr, n_bins_eff)
    fre_te = bp_te @ bin_centers
    return {"belief_tr": bp_tr, "belief_te": bp_te, "state_tr": s_tr, "state_te": s_te,
            "confidence_te": conf_te, "future_return_embedding_te": fre_te,
            "n_states_used": int(len(used)), "states": states_named,
            "I_state_future_oos_bits": round(mi_state_oos, 4), **q}


def _bin_centers(y: np.ndarray, ybin: np.ndarray, n: int) -> np.ndarray:
    return np.array([y[ybin == b].mean() if (ybin == b).any() else 0.0 for b in range(n)])


def _kmeans_numpy(X, k, iters=25):
    rng = np.random.RandomState(SEED)
    cen = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(len(cen))])
        if np.allclose(new, cen):
            break
        cen = new
    return lab, cen


# ── downstream predictive check (does the latent predict future returns OOS?) ────────
def decile_spread(signal: np.ndarray, y: np.ndarray) -> float:
    """Mean future return of the top decile minus the bottom decile, ranked by `signal`."""
    if len(signal) < 10:
        return 0.0
    order = np.argsort(signal); n = len(signal) // 10
    return float(y[order[-n:]].mean() - y[order[:n]].mean())


# ── validation / approval gate (mirrors the deployment-gate discipline) ──────────────
def approve(method: str, dim: int, n_raw: int, retain_ratio: float,
            ds_method: float, ds_raw: float, quantum_ok: bool = True) -> Tuple[str, Dict[str, Any]]:
    """A compressed representation is APPROVED only if it (a) preserves OOS predictive info
    (retain_ratio ≥ RETAIN_TAU), (b) compresses dimension (dim < n_raw), (c) gives a downstream
    OOS lift at least as good as the raw-PCA baseline, and (d) (causal_states) achieves quantum
    compression. Fit-IS/eval-OOS ⇒ leakage-free by construction. Returns (recommendation, notes)."""
    notes = {"dim": dim, "n_raw": n_raw, "retain_ratio": round(retain_ratio, 4),
             "downstream_decile_spread": round(ds_method, 4),
             "raw_decile_spread": round(ds_raw, 4), "leakage_safe": True}
    compresses = dim < n_raw
    retains = retain_ratio >= RETAIN_TAU
    lifts = ds_method >= ds_raw - 1e-9 and ds_method > 0
    if not compresses:
        rec = "rejected"; notes["reason"] = "no_dimensionality_reduction"
    elif method == "causal_states" and not quantum_ok:
        rec = "rejected"; notes["reason"] = "no_quantum_compression(C_q>=C_mu)"
    elif retains and lifts:
        rec = "approved"; notes["reason"] = "retains_oos_predictive_info_and_lifts_downstream"
    elif retains or lifts:
        rec = "shadow_only"; notes["reason"] = "partial(retains_xor_lifts)_needs_more_evidence"
    else:
        rec = "rejected"; notes["reason"] = "loses_oos_predictive_info"
    return rec, notes


def compress(dataset: str = "alpha_entry_dataset") -> Dict[str, Any]:
    """Full pass: load X/future-y, time-split, run every available compressor, measure OOS
    predictive information + downstream lift + (causal) quantum complexity, and emit an approval
    per method. Picks the best APPROVED method by OOS predictive-info DENSITY (info per dim)."""
    X, y, names, df = load_xy(dataset)
    if X.size == 0:
        return {"error": f"no data in {dataset}"}
    n = len(X); k = max(1, int(n * TRAIN_FRAC))
    Xtr_raw, Xte_raw = X[:k], X[k:]
    ytr, yte = y[:k], y[k:]
    (Xtr, Xte), mu, sd = _standardize(Xtr_raw, Xte_raw)
    (ybin_tr, ybin_te), edges = _bin_future(ytr, yte)
    n_raw = X.shape[1]

    # baselines on the FULL raw feature set (OOS)
    mi_raw_oos = _mi_continuous(Xte, yte)
    # raw baseline signal = correlation-weighted feature sum (zero-std/neutral cols → 0 weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([np.corrcoef(Xtr[:, j], ytr)[0, 1] if Xtr[:, j].std() > 1e-9 else 0.0
                         for j in range(Xtr.shape[1])])
    corr = np.nan_to_num(corr)
    ds_raw = decile_spread(Xte @ corr, yte)

    methods: Dict[str, Any] = {}
    embeddings: Dict[str, np.ndarray] = {}

    # linear / NN compressors
    runners = [("pca", lambda: compress_pca(Xtr, Xte, kind="pca")),
               ("sparse_pca", lambda: compress_pca(Xtr, Xte, kind="sparse_pca")),
               ("incremental_pca", lambda: compress_pca(Xtr, Xte, kind="incremental_pca")),
               ("pls", lambda: compress_pls(Xtr, ytr, Xte)),
               ("autoencoder", lambda: compress_autoencoder(Xtr, Xte))]
    for mname, fn in runners:
        try:
            res = fn()
        except Exception as exc:
            logger.debug("%s errored: %s", mname, exc); res = None
        if res is None:
            methods[mname] = {"status": "unavailable"}; continue
        Ztr, Zte, info = res
        mi_oos = _mi_continuous(Zte, yte)
        retain = mi_oos / mi_raw_oos if mi_raw_oos > 1e-9 else 0.0
        # downstream signal = first component (or PLS predicted score)
        ds = decile_spread(Zte[:, 0] if Zte.ndim == 2 and Zte.shape[1] else Zte.ravel(), yte)
        rec, notes = approve(mname, Zte.shape[1], n_raw, retain, ds, ds_raw)
        methods[mname] = {"status": "ok", "dim": int(Zte.shape[1]), "oos_pred_info_nats": round(mi_oos, 4),
                          "retain_ratio": round(retain, 4), "recommendation": rec, **info, **notes}
        embeddings[mname] = Zte

    # quantum-inspired causal-state discovery (the centerpiece)
    try:
        cs = discover_causal_states(Xtr, ytr, ybin_tr, Xte, yte, ybin_te, names)
        mi_cs_oos = _mi_continuous(cs["belief_te"], yte)
        retain_cs = mi_cs_oos / mi_raw_oos if mi_raw_oos > 1e-9 else 0.0
        ds_cs = decile_spread(cs["future_return_embedding_te"], yte)
        quantum_ok = cs["C_q_quantum_memory"] < cs["C_mu_statistical_complexity"] - 1e-9
        rec, notes = approve("causal_states", cs["belief_te"].shape[1], n_raw, retain_cs, ds_cs, ds_raw, quantum_ok)
        methods["causal_states"] = {
            "status": "ok", "dim": int(cs["belief_te"].shape[1]),
            "n_states_used": cs["n_states_used"], "oos_pred_info_nats": round(mi_cs_oos, 4),
            "retain_ratio": round(retain_cs, 4),
            "I_state_future_oos_bits": cs["I_state_future_oos_bits"],
            "C_mu_statistical_complexity": cs["C_mu_statistical_complexity"],
            "C_q_quantum_memory": cs["C_q_quantum_memory"],
            "quantum_compression_ratio": cs["quantum_compression_ratio"],
            "memory_saving_bits": cs["memory_saving_bits"],
            "D_mu_topological": cs["D_mu_topological"], "D_q_quantum_topological": cs["D_q_quantum_topological"],
            "states": cs["states"], "recommendation": rec, **notes}
        embeddings["causal_states"] = np.column_stack(
            [cs["belief_te"], cs["state_te"], cs["future_return_embedding_te"], cs["confidence_te"]])
    except Exception as exc:
        logger.warning("causal-state discovery failed: %s", exc)
        methods["causal_states"] = {"status": f"failed:{str(exc)[:80]}"}

    # pick best APPROVED method by predictive-info density (info per latent dim)
    approved = {m: v for m, v in methods.items()
                if v.get("recommendation") == "approved" and v.get("dim", 0) > 0}
    def density(v):
        return v.get("oos_pred_info_nats", 0.0) / max(1, v.get("dim", 1))
    best = max(approved, key=lambda m: density(approved[m])) if approved else None

    # Persist the validated OOS embeddings of the best (or richest) method for review/retraining.
    emb_method = best or ("causal_states" if "causal_states" in embeddings else
                          ("pls" if "pls" in embeddings else ("pca" if "pca" in embeddings else None)))
    emb_path = None
    if emb_method and emb_method in embeddings:
        emb_path = _persist_embeddings(emb_method, embeddings[emb_method], df, k, yte)

    report = {
        "generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(), "dataset": dataset,
        "samples": n, "train": k, "oos": n - k, "n_raw_features": n_raw,
        "future_target": TARGET, "n_future_bins": int(N_BINS),
        "raw_oos_pred_info_nats": round(mi_raw_oos, 4), "raw_decile_spread": round(ds_raw, 4),
        "feature_groups_present": {g: [c for c in cols if c in names] for g, cols in FEATURE_GROUPS.items()},
        "methods": methods, "best_approved_method": best,
        "embeddings_method_persisted": emb_method, "embeddings_path": emb_path,
        "embedding_columns": PSC_FEATURES,
        "predictive_information_score": round(methods.get(best, {}).get("oos_pred_info_nats", 0.0), 4) if best else 0.0,
        "approved": best is not None,
        "leakage_safe": True, "fit_policy": "all maps fit IN-SAMPLE; predictive info measured OUT-OF-SAMPLE",
    }
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "predictive_state_compression.json", report)
    return report


# ── RL-facing embedding columns + persistence/transform/embed hooks ──────────────────
# The RL consumes (per the brief):  Raw Features + Approved Alpha Features + these Predictive
# State Embeddings — treated as HIGH-PRIORITY features. They are wired into the RL observation
# only when RL_PSC_FEATURES=1 (default off) AND a representation has been APPROVED, preserving
# the validation discipline (no unvalidated state enters live behavior).
PSC_FEATURES = ["predictive_state_1", "predictive_state_2", "predictive_state_3",
                "latent_market_state", "future_return_embedding", "state_confidence",
                "predictive_information_score"]


def _persist_embeddings(method: str, Z: np.ndarray, df, k: int, yte: np.ndarray) -> Optional[str]:
    """Write OOS embeddings (+ join keys) to data/rl_training/<dataset>_psc_embeddings.parquet."""
    try:
        import pandas as pd
        Z = np.atleast_2d(Z)
        cols = {}
        # first up-to-3 latent coords → predictive_state_1..3
        for i in range(min(3, Z.shape[1])):
            cols[f"predictive_state_{i+1}"] = Z[:, i]
        if method == "causal_states":
            # causal layout: [belief(nbins) | state_id | future_return_embedding | confidence]
            cols["latent_market_state"] = Z[:, -3]
            cols["future_return_embedding"] = Z[:, -2]
            cols["state_confidence"] = Z[:, -1]
        else:
            cols["latent_market_state"] = np.zeros(len(Z))
            cols["future_return_embedding"] = Z[:, 0]
            cols["state_confidence"] = 0.5 * np.ones(len(Z))
        out = pd.DataFrame(cols)
        if df is not None:
            tail = df.iloc[k:].reset_index(drop=True)
            for key in ("date_tag", "symbol"):
                if key in tail:
                    out[key] = tail[key].values
        out["realized_R"] = yte
        out["psc_method"] = method
        p = rc.RL_DATA_DIR / "alpha_entry_psc_embeddings.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p, index=False)
        return str(p)
    except Exception as exc:
        logger.debug("persist embeddings failed: %s", exc)
        return None


def transform_all(dataset: str = "alpha_entry_dataset", method: str = "pls") -> Optional[str]:
    """Leakage-safe FULL-coverage embeddings for retraining: fit the chosen approved compressor
    on the IN-SAMPLE slice, then transform EVERY row (IS maps applied to all → no look-ahead).
    Merges the 7 PSC_FEATURES columns into a sibling parquet `<dataset>_psc.parquet` that the RL
    can train on with RL_PSC_FEATURES=1. Returns the path (or None). Causal-state method emits the
    belief→future_return_embedding + state id + confidence; linear methods emit components."""
    X, y, names, df = load_xy(dataset)
    if X.size == 0 or df is None:
        return None
    n = len(X); k = max(1, int(n * TRAIN_FRAC))
    (Xtr, Xall), _, _ = _standardize(X[:k], X)
    (ybin_tr, _), _ = _bin_future(y[:k], y)
    cols: Dict[str, Any] = {f: np.zeros(n) for f in rc.PSC_FEATURES}
    try:
        if method == "causal_states":
            cs = discover_causal_states(Xtr, y[:k], ybin_tr, Xall, y, np.digitize(y, _bin_future(y[:k])[1]),
                                        names)
            bp = cs["belief_te"]
            for i in range(min(3, bp.shape[1])):
                cols[f"predictive_state_{i+1}"] = bp[:, i]
            cols["latent_market_state"] = cs["state_te"].astype(float)
            cols["future_return_embedding"] = cs["future_return_embedding_te"]
            cols["state_confidence"] = cs["confidence_te"]
        else:
            res = compress_pls(Xtr, y[:k], Xall) if method == "pls" else compress_pca(Xtr, Xall, kind=method)
            if res is None:
                return None
            _, Zall, _ = res
            for i in range(min(3, Zall.shape[1])):
                cols[f"predictive_state_{i+1}"] = Zall[:, i]
            cols["future_return_embedding"] = Zall[:, 0]
            cols["state_confidence"] = 0.5 * np.ones(n)
    except Exception as exc:
        logger.warning("transform_all(%s) failed: %s", method, exc)
        return None
    import pandas as pd
    out = df.copy()
    for f in rc.PSC_FEATURES:
        out[f] = cols[f]
    p = rc.RL_DATA_DIR / f"{dataset}_psc.parquet"
    out.to_parquet(p, index=False)
    logger.info("full PSC embeddings (%s) merged → %s", method, p)
    return str(p)


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Quantum-inspired predictive state compression (Elliott et al. 2020)")
    ap.add_argument("--dataset", default="alpha_entry_dataset")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--emit-full-embeddings", metavar="METHOD", default=None,
                    help="fit-IS/transform-all into <dataset>_psc.parquet (e.g. pls | causal_states)")
    args = ap.parse_args()
    r = compress(args.dataset)
    if args.emit_full_embeddings:
        p = transform_all(args.dataset, args.emit_full_embeddings)
        print(f"full PSC embeddings ({args.emit_full_embeddings}) → {p}")
    if "error" in r:
        print(r["error"]); raise SystemExit(0)
    if args.json_only:
        import json; print(json.dumps(r, indent=2, default=str)); return
    print("=" * 80)
    print("  PREDICTIVE STATE COMPRESSION  —  quantum-inspired (Elliott et al. 2020)")
    print("=" * 80)
    print(f"  dataset={r['dataset']}  samples={r['samples']}  IS={r['train']} OOS={r['oos']}  "
          f"raw_features={r['n_raw_features']}  future={r['future_target']}")
    print(f"  raw OOS predictive-info={r['raw_oos_pred_info_nats']} nats  raw decile-spread={r['raw_decile_spread']}")
    print(f"\n  {'method':18s}{'dim':>5}{'oos_MI':>9}{'retain':>8}{'decile':>9}  recommendation")
    for m, v in r["methods"].items():
        if v.get("status") != "ok":
            print(f"  {m:18s}{'--':>5}{'--':>9}{'--':>8}{'--':>9}  {v.get('status')}"); continue
        print(f"  {m:18s}{v.get('dim',0):>5}{v.get('oos_pred_info_nats',0):>9}{v.get('retain_ratio',0):>8}"
              f"{v.get('downstream_decile_spread',0):>9}  {v.get('recommendation')}")
    cs = r["methods"].get("causal_states", {})
    if cs.get("status") == "ok":
        print(f"\n  QUANTUM COMPRESSION (causal states): C_mu={cs['C_mu_statistical_complexity']} bits  "
              f"C_q={cs['C_q_quantum_memory']} bits  ratio={cs['quantum_compression_ratio']}  "
              f"saving={cs['memory_saving_bits']} bits  (D_mu={cs['D_mu_topological']} D_q={cs['D_q_quantum_topological']})")
        print(f"  discovered {cs['n_states_used']} causal market states  "
              f"I(state;future)_oos={cs['I_state_future_oos_bits']} bits:")
        for s in cs.get("states", []):
            print(f"    state {s['state_id']:>2}  {s['label']:24s}  P={s['probability']:.3f}  "
                  f"future_R mean={s['future_mean_R']:+.3f} std={s['future_std_R']:.3f}")
    print(f"\n  BEST APPROVED: {r['best_approved_method']}  predictive_information_score={r['predictive_information_score']}")
    print(f"  embeddings persisted ({r['embeddings_method_persisted']}): {r['embeddings_path']}")
    print(f"  approved={r['approved']}  leakage_safe={r['leakage_safe']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
