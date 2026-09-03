#!/usr/bin/env python3
"""
alpha_discovery/exante_feature_research.py — honest, leakage-resistant discovery of
genuinely predictive EX-ANTE intraday entry features.

Context: the falsification audit proved the alpha_entry options pipeline's only
"predictive" feature (probability_of_profit) was leakage; every legitimate feature had
near-zero out-of-sample signal. This module sources candidate ex-ante features from the
CLEAN intraday microstructure dataset (system-data/<date>/<date>_intraday_training_dataset.csv)
— point-in-time technicals (features at time t) vs strictly-forward returns (t→t+k) — and
measures which truly predict forward returns, with discipline designed to NOT manufacture a
false edge:

  * Target = strictly-forward return (future_return_15m primary; 5m/30m secondary).
  * Unit of significance = (symbol, day) Information Coefficient (Spearman rank corr of the
    feature vs the forward return over that symbol's intraday series). Aggregating ICs across
    symbol-days defeats the t-stat inflation that overlapping 15m-forward 1m-bar labels would
    otherwise cause (each symbol-day contributes ONE ~independent IC).
  * Out-of-sample: ICs computed separately on IS days (earlier) and OOS days (later); a feature
    is only credible if its OOS IC keeps the same sign and magnitude.
  * Multiple testing: ~120 candidates → require a high |t| bar + Benjamini-Hochberg FDR.
  * LEAKAGE TRIPWIRE: any feature with |mean IC| > IC_TRIPWIRE (default 0.15) is FLAGGED as
    "suspiciously high — audit for leakage", NOT auto-approved. (Real intraday 15m-forward ICs
    are small, ~0.02–0.08; anything near PoP's 0.65 is a red flag.)

Outputs:
  reports/exante/exante_feature_validation.csv   per-feature IC stats (all candidates)
  reports/exante/exante_approved_features.json    the approved registry (+ flagged list)

Read-only / offline. No Alpaca, no orders.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTRADAY_GLOB = str(PROJECT_ROOT / "system-data" / "*" / "*_intraday_training_dataset.csv")
OUT_DIR = PROJECT_ROOT / "reports" / "exante"

PRIMARY_TARGET = os.getenv("EXANTE_TARGET", "future_return_15m")
SECONDARY_TARGETS = ("future_return_5m", "future_return_30m")
IC_TRIPWIRE = float(os.getenv("EXANTE_IC_TRIPWIRE", "0.15"))
MIN_T = float(os.getenv("EXANTE_MIN_TSTAT", "3.0"))
MIN_ABS_IC = float(os.getenv("EXANTE_MIN_ABS_IC", "0.02"))
MIN_SIGN_CONSISTENCY = float(os.getenv("EXANTE_MIN_SIGN_CONSISTENCY", "0.60"))
MIN_SYMBOL_DAY_OBS = int(os.getenv("EXANTE_MIN_OBS", "40"))   # min bars per symbol-day to count

# Columns that are LABELS or otherwise forward-looking — never candidate features.
_LABEL_PREFIXES = ("future_", )
_LABEL_EXACT = {
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    "exit_label", "exit_label_id", "prob_hold_training_target",
    "giveback_risk_training_target", "reversal_risk_training_target",
    "expected_value_next_5m", "expected_value_next_15m", "expected_value_next_30m",
    "rolling_peak_close_30", "distance_from_recent_peak", "profit_giveback_from_peak",
    "max_adverse_excursion_15m", "max_favorable_excursion_15m",
    "cum_vol", "cum_pv",  # raw accumulators, not features
}


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_SYMBOL_DAY_OBS:
        return None
    xr = _rankdata(x[m]); yr = _rankdata(y[m])
    if xr.std() == 0 or yr.std() == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def load_intraday() -> "Any":
    import pandas as pd
    files = sorted(f for f in glob.glob(INTRADAY_GLOB) if os.path.getsize(f) > 1000)
    if not files:
        return None, []
    frames = []
    for f in files:
        day = Path(f).name.split("_")[0]
        df = pd.read_csv(f)
        df["_day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    days = sorted(df["_day"].unique())
    return df, days


# Non-stationary price LEVELS and resampled multi-timeframe features are excluded by default:
# levels correlate with forward price only via spurious non-stationarity / intraday seasonality,
# and the resampled Xm_* features carry look-ahead risk (the timeframe bar containing minute t
# includes minutes after t). Set EXANTE_STATIONARY_ONLY=0 to include them (research only).
_STATIONARY_ONLY = os.getenv("EXANTE_STATIONARY_ONLY", "1") == "1"
_LEVEL_PREFIXES = ("sma_", "ema_", "volume_sma_", "rolling_high_", "rolling_low_")
_LEVEL_EXACT = {"vwap", "true_range", "atr_14"}
_MULTITF_PREFIXES = ("5m_", "15m_", "1h_", "4h_")


def _is_excluded_nonstationary(c: str) -> bool:
    if c in _LEVEL_EXACT:
        return True
    if any(c.startswith(p) for p in _LEVEL_PREFIXES):
        return True
    if any(c.startswith(p) for p in _MULTITF_PREFIXES):
        return True
    return False


def candidate_features(df) -> list[str]:
    feats = []
    for c in df.columns:
        if c in _LABEL_EXACT or c == "_day":
            continue
        if any(c.startswith(p) for p in _LABEL_PREFIXES):
            continue
        if df[c].dtype.kind not in "fiu":
            continue
        if _STATIONARY_ONLY and _is_excluded_nonstationary(c):
            continue
        feats.append(c)
    return feats


def per_symbol_day_ic(df, feature: str, target: str) -> list[float]:
    ics = []
    for (_sym, _day), g in df.groupby(["symbol", "_day"], sort=False):
        ic = _spearman(g[feature].to_numpy(dtype=float), g[target].to_numpy(dtype=float))
        if ic is not None and math.isfinite(ic):
            ics.append(ic)
    return ics


def validate_feature(df_is, df_oos, feature: str, target: str) -> dict[str, Any]:
    ic_is = per_symbol_day_ic(df_is, feature, target)
    ic_oos = per_symbol_day_ic(df_oos, feature, target)
    def stats(ics):
        if len(ics) < 3:
            return {"n": len(ics), "mean_ic": 0.0, "t": 0.0, "sign_consistency": 0.0}
        arr = np.array(ics)
        mean = float(arr.mean()); se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
        t = mean / se if se > 0 else 0.0
        sign = float(np.mean(np.sign(arr) == np.sign(mean)))
        return {"n": len(ics), "mean_ic": round(mean, 5), "t": round(t, 3),
                "sign_consistency": round(sign, 3)}
    s_is, s_oos = stats(ic_is), stats(ic_oos)
    return {
        "feature": feature, "target": target,
        "is_n": s_is["n"], "is_mean_ic": s_is["mean_ic"], "is_t": s_is["t"],
        "oos_n": s_oos["n"], "oos_mean_ic": s_oos["mean_ic"], "oos_t": s_oos["t"],
        "oos_sign_consistency": s_oos["sign_consistency"],
        "sign_stable": int(np.sign(s_is["mean_ic"]) == np.sign(s_oos["mean_ic"]) and s_oos["mean_ic"] != 0),
    }


def _bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    n = len(pvals)
    order = np.argsort(pvals)
    passed = [False] * n
    thresh = 0.0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / n) * q:
            thresh = pvals[idx]
    for i, p in enumerate(pvals):
        passed[i] = p <= thresh
    return passed


def _t_to_p(t: float, dof: int) -> float:
    # two-sided p from t via normal approx (dof large here) — conservative enough for ranking
    from math import erf, sqrt
    z = abs(t)
    return max(1e-12, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))


def run(target: str = PRIMARY_TARGET) -> dict[str, Any]:
    import pandas as pd
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, days = load_intraday()
    if df is None or not days:
        return {"error": "no intraday dataset found", "glob": INTRADAY_GLOB}
    if target not in df.columns:
        return {"error": f"target {target} missing"}
    # temporal IS/OOS split by day
    cut = max(1, int(len(days) * 0.7))
    is_days, oos_days = set(days[:cut]), set(days[cut:]) or {days[-1]}
    df = df[np.isfinite(df[target])]
    df_is = df[df["_day"].isin(is_days)]
    df_oos = df[df["_day"].isin(oos_days)]
    feats = candidate_features(df)

    rows = []
    for f in feats:
        try:
            rows.append(validate_feature(df_is, df_oos, f, target))
        except Exception:
            continue

    # multiple-testing on OOS t-stats
    pvals = [_t_to_p(r["oos_t"], max(1, r["oos_n"] - 1)) for r in rows]
    fdr_pass = _bh_fdr(pvals, q=0.10)
    for r, p, fp in zip(rows, pvals, fdr_pass):
        r["oos_p"] = round(p, 6)
        r["fdr_pass"] = int(fp)
        # classification
        absic = abs(r["oos_mean_ic"])
        flagged_leak = absic > IC_TRIPWIRE
        approved = bool(
            not flagged_leak
            and abs(r["oos_t"]) >= MIN_T
            and absic >= MIN_ABS_IC
            and r["oos_sign_consistency"] >= MIN_SIGN_CONSISTENCY
            and r["sign_stable"] == 1
            and fp
        )
        r["leakage_flag"] = int(flagged_leak)
        r["approved"] = int(approved)

    rows.sort(key=lambda r: abs(r["oos_mean_ic"]), reverse=True)
    val_df = pd.DataFrame(rows)
    val_df.to_csv(OUT_DIR / "exante_feature_validation.csv", index=False)

    approved = [r for r in rows if r["approved"]]
    flagged = [r for r in rows if r["leakage_flag"]]
    registry = {
        "generated": "2026-06-01", "target": target,
        "is_days": sorted(is_days), "oos_days": sorted(oos_days),
        "n_candidates": len(rows),
        "thresholds": {"min_t": MIN_T, "min_abs_ic": MIN_ABS_IC,
                       "min_sign_consistency": MIN_SIGN_CONSISTENCY, "ic_tripwire": IC_TRIPWIRE,
                       "fdr_q": 0.10},
        "n_approved": len(approved), "n_leakage_flagged": len(flagged),
        "approved_features": [r["feature"] for r in approved],
        "approved_detail": approved,
        "leakage_flagged": [{"feature": r["feature"], "oos_mean_ic": r["oos_mean_ic"]} for r in flagged],
        "top10_by_oos_ic": [{"feature": r["feature"], "oos_mean_ic": r["oos_mean_ic"],
                             "oos_t": r["oos_t"], "sign_stable": r["sign_stable"]} for r in rows[:10]],
    }
    (OUT_DIR / "exante_approved_features.json").write_text(json.dumps(registry, indent=2))
    return registry


if __name__ == "__main__":
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else PRIMARY_TARGET
    reg = run(tgt)
    print(json.dumps({k: reg.get(k) for k in
                      ("target", "n_candidates", "n_approved", "n_leakage_flagged",
                       "approved_features", "top10_by_oos_ic")}, indent=2, default=str))
