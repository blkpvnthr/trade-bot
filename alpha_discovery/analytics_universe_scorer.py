#!/usr/bin/env python3
"""
alpha_discovery/analytics_universe_scorer.py — Advanced Analytics universe,
risk, and correlation layer (§4, §7, §8, §9).

Consumes the *batch* Alpha Vantage Advanced Analytics result (one request over
the whole candidate universe + benchmarks) and produces:

    §4  per-symbol universe ranking inputs + a composite ``universe_score``;
    §7  advisory risk flags (drawdown / volatility / crowding / instability);
    §8  correlation + covariance matrices and correlation clusters;
    §9  CSV reports under reports/analytics/ and reports/risk/.

CRITICAL SAFETY CONTRACT (per CLAUDE.md): everything here is ADVISORY. The risk
flags and ranks are inputs to selection / sizing — they NEVER override the hard
risk controls (max daily loss, kill switches, exits, cash buffers). Pure,
neutral on missing data, never raises.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

from alpha_discovery.analytics_feature_builder import (
    build_fixed_window_features,
    build_sliding_window_features,
    relationship_value,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_REPORT_DIR = PROJECT_ROOT / "reports" / "analytics"
RISK_REPORT_DIR = PROJECT_ROOT / "reports" / "risk"


# =============================================================================
# Advisory risk thresholds (env-tunable defaults; never hard controls)
# =============================================================================

DEFAULT_THRESHOLDS = {
    "max_drawdown_risk": -0.20,        # drawdown worse than -20% -> flag
    "volatility_expansion_risk": 0.30,  # rolling vol-expansion score above this -> flag
    "correlation_crowding": 0.85,       # |corr to SPY| above this -> flag
    "unstable_relationship": 0.50,      # relationship-shift score above this -> flag
    "high_covariance_to_market": 0.0010,  # covariance to SPY above this -> flag
    "cluster_correlation": 0.70,        # |corr| at/above this groups two symbols
}


def _to_f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f) else f


# =============================================================================
# §7 Risk flags (advisory)
# =============================================================================

def compute_risk_flags(
    fixed_feats: dict[str, Any],
    rolling_feats: dict[str, Any],
    thresholds: Optional[dict[str, float]] = None,
) -> dict[str, int]:
    """Advisory risk flags for one symbol. 1 = flagged. Never a hard block."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    mdd = _to_f(fixed_feats.get("fixed_max_drawdown"))
    vol_exp = _to_f(rolling_feats.get("rolling_volatility_slope"))
    vol_exp_score = max(0.0, (vol_exp or 0.0) * 50.0)
    corr_spy = _to_f(fixed_feats.get("fixed_correlation_to_spy"))
    if corr_spy is None:
        corr_spy = _to_f(rolling_feats.get("rolling_correlation_to_spy"))
    shift = _to_f(rolling_feats.get("rolling_relationship_shift_score"))
    cov_spy = _to_f(fixed_feats.get("fixed_covariance_to_spy"))
    return {
        "max_drawdown_risk_flag": int(mdd is not None and mdd <= t["max_drawdown_risk"]),
        "volatility_expansion_risk_flag": int(vol_exp_score >= t["volatility_expansion_risk"]),
        "correlation_crowding_flag": int(corr_spy is not None and abs(corr_spy) >= t["correlation_crowding"]),
        "unstable_relationship_flag": int(shift is not None and shift >= t["unstable_relationship"]),
        "high_covariance_to_market_flag": int(cov_spy is not None and cov_spy >= t["high_covariance_to_market"]),
    }


# =============================================================================
# §4 Universe ranking
# =============================================================================

def _rank(values: dict[str, Optional[float]], *, descending: bool, use_abs: bool = False) -> dict[str, int]:
    """Dense rank (1 = best) over the present numeric values. Absent -> rank 0."""
    present = {s: (abs(v) if use_abs else v) for s, v in values.items() if v is not None}
    if not present:
        return {s: 0 for s in values}
    order = sorted(present.items(), key=lambda kv: kv[1], reverse=descending)
    ranks = {s: i + 1 for i, (s, _) in enumerate(order)}
    return {s: ranks.get(s, 0) for s in values}


def _minmax_score(values: dict[str, Optional[float]], *, higher_better: bool, use_abs: bool = False) -> dict[str, float]:
    """Normalize present values to [0,1] (1 = best). Absent -> 0.5 (neutral)."""
    present = {s: (abs(v) if use_abs else v) for s, v in values.items() if v is not None}
    if not present:
        return {s: 0.5 for s in values}
    lo, hi = min(present.values()), max(present.values())
    rng = hi - lo
    out: dict[str, float] = {}
    for s in values:
        v = present.get(s)
        if v is None or rng == 0:
            out[s] = 0.5
        else:
            n = (v - lo) / rng
            out[s] = n if higher_better else (1.0 - n)
    return out


def rank_universe(
    symbols: Sequence[str],
    fixed_result: Optional[dict[str, Any]],
    sliding_result: Optional[dict[str, Any]] = None,
    spy: str = "SPY",
    qqq: str = "QQQ",
    sector: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Score + rank candidate ``symbols`` by their analytics statistical edge.

    Returns one row per symbol with the §4 ranking inputs, a composite
    ``universe_score`` in [0,1] (higher = preferred), and advisory risk flags.
    Benchmarks (SPY/QQQ/sector) are excluded from the ranked output.
    """
    cand = [s.upper() for s in symbols if s.upper() not in {spy.upper(), qqq.upper(), (sector or "").upper()}]
    if not cand:
        return []

    fixed = {s: build_fixed_window_features(s, fixed_result, spy, qqq, sector) for s in cand}
    rolling = {s: build_sliding_window_features(s, sliding_result, spy, qqq, sector) for s in cand}

    cum = {s: _to_f(fixed[s].get("fixed_cumulative_return")) for s in cand}
    std = {s: _to_f(fixed[s].get("fixed_return_stddev")) for s in cand}
    mdd = {s: _to_f(fixed[s].get("fixed_max_drawdown")) for s in cand}
    ac = {s: _to_f(fixed[s].get("fixed_autocorrelation_lag1")) for s in cand}
    corr_spy = {s: _to_f(fixed[s].get("fixed_correlation_to_spy")) for s in cand}
    shift = {s: _to_f(rolling[s].get("rolling_relationship_shift_score")) for s in cand}
    ram = {s: (_div(cum[s], std[s])) for s in cand}  # risk-adjusted momentum

    cum_rank = _rank(cum, descending=True)
    std_rank = _rank(std, descending=False)
    mdd_rank = _rank(mdd, descending=True)         # closer to 0 (larger) = better
    ac_rank = _rank(ac, descending=True, use_abs=True)
    corr_rank = _rank(corr_spy, descending=False, use_abs=True)  # lower |corr| = better
    shift_rank = _rank(shift, descending=True)
    ram_rank = _rank(ram, descending=True)

    # Composite score: reward return + risk-adjusted momentum + independence;
    # penalize drawdown depth + correlation crowding.
    s_cum = _minmax_score(cum, higher_better=True)
    s_ram = _minmax_score(ram, higher_better=True)
    s_mdd = _minmax_score(mdd, higher_better=True)            # less-negative is higher
    s_indep = _minmax_score(corr_spy, higher_better=False, use_abs=True)
    s_std = _minmax_score(std, higher_better=False)

    rows: list[dict[str, Any]] = []
    for s in cand:
        flags = compute_risk_flags(fixed[s], rolling[s])
        score = round(
            0.30 * s_cum[s] + 0.25 * s_ram[s] + 0.20 * s_mdd[s]
            + 0.15 * s_indep[s] + 0.10 * s_std[s], 4)
        # advisory down-weight when any risk flag trips (never zeroes it out)
        n_flags = sum(flags.values())
        score = round(score * (1.0 - 0.10 * min(n_flags, 3)), 4)
        rows.append({
            "symbol": s,
            "universe_score": score,
            "cumulative_return": cum[s],
            "stddev": std[s],
            "max_drawdown": mdd[s],
            "autocorrelation": ac[s],
            "correlation_to_spy": corr_spy[s],
            "risk_adjusted_momentum": ram[s],
            "relationship_shift": shift[s],
            "cumulative_return_rank": cum_rank[s],
            "stddev_rank": std_rank[s],
            "max_drawdown_rank": mdd_rank[s],
            "autocorrelation_rank": ac_rank[s],
            "correlation_to_spy_rank": corr_rank[s],
            "correlation_shift_rank": shift_rank[s],
            "risk_adjusted_momentum_rank": ram_rank[s],
            **flags,
        })
    rows.sort(key=lambda r: r["universe_score"], reverse=True)
    for i, r in enumerate(rows):
        r["universe_rank"] = i + 1
    return rows


def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0.0):
        return None
    return a / b


# =============================================================================
# §8 Correlation / covariance matrices + clusters
# =============================================================================

def extract_matrix(result: Optional[dict[str, Any]], calc: str) -> Optional[dict[str, Any]]:
    """Pull a CORRELATION / COVARIANCE matrix from a result's calculations."""
    if not isinstance(result, dict):
        return None
    node = (result.get("calculations") or {}).get(calc.upper())
    if isinstance(node, dict) and isinstance(node.get("index"), list) and isinstance(node.get("matrix"), list):
        return node
    return None


def correlation_clusters(matrix: Optional[dict[str, Any]], threshold: float = 0.70) -> list[list[str]]:
    """Greedy union-find clustering: symbols linked when |corr| >= threshold."""
    if not matrix:
        return []
    index = [str(s).upper() for s in matrix.get("index", [])]
    mat = matrix.get("matrix", [])
    n = len(index)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            try:
                v = mat[i][j]
            except (IndexError, TypeError):
                continue
            if v is not None and abs(float(v)) >= threshold:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(index[i])
    # only return non-trivial clusters first, then singletons
    clusters = sorted(groups.values(), key=len, reverse=True)
    return clusters


# =============================================================================
# §9 Reporting
# =============================================================================

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
    except Exception:  # pragma: no cover - reporting must never break trading
        logger.warning("analytics report write failed: %s", path)


def _matrix_to_rows(matrix: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return []
    index = [str(s).upper() for s in matrix.get("index", [])]
    mat = matrix.get("matrix", [])
    rows: list[dict[str, Any]] = []
    for i, sym in enumerate(index):
        row: dict[str, Any] = {"symbol": sym}
        for j, col in enumerate(index):
            try:
                row[col] = mat[i][j]
            except (IndexError, TypeError):
                row[col] = None
        rows.append(row)
    return rows


def write_universe_report(rows: list[dict[str, Any]], report_dir: Optional[Path] = None) -> None:
    d = Path(report_dir) if report_dir else ANALYTICS_REPORT_DIR
    _write_csv(d / "latest_universe_analytics_scores.csv", rows)


def write_fixed_window_report(rows: list[dict[str, Any]], report_dir: Optional[Path] = None) -> None:
    d = Path(report_dir) if report_dir else ANALYTICS_REPORT_DIR
    _write_csv(d / "latest_fixed_window_analytics.csv", rows)


def write_sliding_window_report(rows: list[dict[str, Any]], report_dir: Optional[Path] = None) -> None:
    d = Path(report_dir) if report_dir else ANALYTICS_REPORT_DIR
    _write_csv(d / "latest_sliding_window_analytics.csv", rows)


def write_relationship_shift_report(rows: list[dict[str, Any]], report_dir: Optional[Path] = None) -> None:
    d = Path(report_dir) if report_dir else ANALYTICS_REPORT_DIR
    shift_rows = [{"symbol": r["symbol"],
                   "relationship_shift": r.get("relationship_shift"),
                   "correlation_to_spy": r.get("correlation_to_spy"),
                   "unstable_relationship_flag": r.get("unstable_relationship_flag", 0)}
                  for r in rows]
    _write_csv(d / "latest_relationship_shift_report.csv", shift_rows)


def write_correlation_reports(
    fixed_result: Optional[dict[str, Any]],
    threshold: float = 0.70,
    report_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Write correlation matrix, covariance matrix, and clusters (§8)."""
    d = Path(report_dir) if report_dir else RISK_REPORT_DIR
    corr = extract_matrix(fixed_result, "CORRELATION")
    cov = extract_matrix(fixed_result, "COVARIANCE")
    _write_csv(d / "latest_correlation_matrix.csv", _matrix_to_rows(corr))
    _write_csv(d / "latest_covariance_matrix.csv", _matrix_to_rows(cov))
    clusters = correlation_clusters(corr, threshold)
    cluster_rows = [{"cluster_id": i + 1, "size": len(c), "symbols": ",".join(c)}
                    for i, c in enumerate(clusters)]
    _write_csv(d / "latest_correlation_clusters.csv", cluster_rows)
    return {"correlation": corr, "covariance": cov, "clusters": clusters}


# =============================================================================
# Driver
# =============================================================================

DEFAULT_UNIVERSE_CALCS = (
    "MEAN", "MEDIAN", "STDDEV", "VARIANCE", "CUMULATIVE_RETURN", "MAX_DRAWDOWN",
    "MIN", "MAX", "AUTOCORRELATION(lag=1)", "CORRELATION(method=PEARSON)", "COVARIANCE",
)


def run_universe_analytics(
    symbols: Sequence[str],
    analytics_client: Optional[Any] = None,
    range_fixed: str = "60day",
    interval_fixed: str = "DAILY",
    range_sliding: str = "3month",
    interval_sliding: str = "5min",
    window_size: int = 20,
    spy: str = "SPY",
    qqq: str = "QQQ",
    sector: Optional[str] = None,
    write_reports: bool = True,
    report_dir: Optional[Path] = None,
    risk_report_dir: Optional[Path] = None,
    local_series: Optional[dict[str, Sequence[float]]] = None,
) -> dict[str, Any]:
    """Fetch batch analytics for the universe and produce ranks + risk + matrices.

    One fixed-window request and one sliding-window request cover the whole
    universe plus benchmarks (the efficient path vs per-symbol calls). Returns a
    dict with ``universe`` rows, ``correlation`` artifacts, and the raw results.
    Advisory-safe end to end.
    """
    syms = [s.upper() for s in symbols]
    request_syms = list(dict.fromkeys(syms + [spy, qqq] + ([sector] if sector else [])))

    try:
        if analytics_client is None:
            from data_sources.alpha_vantage_analytics_client import get_default_analytics_client
            analytics_client = get_default_analytics_client()
    except Exception:  # pragma: no cover - import guard
        analytics_client = None

    fixed_result = sliding_result = None
    if analytics_client is not None:
        try:
            fixed_result = analytics_client.get_fixed_window(
                request_syms, list(DEFAULT_UNIVERSE_CALCS), range_=range_fixed,
                interval=interval_fixed, local_series=local_series)
        except Exception as exc:
            logger.warning("fixed-window universe analytics failed: %s", exc)
        try:
            sliding_result = analytics_client.get_sliding_window(
                request_syms, list(DEFAULT_UNIVERSE_CALCS), range_=range_sliding,
                interval=interval_sliding, window_size=window_size, local_series=local_series)
        except Exception as exc:
            logger.warning("sliding-window universe analytics failed: %s", exc)

    rows = rank_universe(syms, fixed_result, sliding_result, spy, qqq, sector)

    fixed_feature_rows = [
        {"symbol": s, **build_fixed_window_features(s, fixed_result, spy, qqq, sector)}
        for s in syms]
    sliding_feature_rows = [
        {"symbol": s, **build_sliding_window_features(s, sliding_result, spy, qqq, sector)}
        for s in syms]

    corr_artifacts: dict[str, Any] = {}
    if write_reports:
        write_universe_report(rows, report_dir)
        write_fixed_window_report(fixed_feature_rows, report_dir)
        write_sliding_window_report(sliding_feature_rows, report_dir)
        write_relationship_shift_report(rows, report_dir)
        corr_artifacts = write_correlation_reports(
            fixed_result, DEFAULT_THRESHOLDS["cluster_correlation"], risk_report_dir)

    return {
        "universe": rows,
        "fixed_features": fixed_feature_rows,
        "sliding_features": sliding_feature_rows,
        "correlation": corr_artifacts,
        "fixed_result": fixed_result,
        "sliding_result": sliding_result,
    }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["AAPL", "MSFT", "NVDA", "TSLA"]
    out = run_universe_analytics(syms)
    print(json.dumps(out["universe"], indent=2, default=str))
