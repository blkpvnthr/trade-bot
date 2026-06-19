"""Strategy independence diagnostic (population-phase, NOT an RL feature).

Answers a relational question the per-strategy axes can't: is a newly added
strategy a genuinely new edge, or a correlated clone of one you already have?
Five strategies at 0.95 correlation are still one edge.

This module does NOT touch the RL observation space, reward, PPO model, or risk
engine. It is a read-only diagnostic over already-computed strategy trades.

Three pieces:
  1. strategy_family   -- coarse tag (trend/breakout/mean_reversion/flow/...).
  2. correlation matrix -- pairwise Pearson on TIME-ALIGNED return series,
                           with overlap counts + reliability flags.
  3. effective_breadth -- participation ratio of the matrix eigenvalues: the
                          effective number of independent bets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

FAMILIES = ("trend", "breakout", "mean_reversion", "flow", "volatility", "unknown")

# Adapter -> family (exact, highest priority after an explicit override).
ADAPTER_FAMILY = {
    "sma_crossover": "trend", "ema_crossover": "trend", "tjl_trend_join": "trend",
    "pdh_pdl_break_retest": "breakout",
    "rsi_mean_reversion": "mean_reversion", "vwap_reclaim": "mean_reversion",
}

# Name keyword -> family (inference fallback, checked in this order).
NAME_KEYWORDS = [
    ("breakout", ("breakout", "orb", "opening range", "pdh", "pdl",
                  "break-retest", "break retest", "range break")),
    ("mean_reversion", ("rsi", "vwap", "reclaim", "mean reversion", "reversion",
                        "gap fill", "gap-fill", "oversold", "bollinger")),
    ("flow", ("insider", "options flow", "order flow", "unusual volume",
              "dark pool", "sweep", "flow")),
    ("volatility", ("atr", "squeeze", "volatility", "expansion", "vol exp")),
    ("trend", ("tjl", "trend", "ema", "sma", "crossover", "momentum",
               "continuation", "relative strength", "relative-strength")),
]

# Minimum overlapping periods for a correlation to be considered reliable.
MIN_OVERLAP = {"monthly": 6, "daily": 30}

# Conservative fill for unreliable / undefined correlations (used for breadth).
FILL_SAME_FAMILY = 0.75
FILL_DIFF_FAMILY = 0.25

_OVERRIDE_RE = re.compile(
    r"//\s*(?:strategy_)?family\s*[:=]\s*([A-Za-z_][A-Za-z_\- ]*)", re.IGNORECASE)


def _norm_family(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def parse_family_override(text: str) -> str | None:
    """Extract a `// family: X` / `// strategy_family = X` comment, or None."""
    m = _OVERRIDE_RE.search(text or "")
    return _norm_family(m.group(1)) if m else None


def family_for(name: str, adapter_type: str | None = None,
               override: str | None = None) -> str:
    """Resolve a strategy family: explicit override > adapter > name inference."""
    if override:
        fam = _norm_family(override)
        return fam if fam in FAMILIES else fam        # trust explicit intent
    if adapter_type and adapter_type in ADAPTER_FAMILY:
        return ADAPTER_FAMILY[adapter_type]
    low = (name or "").lower()
    for fam, kws in NAME_KEYWORDS:
        if any(k in low for k in kws):
            return fam
    return "unknown"


# -- return series ---------------------------------------------------------
def _period_key(exit_date: str, freq: str) -> str:
    return exit_date[:7] if freq == "monthly" else exit_date   # YYYY-MM vs YYYY-MM-DD


def returns_series(trades: list, freq: str = "monthly") -> dict:
    """Dated fractional-return series. Multiple trades in a period compound."""
    by_period: dict = {}
    for t in trades:
        r = getattr(t, "return_pct", None)
        if r is None:
            continue
        key = _period_key(t.exit_date, freq)
        by_period[key] = by_period.get(key, 1.0) * (1.0 + float(r))
    return {k: v - 1.0 for k, v in by_period.items()}


# -- pairwise correlation --------------------------------------------------
def pairwise(series_a: dict, series_b: dict, min_overlap: int) -> tuple:
    """(corr|nan, overlap_count, reliable). Uses overlapping periods only."""
    common = sorted(set(series_a) & set(series_b))
    n = len(common)
    if n < 2:
        return float("nan"), n, False
    a = np.array([series_a[k] for k in common], float)
    b = np.array([series_b[k] for k in common], float)
    if a.std() == 0 or b.std() == 0:
        return float("nan"), n, False
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr, n, (n >= min_overlap)


@dataclass
class IndependenceReport:
    ids: list
    families: dict
    freq: str
    min_overlap: int
    raw_matrix: list           # NxN, nan where undefined
    overlap_matrix: list       # NxN ints
    reliable_matrix: list      # NxN bools
    cleaned_matrix: list       # NxN, conservative fills applied (used for breadth)
    effective_breadth: float
    n_strategies: int
    per_strategy: dict         # id -> {family, avg_correlation, independence_score}
    insufficient: list = field(default_factory=list)   # ids dropped (too little data)


def _build_cleaned(ids, raw, reliable, families):
    n = len(ids)
    cleaned = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if reliable[i][j] and not np.isnan(raw[i][j]):
                cleaned[i][j] = raw[i][j]
            else:
                same = families.get(ids[i]) == families.get(ids[j])
                cleaned[i][j] = FILL_SAME_FAMILY if same else FILL_DIFF_FAMILY
    return cleaned


def effective_breadth(cleaned: np.ndarray) -> float:
    """Participation ratio of eigenvalues: (Σλ)² / Σλ². Clamped to [1, N]."""
    n = len(cleaned)
    if n <= 1:
        return float(n)
    lam = np.linalg.eigvalsh(np.asarray(cleaned, float))
    s2 = float((lam ** 2).sum())
    if s2 <= 0:
        return 1.0
    pr = float(lam.sum() ** 2 / s2)
    return float(min(max(pr, 1.0), n))


def correlation_matrices(series_by_id: dict, family_by_id: dict,
                         freq: str = "monthly", min_overlap: int | None = None):
    """Build raw/overlap/reliable/cleaned matrices over strategies with >=2
    periods. Returns (ids, raw, overlap, reliable, cleaned, insufficient)."""
    if min_overlap is None:
        min_overlap = MIN_OVERLAP.get(freq, 6)
    ids = [s for s in series_by_id if len(series_by_id[s]) >= 2]
    insufficient = [s for s in series_by_id if s not in ids]
    n = len(ids)
    raw = [[float("nan")] * n for _ in range(n)]
    overlap = [[0] * n for _ in range(n)]
    reliable = [[False] * n for _ in range(n)]
    for i in range(n):
        raw[i][i], overlap[i][i], reliable[i][i] = 1.0, len(series_by_id[ids[i]]), True
        for j in range(i + 1, n):
            c, ov, rel = pairwise(series_by_id[ids[i]], series_by_id[ids[j]],
                                  min_overlap)
            raw[i][j] = raw[j][i] = c
            overlap[i][j] = overlap[j][i] = ov
            reliable[i][j] = reliable[j][i] = rel
    cleaned = _build_cleaned(ids, raw, reliable, family_by_id)
    return ids, raw, overlap, reliable, cleaned, insufficient


def independence_report(trades_by_strategy: dict, family_by_id: dict,
                        freq: str = "monthly",
                        min_overlap: int | None = None) -> IndependenceReport:
    """Full diagnostic from {strategy_id: [TradeRecord]} + {strategy_id: family}."""
    if min_overlap is None:
        min_overlap = MIN_OVERLAP.get(freq, 6)
    series = {sid: returns_series(tr, freq) for sid, tr in trades_by_strategy.items()}
    ids, raw, overlap, reliable, cleaned, insufficient = correlation_matrices(
        series, family_by_id, freq, min_overlap)

    breadth = effective_breadth(np.asarray(cleaned, float))
    per = {}
    n = len(ids)
    for i, sid in enumerate(ids):
        if n > 1:
            others = [abs(cleaned[i][j]) for j in range(n) if j != i]
            avg = float(np.mean(others))
        else:
            avg = 0.0
        per[sid] = {
            "family": family_by_id.get(sid, "unknown"),
            "avg_correlation": round(avg, 4),
            "independence_score": round(max(0.0, min(1.0, 1.0 - avg)), 4),
        }
    return IndependenceReport(
        ids=ids, families={s: family_by_id.get(s, "unknown") for s in ids},
        freq=freq, min_overlap=min_overlap,
        raw_matrix=raw, overlap_matrix=overlap, reliable_matrix=reliable,
        cleaned_matrix=[list(row) for row in np.asarray(cleaned).tolist()],
        effective_breadth=round(breadth, 2), n_strategies=n,
        per_strategy=per, insufficient=insufficient)


def report_from_results(results: list, freq: str = "monthly",
                        min_overlap: int | None = None) -> IndependenceReport:
    """Convenience: build the report from StrategyResult objects (uses each
    result's pooled trades + resolved family). Includes every strategy that
    produced trades, regardless of RL eligibility (this is population-phase)."""
    trades = {r.strategy_id: r.pooled_trades for r in results
              if getattr(r, "pooled_trades", None)}
    fams = {r.strategy_id: getattr(r, "strategy_family", "unknown") for r in results}
    return independence_report(trades, fams, freq, min_overlap)


def _demo():
    # Five near-identical series -> effective breadth ~1; five orthogonal ~5.
    keys = [f"p{i:02d}" for i in range(60)]
    base = np.sin(np.linspace(0, 6, 60))
    clones = {f"c{k}": dict(zip(keys, (base + 0.01 * np.sin(np.linspace(k, k + 1, 60))).tolist()))
              for k in range(5)}
    fams = {k: "trend" for k in clones}
    _, _, _, _, cl, _ = correlation_matrices(clones, fams, "monthly", 6)
    b_clone = effective_breadth(np.asarray(cl, float))

    indep = {f"i{k}": dict(zip(keys, np.sin(np.linspace(0, (k + 1) * 6.28, 60) * (k + 1)).tolist()))
             for k in range(5)}
    famsi = {k: "unknown" for k in indep}
    _, _, _, _, cli, _ = correlation_matrices(indep, famsi, "monthly", 6)
    b_indep = effective_breadth(np.asarray(cli, float))

    assert b_clone < 1.6, b_clone
    assert b_indep > 3.5, b_indep
    assert family_for("TJL Trend Join", "tjl_trend_join") == "trend"
    assert family_for("Mystery", None, override="mean-reversion") == "mean_reversion"
    assert parse_family_override("// family: breakout\nx=1") == "breakout"
    print(f"correlation._demo OK breadth clones={b_clone:.2f} indep={b_indep:.2f}")


if __name__ == "__main__":
    _demo()
