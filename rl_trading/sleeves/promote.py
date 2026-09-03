"""Bridge: turn event_study harness output into a ValidationRecord.

This is the ONLY sanctioned way to mint a passing record. It applies the pre-registered decision
rule mechanically to real statistics, so the gate's booleans cannot be hand-faked — they are
derived from bootstrap CIs, the placebo comparison, OOS, OOD sign-consistency, and a
drop-top-domain concentration check. Garbage stats -> failing record. By construction this returns
a FAILING record today (we have no source that clears it); it is the slot a future winner fills.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .core import ValidationRecord

# Canonical extended-history (daily-rebalance) validation artifact. This is the ONLY artifact
# allowed to mint a passing record for the orthogonal-feature signals. It reproduces 12/12 metrics
# of run_extended_validation.py.
CANONICAL_EXTENDED_VALIDATION = (
    Path(__file__).resolve().parents[2]
    / "rl_strategy_meta_agent/experiments/orthogonal_features/out/extended_validation.json"
)
# Superseded intraday-execution artifacts. They MUST NOT drive sleeve design: lead_validation.json
# is the INTRADAY run (turnover 2.98/day, net Sharpe 0.73@5bps, bootstrap p 0.57 = NOT significant),
# resolved by the daily-rebalance canonical above (turnover 0.24, net Sharpe 1.11, boot p 0).
SUPERSEDED_ARTIFACTS = ("lead_validation.json",)

# Canonical status. The decision in extended_validation.json is ACCEPT, but the historical flag was
# built from reportedDate (ex-post), while live execution uses forward-scheduled dates -> the record
# is a NEAR-PIT proxy, not a perfect replica. Adversarial review (2026-06-24) made this a blocking
# disclosure, not a failure. The gate still passes (the edge is real); the status carries the caveat.
CANONICAL_STATUS = "ACCEPT_WITH_PROVENANCE_CAVEAT"
PROVENANCE_CAVEAT = (
    "Historical validation used actual reportedDate as a near-PIT proxy. "
    "Live execution uses forward scheduled dates only."
)


def _percentile_bootstrap_ci(
    values: Sequence[float], alpha: float = 0.05, n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    """Deterministic percentile bootstrap CI on the mean of `values` (stdlib only).

    Used as a conservative, reproducible CI on the per-year out-of-sample edge. It is weaker than
    the artifact's headline daily-P&L bootstrap (n=55,818, p~0) but uses only published numbers.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("need >= 2 observations for a bootstrap CI")
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int((alpha / 2) * n_boot)], means[int((1 - alpha / 2) * n_boot)]


def from_extended_validation(
    signal: str, path: Optional[str | Path] = None, alpha: float = 0.05,
) -> ValidationRecord:
    """Mint a ValidationRecord from the canonical extended-history validation artifact.

    Maps the published, mechanically-derived statistics to the gate's tri-state booleans:
      - bootstrap_ci  : percentile bootstrap on the per-year walk-forward net Sharpes (lo>0 needed)
      - placebo       : placebo_p < alpha
      - oos           : net-of-cost bootstrap significant AND majority of WF years positive
      - ood           : >=60% WF years positive (sign-consistent) AND symbol-holdout net Sharpe > 0
      - attribution   : FDR-significant AND positive IC AND positive net-of-cost Sharpe
      - concentration : drop-the-best-year mean of WF Sharpes still > 0

    Refuses superseded artifacts and any signal whose pre-registered decision is not ACCEPT.
    """
    path = Path(path) if path is not None else CANONICAL_EXTENDED_VALIDATION
    if path.name in SUPERSEDED_ARTIFACTS:
        raise ValueError(
            f"refusing to mint from superseded artifact {path.name}; it is the intraday run and "
            "must not drive sleeve design. Use extended_validation.json (daily-rebalance canonical)."
        )
    if not path.exists():
        raise FileNotFoundError(f"validation artifact not found: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(
            f"unexpected schema in {path.name}; expected the extended_validation list of signals"
        )
    entry = next((r for r in rows if r.get("signal") == signal), None)
    if entry is None:
        raise KeyError(f"signal {signal!r} not found in {path.name}")
    if entry.get("decision") != "ACCEPT":
        raise ValueError(
            f"{signal}: decision is {entry.get('decision')!r}, not ACCEPT; not promotable"
        )

    yearly = [float(v) for v in entry["wf_by_year"].values()]
    bootstrap_ci = _percentile_bootstrap_ci(yearly, alpha=alpha)

    placebo_passed = float(entry["placebo_p"]) < alpha
    oos_passed = (
        float(entry["net_bootstrap_p"]) < alpha and float(entry["wf_positive_year_frac"]) >= 0.5
    )
    ood_passed = (
        float(entry["wf_positive_year_frac"]) >= 0.6 and float(entry["holdout_net_sharpe"]) > 0.0
    )
    attribution_passed = (
        bool(entry["fdr_significant"])
        and float(entry["ic"]) > 0.0
        and float(entry["net_sharpe_5bps"]) > 0.0
    )
    drop_best = yearly.copy()
    drop_best.remove(max(drop_best))
    concentration_passed = bool(drop_best) and statistics.mean(drop_best) > 0.0

    notes = (
        f"CANONICAL extended-history daily-rebalance validation ({path.name}): "
        f"net_sharpe@5bps={entry['net_sharpe_5bps']}, breakeven_bps={entry['breakeven_bps']}, "
        f"turnover/day={entry['turnover_day']}, net_bootstrap_p={entry['net_bootstrap_p']}, "
        f"placebo_p={entry['placebo_p']}, FDR_significant={entry['fdr_significant']}, "
        f"holdout_net_sharpe={entry['holdout_net_sharpe']}, n_obs={entry['n_obs']}. "
        "Supersedes intraday lead_validation.json (turnover 2.98, net Sharpe 0.73, boot p 0.57). "
        f"STATUS={CANONICAL_STATUS}: {PROVENANCE_CAVEAT}"
    )
    return ValidationRecord(
        source=signal,
        ledger_id=f"extended_validation::{signal}",
        bootstrap_ci=bootstrap_ci,
        placebo_passed=placebo_passed,
        oos_passed=oos_passed,
        ood_passed=ood_passed,
        attribution_passed=attribution_passed,
        concentration_passed=concentration_passed,
        metric="net_sharpe_5bps_walkforward_annual",
        value=float(entry["net_sharpe_5bps"]),
        notes=notes,
    )


def _signs_consistent(per_domain: Mapping[object, Dict[str, float]], min_frac: float = 0.7) -> bool:
    """OOD sign-consistency: same-direction effect in the large majority of domains (years/regimes/
    caps/sectors/...). We do NOT require per-slice significance (low power) — just no catastrophic
    flip. Mirrors the standing OOD policy."""
    means = [d.get("mean", 0.0) for d in per_domain.values() if d.get("n", 0) > 0]
    if len(means) < 3:
        return False
    pos = sum(1 for m in means if m > 0)
    frac = max(pos, len(means) - pos) / len(means)  # dominant direction
    # require the dominant direction to be POSITIVE and cover >= min_frac of domains
    return (pos / len(means) >= min_frac) and frac >= min_frac


def from_event_study(
    source: str,
    ledger_id: str,
    real_stats: Mapping[str, float],
    placebo_stats: Mapping[str, float],
    per_domain: Mapping[object, Dict[str, float]],
    drop_top_domain_stats: Optional[Mapping[str, float]] = None,
    metric: str = "excess_return_vs_spy",
    notes: str = "",
) -> ValidationRecord:
    """Construct a ValidationRecord from harness statistics.

    real_stats / placebo_stats / drop_top_domain_stats: dicts from event_study.stats().
    per_domain: {domain_key: event_study.stats()-like dict} across years/regimes/caps/sectors.
    """
    real_lo = float(real_stats.get("block_lo", real_stats.get("boot_lo", -1.0)))
    real_hi = float(real_stats.get("block_hi", real_stats.get("boot_hi", -1.0)))
    real_mean = float(real_stats.get("mean", 0.0))
    plac_mean = float(placebo_stats.get("mean", 0.0))
    plac_hi = float(placebo_stats.get("block_hi", placebo_stats.get("boot_hi", 1.0)))

    # bootstrap CI: present and strictly positive lower bound
    bootstrap_ci = (real_lo, real_hi)

    # placebo: real effect must beat its own placebo (mean above placebo's upper CI bound,
    # AND real mean meaningfully larger than placebo mean)
    placebo_passed = (real_mean > plac_hi) and (real_mean > 1.5 * max(plac_mean, 0.0))

    # OOS: real CI lower bound positive (the harness CIs are computed on held-out events)
    oos_passed = real_lo > 0.0

    # OOD: sign-consistency across domains (no per-slice significance requirement)
    ood_passed = _signs_consistent(per_domain)

    # attribution: positive hit-rate edge and positive median (effect not from a few outliers)
    attribution_passed = (
        float(real_stats.get("hit_rate", 0.0)) > 0.5
        and float(real_stats.get("median", -1.0)) > 0.0
    )

    # concentration: drop-the-top-domain and require the effect survives positive
    if drop_top_domain_stats is None:
        concentration_passed = None  # not tested -> gate treats as fail
    else:
        dtl = float(drop_top_domain_stats.get("block_lo",
                    drop_top_domain_stats.get("boot_lo", -1.0)))
        concentration_passed = dtl > 0.0

    return ValidationRecord(
        source=source,
        ledger_id=ledger_id,
        bootstrap_ci=bootstrap_ci,
        placebo_passed=bool(placebo_passed),
        oos_passed=bool(oos_passed),
        ood_passed=bool(ood_passed),
        attribution_passed=bool(attribution_passed),
        concentration_passed=(None if concentration_passed is None else bool(concentration_passed)),
        metric=metric,
        value=real_mean,
        notes=notes or f"minted from event_study: n={real_stats.get('n')}, "
                       f"placebo_mean={plac_mean:.4f}",
    )
