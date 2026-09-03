#!/usr/bin/env python3
"""
alpha_discovery/options_alpha_discovery.py — promote options-sentiment data into
the hidden-alpha discovery framework: validate → approve → attribute.

This is the RESEARCH/TRAINING layer for options flow. It is completely
independent of the live booster overlay (options_sentiment_av.py /
options_booster.py): nothing here can place, size, or gate a live trade. Its job
is to decide which options-derived features have earned a place in the RL state.

Pipeline
--------
    1. Candidate features come from alpha_discovery.options_feature_builder
       (the `options_sentiment` family) + the compressed latent options states
       from alpha_discovery.latent_feature_discovery.
    2. validate()  — measure each feature against forward returns (5m/15m/30m/60m
       + EOD), MFE, MAE, win-rate, expectancy, profit-factor, Sharpe. Reuses the
       proven alpha_discovery.news_price_reaction_model.validate_feature.
    3. promote()   — apply conservative out-of-sample thresholds and write the
       Approved Alpha Registry (only validated features are promoted).
    4. attribution()— write reports/options/options_alpha_contribution.csv
       (Sharpe / PnL / Trade-Selection / Timing contribution per feature).

A "labeled record" is a dict carrying each candidate feature value plus realized
forward outcomes, e.g.:
    {"put_call_pressure_factor": 0.4, "iv_skew_factor": 0.03, ...,
     "ret_5m": 0.001, "ret_15m": 0.002, "ret_30m": 0.0, "ret_60m": 0.004,
     "ret_eod": 0.01, "mfe": 0.012, "mae": -0.004}

Read-only / offline; never raises. Promotes NOTHING when data is insufficient
(validation discipline — an unvalidated feature never enters the RL observation).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from alpha_discovery.news_price_reaction_model import (
    HORIZONS,
    validate_features,
    write_validation_report,
)
from alpha_discovery.latent_feature_discovery import discover_options_latent_factors
from alpha_discovery.options_feature_builder import (
    OPTIONS_SENTIMENT_FACTORS,
    OPTIONS_SENTIMENT_INTERACTIONS,
    OPTIONS_SENTIMENT_LATENT,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "options"

VALIDATION_REPORT = REPORT_DIR / "options_validation_report.csv"
APPROVED_REGISTRY_JSON = REPORT_DIR / "options_approved_features.json"
APPROVED_REGISTRY_CSV = REPORT_DIR / "options_approved_features.csv"
ATTRIBUTION_REPORT = REPORT_DIR / "options_alpha_contribution.csv"

# The full options_sentiment candidate family validated by this module.
OPTIONS_SENTIMENT_ALL_FEATURES = (
    list(OPTIONS_SENTIMENT_FACTORS)
    + list(OPTIONS_SENTIMENT_INTERACTIONS)
    + list(OPTIONS_SENTIMENT_LATENT)
)


# =============================================================================
# Config (conservative promotion thresholds; env-tunable)
# =============================================================================

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PromotionConfig:
    """Out-of-sample gates a feature must clear to enter the Approved registry."""

    def __init__(self) -> None:
        self.min_samples = _env_int("OPTIONS_ALPHA_MIN_SAMPLES", 30)
        self.min_abs_ic = _env_float("OPTIONS_ALPHA_MIN_ABS_IC", 0.03)
        self.min_ic_horizons = _env_int("OPTIONS_ALPHA_MIN_IC_HORIZONS", 2)
        # IC significance: |IC|*sqrt(n) must exceed this z-like threshold so a
        # spurious small-sample correlation cannot promote a noise feature.
        self.min_ic_tstat = _env_float("OPTIONS_ALPHA_MIN_IC_TSTAT", 2.0)
        self.min_profit_factor = _env_float("OPTIONS_ALPHA_MIN_PROFIT_FACTOR", 1.10)
        self.min_sharpe = _env_float("OPTIONS_ALPHA_MIN_SHARPE", 0.30)
        self.min_expectancy = _env_float("OPTIONS_ALPHA_MIN_EXPECTANCY", 0.0)


# =============================================================================
# Record assembly
# =============================================================================

def attach_latent_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute the compressed latent options states in-sample and merge them in.

    So `latent_options_factor_1..3`, `latent_options_sentiment_state` and
    `latent_gamma_state` can be validated alongside the raw candidate factors.
    Returns a new list; never mutates the inputs. Safe on empty/degenerate data.
    """
    if not records:
        return list(records)
    try:
        latents = discover_options_latent_factors(records)
    except Exception:  # pragma: no cover - latent discovery is advisory
        return [dict(r) for r in records]
    out = []
    for rec, lat in zip(records, latents):
        merged = dict(rec)
        merged.update(lat)
        out.append(merged)
    return out


def load_records_from_dataset(
    dataset_path: str | Path,
    horizon_map: Optional[dict[str, str]] = None,
    mfe_col: str = "mfe",
    mae_col: str = "mae",
) -> list[dict[str, Any]]:
    """Load labeled records from a parquet/CSV that has options features + labels.

    `horizon_map` renames dataset columns to the canonical HORIZONS
    (ret_5m/ret_15m/ret_30m/ret_60m/ret_eod). Returns [] if the file or its
    label columns are absent (caller then reports no_data — nothing is promoted).
    """
    path = Path(dataset_path)
    if not path.exists():
        logger.warning("options_alpha_discovery: dataset not found: %s", path)
        return []
    try:
        import pandas as pd  # local import; optional dependency

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - optional/IO
        logger.warning("options_alpha_discovery: failed to load %s: %s", path, exc)
        return []

    rename = dict(horizon_map or {})
    if mfe_col != "mfe":
        rename[mfe_col] = "mfe"
    if mae_col != "mae":
        rename[mae_col] = "mae"
    if rename:
        df = df.rename(columns=rename)
    return df.to_dict(orient="records")


# =============================================================================
# Validate / promote / attribute
# =============================================================================

def validate(records: list[dict[str, Any]],
             features: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    """Validate the options_sentiment candidate features against forward returns."""
    feats = list(features) if features is not None else list(OPTIONS_SENTIMENT_ALL_FEATURES)
    return validate_features(records, feats)


def _approval_decision(row: dict[str, Any], cfg: PromotionConfig) -> tuple[bool, list[str]]:
    """Conservative AND-gate: a feature must show *consistent, significant*
    forward-return predictivity AND positive risk-adjusted economics. All gates
    must pass — no single strong metric can carry a feature past the others."""
    reasons: list[str] = []
    if row.get("status") != "ok":
        return False, ["no_data"]
    n = int(row.get("n", 0))
    strong_ic = sum(
        1 for h in HORIZONS if abs(float(row.get(f"ic_{h}", 0.0) or 0.0)) >= cfg.min_abs_ic
    )
    best_ic = max((abs(float(row.get(f"ic_{h}", 0.0) or 0.0)) for h in HORIZONS), default=0.0)
    tstat = best_ic * (n ** 0.5)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    pf = float(row.get("profit_factor", 0.0) or 0.0)
    expectancy = float(row.get("expectancy", 0.0) or 0.0)

    if n < cfg.min_samples:
        reasons.append(f"insufficient_samples({n}<{cfg.min_samples})")
    if strong_ic < cfg.min_ic_horizons:
        reasons.append(f"inconsistent_ic(strong_ic={strong_ic}<{cfg.min_ic_horizons})")
    if tstat < cfg.min_ic_tstat:
        reasons.append(f"insignificant_ic(tstat={tstat:.2f}<{cfg.min_ic_tstat})")
    if sharpe < cfg.min_sharpe:
        reasons.append(f"low_sharpe({sharpe:.2f}<{cfg.min_sharpe})")
    if pf < cfg.min_profit_factor:
        reasons.append(f"profit_factor({pf:.2f}<{cfg.min_profit_factor})")
    if expectancy <= cfg.min_expectancy:
        reasons.append(f"expectancy({expectancy:.4f}<=0)")

    approved = not reasons
    if approved:
        reasons.append(
            f"ok(n={n}, strong_ic={strong_ic}, tstat={tstat:.2f}, sharpe={sharpe:.2f}, pf={pf:.2f}, exp={expectancy:.4f})"
        )
    return approved, reasons


def promote(validation_rows: list[dict[str, Any]],
            cfg: Optional[PromotionConfig] = None,
            write: bool = True) -> dict[str, Any]:
    """Apply promotion gates and (optionally) write the Approved Alpha Registry."""
    cfg = cfg or PromotionConfig()
    entries: list[dict[str, Any]] = []
    approved_names: list[str] = []
    for row in validation_rows:
        approved, reasons = _approval_decision(row, cfg)
        entry = {
            "feature": row.get("feature"),
            "family": "options_sentiment",
            "approved": bool(approved),
            "reasons": "; ".join(reasons),
            "n": row.get("n", 0),
            "best_abs_ic": round(max((abs(float(row.get(f"ic_{h}", 0.0) or 0.0)) for h in HORIZONS), default=0.0), 4),
            "sharpe": row.get("sharpe", 0.0),
            "profit_factor": row.get("profit_factor", 0.0),
            "expectancy": row.get("expectancy", 0.0),
            "win_rate": row.get("win_rate", 0.0),
            "approved_at": _now_iso() if approved else "",
        }
        entries.append(entry)
        if approved:
            approved_names.append(row.get("feature"))

    registry = {
        "family": "options_sentiment",
        "generated_at": _now_iso(),
        "thresholds": {
            "min_samples": cfg.min_samples, "min_abs_ic": cfg.min_abs_ic,
            "min_ic_horizons": cfg.min_ic_horizons, "min_profit_factor": cfg.min_profit_factor,
            "min_sharpe": cfg.min_sharpe, "min_expectancy": cfg.min_expectancy,
        },
        "approved_features": approved_names,
        "approved_count": len(approved_names),
        "evaluated_count": len(entries),
        "entries": entries,
    }
    if write:
        _write_registry(registry, entries)
    return registry


def _write_registry(registry: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        APPROVED_REGISTRY_JSON.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
        cols = ["feature", "family", "approved", "reasons", "n", "best_abs_ic",
                "sharpe", "profit_factor", "expectancy", "win_rate", "approved_at"]
        with APPROVED_REGISTRY_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for e in entries:
                w.writerow({k: e.get(k) for k in cols})
    except Exception:  # pragma: no cover - non-fatal
        logger.warning("options_alpha_discovery: failed to write approved registry")


def attribution(validation_rows: list[dict[str, Any]], write: bool = True) -> list[dict[str, Any]]:
    """Per-feature alpha attribution: Sharpe / PnL / Trade-Selection / Timing.

    These are deterministic research proxies derived from the validation metrics:
        Sharpe Contribution        = signed-return Sharpe of the feature
        PnL Contribution           = expectancy * n   (total signed return, return units)
        Trade Selection Contribution = win_rate - 0.5  (edge over a coin-flip selector)
        Timing Contribution        = MFE / (MFE + |MAE|)  (favorable-excursion capture),
                                     falling back to the best |IC| across horizons
    """
    rows: list[dict[str, Any]] = []
    for row in validation_rows:
        if row.get("status") != "ok":
            continue
        n = int(row.get("n", 0))
        expectancy = float(row.get("expectancy", 0.0) or 0.0)
        win_rate = float(row.get("win_rate", 0.0) or 0.0)
        sharpe = float(row.get("sharpe", 0.0) or 0.0)
        mfe = float(row.get("mfe_mean", 0.0) or 0.0)
        mae = abs(float(row.get("mae_mean", 0.0) or 0.0))
        if (mfe + mae) > 1e-12:
            timing = mfe / (mfe + mae)
        else:
            timing = max((abs(float(row.get(f"ic_{h}", 0.0) or 0.0)) for h in HORIZONS), default=0.0)
        rows.append({
            "Feature": row.get("feature"),
            "Sharpe Contribution": round(sharpe, 4),
            "PnL Contribution": round(expectancy * n, 6),
            "Trade Selection Contribution": round(win_rate - 0.5, 4),
            "Timing Contribution": round(timing, 4),
        })
    if write:
        _write_attribution(rows)
    return rows


def _write_attribution(rows: list[dict[str, Any]]) -> None:
    cols = ["Feature", "Sharpe Contribution", "PnL Contribution",
            "Trade Selection Contribution", "Timing Contribution"]
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with ATTRIBUTION_REPORT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception:  # pragma: no cover - non-fatal
        logger.warning("options_alpha_discovery: failed to write attribution report")


# =============================================================================
# Orchestrator
# =============================================================================

def run(
    records: Optional[list[dict[str, Any]]] = None,
    dataset_path: Optional[str | Path] = None,
    horizon_map: Optional[dict[str, str]] = None,
    mfe_col: str = "mfe",
    mae_col: str = "mae",
    add_latents: bool = True,
    cfg: Optional[PromotionConfig] = None,
) -> dict[str, Any]:
    """Validate → promote → attribute the options_sentiment family end-to-end.

    Provide `records` directly, or a `dataset_path` to a labeled parquet/CSV.
    Returns a summary dict and writes the validation report, Approved registry,
    and attribution CSV under reports/options/. Promotes nothing on no_data.
    """
    cfg = cfg or PromotionConfig()
    if records is None and dataset_path is not None:
        records = load_records_from_dataset(dataset_path, horizon_map, mfe_col, mae_col)
    records = records or []

    if not records:
        summary = {
            "status": "no_data",
            "evaluated_count": 0,
            "approved_count": 0,
            "approved_features": [],
            "validation_report": str(VALIDATION_REPORT),
            "approved_registry": str(APPROVED_REGISTRY_JSON),
            "attribution_report": str(ATTRIBUTION_REPORT),
        }
        logger.info("options_alpha_discovery: no labeled records — nothing promoted")
        return summary

    if add_latents:
        records = attach_latent_features(records)

    validation_rows = validate(records)
    write_validation_report(validation_rows, VALIDATION_REPORT)
    registry = promote(validation_rows, cfg=cfg, write=True)
    attribution(validation_rows, write=True)

    return {
        "status": "ok",
        "evaluated_count": registry["evaluated_count"],
        "approved_count": registry["approved_count"],
        "approved_features": registry["approved_features"],
        "validation_report": str(VALIDATION_REPORT),
        "approved_registry": str(APPROVED_REGISTRY_JSON),
        "attribution_report": str(ATTRIBUTION_REPORT),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    ds = sys.argv[1] if len(sys.argv) > 1 else None
    out = run(dataset_path=ds)
    print(json.dumps(out, indent=2, default=str))
