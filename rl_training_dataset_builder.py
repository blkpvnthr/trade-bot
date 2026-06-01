#!/usr/bin/env python3
"""
rl_training_dataset_builder.py — Build offline RL datasets from the new system outputs.
=======================================================================================

Consumes the Research Lab events + daily output JSON written by the two-engine run.py
and produces one supervised/contextual dataset per RL agent:

    data/rl_training/alpha_entry_dataset.parquet
    data/rl_training/alpha_exit_dataset.parquet
    data/rl_training/buffett_dca_dataset.parquet
    data/rl_training/capital_allocation_dataset.parquet

Parquet when pyarrow/pandas are available; CSV fallback otherwise (logged). Missing
input files are handled safely (empty dataset, never raises). OFFLINE only.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import rl_common as rc
from oops_option_utility import OOPS_FIELDS, enrich_event_with_oops

logger = logging.getLogger(__name__)


def _proxy_realized_R(e: dict) -> float:
    """When no matched exit exists, estimate R from probability_of_profit / expected_R."""
    if e.get("expected_R") is not None:
        return rc.to_f(e.get("expected_R"))
    pop = rc.to_f(e.get("probability_of_profit"), 0.5)
    return round(pop * 3.0 - (1.0 - pop) * 1.0, 4)   # 3R target, 1R stop expectation


def _det_noise(key: str) -> float:
    """Deterministic pseudo-noise in [-0.5, 0.5] from a string key (no RNG)."""
    h = abs(hash(key)) % 1000
    return (h / 1000.0) - 0.5


# ── per-agent dataset builders ──────────────────────────────────────────────────────
def build_alpha_entry_dataset(events: List[dict]) -> List[dict]:
    # map symbol -> list of exit r_multiples (chronological) to attach a realized outcome
    exits_by_symbol: Dict[str, List[float]] = {}
    for e in events:
        if e.get("event_type") == "alpha_exit":
            sym = str(e.get("symbol") or e.get("underlying") or "").upper()
            exits_by_symbol.setdefault(sym, []).append(rc.to_f(e.get("r_multiple", e.get("realized_R"))))
    rows = []
    for e in events:
        if e.get("event_type") not in ("alpha_entry", "alpha_rejection"):
            continue
        sym = str(e.get("symbol") or "").upper()
        matched = exits_by_symbol.get(sym, [])
        realized_R = matched.pop(0) if matched else _proxy_realized_R(e)
        feats = rc.alpha_entry_features(e)
        row = {f: feats[i] for i, f in enumerate(rc.ALPHA_ENTRY_FEATURES)}
        # OOPS enhancement: expected-utility / payoff-simulation features from the option paper.
        # If the event lacks a full option chain, the scorer falls back safely to available
        # price/IV/liquidity proxies and still emits every field for stable model shapes.
        row.update(enrich_event_with_oops(e))
        row.update({"symbol": sym, "date_tag": e.get("date_tag", ""),
                    "event_type": e.get("event_type"), "realized_R": realized_R})
        rows.append(row)
    return rows


def build_alpha_exit_dataset(events: List[dict]) -> List[dict]:
    rows = []
    for e in events:
        if e.get("event_type") != "alpha_exit":
            continue
        r = rc.to_f(e.get("r_multiple", e.get("realized_R")))
        feats = rc.alpha_exit_features(e)
        row = {f: feats[i] for i, f in enumerate(rc.ALPHA_EXIT_FEATURES)}
        # Prefer the real peak excursion from replay/live when present; else proxy.
        peak_R = rc.to_f(e.get("peak_R")) if e.get("peak_R") is not None else max(r, r * 1.2, 0.0)
        overnight_loss = bool(rc.to_f(e.get("holding_period_days")) >= 1 and r < 0)
        row.update({"symbol": str(e.get("symbol") or e.get("underlying") or "").upper(),
                    "date_tag": e.get("date_tag", ""), "realized_R": r,
                    "peak_R": round(peak_R, 4), "overnight_loss": overnight_loss})
        rows.append(row)
    return rows


def build_buffett_dca_dataset(events: List[dict]) -> List[dict]:
    rows = []
    for e in events:
        if e.get("event_type") not in ("bal_buy", "bal_dca", "bal_rejection"):
            continue
        feats = rc.buffett_dca_features(e)
        row = {f: feats[i] for i, f in enumerate(rc.BUFFETT_DCA_FEATURES)}
        sym = str(e.get("symbol") or "").upper()
        discount = row["intrinsic_value_discount"]
        conv = row["conviction_score"]
        # Forward-return proxy from valuation discount + conviction (deterministic),
        # over 5d/20d/60d horizons. Real forward returns can replace this once available.
        base = 0.5 * discount + 0.3 * conv + 0.05 * _det_noise(sym + e.get("date_tag", ""))
        row.update({
            "symbol": sym, "date_tag": e.get("date_tag", ""), "event_type": e.get("event_type"),
            "fwd_return_5d": round(base * 0.25, 4),
            "fwd_return_20d": round(base * 0.6, 4),
            "fwd_return_60d": round(base, 4),
            "valuation_improvement": round(discount, 4),
        })
        rows.append(row)
    return rows


def build_capital_allocation_dataset(events: List[dict]) -> List[dict]:
    # Aggregate per date_tag: alpha expectancy/perf, buffett opportunity quality, etc.
    by_day: Dict[str, Dict[str, List[float]]] = {}
    for e in events:
        d = e.get("date_tag", "")
        by_day.setdefault(d, {"alpha_R": [], "bal_comp": [], "bal_score": []})
        et = e.get("event_type")
        if et == "alpha_exit":
            by_day[d]["alpha_R"].append(rc.to_f(e.get("r_multiple", e.get("realized_R"))))
        elif et in ("bal_buy", "bal_dca"):
            by_day[d]["bal_comp"].append(rc.to_f(e.get("composite_dca_score")) / 100.0)
            by_day[d]["bal_score"].append(rc.to_f(e.get("bal_score")) / 100.0)
    rows = []
    for d, agg in sorted(by_day.items()):
        alpha_R = agg["alpha_R"]
        alpha_exp = sum(alpha_R) / len(alpha_R) if alpha_R else 0.0
        alpha_perf = sum(alpha_R)
        buffett_quality = (sum(agg["bal_comp"]) / len(agg["bal_comp"])) if agg["bal_comp"] else 0.0
        rows.append({
            "date_tag": d,
            "alpha_expectancy": round(alpha_exp, 4),
            "alpha_perf": round(alpha_perf, 4),
            "buffett_quality": round(buffett_quality, 4),
            "market_state_confidence": 0.5,
            "drawdown": round(abs(min(0.0, alpha_perf)) * 0.1, 4),
            "volatility": round(_safe_std(alpha_R), 4),
            "waterfall_ratio": 0.5,
            "optimizer_perf": round(buffett_quality, 4),
            # realized bucket returns proxy for reward simulation
            "alpha_bucket_return": round(alpha_exp * 0.01, 6),
            "buffett_bucket_return": round(buffett_quality * 0.008, 6),
            "cash_bucket_return": 0.0001,
        })
    return rows


def _safe_std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ── writer (parquet → CSV fallback) ─────────────────────────────────────────────────
def write_dataset(rows: List[dict], name: str, out_dir: Path = None) -> Dict[str, Any]:
    out_dir = Path(out_dir) if out_dir else rc.RL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"name": name, "rows": len(rows), "format": None, "path": None}
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        try:
            import pyarrow  # noqa: F401
            p = out_dir / f"{name}.parquet"
            df.to_parquet(p, index=False)
            info["format"] = "parquet"; info["path"] = str(p)
            return info
        except Exception as exc:
            logger.warning("parquet unavailable for %s (%s) — writing CSV fallback", name, exc)
            p = out_dir / f"{name}.csv"
            df.to_csv(p, index=False)
            info["format"] = "csv"; info["path"] = str(p)
            return info
    except Exception as exc:
        logger.warning("pandas unavailable (%s) — writing JSON fallback for %s", exc, name)
        p = out_dir / f"{name}.json"
        rc.write_json(p, rows)
        info["format"] = "json"; info["path"] = str(p)
        return info


def build_all(out_dir: Path = None) -> Dict[str, Any]:
    """Build all four datasets. Safe on missing inputs (empty datasets, no raise).

    Includes both live Research-Lab events and replayed Alpaca-backfill events (Phase 1),
    so the offline historical replay enlarges the RL training set."""
    events = rc.load_research_events() + rc.load_backfill_events()
    builders = {
        "alpha_entry_dataset": build_alpha_entry_dataset,
        "alpha_exit_dataset": build_alpha_exit_dataset,
        "buffett_dca_dataset": build_buffett_dca_dataset,
        "capital_allocation_dataset": build_capital_allocation_dataset,
    }
    summary = {"generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(),
               "source_events": len(events), "datasets": {}}
    for name, fn in builders.items():
        try:
            rows = fn(events)
            # Chronological order so the evaluator's positional 70/30 split is a genuine
            # temporal walk-forward (date_tag is YYYYMMDD → lexicographic == chronological).
            # Without this, source-ordered rows put one source entirely in the test slice.
            # RL_DATASET_SORT=position (or 0) keeps legacy source order for A/B comparison.
            if os.getenv("RL_DATASET_SORT", "chrono").lower() not in ("position", "source", "0"):
                rows = sorted(rows, key=lambda r: str(r.get("date_tag", "")))
        except Exception as exc:
            logger.warning("builder %s failed (%s) — empty dataset", name, exc)
            rows = []
        summary["datasets"][name] = write_dataset(rows, name, out_dir)
    rc.write_json(rc.RL_DATA_DIR / "dataset_summary.json", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = build_all()
    print("RL dataset build complete:")
    for n, info in s["datasets"].items():
        print(f"  {n}: {info['rows']} rows -> {info['format']} ({info['path']})")
