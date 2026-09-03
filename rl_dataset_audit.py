#!/usr/bin/env python3
"""
rl_dataset_audit.py — Missingness report for the offline RL datasets.
=====================================================================

Inspects each built RL dataset and reports, per enrichment field, whether the column
is present and what fraction of rows carry a non-neutral (non-zero) value. Used to
quantify the "seriesless" gap that the Alpaca bar-enrichment phase fills. OFFLINE /
read-only — loads data/rl_training/*.{parquet,csv,json} only; no network, no orders.
"""

from __future__ import annotations

from typing import Any, Dict, List

import rl_common as rc

DATASETS = ["alpha_entry_dataset", "alpha_exit_dataset",
            "buffett_dca_dataset", "capital_allocation_dataset"]

# Enrichment fields the bar-context phase should populate (bars themselves are inputs,
# represented here by the trend/relative features derived from them).
ENRICH_FIELDS = [
    "mtf_trend_fast",      # 5m trend
    "mtf_trend_mid",       # 15m / 1h trend
    "mtf_trend_slow",      # daily trend
    "mtf_alignment",       # cross-timeframe agreement
    "rel_volume",          # relative volume
    "vwap_distance",       # VWAP distance
    "atr_pct",             # ATR as % of price
    "spy_regime",          # market (SPY) regime
    "sector_regime",       # sector ETF regime
    "catalyst_strength",   # catalyst / volume-spike signal
]


def _nonzero_frac(rows: List[dict], field: str) -> float:
    vals = [rc.to_f(r.get(field)) for r in rows if field in r]
    if not vals:
        return 0.0
    return round(sum(1 for v in vals if abs(v) > 1e-9) / len(vals), 4)


def audit_dataset(name: str) -> Dict[str, Any]:
    rows = rc.load_dataset(name)
    report: Dict[str, Any] = {"dataset": name, "rows": len(rows), "fields": {}}
    cols = set().union(*[set(r.keys()) for r in rows]) if rows else set()
    for f in ENRICH_FIELDS:
        present = f in cols
        report["fields"][f] = {"present": present,
                               "nonzero_frac": _nonzero_frac(rows, f) if present else 0.0}
    return report


def audit_all() -> Dict[str, Any]:
    return {"datasets": [audit_dataset(n) for n in DATASETS]}


def print_report(report: Dict[str, Any]) -> None:
    print("RL dataset enrichment-missingness report")
    print("=" * 78)
    for d in report["datasets"]:
        print(f"\n{d['dataset']}  ({d['rows']} rows)")
        if d["rows"] == 0:
            print("  (empty / not built)")
            continue
        for f, info in d["fields"].items():
            if not info["present"]:
                status = "ABSENT (column missing)"
            elif info["nonzero_frac"] == 0.0:
                status = "present but ALL-NEUTRAL (0.0 nonzero)"
            else:
                status = f"populated  nonzero={info['nonzero_frac']:.1%}"
            print(f"    {f:18s} {status}")


if __name__ == "__main__":
    rep = audit_all()
    print_report(rep)
    rc.write_json(rc.RL_DATA_DIR / "dataset_audit_report.json", rep)
    print(f"\n-> {rc.RL_DATA_DIR / 'dataset_audit_report.json'}")
