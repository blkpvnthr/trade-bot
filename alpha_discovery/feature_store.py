#!/usr/bin/env python3
"""
alpha_discovery/feature_store.py — persist + load realtime feature snapshots.

A thin, dependency-light store for the realtime feature families produced by the
builders. Snapshots are written per-symbol as JSONL (append-only history) and a
flat "latest" CSV/JSON report is emitted for monitoring.

Layout:
    feature_store/realtime/<SYMBOL>.jsonl          rolling per-symbol snapshots
    reports/alpha_discovery/latest_realtime_features.csv
    reports/alpha_discovery/latest_realtime_features.json

Never raises; all IO failures are logged and swallowed.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE_DIR = PROJECT_ROOT / "feature_store" / "realtime"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "alpha_discovery"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RealtimeFeatureStore:
    def __init__(self, store_dir: Optional[Path] = None, report_dir: Optional[Path] = None):
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
        for d in (self.store_dir, self.report_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - non-fatal
                pass

    def save_snapshot(self, symbol: str, features: dict[str, Any]) -> None:
        symbol = (symbol or "").upper().strip()
        if not symbol:
            return
        rec = {"symbol": symbol, "snapshot_at": _now_iso(), **features}
        try:
            path = self.store_dir / f"{symbol}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:  # pragma: no cover - non-fatal
            logger.warning("feature_store: save failed for %s", symbol)

    def load_history(self, symbol: str) -> list[dict[str, Any]]:
        symbol = (symbol or "").upper().strip()
        path = self.store_dir / f"{symbol}.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception:  # pragma: no cover - non-fatal
            return []
        return out

    def write_latest_report(self, rows: list[dict[str, Any]]) -> None:
        """Write the latest per-symbol feature snapshot to CSV + JSON reports."""
        self._write_report(rows, "latest_realtime_features")

    def write_family_report(self, rows: list[dict[str, Any]], family: str,
                            names: Optional[list[str]] = None) -> None:
        """Write a family-scoped feature report (e.g. options_sentiment).

        When ``names`` is given, each row is projected to ``symbol`` + those
        columns so the report stays focused on one feature family; otherwise the
        full rows are written. Output: reports/alpha_discovery/latest_{family}_features.{csv,json}.
        """
        family = (family or "family").strip().replace(" ", "_")
        if names:
            keep = ["symbol", "snapshot_at"] + list(names)
            rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
        self._write_report(rows, f"latest_{family}_features")

    def _write_report(self, rows: list[dict[str, Any]], stem: str) -> None:
        if not rows:
            return
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        try:
            csv_path = self.report_dir / f"{stem}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in cols})
            json_path = self.report_dir / f"{stem}.json"
            json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        except Exception:  # pragma: no cover - non-fatal
            logger.warning("feature_store: report write failed (%s)", stem)
