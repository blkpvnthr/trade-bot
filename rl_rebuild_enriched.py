#!/usr/bin/env python3
"""
rl_rebuild_enriched.py — Enrich offline events, then rebuild the RL datasets.
=============================================================================

Wires the documented activation flow into one offline pass:
    load research + backfill events
      → rl_alpaca_bar_enrichment.enrich_events  (per-symbol GA daily technicals + SPY/sector
        regime from READ-ONLY Alpaca historical bars, cached to data/cache/alpaca_bars/)
      → rl_training_dataset_builder builders     (map features → the new 34-wide schema)

So the Generating-Alpha daily features (ga_ema_ratio / ga_macd_hist / ga_rsi /
ga_bollinger_pctb / ga_regime_switch) and spy_regime / sector_regime get REAL values in the
rebuilt datasets instead of neutral zeros. Opening-Range and OOPS-payoff-shape features stay
neutral here (they need intraday frames / a live OOPS run) and populate going forward.

OFFLINE: read-only market data only; never imports AlpacaGateway; never submits an order.
"""
from __future__ import annotations

import logging
import os

import rl_common as rc
import rl_alpaca_bar_enrichment as enr
import rl_training_dataset_builder as builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rl_rebuild_enriched")


def main() -> None:
    events = rc.load_research_events() + rc.load_backfill_events()
    log.info("loaded %d events", len(events))

    # Enrich alpha events with GA daily technicals + regime (read-only Alpaca, cached).
    try:
        counts = enr.enrich_events(events, add_regime=True)
        log.info("enrichment counts: %s", counts)
    except Exception as exc:
        log.warning("enrichment failed (%s) — GA daily/regime stay neutral", exc)

    # Rebuild each dataset from the enriched events (mirrors build_all, chronological sort).
    builders = {
        "alpha_entry_dataset": builder.build_alpha_entry_dataset,
        "alpha_exit_dataset": builder.build_alpha_exit_dataset,
        "buffett_dca_dataset": builder.build_buffett_dca_dataset,
        "capital_allocation_dataset": builder.build_capital_allocation_dataset,
    }
    summary = {"generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(),
               "source_events": len(events), "enriched": True, "datasets": {}}
    for name, fn in builders.items():
        try:
            rows = fn(events)
            if os.getenv("RL_DATASET_SORT", "chrono").lower() not in ("position", "source", "0"):
                rows = sorted(rows, key=lambda r: str(r.get("date_tag", "")))
        except Exception as exc:
            log.warning("builder %s failed (%s)", name, exc)
            rows = []
        summary["datasets"][name] = builder.write_dataset(rows, name)
        log.info("%s: %d rows -> %s", name, summary["datasets"][name]["rows"],
                 summary["datasets"][name]["format"])
    rc.write_json(rc.RL_DATA_DIR / "dataset_summary.json", summary)
    print("ENRICHED REBUILD COMPLETE")
    for n, info in summary["datasets"].items():
        print(f"  {n}: {info['rows']} rows -> {info['path']}")


if __name__ == "__main__":
    main()
