#!/usr/bin/env python3
"""Reorganize raw TradingView trade exports into the canonical structure.

Outcome (win/loss/flat) is derived from Net PnL inside each CSV, NOT from the
WINS/LOSSES/FLAT folder names (those are recorded for traceability only).

    # dry-run (default): print the mapping, write nothing
    python scripts/organize_tradingview_exports.py \
        --source pinescripts/trades --dest data/tradingview_exports --dry-run

    # apply: copy (default) into the canonical tree + write manifest.json
    python scripts/organize_tradingview_exports.py \
        --source pinescripts/trades --dest data/tradingview_exports --apply

Originals are preserved unless --move is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.export_organizer import build_plan, apply_plan, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pinescripts/trades")
    ap.add_argument("--dest", default="data/tradingview_exports")
    ap.add_argument("--dry-run", action="store_true", help="print mapping, write nothing (default)")
    ap.add_argument("--apply", action="store_true", help="actually copy/move files")
    ap.add_argument("--move", action="store_true", help="move instead of copy (default: copy)")
    args = ap.parse_args()

    apply = args.apply and not args.dry_run
    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}")
        return

    plan = build_plan(source, args.dest)
    print(f"Discovered {len(plan)} CSV(s) under {source}\n")
    _print_table(plan)

    unknown_sym = [r for r in plan if r["symbol"] is None]
    unknown_tf = [r for r in plan if r["timeframe"] == "unknown_timeframe"]
    unknown_range = [r for r in plan if r["date_range"] == "unknown_range"]

    if not apply:
        print(f"\n[DRY-RUN] nothing written. Re-run with --apply to organize.")
        _summary(plan, {"organized": 0, "skipped": 0, "collisions": 0},
                 unknown_sym, unknown_tf, unknown_range, args.dest, moved=False)
        return

    stats = apply_plan(plan, move=args.move)
    manifest = write_manifest(plan, args.dest)
    _summary(plan, stats, unknown_sym, unknown_tf, unknown_range, args.dest,
             moved=args.move, manifest=manifest)


def _print_table(plan):
    print(f"{'strategy_id':<20} {'sym':<6} {'tf':<10} {'range':<19} {'rows':>5} "
          f"{'trades':>6} {'netPnL':>9} {'label':<7} dest")
    print("-" * 130)
    for r in plan:
        sym = r["symbol"] or "_unknown_"
        dest_short = "/".join(Path(r["dest_path"]).parts[-4:])
        print(f"{r['strategy_id']:<20} {sym:<6} {r['timeframe']:<10} "
              f"{r['date_range']:<19} {r['row_count']:>5} {r['trade_count']:>6} "
              f"{r['total_net_pnl']:>9.2f} {(r['source_folder_label'] or '-'):<7} "
              f"{dest_short}")


def _summary(plan, stats, unknown_sym, unknown_tf, unknown_range, dest, moved,
             manifest=None):
    print("\n=== Summary ===")
    print(f"CSVs discovered:     {len(plan)}")
    print(f"Organized:           {stats['organized']}")
    print(f"Skipped (identical): {stats['skipped']}")
    print(f"Collisions (hashed): {stats['collisions']}")
    print(f"Unknown symbol:      {len(unknown_sym)}"
          + (f"  -> {[Path(r['source_path']).name for r in unknown_sym][:5]}"
             if unknown_sym else ""))
    print(f"Unknown timeframe:   {len(unknown_tf)}")
    print(f"Unknown date range:  {len(unknown_range)}")
    if manifest:
        print(f"Manifest:            {manifest}")
        examples = [r["dest_path"] for r in plan if r["copied_or_moved"] in
                    ("copied", "moved")][:3]
        for e in examples:
            print(f"  example: {e}")
    print(f"Originals preserved: {'NO (moved)' if moved else 'yes (copied)'}")


if __name__ == "__main__":
    main()
