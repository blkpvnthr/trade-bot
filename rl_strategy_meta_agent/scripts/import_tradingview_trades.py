#!/usr/bin/env python3
"""Import TradingView Strategy Tester "List of Trades" CSV exports.

Imported trades are the EVALUATION SOURCE OF TRUTH. Three input modes:

  --manifest data/tradingview_exports/manifest.json   (canonical, preferred)
  --dir DIR    (recurses; flat 3-part names or nested {id}/{sym}/{tf}/*.csv)
  CSV --strategy-id ... --symbol ... --timeframe ...   (single file)

Position-size normalization (default ON): TradingView backtests often use
unrealistic share counts (500-5000). Trades are normalized to a reference
quantity so strategies are compared on roughly equal risk -- dollar PnL is
scaled, percentage returns / direction / timing are NOT. Use --no-normalize for
raw mode. Outcomes are derived from Net PnL, never from WINS/LOSSES/FLAT labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.registry import read_registry, file_hash
from rl_meta.tv_import import (parse_tv_csv, parse_canonical, normalize_trades,
                              REFERENCE_QTY)
from rl_meta.performance import PerformanceDB


def _resolve_vh(strategy_id: str, override: str | None) -> str:
    """version_hash: explicit > registry > a stable synthetic hash so strategies
    that exist ONLY as TradingView exports (no pinescript .txt) still import."""
    if override:
        return override
    for m in read_registry():
        if m.strategy_id == strategy_id:
            return m.version_hash
    return "tv_" + hashlib.sha256(strategy_id.encode()).hexdigest()[:12]


def _import_one(db, path, sid, symbol, tf, vh, content_hash, normalize,
                ref_qty, force, stats):
    if not Path(path).exists():
        print(f"  MISSING: {path}")
        stats["errors"] += 1
        return
    try:
        trades = parse_tv_csv(path)
    except Exception as e:
        print(f"  {Path(path).name}: ERROR -- {e}")
        stats["errors"] += 1
        return
    if not trades:
        print(f"  {Path(path).name}: SKIP -- no closed trades")
        stats["skipped"] += 1
        return

    normalize_trades(trades, ref_qty, enabled=normalize)
    if content_hash is None:
        content_hash = file_hash(Path(path).read_text(encoding="utf-8-sig",
                                                      errors="replace"))
    status = db.record_import(sid, symbol, tf, vh, Path(path).name, content_hash,
                              trades, date.today().isoformat(), force)
    if status == "duplicate":
        stats["skipped"] += 1
        return

    stats["imported"] += 1
    for t in trades:
        stats["n_trades"] += 1
        if t.scale_factor != 1.0:
            stats["n_normalized"] += 1
        stats["scale_sum"] += t.scale_factor
        stats["max_raw_qty"] = max(stats["max_raw_qty"], t.raw_qty)
        stats["max_norm_qty"] = max(stats["max_norm_qty"], t.normalized_qty)
        stats["raw_pnl"] += t.raw_pnl_usd
        stats["norm_pnl"] += t.pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="single CSV file")
    ap.add_argument("--manifest", default=None, help="manifest.json to ingest")
    ap.add_argument("--dir", default=None, help="directory (recursed)")
    ap.add_argument("--strategy-id")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe")
    ap.add_argument("--version-hash", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--normalize-qty", dest="normalize", action="store_true",
                    default=True, help="normalize position sizing (default)")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--reference-qty", type=float, default=REFERENCE_QTY)
    ap.add_argument("--manifest-timeframe", default="15Min",
                    help="only import manifest records on this timeframe "
                         "('all' to import every timeframe)")
    args = ap.parse_args()

    db = PerformanceDB()
    stats = {"imported": 0, "skipped": 0, "errors": 0, "n_trades": 0,
             "n_normalized": 0, "scale_sum": 0.0, "max_raw_qty": 0.0,
             "max_norm_qty": 0.0, "raw_pnl": 0.0, "norm_pnl": 0.0, "labels": 0}
    mode = "normalized" if args.normalize else "RAW"
    print(f"Importing TradingView exports ({mode}, reference_qty="
          f"{args.reference_qty:g})...\n")

    if args.manifest:
        recs = json.loads(Path(args.manifest).read_text()).get("records", [])
        base = Path(args.manifest).resolve().parent
        today = date.today().isoformat()
        for r in recs:
            tf = r["timeframe"]
            if args.manifest_timeframe != "all" and tf != args.manifest_timeframe:
                stats["skipped"] += 1
                continue
            csv = Path(r["dest_path"])
            if not csv.exists():
                csv = base / Path(r["dest_path"]).name  # fallback
            sym = r.get("symbol") or "UNKNOWN"
            vh = _resolve_vh(r["strategy_id"], args.version_hash)
            _import_one(db, csv, r["strategy_id"], sym, tf, vh,
                        r.get("file_hash"), args.normalize, args.reference_qty,
                        args.force, stats)
            # Preserve the human WINS/LOSSES/FLAT prior as metadata (not outcome).
            from rl_meta.labels import normalize_bucket
            db.record_label(r["strategy_id"], sym, tf,
                            normalize_bucket(r.get("source_folder_label")),
                            Path(r["dest_path"]).name, today)
            stats["labels"] += 1
    elif args.dir:
        files = sorted(Path(args.dir).rglob("*.csv"))
        if not files:
            print(f"No CSVs found under {args.dir}")
            return
        for path in files:
            parsed = parse_canonical(path)
            if not parsed:
                print(f"  {path.name}: SKIP -- cannot resolve id/symbol/timeframe")
                stats["skipped"] += 1
                continue
            sid, sym, tf = parsed
            vh = _resolve_vh(sid, args.version_hash)
            _import_one(db, path, sid, sym, tf, vh, None, args.normalize,
                        args.reference_qty, args.force, stats)
    elif args.csv:
        if not (args.strategy_id and args.symbol and args.timeframe):
            ap.error("single-file import needs --strategy-id, --symbol, --timeframe")
        vh = _resolve_vh(args.strategy_id, args.version_hash)
        _import_one(db, Path(args.csv), args.strategy_id, args.symbol,
                    args.timeframe, vh, None, args.normalize, args.reference_qty,
                    args.force, stats)
    else:
        ap.error("provide --manifest, --dir, or a CSV file")

    _report(stats, db.path, args.normalize)


def _report(s, db_path, normalized):
    n = s["n_trades"]
    avg_sf = (s["scale_sum"] / n) if n else 1.0
    print("\n=== Import summary ===")
    print(f"Files imported:        {s['imported']}")
    print(f"Files skipped:         {s['skipped']}  (duplicate / no trades)")
    print(f"Errors:                {s['errors']}")
    print(f"Human-bucket labels:   {s['labels']}  (WINS/LOSSES/FLAT metadata)")
    print(f"Trades imported:       {n}")
    print(f"Trades normalized:     {s['n_normalized']}  (raw_qty > reference)")
    print(f"Average scale factor:  {avg_sf:.4f}")
    print(f"Largest raw qty:       {s['max_raw_qty']:.0f}")
    print(f"Largest normalized qty:{s['max_norm_qty']:.1f}")
    print(f"Raw total PnL:         {s['raw_pnl']:,.2f}")
    print(f"Normalized total PnL:  {s['norm_pnl']:,.2f}  "
          f"({'used for metrics' if normalized else 'RAW MODE'})")
    print(f"\nimported trades -> {db_path}")
    print("Next: scripts/backtest_strategies.py --timeframe 15Min")


if __name__ == "__main__":
    main()
