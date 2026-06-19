#!/usr/bin/env python3
"""Import TradingView Strategy Tester "List of Trades" CSV exports.

These imported trades become the EVALUATION SOURCE OF TRUTH: when present, the
backtester uses them instead of the Python adapter approximation.

Single file (explicit mapping):
    python scripts/import_tradingview_trades.py path/to/export.csv \
        --strategy-id sample_strategy --symbol NVDA --timeframe 1Day

Bulk by filename convention '{strategy_id}__{symbol}__{timeframe}.csv':
    python scripts/import_tradingview_trades.py --dir data/tradingview_exports

The version_hash is resolved from data/strategy_registry.json (run
scripts/parse_pine_strategies.py first) so each import is pinned to the exact
PineScript version that produced it. Re-importing identical content is a no-op
(duplicate protection).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.registry import read_registry, file_hash
from rl_meta.tv_import import parse_tv_csv, parse_filename, EXPORTS_DIR
from rl_meta.performance import PerformanceDB, compute_metrics


def _version_hash(strategy_id: str, override: str | None) -> str | None:
    if override:
        return override
    for m in read_registry():
        if m.strategy_id == strategy_id:
            return m.version_hash
    return None


def _import_one(db, path: Path, strategy_id: str, symbol: str, timeframe: str,
                version_hash: str | None, force: bool) -> None:
    vh = _version_hash(strategy_id, version_hash)
    if vh is None:
        print(f"  {path.name}: SKIP -- strategy_id '{strategy_id}' not in registry "
              "(run parse_pine_strategies.py or pass --version-hash)")
        return
    trades = parse_tv_csv(path)
    if not trades:
        print(f"  {path.name}: SKIP -- no closed trades parsed")
        return
    content_hash = file_hash(path.read_text(encoding="utf-8-sig", errors="replace"))
    status = db.record_import(strategy_id, symbol, timeframe, vh, path.name,
                              content_hash, trades, date.today().isoformat(), force)
    m = compute_metrics(trades)
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    print(f"  {path.name}: {status:<9} {len(trades)} trades  "
          f"win {m.win_rate*100:.1f}%  PF {pf}  expR$ {m.expectancy:.1f}  "
          f"[{strategy_id}/{symbol}/{timeframe} @ {vh}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="single CSV file to import")
    ap.add_argument("--dir", default=None,
                    help="bulk-import every *.csv in this directory by filename")
    ap.add_argument("--strategy-id")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe")
    ap.add_argument("--version-hash", default=None,
                    help="override (default: resolved from registry)")
    ap.add_argument("--force", action="store_true",
                    help="re-import even if identical content was already imported")
    args = ap.parse_args()

    db = PerformanceDB()
    print("Importing TradingView List-of-Trades exports...\n")

    if args.dir:
        d = Path(args.dir)
        files = sorted(d.glob("*.csv"))
        if not files:
            print(f"No CSVs found in {d}")
            return
        for path in files:
            parsed = parse_filename(path.name)
            if not parsed:
                print(f"  {path.name}: SKIP -- name not "
                      "'{{strategy_id}}__{{symbol}}__{{timeframe}}.csv'")
                continue
            sid, sym, tf = parsed
            _import_one(db, path, sid, sym, tf, args.version_hash, args.force)
    elif args.csv:
        if not (args.strategy_id and args.symbol and args.timeframe):
            ap.error("single-file import needs --strategy-id, --symbol, --timeframe")
        _import_one(db, Path(args.csv), args.strategy_id, args.symbol,
                    args.timeframe, args.version_hash, args.force)
    else:
        ap.error("provide a CSV file or --dir")

    print(f"\nimported trades -> {db.path}")
    print("Next: scripts/backtest_strategies.py  (TradingView trades now override "
          "the adapter)")


if __name__ == "__main__":
    main()
