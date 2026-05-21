#!/usr/bin/env python3
"""
scripts/compare_strategy_overlap.py

Compare pairwise overlap between strategy variants:
  - Signal correlation (Pearson on binary buy/short vectors)
  - Overlapping trade symbols on the same bar
  - Potentially conflicting positions (one buys, another shorts same symbol)
  - Duplicate exposure (both strategies long the same symbol simultaneously)

Operates on replay_results.json signals or live execution logs.
Writes daily_system_outputs/strategy_overlap.json.

Usage:
  python scripts/compare_strategy_overlap.py [--date YYYY-MM-DD]
  python scripts/compare_strategy_overlap.py --use-replay   # use replay_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY, get_strategy

OUTPUT_DIR  = PROJECT_ROOT / "daily_system_outputs"
LOG_BASE    = PROJECT_ROOT / "logs" / "execution"


# ─────────────────────────────────────────────────────────────────────────────
# Build signal matrices from replay or synthetic run
# ─────────────────────────────────────────────────────────────────────────────

def _signals_from_replay(replay_path: Path) -> dict[str, dict[str, list[str]]]:
    """
    Returns {strategy_id: {symbol: [bar_timestamps where signal fired]}}.
    Reads replay_results.json.
    """
    if not replay_path.exists():
        return {}
    try:
        data = json.loads(replay_path.read_text())
    except Exception:
        return {}

    result: dict[str, dict[str, list[str]]] = {}
    for strat in data.get("strategies", []):
        sid = strat["strategy_id"]
        result[sid] = {}
        for sig in strat.get("signals", []):
            sym = sig["symbol"]
            if sym not in result[sid]:
                result[sid][sym] = []
            result[sid][sym].append(sig["first_bar"])
    return result


def _run_synthetic_comparison(n_bars: int = 250, seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Run all strategies on the same synthetic bars.
    Returns {strategy_id: signals_DataFrame}.
    """
    from scripts.smoke_test_variants import (
        _make_tux_bars, _make_generic_bars, _make_pivot_bars,
        _make_orb_bars, _make_sweep_bars, _make_fv_bars,
    )

    bar_map = {
        "tux_mtf_original":  {"bars": _make_tux_bars(), "context": {}},
        "tux_mtf_long_only": {"bars": _make_tux_bars(), "context": {}},
        "pivot_points":      _make_pivot_bars(),
        "orb_8am":           {"bars": _make_orb_bars(), "context": {}},
        "liquidity_sweep":   {"bars": _make_sweep_bars(), "context": {}},
        "flawless_victory":  _make_fv_bars(),
    }

    results: dict[str, pd.DataFrame] = {}
    for sid in STRATEGY_REGISTRY:
        spec = bar_map.get(sid)
        if spec is None:
            continue
        try:
            strategy = get_strategy(sid)
            signals = strategy.generate_signals(spec["bars"], spec.get("context", {}))
            results[sid] = signals
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Overlap analysis
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_correlation(signals_by_strategy: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Pearson correlation on binary buy/short vectors, same index."""
    strategy_ids = list(signals_by_strategy.keys())
    pairs = []

    for sid_a, sid_b in combinations(strategy_ids, 2):
        df_a = signals_by_strategy[sid_a]
        df_b = signals_by_strategy[sid_b]

        # Align on common index
        common = df_a.index.intersection(df_b.index)
        if len(common) < 10:
            corr_buy = corr_short = None
        else:
            a_buy   = df_a.loc[common, "buy_signal"].astype(float)
            b_buy   = df_b.loc[common, "buy_signal"].astype(float)
            a_short = df_a.loc[common, "short_signal"].astype(float)
            b_short = df_b.loc[common, "short_signal"].astype(float)

            corr_buy   = float(a_buy.corr(b_buy))   if a_buy.std() > 0 and b_buy.std() > 0   else 0.0
            corr_short = float(a_short.corr(b_short)) if a_short.std() > 0 and b_short.std() > 0 else 0.0

        # Count overlapping signal bars
        overlap_buy = int((df_a.loc[common, "buy_signal"].astype(bool) &
                           df_b.loc[common, "buy_signal"].astype(bool)).sum()) if len(common) else 0
        overlap_short = int((df_a.loc[common, "short_signal"].astype(bool) &
                             df_b.loc[common, "short_signal"].astype(bool)).sum()) if len(common) else 0

        # Conflicting: one buys, other shorts on the same bar
        conflicts = int((df_a.loc[common, "buy_signal"].astype(bool) &
                         df_b.loc[common, "short_signal"].astype(bool)).sum() +
                        (df_a.loc[common, "short_signal"].astype(bool) &
                         df_b.loc[common, "buy_signal"].astype(bool)).sum()) if len(common) else 0

        pairs.append({
            "strategy_a": sid_a,
            "strategy_b": sid_b,
            "corr_buy": round(corr_buy, 4) if corr_buy is not None else None,
            "corr_short": round(corr_short, 4) if corr_short is not None else None,
            "overlap_buy_bars": overlap_buy,
            "overlap_short_bars": overlap_short,
            "conflicting_bars": conflicts,
            "common_bars": len(common),
        })

    return pairs


def _symbol_overlap_from_replay(replay_signals: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
    """Find symbols traded by multiple strategies on the same timestamp."""
    # Build: {(symbol, timestamp): [strategy_ids that fired]}
    bar_signal_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid, sym_map in replay_signals.items():
        for sym, timestamps in sym_map.items():
            for ts in timestamps:
                bar_signal_map[(sym, ts)].append(sid)

    overlaps = []
    for (sym, ts), strats in bar_signal_map.items():
        if len(strats) > 1:
            overlaps.append({"symbol": sym, "timestamp": ts, "strategies": strats})

    return sorted(overlaps, key=lambda x: x["timestamp"])


def _duplicate_exposure_risk(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag strategy pairs with high correlation (>0.5) as duplicate exposure risk."""
    risks = []
    for p in pairs:
        corr = p.get("corr_buy")
        if corr is not None and abs(corr) > 0.5:
            risks.append({
                "pair": f"{p['strategy_a']} + {p['strategy_b']}",
                "correlation": corr,
                "risk": "high" if abs(corr) > 0.75 else "moderate",
            })
    return risks


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_overlap_analysis(
    trade_date: date,
    use_replay: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    replay_path = OUTPUT_DIR / "replay_results.json"
    replay_signals = _signals_from_replay(replay_path)

    if verbose:
        source = "replay_results.json" if replay_signals else "synthetic bars"
        print(f"[overlap] Analysing on: {source}")

    # Always run synthetic comparison for correlation matrix (same index = fair comparison)
    synthetic_signals = _run_synthetic_comparison()

    pairs = _pairwise_correlation(synthetic_signals)
    symbol_overlaps = _symbol_overlap_from_replay(replay_signals)
    dup_risks = _duplicate_exposure_risk(pairs)

    if verbose:
        print(f"\nPairwise signal correlation (synthetic bars, same index):")
        for p in pairs:
            corr_str = f"corr_buy={p['corr_buy']}" if p["corr_buy"] is not None else "corr_buy=n/a"
            conflict_str = f" CONFLICT={p['conflicting_bars']}" if p["conflicting_bars"] > 0 else ""
            print(f"  {p['strategy_a']} × {p['strategy_b']}: {corr_str}  "
                  f"overlap_buy={p['overlap_buy_bars']}{conflict_str}")

        if dup_risks:
            print(f"\nDuplicate exposure risks:")
            for r in dup_risks:
                print(f"  [{r['risk'].upper()}] {r['pair']}  corr={r['correlation']}")
        else:
            print("\nNo duplicate exposure risks detected.")

        if symbol_overlaps:
            print(f"\nSymbol bar overlaps from replay: {len(symbol_overlaps)} bars")
            for o in symbol_overlaps[:5]:
                print(f"  {o['symbol']} @ {o['timestamp']}: {', '.join(o['strategies'])}")
        else:
            print("\nNo symbol bar overlaps found in replay data.")

    summary = {
        "date": str(trade_date),
        "pairwise_correlation": pairs,
        "symbol_bar_overlaps": symbol_overlaps,
        "duplicate_exposure_risks": dup_risks,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strategy signal overlap")
    parser.add_argument("--date", default=None, help="Trade date YYYY-MM-DD")
    parser.add_argument("--use-replay", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.date:
        trade_date = date.fromisoformat(args.date)
    else:
        today = date.today()
        offset = max(1, today.weekday() - 4) if today.weekday() >= 5 else 1
        trade_date = today - timedelta(days=offset)

    if not args.quiet:
        print(f"=== Strategy Overlap Analysis: {trade_date} ===\n")

    summary = run_overlap_analysis(trade_date, use_replay=args.use_replay, verbose=not args.quiet)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "strategy_overlap.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
