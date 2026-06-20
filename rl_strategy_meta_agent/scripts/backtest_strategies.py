#!/usr/bin/env python3
"""Multi-symbol, walk-forward backtest of every executable PineScript strategy.

RESEARCH-ONLY. Reads cached/Alpaca bars, runs adapters, computes per-symbol +
aggregate + per-split metrics, decides eligibility, writes the performance DB,
and prints the scoreboard.

    python scripts/backtest_strategies.py --timeframe 1Day --start 2018-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.registry import load_strategies, write_registry
from rl_meta.types import StrategyMeta
from rl_meta.data import load_universe_bars, read_universe, TIMEFRAMES
from rl_meta.backtest import run_backtests, persist, population_summary
from rl_meta.ranking import rank_results
from rl_meta.performance import PerformanceDB
from rl_meta.adapters import annotate_executability
from rl_meta.correlation import report_from_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1Day", choices=sorted(TIMEFRAMES))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="cache-only; never fetch (no credentials needed)")
    args = ap.parse_args()

    metas = load_strategies()
    db = PerformanceDB()

    # Synthesize stub metas for TradingView strategies that exist ONLY as imports
    # (no pinescript .txt) so they flow through evaluation as tradingview_verified.
    existing = {m.strategy_id for m in metas}
    for sid, vh in db.imported_strategies(args.timeframe):
        if sid not in existing:
            metas.append(StrategyMeta(sid, sid, vh, "", "manifest",
                                      declares_strategy=True))
            existing.add(sid)
    if not metas:
        print("No strategies (no pinescripts and no imported trades).")
        return

    symbols = args.symbols or read_universe()
    bars = load_universe_bars(symbols, args.timeframe, args.start, args.end,
                              refresh=args.refresh, offline=args.offline)

    any_imports = any(db.imported_trades(m.strategy_id, args.timeframe,
                                         m.version_hash) for m in metas)
    if not bars and not any_imports:
        print("No bars and no imported TradingView trades. Run "
              "scripts/fetch_alpaca_bars.py (needs APCA creds) or "
              "scripts/import_tradingview_trades.py first.")
        return
    print(f"Loaded bars for {len(bars)}/{len(symbols)} symbols @ {args.timeframe}; "
          f"TradingView imports present: {any_imports}\n")

    # universe=None -> each strategy's coverage is judged on its own symbols
    # (TV strategies were exported on their own tickers, not the config universe).
    results = run_backtests(metas, bars, args.timeframe, db=db, universe=None)

    for m in metas:
        annotate_executability(m)

    # Independence diagnostic (population-phase): are these distinct edges?
    report = report_from_results(results, freq="monthly")
    by_id = {m.strategy_id: m for m in metas}
    for sid, row in report.per_strategy.items():
        if sid in by_id:
            by_id[sid].independence_score = row["independence_score"]
            by_id[sid].avg_correlation = row["avg_correlation"]

    # Registry carries executable/adapter_type/evaluation_source/family/independence.
    write_registry(metas)
    persist(db, results, on_date=date.today().isoformat())

    _population_dashboard(population_summary(metas, results, report))
    _scoreboard(results)
    _independence_section(report)
    print(f"\nperformance DB -> {db.path}")


def _population_dashboard(s):
    """Headline you watch as pinescripts/ grows: are you discovering edges or
    just adding clones?"""
    print("=== Population ===")
    extra = (f"   (+ {s['non_strategy_files']} indicator/study file(s) excluded)"
             if s["non_strategy_files"] else "")
    print(f"Strategies discovered: {s['discovered']}{extra}")
    print(f"Eligible (metrics):    {s['eligible']}")
    print(f"RL eligible:           {s['rl_eligible']}")
    print(f"Effective breadth:     {s['effective_breadth']} "
          f"(of {s['breadth_universe']} with trade history)\n")


_SRC_SHORT = {"tradingview_verified": "tv_verified",
              "adapter_screening": "adapter", "parsed_but_not_executable": "not_exec"}


def _scoreboard(results):
    ranked = rank_results(results)
    res_by_id = {r.strategy_id: r for r in results}
    print("=== Aggregate strategy ranking (single-source; no TV/adapter mixing) ===")
    print(f"{'#':>2}  {'strategy':<22} {'source':<12} {'tier':<6} {'trades':>6} "
          f"{'win%':>6} {'PF':>6} {'oosN':>5} {'score':>6} {'conf':>5}  status")
    print("-" * 112)
    for r in ranked:
        sr = res_by_id[r.strategy_id]
        m = r.metrics
        pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        oosN = sr.aggregate.get("oos").trade_count if sr.aggregate else 0
        src = _SRC_SHORT.get(sr.evaluation_source, sr.evaluation_source)
        if sr.rl_eligible:
            flag = "RL-ELIGIBLE"
        elif sr.eligible:
            flag = "SCREEN-ONLY (not RL)"
        else:
            flag = f"REJECT({r.reason})"
        # score = how good it looks; conf = how trustworthy the evidence is.
        print(f"{r.rank:>2}  {r.strategy_name[:22]:<22} {src:<12} {r.tier:<6} "
              f"{m.trade_count:>6} {m.win_rate*100:>5.1f}% {pf:>6} {oosN:>5} "
              f"{r.composite_score:>6.3f} {sr.strategy_confidence:>5.1f}  {flag}")
        if sr.evaluation_source == "tradingview_verified" and sr.missing_tv_symbols:
            print(f"      missing_tv_validation: {', '.join(sr.missing_tv_symbols)}")

    # Per-symbol breakdown for the top eligible strategy.
    top = next((r for r in ranked if r.eligible), None)
    if top:
        sr = res_by_id[top.strategy_id]
        print(f"\n=== Per-symbol metrics: {top.strategy_name} ===")
        print(f"{'symbol':<8} {'trades':>6} {'win%':>6} {'PF':>6} {'expR$':>8} "
              f"{'maxDD':>6} {'totRet%':>8}")
        print("-" * 56)
        for sym in sorted(sr.per_symbol):
            m = sr.per_symbol[sym]["full"]
            pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
            print(f"{sym:<8} {m.trade_count:>6} {m.win_rate*100:>5.1f}% {pf:>6} "
                  f"{m.expectancy:>8.1f} {m.max_drawdown*100:>5.1f}% "
                  f"{m.total_return*100:>7.1f}%")


def _independence_section(report):
    """Portfolio-level diversity diagnostic: are these distinct edges or clones?"""
    n = report.n_strategies
    print(f"\n=== Strategy independence ({report.freq}, min_overlap="
          f"{report.min_overlap}) ===")
    if n == 0:
        print("  no strategies with sufficient trade history")
        return
    print(f"Effective breadth: {report.effective_breadth} / {n} strategies "
          f"(1 = one edge cloned, {n} = all independent)")
    if report.insufficient:
        print(f"  insufficient data (excluded): {', '.join(report.insufficient)}")

    ranked = sorted(report.per_strategy.items(),
                    key=lambda kv: kv[1]["independence_score"])
    if len(ranked) > 1:
        print("Lowest independence (most clone-like):")
        for sid, row in ranked[:3]:
            print(f"  {sid:<24} avg_corr={row['avg_correlation']:.2f} "
                  f"family={row['family']}")
        print("Highest independence (most distinct):")
        for sid, row in reversed(ranked[-3:]):
            print(f"  {sid:<24} avg_corr={row['avg_correlation']:.2f} "
                  f"family={row['family']}")


if __name__ == "__main__":
    main()
