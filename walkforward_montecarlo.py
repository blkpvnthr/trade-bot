#!/usr/bin/env python3
"""Walk-forward Monte Carlo for the run.py alpha strategy.

run.py is the live/paper orchestrator and has no backtest entry point; its
strategy is replayed historically by backtest_alpha_replay.py (same engine
oos_validate.py / bc_runs.py drive). This script:

  1. Walk-forward: replays the strategy over several sequential historical
     windows on a universe file (default tickers4.txt), one FIXED config.
  2. Monte Carlo: daily-PnL bootstrap — resamples the observed per-day realized
     PnL (day_realized_pnl in daily.csv) with replacement to build a
     distribution of $/day, total$, drawdown, prob(>0), prob(>=target/day).

Windows where the replay produces no output (no data / engine failure) are
skipped and reported in a coverage line. Reproducible: fixed RNG seed.

Usage:
    python walkforward_montecarlo.py                 # tickers4.txt, default config
    python walkforward_montecarlo.py tickers2.txt    # different universe
    python walkforward_montecarlo.py --trials 5000 --target 450
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/Users/blkpvnthr/Desktop/trader"
PY = sys.executable

# Sequential walk-forward windows (same set oos_validate.py uses).
WINDOWS = [
    ("2026-01-20", "2026-02-12", "oos-4"),
    ("2026-02-17", "2026-03-12", "oos-3"),
    ("2026-03-16", "2026-04-09", "oos-2"),
    ("2026-04-13", "2026-05-08", "oos-1"),
    ("2026-05-12", "2026-06-04", "latest"),
]

# Fixed config under test (matches acct_tickers4_noovn: $500/trade, 15% expo, costs ON).
FIXED_CONFIG = ["--risk-dollars", "500", "--exposure-pct", "0.15", "--costs"]


def parse_universe(path):
    raw = open(path).read()
    return [t.strip().strip('"').upper() for t in raw.replace("\n", ",").split(",")
            if t.strip().strip('"')]


def run_window(symbols, start, end, tag, extra_config):
    """Run the replay engine for one window; return (summary, daily_pnl_list) or (None, err)."""
    args = [PY, "backtest_alpha_replay.py", "--symbols", symbols,
            "--start", start, "--end", end, *FIXED_CONFIG, *extra_config,
            "--quiet", "--run-tag", tag]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    sp = os.path.join(ROOT, "backtest_outputs", tag, "summary.json")
    dp = os.path.join(ROOT, "backtest_outputs", tag, "daily.csv")
    if not (os.path.exists(sp) and os.path.exists(dp)):
        return None, (r.stderr.strip()[-160:] or "no output produced")
    summary = json.load(open(sp))
    daily = []
    with open(dp) as f:
        header = f.readline().strip().split(",")
        idx = header.index("day_realized_pnl")
        for line in f:
            cells = line.rstrip("\n").split(",")
            if len(cells) > idx and cells[idx] != "":
                daily.append(float(cells[idx]))
    return summary, daily


def max_drawdown_pct(equity_path):
    """Max peak-to-trough drawdown (%) of an equity curve array."""
    peak = np.maximum.accumulate(equity_path)
    dd = (peak - equity_path) / peak
    return float(np.max(dd) * 100.0)


def mc_daily_bootstrap(daily_pnl, start_equity, trials, target, rng):
    """Resample observed daily PnL with replacement; return distribution dict."""
    arr = np.asarray(daily_pnl, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    # (trials, n) matrix of resampled days
    samples = rng.choice(arr, size=(trials, n), replace=True)
    totals = samples.sum(axis=1)
    per_day = totals / n
    # equity paths for drawdown: start_equity + cumulative resampled PnL
    cum = start_equity + np.cumsum(samples, axis=1)
    cum = np.concatenate([np.full((trials, 1), start_equity), cum], axis=1)
    peaks = np.maximum.accumulate(cum, axis=1)
    dd = np.max((peaks - cum) / peaks, axis=1) * 100.0
    return {
        "n_days": n,
        "per_day_median": float(np.median(per_day)),
        "per_day_p5": float(np.percentile(per_day, 5)),
        "per_day_p95": float(np.percentile(per_day, 95)),
        "total_median": float(np.median(totals)),
        "prob_profit": float(np.mean(per_day > 0) * 100.0),
        "prob_target": float(np.mean(per_day >= target) * 100.0),
        "dd_median": float(np.median(dd)),
        "dd_p95": float(np.percentile(dd, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("universe", nargs="?", default="tickers4.txt")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--target", type=float, default=450.0, help="$/day target for prob")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--force-overnight", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    extra = ["--force-overnight"] if args.force_overnight else []

    upath = args.universe if os.path.isabs(args.universe) else os.path.join(ROOT, args.universe)
    stem = os.path.splitext(os.path.basename(upath))[0]
    names = parse_universe(upath)
    syms = ",".join(names)

    print("=" * 100)
    print(f"WALK-FORWARD MONTE CARLO  (run.py strategy via backtest_alpha_replay)")
    print(f"  universe : {os.path.basename(upath)}  ({len(names)} names)")
    print(f"  config   : {' '.join(FIXED_CONFIG + extra)}")
    print(f"  MC       : daily-PnL bootstrap, {args.trials} trials/window, "
          f"target ${args.target:.0f}/day, seed {args.seed}")
    print("=" * 100)

    hdr = (f"{'window':>22} {'label':>7} | {'days':>4} {'act$/d':>7} "
           f"| {'MC$/d med':>9} {'p5':>6} {'p95':>6} {'P(>0)':>6} {'P(>=t)':>7} "
           f"{'DDmed%':>7} {'DDp95%':>7}")
    print(hdr)
    print("-" * len(hdr))

    pooled = []          # all observed daily PnL across covered windows
    pooled_eq = []       # (start_equity) used; take first window's
    covered = skipped = 0
    skip_notes = []

    for st, en, label in WINDOWS:
        tag = f"wfmc_{stem}_{st}_{en}".replace("-", "")
        summary, daily = run_window(syms, st, en, tag, extra)
        if summary is None:
            skipped += 1
            skip_notes.append(f"  {st}..{en} ({label}): SKIPPED — {daily}")
            print(f"{st+'..'+en:>22} {label:>7} | SKIPPED ({daily[:50]})")
            continue
        covered += 1
        start_eq = float(summary.get("start_equity", 6500.0))
        if not pooled_eq:
            pooled_eq.append(start_eq)
        pooled.extend(daily)
        mc = mc_daily_bootstrap(daily, start_eq, args.trials, args.target, rng)
        act = summary.get("avg_realized_pnl_per_day", float("nan"))
        print(f"{st+'..'+en:>22} {label:>7} | {summary['trading_days']:>4} {act:>7.0f} "
              f"| {mc['per_day_median']:>9.0f} {mc['per_day_p5']:>6.0f} {mc['per_day_p95']:>6.0f} "
              f"{mc['prob_profit']:>5.0f}% {mc['prob_target']:>6.0f}% "
              f"{mc['dd_median']:>7.2f} {mc['dd_p95']:>7.2f}")

    print("-" * len(hdr))
    if pooled:
        start_eq = pooled_eq[0]
        mc = mc_daily_bootstrap(pooled, start_eq, args.trials, args.target, rng)
        print(f"{'POOLED (all days)':>22} {'all':>7} | {mc['n_days']:>4} "
              f"{np.mean(pooled):>7.0f} "
              f"| {mc['per_day_median']:>9.0f} {mc['per_day_p5']:>6.0f} {mc['per_day_p95']:>6.0f} "
              f"{mc['prob_profit']:>5.0f}% {mc['prob_target']:>6.0f}% "
              f"{mc['dd_median']:>7.2f} {mc['dd_p95']:>7.2f}")
        print()
        print(f"  Coverage: {covered}/{len(WINDOWS)} windows ran, {skipped} skipped.")
        print(f"  Pooled observed: {len(pooled)} trading days, "
              f"mean ${np.mean(pooled):.1f}/day, total ${np.sum(pooled):.0f}")
        print(f"  Bootstrap $/day 90% CI: "
              f"[${mc['per_day_p5']:.0f}, ${mc['per_day_p95']:.0f}]  "
              f"| P(profitable day-avg)={mc['prob_profit']:.0f}%  "
              f"P(>=${args.target:.0f}/day)={mc['prob_target']:.0f}%")
    else:
        print("  No windows produced output — nothing to Monte Carlo.")

    for note in skip_notes:
        print(note)
    print("\nWFMC_DONE")


if __name__ == "__main__":
    main()
