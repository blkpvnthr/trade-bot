#!/usr/bin/env python3
"""
backtest_vol_gated_insider.py — the VALIDATED edge, hardened: insider-buy SHARES gated on
ENTRY-DAY VOLATILITY (the OOS-surviving filter), with a liquidity filter + slippage sweep.

The discrete KMeans-regime gate FAILED OOS (regime_insider_oos.md). The continuous volatility
relationship PASSED (Q5 +5.53% vs Q1 −0.40%, net 1% slip). This needs NO PCA/KMeans — just
realized_vol_20 percentile. Here it's hardened:
  - OOS: vol thresholds + (optional) any fit use TRAIN only; metrics on TEST (>= train_end).
  - Liquidity filter: avg dollar-volume(20d) >= min_adv → so fixed slippage is realistic.
  - Slippage sweep: 50 / 100 / 150 bps round-trip.
  - No lookahead: vol/ADV computed from bars up to the signal day; entry next open, exit +hold close.
Reports unfiltered vs vol-gated per slippage: n, mean, median, win%, Sharpe, PF, MaxDD, edge.
"""
from __future__ import annotations

import argparse
import statistics
from datetime import date as _date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab


def _collect(hold, min_adv):
    """All insider-entry trades with entry-day vol + ADV (no lookahead). Returns list of dicts."""
    pit = base.load_pit_buy_events()
    rows = []
    for sym in sorted(pit.keys()):
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        op = [float(bars.loc[d, "open"]) for d in td]
        cl = [float(bars.loc[d, "close"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        r1 = np.diff(np.log(np.maximum(cl, 1e-9)))
        td_dates = [d.date() if hasattr(d, "date") else d for d in td]
        for i in ab._insider_entry_indices(td_dates, sym, pit):
            if i < 25 or i + 1 + hold >= len(td):
                continue
            rv20 = float(np.std(r1[i - 20:i])) if i >= 21 else np.nan      # entry-day realized vol
            adv = float(np.mean([cl[k] * vol[k] for k in range(i - 20, i)]))  # 20d $ volume
            if not np.isfinite(rv20) or adv < min_adv:
                continue
            entry = op[i + 1]; exit_px = cl[i + 1 + hold]
            if entry <= 0 or exit_px <= 0:
                continue
            rows.append({"sym": sym, "date": td_dates[i + 1], "gross": exit_px / entry - 1.0,
                         "rv20": rv20, "adv": adv})
    return rows


def _stat(rows, slip):
    if not rows:
        return None
    v = np.array([r["gross"] - slip for r in rows])
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    # MaxDD on chronological 10%-fixed-fraction equity (approx; overlapping trades sequential)
    eq = peak = 1.0; mdd = 0.0
    for r in sorted(rows, key=lambda x: x["date"]):
        eq *= (1 + 0.10 * (r["gross"] - slip))
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)), "win": (v > 0).mean(),
            "sharpe": (v.mean() / sd) if sd > 0 else 0.0, "pf": pf, "mdd": mdd}


def _fmt(s):
    if not s:
        return "| — | 0 |—|—|—|—|—|—|"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {s['n']:,} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | {s['win']*100:.0f}% | "
            f"{s['sharpe']:+.3f} | {pf} | {s['mdd']*100:.0f}% |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-01-01")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--vol-pctile", type=float, default=80, help="trade if entry vol >= this train pctile")
    ap.add_argument("--min-adv", type=float, default=1e6, help="min 20d avg $-volume (liquidity)")
    args = ap.parse_args()

    print(f"Collecting insider trades (hold={args.hold}d, min ADV=${args.min_adv:,.0f})...")
    rows = _collect(args.hold, args.min_adv)
    te = _date.fromisoformat(args.train_end)
    train = [r for r in rows if r["date"] < te]
    test = [r for r in rows if r["date"] >= te]
    if not train or not test:
        print("insufficient data"); return
    vthr = float(np.percentile([r["rv20"] for r in train], args.vol_pctile))   # threshold from TRAIN
    print(f"  {len(rows):,} trades (train {len(train):,} / test {len(test):,}); vol P{args.vol_pctile:.0f} = {vthr:.4f}")

    test_hi = [r for r in test if r["rv20"] >= vthr]      # OOS vol-gated
    test_lo = [r for r in test if r["rv20"] < vthr]

    L = [f"# Vol-Gated Insider SHARES — OOS + Liquidity + Slippage Sweep (hold {args.hold}d)", "",
         f"Insider-buy shares, gated on entry-day realized_vol_20 ≥ TRAIN P{args.vol_pctile:.0f} "
         f"({vthr:.4f}). Liquidity filter: 20d avg $-vol ≥ ${args.min_adv:,.0f}. Train <{args.train_end}, "
         f"test ≥. No lookahead. Hold {args.hold}d, enter next open / exit close.", "",
         f"Trades: {len(rows):,} total, OOS test {len(test):,} ({len(test_hi):,} high-vol gated, "
         f"{len(test_lo):,} low-vol excluded).", ""]
    for slip_bps in (50, 100, 150):
        slip = slip_bps / 10000.0
        L += [f"## Round-trip slippage {slip_bps} bps", "",
              "| Set | n | Mean | Median | Win% | Sharpe | PF | MaxDD |", "|---|---|---|---|---|---|---|---|"]
        L.append("| OOS unfiltered " + _fmt(_stat(test, slip)))
        L.append("| **OOS vol-gated (high vol)** " + _fmt(_stat(test_hi, slip)))
        L.append("| OOS excluded (low vol) " + _fmt(_stat(test_lo, slip)))
        u, g = _stat(test, slip), _stat(test_hi, slip)
        if u and g:
            L.append(f"\n_Gate lift: **{(g['mean']-u['mean'])*100:+.2f}%** mean, PF {g['pf']:.2f} vs {u['pf']:.2f}._")
        L.append("")
    # verdict at 100 bps
    g100 = _stat(test_hi, 0.010); u100 = _stat(test, 0.010)
    ok = bool(g100 and g100["pf"] >= 2.0 and g100["mean"] > (u100["mean"] if u100 else 0) and g100["n"] >= 30)
    L += ["## VERDICT (at 100 bps)", "",
          f"- Vol-gated PF ≥ 2.0: {'YES' if g100 and g100['pf']>=2.0 else 'NO'} (PF {g100['pf']:.2f})" if g100 else "- n/a",
          f"- Beats unfiltered: {'YES' if g100 and u100 and g100['mean']>u100['mean'] else 'NO'}",
          f"- OOS gated n ≥ 30: {'YES' if g100 and g100['n']>=30 else 'NO'} (n={g100['n'] if g100 else 0})",
          "",
          ("**PASS → vol-gated insider shares is a paper live-candidate. Next: wire a VOLATILITY gate "
           "(not KMeans) + liquidity filter into run_insider_trade_copier (shares), keep risk gates, "
           "kill if live PF<1.8. NO options overlay.**" if ok else
           "**MARGINAL — tighten the vol percentile / liquidity filter or widen sample before live.**")]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "vol_gated_insider_oos.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
