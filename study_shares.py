#!/usr/bin/env python3
"""
study_shares.py — THE clean test: the production entry signal on SHARES, no options.

Every prior study was Signal → option wrapper → friction (spread/IV/theta/skew/commission) →
outcome. This removes the wrapper entirely: Signal → Stock → Exit. If the signal can't make
money in shares, no option structure ever will, and the search should move to a better directional
edge. If it CAN, then options merely destroyed it and the question becomes efficient expression.

Entry: current production momentum signal (breakout>20d-high & >SMA50 & rel-vol>1.5), enter
next-day OPEN. Exit: fixed N-day hold at close (the horizon all option studies used). Fixed 1R
(per-trade % return). Liquid mega-caps. Benchmark: UNCONDITIONAL forward return on ALL days
(same horizon) — the signal only has edge if it beats simply being long.

Metrics per horizon: n, mean, median, Sharpe, win%, MaxDD, profit factor.
Saves backtest_outputs/shares_signal.md
"""
from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List

import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab

UNIVERSE = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "AVGO", "TSLA", "AMD", "PLTR", "QQQ", "SPY"]
HORIZONS = [3, 5, 10, 20]


def _mdd(returns_in_order, frac=1.0):
    eq = peak = 1.0; mdd = 0.0
    for r in returns_in_order:
        eq *= (1.0 + frac * r)
        if eq <= 0:
            return -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return mdd


def _stat(rows):
    """rows = [(date, ret)] chronological-sortable."""
    if not rows:
        return None
    v = [r[1] for r in rows]; n = len(v); m = statistics.mean(v)
    sd = statistics.pstdev(v) if n > 1 else 0.0
    wins = [x for x in v if x > 0]; losses = [x for x in v if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    mdd = _mdd([r[1] for r in sorted(rows)])
    return {"n": n, "mean": m, "median": statistics.median(v), "win": len(wins) / n,
            "sharpe": (m / sd) if sd > 0 else 0.0, "mdd": mdd, "pf": pf}


def run():
    # signal[h] and allday[h] = list of (date, fwd_return)
    sig = {h: [] for h in HORIZONS}; allday = {h: [] for h in HORIZONS}
    for sym in UNIVERSE:
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        op = [float(bars.loc[d, "open"]) for d in td]
        close = [float(bars.loc[d, "close"]) for d in td]
        high = [float(bars.loc[d, "high"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        cd = {d: close[k] for k, d in enumerate(td)}
        hd = {d: high[k] for k, d in enumerate(td)}
        vd = {d: vol[k] for k, d in enumerate(td)}
        ents = set(ab._momentum_entry_indices(td, cd, hd, vd))
        for i in range(60, len(td)):
            for h in HORIZONS:
                if i + 1 + h >= len(td):
                    continue
                entry_px = op[i + 1]; exit_px = close[i + 1 + h]   # next-day open → close at horizon
                if entry_px <= 0 or exit_px <= 0:
                    continue
                ret = exit_px / entry_px - 1.0
                allday[h].append((td[i + 1].isoformat(), ret))
                if i in ents:
                    sig[h].append((td[i + 1].isoformat(), ret))
    return sig, allday


def report(sig, allday):
    L = ["# Production Signal on SHARES — the clean no-options test", "",
         "Momentum signal → next-day OPEN → close at horizon. NO options/greeks/IV/theta/spread. "
         f"Universe: {', '.join(UNIVERSE)}. Benchmark = UNCONDITIONAL forward return on ALL days "
         "(same horizon) — the signal has edge ONLY if it beats simply being long.", ""]
    for h in HORIZONS:
        s = _stat(sig[h]); a = _stat(allday[h])
        L += [f"## {h}-day hold", "",
              "| Set | n | Mean | Median | Sharpe | Win% | MaxDD | PF |",
              "|---|---|---|---|---|---|---|---|"]
        for name, st in (("Signal entries", s), ("All-days benchmark", a)):
            if not st:
                L.append(f"| {name} | 0 |—|—|—|—|—|—|"); continue
            pf = "∞" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
            L.append(f"| **{name}** | {st['n']} | {st['mean']*100:+.2f}% | {st['median']*100:+.2f}% | "
                     f"{st['sharpe']:+.3f} | {st['win']*100:.0f}% | {st['mdd']*100:.0f}% | {pf} |")
        # edge = signal mean − all-days mean
        if s and a:
            edge = (s["mean"] - a["mean"]) * 100
            L.append("")
            L.append(f"_Signal edge over unconditional: **{edge:+.2f}%** mean per {h}-day trade "
                     f"(signal {s['mean']*100:+.2f}% vs market {a['mean']*100:+.2f}%)._")
        L.append("")
    L += ["## Interpretation", "",
          "- Signal PF>1 & mean>0 & beats benchmark → the signal has edge; **options destroyed it**.",
          "- Signal PF<1 / mean<0 / ≈benchmark → the **signal itself is weak** (stop optimizing option structures).",
          "_Fixed 1R = raw per-trade share return. MaxDD on chronological 100%-allocation equity._"]
    return "\n".join(L)


def main():
    argparse.ArgumentParser().parse_args()
    print("Running production signal on shares (local bars, no fetches)...")
    sig, allday = run()
    md = report(sig, allday)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "shares_signal.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
