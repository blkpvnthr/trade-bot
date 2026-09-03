#!/usr/bin/env python3
"""
study_liq_sweep.py — backtest the Liquidity-Sweep Trend+Delta signal as the options ENTRY signal.

Signal computed on the UNDERLYING bars (liquidity_sweep_signal). Option leg = ATM call δ0.45-0.65,
spread≤10%, real AV chain, enter ASK / exit BID + commission. Liquid mega-caps.

Arms:
  1. ATM baseline               — momentum entries, 10-day hold (the established control)
  2. ATM + buySignal            — enter on liq-sweep buySignal, 10-day hold
  3. ATM + buySignal + sellExit — enter on buySignal, exit at first sellSignal (≤25d cap)

Reports mean/median/Sharpe/win/MaxDD/trade-count per arm.
Usage: python study_liq_sweep.py --max-entries 300 --len 12 --mult 2.0
"""
from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional

import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo
import study_atm_filters as atm
import liquidity_sweep_signal as lss

HOLD = 10
SELL_CAP = 25     # max trading-day hold for the sellSignal-exit arm (keep the option alive)


def _atm_ret(sym, ed, xd, S1):
    chain = avo.get_chain(sym, ed, calls_only=True)
    c = atm._atm_in_bucket(chain, 21, 49)
    if not c:
        return None
    return atm._realized(c, sym, xd, S1)


def run(max_entries: int, length: int, mult: float) -> Dict[str, List[dict]]:
    arms = {"baseline": [], "buySignal": [], "buySignal+sellExit": []}
    nb = 0
    for sym in atm.LIQUID:
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        # baseline: momentum entries (control)
        for i in ab._momentum_entry_indices(td, close, high, vol):
            if i + 1 + HOLD >= len(td):
                continue
            ed = td[i + 1].isoformat(); xd = td[i + 1 + HOLD].isoformat()
            S1 = float(bars.loc[td[i + 1 + HOLD], "close"])
            if S1 <= 0:
                continue
            r = _atm_ret(sym, ed, xd, S1)
            if r is not None:
                arms["baseline"].append({"sym": sym, "date": ed, "ret": r})
        # liquidity-sweep buy/sell signals
        sig = lss.liquidity_sweep_signals(bars, length, mult)
        buy = sig["buySignal"].values; sell = sig["sellSignal"].values
        for i in range(len(td) - HOLD - 2):
            if nb >= max_entries:
                break
            if not buy[i]:
                continue
            nb += 1
            ed = td[i + 1].isoformat()
            # arm 2: fixed 10-day hold
            xd10 = td[i + 1 + HOLD].isoformat(); S10 = float(bars.loc[td[i + 1 + HOLD], "close"])
            if S10 > 0:
                r2 = _atm_ret(sym, ed, xd10, S10)
                if r2 is not None:
                    arms["buySignal"].append({"sym": sym, "date": ed, "ret": r2})
            # arm 3: exit at first sellSignal in (i+1, i+1+SELL_CAP], else cap
            jx = i + 1 + SELL_CAP
            for j in range(i + 2, min(i + 1 + SELL_CAP, len(td) - 1) + 1):
                if sell[j]:
                    jx = j; break
            jx = min(jx, len(td) - 1)
            xde = td[jx].isoformat(); Se = float(bars.loc[td[jx], "close"])
            if Se > 0:
                r3 = _atm_ret(sym, ed, xde, Se)
                if r3 is not None:
                    arms["buySignal+sellExit"].append({"sym": sym, "date": ed, "ret": r3})
    return arms


def _mdd(rows):
    eq = peak = 1.0; mdd = 0.0
    for r in sorted(rows, key=lambda x: x["date"]):
        eq *= (1.0 + 0.20 * r["ret"])
        if eq <= 0:
            return -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return mdd


def _stat(rows):
    if not rows:
        return None
    v = [r["ret"] for r in rows]; nn = len(v); m = statistics.mean(v)
    sd = statistics.pstdev(v) if nn > 1 else 0.0
    return {"n": nn, "mean": m, "median": statistics.median(v),
            "sharpe": (m / sd) if sd > 0 else 0.0, "win": sum(1 for x in v if x > 0) / nn, "mdd": _mdd(rows)}


def report(arms, length, mult):
    L = [f"# Liquidity-Sweep Trend+Delta as Options Entry Signal (len={length}, mult={mult})", "",
         "Signal on the UNDERLYING; option = ATM call δ0.45-0.65, spread≤10%, real AV chains, "
         "ASK in / BID out + $0.65/side. Liquid mega-caps. **Judge by median + win + n.**", "",
         "| Arm | n | Mean | Median | Sharpe | Win% | MaxDD |", "|---|---|---|---|---|---|---|"]
    for k in ("baseline", "buySignal", "buySignal+sellExit"):
        s = _stat(arms[k])
        if not s:
            L.append(f"| {k} | 0 |—|—|—|—|—|"); continue
        flag = "  ⚠️small-n" if s["n"] < 25 else ""
        L.append(f"| **{k}** | {s['n']}{flag} | {s['mean']*100:+.1f}% | **{s['median']*100:+.1f}%** | "
                 f"{s['sharpe']:+.3f} | {s['win']*100:.0f}% | {s['mdd']*100:.0f}% |")
    L += ["", "_buySignal fires at trend-change crossovers (selective). If it lifts median+win over baseline "
          "at trustworthy n, it's the entry signal; sellExit tests whether trend-reversal exit beats fixed hold._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=400)
    ap.add_argument("--len", type=int, default=12)
    ap.add_argument("--mult", type=float, default=2.0)
    args = ap.parse_args()
    print(f"Liquidity-sweep entry study (len={args.len}, mult={args.mult})...")
    arms = run(args.max_entries, args.len, args.mult)
    md = report(arms, args.len, args.mult)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "liq_sweep.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
