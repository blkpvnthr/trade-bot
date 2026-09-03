#!/usr/bin/env python3
"""
study_trend_structure.py — DIRECTION-SELECTION filters on the ATM long-call baseline.

Tests whether trading WITH the trend / on a structural (supply-zone) breakout improves the
distribution — a different lever than the (failed) ORB entry-timing studies. Objective,
non-repainting definitions; same real-chain infrastructure; n reported honestly per arm
(the IV/ORB studies showed how easily a small-n 'edge' is manufactured).

Filters on the SAME momentum entries (control = baseline):
  +200SMA        close > SMA200                       (primary trend up)
  +50>200        SMA50 > SMA200                        (golden-cross regime)
  +supply        breakout of a prior SUPPLY zone with volume confirmation
  +supply+200SMA / +supply+50>200  (combined)

SUPPLY ZONE (objective): a tight BASE of 3-10 candles (range ≤ 1 ATR) immediately followed by
an impulse DOWN > 2 ATR. zone_high = base high. BREAKOUT at i: close[i] freshly crosses above an
unbroken zone_high AND volume[i] > 1.5 × avg20. (Buyers absorbed the supply that rejected before.)

Option leg: ATM call δ0.45-0.65, spread≤10%, real AV chain, enter ASK / exit BID + commission,
10-day hold. Liquid mega-caps. Reports mean/median/Sharpe/win/MaxDD/n per arm.

Usage: python study_trend_structure.py --max-entries 250
"""
from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional

import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo
import study_atm_filters as atm

HOLD = 10


def _atr(high, low, close, i, n=14):
    if i < n + 1:
        return 0.0
    trs = []
    for k in range(i - n + 1, i + 1):
        trs.append(max(high[k] - low[k], abs(high[k] - close[k - 1]), abs(low[k] - close[k - 1])))
    return sum(trs) / n


def supply_breakout(close, high, low, vol, i, lookback=60) -> bool:
    """close[i] freshly breaks above an unbroken prior supply-zone high with volume>1.5×avg20."""
    if i < 40:
        return False
    avg20 = statistics.mean(vol[i - 20:i]) if i >= 20 else 0.0
    if avg20 <= 0 or vol[i] <= 1.5 * avg20:
        return False
    ci, ci1 = close[i], close[i - 1]
    for b in range(i - 5, max(i - lookback, 25), -1):
        a = _atr(high, low, close, b)
        if a <= 0:
            continue
        for B in range(3, 11):                       # base length 3-10
            lo_idx = b - B + 1
            if lo_idx < 1:
                continue
            base_high = max(high[lo_idx:b + 1]); base_low = min(low[lo_idx:b + 1])
            if (base_high - base_low) > 1.0 * a:     # base must be TIGHT (consolidation)
                continue
            imp_lo = min(low[b + 1:min(b + 4, i)] or [base_low])
            if (base_low - imp_lo) < 2.0 * a:        # impulse DOWN > 2 ATR
                continue
            zone_high = base_high
            if ci1 <= zone_high < ci and all(close[k] <= zone_high for k in range(b + 1, i)):
                return True                          # fresh cross of an unbroken zone
    return False


def run(max_entries: int) -> Dict[str, List[dict]]:
    arms = {k: [] for k in ("baseline", "+200SMA", "+50>200", "+supply",
                            "+supply+200SMA", "+supply+50>200")}
    n = 0
    for sym in atm.LIQUID:
        if n >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 260:
            continue
        td = list(bars.index)
        close = [float(bars.loc[d, "close"]) for d in td]
        high = [float(bars.loc[d, "high"]) for d in td]
        low = [float(bars.loc[d, "low"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        cd = {d: close[k] for k, d in enumerate(td)}
        hd = {d: high[k] for k, d in enumerate(td)}
        vd = {d: vol[k] for k, d in enumerate(td)}
        for i in ab._momentum_entry_indices(td, cd, hd, vd):
            if n >= max_entries:
                break
            if i < 200:                               # need SMA200 history
                continue
            entry_dt = td[i + 1]; exit_dt = td[i + 1 + HOLD]
            ed, xd = entry_dt.isoformat(), exit_dt.isoformat()
            S1 = float(bars.loc[exit_dt, "close"])
            if S1 <= 0:
                continue
            chain = avo.get_chain(sym, ed, calls_only=True)
            c = atm._atm_in_bucket(chain, 21, 49)
            if not c:
                continue
            ret = atm._realized(c, sym, xd, S1)
            if ret is None:
                continue
            n += 1
            sma200 = statistics.mean(close[i - 200:i]); sma50 = statistics.mean(close[i - 50:i])
            f200 = close[i] > sma200
            f5020 = sma50 > sma200
            fsup = supply_breakout(close, high, low, vol, i)
            row = {"sym": sym, "date": ed, "ret": ret}
            arms["baseline"].append(row)
            if f200: arms["+200SMA"].append(row)
            if f5020: arms["+50>200"].append(row)
            if fsup: arms["+supply"].append(row)
            if fsup and f200: arms["+supply+200SMA"].append(row)
            if fsup and f5020: arms["+supply+50>200"].append(row)
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


def report(arms):
    L = ["# Direction-Selection Filters on ATM Long Calls (real chains, liquid mega-caps)", "",
         "Trend + structure filters on the SAME momentum entries. Real AV chains, ATM δ0.45-0.65, "
         "spread≤10%, enter ASK/exit BID + $0.65/side, 10-day hold. **Judge by median + win + n** "
         "(small n = manufactured edge — the lesson from the IV/ORB studies).", "",
         "| Arm | n | Mean | Median | Sharpe | Win% | MaxDD |", "|---|---|---|---|---|---|---|"]
    for k in ("baseline", "+200SMA", "+50>200", "+supply", "+supply+200SMA", "+supply+50>200"):
        s = _stat(arms[k])
        if not s:
            L.append(f"| {k} | 0 |—|—|—|—|—|"); continue
        flag = "  ⚠️small-n" if s["n"] < 25 else ""
        L.append(f"| **{k}** | {s['n']}{flag} | {s['mean']*100:+.1f}% | **{s['median']*100:+.1f}%** | "
                 f"{s['sharpe']:+.3f} | {s['win']*100:.0f}% | {s['mdd']*100:.0f}% |")
    L += ["", "_A filter only counts if it lifts median+win at a sample large enough to trust (n≥~25-30). "
          "Prefer the single simplest filter that survives; do NOT stack._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=250)
    args = ap.parse_args()
    print(f"Direction-selection study (trend + supply-zone, real chains, max {args.max_entries})...")
    arms = run(args.max_entries)
    md = report(arms)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "trend_structure.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
