#!/usr/bin/env python3
"""
backtest_insider_quality_combo.py — prototype: insider signal ∩ quality/trend filter,
expressed as a DEEP-ITM long call.

Thesis (from prior backtests):
    * Insider deep-ITM calls WORK because insider buys carry a real +3-4% underlying edge
      (backtest_insider_options.py: delta0.80 → +9% net @10d).
    * The quality-growth screen alone does NOT (backtest_naked_calls.py): its names have
      ~0% short-horizon drift, so calls just bleed.
    * Combo idea: keep the insider directional edge, ADD a quality/trend gate to drop the
      weakest names, express as a cost-robust deep-ITM call.

What's testable here:
    The screen's FUNDAMENTAL fields (revenue/EPS growth, analyst rating, debt/equity) can't
    be reconstructed point-in-time from price data, so they're a LIVE-ONLY gate (see the
    runtime prototype in run.py). What we CAN compute historically at entry — and what this
    backtest gates on — are the screen's TECHNICAL conditions:
        * price > --price-min (3.99)
        * close > SMA(--trend-sma) at entry   (uptrend / quality proxy, ~ the SMA50>SMA200 idea)
        * relative volume > --relvol-min       (vol / trailing-20d avg)
    It then compares the insider deep-ITM result WITH vs WITHOUT this gate, on the same
    point-in-time insider universe (build_pit_insider_universe.py).

Pricing: Black-Scholes (shared engine). Deep-ITM call (--delta 0.80, --dte 30), exit at
--hold trading days, IV=realised×--iv-mult, extrinsic-value round-trip spread. NET is an
optimistic ceiling (no vol-crush; modelled spread).

Usage:
    python backtest_insider_quality_combo.py
    python backtest_insider_quality_combo.py --trend-sma 50 --relvol-min 1.0 --hold 10
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import timedelta
from typing import Any, Dict, List

import backtest_insider_copier as base
import backtest_insider_options as opt

OUT_DIR = base.OUT_DIR
MIN_VALUE = 1_000_000.0
WINDOW_DAYS = 2
FILING_LAG = 2


def _stat(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    n = len(vals); m = statistics.mean(vals); sd = statistics.pstdev(vals) if n > 1 else 0.0
    t = (m / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "win": sum(1 for v in vals if v > 0) / n, "mean": m,
            "median": statistics.median(vals), "t": t,
            "wipe": sum(1 for v in vals if v <= -0.99) / n}


def run(args) -> Dict[str, Any]:
    import glob, os
    price_syms = {os.path.basename(p).rsplit("_day", 1)[0]
                  for p in glob.glob(str(base.BARS_DIR / "*_day.parquet"))}
    events = base.load_pit_buy_events()
    universe = sorted(set(events) & price_syms)
    half = args.extrinsic_spread / 2.0
    T0 = args.dte / 365.0

    # buckets: "all" insider signals, "combo" = insider ∩ quality/trend gate
    rows = {"all": [], "combo": []}
    for sym in universe:
        buys = events.get(sym, [])
        bars = opt._full_bars(sym)
        if not buys or bars is None or bars.empty:
            continue
        tdates = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in tdates}
        vol = {d: float(bars.loc[d, "volume"]) for d in tdates}
        for sd in base.generate_signals(buys, MIN_VALUE, WINDOW_DAYS, args.hold, tdates, FILING_LAG):
            entry_dt = base._next_trading_on_or_after(sd + timedelta(days=FILING_LAG), tdates)
            if entry_dt is None:
                continue
            exit_dt = base._shift_trading_days(entry_dt, args.hold, tdates)
            if exit_dt is None:
                continue
            ei = tdates.index(entry_dt)
            S0 = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
            if S0 <= 0 or S1 <= 0:
                continue
            sig = opt.realised_vol([close[d] for d in tdates[max(0, ei - 30):ei]]) * args.iv_mult
            K = opt.strike_for_delta(S0, T0, sig, args.delta)
            entry_prem = opt.bs_call(S0, K, T0, sig)
            if entry_prem <= 0:
                continue
            T1 = max(0.5 / 365.0, (args.dte - (exit_dt - entry_dt).days) / 365.0)
            exit_prem = opt.bs_call(S1, K, T1, sig)
            extr_in = max(0.0, entry_prem - max(0.0, S0 - K))
            extr_out = max(0.0, exit_prem - max(0.0, S1 - K))
            net = max(-1.0, (max(0.0, exit_prem - half * extr_out)) / (entry_prem + half * extr_in) - 1.0)
            rec = {"symbol": sym, "net": net, "share": S1 / S0 - 1.0}
            rows["all"].append(rec)

            # quality / trend gate (computable technical part of the screen)
            sma_win = [close[d] for d in tdates[max(0, ei - args.trend_sma):ei]]
            sma = statistics.mean(sma_win) if sma_win else 0.0
            avgvol = statistics.mean([vol[d] for d in tdates[max(0, ei - 20):ei]] or [0])
            relvol = (vol[entry_dt] / avgvol) if avgvol > 0 else 0.0
            price_entry = float(bars.loc[entry_dt, "open"])
            if (price_entry > args.price_min and sma > 0 and price_entry > sma
                    and relvol > args.relvol_min):
                rows["combo"].append(rec)

    return {"params": vars(args), "universe_size": len(universe), "rows": rows}


def markdown(res) -> str:
    p = res["params"]
    a = _stat([r["net"] for r in res["rows"]["all"]])
    c = _stat([r["net"] for r in res["rows"]["combo"]])
    ashr = _stat([r["share"] for r in res["rows"]["all"]])
    cshr = _stat([r["share"] for r in res["rows"]["combo"]])
    L = ["# Insider ∩ Quality/Trend — Deep-ITM Call Combo (prototype backtest)", "",
         f"PIT insider universe {res['universe_size']} names. Deep-ITM call: delta≈{p['delta']}, "
         f"{p['dte']}DTE, {p['hold']}d hold, IV=realised×{p['iv_mult']}, extrinsic spread {p['extrinsic_spread']:.0%}.",
         f"Quality/trend gate (technical part of the screen, computable PIT): price>${p['price_min']}, "
         f"close>SMA{p['trend_sma']}, rel-vol>{p['relvol_min']}.", "",
         "| Set | n | Win% | **Opt net mean** | net median | t | Wipeouts | Share mean |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s, shr in [("insider ALL (no gate)", a, ashr), ("insider ∩ quality/trend", c, cshr)]:
        if not s.get("n"):
            L.append(f"| {name} | 0 |—|—|—|—|—|—|"); continue
        L.append(f"| {name} | {s['n']} | {s['win']*100:.0f}% | **{s['mean']*100:+.1f}%** | "
                 f"{s['median']*100:+.1f}% | {s['t']:+.2f} | {s['wipe']*100:.0f}% | {shr['mean']*100:+.2f}% |")
    keep = (c["n"] / a["n"] * 100) if a.get("n") else 0
    L += ["", f"Gate keeps {c.get('n',0)}/{a.get('n',0)} signals ({keep:.0f}%).", "",
          "## Read",
          "- Does the quality/trend gate RAISE net mean / win% / share edge and CUT wipeouts vs ungated?",
          "  If yes, the combo concentrates the insider edge into better names.",
          "- FUNDAMENTAL screen fields (rev/EPS growth, analyst rating, D/E) are NOT in this backtest "
          "(no PIT fundamentals) — they're a live-only gate in run.py, expected to filter further.",
          "- NET is an optimistic ceiling (BS, modelled spread). Universe still has the usual "
          "survivorship caveat on the unpriced delisted tail.",
          "", "_Same no-lookahead entry/exit as the insider backtests._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.80)
    ap.add_argument("--dte", type=int, default=30)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--iv-mult", type=float, default=1.3)
    ap.add_argument("--extrinsic-spread", type=float, default=0.40)
    ap.add_argument("--price-min", type=float, default=3.99)
    ap.add_argument("--trend-sma", type=int, default=50)
    ap.add_argument("--relvol-min", type=float, default=1.0)
    args = ap.parse_args()
    res = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "insider_quality_combo_backtest.json").write_text(
        json.dumps({**res, "rows": {k: len(v) for k, v in res["rows"].items()}}, indent=2, default=str))
    md = markdown(res)
    (OUT_DIR / "insider_quality_combo_backtest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
