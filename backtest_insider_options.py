#!/usr/bin/env python3
"""
backtest_insider_options.py — approximate options P&L for the insider copier, with a
realistic (extrinsic-based) spread model and liquidity filter for TUNING.

The copier buys a long naked CALL to mirror an insider buy. This re-prices the SAME
point-in-time insider signals (build_pit_insider_universe.py) as a call via Black-Scholes
so we can tune DTE / delta / hold / liquidity toward a config that survives real costs.

Why the earlier "options destroy the edge" result was too harsh on ITM calls:
    Real option bid/ask is priced off TIME VALUE (extrinsic), not total premium. A deep-
    ITM call is mostly intrinsic, so its spread (and theta and IV exposure) is small
    relative to its price. The old model charged spread on the whole premium, punishing
    ITM. This version charges the round-trip spread on EXTRINSIC value (--extrinsic-spread),
    which is what market-makers actually quote — so the ITM-vs-ATM tradeoff is fair.

Tuning knobs:
    --target-delta   entry delta (moneyness). Higher = deeper ITM = less bleed, less leverage.
    --dte            modelled days-to-expiry at entry.
    --horizons       holding periods in trading days (exit). 10 = the copier time-exit.
    --extrinsic-spread  round-trip spread as a fraction of TIME VALUE (small-caps: 0.3-0.6).
    --iv-mult        IV as a multiple of trailing realised vol (event premium: 1.2-1.5).
    --min-advdollar  drop signals whose underlying trailing-30d avg $-volume < this
                     (proxy for "liquid options only").

Still OPTIMISTIC (no vol-crush, IV proxied by realised, fills at modelled spread): treat
NET as a ceiling. Usage:
    python backtest_insider_options.py --target-delta 0.70 --dte 45 --extrinsic-spread 0.4
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from datetime import timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List

import pandas as pd

import backtest_insider_copier as base

OUT_DIR = base.OUT_DIR
_N = NormalDist()
R = 0.03
VOL_FLOOR, VOL_CAP = 0.30, 2.5


def bs_call(S, K, T, sig, r=R):
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    srt = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / srt
    return S * _N.cdf(d1) - K * math.exp(-r * T) * _N.cdf(d1 - srt)


def strike_for_delta(S, T, sig, target_delta, r=R):
    d1 = _N.inv_cdf(min(0.999, max(0.001, target_delta)))
    return S * math.exp((r + 0.5 * sig * sig) * T - d1 * sig * math.sqrt(T))


def realised_vol(closes):
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i] > 0 and closes[i - 1] > 0]
    if len(rets) < 5:
        return VOL_FLOOR
    return max(VOL_FLOOR, min(VOL_CAP, statistics.pstdev(rets) * math.sqrt(252)))


def _full_bars(sym):
    f = base.BARS_DIR / f"{sym}_day.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)[["open", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_convert("UTC").date
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        return None


def run(args) -> Dict[str, Any]:
    price_syms = {os.path.basename(p).rsplit("_day", 1)[0]
                  for p in glob.glob(str(base.BARS_DIR / "*_day.parquet"))}
    events = base.load_pit_buy_events()
    universe = sorted(set(events) & price_syms)
    half = args.extrinsic_spread / 2.0
    T0 = args.dte / 365.0
    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    dropped_illiquid = 0
    for sym in universe:
        buys = events.get(sym, [])
        bars = _full_bars(sym)
        if not buys or bars is None or bars.empty:
            continue
        tdates = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in tdates}
        dvol = {d: float(bars.loc[d, "close"]) * float(bars.loc[d, "volume"]) for d in tdates}
        for hd in horizons:
            for sd in base.generate_signals(buys, args.min_value, args.window_days, hd, tdates, args.filing_lag):
                entry_dt = base._next_trading_on_or_after(sd + timedelta(days=args.filing_lag), tdates)
                if entry_dt is None:
                    continue
                exit_dt = base._shift_trading_days(entry_dt, hd, tdates)
                if exit_dt is None:
                    continue
                ei = tdates.index(entry_dt)
                if args.min_advdollar > 0:
                    adv = statistics.mean([dvol[d] for d in tdates[max(0, ei - 30):ei]] or [0])
                    if adv < args.min_advdollar:
                        dropped_illiquid += 1
                        continue
                S0 = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
                if S0 <= 0 or S1 <= 0:
                    continue
                sig = realised_vol([close[d] for d in tdates[max(0, ei - 30):ei]]) * args.iv_mult
                K = strike_for_delta(S0, T0, sig, args.target_delta)
                entry_prem = bs_call(S0, K, T0, sig)
                if entry_prem <= 0:
                    continue
                T1 = max(0.5 / 365.0, (args.dte - (exit_dt - entry_dt).days) / 365.0)
                exit_prem = bs_call(S1, K, T1, sig)
                # round-trip spread charged on TIME VALUE (extrinsic), per leg
                extr_in = max(0.0, entry_prem - max(0.0, S0 - K))
                extr_out = max(0.0, exit_prem - max(0.0, S1 - K))
                entry_cost = entry_prem + half * extr_in
                exit_proceeds = max(0.0, exit_prem - half * extr_out)
                rows.append({
                    "symbol": sym, "horizon": hd, "signal_date": sd.isoformat(),
                    "share_ret": S1 / S0 - 1.0,
                    "opt_ret_gross": max(-1.0, exit_prem / entry_prem - 1.0),
                    "opt_ret_net": max(-1.0, exit_proceeds / entry_cost - 1.0),
                })
    return {"params": vars(args), "universe_size": len(universe),
            "dropped_illiquid": dropped_illiquid, "horizons": horizons, "rows": rows}


def _stat(vals):
    if not vals:
        return {"n": 0}
    n = len(vals); m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if n > 1 else 0.0
    t = (m / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "win": sum(1 for v in vals if v > 0) / n, "mean": m,
            "median": statistics.median(vals), "t": t,
            "wipe": sum(1 for v in vals if v <= -0.99) / n}


def markdown(res) -> str:
    p = res["params"]
    L = ["# Insider Copier — Options (Black-Scholes) Backtest", "",
         f"PIT signals, {res['universe_size']} names. Call: {p['dte']}DTE, entry delta≈{p['target_delta']}, "
         f"IV=realised×{p['iv_mult']}, extrinsic round-trip spread {p['extrinsic_spread']:.0%}, "
         f"min ADV$ {p['min_advdollar']:,.0f} (dropped {res['dropped_illiquid']} illiquid).", "",
         "| Horizon | n | Win% | **Opt net mean** | net median | t | Wipeouts | gross mean | Share mean | Lev |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for hd in res["horizons"]:
        rws = [r for r in res["rows"] if r["horizon"] == hd]
        net = _stat([r["opt_ret_net"] for r in rws]); gro = _stat([r["opt_ret_gross"] for r in rws])
        shr = _stat([r["share_ret"] for r in rws])
        if not net.get("n"):
            L.append(f"| {hd}d | 0 |—|—|—|—|—|—|—|—|"); continue
        lev = (gro["mean"] / shr["mean"]) if shr.get("mean") else 0.0
        L.append(f"| {hd}d | {net['n']} | {net['win']*100:.0f}% | **{net['mean']*100:+.1f}%** | "
                 f"{net['median']*100:+.1f}% | {net['t']:+.2f} | {net['wipe']*100:.0f}% | "
                 f"{gro['mean']*100:+.1f}% | {shr['mean']*100:+.2f}% | {lev:.1f}× |")
    L += ["", "_Spread charged on TIME VALUE (fair to ITM). NET still optimistic (no vol-crush, "
          "realised-vol IV). 10d = copier time-exit._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-value", type=float, default=1_000_000.0)
    ap.add_argument("--window-days", type=int, default=2)
    ap.add_argument("--filing-lag", type=int, default=2)
    ap.add_argument("--horizons", type=str, default="5,10")
    ap.add_argument("--dte", type=int, default=30)
    ap.add_argument("--target-delta", type=float, default=0.48)
    ap.add_argument("--iv-mult", type=float, default=1.3)
    ap.add_argument("--extrinsic-spread", type=float, default=0.40)
    ap.add_argument("--min-advdollar", type=float, default=0.0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    res = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ""
    (OUT_DIR / f"insider_copier_options_backtest{tag}.json").write_text(
        json.dumps({**res, "rows": len(res["rows"])}, indent=2, default=str))
    md = markdown(res)
    (OUT_DIR / f"insider_copier_options_backtest{tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
