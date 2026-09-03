#!/usr/bin/env python3
"""
backtest_alpha_momentum_options.py — ATM vs deep-ITM calls on the ALPHA MOMENTUM signal.

Decides empirically whether alpha_call_engine.py should default to deep-ITM (δ≈0.80) or
keep ATM (δ≈0.50) for its momentum signal — the open question after wiring options_quant's
leverage tilt. The insider signal (multi-day mean-reversion) validated deep-ITM; momentum
(breakout continuation) may instead want ATM gamma. This settles it on data.

Signal — daily proxy for the engine's intraday ORB/VWAP/RVOL breakout:
    entry when  close > prior-`breakout`-day high  (breakout, ~ORB)
            AND close > SMA(`trend`)               (uptrend, ~above-VWAP/trend)
            AND volume/avg20 > `relvol`            (relative-volume confirmation, ~RVOL)
    No-lookahead: enter next day's OPEN, exit `hold` days later at CLOSE.

Both calls priced head-to-head on the SAME entries via Black-Scholes (shared engine):
    DTE `dte` (alpha band 21-60, peak ~37), IV = realised×`iv_mult`, round-trip spread on
    EXTRINSIC value (`spread`; alpha-core names are very liquid → tight). NET is an optimistic
    ceiling (no vol-crush), but it is APPLES-TO-APPLES across the two deltas, which is what
    the decision needs.

Universe: alpha_call_engine.ALPHA_STATIC_CORE (liquid optionable momentum names).

Usage:
    python backtest_alpha_momentum_options.py
    python backtest_alpha_momentum_options.py --holds 3,5,10 --relvol 1.5 --spread 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Dict, List

import pandas as pd

import backtest_insider_copier as base
import backtest_insider_options as opt

OUT_DIR = base.OUT_DIR


def _bars(sym):
    """Load daily bars with high/low intact (opt._full_bars drops high)."""
    f = base.BARS_DIR / f"{sym}_day.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_convert("UTC").date
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        return None


def _universe() -> List[str]:
    try:
        import alpha_call_engine as ace
        return list(ace.ALPHA_STATIC_CORE)
    except Exception:
        return ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL"]


def _stat(vals):
    if not vals:
        return {"n": 0}
    n = len(vals); m = statistics.mean(vals); sd = statistics.pstdev(vals) if n > 1 else 0.0
    t = (m / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "win": sum(1 for v in vals if v > 0) / n, "mean": m,
            "median": statistics.median(vals), "t": t,
            "wipe": sum(1 for v in vals if v <= -0.99) / n}


def _opt_net(S0, S1, delta, dte, sigma, hold_cal, half):
    T0 = dte / 365.0
    K = opt.strike_for_delta(S0, T0, sigma, delta)
    entry = opt.bs_call(S0, K, T0, sigma)
    if entry <= 0:
        return None
    T1 = max(0.5 / 365.0, (dte - hold_cal) / 365.0)
    ex = opt.bs_call(S1, K, T1, sigma)
    extr_in = max(0.0, entry - max(0.0, S0 - K))
    extr_out = max(0.0, ex - max(0.0, S1 - K))
    return max(-1.0, (max(0.0, ex - half * extr_out)) / (entry + half * extr_in) - 1.0)


def run(args) -> Dict[str, Any]:
    half = args.spread / 2.0
    holds = [int(x) for x in str(args.holds).split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    covered = []
    for sym in _universe():
        bars = _bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        covered.append(sym)
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        for hd in holds:
            i = 55  # need SMA + breakout window
            while i < len(td) - 1:
                d = td[i]
                prior_high = max(high[td[j]] for j in range(i - args.breakout, i))
                sma = statistics.mean(close[td[j]] for j in range(i - args.trend, i))
                avgvol = statistics.mean([vol[td[j]] for j in range(i - 20, i)] or [0])
                relvol = vol[d] / avgvol if avgvol > 0 else 0.0
                is_signal = (close[d] > prior_high and close[d] > sma and relvol > args.relvol)
                if not is_signal:
                    i += 1
                    continue
                entry_dt = td[i + 1]
                exit_idx = i + 1 + hd
                if exit_idx >= len(td):
                    break
                exit_dt = td[exit_idx]
                S0 = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
                if S0 <= 0 or S1 <= 0:
                    i += hd  # non-overlapping after a signal
                    continue
                sigma = opt.realised_vol([close[td[j]] for j in range(max(0, i - 30), i)]) * args.iv_mult
                hold_cal = (exit_dt - entry_dt).days
                atm = _opt_net(S0, S1, args.atm_delta, args.dte, sigma, hold_cal, half)
                itm = _opt_net(S0, S1, args.itm_delta, args.dte, sigma, hold_cal, half)
                if atm is not None and itm is not None:
                    rows.append({"symbol": sym, "hold": hd, "share": S1 / S0 - 1.0,
                                 "atm_net": atm, "itm_net": itm})
                i += hd  # non-overlapping
    return {"params": vars(args), "covered": covered, "holds": holds, "rows": rows}


def markdown(res) -> str:
    p = res["params"]
    L = ["# Alpha Momentum — ATM vs Deep-ITM Calls (head-to-head)", "",
         f"Universe ({len(res['covered'])}): {', '.join(res['covered'])}",
         f"Signal: breakout>{p['breakout']}d-high & close>SMA{p['trend']} & rel-vol>{p['relvol']}. "
         f"Call: {p['dte']}DTE, IV=realised×{p['iv_mult']}, extrinsic spread {p['spread']:.0%}. "
         f"ATM δ={p['atm_delta']} vs deep-ITM δ={p['itm_delta']}.", "",
         "| Hold | n | Share mean | **ATM net** | ATM win/t | **ITM net** | ITM win/t | Winner |",
         "|---|---|---|---|---|---|---|---|"]
    for hd in res["holds"]:
        rws = [r for r in res["rows"] if r["hold"] == hd]
        if not rws:
            L.append(f"| {hd}d | 0 |—|—|—|—|—|— |"); continue
        a = _stat([r["atm_net"] for r in rws]); it = _stat([r["itm_net"] for r in rws])
        sh = _stat([r["share"] for r in rws])
        winner = "ATM" if a["mean"] > it["mean"] else "deep-ITM"
        L.append(f"| {hd}d | {a['n']} | {sh['mean']*100:+.2f}% | **{a['mean']*100:+.1f}%** | "
                 f"{a['win']*100:.0f}%/{a['t']:+.1f} | **{it['mean']*100:+.1f}%** | "
                 f"{it['win']*100:.0f}%/{it['t']:+.1f} | **{winner}** |")
    L += ["", "## Read",
          "- Head-to-head on identical momentum entries → directly answers the engine's default.",
          "- If ATM wins, the convexity (gamma) of momentum continuation pays for its theta → keep ATM default",
          "  (OPTIONS_LEV_SELECT_WEIGHT low). If deep-ITM wins, flip the default like the insider engine.",
          "- Share mean shows whether momentum entries actually have forward drift (vs the ~0 of the quality screen).",
          "", "_NET is an optimistic ceiling; the COMPARISON across deltas is the robust part._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakout", type=int, default=20)
    ap.add_argument("--trend", type=int, default=50)
    ap.add_argument("--relvol", type=float, default=1.5)
    ap.add_argument("--holds", type=str, default="3,5,10")
    ap.add_argument("--dte", type=int, default=35)
    ap.add_argument("--atm-delta", type=float, default=0.50)
    ap.add_argument("--itm-delta", type=float, default=0.80)
    ap.add_argument("--iv-mult", type=float, default=1.2)
    ap.add_argument("--spread", type=float, default=0.15)
    args = ap.parse_args()
    res = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "alpha_momentum_options_backtest.json").write_text(
        json.dumps({**res, "rows": len(res["rows"])}, indent=2, default=str))
    md = markdown(res)
    (OUT_DIR / "alpha_momentum_options_backtest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
