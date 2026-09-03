#!/usr/bin/env python3
"""
backtest_naked_calls.py — backtest the FULL naked-calls path on the screened universe.

Path replayed:
    tv-screener.py scan_quality_growth  ->  tv_screener_tickers.txt  (the universe)
      ->  naked-calls.py contract selection  ->  long CALL purchase.

naked-calls.py selection rules faithfully modelled (naked-calls.py:149-520):
    * calls only, DTE in [7,45], delta in [0.30,0.70] (score peaks at ~0.50 = ATM),
      IV <= 2.50, spread <= 20%, ranked by a liquidity/greeks score → top contract/name.
    Representative pick ≈ ATM (--delta 0.50), low-theta end of the DTE band (--dte 35).
    No exit logic exists in naked-calls.py, so we test fixed holds (--holds).

Pricing: Black-Scholes (shared with backtest_insider_options.py). IV proxied by trailing
realised vol × --iv-mult, capped at naked-calls' MAX_IV (2.5). Round-trip spread charged
on EXTRINSIC value (--extrinsic-spread; these are $300M-10B names so tighter than micro).

Entry timing: naked-calls has NO signal — it buys whenever run on whatever passes the
screen. We model continuous coverage: a non-overlapping entry every --cadence trading
days per name (cadence == hold ⇒ a perpetually-rolled call).

⚠ BIASES (read before trusting):
    * SURVIVORSHIP / LOOK-AHEAD: the universe is TODAY's screen. These names pass the
      quality filter NOW; they would not necessarily have passed in 2021-2024 (the screen
      uses current fundamentals/analyst ratings, which can't be reconstructed historically
      from TradingView). So this measures the naked-calls OPTION MECHANICS on a fixed
      quality basket — NOT a tradeable historical strategy. The live universe rotates daily.
    * BS optimism: realised-vol IV proxy, no vol-crush, fills at modelled spread. NET is a
      ceiling; real fills are worse.

Usage:
    python backtest_naked_calls.py
    python backtest_naked_calls.py --delta 0.50 --dte 35 --holds 10,21 --extrinsic-spread 0.25
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import backtest_insider_copier as base
import backtest_insider_options as opt

OUT_DIR = base.OUT_DIR
UNIVERSE_FILE = base.PROJECT_ROOT / "tv_screener_tickers.txt"
MAX_IV = 2.50   # naked-calls.py MAX_IV


def _universe() -> List[str]:
    if not UNIVERSE_FILE.exists():
        return []
    txt = UNIVERSE_FILE.read_text()
    out, seen = [], set()
    for chunk in txt.replace("\n", ",").split(","):
        s = chunk.strip().strip('"').strip("'").upper()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out


def run(args) -> Dict[str, Any]:
    holds = [int(x) for x in str(args.holds).split(",") if x.strip()]
    half = args.extrinsic_spread / 2.0
    T0 = args.dte / 365.0
    universe = _universe()
    rows: List[Dict[str, Any]] = []
    covered: List[str] = []
    for sym in universe:
        bars = opt._full_bars(sym)
        if bars is None or bars.empty:
            continue
        covered.append(sym)
        tdates = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in tdates}
        for hd in holds:
            # non-overlapping rolling entries every `cadence` trading days
            i = 35  # need trailing window for realised vol
            cadence = max(1, args.cadence)
            while i < len(tdates):
                entry_dt = tdates[i]
                exit_dt = base._shift_trading_days(entry_dt, hd, tdates)
                if exit_dt is None:
                    break
                S0 = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
                if S0 > 0 and S1 > 0:
                    sig = opt.realised_vol([close[d] for d in tdates[max(0, i - 30):i]]) * args.iv_mult
                    if sig <= MAX_IV:   # naked-calls IV filter
                        K = opt.strike_for_delta(S0, T0, sig, args.delta)
                        entry_prem = opt.bs_call(S0, K, T0, sig)
                        if entry_prem > 0:
                            T1 = max(0.5 / 365.0, (args.dte - (exit_dt - entry_dt).days) / 365.0)
                            exit_prem = opt.bs_call(S1, K, T1, sig)
                            extr_in = max(0.0, entry_prem - max(0.0, S0 - K))
                            extr_out = max(0.0, exit_prem - max(0.0, S1 - K))
                            cost = entry_prem + half * extr_in
                            proceeds = max(0.0, exit_prem - half * extr_out)
                            rows.append({"symbol": sym, "hold": hd,
                                         "share_ret": S1 / S0 - 1.0,
                                         "opt_ret_net": max(-1.0, proceeds / cost - 1.0),
                                         "opt_ret_gross": max(-1.0, exit_prem / entry_prem - 1.0)})
                i += cadence
    return {"params": vars(args), "universe": universe, "covered": covered,
            "holds": holds, "rows": rows}


def markdown(res) -> str:
    p = res["params"]
    L = ["# Naked-Calls Path — Backtest on Screened Universe", "",
         f"Universe ({len(res['covered'])}/{len(res['universe'])} priced): {', '.join(res['covered'])}",
         f"Modelled pick: delta≈{p['delta']} (ATM), {p['dte']}DTE, IV=realised×{p['iv_mult']} (cap {MAX_IV}), "
         f"extrinsic round-trip spread {p['extrinsic_spread']:.0%}, rolling entry every {p['cadence']}d.", "",
         "| Hold | n | Win% | **Opt net mean** | net median | t | Wipeouts | gross mean | Share mean | Lev |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for hd in res["holds"]:
        rws = [r for r in res["rows"] if r["hold"] == hd]
        net = opt._stat([r["opt_ret_net"] for r in rws]) if hasattr(opt, "_stat") else None
        # opt has no _stat; compute inline
        nets = [r["opt_ret_net"] for r in rws]; gros = [r["opt_ret_gross"] for r in rws]
        shrs = [r["share_ret"] for r in rws]
        if not nets:
            L.append(f"| {hd}d | 0 |—|—|—|—|—|—|—|—|"); continue
        def st(v):
            n=len(v); m=statistics.mean(v); sd=statistics.pstdev(v) if n>1 else 0.0
            t=(m/(sd/math.sqrt(n))) if sd>0 else 0.0
            return n,m,statistics.median(v),t,sum(1 for x in v if x<=-0.99)/n,sum(1 for x in v if x>0)/n
        n,m,med,t,wipe,win = st(nets); _,gm,_,_,_,_ = st(gros); _,sm,_,_,_,_ = st(shrs)
        lev = gm/sm if sm else 0.0
        L.append(f"| {hd}d | {n} | {win*100:.0f}% | **{m*100:+.1f}%** | {med*100:+.1f}% | {t:+.2f} | "
                 f"{wipe*100:.0f}% | {gm*100:+.1f}% | {sm*100:+.2f}% | {lev:.1f}× |")
    L += ["", "## Read",
          "- **⚠ Survivorship/look-ahead**: universe = TODAY's screen; not a tradeable historical result. "
          "Measures naked-calls option mechanics on a fixed quality basket only.",
          "- ATM ~30DTE calls are the naked-calls profile; cross-check backtest_insider_options.py which "
          "showed ATM is net-negative once realistic IV+spread are priced. NET here is still an optimistic ceiling.",
          "_No exit logic in naked-calls.py — holds are imposed by this backtest._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.50)
    ap.add_argument("--dte", type=int, default=35)
    ap.add_argument("--holds", type=str, default="10,21")
    ap.add_argument("--cadence", type=int, default=10)
    ap.add_argument("--iv-mult", type=float, default=1.2)
    ap.add_argument("--extrinsic-spread", type=float, default=0.25)
    args = ap.parse_args()
    res = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "naked_calls_backtest.json").write_text(
        json.dumps({**res, "rows": len(res["rows"])}, indent=2, default=str))
    md = markdown(res)
    (OUT_DIR / "naked_calls_backtest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
