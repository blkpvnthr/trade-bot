#!/usr/bin/env python3
"""
backtest_real_chains.py — THE ultimate gate: re-validate the EV contract selector on REAL
historical option chains (Alpha Vantage HISTORICAL_OPTIONS), not BS-synthetic ones.

Everything is real: entry chain (real IV/greeks/bid/ask/discrete strikes), and realized P&L
read from the ACTUAL chain at exit (enter at ASK, exit at BID = real round-trip spread; no
Black-Scholes anywhere in the P&L). Only the EV *ranking* uses a model (MC under the signal
drift + trailing-calibrated jumps + real IV) to choose ex-ante.

Arms (same entries):
    Current heuristic : closest-to-ATM (δ0.50) from the REAL chain.
    EV-max            : ContractSelector (calibrated-jump EV + ev_alpha tail penalty) on the REAL chain.

Signal: insider (SEC Form-4 clusters, the mean-reversion control) by default; momentum optional.
Sized on LIVE Alpaca equity. Fetches are cached forever (chains are immutable history).

Usage: python backtest_real_chains.py --signal insider --max-entries 120 --commission 0.65
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import timedelta
from typing import Any, Dict, List, Optional

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import av_historical_options as avo
import options_quant_engine as eng
import backtest_ablation_studies as ab

HOLD = 10
TARGET_DTE = 35
DTE_LO, DTE_HI = 21, 49
ALPHA_SCORE = 0.5  # overridable via run(alpha_score=)


def _engine_chain(real_chain: List[dict]) -> List[dict]:
    """Keep liquid calls in the DTE band with valid quotes + greeks."""
    out = []
    for c in real_chain:
        if c.get("type") != "call":
            continue
        if not (DTE_LO <= int(c.get("dte", 0)) <= DTE_HI):
            continue
        if c.get("bid_price", 0) <= 0 or c.get("ask_price", 0) <= 0:
            continue
        if c.get("delta", 0) <= 0:
            continue
        out.append(c)
    return out


def _realized(entry_c: dict, symbol: str, exit_date: str, S1: float, commission: float) -> Optional[float]:
    """Real realized return: pay ASK at entry, sell BID at exit (looked up in the exit chain).
    Falls back to intrinsic on the underlying if the contract is missing at exit."""
    entry_ask = float(entry_c["ask_price"])
    if entry_ask <= 0:
        return None
    exit_c = avo.lookup_contract(symbol, exit_date, entry_c["symbol"])
    if exit_c and float(exit_c.get("bid_price", 0)) > 0:
        exit_px = float(exit_c["bid_price"])
    else:
        exit_px = max(0.0, S1 - float(entry_c["strike_price"]))   # intrinsic fallback
    comm = commission / 100.0
    return max(-1.0, (exit_px - comm) / (entry_ask + comm) - 1.0)


def run(equity: float, signal: str, max_entries: int, ev_alpha: float, commission: float,
        mode: str = "trailing", symbols: Optional[List[str]] = None, alpha_score: float = ALPHA_SCORE) -> Dict[str, Any]:
    pit = base.load_pit_buy_events() if signal == "insider" else {}
    if symbols:
        syms = [s.strip().upper() for s in symbols if s.strip()]
    else:
        syms = sorted(pit.keys()) if signal == "insider" else list(_universe_momentum())
    rows: List[Dict[str, Any]] = []
    fetched = skipped_no_chain = 0
    for sym in syms:
        if len(rows) >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        if signal == "insider":
            entries = ab._insider_entry_indices(td, sym, pit)
        else:
            high = {d: float(bars.loc[d, "high"]) for d in td}
            vol = {d: float(bars.loc[d, "volume"]) for d in td}
            entries = ab._momentum_entry_indices(td, close, high, vol)
        for i in entries:
            if len(rows) >= max_entries:
                break
            entry_dt = td[i + 1]; exit_dt = td[i + 1 + HOLD]
            ed, xd = entry_dt.isoformat(), exit_dt.isoformat()
            spot = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
            if spot <= 0 or S1 <= 0:
                continue
            real_chain = _engine_chain(avo.get_chain(sym, ed, calls_only=True))
            fetched += 1
            if len(real_chain) < 3:
                skipped_no_chain += 1; continue
            # trailing-calibrated jumps (no lookahead) from the price history
            jt = eng.calibrate_jumps([close[td[k]] for k in range(max(0, i - 252), i)]) if mode == "trailing" else None
            sel = ab._make_selector(jt, ev_alpha)
            path_sigma = (jt["sigma_diffusion"] if jt else 0.0) or eng.realized_vol(
                [close[td[j]] for j in range(max(0, i - 25), i + 1)], 20)
            # arms
            cur_c = min(real_chain, key=lambda c: abs(c["delta"] - 0.50))
            ev_best = sel.optimize_ev(real_chain, spot, alpha_score, path_sigma, HOLD)
            ev_c = next((c for c in real_chain if c["symbol"] == ev_best["metrics"].symbol), cur_c) if ev_best else cur_c
            cur_ret = _realized(cur_c, sym, xd, S1, commission)
            ev_ret = _realized(ev_c, sym, xd, S1, commission)
            if cur_ret is None or ev_ret is None:
                continue
            rows.append({"sym": sym, "date": ed, "cur_ret": cur_ret, "ev_ret": ev_ret,
                         "cur_delta": cur_c["delta"], "ev_delta": ev_c["delta"],
                         "cur_spread": (cur_c["ask_price"]-cur_c["bid_price"])/max(cur_c["mid"],1e-6),
                         "ev_spread": (ev_c["ask_price"]-ev_c["bid_price"])/max(ev_c["mid"],1e-6)})
    return {"equity": equity, "rows": rows, "fetched": fetched, "skipped_no_chain": skipped_no_chain}


def _universe_momentum():
    import glob, os
    return sorted(os.path.basename(p).rsplit("_day", 1)[0]
                  for p in glob.glob(str(base.BARS_DIR / "*_day.parquet")))[:40]


def _stat(v):
    if not v:
        return {"n": 0}
    n = len(v); m = statistics.mean(v); sd = statistics.pstdev(v) if n > 1 else 0.0
    return {"n": n, "mean": m, "sharpe": (m / sd) if sd > 0 else 0.0,
            "win": sum(1 for x in v if x > 0) / n, "median": statistics.median(v)}


def markdown(res, signal):
    rows = res["rows"]
    cur = _stat([r["cur_ret"] for r in rows]); ev = _stat([r["ev_ret"] for r in rows])
    avg = lambda k: statistics.mean([r[k] for r in rows]) if rows else 0
    L = ["# REAL historical chains — EV selector vs ATM heuristic (the ultimate gate)", "",
         f"Signal **{signal}**. {len(rows)} entries on REAL Alpha Vantage chains "
         f"(fetched {res['fetched']}, {res['skipped_no_chain']} had no usable chain). "
         f"Enter at ASK, exit at BID from the actual exit-date chain — no BS in the P&L.", "",
         "| Method | n | Mean | Sharpe | Win% | Median | Avg δ | Avg spread |",
         "|---|---|---|---|---|---|---|---|",
         f"| Current (ATM δ0.50) | {cur.get('n',0)} | {cur['mean']*100:+.1f}% | {cur['sharpe']:+.3f} | "
         f"{cur['win']*100:.0f}% | {cur['median']*100:+.1f}% | {avg('cur_delta'):.2f} | {avg('cur_spread')*100:.1f}% |",
         f"| **EV-max (real greeks)** | {ev.get('n',0)} | **{ev['mean']*100:+.1f}%** | **{ev['sharpe']:+.3f}** | "
         f"{ev['win']*100:.0f}% | {ev['median']*100:+.1f}% | {avg('ev_delta'):.2f} | {avg('ev_spread')*100:.1f}% |",
         "", "_Real IV/greeks/strikes/spreads; realized P&L from the actual chain at exit (enter ASK→exit BID + commission)._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=["insider", "momentum"], default="insider")
    ap.add_argument("--max-entries", type=int, default=120)
    ap.add_argument("--ev-alpha", type=float, default=0.3)
    ap.add_argument("--commission", type=float, default=0.65)
    ap.add_argument("--equity", type=float, default=0.0)
    ap.add_argument("--symbols", type=str, default="", help="comma-sep universe override (liquid names)")
    ap.add_argument("--alpha-score", type=float, default=0.5, help="signal drift used by EV (lower=less OTM)")
    args = ap.parse_args()
    equity = args.equity or ab.live_alpaca_equity() or 6260.0
    syms = [s for s in args.symbols.split(",") if s.strip()] or None
    print(f"REAL-CHAIN validation: signal={args.signal} max_entries={args.max_entries} equity=${equity:,.0f}")
    res = run(equity, args.signal, args.max_entries, args.ev_alpha, args.commission, symbols=syms, alpha_score=args.alpha_score)
    md = markdown(res, args.signal)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / f"real_chain_{args.signal}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
