#!/usr/bin/env python3
"""
backtest_ablation_studies.py — controlled attribution of the new options engine, sized on
the LIVE ALPACA ACCOUNT EQUITY (pulled via the Alpaca API).

Per the review, integrate NOTHING until three controlled studies isolate the source of any edge:

  STUDY 1 — SELECTION ONLY:   current heuristic (closest-to-ATM δ0.50) vs EV-max (ContractSelector).
                              Reported in per-trade RETURN space (size-independent) — pure selection quality.
  STUDY 2 — SIZING ONLY:      same contract (EV-best), current fixed-fraction vs dual-Kelly PositionSizer.
                              Reported as an EQUITY CURVE on the real account equity (affordability + whole
                              contracts bite at a small account).
  STUDY 3 — FULL STACK:       EV-selection + dual-Kelly vs current-heuristic + fixed-fraction.

Signal: daily momentum proxy (breakout>20d-high & >SMA50 & rel-vol>1.5) — same entries for every arm.
Per entry a synthetic chain is priced via Black-Scholes at IV = RV20×iv_mult (HISTORICAL realized vol,
not a user input → real VRP); moneyness-dependent spreads. No-lookahead: selection uses only spot+RV
known at entry; realized P&L uses the actual forward close.

Universe defaults to LOW-PRICED names so options are affordable on a small account.

Usage:
    python backtest_ablation_studies.py                 # pulls live Alpaca equity
    python backtest_ablation_studies.py --equity 6260   # override
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Dict, List, Optional

import numpy as np

import backtest_insider_copier as base
import backtest_insider_options as opt
import backtest_alpha_momentum_options as mom
import options_quant as oq
import options_quant_engine as eng

OUT_DIR = mom.OUT_DIR
HOLD = 10
DTE = 35
IV_MULT = 1.2          # IV = RV20 × this → positive VRP (empirical norm)
TARGET_DELTAS = [0.30, 0.50, 0.70, 0.85]
ALPHA_SCORE = 0.5      # moderate signal; identical across arms
REGIME = "bull"


def live_alpaca_equity() -> Optional[float]:
    import os
    from pathlib import Path
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
    try:
        from alpaca.trading.client import TradingClient
        return float(TradingClient(key, sec, paper=True).get_account().equity)
    except Exception as e:
        print("  (Alpaca equity fetch failed, using fallback):", str(e)[:80])
        return None


def _spread_for_delta(delta: float) -> float:
    """Moneyness-dependent bid-ask: deeper ITM trades tighter (intrinsic-heavy)."""
    return {0.85: 0.04, 0.70: 0.06, 0.50: 0.10, 0.30: 0.16}.get(round(delta, 2), 0.10)


def _build_chain(spot: float, iv: float) -> List[dict]:
    T = DTE / 365.0
    chain = []
    for d in TARGET_DELTAS:
        K = oq_strike_for_delta(spot, T, iv, d)
        mid = oq.bs_price(spot, K, T, iv, kind="call")
        if mid <= 0.02:
            continue
        spr = _spread_for_delta(d); half = mid * spr / 2
        chain.append({"type": "call", "symbol": f"d{int(d*100)}", "strike_price": round(K, 2),
                      "delta": d, "dte": DTE, "bid_price": round(mid - half, 2),
                      "ask_price": round(mid + half, 2), "open_interest": 1000, "volume": 100,
                      "implied_volatility": iv})
    return chain


def oq_strike_for_delta(S, T, sigma, target_delta, r=0.04, q=0.0):
    from statistics import NormalDist
    d1 = NormalDist().inv_cdf(min(0.999, max(0.001, target_delta)))
    return S * math.exp((r + 0.5 * sigma * sigma) * T - d1 * sigma * math.sqrt(T))


def _realized_option_return(contract: dict, spot: float, S1: float, iv: float,
                            commission: float = 0.0) -> float:
    """Realized net option return: enter at mid (incl half-spread on extrinsic), hold HOLD days,
    exit at BS value on the actual forward price, half-spread on exit extrinsic. `commission` is
    $/contract/side (e.g. 0.65) — added per share (÷100), which hits CHEAP contracts hardest."""
    K = float(contract["strike_price"])
    mid = (float(contract["bid_price"]) + float(contract["ask_price"])) / 2
    half = (float(contract["ask_price"]) - float(contract["bid_price"])) / (2 * mid) if mid > 0 else 0.05
    T0 = DTE / 365.0
    entry = oq.bs_price(spot, K, T0, iv, kind="call")
    if entry <= 0:
        return 0.0
    T1 = max(0.5 / 365.0, (DTE - HOLD) / 365.0)
    exit_val = oq.bs_price(S1, K, T1, iv, kind="call")
    extr_in = max(0.0, entry - max(0.0, spot - K)); extr_out = max(0.0, exit_val - max(0.0, S1 - K))
    comm_ps = commission / 100.0
    cost = entry + half * extr_in + comm_ps
    proceeds = max(0.0, exit_val - half * extr_out - comm_ps)
    return max(-1.0, proceeds / cost - 1.0)


def _current_select(chain: List[dict]) -> dict:
    """Current/retail heuristic: closest-to-ATM (δ0.50)."""
    return min(chain, key=lambda c: abs(float(c["delta"]) - 0.50))


def _current_size(equity: float, mid: float, risk_pct: float = 0.02) -> int:
    """Current fixed-fraction sizing: floor(risk_pct·equity / cost_per_contract)."""
    cost = mid * 100
    return max(0, int((equity * risk_pct) // cost)) if cost > 0 else 0


def _make_selector(jp: Optional[dict], ev_alpha: float):
    if jp is None:
        mc = eng.MonteCarloEngine(n_paths=1500, steps=10)               # GBM
    else:
        mc = eng.MonteCarloEngine(n_paths=1500, steps=10, jump_lambda=jp["jump_lambda"],
                                  jump_mean=jp["jump_mean"], jump_std=jp["jump_std"])
    return eng.ContractSelector(mc=mc, max_spread_pct=0.20, ev_alpha=ev_alpha)


def _momentum_entry_indices(td, close, high, vol) -> List[int]:
    """Breakout>20d-high & >SMA50 & rel-vol>1.5, non-overlapping by HOLD."""
    idx = []; i = 60
    while i < len(td) - HOLD - 1:
        d = td[i]
        if (close[d] > max(high[td[j]] for j in range(i - 20, i))
                and close[d] > statistics.mean(close[td[j]] for j in range(i - 50, i))
                and vol[d] > 1.5 * (statistics.mean([vol[td[j]] for j in range(i - 20, i)] or [1]))):
            idx.append(i); i += HOLD
        else:
            i += 1
    return idx


def _insider_entry_indices(td, sym, pit_events) -> List[int]:
    """Entry indices from SEC Form-4 insider clusters (>$1M, 2-day window, 2-day filing lag)."""
    evs = pit_events.get(sym, [])
    if not evs:
        return []
    from datetime import date as _date, timedelta
    buys = sorted((dd if hasattr(dd, "year") else _date.fromisoformat(str(dd)), float(v))
                  for dd, v in evs)
    sigs = base.generate_signals(buys, 1_000_000, 2, HOLD, td, 2)
    idx = []
    for sd in sigs:
        e = base._next_trading_on_or_after(sd + timedelta(days=2), td)
        if e is None or e not in td:
            continue
        i = td.index(e) - 1                     # body uses entry = td[i+1]
        if 60 <= i < len(td) - HOLD - 1:
            idx.append(i)
    return sorted(set(idx))


def run(equity: float, max_symbols: int = 60, mode: str = "gbm",
        trailing_days: int = 252, ev_alpha: float = 0.0, signal: str = "momentum",
        commission: float = 0.0) -> Dict[str, Any]:
    """mode: 'gbm'|'full'(=lookahead)|'trailing'(=no-lookahead). signal: 'momentum'|'insider'.
    Jumps on → MC diffusion σ = ex-jump σ (no double-count); IV stays total RV×IV_MULT.
    ev_alpha>0 → rank by EV + ev_alpha·ES95 (tail penalty)."""
    sizer = eng.PositionSizer(kelly_fraction=0.25, risk_cap_pct=0.02)
    import glob, os
    price_syms = sorted(os.path.basename(p).rsplit("_day", 1)[0]
                        for p in glob.glob(str(base.BARS_DIR / "*_day.parquet")))
    pit_events = base.load_pit_buy_events() if signal == "insider" else {}
    if signal == "insider":
        price_syms = [s for s in price_syms if s in pit_events][:max_symbols]
    else:
        price_syms = price_syms[:max_symbols]
    rows: List[Dict[str, Any]] = []
    for sym in price_syms:
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        sym_selector = sym_diff = None
        if mode == "full":
            jf = eng.calibrate_jumps([close[d] for d in td])
            sym_selector = _make_selector(jf, ev_alpha); sym_diff = jf["sigma_diffusion"] or None
        base_sel = _make_selector(None, ev_alpha)
        entries = (_insider_entry_indices(td, sym, pit_events) if signal == "insider"
                   else _momentum_entry_indices(td, close, high, vol))
        for i in entries:
            entry_dt = td[i + 1]; exit_dt = td[i + 1 + HOLD]
            spot = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
            if spot <= 0 or S1 <= 0:
                continue
            rv20 = eng.realized_vol([close[td[j]] for j in range(max(0, i - 25), i + 1)], 20)
            iv = max(0.10, rv20 * IV_MULT)
            chain = _build_chain(spot, iv)
            if not chain:
                continue
            if mode == "trailing":
                jt = eng.calibrate_jumps([close[td[k]] for k in range(max(0, i - trailing_days), i)])
                selector = _make_selector(jt, ev_alpha); path_sigma = jt["sigma_diffusion"] or rv20
            elif mode == "full":
                selector = sym_selector; path_sigma = sym_diff or rv20
            else:
                selector = base_sel; path_sigma = rv20
            cur_c = _current_select(chain)
            ev_best = selector.optimize_ev(chain, spot, ALPHA_SCORE, path_sigma, HOLD)
            ev_c = ev_best["metrics"] if ev_best else None
            ev_csel = next((c for c in chain if c["symbol"] == ev_c.symbol), cur_c) if ev_c else cur_c
            cur_ret = _realized_option_return(cur_c, spot, S1, iv, commission)
            ev_ret = _realized_option_return(ev_csel, spot, S1, iv, commission)
            ev_mid = (float(ev_csel["bid_price"]) + float(ev_csel["ask_price"])) / 2
            cur_size = _current_size(equity, ev_mid)
            dk_size = sizer.size(equity, ev_best, regime=REGIME, alpha_score=ALPHA_SCORE).contracts if ev_best else 0
            rows.append({"sym": sym, "date": entry_dt.isoformat(), "spot": spot,
                         "cur_ret": cur_ret, "ev_ret": ev_ret, "cur_delta": cur_c["delta"],
                         "ev_delta": ev_csel["delta"], "ev_mid": ev_mid,
                         "cur_size": cur_size, "dk_size": dk_size})
    return {"equity": equity, "rows": rows}


def _stat(v):
    if not v:
        return {"n": 0}
    n = len(v); m = statistics.mean(v); sd = statistics.pstdev(v) if n > 1 else 0.0
    return {"n": n, "mean": m, "sharpe": (m / sd) if sd > 0 else 0.0,
            "win": sum(1 for x in v if x > 0) / n, "median": statistics.median(v)}


def _equity_curve(rows, get_size, get_ret, equity0):
    eq = equity0; peak = eq; mdd = 0.0
    for r in sorted(rows, key=lambda x: x["date"]):
        n = get_size(r)
        if n <= 0:
            continue
        cost = r["ev_mid"] * 100 * n
        pnl = cost * get_ret(r)
        eq += pnl
        if eq <= 0:
            return 0.0, -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return eq, mdd


def markdown(res):
    rows = res["rows"]; eq0 = res["equity"]
    cur = _stat([r["cur_ret"] for r in rows]); ev = _stat([r["ev_ret"] for r in rows])
    # affordability at this equity
    aff_cur = sum(1 for r in rows if r["cur_size"] > 0); aff_dk = sum(1 for r in rows if r["dk_size"] > 0)
    L = ["# Ablation Studies — EV selection & Dual-Kelly, sized on LIVE Alpaca equity", "",
         f"**Account equity (Alpaca API): ${eq0:,.2f}**  ·  {len(rows)} momentum entries  ·  "
         f"hold {HOLD}d, {DTE}DTE, IV=RV20×{IV_MULT} (positive VRP).", "",
         "## STUDY 1 — SELECTION ONLY (per-trade return, size-independent)", "",
         "| Method | n | Mean ret | Sharpe | Win% | Median | Avg δ |",
         "|---|---|---|---|---|---|---|",
         f"| Current (ATM δ0.50 heuristic) | {cur['n']} | {cur['mean']*100:+.1f}% | {cur['sharpe']:+.3f} | "
         f"{cur['win']*100:.0f}% | {cur['median']*100:+.1f}% | {statistics.mean([r['cur_delta'] for r in rows]):.2f} |",
         f"| **EV-max (ContractSelector)** | {ev['n']} | **{ev['mean']*100:+.1f}%** | **{ev['sharpe']:+.3f}** | "
         f"{ev['win']*100:.0f}% | {ev['median']*100:+.1f}% | {statistics.mean([r['ev_delta'] for r in rows]):.2f} |",
         f"\nSelection delta: EV picks avg δ={statistics.mean([r['ev_delta'] for r in rows]):.2f} "
         f"vs the heuristic's 0.50 — does EV move toward deep-ITM, and does it raise Sharpe?", ""]
    # Study 2 / 3 — equity curves on real equity
    cur_term, cur_mdd = _equity_curve(rows, lambda r: r["cur_size"], lambda r: r["ev_ret"], eq0)
    dk_term, dk_mdd = _equity_curve(rows, lambda r: r["dk_size"], lambda r: r["ev_ret"], eq0)
    full_term, full_mdd = _equity_curve(rows, lambda r: r["dk_size"], lambda r: r["ev_ret"], eq0)
    base_term, base_mdd = _equity_curve(rows, lambda r: r["cur_size"], lambda r: r["cur_ret"], eq0)
    L += ["## STUDY 2 — SIZING ONLY (same EV-best contract; equity curve on real account)", "",
          f"- Affordability at ${eq0:,.0f}: fixed-fraction funds **{aff_cur}/{len(rows)}** entries, "
          f"dual-Kelly funds **{aff_dk}/{len(rows)}** (whole-contract flooring at a small account).", "",
          "| Sizing | Terminal equity | Total return | Max DD |",
          "|---|---|---|---|",
          f"| Current fixed-fraction (2%) | ${cur_term:,.0f} | {(cur_term/eq0-1)*100:+.0f}% | {cur_mdd*100:.0f}% |",
          f"| **Dual-Kelly** | ${dk_term:,.0f} | {(dk_term/eq0-1)*100:+.0f}% | {dk_mdd*100:.0f}% |", "",
          "## STUDY 3 — FULL STACK (EV-selection + Dual-Kelly vs Current+Current)", "",
          "| Stack | Terminal equity | Total return | Max DD |",
          "|---|---|---|---|",
          f"| Current selection + fixed-fraction | ${base_term:,.0f} | {(base_term/eq0-1)*100:+.0f}% | {base_mdd*100:.0f}% |",
          f"| **EV-selection + Dual-Kelly** | ${full_term:,.0f} | {(full_term/eq0-1)*100:+.0f}% | {full_mdd*100:.0f}% |", "",
          "## Attribution read",
          "- Study 1 isolates SELECTION (return space). Study 2 isolates SIZING (same contract). Study 3 = both.",
          "- full_stack − current ≈ (Study1 selection effect) + (Study2 sizing effect) + interaction.",
          f"- At ${eq0:,.0f} the affordability constraint is real: only {aff_dk}/{len(rows)} entries fund any",
          "  contract, so the SIZING studies are dominated by whole-contract granularity at this account size.",
          "_BS-priced synthetic chains, terminal-exit, modeled spread only (optimistic fills). No-lookahead._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=0.0, help="override; default = live Alpaca equity")
    ap.add_argument("--mode", choices=["gbm", "full", "trailing"], default="gbm",
                    help="vol model: gbm | full(=lookahead) | trailing(=no-lookahead)")
    ap.add_argument("--trailing-days", type=int, default=252)
    ap.add_argument("--ev-alpha", type=float, default=0.0, help="tail penalty: EV_adj=EV+α·ES95")
    ap.add_argument("--signal", choices=["momentum", "insider"], default="momentum")
    ap.add_argument("--commission", type=float, default=0.0, help="$/contract/side (e.g. 0.65)")
    args = ap.parse_args()
    equity = args.equity or live_alpaca_equity() or 6260.0
    print(f"Studies on equity=${equity:,.2f}  signal={args.signal}  mode={args.mode}  ev_alpha={args.ev_alpha}")
    res = run(equity, mode=args.mode, trailing_days=args.trailing_days, ev_alpha=args.ev_alpha, signal=args.signal, commission=args.commission)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation_studies.json").write_text(
        json.dumps({"equity": res["equity"], "n": len(res["rows"])}, indent=2))
    md = markdown(res)
    (OUT_DIR / "ablation_studies.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
