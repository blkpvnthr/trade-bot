#!/usr/bin/env python3
"""
backtest_spreads.py — the pivot: VERTICAL SPREADS vs naked long calls on REAL chains.

The whole project's strongest finding is that long ATM calls are structurally fragile (positive
mean, negative median, −97% drawdown). Spreads attack that directly: a bull CALL debit spread
caps the loss at the net debit and sells back some theta/spread; a bull PUT credit spread
collects premium (profits if the underlying merely holds). This tests both against the long-call
baseline on identical entries.

Real AV daily chains (full liquid history), enter/exit at real bid/ask, commission per leg.
Same liquid mega-cap universe + momentum entries + 10-day hold as the long-call studies.

Strategies (all bullish, same entries):
  long_call    — buy ATM call (δ≈0.50)                         [baseline]
  bull_call    — buy δ≈0.50 call, sell δ≈0.30 call (debit)     [capped loss]
  bull_put     — sell δ≈0.30 put, buy δ≈0.15 put (credit)      [premium collection]

Usage: python backtest_spreads.py --max-entries 250
"""
from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo
import study_atm_filters as atm

HOLD = 10
COMMISSION = 0.65        # $/contract/side
DTE_LO, DTE_HI = 21, 49
MAX_SPREAD = 0.10


def _legs(chain, kind, dte_lo, dte_hi):
    """Liquid legs of one kind with valid quotes, in the DTE band, grouped by expiry."""
    out = {}
    for c in chain:
        if c.get("type") != kind or not (dte_lo <= int(c.get("dte", 0)) <= dte_hi):
            continue
        if c.get("bid_price", 0) <= 0 or c.get("ask_price", 0) <= 0:
            continue
        if ((c["ask_price"] - c["bid_price"]) / max(c["mid"], 1e-6)) > MAX_SPREAD:
            continue
        out.setdefault(c["expiration_date"], []).append(c)
    return out


def _nearest(legs, target_abs_delta):
    return min(legs, key=lambda c: abs(abs(c["delta"]) - target_abs_delta)) if legs else None


def _exit_quote(sym, xd, leg, S1, side):
    """Exit price for one leg: 'bid' to sell, 'ask' to buy-to-close. Intrinsic fallback."""
    ec = avo.lookup_contract(sym, xd, leg["symbol"])
    if ec and float(ec.get(f"{side}_price", 0)) > 0:
        return float(ec[f"{side}_price"])
    K = float(leg["strike_price"])
    return max(0.0, (S1 - K) if leg["type"] == "call" else (K - S1))   # intrinsic


def _comm_ps(n_legs):
    return n_legs * 2 * COMMISSION / 100.0    # both sides, per share (÷100)


def long_call(sym, chain, xd, S1):
    c = atm._atm_in_bucket(chain, DTE_LO, DTE_HI)
    if not c:
        return None
    entry = c["ask_price"] + _comm_ps(1) / 2
    exit_px = _exit_quote(sym, xd, c, S1, "bid") - _comm_ps(1) / 2
    if entry <= 0:
        return None
    return max(-1.0, exit_px / entry - 1.0)


def bull_call(sym, chain, xd, S1):
    by_exp = _legs(chain, "call", DTE_LO, DTE_HI)
    best = None
    for exp, legs in by_exp.items():
        lng = _nearest(legs, 0.50); sht = _nearest(legs, 0.30)
        if not lng or not sht or sht["strike_price"] <= lng["strike_price"]:
            continue
        debit = lng["ask_price"] - sht["bid_price"] + _comm_ps(2)
        if debit <= 0:
            continue
        exit_val = (_exit_quote(sym, xd, lng, S1, "bid")
                    - _exit_quote(sym, xd, sht, S1, "ask")) - _comm_ps(2)
        ret = max(-1.0, exit_val / debit - 1.0)
        if best is None or debit < best[1]:        # cheapest valid spread
            best = (ret, debit)
    return best[0] if best else None


def bull_put(sym, chain, xd, S1):
    by_exp = _legs(chain, "put", DTE_LO, DTE_HI)
    best = None
    for exp, legs in by_exp.items():
        sht = _nearest(legs, 0.30); lng = _nearest(legs, 0.15)   # sell 0.30 put, buy 0.15 put
        if not sht or not lng or lng["strike_price"] >= sht["strike_price"]:
            continue
        credit = sht["bid_price"] - lng["ask_price"] - _comm_ps(2)
        if credit <= 0:
            continue
        width = sht["strike_price"] - lng["strike_price"]
        risk = width - credit
        if risk <= 0:
            continue
        cost_close = (_exit_quote(sym, xd, sht, S1, "ask")
                      - _exit_quote(sym, xd, lng, S1, "bid")) + _comm_ps(2)
        pnl = credit - cost_close                  # per share
        ret = max(-1.0, pnl / risk)                # return on capital-at-risk
        if best is None or credit > best[1]:        # richest valid credit
            best = (ret, credit)
    return best[0] if best else None


def run(max_entries: int) -> Dict[str, List[dict]]:
    out = {"long_call": [], "bull_call": [], "bull_put": []}
    n = 0
    for sym in atm.LIQUID:
        if n >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        for i in ab._momentum_entry_indices(td, close, high, vol):
            if n >= max_entries:
                break
            entry_dt = td[i + 1]; exit_dt = td[i + 1 + HOLD]
            ed, xd = entry_dt.isoformat(), exit_dt.isoformat()
            S1 = float(bars.loc[exit_dt, "close"])
            chain = avo.get_chain(sym, ed)         # calls AND puts
            if not chain or S1 <= 0:
                continue
            n += 1
            for name, fn in (("long_call", long_call), ("bull_call", bull_call), ("bull_put", bull_put)):
                r = fn(sym, chain, xd, S1)
                if r is not None:
                    out[name].append({"sym": sym, "date": ed, "ret": r})
    return out


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
            "sharpe": (m / sd) if sd > 0 else 0.0, "win": sum(1 for x in v if x > 0) / nn,
            "mdd": _mdd(rows)}


def report(out):
    L = ["# Vertical Spreads vs Long Calls — real chains, liquid mega-caps", "",
         "Same momentum entries / 10-day hold as the long-call studies. Real AV daily chains, real "
         f"bid/ask, ${COMMISSION}/contract/side. bull_call=δ0.50/0.30 debit; bull_put=δ0.30/0.15 "
         "credit (return on capital-at-risk).", "",
         "| Strategy | n | Mean | Median | Sharpe | Win% | MaxDD |",
         "|---|---|---|---|---|---|---|"]
    for name in ("long_call", "bull_call", "bull_put"):
        s = _stat(out[name])
        if not s:
            L.append(f"| {name} | 0 |—|—|—|—|—|"); continue
        L.append(f"| **{name}** | {s['n']} | {s['mean']*100:+.1f}% | **{s['median']*100:+.1f}%** | "
                 f"{s['sharpe']:+.3f} | {s['win']*100:.0f}% | {s['mdd']*100:.0f}% |")
    L += ["", "_Judge by median + win + MaxDD. Bull_call should compress the long-call −90%+ drawdown; "
          "bull_put (credit) wins when the underlying merely holds — a different risk shape entirely._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=250)
    args = ap.parse_args()
    print(f"Spread backtest vs long calls (real chains, liquid, max {args.max_entries})...")
    out = run(args.max_entries)
    md = report(out)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "spreads.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
