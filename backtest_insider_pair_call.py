#!/usr/bin/env python3
"""
backtest_insider_pair_call.py — does the PAIRED-CALL overlay add value on REAL fills?

Tests the live feature (run.py INSIDER_COPY_PAIR_CALL): every vol-gated insider SHARE buy ALSO
buys a just-ITM ~31-DTE long call; both legs exit together on the share's hold-day close.

Honest by construction:
  - SHARE leg: the VALIDATED rule — entry-day realized_vol_20 >= TRAIN P80, 20d $-vol >= min-adv,
    enter next open / exit +hold close, round-trip slippage. (Mirrors backtest_vol_gated_insider.)
  - CALL leg: REAL historical chains (av_historical_options, AV HISTORICAL_OPTIONS premium).
    Entry pays the real ASK; exit sells at the real BID on the share's exit date (intrinsic floor
    only if that day's chain lacks the contract). NO Black-Scholes — synthetic-chain profits are a
    known artifact in this repo (see project_av_chains_backtest).
  - Selection mirrors run._select_insider_pair_call: nearest target-DTE, highest strike <= spot
    (just ITM), liquidity floor.

Reports shares-only vs call-only vs shares+call: n, mean, median, win%, Sharpe, PF, total $PnL,
plus real-chain coverage and how many call exits used the intrinsic fallback.

Usage:
  python backtest_insider_pair_call.py --hold 3 --min-adv 1000000 --call-dte 31 \
      --max-entries 150 --share-dollars 1000 --call-dollars 1500 --slip-bps 100
"""
from __future__ import annotations

import argparse
import statistics
from datetime import date as _date
from typing import Dict, List, Optional

import numpy as np

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo


def collect_signals(hold: int, min_adv: float) -> List[dict]:
    """Every insider-entry SHARE trade with entry-day vol + ADV (no lookahead). Keeps the entry/exit
    DATES, share prices, and entry spot needed to price a paired call on real chains."""
    pit = base.load_pit_buy_events()
    rows = []
    for sym in sorted(pit.keys()):
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        op = [float(bars.loc[d, "open"]) for d in td]
        cl = [float(bars.loc[d, "close"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        r1 = np.diff(np.log(np.maximum(cl, 1e-9)))
        td_dates = [d.date() if hasattr(d, "date") else d for d in td]
        for i in ab._insider_entry_indices(td_dates, sym, pit):
            if i < 25 or i + 1 + hold >= len(td):
                continue
            rv20 = float(np.std(r1[i - 20:i])) if i >= 21 else np.nan
            adv = float(np.mean([cl[k] * vol[k] for k in range(i - 20, i)]))
            if not np.isfinite(rv20) or adv < min_adv:
                continue
            entry = op[i + 1]; exit_px = cl[i + 1 + hold]
            if entry <= 0 or exit_px <= 0:
                continue
            rows.append({"sym": sym, "entry_date": td_dates[i + 1], "exit_date": td_dates[i + 1 + hold],
                         "share_entry": entry, "share_exit": exit_px, "spot": entry,
                         "rv20": rv20, "adv": adv})
    return rows


def price_pair_call(sym: str, entry_date: _date, exit_date: _date, spot: float, share_exit: float,
                    target_dte: int, min_oi: int, commission: float, call_budget: float,
                    target_delta: float = 0.0) -> Optional[dict]:
    """Select a ~target-DTE call on the entry-date REAL chain, enter at ask, exit at the real bid on
    exit_date (intrinsic floor if that day's chain lacks the contract). Selection:
      target_delta<=0 → JUST-ITM (highest strike ≤ spot);
      target_delta>0  → DEEP-ITM (call delta nearest target_delta, requires a real delta).
    Returns leg dict or None when no tradable contract / no chain."""
    chain = avo.get_chain(sym, entry_date.isoformat(), calls_only=True)
    if not chain:
        return {"status": "no_chain"}
    best, bk = None, None
    for c in chain:
        try:
            strike = float(c.get("strike_price") or 0.0)
            ask = float(c.get("ask_price") or 0.0)
            dte = int(c.get("dte") or 0)
            if strike <= 0 or ask <= 0 or dte <= 0:
                continue
            if int(c.get("open_interest") or 0) < min_oi and int(c.get("volume") or 0) <= 0:
                continue
            if target_delta > 0:
                delta = float(c.get("delta") or 0.0)
                if not (0.0 < delta <= 0.98):       # deep-ITM needs a usable delta
                    continue
                key = (abs(dte - target_dte), abs(delta - target_delta))
            else:
                itm = strike <= spot
                tier = 0 if itm else 1
                dist = (spot - strike) if itm else (strike - spot)
                key = (abs(dte - target_dte), tier, dist)
            if bk is None or key < bk:
                bk, best = key, c
        except Exception:
            continue
    if not best:
        return {"status": "no_contract"}
    cid = best["symbol"]
    strike = float(best["strike_price"]); entry_ask = float(best["ask_price"])
    qty = int(call_budget // (entry_ask * 100.0))
    if qty < 1:
        return {"status": "too_expensive", "premium": entry_ask * 100.0}
    # exit at the real bid on the share's exit date; intrinsic floor only if the contract is missing.
    ex = avo.lookup_contract(sym, exit_date.isoformat(), cid)
    fallback = False
    if ex and float(ex.get("bid_price") or 0.0) > 0:
        exit_px = float(ex["bid_price"])
    else:
        exit_px = max(0.0, share_exit - strike)      # conservative intrinsic floor (undervalues time)
        fallback = True
    premium = qty * entry_ask * 100.0
    pnl = qty * (exit_px - entry_ask) * 100.0 - 2.0 * commission * qty
    return {"status": "ok", "qty": qty, "strike": strike, "dte": int(best.get("dte") or 0),
            "delta": float(best.get("delta") or 0.0), "entry_ask": entry_ask, "exit_px": exit_px,
            "premium": premium, "pnl": pnl, "ret": pnl / premium if premium > 0 else 0.0,
            "fallback": fallback}


def stat(rets: List[float], pnls: Optional[List[float]] = None) -> Optional[dict]:
    if not rets:
        return None
    v = np.array(rets, dtype=float)
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)), "win": (v > 0).mean(),
            "sharpe": (v.mean() / sd) if sd > 0 else 0.0, "pf": pf,
            "total": float(np.sum(pnls)) if pnls is not None else None}


def fmt(s: Optional[dict]) -> str:
    if not s:
        return "| 0 | — | — | — | — | — | — |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    tot = f"${s['total']:,.0f}" if s.get("total") is not None else "—"
    return (f"| {s['n']:,} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | {s['win']*100:.0f}% | "
            f"{s['sharpe']:+.3f} | {pf} | {tot} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--min-adv", type=float, default=1e6)
    ap.add_argument("--vol-pctile", type=float, default=80)
    ap.add_argument("--train-end", default="2024-01-01")
    ap.add_argument("--call-dte", type=int, default=31)
    ap.add_argument("--call-min-oi", type=int, default=10)
    ap.add_argument("--target-delta", type=float, default=0.0,
                    help="0 = just-ITM (highest strike <= spot); >0 = deep-ITM call nearest this delta (e.g. 0.78)")
    ap.add_argument("--share-dollars", type=float, default=1000.0)
    ap.add_argument("--call-dollars", type=float, default=1500.0)
    ap.add_argument("--slip-bps", type=float, default=100.0)
    ap.add_argument("--commission", type=float, default=0.65)
    ap.add_argument("--max-entries", type=int, default=150, help="cap real-chain lookups (most recent OOS)")
    args = ap.parse_args()

    print(f"Collecting vol-gated insider signals (hold={args.hold}d, min ADV=${args.min_adv:,.0f})...")
    rows = collect_signals(args.hold, args.min_adv)
    te = _date.fromisoformat(args.train_end)
    train, test = [r for r in rows if r["entry_date"] < te], [r for r in rows if r["entry_date"] >= te]
    if not train or not test:
        print("insufficient data"); return
    vthr = float(np.percentile([r["rv20"] for r in train], args.vol_pctile))
    gated = [r for r in test if r["rv20"] >= vthr]
    gated.sort(key=lambda r: r["entry_date"], reverse=True)
    sample = gated[:args.max_entries]
    print(f"  {len(rows):,} total / OOS test {len(test):,} / vol-gated {len(gated):,}; "
          f"vol P{args.vol_pctile:.0f}={vthr:.4f}. Pricing real chains for {len(sample)} most-recent...")

    slip = args.slip_bps / 10000.0
    sh_ret, sh_pnl = [], []
    cl_ret, cl_pnl = [], []
    cb_ret, cb_pnl = [], []
    cov = {"ok": 0, "no_chain": 0, "no_contract": 0, "too_expensive": 0, "fallback": 0}
    for k, r in enumerate(sample):
        share_ret = r["share_exit"] / r["share_entry"] - 1.0 - slip
        leg = price_pair_call(r["sym"], r["entry_date"], r["exit_date"], r["spot"], r["share_exit"],
                              args.call_dte, args.call_min_oi, args.commission, args.call_dollars,
                              target_delta=args.target_delta)
        st = (leg or {}).get("status", "no_chain")
        # share leg always counts (the baseline)
        sh_ret.append(share_ret); sh_pnl.append(args.share_dollars * share_ret)
        if st == "ok":
            cov["ok"] += 1
            if leg["fallback"]:
                cov["fallback"] += 1
            cl_ret.append(leg["ret"]); cl_pnl.append(leg["pnl"])
            cap = args.share_dollars + leg["premium"]
            pnl = args.share_dollars * share_ret + leg["pnl"]
            cb_ret.append(pnl / cap); cb_pnl.append(pnl)
        else:
            cov[st] = cov.get(st, 0) + 1
            # no tradable call → shares-only for this name
            cb_ret.append(share_ret); cb_pnl.append(args.share_dollars * share_ret)
        if (k + 1) % 25 == 0:
            print(f"    {k+1}/{len(sample)} priced (calls ok={cov['ok']}, no_chain={cov['no_chain']})...")
        if getattr(avo, "_quota_hit", False):
            print(f"    AV quota hit at {k+1}; reporting on what priced so far.")
            break

    s_sh, s_cl, s_cb = stat(sh_ret, sh_pnl), stat(cl_ret, cl_pnl), stat(cb_ret, cb_pnl)
    _mode = f"deep-ITM δ≈{args.target_delta:.2f}" if args.target_delta > 0 else "just-ITM"
    L = [f"# Insider Paired-Call Overlay — REAL chains (hold {args.hold}d, call ~{args.call_dte}DTE {_mode})", "",
         f"Vol-gated insider SHARES (entry-day rv20 ≥ TRAIN P{args.vol_pctile:.0f}={vthr:.4f}, 20d $-vol ≥ "
         f"${args.min_adv:,.0f}), {args.slip_bps:.0f}bps share slippage. Paired just-ITM ~{args.call_dte}DTE "
         f"long call on REAL AV historical chains (entry ask → exit bid, ${args.commission:.2f}/contract). "
         f"Sizing per name: ${args.share_dollars:,.0f} shares + up to ${args.call_dollars:,.0f} call premium.", "",
         f"OOS sample: {len(sample)} most-recent vol-gated signals (of {len(gated):,}).", "",
         f"Real-chain coverage: {cov['ok']} tradable calls, {cov['no_chain']} no-chain, "
         f"{cov['no_contract']} no-qualifying-contract, {cov['too_expensive']} premium>budget, "
         f"{cov['fallback']} exits used intrinsic fallback.", "",
         "| Strategy | n | Mean | Median | Win% | Sharpe | PF | Total $PnL |",
         "|---|---|---|---|---|---|---|---|",
         "| Shares only (baseline) " + fmt(s_sh),
         "| Call only (real fills) " + fmt(s_cl),
         "| **Shares + paired call** " + fmt(s_cb), ""]
    if s_sh and s_cb:
        d_mean = (s_cb["mean"] - s_sh["mean"]) * 100
        d_tot = (s_cb["total"] - s_sh["total"]) if (s_cb.get("total") and s_sh.get("total")) else 0.0
        L += [f"_Overlay marginal effect: **{d_mean:+.2f}%** mean/name, **${d_tot:+,.0f}** total $PnL "
              f"vs shares-only; Sharpe {s_cb['sharpe']:+.3f} vs {s_sh['sharpe']:+.3f}._", ""]
        verdict = ("ADDS value" if (d_mean > 0 and s_cb["sharpe"] >= s_sh["sharpe"]) else "DOES NOT add value")
        L += [f"**VERDICT: the paired-call overlay {verdict} on real fills in this sample.**",
              "_Caveat: long calls bleed theta/spread on real chains (repo prior). Treat a positive "
              "result as provisional until confirmed on a larger sample + multiple slippage assumptions._"]
    out = "\n".join(L)
    _tag = f"deepitm{int(args.target_delta*100)}" if args.target_delta > 0 else "justitm"
    (base.PROJECT_ROOT / "backtest_outputs" / f"insider_pair_call_real_chains_{_tag}.md").write_text(out)
    print("\n" + out)


if __name__ == "__main__":
    main()
