#!/usr/bin/env python3
"""
backtest_alpha_rtarget.py — does the alpha engine do better aiming for 4R / 6R (vs 3R)?

R is a PREMIUM multiple (alpha_call_engine: target_value = entry × (1 + r_target); stop at
entry × (1 − stop_loss_pct)). 3R = +300% on the option, stop −28%. This path-simulates each
momentum entry day-by-day to first-touch of the target or stop (else exit at max-hold), for
ATM(δ0.5) and deep-ITM(δ0.8), across r_target ∈ {3,4,6}.

Hypothesis: deep-ITM (~3-4× leverage) can almost never reach +400-600%, so high R targets
mostly leave it exiting at max-hold/stop; ATM (~10× leverage) can hit them on a real run —
so higher R targets should favour ATM. This tells us where to set OPTIONS_TP_R_MULTIPLE.

Signal + universe + BS pricing reuse backtest_alpha_momentum_options. Daily-close touches
(conservative vs intraday). NET applies the extrinsic-spread haircut on exit.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Dict, List

import backtest_insider_options as opt
import backtest_alpha_momentum_options as mom

OUT_DIR = mom.OUT_DIR


def _simulate(bars, td, close, i, delta, dte, sigma, r_target, stop_pct, max_hold, half):
    """Path-simulate one option trade to first target/stop touch. Returns net return."""
    entry_dt = td[i + 1]
    S0 = float(bars.loc[entry_dt, "open"])
    if S0 <= 0:
        return None
    T0 = dte / 365.0
    K = opt.strike_for_delta(S0, T0, sigma, delta)
    entry = opt.bs_call(S0, K, T0, sigma)
    if entry <= 0:
        return None
    extr_in = max(0.0, entry - max(0.0, S0 - K))
    cost = entry + half * extr_in                      # pay half-spread on entry extrinsic
    target = entry * (1.0 + r_target)
    stop = entry * (1.0 - stop_pct)
    for d in range(1, max_hold + 1):
        if i + 1 + d >= len(td):
            break
        ex_dt = td[i + 1 + d]
        S = float(close[ex_dt])
        T = max(0.5 / 365.0, (dte - (ex_dt - entry_dt).days) / 365.0)
        val = opt.bs_call(S, K, T, sigma)
        if val >= target:
            proceeds = target - half * max(0.0, target - max(0.0, S - K))
            return max(-1.0, proceeds / cost - 1.0)
        if val <= stop:
            proceeds = max(0.0, stop - half * max(0.0, stop - max(0.0, S - K)))
            return max(-1.0, proceeds / cost - 1.0)
    # exit at max-hold
    ex_dt = td[min(i + 1 + max_hold, len(td) - 1)]
    S = float(close[ex_dt]); T = max(0.5 / 365.0, (dte - (ex_dt - entry_dt).days) / 365.0)
    val = opt.bs_call(S, K, T, sigma)
    proceeds = max(0.0, val - half * max(0.0, val - max(0.0, S - K)))
    return max(-1.0, proceeds / cost - 1.0)


def _sell(val, S, K, half):
    """Exit proceeds after half-spread haircut on the option's extrinsic value."""
    return max(0.0, val - half * max(0.0, val - max(0.0, S - K)))


def _simulate_ladder(bars, td, close, i, delta, dte, sigma, r_target, stop_pct, max_hold, half,
                     partial_pct, partial_size, trail_activate_r, trail_offset_r):
    """Faithful alpha_call_engine exit ladder (R = full premium):
       hard stop (−stop_pct) → full target (+r_target) → +partial_pct partial (sell partial_size)
       → trailing stop (activate at +trail_activate_r, trail trail_offset_r below peak).
    Daily-close marks; returns net return on the full initial cost."""
    entry_dt = td[i + 1]
    S0 = float(bars.loc[entry_dt, "open"])
    if S0 <= 0:
        return None
    T0 = dte / 365.0
    K = opt.strike_for_delta(S0, T0, sigma, delta)
    entry = opt.bs_call(S0, K, T0, sigma)
    if entry <= 0:
        return None
    cost = entry + half * max(0.0, entry - max(0.0, S0 - K))   # entry incl. half-spread
    hard_stop = entry * (1.0 - stop_pct)
    partial_trig = entry * (1.0 + partial_pct)
    target = entry * (1.0 + r_target)
    trail_activate = entry * (1.0 + trail_activate_r)
    trail_off = trail_offset_r * entry
    remaining, proceeds = 1.0, 0.0
    peak, trail_active, trail_stop, partial_done = entry, False, 0.0, False
    for d in range(1, max_hold + 1):
        if i + 1 + d >= len(td):
            break
        dt_ = td[i + 1 + d]
        S = float(close[dt_]); T = max(0.5 / 365.0, (dte - (dt_ - entry_dt).days) / 365.0)
        val = opt.bs_call(S, K, T, sigma)
        peak = max(peak, val)
        if val <= hard_stop:
            proceeds += remaining * _sell(val, S, K, half); remaining = 0.0; break
        if val >= target:
            proceeds += remaining * _sell(val, S, K, half); remaining = 0.0; break
        if not partial_done and val >= partial_trig:
            proceeds += partial_size * _sell(val, S, K, half); remaining -= partial_size; partial_done = True
        if val >= trail_activate:
            trail_active = True
        if trail_active:
            trail_stop = max(trail_stop, peak - trail_off)
            if val <= trail_stop:
                proceeds += remaining * _sell(val, S, K, half); remaining = 0.0; break
    if remaining > 0:
        dt_ = td[min(i + 1 + max_hold, len(td) - 1)]
        S = float(close[dt_]); T = max(0.5 / 365.0, (dte - (dt_ - entry_dt).days) / 365.0)
        proceeds += remaining * _sell(opt.bs_call(S, K, T, sigma), S, K, half)
    return max(-1.0, proceeds / cost - 1.0)


def _stat(v):
    if not v:
        return {"n": 0}
    n = len(v); m = statistics.mean(v); sd = statistics.pstdev(v) if n > 1 else 0.0
    t = (m / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "mean": m, "win": sum(1 for x in v if x > 0) / n, "t": t,
            "hit": sum(1 for x in v if x > 1.0) / n}  # share returning >+100% (target-ish)


def _max_drawdown(returns_in_order):
    """Max drawdown of a sequential equal-fraction equity curve (proxy: one trade at a time,
    ignores overlap). Each trade compounds (1 + frac·r); frac small so a wipeout can't ruin it."""
    frac = 0.20  # 20% of equity per trade (fixed-fraction); comparable across configs
    eq = 1.0; peak = 1.0; mdd = 0.0
    for r in returns_in_order:
        eq *= (1.0 + frac * r)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    return mdd


def _risk(returns):
    """Risk-adjusted per-trade stats for a return list (NOT pre-ordered)."""
    if not returns:
        return {"n": 0}
    v = sorted(returns)
    n = len(v); m = statistics.mean(v); sd = statistics.pstdev(v) if n > 1 else 0.0
    wins = [x for x in v if x > 0]; losses = [x for x in v if x < 0]
    sharpe = (m / sd) if sd > 0 else 0.0                    # per-trade Sharpe (mean/stdev)
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")  # profit factor
    p5 = v[max(0, int(0.05 * n) - 1)]
    return {"n": n, "mean": m, "std": sd, "sharpe": sharpe,
            "win": len(wins) / n, "median": statistics.median(v),
            "p5": p5, "worst": v[0], "wipe": sum(1 for x in v if x <= -0.99) / n, "pf": pf}


def _equity_curve(returns_in_order, frac):
    """Sequential fixed-fraction equity curve → (terminal_multiple, max_drawdown)."""
    eq = 1.0; peak = 1.0; mdd = 0.0
    for r in returns_in_order:
        eq *= (1.0 + frac * r)
        if eq <= 0:
            return 0.0, -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return eq, mdd


def kelly_markdown(res):
    """Risk as a function of the Kelly fraction — the lever the drawdowns say matters most.
    Sweeps fractions of the ladder-optimal full Kelly for the deep-ITM (winner) strategy."""
    p = res["params"]
    rows = sorted(res["rows"], key=lambda r: r.get("entry_date", ""))
    rt = res["rtargets"][0]
    L = ["", "## Risk vs KELLY FRACTION — deep-ITM ladder (the sizing lever)", ""]
    try:
        import options_quant as oq
    except Exception:
        return "\n".join(L + ["(options_quant unavailable)"])
    for tag, label in [("itm", "deep-ITM δ0.8"), ("atm", "ATM δ0.5")]:
        col = f"{tag}_{rt}_ladder"
        rets = [r[col] for r in rows if r.get(col) is not None]
        if not rets:
            continue
        f_full = oq.kelly_from_scenarios(rets) or 0.0
        L += [f"### {label} — empirical full-Kelly f* = **{f_full:.3f}** (growth-optimal fraction of equity/trade)",
              "",
              "| Kelly mult | fraction/trade | Terminal ×equity | **Max DD** | Calmar (growth/|DD|) |",
              "|---|---|---|---|---|"]
        for mult, mlabel in [(0.25, "¼-Kelly"), (0.5, "½-Kelly"), (1.0, "full-Kelly"),
                             (1.5, "1.5×-Kelly"), (2.0, "2×-Kelly")]:
            frac = f_full * mult
            term, mdd = _equity_curve(rets, frac)
            growth = term - 1.0
            calmar = (growth / abs(mdd)) if mdd < 0 else float("inf")
            cal = "∞" if calmar == float("inf") else f"{calmar:.2f}"
            L.append(f"| {mlabel} | {frac*100:.1f}% | {term:,.1f}× | **{mdd*100:.0f}%** | {cal} |")
        L.append("")
    L += ["## Read (sizing)",
          "- Growth rises toward full-Kelly then the variance drag flattens it; DRAWDOWN keeps exploding past it.",
          "- ¼–½-Kelly is the standard sweet spot: most of the growth, a fraction of the drawdown.",
          "- This is why the R-target is a sideshow — the fraction-of-equity choice dominates the risk profile.",
          "_Sequential one-trade-at-a-time curve (ignores overlap); comparative, not a literal account path._", ""]
    return "\n".join(L)


def risk_markdown(res):
    p = res["params"]
    rows = sorted(res["rows"], key=lambda r: r.get("entry_date", ""))   # chronological for DD
    L = ["", "## Risk-adjusted view — LADDER (real engine), per-trade", "",
         f"Sharpe = per-trade mean/stdev. MaxDD = sequential 20%-fixed-fraction equity curve "
         f"(proxy, ignores overlap). PF = profit factor. Spread {p['spread']:.0%}, "
         f"trail +{p['trail_activate']*100:.0f}%.", "",
         "| Instrument | R | Mean | Std | **Sharpe** | Win% | Median | p5 (tail) | **MaxDD** | PF |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, label in [("atm", "ATM δ0.5"), ("itm", "deep-ITM δ0.8")]:
        for rt in res["rtargets"]:
            col = f"{tag}_{rt}_ladder"
            vals = [r[col] for r in rows if r.get(col) is not None]
            s = _risk(vals)
            if not s.get("n"):
                continue
            mdd = _max_drawdown([r[col] for r in rows if r.get(col) is not None])
            pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
            L.append(f"| {label} | {rt:g}R | {s['mean']*100:+.1f}% | {s['std']*100:.0f}% | "
                     f"**{s['sharpe']:+.3f}** | {s['win']*100:.0f}% | {s['median']*100:+.1f}% | "
                     f"{s['p5']*100:+.0f}% | **{mdd*100:.1f}%** | {pf} |")
    L += ["", "## Read (risk-adjusted)",
          "- **Sharpe** (mean/stdev) is the real protective-vs-aggressive decider — not mean alone.",
          "- deep-ITM should show higher Sharpe + shallower MaxDD + higher win% than ATM (its intrinsic",
          "  value dampens variance); ATM is a lower-Sharpe lottery (fat tail, deep p5).",
          "- If Sharpe is ~flat across R-targets, the target is irrelevant risk-adjusted → pick the lowest",
          "  (3R) for simplicity; the trailing stop is doing the real work."]
    return "\n".join(L)


def run(args):
    half = args.spread / 2.0
    rtargets = [float(x) for x in str(args.rtargets).split(",") if x.strip()]
    out: Dict[str, Any] = {"params": vars(args), "rtargets": rtargets, "rows": []}
    for sym in mom._universe():
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        i = 55
        while i < len(td) - 1:
            d = td[i]
            prior_high = max(high[td[j]] for j in range(i - args.breakout, i))
            sma = statistics.mean(close[td[j]] for j in range(i - args.trend, i))
            avgvol = statistics.mean([vol[td[j]] for j in range(i - 20, i)] or [0])
            relvol = vol[d] / avgvol if avgvol > 0 else 0.0
            if not (close[d] > prior_high and close[d] > sma and relvol > args.relvol):
                i += 1; continue
            sigma = opt.realised_vol([close[td[j]] for j in range(max(0, i - 30), i)]) * args.iv_mult
            rec = {"symbol": sym, "entry_date": td[i + 1].isoformat()}
            for rt in rtargets:
                for tag, delta in [("atm", args.atm_delta), ("itm", args.itm_delta)]:
                    rec[f"{tag}_{rt}_simple"] = _simulate(bars, td, close, i, delta, args.dte, sigma,
                                                          rt, args.stop_pct, args.max_hold, half)
                    rec[f"{tag}_{rt}_ladder"] = _simulate_ladder(
                        bars, td, close, i, delta, args.dte, sigma, rt, args.stop_pct, args.max_hold,
                        half, args.partial_pct, args.partial_size, args.trail_activate, args.trail_offset)
            out["rows"].append(rec)
            i += args.max_hold  # non-overlapping
    return out


def markdown(res):
    p = res["params"]
    L = ["# Alpha Momentum — R-target sweep with TRAILING-STOP ladder", "",
         f"Two exit models per R-target: SIMPLE (target/stop only) vs LADDER (faithful engine: "
         f"hard stop −{p['stop_pct']*100:.0f}%, +{p['partial_pct']*100:.0f}% partial sells "
         f"{p['partial_size']*100:.0f}%, trail activates +{p['trail_activate']*100:.0f}% / "
         f"{p['trail_offset']*100:.0f}% behind peak, full target +R). "
         f"Max-hold {p['max_hold']}d, {p['dte']}DTE, IV×{p['iv_mult']}, spread {p['spread']:.0%}. "
         f"n={len(res['rows'])}.", "",
         "| R-target | ATM simple | **ATM ladder** | deep-ITM simple | **deep-ITM ladder** | ladder winner |",
         "|---|---|---|---|---|---|"]
    for rt in res["rtargets"]:
        def m(tag, mode):
            return _stat([r.get(f"{tag}_{rt}_{mode}") for r in res["rows"]
                          if r.get(f"{tag}_{rt}_{mode}") is not None])
        a_s, a_l = m("atm", "simple"), m("atm", "ladder")
        i_s, i_l = m("itm", "simple"), m("itm", "ladder")
        if not a_l.get("n"):
            continue
        win = "ATM" if a_l["mean"] > i_l["mean"] else "deep-ITM"
        L.append(f"| {rt:g}R | {a_s['mean']*100:+.1f}% | **{a_l['mean']*100:+.1f}%** | "
                 f"{i_s['mean']*100:+.1f}% | **{i_l['mean']*100:+.1f}%** | **{win}** |")
    L += ["", "## Read",
          "- **LADDER** is the real engine: the trailing stop + partial cash in winners early, so it",
          "  shows how much of the simple-target 4R/6R gain actually SURVIVES live management.",
          "- If ladder net is FLAT across 3R→6R, the trailing stop is capping winners before the high",
          "  target → raising r_target alone won't help; raise r_trail_activate too (let winners run).",
          "- If ladder net still RISES with R, the higher target is reachable under trailing → aim higher.",
          "", "_Daily-close marks (conservative on intraday touches); NET is an optimistic fill ceiling._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtargets", type=str, default="3,4,6")
    ap.add_argument("--stop-pct", type=float, default=0.28)
    ap.add_argument("--max-hold", type=int, default=25)
    ap.add_argument("--breakout", type=int, default=20)
    ap.add_argument("--trend", type=int, default=50)
    ap.add_argument("--relvol", type=float, default=1.5)
    ap.add_argument("--dte", type=int, default=35)
    ap.add_argument("--atm-delta", type=float, default=0.50)
    ap.add_argument("--itm-delta", type=float, default=0.80)
    ap.add_argument("--iv-mult", type=float, default=1.2)
    ap.add_argument("--spread", type=float, default=0.03)
    # engine ladder params (alpha_call_engine defaults)
    ap.add_argument("--partial-pct", type=float, default=0.5)      # +50% partial trigger
    ap.add_argument("--partial-size", type=float, default=0.5)     # sell 50%
    ap.add_argument("--trail-activate", type=float, default=1.7)   # activate at +170% (1.7R)
    ap.add_argument("--trail-offset", type=float, default=0.8)     # trail 0.8R behind peak
    args = ap.parse_args()
    res = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "alpha_rtarget_backtest.json").write_text(
        json.dumps({**res, "rows": len(res["rows"])}, indent=2, default=str))
    md = markdown(res) + "\n" + risk_markdown(res) + "\n" + kelly_markdown(res)
    (OUT_DIR / "alpha_rtarget_backtest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
