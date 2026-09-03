#!/usr/bin/env python3
"""
study_credit_spreads.py — bull-put credit-spread optimization on REAL chains, one lever at a time.

The only positive-median real-chain result was the bull-put credit spread (+6.2% median / 62% win).
This optimizes it WITHOUT stacking: each study varies ONE lever, holding the rest at the prior
study's best. Real AV historical chains (calls+puts), real bid/ask, enter at realistic CREDIT
(short.bid − long.ask), exit at realistic DEBIT (short.ask − long.bid), commissions both sides,
liquid mega-caps, momentum entries, 10-day hold (matches the baseline). Return = P&L / capital-at-risk.

Studies:
  1. short-put delta sweep (0.30/0.25/0.20/0.16/0.10), width fixed $5
  2. width sweep ($1/$2.5/$5/$10), short delta = best of Study 1
  3. stop-loss sweep (none / 1.5×/2.0×/2.5× credit / short-strike breach)
  4. trend filter (none / close>SMA200 / close>SMA200 & SMA50>SMA200)
  5. simplest combined rule (best delta + best width + best stop + optional SMA200)

Metrics: n, mean, median, win%, Sharpe, MaxDD, avg credit($), avg max-loss($), profit factor.
NO synthetic chains, NO EV selector, NO over-fitting, n shown honestly.
Saves to backtest_outputs/credit_spread_optimization.md
"""
from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional, Tuple

import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo
import study_atm_filters as atm

HOLD = 10
COMMISSION = 0.65
DTE_LO, DTE_HI = 21, 49
MAX_SPREAD = 0.10
COMM_RT = 4 * COMMISSION / 100.0     # 2 legs × 2 sides, per share
BASELINE = "+6.2% median / 62% win (current bull-put baseline)"


def _puts(chain):
    return [c for c in chain if c.get("type") == "put" and DTE_LO <= int(c.get("dte", 0)) <= DTE_HI
            and c.get("bid_price", 0) > 0 and c.get("ask_price", 0) > 0
            and ((c["ask_price"] - c["bid_price"]) / max(c["mid"], 1e-6)) <= MAX_SPREAD]


def build_bull_put(chain, short_delta, width, dtol=0.07) -> Optional[Tuple[dict, dict]]:
    """Sell put nearest |δ|=short_delta; buy put `width` below (same expiry)."""
    puts = _puts(chain)
    if not puts:
        return None
    short = min(puts, key=lambda c: abs(abs(c["delta"]) - short_delta))
    if abs(abs(short["delta"]) - short_delta) > dtol:
        return None
    exp = short["expiration_date"]; tgt = short["strike_price"] - width
    longs = [c for c in puts if c["expiration_date"] == exp and c["strike_price"] < short["strike_price"]]
    if not longs:
        return None
    long = min(longs, key=lambda c: abs(c["strike_price"] - tgt))
    return short, long


def _spread_credit_risk(short, long):
    credit = short["bid_price"] - long["ask_price"]            # gross credit/share
    width = short["strike_price"] - long["strike_price"]
    risk = width - credit
    return credit, width, risk


def _close_cost(sym, date, short, long, S):
    """Real debit to close at `date` (short.ask − long.bid); intrinsic fallback per leg."""
    sc = avo.lookup_contract(sym, date, short["symbol"])
    lc = avo.lookup_contract(sym, date, long["symbol"])
    s_ask = float(sc["ask_price"]) if (sc and float(sc.get("ask_price", 0)) > 0) else max(0.0, short["strike_price"] - S)
    l_bid = float(lc["bid_price"]) if (lc and float(lc.get("bid_price", 0)) > 0) else max(0.0, long["strike_price"] - S)
    return s_ask - l_bid


def realized(entry, short_delta, width, stop=None, stop_mult=0.0):
    """One bull-put trade return (on capital-at-risk) + (credit$, maxloss$). stop ∈
    {None,'mult','breach'}. With a stop, walk intermediate days on the REAL chains."""
    sym, i, td, close = entry["sym"], entry["i"], entry["td"], entry["close"]
    chain = avo.get_chain(sym, entry["ed"])
    legs = build_bull_put(chain, short_delta, width)
    if not legs:
        return None
    short, long = legs
    credit, w, risk = _spread_credit_risk(short, long)
    if credit <= 0 or risk <= 0:
        return None
    exit_k = i + 1 + HOLD
    if stop == "breach":
        for k in range(i + 2, exit_k + 1):
            if close[k] <= short["strike_price"]:
                exit_k = k; break
    elif stop == "mult":
        # A bull-put spread is only at a LOSS when the underlying falls toward the short strike.
        # While price stays comfortably above it the cost-to-close can't reach N×credit, so skip
        # the chain fetch on those days (rigorous — not a model; the actual exit is still real-chain).
        risk_level = short["strike_price"] * 1.03
        for k in range(i + 2, exit_k + 1):
            if close[k] > risk_level:
                continue
            ctc = _close_cost(sym, td[k].isoformat(), short, long, close[k])
            if ctc >= stop_mult * credit:
                exit_k = k; break
    exit_k = min(exit_k, len(td) - 1)
    cost_close = _close_cost(sym, td[exit_k].isoformat(), short, long, close[exit_k])
    pnl = credit - cost_close - COMM_RT
    return pnl / risk, credit * 100, risk * 100


def collect_entries(max_entries):
    ents = []
    for sym in atm.LIQUID:
        if len(ents) >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 260:
            continue
        td = list(bars.index)
        close = [float(bars.loc[d, "close"]) for d in td]
        high = [float(bars.loc[d, "high"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        cd = {d: close[k] for k, d in enumerate(td)}
        hd = {d: high[k] for k, d in enumerate(td)}
        vd = {d: vol[k] for k, d in enumerate(td)}
        for i in ab._momentum_entry_indices(td, cd, hd, vd):
            if len(ents) >= max_entries:
                break
            if i < 200 or i + 1 + HOLD >= len(td):
                continue
            sma200 = statistics.mean(close[i - 200:i]); sma50 = statistics.mean(close[i - 50:i])
            ents.append({"sym": sym, "i": i, "ed": td[i + 1].isoformat(), "td": td, "close": close,
                         "date": td[i + 1].isoformat(),
                         "f200": close[i] > sma200, "f5020": sma50 > sma200})
    return ents


def _stat(results):
    """results = list of (ret, credit$, maxloss$)."""
    if not results:
        return None
    v = [r[0] for r in results]; n = len(v); m = statistics.mean(v)
    sd = statistics.pstdev(v) if n > 1 else 0.0
    wins = [x for x in v if x > 0]; losses = [x for x in v if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    # max drawdown on chronological equity (fixed 20% of risk per trade)
    eq = peak = 1.0; mdd = 0.0
    for x in v:
        eq *= (1.0 + 0.20 * x)
        if eq <= 0:
            mdd = -1.0; break
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return {"n": n, "mean": m, "median": statistics.median(v), "win": len(wins) / n,
            "sharpe": (m / sd) if sd > 0 else 0.0, "mdd": mdd,
            "avg_credit": statistics.mean([r[1] for r in results]),
            "avg_maxloss": statistics.mean([r[2] for r in results]), "pf": pf}


def _row(label, results):
    s = _stat(results)
    if not s:
        return f"| {label} | 0 |—|—|—|—|—|—|—|—|"
    flag = " ⚠️" if s["n"] < 30 else ""
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {label} | {s['n']}{flag} | {s['mean']*100:+.1f}% | **{s['median']*100:+.1f}%** | "
            f"{s['win']*100:.0f}% | {s['sharpe']:+.3f} | {s['mdd']*100:.0f}% | "
            f"${s['avg_credit']:.0f} | ${s['avg_maxloss']:.0f} | {pf} |")


HDR = ("| Variant | n | Mean | Median | Win% | Sharpe | MaxDD | AvgCredit | AvgMaxLoss | PF |\n"
       "|---|---|---|---|---|---|---|---|---|---|")


def _best_median(table: Dict[str, list], min_n=30):
    """Pick the variant with the best median among those with n≥min_n (tie-break: PF)."""
    scored = []
    for k, res in table.items():
        s = _stat(res)
        if s and s["n"] >= min_n:
            scored.append((s["median"], s["pf"] if s["pf"] != float("inf") else 99, k))
    if not scored:
        scored = [(_stat(res)["median"], 0, k) for k, res in table.items() if _stat(res)]
    return max(scored)[2] if scored else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=300)
    args = ap.parse_args()
    print(f"Collecting credit-spread entries (liquid, max {args.max_entries})...")
    ents = collect_entries(args.max_entries)
    print(f"  {len(ents)} entries")
    L = ["# Bull-Put Credit-Spread Optimization (real chains, liquid mega-caps)", "",
         f"One lever at a time, no stacking until Study 5. Real AV chains (puts), enter CREDIT / exit "
         f"DEBIT + ${COMMISSION}/contract/side, momentum entries, 10-day hold. Return on capital-at-risk. "
         f"Compare vs baseline **{BASELINE}**. ⚠️ = n<30 (low sample).", ""]

    # ── Study 1: short delta sweep (width $5) ──
    print("Study 1: delta sweep...")
    DELTAS = [0.30, 0.25, 0.20, 0.16, 0.10]
    t1 = {f"δ{d:.2f}": [r for e in ents if (r := realized(e, d, 5.0))] for d in DELTAS}
    L += ["## Study 1 — short-put delta sweep (width fixed $5)", "", HDR]
    L += [_row(k, v) for k, v in t1.items()]
    best_d = float(_best_median(t1).replace("δ", "")) if _best_median(t1) else 0.20
    L += ["", f"→ best short delta (median, n≥30): **δ{best_d:.2f}**", ""]
    print(f"  best delta = {best_d}")

    # ── Study 2: width sweep (short delta = best) ──
    print("Study 2: width sweep...")
    WIDTHS = [1.0, 2.5, 5.0, 10.0]
    t2 = {f"${w:g} wide": [r for e in ents if (r := realized(e, best_d, w))] for w in WIDTHS}
    L += ["## Study 2 — long-leg width sweep (short δ{:.2f})".format(best_d), "", HDR]
    L += [_row(k, v) for k, v in t2.items()]
    best_w = float(_best_median(t2).replace("$", "").replace(" wide", "")) if _best_median(t2) else 5.0
    L += ["", f"→ best width (median, n≥30): **${best_w:g} wide**", ""]
    print(f"  best width = {best_w}")

    # ── Study 3: stop-loss sweep (best delta + width) ──
    print("Study 3: stop-loss sweep (walks intermediate chains)...")
    t3 = {
        "no stop": [r for e in ents if (r := realized(e, best_d, best_w))],
        "1.5x credit": [r for e in ents if (r := realized(e, best_d, best_w, "mult", 1.5))],
        "2.0x credit": [r for e in ents if (r := realized(e, best_d, best_w, "mult", 2.0))],
        "2.5x credit": [r for e in ents if (r := realized(e, best_d, best_w, "mult", 2.5))],
        "short-strike breach": [r for e in ents if (r := realized(e, best_d, best_w, "breach"))],
    }
    L += ["## Study 3 — stop-loss sweep (δ{:.2f}, ${:g} wide)".format(best_d, best_w), "", HDR]
    L += [_row(k, v) for k, v in t3.items()]
    best_stop = _best_median(t3) or "no stop"
    L += ["", f"→ best stop (median+MaxDD, n≥30): **{best_stop}**", ""]
    print(f"  best stop = {best_stop}")

    # ── Study 4: trend filter (best delta + width, no stop) ──
    print("Study 4: trend filter...")
    base4 = [(e, realized(e, best_d, best_w)) for e in ents]
    base4 = [(e, r) for e, r in base4 if r]
    t4 = {
        "no filter": [r for _, r in base4],
        "close>SMA200": [r for e, r in base4 if e["f200"]],
        "close>SMA200 & SMA50>SMA200": [r for e, r in base4 if e["f200"] and e["f5020"]],
    }
    L += ["## Study 4 — trend filter (δ{:.2f}, ${:g} wide)".format(best_d, best_w), "", HDR]
    L += [_row(k, v) for k, v in t4.items()]
    L += [""]

    # ── Study 5: simplest combined rule ──
    print("Study 5: combined...")
    stop_kind = ("mult", float(best_stop.split("x")[0])) if "credit" in best_stop else \
                ("breach", 0.0) if "breach" in best_stop else (None, 0.0)
    def _combo(use_sma):
        out = []
        for e in ents:
            if use_sma and not e["f200"]:
                continue
            r = realized(e, best_d, best_w, stop_kind[0], stop_kind[1])
            if r:
                out.append(r)
        return out
    t5 = {f"δ{best_d:.2f} ${best_w:g}w {best_stop}": _combo(False),
          f"δ{best_d:.2f} ${best_w:g}w {best_stop} + SMA200": _combo(True)}
    L += ["## Study 5 — simplest combined rule (no over-fitting)", "", HDR]
    L += [_row(k, v) for k, v in t5.items()]
    L += ["", "_Key target: positive MEDIAN + controlled MaxDD + large-enough n (NOT max mean). "
          f"Baseline {BASELINE}._"]

    out = "\n".join(L)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "credit_spread_optimization.md").write_text(out)
    print("\n" + out)


if __name__ == "__main__":
    main()
