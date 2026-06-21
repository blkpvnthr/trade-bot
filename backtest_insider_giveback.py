#!/usr/bin/env python3
"""backtest_insider_giveback.py - Track-1-step-3: give-back EXIT falsification on the insider book.

Tests whether a give-back-aware early exit beats the incumbent fixed 3-day hold on the
vol-gated insider-shares edge, AT THE DAILY 3-DAY HORIZON (the intraday give_back AUC 0.755 is
horizon-mismatched; the daily variant is only 0.668 and monetization is twice-unproven, so we
build the exit RULE directly and grade it, not borrow the AUC).

Give-back exit = a trailing peak-retracement stop over the hold: track MFE via daily highs;
if the position has run up >= GB_MFE and then gives back >= GB_FRAC of that peak, exit at that
day's close; else exit at the 3-day timer. One round trip either way (100bps). Pure exit
overlay - same entries, same vol gate, flat sizing (exposure scaling already falsified).

STRICT BAR (must pass ALL):
  1. net improvement: total return AND Sharpe beat the fixed-hold baseline,
  2. tail preserved: does NOT truncate the top winners (the right-tail IS the edge),
  3. not luck: beats a placebo of random early exits at the same frequency.
If it fails -> conclusion: flat sizing + fixed 3-day hold; do not over-manage the edge.

  python backtest_insider_giveback.py
  python backtest_insider_giveback.py --selftest
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

import backtest_vol_gated_insider as vg
import backtest_alpha_momentum_options as mom
from backtest_insider_exposure import (COST, ALLOC, MAX_OPEN, START_EQUITY, HOLD,
                                       TRAIN_END, OUT)

GB_MFE_GRID = [0.005, 0.01, 0.02]    # min peak gain before the give-back stop arms
GB_FRAC_GRID = [0.3, 0.5, 0.7]       # fraction of peak surrendered that triggers exit
PLACEBO_N = 200


# ---- path-aware trade reconstruction (mirrors vg._collect entry/exit) ---------------
def collect_paths(hold: int = HOLD) -> list[dict]:
    """The same vol-gated trades as the exposure study, each with its daily holding path."""
    ins = vg._collect(hold=hold, min_adv=1_000_000)
    vthr = float(np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80))
    trades = sorted([r for r in ins if r["rv20"] >= vthr], key=lambda r: r["date"])
    out = []
    bars_cache: dict = {}
    for t in trades:
        sym = t["sym"]
        if sym not in bars_cache:
            bars_cache[sym] = mom._bars(sym)
        b = bars_cache[sym]
        if b is None or t["date"] not in b.index:
            continue
        idx = list(b.index).index(t["date"])
        if idx + hold >= len(b):
            continue
        entry = float(b.iloc[idx]["open"])
        if entry <= 0:
            continue
        # path over holding days k=0..hold (decision at each day's close, peak via daily high)
        highs = [float(b.iloc[idx + k]["high"]) / entry - 1.0 for k in range(hold + 1)]
        closes = [float(b.iloc[idx + k]["close"]) / entry - 1.0 for k in range(hold + 1)]
        out.append({"sym": sym, "date": t["date"], "gross": closes[hold],
                    "highs": highs, "closes": closes, "hold": hold})
    return out


def giveback_exit(tr: dict, gb_mfe: float, gb_frac: float) -> tuple[float, int]:
    """Return (realized_return, days_held). Trailing give-back stop on daily closes."""
    peak = 0.0
    for k in range(tr["hold"] + 1):
        peak = max(peak, tr["highs"][k])
        if k == 0:
            continue                       # need at least one full day before stopping out
        cur = tr["closes"][k]
        if peak >= gb_mfe and (peak - cur) >= gb_frac * peak:
            return cur, k                  # exit at this close
    return tr["closes"][tr["hold"]], tr["hold"]


# ---- portfolio sim (flat sizing, VARIABLE hold so capacity frees at the real exit) --
def equity_curve(legs: list[dict]):
    """legs: [{date, net, hold_k}]. Risk-sized, max-3-concurrent. Returns eq, daily, per-trade net."""
    daily: dict = {}
    open_until: list = []
    pt = []
    for r in sorted(legs, key=lambda x: x["date"]):
        d0 = r["date"]
        open_until = [x for x in open_until if x > d0]
        if len(open_until) >= MAX_OPEN:
            continue
        net, hk = r["net"], max(1, r["hold_k"])
        pt.append(net)
        per_day = (1 + net) ** (1 / hk) - 1
        for k in range(hk):
            dd = d0 + timedelta(days=k)
            daily[dd] = daily.get(dd, 0.0) + ALLOC * per_day
        open_until.append(d0 + timedelta(days=hk))
    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)
    eq = START_EQUITY * (1 + s).cumprod()
    return eq, s, np.array(pt)


def metrics(legs: list[dict]) -> dict:
    eq, ret, pt = equity_curve(legs)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    tot = float(eq.iloc[-1] / eq.iloc[0] - 1)
    maxdd = float((eq / eq.cummax() - 1).min())
    act = ret[ret != 0]
    sharpe = float(act.mean() / act.std() * np.sqrt(252)) if len(act) and act.std() > 0 else 0.0
    w, l = pt[pt > 0], pt[pt < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) and l.sum() != 0 else float("inf")
    order = np.argsort(pt)[::-1]
    return {
        "trades_taken": int(len(pt)),
        "win_rate": round(float((pt > 0).mean()), 4),
        "avg_R_net": round(float(pt.mean()), 5),
        "total_return": round(tot, 4),
        "sharpe_active": round(sharpe, 3),
        "max_drawdown": round(maxdd, 4),
        "calmar": round((tot / yrs) / abs(maxdd), 3) if maxdd < 0 and yrs > 0 else None,
        "pf": "inf" if pf == float("inf") else round(pf, 3),
        "drop_top3_total": round(_drop_total(legs, 3), 4),
        "drop_top5_total": round(_drop_total(legs, 5), 4),
    }


def _drop_total(legs, n):
    order = sorted(range(len(legs)), key=lambda i: legs[i]["net"], reverse=True)
    keep = [legs[i] for i in order[n:]]
    if not keep:
        return 0.0
    eq, *_ = equity_curve(keep)
    return float(eq.iloc[-1] / eq.iloc[0] - 1)


def _legs(trades, realized_fn):
    out = []
    for t in trades:
        r, hk = realized_fn(t)
        out.append({"date": t["date"], "net": r - COST, "hold_k": hk})
    return out


def winner_retention(trades, gb_mfe, gb_frac, k=10) -> dict:
    """Of the top-k baseline winners, how much of their gross does give-back keep?"""
    top = sorted(trades, key=lambda t: t["gross"], reverse=True)[:k]
    rets = [(t["gross"], giveback_exit(t, gb_mfe, gb_frac)[0]) for t in top]
    cut = sum(1 for g, r in rets if r < g - 1e-6)
    retention = float(np.mean([r / g for g, r in rets if g > 1e-6])) if rets else 1.0
    return {"top_k": k, "winners_cut": cut, "mean_retention": round(retention, 3),
            "baseline_mean": round(float(np.mean([g for g, _ in rets])), 4),
            "giveback_mean": round(float(np.mean([r for _, r in rets])), 4)}


def placebo_total(trades, exit_rate, rng) -> float:
    """Random early exits at the same frequency: exit a random fraction on a random day."""
    legs = []
    for t in trades:
        if rng.random() < exit_rate:
            k = int(rng.integers(1, t["hold"]))      # random day 1..hold-1
            legs.append({"date": t["date"], "net": t["closes"][k] - COST, "hold_k": k})
        else:
            legs.append({"date": t["date"], "net": t["gross"] - COST, "hold_k": t["hold"]})
    eq, *_ = equity_curve(legs)
    return float(eq.iloc[-1] / eq.iloc[0] - 1)


def overlay_promotable(best: dict, baseline: dict, placebo_q90: float) -> tuple[bool, dict]:
    """The promotion bar for ANY insider-book overlay (sizing or exit).

    Headline Sharpe/return improvement is necessary but NOT sufficient: an overlay must also
    preserve the right tail (the edge) and beat a same-frequency random early-exit placebo.
    `best` carries a `winner_retention` dict {mean_retention, winners_cut}.
    """
    wr = best["winner_retention"]
    checks = {
        "net_improves": best["total_return"] > baseline["total_return"],
        "sharpe_holds": best["sharpe_active"] >= baseline["sharpe_active"],
        "tail_preserved": wr["mean_retention"] >= 0.85 and wr["winners_cut"] <= 2,
        "beats_placebo": best["total_return"] > placebo_q90,
    }
    return all(checks.values()), checks


def run() -> dict:
    trades = collect_paths()
    base_legs = _legs(trades, lambda t: (t["gross"], t["hold"]))
    base_m = metrics(base_legs)

    grid = {}
    for mfe in GB_MFE_GRID:
        for frac in GB_FRAC_GRID:
            legs = _legs(trades, lambda t, a=mfe, b=frac: giveback_exit(t, a, b))
            m = metrics(legs)
            exited = sum(1 for t in trades if giveback_exit(t, mfe, frac)[1] < t["hold"])
            m["early_exit_rate"] = round(exited / len(trades), 3)
            m["winner_retention"] = winner_retention(trades, mfe, frac)
            grid[f"mfe{mfe}_frac{frac}"] = m

    # best by total return; grade against the strict bar
    best_key = max(grid, key=lambda k: grid[k]["total_return"])
    best = grid[best_key]
    rng = np.random.default_rng(0)
    placebo = np.array([placebo_total(trades, best["early_exit_rate"], rng)
                        for _ in range(PLACEBO_N)])
    p_q90 = float(np.quantile(placebo, 0.90))

    passes, checks = overlay_promotable(best, base_m, p_q90)
    verdict = ("PASS - give-back exit improves net AND preserves the tail"
               if passes else
               "FAIL - flat sizing + fixed 3-day hold; do not over-manage the edge")

    payload = {
        "strategy": "vol_gated_insider_shares",
        "test": "giveback_exit_overlay_daily_3d",
        "params": {"hold": HOLD, "cost_bps": COST * 1e4, "gb_mfe_grid": GB_MFE_GRID,
                   "gb_frac_grid": GB_FRAC_GRID, "placebo_n": PLACEBO_N},
        "n_trades": len(trades),
        "baseline_fixed_hold": base_m,
        "grid": grid,
        "best_variant": best_key,
        "placebo_random_exits": {"mean": round(float(placebo.mean()), 4), "q90": round(p_q90, 4)},
        "strict_bar": {k: bool(v) for k, v in checks.items()},
        "verdict": verdict,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "insider_giveback_backtest.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "insider_giveback_backtest.md").write_text(_md(payload))
    return payload


def _md(p: dict) -> str:
    b = p["baseline_fixed_hold"]; best = p["grid"][p["best_variant"]]; wr = best["winner_retention"]
    L = ["# Insider book - give-back EXIT falsification (Track-1-step-3)", "",
         f"_{p['n_trades']} vol-gated trades, {p['params']['cost_bps']:.0f}bps, daily 3d horizon. "
         "Trailing give-back stop vs fixed-hold baseline._", "",
         "| Metric | baseline (fixed 3d) | best give-back |", "|---|---|---|",
         f"| variant | fixed_hold | {p['best_variant']} |",
         f"| total_return | {b['total_return']} | {best['total_return']} |",
         f"| sharpe_active | {b['sharpe_active']} | {best['sharpe_active']} |",
         f"| max_drawdown | {b['max_drawdown']} | {best['max_drawdown']} |",
         f"| pf | {b['pf']} | {best['pf']} |",
         f"| drop_top5_total | {b['drop_top5_total']} | {best['drop_top5_total']} |",
         f"| early_exit_rate | - | {best['early_exit_rate']} |", "",
         "## Tail check (top-10 baseline winners)",
         f"- winners cut early: **{wr['winners_cut']}/10**, mean retention of their gross: "
         f"**{wr['mean_retention']*100:.0f}%** (baseline {wr['baseline_mean']*100:+.1f}% -> "
         f"give-back {wr['giveback_mean']*100:+.1f}%)", "",
         "## Placebo (random early exits, matched frequency)",
         f"- give-back best total {best['total_return']*100:+.1f}% vs placebo q90 "
         f"{p['placebo_random_exits']['q90']*100:+.1f}% (mean {p['placebo_random_exits']['mean']*100:+.1f}%)", "",
         "## Strict bar", ""]
    for k, v in p["strict_bar"].items():
        L.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    L += ["", f"## VERDICT: {p['verdict']}"]
    return "\n".join(L)


def _selftest() -> None:
    # synthetic trade: runs to +10% high on day1 then closes flat by day3 (classic give-back)
    tr = {"sym": "T", "date": None, "gross": 0.0, "hold": 3,
          "highs": [0.0, 0.10, 0.10, 0.10], "closes": [0.0, 0.08, 0.02, 0.0]}
    # tight stop (arm at 1% peak, exit on 30% give-back) must fire early and keep > final 0%
    r, k = giveback_exit(tr, 0.01, 0.30)
    assert k < 3 and r > 0.0, (r, k)                       # exited early, banked some gain
    # un-armable stop (need 50% peak) never triggers -> baseline
    r2, k2 = giveback_exit(tr, 0.50, 0.30)
    assert k2 == 3 and abs(r2 - 0.0) < 1e-9, (r2, k2)
    # a monotone winner (no give-back) is never cut -> tail preserved
    win = {"highs": [0, 0.05, 0.10, 0.15], "closes": [0, 0.05, 0.10, 0.15], "hold": 3}
    assert giveback_exit(win, 0.01, 0.30)[1] == 3, "monotone winner must not be truncated"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        out = run()
        print((OUT / "insider_giveback_backtest.md").read_text())
