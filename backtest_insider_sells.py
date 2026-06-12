#!/usr/bin/env python3
"""
backtest_insider_sells.py — Form-4 SELLS as an information signal (upgraded gauntlet).

PIT-clean sells (data/cache/pit_insider_sells.json, extracted from the same zips as buys). Tests
whether large insider SELL clusters (>$1M, 2-day window, 2-day filing lag) predict UNDERPERFORMANCE.
Two framings:
  - SHORT P&L: short next open, cover +3d close, 1% round-trip (small-cap). Profits if stock falls.
  - AVOID/predictive: post-sell forward return vs the unconditional all-days baseline.
Also tests the high-vol subset (vol amplified the BUY edge). Gauntlet: per-year regime stability,
OOS, MC, |corr vs vol-gated insider-BUY|<0.3, realistic costs.
"""
from __future__ import annotations

import json
from datetime import date as _date, datetime
from typing import Dict, List

import numpy as np

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import backtest_vol_gated_insider as vg

HOLD = 3
COST = 0.010
TRAIN_END = _date(2024, 1, 1)
VOL_MIN = 0.0563
SELLS_PATH = base.PROJECT_ROOT / "data" / "cache" / "pit_insider_sells.json"


def _load_sells() -> Dict[str, list]:
    raw = json.loads(SELLS_PATH.read_text())
    out: Dict[str, list] = {}
    for sym, evs in raw.items():
        cleaned = []
        for d, v in evs:
            try:
                dt = datetime.fromisoformat(str(d)[:10]).date()
                if 2018 <= dt.year <= 2026:
                    cleaned.append((dt, float(v)))
            except Exception:
                continue
        if cleaned:
            out[sym] = cleaned
    return out


def collect():
    sells = _load_sells()
    rows = []
    for sym in sorted(sells.keys()):
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        op = [float(bars.loc[d, "open"]) for d in td]
        cl = [float(bars.loc[d, "close"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        r1 = np.diff(np.log(np.maximum(cl, 1e-9)))
        td_dates = [d.date() if hasattr(d, "date") else d for d in td]
        for i in ab._insider_entry_indices(td_dates, sym, sells):   # >$1M cluster, 2d window, 2d lag
            if i < 25 or i + 1 + HOLD >= len(td):
                continue
            rv20 = float(np.std(r1[i - 20:i])) if i >= 21 else np.nan
            adv = float(np.mean([cl[k] * vol[k] for k in range(i - 20, i)]))
            entry = op[i + 1]; exit_px = cl[i + 1 + HOLD]
            if entry <= 0 or exit_px <= 0 or not np.isfinite(rv20) or adv < 1e6:
                continue
            fwd = exit_px / entry - 1.0                 # the stock's forward return
            rows.append({"sym": sym, "date": td_dates[i + 1], "fwd": fwd,
                         "short": -fwd - COST, "rv20": rv20})
    return rows


def _pf(v):
    w, l = v[v > 0], v[v < 0]
    return (w.sum() / abs(l.sum())) if len(l) else float("inf")


def _gaunt(rows, key, name, ins_daily):
    rng = np.random.RandomState(7)
    test = [r for r in rows if r["date"] >= TRAIN_END]
    tv = np.array([r[key] for r in test])
    L = [f"### {name}", ""]
    if len(tv) < 20:
        return "\n".join(L + [f"n={len(tv)} too small → REJECT", ""])
    sd = tv.std(); eq = peak = 1.0; mdd = 0.0
    for r in sorted(test, key=lambda x: x["date"]):
        eq *= (1 + 0.10 * r[key]); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    pf = _pf(tv)
    L.append(f"OOS 2024+: n={len(tv):,} mean {tv.mean()*100:+.2f}% median {np.median(tv)*100:+.2f}% "
             f"win {(tv>0).mean()*100:.0f}% PF {pf:.2f} MaxDD {mdd*100:.0f}%")
    L += ["", "| Year | n | Mean | Win% | PF |", "|---|---|---|---|---|"]
    yr_fail = False; n_yr = 0
    for yr in range(2021, 2027):
        yv = np.array([r[key] for r in rows if r["date"].year == yr])
        if len(yv) < 10:
            continue
        n_yr += 1; ypf = _pf(yv)
        if ypf < 1.2:
            yr_fail = True
        L.append(f"| {yr}{' ⚠' if ypf<1.2 else ''} | {len(yv):,} | {yv.mean()*100:+.2f}% | {(yv>0).mean()*100:.0f}% | {ypf:.2f} |")
    meds = [np.prod(1 + rng.choice(tv, size=min(73, len(tv)), replace=True)) - 1 for _ in range(4000)]
    mc_p = np.mean([m > 0 for m in meds]); mc_med = float(np.median(meds))
    # corr vs insider buy daily (only meaningful for the short P&L series)
    daily = {}
    for r in rows:
        daily.setdefault(r["date"], []).append(r[key])
    dser = {d: float(np.mean(v)) for d, v in daily.items()}
    common = sorted(set(dser) & set(ins_daily))
    corr = (float(np.corrcoef([dser[d] for d in common], [ins_daily[d] for d in common])[0, 1])
            if len(common) > 30 else float("nan"))
    ok = (pf >= 2.0 and not yr_fail and n_yr >= 4 and mc_med > 0 and mc_p >= 0.6 and mdd > -0.25
          and (abs(corr) < 0.3 if corr == corr else True) and len(tv) >= 30)
    L += ["", f"MC: P(profit)={mc_p*100:.0f}%, median={mc_med*100:+.2f}% | corr vs insider-buy: {corr:+.2f}",
          f"**VERDICT: {'PASS' if ok else 'FAIL'}** "
          f"(PF≥2 {'✓' if pf>=2 else '✗'}, every-year {'✓' if not yr_fail else '✗'}, "
          f"MC {'✓' if mc_med>0 and mc_p>=0.6 else '✗'}, DD {'✓' if mdd>-0.25 else '✗'})", ""]
    return "\n".join(L)


def main():
    print("Collecting Form-4 sell-cluster events...")
    rows = collect()
    print(f"  {len(rows):,} sell-cluster trades")
    # insider-buy daily for correlation
    ins = vg._collect(hold=3, min_adv=1_000_000)
    vthr = np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80)
    ib = {}
    for r in ins:
        if r["rv20"] >= vthr:
            ib.setdefault(r["date"], []).append((1 + r["gross"]) ** (1 / 3) - 1)
    ins_daily = {d: float(np.mean(v)) for d, v in ib.items()}
    # unconditional baseline forward return (does post-sell underperform?)
    fwd_all = np.array([r["fwd"] for r in rows])
    hi = [r for r in rows if r["rv20"] >= VOL_MIN]
    L = ["# Form-4 SELLS — Information-Signal Gauntlet", "",
         f"{len(rows):,} >$1M sell-cluster trades (PIT-clean), 3d hold, 1% cost. Prior: insider sells "
         "are noisy (diversification/tax/10b5-1). Predictive check: post-sell forward return.", "",
         f"Post-sell forward 3d return: mean {fwd_all.mean()*100:+.2f}%, median {np.median(fwd_all)*100:+.2f}%, "
         f"win {(fwd_all>0).mean()*100:.0f}% (if << +ve baseline → sells predict underperformance)", "",
         _gaunt(rows, "short", "Form-4 SELL → SHORT (all)", ins_daily),
         _gaunt(hi, "short", "Form-4 SELL → SHORT (high-vol subset)", ins_daily)]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "insider_sells.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
