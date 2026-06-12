#!/usr/bin/env python3
"""
backtest_candidates_bc.py — Candidates B & C of the independent-signal gauntlet (UPGRADED bar).

B = cross-sectional 5-day REVERSAL on liquid large-caps (long the worst-5d-return decile, 3d hold).
C = sector-rotation MOMENTUM on 7 sector ETFs (long top-2 by 3-mo return, ~21d hold, the diversifier).

UPGRADED GAUNTLET (per-year regime stability is the discriminating test that caught Candidate A):
  PASS only if: PF≥2.0 OOS(2024+) AND positive PF in EVERY regime-year (none losing/flat<1.2)
  AND positive MC median AND acceptable DD AND |corr vs insider|<0.3 AND realistic costs.
Shares/ETFs only. NO run.py wiring regardless of result.
"""
from __future__ import annotations

import glob
from datetime import date as _date
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

TRAIN_END = _date(2024, 1, 1)
SECTORS = ["XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI"]
SEED = 7


# ── shared gauntlet evaluator ───────────────────────────────────────────────────────────────
def _insider_daily():
    ins = vg._collect(hold=3, min_adv=1_000_000)
    if not ins:
        return {}
    vthr = np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80)
    daily: Dict = {}
    for r in ins:
        if r["rv20"] >= vthr:
            daily.setdefault(r["date"], []).append((1 + r["gross"]) ** (1 / 3) - 1)
    return {d: float(np.mean(v)) for d, v in daily.items()}


def _pf(v):
    w, l = v[v > 0], v[v < 0]
    return (w.sum() / abs(l.sum())) if len(l) else float("inf")


def gauntlet(trades: List[dict], daily: Dict, name: str, ins_daily: Dict) -> str:
    rng = np.random.RandomState(SEED)
    test = [t for t in trades if t["date"] >= TRAIN_END]
    tv = np.array([t["ret"] for t in test]) if test else np.array([])
    # OOS stats
    if len(tv):
        sd = tv.std()
        eq = peak = 1.0; mdd = 0.0
        for t in sorted(test, key=lambda x: x["date"]):
            eq *= (1 + 0.10 * t["ret"]); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        oos = {"n": len(tv), "mean": tv.mean(), "median": float(np.median(tv)), "win": (tv > 0).mean(),
               "sharpe": tv.mean() / sd if sd > 0 else 0, "pf": _pf(tv), "mdd": mdd}
    else:
        oos = None
    # per-year regime stability
    yr_rows, yr_fail = [], False
    for yr in range(2021, 2027):
        yv = np.array([t["ret"] for t in trades if t["date"].year == yr])
        if len(yv) < 10:
            continue
        pf = _pf(yv)
        yr_rows.append((yr, len(yv), yv.mean(), (yv > 0).mean(), pf))
        if pf < 1.2:
            yr_fail = True
    # MC bootstrap on OOS
    mc_p, mc_med = 0.0, 0.0
    if len(tv) >= 20:
        meds = []
        for _ in range(4000):
            s = rng.choice(tv, size=min(73, len(tv)), replace=True)
            tot = np.prod(1 + s) - 1; meds.append(tot); mc_p += int(tot > 0)
        mc_p /= 4000; mc_med = float(np.median(meds))
    # correlation vs insider
    common = sorted(set(daily) & set(ins_daily))
    corr = (float(np.corrcoef([daily[d] for d in common], [ins_daily[d] for d in common])[0, 1])
            if len(common) > 30 else float("nan"))
    # verdict
    ok_pf = bool(oos and oos["pf"] >= 2.0)
    ok_yr = (not yr_fail) and len(yr_rows) >= 4
    ok_mc = mc_med > 0 and mc_p >= 0.6
    ok_dd = bool(oos and oos["mdd"] > -0.25)
    ok_corr = (abs(corr) < 0.3) if corr == corr else True
    ok_n = bool(oos and oos["n"] >= 30)
    passed = ok_pf and ok_yr and ok_mc and ok_dd and ok_corr and ok_n

    L = [f"## {name}", ""]
    if oos:
        pf = "∞" if oos["pf"] == float("inf") else f"{oos['pf']:.2f}"
        L += [f"OOS 2024+: n={oos['n']:,} mean {oos['mean']*100:+.2f}% median {oos['median']*100:+.2f}% "
              f"win {oos['win']*100:.0f}% Sharpe {oos['sharpe']:+.3f} PF {pf} MaxDD {oos['mdd']*100:.0f}%"]
    L += ["", "Per-year regime stability (the decisive gate):",
          "| Year | n | Mean | Win% | PF |", "|---|---|---|---|---|"]
    for yr, n, m, w, pf in yr_rows:
        flag = " ⚠" if pf < 1.2 else ""
        L.append(f"| {yr}{flag} | {n:,} | {m*100:+.2f}% | {w*100:.0f}% | {pf:.2f} |")
    L += ["", f"MC (OOS bootstrap): P(profit)={mc_p*100:.0f}%, median={mc_med*100:+.2f}%",
          f"Correlation vs insider ({len(common)} days): {corr:+.2f}", "",
          f"**VERDICT: {'PASS' if passed else 'FAIL'}** — "
          f"PF≥2 OOS {'✓' if ok_pf else '✗'}, every-year-positive {'✓' if ok_yr else '✗'}, "
          f"MC {'✓' if ok_mc else '✗'}, DD {'✓' if ok_dd else '✗'}, low-corr {'✓' if ok_corr else '✗'}, "
          f"n≥30 {'✓' if ok_n else '✗'}", ""]
    return "\n".join(L)


# ── Candidate B: cross-sectional 5-day reversal on liquid large-caps ─────────────────────────
def candidate_B(max_names=250, min_adv=20e6, hold=3, cost=0.0015):
    rows, prices = [], {}
    for fp in glob.glob(str(base.BARS_DIR / "*_day.parquet")):
        sym = fp.rsplit("/", 1)[-1].rsplit("_day", 1)[0]
        if sym in SECTORS or sym in ("SPY", "QQQ", "IWM"):
            continue
        try:
            b = pd.read_parquet(fp)
        except Exception:
            continue
        if len(b) < 300:
            continue
        adv = float((b["close"] * b["volume"]).mean())
        if adv >= min_adv:
            prices[sym] = b
    # top liquid names
    liq = sorted(prices, key=lambda s: float((prices[s]["close"] * prices[s]["volume"]).mean()), reverse=True)[:max_names]
    op = pd.DataFrame({s: prices[s]["open"] for s in liq})
    cl = pd.DataFrame({s: prices[s]["close"] for s in liq})
    ret5 = cl.pct_change(5)
    dates = list(cl.index)
    daily: Dict = {}
    for i in range(60, len(dates) - hold - 1):
        row = ret5.iloc[i].dropna()
        if len(row) < 30:
            continue
        losers = row.nsmallest(max(1, len(row) // 10)).index    # worst-5d decile
        for s in losers:
            e = op[s].iloc[i + 1]; x = cl[s].iloc[i + 1 + hold]
            if e > 0 and x > 0:
                ret = x / e - 1 - cost
                d = dates[i + 1].date() if hasattr(dates[i + 1], "date") else dates[i + 1]
                rows.append({"sym": s, "date": d, "ret": ret})
                pd_ret = (1 + ret) ** (1 / hold) - 1
                for k in range(i + 1, i + 1 + hold):
                    dd = dates[k].date() if hasattr(dates[k], "date") else dates[k]
                    daily.setdefault(dd, []).append(pd_ret)
    return rows, {d: float(np.mean(v)) for d, v in daily.items()}


# ── Candidate C: sector-rotation momentum (top-2 of 7, 3-mo lookback, 21d hold) ──────────────
def candidate_C(lookback=63, hold=21, top=2, cost=0.0005):
    cl = {}
    for s in SECTORS:
        try:
            cl[s] = pd.read_parquet(base.BARS_DIR / f"{s}_day.parquet")["close"].astype(float)
        except Exception:
            pass
    panel = pd.DataFrame(cl).dropna()
    mom = panel.pct_change(lookback)
    dates = list(panel.index)
    rows, daily = [], {}
    i = lookback + 1
    while i < len(dates) - hold - 1:
        r = mom.iloc[i].dropna()
        if len(r) >= top:
            picks = r.nlargest(top).index
            for s in picks:
                e = panel[s].iloc[i + 1]; x = panel[s].iloc[i + 1 + hold]
                if e > 0 and x > 0:
                    ret = x / e - 1 - cost
                    d = dates[i + 1].date() if hasattr(dates[i + 1], "date") else dates[i + 1]
                    rows.append({"sym": s, "date": d, "ret": ret})
                    pd_ret = (1 + ret) ** (1 / hold) - 1
                    for k in range(i + 1, i + 1 + hold):
                        dd = dates[k].date() if hasattr(dates[k], "date") else dates[k]
                        daily.setdefault(dd, []).append(pd_ret)
        i += hold      # monthly rebalance (non-overlapping)
    return rows, {d: float(np.mean(v)) for d, v in daily.items()}


def main():
    ins = _insider_daily()
    L = ["# Candidates B & C — Independent-Signal Gauntlet (upgraded bar)", "",
         "Shares/ETFs only. PASS requires PF≥2 OOS + positive PF EVERY regime-year + MC median>0 + "
         "DD>-25% + |corr-insider|<0.3 + n≥30. No run.py wiring.", ""]
    print("Candidate B: cross-sectional 5d reversal (liquid large-caps)...")
    bt, bd = candidate_B()
    L.append(gauntlet(bt, bd, "Candidate B — Cross-Sectional 5-Day Reversal (liquid large-caps, 15bps)", ins))
    print("Candidate C: sector-rotation momentum...")
    ct, cd = candidate_C()
    L.append(gauntlet(ct, cd, "Candidate C — Sector-Rotation Momentum (7 sector ETFs, 5bps)", ins))
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "candidates_bc.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
