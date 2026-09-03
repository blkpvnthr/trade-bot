#!/usr/bin/env python3
"""
study_insider_amplifiers.py — do RICH Form-4 features AMPLIFY the validated vol-gated insider edge?

For each vol-gated insider entry, compute cluster features from pit_insider_rich.json (owner
identity + role) in the pre-entry window: breadth (# distinct insiders), C-suite, officer, repeat
buyer. Test whether any cohort beats the base edge — same bar: per-year stable (no losing/flat year),
beats base on mean+PF, n≥30. (Refinement/amplifier of the insider edge, not an independent signal.)
"""
from __future__ import annotations

import json
import re
from datetime import date as _date, datetime, timedelta
from typing import Dict, List

import numpy as np

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

RICH = base.PROJECT_ROOT / "data" / "cache" / "pit_insider_rich.json"
CSUITE = re.compile(r"\b(ceo|chief executive|cfo|chief financial|president|chairman|chief)\b", re.I)


def _load_rich():
    raw = json.loads(RICH.read_text())
    out: Dict[str, list] = {}
    for sym, evs in raw.items():
        rows = []
        for e in evs:
            try:
                d = datetime.fromisoformat(e[0][:10]).date()
                rows.append((d, float(e[1]), str(e[2]), int(e[3]), int(e[4]), int(e[5]), str(e[6])))
            except Exception:
                continue
        if rows:
            out[sym] = rows
    return out


def _features(rich, sym, entry_date):
    """Cluster features in [entry-7d, entry-1d] (the buys that triggered the signal)."""
    evs = rich.get(sym, [])
    lo, hi = entry_date - timedelta(days=10), entry_date
    win = [e for e in evs if lo <= e[0] < hi]
    if not win:
        return None
    ciks = {e[2] for e in win if e[2]}
    csuite = any(CSUITE.search(e[6] or "") for e in win)
    officer = any(e[3] for e in win)
    # repeat buyer: any owner who also bought this ticker in the prior 90d (before the window)
    prior = [e for e in evs if entry_date - timedelta(days=100) <= e[0] < lo]
    repeat = bool({e[2] for e in win if e[2]} & {e[2] for e in prior if e[2]})
    return {"breadth": len(ciks), "csuite": csuite, "officer": officer, "repeat": repeat}


def _pf(v):
    w, l = v[v > 0], v[v < 0]
    return (w.sum() / abs(l.sum())) if len(l) else float("inf")


def _stat(rows):
    if len(rows) < 5:
        return None
    v = np.array([r["gross"] for r in rows])
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)), "win": (v > 0).mean(), "pf": _pf(v)}


def _peryear_ok(rows):
    nyr = 0; fail = False
    out = []
    for yr in (2022, 2023, 2024):
        yv = np.array([r["gross"] for r in rows if r["date"].year == yr])
        if len(yv) >= 10:
            nyr += 1; p = _pf(yv); out.append((yr, len(yv), yv.mean(), p))
            if p < 1.2:
                fail = True
    return (not fail and nyr >= 2), out


def _row(name, rows, base_stat):
    s = _stat(rows)
    if not s:
        return f"| {name} | {len(rows)} |—|—|—|— | — |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    lift = (s["mean"] - base_stat["mean"]) * 100 if base_stat else 0
    return (f"| {name} | {s['n']} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | "
            f"{s['win']*100:.0f}% | {pf} | {lift:+.2f}% |")


def main():
    rich = _load_rich()
    ins = vg._collect(hold=3, min_adv=1_000_000)
    vthr = np.percentile([r["rv20"] for r in ins if r["date"] < _date(2024, 1, 1)], 80)
    gated = [r for r in ins if r["rv20"] >= vthr]
    print(f"{len(gated)} vol-gated insider entries; computing rich cluster features...")
    rows = []
    for r in gated:
        f = _features(rich, r["sym"], r["date"])
        if f:
            rows.append({**r, **f})
    cov = len(rows) / max(len(gated), 1)
    base_s = _stat(rows)
    cohorts = {
        "ALL gated (base)": rows,
        "breadth >=2 insiders": [r for r in rows if r["breadth"] >= 2],
        "breadth ==1 (single)": [r for r in rows if r["breadth"] == 1],
        "C-suite buyer (CEO/CFO/Chair)": [r for r in rows if r["csuite"]],
        "officer buyer": [r for r in rows if r["officer"]],
        "repeat buyer (prior 90d)": [r for r in rows if r["repeat"]],
    }
    L = ["# Insider-Edge Amplifiers — rich Form-4 (breadth / role / repeat)", "",
         f"Vol-gated insider entries, 3d hold (gross). Rich-feature coverage {cov*100:.0f}% "
         f"({len(rows)}/{len(gated)}). Bar: beat base on mean+PF AND per-year stable (no year PF<1.2), n≥30.", "",
         "| Cohort | n | Mean | Median | Win% | PF | lift vs base |", "|---|---|---|---|---|---|---|"]
    for name, sub in cohorts.items():
        L.append(_row(name, sub, base_s))
    L += [""]
    # verdict for each amplifier
    L += ["## Verdict per amplifier (must beat base mean+PF AND be per-year stable, n≥30)", ""]
    for name in ["breadth >=2 insiders", "C-suite buyer (CEO/CFO/Chair)", "officer buyer", "repeat buyer (prior 90d)"]:
        sub = cohorts[name]; s = _stat(sub)
        if not s or s["n"] < 30:
            L.append(f"- **{name}**: n={len(sub)} — too small / no edge"); continue
        beats = s["mean"] > base_s["mean"] and s["pf"] >= base_s["pf"]
        stable, yrs = _peryear_ok(sub)
        yrstr = ", ".join(f"{y}:PF{p:.2f}" for y, n, m, p in yrs)
        L.append(f"- **{name}** (n={s['n']}, PF {s['pf']:.2f}, +{(s['mean']-base_s['mean'])*100:+.2f}%): "
                 f"{'**AMPLIFIES**' if beats and stable else 'no'} | per-year [{yrstr}]")
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "insider_amplifiers.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
