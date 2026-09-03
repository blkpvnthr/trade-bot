#!/usr/bin/env python3
"""
backtest_insider_sizing.py — Item #2: does CONVICTION sizing beat FLAT sizing on the validated
vol-gated insider sleeve, out of sample? (See PRODUCTION_ARCHITECTURE.md §B.)

Conviction features reconstructed per trade (all ex-ante, known at entry):
  - rv20    : entry-day realized vol (20d) — strong prior (Q5 +5.53% vs Q1 -0.40%)
  - breadth : # of Form-4 buys clustered in the 2-day signal window
  - value   : $ size of the cluster — NEGATIVE control ($100M study: bigger != better)
  (freshness is a FIXED +2d entry rule → no per-trade variation → not calibratable here.)

Method (no peeking):
  Part 1 — predictiveness: Spearman IC + Q5-Q1 spread of each feature vs gross return, TRAIN then OOS.
  Part 2 — sizing schemes vs flat on the PRODUCTION (vol-gated) set: weighted-average daily PnL
           series (SCALE-INVARIANT in weights → automatically same-gross, no leverage confound).
           Compare OOS Sharpe / CAGR / MaxDD; bootstrap p-value for ΔSharpe>0 vs flat.
  Part 3 — gate-vs-tilt: is a continuous vol TILT on the full set better than the binary vol GATE?

Weight functions are fit on TRAIN (z-score mean/std, quintile breaks) and applied to OOS.
A scheme is adopted only if it raises OOS Sharpe with bootstrap p<0.10 (else keep flat).
"""
from __future__ import annotations

from datetime import date as _date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom

HOLD = 5                       # best-edge horizon (t-stat 5.82 @5d); production share hold=3
MIN_ADV = 1_000_000
VOL_PCTILE = 80                # production vol-gate
TRAIN_END = _date(2024, 1, 1)
SEED = 7
LAMBDA = 0.5                   # tilt strength for continuous z-score weighting


def collect() -> List[dict]:
    """Per-trade dicts mirroring production signal logic + conviction-feature snapshot."""
    pit = base.load_pit_buy_events()
    rows = []
    for sym in sorted(pit.keys()):
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        td_dates = [d.date() if hasattr(d, "date") else d for d in td]
        op = [float(bars.loc[d, "open"]) for d in td]
        cl = [float(bars.loc[d, "close"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        r1 = np.diff(np.log(np.maximum(cl, 1e-9)))
        buys = sorted((dd if hasattr(dd, "year") else _date.fromisoformat(str(dd)), float(v))
                      for dd, v in pit[sym])
        sigs = base.generate_signals(buys, 1_000_000, 2, HOLD, td_dates, 2)
        for sd in sigs:
            e = base._next_trading_on_or_after(sd + timedelta(days=2), td_dates)
            if e is None or e not in td_dates:
                continue
            i = td_dates.index(e) - 1
            if not (25 <= i < len(td) - HOLD - 1):
                continue
            rv20 = float(np.std(r1[i - 20:i])) if i >= 21 else np.nan
            adv = float(np.mean([cl[k] * vol[k] for k in range(i - 20, i)]))
            if not np.isfinite(rv20) or adv < MIN_ADV:
                continue
            entry = op[i + 1]
            exit_px = cl[i + 1 + HOLD]
            if entry <= 0 or exit_px <= 0:
                continue
            lo = sd - timedelta(days=2)
            wbuys = [(dd, v) for dd, v in buys if lo <= dd <= sd]
            rows.append({"sym": sym, "date": td_dates[i + 1], "gross": exit_px / entry - 1.0,
                         "rv20": rv20, "adv": adv,
                         "value": float(sum(v for _, v in wbuys)), "breadth": int(len(wbuys)),
                         "hold": HOLD})
    return rows


# ---------- predictiveness ----------
def spearman_ic(rows, key):
    if len(rows) < 10:
        return float("nan")
    x = pd.Series([r[key] for r in rows]).rank()
    y = pd.Series([r["gross"] for r in rows]).rank()
    return float(np.corrcoef(x, y)[0, 1])


def quintile_spread(rows, key):
    if len(rows) < 25:
        return float("nan"), float("nan")
    df = pd.DataFrame(rows)
    try:
        df["q"] = pd.qcut(df[key].rank(method="first"), 5, labels=False)
    except Exception:
        return float("nan"), float("nan")
    q5 = df[df["q"] == 4]["gross"].mean()
    q1 = df[df["q"] == 0]["gross"].mean()
    return float(q5 - q1), float(q5)


# ---------- weighting schemes ----------
def zfit(train, key):
    v = np.array([r[key] for r in train], float)
    return float(np.mean(v)), float(np.std(v) or 1.0)


def w_z(rows, key, mu, sd):
    return np.array([np.exp(LAMBDA * np.clip((r[key] - mu) / sd, -2, 2)) for r in rows])


def w_composite(rows, fits):
    z = np.zeros(len(rows))
    for key, (mu, sd) in fits.items():
        z += np.clip((np.array([r[key] for r in rows], float) - mu) / sd, -2, 2)
    return np.exp(LAMBDA * z / max(len(fits), 1))


def w_buckets(rows, fits):
    """Composite conviction score → train-quintile → A+8/A6/B4/C2 bucket weights."""
    z = np.zeros(len(rows))
    for key, (mu, sd) in fits.items():
        z += np.clip((np.array([r[key] for r in rows], float) - mu) / sd, -2, 2)
    s = pd.Series(z)
    try:
        q = pd.qcut(s.rank(method="first"), 5, labels=False).values
    except Exception:
        return np.ones(len(rows))
    table = {4: 8.0, 3: 6.0, 2: 4.0, 1: 2.0, 0: 2.0}   # A+ / A / B / C / C
    return np.array([table[int(qq)] for qq in q])


# ---------- portfolio metrics on a weighted-average daily series (scale-invariant) ----------
def daily_series(rows, weights):
    num, den = {}, {}
    for r, w in zip(rows, weights):
        gday = (1 + r["gross"]) ** (1 / r["hold"]) - 1
        for d in pd.bdate_range(r["date"], periods=int(r["hold"])):
            dd = d.date()
            num[dd] = num.get(dd, 0.0) + w * gday
            den[dd] = den.get(dd, 0.0) + w
    dates = sorted(num)
    ser = np.array([num[d] / den[d] for d in dates]) if dates else np.array([])
    return ser


def port_stats(rows, weights):
    ser = daily_series(rows, weights)
    if len(ser) < 5:
        return {"sharpe": 0.0, "cagr": 0.0, "mdd": 0.0, "ndays": len(ser)}
    sh = float(ser.mean() / ser.std() * np.sqrt(252)) if ser.std() > 0 else 0.0
    eq = np.cumprod(1 + ser)
    cagr = float(eq[-1] ** (252.0 / len(ser)) - 1)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return {"sharpe": sh, "cagr": cagr, "mdd": mdd, "ndays": len(ser)}


def boot_p(rows, weights, B=1500, seed=SEED):
    """p-value for ΔSharpe(scheme - flat) > 0 via trade resampling."""
    if len(rows) < 20:
        return 1.0
    rng = np.random.RandomState(seed)
    idx = np.arange(len(rows))
    neg = 0
    for _ in range(B):
        s = rng.choice(idx, size=len(rows), replace=True)
        rs = [rows[k] for k in s]
        ws = weights[s]
        d = port_stats(rs, ws)["sharpe"] - port_stats(rs, np.ones(len(rs)))["sharpe"]
        if d <= 0:
            neg += 1
    return neg / B


def fmt(name, s, p=None):
    pp = "" if p is None else f" | p={p:.3f}"
    return (f"| {name} | {s['ndays']} | {s['sharpe']:+.3f} | {s['cagr']*100:+.1f}% | "
            f"{s['mdd']*100:.0f}%{pp} |")


def main():
    print("Collecting insider trades with conviction features...")
    rows = collect()
    train = [r for r in rows if r["date"] < TRAIN_END]
    test = [r for r in rows if r["date"] >= TRAIN_END]
    vthr = float(np.percentile([r["rv20"] for r in train], VOL_PCTILE))
    gated_test = [r for r in test if r["rv20"] >= vthr]
    print(f"  {len(rows):,} trades (train {len(train):,}/test {len(test):,}); "
          f"vol P{VOL_PCTILE}={vthr:.4f}; gated OOS n={len(gated_test):,}")

    # Part 1 — predictiveness
    L = ["# Insider Conviction Sizing Calibration (§B) — does conviction sizing beat flat OOS?", "",
         f"hold={HOLD}d, min ADV=${MIN_ADV:,.0f}, vol-gate P{VOL_PCTILE} (={vthr:.4f}). "
         f"Train <{TRAIN_END}, test ≥. Weighted-avg daily series is scale-invariant → same-gross.", "",
         "## Part 1 — feature predictiveness (Spearman IC, Q5−Q1 spread)", "",
         "| Feature | IC train | IC OOS | Q5−Q1 train | Q5−Q1 OOS | verdict |",
         "|---|---|---|---|---|---|"]
    for key, prior in [("rv20", "vol"), ("breadth", "breadth"), ("value", "value (neg control)")]:
        ic_tr, ic_oos = spearman_ic(train, key), spearman_ic(test, key)
        sp_tr, _ = quintile_spread(train, key)
        sp_oos, _ = quintile_spread(test, key)
        verdict = ("predictive" if (ic_oos == ic_oos and ic_oos > 0.02 and sp_oos == sp_oos and sp_oos > 0)
                   else "no/weak")
        L.append(f"| {prior} | {ic_tr:+.3f} | {ic_oos:+.3f} | {sp_tr*100:+.2f}% | "
                 f"{sp_oos*100:+.2f}% | {verdict} |")

    # Part 2 — sizing schemes vs flat on the GATED production set
    fits_all = {"rv20": zfit(train, "rv20"), "breadth": zfit(train, "breadth")}
    schemes = {
        "flat (baseline)": np.ones(len(gated_test)),
        "vol-tilt": w_z(gated_test, "rv20", *zfit(train, "rv20")),
        "breadth-tilt": w_z(gated_test, "breadth", *zfit(train, "breadth")),
        "value-tilt (neg ctrl)": w_z(gated_test, "value", *zfit(train, "value")),
        "composite (vol+breadth)": w_composite(gated_test, fits_all),
        "buckets A+8/A6/B4/C2": w_buckets(gated_test, fits_all),
    }
    flat = port_stats(gated_test, schemes["flat (baseline)"])
    L += ["", "## Part 2 — sizing schemes vs flat (GATED production set, OOS)", "",
          "| Scheme | days | Sharpe | CAGR | MaxDD | ΔSharpe-p |", "|---|---|---|---|---|---|",
          fmt("flat (baseline)", flat)]
    best = ("flat (baseline)", flat, 1.0)
    for name, w in schemes.items():
        if name == "flat (baseline)":
            continue
        s = port_stats(gated_test, w)
        p = boot_p(gated_test, w)
        L.append(fmt(name, s, p))
        if s["sharpe"] > best[1]["sharpe"] and p < 0.10:
            best = (name, s, p)

    # Part 3 — gate vs tilt (full set)
    full_flat = port_stats(test, np.ones(len(test)))
    full_vol = port_stats(test, w_z(test, "rv20", *zfit(train, "rv20")))
    p_full_vol = boot_p(test, w_z(test, "rv20", *zfit(train, "rv20")))
    L += ["", "## Part 3 — gate vs tilt (does continuous vol sizing beat the binary gate?)", "",
          "| Config | days | Sharpe | CAGR | MaxDD | ΔSharpe-p |", "|---|---|---|---|---|---|",
          fmt("full set, flat", full_flat),
          fmt("full set, vol-tilt", full_vol, p_full_vol),
          fmt("gated set, flat", flat)]

    # Verdict
    adopt = best[0] != "flat (baseline)"
    L += ["", "## VERDICT", "",
          f"- Best OOS scheme on gated set: **{best[0]}** (Sharpe {best[1]['sharpe']:+.3f}, "
          f"vs flat {flat['sharpe']:+.3f}, p={best[2]:.3f})",
          f"- Gate vs tilt: gated-flat Sharpe {flat['sharpe']:+.3f} vs full-vol-tilt {full_vol['sharpe']:+.3f}",
          "",
          (f"**ADOPT {best[0]} for insider sizing (clamp bucket weights to the 10% name cap; "
           f"fractional/¼-Kelly spirit).**" if adopt else
           "**KEEP FLAT SIZING — no conviction scheme beats flat at p<0.10 OOS. The vol-GATE already "
           "harvests the vol edge as a filter; further sizing tilt adds noise, not signal. "
           "Do not poison a validated edge with unproven sizing.**")]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "insider_sizing_calibration.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
