#!/usr/bin/env python3
"""
backtest_options_positioning.py — does BEARISH options crowding predict stock mean-reversion UP,
and does measuring it vs each ticker's OWN baseline beat a cross-sectional read?

Signal inputs (PIT, real historical AV): HISTORICAL_PUT_CALL_RATIO (full chain) and
HISTORICAL_VOLUME_OPEN_INTEREST_RATIO aggregated to a call-share. NO sentiment/surprise/news/
fundamentals — options positioning ONLY.

Two modes:
  --mode earnings  : cross-sectional. One event per earnings report (T+1 after reportedDate).
                     Features = raw pcr, call-share.
  --mode baseline  : name-relative. Weekly grid per ticker. Features = pcr & call-share expressed
                     as z-score (or percentile) vs the ticker's OWN trailing rolling baseline.

Discipline:
  - WALK-FORWARD (expanding window): for each test YEAR, orientation signs + z-stats are fit on
    PRIOR years only, then applied to that year. Pooled OOS = all walk-forward test years.
  - Sign orientation (pcr +/-, voi +/-) LEARNED on train each window → reports sign STABILITY.
  - Horizons 5/10/21d (close→+H close), honest round-trip slippage.
  - Year-by-year table + sign-stability table. Saves markdown + per-event CSV.
  - Does NOT touch the live bot. A result is only credible if signs are stable across regimes.

Usage:
  python backtest_options_positioning.py --mode earnings  --start 2019-01-01 --max-fetch 4000
  python backtest_options_positioning.py --mode baseline  --cadence 7 --baseline-window 26 \
        --baseline-method z --start 2021-01-01 --symbols AAPL,MSFT,... --max-fetch 6000
"""
from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path
from datetime import date as _date, datetime, timedelta
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np

import backtest_alpha_momentum_options as mom
import av_catalyst as cat

ROOT = Path(__file__).resolve().parent
HORIZONS = [5, 10, 21]
UNIVERSE = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AMD AVGO CRM ORCL ADBE NFLX INTC CSCO QCOM TXN "
            "IBM NOW UBER PYPL SHOP COIN PLTR SNOW DDOG NET CRWD ABNB MU BA CAT GE JPM BAC WMT TGT "
            "DIS NKE PFE MRNA F GM DAL AAL CCL RIVN LCID SOFI HOOD DKNG ROKU SQ ZM ETSY CVNA UPST "
            "AFRM PINS SNAP TWLO").split()


def _bars(sym):
    b = mom._bars(sym)
    if b is None or b.empty or len(b) < 80:
        return None
    return b


def _idx_on_or_after(bdates, d):
    return next((i for i, x in enumerate(bdates) if x >= d), None)


def _fwd_returns(cl, idx, slip):
    """close[idx] -> close[idx+H] minus slip, for each horizon. None if insufficient bars."""
    out = {}
    base = float(cl[idx])
    if base <= 0:
        return None
    for h in HORIZONS:
        if idx + h >= len(cl):
            out[h] = None
        else:
            px = float(cl[idx + h])
            out[h] = (px / base - 1.0 - slip) if px > 0 else None
    return out


def build_earnings_events(universe, start, slip, max_fetch):
    spy = _bars("SPY")
    fetches = [0]
    events = []
    for sym in universe:
        b = _bars(sym)
        if b is None:
            continue
        bdates = [d.date() if hasattr(d, "date") else d for d in b.index]
        cl = b["close"].tolist()
        for e in cat.earnings(sym):
            rd = e.get("reportedDate")
            if not rd:
                continue
            try:
                rep = datetime.strptime(rd[:10], "%Y-%m-%d").date()
            except Exception:
                continue
            if rep < start:
                continue
            entry_idx = next((i for i, d in enumerate(bdates) if d > rep), None)
            if entry_idx is None:
                continue
            fwd = _fwd_returns(cl, entry_idx, slip)
            if not fwd or fwd[max(HORIZONS)] is None:
                continue
            pcr = cat.put_call_ratio(sym, rep.isoformat()); fetches[0] += 1
            cs = cat.volume_oi_call_share(sym, rep.isoformat()); fetches[0] += 1
            if pcr is None and cs is None:
                continue
            events.append({"sym": sym, "date": bdates[entry_idx], "pcr": pcr, "cshare": cs,
                           **{f"fwd{h}": fwd[h] for h in HORIZONS}})
            if cat.quota_hit() or fetches[0] >= max_fetch:
                print(f"  stop (quota={cat.quota_hit()}, fetches={fetches[0]}) — {len(events)} events")
                return events
    return events


def build_baseline_events(universe, start, end, cadence, window, method, slip, max_fetch):
    """Weekly grid per ticker; features = pcr & call-share vs the ticker's OWN trailing baseline."""
    fetches = [0]
    events = []
    for sym in universe:
        b = _bars(sym)
        if b is None:
            continue
        bdates = [d.date() if hasattr(d, "date") else d for d in b.index]
        cl = b["close"].tolist()
        # weekly grid of trading days within [start, end]
        grid, d = [], start
        while d <= end:
            gi = _idx_on_or_after(bdates, d)
            if gi is not None and gi < len(bdates) and (not grid or grid[-1][0] != gi):
                grid.append((gi, bdates[gi]))
            d += timedelta(days=cadence)
        series = []   # (idx, date, pcr, cshare)
        for gi, gd in grid:
            pcr = cat.put_call_ratio(sym, gd.isoformat()); fetches[0] += 1
            cs = cat.volume_oi_call_share(sym, gd.isoformat()); fetches[0] += 1
            if pcr is not None or cs is not None:
                series.append((gi, gd, pcr, cs))
            if cat.quota_hit() or fetches[0] >= max_fetch:
                break
        # baseline-relative features (trailing window, no lookahead) + forward returns
        for j, (gi, gd, pcr, cs) in enumerate(series):
            if j < window:
                continue
            hist_pcr = [s[2] for s in series[j - window:j] if s[2] is not None]
            hist_cs = [s[3] for s in series[j - window:j] if s[3] is not None]
            if len(hist_pcr) < max(8, window // 2) or len(hist_cs) < max(8, window // 2):
                continue
            bz_pcr = _rel(pcr, hist_pcr, method)
            bz_cs = _rel(cs, hist_cs, method)
            if bz_pcr is None and bz_cs is None:
                continue
            fwd = _fwd_returns(cl, gi, slip)
            if not fwd or fwd[max(HORIZONS)] is None:
                continue
            events.append({"sym": sym, "date": gd, "pcr": bz_pcr, "cshare": bz_cs,
                           **{f"fwd{h}": fwd[h] for h in HORIZONS}})
        if cat.quota_hit() or fetches[0] >= max_fetch:
            print(f"  stop (quota={cat.quota_hit()}, fetches={fetches[0]}) — {len(events)} events so far")
            break
    return events


def _rel(x, hist, method):
    if x is None or len(hist) < 3:
        return None
    if method == "pct":
        return sum(1 for h in hist if h <= x) / len(hist)        # percentile rank in [0,1]
    mu, sd = mean(hist), (pstdev(hist) or 1.0)
    return (x - mu) / sd                                          # z-score vs own baseline


def _stat(arr):
    if not arr:
        return None
    v = np.array(arr, float)
    losses = v[v < 0]
    pf = (v[v > 0].sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    return {"n": len(v), "mean": float(v.mean()), "win": float((v > 0).mean()),
            "sharpe": float(v.mean() / sd) if sd > 0 else 0.0, "pf": pf}


def _orient_and_score(train, test, feats, h):
    """Learn z-stats + sign orientation on train (by corr with fwd{h}), score test in place. Returns signs."""
    yk = f"fwd{h}"
    fwd_tr = np.array([e[yk] for e in train], float)
    signs, stats = {}, {}
    for f in feats:
        tv = [e[f] for e in train if e.get(f) is not None]
        mu = mean(tv) if tv else 0.0
        sd = (pstdev(tv) if len(tv) > 1 else 1.0) or 1.0
        stats[f] = (mu, sd)
        fv = np.array([(e[f] if e.get(f) is not None else mu) for e in train], float)
        signs[f] = (1.0 if np.corrcoef(fv, fwd_tr)[0, 1] >= 0 else -1.0) if fv.std() > 0 else 0.0
    for e in test:
        e["score"] = sum(signs[f] * (((e[f] if e.get(f) is not None else stats[f][0]) - stats[f][0]) / stats[f][1])
                         for f in feats)
    return signs


def walk_forward(events, feats, h, min_train=60):
    """Expanding-window walk-forward by YEAR. Returns (pooled_scored_test, per_year_rows)."""
    ev = [e for e in events if e.get(f"fwd{h}") is not None]
    years = sorted({e["date"].year for e in ev})
    pooled, per_year = [], []
    for y in years:
        train = [e for e in ev if e["date"].year < y]
        test = [e for e in ev if e["date"].year == y]
        if len(train) < min_train or len(test) < 10:
            continue
        signs = _orient_and_score(train, test, feats, h)
        test.sort(key=lambda e: e["score"])
        qn = max(1, len(test) // 5)
        q1, q5 = test[:qn], test[-qn:]
        s1, s5 = _stat([e[f"fwd{h}"] for e in q1]), _stat([e[f"fwd{h}"] for e in q5])
        per_year.append({"year": y, "n": len(test), "signs": dict(signs),
                         "q5_q1": (s5["mean"] - s1["mean"]) if (s1 and s5) else None,
                         "q5_mean": s5["mean"] if s5 else None, "q5_win": s5["win"] if s5 else None,
                         "q5_pf": s5["pf"] if s5 else None})
        pooled.extend(test)
    return pooled, per_year


def _pf(s):
    return "∞" if s and s["pf"] == float("inf") else (f"{s['pf']:.2f}" if s else "—")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["earnings", "baseline"], default="earnings")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--slip-bps", type=float, default=15.0)
    ap.add_argument("--max-fetch", type=int, default=4000, help="hard cap on AV calls this run")
    ap.add_argument("--cadence", type=int, default=7, help="baseline: days between grid points")
    ap.add_argument("--baseline-window", type=int, default=26, help="baseline: trailing obs (26w≈6m)")
    ap.add_argument("--baseline-method", choices=["z", "pct"], default="z")
    args = ap.parse_args()

    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or UNIVERSE
    start = _date.fromisoformat(args.start); end = _date.fromisoformat(args.end)
    slip = args.slip_bps / 10000.0
    feats = ["pcr", "cshare"]
    print(f"[{args.mode}] {len(universe)} symbols, {start}..{end}, slip {args.slip_bps}bps, "
          f"max_fetch {args.max_fetch}. Fetching put/call + vol/OI (cached)...")
    if args.mode == "earnings":
        ev = build_earnings_events(universe, start, slip, args.max_fetch)
        sub = f"cross-sectional @ earnings (T+1 after reportedDate), raw pcr & call-share"
    else:
        ev = build_baseline_events(universe, start, end, args.cadence, args.baseline_window,
                                   args.baseline_method, slip, args.max_fetch)
        sub = (f"name-relative: pcr & call-share vs each ticker's trailing {args.baseline_window}-obs "
               f"baseline ({args.baseline_method}), weekly grid")
    print(f"  built {len(ev)} events.")
    if len(ev) < 80:
        print("insufficient events — raise --max-fetch / widen window."); return

    L = [f"# Options-Positioning Signal — {args.mode.upper()} ({sub})", "",
         f"Bearish-crowding → mean-reversion-up test. Walk-forward by year (orientation+z fit on PRIOR "
         f"years only). Stock close→+H close, {args.slip_bps:.0f}bps round-trip slip. {len(ev)} events, "
         f"{start}→{end}. Q5 = signal says LONG, Q1 = avoid/short.", ""]
    csv_rows = []
    for h in HORIZONS:
        pooled, per_year = walk_forward(ev, feats, h)
        if not pooled:
            L += [f"## {h}d — insufficient walk-forward windows", ""]
            continue
        pooled.sort(key=lambda e: e["score"]); qn = max(1, len(pooled) // 5)
        s1 = _stat([e[f"fwd{h}"] for e in pooled[:qn]]); s5 = _stat([e[f"fwd{h}"] for e in pooled[-qn:]])
        spread = (s5["mean"] - s1["mean"]) * 100 if (s1 and s5) else 0.0
        # sign stability across years
        stab = {f: {yr["signs"].get(f, 0.0) for yr in per_year} for f in feats}
        stab_lbl = ", ".join(f"{f}:{'STABLE ' + ('+' if list(stab[f])[0] >= 0 else '−') if len(stab[f]) == 1 else 'UNSTABLE'}"
                             for f in feats)
        L += [f"## Horizon {h}d — pooled OOS (walk-forward)", "",
              f"Pooled Q5−Q1 **{spread:+.2f}%** · Q5 mean {s5['mean']*100:+.2f}% win {s5['win']*100:.0f}% "
              f"PF {_pf(s5)} (n={s5['n']}) · Q1 mean {s1['mean']*100:+.2f}% PF {_pf(s1)}.",
              f"Sign stability: **{stab_lbl}**.", "",
              "| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |",
              "|---|---|---|---|---|---|---|---|"]
        for yr in per_year:
            ps = "+" if yr["signs"].get("pcr", 0) >= 0 else "−"
            vs = "+" if yr["signs"].get("cshare", 0) >= 0 else "−"
            L.append(f"| {yr['year']} | {yr['n']} | {ps} | {vs} | "
                     f"{(yr['q5_q1'] or 0)*100:+.2f}% | {(yr['q5_mean'] or 0)*100:+.2f}% | "
                     f"{(yr['q5_win'] or 0)*100:.0f}% | {_pf({'pf': yr['q5_pf']}) if yr['q5_pf'] is not None else '—'} |")
        L.append("")
        if h == HORIZONS[1]:
            csv_rows = [{"sym": e["sym"], "date": e["date"].isoformat(), "pcr": e.get("pcr"),
                         "cshare": e.get("cshare"), "score": round(e.get("score", 0), 4),
                         **{f"fwd{hh}": e.get(f"fwd{hh}") for hh in HORIZONS}} for e in pooled]
    L += ["---", "_Honest costs: stock round-trip slippage only; no borrow/short costs modeled for the "
          "Q1 (short) leg. NOT wired to the live bot — deploy only if signs stay STABLE across years."
          " Baseline-relative is the fairer test; compare its sign-stability to the cross-sectional run._"]
    out = "\n".join(L)
    if ROOT:
        tag = f"{args.mode}_{args.baseline_method}" if args.mode == "baseline" else args.mode
        (ROOT / "backtest_outputs" / f"options_positioning_{tag}.md").write_text(out)
        if csv_rows:
            with open(ROOT / "backtest_outputs" / f"options_positioning_{tag}.csv", "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    print("\n" + out)


if __name__ == "__main__":
    main()
