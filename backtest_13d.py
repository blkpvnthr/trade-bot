#!/usr/bin/env python3
"""
backtest_13d.py — Schedule 13D (activist) event signal through the upgraded gauntlet.

Long the subject at the next OPEN after the 13D filing date (PIT-correct: public when filed),
hold H days. 13D targets aren't in the repo bar set, so daily bars are fetched on demand
(Alpaca IEX, cached). Liquidity filter (20d avg $-vol ≥ min_adv) so costs are realistic.

Gauntlet: OOS(2024+) PF, per-year regime stability, daily-PnL MC, |corr vs vol-gated insider|<0.3,
realistic costs. PASS only if PF≥2 OOS AND positive every regime-year AND MC median>0 AND DD ok
AND low insider-corr AND n≥30.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

EVENTS = base.PROJECT_ROOT / "data" / "cache" / "pit_13d_events.json"
BARS_DIR = base.PROJECT_ROOT / "data" / "cache" / "daily_bars_iex"
TRAIN_END = _date(2024, 1, 1)
COST = 0.010                      # 1% round-trip — realistic for $1M-ADV small/mid-caps (insider-consistent)
MIN_ADV = 1e6                     # match the insider strategy floor (activists target small/mid-caps)
HORIZONS = [3, 5, 10, 20]
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
        key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
        from alpaca.data.historical import StockHistoricalDataClient
        _CLIENT = StockHistoricalDataClient(key, sec)
    return _CLIENT


def _bars(sym: str) -> Optional[pd.DataFrame]:
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    cf = BARS_DIR / f"{sym}.parquet"
    if cf.exists():
        try:
            df = pd.read_parquet(cf); return df if not df.empty else None
        except Exception:
            pass
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                               start=datetime(2020, 6, 1), end=datetime(2025, 1, 1), feed=DataFeed.IEX)
        raw = _client().get_stock_bars(req).df
        if raw is None or raw.empty:
            cf.write_bytes(b""); return None
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(sym, level=0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None) if pd.to_datetime(raw.index).tz is None \
            else pd.to_datetime(raw.index).tz_convert("America/New_York").tz_localize(None)
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df.to_parquet(cf); return df
    except Exception:
        try: cf.write_bytes(b"")
        except Exception: pass
        return None


def collect(hold):
    events = json.loads(EVENTS.read_text())
    rows = []
    for sym, dates in events.items():
        bars = _bars(sym)
        if bars is None or len(bars) < 60:
            continue
        idx = [d.date() if hasattr(d, "date") else d for d in bars.index]
        op = bars["open"].astype(float).values; cl = bars["close"].astype(float).values
        vol = bars["volume"].astype(float).values
        for ds in dates:
            try:
                fd = datetime.fromisoformat(ds[:10]).date()
            except Exception:
                continue
            # next trading day at/after filing date
            j = None
            for k, dd in enumerate(idx):
                if dd > fd:
                    j = k; break
            if j is None or j + hold >= len(idx) or j < 21:
                continue
            adv = float(np.mean([cl[m] * vol[m] for m in range(j - 20, j)]))
            entry = op[j]; exit_px = cl[j + hold]
            if entry <= 0 or exit_px <= 0 or adv < MIN_ADV:
                continue
            rows.append({"sym": sym, "date": idx[j], "ret": exit_px / entry - 1 - COST, "adv": adv})
    return rows


def _pf(v):
    w, l = v[v > 0], v[v < 0]
    return (w.sum() / abs(l.sum())) if len(l) else float("inf")


def gauntlet(rows, name, ins_daily):
    rng = np.random.RandomState(7)
    test = [r for r in rows if r["date"] >= TRAIN_END]
    tv = np.array([r["ret"] for r in test])
    if len(tv) < 20:
        return f"### {name}\nOOS n={len(tv)} too small → REJECT\n"
    sd = tv.std(); eq = peak = 1.0; mdd = 0.0
    for r in sorted(test, key=lambda x: x["date"]):
        eq *= (1 + 0.10 * r["ret"]); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    pf = _pf(tv)
    L = [f"### {name}", "",
         f"OOS 2024+: n={len(tv):,} mean {tv.mean()*100:+.2f}% median {np.median(tv)*100:+.2f}% "
         f"win {(tv>0).mean()*100:.0f}% PF {pf:.2f} MaxDD {mdd*100:.0f}%", "",
         "| Year | n | Mean | Win% | PF |", "|---|---|---|---|---|"]
    yr_fail = False; n_yr = 0
    for yr in range(2021, 2027):
        yv = np.array([r["ret"] for r in rows if r["date"].year == yr])
        if len(yv) < 10:
            continue
        n_yr += 1; ypf = _pf(yv)
        if ypf < 1.2:
            yr_fail = True
        L.append(f"| {yr}{' ⚠' if ypf<1.2 else ''} | {len(yv):,} | {yv.mean()*100:+.2f}% | {(yv>0).mean()*100:.0f}% | {ypf:.2f} |")
    meds = [np.prod(1 + rng.choice(tv, size=min(73, len(tv)), replace=True)) - 1 for _ in range(4000)]
    mc_p = float(np.mean([m > 0 for m in meds])); mc_med = float(np.median(meds))
    daily = {}
    for r in rows:
        daily.setdefault(r["date"], []).append(r["ret"])
    dser = {d: float(np.mean(v)) for d, v in daily.items()}
    common = sorted(set(dser) & set(ins_daily))
    corr = (float(np.corrcoef([dser[d] for d in common], [ins_daily[d] for d in common])[0, 1])
            if len(common) > 30 else float("nan"))
    ok = (pf >= 2.0 and not yr_fail and n_yr >= 4 and mc_med > 0 and mc_p >= 0.6 and mdd > -0.25
          and (abs(corr) < 0.3 if corr == corr else True) and len(tv) >= 30)
    L += ["", f"MC: P(profit)={mc_p*100:.0f}%, median={mc_med*100:+.2f}% | corr vs insider: {corr:+.2f}",
          f"**VERDICT: {'PASS' if ok else 'FAIL'}** (PF≥2 {'✓' if pf>=2 else '✗'}, every-year "
          f"{'✓' if not yr_fail else '✗'}, MC {'✓' if mc_med>0 and mc_p>=0.6 else '✗'}, "
          f"DD {'✓' if mdd>-0.25 else '✗'}, corr {'✓' if (abs(corr)<0.3 if corr==corr else True) else '✗'})", ""]
    return "\n".join(L)


def main():
    if not EVENTS.exists():
        print("pit_13d_events.json not built yet"); return
    ev = json.loads(EVENTS.read_text())
    print(f"13D events: {sum(len(v) for v in ev.values()):,} across {len(ev):,} tickers. Fetching bars...")
    ins = vg._collect(hold=3, min_adv=1_000_000)
    vthr = np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80)
    ib = {}
    for r in ins:
        if r["rv20"] >= vthr:
            ib.setdefault(r["date"], []).append((1 + r["gross"]) ** (1 / 3) - 1)
    ins_daily = {d: float(np.mean(v)) for d, v in ib.items()}
    L = ["# Schedule 13D (Activist) Event Signal — Upgraded Gauntlet", "",
         f"Long subject at next open after 13D filing, hold H days. Real IEX daily bars, "
         f"{COST*1e4:.0f}bps cost, ADV≥${MIN_ADV:,.0f}. {len(ev):,} tickers.", ""]
    for h in HORIZONS:
        rows = collect(h)
        L.append(gauntlet(rows, f"Hold {h} days (n_total={len(rows):,})", ins_daily))
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "candidate_13d.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
