#!/usr/bin/env python3
"""
rl_regime_study.py — Does a market-regime filter actually cut drawdown? (multi-regime test)
===========================================================================================

The strategy-overlay finding (docs/rl_strategy_overlay.md) was that on a 2-week, uniformly
bullish sample no filter can be tested. This module fetches a MULTI-REGIME daily history
(read-only Alpaca IEX, ~2020-07 → present: 2021/23/24 bull, 2022/25/26 bear) and tests the
literature's #1 lever in isolation: a long-only daily-momentum book, where the ONLY difference
between the two arms is a market-regime gate.

    signal (both arms):   hold names trading above their 50-DMA (long momentum)
    regime gate (arm B):  additionally require SPY above its 200-DMA → otherwise go to CASH

Same signal, same universe, same period — so any difference in drawdown / Calmar is the
regime filter's doing. Equal-weight across held names; next-day returns (no overlap).

SAFETY: read-only historical bars (IEX, never SIP by default), cached under
data/cache/alpaca_bars/. No orders, no live wiring, no gate/risk-control changes.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import rl_common as rc
import rl_alpaca_bar_enrichment as enr

# Tradeable universe (liquid large/mega caps + sector ETFs); SPY is the market-regime gauge.
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "NFLX",
            "JPM", "V", "MA", "UNH", "HD", "JNJ", "KO", "PG", "PEP", "COST", "WMT", "XOM", "CVX",
            "XLK", "XLF", "XLY", "XLV", "XLE", "XLI", "XLP", "XLC"]
START = os.getenv("REGIME_START", "2019-01-01")
WARMUP = 200


def _fetch_daily(symbols: List[str]) -> Dict[str, Any]:
    cache = enr.BarCache()
    start = datetime.fromisoformat(START).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    out = {}
    for s in list(dict.fromkeys(symbols + ["SPY"])):       # SPY is the regime gauge — always fetch it
        df = cache.get(s, "day", start, end)
        if df is not None and len(df):
            out[s] = df
    return out


def _sma(xs: List[float], w: int, i: int) -> Optional[float]:
    if i + 1 < w:
        return None
    seg = xs[i + 1 - w:i + 1]
    return sum(seg) / len(seg)


def build_book(daily: Dict[str, Any]) -> Dict[str, Any]:
    """Build aligned next-day-return series + signals. Returns per-date held sets and SPY regime."""
    import pandas as pd
    spy = daily.get("SPY")
    if spy is None or len(spy) < WARMUP + 5:
        return {}
    spy_close = [float(x) for x in spy["close"]]
    spy_dates = [d.date() for d in spy.index]
    spy_sma200 = {spy_dates[i]: _sma(spy_close, 200, i) for i in range(len(spy_close))}
    spy_px = {spy_dates[i]: spy_close[i] for i in range(len(spy_close))}

    # per symbol: date → (above_sma50, next_day_return)
    sym_signal: Dict[str, Dict[Any, tuple]] = {}
    for s, df in daily.items():
        if s == "SPY" or len(df) < WARMUP + 5:
            continue
        c = [float(x) for x in df["close"]]
        dates = [d.date() for d in df.index]
        sig = {}
        for i in range(WARMUP, len(c) - 1):
            sma50 = _sma(c, 50, i)
            if sma50 and c[i] > 0:
                nxt = c[i + 1] / c[i] - 1.0
                sig[dates[i]] = (c[i] > sma50, nxt)
        sym_signal[s] = sig

    all_dates = sorted(set().union(*[set(v) for v in sym_signal.values()]))
    return {"sym_signal": sym_signal, "all_dates": all_dates,
            "spy_sma200": spy_sma200, "spy_px": spy_px}


def simulate_book(book: Dict[str, Any], regime_gate: bool) -> Dict[str, Any]:
    """Daily equal-weight long book. regime_gate=True → flat (cash) when SPY ≤ its 200-DMA."""
    sym_signal = book["sym_signal"]
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    day_rets: List[float] = []
    dated: List[tuple] = []
    pos_days = 0
    invested_days = 0
    for d in book["all_dates"]:
        spy_px = book["spy_px"].get(d)
        spy_sma = book["spy_sma200"].get(d)
        bull = (spy_px is not None and spy_sma is not None and spy_px > spy_sma)
        if regime_gate and not bull:
            day_ret = 0.0                                  # go to cash in bear regime
        else:
            held = [sig[d][1] for s, sig in sym_signal.items()
                    if d in sig and sig[d][0]]             # above-50DMA longs
            day_ret = (sum(held) / len(held)) if held else 0.0
            if held:
                invested_days += 1
        equity *= (1.0 + day_ret)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        day_rets.append(day_ret)
        dated.append((d, day_ret))
        if day_ret > 0:
            pos_days += 1
    n = len(day_rets)
    mean = sum(day_rets) / n if n else 0.0
    std = math.sqrt(sum((r - mean) ** 2 for r in day_rets) / n) if n else 0.0
    total_ret = equity - 1.0
    years = n / 252.0 if n else 1.0
    cagr = (equity ** (1.0 / years) - 1.0) if years > 0 and equity > 0 else 0.0
    return {"days": n, "invested_days": invested_days,
            "total_return_pct": round(total_ret * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar": round(cagr / max_dd, 3) if max_dd > 0 else 0.0,
            "sharpe_annual": round((mean / std) * math.sqrt(252), 3) if std > 0 else 0.0,
            "positive_day_pct": round(pos_days / n * 100, 2) if n else 0.0,
            "_dated": dated}


def _year_drawdowns(dated: List[tuple]) -> Dict[str, float]:
    """Per-year max drawdown of the equity path (re-based each year) for regime comparison."""
    by_year: Dict[str, List[float]] = {}
    for d, r in dated:
        by_year.setdefault(str(d.year), []).append(r)
    out = {}
    for y, rets in by_year.items():
        eq = 1.0; peak = 1.0; mdd = 0.0
        for r in rets:
            eq *= (1 + r); peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
        out[y] = round(mdd * 100, 2)
    return out


def run() -> Dict[str, Any]:
    daily = _fetch_daily(UNIVERSE)
    book = build_book(daily)
    if not book:
        return {"error": "insufficient daily history (need SPY + universe)"}
    take_all = simulate_book(book, regime_gate=False)
    gated = simulate_book(book, regime_gate=True)
    res = {"universe": len([s for s in daily if s != "SPY"]), "start": START,
           "span_days": take_all["days"],
           "take_all": {k: v for k, v in take_all.items() if k != "_dated"},
           "regime_gated": {k: v for k, v in gated.items() if k != "_dated"},
           "per_year_max_dd": {"take_all": _year_drawdowns(take_all["_dated"]),
                               "regime_gated": _year_drawdowns(gated["_dated"])}}
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "rl_regime_study.json", res)
    return res


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    r = run()
    if "error" in r:
        print(r["error"]); raise SystemExit(0)
    print(f"Multi-regime daily long-only momentum book — {r['universe']} names, {r['span_days']} days from {r['start']}")
    cols = ["total_return_pct", "cagr_pct", "max_drawdown_pct", "calmar", "sharpe_annual", "positive_day_pct", "invested_days"]
    print(f"\n{'arm':14s}" + "".join(f"{c:>18s}" for c in cols))
    for arm in ("take_all", "regime_gated"):
        m = r[arm]
        print(f"{arm:14s}" + "".join(f"{m[c]:>18}" for c in cols))
    print("\nPer-year max drawdown (%) — take_all vs regime_gated:")
    yrs = sorted(r["per_year_max_dd"]["take_all"])
    print(f"{'year':6s}" + "".join(f"{y:>9s}" for y in yrs))
    print(f"{'all-in':6s}" + "".join(f"{r['per_year_max_dd']['take_all'][y]:>9}" for y in yrs))
    print(f"{'gated':6s}" + "".join(f"{r['per_year_max_dd']['regime_gated'][y]:>9}" for y in yrs))
