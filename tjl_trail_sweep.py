#!/usr/bin/env python3
"""
tjl_trail_sweep.py — sweep the trailing-stop gap under the swing+trail+PDH config and
report the aggregate for each gap on the SAME data (bars fetched once per symbol, reused
across gaps — no redundant Alpaca calls, so the only variable is trail_gap_pct).

Apples-to-apples on the FULL universe (no PF filter) — this is a parameter sensitivity
scan, not winner selection. Needs Alpaca data keys in env.

Usage: python tjl_trail_sweep.py [SYM,SYM,...] [days]
"""
from __future__ import annotations
import sys

import tjl_backtest as B
import tjl_signal as T

# ---- fetch-once cache: get_bars(symbol, tf_min, days) is called 3x/symbol/gap ----
_orig_get_bars = B.get_bars
_cache: dict = {}
def _cached_get_bars(symbol, tf_min, days):
    key = (symbol, tf_min, days)
    if key not in _cache:
        _cache[key] = _orig_get_bars(symbol, tf_min, days)
    return _cache[key]
B.get_bars = _cached_get_bars

UNIVERSE = (sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1]
            else ["AMD", "NVDA", "MU", "AAPL", "MSFT"])
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
GAPS = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
WATCH = ["AMAT", "QBTS", "IONQ", "GTLB", "ASML", "SHLS"]   # names to track per gap


def run(gap: float):
    cfg = T.TJLConfig(swing_mode=True, use_trail=True, trail_require_pdh=True, trail_gap_pct=gap)
    rets, names, per = [], 0, {}
    for sym in UNIVERSE:
        trades, st = B.backtest(sym, cfg, DAYS)
        if st["source"] != "alpaca" or st["trades"] == 0:
            continue
        names += 1
        r = [tr.ret_pct for tr in trades]
        rets += r
        per[sym] = round(sum(r), 2)
    wins = sum(1 for r in rets if r > 0)
    gw = sum(r for r in rets if r > 0)
    gl = -sum(r for r in rets if r <= 0)
    n_keep = sum(1 for s, v in per.items() if v > 0)   # rough PF>=1 proxy: positive total
    return {
        "gap": gap, "names": names, "trades": len(rets),
        "win": round(wins / len(rets) * 100, 1) if rets else 0.0,
        "total": round(sum(rets), 2),
        "avg": round(sum(rets) / len(rets), 3) if rets else 0.0,
        "pf": (round(gw / gl, 2) if gl > 0 else float("inf")),
        "pos_names": n_keep, "per": per,
    }


def main():
    print(f"TJL trail-gap sweep — swing+trail+PDH, {len(UNIVERSE)} names, {DAYS}d, full universe\n")
    rows = [run(g) for g in GAPS]
    print(f"{'gap%':>5}{'names':>7}{'trades':>8}{'win%':>7}{'total%':>9}{'avgR%':>8}{'PF':>7}{'+names':>8}")
    best = max(rows, key=lambda r: r["pf"] if r["pf"] != float("inf") else -1)
    for r in rows:
        pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        star = "  <- best PF" if r is best else ""
        print(f"{r['gap']*100:>5.1f}{r['names']:>7}{r['trades']:>8}{r['win']:>6.1f}%"
              f"{r['total']:>8.2f}%{r['avg']:>7.3f}%{pf:>7}{r['pos_names']:>8}{star}")
    print(f"\nper-name total% by gap ({', '.join(f'{int(g*1000)/10}' for g in GAPS)}):")
    for s in WATCH:
        cells = " ".join(f"{r['per'].get(s, '--'):>7}" for r in rows)
        print(f"  {s:<6}{cells}")


if __name__ == "__main__":
    main()
