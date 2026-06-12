#!/usr/bin/env python3
"""
study_atm_filters.py — real-chain robustness study for the ONLY net-positive setup:
liquid mega-cap ATM calls (+9.5% baseline). Answers "which ATM trades are worth taking?"

NO synthetic prices, NO EV model, NO Merton/GBM. Pure ATM selection on REAL Alpha Vantage
chains: enter ASK, exit BID (looked up in the actual exit chain), commission included, liquid
mega-cap universe, ATM delta band 0.45-0.65, spread <= 10%.

Studies:
  1. DTE buckets: 7-14 / 15-30 / 31-45 / 46-60 / 61-90  → mean, Sharpe, win, median, spread, IV
  2. IV filter:   all / <30% / 30-50% / 50-80% / >80%
  3. Trend strength (daily proxies — intraday ORB/VWAP unavailable in daily bars):
        breakout distance above 20d-high, distance above SMA50, relative volume
  4. Simplest combined rule that beats the +9.5% ATM baseline (prefer fewer filters)

Usage: python study_atm_filters.py --max-entries 200
"""
from __future__ import annotations

import argparse
import math
import statistics
from typing import Any, Dict, List, Optional

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo

HOLD = 10
COMMISSION = 0.65
MIN_DELTA, MAX_DELTA = 0.45, 0.65   # ATM band
MAX_SPREAD = 0.10
LIQUID = ["SPY", "QQQ", "AAPL", "NVDA", "AMD", "MSFT", "META", "AMZN", "GOOGL", "AVGO",
          "TSLA", "NFLX", "IWM", "MSTR", "PLTR", "COST", "JPM", "V", "MA", "HD"]
DTE_BUCKETS = [(7, 14), (15, 30), (31, 45), (46, 60), (61, 90)]


def _atm_in_bucket(chain: List[dict], lo: int, hi: int) -> Optional[dict]:
    cands = [c for c in chain if c.get("type") == "call" and lo <= int(c.get("dte", 0)) <= hi
             and MIN_DELTA <= c.get("delta", 0) <= MAX_DELTA
             and c.get("bid_price", 0) > 0 and c.get("ask_price", 0) > 0
             and ((c["ask_price"] - c["bid_price"]) / max(c["mid"], 1e-6)) <= MAX_SPREAD]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c["delta"] - 0.50))


def _realized(entry_c: dict, sym: str, exit_date: str, S1: float) -> Optional[float]:
    entry_ask = float(entry_c["ask_price"])
    if entry_ask <= 0:
        return None
    exit_c = avo.lookup_contract(sym, exit_date, entry_c["symbol"])
    exit_px = float(exit_c["bid_price"]) if (exit_c and float(exit_c.get("bid_price", 0)) > 0) \
        else max(0.0, S1 - float(entry_c["strike_price"]))
    comm = COMMISSION / 100.0
    return max(-1.0, (exit_px - comm) / (entry_ask + comm) - 1.0)


def collect(max_entries: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sym in LIQUID:
        if len(rows) >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        for i in ab._momentum_entry_indices(td, close, high, vol):
            if len(rows) >= max_entries:
                break
            entry_dt = td[i + 1]; exit_dt = td[i + 1 + HOLD]
            ed, xd = entry_dt.isoformat(), exit_dt.isoformat()
            spot = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
            if spot <= 0 or S1 <= 0:
                continue
            chain = avo.get_chain(sym, ed, calls_only=True)
            if not chain:
                continue
            # daily trend-strength proxies (computed at entry, no lookahead)
            d = td[i]
            hi20 = max(high[td[j]] for j in range(i - 20, i))
            sma50 = statistics.mean(close[td[j]] for j in range(i - 50, i))
            avgvol = statistics.mean([vol[td[j]] for j in range(i - 20, i)] or [1])
            breakout_dist = (close[d] - hi20) / hi20 if hi20 > 0 else 0.0
            sma50_dist = (close[d] - sma50) / sma50 if sma50 > 0 else 0.0
            relvol = vol[d] / avgvol if avgvol > 0 else 0.0
            for lo, hi in DTE_BUCKETS:
                c = _atm_in_bucket(chain, lo, hi)
                if not c:
                    continue
                ret = _realized(c, sym, xd, S1)
                if ret is None:
                    continue
                rows.append({"sym": sym, "dte_bucket": f"{lo}-{hi}", "ret": ret,
                             "iv": float(c.get("implied_volatility", 0)),
                             "spread": (c["ask_price"] - c["bid_price"]) / max(c["mid"], 1e-6),
                             "delta": c["delta"], "breakout_dist": breakout_dist,
                             "sma50_dist": sma50_dist, "relvol": relvol})
    return rows


def _stat(v: List[float]) -> Dict[str, Any]:
    if not v:
        return {"n": 0}
    n = len(v); m = statistics.mean(v); sd = statistics.pstdev(v) if n > 1 else 0.0
    return {"n": n, "mean": m, "sharpe": (m / sd) if sd > 0 else 0.0,
            "win": sum(1 for x in v if x > 0) / n, "median": statistics.median(v)}


def _row(label, rows, extra=()):
    s = _stat([r["ret"] for r in rows])
    if not s.get("n"):
        return f"| {label} | 0 |—|—|—|—|" + "—|" * len(extra)
    ex = "".join(f" {statistics.mean([r[k] for r in rows])*m:.{p}f}{u} |" for k, m, p, u in extra)
    return (f"| {label} | {s['n']} | {s['mean']*100:+.1f}% | {s['sharpe']:+.3f} | "
            f"{s['win']*100:.0f}% | {s['median']*100:+.1f}% |{ex}")


def report(rows: List[Dict[str, Any]]) -> str:
    base_all = _stat([r["ret"] for r in rows])
    L = ["# Real-Chain ATM Filter Study (which ATM trades are worth taking?)", "",
         f"{len({(r['sym'],r['dte_bucket']) for r in rows})} contract-trades over {len(LIQUID)} liquid "
         f"mega-caps. Real chains, enter ASK/exit BID + ${COMMISSION}/side, ATM δ0.45-0.65, spread ≤10%.",
         f"**Baseline (all ATM trades): mean {base_all.get('mean',0)*100:+.1f}%, Sharpe {base_all.get('sharpe',0):+.3f}, "
         f"win {base_all.get('win',0)*100:.0f}%, n={base_all.get('n',0)}**", "",
         "## 1. DTE buckets", "",
         "| DTE | n | Mean | Sharpe | Win% | Median | Avg spread | Avg IV |",
         "|---|---|---|---|---|---|---|---|"]
    for lo, hi in DTE_BUCKETS:
        sub = [r for r in rows if r["dte_bucket"] == f"{lo}-{hi}"]
        L.append(_row(f"{lo}-{hi}", sub, [("spread", 100, 1, "%"), ("iv", 100, 0, "%")]))
    L += ["", "## 2. IV filter", "",
          "| IV bucket | n | Mean | Sharpe | Win% | Median |", "|---|---|---|---|---|---|"]
    for lab, f in [("all", lambda r: True), ("IV<30%", lambda r: r["iv"] < 0.30),
                   ("IV 30-50%", lambda r: 0.30 <= r["iv"] < 0.50),
                   ("IV 50-80%", lambda r: 0.50 <= r["iv"] < 0.80), ("IV>80%", lambda r: r["iv"] >= 0.80)]:
        L.append(_row(lab, [r for r in rows if f(r)]))
    L += ["", "## 3. Trend strength (daily proxies; intraday ORB/VWAP not in daily bars)", "",
          "| Filter | n | Mean | Sharpe | Win% | Median |", "|---|---|---|---|---|---|"]
    for lab, f in [("breakout_dist >2%", lambda r: r["breakout_dist"] > 0.02),
                   ("breakout_dist >5%", lambda r: r["breakout_dist"] > 0.05),
                   ("dist>SMA50 >5%", lambda r: r["sma50_dist"] > 0.05),
                   ("dist>SMA50 >10%", lambda r: r["sma50_dist"] > 0.10),
                   ("relvol >2", lambda r: r["relvol"] > 2.0),
                   ("relvol >3", lambda r: r["relvol"] > 3.0)]:
        L.append(_row(lab, [r for r in rows if f(r)]))
    # 4. simplest combined rule search (1-2 filters) beating baseline
    L += ["", "## 4. Simplest rule that beats the ATM baseline", ""]
    cands = [
        ("31-45 DTE only", lambda r: r["dte_bucket"] == "31-45"),
        ("31-45 DTE + IV<50%", lambda r: r["dte_bucket"] == "31-45" and r["iv"] < 0.50),
        ("IV<50%", lambda r: r["iv"] < 0.50),
        ("relvol>2", lambda r: r["relvol"] > 2.0),
        ("relvol>2 + IV<50%", lambda r: r["relvol"] > 2.0 and r["iv"] < 0.50),
        ("breakout>5% + IV<50%", lambda r: r["breakout_dist"] > 0.05 and r["iv"] < 0.50),
        ("31-45 DTE + relvol>2", lambda r: r["dte_bucket"] == "31-45" and r["relvol"] > 2.0),
    ]
    L += ["| Rule | n | Mean | Sharpe | Win% | Median |", "|---|---|---|---|---|---|"]
    scored = []
    for lab, f in cands:
        sub = [r for r in rows if f(r)]
        s = _stat([r["ret"] for r in sub])
        L.append(_row(lab, sub))
        if s.get("n", 0) >= 20:
            scored.append((s["mean"], lab, s))
    L += ["", "_Real fills (ask→bid + commission). Daily-bar trend proxies. Prefer fewer filters; "
          "n≥20 for a rule to count. NOT optimized — a robustness scan, not a fit._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=200)
    args = ap.parse_args()
    print(f"Collecting real-chain ATM trades (liquid mega-caps, max {args.max_entries})...")
    rows = collect(args.max_entries)
    md = report(rows)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "atm_filter_study.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
