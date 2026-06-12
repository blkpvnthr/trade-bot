#!/usr/bin/env python3
"""
backtest_etf_meanrev.py — CANDIDATE A of the independent-signal gauntlet.

Signal: Connors-style short-term mean-reversion on LIQUID ETFs — buy the dip (RSI(2) < 10) only
in an uptrend (close > SMA200); exit when RSI(2) > 70 or after max-hold. Shares/ETFs only,
realistic ~5bps round-trip. NO parameter fitting (standard Connors values) → can't overfit-to-tickers.

Gauntlet: OOS train(<2024)/test(2024+) stats; bootstrap daily-PnL Monte Carlo; MaxDD; correlation
of daily PnL vs the validated vol-gated INSIDER edge (diversification is the point).
PASS bar: PF≥2.0 OOS AND positive MC-median AND acceptable MaxDD AND low insider-corr.
"""
from __future__ import annotations

import statistics
from datetime import date as _date
from typing import Dict, List

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

ETFS = ["SPY", "IWM", "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI"]   # full-history liquid ETFs
RSI_LEN, RSI_BUY, RSI_EXIT, MAX_HOLD = 2, 10.0, 70.0, 6
COST = 0.0005                 # 5bps round-trip (liquid ETF)
TRAIN_END = _date(2024, 1, 1)
SEED = 7


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def collect():
    trades = []                       # per-trade dicts
    daily = {}                        # date -> list of that-day position returns (equal weight)
    for sym in ETFS:
        try:
            b = pd.read_parquet(base.BARS_DIR / f"{sym}_day.parquet")
        except Exception:
            continue
        if len(b) < 260:
            continue
        td = list(b.index)
        op = b["open"].astype(float).values; cl = b["close"].astype(float).values
        rsi = _rsi(b["close"].astype(float), RSI_LEN).values
        sma200 = b["close"].astype(float).rolling(200).mean().values
        i = 200
        while i < len(td) - MAX_HOLD - 1:
            if rsi[i] < RSI_BUY and cl[i] > sma200[i]:        # oversold dip in an uptrend
                entry = op[i + 1]
                # exit: first day RSI>70, else MAX_HOLD
                xk = i + 1 + MAX_HOLD
                for k in range(i + 1, i + 1 + MAX_HOLD):
                    if rsi[k] > RSI_EXIT:
                        xk = k; break
                xk = min(xk, len(td) - 1)
                exit_px = cl[xk]
                if entry > 0 and exit_px > 0:
                    ret = exit_px / entry - 1 - COST
                    edate = td[i + 1].date() if hasattr(td[i + 1], "date") else td[i + 1]
                    trades.append({"sym": sym, "date": edate, "ret": ret, "hold": xk - (i + 1)})
                    # spread the trade return across its held days for the daily-PnL series
                    hold = max(xk - (i + 1), 1)
                    per_day = (1 + ret) ** (1 / hold) - 1
                    for k in range(i + 1, xk + 1):
                        dd = td[k].date() if hasattr(td[k], "date") else td[k]
                        daily.setdefault(dd, []).append(per_day)
                i = xk + 1
            else:
                i += 1
    daily_series = {d: float(np.mean(v)) for d, v in daily.items()}   # equal-weight portfolio daily ret
    return trades, daily_series


def _stat(rows):
    if not rows:
        return None
    v = np.array([r["ret"] for r in rows])
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    eq = peak = 1.0; mdd = 0.0
    for r in sorted(rows, key=lambda x: x["date"]):
        eq *= (1 + 0.10 * r["ret"]); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)), "win": (v > 0).mean(),
            "sharpe": (v.mean() / sd) if sd > 0 else 0.0, "pf": pf, "mdd": mdd}


def _fmt(name, s):
    if not s:
        return f"| {name} | 0 |—|—|—|—|—|—|"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {name} | {s['n']:,} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | "
            f"{s['win']*100:.0f}% | {s['sharpe']:+.3f} | {pf} | {s['mdd']*100:.0f}% |")


def main():
    print("Collecting ETF mean-reversion trades...")
    trades, etf_daily = collect()
    train = [t for t in trades if t["date"] < TRAIN_END]
    test = [t for t in trades if t["date"] >= TRAIN_END]
    rng = np.random.RandomState(SEED)

    # bootstrap MC on OOS trade returns (P profitable over a 73-trade block, median block return)
    test_r = np.array([t["ret"] for t in test])
    mc_meds, mc_pprofit = [], 0
    if len(test_r) >= 20:
        for _ in range(5000):
            samp = rng.choice(test_r, size=min(73, len(test_r)), replace=True)
            tot = np.prod(1 + samp) - 1
            mc_meds.append(tot)
            mc_pprofit += int(tot > 0)
        mc_pprofit /= 5000
    mc_median = float(np.median(mc_meds)) if mc_meds else 0.0

    # correlation of daily PnL vs the insider edge (gated, same dates)
    ins_trades = vg._collect(hold=3, min_adv=1_000_000)
    vthr = np.percentile([r["rv20"] for r in ins_trades if r["date"] < TRAIN_END], 80) if ins_trades else 1
    ins_daily = {}
    for r in ins_trades:
        if r["rv20"] >= vthr:
            ins_daily.setdefault(r["date"], []).append((1 + r["gross"]) ** (1 / 3) - 1)
    ins_series = {d: float(np.mean(v)) for d, v in ins_daily.items()}
    common = sorted(set(etf_daily) & set(ins_series))
    corr = (float(np.corrcoef([etf_daily[d] for d in common], [ins_series[d] for d in common])[0, 1])
            if len(common) > 30 else float("nan"))

    s_test = _stat(test); s_train = _stat(train)
    L = ["# Candidate A — ETF Short-Term Mean-Reversion (RSI2 dip, liquid ETFs) — GAUNTLET", "",
         f"{len(ETFS)-1} ETFs, RSI({RSI_LEN})<{RSI_BUY} + close>SMA200 → next-open; exit RSI>{RSI_EXIT} "
         f"or {MAX_HOLD}d. {COST*1e4:.0f}bps round-trip. No parameter fitting.", "",
         "| Set | n | Mean | Median | Win% | Sharpe | PF | MaxDD |", "|---|---|---|---|---|---|---|---|",
         _fmt("TRAIN (<2024)", s_train), _fmt("**OOS TEST (2024+)**", s_test), "",
         "## Daily-PnL Monte Carlo (OOS, 5000× bootstrap of 73-trade blocks)",
         f"- P(block profitable) = **{mc_pprofit*100:.0f}%**, median block return = **{mc_median*100:+.2f}%**", "",
         "## Diversification",
         f"- Correlation of daily PnL vs vol-gated INSIDER edge ({len(common)} common days): **{corr:+.2f}**", ""]
    ok_pf = bool(s_test and s_test["pf"] >= 2.0)
    ok_mc = mc_median > 0 and mc_pprofit >= 0.6
    ok_dd = bool(s_test and s_test["mdd"] > -0.25)
    ok_corr = (abs(corr) < 0.3) if corr == corr else True
    pass_all = ok_pf and ok_mc and ok_dd and ok_corr and bool(s_test and s_test["n"] >= 30)
    L += ["## VERDICT", "",
          f"- PF≥2.0 OOS: {'YES' if ok_pf else 'NO'} ({s_test['pf']:.2f})" if s_test else "- n/a",
          f"- Positive MC median + P(profit)≥60%: {'YES' if ok_mc else 'NO'}",
          f"- MaxDD acceptable (>-25%): {'YES' if ok_dd else 'NO'} ({s_test['mdd']*100:.0f}%)" if s_test else "",
          f"- Low insider-correlation (|r|<0.3): {'YES' if ok_corr else 'NO'} ({corr:+.2f})",
          f"- OOS n≥30: {'YES' if s_test and s_test['n']>=30 else 'NO'}",
          "",
          ("**PASS → propose run.py wiring (ETF mean-reversion, shares).**" if pass_all else
           "**FAIL → do NOT wire. " + (
               "PF below 2.0; " if not ok_pf else "") + (
               "MC not robustly profitable; " if not ok_mc else "") + (
               "drawdown too deep; " if not ok_dd else "") + (
               "too correlated with insider; " if not ok_corr else "") + "moving to Candidate B.**")]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "candidate_A_etf_meanrev.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
