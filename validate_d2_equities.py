#!/usr/bin/env python3
"""Validate the D2 + chop-filter lead on a BROAD liquid-equities universe (RESEARCH ONLY).

The equities bucket was the only place the D2 (DI+>DI-, rising ADX) + chop (ADX>=thr)
rule looked net-positive, but on tiny n (26-53 trades) with a hand-picked threshold.
This re-tests on ~60 liquid large-caps with an HONEST train/test split: pick the ADX
chop threshold on the TRAIN half, then report it OUT-OF-SAMPLE on the TEST half. If the
edge is real it survives more symbols + a holdout; if it's an artifact it won't.
"""
from __future__ import annotations
import argparse
import pandas as pd

from backtest_daily_green import fetch_5min
from backtest_dmi_adx import indicators, run_dmi, metrics, COST_BPS

EQUITIES = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "MA", "UNH", "HD", "PG", "XOM", "JNJ", "COST", "WMT", "LLY", "ABBV",
    "BAC", "KO", "PEP", "CVX", "MRK", "ADBE", "CRM", "NFLX", "AMD", "INTC",
    "CSCO", "QCOM", "TXN", "ORCL", "ACN", "MCD", "NKE", "DIS", "WFC", "GS",
    "MS", "CAT", "BA", "GE", "HON", "UNP", "LMT", "RTX", "LIN", "PFE",
    "TMO", "DHR", "ABT", "NEE", "PM", "T", "VZ", "CMCSA", "INTU", "AMAT",
]
EQ_COST = COST_BPS["equities"]  # 3 bps/side


def trades_for(frames, atr_mult, cross_lb, exit_mode, chop_adx):
    rows = []
    for sym, raw in frames.items():
        d = indicators(raw, 5)
        for t in run_dmi(d, "D2", atr_mult, cross_lb, EQ_COST, exit_mode, chop_adx):
            t2 = dict(t); t2["symbol"] = sym
            rows.append(t2)
    return pd.DataFrame(rows)


def split_metrics(df, cut, label):
    tr = df[df["date"] <= cut].to_dict("records")
    te = df[df["date"] > cut].to_dict("records")
    return metrics(tr), metrics(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--atr-mult", type=float, default=1.5)
    ap.add_argument("--cross-lb", type=int, default=3)
    ap.add_argument("--exit", choices=["bracket", "trend"], default="bracket")
    ap.add_argument("--train-frac", type=float, default=0.65)
    args = ap.parse_args()

    frames = fetch_5min(EQUITIES, args.days)
    print(f"[val] equities fetched={len(frames)} exit={args.exit} cost={EQ_COST}bps/side")

    # candidate chop thresholds (fixed grid; selection happens ON TRAIN only)
    grid = [0, 20, 25, 30]
    per = {}
    all_dates = None
    for thr in grid:
        df = trades_for(frames, args.atr_mult, args.cross_lb, args.exit, thr)
        per[thr] = df
        if len(df):
            all_dates = sorted(df["date"].unique()) if all_dates is None else all_dates
    if not all_dates:
        print("no trades"); return
    cut = all_dates[int(len(all_dates) * args.train_frac)]
    print(f"train<= {cut} < test\n")

    print("threshold |         TRAIN (Sharpe|PF|bps|n)        |          TEST/OOS (Sharpe|PF|bps|n)")
    best_thr, best_train_sh = None, -1e9
    for thr in grid:
        df = per[thr]
        if df.empty:
            print(f"ADX>={thr:<4} | no trades"); continue
        tr_m, te_m = split_metrics(df, cut, thr)
        print(f"ADX>={thr:<4} | {tr_m.get('net_Sharpe')}|{tr_m.get('PF')}|{tr_m.get('net_bps')}|{tr_m.get('trades')}"
              f"\t| {te_m.get('net_Sharpe')}|{te_m.get('PF')}|{te_m.get('net_bps')}|{te_m.get('trades')}")
        if tr_m.get("net_Sharpe", -9) > best_train_sh:
            best_train_sh, best_thr = tr_m.get("net_Sharpe", -9), thr

    # Honest OOS: threshold chosen on TRAIN, reported on TEST
    df = per[best_thr]
    _, te_best = split_metrics(df, cut, best_thr)
    print(f"\n==> Threshold selected on TRAIN = ADX>={best_thr} (train Sharpe {best_train_sh})")
    print(f"==> OUT-OF-SAMPLE (test) with that threshold: {te_best}")
    accept = (te_best.get("net_Sharpe", -9) > 0.5 and te_best.get("PF", 0) > 1.1
              and te_best.get("net_bps", -9) > 0 and te_best.get("trades", 0) >= 50)
    print(f"==> VERDICT: {'ACCEPT (survives honest OOS)' if accept else 'INCONCLUSIVE/REJECT (does not survive OOS)'}")


if __name__ == "__main__":
    main()
