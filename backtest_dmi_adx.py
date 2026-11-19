#!/usr/bin/env python3
"""M5 Variant D - DMI/ADX trend-continuation research (RESEARCH ONLY; no engine changes).

Falsification harness for three long-entry variants on completed 5-min candles.

Common SETUP filters (uptrend regime): DI+ > DI-, Close > SMA50, SMA50 slope up,
Bollinger lower-band slope up. ENTRY TRIGGER (all variants): close crosses ABOVE the
Bollinger midline (SMA20) while the setup is valid - enter the NEXT bar's open.

Variants (differ only in the ADX condition, so D1 vs B0 isolates ADX>DI+):
  B0  baseline (no ADX)      : setup + midline-cross
  D1  ADX > DI+              : setup + ADX>DI+ + midline-cross
  D2  rising ADX             : setup + ADX>20 + ADX>prev + midline-cross
  D3  DI+ crossover (recent) : DI+ crossed>DI- within N bars + ADX>20 + setup + midline-cross
Controls:
  SMA  simple SMA50 cross (naive trend control)
  DGA  daily-green (M4 Variant A: green vs open, open stop)

Management (identical across the DMI/SMA family): Wilder ATR(14) stop = entry - k*ATR,
2:1 target, EOD-flat, next-bar entry, one position per symbol at a time, risk-normalized
(results in R multiples). DGA uses its native open-stop. Instrument-class spread/slippage.
Walk-forward: K contiguous folds (no parameter fitting -> each fold is out-of-sample; the
previously-inspected 2026-07-02 holdout is NOT treated as unseen).

Usage: python backtest_dmi_adx.py [--atr-mult 1.5] [--slope-lb 5] [--folds 4]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

import trade
from backtest_daily_green import fetch_5min, ET

# Instrument classes -> per-side cost (bps). Majors tightest; leveraged/inverse widest.
MAJORS = {"SPY", "QQQ", "DIA"}
LEV_ETF = {"SQQQ", "TQQQ", "SOXL", "SOXS", "QID", "DRIP", "TECS", "TECL", "DOG", "SPXU",
           "SDS", "SH", "SDOW", "NVD", "NVDL", "PSQ", "SPXS", "UPRO", "SPXL"}
COST_BPS = {"majors": 1.0, "equities": 3.0, "lev_etf": 6.0}


def bucket_of(sym: str) -> str:
    if sym in MAJORS:
        return "majors"
    if sym in LEV_ETF:
        return "lev_etf"
    return "equities"


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def indicators(df: pd.DataFrame, slope_lb: int) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").reset_index(drop=True)
    h, l, c = d["high"].astype(float), d["low"].astype(float), d["close"].astype(float)
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    pdi = 100 * rma(plus_dm, 14) / atr.replace(0, np.nan)
    mdi = 100 * rma(minus_dm, 14) / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = rma(dx, 14)
    sma50 = c.rolling(50).mean()
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    bb_low = mid - 2 * std
    d["pdi"], d["mdi"], d["adx"], d["atr"] = pdi, mdi, adx, atr
    d["sma50"], d["mid"], d["bb_low"] = sma50, mid, bb_low
    d["sma50_up"] = sma50 > sma50.shift(slope_lb)
    d["bb_low_up"] = bb_low > bb_low.shift(slope_lb)
    d["mid_cross"] = (c > mid) & (c.shift() <= mid.shift())
    d["di_cross"] = (pdi > mdi) & (pdi.shift() <= mdi.shift())
    d["et"] = d["timestamp"].dt.tz_convert(ET)
    d["date"] = d["et"].dt.date
    d["rth"] = (d["et"].dt.time >= pd.Timestamp("09:30").time()) & (d["et"].dt.time < pd.Timestamp("16:00").time())
    return d


def entry_mask(d: pd.DataFrame, variant: str, cross_lb: int, chop_adx: float = 0.0) -> pd.Series:
    setup = (d["pdi"] > d["mdi"]) & (d["close"] > d["sma50"]) & d["sma50_up"] & d["bb_low_up"]
    if chop_adx > 0:  # "do nothing on chop": require a trending tape (ADX >= threshold)
        setup = setup & (d["adx"] >= chop_adx)
    if variant == "B0":
        return setup & d["mid_cross"]
    if variant == "D1":
        return setup & (d["adx"] > d["pdi"]) & d["mid_cross"]
    if variant == "D2":
        return setup & (d["adx"] > 20) & (d["adx"] > d["adx"].shift()) & d["mid_cross"]
    if variant == "D3":
        recent = d["di_cross"].rolling(cross_lb, min_periods=1).max() > 0
        return (recent & (d["pdi"] > d["mdi"]) & (d["adx"] > 20)
                & (d["close"] > d["sma50"]) & d["sma50_up"] & d["bb_low_up"] & d["mid_cross"])
    if variant == "SMA":  # naive control: close crosses above SMA50
        return (d["close"] > d["sma50"]) & (d["close"].shift() <= d["sma50"].shift())
    raise ValueError(variant)


def manage(d, entry_i, entry, stop, target, risk, sess_end_i, cost_side):
    highs, lows, closes = d["high"].values, d["low"].values, d["close"].values
    mfe = mae = 0.0
    exit_px, reason = None, None
    for j in range(entry_i, sess_end_i + 1):
        mfe = max(mfe, (highs[j] - entry) / risk)
        mae = max(mae, (entry - lows[j]) / risk)
        if lows[j] <= stop:
            exit_px, reason = stop, "stop"; break
        if highs[j] >= target:
            exit_px, reason = target, "target"; break
    if exit_px is None:
        exit_px, reason = float(closes[sess_end_i]), "eod"
    net = (exit_px - entry) / entry - 2 * (cost_side / 1e4)
    return {"net": net, "R": net / (risk / entry), "mfe_R": mfe, "mae_R": mae, "reason": reason}


def manage_trend(d, entry_i, entry, stop, risk, end_i, cost_side):
    """Trend-trailing exit: hold while SMA50 slope up AND BB-lower slope up (multi-day);
    exit when either flattens/curves down, or the ATR stop is hit. No 2:1, no EOD-flat."""
    highs, lows, closes = d["high"].values, d["low"].values, d["close"].values
    sma_up, bb_up = d["sma50_up"].values, d["bb_low_up"].values
    pdi, mdi = d["pdi"].values, d["mdi"].values
    mfe = mae = 0.0
    exit_px, reason, exit_i = None, None, end_i
    for j in range(entry_i, end_i + 1):
        mfe = max(mfe, (highs[j] - entry) / risk)
        mae = max(mae, (entry - lows[j]) / risk)
        if lows[j] <= stop:
            exit_px, reason, exit_i = stop, "stop", j; break
        if pdi[j] < mdi[j]:  # hard exit: directional momentum flipped bearish
            exit_px, reason, exit_i = float(closes[j]), "di_flip", j; break
        if not (bool(sma_up[j]) and bool(bb_up[j])):  # trend flattened / turned down
            exit_px, reason, exit_i = float(closes[j]), "trend_exit", j; break
    if exit_px is None:
        exit_px, reason = float(closes[end_i]), "end"
    net = (exit_px - entry) / entry - 2 * (cost_side / 1e4)
    return {"net": net, "R": net / (risk / entry), "mfe_R": mfe, "mae_R": mae, "reason": reason}, exit_i


def run_dmi(d, variant, atr_mult, cross_lb, cost_side, exit_mode="bracket", chop_adx=0.0):
    mask = entry_mask(d, variant, cross_lb, chop_adx).values
    rth = d["rth"].values
    dates = d["date"].values
    opens, atrs = d["open"].values, d["atr"].values
    n = len(d)
    sess_end = {}  # date -> last RTH index
    for i in range(n):
        if rth[i]:
            sess_end[dates[i]] = i
    trades, open_until = [], -1
    for i in range(n - 1):
        if i <= open_until or not rth[i] or not mask[i]:
            continue
        if not rth[i + 1] or dates[i + 1] != dates[i]:
            continue  # need a next bar to enter on
        entry = float(opens[i + 1])
        a = float(atrs[i]) if atrs[i] == atrs[i] else 0.0
        if entry <= 0 or a <= 0:
            continue
        risk = atr_mult * a
        if exit_mode == "trend":
            tr, exit_i = manage_trend(d, i + 1, entry, entry - risk, risk, n - 1, cost_side)
            open_until = exit_i  # one position per symbol until the trend trade closes
        else:
            tr = manage(d, i + 1, entry, entry - risk, entry + 2 * risk, risk, sess_end[dates[i]], cost_side)
            open_until = sess_end[dates[i]]
        tr.update({"date": dates[i]})
        trades.append(tr)
    return trades


def run_dga(d, cost_side):
    """Daily-green control: enter when close>today's open; stop at open; 2:1; EOD."""
    trades = []
    for date, g in d[d["rth"]].groupby("date"):
        g = g.reset_index(drop=True)
        o = float(g["open"].iloc[0])
        if o <= 0:
            continue
        closes, opens = g["close"].values, g["open"].values
        ncols = len(g)
        ei = None
        for i in range(ncols - 1):
            if closes[i] > o:
                ei = i + 1; break
        if ei is None:
            continue
        entry = float(opens[ei])
        if entry <= o:
            continue
        risk = max(entry - o, 0.0015 * entry)
        tr = manage(g, ei, entry, o, entry + 2 * risk, risk, ncols - 1, cost_side)
        tr.update({"date": date})
        trades.append(tr)
    return trades


def metrics(trades, max_positions=5):
    if not trades:
        return {"trades": 0}
    df = pd.DataFrame(trades)
    wins = df[df["net"] > 0]["net"]; losses = df[df["net"] <= 0]["net"]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    # risk-normalized portfolio: equal risk per trade, cap positions/day, compound 1% risk
    daily = df.groupby("date").apply(lambda x: x.head(max_positions)["R"].mean())
    daily = daily.sort_index()
    eq = (1 + 0.01 * daily).cumprod()
    years = max(len(daily) / 252.0, 1e-9)
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if len(eq) else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    return {
        "trades": len(df), "win%": round(100 * len(wins) / len(df), 1),
        "net_bps": round(1e4 * df["net"].mean(), 1), "PF": round(pf, 2),
        "net_Sharpe": round(sharpe, 2), "CAGR%": round(100 * cagr, 1), "maxDD%": round(100 * dd, 1),
        "avg_MFE_R": round(df["mfe_R"].mean(), 2), "avg_MAE_R": round(df["mae_R"].mean(), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--atr-mult", type=float, default=1.5)
    ap.add_argument("--slope-lb", type=int, default=5)
    ap.add_argument("--cross-lb", type=int, default=3)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--exit", choices=["bracket", "trend"], default="bracket",
                    help="bracket=ATR stop+2:1+EOD; trend=hold while slopes up, exit on flatten/DI-flip/stop (multi-day)")
    ap.add_argument("--chop-adx", type=float, default=0.0, help="skip entries when ADX < this (do nothing on chop)")
    args = ap.parse_args()

    syms = trade.load_trading_universe()
    frames = fetch_5min(syms, args.days)
    print(f"[m5] symbols={len(frames)} atr_mult={args.atr_mult} slope_lb={args.slope_lb} folds={args.folds} exit={args.exit}")

    variants = ["B0", "D1", "D2", "D3", "SMA", "DGA"]
    # collect trades tagged by symbol + bucket
    all_trades = {v: [] for v in variants}
    for sym, raw in frames.items():
        d = indicators(raw, args.slope_lb)
        b = bucket_of(sym); cost = COST_BPS[b]
        for v in variants:
            trs = run_dga(d, cost) if v == "DGA" else run_dmi(d, v, args.atr_mult, args.cross_lb, cost, args.exit, args.chop_adx)
            for t in trs:
                t2 = dict(t); t2["symbol"] = sym; t2["bucket"] = b
                all_trades[v].append(t2)

    buckets = ["majors", "equities", "lev_etf", "ALL"]
    print("\n================ COMPARATIVE PERFORMANCE MATRIX (net_Sharpe | PF | net_bps | n) ================")
    header = f"{'variant':<6}" + "".join(f"{b:>26}" for b in buckets)
    print(header)
    matrix = {}
    for v in variants:
        row = f"{v:<6}"
        matrix[v] = {}
        for b in buckets:
            trs = all_trades[v] if b == "ALL" else [t for t in all_trades[v] if t["bucket"] == b]
            m = metrics(trs)
            matrix[v][b] = m
            cell = f"{m.get('net_Sharpe','-')}|{m.get('PF','-')}|{m.get('net_bps','-')}|{m.get('trades',0)}"
            row += f"{cell:>26}"
        print(row)

    # Walk-forward: contiguous folds over dates (no param fitting -> each fold is OOS)
    def fold_sharpes(trs):
        if not trs:
            return []
        df = pd.DataFrame(trs); dts = sorted(df["date"].unique())
        out = []
        for k in range(args.folds):
            lo = dts[int(len(dts) * k / args.folds)]; hi = dts[int(len(dts) * (k + 1) / args.folds) - 1]
            sub = df[(df["date"] >= lo) & (df["date"] <= hi)].to_dict("records")
            out.append(metrics(sub).get("net_Sharpe", 0.0))
        return out

    print("\n================ WALK-FORWARD net_Sharpe per fold (ALL symbols) ================")
    for v in variants:
        print(f"{v:<6} folds={fold_sharpes(all_trades[v])}")

    # Verdicts
    print("\n================ VERDICTS ================")
    b0 = matrix["B0"]["ALL"]
    for v in ["D1", "D2", "D3"]:
        m = matrix[v]["ALL"]
        folds = fold_sharpes(all_trades[v])
        pos_folds = sum(1 for s in folds if s > 0)
        beats_dga = m.get("net_Sharpe", -9) > matrix["DGA"]["ALL"].get("net_Sharpe", -9)
        beats_sma = m.get("net_Sharpe", -9) > matrix["SMA"]["ALL"].get("net_Sharpe", -9)
        accept = (m.get("net_Sharpe", -9) > 0.5 and m.get("PF", 0) > 1.1 and m.get("net_bps", -9) > 0
                  and pos_folds >= max(3, args.folds - 1) and beats_dga and beats_sma)
        verdict = "ACCEPT" if accept else "INCONCLUSIVE"
        print(f"{v}: {verdict} | net_Sharpe={m.get('net_Sharpe')} PF={m.get('PF')} net_bps={m.get('net_bps')} "
              f"pos_folds={pos_folds}/{args.folds} beats_DGA={beats_dga} beats_SMA={beats_sma}")
    d1, b0m = matrix["D1"]["ALL"], matrix["B0"]["ALL"]
    print(f"\nIncremental ADX>DI+ value (D1 vs B0 baseline): "
          f"net_bps {b0m.get('net_bps')} -> {d1.get('net_bps')}, "
          f"Sharpe {b0m.get('net_Sharpe')} -> {d1.get('net_Sharpe')}, "
          f"PF {b0m.get('PF')} -> {d1.get('PF')}, trades {b0m.get('trades')} -> {d1.get('trades')}")


if __name__ == "__main__":
    main()
