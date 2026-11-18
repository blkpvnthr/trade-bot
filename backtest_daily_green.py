#!/usr/bin/env python3
"""Historical + OOS backtest of the daily-green 2:1 strategy (intraday, 5-min bars).

Falsification-first: no look-ahead (entry at the NEXT bar's open after the green-vs-open
signal; stop assumed to fill before target within a bar - conservative), realistic
slippage, and an out-of-sample holdout split. This measures the CORE edge; it does not
model pyramiding (position-level, path-dependent) - that is a sizing overlay, not the edge.

Variants:
  A core      : enter when price > today's open; exit at 2:1 target, open stop, or EOD.
  B +mtf5m    : A, but only enter when the day's 5-min momentum is bullish at entry
                (close>VWAP, EMA9>EMA21, rising) - the look-ahead-free LTF part of M4.
  C +reversal : B, plus the sensitive reversal profit-exit (exit in profit when 5m
                momentum is lost) instead of always waiting for the 2:1 target.

Usage: python backtest_daily_green.py [--days 400] [--slippage-bps 5] [--max-positions 5]
"""
from __future__ import annotations
import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import trade
import mtf_gate
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

ET = trade.ET
SCRATCH = "/private/tmp/claude-501/-Users-blkpvnthr-Desktop-trader/1712605e-ef05-4b7b-81f4-fef20f8ea931/scratchpad"
MIN_RISK_PCT = 0.0015


def fetch_5min(symbols, days):
    # Cache is CSV (not pickle): self-generated market data, safe + dependency-free.
    cache = os.path.join(SCRATCH, f"bt_5min_{days}d.csv")
    if os.path.exists(cache):
        alldf = pd.read_csv(cache)
        have = set(alldf["symbol"].unique())
        if set(symbols) <= have:
            print(f"[data] using CSV cache ({len(have)} symbols)")
            return {s: g.reset_index(drop=True) for s, g in alldf[alldf["symbol"].isin(symbols)].groupby("symbol")}
    cli = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=days)
    frames = {}
    for i, s in enumerate(symbols):
        try:
            req = StockBarsRequest(symbol_or_symbols=[s], timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                   start=start, end=end, feed=DataFeed.IEX)
            df = cli.get_stock_bars(req).df
            if df is None or df.empty:
                continue
            df = df.reset_index()
            df = df.rename(columns={c: c.lower() for c in df.columns})
            if "symbol" not in df.columns:
                df["symbol"] = s
            frames[s] = df
            print(f"[data] {i+1}/{len(symbols)} {s}: {len(df)} bars")
        except Exception as exc:
            print(f"[data] {s} failed: {exc}")
        time.sleep(0.1)
    if frames:
        pd.concat(frames.values(), ignore_index=True).to_csv(cache, index=False)
    return frames


def rth_days(df):
    """Yield (date_et, day_df) for regular-hours bars, chronological."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["et"] = d["timestamp"].dt.tz_convert(ET)
    d = d[(d["et"].dt.time >= pd.Timestamp("09:30").time()) & (d["et"].dt.time < pd.Timestamp("16:00").time())]
    d = d.sort_values("et")
    for date, g in d.groupby(d["et"].dt.date, sort=True):
        if len(g) >= 6:
            yield date, g.reset_index(drop=True)


def simulate_day(day_df, variant, slippage_bps, rr, cfg=None):
    """Return a trade dict or None for one symbol-day. No look-ahead.
    rr = reward:risk multiple (reward = rr x risk); 2.0 == risk:reward 1:2.
    cfg (cost-aware redesign): {stop_model: 'open'|'pct', stop_pct, min_green_pct}."""
    cfg = cfg or {"stop_model": "open", "stop_pct": 0.006, "min_green_pct": 0.0}
    o = float(day_df["open"].iloc[0])  # today's session open
    if o <= 0:
        return None
    closes = day_df["close"].astype(float).values
    opens = day_df["open"].astype(float).values
    highs = day_df["high"].astype(float).values
    lows = day_df["low"].astype(float).values
    n = len(day_df)
    min_green = cfg["min_green_pct"]
    # entry: first bar at least min_green% above open; enter NEXT bar open (no look-ahead)
    entry_i = None
    for i in range(n - 1):
        if closes[i] > o and (closes[i] - o) / o >= min_green:
            if variant in ("B", "C"):
                sub = day_df.iloc[: i + 1]
                if not mtf_gate.ltf_momentum_bullish(sub, ret_lookback=min(6, max(1, i))):
                    continue
            entry_i = i + 1
            break
    if entry_i is None or entry_i >= n:
        return None
    entry = float(opens[entry_i])
    if entry <= o:
        return None
    if cfg["stop_model"] == "pct":
        risk = cfg["stop_pct"] * entry        # wider fixed stop -> less whipsaw
        stop = entry - risk
        target = entry + rr * risk
    else:
        risk = max(entry - o, MIN_RISK_PCT * entry)
        stop = o
        target = entry + rr * risk
    exit_price, reason = None, None
    optimistic = os.getenv("BT_OPTIMISTIC") == "1"  # target-before-stop upper bound
    for j in range(entry_i, n):
        hit_stop = lows[j] <= stop
        hit_target = highs[j] >= target
        if hit_stop and hit_target:
            exit_price, reason = (target, "target") if optimistic else (stop, "stop")
            break
        if hit_stop:
            exit_price, reason = stop, "stop"
            break
        if hit_target:
            exit_price, reason = target, "target"
            break
        if variant == "C" and float(closes[j]) > entry:  # reversal profit-exit
            if mtf_gate.momentum_lost(day_df.iloc[: j + 1]):
                exit_price, reason = float(closes[j]), "reversal"
                break
    if exit_price is None:
        exit_price, reason = float(closes[-1]), "eod"
    gross = (exit_price - entry) / entry
    net = gross - 2 * (slippage_bps / 10000.0)  # round-trip slippage
    return {"entry": entry, "exit": exit_price, "ret": net, "reason": reason}


def liquidity_rank(frames, top_n):
    """Keep the top_n symbols by avg daily dollar volume (tight-spread proxy)."""
    if not top_n or top_n <= 0:
        return frames
    dv = {}
    for s, df in frames.items():
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        ndays = max(d["timestamp"].dt.tz_convert(ET).dt.date.nunique(), 1)
        dv[s] = float((d["close"].astype(float) * d["volume"].astype(float)).sum()) / ndays
    keep = sorted(dv, key=dv.get, reverse=True)[:top_n]
    print(f"[liq] top-{top_n} by $vol: {keep}")
    return {s: frames[s] for s in keep}


def run_variant(frames, variant, slippage_bps, max_positions, rr, cfg):
    rows = []
    for sym, df in frames.items():
        for date, day_df in rth_days(df):
            tr = simulate_day(day_df, variant, slippage_bps, rr, cfg)
            if tr:
                tr.update({"symbol": sym, "date": date})
                rows.append(tr)
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)
    trades = pd.DataFrame(rows)
    # portfolio: equal-weight, cap max_positions per day
    daily = {}
    for date, g in trades.groupby("date"):
        taken = g.head(max_positions)
        daily[date] = float(taken["ret"].mean())
    day_ret = pd.Series(daily).sort_index()
    return trades, day_ret


def metrics(trades, day_ret, label):
    if trades.empty:
        return {"label": label, "trades": 0}
    wins = trades[trades["ret"] > 0]["ret"]
    losses = trades[trades["ret"] <= 0]["ret"]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    eq = (1 + day_ret).cumprod()
    n_days = len(day_ret)
    years = max(n_days / 252.0, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1 if len(eq) else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    sharpe = (day_ret.mean() / day_ret.std() * np.sqrt(252)) if day_ret.std() > 0 else 0.0
    downside = day_ret[day_ret < 0].std()
    sortino = (day_ret.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0
    return {
        "label": label, "trades": len(trades), "win%": round(100 * len(wins) / len(trades), 1),
        "avg_ret_bps": round(1e4 * trades["ret"].mean(), 1), "PF": round(pf, 2),
        "net_Sharpe": round(sharpe, 2), "Sortino": round(sortino, 2),
        "CAGR%": round(100 * cagr, 1), "maxDD%": round(100 * dd, 1),
        "trade_days": n_days, "avg_trades/day": round(len(trades) / max(n_days, 1), 1),
        "reasons": trades["reason"].value_counts().to_dict(),
    }


def split_oos(trades, day_ret, frac=0.65):
    dates = sorted(day_ret.index)
    if len(dates) < 10:
        return None
    cut = dates[int(len(dates) * frac)]
    is_tr = trades[trades["date"] <= cut]; oos_tr = trades[trades["date"] > cut]
    is_dr = day_ret[day_ret.index <= cut]; oos_dr = day_ret[day_ret.index > cut]
    return cut, (is_tr, is_dr), (oos_tr, oos_dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--rr", type=float, default=2.0, help="reward:risk multiple (reward = rr x risk); 2.0 == risk:reward 1:2")
    ap.add_argument("--stop-model", choices=["open", "pct"], default="open")
    ap.add_argument("--stop-pct", type=float, default=0.006)
    ap.add_argument("--min-green-pct", type=float, default=0.0)
    ap.add_argument("--top-liquidity", type=int, default=0, help="keep top-N symbols by $vol (0=all)")
    args = ap.parse_args()

    cfg = {"stop_model": args.stop_model, "stop_pct": args.stop_pct, "min_green_pct": args.min_green_pct}
    symbols = trade.load_trading_universe()
    print(f"[bt] universe={len(symbols)} days={args.days} slippage={args.slippage_bps}bps/side maxpos={args.max_positions} rr={args.rr}")
    print(f"[bt] redesign cfg={cfg} top_liquidity={args.top_liquidity}")
    frames = fetch_5min(symbols, args.days)
    frames = liquidity_rank(frames, args.top_liquidity)
    print(f"[bt] trading {len(frames)} symbols")

    for variant, name in [("A", "core daily-green"), ("B", "+mtf 5m momentum"), ("C", "+reversal profit-exit")]:
        trades, day_ret = run_variant(frames, variant, args.slippage_bps, args.max_positions, args.rr, cfg)
        print("\n" + "=" * 78)
        print(f"VARIANT {variant}: {name}")
        if trades.empty:
            print("  no trades")
            continue
        full = metrics(trades, day_ret, "ALL")
        sp = split_oos(trades, day_ret)
        print(f"  ALL : {full}")
        if sp:
            cut, (is_tr, is_dr), (oos_tr, oos_dr) = sp
            print(f"  IS  (<= {cut}): {metrics(is_tr, is_dr, 'IS')}")
            print(f"  OOS ( > {cut}): {metrics(oos_tr, oos_dr, 'OOS')}")


if __name__ == "__main__":
    main()
