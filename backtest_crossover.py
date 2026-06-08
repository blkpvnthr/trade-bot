#!/usr/bin/env python3
"""
backtest_crossover.py — backtest the live crossover trend rules on $10,000.

Rules (exactly as deployed):
  BUY  when price crosses UP through the 200-SMA, OR crosses DOWN through the 9-EMA while > 200-SMA.
  EXIT when price drops below the 200-SMA (full sell).
Sizing mirrors the live framework: deploy a fraction of AVAILABLE cash (cash above the 45% buffer),
capped at 10% of equity per single name. Daily bars, ~7y, 5bps slippage/side. Benchmarked vs SPY.
"""
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd


def load_env():
    for l in open(".env"):
        s = l.strip()
        if "=" in s and not s.startswith("#"):
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

dc = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])

START_CASH = 10_000.0
CASH_BUFFER = 0.45
SINGLE_CAP = 0.10
BUY_FRAC = 0.20            # fraction of AVAILABLE cash deployed per signal
MIN_ORDER = 100.0
SLIP = 0.0005             # 5 bps per side
EMA_P, SMA_P = 9, 200
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)

watch = [s.strip().upper() for s in os.environ.get("BUFFETT_WATCHLIST", "").split(",") if s.strip()]
seed = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSM", "ADBE", "CRM",
        "COST", "UNH", "V", "MA", "HD", "JPM", "QCOM", "AMAT", "INTC", "MRVL"]
UNIVERSE = sorted(set(watch + seed))


def fetch(sym):
    try:
        df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                                start=START, end=END, feed=DataFeed.IEX)).df
        if df is None or df.empty:
            return None
        df = df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == sym]
        s = pd.Series(df["close"].astype(float).values,
                      index=pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize())
        return s[~s.index.duplicated()]
    except Exception:
        return None


print(f"Fetching {len(UNIVERSE)} symbols + SPY …")
data, ema, sma = {}, {}, {}
for sym in UNIVERSE:
    s = fetch(sym)
    if s is not None and len(s) >= SMA_P + 30:
        data[sym] = s
        ema[sym] = s.ewm(span=EMA_P, adjust=False).mean()
        sma[sym] = s.rolling(SMA_P).mean()
spy = fetch("SPY")
print(f"Usable symbols ({len(data)}): {', '.join(sorted(data))}\n")

# common trading calendar
all_dates = sorted(set().union(*[set(s.index) for s in data.values()]))
all_dates = [d for d in all_dates if d >= all_dates[0]]

cash = START_CASH
pos = {}                  # sym -> qty (fractional)
eq_curve, trades = [], []
entry_cost = {}           # sym -> avg cost basis $

for d in all_dates:
    px = {s: data[s].get(d) for s in data if d in data[s].index}
    px = {s: p for s, p in px.items() if p is not None and not np.isnan(p)}
    equity = cash + sum(pos.get(s, 0) * px.get(s, 0) for s in pos)
    # 1) EXITS — price below 200-SMA
    for s in list(pos):
        p = px.get(s)
        sm = sma[s].get(d)
        if p is None or sm is None or np.isnan(sm):
            continue
        if p < sm and pos[s] > 0:
            proceeds = pos[s] * p * (1 - SLIP)
            cost = entry_cost.get(s, pos[s] * p)
            trades.append({"sym": s, "pnl": proceeds - cost, "exit": d})
            cash += proceeds
            pos[s] = 0.0
            entry_cost[s] = 0.0
    pos = {s: q for s, q in pos.items() if q > 1e-9}
    equity = cash + sum(pos.get(s, 0) * px.get(s, 0) for s in pos)
    # 2) BUYS — crossover signals
    for s in data:
        p = px.get(s)
        if p is None:
            continue
        ser = data[s]
        try:
            i = ser.index.get_loc(d)
        except KeyError:
            continue
        if i < SMA_P:
            continue
        prev = ser.iloc[i - 1]
        em, sm = ema[s].get(d), sma[s].get(d)
        if em is None or sm is None or np.isnan(em) or np.isnan(sm):
            continue
        cross_up_sma = prev < sm <= p
        cross_dn_ema = prev > em >= p
        if not (cross_up_sma or (cross_dn_ema and p > sm)):
            continue
        avail = max(0.0, cash - CASH_BUFFER * equity)
        cur_val = pos.get(s, 0) * p
        room = max(0.0, SINGLE_CAP * equity - cur_val)
        order = min(BUY_FRAC * avail, room, avail)
        if order >= MIN_ORDER:
            qty = order / p
            cash -= order * (1 + SLIP)
            pos[s] = pos.get(s, 0) + qty
            entry_cost[s] = entry_cost.get(s, 0) + order
    eq_curve.append((d, cash + sum(pos.get(s, 0) * px.get(s, 0) for s in pos)))

# ── metrics ──
eq = pd.Series([v for _, v in eq_curve], index=[d for d, _ in eq_curve])
ret = eq.pct_change().dropna()
years = (eq.index[-1] - eq.index[0]).days / 365.25
final = eq.iloc[-1]
cagr = (final / START_CASH) ** (1 / years) - 1
sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() else 0
dn = ret[ret < 0].std()
sortino = ret.mean() / dn * np.sqrt(252) if dn else 0
dd = (eq / eq.cummax() - 1).min()
wins = [t for t in trades if t["pnl"] > 0]
wr = len(wins) / len(trades) * 100 if trades else 0
gw = sum(t["pnl"] for t in wins)
gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
pf = gw / gl if gl else float("inf")

# SPY benchmark
spy_final = spy_cagr = None
if spy is not None:
    sp = spy[(spy.index >= eq.index[0]) & (spy.index <= eq.index[-1])]
    spy_final = START_CASH * sp.iloc[-1] / sp.iloc[0]
    spy_cagr = (spy_final / START_CASH) ** (1 / years) - 1

print("=" * 56)
print(f"CROSSOVER STRATEGY BACKTEST  (${START_CASH:,.0f} start)")
print(f"Period      : {eq.index[0]:%Y-%m-%d} → {eq.index[-1]:%Y-%m-%d}  ({years:.1f}y)")
print("=" * 56)
print(f"Final equity: ${final:,.0f}   ({final/START_CASH-1:+.1%} total)")
print(f"CAGR        : {cagr:+.2%}")
print(f"Sharpe      : {sharpe:.2f}   Sortino: {sortino:.2f}")
print(f"Max drawdown: {dd:.1%}")
print(f"Round-trips : {len(trades)}   Win%: {wr:.0f}   PF: {pf:.2f}")
print(f"Avg trade   : ${np.mean([t['pnl'] for t in trades]):+,.0f}" if trades else "no trades")
print("-" * 56)
if spy_final:
    print(f"SPY buy&hold: ${spy_final:,.0f}   CAGR {spy_cagr:+.2%}   (benchmark)")
    print(f"vs SPY      : {'OUTPERFORM' if final>spy_final else 'UNDERPERFORM'} by ${abs(final-spy_final):,.0f}")
print("-" * 56)
print("Per-year return:")
for yr, grp in eq.groupby(eq.index.year):
    r = grp.iloc[-1] / grp.iloc[0] - 1
    print(f"  {yr}: {r:+.1%}   (end ${grp.iloc[-1]:,.0f})")
