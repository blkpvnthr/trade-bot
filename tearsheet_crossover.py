#!/usr/bin/env python3
"""
tearsheet_crossover.py — vectorbt backtest + interactive HTML tearsheet of the crossover strategy
on $10,000, with the live −1.5% drawdown kill switch modeled.

Rules: BUY on 200-SMA cross-up OR 9-EMA cross-down (while > 200-SMA); EXIT below 200-SMA.
Sizing: 20% of AVAILABLE cash (above the 45% buffer), capped 10%/name. 5bps slippage/side.
Kill switch: halt new buys whenever portfolio drawdown from its high-water mark exceeds 1.5%.

Output: crossover_tearsheet.html — equity curves (no-kill vs −1.5% kill vs SPY) with buy ▲ / sell ▼
markers, a drawdown panel, and a metrics table.
"""
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

START_CASH, CASH_BUFFER, SINGLE_CAP, BUY_FRAC = 10_000.0, 0.45, 0.10, 0.20
MIN_ORDER, SLIP, EMA_P, SMA_P, KILL = 100.0, 0.0005, 9, 200, 0.015
START, END = datetime(2019, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)

watch = [s.strip().upper() for s in os.environ.get("BUFFETT_WATCHLIST", "").split(",") if s.strip()]
seed = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSM", "ADBE", "CRM",
        "COST", "UNH", "V", "MA", "HD", "JPM", "QCOM", "AMAT", "INTC", "MRVL"]
EXCLUDE = {"TQQQ", "SOXL", "SQQQ", "SPXL", "SPXS", "UVXY", "SVXY",   # leveraged ETFs (vol decay)
           "QQQ", "SPY", "DIA", "IWM", "VOO", "VTI"}                # index ETFs
UNIVERSE = sorted(set(watch + seed) - EXCLUDE)


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
series = {}
for sym in UNIVERSE:
    s = fetch(sym)
    if s is not None and len(s) >= SMA_P + 30:
        series[sym] = s
spy = fetch("SPY")
closes = pd.DataFrame(series).sort_index()
closes = closes[closes.index >= closes.dropna(how="all").index[0]]
print(f"Usable symbols ({closes.shape[1]}): {', '.join(closes.columns)}\n")

ema = closes.ewm(span=EMA_P, adjust=False).mean()
sma = closes.rolling(SMA_P).mean()
prev = closes.shift(1)
entries = (((prev < sma) & (closes >= sma)) | ((prev > ema) & (closes <= ema) & (closes > sma))).fillna(False)
exits = (closes < sma).fillna(False)

# ── genuine vectorbt portfolio (base, no kill switch) — validation + native stats ──
print("Running vectorbt Portfolio.from_signals …")
pf = vbt.Portfolio.from_signals(closes, entries, exits, init_cash=START_CASH, cash_sharing=True,
                                group_by=True, size=BUY_FRAC, size_type="percent",
                                fees=SLIP, slippage=SLIP, freq="1D")
try:
    vbt_stats = pf.stats()
    print("\n=== vectorbt Portfolio.stats() (base, no kill switch) ===")
    print(vbt_stats)
except Exception as e:
    vbt_stats = None
    print("vbt stats unavailable:", e)


# ── sequential sim (identical engine) for no-kill + −1.5% kill, for comparison + buy/sell markers ──
def simulate(kill=None):
    cash, pos, peak = START_CASH, {}, START_CASH
    curve, orders, trades, cost = [], [], [], {}
    for d in closes.index:
        row, emr, smr, pvr = closes.loc[d], ema.loc[d], sma.loc[d], prev.loc[d]
        eq = cash + sum(pos.get(s, 0) * row[s] for s in pos if not np.isnan(row[s]))
        for s in list(pos):                                   # exits
            p, sm = row[s], smr[s]
            if np.isnan(p) or np.isnan(sm):
                continue
            if p < sm and pos[s] > 0:
                proceeds = pos[s] * p * (1 - SLIP)
                trades.append(proceeds - cost.get(s, proceeds))
                cash += proceeds; orders.append((d, "sell")); pos[s] = 0.0; cost[s] = 0.0
        pos = {s: q for s, q in pos.items() if q > 1e-9}
        eq = cash + sum(pos.get(s, 0) * row[s] for s in pos if not np.isnan(row[s]))
        peak = max(peak, eq)
        halted = kill is not None and eq < peak * (1 - kill)
        if not halted:
            for s in closes.columns:                          # buys
                p, em, sm, pv = row[s], emr[s], smr[s], pvr[s]
                if any(np.isnan(x) for x in (p, em, sm, pv)):
                    continue
                if not ((pv < sm <= p) or ((pv > em >= p) and p > sm)):
                    continue
                avail = max(0.0, cash - CASH_BUFFER * eq)
                room = max(0.0, SINGLE_CAP * eq - pos.get(s, 0) * p)
                order = min(BUY_FRAC * avail, room, avail)
                if order >= MIN_ORDER:
                    cash -= order * (1 + SLIP)
                    pos[s] = pos.get(s, 0) + order / p
                    cost[s] = cost.get(s, 0) + order
                    orders.append((d, "buy"))
        curve.append((d, cash + sum(pos.get(s, 0) * row[s] for s in pos if not np.isnan(row[s]))))
    return pd.Series(dict(curve)), orders, trades


base_curve, base_orders, base_trades = simulate(kill=None)
kill_curve, kill_orders, kill_trades = simulate(kill=KILL)
spy_curve = (START_CASH * spy / spy.iloc[0]).reindex(base_curve.index).ffill() if spy is not None else None


def metrics(curve, trades):
    ret = curve.pct_change().dropna()
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / START_CASH) ** (1 / yrs) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() else 0
    dn = ret[ret < 0].std()
    sortino = ret.mean() / dn * np.sqrt(252) if dn else 0
    dd = (curve / curve.cummax() - 1).min()
    wr = (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0
    gw = sum(t for t in trades if t > 0); gl = -sum(t for t in trades if t <= 0)
    pf_ = gw / gl if gl else float("inf")
    return dict(final=curve.iloc[-1], tot=curve.iloc[-1] / START_CASH - 1, cagr=cagr,
                sharpe=sharpe, sortino=sortino, dd=dd, n=len(trades), wr=wr, pf=pf_)


mb, mk = metrics(base_curve, base_trades), metrics(kill_curve, kill_trades)
ms = metrics(spy_curve, []) if spy_curve is not None else None

# ── plotly tearsheet ──
fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32], shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=("Equity ($10k start) — buys ▲ / sells ▼", "Drawdown %"))
fig.add_trace(go.Scatter(x=base_curve.index, y=base_curve, name="No kill switch",
                         line=dict(color="#3fb950", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=kill_curve.index, y=kill_curve, name="−1.5% kill switch",
                         line=dict(color="#58a6ff", width=2)), row=1, col=1)
if spy_curve is not None:
    fig.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve, name="SPY buy&hold",
                             line=dict(color="#8b949e", width=1.4, dash="dot")), row=1, col=1)
bd = [d for d, s in base_orders if s == "buy"]
sd = [d for d, s in base_orders if s == "sell"]
fig.add_trace(go.Scatter(x=bd, y=base_curve.reindex(bd), mode="markers", name="buy",
                         marker=dict(symbol="triangle-up", color="#2ea043", size=7, line=dict(width=0))), row=1, col=1)
fig.add_trace(go.Scatter(x=sd, y=base_curve.reindex(sd), mode="markers", name="sell",
                         marker=dict(symbol="triangle-down", color="#f85149", size=7, line=dict(width=0))), row=1, col=1)
ddb = (base_curve / base_curve.cummax() - 1) * 100
ddk = (kill_curve / kill_curve.cummax() - 1) * 100
fig.add_trace(go.Scatter(x=ddb.index, y=ddb, name="DD no-kill", fill="tozeroy",
                         line=dict(color="#3fb950", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=ddk.index, y=ddk, name="DD −1.5% kill",
                         line=dict(color="#58a6ff", width=1)), row=2, col=1)
fig.update_layout(template="plotly_dark", height=760, margin=dict(t=70, l=50, r=20, b=30),
                  title="Crossover Strategy — vectorbt tearsheet", legend=dict(orientation="h", y=1.06),
                  paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
fig.update_yaxes(title_text="$", row=1, col=1)
chart = fig.to_html(full_html=False, include_plotlyjs="cdn")


def row(name, m, color):
    pf_ = "∞" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (f"<tr><td style='color:{color};font-weight:600'>{name}</td>"
            f"<td>${m['final']:,.0f}</td><td>{m['tot']:+.0%}</td><td>{m['cagr']:+.1%}</td>"
            f"<td>{m['sharpe']:.2f}</td><td>{m['sortino']:.2f}</td><td>{m['dd']:.1%}</td>"
            f"<td>{m['n']}</td><td>{m['wr']:.0f}%</td><td>{pf_}</td></tr>")


tbl = ("<table><thead><tr><th>Variant</th><th>Final</th><th>Total</th><th>CAGR</th><th>Sharpe</th>"
       "<th>Sortino</th><th>MaxDD</th><th>Trades</th><th>Win%</th><th>PF</th></tr></thead><tbody>"
       + row("No kill switch", mb, "#3fb950") + row("−1.5% kill switch", mk, "#58a6ff")
       + (row("SPY buy&hold", ms, "#8b949e") if ms else "") + "</tbody></table>")

html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Crossover Tearsheet</title>
<style>body{{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}}
h1{{font-size:18px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}}
th,td{{padding:7px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:#8b949e}}
.note{{color:#8b949e;font-size:12px;line-height:1.5;margin-top:10px}}.scroll{{overflow-x:auto}}</style></head>
<body><h1>Crossover Strategy — $10,000 · {base_curve.index[0]:%Y-%m-%d} → {base_curve.index[-1]:%Y-%m-%d}</h1>
<div class=scroll>{tbl}</div>{chart}
<div class=note>Rules: buy on 200-SMA cross-up OR 9-EMA cross-down (while &gt; 200-SMA); exit below 200-SMA.
Sizing 20% of available cash, 45% cash buffer, 10%/name cap, 5bps slippage/side. Kill switch halts new
buys when drawdown from the high-water mark exceeds 1.5%. Universe = live watchlist (selection/survivorship
bias) incl. leveraged ETFs; mostly a 2020–2026 bull regime. Backtest, not a forward guarantee.</div>
</body></html>"""
open("crossover_tearsheet.html", "w").write(html)
print(f"\nNo-kill : ${mb['final']:,.0f} ({mb['cagr']:+.1%} CAGR, DD {mb['dd']:.1%}, {mb['n']} trades)")
print(f"−1.5%kill: ${mk['final']:,.0f} ({mk['cagr']:+.1%} CAGR, DD {mk['dd']:.1%}, {mk['n']} trades)")
if ms:
    print(f"SPY     : ${ms['final']:,.0f} ({ms['cagr']:+.1%} CAGR)")
print("wrote crossover_tearsheet.html")
