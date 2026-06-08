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


spy_curve = (START_CASH * spy / spy.iloc[0]).reindex(closes.index).ffill() if spy is not None else None


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
    calmar = cagr / abs(dd) if dd else 0
    return dict(final=curve.iloc[-1], tot=curve.iloc[-1] / START_CASH - 1, cagr=cagr, sharpe=sharpe,
                sortino=sortino, dd=dd, n=len(trades), wr=wr, pf=pf_, calmar=calmar)


# ── sweep the kill switch ──
KILLS = [("none", None), ("1.5%", 0.015), ("3%", 0.03), ("6%", 0.06),
         ("10%", 0.10), ("15%", 0.15), ("20%", 0.20), ("30%", 0.30)]
res = {}
for label, k in KILLS:
    c, o, t = simulate(kill=k)
    res[label] = (c, metrics(c, t))
ms = metrics(spy_curve, []) if spy_curve is not None else None
best = max(KILLS, key=lambda lk: res[lk[0]][1]["calmar"])[0]   # best risk-adjusted return (Calmar)

PALETTE = ["#f85149", "#db6d28", "#d29922", "#3fb950", "#2ea043", "#58a6ff", "#a371f7", "#db61a2"]
fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.1,
                    subplot_titles=("Equity by kill-switch level ($10k start)",
                                    "Risk / return frontier — CAGR vs Max Drawdown (pick the knee)"))
for i, (label, k) in enumerate(KILLS):
    c = res[label][0]
    fig.add_trace(go.Scatter(x=c.index, y=c, name=f"kill {label}" + (" \u2605" if label == best else ""),
                             line=dict(color=PALETTE[i], width=3 if label == best else 1.4)), row=1, col=1)
if spy_curve is not None:
    fig.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve, name="SPY",
                             line=dict(color="#8b949e", width=1.4, dash="dot")), row=1, col=1)
for i, (label, k) in enumerate(KILLS):
    m = res[label][1]
    fig.add_trace(go.Scatter(x=[abs(m["dd"]) * 100], y=[m["cagr"] * 100], mode="markers+text",
                             text=[label], textposition="top center", showlegend=False,
                             marker=dict(color=PALETTE[i], size=15)), row=2, col=1)
if ms:
    fig.add_trace(go.Scatter(x=[abs(ms["dd"]) * 100], y=[ms["cagr"] * 100], mode="markers+text",
                             text=["SPY"], textposition="bottom center", showlegend=False,
                             marker=dict(color="#8b949e", size=12, symbol="diamond")), row=2, col=1)
fig.update_yaxes(title_text="$", row=1, col=1)
fig.update_xaxes(title_text="Max Drawdown %", row=2, col=1)
fig.update_yaxes(title_text="CAGR %", row=2, col=1)
fig.update_layout(template="plotly_dark", height=830, margin=dict(t=70, l=50, r=20, b=30),
                  title="Crossover Strategy — kill-switch sweep", legend=dict(orientation="h", y=1.05),
                  paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
chart = fig.to_html(full_html=False, include_plotlyjs="cdn")


def trow(label, m, star=False):
    pf_ = "\u221e" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    nm = label + (" \u2605" if star else "")
    hi = " style='background:#15301c'" if star else ""
    return (f"<tr{hi}><td style='font-weight:600'>{nm}</td><td>${m['final']:,.0f}</td><td>{m['cagr']:+.1%}</td>"
            f"<td>{m['dd']:.1%}</td><td><b>{m['calmar']:.2f}</b></td><td>{m['sharpe']:.2f}</td>"
            f"<td>{m['sortino']:.2f}</td><td>{m['n']}</td><td>{m['wr']:.0f}%</td><td>{pf_}</td></tr>")


tbl = ("<table><thead><tr><th>Kill switch</th><th>Final</th><th>CAGR</th><th>MaxDD</th><th>Calmar</th>"
       "<th>Sharpe</th><th>Sortino</th><th>Trades</th><th>Win%</th><th>PF</th></tr></thead><tbody>"
       + "".join(trow(label, res[label][1], label == best) for label, k in KILLS)
       + (trow("SPY b&h", ms) if ms else "") + "</tbody></table>")

bm = res[best][1]
html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Kill-switch sweep</title>
<style>body{{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}}
h1{{font-size:18px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}}
th,td{{padding:7px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:#8b949e}}
.note{{color:#8b949e;font-size:12px;line-height:1.5;margin-top:10px}}.scroll{{overflow-x:auto}}
.pick{{background:#161b22;border:1px solid #2ea043;border-radius:10px;padding:10px 12px;margin:8px 0}}</style></head>
<body><h1>Kill-switch sweep — $10,000 \u00b7 {res['none'][0].index[0]:%Y-%m-%d} \u2192 {res['none'][0].index[-1]:%Y-%m-%d}</h1>
<div class=pick>\u2605 Best risk-adjusted (Calmar = CAGR/|MaxDD|): <b>kill {best}</b> \u2014
${bm['final']:,.0f}, CAGR {bm['cagr']:+.1%}, MaxDD {bm['dd']:.1%}, Calmar {bm['calmar']:.2f}</div>
<div class=scroll>{tbl}</div>{chart}
<div class=note>Crossover strategy, clean universe (no leveraged ETFs/indices). Kill switch halts new buys
when drawdown from the high-water mark exceeds the level (existing positions still exit below the 200-SMA,
so realized drawdown can exceed the threshold). Sizing 20% of available cash, 45% buffer, 10%/name, 5bps/side.
Selection/survivorship bias + mostly-bull 2020\u20132026 regime \u2014 backtest, not a forward guarantee.</div>
</body></html>"""
open("crossover_tearsheet.html", "w").write(html)
print("sweep results:")
for label, k in KILLS:
    m = res[label][1]
    print(f"  kill {label:>5}: ${m['final']:>8,.0f}  CAGR {m['cagr']:+6.1%}  DD {m['dd']:>6.1%}  Calmar {m['calmar']:.2f}  Sharpe {m['sharpe']:.2f}  ({m['n']} tr)")
if ms: print(f"  SPY        : ${ms['final']:>8,.0f}  CAGR {ms['cagr']:+6.1%}  DD {ms['dd']:>6.1%}")
print(f"\nBEST by Calmar: kill {best}")
