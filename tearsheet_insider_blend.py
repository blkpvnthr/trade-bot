#!/usr/bin/env python3
"""
tearsheet_insider_blend.py — turn the validated insider-copier edge into a DAILY equity curve and
blend it with an SPY core + a small crisis-diversifier (GLD), vol-targeted to a fixed risk budget.

Insider sleeve: PIT Form-4 buy events (survivorship-clean) → copier rules (trailing-2d buy > $1M,
2-day filing lag, enter next open, hold 10 trading days, a held name can't re-signal) → an equal-weight
daily return stream across all active positions. SPY/GLD from cached + live bars. Vol-target the blend.
"""
import os
from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import backtest_insider_copier as bic   # reuse load_pit_buy_events / load_bars / trading-day helpers

MIN_VALUE, WINDOW, HOLD, LAG = 1_000_000.0, 2, 10, 2
START_CASH, TARGET_VOL, MAX_LEV = 10_000.0, 0.12, 2.0

print("Loading PIT insider events + bars …")
events = bic.load_pit_buy_events()
spy_bars = bic.load_bars("SPY")
cal = sorted({d.date() if hasattr(d, "date") else d for d in spy_bars.index})   # master trading calendar

# ── build the insider daily return stream (equal-weight across active positions) ──
day_sum, day_cnt = defaultdict(float), defaultdict(int)
n_sig = n_sym = 0
for sym, buys in events.items():
    bars = bic.load_bars(sym)
    if bars is None or bars.empty:
        continue
    tdates = sorted({d.date() if hasattr(d, "date") else d for d in bars.index})
    sigs = bic.generate_signals(buys, MIN_VALUE, WINDOW, HOLD, tdates, LAG)
    if not sigs:
        continue
    n_sym += 1
    close = bars["close"]
    close.index = [d.date() if hasattr(d, "date") else d for d in close.index]
    rets = close.pct_change()
    for d in sigs:
        entry = bic._next_trading_on_or_after(d + timedelta(days=LAG), tdates)
        if entry is None:
            continue
        exit_ = bic._shift_trading_days(entry, HOLD, tdates) or entry
        n_sig += 1
        for t in tdates:
            if entry < t <= exit_:                    # close-to-close returns over the hold window
                r = rets.get(t)
                if r is not None and not np.isnan(r):
                    day_sum[t] += float(r); day_cnt[t] += 1

ins_ret = pd.Series({d: day_sum[d] / day_cnt[d] for d in day_cnt if day_cnt[d] > 0}).sort_index()
ins_ret = ins_ret.reindex(cal).fillna(0.0)            # 0 (cash) on no-position days
print(f"  {n_sym} symbols, {n_sig} copy signals; insider stream {ins_ret.index[0]} → {ins_ret.index[-1]}")

# ── benchmark + diversifier daily returns ──
spy_close = spy_bars["close"].copy(); spy_close.index = cal
spy_ret = spy_close.pct_change().reindex(cal).fillna(0.0)
gld_bars = bic.load_bars("GLD")
if gld_bars is None or gld_bars.empty:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from datetime import datetime, timezone
    for l in open(".env"):
        s = l.strip()
        if "=" in s and not s.startswith("#"):
            k, v = s.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    _dc = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
    df = _dc.get_stock_bars(StockBarsRequest(symbol_or_symbols="GLD", timeframe=TimeFrame.Day,
            start=datetime(2019, 1, 1, tzinfo=timezone.utc), end=datetime.now(timezone.utc),
            feed=DataFeed.IEX)).df.reset_index()
    g = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["timestamp"]).dt.date)
    gld_ret = g.pct_change().reindex(cal).fillna(0.0)
else:
    gc = gld_bars["close"].copy(); gc.index = [d.date() if hasattr(d, "date") else d for d in gld_bars.index]
    gld_ret = gc.pct_change().reindex(cal).fillna(0.0)

# trim to the insider-active window (where the edge actually trades)
mask = ins_ret.index >= ins_ret[ins_ret != 0].index[0]
comp = pd.DataFrame({"SPY": spy_ret, "Insider": ins_ret, "GLD": gld_ret}).loc[mask].fillna(0.0)


def curve(r):
    return START_CASH * (1 + r).cumprod()


def m(c, trades_ret=None):
    rr = c.pct_change().dropna()
    yrs = (c.index[-1] - c.index[0]).days / 365.25 if hasattr(c.index[-1], "__sub__") else len(c) / 252
    try:
        yrs = (pd.Timestamp(c.index[-1]) - pd.Timestamp(c.index[0])).days / 365.25
    except Exception:
        yrs = len(c) / 252
    dd = (c / c.cummax() - 1).min()
    dn = rr[rr < 0].std()
    cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / max(yrs, 0.1)) - 1
    d = dict(final=c.iloc[-1], cagr=cagr, vol=rr.std() * np.sqrt(252),
             sharpe=(rr.mean() / rr.std() * np.sqrt(252) if rr.std() else 0),
             sortino=(rr.mean() / dn * np.sqrt(252) if dn else 0), dd=dd,
             calmar=(cagr / abs(dd) if dd else 0))
    if trades_ret is not None and len(trades_ret):
        w = sum(1 for t in trades_ret if t > 0)
        gw = sum(t for t in trades_ret if t > 0); gl = -sum(t for t in trades_ret if t <= 0)
        d["wr"] = w / len(trades_ret) * 100; d["pf"] = gw / gl if gl else float("inf")
    return d


# Blend: SPY core 55% + Insider 35% + GLD 10%, then vol-target the whole thing.
W = {"SPY": 0.55, "Insider": 0.35, "GLD": 0.10}
blend_raw = sum(W[k] * comp[k] for k in W)
pvol = blend_raw.rolling(20).std().shift(1) * np.sqrt(252)
lev = (TARGET_VOL / pvol).clip(0, MAX_LEV).fillna(1.0)
blend_r = lev * blend_raw

start = comp.index[max(40, 1)]
curves = {
    "Blend (vol-targeted)": curve(blend_r.loc[start:]),
    "Insider alpha (standalone)": curve(comp["Insider"].loc[start:]),
    "SPY buy & hold": curve(comp["SPY"].loc[start:]),
}
# per-signal returns for insider win%/PF (already aggregated daily, approximate trade stats from event mean)
M = {k: m(v) for k, v in curves.items()}
cor = comp.corr().round(2)

CLR = {"Blend (vol-targeted)": "#3fb950", "Insider alpha (standalone)": "#a371f7", "SPY buy & hold": "#6e7681"}
fig = make_subplots(rows=2, cols=1, row_heights=[0.64, 0.36], vertical_spacing=0.08,
                    subplot_titles=("Equity ($10k start)", "Drawdown %"))
for k, c in curves.items():
    fig.add_trace(go.Scatter(x=c.index, y=c, name=k, line=dict(color=CLR[k],
                  width=2.6 if k.startswith("Blend") else 1.6)), row=1, col=1)
    dd = (c / c.cummax() - 1) * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name=k, showlegend=False,
                  line=dict(color=CLR[k], width=1.4 if k.startswith("Blend") else 1.0)), row=2, col=1)
fig.update_yaxes(title_text="$", row=1, col=1)
fig.update_layout(template="plotly_dark", height=720, margin=dict(t=66, l=50, r=20, b=24),
                  title="Insider-alpha blend — SPY core + insider edge + GLD, vol-targeted",
                  legend=dict(orientation="h", y=1.07), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
chart = fig.to_html(full_html=False, include_plotlyjs="cdn",
                    config={"responsive": True, "displayModeBar": False},
                    default_width="100%", default_height="560px")
hm = go.Figure(go.Heatmap(z=cor.values, x=cor.columns, y=cor.index, zmin=-1, zmax=1, colorscale="RdBu",
               reversescale=True, text=cor.values, texttemplate="%{text}", showscale=False))
hm.update_layout(template="plotly_dark", height=260, margin=dict(t=40, l=60, r=10, b=24),
                 title="Component correlation", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
heat = hm.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False},
                  default_width="100%", default_height="260px")


def trow(k, mm, star=False):
    hi = " style='background:#15301c'" if star else ""
    mk = " ★" if star else ""
    return (f"<tr{hi}><td style='font-weight:600'>{k}{mk}</td><td>${mm['final']:,.0f}</td>"
            f"<td>{mm['cagr']:+.1%}</td><td>{mm['vol']:.0%}</td><td>{mm['dd']:.1%}</td>"
            f"<td><b>{mm['sharpe']:.2f}</b></td><td>{mm['sortino']:.2f}</td><td><b>{mm['calmar']:.2f}</b></td></tr>")


tbl = ("<table><thead><tr><th>Portfolio</th><th>Final</th><th>CAGR</th><th>Vol</th><th>MaxDD</th>"
       "<th>Sharpe</th><th>Sortino</th><th>Calmar</th></tr></thead><tbody>"
       + "".join(trow(k, M[k], k.startswith("Blend")) for k in curves) + "</tbody></table>")
b, sp = M["Blend (vol-targeted)"], M["SPY buy & hold"]
html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Insider blend</title>
<style>body{{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}}
h1{{font-size:17px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}}
th,td{{padding:7px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:#8b949e}}
.note{{color:#8b949e;font-size:12px;line-height:1.5;margin-top:10px}}.scroll{{overflow-x:auto}}
.pick{{background:#161b22;border:1px solid #2ea043;border-radius:10px;padding:10px 12px;margin:8px 0}}</style></head>
<body><h1>Insider-alpha blend · $10,000 · {list(curves.values())[0].index[0]} → {list(curves.values())[0].index[-1]}</h1>
<div class=pick>★ Blend: ${b['final']:,.0f}, CAGR {b['cagr']:+.1%}, Vol {b['vol']:.0%}, MaxDD {b['dd']:.1%}, <b>Sharpe {b['sharpe']:.2f}</b>
&nbsp;·&nbsp; SPY: ${sp['final']:,.0f}, {sp['cagr']:+.1%}, DD {sp['dd']:.1%}, Sharpe {sp['sharpe']:.2f}</div>
<div class=scroll>{tbl}</div>{chart}{heat}
<div class=note>Insider sleeve = PIT Form-4 buys >$1M (trailing 2d), 2-day filing lag, enter next open, hold 10d,
equal-weight across active positions, survivorship-clean ({n_sym} names, {n_sig} signals). Blend = 55% SPY +
35% insider + 10% GLD, vol-targeted to {TARGET_VOL:.0%} (lev ≤ {MAX_LEV:.0f}x). The insider stream's low SPY
correlation is what improves the blend. In-sample to the PIT window; not a forward guarantee.</div>
</body></html>"""
open("crossover_tearsheet.html", "w").write(html)
print("\nresults:")
for k in curves:
    mm = M[k]
    print(f"  {k:<28}: ${mm['final']:>8,.0f}  CAGR {mm['cagr']:+6.1%}  Vol {mm['vol']:.0%}  DD {mm['dd']:>6.1%}  Sharpe {mm['sharpe']:.2f}  Calmar {mm['calmar']:.2f}")
print("\ncorrelations:\n", cor.to_string())
