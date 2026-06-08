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
SLIP_INS = 0.0015   # 15bps each side on the insider sleeve (small/mid-cap realism)
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
        win_days = [t for t in tdates if entry < t <= exit_]
        for i, t in enumerate(win_days):
            r = rets.get(t)
            if r is None or np.isnan(r):
                continue
            cost = (SLIP_INS if i == 0 else 0.0) + (SLIP_INS if i == len(win_days) - 1 else 0.0)
            day_sum[t] += float(r) - cost; day_cnt[t] += 1

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


def met(c):
    rr = c.pct_change().dropna()
    yrs = (pd.Timestamp(c.index[-1]) - pd.Timestamp(c.index[0])).days / 365.25
    dd = (c / c.cummax() - 1).min(); dn = rr[rr < 0].std()
    cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / max(yrs, 0.1)) - 1
    return dict(final=c.iloc[-1], cagr=cagr, vol=rr.std() * np.sqrt(252),
                sharpe=(rr.mean() / rr.std() * np.sqrt(252) if rr.std() else 0),
                sortino=(rr.mean() / dn * np.sqrt(252) if dn else 0),
                dd=dd, calmar=(cagr / abs(dd) if dd else 0))


start = comp.index[40]
spy_curve = curve(comp["SPY"].loc[start:]); spy_m = met(spy_curve)
ins_curve = curve(comp["Insider"].loc[start:])
W_GLD = 0.15
W_INS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
TVOL = [0.08, 0.10, 0.12, 0.14, 0.16]


def blend_curve(wi, tv):
    ws = max(0.0, 1 - wi - W_GLD)
    raw = ws * comp["SPY"] + wi * comp["Insider"] + W_GLD * comp["GLD"]
    pv = raw.rolling(20).std().shift(1) * np.sqrt(252)
    lev = (tv / pv).clip(0, MAX_LEV).fillna(1.0)
    return curve((lev * raw).loc[start:])


grid = {(wi, tv): met(blend_curve(wi, tv)) for wi in W_INS for tv in TVOL}
beats = lambda g: g["cagr"] > spy_m["cagr"] and g["dd"] > spy_m["dd"]   # more return AND shallower DD
winners = [(wi, tv) for wi in W_INS for tv in TVOL if beats(grid[(wi, tv)])]
pool = winners or [(wi, tv) for wi in W_INS for tv in TVOL]
best_wi, best_tv = max(pool, key=lambda k: grid[k]["calmar"])
bm = grid[(best_wi, best_tv)]
best_curve = blend_curve(best_wi, best_tv)

# ── aggressive cash-only (no leverage): 60% insider / 40% SPY, regime ON vs OFF ──
def _reg_below200(sym):
    b = bic.load_bars(sym)
    if b is None or b.empty:
        return pd.Series(False, index=comp.index)
    c = b["close"].copy(); c.index = [d.date() if hasattr(d, "date") else d for d in b.index]
    c = c.reindex(comp.index).ffill()
    return (c < c.rolling(200).mean())
RISK_OFF = (sum(_reg_below200(x).astype(int) for x in ("SPY", "QQQ", "IWM")) >= 2).reindex(comp.index).fillna(False)
W_AGG = 0.60
agg_raw = W_AGG * comp["Insider"] + (1 - W_AGG) * comp["SPY"]      # weights sum to 1.0 -> cash-only, no margin
agg_off_curve = curve(agg_raw.loc[start:])
agg_on_curve = curve((agg_raw * (~RISK_OFF).astype(float)).loc[start:])   # flat (cash) on market risk-off
agg_off_m, agg_on_m = met(agg_off_curve), met(agg_on_curve)
print(f"AGG 60% insider cash-only, regime OFF: CAGR {agg_off_m['cagr']:+.1%}  DD {agg_off_m['dd']:.1%}  Sharpe {agg_off_m['sharpe']:.2f}")
print(f"AGG 60% insider cash-only, regime ON : CAGR {agg_on_m['cagr']:+.1%}  DD {agg_on_m['dd']:.1%}  Sharpe {agg_on_m['sharpe']:.2f}")

# equity chart: best blend vs SPY vs insider
fig = make_subplots(rows=2, cols=1, row_heights=[0.64, 0.36], vertical_spacing=0.08,
                    subplot_titles=(f"Best blend ({best_wi:.0%} insider, vol-target {best_tv:.0%}) vs SPY", "Drawdown %"))
for nm, c, col, wd in [("Aggressive 60% insider (regime OFF)", agg_off_curve, "#f85149", 2.0),
                       ("Aggressive 60% insider (regime ON)", agg_on_curve, "#3fb950", 2.4),
                       ("SPY", spy_curve, "#6e7681", 1.5), ("Insider standalone", ins_curve, "#a371f7", 1.1)]:
    fig.add_trace(go.Scatter(x=c.index, y=c, name=nm, line=dict(color=col, width=wd)), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=(c / c.cummax() - 1) * 100, showlegend=False,
                  line=dict(color=col, width=wd * 0.5)), row=2, col=1)
fig.update_yaxes(title_text="$", row=1, col=1)
fig.update_layout(template="plotly_dark", height=620, margin=dict(t=66, l=50, r=20, b=24),
                  title="Tuned insider blend", legend=dict(orientation="h", y=1.08),
                  paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
chart = fig.to_html(full_html=False, include_plotlyjs="cdn",
                    config={"responsive": True, "displayModeBar": False}, default_width="100%", default_height="540px")

# sweep heatmap: CAGR, with a marker on cells beating SPY on both
ztxt = [[(f"{grid[(wi, tv)]['cagr']:.0%} / {grid[(wi, tv)]['dd']:.0%}"
          + (" \u2713" if beats(grid[(wi, tv)]) else "")) for tv in TVOL] for wi in W_INS]
zc = [[grid[(wi, tv)]["calmar"] for tv in TVOL] for wi in W_INS]
hm = go.Figure(go.Heatmap(z=zc, x=[f"{t:.0%}" for t in TVOL], y=[f"{w:.0%}" for w in W_INS],
               text=ztxt, texttemplate="%{text}", colorscale="Greens", showscale=False))
hm.update_layout(template="plotly_dark", height=320, margin=dict(t=44, l=70, r=10, b=40),
                 title="Sweep: CAGR / MaxDD per cell (\u2713 = beats SPY on BOTH); color = Calmar",
                 xaxis_title="vol target", yaxis_title="insider weight",
                 paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
heat = hm.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False},
                  default_width="100%", default_height="320px")

STAR, CHK = "\u2605", "\u2713"
rows = ""
for (wi, tv) in sorted(grid, key=lambda k: -grid[k]["calmar"])[:8]:
    g = grid[(wi, tv)]
    star = STAR if (wi, tv) == (best_wi, best_tv) else ""
    chk = CHK if beats(g) else ""
    rows += (f"<tr><td style='font-weight:600'>{wi:.0%} insider / {tv:.0%} vol {star}</td>"
             f"<td>${g['final']:,.0f}</td><td>{g['cagr']:+.1%}</td><td>{g['dd']:.1%}</td>"
             f"<td>{g['sharpe']:.2f}</td><td><b>{g['calmar']:.2f}</b></td><td>{chk}</td></tr>")
spy_row = (f"<tr style='background:#21262d'><td style='font-weight:600'>SPY buy &amp; hold</td><td>${spy_m['final']:,.0f}</td>"
           f"<td>{spy_m['cagr']:+.1%}</td><td>{spy_m['dd']:.1%}</td><td>{spy_m['sharpe']:.2f}</td><td><b>{spy_m['calmar']:.2f}</b></td><td>\u2014</td></tr>")
tbl = ("<table><thead><tr><th>Mix</th><th>Final</th><th>CAGR</th><th>MaxDD</th><th>Sharpe</th><th>Calmar</th><th>Beats SPY</th></tr></thead>"
       f"<tbody>{rows}{spy_row}</tbody></table>")
html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Tuned insider blend</title>
<style>body{{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}}
h1{{font-size:17px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}}
th,td{{padding:7px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:#8b949e}}
.note{{color:#8b949e;font-size:12px;line-height:1.5;margin-top:10px}}.scroll{{overflow-x:auto}}
.pick{{background:#161b22;border:1px solid #2ea043;border-radius:10px;padding:10px 12px;margin:8px 0}}</style></head>
<body><h1>Tuned insider blend (slippage-adjusted) · $10,000 · {best_curve.index[0]} → {best_curve.index[-1]}</h1>
<div class=pick>\u2605 Best: <b>{best_wi:.0%} insider / {W_GLD:.0%} GLD / {1-best_wi-W_GLD:.0%} SPY, vol-target {best_tv:.0%}</b>
\u2014 ${bm['final']:,.0f}, CAGR {bm['cagr']:+.1%}, MaxDD {bm['dd']:.1%}, Sharpe {bm['sharpe']:.2f}
&nbsp;vs SPY {spy_m['cagr']:+.1%} / {spy_m['dd']:.1%} / {spy_m['sharpe']:.2f}
&nbsp;\u2014 {'beats SPY on return AND drawdown' if beats(bm) else 'closest to the goal'}</div>
<div class=scroll>{tbl}</div>{chart}{heat}
<div class=note>Insider sleeve slippage-adjusted (15bps/side). Blend = SPY + insider + 15% GLD, vol-targeted (lev \u2264 {MAX_LEV:.0f}x).
\u2713 = the mix beat SPY on BOTH CAGR and MaxDD. Insider stream still equal-weights all active signals (optimistic vs the
live $1,500/name copier), so treat magnitudes as an upper bound; the RANKING across mixes is the useful part.</div>
</body></html>"""
open("crossover_tearsheet.html","w").write(html)
print(f"SPY: CAGR {spy_m['cagr']:+.1%}  DD {spy_m['dd']:.1%}  Sharpe {spy_m['sharpe']:.2f}")
print(f"insider standalone (post-slip): CAGR {met(ins_curve)['cagr']:+.1%}  DD {met(ins_curve)['dd']:.1%}  Sharpe {met(ins_curve)['sharpe']:.2f}")
print(f"\nwinners (beat SPY on both): {[(f'{wi:.0%}',f'{tv:.0%}') for wi,tv in winners]}")
print(f"BEST: {best_wi:.0%} insider / {best_tv:.0%} vol -> CAGR {bm['cagr']:+.1%}  DD {bm['dd']:.1%}  Sharpe {bm['sharpe']:.2f}  Calmar {bm['calmar']:.2f}")
