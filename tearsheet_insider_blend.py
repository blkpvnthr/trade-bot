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

# ── Strategy page: ONLY the current deployed strategy ──
cur_curve, cur_m = agg_on_curve, agg_on_m     # aggressive insider, regime-ON = the live config

fig = make_subplots(rows=2, cols=1, row_heights=[0.66, 0.34], vertical_spacing=0.08,
                    subplot_titles=("Equity ($10k start)", "Drawdown %"))
for nm, c, col, wd in [("Current strategy", cur_curve, "#3fb950", 2.8), ("SPY (benchmark)", spy_curve, "#6e7681", 1.5)]:
    fig.add_trace(go.Scatter(x=c.index, y=c, name=nm, line=dict(color=col, width=wd)), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=(c / c.cummax() - 1) * 100, showlegend=False,
                  line=dict(color=col, width=wd * 0.5)), row=2, col=1)
fig.update_yaxes(title_text="$", row=1, col=1)
fig.update_layout(template="plotly_dark", height=600, margin=dict(t=56, l=50, r=20, b=24),
                  title="Current strategy - aggressive insider copier (cash-only, regime-on)",
                  legend=dict(orientation="h", y=1.08), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
chart = fig.to_html(full_html=False, include_plotlyjs="cdn",
                    config={"responsive": True, "displayModeBar": False}, default_width="100%", default_height="540px")


def card(k, v, sub, cls=""):
    return f"<div class=card><div class=k>{k}</div><div class='v {cls}'>{v}</div><div class=sub>{sub}</div></div>"


cards = ("<div class=cards>"
         + card("CAGR", f"{cur_m['cagr']:+.1%}", f"SPY {spy_m['cagr']:+.1%}", "pos" if cur_m['cagr'] >= 0 else "neg")
         + card("Max drawdown", f"{cur_m['dd']:.1%}", f"SPY {spy_m['dd']:.1%}", "neg")
         + card("Sharpe", f"{cur_m['sharpe']:.2f}", f"SPY {spy_m['sharpe']:.2f}")
         + card("Sortino", f"{cur_m['sortino']:.2f}", f"SPY {spy_m['sortino']:.2f}")
         + card("Calmar", f"{cur_m['calmar']:.2f}", f"SPY {spy_m['calmar']:.2f}")
         + card("Volatility", f"{cur_m['vol']:.0%}", f"SPY {spy_m['vol']:.0%}")
         + "</div>")

CSS = ("body{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}"
       "h1{font-size:17px;margin:2px 0 4px}.sub2{color:#8b949e;font-size:12px;margin-bottom:10px}"
       ".cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}"
       ".card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px}"
       ".card .k{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.04em}"
       ".card .v{font-size:21px;font-weight:700;margin-top:3px}.card .sub{color:#6e7681;font-size:11px;margin-top:2px}"
       ".pos{color:#3fb950}.neg{color:#f85149}"
       ".cfg{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;margin-top:10px;font-size:12px;color:#9da7b1;line-height:1.6}"
       ".cfg b{color:#e6edf3}.note{color:#8b949e;font-size:11px;line-height:1.5;margin-top:8px}")

html = ("<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'><title>Current strategy</title>"
        f"<style>{CSS}</style></head><body><h1>Current strategy</h1>"
        f"<div class=sub2>Aggressive insider copier &middot; cash-only &middot; regime-on &middot; {cur_curve.index[0]} &rarr; {cur_curve.index[-1]}</div>"
        f"{cards}{chart}"
        "<div class=cfg>LIVE CONFIG: insider copier (primary engine) <b>12 names &times; $5,000</b>, <b>~90% deployed</b> "
        "(10% cash buffer), regime protection <b>ON</b>, per-position stop 15%, equity floor &minus;45% &rarr; ruin kill &minus;50%, "
        "<b>cash-only (no margin &mdash; can't hit $0)</b>. Signals = SEC Form-4 open-market buys &gt;$1M, &le;2 days old, &ge;2 distinct insiders.</div>"
        "<div class=note>Backtest proxy: 60% insider / 40% SPY, regime-gated, cash-only, 15bps/side slippage, in-sample. "
        "The live copier holds discrete names so it captures LESS than the idealized magnitudes &mdash; judge it on "
        "'beats SPY with shallower drawdowns', not the literal CAGR.</div></body></html>")
open("crossover_tearsheet.html", "w").write(html)
print(f"Current strategy: CAGR {cur_m['cagr']:+.1%}  DD {cur_m['dd']:.1%}  Sharpe {cur_m['sharpe']:.2f}  Calmar {cur_m['calmar']:.2f}")
print(f"SPY benchmark   : CAGR {spy_m['cagr']:+.1%}  DD {spy_m['dd']:.1%}  Sharpe {spy_m['sharpe']:.2f}")
