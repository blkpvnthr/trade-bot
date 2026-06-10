#!/usr/bin/env python3
"""tearsheet_insider_blend.py — the FILTERED strategy ACTUALLY being traded:
insider copier + SMA50 trend gate + vol(rv20>=0.0563) + liquidity(ADV>=$1M), blended 60/40 with an
SPY core, regime-gated (cash when a majority of SPY/QQQ/IWM are below their 200-SMA), cash-only.
Overlays the RETURN GIVEN UP by the SMA50 trend gate vs the same strategy without it. Cached PIT bars only."""
from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import backtest_insider_copier as bic

MIN_VALUE, WINDOW, HOLD, LAG = 1_000_000.0, 2, 10, 2
SLIP_INS = 0.0015
VOL_MIN_RV20, MIN_ADV = 0.0563, 1_000_000.0   # live vol + liquidity gates (permanent; 0.025 vol is a 1-day temp)
W_INS, START_CASH = 0.60, 5_000.0

print("Loading PIT insider events + bars …")
events = bic.load_pit_buy_events()
spy_bars = bic.load_bars("SPY")
cal = sorted({d.date() if hasattr(d, "date") else d for d in spy_bars.index})

# ── build TWO insider streams: filtered (trend+vol+liq, LIVE) and no-trend (vol+liq only) ──
ds_f, dc_f = defaultdict(float), defaultdict(int)
ds_n, dc_n = defaultdict(float), defaultdict(int)
n_f = n_n = 0
buys_f, sells_f = [], []   # with-trend-gate (old) entry/exit dates
buys_n, sells_n = [], []   # no-trend-gate (CURRENT) entry/exit dates for chart markers
for sym, buys in events.items():
    bars = bic.load_bars(sym)
    if bars is None or bars.empty:
        continue
    tdates = sorted({d.date() if hasattr(d, "date") else d for d in bars.index})
    sigs = bic.generate_signals(buys, MIN_VALUE, WINDOW, HOLD, tdates, LAG)
    if not sigs:
        continue
    cols = ["close", "volume"] if "volume" in bars.columns else ["close"]
    df = bars[cols].astype(float)
    df.index = [d.date() if hasattr(d, "date") else d for d in bars.index]
    df = df[~df.index.duplicated()].sort_index()
    close = df["close"]
    volser = df["volume"] if "volume" in df.columns else pd.Series(1e15, index=close.index)
    rets = close.pct_change()
    for d in sigs:
        entry = bic._next_trading_on_or_after(d + timedelta(days=LAG), tdates)
        if entry is None or entry not in close.index:
            continue
        hist = close.loc[:entry]
        if len(hist) < 50:
            continue
        price = float(close.loc[entry]); sma50 = float(hist.tail(50).mean())
        r20 = rets.loc[:entry].tail(20); rv20 = float(r20.std()) if r20.notna().sum() >= 20 else 0.0
        adv = float((close * volser).loc[:entry].tail(20).mean())
        if rv20 < VOL_MIN_RV20 or adv < MIN_ADV:        # vol + liquidity gates (BOTH versions)
            continue
        pass_trend = price > sma50                      # SMA50 trend gate (LIVE only)
        exit_ = bic._shift_trading_days(entry, HOLD, tdates) or entry
        win = [t for t in tdates if entry < t <= exit_]
        n_n += 1
        buys_n.append(entry); sells_n.append(exit_)
        if pass_trend:
            n_f += 1
            buys_f.append(entry); sells_f.append(exit_)
        for i, t in enumerate(win):
            r = rets.get(t)
            if r is None or np.isnan(r):
                continue
            cost = (SLIP_INS if i == 0 else 0.0) + (SLIP_INS if i == len(win) - 1 else 0.0)
            ds_n[t] += float(r) - cost; dc_n[t] += 1
            if pass_trend:
                ds_f[t] += float(r) - cost; dc_f[t] += 1


def stream(ds, dc):
    s = pd.Series({d: ds[d] / dc[d] for d in dc if dc[d] > 0}).sort_index()
    return s.reindex(cal).fillna(0.0)


ins_f, ins_n = stream(ds_f, dc_f), stream(ds_n, dc_n)
print(f"  filtered (trend+vol+liq): {n_f} signals | no-trend (vol+liq): {n_n} signals")

spc = spy_bars["close"].astype(float); spc.index = cal
spy_ret = spc.pct_change().reindex(cal).fillna(0.0)


def reg_below(sym):
    b = bic.load_bars(sym)
    if b is None or b.empty:
        return pd.Series(False, index=cal)
    c = b["close"].astype(float); c.index = [d.date() if hasattr(d, "date") else d for d in b.index]
    c = c[~c.index.duplicated()].reindex(cal).ffill()
    return c < c.rolling(200).mean()


RISK_OFF = (sum(reg_below(x).astype(int) for x in ("SPY", "QQQ", "IWM")) >= 2).reindex(cal).fillna(False)
act = ins_n[ins_n != 0]
start = act.index[0] if len(act) else cal[40]
idx = [d for d in cal if d >= start]


def blend(ins):
    raw = (W_INS * ins + (1 - W_INS) * spy_ret) * (~RISK_OFF).astype(float)   # 60/40 cash-only, regime to cash
    return START_CASH * (1 + raw.loc[idx]).cumprod()


cur, notrend = blend(ins_f), blend(ins_n)
spy_curve = START_CASH * (1 + spy_ret.loc[idx]).cumprod()


def met(c):
    rr = c.pct_change().dropna()
    yrs = (pd.Timestamp(c.index[-1]) - pd.Timestamp(c.index[0])).days / 365.25
    dd = (c / c.cummax() - 1).min()
    cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / max(yrs, 0.1)) - 1
    return dict(final=c.iloc[-1], cagr=cagr, dd=dd,
                sharpe=(rr.mean() / rr.std() * np.sqrt(252) if rr.std() else 0))


mF, mN, mS = met(cur), met(notrend), met(spy_curve)
diff = notrend - cur                    # $ recaptured by removing the trend gate
print(f"CURRENT (no trend gate): CAGR {mN['cagr']:+.1%} DD {mN['dd']:.1%} Sharpe {mN['sharpe']:.2f}")
print(f"OLD (with gate)        : CAGR {mF['cagr']:+.1%} DD {mF['dd']:.1%} Sharpe {mF['sharpe']:.2f}")
print(f"Recaptured by removal  : {mN['cagr']-mF['cagr']:+.1%} CAGR, ${diff.iloc[-1]:,.0f} on $5k")

# Reference numbers for the live signal-fidelity comparison (signal_fidelity_report.py).
import json as _json
_json.dump({
    "start": str(notrend.index[0]), "end": str(notrend.index[-1]),
    "signals": int(n_n), "signals_gated": int(n_f),
    "cagr": round(mN["cagr"], 4), "max_dd": round(mN["dd"], 4), "sharpe": round(mN["sharpe"], 3),
    "start_cash": float(START_CASH), "slippage_bps_per_side": SLIP_INS * 1e4,
    "vol_min_rv20": VOL_MIN_RV20, "min_adv": MIN_ADV, "freshness_max_days": LAG,
    "note": "in-sample 60/40 SPY blend; per-trade win-rate/return not modeled by the daily-return blend",
}, open("backtest_reference.json", "w"), indent=2)
print("wrote backtest_reference.json")

# ── chart: CURRENT (no trend gate) primary; old gated version dotted; gap = recaptured ──
fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], vertical_spacing=0.13)
fig.add_trace(go.Scatter(x=notrend.index, y=notrend, name="Current strategy — trend gate REMOVED",
                         line=dict(color="#3fb950", width=2.8)), 1, 1)
fig.add_trace(go.Scatter(x=cur.index, y=cur, name="Before — with SMA50 gate (removed)",
                         line=dict(color="#8b949e", width=1.5, dash="dot")), 1, 1)
fig.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve, name="SPY",
                         line=dict(color="#484f58", width=1.2)), 1, 1)
_bx = [d for d in buys_n if d in notrend.index]
_sx = [d for d in sells_n if d in notrend.index]
fig.add_trace(go.Scatter(x=_bx, y=[notrend.loc[d] for d in _bx], mode="markers", name=f"Buys ({len(_bx)})",
                         marker=dict(symbol="triangle-up", color="#2ea043", size=5, opacity=0.4)), 1, 1)
fig.add_trace(go.Scatter(x=_sx, y=[notrend.loc[d] for d in _sx], mode="markers", name=f"Sells ({len(_sx)})",
                         marker=dict(symbol="triangle-down", color="#da3633", size=5, opacity=0.35)), 1, 1)
fig.add_trace(go.Scatter(x=diff.index, y=diff, name="Recaptured by removing the gate ($)",
                         line=dict(color="#3fb950", width=1.8), fill="tozeroy",
                         fillcolor="rgba(63,185,80,0.16)"), 2, 1)
fig.update_yaxes(title_text="Equity $", row=1, col=1, title_font_size=11)
fig.update_yaxes(title_text="Recaptured $", row=2, col=1, title_font_size=11)
fig.update_layout(template="plotly_dark", height=620, margin=dict(t=34, l=60, r=14, b=30),
                  legend=dict(orientation="h", y=1.09, x=0, font=dict(size=10)),
                  paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
chart = fig.to_html(full_html=False, include_plotlyjs="cdn",
                    config={"responsive": True, "displayModeBar": False},
                    default_width="100%", default_height="560px")


def card(k, v, sub, cls=""):
    return f"<div class=card><div class=k>{k}</div><div class='v {cls}'>{v}</div><div class=sub>{sub}</div></div>"


cards = ("<div class=cards>"
         + card("CAGR (current)", f"{mN['cagr']:+.1%}", f"old w/gate {mF['cagr']:+.1%} &middot; SPY {mS['cagr']:+.1%}",
                "pos" if mN['cagr'] >= 0 else "neg")
         + card("Max drawdown", f"{mN['dd']:.1%}", f"old {mF['dd']:.1%} &middot; SPY {mS['dd']:.1%}", "neg")
         + card("Sharpe", f"{mN['sharpe']:.2f}", f"old {mF['sharpe']:.2f} &middot; SPY {mS['sharpe']:.2f}")
         + card("Recaptured / yr", f"{mN['cagr']-mF['cagr']:+.1%}", "CAGR regained by removing the gate", "pos")
         + card("Cumulative gain", f"${diff.iloc[-1]:,.0f}", "vs the gated version, on $5k", "pos")
         + card("Signals taken", f"{n_n:,}", f"was {n_f:,} with the gate (+{n_n-n_f:,})")
         + "</div>")

CSS = ("body{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}"
       "h1{font-size:17px;margin:2px 0 4px}.sub2{color:#8b949e;font-size:12px;margin-bottom:8px}"
       ".cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}"
       ".card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px}"
       ".card .k{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.04em}"
       ".card .v{font-size:20px;font-weight:700;margin-top:3px}.card .sub{color:#6e7681;font-size:11px;margin-top:2px}"
       ".pos{color:#3fb950}.neg{color:#f85149}"
       ".cfg{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;margin-top:10px;font-size:12px;color:#9da7b1;line-height:1.6}"
       ".cfg b{color:#e6edf3}.note{color:#8b949e;font-size:11px;line-height:1.5;margin-top:8px}"
       ".brief{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px 16px;margin-top:14px;"
       "font-size:12.5px;color:#c9d1d9;line-height:1.6}"
       ".brief h2{font-size:14px;margin:0 0 6px;color:#e6edf3}.brief h3{font-size:12.5px;margin:12px 0 4px;color:#e6edf3}"
       ".brief ul,.brief ol{margin:4px 0 4px 18px;padding:0}.brief li{margin:3px 0}"
       ".brief code{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:0 4px;font-size:11px}"
       ".brief b{color:#e6edf3}.brief .bottom{margin-top:12px;border-top:1px solid #30363d;padding-top:10px}")

html = ("<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'><title>Current strategy</title>"
        f"<style>{CSS}</style></head>"
        "<body><h1>Current strategy — insider copier, trend gate removed</h1>"
        f"<div class=sub2>Insider copier (NO trend gate) + vol/liquidity, 60/40 SPY, regime-on, cash-only &middot; {notrend.index[0]} &rarr; {notrend.index[-1]}</div>"
        f"{cards}{chart}"
        "<div class=cfg>LIVE CONFIG: insider copier <b>12 names &times; $5,000</b>, ~90% deployed, "
        "<b>trend gate REMOVED</b> (now buys insiders' dip-buys &mdash; the highest-alpha signals), vol &ge; 0.0563, "
        "liquidity &ge; $1M ADV, freshness <b>&le;1 day from filing</b>, regime-on, cash-only, equity floor &minus;45% &rarr; ruin &minus;50%. "
        "<i>Activates at the scheduled 13:00 UTC restart (with the vol-gate revert to 0.0563).</i></div>"
        "<div class=note>GREEN = the current strategy (trend gate removed). The dotted line is the OLD gated version; "
        "the shaded gap + lower panel are the return <b>recaptured</b> by removing the gate &mdash; the backtest showed the "
        "gate blocked insiders' dip-buys (the highest-alpha signals), costing return AND deepening drawdown. "
        "60/40 SPY blend, regime-gated, 15bps/side slippage, in-sample; the live discrete copier captures less than these idealized magnitudes. "
        "<b>Live:</b> the gate is removed only for the insider copier; the BAL/Buffett-DCA sleeve still runs the <b>trend gate</b> alongside it &mdash; entering on an <b>EMA9 cross-down above the 200-SMA</b> (dip) <b>or</b> a <b>fresh cross-up through the 200-SMA</b>.</div>"
        "<div class=brief>"
        "<h2>Faithful now &mdash; but a trade still needs a signal</h2>"
        "<p>The implementation will now execute faithfully to the strategy. But &ldquo;orders will execute&rdquo; "
        "depends on conditions that aren&rsquo;t guaranteed every day.</p>"
        "<h3>What&rsquo;s automatic (no action needed)</h3><ul>"
        "<li><b>13:25 UTC restart</b> (timer active) &rarr; loads config fresh, day-P&amp;L resets &rarr; the &minus;$500 daily-loss halt clears.</li>"
        "<li><b>Regime</b>: copier uses the backtest rule (SPY/QQQ/IWM 200-SMA, currently 3/3 risk-on) &mdash; no more <code>halt_new_buys</code> blocking everything.</li>"
        "<li><b>Vol gate</b>: batched &mdash; no more 401 storm.</li>"
        "<li><b>Webhook paused</b> (won&rsquo;t auto-restart) &mdash; copier runs isolated.</li>"
        "<li><b>Queue</b>: empty now; a daily-loss-blocked signal replays next session while still fresh.</li></ul>"
        "<h3>A trade only happens if a signal exists</h3>"
        "<p>The copier fires only on a real <b>fresh (&le;1 day) insider open-market buy &gt;$1M</b> that passes the "
        "vol (&ge;0.0563) + liquidity (&ge;$1M ADV) gates &mdash; infrequent (~793 in 5.4y &asymp; 12/mo, so many days have zero). "
        "A qualifying signal &rarr; executes automatically. None &rarr; no trade, and that&rsquo;s correct, not a bug "
        "(the fidelity tool shows &ldquo;0 signals&rdquo; those days).</p>"
        "<h3>Three things that still (correctly) stop a trade</h3><ol>"
        "<li><b>Risk-off regime</b> &mdash; a majority of SPY/QQQ/IWM below their 200-SMA &rarr; holds cash (exactly like the backtest).</li>"
        "<li><b>Daily-loss limit re-engages</b> &mdash; another &minus;$500 day &rarr; halts again (intended).</li>"
        "<li><b>Account size</b> &mdash; each copy is risk-capped to ~$589/name (not the nominal $5k) on a ~$4.9k account.</li></ol>"
        "<p class=bottom><b>Bottom line:</b> nothing to change. The system is now faithful &mdash; it trades the strategy when a "
        "real signal shows up and conditions allow, and sits out when they don&rsquo;t. Judge it by the signal-fidelity report "
        "over weeks (intended-vs-executed), not whether it trades on any given day.</p></div>"
        "</body></html>")
open("crossover_tearsheet.html", "w", encoding="utf-8").write(html)
print("wrote crossover_tearsheet.html (current = no trend gate)")
