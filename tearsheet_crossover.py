#!/usr/bin/env python3
"""
tearsheet_crossover.py — backtest the ENHANCED crossover strategy on $10,000 and compare to the
baseline, with an interactive HTML tearsheet (equity curves, drawdown, metrics).

Enhancements stacked on the base crossover (buy 200-SMA cross-up OR 9-EMA cross-down>200-SMA):
  ENTRY  : quality large-cap universe; broader-market regime (SPY/QQQ/IWM breadth) must be risk-on;
           market-volatility filter (skip top-quartile SPY 20d realized vol); relative strength
           (name's 63d return > SPY's); volume confirmation (today's vol > 20d average); sector cap.
  EXIT   : ATR(14) trailing stop (3 ATR below the high since entry) OR 200-SMA break OR 15% stop OR
           regime liquidation (market breadth risk-off); partial profit (sell 1/3 at +20%).
Sizing 20% of available cash, 45% buffer, 10%/name, 30%/sector, 5bps slippage/side.
"""
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
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

START_CASH, CASH_BUFFER, SINGLE_CAP, SECTOR_CAP, BUY_FRAC = 10_000.0, 0.45, 0.10, 0.30, 0.20
MIN_ORDER, SLIP, EMA_P, SMA_P = 100.0, 0.0005, 9, 200
STOP, TRAIL_ATR, PARTIAL_TGT, PARTIAL_FRAC = 0.15, 3.0, 0.20, 0.33
START, END = datetime(2019, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)

# Quality, diversified large-cap universe (lower beta than the speculative watchlist) + sector map.
SECTOR = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "ADBE": "Tech", "CRM": "Tech",
    "ORCL": "Tech", "CSCO": "Tech", "ACN": "Tech", "INTU": "Tech", "NOW": "Tech",
    "NVDA": "Semis", "AVGO": "Semis", "TSM": "Semis", "TXN": "Semis", "QCOM": "Semis", "AMAT": "Semis",
    "AMZN": "Consumer", "COST": "Consumer", "WMT": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples",
    "JNJ": "Health", "UNH": "Health", "LLY": "Health", "ABBV": "Health", "MRK": "Health", "TMO": "Health", "ABT": "Health",
    "V": "Financial", "MA": "Financial", "AXP": "Financial", "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
    "LIN": "Industrial", "CAT": "Industrial", "DE": "Industrial", "HON": "Industrial", "UNP": "Industrial",
}
UNIVERSE = sorted(SECTOR)


def fetch(sym):
    try:
        df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                                start=START, end=END, feed=DataFeed.IEX)).df
        if df is None or df.empty:
            return None
        df = df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == sym]
        df = df.set_index(pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize())
        return df[["close", "high", "low", "volume"]].astype(float)[~df.index.duplicated()]
    except Exception:
        return None


print(f"Fetching {len(UNIVERSE)} quality names + SPY/QQQ/IWM …")
bars = {s: fetch(s) for s in UNIVERSE}
bars = {s: d for s, d in bars.items() if d is not None and len(d) >= SMA_P + 70}
spyb, qqqb, iwmb = fetch("SPY"), fetch("QQQ"), fetch("IWM")
closes = pd.DataFrame({s: d["close"] for s, d in bars.items()}).sort_index()
closes = closes[closes.index >= closes.dropna(how="all").index[0]]
idx = closes.index
highs = pd.DataFrame({s: d["high"] for s, d in bars.items()}).reindex(idx)
lows = pd.DataFrame({s: d["low"] for s, d in bars.items()}).reindex(idx)
vols = pd.DataFrame({s: d["volume"] for s, d in bars.items()}).reindex(idx)
print(f"Usable ({closes.shape[1]}): {', '.join(closes.columns)}\n")

ema = closes.ewm(span=EMA_P, adjust=False).mean()
sma = closes.rolling(SMA_P).mean()
prev = closes.shift(1)
ret63 = closes.pct_change(63)
vol20 = vols.rolling(20).mean()
tr = pd.concat([highs - lows, (highs - closes.shift(1)).abs(), (lows - closes.shift(1)).abs()]
               ).groupby(level=0).max()
atr = tr.rolling(14).mean()

spy = spyb["close"].reindex(idx).ffill()
spy_ret63 = spy.pct_change(63)
spy_volann = spy.pct_change().rolling(20).std() * np.sqrt(252)
vol_ok = (spy_volann < spy_volann.rolling(252).quantile(0.75)).fillna(True)   # below top-quartile vol


def _idx_up(b):
    a = b["close"].reindex(idx).ffill()
    s200, s50 = a.rolling(SMA_P).mean(), a.rolling(50).mean()
    return ((a > s200) & (s50 > s50.shift(20))).fillna(False), (a < s200).fillna(False)


_reg = [_idx_up(b) for b in (spyb, qqqb, iwmb) if b is not None]
RISK_ON = (sum(u.astype(int) for u, _ in _reg) >= 2)
RISK_OFF = (sum(dn.astype(int) for _, dn in _reg) >= 2)


def simulate(mode="enhanced"):
    cash = START_CASH
    pos, cost, peak, partial = {}, {}, {}, {}
    curve, orders, trades = [], [], []
    for d in idx:
        c, sm_, em_, pv_, at_ = closes.loc[d], sma.loc[d], ema.loc[d], prev.loc[d], atr.loc[d]
        roff, ron = bool(RISK_OFF.get(d, False)), bool(RISK_ON.get(d, False))

        def sell(s, p, frac=1.0):
            nonlocal cash
            q = pos[s] * frac
            proceeds = q * p * (1 - SLIP)
            trades.append(proceeds - (cost[s] * frac))
            cash += proceeds
            cost[s] *= (1 - frac); pos[s] -= q
            orders.append((d, "sell"))

        if roff:                                                  # regime exit → liquidate
            for s in [x for x in pos if pos[x] > 0]:
                p = c[s]
                if not np.isnan(p):
                    sell(s, p, 1.0)
        for s in [x for x in pos if pos[x] > 0]:                  # per-position exits
            p, sm = c[s], sm_[s]
            if np.isnan(p):
                continue
            hi = highs.loc[d, s]
            if not np.isnan(hi):
                peak[s] = max(peak.get(s, p), hi)
            avg = cost[s] / pos[s] if pos[s] > 0 else 0.0
            a = at_[s]
            trail_hit = (mode == "enhanced" and not np.isnan(a) and a > 0
                         and p < peak.get(s, p) - TRAIL_ATR * a)
            if (not np.isnan(sm) and p < sm) or (avg > 0 and p < avg * (1 - STOP)) or trail_hit:
                sell(s, p, 1.0)
        if mode == "enhanced":                                    # partial profit at +20%
            for s in [x for x in pos if pos[x] > 0 and not partial.get(x)]:
                p = c[s]
                avg = cost[s] / pos[s] if pos[s] > 0 else 0.0
                if not np.isnan(p) and avg > 0 and p >= avg * (1 + PARTIAL_TGT):
                    sell(s, p, PARTIAL_FRAC); partial[s] = True
        pos = {s: q for s, q in pos.items() if q > 1e-9}
        eq = cash + sum(pos.get(s, 0) * c[s] for s in pos if not np.isnan(c[s]))

        block = roff or not ron or (mode == "enhanced" and not bool(vol_ok.get(d, True)))
        if not block:
            for s in closes.columns:
                p, em, sm, pv = c[s], em_[s], sm_[s], pv_[s]
                if any(np.isnan(x) for x in (p, em, sm, pv)):
                    continue
                if not ((pv < sm <= p) or ((pv > em >= p) and p > sm)):
                    continue
                if mode == "enhanced":
                    r63, sr63 = ret63.loc[d, s], spy_ret63.get(d)
                    if np.isnan(r63) or sr63 is None or np.isnan(sr63) or r63 <= sr63:   # relative strength
                        continue
                    v, v20 = vols.loc[d, s], vol20.loc[d, s]
                    if np.isnan(v) or np.isnan(v20) or v <= v20:                          # volume confirm
                        continue
                    sec = SECTOR.get(s)
                    sec_exp = sum(pos.get(x, 0) * c[x] for x in pos
                                  if SECTOR.get(x) == sec and not np.isnan(c[x]))
                    if sec_exp >= SECTOR_CAP * eq:                                        # sector cap
                        continue
                avail = max(0.0, cash - CASH_BUFFER * eq)
                room = max(0.0, SINGLE_CAP * eq - pos.get(s, 0) * p)
                order = min(BUY_FRAC * avail, room, avail)
                if order >= MIN_ORDER:
                    cash -= order * (1 + SLIP)
                    pos[s] = pos.get(s, 0) + order / p
                    cost[s] = cost.get(s, 0) + order
                    peak[s] = max(peak.get(s, p), p); partial.setdefault(s, False)
                    orders.append((d, "buy"))
        curve.append((d, cash + sum(pos.get(s, 0) * c[s] for s in pos if not np.isnan(c[s]))))
    return pd.Series(dict(curve)), orders, trades


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
    return dict(final=curve.iloc[-1], tot=curve.iloc[-1] / START_CASH - 1, cagr=cagr, sharpe=sharpe,
                sortino=sortino, dd=dd, n=len(trades), wr=wr, pf=(gw / gl if gl else float("inf")),
                calmar=(cagr / abs(dd) if dd else 0))


spy_curve = (START_CASH * spy / spy.iloc[0])
VARIANTS = [("Baseline (crossover+regime+stop)", "baseline"), ("Enhanced (all changes)", "enhanced")]
res = {label: (lambda r: (r[0], metrics(r[0], r[2])))(simulate(m)) for label, m in VARIANTS}
ms = metrics(spy_curve.reindex(idx).ffill(), [])
best = "Enhanced (all changes)"

PALETTE = {"Baseline (crossover+regime+stop)": "#8b949e", "Enhanced (all changes)": "#3fb950"}
fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.08,
                    subplot_titles=("Equity ($10k start)", "Drawdown %"))
for label, _ in VARIANTS:
    c = res[label][0]
    fig.add_trace(go.Scatter(x=c.index, y=c, name=label,
                             line=dict(color=PALETTE[label], width=2.4)), row=1, col=1)
    dd = (c / c.cummax() - 1) * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name=label, showlegend=False,
                             line=dict(color=PALETTE[label], width=1.2)), row=2, col=1)
fig.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve, name="SPY",
                         line=dict(color="#6e7681", width=1.3, dash="dot")), row=1, col=1)
fig.update_yaxes(title_text="$", row=1, col=1)
fig.update_layout(template="plotly_dark", height=760, margin=dict(t=70, l=50, r=20, b=30),
                  title="Enhanced Crossover Strategy — quality universe, RS/vol/regime filters, ATR trail + partials",
                  legend=dict(orientation="h", y=1.06), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
chart = fig.to_html(full_html=False, include_plotlyjs="cdn",
                    config={"responsive": True, "displayModeBar": False},
                    default_width="100%", default_height="620px")


def trow(label, m, star=False):
    pf_ = "∞" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    hi = " style='background:#15301c'" if star else ""
    return (f"<tr{hi}><td style='font-weight:600'>{label}{' ★' if star else ''}</td><td>${m['final']:,.0f}</td>"
            f"<td>{m['cagr']:+.1%}</td><td>{m['dd']:.1%}</td><td><b>{m['calmar']:.2f}</b></td>"
            f"<td><b>{m['wr']:.0f}%</b></td><td>{m['sharpe']:.2f}</td><td>{m['sortino']:.2f}</td>"
            f"<td>{m['n']}</td><td>{pf_}</td></tr>")


tbl = ("<table><thead><tr><th>Variant</th><th>Final</th><th>CAGR</th><th>MaxDD</th><th>Calmar</th>"
       "<th>Win%</th><th>Sharpe</th><th>Sortino</th><th>Trades</th><th>PF</th></tr></thead><tbody>"
       + "".join(trow(l, res[l][1], l == best) for l, _ in VARIANTS) + trow("SPY b&h", ms) + "</tbody></table>")
e = res[best][1]
html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Enhanced strategy</title>
<style>body{{background:#0d1117;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px}}
h1{{font-size:17px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}}
th,td{{padding:7px 6px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:#8b949e}}
.note{{color:#8b949e;font-size:12px;line-height:1.5;margin-top:10px}}.scroll{{overflow-x:auto}}
.pick{{background:#161b22;border:1px solid #2ea043;border-radius:10px;padding:10px 12px;margin:8px 0}}</style></head>
<body><h1>Enhanced Crossover Strategy · $10,000 · {res[best][0].index[0]:%Y-%m-%d} → {res[best][0].index[-1]:%Y-%m-%d}</h1>
<div class=pick>★ Enhanced: ${e['final']:,.0f}, CAGR {e['cagr']:+.1%}, MaxDD {e['dd']:.1%}, <b>Win {e['wr']:.0f}%</b>, Calmar {e['calmar']:.2f}</div>
<div class=scroll>{tbl}</div>{chart}
<div class=note>ENTRY: quality large-cap universe; regime risk-on (SPY/QQQ/IWM breadth); SPY-vol filter
(skip top-quartile 20d realized vol); relative strength (63d return &gt; SPY); volume &gt; 20d avg; 30%/sector cap.
EXIT: ATR(14) 3x trailing stop OR 200-SMA OR 15% stop OR regime liquidation; partial 1/3 at +20%.
20% of available cash, 45% buffer, 10%/name, 5bps/side. Backtest, not a forward guarantee.</div>
</body></html>"""
open("crossover_tearsheet.html", "w").write(html)
for l, _ in VARIANTS:
    m = res[l][1]
    print(f"  {l:<34}: ${m['final']:>8,.0f}  CAGR {m['cagr']:+6.1%}  DD {m['dd']:>6.1%}  Win {m['wr']:>3.0f}%  Calmar {m['calmar']:.2f}  ({m['n']} tr)")
print(f"  {'SPY':<34}: ${ms['final']:>8,.0f}  CAGR {ms['cagr']:+6.1%}  DD {ms['dd']:>6.1%}")
