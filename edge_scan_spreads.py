#!/usr/bin/env python3
"""edge_scan_spreads.py — does premium SELLING beat premium BUYING on real chains?

Decisive, signal-free expectancy test of option STRUCTURES on real Alpha Vantage
historical chains, held to expiry. No directional signal — this isolates the
volatility-risk-premium (VRP) edge that the long-call buyer was on the wrong side of.

For each weekly entry date over a long history, on a liquid underlying:
  • LONG ATM CALL          (the current engine's structure — pays theta + spread)
  • SHORT 30Δ/15Δ PUT SPREAD  (defined-risk bullish credit — collects theta)
  • IRON CONDOR 30Δ/15Δ      (defined-risk delta-neutral credit — pure VRP harvest)

Conservative fills: SELL at bid, BUY at ask (cross the spread against us).
Settlement: real underlying close on the expiry date (Alpaca daily bars).
Honest: every losing tail is included; held to expiry (no curve-fit exits).

Output: per-structure expectancy ($/trade), win rate, worst loss, total, Sharpe-ish.
"""
from __future__ import annotations
import os, sys, json, math, statistics
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
import requests

ROOT = "/Users/blkpvnthr/Desktop/trader"
load_dotenv(os.path.join(ROOT, ".env"))
AVK = os.getenv("ALPHA_VANTAGE_API_KEY")
AK = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY")
AS = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET")
sys.path.insert(0, ROOT)
import av_options_adapter as av  # reuse cached AV fetch

START = date(2025, 6, 2)
END = date(2026, 4, 27)        # 30-DTE expiries settle by ~late-May 2026 (in the past)
DTE_LO, DTE_HI = 25, 45

def f(x, d=0.0):
    try: return float(x)
    except: return d

# --- Alpaca daily closes for settlement (per symbol, cached in-process) ---
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
_dc = StockHistoricalDataClient(AK, AS)
_CLOSES: dict = {}

_YF: dict = {}
def yf_daily(symbol):
    """Daily OHLCV via yfinance (covers 2014+; Alpaca stock history is empty <2016)."""
    if symbol in _YF:
        return _YF[symbol]
    import yfinance as yf
    df = yf.download(symbol, start="2014-01-01", end="2026-06-06", progress=False, auto_adjust=False)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)
    out = {}
    for ts, row in df.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        out[d] = dict(open=float(row["Open"]), high=float(row["High"]), low=float(row["Low"]),
                      close=float(row["Close"]), volume=float(row["Volume"]))
    _YF[symbol] = out
    return out


def get_closes(symbol):
    if symbol in _CLOSES:
        return _CLOSES[symbol]
    if os.getenv("DAILY_SOURCE") == "yf":
        _CLOSES[symbol] = {d: v["close"] for d, v in yf_daily(symbol).items()}
        return _CLOSES[symbol]
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=datetime(2021,1,1), end=datetime(2026,6,6))
    bars = _dc.get_stock_bars(req).df.reset_index()
    closes = {}
    for _, r in bars.iterrows():
        d = r["timestamp"].date() if hasattr(r["timestamp"], "date") else datetime.fromisoformat(str(r["timestamp"])).date()
        closes[d] = float(r["close"])
    _CLOSES[symbol] = closes
    return closes

def raw_chain(symbol, d):
    """Full raw AV historical chain (puts+calls) for date d, cached via adapter dir."""
    import av_options_adapter as A
    ds = d.strftime("%Y-%m-%d")
    cf = A._CACHE_DIR / f"{symbol}_{ds.replace('-','')}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    r = requests.get("https://www.alphavantage.co/query",
                     params={"function":"HISTORICAL_OPTIONS","symbol":symbol,"date":ds,"apikey":AVK}, timeout=30)
    rows = r.json().get("data") or []
    cf.write_text(json.dumps(rows))
    return rows

def pick(rows, typ, target_delta, exp):
    cands = [r for r in rows if str(r.get("type")).lower()==typ and r.get("expiration","")[:10]==exp
             and f(r.get("bid"))>0 and f(r.get("ask"))>0]
    if not cands: return None
    return min(cands, key=lambda r: abs(abs(f(r.get("delta")))-target_delta))


def scan(symbol):
    """Held-to-expiry expectancy of 3 structures on real chains. Returns dict of pnl lists."""
    closes = get_closes(symbol)
    tdays = sorted(closes)
    def close_on_or_after(d):
        for t in tdays:
            if t >= d: return closes[t]
        return None
    entries = []
    d = START
    while d <= END:
        t = next((x for x in tdays if x >= d), None)
        if t and t <= END and t not in entries: entries.append(t)
        d += timedelta(days=7)
    results = {"long_call": [], "put_credit_spread": [], "iron_condor": []}
    used = 0
    for ent in entries:
        rows = raw_chain(symbol, ent)
        if not rows or ent not in closes: continue
        exps = {}
        for r in rows:
            e = r.get("expiration","")[:10]
            if not e: continue
            try: dte = (date.fromisoformat(e)-ent).days
            except: continue
            if DTE_LO <= dte <= DTE_HI: exps[e] = exps.get(e,0)+1
        if not exps: continue
        exp = max(exps, key=exps.get)
        S_exp = close_on_or_after(date.fromisoformat(exp))
        if S_exp is None: continue
        used += 1
        c = pick(rows, "call", 0.50, exp)
        if c:
            results["long_call"].append((max(0.0, S_exp - f(c.get("strike"))) - f(c.get("ask"))) * 100)
        ps, pl = pick(rows,"put",0.30,exp), pick(rows,"put",0.15,exp)
        credit_put = None
        if ps and pl and f(ps.get("strike"))>f(pl.get("strike")):
            Ks, Kl = f(ps.get("strike")), f(pl.get("strike"))
            credit = f(ps.get("bid")) - f(pl.get("ask"))
            if credit > 0:
                payoff = max(credit - (max(0.0,Ks-S_exp)-max(0.0,Kl-S_exp)), credit-(Ks-Kl))
                results["put_credit_spread"].append(payoff*100); credit_put=(Ks,Kl,credit)
        cs, cl = pick(rows,"call",0.30,exp), pick(rows,"call",0.15,exp)
        if credit_put and cs and cl and f(cl.get("strike"))>f(cs.get("strike")):
            Ksc, Klc = f(cs.get("strike")), f(cl.get("strike"))
            credit_c = f(cs.get("bid")) - f(cl.get("ask"))
            Ksp,Klp,credit_p = credit_put
            if credit_c>0:
                put_leg=max(credit_p-(max(0.0,Ksp-S_exp)-max(0.0,Klp-S_exp)), credit_p-(Ksp-Klp))
                call_leg=max(credit_c-(max(0.0,S_exp-Ksc)-max(0.0,S_exp-Klc)), credit_c-(Klc-Ksc))
                results["iron_condor"].append((put_leg+call_leg)*100)
    results["_used"] = used
    return results


def summarize(pnls):
    if not pnls: return None
    n=len(pnls); tot=sum(pnls); exp=tot/n
    wins=[p for p in pnls if p>0]
    sd=statistics.pstdev(pnls) if n>1 else 0
    t=(exp/sd*math.sqrt(n)) if sd>0 else 0
    return dict(n=n, exp=exp, total=tot, win=100*len(wins)/n, worst=min(pnls), t=t)


def stat(name, pnls):
    s=summarize(pnls)
    if not s: print(f"  {name:<20} no trades"); return
    print(f"  {name:<20} n={s['n']:<3} exp=${s['exp']:>7.1f}/trade  total=${s['total']:>8.0f}  "
          f"win%={s['win']:>5.1f}  worst=${s['worst']:>7.0f}  t~{s['t']:>4.1f}")


if __name__ == "__main__":
    SYM = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    res = scan(SYM)
    print(f"\nEDGE SCAN — {SYM}, weekly entries {START}..{END}, {DTE_LO}-{DTE_HI} DTE, held to expiry, REAL AV chains")
    print(f"entries used: {res['_used']}  (conservative fills: sell@bid / buy@ask; real settlement)\n")
    stat("LONG ATM CALL", res["long_call"])
    stat("PUT CREDIT SPREAD", res["put_credit_spread"])
    stat("IRON CONDOR", res["iron_condor"])
    print("\nEDGE_SCAN_DONE")
