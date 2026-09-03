#!/usr/bin/env python3
"""regime_score.py — multi-factor market regime (bull / cash / bear) for QQQ (NDX proxy).

A binary 20/50-EMA regime whipsaws — by the time it confirms, you've sold losing put
spreads through the whole top. This scores STRUCTURE so the engine can stand aside in
transition (the regime where the 2022 test showed all the damage occurred).

bear_score (0-5):
  +1  close < EMA50
  +1  EMA20 < EMA50
  +1  ADX(14) > 25            (trend strength)
  +1  VIX > VIX 20DMA         (rising fear; realized-vol proxy if ^VIX unavailable)
  +1  close < prior 20d swing low   (structural breakdown)

regime:  score>=4 -> 'bear' (sell call spreads) | score<=1 -> 'bull' (sell put spreads)
         else 'cash' (NO TRADE — transition).
"""
from __future__ import annotations
import os
from datetime import date, datetime
from dotenv import load_dotenv

ROOT = "/Users/blkpvnthr/Desktop/trader"
load_dotenv(os.path.join(ROOT, ".env"))
AK = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY")
AS = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET")


def _qqq_ohlc():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = StockHistoricalDataClient(AK, AS)
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols="QQQ", timeframe=TimeFrame.Day,
                          start=datetime(2021,1,1), end=datetime(2026,6,6))).df.reset_index()
    rows = []
    for _, r in df.iterrows():
        d = r["timestamp"].date() if hasattr(r["timestamp"], "date") else datetime.fromisoformat(str(r["timestamp"])).date()
        rows.append((d, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
    return rows  # sorted by date


def _ema(vals, span):
    k = 2/(span+1); e=None; out=[]
    for v in vals:
        e = v if e is None else v*k + e*(1-k); out.append(e)
    return out


def _wilder(vals, n):
    out=[]; s=None
    for i,v in enumerate(vals):
        if i < n:
            out.append(None)
            if i == n-1: s = sum(vals[:n]); out[-1]=s/n
        else:
            s = s - (s/n) + v; out.append(s/n)
    return out


def _adx(highs, lows, closes, n=14):
    tr=[0.0]; pdm=[0.0]; ndm=[0.0]
    for i in range(1,len(closes)):
        up=highs[i]-highs[i-1]; dn=lows[i-1]-lows[i]
        pdm.append(up if (up>dn and up>0) else 0.0)
        ndm.append(dn if (dn>up and dn>0) else 0.0)
        tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr=_wilder(tr,n); pdi_s=_wilder(pdm,n); ndi_s=_wilder(ndm,n)
    adx_in=[]
    for i in range(len(closes)):
        if atr[i] and atr[i]>0 and pdi_s[i] is not None:
            pdi=100*pdi_s[i]/atr[i]; ndi=100*ndi_s[i]/atr[i]
            dx=100*abs(pdi-ndi)/((pdi+ndi) or 1)
            adx_in.append(dx)
        else:
            adx_in.append(0.0)
    return _wilder(adx_in, n)


def _vix_series():
    """date->VIX close via yfinance ^VIX; {} if unavailable (caller uses realized-vol proxy)."""
    try:
        import yfinance as yf
        df = yf.download("^VIX", start="2021-01-01", end="2026-06-06", progress=False, auto_adjust=False)
        if df is None or len(df)==0: return {}
        cl = df["Close"]
        out={}
        for ts, v in cl.items():
            d = ts.date() if hasattr(ts,"date") else ts
            try: out[d]=float(v)
            except Exception:
                try: out[d]=float(v.iloc[0])
                except Exception: pass
        return out
    except Exception:
        return {}


def compute(detail=False):
    rows=_qqq_ohlc()
    days=[r[0] for r in rows]; H=[r[2] for r in rows]; L=[r[3] for r in rows]; C=[r[4] for r in rows]
    e20=_ema(C,20); e50=_ema(C,50); adx=_adx(H,L,C,14)
    vix=_vix_series()
    # VIX 20DMA (align to QQQ days; carry-forward last VIX); fallback proxy = 20d realized vol
    vlist=[]
    last=None
    for d in days:
        if d in vix: last=vix[d]
        vlist.append(last)
    def ma(series,i,n):
        s=[x for x in series[max(0,i-n+1):i+1] if x is not None]
        return sum(s)/len(s) if s else None
    # realized vol proxy (annualized 20d) if no VIX
    import math
    rv=[None]*len(C)
    for i in range(len(C)):
        if i>=20:
            rets=[math.log(C[j]/C[j-1]) for j in range(i-19,i+1)]
            mu=sum(rets)/len(rets); var=sum((x-mu)**2 for x in rets)/len(rets)
            rv[i]=math.sqrt(var*252)*100
    reg={}; det={}
    for i,d in enumerate(days):
        score=0; comp={}
        comp["c<e50"]=C[i]<(e50[i] or C[i]); score+=comp["c<e50"]
        comp["e20<e50"]=(e20[i] or 0)<(e50[i] or 0); score+=comp["e20<e50"]
        comp["adx>25"]=(adx[i] or 0)>25; score+=comp["adx>25"]
        if vlist[i] is not None:
            v20=ma(vlist,i,20); comp["vix>20dma"]=bool(v20 and vlist[i]>v20)
        else:
            r60=ma(rv,i,60); comp["vix>20dma"]=bool(rv[i] and r60 and rv[i]>r60)
        score+=comp["vix>20dma"]
        swing=min(C[max(0,i-20):i]) if i>=1 else C[i]
        comp["c<swinglow"]=C[i]<swing; score+=comp["c<swinglow"]
        reg[d] = "bear" if score>=4 else ("bull" if score<=1 else "cash")
        det[d]=(score,comp)
    return (reg, det) if detail else reg


if __name__ == "__main__":
    reg, det = compute(detail=True)
    src = "^VIX" if _vix_series() else "realized-vol proxy"
    for lbl, s, e in [("2022 bear","2022-01-03","2023-01-03"), ("2025-26 bull","2025-06-02","2026-04-27")]:
        sd=date.fromisoformat(s); ed=date.fromisoformat(e)
        win={d:reg[d] for d in reg if sd<=d<=ed}
        from collections import Counter
        print(f"{lbl} [{s}..{e}]  vix_src={src}  " +
              "  ".join(f"{k}={v}" for k,v in Counter(win.values()).items()) + f"  (n={len(win)})")
