#!/usr/bin/env python3
"""directional_volexit_engine.py — long options + bull-regime naked puts, volume-reversal exit.

Encodes every failure lesson from the session:
  • long calls died to theta  -> exit on volume CLIMAX (take profit into exhaustion), not hold
  • credit spreads died to skew -> directional longs + tightly-gated naked puts instead
  • single-stock gaps -> INDEX-ONLY universe (QQQ/SPY/IWM)

REGIME (QQQ 200DMA ±2% band):
  BULL  (>200DMA+2%): Long Calls (~45Δ) + Naked Puts (10-20Δ, IVrank>40, capped)
  BEAR  (<200DMA-2%): Long Puts (~45Δ)
  TRANSITION (±2%)  : CASH (no trade)

EXIT — symmetric volume-reversal-AGAINST (RVOL>=2.0):
  Long Call  : RVOL>=2 AND red candle (close<open) AND close<prev_close   (climax/distribution)
  Long Put   : RVOL>=2 AND green candle(close>open) AND close>prev_close   (capitulation/cover)
  Naked Put  : same as long call (high-vol downside reversal = short put most vulnerable)
  + near-expiry backstop at TIME_STOP_DTE.

NAKED PUT constraints: bull only, DTE 30-60, delta 0.10-0.20, IV Rank>40 (VIX-based proxy),
  max assignment exposure 25% of account total / 10% per underlying. NOT cash-secured (margin),
  risk governed by the assignment-exposure caps. Real AV chains, BS-marked, conservative fills.
"""
from __future__ import annotations
import os, sys, json, argparse, csv
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = "/Users/blkpvnthr/Desktop/trader"
sys.path.insert(0, ROOT)
import edge_scan_spreads as E
import credit_spread_engine as C
import regime_score as RS
_bs_call = C._bs_call; bs_put = C.bs_put

_OHLCV = {}; _HL = {}
def get_ohlcv(symbol):
    if symbol in _OHLCV: return _OHLCV[symbol]
    if os.getenv("DAILY_SOURCE") == "yf":   # pre-2016 history via yfinance
        out = {}; hl = {}
        for d, v in E.yf_daily(symbol).items():
            out[d] = (v["open"], v["close"], v["volume"]); hl[d] = (v["high"], v["low"])
        _OHLCV[symbol] = out; _HL[symbol] = hl
        return out
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cl = StockHistoricalDataClient(E.AK, E.AS)
    df = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=datetime(2021,1,1), end=datetime(2026,6,6))).df.reset_index()
    out={}; hl={}
    for _, r in df.iterrows():
        d = r["timestamp"].date() if hasattr(r["timestamp"],"date") else datetime.fromisoformat(str(r["timestamp"])).date()
        out[d]=(float(r["open"]), float(r["close"]), float(r["volume"]))   # (open, close, volume) — indices unchanged
        hl[d]=(float(r["high"]), float(r["low"]))
    _OHLCV[symbol]=out; _HL[symbol]=hl
    return out


def atr_pct_series(symbol, ohlcv):
    """ATR(20)/Close per date for vol-scaled sizing."""
    o=ohlcv[symbol]; hl=_HL[symbol]; days=sorted(o); trs=[]; out={}; prevc=None
    for i,d in enumerate(days):
        h,l=hl[d]; c=o[d][1]
        tr=(h-l) if prevc is None else max(h-l, abs(h-prevc), abs(l-prevc))
        trs.append(tr); prevc=c
        out[d]=(sum(trs[-20:])/20)/c if (i>=20 and c>0) else 0.0
    return out

# VIX-based IV rank (range over trailing 252d); proxy for QQQ/SPY/IWM IV rank
_VIX = None
def iv_rank(d):
    global _VIX
    if _VIX is None:
        raw = RS._vix_series() or {}
        _VIX={}
        for k,v in raw.items():
            if isinstance(k,str):
                try: k=datetime.fromisoformat(k[:10]).date()
                except: continue
            elif hasattr(k,"date") and not isinstance(k,date): k=k.date()
            if isinstance(k,date): _VIX[k]=v
    vdays=[x for x in sorted(_VIX) if x<=d]
    if len(vdays)<60: return 50.0
    prior=[_VIX[x] for x in vdays[-252:]]; cur=prior[-1]; lo=min(prior); hi=max(prior)
    return 100*(cur-lo)/(hi-lo) if hi>lo else 50.0

def opt_value(p, S, today, r=0.043):
    dte=max((p.exp-today).days,0); T=dte/365.0
    if p.kind=="long_call": return _bs_call(S,p.strike,max(T,1e-6),r,max(p.iv,1e-4))["price"]
    return bs_put(S,p.strike,T,r,p.iv)

class Pos:
    __slots__=("sym","kind","strike","iv","exp","contracts","entry","opened","bullish","assign","edelta")
    def __init__(s,sym,kind,strike,iv,exp,contracts,entry,opened,bullish,assign=0.0,edelta=0.0):
        s.sym=sym;s.kind=kind;s.strike=strike;s.iv=iv;s.exp=exp;s.contracts=contracts
        s.entry=entry;s.opened=opened;s.bullish=bullish;s.assign=assign;s.edelta=edelta


def put_delta(S,K,T,r,sigma):
    if T<=0: return -1.0 if S<K else 0.0
    return _bs_call(S,K,max(T,1e-6),r,max(sigma,1e-4))["delta"]-1.0

def pick_band(rows,typ,lo,hi,exp):
    """pick the contract whose |delta| is within [lo,hi], closest to the midpoint."""
    mid=(lo+hi)/2; best=None; bd=9
    for r in rows:
        if str(r.get("type")).lower()!=typ or r.get("expiration","")[:10]!=exp: continue
        if E.f(r.get("bid"))<=0 or E.f(r.get("ask"))<=0: continue
        dl=abs(E.f(r.get("delta")))
        if lo<=dl<=hi and abs(dl-mid)<bd: best=r; bd=abs(dl-mid)
    return best

def portfolio_metrics(eq_rows, start_eq):
    import math, statistics
    from collections import defaultdict
    eqs=[r["equity"] for r in eq_rows]
    if len(eqs)<3: return {}
    rets=[eqs[i]/eqs[i-1]-1 for i in range(1,len(eqs)) if eqs[i-1]>0]
    mean=statistics.mean(rets); sd=statistics.pstdev(rets) or 1e-9
    ds=[r for r in rets if r<0]; dsd=(statistics.pstdev(ds) if len(ds)>1 else (abs(min(ds)) if ds else 1e-9)) or 1e-9
    peak=eqs[0]; sq=[]
    for e in eqs:
        peak=max(peak,e); dd=(peak-e)/peak*100; sq.append(dd*dd)
    ulcer=math.sqrt(sum(sq)/len(sq))
    cagr=((eqs[-1]/start_eq)**(252/len(eqs))-1)*100 if eqs[-1]>0 else -100
    monthly=defaultdict(list)
    for r in eq_rows: monthly[r["date"][:7]].append(r["equity"])
    mrets=[v[-1]/v[0]-1 for k,v in sorted(monthly.items()) if len(v)>=2 and v[0]>0]
    return dict(cagr_pct=round(cagr,1), sharpe=round(mean/sd*math.sqrt(252),2),
                sortino=round(mean/dsd*math.sqrt(252),2), ulcer=round(ulcer,2),
                worst_month_pct=round(min(mrets)*100,1) if mrets else 0)


def run(args):
    eq0, mode = (C.live_equity() if args.account else (args.equity,"FIXED"))
    rb = args.risk_budget * args.size_mult
    uni=[s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    START=datetime.strptime(args.start,"%Y-%m-%d").date(); END=datetime.strptime(args.end,"%Y-%m-%d").date()
    ohlcv={s:get_ohlcv(s) for s in uni}
    cal=sorted(set().union(*[set(o) for o in ohlcv.values()])); cal=[d for d in cal if START<=d<=END+timedelta(days=60)]
    regime=C.build_regime_200(E.get_closes("QQQ"), band=args.trend_band)
    entry_days=[]; d=START
    while d<=END:
        t=next((x for x in cal if x>=d),None)
        if t and t not in entry_days: entry_days.append(t)
        d+=timedelta(days=7)
    regime_days={"bull":0,"bear":0,"cash":0}
    for ed in entry_days: regime_days[regime.get(ed,"bull")]=regime_days.get(regime.get(ed,"bull"),0)+1
    # 20-EMA of close per symbol (slower trailing exit: 2 consecutive closes against it)
    ema20={}
    for s in uni:
        e=None; ema20[s]={}
        for dd in sorted(ohlcv[s]):
            c=ohlcv[s][dd][1]; e=c if e is None else c*(2/21)+e*(1-2/21); ema20[s][dd]=e
    atrp={s:atr_pct_series(s,ohlcv) for s in uni}   # ATR(20)/Close per symbol
    def atr_factor(sym,d):
        if not args.atr_scale: return 1.0
        ap=atrp[sym].get(d,0.0)
        return 0.25 if ap>0.05 else (0.50 if ap>0.03 else 1.0)

    cash=eq0; realized=0.0; peak=eq0; maxdd=0.0; pos=[]; trades=[]; eq_rows=[]; suspend=0
    def bar(sym,d):
        o=ohlcv[sym];
        if d in o: return o[d]
        prior=[x for x in o if x<=d]; return o[prior[-1]] if prior else None
    def prevclose(sym,d):
        o=ohlcv[sym]; days=[x for x in sorted(o) if x<d]
        return o[days[-1]][1] if days else None
    def rvol(sym,d):
        o=ohlcv[sym]; days=[x for x in sorted(o) if x<=d]
        if len(days)<21: return 0.0
        avg=sum(o[x][2] for x in days[-21:-1])/20
        return o[days[-1]][2]/avg if avg>0 else 0.0

    for today in cal:
        if suspend>0: suspend-=1   # portfolio DD-throttle countdown (suspends new entries)
        # ---- manage: exit = vol-reversal OR (optional 20EMA-2close) OR 7DTE ----
        keep=[]
        for p in pos:
            b=bar(p.sym,today)
            if b is None or today not in ohlcv[p.sym]: keep.append(p); continue
            o,c,_=b; pc=prevclose(p.sym,today); rv=rvol(p.sym,today); dte=(p.exp-today).days
            S=c; V=opt_value(p,S,today)
            # 2 consecutive closes against the 20-EMA (slow trailing exit)
            sdays=[x for x in sorted(ohlcv[p.sym]) if x<=today]; consec=False
            if len(sdays)>=2:
                d0,d1=sdays[-1],sdays[-2]; e0=ema20[p.sym].get(d0); e1=ema20[p.sym].get(d1)
                c0=ohlcv[p.sym][d0][1]; c1=ohlcv[p.sym][d1][1]
                if e0 and e1:
                    consec=(c0<e0 and c1<e1) if p.bullish else (c0>e0 and c1>e1)
            red = c<o and (pc is not None and c<pc); green = c>o and (pc is not None and c>pc)
            reason=None
            if rv>=args.rvol_th and ((p.bullish and red) or ((not p.bullish) and green)): reason="vol_reversal"
            elif args.ema20_exit and consec: reason="ema20_2close"
            elif dte<=args.time_stop_dte: reason="time_stop"
            if reason is None and dte<=0: reason="expiry"
            if reason:
                if p.kind=="short_put":
                    pnl=(p.entry - V)*100*p.contracts; cash += (p.entry - V)*100*p.contracts
                else:
                    pnl=(V - p.entry)*100*p.contracts; cash += V*100*p.contracts
                realized+=pnl
                trades.append(dict(close=today.isoformat(),sym=p.sym,kind=p.kind,strike=p.strike,
                    contracts=p.contracts,entry=round(p.entry,2),exitv=round(V,2),pnl=round(pnl,2),
                    reason=reason,held=(today-p.opened).days))
            else: keep.append(p)
        pos=keep

        # ---- entries (weekly) ----
        if today in entry_days and suspend==0:   # DD-throttle suspends new entries
            reg=regime.get(today,"bull")
            if reg!="cash":
                naked_assign=sum(p.assign for p in pos if p.kind=="short_put")
                for sym in uni:
                    if len(pos)>=args.max_positions: break
                    rows=E.raw_chain(sym,today)
                    if not rows: continue
                    def exps_for(lodte,hidte):
                        e={}
                        for rr in rows:
                            ex=rr.get("expiration","")[:10]
                            if not ex: continue
                            try: dt=(date.fromisoformat(ex)-today).days
                            except: continue
                            if lodte<=dt<=hidte: e[ex]=e.get(ex,0)+1
                        return max(e,key=e.get) if e else None
                    if reg=="bull":
                        exp=exps_for(30,45)
                        if exp and not any(x.sym==sym and x.kind=="long_call" for x in pos):
                            c45=pick_band(rows,"call",0.40,0.55,exp)
                            if c45:
                                ask=E.f(c45.get("ask")); n=int(rb*atr_factor(sym,today)//(ask*100))
                                if n>=1:
                                    cash-=ask*100*n
                                    pos.append(Pos(sym,"long_call",E.f(c45.get("strike")),E.f(c45.get("implied_volatility")) or .3,
                                        date.fromisoformat(exp),n,ask,today,True,0.0,abs(E.f(c45.get("delta")))))
                                    trades.append(dict(open=today.isoformat(),sym=sym,kind="long_call",contracts=n,entry=round(ask,2)))
                        # (naked-put sleeve removed — never fired in any window: IV-rank>40 and
                        #  bull-regime are near-mutually-exclusive, so it added complexity, no P&L)
                    else:  # bear: long put ~45Δ
                        exp=exps_for(30,45)
                        if exp and not any(x.sym==sym and x.kind=="long_put" for x in pos):
                            p45=pick_band(rows,"put",0.40,0.55,exp)
                            if p45:
                                ask=E.f(p45.get("ask")); n=int(rb*atr_factor(sym,today)//(ask*100))
                                if n>=1:
                                    cash-=ask*100*n
                                    pos.append(Pos(sym,"long_put",E.f(p45.get("strike")),E.f(p45.get("implied_volatility")) or .3,
                                        date.fromisoformat(exp),n,ask,today,False,0.0,abs(E.f(p45.get("delta")))))
                                    trades.append(dict(open=today.isoformat(),sym=sym,kind="long_put",contracts=n,entry=round(ask,2)))

        mtm=cash
        for p in pos:
            b=bar(p.sym,today)
            if b is None: continue
            V=opt_value(p,b[1],today)
            mtm += (V*100*p.contracts) if p.kind!="short_put" else (-(V-p.entry)*100*p.contracts)
        peak=max(peak,mtm); cur_dd=(peak-mtm)/peak*100 if peak>0 else 0; maxdd=max(maxdd,cur_dd)
        if args.dd_throttle and suspend==0 and cur_dd > args.dd_throttle_pct*100:
            suspend=args.dd_throttle_days   # halt new entries for N trading days after a DD breach
        eq_rows.append(dict(date=today.isoformat(),equity=round(mtm,2),open=len(pos)))

    closed=[t for t in trades if "pnl" in t]; pnls=[t["pnl"] for t in closed]; n=len(closed)
    wins=[t for t in closed if t["pnl"]>0]; days=max((cal[-1]-cal[0]).days,1) if cal else 1
    out=Path(ROOT)/"backtest_outputs"/args.run_tag; out.mkdir(parents=True,exist_ok=True)
    with open(out/"trades.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for t in trades for k in t})); w.writeheader(); w.writerows(trades)
    by_kind={k:{"n":sum(1 for t in closed if t.get("kind")==k),"pnl":round(sum(t["pnl"] for t in closed if t.get("kind")==k),0)}
             for k in set(t.get("kind") for t in closed)} if closed else {}
    summary=dict(run_tag=args.run_tag,mode=mode,universe=uni,window=f"{START}..{END}",start_equity=eq0,
        final_equity=round(eq0+realized,2),total_pnl=round(realized,2),return_pct=round(100*realized/eq0,2),
        closed_trades=n,win_rate=round(100*len(wins)/n,1) if n else 0,
        expectancy=round(sum(pnls)/n,1) if n else 0,worst=round(min(pnls),1) if pnls else 0,
        max_drawdown_pct=round(maxdd,2),pnl_per_day=round(realized/days,1),by_kind=by_kind,
        exit_mix={r:sum(1 for t in closed if t["reason"]==r) for r in set(t["reason"] for t in closed)} if closed else {},
        regime_entry_days=regime_days, size_mult=args.size_mult, metrics=portfolio_metrics(eq_rows,eq0))
    json.dump(summary,open(out/"summary.json","w"),indent=2); print(json.dumps(summary,indent=2)); print("DVE_DONE")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",default="QQQ,SPY,IWM")
    p.add_argument("--start",default="2025-06-02"); p.add_argument("--end",default="2026-04-27")
    p.add_argument("--account",action="store_true"); p.add_argument("--equity",type=float,default=100000.0)
    p.add_argument("--risk-budget",dest="risk_budget",type=float,default=2000.0)
    p.add_argument("--per-name-pct",dest="per_name_pct",type=float,default=0.10)
    p.add_argument("--naked-total-pct",dest="naked_total_pct",type=float,default=0.25)
    p.add_argument("--iv-rank-min",dest="iv_rank_min",type=float,default=40.0)
    p.add_argument("--rvol-th",dest="rvol_th",type=float,default=2.0)
    p.add_argument("--time-stop-dte",dest="time_stop_dte",type=int,default=7)
    p.add_argument("--max-positions",dest="max_positions",type=int,default=9)
    p.add_argument("--trend-band",dest="trend_band",type=float,default=0.02)
    p.add_argument("--size-mult",dest="size_mult",type=float,default=1.0,help="position-size multiplier (exposure sweep)")
    p.add_argument("--ema20-exit",dest="ema20_exit",action="store_true",help="enable 20EMA-2close exit (default OFF = pure rare-exit)")
    p.add_argument("--atr-scale",dest="atr_scale",action="store_true",help="vol-scale size: ATR%>5%→0.25x, >3%→0.5x")
    p.add_argument("--dd-throttle",dest="dd_throttle",action="store_true",help="suspend new entries after equity DD breach")
    p.add_argument("--dd-throttle-pct",dest="dd_throttle_pct",type=float,default=0.10)
    p.add_argument("--dd-throttle-days",dest="dd_throttle_days",type=int,default=20)
    p.add_argument("--run-tag",dest="run_tag",default="dve")
    run(p.parse_args())
