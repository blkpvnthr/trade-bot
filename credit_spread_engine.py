#!/usr/bin/env python3
"""credit_spread_engine.py — defined-risk credit-spread engine on REAL option chains.

The session's evidence: buying calls loses to spread+theta; SELLING defined-risk premium
(put credit spreads) had the highest t-stat / 92% win on real chains (VRP edge), but with a
fat left tail. This engine harvests that edge WITH active management to cut the tail:

  • Entry  : weekly per name, sell ~30Δ put / buy ~15Δ put, 30-45 DTE (real AV-chain credit,
             conservative fills sell@bid / buy@ask). Defined max loss = (width - credit).
  • Manage : daily mark via Black-Scholes (entry IV, real underlying path). Close on
               - take-profit  : capture TP_FRAC of the credit (default 50%)
               - stop-loss    : loss reaches SL_MULT × credit (default 2×)
               - time stop    : at TIME_STOP_DTE (default 21 DTE)
  • Size   : risk per spread sized to a $ budget (default $500–$1000) and exposure cap,
             anchored to LIVE Alpaca account equity.

Honest: real entry credits, every tail included, managed exits (no curve-fit), defined risk.
Outputs: trades.csv, equity_curve.csv, summary.json under backtest_outputs/<run_tag>/.
"""
from __future__ import annotations
import os, sys, json, math, argparse, statistics
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = "/Users/blkpvnthr/Desktop/trader"
sys.path.insert(0, ROOT)
import edge_scan_spreads as E      # raw_chain, pick, get_closes, f
import alpha_call_engine as ace
_bs_call = ace.black_scholes_call


def bs_put(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return max(K - S, 0.0)
    c = _bs_call(S, K, max(T, 1e-6), r, max(sigma, 1e-4), q)["price"]
    return max(c - S * math.exp(-q * T) + K * math.exp(-r * T), 0.0)


def live_equity():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    k = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY")
    s = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET")
    paper = str(os.getenv("ALPACA_PAPER_ONLY", "true")).lower() in {"1", "true", "yes", "paper"}
    from alpaca.trading.client import TradingClient
    a = TradingClient(k, s, paper=paper).get_account()
    return round(float(a.equity), 2), ("PAPER" if paper else "LIVE")


class Spread:
    __slots__ = ("sym","kind","Ks","Kl","credit","iv","exp","contracts","opened","width","entry_risk","regime")
    def __init__(s, sym, kind, Ks, Kl, credit, iv, exp, contracts, opened, regime):
        s.sym=sym; s.kind=kind; s.Ks=Ks; s.Kl=Kl; s.credit=credit; s.iv=iv; s.exp=exp
        s.contracts=contracts; s.opened=opened; s.regime=regime; s.width=abs(Ks-Kl)
        s.entry_risk=(s.width-credit)*100*contracts


def value_to_close(sp: Spread, S, today, r=0.043):
    """Per-share cost to buy the spread back (short leg - long leg), BS-marked."""
    dte = max((sp.exp - today).days, 0); T = dte/365.0
    if sp.kind == "put":   # bull put spread: short Ks(high) / long Kl(low)
        return max(bs_put(S, sp.Ks, T, r, sp.iv) - bs_put(S, sp.Kl, T, r, sp.iv), 0.0)
    # bear call spread: short Ks(low) / long Kl(high)
    sc = _bs_call(S, sp.Ks, max(T,1e-6), r, max(sp.iv,1e-4))["price"]
    lc = _bs_call(S, sp.Kl, max(T,1e-6), r, max(sp.iv,1e-4))["price"]
    return max(sc - lc, 0.0)


def _ema(days, closes, span):
    k = 2/(span+1); ema=None; out={}
    for d in days:
        c=closes[d]; ema = c if ema is None else c*k + ema*(1-k); out[d]=ema
    return out


def build_regime(proxy_closes):
    """NDX-proxy (QQQ) regime: 'bear' when close < EMA20 AND < EMA50, else 'bull'."""
    days=sorted(proxy_closes); e20=_ema(days,proxy_closes,20); e50=_ema(days,proxy_closes,50)
    return {d: ("bear" if (proxy_closes[d]<e20[d] and proxy_closes[d]<e50[d]) else "bull") for d in days}


def build_regime_200(proxy_closes, band=0.0):
    """Primary-trend gate on the 200-day SMA: above->bull (put spreads), below->bear
    (call spreads). Optional cash band of +/-`band` around the 200DMA. The point: NEVER
    sell put spreads while QQQ is below its 200DMA (kills the 2022 bear-rally put bleed)."""
    days=sorted(proxy_closes)
    out={}
    from collections import deque
    win=deque(maxlen=200)
    for d in days:
        c=proxy_closes[d]; win.append(c)
        sma=(sum(win)/len(win)) if len(win)==200 else None
        if sma is None:
            out[d]="bull"   # pre-warmup default (history starts 2021-01 so 200DMA is set by the test windows)
        elif c > sma*(1+band): out[d]="bull"
        elif c < sma*(1-band): out[d]="bear"
        else: out[d]="cash"
    return out


def run(args):
    eq0, mode = (live_equity() if args.account else (args.equity, "FIXED"))
    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    START = datetime.strptime(args.start, "%Y-%m-%d").date()
    END = datetime.strptime(args.end, "%Y-%m-%d").date()
    closes = {s: E.get_closes(s) for s in universe}
    cal = sorted(set().union(*[set(c) for c in closes.values()]))
    cal = [d for d in cal if START <= d <= END + timedelta(days=50)]
    entry_days = []
    d = START
    while d <= END:
        t = next((x for x in cal if x >= d), None)
        if t and t not in entry_days: entry_days.append(t)
        d += timedelta(days=7)

    # NDX-proxy regime from QQQ (market-wide; applied to all names)
    regime_map = {}
    if args.regime_switch:
        if args.regime_mode == "score":
            import regime_score
            regime_map = regime_score.compute()      # bull / cash / bear (multi-factor)
        elif args.regime_mode == "trend200":
            regime_map = build_regime_200(E.get_closes("QQQ"), band=args.trend_band)  # 200DMA primary-trend gate
        else:
            regime_map = build_regime(E.get_closes("QQQ"))   # bull / bear (EMA only)

    equity = eq0; cash = eq0
    open_pos: list[Spread] = []
    trades=[]; eq_rows=[]; realized=0.0; peak=eq0; maxdd=0.0
    regime_days={"bull":0,"bear":0,"cash":0}; peak_risk=0.0
    # regime distribution over ALL entry days in the window
    for ed in entry_days:
        regime_days[regime_map.get(ed,"bull")] = regime_days.get(regime_map.get(ed,"bull"),0)+1

    def px(sym, d):
        c = closes[sym]
        if d in c: return c[d]
        prior=[x for x in c if x<=d]
        return c[prior[-1]] if prior else None

    for today in cal:
        # ---- manage open positions ----
        still=[]
        for sp in open_pos:
            S = px(sp.sym, today)
            if S is None: still.append(sp); continue
            V = value_to_close(sp, S, today)
            dte = (sp.exp - today).days
            pnl_per = (sp.credit - V)                      # per share
            reason=None
            if V <= (1-args.tp_frac)*sp.credit: reason="take_profit"
            elif (V - sp.credit) >= args.sl_mult*sp.credit: reason="stop_loss"
            elif dte <= args.time_stop_dte: reason="time_stop"
            elif dte <= 0: reason="expiry"
            if reason:
                # cap realized at defined max loss
                pnl_per = max(pnl_per, -(sp.width - sp.credit))
                pnl = pnl_per*100*sp.contracts
                cash += pnl; realized += pnl
                trades.append(dict(close=today.isoformat(), sym=sp.sym, kind=sp.kind, regime=sp.regime,
                    Ks=sp.Ks, Kl=sp.Kl, contracts=sp.contracts, credit=round(sp.credit,2),
                    exit_val=round(V,2), pnl=round(pnl,2), reason=reason, held_days=(today-sp.opened).days))
            else:
                still.append(sp)
        open_pos=still

        # ---- entries on weekly cadence ----
        if today in entry_days:
            risk_cap_total = equity*args.exposure_pct
            cur_risk = sum(s.entry_risk for s in open_pos)
            for sym in universe:
                if any(s.sym==sym for s in open_pos): continue
                if len(open_pos) >= args.max_positions: break
                rows = E.raw_chain(sym, today)
                if not rows: continue
                exps={}
                for rr in rows:
                    e=rr.get("expiration","")[:10]
                    if not e: continue
                    try: dte=(date.fromisoformat(e)-today).days
                    except: continue
                    if 30<=dte<=45: exps[e]=exps.get(e,0)+1
                if not exps: continue
                exp=max(exps,key=exps.get)
                reg = regime_map.get(today, "bull")
                if reg == "cash":
                    continue   # transition regime → stand aside (this is where 2022 bled)
                # bull → bull PUT credit spread (sell 30Δ/buy 15Δ put)
                # bear → bear CALL credit spread (sell 30Δ/buy 15Δ call) — same underlying
                if reg == "bear":
                    short, lng, kind = E.pick(rows,"call",0.30,exp), E.pick(rows,"call",0.15,exp), "call"
                else:
                    short, lng, kind = E.pick(rows,"put",0.30,exp), E.pick(rows,"put",0.15,exp), "put"
                if not (short and lng): continue
                Ks,Kl=E.f(short.get("strike")),E.f(lng.get("strike"))
                if kind=="put" and Ks<=Kl: continue      # put: short strike > long strike
                if kind=="call" and Ks>=Kl: continue      # call: short strike < long strike
                credit=E.f(short.get("bid"))-E.f(lng.get("ask"))
                if credit<=0: continue
                iv=E.f(short.get("implied_volatility")) or 0.3
                width=abs(Ks-Kl); risk_per=(width-credit)*100
                if risk_per<=0: continue
                budget = args.risk_budget * (0.5 if reg=="bear" else 1.0)   # half size in bear regime
                n=int(min(budget//risk_per, (risk_cap_total-cur_risk)//risk_per))
                if n<1: continue
                exp_d=date.fromisoformat(exp)
                sp=Spread(sym,kind,Ks,Kl,credit,iv,exp_d,n,today,reg)
                open_pos.append(sp); cur_risk+=sp.entry_risk
                trades.append(dict(open=today.isoformat(), sym=sym, kind=kind, regime=reg, Ks=Ks, Kl=Kl,
                    contracts=n, credit=round(credit,2), iv=round(iv,3), dte=(exp_d-today).days,
                    max_risk=round(risk_per*n,2)))

        # ---- mark equity ----
        mtm = cash
        for sp in open_pos:
            S=px(sp.sym,today)
            if S is None: continue
            V=value_to_close(sp, S, today)
            mtm += (sp.credit - V)*100*sp.contracts   # unrealized
        equity = mtm
        peak_risk=max(peak_risk, sum(s.entry_risk for s in open_pos))
        peak=max(peak,equity); maxdd=max(maxdd,(peak-equity)/peak*100 if peak>0 else 0)
        eq_rows.append(dict(date=today.isoformat(), equity=round(equity,2), open=len(open_pos)))

    # ---- summary ----
    closed=[t for t in trades if "pnl" in t]
    wins=[t for t in closed if t["pnl"]>0]
    pnls=[t["pnl"] for t in closed]
    n=len(closed)
    days=max((cal[-1]-cal[0]).days,1) if cal else 1
    sd=statistics.pstdev(pnls) if n>1 else 0
    out=Path(ROOT)/"backtest_outputs"/args.run_tag; out.mkdir(parents=True,exist_ok=True)
    import csv
    with open(out/"trades.csv","w",newline="") as fcsv:
        w=csv.DictWriter(fcsv, fieldnames=sorted({k for t in trades for k in t})); w.writeheader(); w.writerows(trades)
    with open(out/"equity_curve.csv","w",newline="") as fcsv:
        w=csv.DictWriter(fcsv, fieldnames=["date","equity","open"]); w.writeheader(); w.writerows(eq_rows)
    summary=dict(run_tag=args.run_tag, mode=mode, universe=universe, start_equity=eq0,
        final_equity=round(equity,2), total_pnl=round(realized,2),
        return_pct=round(100*(equity-eq0)/eq0,2), closed_trades=n,
        win_rate=round(100*len(wins)/n,1) if n else 0,
        expectancy=round(sum(pnls)/n,1) if n else 0,
        avg_win=round(sum(t["pnl"] for t in wins)/len(wins),1) if wins else 0,
        avg_loss=round(sum(t["pnl"] for t in closed if t["pnl"]<=0)/max(1,n-len(wins)),1),
        worst=round(min(pnls),1) if pnls else 0, max_drawdown_pct=round(maxdd,2),
        regime_entry_days=regime_days, peak_concurrent_risk=round(peak_risk,0),
        pnl_per_calendar_day=round(realized/days,1),
        exit_mix={r:sum(1 for t in closed if t["reason"]==r) for r in set(t["reason"] for t in closed)} if closed else {},
        by_kind={kd:{"n":sum(1 for t in closed if t.get("kind")==kd),
                     "pnl":round(sum(t["pnl"] for t in closed if t.get("kind")==kd),0)}
                 for kd in set(t.get("kind") for t in closed)} if closed else {},
        params=dict(risk_budget=args.risk_budget, exposure_pct=args.exposure_pct,
            tp_frac=args.tp_frac, sl_mult=args.sl_mult, time_stop_dte=args.time_stop_dte,
            max_positions=args.max_positions,
            regime_mode=(args.regime_mode if args.regime_switch else "off")))
    json.dump(summary, open(out/"summary.json","w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("CSE_DONE")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Defined-risk credit-spread engine on real chains.")
    p.add_argument("--symbols", default="SPY,QQQ,IWM")
    p.add_argument("--start", default="2025-06-02")
    p.add_argument("--end", default="2026-04-27")
    p.add_argument("--account", action="store_true", help="use live Alpaca equity as capital")
    p.add_argument("--equity", type=float, default=6260.0)
    p.add_argument("--risk-budget", dest="risk_budget", type=float, default=1000.0, help="max $ risk per spread")
    p.add_argument("--exposure-pct", dest="exposure_pct", type=float, default=0.35)
    p.add_argument("--tp-frac", dest="tp_frac", type=float, default=0.5, help="take profit at this fraction of credit")
    p.add_argument("--sl-mult", dest="sl_mult", type=float, default=2.0, help="stop at this multiple of credit loss")
    p.add_argument("--time-stop-dte", dest="time_stop_dte", type=int, default=21)
    p.add_argument("--max-positions", dest="max_positions", type=int, default=6)
    p.add_argument("--no-regime-switch", dest="regime_switch", action="store_false",
                   help="disable regime switch (bull put spreads only)")
    p.add_argument("--regime-mode", dest="regime_mode", choices=["ema","score","trend200"], default="score",
                   help="ema=binary 20/50; score=multi-factor bull/cash/bear; trend200=200DMA primary-trend gate")
    p.add_argument("--trend-band", dest="trend_band", type=float, default=0.0,
                   help="cash band +/- around the 200DMA for trend200 (e.g. 0.01 = +/-1%)")
    p.set_defaults(regime_switch=True)
    p.add_argument("--run-tag", dest="run_tag", default="credit_spread_engine")
    run(p.parse_args())
