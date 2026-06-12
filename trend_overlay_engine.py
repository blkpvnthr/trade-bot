#!/usr/bin/env python3
"""trend_overlay_engine.py — underlying for trend, options only for crashes.

The session's core finding: profitable long-CALL trades were really harvesting trend/
direction while PAYING theta+spread+IV to express it. So express the bull trend with the
UNDERLYING (no theta), and reserve options for the one place convexity actually pays — bear
downtrends/crashes.

  REGIME (QQQ 200DMA ±2%):
    BULL  -> LONG UNDERLYING shares (QQQ/SPY/IWM, equal weight, `exposure` of equity)
    BEAR  -> LONG ~45Δ PUTS (real AV chains; RVOL>=2 reversal / 7-DTE / regime-flip exits)
    TRANSITION -> CASH

Shares entered on any bull day (held while bull), exited when regime leaves bull. Puts
entered weekly in bear. Real chains, BS-marked puts, conservative fills. Same 5-window OOS
discipline + full metric panel as the directional engine.
"""
from __future__ import annotations
import sys, json, argparse
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = "/Users/blkpvnthr/Desktop/trader"; sys.path.insert(0, ROOT)
import edge_scan_spreads as E
import credit_spread_engine as C
import directional_volexit_engine as D   # get_ohlcv, pick_band, opt_value, portfolio_metrics, Pos


def run(args):
    eq0 = args.equity
    uni = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    START = datetime.strptime(args.start, "%Y-%m-%d").date(); END = datetime.strptime(args.end, "%Y-%m-%d").date()
    ohlcv = {s: D.get_ohlcv(s) for s in uni}
    cal = sorted(set().union(*[set(o) for o in ohlcv.values()])); cal = [d for d in cal if START <= d <= END + timedelta(days=60)]
    regime = C.build_regime_200(E.get_closes("QQQ"), band=args.trend_band)
    # 200-SMA slope on QQQ: only go long the underlying when the long-term trend is RISING
    from collections import deque as _dq
    _qqq = E.get_closes("QQQ"); _qd = sorted(_qqq); _w = _dq(maxlen=200); _sma = {}
    for _dd in _qd:
        _w.append(_qqq[_dd]); _sma[_dd] = (sum(_w) / len(_w)) if len(_w) == 200 else None
    def rising200(dd):
        days = [x for x in _qd if x <= dd]
        if len(days) < 221: return True
        cur, prev = _sma[days[-1]], _sma[days[-21]]
        return bool(cur and prev and cur > prev)
    entry_days = []; d = START
    while d <= END:
        t = next((x for x in cal if x >= d), None)
        if t and t not in entry_days: entry_days.append(t)
        d += timedelta(days=7)
    rb = args.risk_budget * args.size_mult

    def close(sym, dd):
        o = ohlcv[sym]
        if dd in o: return o[dd][1]
        prior = [x for x in o if x <= dd]; return o[prior[-1]][1] if prior else None
    def rvol(sym, dd):
        o = ohlcv[sym]; days = [x for x in sorted(o) if x <= dd]
        if len(days) < 21: return 0.0
        avg = sum(o[x][2] for x in days[-21:-1]) / 20
        return o[days[-1]][2] / avg if avg > 0 else 0.0

    cash = eq0; realized = 0.0; peak = eq0; maxdd = 0.0
    shares = {}   # sym -> (qty, entry_price)
    puts = []     # D.Pos (long_put)
    trades = []; eq_rows = []

    for today in cal:
        reg = regime.get(today, "bull")

        # ---- shares: exit if no longer bull ----
        if reg != "bull" and shares:
            for sym, (qty, entry) in list(shares.items()):
                px = close(sym, today)
                if px is None: continue
                pnl = qty * (px - entry); cash += qty * px; realized += pnl
                trades.append(dict(close=today.isoformat(), sym=sym, kind="shares", qty=qty,
                                   entry=round(entry, 2), exitv=round(px, 2), pnl=round(pnl, 2), reason="regime_exit"))
            shares = {}

        # ---- puts: manage (vol-reversal / 7DTE / not-bear) ----
        keep = []
        for p in puts:
            o = ohlcv.get(p.sym, {})
            if today not in o: keep.append(p); continue
            op, c, _ = o[today]; days = [x for x in sorted(o) if x < today]
            pc = o[days[-1]][1] if days else c
            rv = rvol(p.sym, today); dte = (p.exp - today).days
            S = c; V = D.opt_value(p, S, today)
            green = c > op and c > pc
            reason = None
            if rv >= args.rvol_th and green: reason = "vol_reversal"
            elif dte <= args.time_stop_dte: reason = "time_stop"
            elif reg != "bear": reason = "regime_exit"
            if reason:
                pnl = (V - p.entry) * 100 * p.contracts; cash += V * 100 * p.contracts; realized += pnl
                trades.append(dict(close=today.isoformat(), sym=p.sym, kind="long_put", qty=p.contracts,
                                   entry=round(p.entry, 2), exitv=round(V, 2), pnl=round(pnl, 2), reason=reason))
            else: keep.append(p)
        puts = keep

        # ---- entries ----
        if reg == "bull" and (not args.require_rising_200 or rising200(today)):
            target = args.exposure * eq0 / len(uni)
            for sym in uni:
                if sym in shares: continue
                px = close(sym, today)
                if not px: continue
                qty = int(target // px)
                if qty < 1: continue
                cash -= qty * px; shares[sym] = (qty, px)
                trades.append(dict(open=today.isoformat(), sym=sym, kind="shares", qty=qty, entry=round(px, 2)))
        elif reg == "bear" and today in entry_days and not args.no_puts:
            for sym in uni:
                if any(x.sym == sym for x in puts): continue
                rows = E.raw_chain(sym, today)
                if not rows: continue
                exps = {}
                for r in rows:
                    e = (r.get("expiration") or "")[:10]
                    if not e: continue
                    try: dt = (date.fromisoformat(e) - today).days
                    except: continue
                    if 30 <= dt <= 45: exps[e] = exps.get(e, 0) + 1
                if not exps: continue
                exp = max(exps, key=exps.get)
                p45 = D.pick_band(rows, "put", 0.40, 0.55, exp)
                if not p45: continue
                ask = E.f(p45.get("ask")); n = int(rb // (ask * 100)) if ask > 0 else 0
                if n < 1: continue
                cash -= ask * 100 * n
                puts.append(D.Pos(sym, "long_put", E.f(p45.get("strike")), E.f(p45.get("implied_volatility")) or .3,
                                  date.fromisoformat(exp), n, ask, today, False))
                trades.append(dict(open=today.isoformat(), sym=sym, kind="long_put", qty=n, entry=round(ask, 2)))

        # ---- equity mark ----
        mtm = cash
        for sym, (qty, _e) in shares.items():
            px = close(sym, today)
            if px: mtm += qty * px
        for p in puts:
            o = ohlcv.get(p.sym, {})
            if today in o: mtm += D.opt_value(p, o[today][1], today) * 100 * p.contracts
        peak = max(peak, mtm); maxdd = max(maxdd, (peak - mtm) / peak * 100 if peak > 0 else 0)
        eq_rows.append(dict(date=today.isoformat(), equity=round(mtm, 2), open=len(shares) + len(puts)))

    # ---- flatten open positions at window end (so realized = true mark-to-market) ----
    last = cal[-1]
    for sym, (qty, entry) in shares.items():
        px = close(sym, last)
        if px:
            realized += qty * (px - entry); cash += qty * px
            trades.append(dict(close=last.isoformat(), sym=sym, kind="shares", qty=qty,
                               entry=round(entry, 2), exitv=round(px, 2), pnl=round(qty * (px - entry), 2), reason="eow_flatten"))
    shares = {}
    for p in puts:
        o = ohlcv.get(p.sym, {})
        if last in o:
            V = D.opt_value(p, o[last][1], last); pnl = (V - p.entry) * 100 * p.contracts
            realized += pnl; cash += V * 100 * p.contracts
            trades.append(dict(close=last.isoformat(), sym=p.sym, kind="long_put", qty=p.contracts,
                               entry=round(p.entry, 2), exitv=round(V, 2), pnl=round(pnl, 2), reason="eow_flatten"))
    puts = []

    closed = [t for t in trades if "pnl" in t]; pnls = [t["pnl"] for t in closed]; n = len(closed)
    wins = [t for t in closed if t["pnl"] > 0]; days = max((cal[-1] - cal[0]).days, 1) if cal else 1
    by_kind = {k: {"n": sum(1 for t in closed if t.get("kind") == k), "pnl": round(sum(t["pnl"] for t in closed if t.get("kind") == k), 0)}
               for k in set(t.get("kind") for t in closed)} if closed else {}
    out = Path(ROOT) / "backtest_outputs" / args.run_tag; out.mkdir(parents=True, exist_ok=True)
    summary = dict(run_tag=args.run_tag, universe=uni, window=f"{START}..{END}", start_equity=eq0,
                   final_equity=round(eq0 + realized, 2), total_pnl=round(realized, 2),
                   return_pct=round(100 * realized / eq0, 2), closed_trades=n,
                   win_rate=round(100 * len(wins) / n, 1) if n else 0,
                   max_drawdown_pct=round(maxdd, 2), by_kind=by_kind, metrics=D.portfolio_metrics(eq_rows, eq0))
    json.dump(summary, open(out / "summary.json", "w"), indent=2); print(json.dumps(summary, indent=2)); print("TREND_DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="QQQ,SPY,IWM")
    p.add_argument("--start", default="2025-06-02"); p.add_argument("--end", default="2026-04-27")
    p.add_argument("--equity", type=float, default=100000.0)
    p.add_argument("--exposure", type=float, default=1.0, help="bull underlying exposure (fraction of equity)")
    p.add_argument("--risk-budget", dest="risk_budget", type=float, default=2000.0)
    p.add_argument("--size-mult", dest="size_mult", type=float, default=0.75)
    p.add_argument("--rvol-th", dest="rvol_th", type=float, default=2.0)
    p.add_argument("--time-stop-dte", dest="time_stop_dte", type=int, default=7)
    p.add_argument("--trend-band", dest="trend_band", type=float, default=0.02)
    p.add_argument("--require-rising-200", dest="require_rising_200", action="store_true",
                   help="only buy shares when 200DMA slope is rising (kills bear-rally whipsaw)")
    p.add_argument("--no-puts", dest="no_puts", action="store_true",
                   help="shares-only: hold cash in bear instead of buying ~45Δ puts (drops options dependency)")
    p.add_argument("--run-tag", dest="run_tag", default="trend")
    run(p.parse_args())
