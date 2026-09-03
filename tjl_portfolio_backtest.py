#!/usr/bin/env python3
"""
tjl_portfolio_backtest.py — "what would my actual account have done?" backtest.

Runs the TJL strategy across a universe over a long window on REAL dollar equity:
shared capital, N equal-weight concurrent slots, compounding, fills at the signal bar's
close. Reports the dollar equity curve, drawdown, Sharpe, and SPY buy & hold over the
same window. Default config = the ORIGINAL TJL (intraday: 1.5% stop / 3% TP / EOD-flat,
no swing/trail) — flip TJL_SWING_MODE/TJL_USE_TRAIL to test the current config instead.

Faithful + fast: indicators are computed ONCE per symbol with tjl_signal's causal
primitives (ema/sma/stoch are causal, so full-series values equal truncated-series
values), and HTF bars are mapped to the last COMPLETED bar via a moving pointer — same
non-repaint contract as the engine, but O(n) instead of the per-bar O(n^2) harness.

Usage: python tjl_portfolio_backtest.py [SYM,SYM,...] [days] [start_equity] [slots]
"""
from __future__ import annotations
import os, sys
from datetime import timedelta, timezone
from pathlib import Path
import heapq

import tjl_signal as T
import tjl_backtest as B

ROOT = Path(__file__).resolve().parent
ET = B.ET

UNIVERSE = (sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1]
            else ["AMD", "NVDA", "MU", "AAPL", "MSFT"])
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 365
START_EQUITY = float(sys.argv[3]) if len(sys.argv) > 3 else 100_000.0
SLOTS = int(sys.argv[4]) if len(sys.argv) > 4 else 10   # max concurrent equal-weight positions


def symbol_trades(sym: str, cfg: T.TJLConfig, days: int):
    """All TJL trades for one symbol over the window, as (entry_t, exit_t, ret_pct).
    Indicators precomputed once (causal) → faithful to tjl_signal, O(n)."""
    e, src = B.get_bars(sym, 5, days)
    if src != "alpaca" or len(e) < 50:
        return [], {}, src
    # HTF needs warmup BEFORE the window (SMA200 on 60m ≈ 30 trading days); over-fetch,
    # bounded so a 1y window doesn't pull 6y of bars.
    m = B.get_bars(sym, cfg.momo_tf_min, min(days * 2, days + 25))[0]
    t = B.get_bars(sym, cfg.trend_tf_min, min(days * 6, days + 90))[0]
    # HTF booleans per completed bar
    tcl = [b.c for b in t]
    tsma = T.sma_series(tcl, cfg.trend_sma_len)
    trend_ok = [tsma[j] is not None and tcl[j] > tsma[j] for j in range(len(t))]
    mk, md = T.stoch_kd([b.h for b in m], [b.l for b in m], [b.c for b in m],
                        cfg.stoch_len, cfg.smooth_k, cfg.smooth_d)
    momo_ok = [mk[j] is not None and md[j] is not None and mk[j] > md[j] and mk[j] < cfg.stoch_ob
               for j in range(len(m))]
    ecl = [b.c for b in e]
    ef = T.ema_series(ecl, cfg.fast_ema)
    es = T.ema_series(ecl, cfg.slow_ema)

    # prior-day RTH high per ET date (trail arming) + daily RTH close (mark-to-market)
    _dh, daily_close = {}, {}
    for _b in e:
        if T.is_rth(_b.t):
            _dk = _b.t.astimezone(ET).date()
            _dh[_dk] = max(_dh.get(_dk, _b.h), _b.h)
            daily_close[_dk] = _b.c
    _ds = sorted(_dh)
    prev_day_high = {dd: _dh[_ds[j - 1]] for j, dd in enumerate(_ds) if j > 0}

    trd, mom = cfg.trend_tf_min, cfg.momo_tf_min
    trades = []
    in_pos = False
    ep = et = hw = None
    armed = not cfg.trail_require_pdh
    pt = pm = 0
    for i, b in enumerate(e):
        while pt < len(t) and t[pt].t <= b.t - timedelta(minutes=trd):
            pt += 1
        while pm < len(m) and m[pm].t <= b.t - timedelta(minutes=mom):
            pm += 1
        if in_pos:
            stop = T.trailing_stop_level(cfg, ep, hw, armed=armed)
            tp = ep * (1 + cfg.tp_pct)
            prev = e[i - 1] if i > 0 else None
            eod = (cfg.eod_flat and not cfg.swing_mode and prev is not None
                   and T.is_rth(prev.t) and not T.is_rth(b.t))
            if b.l <= stop:
                trades.append((et, b.t, ep, stop)); in_pos = False
            elif (not cfg.use_trail) and b.h >= tp:
                trades.append((et, b.t, ep, tp)); in_pos = False
            elif eod:
                trades.append((et, b.t, ep, b.c)); in_pos = False
            elif cfg.use_trail:
                hw = b.c if hw is None else max(hw, b.c)
                if cfg.trail_require_pdh and not armed:        # arm on a close above PDH
                    pdh = prev_day_high.get(b.t.astimezone(ET).date())
                    if pdh is not None and b.c > pdh:
                        armed = True
            continue
        if not T.is_rth(b.t) or i < 1:
            continue
        if pt == 0 or not trend_ok[pt - 1]:
            continue
        if pm == 0 or not momo_ok[pm - 1]:
            continue
        if not (ef[i - 1] <= es[i - 1] and ef[i] > es[i]):   # crossover up
            continue
        in_pos, ep, et, hw = True, b.c, b.t, b.c
        armed = not cfg.trail_require_pdh
    return trades, daily_close, src


def metrics(curve):
    """curve: sorted list of (datetime, equity). Returns dict of $/%/Sharpe/maxDD."""
    import math
    eqs = [v for _, v in curve]
    e0, e1 = eqs[0], eqs[-1]
    peak = -1e18; mdd = 0.0
    for v in eqs:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak > 0 else 0.0)
    # daily resample for vol/Sharpe
    daily = {}
    for ts, v in curve:
        daily[ts.astimezone(ET).date()] = v
    dv = [daily[d] for d in sorted(daily)]
    rets = [dv[i] / dv[i - 1] - 1 for i in range(1, len(dv)) if dv[i - 1] > 0]
    mu = sum(rets) / len(rets) if rets else 0.0
    sd = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5 if len(rets) > 1 else 0.0
    sharpe = (mu / sd * math.sqrt(252)) if sd > 0 else 0.0
    years = max((curve[-1][0] - curve[0][0]).days / 365.25, 1e-9)
    cagr = (e1 / e0) ** (1 / years) - 1 if e0 > 0 else 0.0
    return {"final": e1, "ret": (e1 / e0 - 1) * 100, "cagr": cagr * 100,
            "vol": sd * math.sqrt(252) * 100, "sharpe": sharpe, "maxdd": mdd * 100}


def _spy_daily(days):
    """SPY daily closes over the window as [(datetime, close)] for the benchmark."""
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not (key and sec):
        return []
    from datetime import datetime, timezone as _tz, timedelta as _td
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(key, sec)
    end = datetime.now(_tz.utc); start = end - _td(days=days + 5)
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day,
                                           start=start, end=end, feed=DataFeed.IEX)).df
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.reset_index().iterrows():
        ts = r["timestamp"]; ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append((ts, float(r["close"])))
    return out


def portfolio(trades, daily_px, calendar, e0, slots, slip_frac):
    """Replay trades through shared capital: <= `slots` concurrent equal-weight positions,
    each sized equity/slots at entry, compounding. Open positions are MARKED TO MARKET at
    every trading day's close (daily_px[sym][date]), so the equity curve and drawdown
    include intra-trade overnight/multi-day unrealized swings (not just realized P&L).
    Round-trip slippage (fraction) is charged on the cost basis at exit. Capacity-limited
    trades (no free slot/cash) are skipped. Returns (curve, taken, skipped).
    trades: list of (sym, entry_t, exit_t, entry_px, exit_px)."""
    import datetime as _dt
    events = []                          # (ts, priority, key) — exits(0) < entries(1) < marks(2)
    for k, (sym, et, xt, epx, xpx) in enumerate(trades):
        events.append((et, 1, k))
        events.append((xt, 0, k))
    for d in calendar:
        events.append((_dt.datetime(d.year, d.month, d.day, 23, 0, tzinfo=timezone.utc), 2, d))
    events.sort(key=lambda x: (x[0], x[1]))

    cash = e0
    openp = {}                           # key -> (sym, shares, entry_px)
    last_px = {}                         # sym -> last seen price (mark/size)
    curve = []
    taken = skipped = 0

    def _equity():
        return cash + sum(sh * last_px.get(s, epx) for (s, sh, epx) in openp.values())

    for ts, kind, key in events:
        if kind == 0:                                    # EXIT (realize)
            if key in openp:
                sym, sh, epx = openp.pop(key)
                xpx = trades[key][4]
                last_px[sym] = xpx
                cash += sh * xpx - slip_frac * sh * epx   # round-trip slippage on cost basis
                curve.append((ts, _equity()))
        elif kind == 1:                                  # ENTRY (decide taken/skipped)
            sym, et, xt, epx, xpx = trades[key]
            last_px[sym] = epx
            if len(openp) < slots and cash > 1.0:
                alloc = min(cash, _equity() / slots)
                if alloc >= 1.0:
                    openp[key] = (sym, alloc / epx, epx)
                    cash -= alloc
                    taken += 1
                    curve.append((ts, _equity()))
                    continue
            skipped += 1
        else:                                            # MARK (daily close → unrealized)
            for s, dpx in daily_px.items():
                if key in dpx:
                    last_px[s] = dpx[key]
            curve.append((ts, _equity()))
    curve.sort(key=lambda x: x[0])
    return (curve or [(events[0][0] if events else None, e0)]), taken, skipped


def main():
    cfg = T.TJLConfig.from_env()
    mode = ("swing+trail" + ("(PDH)" if cfg.trail_require_pdh else "")) if cfg.use_trail else "ORIGINAL intraday"
    print(f"TJL portfolio backtest — {len(UNIVERSE)} names · {DAYS}d · ${START_EQUITY:,.0f} start · "
          f"{SLOTS} equal-weight slots")
    print(f"  config: {mode} | stop {cfg.stop_pct*100:.1f}% / tp {cfg.tp_pct*100:.1f}% / "
          f"eod_flat={cfg.eod_flat and not cfg.swing_mode}")
    all_tr, daily_px, got = [], {}, 0
    for sym in UNIVERSE:
        trs, dclose, src = symbol_trades(sym, cfg, DAYS)
        if src == "alpaca":
            got += 1
            for (et, xt, epx, xpx) in trs:
                all_tr.append((sym, et, xt, epx, xpx))
            daily_px[sym] = dclose
    if not all_tr:
        print("no trades"); return
    t0 = min(x[1] for x in all_tr)
    t1 = max(x[2] for x in all_tr)
    d0, d1 = t0.astimezone(ET).date(), t1.astimezone(ET).date()
    calendar = sorted({d for dc in daily_px.values() for d in dc if d0 <= d <= d1})

    spy = [(ts, px) for ts, px in _spy_daily(DAYS) if t0 <= ts <= t1 + timedelta(days=2)]
    spy_curve = [(ts, START_EQUITY * px / spy[0][1]) for ts, px in spy] if len(spy) > 2 else []
    sm = metrics(spy_curve) if spy_curve else None

    base_curve = None
    _t = _s = 0
    rows_data = []
    print(f"\nWINDOW {t0.astimezone(ET).date()} -> {t1.astimezone(ET).date()}  "
          f"({(t1 - t0).days/365.25:.2f}y) · {got}/{len(UNIVERSE)} names priced")
    hdr = f"{'strategy':<30}{'final $':>14}{'tot ret':>10}{'CAGR':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    # round-trip slippage/cost sensitivity (bps charged on cost basis at exit)
    slip_bps = [float(x) for x in (os.environ.get("TJL_SLIP_GRID", "0,5,10").split(","))]
    for slip in slip_bps:
        curve, taken, skipped = portfolio(all_tr, daily_px, calendar, START_EQUITY, SLOTS, slip / 10000.0)
        m = metrics(curve)
        if base_curve is None:
            base_curve, _t, _s = curve, taken, skipped
        rows_data.append({"slip_bps": slip, "final": round(m["final"]), "ret": round(m["ret"], 1),
                          "cagr": round(m["cagr"], 1), "vol": round(m["vol"], 1),
                          "sharpe": round(m["sharpe"], 2), "maxdd": round(m["maxdd"], 1)})
        print(f"{('TJL '+mode+' · '+str(int(slip))+'bp slip'):<30}{m['final']:>14,.0f}{m['ret']:>9.1f}%"
              f"{m['cagr']:>8.1f}%{m['vol']:>7.1f}%{m['sharpe']:>8.2f}{m['maxdd']:>8.1f}%")
    if sm:
        print(f"{'SPY buy & hold':<30}{sm['final']:>14,.0f}{sm['ret']:>9.1f}%{sm['cagr']:>8.1f}%"
              f"{sm['vol']:>7.1f}%{sm['sharpe']:>8.2f}{sm['maxdd']:>8.1f}%")
    print(f"\nsignals {len(all_tr)} · taken {_t} · skipped (no free slot/cash) {_s}")

    # ── JSON artifact for the dashboard (daily equity curves, normalized x in 0..1) ──
    def _daily(curve):
        dd = {}
        for ts, v in curve:
            dd[ts.astimezone(ET).date()] = v
        ks = sorted(dd)
        if len(ks) < 2:
            return []
        span = (ks[-1] - ks[0]).days or 1
        return [{"x": round((d - ks[0]).days / span, 4), "v": round(dd[d], 0)} for d in ks]
    import json as _json
    doc = {
        "strategy": "TJL portfolio", "config": mode, "start_equity": START_EQUITY,
        "slots": SLOTS, "days": DAYS, "names": got,
        "window": {"start": d0.isoformat(), "end": d1.isoformat()},
        "signals": len(all_tr), "taken": _t, "skipped": _s,
        "table": rows_data,
        "spy": {"ret": round(sm["ret"], 1), "sharpe": round(sm["sharpe"], 2),
                "maxdd": round(sm["maxdd"], 1), "final": round(sm["final"])} if sm else None,
        "equity_curve": _daily(base_curve),
        "spy_curve": _daily(spy_curve) if spy_curve else [],
    }
    outp = ROOT / "backtest_outputs" / os.environ.get("TJL_PF_OUT", "tjl_portfolio.json")
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(_json.dumps(doc, indent=2))
    print(f"\nartifact -> {outp}")


if __name__ == "__main__":
    main()
