#!/usr/bin/env python3
"""
backtest_bal_gates_off.py — replay the GATES-OFF Buffett Accumulation Layer (BAL) DCA config
on $1,000,000 over real daily bars.

"Gates-off" config (instance2 / nonl0cal.2 study):
  BAL_DCA_BYPASS_BUFFETT_SCORE=true   -> quality/score gate skipped
  BAL_DCA_BYPASS_VALUATION=true       -> valuation (p/iv) gate skipped
  BAL_MIN_SCORE_TO_BUY=0, BAL_CONVICTION_SMALL=0, BAL_DCA_SMALL_SCORE bypassed
  MAX_ORDER_DOLLARS=15000

With the quality/valuation/composite/conviction gates removed, EVERY name in the 44-name
exclusive universe becomes eligible every cycle, so the engine DCAs into all of them in
parallel until the 80% stock-exposure cap binds -> it dilutes to a roughly equal-weight,
~80%-invested basket. The per-name conviction target (~4.5-7%) is never reached because the
total-exposure cap stops it first. The controls that STAY ON are faithfully modeled:
  - regime gate: new buys only when SPY > SPY 200-DMA  (step 0, kept)
  - max 5 tranches/name, >=3 trading days apart        (step 5, kept)
  - 10% per-name cap, 80% total stock-exposure cap     (step 5, kept)
  - 8% cash buffer                                     (step 5, kept)

Abstractions (stated honestly): conviction/p-iv weighting is not PIT-backtestable, so tranches
target equal weight (the gates-off limit). No score-based exits are modeled (gates-off is an
accumulation study; exits are score-driven and bypassed) -> the basket is held. Fills at the
daily close, dividend+split-adjusted (total return), zero commission (Alpaca).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache" / "bal_backtest"
CACHE.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["COMP","ARL","TCI","CWK","NMRK","MLP","VLY","ASB","UMBF","ONB","FULT","AUB",
            "BANF","FFBC","NUVL","PTGX","SYRE","TGTX","CELC","ORKA","AXSM","CRNX","ERAS",
            "CGON","DNTH","SRRK","KYMR","BLDR","CARR","MAS","LII","TT","JCI","TMO","RVTY",
            "DGX","LH","IDXX","DHR","BX","APO","ARES","KKR","BLK"]
BENCH = "SPY"

START = sys.argv[1] if len(sys.argv) > 1 else "2022-01-01"  # sim start (200-DMA warmup uses prior data)
FETCH_START = "2021-01-01"

# ── config (gates-off; controls that stay ON) ────────────────────────────────
START_CASH      = 1_000_000.0
MAX_EXPOSURE    = 0.80    # MAX_STOCK_EXPOSURE_PCT
CASH_BUFFER     = 0.08    # CASH_BUFFER_PCT
MAX_SINGLE      = 0.10    # MAX_SINGLE_STOCK_WEIGHT
MAX_TRANCHES    = 5       # BAL_MAX_DCA_COUNT_PER_SYMBOL
MIN_DAYS        = 3       # BAL_MIN_DAYS_BETWEEN_DCA (trading days)
MIN_ORDER       = 100.0   # BAL_DCA_MIN_ORDER_DOLLARS
MAX_ORDER       = 15000.0 # MAX_ORDER_DOLLARS (the fix)


def load_env():
    f = ROOT / ".env"
    if f.exists():
        for ln in f.read_text().splitlines():
            ln = ln.strip()
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch(symbols):
    """Return {symbol: Series(date->close)} from cache, fetching misses via Alpaca (total-return)."""
    load_env()
    out = {}
    need = []
    for s in symbols:
        cf = CACHE / f"{s}.parquet"
        if cf.exists():
            d = pd.read_parquet(cf)
            out[s] = d["close"]
        else:
            need.append(s)
    if need:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment, DataFeed
        cli = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
        for feed in (DataFeed.SIP, DataFeed.IEX):
            try:
                req = StockBarsRequest(symbol_or_symbols=need, timeframe=TimeFrame.Day,
                                       start=datetime(2021, 1, 1), adjustment=Adjustment.ALL, feed=feed)
                df = cli.get_stock_bars(req).df
                break
            except Exception as e:
                print(f"  feed {feed} failed: {repr(e)[:100]}")
                df = None
        if df is None or df.empty:
            raise RuntimeError("Alpaca fetch returned nothing")
        for s in need:
            try:
                sub = df.xs(s, level="symbol")
            except KeyError:
                print(f"  no data: {s}")
                continue
            ser = sub["close"].copy()
            ser.index = pd.to_datetime(ser.index).tz_convert("UTC").normalize().tz_localize(None)
            pd.DataFrame({"close": ser}).to_parquet(CACHE / f"{s}.parquet")
            out[s] = ser
    # normalize all indices to naive date
    for s in list(out.keys()):
        ser = out[s]
        idx = pd.to_datetime(ser.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC")
            idx = idx.tz_localize(None)
        ser.index = idx.normalize()
        out[s] = ser[~ser.index.duplicated(keep="last")]
    return out


def metrics(equity: pd.Series, label: str) -> dict:
    eq = equity.dropna()
    ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    days = (eq.index[-1] - eq.index[0]).days or 1
    yrs = days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1.0 if yrs > 0 else 0.0
    dr = eq.pct_change().dropna()
    vol = dr.std() * np.sqrt(252)
    sharpe = (dr.mean() * 252) / vol if vol > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {"label": label, "final": eq.iloc[-1], "total_return": ret, "cagr": cagr,
            "vol": vol, "sharpe": sharpe, "max_dd": dd}


def main():
    print("Fetching daily bars (total-return) for 44 + SPY ...")
    data = fetch(UNIVERSE + [BENCH])
    spy = data.pop(BENCH)
    names = [s for s in UNIVERSE if s in data]
    print(f"  have {len(names)}/{len(UNIVERSE)} names + SPY")
    missing = [s for s in UNIVERSE if s not in data]
    if missing:
        print(f"  missing (skipped): {missing}")

    # price panel
    panel = pd.DataFrame({s: data[s] for s in names}).sort_index()
    spy = spy.sort_index()
    spy_ma = spy.rolling(200).mean()

    cal = panel.loc[START:].index
    cal = cal[cal.isin(spy.index)]

    # survivorship test: drop the top-N names by their own return over the window
    DROP_TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dropped = []
    if DROP_TOP > 0:
        rets = {}
        for s in names:
            ser = panel[s].reindex(cal).dropna()
            if len(ser) > 1 and ser.iloc[0] > 0:
                rets[s] = ser.iloc[-1] / ser.iloc[0] - 1.0
        ranked = sorted(rets.items(), key=lambda x: -x[1])
        dropped = [s for s, _ in ranked[:DROP_TOP]]
        names = [s for s in names if s not in dropped]
        panel = panel[names]
        print("DROPPED top-%d by window return: %s" % (
            DROP_TOP, ", ".join("%s +%.0f%%" % (s, rets[s] * 100) for s in dropped)))

    # ── ablation toggles (env-driven); defaults = gates-off baseline ──────────
    rgate     = os.getenv("ABL_REGIME", "1") == "1"          # SPY>200DMA regime gate
    name_cap  = MAX_SINGLE if os.getenv("ABL_NAMECAP", "1") == "1" else 1.0
    exp_cap   = MAX_EXPOSURE if os.getenv("ABL_EXPCAP", "1") == "1" else 1.0
    max_tr    = MAX_TRANCHES if os.getenv("ABL_TRANCHES", "1") == "1" else 10**6
    min_d     = MIN_DAYS if os.getenv("ABL_COOLDOWN", "1") == "1" else 0
    buf       = CASH_BUFFER if os.getenv("ABL_BUFFER", "1") == "1" else 0.0
    trendfilt = os.getenv("ABL_TRENDFILT", "0") == "1"       # proxy quality/momentum gate
    dipfilt   = os.getenv("ABL_DIPFILT", "0") == "1"         # proxy valuation/dislocation gate
    label     = os.getenv("ABL_LABEL", "baseline (gates-off)")
    ma200 = panel.rolling(200).mean()
    ma50 = panel.rolling(50).mean()

    shares = {s: 0.0 for s in names}
    last_buy = {s: -999 for s in names}
    tranches = {s: 0 for s in names}
    cash = START_CASH
    eq_curve, invested_curve = [], []

    for i, day in enumerate(cal):
        px = panel.loc[day]
        live = {s: float(px[s]) for s in names if pd.notna(px[s]) and px[s] > 0}
        pos_val = sum(shares[s] * live.get(s, 0.0) for s in names)
        equity = cash + pos_val

        regime_ok = (not rgate) or bool(
            day in spy_ma.index and pd.notna(spy_ma.loc[day]) and spy.loc[day] > spy_ma.loc[day])
        n_elig = len(live)
        if regime_ok and n_elig > 0:
            w_target = min(name_cap, exp_cap / n_elig)
            for s in sorted(live.keys()):
                price = live[s]
                if trendfilt:
                    m = ma200.loc[day, s] if day in ma200.index else None
                    if not (pd.notna(m) and price > m):  continue
                if dipfilt:
                    m = ma50.loc[day, s] if day in ma50.index else None
                    if not (pd.notna(m) and price < m):   continue
                pos_val = sum(shares[k] * live.get(k, 0.0) for k in names)
                equity = cash + pos_val
                exposure = pos_val / equity if equity > 0 else 0.0
                cur_w = (shares[s] * price) / equity if equity > 0 else 0.0
                if tranches[s] >= max_tr:                continue
                if i - last_buy[s] < min_d:              continue
                if cur_w >= w_target:                    continue
                if exposure >= exp_cap:                  break
                spendable = cash - buf * equity
                if spendable < MIN_ORDER:                continue
                tranche = min(w_target * equity / MAX_TRANCHES,
                              (w_target - cur_w) * equity,
                              (exp_cap - exposure) * equity,
                              spendable, MAX_ORDER)
                if tranche < MIN_ORDER:                  continue
                qty = int(tranche / price)
                if qty < 1:                              continue
                cost = qty * price
                if cost > cash - buf * equity:           continue
                shares[s] += qty
                cash -= cost
                tranches[s] += 1
                last_buy[s] = i

        pos_val = sum(shares[s] * live.get(s, 0.0) for s in names)
        eq_curve.append(cash + pos_val)
        invested_curve.append(pos_val / (cash + pos_val) if (cash + pos_val) > 0 else 0.0)

    strat = pd.Series(eq_curve, index=cal)
    inv = pd.Series(invested_curve, index=cal)

    # benchmarks over the same window
    spy_w = spy.loc[cal[0]:cal[-1]].reindex(cal).ffill()
    spy_bh = START_CASH * (spy_w / spy_w.iloc[0])
    # equal-weight basket buy&hold (100% at start among names live on day 0), held
    p0 = panel.reindex(cal).ffill()
    live0 = [s for s in names if pd.notna(p0.iloc[0][s]) and p0.iloc[0][s] > 0]
    ew_sh = {s: (START_CASH / len(live0)) / p0.iloc[0][s] for s in live0}
    ew_bh = pd.Series([sum(ew_sh[s] * p0.loc[d, s] for s in live0) for d in cal], index=cal)

    # ── report ───────────────────────────────────────────────────────────────
    rows = [metrics(strat, label),
            metrics(ew_bh, "Equal-weight basket B&H (100%, no regime)"),
            metrics(spy_bh, "SPY buy & hold")]
    m0 = rows[0]
    print("ABLLINE|%s|%.0f|%.2f|%.2f|%.2f|%.1f" % (
        label, m0["final"], m0["total_return"] * 100, m0["sharpe"], m0["max_dd"] * 100, inv.mean() * 100))
    out = []
    out.append("")
    out.append(f"WINDOW: {cal[0].date()} -> {cal[-1].date()}  ({(cal[-1]-cal[0]).days/365.25:.2f}y, {len(cal)} bars)")
    out.append(f"NAMES TRADED: {len(names)}   START CASH: ${START_CASH:,.0f}")
    out.append(f"AVG INVESTED: {inv.mean()*100:.1f}%   FINAL INVESTED: {inv.iloc[-1]*100:.1f}%   "
               f"DAYS IN-MARKET: {(inv>0.05).mean()*100:.0f}%")
    out.append("")
    hdr = f"{'strategy':<42}{'final $':>14}{'tot ret':>10}{'CAGR':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}"
    out.append(hdr); out.append("-" * len(hdr))
    for m in rows:
        out.append(f"{m['label']:<42}{m['final']:>14,.0f}{m['total_return']*100:>9.1f}%"
                   f"{m['cagr']*100:>8.1f}%{m['vol']*100:>7.1f}%{m['sharpe']:>8.2f}{m['max_dd']*100:>8.1f}%")
    out.append("")
    # per-name final value contribution
    final_px = panel.reindex(cal).ffill().iloc[-1]
    contrib = sorted(((s, shares[s] * final_px[s]) for s in names if shares[s] > 0),
                     key=lambda x: -x[1])
    fin_eq = strat.iloc[-1]
    out.append("TOP / BOTTOM HOLDINGS (final weight, tranches filled):")
    for s, v in contrib[:6]:
        out.append(f"   {s:<6} ${v:>11,.0f}  {v/fin_eq*100:5.1f}%  ({tranches[s]} tranches)")
    out.append("   ...")
    for s, v in contrib[-4:]:
        out.append(f"   {s:<6} ${v:>11,.0f}  {v/fin_eq*100:5.1f}%  ({tranches[s]} tranches)")
    n_zero = sum(1 for s in names if shares[s] == 0)
    if n_zero:
        out.append(f"   ({n_zero} names never filled — no price at start / IPO'd later)")
    report = "\n".join(out)
    print(report)

    strat.to_frame("bal_gates_off").assign(spy=spy_bh, ew_basket=ew_bh, invested=inv) \
        .to_csv(ROOT / "backtest_outputs" / "bal_gates_off_equity.csv")
    print(f"\nequity curve -> backtest_outputs/bal_gates_off_equity.csv")


if __name__ == "__main__":
    (ROOT / "backtest_outputs").mkdir(exist_ok=True)
    main()
