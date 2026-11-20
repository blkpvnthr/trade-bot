#!/usr/bin/env python3
"""
backtest_di_bb.py -- evaluate the DI/Bollinger entry/exit rule as five variants, across
1/5/15-min completed candles, in-sample vs out-of-sample, net of costs, with regime splits.

RULE (completed candle; no ADX/order-flow as mandatory conditions):
    ENTRY  long : DI+ > DI-  AND  close > BB_middle(20)
    EXIT   long : DI- > DI+  AND  close < BB_middle(20)

VARIANTS (same symbols / periods / execution / costs):
    A  baseline        : buy-and-hold the session-series (market benchmark).
    B  DI/BB entry     + "existing" protective exit (stop -2% / +4%-lock giveback 1.5%).
    C  "existing" trend entry (close>SMA50 & SMA50 rising) + DI/BB exit.
    D  DI/BB entry     + DI/BB exit (the pure rule).
    E  D + order-flow ranking + profit protection -- evaluated SEPARATELY on genuine IEX
       tick days (see note), because order-flow needs ticks (never fabricated from bars).

NO LOOK-AHEAD: a signal on completed bar i executes at bar i+1 OPEN. Realistic spread/
slippage/fees via costs.py buckets. DI/Bollinger use ACTUAL historical bars.

Data limits: Alpaca IEX bars (single-exchange). 1-min over 5y is large; cached per
(symbol,timeframe). Variant E uses the sampled genuine-tick cache (smaller real-tick sample).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
for _l in (PROJECT_ROOT / ".env").read_text().splitlines() if (PROJECT_ROOT / ".env").exists() else []:
    if "=" in _l and not _l.strip().startswith("#"):
        import os
        _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip())

from backtest_dmi_adx import indicators, bucket_of, COST_BPS, ET

CACHE = PROJECT_ROOT / "data" / "cache" / "bars_multi_tf"
OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")
SLIPPAGE_BPS = 2.0   # modeled per-side execution slippage on top of the spread/commission bucket


# ----------------------------------------------------------------- data (actual historical bars)
def _client():
    import os
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(key, sec)


def fetch_bars(client, symbol: str, tf_min: int, start: date, end: date) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{symbol}_{tf_min}m_{start.isoformat()}_{end.isoformat()}.csv"
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            return df if not df.empty else pd.DataFrame()
        except Exception:
            pass
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame(tf_min, TimeFrameUnit.Minute),
                           start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
                           end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc), feed=DataFeed.IEX)
    try:
        df = client.get_stock_bars(req).df
    except Exception as exc:
        print(f"  [{symbol} {tf_min}m] fetch error: {str(exc)[:90]}")
        return pd.DataFrame()
    if df is None or df.empty:
        pd.DataFrame().to_csv(cache, index=False); return pd.DataFrame()
    d = df.xs(symbol, level=0).reset_index()
    d = d.rename(columns={"timestamp": "timestamp"})[["timestamp", "open", "high", "low", "close", "volume"]]
    d.to_csv(cache, index=False)
    return d


# ----------------------------------------------------------------- regime (SPY daily)
def _fetch_daily(client, symbol: str, start: date, end: date) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{symbol}_1D_{start.isoformat()}_{end.isoformat()}.csv"
    if cache.exists():
        try:
            return pd.read_csv(cache)
        except Exception:
            pass
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
                           start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
                           end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc), feed=DataFeed.IEX)
    try:
        df = client.get_stock_bars(req).df
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.xs(symbol, level=0).reset_index()[["timestamp", "open", "high", "low", "close", "volume"]]
    d.to_csv(cache, index=False)
    return d


def spy_regime_map(client, start: date, end: date) -> dict[date, str]:
    df = _fetch_daily(client, "SPY", start, end)
    if df.empty:
        return {}
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["date"] = d["timestamp"].dt.tz_convert(ET).dt.date
    daily = d.groupby("date")["close"].last()
    sma = daily.rolling(50).mean()
    out = {}
    for dt, px in daily.items():
        s = sma.get(dt, float("nan"))
        s_prev = sma.shift(10).get(dt, float("nan"))
        if not (s == s) or not (s_prev == s_prev):
            out[dt] = "unknown"
        elif px > s and s > s_prev:
            out[dt] = "bull"
        elif px < s and s < s_prev:
            out[dt] = "bear"
        else:
            out[dt] = "sideways"
    return out


# ----------------------------------------------------------------- trade sim
@dataclass
class BTrade:
    symbol: str
    entry_dt: Any
    exit_dt: Any
    entry_px: float
    exit_px: float
    ret: float          # net of costs
    gross_ret: float
    hold_bars: int
    hold_min: float
    mfe: float
    mae: float
    slippage_bps: float
    regime: str


def _signals(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["di_bb_entry"] = (d["pdi"] > d["mdi"]) & (d["close"] > d["mid"])
    d["di_bb_exit"] = (d["mdi"] > d["pdi"]) & (d["close"] < d["mid"])
    d["trend_entry"] = (d["close"] > d["sma50"]) & d["sma50_up"]     # "existing" entry proxy
    return d


def simulate(d: pd.DataFrame, symbol: str, variant: str, tf_min: int, cost_bps: float,
             regime_map: dict) -> list[BTrade]:
    """One long position at a time. Signal on completed bar i -> execute at bar i+1 OPEN."""
    d = d[d["rth"]].reset_index(drop=True) if "rth" in d.columns else d.reset_index(drop=True)
    if len(d) < 60:
        return []
    o = d["open"].astype(float).values; c = d["close"].astype(float).values
    hi = d["high"].astype(float).values; lo = d["low"].astype(float).values
    ent = d["di_bb_entry"].values; ext = d["di_bb_exit"].values
    tent = d["trend_entry"].values
    dates = d["date"].values
    n = len(d)
    rt_cost = (cost_bps + SLIPPAGE_BPS) / 1e4  # per side

    def entry_ok(i):
        if variant in ("B", "D", "E"):
            return bool(ent[i])
        if variant == "C":
            return bool(tent[i])
        return False

    def exit_ok(i, entry_px, peak):
        if variant in ("C", "D", "E"):
            return bool(ext[i])
        if variant == "B":  # "existing" protective exit: stop -2% or +4%-lock giveback 1.5%
            if c[i] <= entry_px * 0.98:
                return True
            if peak >= entry_px * 1.04 and c[i] <= peak - 0.015 * entry_px:
                return True
            return False
        return False

    trades: list[BTrade] = []
    if variant == "A":  # buy-and-hold the series
        ei, xi = 1, n - 1
        entry_px = o[ei] * (1 + rt_cost); exit_px = c[xi] * (1 - rt_cost)
        gret = c[xi] / o[ei] - 1.0
        peak = max(hi[ei:xi + 1]); dip = min(lo[ei:xi + 1])
        trades.append(BTrade(symbol, dates[ei], dates[xi], entry_px, exit_px,
                             exit_px / entry_px - 1.0, gret, xi - ei, (xi - ei) * tf_min,
                             peak / o[ei] - 1.0, dip / o[ei] - 1.0, (cost_bps + SLIPPAGE_BPS) * 2,
                             regime_map.get(dates[ei], "unknown")))
        return trades

    i = 0
    while i < n - 1:
        if not entry_ok(i):
            i += 1; continue
        ei = i + 1                     # next-bar-open execution (no look-ahead)
        entry_px = o[ei] * (1 + rt_cost)
        peak = hi[ei]; dip = lo[ei]
        j = ei
        xi = None
        while j < n - 1:
            peak = max(peak, hi[j]); dip = min(dip, lo[j])
            if exit_ok(j, o[ei], peak):
                xi = j + 1; break
            j += 1
        if xi is None:
            xi = n - 1
        exit_px = o[xi] * (1 - rt_cost) if xi < n else c[n - 1] * (1 - rt_cost)
        gret = o[xi] / o[ei] - 1.0
        trades.append(BTrade(symbol, dates[ei], dates[xi], entry_px, exit_px,
                             exit_px / entry_px - 1.0, gret, xi - ei, (xi - ei) * tf_min,
                             peak / o[ei] - 1.0, dip / o[ei] - 1.0, (cost_bps + SLIPPAGE_BPS) * 2,
                             regime_map.get(dates[ei], "unknown")))
        i = xi + 1
    return trades


# ----------------------------------------------------------------- metrics
def metrics(trades: list[BTrade], notional: float = 2000.0) -> dict:
    if not trades:
        return {"trades": 0}
    rets = [t.ret for t in trades]
    pnls = [r * notional for r in rets]
    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe = (statistics.mean(rets) / sd * math.sqrt(252)) if sd > 0 else 0.0  # annualized-ish
    cum = pk = mdd = 0.0
    for x in pnls:
        cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    total_hold = sum(t.hold_min for t in trades)
    span_min = max(1.0, total_hold)  # exposure proxy
    return {
        "trades": len(trades),
        "net_pnl": round(sum(pnls), 2),
        "net_expectancy": round(statistics.mean(pnls), 3),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "win_rate": round(len(wins) / len(pnls), 3),
        "max_drawdown": round(mdd, 2),
        "sharpe": round(sharpe, 3),
        "avg_hold_min": round(statistics.mean([t.hold_min for t in trades]), 1),
        "avg_mfe": round(statistics.mean([t.mfe for t in trades]) * 100, 3),
        "avg_mae": round(statistics.mean([t.mae for t in trades]) * 100, 3),
        "avg_slippage_bps": round(statistics.mean([t.slippage_bps for t in trades]), 2),
        "exposure_adj_return_per_hr": round(sum(pnls) / (span_min / 60.0), 4),
    }


def by_regime(trades: list[BTrade]) -> dict:
    out = {}
    for reg in ("bull", "bear", "sideways"):
        ts = [t for t in trades if t.regime == reg]
        if ts:
            out[reg] = {"trades": len(ts),
                        "net_expectancy": round(statistics.mean([t.ret for t in ts]) * 2000, 3),
                        "win_rate": round(sum(1 for t in ts if t.ret > 0) / len(ts), 3)}
    return out


def verdict(oos: dict, baseline_oos: dict) -> str:
    n = oos.get("trades", 0)
    if n < 30:
        return "INCONCLUSIVE (too few OOS trades)"
    exp = oos.get("net_expectancy", 0) or 0
    base = baseline_oos.get("net_expectancy", 0) or 0
    pf = oos.get("profit_factor", 0)
    pf = 999 if pf == "inf" else (pf or 0)
    if exp > 0 and exp > base and pf > 1.1:
        return "ACCEPT (OOS net-positive, beats baseline, PF>1.1)"
    if exp > 0:
        return "INCONCLUSIVE (OOS net-positive but does not clearly beat baseline/PF)"
    return "REJECT (OOS not net-positive after costs)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,META,GOOGL")
    ap.add_argument("--timeframes", default="1,5,15")
    ap.add_argument("--start", default="2020-11-01")
    ap.add_argument("--end", default="2025-11-01")
    ap.add_argument("--train-end", default="2023-05-01")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    client = _client()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [int(x) for x in args.timeframes.split(",") if x.strip()]
    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    train_end = date.fromisoformat(args.train_end)
    regime_map = spy_regime_map(client, start, end)
    print(f"[di-bb] {len(symbols)} symbols x timeframes {tfs}, {start}..{end}, "
          f"train<{train_end} (actual historical bars)")

    variants = ["A", "B", "C", "D"]  # E handled separately (genuine ticks)
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                              "data_source": "Alpaca IEX bars (single-exchange)", "symbols": symbols,
                              "window": [args.start, args.end], "train_end": args.train_end,
                              "timeframes": {}}
    for tf in tfs:
        print(f"\n[tf={tf}m] fetching + simulating...", flush=True)
        per_variant: dict[str, list[BTrade]] = {v: [] for v in variants}
        for sym in symbols:
            bars = fetch_bars(client, sym, tf, start, end)
            if bars.empty:
                continue
            d = _signals(indicators(bars, slope_lb=3))
            cost = COST_BPS[bucket_of(sym)]
            for v in variants:
                if v == "A":  # buy-hold split per period so IS/OOS each get a benchmark trade
                    is_slice = d[d["date"] < train_end].reset_index(drop=True)
                    oos_slice = d[d["date"] >= train_end].reset_index(drop=True)
                    per_variant[v].extend(simulate(is_slice, sym, "A", tf, cost, regime_map))
                    per_variant[v].extend(simulate(oos_slice, sym, "A", tf, cost, regime_map))
                else:
                    per_variant[v].extend(simulate(d, sym, v, tf, cost, regime_map))
            print(f"  {sym}: bars={len(bars)}", flush=True)
        tfkey = f"{tf}m"
        report["timeframes"][tfkey] = {}
        for v in variants:
            tr = per_variant[v]
            is_tr = [t for t in tr if t.entry_dt < train_end]
            oos_tr = [t for t in tr if t.entry_dt >= train_end]
            report["timeframes"][tfkey][v] = {
                "overall": metrics(tr), "in_sample": metrics(is_tr), "out_of_sample": metrics(oos_tr),
                "regime_oos": by_regime(oos_tr),
            }
        base_oos = report["timeframes"][tfkey]["A"]["out_of_sample"]
        for v in ("B", "C", "D"):
            report["timeframes"][tfkey][v]["verdict"] = verdict(
                report["timeframes"][tfkey][v]["out_of_sample"], base_oos)
        report["timeframes"][tfkey]["A"]["verdict"] = "BASELINE"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"di_bb_backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    # print summary
    for tf in tfs:
        tfkey = f"{tf}m"
        print(f"\n================= TIMEFRAME {tfkey} =================")
        cols = ["trades", "net_expectancy", "profit_factor", "win_rate", "sharpe",
                "max_drawdown", "avg_hold_min", "avg_mfe", "avg_mae"]
        print(f"{'OOS metric':<16}" + "".join(f"{v:>13}" for v in variants))
        for col in cols:
            print(f"{col:<16}" + "".join(
                f"{str(report['timeframes'][tfkey][v]['out_of_sample'].get(col)):>13}" for v in variants))
        print("verdict:")
        for v in ("B", "C", "D"):
            print(f"   {v}: {report['timeframes'][tfkey][v]['verdict']}")
    print(f"\n[di-bb] wrote {OUT_DIR / f'di_bb_backtest_{tag}.json'}")
    print("NOTE: IEX bars proxy; variant E (order-flow) needs genuine ticks -> run backtest_di_bb_E.py.")


if __name__ == "__main__":
    main()
