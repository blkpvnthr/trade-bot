#!/usr/bin/env python3
"""
+15% premarket gapper intraday short-fade research framework.  RESEARCH ONLY.

Question: do stocks gapping >=15% in the premarket fade intraday enough to be a
tradable SHORT after realistic costs?

Pipeline:
  1. Scan TradingView for premarket gappers -> universe CSV (today's snapshot).
  2. Pull Alpaca 1-min bars (symbol x month, cached, resumable, offline-capable).
  3. Rebuild qualifying gapper-days from the bars (descriptive baseline).
  4. Backtest two short entries (A1/A2) x two exit models with NEXT-BAR execution,
     no lookahead, and slippage/commission/borrow costs.
  5. Report a robustness grid and write all artifacts to CSV.

DO NOT wire this to live trading.  It places no orders.

SELECTION-BIAS WARNING: the TradingView scan returns names that are gapping *today*,
so the historical lookback over those same names is survivor/selection biased.  The
baseline fade rate is descriptive, not an edge.  Only the post-cost strategy stats
(sections 4-9), tested for sign and stability, speak to tradability.

SMALL-CAP SHORT-SALE REALITY this backtest does NOT model (read before trusting it):
  - Locate/shortability: many +15% small-cap gappers are HARD-TO-BORROW or have NO
    locate -> the short is simply not enterable.  Fills here are hypothetical.
  - Borrow fees: HTB names can cost hundreds of bps/day; --borrow-fee-bps-per-day is
    one flat knob; real fees are name/day specific and often far higher.
  - SSR (uptick rule): a -10% prior day restricts shorts next session.
  - Halts: LULD volatility halts are common and can trap a position past any stop.
  - Slippage/spreads: spreads on $1-20 low-float names dwarf 10 bps.  Costs here are
    optimistic.
  - Rejections / partial fills on marketable short orders.
Treat every positive result here as an UPPER bound that reality will erode.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import dotenv
from dotenv import load_dotenv

load_dotenv(dotenv.find_dotenv())

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

ET = ZoneInfo("America/New_York")

# Session boundaries (ET).
PRE_START = time(4, 0)
RTH_OPEN = time(9, 30)
FIRST15_END = time(9, 45)   # first-15 window is [09:30, 09:45)
ENTRY_AFTER = time(9, 45)   # signals only allowed at/after 09:45
TIME_EXIT = time(15, 55)    # forced exit
RTH_CLOSE = time(16, 0)
RTH_MINUTES = 390.0         # for borrow-fee day fraction

VARIANTS = ("A1", "A2")
EXIT_MODELS = (1, 2)        # 1 = stop/2R/time ; 2 = adds VWAP-reclaim exit


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    start: str
    end: str
    gap_threshold: float          # percent, e.g. 15
    max_symbols: int
    feed: str
    slippage_bps: float
    commission_bps: float
    borrow_fee_bps_per_day: float
    cache_dir: str
    strict_low_break: bool        # run A2 only
    min_rel_volume: float         # hard pre-filter on qualifying days (fraction)
    min_price: float              # hard pre-filter on entry price
    max_price: float
    max_market_cap: float         # dollars; inf = off
    skip_fetch_existing_cache: bool  # offline: only use cached chunks
    universe_source: str          # "tradingview" (today's scan) | "pit_archive"
    pit_archive_dir: str          # dir of YYYY-MM-DD_premarket_gappers.csv files


# --------------------------------------------------------------------------- #
# 1. Universe
# --------------------------------------------------------------------------- #
def scan_tradingview_gappers(cfg: Config) -> pd.DataFrame:
    """Scan TradingView premarket gappers; persist + return the universe frame.

    Falls back to a pre-existing tradingview_premarket_gappers.csv if the live
    scan returns nothing (e.g. run outside premarket hours), so the backtest is
    reproducible at any time of day.
    """
    from tradingview_screener import Query, col

    _, df = (
        Query()
        .select(
            "logoid", "name", "premarket_close", "premarket_change_abs",
            "premarket_change", "premarket_volume", "premarket_gap", "close",
            "change", "volume", "market_cap_basic", "Perf.1Y.MarketCap",
            "description", "type", "typespecs", "update_mode", "pricescale",
            "minmov", "fractional", "minmove2", "fundamental_currency_code",
            "currency",
        )
        .where(
            col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]),
            col("is_primary") == True,  # noqa: E712  (screener DSL needs ==)
            col("typespecs").has("common"),
            col("typespecs").has_none_of("preferred"),
            col("type") == "stock",
            col("premarket_gap").not_empty(),
            col("active_symbol") == True,  # noqa: E712
        )
        .order_by("premarket_gap", ascending=False, nulls_first=False)
        .limit(cfg.max_symbols)
        .set_markets("america")
        .set_property("symbols", {"query": {"types": ["stock", "fund", "dr", "structured"]}})
        .set_property("preset", "pre_market_gappers")
        .get_scanner_data()
    )

    df = df.copy()
    df["premarket_gap"] = pd.to_numeric(df["premarket_gap"], errors="coerce")
    df = df[df["premarket_gap"] >= cfg.gap_threshold].copy()
    df["symbol"] = df["name"].astype(str).str.replace(r"^[A-Z]+:", "", regex=True)

    if df.empty and os.path.exists("tradingview_premarket_gappers.csv"):
        print("Live scan empty; reusing existing tradingview_premarket_gappers.csv")
        df = pd.read_csv("tradingview_premarket_gappers.csv")
        if "symbol" not in df.columns:
            df["symbol"] = df["name"].astype(str).str.replace(r"^[A-Z]+:", "", regex=True)
        df = df.head(cfg.max_symbols)

    df.to_csv("tradingview_premarket_gappers.csv", index=False)
    symbols = sorted(df["symbol"].dropna().unique().tolist())[: cfg.max_symbols]
    print(f"\n=== Universe: {len(symbols)} symbols, premarket_gap >= {cfg.gap_threshold}% ===")
    print(symbols)
    return df[df["symbol"].isin(symbols)].copy()


# --------------------------------------------------------------------------- #
# 2. Historical data (symbol x month, cached, resumable)
# --------------------------------------------------------------------------- #
def month_ranges(start: str, end: str):
    starts = pd.date_range(start=start, end=end, freq="MS")
    if pd.Timestamp(start) not in starts:
        starts = starts.insert(0, pd.Timestamp(start))
    for s in starts:
        e = s + pd.offsets.MonthBegin(1)
        if e > pd.Timestamp(end):
            e = pd.Timestamp(end)
        if s >= e:
            continue
        yield s, e


def fetch_symbol_month(client, symbol, start, end, cfg: Config) -> pd.DataFrame:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = f"{cfg.cache_dir}/{symbol}_{start.date()}_{end.date()}.csv"

    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, parse_dates=["timestamp"])
        except Exception:
            os.remove(cache_path)  # corrupt partial -> refetch

    if cfg.skip_fetch_existing_cache:
        return pd.DataFrame()  # offline mode: only use what's cached

    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    print(f"Fetching {symbol} {start.date()} -> {end.date()}")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz=ET),
        end=pd.Timestamp(end, tz=ET),
        feed=cfg.feed,
    )
    df = client.get_stock_bars(req).df
    # atomic write (tmp + replace) so an interrupted run never leaves a
    # truncated-but-parseable chunk that the resume path would silently trust.
    if df.empty:
        # empty marker so we don't refetch known-empty chunks
        out = pd.DataFrame(columns=["symbol", "timestamp"])
    else:
        out = df.reset_index()
    out.to_csv(cache_path + ".tmp", index=False)
    os.replace(cache_path + ".tmp", cache_path)
    return out if not out.empty else pd.DataFrame()


def load_symbol_bars(client, symbol, cfg: Config) -> pd.DataFrame:
    """Concat one symbol's monthly chunks and add ET date/tod columns."""
    frames = []
    for s, e in month_ranges(cfg.start, cfg.end):
        try:
            df = fetch_symbol_month(client, symbol, s, e, cfg)
            if not df.empty and "timestamp" in df.columns:
                frames.append(df)
        except Exception as exc:  # network/rate-limit/etc -> skip chunk, keep going
            print(f"FAILED {symbol} {s.date()}->{e.date()}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["tod"] = df["timestamp"].dt.time
    return df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Descriptive baseline (per qualifying gapper-day)
# --------------------------------------------------------------------------- #
def _vwap(df: pd.DataFrame) -> np.ndarray:
    typ = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    pv = typ * df["volume"].values
    cum_pv = np.cumsum(pv)
    cum_v = np.cumsum(df["volume"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cum_v > 0, cum_pv / cum_v, np.nan)


def baseline_for_symbol(sdf: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per qualifying (gap >= threshold) day for this symbol."""
    rth_all = sdf[(sdf["tod"] >= RTH_OPEN) & (sdf["tod"] <= RTH_CLOSE)]
    pre_all = sdf[(sdf["tod"] >= PRE_START) & (sdf["tod"] < RTH_OPEN)]
    if rth_all.empty:
        return pd.DataFrame()

    rth_daily = rth_all.groupby("date", sort=True).agg(
        rth_open=("open", "first"),
        rth_close=("close", "last"),
        rth_high=("high", "max"),
        rth_volume=("volume", "sum"),
    )
    # prior RTH close & 20d avg volume use only PAST days (shift) -> no leakage.
    rth_daily["prior_close"] = rth_daily["rth_close"].shift(1)
    rth_daily["avg_20d_vol"] = rth_daily["rth_volume"].rolling(20).mean().shift(1)

    pre_daily = pre_all.groupby("date", sort=True).agg(
        premarket_high=("high", "max"),
        premarket_volume=("volume", "sum"),
    )

    out = rth_daily.join(pre_daily, how="inner").dropna(subset=["prior_close", "premarket_high"])
    if out.empty:
        return out
    out["gap_pct"] = out["premarket_high"] / out["prior_close"] - 1.0
    out = out[out["gap_pct"] >= cfg.gap_threshold / 100.0].copy()
    if out.empty:
        return out

    # per-day intraday descriptives (VWAP / first-15 / excursions from minute bars)
    f15h, f15l, cav, hf15, o2c = {}, {}, {}, {}, {}
    o10, o12, mae, mfe = {}, {}, {}, {}
    for d in out.index:
        day = rth_all[rth_all["date"] == d]
        if day.empty:
            continue
        vw = _vwap(day)
        tod = day["tod"].values
        highs, lows, closes = day["high"].values, day["low"].values, day["close"].values
        m15 = (tod >= RTH_OPEN) & (tod < FIRST15_END)
        if not m15.any():
            continue
        f15h[d] = float(highs[m15].max())
        f15l[d] = float(lows[m15].min())
        o0 = float(day["open"].values[0])
        last_close = float(closes[-1])
        cav[d] = bool(last_close > vw[-1]) if np.isfinite(vw[-1]) else np.nan
        hf15[d] = bool(last_close > f15h[d])
        o2c[d] = last_close / o0 - 1.0

        def _px_at(cut):  # last close at/before a cutoff time
            m = tod <= cut
            return float(closes[m][-1]) if m.any() else np.nan
        o10[d] = _px_at(time(10, 0)) / o0 - 1.0
        o12[d] = _px_at(time(12, 0)) / o0 - 1.0
        # fade = short from RTH open; "against" = price up, "in favor" = price down
        mae[d] = float(highs.max()) / o0 - 1.0          # max adverse excursion (positive)
        mfe[d] = 1.0 - float(lows.min()) / o0           # max favorable excursion (positive)

    out["first_15_high"] = pd.Series(f15h)
    out["first_15_low"] = pd.Series(f15l)
    out["close_above_vwap"] = pd.Series(cav)
    out["held_first_15_high"] = pd.Series(hf15)
    out["open_to_close_return"] = pd.Series(o2c)
    out["open_to_10am_return"] = pd.Series(o10)
    out["open_to_noon_return"] = pd.Series(o12)
    out["mae_against_fade"] = pd.Series(mae)
    out["mfe_for_fade"] = pd.Series(mfe)
    out["premarket_relative_volume"] = out["premarket_volume"] / out["avg_20d_vol"]
    out["closed_below_open"] = out["rth_close"] < out["rth_open"]
    return out.dropna(subset=["first_15_high", "first_15_low"]).reset_index().rename(columns={"index": "date"})


# --------------------------------------------------------------------------- #
# 4-6. Strategy simulation (next-bar execution, no lookahead)
# --------------------------------------------------------------------------- #
def simulate_day(day_rth: pd.DataFrame, variant: str, exit_model: int, cfg: Config):
    """Simulate one short trade on one gapper-day.

    Returns a trade dict, or ("skip", reason) if no trade is taken.
    Lookahead guards:
      * signal evaluated on a bar's CLOSE; entry is the NEXT bar's OPEN.
      * exits only ever read bars at/after the entry bar.
      * intrabar stop vs target tie -> stop assumed first (conservative).
    """
    day = day_rth.sort_values("timestamp").reset_index(drop=True)
    o = day["open"].values.astype(float)
    h = day["high"].values.astype(float)
    low = day["low"].values.astype(float)
    c = day["close"].values.astype(float)
    tod = day["tod"].values
    ts = day["timestamp"].values
    vwap = _vwap(day)
    n = len(day)

    m15 = np.array([(t >= RTH_OPEN) and (t < FIRST15_END) for t in tod])
    if not m15.any():
        return ("skip", "no_first15")
    f15_high = float(h[m15].max())
    f15_low = float(low[m15].min())
    thresh = f15_high if variant == "A1" else f15_low

    # --- signal: first bar at/after 09:45 with close < VWAP and close < thresh ---
    signal_i = None
    for i in range(n):
        if tod[i] < ENTRY_AFTER:
            continue
        if np.isfinite(vwap[i]) and c[i] < vwap[i] and c[i] < thresh:
            signal_i = i
            break
    if signal_i is None:
        return ("skip", "no_signal")

    entry_i = signal_i + 1
    if entry_i >= n:
        return ("skip", "no_next_bar")
    entry_price = o[entry_i]
    stop = f15_high
    if not (entry_price < stop):           # risk must be positive for a short
        return ("skip", "entry_above_stop")
    R = stop - entry_price
    target = entry_price - 2.0 * R

    # eligible bars: entry bar onward, up to the time-exit cutoff
    elig = [j for j in range(entry_i, n) if tod[j] <= TIME_EXIT]
    if not elig:
        return ("skip", "no_session_after_entry")

    exit_price = exit_reason = exit_i = None
    for j in elig:
        # intrabar: stop checked before target (tie -> stop first)
        if h[j] >= stop:
            exit_price, exit_reason, exit_i = stop, "stop", j
            break
        if low[j] <= target:
            exit_price, exit_reason, exit_i = target, "target", j
            break
        if exit_model == 2 and np.isfinite(vwap[j]) and c[j] > vwap[j]:
            exit_price, exit_reason, exit_i = c[j], "vwap_reclaim", j
            break
    if exit_price is None:                  # forced time exit at last eligible bar
        exit_i = elig[-1]
        exit_price, exit_reason = c[exit_i], "time"

    hold_min = (pd.Timestamp(ts[exit_i]) - pd.Timestamp(ts[entry_i])).total_seconds() / 60.0
    gross = (entry_price - exit_price) / entry_price      # SHORT return
    cost_bps = (2 * cfg.slippage_bps + 2 * cfg.commission_bps
                + cfg.borrow_fee_bps_per_day * (hold_min / RTH_MINUTES))
    net = gross - cost_bps / 1e4

    return {
        "variant": variant, "exit_model": exit_model,
        "entry_time": pd.Timestamp(ts[entry_i]), "entry_price": entry_price,
        "exit_time": pd.Timestamp(ts[exit_i]), "exit_price": exit_price,
        "exit_reason": exit_reason, "stop": stop, "target": target, "R": R,
        "hold_min": hold_min, "gross_return": gross, "net_return": net,
        "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# 9. Metrics
# --------------------------------------------------------------------------- #
def trade_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    base = {"trades": n}
    if n == 0:
        return {**base, **{k: np.nan for k in (
            "win_rate", "avg_return", "median_return", "avg_winner", "avg_loser",
            "payoff_ratio", "expectancy", "profit_factor", "sharpe", "gross_profit",
            "gross_loss", "max_drawdown", "pct_stop", "pct_target",
            "pct_vwap_reclaim", "pct_time")}}
    r = df["net_return"].values.astype(float)
    wins, losses = r[r > 0], r[r < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if n else 0.0
    rc = df["exit_reason"].value_counts()
    return {
        **base,
        "win_rate": len(wins) / n,
        "avg_return": float(r.mean()),
        "median_return": float(np.median(r)),
        "avg_winner": float(wins.mean()) if len(wins) else np.nan,
        "avg_loser": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio": (float(wins.mean()) / abs(float(losses.mean())))
                        if len(wins) and len(losses) else np.nan,
        "expectancy": float(r.mean()),                 # mean net return / trade
        "profit_factor": (gp / gl) if gl > 0 else np.inf,
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std(ddof=1) > 0 else np.nan,
        "gross_profit": gp, "gross_loss": gl, "max_drawdown": dd,
        "pct_stop": rc.get("stop", 0) / n,
        "pct_target": rc.get("target", 0) / n,
        "pct_vwap_reclaim": rc.get("vwap_reclaim", 0) / n,
        "pct_time": rc.get("time", 0) / n,
    }


# --------------------------------------------------------------------------- #
# 8. Robustness filter grid
# --------------------------------------------------------------------------- #
def filter_specs():
    """name -> predicate(trades_df) -> boolean mask. Single-axis (marginal) filters."""
    return {
        "all": lambda d: pd.Series(True, index=d.index),
        "relvol>=0.01": lambda d: d["premarket_relative_volume"] >= 0.01,
        "relvol>=0.03": lambda d: d["premarket_relative_volume"] >= 0.03,
        "relvol>=0.05": lambda d: d["premarket_relative_volume"] >= 0.05,
        "mktcap<500M": lambda d: d["market_cap"] < 500e6,
        "mktcap<250M": lambda d: d["market_cap"] < 250e6,
        "price_1_20": lambda d: (d["entry_price"] >= 1) & (d["entry_price"] <= 20),
        "samples>=25": lambda d: d["sym_samples"] >= 25,
        # WARNING: sym_avg_o2c / sym_cav_rate are computed over the WHOLE sample,
        # so this filter conditions trades on a symbol's realized fade behavior in
        # the same window -> it is IN-SAMPLE selection (look-ahead-flavored), NOT a
        # tradeable ex-ante filter. Diagnostic only; never read its result as edge.
        "descriptive_edge_INSAMPLE": lambda d: (d["sym_avg_o2c"] < -0.005) & (d["sym_cav_rate"] < 0.45),
    }


# --------------------------------------------------------------------------- #
# Point-in-time universe (archived daily scans -> no survivorship bias)
# --------------------------------------------------------------------------- #
def load_pit_archive(cfg: Config):
    """Load archived daily premarket-gapper scans in [start, end].

    Returns (symbols, date_to_symbols, coverage_df, mktcap):
      * symbols          : union of all symbols across archived dates (bars to fetch)
      * date_to_symbols  : {'YYYY-MM-DD' -> set(symbols on that day's scan)}
      * coverage_df      : one row per archived date present in range
      * mktcap           : {symbol -> last-seen market_cap_basic from the archive}
    Membership only: a day trades a symbol iff it appeared on that day's scan AND
    the bar-derived gap still clears cfg.gap_threshold (checked in baseline).
    """
    import glob

    files = sorted(glob.glob(os.path.join(cfg.pit_archive_dir, "*_premarket_gappers.csv")))
    start = pd.Timestamp(cfg.start).date()
    end = pd.Timestamp(cfg.end).date()
    date_to_symbols, mktcap, cov_rows = {}, {}, []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "scan_date" not in df.columns or "symbol" not in df.columns:
            continue
        sd = str(df["scan_date"].iloc[0])
        try:
            dd = pd.Timestamp(sd).date()
        except Exception:
            continue
        if dd < start or dd > end:
            continue
        syms = set(df["symbol"].dropna().astype(str))
        if not syms:
            continue
        date_to_symbols.setdefault(sd, set()).update(syms)
        gaps = pd.to_numeric(df.get("premarket_gap"), errors="coerce")
        caps = pd.to_numeric(df.get("market_cap_basic"), errors="coerce")
        for sym, cap in zip(df["symbol"].astype(str), caps):
            if pd.notna(cap):
                mktcap[sym] = float(cap)
        cov_rows.append({
            "scan_date": sd, "n_symbols": len(syms),
            "n_snapshots": int(df["scan_timestamp_et"].nunique())
            if "scan_timestamp_et" in df.columns else 1,
            "min_gap": float(gaps.min()) if gaps.notna().any() else np.nan,
            "max_gap": float(gaps.max()) if gaps.notna().any() else np.nan,
        })
    symbols = sorted({s for ss in date_to_symbols.values() for s in ss})
    coverage = (pd.DataFrame(cov_rows).sort_values("scan_date").reset_index(drop=True)
                if cov_rows else pd.DataFrame(
                    columns=["scan_date", "n_symbols", "n_snapshots", "min_gap", "max_gap"]))
    return symbols, date_to_symbols, coverage, mktcap


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> None:
    date_to_symbols, archive_dates = None, None
    if cfg.universe_source == "pit_archive":
        symbols, date_to_symbols, coverage, mktcap = load_pit_archive(cfg)
        coverage.to_csv("pit_archive_coverage.csv", index=False)
        archive_dates = set(date_to_symbols)
        print(f"\n=== PIT archive universe: {len(symbols)} symbols across "
              f"{len(archive_dates)} archived dates ({cfg.pit_archive_dir}) ===")
        if not symbols:
            print("No archived scans found in range. Run pit_gapper_archive.py first.")
            return
    else:
        tv_df = scan_tradingview_gappers(cfg)
        symbols = sorted(tv_df["symbol"].dropna().unique().tolist())
        if not symbols:
            print("No qualifying TradingView gappers. Nothing to test.")
            return
        mktcap = (tv_df.assign(market_cap=pd.to_numeric(tv_df.get("market_cap_basic"),
                                                        errors="coerce"))
                  .set_index("symbol")["market_cap"].to_dict())

    client = None
    if not cfg.skip_fetch_existing_cache:
        from alpaca.data.historical import StockHistoricalDataClient
        client = StockHistoricalDataClient(API_KEY, API_SECRET)

    variants = ("A2",) if cfg.strict_low_break else VARIANTS
    baseline_rows, trade_rows, skip_rows = [], [], []
    pit_skipped_dates = {}  # 'YYYY-MM-DD' -> count of symbol-days dropped (no archive)

    for sym in symbols:
        sdf = load_symbol_bars(client, sym, cfg)
        if sdf.empty:
            skip_rows.append({"symbol": sym, "date": None, "gap_pct": None, "reason": "no_bars"})
            continue
        base = baseline_for_symbol(sdf, cfg)
        if base.empty:
            continue
        base["symbol"] = sym
        baseline_rows.append(base)

        rth = sdf[(sdf["tod"] >= RTH_OPEN) & (sdf["tod"] <= RTH_CLOSE)]
        by_date = {d: g for d, g in rth.groupby("date", sort=True)}
        for _, row in base.iterrows():
            d, gap, relvol = row["date"], row["gap_pct"], row["premarket_relative_volume"]
            # point-in-time membership: only trade symbols on that day's archived scan
            if cfg.universe_source == "pit_archive":
                d_str = str(d)
                if d_str not in archive_dates:
                    pit_skipped_dates[d_str] = pit_skipped_dates.get(d_str, 0) + 1
                    skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap,
                                      "reason": "no_pit_archive_for_date"})
                    continue
                if sym not in date_to_symbols.get(d_str, set()):
                    skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap,
                                      "reason": "symbol_not_in_pit_archive"})
                    continue
            # hard CLI pre-filters (logged, not silently dropped)
            if pd.notna(relvol) and relvol < cfg.min_rel_volume:
                skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap, "reason": "prefilter_relvol"})
                continue
            cap = mktcap.get(sym, np.nan)
            if math.isfinite(cfg.max_market_cap) and pd.notna(cap) and cap > cfg.max_market_cap:
                skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap, "reason": "prefilter_mktcap"})
                continue
            day = by_date.get(d)
            if day is None or len(day) < 20:
                skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap, "reason": "insufficient_bars"})
                continue
            for variant in variants:
                for em in EXIT_MODELS:
                    res = simulate_day(day, variant, em, cfg)
                    if isinstance(res, tuple):
                        if variant == variants[0] and em == EXIT_MODELS[0]:
                            skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap, "reason": res[1]})
                        continue
                    if not (cfg.min_price <= res["entry_price"] <= cfg.max_price):
                        if variant == variants[0] and em == EXIT_MODELS[0]:
                            skip_rows.append({"symbol": sym, "date": d, "gap_pct": gap, "reason": "prefilter_price"})
                        continue
                    res.update(symbol=sym, date=d, gap_pct=gap,
                               premarket_relative_volume=relvol, market_cap=cap)
                    trade_rows.append(res)

    # ---- baseline outputs -------------------------------------------------- #
    if not baseline_rows:
        print("No historical qualifying gapper-days found in the universe.")
        return
    baseline = pd.concat(baseline_rows, ignore_index=True)
    baseline.to_csv("baseline_gapper_fade_results.csv", index=False)

    bsum = baseline.groupby("symbol").agg(
        samples=("closed_below_open", "count"),
        fade_rate=("closed_below_open", "mean"),
        avg_open_to_close_return=("open_to_close_return", "mean"),
        median_open_to_close_return=("open_to_close_return", "median"),
        close_above_vwap_rate=("close_above_vwap", "mean"),
        held_first_15_high_rate=("held_first_15_high", "mean"),
        avg_premarket_rel_vol=("premarket_relative_volume", "mean"),
        avg_gap=("gap_pct", "mean"),
    )
    bsum["market_cap"] = bsum.index.map(lambda s: mktcap.get(s, np.nan))
    bsum.to_csv("baseline_gapper_fade_summary.csv")

    total = len(baseline)
    print("\n=== Descriptive baseline (+%g%% gappers) ===" % cfg.gap_threshold)
    print(f"Qualifying gapper-days : {total}")
    print(f"Fade rate (close<open) : {baseline['closed_below_open'].mean():.2%}")
    print(f"Avg open->close return : {baseline['open_to_close_return'].mean():.3%}")
    print(f"Median open->close ret : {baseline['open_to_close_return'].median():.3%}")
    print(f"Close-above-VWAP rate  : {baseline['close_above_vwap'].mean():.2%}")
    print(f"Held first-15 high rate: {baseline['held_first_15_high'].mean():.2%}")

    pd.DataFrame(skip_rows).to_csv("rejected_or_skipped_days.csv", index=False)

    if cfg.universe_source == "pit_archive":
        sd = pd.DataFrame(
            [{"date": k, "symbol_days_dropped": v, "reason": "no_pit_archive_for_date"}
             for k, v in sorted(pit_skipped_dates.items())],
            columns=["date", "symbol_days_dropped", "reason"])
        sd.to_csv("skipped_dates.csv", index=False)
        print(f"\nPIT archive: {len(archive_dates)} dates covered, "
              f"{len(pit_skipped_dates)} qualifying-day dates skipped (no archive). "
              f"-> pit_archive_coverage.csv, skipped_dates.csv")

    # ---- strategy outputs -------------------------------------------------- #
    if not trade_rows:
        print("\nNo strategy trades generated (no signals cleared the filters).")
        # still write empty schemas so downstream tooling has the files
        pd.DataFrame(columns=["symbol", "date", "variant", "exit_model", "net_return"]).to_csv(
            "strategy_trades.csv", index=False)
        pd.DataFrame().to_csv("strategy_variant_summary.csv", index=False)
        return

    trades = pd.DataFrame(trade_rows)
    # attach symbol-level descriptive stats for the descriptive-edge / sample filters
    sym_stats = bsum[["samples", "avg_open_to_close_return", "close_above_vwap_rate"]].rename(
        columns={"samples": "sym_samples", "avg_open_to_close_return": "sym_avg_o2c",
                 "close_above_vwap_rate": "sym_cav_rate"})
    trades = trades.merge(sym_stats, left_on="symbol", right_index=True, how="left")
    trades.to_csv("strategy_trades.csv", index=False)

    specs = filter_specs()
    rows = []
    for variant in variants:
        for em in EXIT_MODELS:
            sub0 = trades[(trades["variant"] == variant) & (trades["exit_model"] == em)]
            for fname, pred in specs.items():
                sub = sub0[pred(sub0)] if len(sub0) else sub0
                rows.append({"variant": variant, "exit_model": em, "filter": fname,
                             **trade_metrics(sub)})
    vsum = pd.DataFrame(rows)
    vsum.to_csv("strategy_variant_summary.csv", index=False)

    show = ["variant", "exit_model", "filter", "trades", "win_rate", "expectancy",
            "profit_factor", "sharpe", "max_drawdown", "pct_stop", "pct_target"]
    base_grid = vsum[vsum["filter"] == "all"][show]
    print("\n=== Strategy variants (filter=all, after costs) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(base_grid.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nWrote: tradingview_premarket_gappers.csv, baseline_gapper_fade_results.csv,")
    print("       baseline_gapper_fade_summary.csv, strategy_trades.csv,")
    print("       strategy_variant_summary.csv, rejected_or_skipped_days.csv")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="+15% premarket gapper intraday short-fade research (research only)")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-06-24")
    p.add_argument("--gap-threshold", type=float, default=15.0, help="percent, e.g. 15")
    p.add_argument("--max-symbols", type=int, default=100)
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--slippage-bps", type=float, default=10.0, help="per side")
    p.add_argument("--commission-bps", type=float, default=0.0, help="per side")
    p.add_argument("--borrow-fee-bps-per-day", type=float, default=0.0)
    p.add_argument("--cache-dir", default="alpaca_minute_cache")
    p.add_argument("--strict-low-break", action="store_true", help="run only A2 (close<first_15_low)")
    p.add_argument("--min-rel-volume", type=float, default=0.0, help="hard pre-filter on premarket rel volume")
    p.add_argument("--min-price", type=float, default=0.0)
    p.add_argument("--max-price", type=float, default=float("inf"))
    p.add_argument("--max-market-cap", type=float, default=float("inf"), help="dollars; inf = off")
    p.add_argument("--skip-fetch-existing-cache", action="store_true", help="offline: only use cached chunks")
    p.add_argument("--universe-source", default="tradingview",
                   choices=["tradingview", "pit_archive"],
                   help="tradingview = today's scan (survivorship-biased); "
                        "pit_archive = point-in-time daily scans")
    p.add_argument("--pit-archive-dir", default="data/pit_gappers")
    a = p.parse_args()
    return Config(
        start=a.start, end=a.end, gap_threshold=a.gap_threshold, max_symbols=a.max_symbols,
        feed=a.feed, slippage_bps=a.slippage_bps, commission_bps=a.commission_bps,
        borrow_fee_bps_per_day=a.borrow_fee_bps_per_day, cache_dir=a.cache_dir,
        strict_low_break=a.strict_low_break, min_rel_volume=a.min_rel_volume,
        min_price=a.min_price, max_price=a.max_price, max_market_cap=a.max_market_cap,
        skip_fetch_existing_cache=a.skip_fetch_existing_cache,
        universe_source=a.universe_source, pit_archive_dir=a.pit_archive_dir,
    )


if __name__ == "__main__":
    run(parse_args())
