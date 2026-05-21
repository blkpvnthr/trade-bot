#!/usr/bin/env python3
"""
scripts/replay_market_day.py

Replay a historical market day against every registered strategy variant.
Loads OHLCV bars from the Alpaca paper data API (or falls back to yfinance),
feeds them bar-by-bar to generate_signals(), accumulates signal events,
and writes daily_system_outputs/replay_results.json.

Usage:
  python scripts/replay_market_day.py [--date YYYY-MM-DD] [--symbols AAPL,TSLA]
  python scripts/replay_market_day.py --date 2026-01-02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY, get_strategy

OUTPUT_DIR = PROJECT_ROOT / "daily_system_outputs"

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
DEFAULT_TIMEFRAME = "1Min"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_alpaca_bars(
    symbols: list[str],
    trade_date: date,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> dict[str, pd.DataFrame]:
    """Load 1-min OHLCV bars from Alpaca paper API. Returns {symbol: DataFrame}."""
    api_key = os.getenv("APCA_API_KEY_ID", "")
    api_secret = os.getenv("APCA_API_SECRET_KEY", "")
    if not api_key or not api_secret or api_key.startswith(("YOUR_", "REPLACE")):
        return {}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(api_key, api_secret)
        start = datetime(trade_date.year, trade_date.month, trade_date.day, 4, 0, tzinfo=timezone.utc)
        end   = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 59, tzinfo=timezone.utc)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
            end=end,
        )
        bars_multi = client.get_stock_bars(req).df

        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                sym_df = bars_multi.loc[sym].copy() if sym in bars_multi.index.get_level_values(0) else pd.DataFrame()
                if not sym_df.empty:
                    sym_df = sym_df.rename(columns={
                        "open": "open", "high": "high", "low": "low",
                        "close": "close", "volume": "volume",
                    })
                    result[sym] = sym_df[["open", "high", "low", "close", "volume"]]
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _load_yfinance_bars(
    symbols: list[str],
    trade_date: date,
) -> dict[str, pd.DataFrame]:
    """Fallback: load 1-min bars from yfinance."""
    try:
        import yfinance as yf
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                start_str = trade_date.strftime("%Y-%m-%d")
                end_str   = (trade_date + timedelta(days=1)).strftime("%Y-%m-%d")
                df = ticker.history(start=start_str, end=end_str, interval="1m")
                if df.empty:
                    continue
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                df.index = pd.to_datetime(df.index, utc=True)
                result[sym] = df[["open", "high", "low", "close", "volume"]]
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _synthetic_bars(
    symbols: list[str],
    trade_date: date,
    n_bars: int = 390,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic intraday bars when no real data is available."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    start_et = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=ET)
    ts = pd.DatetimeIndex([
        (start_et + timedelta(minutes=i)).astimezone(timezone.utc)
        for i in range(n_bars)
    ])
    result: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(hash(sym) % (2**32) + i)
        close = 100.0 + np.cumsum(rng.normal(0, 0.1, n_bars))
        close = np.maximum(close, 1.0)
        noise = rng.uniform(0.001, 0.004, n_bars)
        df = pd.DataFrame({
            "open":   close * (1 - noise / 2),
            "high":   close * (1 + noise),
            "low":    close * (1 - noise),
            "close":  close,
            "volume": rng.integers(5_000, 80_000, n_bars).astype(float),
        }, index=ts)
        result[sym] = df
    return result


def _load_bars(
    symbols: list[str],
    trade_date: date,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Try Alpaca → yfinance → synthetic. Returns (bars_dict, source_label)."""
    bars = _load_alpaca_bars(symbols, trade_date)
    if bars:
        return bars, "alpaca"

    bars = _load_yfinance_bars(symbols, trade_date)
    if bars:
        return bars, "yfinance"

    bars = _synthetic_bars(symbols, trade_date)
    return bars, "synthetic"


# ─────────────────────────────────────────────────────────────────────────────
# Replay engine
# ─────────────────────────────────────────────────────────────────────────────

def _build_pivot_context(bars: pd.DataFrame) -> dict[str, float]:
    """Simple pivot calc from prior-day high/low/close (uses first 30 bars as proxy)."""
    if len(bars) < 30:
        return {}
    h = float(bars["high"].iloc[:30].max())
    l = float(bars["low"].iloc[:30].min())
    c = float(bars["close"].iloc[29])
    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    r2 = pp + (h - l)
    s1 = 2 * pp - h
    s2 = pp - (h - l)
    return {
        "pivot_pp": pp, "pivot_r1": r1, "pivot_r2": r2,
        "pivot_r3": r2 + (r1 - pp), "pivot_s1": s1, "pivot_s2": s2,
        "pivot_s3": s2 - (pp - s1),
    }


def _replay_strategy(
    strategy_id: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    min_bars: int = 50,
) -> dict[str, Any]:
    """Replay one strategy across all symbols. Returns replay result dict."""
    result: dict[str, Any] = {
        "strategy_id": strategy_id,
        "symbols_tested": 0,
        "signals": [],
        "exceptions": [],
        "total_buy": 0,
        "total_sell": 0,
        "total_short": 0,
        "total_cover": 0,
        "passed": True,
    }

    try:
        strategy = get_strategy(strategy_id)
    except Exception as exc:
        result["exceptions"].append(f"instantiation: {exc}")
        result["passed"] = False
        return result

    for sym, bars in bars_by_symbol.items():
        if len(bars) < min_bars:
            continue

        result["symbols_tested"] += 1

        ctx: dict[str, Any] = {}
        if strategy_id == "pivot_points":
            ctx = _build_pivot_context(bars)

        try:
            signals = strategy.generate_signals(bars, ctx)
            for col, total_key in [
                ("buy_signal", "total_buy"),
                ("sell_signal", "total_sell"),
                ("short_signal", "total_short"),
                ("cover_signal", "total_cover"),
            ]:
                count = int(signals[col].astype(bool).sum())
                result[total_key] += count
                if count > 0:
                    # Record the first signal row
                    idx = signals[signals[col].astype(bool)].index[0]
                    row = signals.loc[idx]
                    result["signals"].append({
                        "symbol": sym,
                        "signal_type": col,
                        "count": count,
                        "first_bar": str(idx),
                        "entry_reason": str(row.get("entry_reason", "")),
                        "stop_price": float(row["stop_price"]) if not pd.isna(row["stop_price"]) else None,
                        "target_price": float(row["target_price"]) if not pd.isna(row["target_price"]) else None,
                        "risk_reward": float(row["risk_reward"]) if not pd.isna(row["risk_reward"]) else None,
                    })
        except Exception as exc:
            result["exceptions"].append(f"{sym}: {type(exc).__name__}: {exc}")
            result["passed"] = False

    return result


def run_replay(
    symbols: list[str],
    trade_date: date,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run full replay for all strategies. Returns summary dict."""
    bars_by_symbol, data_source = _load_bars(symbols, trade_date)

    if verbose and bars_by_symbol:
        first_sym = next(iter(bars_by_symbol))
        print(f"[replay] date={trade_date}  symbols={len(bars_by_symbol)}  "
              f"source={data_source}  bars_per_symbol~{len(bars_by_symbol[first_sym])}")

    results: list[dict[str, Any]] = []
    for strategy_id in STRATEGY_REGISTRY:
        res = _replay_strategy(strategy_id, bars_by_symbol)
        results.append(res)
        status = "PASS" if res["passed"] else "FAIL"
        sig_counts = (f"buy={res['total_buy']} sell={res['total_sell']} "
                      f"short={res['total_short']} cover={res['total_cover']}")
        if verbose:
            print(f"  [{status}] {strategy_id}: {sig_counts}  "
                  f"exceptions={len(res['exceptions'])}")
            for exc in res["exceptions"][:2]:
                print(f"         {exc}")

    summary = {
        "date": str(trade_date),
        "data_source": data_source,
        "symbols": list(bars_by_symbol.keys()),
        "strategies": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a market day against all strategy variants")
    parser.add_argument("--date", default=None,
                        help="Trade date YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.date:
        trade_date = date.fromisoformat(args.date)
    else:
        today = date.today()
        offset = max(1, today.weekday() - 4) if today.weekday() >= 5 else 1
        trade_date = today - timedelta(days=offset)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not args.quiet:
        print(f"=== Replay: {trade_date} | {len(symbols)} symbols ===\n")

    summary = run_replay(symbols, trade_date, verbose=not args.quiet)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "replay_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        total_signals = sum(
            s["total_buy"] + s["total_sell"] + s["total_short"] + s["total_cover"]
            for s in summary["strategies"]
        )
        all_pass = all(s["passed"] for s in summary["strategies"])
        print(f"\nTotal signals across all strategies: {total_signals}")
        print(f"{'ALL PASS' if all_pass else 'SOME EXCEPTIONS'}")
        print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
