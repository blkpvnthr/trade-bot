#!/usr/bin/env python3
"""
scanner.py

Standalone bullish trading-universe creator.

Runs:
    python scanner.py

Purpose:
    - Create a locked daily long-only trading universe.
    - Never launches the executor.
    - Never calls scanner.py recursively.
    - Removes symbols with special characters entirely.
    - Excludes inverse / leveraged ETFs.
    - Prefers existing momentum scanner outputs when available.
    - Falls back to Alpaca daily bars when the existing input list is too small.
    - Outputs a practical universe sorted by:
        1. avg_dollar_volume20
        2. daily_chg_pct
        3. momentum_score

Outputs:
    locked_trading_universe.txt
    locked_trading_universe_csv.txt
    locked_trading_universe.csv
    locked_trading_universe_rejected.csv
    market_movers_diagnostics.csv
    regular_positive_momentum_universe.txt
    leading_index_universe.txt
    leading_index_scores.csv

Notes:
    This file is intentionally a universe creator only.
    It does not import, create, or run ProductionOrderExecutor.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    DataFeed = None
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None


# =============================================================================
# Paths
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

OUT_DIR = PROJECT_ROOT
DAILY_OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
DAILY_OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SYMBOLS_TXT = PROJECT_ROOT / "latest_breakout_momentum_symbols.txt"
INPUT_SYMBOLS_CSV_TXT = PROJECT_ROOT / "latest_breakout_momentum_symbols_csv.txt"
INPUT_CANDIDATES_CSV = PROJECT_ROOT / "latest_breakout_momentum_candidates.csv"

LOCKED_TXT = PROJECT_ROOT / "locked_trading_universe.txt"
LOCKED_CSV_LIST_TXT = PROJECT_ROOT / "locked_trading_universe_csv.txt"
LOCKED_CSV = PROJECT_ROOT / "locked_trading_universe.csv"
REJECTED_CSV = PROJECT_ROOT / "locked_trading_universe_rejected.csv"
DIAGNOSTICS_CSV = PROJECT_ROOT / "market_movers_diagnostics.csv"

REGULAR_UNIVERSE_TXT = PROJECT_ROOT / "regular_positive_momentum_universe.txt"
LEADING_INDEX_UNIVERSE_TXT = PROJECT_ROOT / "leading_index_universe.txt"
LEADING_INDEX_SCORES_CSV = PROJECT_ROOT / "leading_index_scores.csv"


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ScannerSettings:
    target_count: int = 30
    min_output_count_before_alpaca_fallback: int = 20

    max_price: float = 100.0
    min_price: float = 1.00

    # This is intentionally loose. If fewer than target_count pass, the scanner
    # automatically relaxes liquidity by ranking instead of failing.
    preferred_min_avg_dollar_volume20: float = 3_000_000.0
    hard_min_avg_dollar_volume20: float = 250_000.0

    lookback_days: int = 220
    data_feed: str = "iex"

    # Leadership scoring.
    leadership_lookback: int = 90
    leadership_min_obs: int = 40
    leadership_lags: tuple[int, ...] = (1, 2, 3, 5)
    benchmarks: tuple[str, ...] = ("SPY", "QQQ")

    # If the existing input file contains only one symbol, this prevents a
    # locked universe of one ticker unless Alpaca is unavailable.
    force_fallback_when_input_count_below: int = 5


SETTINGS = ScannerSettings()


# A practical fallback list biased toward liquid common stocks under/near $100,
# plus benchmarks for leadership scoring. Leveraged/inverse ETFs are filtered out.
DEFAULT_SCAN_UNIVERSE = [
    "SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "AMD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AVGO",
    "INTC", "MU", "PLTR", "COIN", "MARA", "RIOT", "RBLX", "SNAP", "SOFI",
    "HOOD", "OPEN", "RIVN", "LCID", "NIO", "F", "GM", "AAL", "UAL", "DAL",
    "CCL", "NCLH", "RCL", "WBD", "PARA", "PFE", "BAC", "C", "WFC", "KEY",
    "T", "VZ", "PYPL", "SHOP", "SQ", "UBER", "LYFT", "DKNG", "RKT", "AFRM",
    "ROKU", "SOUN", "IONQ", "QBTS", "RGTI", "QUBT", "APLD", "OKLO", "ACHR",
    "JOBY", "RKLB", "ASTS", "LUNR", "DNA", "BBAI", "AI", "PATH", "U", "NET",
    "DDOG", "SNOW", "CRWD", "MRVL", "ON", "WDC", "STX", "HPQ", "DELL",
    "DIS", "NKE", "SBUX", "KO", "KHC", "CMCSA", "PINS", "ETSY", "CHWY",
    "PTON", "UPST", "CVNA", "GME", "AMC", "KSS", "M", "JWN", "BB", "NOK",
]

# Remove these even if they appear in input files.
EXCLUDED_EXACT_SYMBOLS = {
    # Inverse / short ETFs.
    "SQQQ", "SPXU", "SH", "PSQ", "DOG", "DXD", "SDOW", "SDS", "QID",
    "SRTY", "TZA", "SOXS", "TECS", "LABD", "FAZ", "DRV", "ERY", "YANG",
    "VIXY", "UVXY", "SVXY",

    # Leveraged long ETFs.
    "TQQQ", "UPRO", "SPXL", "SSO", "QLD", "UDOW", "SOXL", "TECL", "FAS",
    "LABU", "TNA", "FNGU", "BULZ", "NAIL", "WEBL", "CURE", "ERX", "YINN",
}

LEVERAGED_OR_INVERSE_NAME_TERMS = (
    "2X", "3X", "ULTRA", "ULTRAPRO", "DAILY BULL", "DAILY BEAR",
    "INVERSE", "SHORT", "BEAR", "LEVERAGED",
)


# =============================================================================
# Utility helpers
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def clean_symbol(value: Any) -> str:
    """Uppercase a symbol without converting dots/dashes.

    Symbols containing dots, dashes, slashes, spaces, or other special chars are
    rejected later. This intentionally removes BF-B / BRK.B style symbols.
    """
    return str(value or "").upper().strip().replace('"', "").replace("'", "")


def is_plain_symbol(symbol: str) -> bool:
    """Allow only simple common-stock style tickers.

    This intentionally excludes special-character symbols such as BF-B.
    """
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]{0,4}", symbol))


def is_excluded_symbol(symbol: str) -> bool:
    return clean_symbol(symbol) in EXCLUDED_EXACT_SYMBOLS


def to_float(value: Any, default: float = np.nan) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def pct_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return s.rank(pct=True).fillna(0.0)


def unique_keep_order(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        sym = clean_symbol(value)
        if sym and sym not in out:
            out.append(sym)
    return out


def reject_row(symbol: str, reason: str, source: str = "") -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc().isoformat(),
        "symbol": symbol,
        "reason": reason,
        "source": source,
    }


# =============================================================================
# Input readers
# =============================================================================

def read_symbol_text_file(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\n", ",").replace("\t", ",")
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    return unique_keep_order(tokens)


def find_symbol_column(df: pd.DataFrame) -> str | None:
    candidates = ["symbol", "ticker", "Symbol", "Ticker"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def read_candidates_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[input] failed reading {path}: {exc}")
        return pd.DataFrame()

    symbol_col = find_symbol_column(df)
    if symbol_col is None:
        return pd.DataFrame()

    df = df.copy()
    df["symbol"] = df[symbol_col].map(clean_symbol)
    return df


def load_existing_inputs() -> tuple[list[str], pd.DataFrame]:
    symbols: list[str] = []

    for path in [INPUT_SYMBOLS_CSV_TXT, INPUT_SYMBOLS_TXT]:
        file_symbols = read_symbol_text_file(path)
        if file_symbols:
            print(f"[input] loaded {len(file_symbols)} symbols from {path}")
            symbols.extend(file_symbols)

    candidates = read_candidates_csv(INPUT_CANDIDATES_CSV)
    if not candidates.empty:
        print(f"[input] loaded {len(candidates)} candidate rows from {INPUT_CANDIDATES_CSV}")
        symbols.extend(candidates["symbol"].tolist())

    return unique_keep_order(symbols), candidates


# =============================================================================
# Alpaca data
# =============================================================================

def get_data_feed(settings: ScannerSettings):
    if DataFeed is None:
        return None
    return DataFeed.IEX if settings.data_feed.lower() == "iex" else DataFeed.SIP


def make_data_client() -> StockHistoricalDataClient | None:
    if StockHistoricalDataClient is None:
        print("[alpaca] alpaca-py data client unavailable; install alpaca-py for fallback scanning")
        return None

    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        print(f"[alpaca] missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in {ENV_PATH}")
        return None

    return StockHistoricalDataClient(api_key, secret)


def download_daily_bars(symbols: list[str], settings: ScannerSettings) -> pd.DataFrame:
    client = make_data_client()
    if client is None:
        return pd.DataFrame()

    symbols = unique_keep_order(symbols)
    symbols = [s for s in symbols if is_plain_symbol(s) and not is_excluded_symbol(s)]

    if not symbols:
        return pd.DataFrame()

    end = now_utc()
    start = end - timedelta(days=max(45, settings.lookback_days))

    all_frames: list[pd.DataFrame] = []
    chunk_size = 75

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=get_data_feed(settings),
            )
            bars = client.get_stock_bars(req).df
            if bars is None or bars.empty:
                continue

            df = bars.reset_index()
            if "symbol" not in df.columns and "level_0" in df.columns:
                df = df.rename(columns={"level_0": "symbol"})
            if "timestamp" not in df.columns and "level_1" in df.columns:
                df = df.rename(columns={"level_1": "timestamp"})

            df["symbol"] = df["symbol"].map(clean_symbol)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            all_frames.append(df)
            print(f"[alpaca] downloaded chunk {i // chunk_size + 1} symbols={len(chunk)} rows={len(df)}")
            time.sleep(0.35)
        except Exception as exc:
            print(f"[alpaca] failed chunk {i // chunk_size + 1} symbols={chunk[:5]}... error={exc}")
            time.sleep(1.0)

    if not all_frames:
        return pd.DataFrame()

    out = pd.concat(all_frames, ignore_index=True)
    out = out.dropna(subset=["symbol", "timestamp", "close"])
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


# =============================================================================
# Feature engineering
# =============================================================================

def build_features_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()

    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        print(f"[features] missing columns: {sorted(missing)}")
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []

    for symbol, g in bars.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").copy()
        if len(g) < 25:
            continue

        for col in ["open", "high", "low", "close", "volume"]:
            g[col] = pd.to_numeric(g[col], errors="coerce")

        close = g["close"]
        volume = g["volume"]

        g["prev_close"] = close.shift(1)
        g["prev_high"] = g["high"].shift(1)

        g["daily_chg_pct"] = close / g["prev_close"].replace(0, np.nan) - 1.0
        g["ret_3d"] = close.pct_change(3, fill_method=None)
        g["ret_5d"] = close.pct_change(5, fill_method=None)
        g["ret_10d"] = close.pct_change(10, fill_method=None)

        g["sma5"] = close.rolling(5).mean()
        g["sma10"] = close.rolling(10).mean()
        g["sma20"] = close.rolling(20).mean()
        g["sma50"] = close.rolling(50).mean()

        g["avg_volume20"] = volume.rolling(20).mean()
        g["avg_dollar_volume20"] = (close * volume).rolling(20).mean()
        g["relative_volume"] = volume / g["avg_volume20"].replace(0, np.nan)

        g["close_above_prev_high"] = close > g["prev_high"]
        g["close_above_sma20"] = close > g["sma20"]
        g["sma5_above_sma20"] = g["sma5"] > g["sma20"]
        g["sma20_above_sma50"] = g["sma20"] > g["sma50"]

        rows.append(g)

    if not rows:
        return pd.DataFrame()

    features = pd.concat(rows, ignore_index=True)
    latest = features.groupby("symbol", as_index=False).tail(1).copy()

    if latest.empty:
        return latest

    latest["volume_rank"] = pct_rank(latest["avg_dollar_volume20"])
    latest["daily_chg_rank"] = pct_rank(latest["daily_chg_pct"])
    latest["ret_5d_rank"] = pct_rank(latest["ret_5d"])
    latest["ret_10d_rank"] = pct_rank(latest["ret_10d"])
    latest["relative_volume_rank"] = pct_rank(latest["relative_volume"])

    trend_bonus = (
        latest["close_above_sma20"].astype(float).fillna(0.0) * 0.25
        + latest["sma5_above_sma20"].astype(float).fillna(0.0) * 0.20
        + latest["sma20_above_sma50"].astype(float).fillna(0.0) * 0.15
        + latest["close_above_prev_high"].astype(float).fillna(0.0) * 0.15
    )

    latest["momentum_score"] = (
        0.25 * latest["daily_chg_rank"]
        + 0.20 * latest["ret_5d_rank"]
        + 0.15 * latest["ret_10d_rank"]
        + 0.15 * latest["relative_volume_rank"]
        + trend_bonus
    ).clip(0, 1)

    return latest.reset_index(drop=True)


def build_candidate_frame_from_existing(symbols: list[str], candidates: pd.DataFrame) -> pd.DataFrame:
    """Use existing files if they already contain ranked candidate data."""
    if not candidates.empty and "symbol" in candidates.columns:
        df = candidates.copy()
    else:
        df = pd.DataFrame({"symbol": symbols})

    if df.empty:
        return df

    df["symbol"] = df["symbol"].map(clean_symbol)

    defaults = {
        "close": np.nan,
        "daily_chg_pct": np.nan,
        "avg_dollar_volume20": np.nan,
        "momentum_score": np.nan,
        "relative_volume": np.nan,
        "source": "existing_input",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Normalize common alternative column names.
    rename_map = {
        "price": "close",
        "chg_pct": "daily_chg_pct",
        "change_pct": "daily_chg_pct",
        "daily_change_pct": "daily_chg_pct",
        "dollar_volume20": "avg_dollar_volume20",
        "avg_dollar_volume": "avg_dollar_volume20",
        "score": "momentum_score",
    }
    for old, new in rename_map.items():
        if old in df.columns and new in df.columns:
            df[new] = df[new].where(df[new].notna(), df[old])

    for col in ["close", "daily_chg_pct", "avg_dollar_volume20", "momentum_score", "relative_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text lists have no data. Give them neutral score so they can be validated
    # by Alpaca fallback if needed.
    df["momentum_score"] = df["momentum_score"].fillna(0.50)

    return df


# =============================================================================
# Filtering and ranking
# =============================================================================

def filter_and_rank(df: pd.DataFrame, settings: ScannerSettings, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rejected: list[dict[str, Any]] = []

    if df.empty:
        return df, pd.DataFrame()

    work = df.copy()
    work["symbol"] = work["symbol"].map(clean_symbol)

    keep_mask = []
    for _, row in work.iterrows():
        symbol = clean_symbol(row.get("symbol", ""))
        reason = ""

        if not symbol:
            reason = "blank_symbol"
        elif not is_plain_symbol(symbol):
            reason = "special_character_or_invalid_symbol"
        elif is_excluded_symbol(symbol):
            reason = "inverse_or_leveraged_etf_excluded"

        close = to_float(row.get("close"), np.nan)
        if not reason and not pd.isna(close):
            if close < settings.min_price:
                reason = "below_min_price"
            elif close > settings.max_price:
                reason = "above_max_price"

        daily_chg = to_float(row.get("daily_chg_pct"), np.nan)
        if not reason and not pd.isna(daily_chg) and daily_chg <= 0:
            reason = "non_positive_daily_change"

        avg_dv = to_float(row.get("avg_dollar_volume20"), np.nan)
        if not reason and not pd.isna(avg_dv) and avg_dv < settings.hard_min_avg_dollar_volume20:
            reason = "below_hard_liquidity_floor"

        if reason:
            rejected.append(reject_row(symbol, reason, source))
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    accepted = work.loc[keep_mask].copy()

    if accepted.empty:
        return accepted, pd.DataFrame(rejected)

    # Prefer high liquidity, but do not over-filter. This prevents empty output.
    accepted["liquidity_preferred"] = (
        pd.to_numeric(accepted["avg_dollar_volume20"], errors="coerce")
        .fillna(0.0)
        >= settings.preferred_min_avg_dollar_volume20
    )

    # If existing text input has no numeric data, keep order but still write CSV.
    for col in ["avg_dollar_volume20", "daily_chg_pct", "momentum_score"]:
        if col not in accepted.columns:
            accepted[col] = np.nan

    accepted["avg_dollar_volume20_sort"] = pd.to_numeric(accepted["avg_dollar_volume20"], errors="coerce").fillna(-1)
    accepted["daily_chg_pct_sort"] = pd.to_numeric(accepted["daily_chg_pct"], errors="coerce").fillna(-1)
    accepted["momentum_score_sort"] = pd.to_numeric(accepted["momentum_score"], errors="coerce").fillna(0)

    accepted = accepted.sort_values(
        ["liquidity_preferred", "avg_dollar_volume20_sort", "daily_chg_pct_sort", "momentum_score_sort"],
        ascending=[False, False, False, False],
    )

    accepted = accepted.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    accepted["rank"] = np.arange(1, len(accepted) + 1)
    accepted["source"] = source

    drop_cols = [c for c in accepted.columns if c.endswith("_sort")]
    accepted = accepted.drop(columns=drop_cols)

    return accepted, pd.DataFrame(rejected)


# =============================================================================
# Leadership scoring
# =============================================================================

def build_leadership_scores(bars: pd.DataFrame, selected_symbols: list[str], settings: ScannerSettings) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()

    needed = {"timestamp", "symbol", "close"}
    if not needed.issubset(bars.columns):
        return pd.DataFrame()

    df = bars.copy()
    df["symbol"] = df["symbol"].map(clean_symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "close"])

    close = (
        df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )

    if close.empty:
        return pd.DataFrame()

    returns = close.pct_change(fill_method=None).dropna(how="all").tail(settings.leadership_lookback)
    benchmarks = [b for b in settings.benchmarks if b in returns.columns]
    if not benchmarks:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for symbol in selected_symbols:
        if symbol not in returns.columns or symbol in benchmarks:
            continue

        symbol_ret = returns[symbol]
        lag_scores = []

        for bench in benchmarks:
            bench_ret = returns[bench]
            for lag in settings.leadership_lags:
                aligned = pd.concat(
                    [
                        symbol_ret.shift(lag).rename("symbol_ret"),
                        bench_ret.rename("bench_ret"),
                    ],
                    axis=1,
                ).dropna()

                if len(aligned) < settings.leadership_min_obs:
                    continue

                corr = aligned["symbol_ret"].corr(aligned["bench_ret"])
                if pd.isna(corr):
                    continue

                weighted = float(corr) / max(float(lag), 1.0)
                lag_scores.append(weighted)

                rows.append(
                    {
                        "symbol": symbol,
                        "benchmark": bench,
                        "lag": lag,
                        "lagged_corr": float(corr),
                        "weighted_lagged_corr": weighted,
                        "observations": len(aligned),
                    }
                )

    if not rows:
        return pd.DataFrame()

    matrix = pd.DataFrame(rows)
    scores = (
        matrix.groupby("symbol", as_index=False)
        .agg(
            leadership_score=("weighted_lagged_corr", "mean"),
            leadership_power=("weighted_lagged_corr", lambda s: float(np.nanmean(np.abs(s)))),
            best_lagged_corr=("lagged_corr", lambda s: float(s.iloc[np.nanargmax(np.abs(s.to_numpy()))]) if len(s) else 0.0),
            observations=("observations", "max"),
        )
    )
    scores = scores.sort_values(["leadership_power", "leadership_score"], ascending=[False, False]).reset_index(drop=True)
    scores["leadership_rank"] = np.arange(1, len(scores) + 1)
    return scores


# =============================================================================
# Output writers
# =============================================================================

def write_outputs(final_df: pd.DataFrame, rejected_df: pd.DataFrame, leadership_df: pd.DataFrame) -> None:
    final_df = final_df.copy()
    symbols = final_df["symbol"].astype(str).tolist()

    LOCKED_TXT.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    LOCKED_CSV_LIST_TXT.write_text(",".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    final_df.to_csv(LOCKED_CSV, index=False)

    if rejected_df is None or rejected_df.empty:
        rejected_df = pd.DataFrame(columns=["timestamp_utc", "symbol", "reason", "source"])
    rejected_df.to_csv(REJECTED_CSV, index=False)

    final_df.to_csv(DIAGNOSTICS_CSV, index=False)

    regular_symbols = final_df["symbol"].head(SETTINGS.target_count).tolist()
    REGULAR_UNIVERSE_TXT.write_text("\n".join(regular_symbols) + ("\n" if regular_symbols else ""), encoding="utf-8")

    if leadership_df is not None and not leadership_df.empty:
        leadership_df.to_csv(LEADING_INDEX_SCORES_CSV, index=False)
        lead_symbols = leadership_df["symbol"].head(SETTINGS.target_count).astype(str).tolist()
    else:
        pd.DataFrame().to_csv(LEADING_INDEX_SCORES_CSV, index=False)
        lead_symbols = []

    LEADING_INDEX_UNIVERSE_TXT.write_text("\n".join(lead_symbols) + ("\n" if lead_symbols else ""), encoding="utf-8")

    print("[universe] created locked trading universe")
    print(f"[universe] input={INPUT_SYMBOLS_CSV_TXT}")
    print(f"[universe] symbols={len(symbols)}")
    print(f"[universe] rejected={len(rejected_df)}")
    print(f"[universe] output_txt={LOCKED_TXT}")
    print(f"[universe] output_csv_list={LOCKED_CSV_LIST_TXT}")
    print(f"[universe] output_csv={LOCKED_CSV}")
    print(f"[universe] rejected_csv={REJECTED_CSV}")
    print(f"[universe] regular_universe={REGULAR_UNIVERSE_TXT}")
    print(f"[universe] leading_index_universe={LEADING_INDEX_UNIVERSE_TXT}")
    if symbols:
        print("[universe] " + ", ".join(symbols))


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    settings = SETTINGS

    raw_symbols, existing_candidates = load_existing_inputs()
    existing_df = build_candidate_frame_from_existing(raw_symbols, existing_candidates)

    accepted_existing, rejected_existing = filter_and_rank(existing_df, settings, source="existing_files")

    use_alpaca_fallback = (
        len(accepted_existing) < settings.force_fallback_when_input_count_below
        or len(accepted_existing) < settings.min_output_count_before_alpaca_fallback
        or accepted_existing[["close", "daily_chg_pct", "avg_dollar_volume20"]].isna().all(axis=None)
    )

    bars = pd.DataFrame()
    accepted_fallback = pd.DataFrame()
    rejected_fallback = pd.DataFrame()

    if use_alpaca_fallback:
        print(
            "[fallback] existing inputs are too small or lack market data; "
            "building a fresh bullish market-movers universe from Alpaca"
        )

        seed_symbols = unique_keep_order(list(raw_symbols) + DEFAULT_SCAN_UNIVERSE + list(settings.benchmarks))
        bars = download_daily_bars(seed_symbols, settings)

        if not bars.empty:
            features = build_features_from_bars(bars)
            accepted_fallback, rejected_fallback = filter_and_rank(features, settings, source="alpaca_daily_bars")
        else:
            print("[fallback] no Alpaca bars available; using validated existing inputs only")

    frames = []
    if not accepted_fallback.empty:
        frames.append(accepted_fallback)
    if not accepted_existing.empty:
        frames.append(accepted_existing)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["symbol"])

    if combined.empty:
        rejected = pd.concat(
            [df for df in [rejected_existing, rejected_fallback] if df is not None and not df.empty],
            ignore_index=True,
        ) if (not rejected_existing.empty or not rejected_fallback.empty) else pd.DataFrame()
        write_outputs(combined, rejected, pd.DataFrame())
        print("[error] no valid symbols found")
        return 1

    # Re-rank after merging fallback + existing candidates.
    combined["symbol"] = combined["symbol"].map(clean_symbol)
    combined = combined.drop_duplicates(subset=["symbol"], keep="first")

    for col in ["avg_dollar_volume20", "daily_chg_pct", "momentum_score"]:
        if col not in combined.columns:
            combined[col] = np.nan
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["avg_dollar_volume20_sort"] = combined["avg_dollar_volume20"].fillna(-1)
    combined["daily_chg_pct_sort"] = combined["daily_chg_pct"].fillna(-1)
    combined["momentum_score_sort"] = combined["momentum_score"].fillna(0)

    combined = combined.sort_values(
        ["avg_dollar_volume20_sort", "daily_chg_pct_sort", "momentum_score_sort"],
        ascending=[False, False, False],
    ).drop(columns=["avg_dollar_volume20_sort", "daily_chg_pct_sort", "momentum_score_sort"])

    final = combined.head(settings.target_count).reset_index(drop=True)
    final["rank"] = np.arange(1, len(final) + 1)

    if bars.empty:
        # Try to fetch only final + benchmarks for leadership scoring if possible.
        leadership_seed = unique_keep_order(final["symbol"].tolist() + list(settings.benchmarks))
        bars = download_daily_bars(leadership_seed, settings)

    leadership = build_leadership_scores(bars, final["symbol"].tolist(), settings)
    if not leadership.empty:
        # Keep only selected universe symbols for leader file.
        leadership = leadership[leadership["symbol"].isin(set(final["symbol"]))].reset_index(drop=True)

    rejected = pd.concat(
        [df for df in [rejected_existing, rejected_fallback] if df is not None and not df.empty],
        ignore_index=True,
    ) if (not rejected_existing.empty or not rejected_fallback.empty) else pd.DataFrame()

    write_outputs(final, rejected, leadership)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
