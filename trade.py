#!/usr/bin/env python3
"""
run_daily_system_walkforward_emulated.py

Daily orchestrator that emulates the walk-forward explosive alpha logic.

Pipeline:
    1. Download Alpaca tradable universe.
    2. Download daily bars once.
    3. Select a dynamic liquid universe.
    4. Build features.
    5. Train an explosive-alpha model using historical feature rows.
    6. Score only the latest rows.
    7. Rank sparse high-conviction breakout/continuation candidates.
    8. Build convex/Kelly/volatility-adjusted target weights.
    9. Build Alpaca order plan through the existing execution layer.
   10. Execute only if --execute is supplied.

Default:
    DRY RUN ONLY.

Run:
    python run_daily_system_walkforward_emulated.py

Paper execution:
    python run_daily_system_walkforward_emulated.py --execute

Useful:
    python run_daily_system_walkforward_emulated.py --top-n 3
    python run_daily_system_walkforward_emulated.py --cash-buffer 0.08
    python run_daily_system_walkforward_emulated.py --max-symbols 3000
"""

from __future__ import annotations

import argparse
import os
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from alpaca_executor import (
    ExecutorConfig,
    get_account_snapshot,
    get_positions_df,
    get_trading_client,
    run_execution,
    save_execution_results,
    scale_executor_config_for_account,
)
from features import (
    REGIME_SYMBOLS,
    build_features,
    normalize_symbol,
)
from portfolio_optimizer import (
    OptimizerConfig,
    optimize_portfolio,
    print_portfolio_summary,
    scale_optimizer_config_for_equity,
)


# =============================================================================
# Paths / environment
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
LIVE_DIR = PROJECT_ROOT


def find_env(start_path: Path) -> Path:
    for parent in [start_path.parent] + list(start_path.parents):
        env_path = parent / ".env"
        if env_path.exists():
            return env_path
    raise FileNotFoundError(f"Could not find .env starting from {start_path}")


ENV_PATH = find_env(THIS_FILE)
load_dotenv(ENV_PATH)

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

print(f"[env] Loaded .env from: {ENV_PATH}")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        f"Missing Alpaca credentials. Found .env at {ENV_PATH}, "
        "but APCA_API_KEY_ID or APCA_API_SECRET_KEY is missing."
    )


OUT_DIR = LIVE_DIR / "daily_system_outputs"
LOG_DIR = LIVE_DIR / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RAW_BARS_PATH = OUT_DIR / "daily_raw_bars.csv"
FEATURES_PATH = OUT_DIR / "daily_features.csv"
TRAIN_EVENTS_PATH = OUT_DIR / "daily_walkforward_emulated_training_events.csv"
PREDICTIONS_PATH = OUT_DIR / "daily_walkforward_emulated_predictions.csv"
RANKED_PATH = OUT_DIR / "daily_walkforward_emulated_ranked_candidates.csv"
TARGETS_PATH = OUT_DIR / "daily_walkforward_emulated_portfolio_targets.csv"
REBALANCE_ORDERS_PATH = OUT_DIR / "daily_walkforward_emulated_rebalance_orders.csv"
EXIT_ORDERS_PATH = OUT_DIR / "daily_walkforward_emulated_exit_orders.csv"
SUMMARY_PATH = OUT_DIR / "daily_walkforward_emulated_summary.txt"
SYMBOLS_PATH = OUT_DIR / "latest_walkforward_emulated_trade_universe.txt"


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42

LOOKBACK_DAYS = 520
CHUNK_SIZE = 20
UNIVERSE_LIMIT = 500

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

MIN_PRICE = 5.00
MAX_PRICE = 250.00
MIN_VOLUME = 500_000
MIN_DAILY_DOLLAR_VOLUME = 10_000_000
MIN_AVG_DOLLAR_VOLUME20 = 10_000_000
MIN_ATR_PCT = 0.004
MIN_DAILY_RANGE_PCT = 0.004
MIN_RELATIVE_VOLUME = 0.85
MIN_RANGE_EXPANSION = 0.006

# Explosive target emulates the walk-forward target.
FUTURE_RET_1D = "future_return_1d"
FUTURE_RET_3D = "future_return_3d"
FUTURE_RET_5D = "future_return_5d"
TARGET_LABEL = "explosive_alpha_label"
MIN_FUTURE_3D_RETURN = 0.010
MIN_FUTURE_5D_RETURN = 0.015
TOP_RETURN_QUANTILE = 0.80

# Sparse entry / ranking.
SOFT_MIN_PROBABILITY = 0.38
MIN_EXPLOSIVE_SCORE = 0.45
MIN_EXPLOSION_SETUP_SCORE = 0.45

# Convex sizing.
BASE_MAX_GROSS_EXPOSURE = 0.95
MAX_WEIGHT_PER_POSITION = 0.45
MIN_WEIGHT_PER_POSITION = 0.02
MAX_OPEN_POSITIONS = 3
CONVEX_POWER = 5.0

KELLY_WIN_R = 3.0
KELLY_LOSS_R = 1.0
KELLY_FRACTION = 0.55
MAX_KELLY_WEIGHT = 0.45

TARGET_POSITION_VOL = 0.030
VOL_ADJUST_MIN = 0.30
VOL_ADJUST_MAX = 1.75

REGIME_GROSS_EXPOSURE = {
    "TREND_RISK_ON": 1.00,
    "VOLATILE_RISK_ON": 0.75,
    "CHOP_NEUTRAL": 0.45,
    "DEFENSIVE_RISK_OFF": 0.20,
}

REGIME_WEIGHT_MULTIPLIER = {
    "TREND_RISK_ON": 1.55,
    "VOLATILE_RISK_ON": 1.10,
    "CHOP_NEUTRAL": 0.70,
    "DEFENSIVE_RISK_OFF": 0.40,
}

DROP_MACRO_DOMINANT_FEATURES = {
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "spy_volatility_20d",
    "spy_close_vs_sma_20",
    "qqq_return_1d",
    "qqq_return_5d",
    "qqq_return_20d",
    "qqq_volatility_20d",
    "qqq_close_vs_sma_20",
    "vixy_return_1d",
    "vixy_return_5d",
    "vixy_return_20d",
    "vixy_close_vs_sma_20",
}

LEAKAGE_COLUMNS = {
    FUTURE_RET_1D,
    FUTURE_RET_3D,
    FUTURE_RET_5D,
    "future_close_3d",
    "future_close_5d",
    "next_close",
    "next_ret_1d",
    "forward_sharpe_proxy",
    "continuation_label",
    "next_bar_positive_label",
    "next_bar_tradeable_positive_label",
    "multi_bar_continuation_label",
    "explosive_continuation_label",
    TARGET_LABEL,
    "prediction",
    "model_prediction",
    "selected_threshold",
    "entry_threshold",
    "rank_for_day",
    "target_weight",
    "target_dollars",
    "recommended_shares",
}

ID_COLUMNS = {"timestamp", "symbol", "symbol_id"}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=3000,
        help="Maximum Alpaca tradable assets to initially scan.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--top-n", type=int, default=3)

    parser.add_argument("--min-probability", type=float, default=SOFT_MIN_PROBABILITY)
    parser.add_argument("--min-explosive-score", type=float, default=MIN_EXPLOSIVE_SCORE)

    parser.add_argument("--cash-buffer", type=float, default=None)
    parser.add_argument("--max-gross-exposure", type=float, default=None)
    parser.add_argument("--max-position-weight", type=float, default=None)
    parser.add_argument("--min-position-weight", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--min-trade-dollars", type=float, default=None)
    parser.add_argument("--disable-account-scaling", action="store_true")

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit paper/live orders. Without this flag, dry-run mode is used.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live Alpaca account instead of paper. Not recommended.",
    )
    parser.add_argument(
        "--no-exit-orders",
        action="store_true",
        help="Do not build exit orders for currently held names outside target list.",
    )

    return parser.parse_args()


# =============================================================================
# Utility helpers
# =============================================================================

def save_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def numeric_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def clean_symbol_list(symbols: Iterable[str]) -> list[str]:
    cleaned = []
    for s in symbols:
        try:
            sym = normalize_symbol(str(s).strip().upper())
        except Exception:
            continue

        if not sym:
            continue
        if "/" in sym or "^" in sym:
            continue

        cleaned.append(sym)

    return sorted(set(cleaned))


# =============================================================================
# Alpaca universe / data
# =============================================================================


def _asset_attr_to_str(value) -> str:
    """
    Alpaca SDK enum-safe string conversion.

    Some Alpaca fields are enums such as AssetClass.US_EQUITY or
    AssetStatus.ACTIVE. str(value) may produce 'AssetClass.US_EQUITY',
    so this normalizes enum and string values to lowercase comparable tokens.
    """
    if value is None:
        return ""

    if hasattr(value, "value"):
        return str(value.value).lower().strip()

    text = str(value).lower().strip()

    if "." in text:
        text = text.split(".")[-1]

    return text


def _web_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }


def get_sp500_fallback_symbols() -> list[str]:
    print("[universe fallback] fetching S&P 500 symbols...")

    try:
        resp = requests.get(WIKI_SP500_URL, headers=_web_headers(), timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        if not tables or "Symbol" not in tables[0].columns:
            raise RuntimeError("Could not parse S&P 500 symbols from Wikipedia.")

        symbols = tables[0]["Symbol"].astype(str).map(normalize_symbol).tolist()
        symbols = sorted(set(s for s in symbols if s and s != "NAN"))
        print(f"[universe fallback] S&P 500 symbols={len(symbols):,}")
        return symbols

    except Exception as exc:
        print(f"[universe fallback warning] S&P 500 fetch failed: {exc}")
        return []


def get_nasdaq_fallback_symbols(max_symbols: int | None = None) -> list[str]:
    print("[universe fallback] fetching NASDAQ-listed symbols...")

    try:
        resp = requests.get(NASDAQ_LISTED_URL, headers=_web_headers(), timeout=30)
        resp.raise_for_status()

        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
        data_lines = [line for line in lines if not line.startswith("File Creation Time")]
        df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|")

        if "Symbol" not in df.columns:
            raise RuntimeError("NASDAQ-listed file missing Symbol column.")

        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() == "N"].copy()

        symbols: list[str] = []
        for raw_sym in df["Symbol"].tolist():
            if pd.isna(raw_sym):
                continue
            sym = normalize_symbol(str(raw_sym))
            if not sym or sym == "NAN":
                continue
            # Drop preferreds, warrants, units, rights, special classes, etc.
            if "." in sym or "$" in sym or "/" in sym or "^" in sym:
                continue
            symbols.append(sym)

        symbols = sorted(set(symbols))
        if max_symbols is not None:
            symbols = symbols[:max_symbols]

        print(f"[universe fallback] NASDAQ symbols={len(symbols):,}")
        return symbols

    except Exception as exc:
        print(f"[universe fallback warning] NASDAQ fetch failed: {exc}")
        return []


def get_static_emergency_symbols() -> list[str]:
    """Hard fallback so the script never downloads only regime symbols."""
    return sorted(set([
        "AAPL", "AMD", "AMZN", "AVGO", "COIN", "GOOGL", "META", "MSFT", "MU",
        "NFLX", "NVDA", "PLTR", "SMCI", "SOFI", "TSLA", "TSM", "UBER",
        "ARM", "CRWD", "DDOG", "MSTR", "NET", "PANW", "SHOP", "SNOW",
        "QQQ", "SPY", "DIA", "IWM", "VIXY",
    ]))


def get_fallback_universe(max_symbols: int = 3000) -> list[str]:
    sp500 = get_sp500_fallback_symbols()
    nasdaq = get_nasdaq_fallback_symbols(max_symbols=max_symbols)

    symbols = sorted(set(sp500 + nasdaq + get_static_emergency_symbols() + list(REGIME_SYMBOLS)))

    if max_symbols is not None and len(symbols) > max_symbols:
        # Keep liquid/familiar emergency + regime names, then fill with public fallback.
        required = set(get_static_emergency_symbols()) | set(REGIME_SYMBOLS)
        remaining = [s for s in symbols if s not in required]
        symbols = sorted(set(list(required) + remaining[: max(0, max_symbols - len(required))]))

    print(f"[universe fallback] final fallback symbols={len(symbols):,}")
    return symbols


def get_alpaca_tradeable_symbols(
    trading_client: TradingClient,
    max_symbols: int = 3000,
) -> list[str]:
    print("[universe] downloading Alpaca tradable assets...")

    symbols: list[str] = []
    total_assets = 0
    active_assets = 0
    tradable_assets = 0
    equity_assets = 0

    try:
        assets = trading_client.get_all_assets()
    except Exception as exc:
        print(f"[universe warning] Alpaca get_all_assets failed: {exc}")
        return get_fallback_universe(max_symbols=max_symbols)

    for asset in assets:
        total_assets += 1
        try:
            tradable = bool(getattr(asset, "tradable", False))
            status = _asset_attr_to_str(getattr(asset, "status", ""))
            asset_class = _asset_attr_to_str(getattr(asset, "asset_class", ""))

            if status == "active":
                active_assets += 1
            if tradable:
                tradable_assets += 1

            is_equity = asset_class in {"us_equity", "us-equity", "equity", "stock", "us equity"}
            if is_equity:
                equity_assets += 1

            if not tradable:
                continue
            if status and status != "active":
                continue
            if not is_equity:
                continue

            symbol = normalize_symbol(getattr(asset, "symbol", ""))
            if not symbol:
                continue
            if "/" in symbol or "^" in symbol or "$" in symbol:
                continue

            symbols.append(symbol)

        except Exception as exc:
            print(f"[universe warning] skipped asset due to error: {exc}")
            continue

    symbols = sorted(set(symbols))

    print(
        "[universe debug] "
        f"assets={total_assets:,} active={active_assets:,} "
        f"tradable={tradable_assets:,} equity={equity_assets:,} usable={len(symbols):,}"
    )

    # Critical fix: if Alpaca assets produced no usable symbols, use a broad public fallback.
    if len(symbols) == 0:
        print("[universe warning] Alpaca assets returned zero usable equities. Using public fallback universe.")
        symbols = get_fallback_universe(max_symbols=max_symbols)
    else:
        if max_symbols is not None:
            symbols = symbols[:max_symbols]
        for required in REGIME_SYMBOLS:
            if required not in symbols:
                symbols.append(required)
        symbols = sorted(set(symbols))

    print(f"[universe] final tradable/download symbols={len(symbols):,}")
    return symbols


def fetch_chunk(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if df.empty:
        return pd.DataFrame()

    needed = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise RuntimeError(f"Alpaca response missing columns: {missing}")

    df = df[needed].copy()
    df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_daily_bars(
    symbols: list[str],
    lookback_days: int,
    chunk_size: int,
) -> pd.DataFrame:
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    symbols = clean_symbol_list(symbols)
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    frames: list[pd.DataFrame] = []

    print(f"[download] {start.date()} -> {end.date()} chunks={len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"[download] chunk={i}/{len(chunks)} symbols={len(chunk)}")

        try:
            df = fetch_chunk(client, chunk, start, end)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"[warning] failed chunk={chunk[:5]}... error={exc}")

    if not frames:
        raise RuntimeError("No daily bars downloaded from Alpaca.")

    data = pd.concat(frames, ignore_index=True)

    data = (
        data.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["symbol", "timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    print(f"[download] rows={len(data):,} symbols={data['symbol'].nunique():,}")
    return data



def select_dynamic_universe_from_bars(
    raw: pd.DataFrame,
    max_symbols: int = UNIVERSE_LIMIT,
) -> list[str]:
    print("[universe] selecting dynamic universe from downloaded bars...")

    if raw is None or raw.empty:
        raise RuntimeError("No raw bars were downloaded from Alpaca. Check API keys/feed/symbol list.")

    raw = raw.copy()
    raw["symbol"] = raw["symbol"].astype(str).map(normalize_symbol)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    print(
        f"[universe debug] raw rows={len(raw):,} "
        f"symbols={raw['symbol'].nunique():,} "
        f"date_min={raw['timestamp'].min()} "
        f"date_max={raw['timestamp'].max()}"
    )

    latest_rows = []

    for sym, g in raw.groupby("symbol", sort=False):
        # Regime symbols are context/proxy symbols, not trade candidates.
        if sym in REGIME_SYMBOLS:
            continue

        g = g.sort_values("timestamp").copy()

        # Use 30 bars instead of 80 so newer listings do not kill the universe.
        if len(g) < 30:
            continue

        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) >= 2 else g.iloc[-1]

        close = float(last["close"])
        prev_close = float(prev["close"])
        volume = float(last["volume"])

        if close <= 0 or prev_close <= 0:
            continue

        dailychg_pct = ((close / prev_close) - 1.0) * 100.0
        dollar_volume = close * volume

        # Keep this permissive here. The model/ranking filters handle quality later.
        if dollar_volume < 1_000_000:
            continue

        latest_rows.append(
            {
                "symbol": sym,
                "price": close,
                "dailychg_pct": dailychg_pct,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "bar_count": len(g),
            }
        )

    latest = pd.DataFrame(latest_rows)

    if latest.empty:
        non_regime = sorted(set(raw["symbol"]) - set(REGIME_SYMBOLS))
        print("[universe warning] no symbols passed dynamic selector.")
        print(f"[universe debug] non_regime_symbols_in_raw={len(non_regime)}")

        if non_regime:
            print("[universe fallback] using downloaded non-regime symbols directly.")
            return non_regime[:max_symbols]

        raise RuntimeError(
            "No valid non-regime symbols available after downloading Alpaca bars. "
            "The download universe contained only REGIME_SYMBOLS. "
            "This means get_alpaca_tradeable_symbols() did not return a usable stock universe."
        )

    latest = latest[
        (latest["price"] >= MIN_PRICE)
        & (latest["price"] <= MAX_PRICE)
        & (latest["dollar_volume"] >= 1_000_000)
    ].copy()

    if latest.empty:
        non_regime = sorted(set(raw["symbol"]) - set(REGIME_SYMBOLS))
        print("[universe warning] no symbols passed final price/liquidity selector.")
        if non_regime:
            print("[universe fallback] using downloaded non-regime symbols directly.")
            return non_regime[:max_symbols]
        raise RuntimeError("No symbols passed dynamic universe liquidity/price filters.")

    latest["universe_score"] = (
        latest["dailychg_pct"].clip(lower=0.0) * 0.45
        + np.log1p(latest["dollar_volume"]) * 0.40
        + np.log1p(latest["volume"]) * 0.15
    )

    latest = latest.sort_values("universe_score", ascending=False)
    symbols = latest["symbol"].head(max_symbols).tolist()

    print(f"[universe] selected={len(symbols)}")
    print("[universe]", ", ".join(symbols[:50]))
    return symbols


# =============================================================================
# Walk-forward-emulated features / target
# =============================================================================

def ensure_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    aliases = {
        "ret_1d": "return_1d",
        "ret_3d": "return_3d",
        "ret_5d": "return_5d",
        "ret_10d": "return_10d",
        "ret_20d": "return_20d",
        "ret_21d": "return_20d",
        "volume_ratio20": "relative_volume_20",
        "dist_sma5": "close_vs_sma_5",
        "dist_sma20": "close_vs_sma_20",
        "dist_sma50": "close_vs_sma_50",
        "dist_sma200": "close_vs_sma_200",
        "rsi": "rsi_14",
        "sma5": "sma_5",
        "sma20": "sma_20",
        "sma50": "sma_50",
        "sma200": "sma_200",
    }

    for new_col, old_col in aliases.items():
        if new_col not in df.columns and old_col in df.columns:
            df[new_col] = df[old_col]

    if "dollar_volume" not in df.columns and {"close", "volume"}.issubset(df.columns):
        df["dollar_volume"] = df["close"] * df["volume"]

    if "avg_dollar_volume20" not in df.columns and "dollar_volume" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).copy()
        df["avg_dollar_volume20"] = (
            df.groupby("symbol")["dollar_volume"]
            .rolling(20)
            .mean()
            .reset_index(level=0, drop=True)
        )

    if "daily_range_pct" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        df["daily_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    if "atr_pct" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        df = df.sort_values(["symbol", "timestamp"]).copy()
        prev_close = df.groupby("symbol")["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_pct"] = (
            tr.groupby(df["symbol"])
            .rolling(14)
            .mean()
            .reset_index(level=0, drop=True)
            / df["close"].replace(0, np.nan)
        )

    if "relative_volume_20" not in df.columns:
        if "volume_sma_20" in df.columns:
            df["relative_volume_20"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
        else:
            df = df.sort_values(["symbol", "timestamp"]).copy()
            vol_sma = (
                df.groupby("symbol")["volume"]
                .rolling(20)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df["relative_volume_20"] = df["volume"] / vol_sma.replace(0, np.nan)

    for col in ["trend_score", "momentum_score", "breakout_score"]:
        if col not in df.columns:
            df[col] = 0.0

    if "risk_score" not in df.columns:
        df["risk_score"] = numeric_series(df, "atr_pct") + numeric_series(df, "daily_range_pct")

    if "high_conviction_score" not in df.columns:
        df["high_conviction_score"] = (
            numeric_series(df, "momentum_score")
            + numeric_series(df, "trend_score")
            + numeric_series(df, "breakout_score")
            - numeric_series(df, "risk_score")
        )

    return df


def add_explosive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    close = numeric_series(df, "close")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    volume = numeric_series(df, "volume")

    if "volume_sma_20" in df.columns:
        volume_sma20 = numeric_series(df, "volume_sma_20", np.nan)
    else:
        volume_sma20 = (
            volume.groupby(df["symbol"])
            .rolling(20)
            .mean()
            .reset_index(level=0, drop=True)
        )

    df["range_expansion"] = numeric_series(df, "daily_range_pct")
    df["volume_spike"] = volume / volume_sma20.replace(0, np.nan)
    df["volume_spike"] = df["volume_spike"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    if {"sma5", "sma20", "sma50"}.issubset(df.columns):
        df["trend_alignment"] = (
            (numeric_series(df, "sma5") > numeric_series(df, "sma20"))
            & (numeric_series(df, "sma20") > numeric_series(df, "sma50"))
        ).astype(int)
    else:
        df["trend_alignment"] = (
            (numeric_series(df, "dist_sma5") > 0)
            & (numeric_series(df, "dist_sma20") > -0.005)
            & (numeric_series(df, "dist_sma50") > -0.02)
        ).astype(int)

    prev_high = high.groupby(df["symbol"]).shift(1)
    prev_low = low.groupby(df["symbol"]).shift(1)

    df["breakout_above_prev_high"] = (close > prev_high).astype(int)
    df["breakdown_below_prev_low"] = (close < prev_low).astype(int)

    df["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    df["close_location"] = df["close_location"].clip(0, 1).fillna(0.5)

    df["explosion_setup_score"] = (
        0.20 * (numeric_series(df, "ret_1d") > 0).astype(int)
        + 0.20 * (numeric_series(df, "ret_5d") > 0).astype(int)
        + 0.15 * df["trend_alignment"]
        + 0.20 * df["breakout_above_prev_high"]
        + 0.15 * (df["volume_spike"] >= 1.2).astype(int)
        + 0.10 * (df["range_expansion"] >= 0.012).astype(int)
    ).clip(0, 1)

    df["late_reversal_risk"] = (
        ((numeric_series(df, "rsi", 50) > 84) & (numeric_series(df, "macd_hist") < 0))
        | ((df["close_location"] < 0.25) & (df["range_expansion"] > 0.04))
        | (df["breakdown_below_prev_low"] == 1)
    ).astype(int)

    return df


def add_tradability_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = numeric_series(df, "close")
    volume = numeric_series(df, "volume")
    dollar_volume = numeric_series(df, "dollar_volume")
    avg_dollar_volume20 = numeric_series(df, "avg_dollar_volume20")
    atr_pct = numeric_series(df, "atr_pct")
    daily_range_pct = numeric_series(df, "daily_range_pct")

    df["tradable_liquid"] = (
        (close >= MIN_PRICE)
        & (volume >= MIN_VOLUME)
        & (dollar_volume >= MIN_DAILY_DOLLAR_VOLUME)
        & (avg_dollar_volume20 >= MIN_AVG_DOLLAR_VOLUME20)
    ).astype(int)

    df["tradable_movement"] = (
        (atr_pct >= MIN_ATR_PCT)
        & (daily_range_pct >= MIN_DAILY_RANGE_PCT)
    ).astype(int)

    df["tradable_symbol"] = (
        (df["tradable_liquid"] == 1)
        & (df["tradable_movement"] == 1)
    ).astype(int)

    return df



def add_momentum_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the momentum_favorable / momentum_negative flags expected by the
    walk-forward-emulated ranking logic.

    This keeps trade.py self-contained even when features.py does not produce
    these exact columns.
    """
    df = df.copy()

    ret_1d = numeric_series(df, "ret_1d")
    ret_3d = numeric_series(df, "ret_3d")
    ret_5d = numeric_series(df, "ret_5d")

    dist_sma5 = numeric_series(df, "dist_sma5")
    dist_sma20 = numeric_series(df, "dist_sma20")
    macd_hist = numeric_series(df, "macd_hist")
    rsi_val = numeric_series(df, "rsi", 50.0)
    rel_vol = numeric_series(df, "relative_volume_20", 1.0)
    high_conviction = numeric_series(df, "high_conviction_score")
    late_reversal = numeric_series(df, "late_reversal_risk")

    df["momentum_favorable"] = (
        (ret_1d > -0.003)
        & (ret_3d > -0.010)
        & (ret_5d > -0.015)
        & (dist_sma5 > -0.015)
        & (dist_sma20 > -0.025)
        & (macd_hist > -0.004)
        & (rsi_val >= 38)
        & (rsi_val <= 86)
        & (rel_vol >= 0.70)
        & (late_reversal == 0)
    ).astype(int)

    df["momentum_negative"] = (
        (ret_1d < 0)
        & (
            (ret_3d < -0.025)
            | (ret_5d < -0.045)
            | (dist_sma5 < -0.035)
            | (dist_sma20 < -0.055)
            | (high_conviction < -0.75)
            | (late_reversal == 1)
        )
    ).astype(int)

    return df


def make_explosive_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if FUTURE_RET_3D not in df.columns:
        df[FUTURE_RET_3D] = numeric_series(df, FUTURE_RET_1D)

    if FUTURE_RET_5D not in df.columns:
        df[FUTURE_RET_5D] = numeric_series(df, FUTURE_RET_1D)

    future3 = numeric_series(df, FUTURE_RET_3D)
    future5 = numeric_series(df, FUTURE_RET_5D)

    q = (
        df.assign(_future3=future3)
        .groupby("timestamp")["_future3"]
        .transform(lambda x: x.quantile(TOP_RETURN_QUANTILE))
    )

    df[TARGET_LABEL] = (
        (
            (future3 >= MIN_FUTURE_3D_RETURN)
            | (future5 >= MIN_FUTURE_5D_RETURN)
            | (future3 >= q)
        )
        & (df["tradable_symbol"] == 1)
        & (numeric_series(df, "range_expansion") >= MIN_DAILY_RANGE_PCT)
    ).astype(int)

    return df


def prepare_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    df = ensure_aliases(df)
    df = add_explosive_features(df)
    df = add_tradability_flags(df)
    df = add_momentum_state(df)
    df = make_explosive_target(df)

    return df.replace([np.inf, -np.inf], np.nan)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(ID_COLUMNS) | set(LEAKAGE_COLUMNS)
    blocked |= {"open", "high", "low", "close"}
    blocked |= DROP_MACRO_DOMINANT_FEATURES
    blocked |= {"market_regime", "route"}

    feature_cols = []

    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    keep = [c for c in feature_cols if df[c].nunique(dropna=True) > 1]

    if not keep:
        raise ValueError("No valid feature columns found.")

    return keep


# =============================================================================
# Model
# =============================================================================

def build_model() -> Pipeline:
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.10,
        min_samples_leaf=35,
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    ensemble = VotingClassifier(
        estimators=[("hgb", hgb), ("rf", rf)],
        voting="soft",
        weights=[0.75, 0.25],
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ensemble),
        ]
    )


@dataclass
class ModelBundle:
    model: Pipeline
    feature_cols: list[str]
    train_positive_rate: float
    validation_auc: float | None


def train_walkforward_emulated_model(df: pd.DataFrame) -> ModelBundle:
    latest_ts = df["timestamp"].max()

    train_df = df[
        (df["timestamp"] < latest_ts)
        & (df["close"] >= MIN_PRICE)
        & (df["tradable_symbol"] == 1)
    ].copy()

    train_df = train_df.dropna(subset=[TARGET_LABEL]).copy()

    if train_df.empty:
        raise RuntimeError("No valid training rows after walk-forward-emulated filters.")

    feature_cols = get_feature_columns(train_df)

    # Small terminal validation split by date to emulate out-of-sample behavior.
    unique_dates = sorted(train_df["timestamp"].dropna().unique())
    if len(unique_dates) >= 80:
        split_idx = int(len(unique_dates) * 0.85)
        split_date = unique_dates[split_idx]
        fit_df = train_df[train_df["timestamp"] < split_date].copy()
        val_df = train_df[train_df["timestamp"] >= split_date].copy()
    else:
        fit_df = train_df
        val_df = pd.DataFrame()

    model = build_model()

    X_train = fit_df[feature_cols]
    y_train = fit_df[TARGET_LABEL].astype(int)

    print(f"[model] training rows={len(fit_df):,} features={len(feature_cols):,} positive_rate={y_train.mean():.2%}")
    model.fit(X_train, y_train)

    validation_auc = None

    if not val_df.empty and val_df[TARGET_LABEL].nunique() > 1:
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET_LABEL].astype(int)
        p_val = model.predict_proba(X_val)[:, 1]
        pred_val = (p_val >= SOFT_MIN_PROBABILITY).astype(int)

        validation_auc = float(roc_auc_score(y_val, p_val))

        print("\n===== WALK-FORWARD-EMULATED VALIDATION =====")
        print(f"val_rows={len(val_df):,}")
        print(f"auc={validation_auc:.4f}")
        print(f"precision={precision_score(y_val, pred_val, zero_division=0):.4f}")
        print(f"recall={recall_score(y_val, pred_val, zero_division=0):.4f}")
        print(classification_report(y_val, pred_val, zero_division=0))

    return ModelBundle(
        model=model,
        feature_cols=feature_cols,
        train_positive_rate=float(y_train.mean()),
        validation_auc=validation_auc,
    )


# =============================================================================
# Ranking / sizing
# =============================================================================

def expected_value_score(prob: float, upside_r: float = KELLY_WIN_R, downside_r: float = KELLY_LOSS_R) -> float:
    p = float(np.clip(prob, 0.0, 1.0))
    return p * upside_r - (1.0 - p) * downside_r


def kelly_weight(prob: float) -> float:
    ev = expected_value_score(prob)
    if ev <= 0:
        return 0.0
    full_kelly = ev / KELLY_WIN_R
    return float(np.clip(full_kelly * KELLY_FRACTION, 0.0, MAX_KELLY_WEIGHT))


def volatility_adjustment(atr_pct: float) -> float:
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        atr_pct = TARGET_POSITION_VOL
    return float(np.clip(TARGET_POSITION_VOL / atr_pct, VOL_ADJUST_MIN, VOL_ADJUST_MAX))


def convex_rank_weights(scores: pd.Series, power: float = CONVEX_POWER) -> pd.Series:
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0).clip(lower=0.0)
    if s.sum() <= 0:
        ranks = pd.Series(np.arange(len(s), 0, -1), index=s.index, dtype=float)
        s = ranks / ranks.max()
    powered = s ** power
    if powered.sum() <= 0:
        return pd.Series(1.0 / max(len(s), 1), index=s.index)
    return powered / powered.sum()


def score_latest_rows(latest: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    latest = latest.copy()

    proba = bundle.model.predict_proba(latest[bundle.feature_cols])[:, 1]
    latest["explosive_alpha_probability"] = proba

    prob = latest["explosive_alpha_probability"].fillna(0)
    ev = prob.map(expected_value_score)

    rel_vol = numeric_series(latest, "relative_volume_20", 1.0).clip(0, 3) / 3.0
    range_exp = numeric_series(latest, "range_expansion").clip(0, 0.08) / 0.08
    breakout = numeric_series(latest, "breakout_above_prev_high")
    trend = numeric_series(latest, "trend_alignment")

    latest["expected_value"] = ev

    latest["explosive_score"] = (
        0.32 * prob
        + 0.22 * numeric_series(latest, "explosion_setup_score")
        + 0.18 * breakout
        + 0.12 * range_exp
        + 0.10 * rel_vol
        + 0.08 * numeric_series(latest, "momentum_score")
        + 0.06 * trend
        + 0.10 * ev.clip(lower=0)
        - 0.20 * numeric_series(latest, "momentum_negative")
        - 0.10 * numeric_series(latest, "late_reversal_risk")
    )

    latest["confidence_score"] = (
        0.50 * prob
        + 0.20 * numeric_series(latest, "explosion_setup_score")
        + 0.15 * breakout
        + 0.10 * numeric_series(latest, "momentum_score")
        + 0.05 * ev.clip(lower=0)
    ).clip(0, 1)

    return latest


def rank_walkforward_emulated_candidates(
    scored: pd.DataFrame,
    *,
    top_n: int,
    min_probability: float,
    min_explosive_score: float,
) -> pd.DataFrame:
    ranked = scored.copy()

    ranked = ranked[
        (ranked["tradable_symbol"] == 1)
        & (ranked["close"] >= MIN_PRICE)
        & (ranked["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20)
        & (ranked["relative_volume_20"] >= MIN_RELATIVE_VOLUME)
        & (ranked["range_expansion"] >= MIN_RANGE_EXPANSION)
        & (ranked["momentum_negative"] == 0)
        & (ranked["late_reversal_risk"] == 0)
        & (
            (ranked["breakout_above_prev_high"] == 1)
            | (ranked["trend_alignment"] == 1)
            | (ranked["explosion_setup_score"] >= MIN_EXPLOSION_SETUP_SCORE)
        )
        & (
            (ranked["explosive_alpha_probability"] >= min_probability)
            | (ranked["breakout_above_prev_high"] == 1)
            | (ranked["expected_value"] > 0.10)
        )
        & (ranked["explosive_score"] >= min_explosive_score)
    ].copy()

    if ranked.empty:
        return ranked

    ranked = ranked.sort_values("explosive_score", ascending=False).head(top_n).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    # Compatibility columns for portfolio_optimizer.
    ranked["continuation_probability"] = ranked["explosive_alpha_probability"]
    ranked["confidence"] = ranked["confidence_score"]
    ranked["model_score"] = ranked["explosive_score"]
    ranked["expected_return_5d"] = ranked.get(FUTURE_RET_5D, np.nan)
    ranked["expected_sharpe_score"] = ranked["expected_value"]

    return ranked


def build_convex_target_weights(
    ranked: pd.DataFrame,
    *,
    equity: float,
    max_gross_exposure: float,
    max_position_weight: float,
    min_position_weight: float,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked

    out = ranked.copy()

    regime = str(out["market_regime"].iloc[0]) if "market_regime" in out.columns else "CHOP_NEUTRAL"
    regime_gross = REGIME_GROSS_EXPOSURE.get(regime, 0.45)
    regime_mult = REGIME_WEIGHT_MULTIPLIER.get(regime, 0.70)

    gross_cap = min(max_gross_exposure, regime_gross)

    raw_weights = []

    for _, row in out.iterrows():
        prob = float(row.get("explosive_alpha_probability", 0.0))
        atr_pct = float(row.get("atr_pct", TARGET_POSITION_VOL))

        kw = kelly_weight(prob)
        vol_adj = volatility_adjustment(atr_pct)
        confidence = float(row.get("confidence_score", prob))

        raw_weights.append(max(kw * vol_adj * regime_mult * (0.50 + confidence), 0.0))

    out["raw_weight"] = raw_weights

    convex = convex_rank_weights(out["explosive_score"], power=CONVEX_POWER)
    out["raw_weight"] = 0.50 * out["raw_weight"] + 0.50 * convex

    total_raw = out["raw_weight"].sum()

    if total_raw <= 0:
        return pd.DataFrame()

    out["target_weight"] = out["raw_weight"] / total_raw * gross_cap
    out["target_weight"] = out["target_weight"].clip(0, max_position_weight)
    out.loc[out["target_weight"] < min_position_weight, "target_weight"] = 0.0

    total_weight = out["target_weight"].sum()
    if total_weight > gross_cap:
        out["target_weight"] *= gross_cap / total_weight

    out["target_dollars"] = equity * out["target_weight"]
    out["sizing_method"] = "WALKFORWARD_EMULATED_EXPLOSIVE_CONVEX_KELLY_VOL"

    out = out[out["target_weight"] > 0].copy()

    return out.sort_values("target_weight", ascending=False).reset_index(drop=True)



# =============================================================================
# Optimizer compatibility
# =============================================================================

def add_optimizer_compatibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    portfolio_optimizer.py expects an older candidate schema.
    The walk-forward-emulated model creates newer names.

    This adapter preserves the explosive-alpha scores while adding the columns
    required by portfolio_optimizer._validate_candidates().
    """
    out = df.copy()

    if "price" not in out.columns:
        out["price"] = numeric_series(out, "close")

    if "trade_confidence" not in out.columns:
        out["trade_confidence"] = numeric_series(out, "confidence_score")

    if "continuation_prob" not in out.columns:
        out["continuation_prob"] = numeric_series(out, "explosive_alpha_probability")

    if "continuation_probability" not in out.columns:
        out["continuation_probability"] = out["continuation_prob"]

    if "volatility20" not in out.columns:
        if "volatility_20d" in out.columns:
            out["volatility20"] = numeric_series(out, "volatility_20d")
        elif "atr_pct" in out.columns:
            out["volatility20"] = numeric_series(out, "atr_pct")
        else:
            out["volatility20"] = 0.02

    if "confluence_reversal_risk" not in out.columns:
        out["confluence_reversal_risk"] = numeric_series(out, "late_reversal_risk")

    if "bear" not in out.columns:
        if "market_regime" in out.columns:
            out["bear"] = (
                out["market_regime"].astype(str).str.upper().str.contains("DEFENSIVE")
            ).astype(int)
        else:
            out["bear"] = 0

    # Helpful aliases used by some optimizer versions.
    if "confidence" not in out.columns:
        out["confidence"] = out["trade_confidence"]

    if "model_score" not in out.columns:
        out["model_score"] = numeric_series(out, "explosive_score")

    if "expected_sharpe_score" not in out.columns:
        out["expected_sharpe_score"] = numeric_series(out, "expected_value")

    # Clean required numeric columns.
    required_numeric = [
        "price",
        "trade_confidence",
        "continuation_prob",
        "continuation_probability",
        "volatility20",
        "confluence_reversal_risk",
        "bear",
        "confidence",
        "model_score",
        "expected_sharpe_score",
    ]

    for col in required_numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # Do not allow zero/negative price into optimizer.
    out["price"] = out["price"].where(out["price"] > 0, numeric_series(out, "close"))

    return out


# =============================================================================
# Summary
# =============================================================================

def write_summary(
    *,
    account_equity: float,
    account_cash: float,
    analysis_symbol_count: int,
    bar_symbol_count: int,
    feature_rows: int,
    train_rows: int,
    prediction_rows: int,
    ranked_rows: int,
    target_rows: int,
    execute: bool,
    validation_auc: float | None,
) -> None:
    text = f"""
===== DAILY WALK-FORWARD-EMULATED SYSTEM SUMMARY =====
timestamp_utc              : {datetime.now(timezone.utc).isoformat()}
execute_requested           : {execute}

account_equity              : ${account_equity:,.2f}
account_cash                : ${account_cash:,.2f}

analysis_symbols            : {analysis_symbol_count}
bar_symbols                 : {bar_symbol_count}
feature_rows                : {feature_rows:,}
training_rows               : {train_rows:,}
prediction_rows             : {prediction_rows:,}
ranked_candidates           : {ranked_rows}
portfolio_targets           : {target_rows}
validation_auc              : {validation_auc}

outputs_dir                 : {OUT_DIR}
symbols_file                : {SYMBOLS_PATH}
ranked_candidates_csv       : {RANKED_PATH}
portfolio_targets_csv       : {TARGETS_PATH}
rebalance_orders_csv        : {REBALANCE_ORDERS_PATH}
exit_orders_csv             : {EXIT_ORDERS_PATH}
""".strip()

    SUMMARY_PATH.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"[saved] {SUMMARY_PATH}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    print("===== DAILY WALK-FORWARD-EMULATED EXPLOSIVE ALPHA SYSTEM =====")

    paper = not args.live

    trading_config = ExecutorConfig(paper=paper)

    trading_client = get_trading_client(trading_config)
    account = get_account_snapshot(trading_client)

    optimizer_config = OptimizerConfig(
        starting_equity=account.equity,
        cash_buffer_pct=args.cash_buffer if args.cash_buffer is not None else OptimizerConfig().cash_buffer_pct,
        max_gross_exposure=args.max_gross_exposure if args.max_gross_exposure is not None else BASE_MAX_GROSS_EXPOSURE,
        max_position_weight=args.max_position_weight if args.max_position_weight is not None else MAX_WEIGHT_PER_POSITION,
        min_position_weight=args.min_position_weight if args.min_position_weight is not None else MIN_WEIGHT_PER_POSITION,
        max_positions=args.max_positions if args.max_positions is not None else MAX_OPEN_POSITIONS,
        min_trade_dollars=args.min_trade_dollars if args.min_trade_dollars is not None else OptimizerConfig().min_trade_dollars,
        dynamic_account_scaling=not args.disable_account_scaling,
    )
    optimizer_config = scale_optimizer_config_for_equity(account.equity, optimizer_config)

    trading_config.cash_buffer_pct = optimizer_config.cash_buffer_pct
    trading_config.min_trade_dollars = optimizer_config.min_trade_dollars
    trading_config.dynamic_account_scaling = not args.disable_account_scaling
    trading_config = scale_executor_config_for_account(account, trading_config)

    current_positions = get_positions_df(trading_client)

    print(f"[account] equity=${account.equity:,.2f} cash=${account.cash:,.2f}")
    print(
        "[sizing] "
        f"gross_cap={optimizer_config.max_gross_exposure:.2%} "
        f"max_position={optimizer_config.max_position_weight:.2%} "
        f"max_positions={optimizer_config.max_positions} "
        f"min_trade=${optimizer_config.min_trade_dollars:,.2f} "
        f"max_order=${trading_config.max_order_dollars:,.2f} "
        f"cash_buffer={trading_config.cash_buffer_pct:.2%}"
    )
    print(f"[positions] count={len(current_positions)}")

    raw_universe_symbols = get_alpaca_tradeable_symbols(
        trading_client=trading_client,
        max_symbols=args.max_symbols,
    )

    raw_symbol_pool = sorted(set(raw_universe_symbols) | set(REGIME_SYMBOLS))

    raw_pool_bars = download_daily_bars(
        symbols=raw_symbol_pool,
        lookback_days=args.lookback_days,
        chunk_size=args.chunk_size,
    )

    analysis_symbols = select_dynamic_universe_from_bars(
        raw=raw_pool_bars,
        max_symbols=UNIVERSE_LIMIT,
    )

    all_symbols = sorted(set(analysis_symbols) | set(REGIME_SYMBOLS))
    raw = raw_pool_bars[raw_pool_bars["symbol"].isin(all_symbols)].copy()

    save_frame(raw, RAW_BARS_PATH)

    print(f"[universe] analysis_symbols={len(analysis_symbols)}")
    print(f"[universe] total_symbols_with_regime={len(all_symbols)}")

    features = build_features(raw, regime_symbols=REGIME_SYMBOLS, add_regime=True)
    save_frame(features, FEATURES_PATH)

    model_frame = prepare_model_frame(features)
    latest_ts = model_frame["timestamp"].max()

    train_rows = model_frame[
        (model_frame["timestamp"] < latest_ts)
        & (model_frame["symbol"].isin(analysis_symbols))
        & (model_frame["tradable_symbol"] == 1)
    ].copy()

    save_frame(train_rows, TRAIN_EVENTS_PATH)

    if train_rows.empty:
        raise RuntimeError("No valid training rows after feature/model-frame filtering.")

    bundle = train_walkforward_emulated_model(model_frame)

    latest = model_frame[
        (model_frame["timestamp"] == latest_ts)
        & (model_frame["symbol"].isin(analysis_symbols))
    ].copy()

    if latest.empty:
        raise RuntimeError("No latest rows available for prediction.")

    predicted = score_latest_rows(latest, bundle)
    save_frame(predicted, PREDICTIONS_PATH)

    ranked = rank_walkforward_emulated_candidates(
        predicted,
        top_n=args.top_n,
        min_probability=args.min_probability,
        min_explosive_score=args.min_explosive_score,
    )

    save_frame(ranked, RANKED_PATH)

    if ranked.empty:
        print("[rank] no candidates passed walk-forward-emulated explosive filters.")
        SYMBOLS_PATH.write_text("", encoding="utf-8")

        empty = pd.DataFrame()
        save_frame(empty, TARGETS_PATH)
        save_frame(empty, REBALANCE_ORDERS_PATH)
        save_frame(empty, EXIT_ORDERS_PATH)

        write_summary(
            account_equity=account.equity,
            account_cash=account.cash,
            analysis_symbol_count=len(analysis_symbols),
            bar_symbol_count=raw["symbol"].nunique(),
            feature_rows=len(features),
            train_rows=len(train_rows),
            prediction_rows=len(predicted),
            ranked_rows=0,
            target_rows=0,
            execute=args.execute,
            validation_auc=bundle.validation_auc,
        )
        return

    targets = build_convex_target_weights(
        ranked,
        equity=account.equity,
        max_gross_exposure=optimizer_config.max_gross_exposure,
        max_position_weight=optimizer_config.max_position_weight,
        min_position_weight=optimizer_config.min_position_weight,
    )

    if targets.empty:
        print("[targets] no targets after convex/Kelly/vol sizing.")
        SYMBOLS_PATH.write_text("", encoding="utf-8")
        save_frame(targets, TARGETS_PATH)
        save_frame(pd.DataFrame(), REBALANCE_ORDERS_PATH)
        save_frame(pd.DataFrame(), EXIT_ORDERS_PATH)

        write_summary(
            account_equity=account.equity,
            account_cash=account.cash,
            analysis_symbol_count=len(analysis_symbols),
            bar_symbol_count=raw["symbol"].nunique(),
            feature_rows=len(features),
            train_rows=len(train_rows),
            prediction_rows=len(predicted),
            ranked_rows=len(ranked),
            target_rows=0,
            execute=args.execute,
            validation_auc=bundle.validation_auc,
        )
        return

    symbols_text = ", ".join(f'"{s}"' for s in targets["symbol"].astype(str).tolist())
    SYMBOLS_PATH.write_text(symbols_text, encoding="utf-8")

    print("\n===== WALK-FORWARD-EMULATED TRADE UNIVERSE =====")
    print(symbols_text)

    targets = add_optimizer_compatibility_columns(targets)
    save_frame(targets, TARGETS_PATH)

    # Use the existing optimizer/execution layer for order planning.
    # The optimizer still handles current broker positions, cash buffer, and trade sizing.

    optimizer_input = add_optimizer_compatibility_columns(targets)

    # Preserve optimizer compatibility while feeding walk-forward-emulated scores.
    optimizer_input["continuation_probability"] = optimizer_input["explosive_alpha_probability"]
    optimizer_input["continuation_prob"] = optimizer_input["explosive_alpha_probability"]
    optimizer_input["confidence"] = optimizer_input["confidence_score"]
    optimizer_input["trade_confidence"] = optimizer_input["confidence_score"]
    optimizer_input["model_score"] = optimizer_input["explosive_score"]
    optimizer_input["expected_sharpe_score"] = optimizer_input["expected_value"]

    opt_targets, rebalance_orders, exit_orders, portfolio_summary = optimize_portfolio(
        candidates=optimizer_input,
        equity=account.equity,
        current_positions=current_positions,
        config=optimizer_config,
    )

    if args.no_exit_orders:
        exit_orders = pd.DataFrame()
        print("[orders] exit orders disabled by --no-exit-orders")

    print_portfolio_summary(portfolio_summary)

    save_frame(opt_targets, TARGETS_PATH)
    save_frame(rebalance_orders, REBALANCE_ORDERS_PATH)
    save_frame(exit_orders, EXIT_ORDERS_PATH)

    order_plan, execution_results, _, _ = run_execution(
        rebalance_orders=rebalance_orders,
        exit_orders=exit_orders,
        config=trading_config,
    )

    save_execution_results(
        order_plan=order_plan,
        results=execution_results,
        output_dir=LOG_DIR,
    )

    write_summary(
        account_equity=account.equity,
        account_cash=account.cash,
        analysis_symbol_count=len(analysis_symbols),
        bar_symbol_count=raw["symbol"].nunique(),
        feature_rows=len(features),
        train_rows=len(train_rows),
        prediction_rows=len(predicted),
        ranked_rows=len(ranked),
        target_rows=len(opt_targets),
        execute=args.execute,
        validation_auc=bundle.validation_auc,
    )


if __name__ == "__main__":
    main()
