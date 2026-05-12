#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca_executor import (
    ExecutorConfig,
    get_account_snapshot,
    get_positions_df,
    get_trading_client,
    run_execution,
    save_execution_results,
    scale_executor_config_for_account,
)
from features import REGIME_SYMBOLS, build_features, normalize_symbol
from portfolio_optimizer import (
    OptimizerConfig,
    optimize_portfolio,
    print_portfolio_summary,
    scale_optimizer_config_for_equity,
)

try:
    from llm_sentiment_universe_selector import build_sentiment_universe
except Exception:
    build_sentiment_universe = None


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(f"Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in {ENV_PATH}")

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

LLM_SENTIMENT_UNIVERSE_JSON_PATH = OUT_DIR / "latest_llm_sentiment_universe.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / "locked_trading_universe.txt"

RANDOM_STATE = 42
LOOKBACK_DAYS = 520
CHUNK_SIZE = 20
TARGET_LLM_UNIVERSE_SIZE = 30

MIN_PRICE = 5.00
MAX_PRICE = 250.00
MIN_VOLUME = 500_000
MIN_DAILY_DOLLAR_VOLUME = 10_000_000
MIN_AVG_DOLLAR_VOLUME20 = 10_000_000
MIN_ATR_PCT = 0.004
MIN_DAILY_RANGE_PCT = 0.004
MIN_RELATIVE_VOLUME = 0.85
MIN_RANGE_EXPANSION = 0.006

FUTURE_RET_1D = "future_return_1d"
FUTURE_RET_3D = "future_return_3d"
FUTURE_RET_5D = "future_return_5d"
TARGET_LABEL = "explosive_alpha_label"

MIN_FUTURE_3D_RETURN = 0.010
MIN_FUTURE_5D_RETURN = 0.015
TOP_RETURN_QUANTILE = 0.80

SOFT_MIN_PROBABILITY = 0.38
MIN_EXPLOSIVE_SCORE = 0.45
MIN_EXPLOSION_SETUP_SCORE = 0.45

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
    "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_volatility_20d",
    "spy_close_vs_sma_20", "qqq_return_1d", "qqq_return_5d", "qqq_return_20d",
    "qqq_volatility_20d", "qqq_close_vs_sma_20", "vixy_return_1d",
    "vixy_return_5d", "vixy_return_20d", "vixy_close_vs_sma_20",
}

LEAKAGE_COLUMNS = {
    FUTURE_RET_1D, FUTURE_RET_3D, FUTURE_RET_5D,
    "future_close_3d", "future_close_5d", "next_close", "next_ret_1d",
    "forward_sharpe_proxy", "continuation_label", "next_bar_positive_label",
    "next_bar_tradeable_positive_label", "multi_bar_continuation_label",
    "explosive_continuation_label", TARGET_LABEL, "prediction", "model_prediction",
    "selected_threshold", "entry_threshold", "rank_for_day", "target_weight",
    "target_dollars", "recommended_shares",
}

ID_COLUMNS = {"timestamp", "symbol", "symbol_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--llm-symbols", type=int, default=TARGET_LLM_UNIVERSE_SIZE)
    parser.add_argument("--skip-llm-refresh", action="store_true")
    parser.add_argument("--min-probability", type=float, default=SOFT_MIN_PROBABILITY)
    parser.add_argument("--min-explosive-score", type=float, default=MIN_EXPLOSIVE_SCORE)
    parser.add_argument("--cash-buffer", type=float, default=None)
    parser.add_argument("--max-gross-exposure", type=float, default=None)
    parser.add_argument("--max-position-weight", type=float, default=None)
    parser.add_argument("--min-position-weight", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--min-trade-dollars", type=float, default=None)
    parser.add_argument("--disable-account-scaling", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-exit-orders", action="store_true")
    return parser.parse_args()


def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def numeric_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def clean_symbol(symbol: Any) -> str:
    sym = normalize_symbol(str(symbol or "").upper().strip())
    sym = sym.replace(".", "-")
    if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", sym):
        return ""
    if "-PR" in sym or "-WS" in sym or "-WT" in sym or "-U" in sym or "-RT" in sym:
        return ""
    return sym


def refresh_llm_universe_if_needed(skip: bool = False) -> None:
    if skip:
        print("[llm-universe] refresh skipped by flag")
        return

    if build_sentiment_universe is None:
        print("[llm-universe] selector unavailable; using existing JSON/locked universe")
        return

    try:
        print("===== YAHOO/LLM SENTIMENT UNIVERSE REFRESH =====")
        payload = build_sentiment_universe()
        print(f"[llm-universe] market_tone={payload.get('market_tone', 'unknown')}")
        print(f"[llm-universe] selected_symbols={len(payload.get('symbols', []))}")
    except Exception as exc:
        print(f"[llm-universe] refresh failed: {exc}")


def load_llm_sentiment_universe(max_symbols: int = TARGET_LLM_UNIVERSE_SIZE) -> list[str]:
    symbols: list[Any] = []

    if LLM_SENTIMENT_UNIVERSE_JSON_PATH.exists():
        data = json.loads(LLM_SENTIMENT_UNIVERSE_JSON_PATH.read_text(encoding="utf-8"))

        if isinstance(data.get("symbols"), list):
            symbols.extend(data["symbols"])

        if isinstance(data.get("selections"), list):
            for row in data["selections"]:
                if isinstance(row, dict):
                    symbols.append(row.get("symbol") or row.get("ticker"))

    elif LOCKED_UNIVERSE_PATH.exists():
        raw = LOCKED_UNIVERSE_PATH.read_text(encoding="utf-8", errors="ignore")
        symbols.extend(re.split(r"[\s,]+", raw))

    else:
        raise RuntimeError(
            "No LLM sentiment universe found. Run llm_sentiment_universe_selector.py first."
        )

    cleaned: list[str] = []
    seen: set[str] = set()

    for raw in symbols:
        sym = clean_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        cleaned.append(sym)
        if len(cleaned) >= max_symbols:
            break

    if not cleaned:
        raise RuntimeError("LLM sentiment universe exists but contains no valid symbols.")

    print(f"[universe] loaded Yahoo/LLM sentiment symbols={len(cleaned)}")
    print("[universe] " + ", ".join(cleaned))
    return cleaned


def download_daily_bars(symbols: list[str], lookback_days: int, chunk_size: int) -> pd.DataFrame:
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    symbols = sorted(set(clean_symbol(s) for s in symbols if clean_symbol(s)))
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    frames: list[pd.DataFrame] = []
    print(f"[download] {start.date()} -> {end.date()} chunks={len(chunks)} symbols={len(symbols)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"[download] chunk={i}/{len(chunks)} symbols={len(chunk)}")

        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index()

            if df.empty:
                continue

            needed = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            df = df[needed].copy()
            df["symbol"] = df["symbol"].astype(str).map(clean_symbol)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            frames.append(df.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close"]))

        except Exception as exc:
            print(f"[warning] failed chunk={chunk[:5]} error={exc}")

    if not frames:
        raise RuntimeError("No daily bars downloaded from Alpaca.")

    raw = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["symbol", "timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    print(f"[download] rows={len(raw):,} symbols={raw['symbol'].nunique():,}")
    return raw


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

    volume_sma20 = (
        volume.groupby(df["symbol"])
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["range_expansion"] = numeric_series(df, "daily_range_pct")
    df["volume_spike"] = volume / volume_sma20.replace(0, np.nan)
    df["volume_spike"] = df["volume_spike"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["trend_alignment"] = (
        (numeric_series(df, "sma5") > numeric_series(df, "sma20"))
        & (numeric_series(df, "sma20") > numeric_series(df, "sma50"))
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

    df["tradable_liquid"] = (
        (numeric_series(df, "close") >= MIN_PRICE)
        & (numeric_series(df, "volume") >= MIN_VOLUME)
        & (numeric_series(df, "dollar_volume") >= MIN_DAILY_DOLLAR_VOLUME)
        & (numeric_series(df, "avg_dollar_volume20") >= MIN_AVG_DOLLAR_VOLUME20)
    ).astype(int)

    df["tradable_movement"] = (
        (numeric_series(df, "atr_pct") >= MIN_ATR_PCT)
        & (numeric_series(df, "daily_range_pct") >= MIN_DAILY_RANGE_PCT)
    ).astype(int)

    df["tradable_symbol"] = (
        (df["tradable_liquid"] == 1)
        & (df["tradable_movement"] == 1)
    ).astype(int)

    return df


def add_momentum_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["momentum_favorable"] = (
        (numeric_series(df, "ret_1d") > -0.003)
        & (numeric_series(df, "ret_3d") > -0.010)
        & (numeric_series(df, "ret_5d") > -0.015)
        & (numeric_series(df, "dist_sma5") > -0.015)
        & (numeric_series(df, "dist_sma20") > -0.025)
        & (numeric_series(df, "macd_hist") > -0.004)
        & (numeric_series(df, "rsi", 50) >= 38)
        & (numeric_series(df, "rsi", 50) <= 86)
        & (numeric_series(df, "relative_volume_20", 1.0) >= 0.70)
        & (numeric_series(df, "late_reversal_risk") == 0)
    ).astype(int)

    df["momentum_negative"] = (
        (numeric_series(df, "ret_1d") < 0)
        & (
            (numeric_series(df, "ret_3d") < -0.025)
            | (numeric_series(df, "ret_5d") < -0.045)
            | (numeric_series(df, "dist_sma5") < -0.035)
            | (numeric_series(df, "dist_sma20") < -0.055)
            | (numeric_series(df, "high_conviction_score") < -0.75)
            | (numeric_series(df, "late_reversal_risk") == 1)
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
        & (numeric_series(df, "tradable_symbol") == 1)
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

    feature_cols = [
        col for col in df.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]

    keep = [c for c in feature_cols if df[c].nunique(dropna=True) > 1]

    if not keep:
        raise ValueError("No valid feature columns found.")

    return keep


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

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", VotingClassifier(
                estimators=[("hgb", hgb), ("rf", rf)],
                voting="soft",
                weights=[0.75, 0.25],
                n_jobs=-1,
            )),
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
    ].dropna(subset=[TARGET_LABEL]).copy()

    if train_df.empty:
        raise RuntimeError("No valid training rows after filters.")

    feature_cols = get_feature_columns(train_df)

    unique_dates = sorted(train_df["timestamp"].dropna().unique())
    if len(unique_dates) >= 80:
        split_date = unique_dates[int(len(unique_dates) * 0.85)]
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

        print("\n===== WALK-FORWARD VALIDATION =====")
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


def expected_value_score(prob: float) -> float:
    p = float(np.clip(prob, 0.0, 1.0))
    return p * KELLY_WIN_R - (1.0 - p) * KELLY_LOSS_R


def kelly_weight(prob: float) -> float:
    ev = expected_value_score(prob)
    if ev <= 0:
        return 0.0
    return float(np.clip((ev / KELLY_WIN_R) * KELLY_FRACTION, 0.0, MAX_KELLY_WEIGHT))


def volatility_adjustment(atr_pct: float) -> float:
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        atr_pct = TARGET_POSITION_VOL
    return float(np.clip(TARGET_POSITION_VOL / atr_pct, VOL_ADJUST_MIN, VOL_ADJUST_MAX))


def convex_rank_weights(scores: pd.Series) -> pd.Series:
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0).clip(lower=0.0)
    if s.sum() <= 0:
        return pd.Series(1.0 / max(len(s), 1), index=s.index)
    powered = s ** CONVEX_POWER
    return powered / powered.sum()


def score_latest_rows(latest: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    latest = latest.copy()
    latest["explosive_alpha_probability"] = bundle.model.predict_proba(latest[bundle.feature_cols])[:, 1]

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


def rank_candidates(scored: pd.DataFrame, top_n: int, min_probability: float, min_explosive_score: float) -> pd.DataFrame:
    ranked = scored[
        (scored["tradable_symbol"] == 1)
        & (scored["close"] >= MIN_PRICE)
        & (scored["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20)
        & (scored["relative_volume_20"] >= MIN_RELATIVE_VOLUME)
        & (scored["range_expansion"] >= MIN_RANGE_EXPANSION)
        & (scored["momentum_negative"] == 0)
        & (scored["late_reversal_risk"] == 0)
        & (
            (scored["breakout_above_prev_high"] == 1)
            | (scored["trend_alignment"] == 1)
            | (scored["explosion_setup_score"] >= MIN_EXPLOSION_SETUP_SCORE)
        )
        & (
            (scored["explosive_alpha_probability"] >= min_probability)
            | (scored["breakout_above_prev_high"] == 1)
            | (scored["expected_value"] > 0.10)
        )
        & (scored["explosive_score"] >= min_explosive_score)
    ].copy()

    if ranked.empty:
        return ranked

    ranked = ranked.sort_values("explosive_score", ascending=False).head(top_n).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["continuation_probability"] = ranked["explosive_alpha_probability"]
    ranked["confidence"] = ranked["confidence_score"]
    ranked["model_score"] = ranked["explosive_score"]
    ranked["expected_return_5d"] = ranked.get(FUTURE_RET_5D, np.nan)
    ranked["expected_sharpe_score"] = ranked["expected_value"]

    return ranked


def build_convex_target_weights(
    ranked: pd.DataFrame,
    equity: float,
    max_gross_exposure: float,
    max_position_weight: float,
    min_position_weight: float,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked

    out = ranked.copy()

    regime = str(out["market_regime"].iloc[0]) if "market_regime" in out.columns else "CHOP_NEUTRAL"
    gross_cap = min(max_gross_exposure, REGIME_GROSS_EXPOSURE.get(regime, 0.45))
    regime_mult = REGIME_WEIGHT_MULTIPLIER.get(regime, 0.70)

    raw_weights = []

    for _, row in out.iterrows():
        prob = float(row.get("explosive_alpha_probability", 0.0))
        atr_pct = float(row.get("atr_pct", TARGET_POSITION_VOL))
        confidence = float(row.get("confidence_score", prob))

        raw_weights.append(
            max(kelly_weight(prob) * volatility_adjustment(atr_pct) * regime_mult * (0.50 + confidence), 0.0)
        )

    out["raw_weight"] = 0.50 * pd.Series(raw_weights, index=out.index) + 0.50 * convex_rank_weights(out["explosive_score"])

    if out["raw_weight"].sum() <= 0:
        return pd.DataFrame()

    out["target_weight"] = out["raw_weight"] / out["raw_weight"].sum() * gross_cap
    out["target_weight"] = out["target_weight"].clip(0, max_position_weight)
    out.loc[out["target_weight"] < min_position_weight, "target_weight"] = 0.0

    if out["target_weight"].sum() > gross_cap:
        out["target_weight"] *= gross_cap / out["target_weight"].sum()

    out["target_dollars"] = equity * out["target_weight"]
    out["sizing_method"] = "YAHOO_LLM_WALKFORWARD_CONVEX_KELLY_VOL"

    return out[out["target_weight"] > 0].sort_values("target_weight", ascending=False).reset_index(drop=True)


def add_optimizer_compatibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["price"] = numeric_series(out, "close")
    out["trade_confidence"] = numeric_series(out, "confidence_score")
    out["continuation_prob"] = numeric_series(out, "explosive_alpha_probability")
    out["continuation_probability"] = out["continuation_prob"]
    out["volatility20"] = numeric_series(out, "volatility_20d", numeric_series(out, "atr_pct", 0.02))
    out["confluence_reversal_risk"] = numeric_series(out, "late_reversal_risk")
    out["bear"] = out.get("market_regime", pd.Series("", index=out.index)).astype(str).str.upper().str.contains("DEFENSIVE").astype(int)
    out["confidence"] = out["trade_confidence"]
    out["model_score"] = numeric_series(out, "explosive_score")
    out["expected_sharpe_score"] = numeric_series(out, "expected_value")

    return out


def safe_optimizer_config(args: argparse.Namespace, account_equity: float) -> OptimizerConfig:
    base = OptimizerConfig()

    cfg = OptimizerConfig(
        starting_equity=account_equity,
        cash_buffer_pct=float(args.cash_buffer if args.cash_buffer is not None else (base.cash_buffer_pct or 0.08)),
        max_gross_exposure=float(args.max_gross_exposure if args.max_gross_exposure is not None else BASE_MAX_GROSS_EXPOSURE),
        max_position_weight=float(args.max_position_weight if args.max_position_weight is not None else MAX_WEIGHT_PER_POSITION),
        min_position_weight=float(args.min_position_weight if args.min_position_weight is not None else MIN_WEIGHT_PER_POSITION),
        max_positions=int(args.max_positions if args.max_positions is not None else MAX_OPEN_POSITIONS),
        min_trade_dollars=float(args.min_trade_dollars if args.min_trade_dollars is not None else (base.min_trade_dollars or 100.0)),
        dynamic_account_scaling=not args.disable_account_scaling,
    )

    cfg.cash_buffer_pct = float(cfg.cash_buffer_pct or 0.08)
    cfg.max_gross_exposure = float(cfg.max_gross_exposure or BASE_MAX_GROSS_EXPOSURE)
    cfg.max_position_weight = float(cfg.max_position_weight or MAX_WEIGHT_PER_POSITION)
    cfg.min_position_weight = float(cfg.min_position_weight or MIN_WEIGHT_PER_POSITION)
    cfg.min_trade_dollars = float(cfg.min_trade_dollars or 100.0)

    return scale_optimizer_config_for_equity(account_equity, cfg)


def write_summary(**kwargs: Any) -> None:
    lines = ["===== DAILY YAHOO/LLM WALK-FORWARD SYSTEM SUMMARY ====="]
    lines.append(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")

    for key, value in kwargs.items():
        lines.append(f"{key}: {value}")

    text = "\n".join(lines)
    SUMMARY_PATH.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"[saved] {SUMMARY_PATH}")


def main() -> None:
    args = parse_args()

    print("===== DAILY YAHOO/LLM WALK-FORWARD EXPLOSIVE ALPHA SYSTEM =====")

    refresh_llm_universe_if_needed(skip=args.skip_llm_refresh)

    analysis_symbols = load_llm_sentiment_universe(max_symbols=args.llm_symbols)
    all_symbols = sorted(set(analysis_symbols) | set(REGIME_SYMBOLS))

    paper = not args.live
    trading_config = ExecutorConfig(paper=paper)

    trading_client = get_trading_client(trading_config)
    account = get_account_snapshot(trading_client)
    current_positions = get_positions_df(trading_client)

    optimizer_config = safe_optimizer_config(args, account.equity)

    trading_config.cash_buffer_pct = optimizer_config.cash_buffer_pct
    trading_config.min_trade_dollars = optimizer_config.min_trade_dollars
    trading_config.dynamic_account_scaling = not args.disable_account_scaling
    trading_config = scale_executor_config_for_account(account, trading_config)

    print(f"[account] equity=${account.equity:,.2f} cash=${account.cash:,.2f}")
    print(f"[positions] count={len(current_positions)}")
    print(f"[universe] analysis_symbols={len(analysis_symbols)} source=yahoo_llm_sentiment")
    print(f"[universe] total_symbols_with_regime={len(all_symbols)}")

    raw = download_daily_bars(
        symbols=all_symbols,
        lookback_days=args.lookback_days,
        chunk_size=args.chunk_size,
    )
    save_frame(raw, RAW_BARS_PATH)

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
        raise RuntimeError("No valid training rows after Yahoo/LLM universe filtering.")

    bundle = train_walkforward_emulated_model(model_frame)

    latest = model_frame[
        (model_frame["timestamp"] == latest_ts)
        & (model_frame["symbol"].isin(analysis_symbols))
    ].copy()

    if latest.empty:
        raise RuntimeError("No latest rows available for prediction.")

    predicted = score_latest_rows(latest, bundle)
    save_frame(predicted, PREDICTIONS_PATH)

    ranked = rank_candidates(
        predicted,
        top_n=args.top_n,
        min_probability=args.min_probability,
        min_explosive_score=args.min_explosive_score,
    )
    save_frame(ranked, RANKED_PATH)

    if ranked.empty:
        print("[rank] no candidates passed filters.")
        SYMBOLS_PATH.write_text("", encoding="utf-8")
        save_frame(pd.DataFrame(), TARGETS_PATH)
        save_frame(pd.DataFrame(), REBALANCE_ORDERS_PATH)
        save_frame(pd.DataFrame(), EXIT_ORDERS_PATH)
        write_summary(
            execute=args.execute,
            analysis_symbols=len(analysis_symbols),
            bar_symbols=raw["symbol"].nunique(),
            feature_rows=len(features),
            train_rows=len(train_rows),
            prediction_rows=len(predicted),
            ranked_rows=0,
            target_rows=0,
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
        print("[targets] no targets after sizing.")
        SYMBOLS_PATH.write_text("", encoding="utf-8")
        save_frame(targets, TARGETS_PATH)
        save_frame(pd.DataFrame(), REBALANCE_ORDERS_PATH)
        save_frame(pd.DataFrame(), EXIT_ORDERS_PATH)
        return

    symbols_text = ", ".join(f'"{s}"' for s in targets["symbol"].astype(str).tolist())
    SYMBOLS_PATH.write_text(symbols_text, encoding="utf-8")

    print("\n===== YAHOO/LLM WALK-FORWARD TRADE UNIVERSE =====")
    print(symbols_text)

    targets = add_optimizer_compatibility_columns(targets)
    save_frame(targets, TARGETS_PATH)

    optimizer_input = add_optimizer_compatibility_columns(targets)

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

    if args.execute:
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
    else:
        print("[dry-run] execution skipped. Use --execute to submit orders.")

    write_summary(
        execute=args.execute,
        account_equity=f"${account.equity:,.2f}",
        account_cash=f"${account.cash:,.2f}",
        analysis_symbols=len(analysis_symbols),
        bar_symbols=raw["symbol"].nunique(),
        feature_rows=len(features),
        train_rows=len(train_rows),
        prediction_rows=len(predicted),
        ranked_rows=len(ranked),
        target_rows=len(opt_targets),
        validation_auc=bundle.validation_auc,
    )


if __name__ == "__main__":
    main()