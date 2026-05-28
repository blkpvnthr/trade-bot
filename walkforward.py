#!/usr/bin/env python3
"""
walkforward_explosive_probabilistic_risk_sim.py

Explosive Alpha walk-forward simulator.

Designed for max-return / convex upside capture, not balanced accuracy.

Core philosophy:
    - Do NOT force trades every day.
    - Only trade top explosive setups.
    - Prefer breakout + expansion + volume participation.
    - Concentrate into 1-2 best names.
    - Size risk before entry, not after entry.
    - Avoid breakout-only entries unless model probability confirms the edge.
    - Use model probability, expected value, edge decay, and time decay exits.
    - Use drawdown-based exposure scaling and symbol cooldowns after losses.

This script is analysis-only:
    - no Alpaca orders
    - no broker mutation
    - no live execution

Expected input:
    daily_system_outputs/daily_features.csv

Outputs:
    daily_system_outputs/wf_explosive_predictions.csv
    daily_system_outputs/wf_explosive_window_metrics.csv
    daily_system_outputs/wf_explosive_daily_equity.csv
    daily_system_outputs/wf_explosive_trade_log.csv
    daily_system_outputs/wf_explosive_events.csv
    daily_system_outputs/wf_explosive_equity_curve.png
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score, recall_score, roc_auc_score


# =============================================================================
# Paths
# =============================================================================

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

_DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", __import__("datetime").datetime.now().strftime("%Y%m%d")))


def _resolve_daily_features_path(date_tag: str = _DATA_TAG) -> Path:
    env_override = os.getenv("DAILY_FEATURES_PATH", "")
    if env_override and Path(env_override).exists():
        return Path(env_override)
    new_path = PROJECT_ROOT / "system-data" / date_tag / f"{date_tag}_daily_features.csv"
    if new_path.exists():
        return new_path
    old_dated = PROJECT_ROOT / "daily_system_outputs" / f"{date_tag}_daily_features.csv"
    if old_dated.exists():
        return old_dated
    legacy = PROJECT_ROOT / "daily_system_outputs" / "daily_features.csv"
    if legacy.exists():
        return legacy
    return new_path  # canonical; let caller handle missing


INPUT_PATH = _resolve_daily_features_path()
print(f"[features-path] walkforward resolved={INPUT_PATH} exists={INPUT_PATH.exists()}")
OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREDICTIONS = OUT_DIR / "wf_probabilistic_risk_predictions.csv"
OUT_WINDOW_METRICS = OUT_DIR / "wf_probabilistic_risk_window_metrics.csv"
OUT_DAILY_EQUITY = OUT_DIR / "wf_probabilistic_risk_daily_equity.csv"
OUT_TRADE_LOG = OUT_DIR / "wf_probabilistic_risk_trade_log.csv"
OUT_EVENTS = OUT_DIR / "wf_probabilistic_risk_events.csv"
OUT_EQUITY_PLOT = OUT_DIR / "wf_probabilistic_risk_equity_curve.png"
OUT_MODEL_DIAGNOSTICS = OUT_DIR / "wf_probabilistic_risk_model_diagnostics.csv"


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


STARTING_EQUITY = env_float("STARTING_EQUITY", env_float("TRADING_ACCOUNT_EQUITY", 100_000.00))

# Sequence model configuration.
# The Transformer sees rolling per-symbol windows instead of one isolated row.
SEQUENCE_LENGTH = 20
TRANSFORMER_D_MODEL = 96
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.15
TRANSFORMER_BATCH_SIZE = 256
TRANSFORMER_EPOCHS = 35
TRANSFORMER_LR = 8e-4
TRANSFORMER_WEIGHT_DECAY = 1e-4
TRANSFORMER_PATIENCE = 6
TRANSFORMER_VAL_FRACTION = 0.18
TRANSFORMER_MIN_SEQUENCES = 250

TRAIN_DAYS = 252
TEST_DAYS = 42
STEP_DAYS = 42
MIN_TRAIN_ROWS = 1_000
MIN_TRAIN_POSITIVES = 25

FUTURE_RET_1D = "future_return_1d"
FUTURE_RET_3D = "future_return_3d"
FUTURE_RET_5D = "future_return_5d"
TARGET_LABEL = "wf_explosive_alpha_label"

# Explosive target.
# This trains the model to care about larger forward moves, not tiny positive bars.
MIN_FUTURE_3D_RETURN = 0.005
MIN_FUTURE_5D_RETURN = 0.011
TOP_RETURN_QUANTILE = 0.80

# Tradability.
MIN_PRICE = 10.00
MIN_VOLUME = 500_000
MIN_DOLLAR_VOLUME = 3_000_000
MIN_AVG_DOLLAR_VOLUME20 = 3_000_000
MIN_ATR_PCT = 0.005
MIN_DAILY_RANGE_PCT = 0.005

CORE_SYMBOLS = {"SPY", "QQQ", "DIA", "IWM", "VIXY"}

# Explosive entry constraints.
SOFT_MIN_PROBABILITY = 0.50
MIN_EXPLOSIVE_SCORE = 0.48
MIN_EXPLOSION_SETUP_SCORE = 0.45
MIN_RELATIVE_VOLUME = 0.55
MIN_RANGE_EXPANSION = 0.006

# Portfolio concentration.
MAX_OPEN_POSITIONS = 5
MAX_NEW_ENTRIES_PER_DAY = 5
TARGET_NEW_ENTRIES_PER_DAY = 3
BASE_MAX_GROSS_EXPOSURE = 0.95
MAX_WEIGHT_PER_POSITION = 0.30
MIN_WEIGHT_PER_POSITION = 0.03

# Pyramiding.
ALLOW_SCALE_IN = True
MAX_SCALE_INS_PER_POSITION = 2
SCALE_IN_FRACTION_OF_TARGET = 0.30
MIN_SCALE_IN_PROFIT = 0.010
SCALE_IN_PROBABILITY = 0.62
SCALE_IN_MIN_HOLD_BARS = 1

# Regime exposure.
REGIME_GROSS_EXPOSURE = {
    "TREND_RISK_ON": 0.95,
    "VOLATILE_RISK_ON": 0.65,
    "CHOP_NEUTRAL": 0.35,
    "DEFENSIVE_RISK_OFF": 0.05,
}

REGIME_MAX_NEW_ENTRIES = {
    "TREND_RISK_ON": 3,
    "VOLATILE_RISK_ON": 2,
    "CHOP_NEUTRAL": 1,
    "DEFENSIVE_RISK_OFF": 0,
}

REGIME_WEIGHT_MULTIPLIER = {
    "TREND_RISK_ON": 1.35,
    "VOLATILE_RISK_ON": 1.10,
    "CHOP_NEUTRAL": 0.70,
    "DEFENSIVE_RISK_OFF": 0.40,
}

# Capital preservation.
# Looser than safety versions, but still prevents death spirals.
EQUITY_RECOVERY = 0.97
EQUITY_DEFENSIVE = 0.94
EQUITY_KILL = 0.90

CAPITAL_PRESERVATION_GROSS_MULT = {
    "NORMAL": 1.00,
    "RECOVERY": 0.50,
    "DEFENSIVE": 0.25,
    "KILL": 0.00,
}

# Kelly/convex sizing.
KELLY_WIN_R = 3.0
KELLY_LOSS_R = 1.0
KELLY_FRACTION = 0.55
MAX_KELLY_WEIGHT = 0.30
CONVEX_POWER = 5.0

TARGET_POSITION_VOL = 0.025
VOL_ADJUST_MIN = 0.25
VOL_ADJUST_MAX = 1.5

# Exits.
# No weak exits. Winners are given time.
MODEL_FAILURE_PROB = 0.30
EDGE_DECAY_FRACTION = 0.45
MAX_HOLD_BARS = 35

# The old ATR/trailing-stop exit layer is intentionally removed.
# Risk is now controlled by pre-entry sizing, dynamic exposure, model failure,
# edge decay, and time decay exits.

# Selective scale-outs. These are deliberately delayed so they do not kill convexity.
# Compounding mode: harvest partial gains earlier so multiple 5-20% wins can compound.
# Scale-outs happen when upside appears and model edge starts fading.
SCALE_OUT_1_PROFIT = 0.08
SCALE_OUT_1_FRACTION = 0.35
SCALE_OUT_2_PROFIT = 0.16
SCALE_OUT_2_FRACTION = 0.40
SCALE_OUT_PROB_DROP = 0.52

# Probabilistic risk layer.
MIN_EXPECTED_VALUE_FOR_ENTRY = 0.04

# Compounding target framework.
# Do not wait for rare 100% winners; the goal is to stack repeatable 5-20% trades.
TARGET_TRADE_RETURN = 0.20
MIN_MOONSHOT_SCORE = 0.45
EARLY_FAILURE_BARS = 5
EARLY_FAILURE_MIN_PROFIT = -0.005
PROFIT_LOCK_AFTER = 0.12
PROFIT_LOCK_FLOOR = 0.04

# Moderate holding window for compounding trades.
TIME_DECAY_BARS = 10
MIN_TIME_DECAY_PROFIT = 0.005
LOSS_COOLDOWN_BARS = 5
MAX_QQQ_CORRELATED_NAMES = 2
HIGH_QQQ_CORR_THRESHOLD = 0.70
PORTFOLIO_VOL_CAP = 0.040

DYNAMIC_PROBABILITY_THRESHOLDS = {
    "TREND_RISK_ON": 0.50,
    "VOLATILE_RISK_ON": 0.55,
    "CHOP_NEUTRAL": 0.62,
    "DEFENSIVE_RISK_OFF": 0.75,
}

CAPITAL_STATE_THRESHOLD_ADDON = {
    "NORMAL": 0.00,
    "RECOVERY": 0.03,
    "DEFENSIVE": 0.07,
    "KILL": 1.00,
}


# Slippage.
ENTRY_SLIPPAGE_BPS = 5
EXIT_SLIPPAGE_BPS = 5

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
    "future_return_rank_5d",
    "future_return_rank_3d",
    "future_return_score",
    "future_continuation_quality",
    "future_explosive_quality",
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


@dataclass
class Position:
    symbol: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    shares: int
    weight_at_entry: float
    target_weight: float
    entry_probability: float
    peak_probability: float
    scale_ins: int = 0
    scale_outs: int = 0
    bars_held: int = 0
    max_favorable_price: float = 0.0
    min_adverse_price: float = 0.0
    recent_outcomes: list[int] = field(default_factory=list)


# =============================================================================
# Feature helpers
# =============================================================================

def numeric_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature generation first.")

    df = pd.read_csv(path)

    required = {"timestamp", "symbol", FUTURE_RET_1D}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


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
        elif "volume_ratio20" in df.columns:
            df["relative_volume_20"] = df["volume_ratio20"]
        else:
            df["relative_volume_20"] = 1.0

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


def add_stock_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    close = numeric_series(df, "close")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    volume = numeric_series(df, "volume")
    volume_sma20 = numeric_series(df, "volume_sma_20", np.nan)

    if volume_sma20.isna().all():
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
        & (dollar_volume >= MIN_DOLLAR_VOLUME)
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

    df["core_symbol"] = df["symbol"].isin(CORE_SYMBOLS).astype(int)
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"timestamp", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_market_regime: missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"])

    ts = df[["timestamp"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)

    def proxy(symbol: str, source_col: str, out_col: str) -> pd.DataFrame:
        if source_col not in df.columns:
            return pd.DataFrame({"timestamp": ts["timestamp"], out_col: 0.0})

        tmp = (
            df.loc[df["symbol"].eq(symbol), ["timestamp", source_col]]
            .drop_duplicates("timestamp", keep="last")
            .rename(columns={source_col: out_col})
            .copy()
        )

        if tmp.empty:
            return pd.DataFrame({"timestamp": ts["timestamp"], out_col: 0.0})

        tmp[out_col] = pd.to_numeric(tmp[out_col], errors="coerce")
        return ts.merge(tmp, on="timestamp", how="left")

    regime = ts.copy()

    parts = [
        proxy("QQQ", "ret_5d", "regime_qqq_ret_5d"),
        proxy("SPY", "ret_5d", "regime_spy_ret_5d"),
        proxy("IWM", "ret_5d", "regime_iwm_ret_5d"),
        proxy("VIXY", "ret_5d", "regime_vixy_ret_5d"),
        proxy("QQQ", "volatility_20d", "regime_qqq_volatility_20d"),
        proxy("SPY", "dist_sma20", "regime_spy_dist_sma20"),
        proxy("QQQ", "dist_sma20", "regime_qqq_dist_sma20"),
    ]

    for part in parts:
        regime = regime.merge(part, on="timestamp", how="left")

    for col in regime.columns:
        if col != "timestamp":
            regime[col] = pd.to_numeric(regime[col], errors="coerce").fillna(0.0)

    regime["risk_on_score"] = (
        0.25 * (regime["regime_qqq_ret_5d"] > 0).astype(int)
        + 0.25 * (regime["regime_spy_ret_5d"] > 0).astype(int)
        + 0.15 * (regime["regime_iwm_ret_5d"] > 0).astype(int)
        + 0.15 * (regime["regime_vixy_ret_5d"] < 0).astype(int)
        + 0.10 * (regime["regime_qqq_dist_sma20"] > 0).astype(int)
        + 0.10 * (regime["regime_spy_dist_sma20"] > 0).astype(int)
    ).clip(0.0, 1.0)

    rolling_q = regime["regime_qqq_volatility_20d"].rolling(
        window=120,
        min_periods=20,
    ).quantile(0.75)

    regime["market_volatility_high"] = (
        regime["regime_qqq_volatility_20d"] > rolling_q
    ).fillna(False).astype(int)

    conditions = [
        (regime["risk_on_score"] >= 0.70) & (regime["market_volatility_high"] == 0),
        (regime["risk_on_score"] >= 0.55) & (regime["market_volatility_high"] == 1),
        (regime["risk_on_score"] <= 0.35) | (regime["regime_vixy_ret_5d"] > 0.03),
    ]

    choices = [
        "TREND_RISK_ON",
        "VOLATILE_RISK_ON",
        "DEFENSIVE_RISK_OFF",
    ]

    regime["market_regime"] = np.select(
        conditions,
        choices,
        default="CHOP_NEUTRAL",
    )

    regime_cols = [
        "risk_on_score",
        "market_volatility_high",
        "market_regime",
        "regime_qqq_ret_5d",
        "regime_spy_ret_5d",
        "regime_iwm_ret_5d",
        "regime_vixy_ret_5d",
        "regime_qqq_volatility_20d",
        "regime_spy_dist_sma20",
        "regime_qqq_dist_sma20",
    ]

    out = df.drop(columns=[c for c in regime_cols if c in df.columns], errors="ignore")

    out = out.merge(
        regime[["timestamp"] + regime_cols],
        on="timestamp",
        how="left",
    )

    out["market_regime"] = out["market_regime"].fillna("CHOP_NEUTRAL")
    out["risk_on_score"] = pd.to_numeric(out["risk_on_score"], errors="coerce").fillna(0.5)
    out["market_volatility_high"] = pd.to_numeric(
        out["market_volatility_high"],
        errors="coerce",
    ).fillna(0).astype(int)

    for col in [
        "regime_qqq_ret_5d",
        "regime_spy_ret_5d",
        "regime_iwm_ret_5d",
        "regime_vixy_ret_5d",
        "regime_qqq_volatility_20d",
        "regime_spy_dist_sma20",
        "regime_qqq_dist_sma20",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

def add_momentum_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds momentum state flags used by the walk-forward simulator.

    Required downstream columns:
        - momentum_favorable
        - momentum_negative
    """
    df = df.copy()

    ret_1d = numeric_series(df, "ret_1d")
    ret_3d = numeric_series(df, "ret_3d")
    ret_5d = numeric_series(df, "ret_5d")
    dist_sma5 = numeric_series(df, "dist_sma5")
    dist_sma20 = numeric_series(df, "dist_sma20")
    macd_hist = numeric_series(df, "macd_hist")
    rsi = numeric_series(df, "rsi", 50.0)
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
        & (rsi >= 38)
        & (rsi <= 86)
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

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a higher-quality supervised target for the sequence model.

    The old target mainly asked, "did the future return clear a threshold?"
    This version teaches the model the *type* of move we want:
      - positive 1D follow-through / no immediate failure,
      - strong 3D or 5D continuation,
      - cross-sectional leadership on that date,
      - tradable movement and liquidity.

    All columns beginning with future_* are blocked from features via
    LEAKAGE_COLUMNS, so these are labels/diagnostics only.
    """
    df = df.copy()

    if FUTURE_RET_3D not in df.columns:
        df[FUTURE_RET_3D] = numeric_series(df, FUTURE_RET_1D)
    if FUTURE_RET_5D not in df.columns:
        df[FUTURE_RET_5D] = numeric_series(df, FUTURE_RET_1D)

    future1 = numeric_series(df, FUTURE_RET_1D)
    future3 = numeric_series(df, FUTURE_RET_3D)
    future5 = numeric_series(df, FUTURE_RET_5D)

    # Cross-sectional leadership: a setup is more useful if it beats other choices that day.
    df["future_return_rank_3d"] = df.groupby("timestamp")[FUTURE_RET_3D].rank(pct=True)
    df["future_return_rank_5d"] = df.groupby("timestamp")[FUTURE_RET_5D].rank(pct=True)

    # Smooth quality score for diagnostics and optional later regression/ranking work.
    df["future_return_score"] = (
        0.25 * future1.clip(-0.05, 0.08) / 0.08
        + 0.35 * future3.clip(-0.08, 0.15) / 0.15
        + 0.40 * future5.clip(-0.10, 0.25) / 0.25
    ).clip(-1.0, 1.0)

    no_immediate_failure = future1 > -0.020
    follows_through = (future3 > 0.015) | (future5 > 0.030)
    explosive_move = (future3 >= MIN_FUTURE_3D_RETURN) | (future5 >= MIN_FUTURE_5D_RETURN)
    cross_sectional_leader = (df["future_return_rank_3d"] >= TOP_RETURN_QUANTILE) | (df["future_return_rank_5d"] >= TOP_RETURN_QUANTILE)

    # Diagnostic labels, not features.
    df["future_continuation_quality"] = (no_immediate_failure & follows_through).astype(int)
    df["future_explosive_quality"] = (no_immediate_failure & explosive_move & cross_sectional_leader).astype(int)

    df[TARGET_LABEL] = (
        no_immediate_failure
        & (explosive_move | cross_sectional_leader | follows_through)
        & (df["tradable_symbol"] == 1)
        & (numeric_series(df, "range_expansion") >= MIN_DAILY_RANGE_PCT)
    ).astype(int)

    return df

def add_market_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling market-correlation features used by the risk layer.

    This is intentionally causal: each row only uses returns through the current row.
    The simulator uses this mostly as a concentration brake so it does not stack
    multiple highly QQQ-correlated names at the same time.
    """
    df = df.copy().sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if "ret_1d" not in df.columns:
        df["symbol_corr_20_to_qqq"] = 0.0
        df["symbol_corr_20_to_spy"] = 0.0
        return df

    pivot = (
        df.pivot_table(index="timestamp", columns="symbol", values="ret_1d", aggfunc="last")
        .sort_index()
    )

    for proxy_symbol, out_col in [("QQQ", "symbol_corr_20_to_qqq"), ("SPY", "symbol_corr_20_to_spy")]:
        if proxy_symbol not in pivot.columns:
            corr_long = pd.DataFrame({"timestamp": df["timestamp"], "symbol": df["symbol"], out_col: 0.0})
        else:
            corr = pivot.rolling(20, min_periods=8).corr(pivot[proxy_symbol])
            corr_long = (
                corr.stack()
                .rename(out_col)
                .reset_index()
                .rename(columns={"level_1": "symbol"})
            )
            corr_long["symbol"] = corr_long["symbol"].astype(str).str.upper().str.strip()

        df = df.merge(corr_long[["timestamp", "symbol", out_col]], on=["timestamp", "symbol"], how="left")
        df[out_col] = pd.to_numeric(df[out_col], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

    return df


def add_price_indicator_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly encode price/indicator relationships before the Transformer sees them.

    The sequence model can learn interactions on its own, but these engineered
    relationships make the learning problem easier on small walk-forward windows.
    Everything here is causal and uses only current/past information.
    """
    df = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    close = numeric_series(df, "close")
    open_ = numeric_series(df, "open")
    high = numeric_series(df, "high")
    low = numeric_series(df, "low")
    volume = numeric_series(df, "volume")

    rsi = numeric_series(df, "rsi", 50.0)
    macd_hist = numeric_series(df, "macd_hist")
    rel_vol = numeric_series(df, "relative_volume_20", 1.0)
    atr_pct = numeric_series(df, "atr_pct")
    daily_range_pct = numeric_series(df, "daily_range_pct")

    dist_sma5 = numeric_series(df, "dist_sma5")
    dist_sma20 = numeric_series(df, "dist_sma20")
    dist_sma50 = numeric_series(df, "dist_sma50")
    dist_sma200 = numeric_series(df, "dist_sma200")

    trend_score = numeric_series(df, "trend_score")
    momentum_score = numeric_series(df, "momentum_score")
    breakout_score = numeric_series(df, "breakout_score")

    safe_close = close.replace(0, np.nan)

    # Price structure relative to its own candle.
    df["close_to_open_pct"] = (close - open_) / safe_close
    df["high_low_range_pct"] = (high - low) / safe_close
    df["close_to_high_pct"] = (close - high) / safe_close
    df["close_to_low_pct"] = (close - low) / safe_close
    df["candle_body_to_range"] = ((close - open_).abs() / (high - low).replace(0, np.nan)).clip(0, 5)

    # Price vs moving-average stack. These are direct relationships, not raw prices.
    df["sma5_minus_sma20"] = dist_sma5 - dist_sma20
    df["sma20_minus_sma50"] = dist_sma20 - dist_sma50
    df["sma50_minus_sma200"] = dist_sma50 - dist_sma200
    df["price_above_sma_stack"] = (
        (dist_sma5 > 0)
        & (dist_sma20 > 0)
        & (dist_sma50 > -0.01)
    ).astype(int)

    # Oscillator context: RSI means different things depending on trend and extension.
    df["rsi_centered"] = (rsi - 50.0) / 50.0
    df["rsi_trend_interaction"] = df["rsi_centered"] * trend_score
    df["rsi_price_extension"] = df["rsi_centered"] * dist_sma20
    df["rsi_pullback_buy_zone"] = ((rsi.between(35, 55)) & (dist_sma20 > -0.025) & (numeric_series(df, "ret_5d") > 0)).astype(int)
    df["rsi_overextension_risk"] = ((rsi > 72) & (dist_sma20 > 0.045)).astype(int)

    # MACD/price/volume confirmation.
    df["macd_breakout_interaction"] = macd_hist * breakout_score
    df["macd_trend_interaction"] = macd_hist * trend_score
    df["volume_breakout_interaction"] = rel_vol * numeric_series(df, "breakout_above_prev_high")
    df["volume_trend_interaction"] = rel_vol * trend_score

    # Momentum acceleration and persistence.
    df["momentum_acceleration_3v10"] = numeric_series(df, "ret_3d") - numeric_series(df, "ret_10d")
    df["momentum_acceleration_5v20"] = numeric_series(df, "ret_5d") - numeric_series(df, "ret_20d")
    df["price_momentum_alignment"] = (
        (numeric_series(df, "ret_3d") > 0)
        & (numeric_series(df, "ret_5d") > 0)
        & (dist_sma20 > 0)
    ).astype(int)

    df["up_day_5_count"] = (
        numeric_series(df, "ret_1d")
        .gt(0)
        .groupby(df["symbol"])
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["volume_acceleration_3d"] = (
        volume.groupby(df["symbol"]).pct_change(3).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Compression -> expansion context.
    atr_mean20 = (
        atr_pct.groupby(df["symbol"])
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    range_mean20 = (
        daily_range_pct.groupby(df["symbol"])
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["volatility_compression"] = (atr_pct < atr_mean20).astype(int)
    df["range_expansion_vs_20"] = (daily_range_pct / range_mean20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["compression_breakout"] = (
        (df["volatility_compression"] == 1)
        & (numeric_series(df, "breakout_above_prev_high") == 1)
        & (rel_vol > 1.0)
    ).astype(int)

    # Relationship transitions: these tell the model not only where indicators are,
    # but whether price/indicators are improving, reclaiming, exhausting, or confirming.
    df["rsi_delta_3d"] = rsi.groupby(df["symbol"]).diff(3).fillna(0.0)
    df["macd_hist_delta_3d"] = macd_hist.groupby(df["symbol"]).diff(3).fillna(0.0)
    df["rel_volume_delta_3d"] = rel_vol.groupby(df["symbol"]).diff(3).fillna(0.0)
    df["dist_sma20_delta_3d"] = dist_sma20.groupby(df["symbol"]).diff(3).fillna(0.0)

    prev_dist_sma20 = dist_sma20.groupby(df["symbol"]).shift(1)
    prev_macd_hist = macd_hist.groupby(df["symbol"]).shift(1)
    prev_rsi = rsi.groupby(df["symbol"]).shift(1)

    df["price_reclaim_sma20"] = ((prev_dist_sma20 <= 0) & (dist_sma20 > 0)).astype(int)
    df["macd_turn_positive"] = ((prev_macd_hist <= 0) & (macd_hist > 0)).astype(int)
    df["rsi_reclaim_50"] = ((prev_rsi <= 50) & (rsi > 50)).astype(int)
    df["rsi_rising_3d"] = (df["rsi_delta_3d"] > 0).astype(int)
    df["macd_improving_3d"] = (df["macd_hist_delta_3d"] > 0).astype(int)
    df["volume_confirming_3d"] = (df["rel_volume_delta_3d"] > 0).astype(int)

    df["price_above_sma20_count_5"] = (
        dist_sma20.gt(0)
        .groupby(df["symbol"])
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["indicator_transition_score"] = (
        0.18 * df["price_reclaim_sma20"]
        + 0.16 * df["macd_turn_positive"]
        + 0.14 * df["rsi_reclaim_50"]
        + 0.14 * df["rsi_rising_3d"]
        + 0.14 * df["macd_improving_3d"]
        + 0.12 * df["volume_confirming_3d"]
        + 0.12 * numeric_series(df, "breakout_above_prev_high")
    ).clip(0, 1)

    # Composite relationship score used by the model and entry layer.
    df["price_indicator_alignment_score"] = (
        0.18 * df["price_above_sma_stack"]
        + 0.16 * df["price_momentum_alignment"]
        + 0.14 * numeric_series(df, "breakout_above_prev_high")
        + 0.14 * (macd_hist > 0).astype(int)
        + 0.12 * (rsi.between(45, 72)).astype(int)
        + 0.12 * (rel_vol > 1.0).astype(int)
        + 0.14 * df["compression_breakout"]
        - 0.12 * df["rsi_overextension_risk"]
    ).clip(0, 1)

    df["context_interaction_score"] = (
        0.45 * df["price_indicator_alignment_score"]
        + 0.25 * df["indicator_transition_score"]
        + 0.15 * df["compression_breakout"]
        + 0.15 * df["price_momentum_alignment"]
    ).clip(0, 1)

    return df.replace([np.inf, -np.inf], np.nan)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_aliases(df)
    df = add_stock_specific_features(df)
    df = add_tradability_flags(df)
    df = add_market_regime(df)
    df = add_market_correlation_features(df)
    df = add_price_indicator_interaction_features(df)
    df = add_momentum_state(df)
    df = make_target(df)
    return df.replace([np.inf, -np.inf], np.nan)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(ID_COLUMNS) | set(LEAKAGE_COLUMNS)
    blocked |= {"open", "high", "low", "close"}
    blocked |= DROP_MACRO_DOMINANT_FEATURES
    blocked |= {"market_regime"}

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

class PriceIndicatorTransformer(nn.Module):
    """Transformer encoder for rolling price/indicator sequences."""

    def __init__(
        self,
        n_features: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_LAYERS,
        dropout: float = TRANSFORMER_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x) + self.position_embedding[:, : x.shape[1], :]
        z = self.encoder(z)
        # Use the last timestep because the target is tied to the current row.
        return self.head(z[:, -1, :]).squeeze(-1)


class SequenceTransformerClassifier:
    """
    Walk-forward-safe Transformer wrapper.

    It trains on per-symbol rolling windows and stores the last SEQUENCE_LENGTH - 1
    rows from training so the first test rows can be scored with proper context.
    """

    def __init__(self) -> None:
        self.feature_cols: list[str] = []
        self.feature_median: pd.Series | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.model: PriceIndicatorTransformer | None = None
        self.history_by_symbol: dict[str, pd.DataFrame] = {}
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _clean_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        x_df = df[self.feature_cols].copy()
        x_df = x_df.replace([np.inf, -np.inf], np.nan)

        if fit or self.feature_median is None:
            self.feature_median = x_df.median(numeric_only=True).fillna(0.0)

        x_df = x_df.fillna(self.feature_median).astype(float)
        x = x_df.to_numpy(dtype=np.float32)

        if fit or self.feature_mean is None or self.feature_std is None:
            self.feature_mean = np.nanmean(x, axis=0)
            self.feature_std = np.nanstd(x, axis=0)
            self.feature_std[self.feature_std < 1e-6] = 1.0

        x = (x - self.feature_mean) / self.feature_std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32)

    def _sequence_arrays(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        fit_features: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        if df.empty:
            y_empty = np.array([], dtype=np.float32) if target_col else None
            return np.empty((0, SEQUENCE_LENGTH, len(self.feature_cols)), dtype=np.float32), y_empty, np.array([])

        tmp = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=False).rename(columns={"index": "_orig_index"})
        x_all = self._clean_features(tmp, fit=fit_features)
        tmp["_x_pos"] = np.arange(len(tmp))

        sequences: list[np.ndarray] = []
        targets: list[float] = []
        row_ids: list[int] = []

        for _, g in tmp.groupby("symbol", sort=False):
            positions = g["_x_pos"].to_numpy(dtype=int)
            orig_index = g["_orig_index"].to_numpy()
            if len(positions) < SEQUENCE_LENGTH:
                continue

            for end_i in range(SEQUENCE_LENGTH - 1, len(positions)):
                locs = positions[end_i - SEQUENCE_LENGTH + 1 : end_i + 1]
                sequences.append(x_all[locs])
                row_ids.append(int(orig_index[end_i]))
                if target_col is not None:
                    targets.append(float(g[target_col].iloc[end_i]))

        x_seq = np.stack(sequences).astype(np.float32) if sequences else np.empty((0, SEQUENCE_LENGTH, len(self.feature_cols)), dtype=np.float32)
        y_seq = np.array(targets, dtype=np.float32) if target_col is not None else None
        return x_seq, y_seq, np.array(row_ids)

    def _store_history(self, train_df: pd.DataFrame) -> None:
        self.history_by_symbol = {}
        for sym, g in train_df.sort_values(["symbol", "timestamp"]).groupby("symbol"):
            self.history_by_symbol[str(sym)] = g.tail(SEQUENCE_LENGTH - 1).copy()

    def _context_for_prediction(self, test_df: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        test_symbols = set(test_df["symbol"].astype(str))
        for sym in test_symbols:
            hist = self.history_by_symbol.get(sym)
            if hist is not None and not hist.empty:
                pieces.append(hist)
        pieces.append(test_df)
        return pd.concat(pieces, ignore_index=False).sort_values(["symbol", "timestamp"])

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_LABEL) -> "SequenceTransformerClassifier":
        self.feature_cols = list(feature_cols)
        self._store_history(train_df)

        x_seq, y_seq, _ = self._sequence_arrays(train_df, target_col=target_col, fit_features=True)
        if y_seq is None or len(x_seq) < TRANSFORMER_MIN_SEQUENCES:
            raise RuntimeError(
                f"Not enough sequences for Transformer: {len(x_seq):,}. "
                f"Need at least {TRANSFORMER_MIN_SEQUENCES:,}."
            )

        order = np.arange(len(x_seq))
        # Keep validation chronological to reduce leakage versus random splits.
        split = int(len(order) * (1.0 - TRANSFORMER_VAL_FRACTION))
        split = int(np.clip(split, 1, len(order) - 1))
        train_idx = order[:split]
        val_idx = order[split:]

        x_train = torch.tensor(x_seq[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_seq[train_idx], dtype=torch.float32)
        x_val = torch.tensor(x_seq[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_seq[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=TRANSFORMER_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=TRANSFORMER_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        self.model = PriceIndicatorTransformer(n_features=len(self.feature_cols)).to(self.device)

        pos_count = float(y_train.sum().item())
        neg_count = float(len(y_train) - pos_count)
        pos_weight_value = neg_count / max(pos_count, 1.0)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=self.device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=TRANSFORMER_LR,
            weight_decay=TRANSFORMER_WEIGHT_DECAY,
        )

        best_state = None
        best_val_loss = float("inf")
        patience_left = TRANSFORMER_PATIENCE

        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        for epoch in range(1, TRANSFORMER_EPOCHS + 1):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    val_losses.append(float(loss_fn(logits, yb).detach().cpu()))

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = TRANSFORMER_PATIENCE
            else:
                patience_left -= 1

            if epoch == 1 or epoch % 5 == 0 or patience_left <= 0:
                print(
                    f"[transformer] epoch={epoch:02d} "
                    f"train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} "
                    f"pos_weight={pos_weight_value:.2f}"
                )

            if patience_left <= 0:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, test_df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Transformer model has not been fit.")
        if feature_cols is not None and list(feature_cols) != self.feature_cols:
            raise ValueError("feature_cols differ from training feature columns.")

        test_index = np.array(test_df.index)
        context = self._context_for_prediction(test_df)
        x_seq, _, row_ids = self._sequence_arrays(context, target_col=None, fit_features=False)

        proba_by_index: dict[int, float] = {}
        if len(x_seq) > 0:
            loader = DataLoader(torch.tensor(x_seq, dtype=torch.float32), batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False)
            probs = []
            self.model.eval()
            with torch.no_grad():
                for xb in loader:
                    xb = xb.to(self.device)
                    logits = self.model(xb)
                    probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            proba_by_index = {int(idx): float(prob) for idx, prob in zip(row_ids, probs)}

        # Rows that still lack enough sequence context get a conservative low probability.
        out = np.array([proba_by_index.get(int(idx), 0.0) for idx in test_index], dtype=np.float32)
        return np.column_stack([1.0 - out, out])


def build_model() -> SequenceTransformerClassifier:
    return SequenceTransformerClassifier()


def make_windows(unique_dates: list[pd.Timestamp]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    windows = []
    start = 0
    window_id = 0

    while True:
        train_start = start
        train_end = train_start + TRAIN_DAYS
        test_end = train_end + TEST_DAYS
        if test_end > len(unique_dates):
            break

        train_dates = np.array(unique_dates[train_start:train_end])
        test_dates = np.array(unique_dates[train_end:test_end])
        windows.append((window_id, train_dates, test_dates))
        window_id += 1
        start += STEP_DAYS

    return windows


def score_test_window(
    model: SequenceTransformerClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    window_id: int,
) -> pd.DataFrame:
    out = test_df.copy()
    proba = model.predict_proba(out, feature_cols=feature_cols)[:, 1]
    out["walkforward_window"] = window_id
    out["explosive_alpha_probability"] = proba
    out["model_prediction"] = (proba >= SOFT_MIN_PROBABILITY).astype(int)
    return out


# =============================================================================
# Sizing and selection
# =============================================================================

def capital_state(equity: float, high_water_mark: float) -> str:
    if high_water_mark <= 0:
        return "NORMAL"

    ratio = equity / high_water_mark
    if ratio <= EQUITY_KILL:
        return "KILL"
    if ratio <= EQUITY_DEFENSIVE:
        return "DEFENSIVE"
    if ratio <= EQUITY_RECOVERY:
        return "RECOVERY"
    return "NORMAL"


def regime_gross_exposure(regime: str, cap_state: str) -> float:
    base = REGIME_GROSS_EXPOSURE.get(regime, 0.45)
    mult = CAPITAL_PRESERVATION_GROSS_MULT.get(cap_state, 1.0)
    return base * mult


def regime_new_entries(regime: str, cap_state: str) -> int:
    if cap_state == "KILL":
        return 0
    base = REGIME_MAX_NEW_ENTRIES.get(regime, TARGET_NEW_ENTRIES_PER_DAY)
    if cap_state == "DEFENSIVE":
        return 1
    if cap_state == "RECOVERY":
        return min(2, base)
    return int(np.clip(base, 1, MAX_NEW_ENTRIES_PER_DAY))


def expected_value_score(prob: float, upside_r: float = KELLY_WIN_R, downside_r: float = KELLY_LOSS_R) -> float:
    p = float(np.clip(prob, 0.0, 1.0))
    return p * upside_r - (1.0 - p) * downside_r


def kelly_weight(prob: float) -> float:
    ev = expected_value_score(prob)
    if ev <= 0:
        return 0.0
    full_kelly = ev / KELLY_WIN_R
    return float(np.clip(full_kelly * KELLY_FRACTION, 0.0, MAX_KELLY_WEIGHT))


def realized_volatility_pct(row: pd.Series) -> float:
    """Pick the best available per-symbol volatility estimate for sizing."""
    for col in ["volatility_20d", "atr_pct", "daily_range_pct"]:
        val = float(row.get(col, np.nan))
        if np.isfinite(val) and val > 0:
            return val
    return TARGET_POSITION_VOL


def volatility_adjustment(vol_pct: float) -> float:
    if not np.isfinite(vol_pct) or vol_pct <= 0:
        vol_pct = TARGET_POSITION_VOL
    return float(np.clip(TARGET_POSITION_VOL / vol_pct, VOL_ADJUST_MIN, VOL_ADJUST_MAX))


def dynamic_probability_threshold(regime: str, cap_state: str) -> float:
    base = DYNAMIC_PROBABILITY_THRESHOLDS.get(str(regime), SOFT_MIN_PROBABILITY)
    addon = CAPITAL_STATE_THRESHOLD_ADDON.get(str(cap_state), 0.0)
    return float(np.clip(base + addon, 0.0, 0.99))


def portfolio_vol_scaler(planned: pd.DataFrame) -> float:
    """
    Conservative one-period portfolio volatility approximation.

    Without a full covariance matrix, assume selected names are positively correlated
    and combine their vol-weighted exposures. This protects against oversized stacks
    of high-beta/high-correlation names.
    """
    if planned.empty or "target_weight" not in planned.columns:
        return 1.0
    vols = planned.apply(realized_volatility_pct, axis=1).astype(float)
    approx_vol = float(np.sqrt(np.sum((planned["target_weight"].astype(float) * vols) ** 2)))
    if not np.isfinite(approx_vol) or approx_vol <= PORTFOLIO_VOL_CAP:
        return 1.0
    return float(np.clip(PORTFOLIO_VOL_CAP / approx_vol, 0.25, 1.0))


def convex_rank_weights(scores: pd.Series, power: float = CONVEX_POWER) -> pd.Series:
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0).clip(lower=0.0)
    if s.sum() <= 0:
        ranks = pd.Series(np.arange(len(s), 0, -1), index=s.index, dtype=float)
        s = ranks / ranks.max()
    powered = s ** power
    if powered.sum() <= 0:
        return pd.Series(1.0 / max(len(s), 1), index=s.index)
    return powered / powered.sum()


def score_entry_candidates(day: pd.DataFrame, equity: float, high_water_mark: float) -> pd.DataFrame:
    cap_state = capital_state(equity, high_water_mark)

    base = day[
        (day["tradable_symbol"] == 1)
        & (day["close"] >= MIN_PRICE)
        & (day["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20)
        & (day["relative_volume_20"] >= MIN_RELATIVE_VOLUME)
        & (day["range_expansion"] >= MIN_RANGE_EXPANSION)
    ].copy()

    if base.empty:
        return base

    prob = base["explosive_alpha_probability"].fillna(0).astype(float)
    ev = prob.map(expected_value_score)
    rel_vol = numeric_series(base, "relative_volume_20", 1.0).clip(0, 3) / 3.0
    range_exp = numeric_series(base, "range_expansion").clip(0, 0.08) / 0.08
    breakout = numeric_series(base, "breakout_above_prev_high")
    trend = numeric_series(base, "trend_alignment")

    base["expected_value"] = ev
    base["entry_probability_threshold"] = base["market_regime"].map(
        lambda r: dynamic_probability_threshold(str(r), cap_state)
    )

    base["explosive_score"] = (
        0.38 * prob
        + 0.20 * numeric_series(base, "explosion_setup_score")
        + 0.12 * breakout
        + 0.10 * range_exp
        + 0.08 * rel_vol
        + 0.07 * numeric_series(base, "momentum_score")
        + 0.05 * trend
        + 0.10 * numeric_series(base, "price_indicator_alignment_score")
        + 0.08 * numeric_series(base, "indicator_transition_score")
        + 0.08 * numeric_series(base, "context_interaction_score")
        + 0.10 * ev.clip(lower=0)
        - 0.22 * numeric_series(base, "momentum_negative")
        - 0.12 * numeric_series(base, "late_reversal_risk")
    )

    base["confidence_score"] = (
        0.58 * prob
        + 0.17 * numeric_series(base, "explosion_setup_score")
        + 0.10 * breakout
        + 0.08 * numeric_series(base, "momentum_score")
        + 0.08 * numeric_series(base, "price_indicator_alignment_score")
        + 0.06 * numeric_series(base, "indicator_transition_score")
        + 0.05 * ev.clip(lower=0)
    ).clip(0, 1)

    # Moonshot score: requires momentum, explosive setup quality, volume participation,
    # and a breakout/trend context before attempting to hold for a very large return.
    base["moonshot_score"] = (
        0.40 * numeric_series(base, "momentum_score").clip(0, 1)
        + 0.25 * numeric_series(base, "explosion_setup_score").clip(0, 1)
        + 0.15 * numeric_series(base, "relative_volume_20", 1.0).clip(0, 3) / 3.0
        + 0.10 * numeric_series(base, "breakout_above_prev_high").clip(0, 1)
        + 0.10 * numeric_series(base, "trend_alignment").clip(0, 1)
        + 0.10 * numeric_series(base, "context_interaction_score").clip(0, 1)
    ).clip(0, 1)

    # No breakout override: every entry must pass model probability, EV, and moonshot filters.
    base = base[
        (base["explosive_alpha_probability"] >= base["entry_probability_threshold"])
        & (base["expected_value"] >= MIN_EXPECTED_VALUE_FOR_ENTRY)
        & (base["explosive_score"] >= MIN_EXPLOSIVE_SCORE)
        & ((numeric_series(base, "context_interaction_score") >= 0.35) | (base["explosive_alpha_probability"] >= base["entry_probability_threshold"] + 0.08))
        & (base["momentum_negative"] == 0)
        & (base["late_reversal_risk"] == 0)
        & (
            (base["trend_alignment"] == 1)
            | (base["explosion_setup_score"] >= MIN_EXPLOSION_SETUP_SCORE)
            | (base["breakout_above_prev_high"] == 1)
        )
    ].copy()

    return base


def apply_correlation_filter(candidates: pd.DataFrame, day: pd.DataFrame, open_symbols: set[str]) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    if "symbol_corr_20_to_qqq" not in candidates.columns:
        return candidates

    existing_high_corr = 0
    if open_symbols:
        open_rows = day[day["symbol"].isin(open_symbols)].copy()
        existing_high_corr = int((numeric_series(open_rows, "symbol_corr_20_to_qqq").abs() >= HIGH_QQQ_CORR_THRESHOLD).sum())

    selected = []
    high_corr_count = existing_high_corr

    for _, row in candidates.sort_values("explosive_score", ascending=False).iterrows():
        corr_to_qqq = abs(float(row.get("symbol_corr_20_to_qqq", 0.0)))
        if corr_to_qqq >= HIGH_QQQ_CORR_THRESHOLD and high_corr_count >= MAX_QQQ_CORRELATED_NAMES:
            continue
        selected.append(row)
        if corr_to_qqq >= HIGH_QQQ_CORR_THRESHOLD:
            high_corr_count += 1

    if not selected:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(selected).reset_index(drop=True)


def build_weight_plan(candidates: pd.DataFrame, equity: float, high_water_mark: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    regime = str(candidates["market_regime"].iloc[0]) if "market_regime" in candidates.columns else "CHOP_NEUTRAL"
    cap_state = capital_state(equity, high_water_mark)
    gross_cap = min(regime_gross_exposure(regime, cap_state), BASE_MAX_GROSS_EXPOSURE)
    max_entries = regime_new_entries(regime, cap_state)

    if gross_cap <= 0 or max_entries <= 0:
        return pd.DataFrame()

    filtered = candidates.copy().sort_values("explosive_score", ascending=False).head(max_entries)
    if filtered.empty:
        return filtered

    raw_weights = []
    for _, row in filtered.iterrows():
        prob = float(row.get("explosive_alpha_probability", 0.0))
        vol_pct = realized_volatility_pct(row)
        regime_mult = REGIME_WEIGHT_MULTIPLIER.get(regime, 0.70)
        kw = kelly_weight(prob)
        vol_adj = volatility_adjustment(vol_pct)
        confidence = float(row.get("confidence_score", prob))
        raw_weights.append(max(kw * vol_adj * regime_mult * (0.50 + confidence), 0.0))

    filtered["raw_weight"] = raw_weights
    convex = convex_rank_weights(filtered["explosive_score"], power=CONVEX_POWER)
    filtered["raw_weight"] = 0.65 * filtered["raw_weight"] + 0.35 * convex

    total_raw = filtered["raw_weight"].sum()
    if total_raw <= 0:
        return pd.DataFrame()

    filtered["target_weight"] = filtered["raw_weight"] / total_raw * gross_cap
    filtered["target_weight"] = filtered["target_weight"].clip(0, MAX_WEIGHT_PER_POSITION)
    filtered.loc[filtered["target_weight"] < MIN_WEIGHT_PER_POSITION, "target_weight"] = 0.0

    total_weight = filtered["target_weight"].sum()
    if total_weight > gross_cap:
        filtered["target_weight"] *= gross_cap / total_weight

    vol_scaler = portfolio_vol_scaler(filtered)
    filtered["target_weight"] *= vol_scaler

    filtered["capital_state"] = cap_state
    filtered["regime_gross_cap"] = gross_cap
    filtered["portfolio_vol_scaler"] = vol_scaler
    filtered["sizing_method"] = "PROB_EV_KELLY_VOL_DRAWDOWN_CORR"

    return filtered[filtered["target_weight"] > 0].copy()


def get_entry_candidates(
    day: pd.DataFrame,
    available_slots: int,
    equity: float,
    high_water_mark: float,
    open_symbols: set[str] | None = None,
    cooldowns: dict[str, int] | None = None,
) -> pd.DataFrame:
    if available_slots <= 0:
        return pd.DataFrame()

    open_symbols = open_symbols or set()
    cooldowns = cooldowns or {}

    # Skip only truly defensive markets. Do not block CHOP completely,
    # because some explosive breakouts still happen there.
    if not day.empty:
        regime = str(day["market_regime"].mode().iloc[0]) if "market_regime" in day.columns else "CHOP_NEUTRAL"
        risk_on_score = float(day["risk_on_score"].median()) if "risk_on_score" in day.columns else 0.5

        if regime == "DEFENSIVE_RISK_OFF" and risk_on_score < 0.45:
            return pd.DataFrame()

    candidates = score_entry_candidates(day, equity=equity, high_water_mark=high_water_mark)
    if candidates.empty:
        return candidates

    blocked = set(open_symbols) | {sym for sym, remaining in cooldowns.items() if remaining > 0}
    if blocked:
        candidates = candidates[~candidates["symbol"].isin(blocked)].copy()

    candidates = apply_correlation_filter(candidates, day=day, open_symbols=open_symbols)
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("explosive_score", ascending=False).head(available_slots).copy()
    return build_weight_plan(candidates, equity=equity, high_water_mark=high_water_mark)


# =============================================================================
# Probabilistic risk controller
# =============================================================================

def update_recent_outcomes(pos: Position, close_price: float) -> None:
    bar_return_from_entry = close_price / pos.entry_price - 1.0
    pos.recent_outcomes.append(1 if bar_return_from_entry > 0 else 0)
    if len(pos.recent_outcomes) > 3:
        pos.recent_outcomes.pop(0)


def trade_profit(pos: Position, close_price: float) -> float:
    return close_price / pos.entry_price - 1.0


def meta_controller_decision(pos: Position, row: pd.Series, close_price: float) -> str:
    prob = float(row.get("explosive_alpha_probability", 0.5))
    ev = expected_value_score(prob)
    momentum_negative = bool(row.get("momentum_negative", 0))
    momentum_favorable = bool(row.get("momentum_favorable", 1))
    breakout = bool(row.get("breakout_above_prev_high", 0))
    explosion = float(row.get("explosion_setup_score", 0.0))
    pnl_pct = trade_profit(pos, close_price)

    # Keep tracking model edge while the trade is alive.
    pos.peak_probability = max(pos.peak_probability, prob)
    peak_pnl_pct = pos.max_favorable_price / pos.entry_price - 1.0 if pos.entry_price > 0 else 0.0

    # 0) Moonshot target: if the trade reaches the 100% target, take the win.
    if pnl_pct >= TARGET_TRADE_RETURN:
        return "EXIT_TARGET_100"

    # 0b) Profit lock: after a large move, do not allow a full round trip.
    if peak_pnl_pct >= PROFIT_LOCK_AFTER and pnl_pct <= PROFIT_LOCK_FLOOR:
        return "EXIT_PROFIT_LOCK"

    # 0c) Early failure: moonshot trades should show strength quickly.
    if pos.bars_held >= EARLY_FAILURE_BARS and pnl_pct < EARLY_FAILURE_MIN_PROFIT:
        return "EXIT_EARLY_FAILURE"

    # 1) Model failure: exit when estimated edge has turned clearly negative.
    if prob < MODEL_FAILURE_PROB and ev < -0.20:
        return "EXIT_MODEL_FAILURE"

    # 2) Edge decay: exit when current confidence collapses from its trade peak.
    if pos.peak_probability > 0 and prob < pos.peak_probability * EDGE_DECAY_FRACTION:
        return "EXIT_EDGE_DECAY"

    # 3) Momentum confirmation failure: if the model and price action disagree, leave.
    if momentum_negative and prob < max(MODEL_FAILURE_PROB, 0.40):
        return "EXIT_MOMENTUM_FAILURE"

    # 4) Time decay: explosive trades should work quickly.
    if pos.bars_held >= TIME_DECAY_BARS and pnl_pct < MIN_TIME_DECAY_PROFIT:
        return "EXIT_TIME_DECAY"

    # 5) Hard time stop: avoid capital getting stuck.
    if pos.bars_held >= MAX_HOLD_BARS:
        return "EXIT_TIME_STOP"

    # Selective scale-out: harvest earlier when upside has appeared but edge is fading.
    if pnl_pct >= SCALE_OUT_2_PROFIT and prob < SCALE_OUT_PROB_DROP and pos.scale_outs < 2:
        return "SCALE_OUT_2"

    if pnl_pct >= SCALE_OUT_1_PROFIT and prob < SCALE_OUT_PROB_DROP and pos.scale_outs < 1:
        return "SCALE_OUT_1"

    # Conservative pyramiding: only add once when the trade is already working and edge improves.
    if (
        ALLOW_SCALE_IN
        and pos.scale_ins < MAX_SCALE_INS_PER_POSITION
        and pos.bars_held >= SCALE_IN_MIN_HOLD_BARS
        and pnl_pct >= MIN_SCALE_IN_PROFIT
        and momentum_favorable
        and prob >= max(SCALE_IN_PROBABILITY, pos.entry_probability + 0.05)
        and (breakout or explosion >= 0.70 or ev > 0.50)
    ):
        return "SCALE_IN"

    return "HOLD"


def model_exit_decision(pos: Position, row: pd.Series, close_price: float) -> tuple[str | None, float | None, str]:
    decision = meta_controller_decision(pos, row, close_price)

    if decision.startswith("EXIT_"):
        exit_price = close_price * (1 - EXIT_SLIPPAGE_BPS / 10_000)
        return decision.replace("EXIT_", ""), exit_price, "EXIT"

    return None, None, decision


# =============================================================================
# Simulation
# =============================================================================

def simulate_positions(scored: pd.DataFrame, window_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = scored.sort_values(["timestamp", "symbol"]).copy()
    unique_dates = list(scored["timestamp"].dropna().sort_values().unique())

    positions: dict[str, Position] = {}
    cooldowns: dict[str, int] = {}
    events: list[dict] = []
    daily_rows: list[dict] = []

    equity = STARTING_EQUITY
    high_water_mark = STARTING_EQUITY

    for ts in unique_dates:
        # Decay per-symbol cooldowns after losing exits.
        cooldowns = {sym: bars - 1 for sym, bars in cooldowns.items() if bars - 1 > 0}

        day = scored[scored["timestamp"] == ts].copy()
        day_by_symbol = {r["symbol"]: r for _, r in day.iterrows()}

        day_pnl = 0.0
        closed_today = []

        # Manage existing positions.
        for sym, pos in list(positions.items()):
            if sym not in day_by_symbol:
                continue

            row = day_by_symbol[sym]

            open_price = float(row.get("open", np.nan))
            high_price = float(row.get("high", np.nan))
            low_price = float(row.get("low", np.nan))
            close_price = float(row.get("close", np.nan))

            if not all(np.isfinite(x) and x > 0 for x in [open_price, high_price, low_price, close_price]):
                continue

            pos.bars_held += 1
            pos.max_favorable_price = max(pos.max_favorable_price, high_price)
            pos.min_adverse_price = min(pos.min_adverse_price, low_price)

            update_recent_outcomes(pos, close_price)

            exit_reason, exit_price, controller_action = model_exit_decision(
                pos=pos,
                row=row,
                close_price=close_price,
            )

            if exit_reason is not None and exit_price is not None:
                pnl = (exit_price - pos.entry_price) * pos.shares
                ret = exit_price / pos.entry_price - 1.0
                day_pnl += pnl

                events.append(
                    {
                        "walkforward_window": window_id,
                        "event": "EXIT",
                        "timestamp": ts,
                        "symbol": sym,
                        "entry_timestamp": pos.entry_timestamp,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "shares": pos.shares,
                        "weight_at_entry": pos.weight_at_entry,
                        "target_weight": pos.target_weight,
                        "pnl": pnl,
                        "return": ret,
                        "bars_held": pos.bars_held,
                        "exit_reason": exit_reason,
                        "controller_action": controller_action,
                        "model_prob": float(row.get("explosive_alpha_probability", np.nan)),
                        "recent_outcomes": str(pos.recent_outcomes),
                        "entry_probability": pos.entry_probability,
                        "peak_probability": pos.peak_probability,
                        "scale_ins": pos.scale_ins,
                        "scale_outs": pos.scale_outs,
                        "max_favorable_price": pos.max_favorable_price,
                        "min_adverse_price": pos.min_adverse_price,
                    }
                )
                if pnl < 0:
                    cooldowns[sym] = LOSS_COOLDOWN_BARS
                closed_today.append(sym)
                continue

            if controller_action in {"SCALE_OUT_1", "SCALE_OUT_2"} and pos.shares > 1:
                fraction = SCALE_OUT_1_FRACTION if controller_action == "SCALE_OUT_1" else SCALE_OUT_2_FRACTION
                sell_shares = max(1, int(pos.shares * fraction))
                sell_price = close_price * (1 - EXIT_SLIPPAGE_BPS / 10_000)
                pnl = (sell_price - pos.entry_price) * sell_shares
                ret = sell_price / pos.entry_price - 1.0
                day_pnl += pnl
                pos.shares -= sell_shares
                pos.scale_outs += 1

                events.append(
                    {
                        "walkforward_window": window_id,
                        "event": "SCALE_OUT",
                        "timestamp": ts,
                        "symbol": sym,
                        "entry_timestamp": pos.entry_timestamp,
                        "entry_price": pos.entry_price,
                        "exit_price": sell_price,
                        "shares": sell_shares,
                        "remaining_shares": pos.shares,
                        "weight_at_entry": pos.weight_at_entry,
                        "target_weight": pos.target_weight,
                        "pnl": pnl,
                        "return": ret,
                        "bars_held": pos.bars_held,
                        "exit_reason": controller_action,
                        "controller_action": controller_action,
                        "model_prob": float(row.get("explosive_alpha_probability", np.nan)),
                        "recent_outcomes": str(pos.recent_outcomes),
                        "entry_probability": pos.entry_probability,
                        "peak_probability": pos.peak_probability,
                        "scale_ins": pos.scale_ins,
                        "scale_outs": pos.scale_outs,
                        "scale_outs": pos.scale_outs,
                        "max_favorable_price": pos.max_favorable_price,
                        "min_adverse_price": pos.min_adverse_price,
                    }
                )

                if pos.shares <= 0:
                    closed_today.append(sym)

            elif controller_action == "SCALE_IN":
                add_weight = min(
                    pos.target_weight * SCALE_IN_FRACTION_OF_TARGET,
                    MAX_WEIGHT_PER_POSITION - pos.weight_at_entry,
                )

                if add_weight > MIN_WEIGHT_PER_POSITION:
                    add_dollars = equity * add_weight
                    add_price = close_price * (1 + ENTRY_SLIPPAGE_BPS / 10_000)
                    add_shares = int(add_dollars // add_price)

                    if add_shares > 0:
                        old_cost = pos.entry_price * pos.shares
                        new_cost = add_price * add_shares
                        new_total = pos.shares + add_shares

                        pos.entry_price = (old_cost + new_cost) / new_total
                        pos.shares = new_total
                        pos.weight_at_entry += add_weight
                        pos.scale_ins += 1

                        pos.peak_probability = max(pos.peak_probability, float(row.get("explosive_alpha_probability", pos.peak_probability)))

                        events.append(
                            {
                                "walkforward_window": window_id,
                                "event": "SCALE_IN",
                                "timestamp": ts,
                                "symbol": sym,
                                "entry_timestamp": pos.entry_timestamp,
                                "entry_price": add_price,
                                "average_entry_price": pos.entry_price,
                                "shares": add_shares,
                                "total_shares": pos.shares,
                                "weight_added": add_weight,
                                "weight_at_entry": pos.weight_at_entry,
                                "target_weight": pos.target_weight,
                                "pnl": 0.0,
                                "return": 0.0,
                                "bars_held": pos.bars_held,
                                "exit_reason": "",
                                "controller_action": controller_action,
                                "model_prob": float(row.get("explosive_alpha_probability", np.nan)),
                                "recent_outcomes": str(pos.recent_outcomes),
                                "entry_probability": pos.entry_probability,
                                "peak_probability": pos.peak_probability,
                                "scale_ins": pos.scale_ins,
                        "scale_outs": pos.scale_outs,
                            }
                        )

        for sym in closed_today:
            positions.pop(sym, None)

        equity += day_pnl
        high_water_mark = max(high_water_mark, equity)

        # Open new positions.
        available_slots = MAX_OPEN_POSITIONS - len(positions)
        candidates = get_entry_candidates(
            day,
            available_slots=available_slots,
            equity=equity,
            high_water_mark=high_water_mark,
            open_symbols=set(positions.keys()),
            cooldowns=cooldowns,
        )

        if not candidates.empty:
            candidates = candidates[~candidates["symbol"].isin(set(positions.keys()))].copy()

        if not candidates.empty:
            for _, row in candidates.iterrows():
                sym = str(row["symbol"])
                close_price = float(row.get("close", np.nan))
                target_weight = float(row.get("target_weight", 0.0))

                if not np.isfinite(close_price) or close_price <= 0 or target_weight <= 0:
                    continue

                entry_price = close_price * (1 + ENTRY_SLIPPAGE_BPS / 10_000)
                target_dollars = equity * target_weight
                shares = int(target_dollars // entry_price)

                if shares <= 0:
                    continue

                entry_prob = float(row.get("explosive_alpha_probability", np.nan))

                positions[sym] = Position(
                    symbol=sym,
                    entry_timestamp=ts,
                    entry_price=entry_price,
                    shares=shares,
                    weight_at_entry=target_weight,
                    target_weight=target_weight,
                    entry_probability=entry_prob,
                    peak_probability=entry_prob,
                    scale_ins=0,
                    scale_outs=0,
                    bars_held=0,
                    max_favorable_price=entry_price,
                    min_adverse_price=entry_price,
                    recent_outcomes=[],
                )

                events.append(
                    {
                        "walkforward_window": window_id,
                        "event": "ENTRY",
                        "timestamp": ts,
                        "symbol": sym,
                        "entry_timestamp": ts,
                        "entry_price": entry_price,
                        "shares": shares,
                        "weight_at_entry": target_weight,
                        "target_weight": target_weight,
                        "pnl": 0.0,
                        "return": 0.0,
                        "bars_held": 0,
                        "exit_reason": "",
                        "controller_action": "OPEN_EXPLOSIVE",
                        "model_prob": row.get("explosive_alpha_probability", np.nan),
                        "explosive_score": row.get("explosive_score", np.nan),
                        "expected_value": row.get("expected_value", np.nan),
                        "moonshot_score": row.get("moonshot_score", np.nan),
                        "recent_outcomes": "",
                        "entry_probability": entry_prob,
                        "peak_probability": entry_prob,
                        "entry_probability_threshold": row.get("entry_probability_threshold", np.nan),
                        "portfolio_vol_scaler": row.get("portfolio_vol_scaler", np.nan),
                        "symbol_corr_20_to_qqq": row.get("symbol_corr_20_to_qqq", np.nan),
                        "scale_ins": 0,
                        "scale_outs": 0,
                        "max_favorable_price": entry_price,
                        "min_adverse_price": entry_price,
                        "selection_score": row.get("explosive_score", np.nan),
                        "confidence_score": row.get("confidence_score", np.nan),
                        "capital_state": row.get("capital_state", ""),
                        "regime_gross_cap": row.get("regime_gross_cap", np.nan),
                        "sizing_method": row.get("sizing_method", ""),
                    }
                )

        # Mark open positions.
        open_value = 0.0
        unrealized_pnl = 0.0

        for sym, pos in positions.items():
            row = day_by_symbol.get(sym)
            if row is None:
                continue
            close_price = float(row.get("close", np.nan))
            if not np.isfinite(close_price) or close_price <= 0:
                continue
            open_value += close_price * pos.shares
            unrealized_pnl += (close_price - pos.entry_price) * pos.shares

        marked_equity = equity + unrealized_pnl
        high_water_mark = max(high_water_mark, marked_equity)
        cap_state = capital_state(marked_equity, high_water_mark)

        daily_rows.append(
            {
                "walkforward_window": window_id,
                "timestamp": ts,
                "realized_pnl": day_pnl,
                "unrealized_pnl": unrealized_pnl,
                "equity": marked_equity,
                "cash_equity": equity,
                "high_water_mark": high_water_mark,
                "capital_state": cap_state,
                "open_positions": len(positions),
                "open_market_value": open_value,
                "gross_exposure": open_value / marked_equity if marked_equity > 0 else 0.0,
            }
        )

    # Force close at window end.
    if positions and unique_dates:
        final_ts = unique_dates[-1]
        final_day = scored[scored["timestamp"] == final_ts].copy()
        final_by_symbol = {r["symbol"]: r for _, r in final_day.iterrows()}

        for sym, pos in list(positions.items()):
            row = final_by_symbol.get(sym)
            if row is None:
                continue

            close_price = float(row.get("close", np.nan))
            if not np.isfinite(close_price) or close_price <= 0:
                continue

            exit_price = close_price * (1 - EXIT_SLIPPAGE_BPS / 10_000)
            pnl = (exit_price - pos.entry_price) * pos.shares
            ret = exit_price / pos.entry_price - 1.0
            equity += pnl

            events.append(
                {
                    "walkforward_window": window_id,
                    "event": "EXIT",
                    "timestamp": final_ts,
                    "symbol": sym,
                    "entry_timestamp": pos.entry_timestamp,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "weight_at_entry": pos.weight_at_entry,
                    "target_weight": pos.target_weight,
                    "pnl": pnl,
                    "return": ret,
                    "bars_held": pos.bars_held,
                    "exit_reason": "FORCED_WINDOW_END_EXIT",
                    "controller_action": "EXIT_TIME",
                    "model_prob": float(row.get("explosive_alpha_probability", np.nan)),
                    "recent_outcomes": str(pos.recent_outcomes),
                    "entry_probability": pos.entry_probability,
                    "peak_probability": pos.peak_probability,
                    "scale_ins": pos.scale_ins,
                        "scale_outs": pos.scale_outs,
                    "max_favorable_price": pos.max_favorable_price,
                    "min_adverse_price": pos.min_adverse_price,
                }
            )

        positions.clear()

    return pd.DataFrame(daily_rows), pd.DataFrame(events)


# =============================================================================
# Metrics and plotting
# =============================================================================

def calculate_window_metrics(daily: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "closed_trades": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "avg_open_positions": 0.0,
            "avg_gross_exposure": 0.0,
            "model_exit_count": 0,
            "scale_in_count": 0,
            "scale_out_count": 0,
        }

    d = daily.copy().sort_values("timestamp")
    d["daily_return"] = d["equity"].pct_change().fillna(0.0)
    eq = d["equity"]
    dd = eq / eq.cummax() - 1

    total_return = eq.iloc[-1] / max(eq.iloc[0], 1e-9) - 1
    annual_return = (1 + total_return) ** (252 / max(len(d), 1)) - 1
    annual_vol = d["daily_return"].std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    exits = events[events["event"] == "EXIT"].copy() if not events.empty else pd.DataFrame()
    closed_trades = len(exits)

    if closed_trades > 0:
        win_rate = float((exits["pnl"] > 0).mean())
        avg_trade_return = float(exits["return"].mean())
        model_exit_count = int(exits["exit_reason"].isin(["MODEL_FAILURE", "EDGE_DECAY", "MOMENTUM_FAILURE", "TIME_DECAY", "TIME_STOP", "TARGET_100", "PROFIT_LOCK", "EARLY_FAILURE"]).sum())
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        model_exit_count = 0

    scale_in_count = int((events["event"] == "SCALE_IN").sum()) if not events.empty else 0
    scale_out_count = int((events["event"] == "SCALE_OUT").sum()) if not events.empty else 0

    return {
        "days": int(len(d)),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "closed_trades": int(closed_trades),
        "win_rate": float(win_rate),
        "avg_trade_return": float(avg_trade_return),
        "avg_open_positions": float(d["open_positions"].mean()),
        "avg_gross_exposure": float(d["gross_exposure"].mean()),
        "model_exit_count": int(model_exit_count),
        "scale_in_count": int(scale_in_count),
        "scale_out_count": int(scale_out_count),
    }


def plot_equity_curve(daily_all: pd.DataFrame) -> None:
    if daily_all.empty or "stitched_equity" not in daily_all.columns:
        return
    plt.figure(figsize=(12, 7))
    plt.plot(daily_all["timestamp"], daily_all["stitched_equity"])
    plt.title("Walk-Forward Explosive Alpha Equity Curve - Probabilistic Risk")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PLOT, dpi=150)
    plt.close()


def main() -> None:
    print("===== WALK-FORWARD EXPLOSIVE ALPHA PROBABILISTIC RISK SIM =====")
    print(f"[load] {INPUT_PATH}")

    df = load_data(INPUT_PATH)
    df = prepare_data(df)

    # Need future 3d/5d when available for target, but don't require them for inference.
    df = df.dropna(subset=[FUTURE_RET_1D]).copy()

    trainable = df[
        (df["close"] >= MIN_PRICE)
        & ((df["tradable_symbol"] == 1) | (df["core_symbol"] == 1))
    ].copy()

    if trainable.empty:
        raise RuntimeError("No trainable rows after filters.")

    feature_cols = get_feature_columns(trainable)
    unique_dates = sorted(trainable["timestamp"].dropna().unique())
    windows = make_windows(unique_dates)

    print(f"[data] rows={len(trainable):,}")
    print(f"[data] symbols={trainable['symbol'].nunique():,}")
    print(f"[data] unique_dates={len(unique_dates):,}")
    print(f"[data] feature_cols={len(feature_cols):,}")
    print(f"[walkforward] windows={len(windows)} train_days={TRAIN_DAYS} test_days={TEST_DAYS} step_days={STEP_DAYS}")
    print("[mode] EXPLOSIVE + TRANSFORMER SEQUENCE MODEL: price/indicator windows, EV/time exits, cooldowns, correlation-aware sizing")

    if not windows:
        raise RuntimeError("Not enough dates to build walk-forward windows.")

    all_predictions = []
    all_daily = []
    all_events = []
    window_metrics = []

    for window_id, train_dates, test_dates in windows:
        train_df = trainable[trainable["timestamp"].isin(train_dates)].copy()
        test_df = trainable[trainable["timestamp"].isin(test_dates)].copy()

        y_train = train_df[TARGET_LABEL].astype(int) if TARGET_LABEL in train_df.columns else pd.Series(dtype=int)
        train_positive_count = int(y_train.sum()) if not y_train.empty else 0

        if len(train_df) < MIN_TRAIN_ROWS or train_positive_count < MIN_TRAIN_POSITIVES or test_df.empty:
            print(
                f"[wf{window_id}] skipped "
                f"train_rows={len(train_df):,} "
                f"train_positives={train_positive_count:,} "
                f"test_rows={len(test_df):,}"
            )
            continue

        y_train = train_df[TARGET_LABEL].astype(int)
        y_test = test_df[TARGET_LABEL].astype(int)

        model = build_model()

        print(
            f"[wf{window_id}] train={train_dates[0]}->{train_dates[-1]} "
            f"test={test_dates[0]}->{test_dates[-1]} "
            f"rows train={len(train_df):,} test={len(test_df):,} "
            f"pos_rate={y_train.mean():.2%}"
        )

        model.fit(train_df, feature_cols=feature_cols, target_col=TARGET_LABEL)

        scored = score_test_window(
            model=model,
            test_df=test_df,
            feature_cols=feature_cols,
            window_id=window_id,
        )

        proba = scored["explosive_alpha_probability"].values
        pred = (proba >= SOFT_MIN_PROBABILITY).astype(int)

        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = np.nan

        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)

        daily, events = simulate_positions(scored, window_id=window_id)
        metrics = calculate_window_metrics(daily, events)

        metrics.update(
            {
                "walkforward_window": window_id,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
                "probability_threshold": SOFT_MIN_PROBABILITY,
                "test_auc": auc,
                "test_precision": precision,
                "test_recall": recall,
            }
        )

        window_metrics.append(metrics)
        all_predictions.append(scored)
        all_daily.append(daily)
        all_events.append(events)

        print(
            f"[wf{window_id}] auc={auc:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"return={metrics['total_return']:.2%} sharpe={metrics['sharpe']:.2f} "
            f"maxdd={metrics['max_drawdown']:.2%} trades={metrics['closed_trades']} "
            f"scale_in={metrics['scale_in_count']} scale_out={metrics.get('scale_out_count', 0)} "
            f"model_exit={metrics.get('model_exit_count', 0)}"
        )

    if not all_predictions:
        raise RuntimeError("No walk-forward predictions were produced.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    daily_all = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    metrics_df = pd.DataFrame(window_metrics).sort_values("walkforward_window")

    if not daily_all.empty:
        daily_all = daily_all.sort_values(["timestamp", "walkforward_window"]).copy()
        daily_all["daily_return"] = daily_all.groupby("walkforward_window")["equity"].pct_change().fillna(0.0)
        daily_all["stitched_equity"] = STARTING_EQUITY * (1 + daily_all["daily_return"]).cumprod()
        daily_all["cum_return"] = daily_all["stitched_equity"] / STARTING_EQUITY - 1
        daily_all["drawdown"] = daily_all["stitched_equity"] / daily_all["stitched_equity"].cummax() - 1

    predictions.to_csv(OUT_PREDICTIONS, index=False)

    diagnostic_cols = [
        "walkforward_window", "timestamp", "symbol", "explosive_alpha_probability",
        TARGET_LABEL, "future_return_rank_3d", "future_return_rank_5d",
        "future_continuation_quality", "future_explosive_quality",
        "price_indicator_alignment_score", "indicator_transition_score",
        "context_interaction_score", "rsi_14", "macd_hist", "relative_volume_20",
        "breakout_above_prev_high", "dist_sma20", "momentum_acceleration_3v10",
    ]
    existing_diagnostic_cols = [c for c in diagnostic_cols if c in predictions.columns]
    predictions[existing_diagnostic_cols].sort_values(
        ["walkforward_window", "explosive_alpha_probability"],
        ascending=[True, False],
    ).head(200).to_csv(OUT_MODEL_DIAGNOSTICS, index=False)

    metrics_df.to_csv(OUT_WINDOW_METRICS, index=False)
    daily_all.to_csv(OUT_DAILY_EQUITY, index=False)
    events_all.to_csv(OUT_EVENTS, index=False)

    trade_log = events_all[events_all["event"].isin(["EXIT", "SCALE_OUT"])].copy() if not events_all.empty else pd.DataFrame()
    trade_log.to_csv(OUT_TRADE_LOG, index=False)

    plot_equity_curve(daily_all)

    print(f"[saved] predictions: {OUT_PREDICTIONS}")
    print(f"[saved] window metrics: {OUT_WINDOW_METRICS}")
    print(f"[saved] daily equity: {OUT_DAILY_EQUITY}")
    print(f"[saved] trade log: {OUT_TRADE_LOG}")
    print(f"[saved] events: {OUT_EVENTS}")
    print(f"[saved] equity plot: {OUT_EQUITY_PLOT}")
    print(f"[saved] model diagnostics: {OUT_MODEL_DIAGNOSTICS}")

    print("\n===== WALK-FORWARD EXPLOSIVE ALPHA SUMMARY =====")
    if not daily_all.empty:
        final_equity = daily_all["stitched_equity"].iloc[-1]
        total_return = final_equity / STARTING_EQUITY - 1
        max_dd = daily_all["drawdown"].min()
        daily_r = daily_all["daily_return"].fillna(0)
        ann_ret = (1 + total_return) ** (252 / max(len(daily_r), 1)) - 1
        ann_vol = daily_r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        exits = trade_log
        win_rate = float((exits["pnl"] > 0).mean()) if not exits.empty else 0.0
        model_exit_count = int(exits["exit_reason"].isin(["MODEL_FAILURE", "EDGE_DECAY", "MOMENTUM_FAILURE", "TIME_DECAY", "TIME_STOP", "TARGET_100", "PROFIT_LOCK", "EARLY_FAILURE"]).sum()) if not exits.empty else 0
        scale_count = int((events_all["event"] == "SCALE_IN").sum()) if not events_all.empty else 0
        scale_out_count = int((events_all["event"] == "SCALE_OUT").sum()) if not events_all.empty else 0

        print(f"final_equity=${final_equity:,.2f}")
        print(f"total_return={total_return:.2%}")
        print(f"annual_return={ann_ret:.2%}")
        print(f"annual_vol={ann_vol:.2%}")
        print(f"sharpe={sharpe:.2f}")
        print(f"max_drawdown={max_dd:.2%}")
        print(f"closed_trades={len(exits):,}")
        print(f"win_rate={win_rate:.2%}")
        print(f"model_exit_count={model_exit_count}")
        print(f"scale_in_count={scale_count}")
        print(f"scale_out_count={scale_out_count}")
    else:
        print("No simulated positions were opened.")


if __name__ == "__main__":
    main()
