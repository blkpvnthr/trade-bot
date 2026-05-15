#!/usr/bin/env python3
"""
production_regime_sizing_model.py

Production-grade supervised continuation model with:

    1. Regime-aware routing
    2. Position sizing engine
    3. No execution layer

This script DOES NOT place trades.
It only outputs:
    - predictions
    - trade actions
    - regime route
    - recommended target weights
    - 30-symbol trading universe

Expected input:
    daily_system_outputs/daily_features.csv

Automatic trading-context inputs:
    daily_system_outputs/market_context.json
    daily_system_outputs/premarket_trade_plan.json

These files are loaded automatically when present. No command-line arguments are required.

Outputs:
    daily_system_outputs/regime_predictions.csv
    daily_system_outputs/regime_actions.csv
    daily_system_outputs/regime_universe_30.csv
    daily_system_outputs/regime_universe_30.txt
    daily_system_outputs/regime_position_plan.csv
    daily_system_outputs/regime_feature_importance.csv
    daily_system_outputs/regime_threshold_table.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from market_context_builder import TODAY_TAG


# =============================================================================
# Paths
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

INPUT_PATH = PROJECT_ROOT / "daily_system_outputs" / f"{TODAY_TAG}_daily_features.csv"
OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREDICTIONS = OUT_DIR / f"{TODAY_TAG}_regime_predictions.csv"
OUT_ACTIONS = OUT_DIR / f"{TODAY_TAG}_regime_actions.csv"
OUT_UNIVERSE_CSV = OUT_DIR / f"{TODAY_TAG}_regime_universe_30.csv"
OUT_UNIVERSE_TXT = OUT_DIR / f"{TODAY_TAG}_regime_universe_30.txt"
OUT_POSITION_PLAN = OUT_DIR / f"{TODAY_TAG}_regime_position_plan.csv"
OUT_IMPORTANCE = OUT_DIR / f"{TODAY_TAG}_regime_feature_importance.csv"
OUT_THRESHOLD_TABLE = OUT_DIR / f"{TODAY_TAG}_regime_threshold_table.csv"

DEFAULT_MARKET_CONTEXT_PATH = OUT_DIR / f"{TODAY_TAG}_market_context.json"
DEFAULT_PREMARKET_TRADE_PLAN_PATH = OUT_DIR / f"{TODAY_TAG}_premarket_trade_plan.json"

# Runtime-loaded optional context. These are intentionally globals so the
# sizing/action functions can remain simple while main() controls the inputs.
ACTIVE_MARKET_CONTEXT: dict[str, Any] = {}
ACTIVE_PREMARKET_TRADE_PLAN: dict[str, Any] = {}
ACTIVE_REQUIRE_PREMARKET_VALID_SETUP = os.getenv("MODEL_REQUIRE_PREMARKET_VALID_SETUP", "1").lower() not in {"0", "false", "no"}


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.25
TRADING_UNIVERSE_SIZE = 30

FUTURE_RET_1D = "future_return_1d"
FUTURE_RET_3D = "future_return_3d"
FUTURE_RET_5D = "future_return_5d"
TARGET_LABEL = "explosive_continuation_label"

# Explosive continuation target.
MIN_NEXT_BAR_RETURN = 0.003       # +0.30%
MIN_3D_RETURN = 0.010             # +1.00%
MIN_5D_RETURN = 0.015             # +1.50%
MAX_3D_DRAWDOWN_PROXY = -0.025

MIN_TARGET_RANGE_EXPANSION = 0.015
MIN_TARGET_REL_VOLUME = 1.05

# Tradability.
MIN_PRICE = 5.00
MIN_VOLUME = 250_000
MIN_DOLLAR_VOLUME = 5_000_000
MIN_AVG_DOLLAR_VOLUME20 = 5_000_000
MIN_ATR_PCT = 0.004
MIN_DAILY_RANGE_PCT = 0.003

# Threshold search.
MIN_THRESHOLD = 0.30
MAX_THRESHOLD = 0.90
THRESHOLD_STEP = 0.01
MIN_TRADE_COUNT = 40
MIN_ENTRY_PROBABILITY = 0.55

# Position sizing / portfolio risk.
# Dynamically set this with ACCOUNT_EQUITY or TRADING_ACCOUNT_EQUITY.
# Example: ACCOUNT_EQUITY=13500 python model.py
def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


ACCOUNT_EQUITY = env_float("ACCOUNT_EQUITY", env_float("TRADING_ACCOUNT_EQUITY", 100_000.00))
MAX_TOTAL_GROSS_EXPOSURE = 0.60
MAX_SINGLE_POSITION_WEIGHT = 0.08
MIN_POSITION_WEIGHT = 0.005
BASE_RISK_PER_TRADE = 0.01
AGGRESSIVE_RISK_PER_TRADE = 0.015
DEFENSIVE_RISK_PER_TRADE = 0.005
MIN_STOP_PCT = 0.015
MAX_STOP_PCT = 0.08
ATR_STOP_MULTIPLIER = 2.0
MAX_OPEN_LONGS = 8

# Regime routing.
REGIME_SYMBOLS = {"SPY", "QQQ", "IWM", "VIXY"}
CORE_SYMBOLS = pd.read_csv("data/bullish_2026-05-13.csv")["Symbol"].to_list()
INVERSE_HEDGE_SYMBOLS = {"SQQQ", "SH", "PSQ", "DOG", "VIXY"}


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
    TARGET_LABEL,
    "continuation_probability",
    "next_bar_positive_probability",
    "next_bar_tradeable_positive_probability",
    "multi_bar_continuation_probability",
    "explosive_continuation_probability",
    "model_prediction",
    "prediction",
    "selected_threshold",
    "entry_threshold",
    "trade_action",
    "rank_for_day",
    "target_weight",
    "target_dollars",
    "recommended_shares",
}

ID_COLUMNS = {
    "timestamp",
    "symbol",
    "symbol_id",
}



# =============================================================================
# External market / premarket context helpers
# =============================================================================

def read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        path = Path(path)
        if not path.exists():
            print(f"[context] optional JSON not found: {path}")
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        print(f"[context] failed to read {path}: {exc}")
        return {}


def pct_to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def extract_index_change_pct(market_context: dict[str, Any], symbol: str) -> float:
    """Return decimal change, e.g. -0.006 for -0.60%."""
    if not isinstance(market_context, dict):
        return 0.0

    for section in ["key_etf_risk_signals", "previous_session"]:
        block = market_context.get(section, {})
        if isinstance(block, dict):
            data = block.get(symbol, {})
            if isinstance(data, dict):
                for key in ["prev_close_change_pct", "day_change_pct", "change_pct"]:
                    if key in data:
                        return pct_to_float(data.get(key), 0.0)
    return 0.0


def market_context_trade_permission(market_context: dict[str, Any]) -> dict[str, Any]:
    """
    Converts market_context.json into model-level risk controls.

    market_context_builder.py emits fields such as:
      - overall_market_bias: bullish | bearish | mixed | risk_off | risk_on
      - trade_permission: normal | cautious | halt_new_buys
      - risk_level: low | medium | high

    This helper turns that into a deterministic gate for model position sizing.
    """
    if not isinstance(market_context, dict) or not market_context:
        return {
            "has_market_context": False,
            "market_context_bias": "unknown",
            "market_context_permission": "unknown",
            "market_context_risk_level": "unknown",
            "halt_new_longs": False,
            "target_weight_multiplier": 1.0,
            "reason": "No market_context.json supplied.",
        }

    bias = str(market_context.get("overall_market_bias", "mixed")).lower().strip()
    permission = str(market_context.get("trade_permission", "cautious")).lower().strip()
    risk_level = str(market_context.get("risk_level", "medium")).lower().strip()

    spy_change = extract_index_change_pct(market_context, "SPY")
    qqq_change = extract_index_change_pct(market_context, "QQQ")
    broad_market_down = min(spy_change, qqq_change) <= -0.005

    halt_new_longs = permission in {"halt_new_buys", "halt", "risk_off"} or bias in {"risk_off", "bearish"}

    # Extra protection requested in the trading system: when broad market is down
    # more than -0.5%, do not open fresh longs unless the context classifier has
    # explicitly allowed normal/risk-on trading.
    if broad_market_down and permission != "normal" and bias not in {"bullish", "risk_on"}:
        halt_new_longs = True

    if halt_new_longs:
        multiplier = 0.0
    elif permission == "cautious" or risk_level == "high" or bias == "mixed":
        multiplier = 0.50
    elif risk_level == "medium":
        multiplier = 0.75
    else:
        multiplier = 1.0

    return {
        "has_market_context": True,
        "market_context_bias": bias,
        "market_context_permission": permission,
        "market_context_risk_level": risk_level,
        "market_context_spy_change_pct": spy_change,
        "market_context_qqq_change_pct": qqq_change,
        "halt_new_longs": halt_new_longs,
        "target_weight_multiplier": multiplier,
        "reason": str(market_context.get("summary", ""))[:700],
    }


def extract_premarket_symbol_map(premarket_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize premarket_trade_plan.json into per-symbol records."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(premarket_plan, dict) or not premarket_plan:
        return out

    for item in premarket_plan.get("daily_trade_plans", []) or []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip().replace(".", "-")
        if not symbol:
            continue

        levels = item.get("combined_support_and_resistance_levels", {}) or {}
        entries = item.get("trade_entries_and_exits", {}) or {}
        calls = entries.get("calls", {}) or {}
        puts = entries.get("puts", {}) or {}

        out[symbol] = {
            "premarket_valid_setup": bool(item.get("valid_setup", False)),
            "premarket_setup_reason": str(item.get("setup_reason", ""))[:700],
            "premarket_current_price": item.get("current_price"),
            "premarket_call_entry_price": calls.get("entry_price"),
            "premarket_call_stop_loss": calls.get("stop_loss"),
            "premarket_call_take_profit": calls.get("take_profit"),
            "premarket_put_entry_price": puts.get("entry_price"),
            "premarket_put_stop_loss": puts.get("stop_loss"),
            "premarket_put_take_profit": puts.get("take_profit"),
            "premarket_resistance_1": levels.get("resistance_1"),
            "premarket_resistance_2": levels.get("resistance_2"),
            "premarket_resistance_3": levels.get("resistance_3"),
            "premarket_support_1": levels.get("support_1"),
            "premarket_support_2": levels.get("support_2"),
            "premarket_support_3": levels.get("support_3"),
        }

    # Fallback to raw pattern plans if daily_trade_plans is absent.
    if not out:
        for item in premarket_plan.get("raw_premarket_pattern_plans", []) or []:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper().strip().replace(".", "-")
            if not symbol:
                continue
            out[symbol] = {
                "premarket_valid_setup": bool(item.get("valid", False)),
                "premarket_setup_reason": str(item.get("reason", ""))[:700],
                "premarket_current_price": item.get("premarket_close"),
                "premarket_call_entry_price": item.get("entry_price"),
                "premarket_call_stop_loss": item.get("stop_loss"),
                "premarket_call_take_profit": item.get("primary_target"),
                "premarket_resistance_1": item.get("resistance"),
                "premarket_support_1": item.get("support"),
            }

    return out


def apply_external_trading_context(
    predictions: pd.DataFrame,
    market_context: dict[str, Any],
    premarket_plan: dict[str, Any],
    require_premarket_valid_setup: bool = True,
) -> pd.DataFrame:
    """Gate/annotate model predictions with market_context.json and premarket_trade_plan.json."""
    df = predictions.copy()

    permission = market_context_trade_permission(market_context)
    premarket_by_symbol = extract_premarket_symbol_map(premarket_plan)
    valid_premarket_symbols = {s for s, rec in premarket_by_symbol.items() if rec.get("premarket_valid_setup")}

    df["has_market_context"] = permission["has_market_context"]
    df["market_context_bias"] = permission["market_context_bias"]
    df["market_context_permission"] = permission["market_context_permission"]
    df["market_context_risk_level"] = permission["market_context_risk_level"]
    df["market_context_spy_change_pct"] = permission.get("market_context_spy_change_pct", 0.0)
    df["market_context_qqq_change_pct"] = permission.get("market_context_qqq_change_pct", 0.0)
    df["market_context_halt_new_longs"] = bool(permission["halt_new_longs"])
    df["market_context_weight_multiplier"] = float(permission["target_weight_multiplier"])
    df["market_context_reason"] = permission["reason"]

    df["has_premarket_trade_plan"] = bool(premarket_by_symbol)
    df["premarket_valid_setup"] = df["symbol"].astype(str).str.upper().map(
        lambda s: bool(premarket_by_symbol.get(s, {}).get("premarket_valid_setup", False))
    )

    for col in [
        "premarket_setup_reason",
        "premarket_current_price",
        "premarket_call_entry_price",
        "premarket_call_stop_loss",
        "premarket_call_take_profit",
        "premarket_put_entry_price",
        "premarket_put_stop_loss",
        "premarket_put_take_profit",
        "premarket_resistance_1",
        "premarket_resistance_2",
        "premarket_resistance_3",
        "premarket_support_1",
        "premarket_support_2",
        "premarket_support_3",
    ]:
        df[col] = df["symbol"].astype(str).str.upper().map(lambda s, c=col: premarket_by_symbol.get(s, {}).get(c))

    long_actions = df["trade_action"].isin(["OPEN_LONG", "HOLD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION"])
    open_long = df["trade_action"].eq("OPEN_LONG")

    if permission["halt_new_longs"]:
        df.loc[open_long, "trade_action"] = "WATCH_MARKET_CONTEXT_HALTED"
        df.loc[open_long, "entry_signal"] = 0

    if require_premarket_valid_setup and premarket_by_symbol:
        lacks_valid_premarket = ~df["symbol"].astype(str).str.upper().isin(valid_premarket_symbols)
        downgrade = open_long & lacks_valid_premarket
        df.loc[downgrade, "trade_action"] = "WATCH_NO_VALID_PREMARKET_SETUP"
        df.loc[downgrade, "entry_signal"] = 0

    # Never block LIQUIDATE. Existing hold sizing can be reduced by market context,
    # but exits remain fully available to the downstream executor.
    if "target_weight" in df.columns:
        df.loc[long_actions, "target_weight"] = df.loc[long_actions, "target_weight"] * float(permission["target_weight_multiplier"])

    return df


# =============================================================================
# Data helpers
# =============================================================================

def run_regime_model_if_available() -> bool:
    if not REGIME_MODEL_SCRIPT_PATH.exists():
        print(f"[model] {REGIME_MODEL_SCRIPT_PATH} not found; skipping")
        return False

    try:
        print("[model] running regime-aware continuation model")
        code = os.system(f"{os.sys.executable} {REGIME_MODEL_SCRIPT_PATH}")
        if code != 0:
            print(f"[model] model.py exited with code={code}")
            return False

        return REGIME_POSITION_PLAN_PATH.exists()

    except Exception as exc:
        print(f"[model] failed to run model.py: {exc}")
        return False

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run your Alpaca feature-generation script first."
        )

    df = pd.read_csv(path)

    required = {"timestamp", "symbol", FUTURE_RET_1D, FUTURE_RET_3D, FUTURE_RET_5D}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def numeric_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


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

    defaults = {
        "trend_score": 0.0,
        "momentum_score": 0.0,
        "breakout_score": 0.0,
        "risk_score": numeric_series(df, "atr_pct") + numeric_series(df, "daily_range_pct"),
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

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
        0.25 * (numeric_series(df, "ret_1d") > 0).astype(int)
        + 0.20 * (numeric_series(df, "ret_5d") > 0).astype(int)
        + 0.15 * df["trend_alignment"]
        + 0.15 * df["breakout_above_prev_high"]
        + 0.15 * (df["volume_spike"] >= 1.2).astype(int)
        + 0.10 * (df["range_expansion"] >= 0.015).astype(int)
    ).clip(0, 1)

    df["late_reversal_risk"] = (
        ((numeric_series(df, "rsi", 50) > 78) & (numeric_series(df, "macd_hist") < 0))
        | ((df["close_location"] < 0.35) & (df["range_expansion"] > 0.025))
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


# =============================================================================
# Regime routing
# =============================================================================

def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a causal market regime from available proxy features.
    The regime is attached per timestamp to every symbol.

    Safe to call even if df already contains market_regime/route/regime_* columns.
    """
    df = df.copy()

    required = {"timestamp", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_market_regime: missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"])

    ts = df[["timestamp"]].drop_duplicates().sort_values("timestamp").reset_index(drop=True)

    def proxy_feature(symbol: str, source_col: str, out_col: str) -> pd.DataFrame:
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

    regime_parts = [
        proxy_feature("QQQ", "ret_5d", "regime_qqq_ret_5d"),
        proxy_feature("SPY", "ret_5d", "regime_spy_ret_5d"),
        proxy_feature("IWM", "ret_5d", "regime_iwm_ret_5d"),
        proxy_feature("VIXY", "ret_5d", "regime_vixy_ret_5d"),
        proxy_feature("QQQ", "volatility_20d", "regime_qqq_volatility_20d"),
        proxy_feature("SPY", "dist_sma20", "regime_spy_dist_sma20"),
        proxy_feature("QQQ", "dist_sma20", "regime_qqq_dist_sma20"),
    ]

    regime = ts.copy()
    for part in regime_parts:
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

    vol_q75 = regime["regime_qqq_volatility_20d"].rolling(
        window=120,
        min_periods=20,
    ).quantile(0.75)

    regime["market_volatility_high"] = (
        regime["regime_qqq_volatility_20d"] > vol_q75
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

    route_map = {
        "TREND_RISK_ON": "BREAKOUT_TREND_ROUTE",
        "VOLATILE_RISK_ON": "CONFIRMED_MOMENTUM_ROUTE",
        "CHOP_NEUTRAL": "SELECTIVE_WATCH_ROUTE",
        "DEFENSIVE_RISK_OFF": "DEFENSIVE_OR_CASH_ROUTE",
    }

    regime["route"] = regime["market_regime"].map(route_map).fillna("SELECTIVE_WATCH_ROUTE")

    regime_cols = [
        "risk_on_score",
        "market_volatility_high",
        "market_regime",
        "route",
        "regime_qqq_ret_5d",
        "regime_spy_ret_5d",
        "regime_iwm_ret_5d",
        "regime_vixy_ret_5d",
        "regime_qqq_volatility_20d",
        "regime_spy_dist_sma20",
        "regime_qqq_dist_sma20",
    ]

    # Prevent pandas from creating market_regime_x / market_regime_y.
    out = df.drop(columns=[c for c in regime_cols if c in df.columns], errors="ignore")

    out = out.merge(
        regime[["timestamp"] + regime_cols],
        on="timestamp",
        how="left",
    )

    out["market_regime"] = out["market_regime"].fillna("CHOP_NEUTRAL")
    out["route"] = out["route"].fillna("SELECTIVE_WATCH_ROUTE")
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
    explosion_setup = numeric_series(df, "explosion_setup_score")
    late_reversal = numeric_series(df, "late_reversal_risk")

    df["momentum_favorable"] = (
        (ret_1d > 0)
        & (ret_3d > -0.005)
        & (ret_5d > 0)
        & (dist_sma5 > -0.01)
        & (dist_sma20 > -0.015)
        & (macd_hist > 0)
        & (rsi >= 42)
        & (rsi <= 78)
        & (rel_vol >= 0.75)
        & (explosion_setup >= 0.45)
        & (late_reversal == 0)
    ).astype(int)

    df["momentum_steady"] = (
        (ret_3d > -0.015)
        & (ret_5d > -0.025)
        & (dist_sma20 > -0.025)
        & (macd_hist >= -0.002)
        & (late_reversal == 0)
    ).astype(int)

    df["momentum_negative"] = (
        (ret_1d < 0)
        & (
            (ret_3d < -0.0125)
            | (ret_5d < -0.02)
            | (dist_sma5 < -0.015)
            | (dist_sma20 < -0.025)
            | (macd_hist < 0)
            | (high_conviction < 0)
            | (late_reversal == 1)
        )
    ).astype(int)

    return df


def add_three_bar_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    ret = numeric_series(df, "ret_1d")

    df["bar_positive"] = (ret > 0).astype(int)
    df["bar_negative"] = (ret < 0).astype(int)

    df["last_3_positive_count"] = (
        df.groupby("symbol")["bar_positive"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["last_3_negative_count"] = (
        df.groupby("symbol")["bar_negative"]
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [FUTURE_RET_1D, FUTURE_RET_3D, FUTURE_RET_5D]:
        if col not in df.columns:
            raise ValueError(f"Missing target source column: {col}")

    f1 = numeric_series(df, FUTURE_RET_1D)
    f3 = numeric_series(df, FUTURE_RET_3D)
    f5 = numeric_series(df, FUTURE_RET_5D)

    df[TARGET_LABEL] = (
        (f1 >= MIN_NEXT_BAR_RETURN)
        & (f3 >= MIN_3D_RETURN)
        & (f5 >= MIN_5D_RETURN)
        & (f3 >= MAX_3D_DRAWDOWN_PROXY)
        & (numeric_series(df, "range_expansion") >= MIN_TARGET_RANGE_EXPANSION)
        & (numeric_series(df, "relative_volume_20", 1.0) >= MIN_TARGET_REL_VOLUME)
        & (df["tradable_symbol"] == 1)
    ).astype(int)

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_aliases(df)
    df = add_stock_specific_features(df)
    df = add_tradability_flags(df)
    df = add_market_regime(df)
    df = add_momentum_state(df)
    df = add_three_bar_confirmation(df)
    df = make_target(df)
    return df.replace([np.inf, -np.inf], np.nan)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(ID_COLUMNS) | set(LEAKAGE_COLUMNS)
    blocked |= {"open", "high", "low", "close"}
    blocked |= DROP_MACRO_DOMINANT_FEATURES

    # Non-numeric categorical regime fields are output only, not model inputs.
    blocked |= {"market_regime", "route"}

    feature_cols: list[str] = []

    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    keep = []
    for col in feature_cols:
        if df[col].nunique(dropna=True) > 1:
            keep.append(col)

    if not keep:
        raise ValueError("No valid feature columns found.")

    return keep


def time_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    dates = sorted(df["timestamp"].dropna().unique())

    if len(dates) < 10:
        raise ValueError("Not enough unique timestamps for time split.")

    split_idx = int(len(dates) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(dates) - 1))
    split_date = dates[split_idx]

    train_df = df[df["timestamp"] < split_date].copy()
    test_df = df[df["timestamp"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty frame.")

    return train_df, test_df, split_date


# =============================================================================
# Model
# =============================================================================

def build_model() -> Pipeline:
    hgb = HistGradientBoostingClassifier(
        max_iter=650,
        learning_rate=0.025,
        max_leaf_nodes=31,
        l2_regularization=0.15,
        min_samples_leaf=45,
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=450,
        max_depth=9,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("hgb", hgb),
            ("rf", rf),
        ],
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


# =============================================================================
# Evaluation
# =============================================================================

def find_best_threshold(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    rows = []

    for threshold in np.arange(
        MIN_THRESHOLD,
        MAX_THRESHOLD + THRESHOLD_STEP,
        THRESHOLD_STEP,
    ):
        pred = (proba >= threshold).astype(int)

        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        accuracy = accuracy_score(y_true, pred)
        trade_count = int(pred.sum())

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "trade_count": trade_count,
                "score": 0.85 * precision + 0.15 * recall,
            }
        )

    out = pd.DataFrame(rows)
    eligible = out[out["trade_count"] >= MIN_TRADE_COUNT].copy()

    if eligible.empty:
        eligible = out.copy()

    return eligible.sort_values(
        ["score", "precision", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    proba = model.predict_proba(X_test)[:, 1]

    threshold_table = find_best_threshold(y_test, proba)

    selected_threshold = float(threshold_table.iloc[0]["threshold"])
    entry_threshold = max(selected_threshold, MIN_ENTRY_PROBABILITY)

    pred = (proba >= selected_threshold).astype(int)

    print("\n===== THRESHOLD TABLE =====")
    print(threshold_table.head(15).to_string(index=False))

    print("\n===== MODEL EVALUATION =====")
    print(f"selected_threshold={selected_threshold:.2f}")
    print(f"entry_threshold={entry_threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, pred, zero_division=0):.4f}")

    try:
        print(f"ROC AUC:   {roc_auc_score(y_test, proba):.4f}")
    except Exception:
        print("ROC AUC:   unavailable")

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, pred, zero_division=0))

    eval_df = pd.DataFrame(
        {
            "explosive_continuation_probability": proba,
            "selected_threshold": selected_threshold,
            "entry_threshold": entry_threshold,
            "model_prediction": pred,
        },
        index=X_test.index,
    )

    return eval_df, threshold_table, selected_threshold, entry_threshold


def permutation_importance_simple(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    max_features: int = 80,
) -> pd.DataFrame:
    baseline_proba = model.predict_proba(X_test)[:, 1]

    try:
        baseline_auc = roc_auc_score(y_test, baseline_proba)
    except Exception:
        baseline_auc = 0.0

    rows = []
    rng = np.random.default_rng(RANDOM_STATE)

    for col in feature_cols[:max_features]:
        X_perm = X_test.copy()
        X_perm[col] = rng.permutation(X_perm[col].values)

        try:
            perm_proba = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y_test, perm_proba)
            importance = baseline_auc - perm_auc
        except Exception:
            importance = 0.0

        rows.append(
            {
                "feature": col,
                "importance_auc_drop": importance,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "importance_auc_drop",
        ascending=False,
    ).reset_index(drop=True)


# =============================================================================
# Actions and position sizing
# =============================================================================

def assign_trade_actions(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()

    prob = df["explosive_continuation_probability"]
    regime = df["market_regime"].fillna("CHOP_NEUTRAL")

    risk_on_route = regime.isin(["TREND_RISK_ON", "VOLATILE_RISK_ON"])
    selective_route = regime.eq("CHOP_NEUTRAL")
    defensive_route = regime.eq("DEFENSIVE_RISK_OFF")

    df["entry_signal"] = (
        (prob >= df["entry_threshold"])
        & (df["tradable_symbol"] == 1)
        & (df["momentum_favorable"] == 1)
        & (df["momentum_negative"] == 0)
        & (df["late_reversal_risk"] == 0)
        & (~defensive_route)
    ).astype(int)

    df["hold_signal"] = (
        (prob >= df["selected_threshold"])
        & (df["tradable_symbol"] == 1)
        & (df["momentum_steady"] == 1)
        & (df["momentum_negative"] == 0)
        & (df["late_reversal_risk"] == 0)
        & (~defensive_route)
    ).astype(int)

    # In neutral/chop, new marginal signals should wait for confirmation.
    df["watch_signal"] = (
        (df["entry_signal"] == 0)
        & (df["hold_signal"] == 0)
        & (df["tradable_symbol"] == 1)
        & (df["momentum_negative"] == 0)
        & (df["late_reversal_risk"] == 0)
        & (~defensive_route)
    ).astype(int)

    df["continue_hold_after_watch"] = (
        (df["last_3_positive_count"] >= 2)
        & (df["momentum_steady"] == 1)
        & (df["momentum_negative"] == 0)
        & (df["late_reversal_risk"] == 0)
        & (~defensive_route)
    ).astype(int)

    df["liquidate_signal"] = (
        (df["last_3_negative_count"] >= 2)
        | (df["momentum_negative"] == 1)
        | (df["tradable_symbol"] == 0)
        | (df["late_reversal_risk"] == 1)
        | defensive_route
    ).astype(int)

    df["trade_action"] = "NO_TRADE"

    df.loc[df["watch_signal"] == 1, "trade_action"] = "WATCH_3_BARS"
    df.loc[df["hold_signal"] == 1, "trade_action"] = "HOLD_LONG"
    df.loc[df["continue_hold_after_watch"] == 1, "trade_action"] = "CONTINUE_HOLD_WAIT_CONFIRMATION"
    df.loc[df["entry_signal"] == 1, "trade_action"] = "OPEN_LONG"
    df.loc[df["liquidate_signal"] == 1, "trade_action"] = "LIQUIDATE"

    # OPEN_LONG can override only outside defensive route.
    df.loc[(df["entry_signal"] == 1) & (~defensive_route), "trade_action"] = "OPEN_LONG"

    # In CHOP, downgrade weak opens to watch.
    weak_chop_open = (
        selective_route
        & (df["trade_action"] == "OPEN_LONG")
        & (df["explosion_setup_score"] < 0.70)
    )
    df.loc[weak_chop_open, "trade_action"] = "WATCH_3_BARS"

    return df


def get_regime_risk_parameters(regime: str) -> dict[str, float]:
    if regime == "TREND_RISK_ON":
        return {
            "gross_exposure": MAX_TOTAL_GROSS_EXPOSURE,
            "risk_per_trade": AGGRESSIVE_RISK_PER_TRADE,
            "max_single_weight": MAX_SINGLE_POSITION_WEIGHT,
            "max_open_longs": float(MAX_OPEN_LONGS),
        }

    if regime == "VOLATILE_RISK_ON":
        return {
            "gross_exposure": min(MAX_TOTAL_GROSS_EXPOSURE, 0.45),
            "risk_per_trade": BASE_RISK_PER_TRADE,
            "max_single_weight": min(MAX_SINGLE_POSITION_WEIGHT, 0.06),
            "max_open_longs": float(max(4, MAX_OPEN_LONGS - 2)),
        }

    if regime == "CHOP_NEUTRAL":
        return {
            "gross_exposure": 0.30,
            "risk_per_trade": DEFENSIVE_RISK_PER_TRADE,
            "max_single_weight": 0.04,
            "max_open_longs": 4.0,
        }

    return {
        "gross_exposure": 0.0,
        "risk_per_trade": 0.0,
        "max_single_weight": 0.0,
        "max_open_longs": 0.0,
    }


def apply_position_sizing(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()

    if df.empty:
        return df

    latest_regime = str(df["market_regime"].iloc[0]) if "market_regime" in df.columns else "CHOP_NEUTRAL"
    params = get_regime_risk_parameters(latest_regime)

    df["risk_budget_gross_exposure"] = params["gross_exposure"]
    df["risk_budget_per_trade"] = params["risk_per_trade"]
    df["max_single_position_weight"] = params["max_single_weight"]
    df["max_open_longs_allowed"] = int(params["max_open_longs"])

    if params["gross_exposure"] <= 0:
        df["target_weight"] = 0.0
        df["target_dollars"] = 0.0
        df["recommended_shares"] = 0
        df["risk_dollars"] = 0.0
        df["stop_pct"] = 0.0
        df["sizing_reason"] = "DEFENSIVE_OR_CASH_ROUTE"
        return df

    # Only OPEN/HOLD/CONFIRMED symbols receive target weight.
    eligible = df["trade_action"].isin(
        ["OPEN_LONG", "HOLD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION"]
    )

    # Risk-normalized score.
    atr_pct = numeric_series(df, "atr_pct", 0.02).clip(MIN_STOP_PCT / ATR_STOP_MULTIPLIER, MAX_STOP_PCT)
    stop_pct = (atr_pct * ATR_STOP_MULTIPLIER).clip(MIN_STOP_PCT, MAX_STOP_PCT)

    df["stop_pct"] = stop_pct
    df["risk_dollars"] = ACCOUNT_EQUITY * params["risk_per_trade"]

    raw_score = (
        0.45 * numeric_series(df, "explosive_continuation_probability")
        + 0.20 * numeric_series(df, "explosion_setup_score")
        + 0.15 * numeric_series(df, "momentum_score")
        + 0.10 * numeric_series(df, "relative_volume_20", 1.0).clip(0, 3) / 3.0
        + 0.10 * numeric_series(df, "range_expansion").clip(0, 0.08) / 0.08
        - 0.20 * numeric_series(df, "late_reversal_risk")
    ).clip(lower=0)

    df["position_score"] = raw_score

    df.loc[~eligible, "position_score"] = 0.0

    # Limit eligible OPEN_LONGs.
    open_idx = df[df["trade_action"] == "OPEN_LONG"].sort_values(
        "position_score", ascending=False
    ).index.tolist()

    for idx in open_idx[int(params["max_open_longs"]):]:
        df.loc[idx, "trade_action"] = "WATCH_3_BARS"
        df.loc[idx, "position_score"] = 0.0

    score_sum = df["position_score"].sum()

    if score_sum <= 0:
        df["target_weight"] = 0.0
    else:
        vol_adjusted_weight = df["position_score"] / score_sum * params["gross_exposure"]

        # ATR-risk cap.
        risk_weight_cap = (params["risk_per_trade"] / df["stop_pct"]).replace([np.inf, -np.inf], np.nan)
        risk_weight_cap = risk_weight_cap.fillna(params["max_single_weight"]).clip(0, params["max_single_weight"])

        df["target_weight"] = np.minimum(vol_adjusted_weight, risk_weight_cap)
        df["target_weight"] = df["target_weight"].clip(0, params["max_single_weight"])

        # Remove dust.
        df.loc[df["target_weight"] < MIN_POSITION_WEIGHT, "target_weight"] = 0.0

        # Rescale if cap math undershoots but keep gross exposure cap.
        total_weight = df["target_weight"].sum()
        if total_weight > params["gross_exposure"]:
            df["target_weight"] *= params["gross_exposure"] / total_weight

    # Optional market-context risk downshift. This is populated when main() is
    # called with --market-context-json. It can shrink or zero new long sizing,
    # while still allowing LIQUIDATE actions to pass through.
    if "market_context_weight_multiplier" in df.columns:
        multiplier = pd.to_numeric(df["market_context_weight_multiplier"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        df["target_weight"] = df["target_weight"] * multiplier

    df["target_dollars"] = ACCOUNT_EQUITY * df["target_weight"]
    df["recommended_shares"] = np.floor(
        df["target_dollars"] / numeric_series(df, "close").replace(0, np.nan)
    ).fillna(0).astype(int)

    df["sizing_reason"] = np.where(
        df["target_weight"] > 0,
        "RISK_WEIGHTED_BY_PROBABILITY_MOMENTUM_ATR",
        "NO_SIZE_ACTION_NOT_ELIGIBLE_OR_DUST",
    )

    return df


def build_latest_universe(predictions: pd.DataFrame) -> pd.DataFrame:
    latest_date = predictions["timestamp"].max()
    latest = predictions[predictions["timestamp"] == latest_date].copy()

    base = latest[
        (latest["tradable_symbol"] == 1)
        & (latest["close"] >= MIN_PRICE)
        & (latest["avg_dollar_volume20"] >= MIN_AVG_DOLLAR_VOLUME20)
        & (latest["momentum_negative"] == 0)
        & (latest["late_reversal_risk"] == 0)
    ].copy()

    if base.empty:
        base = latest[
            (latest["tradable_symbol"] == 1)
            & (latest["momentum_negative"] == 0)
        ].copy()

    if base.empty:
        base = latest.copy()

    action_priority = {
        "OPEN_LONG": 0,
        "HOLD_LONG": 1,
        "CONTINUE_HOLD_WAIT_CONFIRMATION": 2,
        "WATCH_3_BARS": 3,
        "WATCH_MARKET_CONTEXT_HALTED": 3,
        "WATCH_NO_VALID_PREMARKET_SETUP": 3,
        "NO_TRADE": 4,
        "LIQUIDATE": 9,
    }

    base["action_priority"] = base["trade_action"].map(action_priority).fillna(5)

    rel_vol = numeric_series(base, "relative_volume_20", 1.0).clip(0, 3) / 3.0
    range_exp = numeric_series(base, "range_expansion").clip(0, 0.08) / 0.08

    base["universe_score"] = (
        0.50 * base["explosive_continuation_probability"].fillna(0)
        + 0.20 * numeric_series(base, "momentum_score")
        + 0.15 * rel_vol
        + 0.15 * range_exp
        + 0.10 * numeric_series(base, "explosion_setup_score")
        + 0.05 * numeric_series(base, "risk_on_score", 0.5)
        - 0.25 * base["momentum_negative"].fillna(0)
        - 0.20 * base["late_reversal_risk"].fillna(0)
    )

    universe = (
        base.sort_values(
            [
                "action_priority",
                "universe_score",
                "explosive_continuation_probability",
            ],
            ascending=[True, False, False],
        )
        .head(TRADING_UNIVERSE_SIZE)
        .copy()
        .reset_index(drop=True)
    )

    universe["rank"] = np.arange(1, len(universe) + 1)

    universe = apply_position_sizing(universe)

    return universe


def save_outputs(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
    threshold_table: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    predictions.to_csv(OUT_PREDICTIONS, index=False)
    predictions.to_csv(OUT_ACTIONS, index=False)
    universe.to_csv(OUT_UNIVERSE_CSV, index=False)
    universe.to_csv(OUT_POSITION_PLAN, index=False)
    threshold_table.to_csv(OUT_THRESHOLD_TABLE, index=False)
    importance.to_csv(OUT_IMPORTANCE, index=False)
    print(F"[saved] predictions: {OUT_UNIVERSE_TXT}")

    with open(OUT_UNIVERSE_TXT, "w", encoding="utf-8") as f:
        f.write("TRADING_UNIVERSE = [\n")
        for sym in universe["symbol"].astype(str).tolist():
            f.write(f'    "{sym}",\n')
        f.write("]\n")

    print(f"[saved] predictions: {OUT_PREDICTIONS}")
    print(f"[saved] actions: {OUT_ACTIONS}")
    print(f"[saved] universe csv: {OUT_UNIVERSE_CSV}")
    print(f"[saved] universe txt: {OUT_UNIVERSE_TXT}")
    print(f"[saved] position plan: {OUT_POSITION_PLAN}")
    print(f"[saved] threshold table: {OUT_THRESHOLD_TABLE}")
    print(f"[saved] feature importance: {OUT_IMPORTANCE}")

    display_cols = [
        "rank",
        "timestamp",
        "symbol",
        "market_regime",
        "route",
        "close",
        "explosive_continuation_probability",
        "entry_threshold",
        "explosion_setup_score",
        "risk_on_score",
        "market_context_bias",
        "market_context_permission",
        "market_context_risk_level",
        "premarket_valid_setup",
        "premarket_call_entry_price",
        "premarket_call_stop_loss",
        "premarket_call_take_profit",
        "momentum_favorable",
        "momentum_steady",
        "momentum_negative",
        "late_reversal_risk",
        "trade_action",
        "target_weight",
        "target_dollars",
        "recommended_shares",
        "stop_pct",
        "risk_budget_per_trade",
        "sizing_reason",
    ]

    display_cols = [c for c in display_cols if c in universe.columns]

    print("\n===== REGIME-AWARE POSITION PLAN =====")
    print(universe[display_cols].to_string(index=False))

    print("\n===== PYTHON LIST FORMAT =====")
    print("TRADING_UNIVERSE = [")
    for sym in universe["symbol"].astype(str).tolist():
        print(f'    "{sym}",')
    print("]")


# =============================================================================
# Main
# =============================================================================
# =============================================================================
# Main
# =============================================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def load_runtime_trading_context() -> None:
    """
    Automatically load model trading context from standard daily_system_outputs files.

    No command-line arguments are required. Optional env toggles:
      MODEL_USE_MARKET_CONTEXT=0 disables market_context.json
      MODEL_USE_PREMARKET_PLAN=0 disables premarket_trade_plan.json
      MODEL_REQUIRE_PREMARKET_VALID_SETUP=0 allows OPEN_LONG without valid premarket setup
    """
    global ACTIVE_MARKET_CONTEXT, ACTIVE_PREMARKET_TRADE_PLAN, ACTIVE_REQUIRE_PREMARKET_VALID_SETUP

    use_market_context = env_bool("MODEL_USE_MARKET_CONTEXT", True)
    use_premarket_plan = env_bool("MODEL_USE_PREMARKET_PLAN", True)

    ACTIVE_MARKET_CONTEXT = read_optional_json(DEFAULT_MARKET_CONTEXT_PATH) if use_market_context else {}
    ACTIVE_PREMARKET_TRADE_PLAN = read_optional_json(DEFAULT_PREMARKET_TRADE_PLAN_PATH) if use_premarket_plan else {}
    ACTIVE_REQUIRE_PREMARKET_VALID_SETUP = env_bool("MODEL_REQUIRE_PREMARKET_VALID_SETUP", True)

    print(f"[context] market_context={bool(ACTIVE_MARKET_CONTEXT)} path={DEFAULT_MARKET_CONTEXT_PATH if use_market_context else 'disabled'}")
    print(f"[context] premarket_trade_plan={bool(ACTIVE_PREMARKET_TRADE_PLAN)} path={DEFAULT_PREMARKET_TRADE_PLAN_PATH if use_premarket_plan else 'disabled'}")
    print(f"[context] require_valid_premarket_setup={ACTIVE_REQUIRE_PREMARKET_VALID_SETUP}")


def main() -> None:
    load_runtime_trading_context()

    print("===== REGIME-AWARE PRODUCTION CONTINUATION MODEL =====")
    print(f"[load] {INPUT_PATH}")

    df = load_data(INPUT_PATH)
    df = prepare_data(df)
    df = df.dropna(subset=[FUTURE_RET_1D, FUTURE_RET_3D, FUTURE_RET_5D]).copy()

    trainable = df[
        (df["close"] >= MIN_PRICE)
        & ((df["tradable_symbol"] == 1) | (df["core_symbol"] == 1))
    ].copy()

    if trainable.empty:
        raise RuntimeError("No trainable rows after filters.")

    feature_cols = get_feature_columns(trainable)

    print(f"[data] rows={len(df):,}")
    print(f"[data] trainable_rows={len(trainable):,}")
    print(f"[data] symbols={df['symbol'].nunique():,}")
    print(f"[data] trainable_symbols={trainable['symbol'].nunique():,}")
    print(f"[data] feature_cols={len(feature_cols):,}")
    print(
        "[target] explosive continuation: "
        f"1d>={MIN_NEXT_BAR_RETURN:.2%}, "
        f"3d>={MIN_3D_RETURN:.2%}, "
        f"5d>={MIN_5D_RETURN:.2%}, "
        f"range>={MIN_TARGET_RANGE_EXPANSION:.2%}, "
        f"relvol>={MIN_TARGET_REL_VOLUME:.2f}"
    )
    print(f"[target] positive_rate={trainable[TARGET_LABEL].mean():.2%}")

    train_df, test_df, split_date = time_split(trainable)

    print(f"[split] train rows={len(train_df):,}")
    print(f"[split] test rows={len(test_df):,}")
    print(f"[split] split_date={split_date}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_LABEL].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_LABEL].astype(int)

    model = build_model()

    print("[train] fitting regime-aware production ensemble model...")
    model.fit(X_train, y_train)

    eval_df, threshold_table, selected_threshold, entry_threshold = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    base_cols = [
        "timestamp",
        "symbol",
        "market_regime",
        "route",
        "risk_on_score",
        "market_volatility_high",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "avg_dollar_volume20",
        "atr_pct",
        "daily_range_pct",
        "range_expansion",
        "volume_spike",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "momentum_score",
        "trend_score",
        "breakout_score",
        "risk_score",
        "high_conviction_score",
        "relative_volume_20",
        "trend_alignment",
        "breakout_above_prev_high",
        "breakdown_below_prev_low",
        "close_location",
        "explosion_setup_score",
        "late_reversal_risk",
        "tradable_liquid",
        "tradable_movement",
        "tradable_symbol",
        "momentum_favorable",
        "momentum_steady",
        "momentum_negative",
        "last_3_positive_count",
        "last_3_negative_count",
        FUTURE_RET_1D,
        FUTURE_RET_3D,
        FUTURE_RET_5D,
        TARGET_LABEL,
    ]

    existing_cols = [c for c in base_cols if c in test_df.columns]
    predictions = test_df[existing_cols].copy()

    predictions["explosive_continuation_probability"] = eval_df[
        "explosive_continuation_probability"
    ].values
    predictions["selected_threshold"] = eval_df["selected_threshold"].values
    predictions["entry_threshold"] = eval_df["entry_threshold"].values
    predictions["model_prediction"] = eval_df["model_prediction"].values

    predictions["rank_for_day"] = predictions.groupby("timestamp")[
        "explosive_continuation_probability"
    ].rank(method="first", ascending=False)

    predictions = assign_trade_actions(predictions)

    predictions = apply_external_trading_context(
        predictions=predictions,
        market_context=ACTIVE_MARKET_CONTEXT,
        premarket_plan=ACTIVE_PREMARKET_TRADE_PLAN,
        require_premarket_valid_setup=ACTIVE_REQUIRE_PREMARKET_VALID_SETUP,
    )

    predictions = predictions.sort_values(
        ["timestamp", "explosive_continuation_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)

    universe = build_latest_universe(predictions)

    print("\n[importance] calculating permutation importance...")
    importance = permutation_importance_simple(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
    )

    print("\n===== TOP FEATURES =====")
    print(importance.head(30).to_string(index=False))

    save_outputs(
        predictions=predictions,
        universe=universe,
        threshold_table=threshold_table,
        importance=importance,
    )

    print("\n===== SUMMARY =====")
    print(f"selected_threshold={selected_threshold:.2f}")
    print(f"entry_threshold={entry_threshold:.2f}")
    print(f"latest_date={predictions['timestamp'].max()}")
    print(f"universe_size={len(universe)}")
    print(f"universe_symbols={universe['symbol'].tolist()}")

    if not universe.empty and "market_regime" in universe.columns:
        print(f"latest_regime={universe['market_regime'].iloc[0]}")
        print(f"latest_route={universe['route'].iloc[0]}")
        print(f"total_target_weight={universe['target_weight'].sum():.2%}")
        print(f"total_target_dollars=${universe['target_dollars'].sum():,.2f}")


if __name__ == "__main__":
    main()