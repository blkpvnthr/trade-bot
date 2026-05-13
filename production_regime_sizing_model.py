#!/usr/bin/env python3
"""
production_regime_sizing_model.py

Regime-aware production sizing layer for run.py.

Default usage:
    python production_regime_sizing_model.py

What it does:
    1. Loads daily_system_outputs/daily_features.csv if present.
    2. If missing, attempts to run model.py to generate daily_features.csv.
    3. Loads:
        daily_system_outputs/market_context.json
        daily_system_outputs/premarket_trade_plan.json
        daily_system_outputs/intraday_structure_map.json
    4. Scores symbols for long continuation/reversal quality.
    5. Applies market-regime, premarket, liquidity, volatility, and PDT-aware sizing.
    6. Writes:
        daily_system_outputs/regime_position_plan.csv
        daily_system_outputs/regime_universe_30.csv
        daily_system_outputs/regime_actions.csv
        daily_system_outputs/regime_sizing_diagnostics.json

This script does NOT place trades.
It only creates a portfolio/sizing plan for run.py.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_context_builder import TODAY_TAG


# =============================================================================
# Paths
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = OUT_DIR / f"{TODAY_TAG}_daily_features.csv"
MODEL_SCRIPT_PATH = PROJECT_ROOT / f"{TODAY_TAG}_model.py"

MARKET_CONTEXT_PATH = OUT_DIR / f"{TODAY_TAG}_market_context.json"
PREMARKET_TRADE_PLAN_PATH = OUT_DIR / f"{TODAY_TAG}_premarket_trade_plan.json"
INTRADAY_STRUCTURE_MAP_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_structure_map.json"

REGIME_POSITION_PLAN_PATH = OUT_DIR / f"{TODAY_TAG}_regime_position_plan.csv"
REGIME_UNIVERSE_PATH = OUT_DIR / f"{TODAY_TAG}_regime_universe_30.csv"
REGIME_ACTIONS_PATH = OUT_DIR / f"{TODAY_TAG}_regime_actions.csv"
DIAGNOSTICS_PATH = OUT_DIR / f"{TODAY_TAG}_regime_sizing_diagnostics.json"

MAX_OUTPUT_SYMBOLS = int(os.getenv("REGIME_MAX_OUTPUT_SYMBOLS", "30"))


# =============================================================================
# Config
# =============================================================================

@dataclass
class RegimeSizingConfig:
    min_price: float = float(os.getenv("REGIME_MIN_PRICE", "2.0"))
    max_price: float = float(os.getenv("REGIME_MAX_PRICE", "500.0"))
    min_avg_dollar_volume: float = float(os.getenv("REGIME_MIN_AVG_DOLLAR_VOLUME", "5000000"))
    min_score: float = float(os.getenv("REGIME_MIN_SCORE", "0.50"))
    min_confidence: float = float(os.getenv("REGIME_MIN_CONFIDENCE", "0.52"))

    base_weight: float = float(os.getenv("REGIME_BASE_WEIGHT", "0.030"))
    max_weight: float = float(os.getenv("REGIME_MAX_WEIGHT", "0.100"))
    min_weight: float = float(os.getenv("REGIME_MIN_WEIGHT", "0.010"))
    max_gross_exposure: float = float(os.getenv("REGIME_MAX_GROSS_EXPOSURE", "0.72"))

    risk_off_gross_mult: float = float(os.getenv("REGIME_RISK_OFF_GROSS_MULT", "0.25"))
    cautious_gross_mult: float = float(os.getenv("REGIME_CAUTIOUS_GROSS_MULT", "0.55"))
    bullish_gross_mult: float = float(os.getenv("REGIME_BULLISH_GROSS_MULT", "1.00"))

    premarket_valid_bonus: float = float(os.getenv("REGIME_PREMARKET_VALID_BONUS", "0.08"))
    structure_long_allowed_bonus: float = float(os.getenv("REGIME_STRUCTURE_LONG_ALLOWED_BONUS", "0.05"))
    bullish_market_bonus: float = float(os.getenv("REGIME_BULLISH_MARKET_BONUS", "0.08"))
    risk_off_penalty: float = float(os.getenv("REGIME_RISK_OFF_PENALTY", "0.20"))

    max_rsi: float = float(os.getenv("REGIME_MAX_RSI", "82.0"))
    allow_cautious_buys: bool = os.getenv("REGIME_ALLOW_CAUTIOUS_BUYS", "1").lower() in {"1", "true", "yes"}
    require_positive_momentum: bool = os.getenv("REGIME_REQUIRE_POSITIVE_MOMENTUM", "0").lower() in {"1", "true", "yes"}

    # Used for action mapping.
    strong_open_threshold: float = float(os.getenv("REGIME_STRONG_OPEN_THRESHOLD", "0.62"))
    hold_wait_threshold: float = float(os.getenv("REGIME_HOLD_WAIT_THRESHOLD", "0.52"))
    liquidate_threshold: float = float(os.getenv("REGIME_LIQUIDATE_THRESHOLD", "0.35"))


CONFIG = RegimeSizingConfig()


# =============================================================================
# Helpers
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    raw = str(value or "").upper().strip().replace(".", "-")
    return re.sub(r"[^A-Z0-9\-]", "", raw)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def clip01(value: Any) -> float:
    return float(np.clip(to_float(value, 0.0), 0.0, 1.0))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    b = to_float(b, 0.0)
    if abs(b) < 1e-12:
        return default
    return to_float(a, 0.0) / b


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_model_py_if_needed() -> None:
    if INPUT_PATH.exists():
        return

    if not MODEL_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found and {MODEL_SCRIPT_PATH} does not exist. "
            "Run your feature-generation process first."
        )

    print(f"[features] {INPUT_PATH} missing; running {MODEL_SCRIPT_PATH.name}")
    proc = subprocess.run(
        [sys.executable, str(MODEL_SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=int(os.getenv("REGIME_MODEL_PY_TIMEOUT_SECONDS", "900")),
    )

    output = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
    if output:
        print(output[-4000:])

    if proc.returncode != 0:
        raise RuntimeError(f"model.py failed with code {proc.returncode}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} still not found after running model.py."
        )


def load_data(path: Path) -> pd.DataFrame:
    run_model_py_if_needed()

    print(f"[load] {path}")
    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(f"{path} is empty.")

    if "symbol" not in df.columns:
        raise ValueError(f"{path} missing required column: symbol")

    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"] != ""].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values(["symbol", "timestamp"])

        # Use latest row per symbol.
        df = df.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)

    return df


def market_context_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    bias = str(payload.get("overall_market_bias", payload.get("market_bias", "mixed"))).lower().strip()
    permission = str(payload.get("trade_permission", "cautious")).lower().strip()
    risk = str(payload.get("risk_level", "medium")).lower().strip()

    risk_off = (
        permission == "halt_new_buys"
        or bias in {"risk_off", "bearish", "defensive"}
        or risk == "high"
    )
    bullish = (
        bias in {"bullish", "risk_on", "trend_risk_on"}
        or permission in {"normal", "allow_buys", "risk_on"}
    )
    cautious = permission == "cautious" or risk in {"medium", "elevated"}

    if risk_off:
        gross_mult = CONFIG.risk_off_gross_mult
    elif cautious:
        gross_mult = CONFIG.cautious_gross_mult
    else:
        gross_mult = CONFIG.bullish_gross_mult

    return {
        "bias": bias,
        "permission": permission,
        "risk": risk,
        "risk_off": risk_off,
        "bullish": bullish,
        "cautious": cautious,
        "gross_mult": gross_mult,
        "raw": payload,
    }


def extract_premarket_symbols(payload: dict[str, Any]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    valid: set[str] = set()
    meta: dict[str, dict[str, Any]] = {}

    if not isinstance(payload, dict):
        return valid, meta

    for s in payload.get("valid_symbols", []) or []:
        sym = normalize_symbol(s)
        if sym:
            valid.add(sym)
            meta.setdefault(sym, {})["premarket_valid"] = True

    for key in ["plans", "daily_trade_plans", "raw_premarket_pattern_plans", "trade_plans"]:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                sym = normalize_symbol(item.get("symbol") or item.get("ticker"))
                if not sym:
                    continue
                meta.setdefault(sym, {}).update(item)
                if bool(item.get("valid", False)) or bool(item.get("long_allowed", False)) or item.get("status") == "valid":
                    valid.add(sym)
                    meta[sym]["premarket_valid"] = True

    return valid, meta


def extract_structure_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}

    for key in ["symbols", "structure_maps", "intraday_structure_maps", "intraday_structure_map", "maps"]:
        value = payload.get(key)
        if isinstance(value, dict):
            return {
                normalize_symbol(k): v
                for k, v in value.items()
                if normalize_symbol(k) and isinstance(v, dict)
            }

    for key in ["structures", "plans", "daily_trade_plans", "intraday_plans"]:
        value = payload.get(key)
        if isinstance(value, list):
            out = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                sym = normalize_symbol(item.get("symbol") or item.get("ticker"))
                if sym:
                    out[sym] = item
            if out:
                return out

    # Direct symbol map.
    metadata_keys = {
        "CREATED_AT_UTC",
        "SYMBOL_COUNT",
        "LONG_ALLOWED_SYMBOLS",
        "STRUCTURES",
        "METADATA",
        "CONFIG",
    }
    out = {}
    for k, v in payload.items():
        sym = normalize_symbol(k)
        if sym and sym not in metadata_keys and isinstance(v, dict):
            out[sym] = v
    return out


def structure_features_for_symbol(symbol: str, structure_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = structure_map.get(symbol, {})
    if not isinstance(item, dict):
        return {}

    trade_structure = item.get("trade_structure", {}) or {}
    signals = item.get("signals", {}) or {}

    long_allowed = bool(trade_structure.get("long_allowed", item.get("long_allowed", True)))
    short_allowed = bool(trade_structure.get("short_or_put_allowed", item.get("short_or_put_allowed", True)))

    breakout_score = to_float(
        item.get("breakout_score")
        or signals.get("breakout_score")
        or item.get("bullish_structure_score"),
        0.0,
    )
    trend_score = to_float(
        item.get("trend_score")
        or signals.get("trend_score")
        or item.get("direction_score"),
        0.0,
    )
    liquidity_score = to_float(
        item.get("liquidity_score")
        or (item.get("liquidity", {}) or {}).get("liquidity_score")
        or (item.get("intraday_liquidity", {}) or {}).get("liquidity_score"),
        0.5,
    )

    return {
        "structure_long_allowed": long_allowed,
        "structure_short_allowed": short_allowed,
        "structure_breakout_score": breakout_score,
        "structure_trend_score": trend_score,
        "structure_liquidity_score": liquidity_score,
    }


# =============================================================================
# Feature scoring
# =============================================================================

def get_col(row: pd.Series, names: list[str], default: float = 0.0) -> float:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return to_float(row.get(name), default)
    return default


def score_symbol(row: pd.Series, context: dict[str, Any], premarket_valid: bool, structure_meta: dict[str, Any]) -> dict[str, Any]:
    close = get_col(row, ["close", "price", "last_price", "current_price"], 0.0)
    avg_dollar_volume = get_col(
        row,
        ["avg_dollar_volume_20", "dollar_volume20", "avg_dollar_volume", "dollar_volume"],
        0.0,
    )
    rsi = get_col(row, ["rsi14", "rsi", "RSI"], 50.0)
    atr_pct = abs(get_col(row, ["atr_pct", "ATR_pct", "atr_percent"], 0.03))

    ret_1 = get_col(row, ["ret_1d", "ret_1", "return_1d"], 0.0)
    ret_3 = get_col(row, ["ret_3d", "ret_3", "return_3d"], 0.0)
    ret_5 = get_col(row, ["ret_5d", "ret_5", "return_5d"], 0.0)
    ret_10 = get_col(row, ["ret_10d", "ret_10", "return_10d"], 0.0)
    ret_20 = get_col(row, ["ret_20d", "ret_20", "return_20d"], 0.0)

    rel_volume = get_col(row, ["rel_volume", "relative_volume", "volume_ratio"], 1.0)
    trend_stack = get_col(row, ["trend_stack"], 0.0)
    above_sma20 = get_col(row, ["above_sma20"], 0.0)
    above_prev_high = get_col(row, ["above_prev_high", "close_above_prior_high"], 0.0)
    lower_band_reclaim = get_col(row, ["lower_band_reclaim"], 0.0)
    macd_bullish = get_col(row, ["macd_bullish"], 0.0)
    macd_hist = get_col(row, ["macd_hist"], 0.0)
    bb_pos = get_col(row, ["bb_pos"], 0.5)

    model_prob = get_col(
        row,
        ["model_probability", "buy_probability", "continuation_probability", "probability", "pred_prob"],
        0.50,
    )
    if model_prob > 1.0:
        model_prob /= 100.0
    model_prob = clip01(model_prob)

    momentum_score = (
        0.18 * np.clip((ret_1 + 0.02) / 0.06, 0, 1)
        + 0.16 * np.clip((ret_3 + 0.03) / 0.09, 0, 1)
        + 0.14 * np.clip((ret_5 + 0.04) / 0.12, 0, 1)
        + 0.10 * np.clip((ret_10 + 0.05) / 0.16, 0, 1)
        + 0.08 * np.clip((ret_20 + 0.08) / 0.24, 0, 1)
        + 0.10 * float(trend_stack > 0)
        + 0.08 * float(above_sma20 > 0)
        + 0.08 * float(macd_bullish > 0 or macd_hist > 0)
        + 0.08 * np.clip((rel_volume - 0.8) / 2.2, 0, 1)
    )

    breakout_score = (
        0.25 * float(above_prev_high > 0)
        + 0.20 * float(bb_pos > 0.80)
        + 0.15 * float(rel_volume >= 1.25)
        + 0.15 * np.clip((ret_1 + 0.01) / 0.05, 0, 1)
        + 0.15 * to_float(structure_meta.get("structure_breakout_score"), 0.0)
        + 0.10 * float(premarket_valid)
    )

    reversal_score = (
        0.30 * float(lower_band_reclaim > 0)
        + 0.20 * np.clip((50.0 - min(rsi, 50.0)) / 25.0, 0, 1)
        + 0.20 * np.clip((ret_1 + 0.025) / 0.05, 0, 1)
        + 0.15 * float(macd_hist > 0)
        + 0.15 * to_float(structure_meta.get("structure_trend_score"), 0.0)
    )

    liquidity_score = np.clip(np.log10(max(avg_dollar_volume, 1.0)) / 9.0, 0, 1)
    liquidity_score = max(float(liquidity_score), to_float(structure_meta.get("structure_liquidity_score"), 0.0))

    risk_penalty = 0.0
    if rsi >= CONFIG.max_rsi:
        risk_penalty += 0.08
    if atr_pct > 0.08:
        risk_penalty += 0.06
    if close < CONFIG.min_price or close > CONFIG.max_price:
        risk_penalty += 0.25
    if avg_dollar_volume > 0 and avg_dollar_volume < CONFIG.min_avg_dollar_volume:
        risk_penalty += 0.16

    raw_score = (
        0.32 * momentum_score
        + 0.24 * breakout_score
        + 0.12 * reversal_score
        + 0.18 * model_prob
        + 0.14 * liquidity_score
        - risk_penalty
    )

    if premarket_valid:
        raw_score += CONFIG.premarket_valid_bonus
    if structure_meta.get("structure_long_allowed", True):
        raw_score += CONFIG.structure_long_allowed_bonus
    else:
        raw_score -= 0.20

    if context["bullish"]:
        raw_score += CONFIG.bullish_market_bonus
    if context["risk_off"]:
        raw_score -= CONFIG.risk_off_penalty
    elif context["cautious"]:
        raw_score -= 0.04

    confidence = clip01(raw_score)
    setup_score = clip01(0.65 * raw_score + 0.35 * max(momentum_score, breakout_score, reversal_score))

    if breakout_score >= max(momentum_score, reversal_score):
        setup_type = "breakout"
    elif reversal_score >= momentum_score:
        setup_type = "reversal"
    else:
        setup_type = "momentum"

    return {
        "close": close,
        "avg_dollar_volume": avg_dollar_volume,
        "rsi14": rsi,
        "atr_pct": atr_pct,
        "ret_1d": ret_1,
        "ret_3d": ret_3,
        "ret_5d": ret_5,
        "momentum_score": float(np.clip(momentum_score, 0, 1)),
        "breakout_score": float(np.clip(breakout_score, 0, 1)),
        "reversal_score": float(np.clip(reversal_score, 0, 1)),
        "liquidity_score": float(np.clip(liquidity_score, 0, 1)),
        "model_probability": model_prob,
        "setup_score": setup_score,
        "trade_confidence": confidence,
        "setup_type": setup_type,
        "premarket_valid": bool(premarket_valid),
        "structure_long_allowed": bool(structure_meta.get("structure_long_allowed", True)),
        "risk_penalty": float(risk_penalty),
    }


def action_from_score(score: float, confidence: float, context: dict[str, Any], current_weight: float = 0.0) -> str:
    if context["risk_off"] and not CONFIG.allow_cautious_buys:
        return "LIQUIDATE" if current_weight > 0 else "AVOID"

    if score < CONFIG.liquidate_threshold:
        return "LIQUIDATE" if current_weight > 0 else "AVOID"

    if score >= CONFIG.strong_open_threshold and confidence >= CONFIG.min_confidence:
        return "OPEN_LONG" if current_weight <= 0 else "HOLD_LONG"

    if score >= CONFIG.hold_wait_threshold:
        return "CONTINUE_HOLD_WAIT_CONFIRMATION" if current_weight > 0 else "WAIT_CONFIRMATION"

    return "AVOID"


def target_weight_from_score(score: float, confidence: float, context: dict[str, Any], setup_type: str) -> float:
    if score < CONFIG.min_score or confidence < CONFIG.min_confidence:
        return 0.0

    gross_mult = to_float(context.get("gross_mult"), 1.0)
    base = CONFIG.base_weight

    if setup_type == "breakout":
        base *= 1.15
    elif setup_type == "reversal":
        base *= 0.75

    weight = base * (0.65 + 0.70 * confidence) * gross_mult

    return float(np.clip(weight, CONFIG.min_weight, CONFIG.max_weight))


def normalize_plan_weights(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df

    max_gross = CONFIG.max_gross_exposure * to_float(context.get("gross_mult"), 1.0)
    max_gross = float(np.clip(max_gross, 0.05, CONFIG.max_gross_exposure))

    out = df.copy()
    buy_mask = out["trade_action"].isin(["OPEN_LONG", "HOLD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION"])
    gross = float(out.loc[buy_mask, "target_weight"].sum())

    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        out.loc[buy_mask, "target_weight"] *= scale
        out["weight_scale_applied"] = scale
    else:
        out["weight_scale_applied"] = 1.0

    return out


# =============================================================================
# Main plan builder
# =============================================================================

def build_regime_plan(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market_context = market_context_summary(read_json(MARKET_CONTEXT_PATH, {}))
    premarket_payload = read_json(PREMARKET_TRADE_PLAN_PATH, {})
    structure_payload = read_json(INTRADAY_STRUCTURE_MAP_PATH, {})

    valid_premarket_symbols, premarket_meta = extract_premarket_symbols(premarket_payload)
    structure_map = extract_structure_map(structure_payload)

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue

        structure_meta = structure_features_for_symbol(symbol, structure_map)
        premarket_valid = symbol in valid_premarket_symbols

        score_payload = score_symbol(
            row=row,
            context=market_context,
            premarket_valid=premarket_valid,
            structure_meta=structure_meta,
        )

        close = score_payload["close"]
        avg_dollar_volume = score_payload["avg_dollar_volume"]

        eligible = True
        reasons: list[str] = []

        if close <= 0:
            eligible = False
            reasons.append("invalid_price")
        if close < CONFIG.min_price:
            eligible = False
            reasons.append("below_min_price")
        if close > CONFIG.max_price:
            eligible = False
            reasons.append("above_max_price")
        if avg_dollar_volume > 0 and avg_dollar_volume < CONFIG.min_avg_dollar_volume:
            eligible = False
            reasons.append("low_liquidity")
        if CONFIG.require_positive_momentum and score_payload["ret_1d"] <= 0 and score_payload["ret_3d"] <= 0:
            eligible = False
            reasons.append("no_positive_momentum")
        if not score_payload["structure_long_allowed"]:
            eligible = False
            reasons.append("structure_long_not_allowed")

        action = action_from_score(
            score_payload["setup_score"],
            score_payload["trade_confidence"],
            market_context,
            current_weight=0.0,
        )

        if not eligible and action in {"OPEN_LONG", "HOLD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION"}:
            action = "WAIT_CONFIRMATION"

        target_weight = target_weight_from_score(
            score_payload["setup_score"],
            score_payload["trade_confidence"],
            market_context,
            score_payload["setup_type"],
        )

        if action not in {"OPEN_LONG", "HOLD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION"}:
            target_weight = 0.0

        rows.append(
            {
                "symbol": symbol,
                "trade_action": action,
                "target_weight": target_weight,
                "trade_confidence": score_payload["trade_confidence"],
                "confidence": score_payload["trade_confidence"],
                "setup_score": score_payload["setup_score"],
                "setup_type": score_payload["setup_type"],
                "price": close,
                "close": close,
                "avg_dollar_volume": avg_dollar_volume,
                "rsi14": score_payload["rsi14"],
                "atr_pct": score_payload["atr_pct"],
                "ret_1d": score_payload["ret_1d"],
                "ret_3d": score_payload["ret_3d"],
                "ret_5d": score_payload["ret_5d"],
                "momentum_score": score_payload["momentum_score"],
                "breakout_score": score_payload["breakout_score"],
                "reversal_score": score_payload["reversal_score"],
                "liquidity_score": score_payload["liquidity_score"],
                "model_probability": score_payload["model_probability"],
                "premarket_valid": score_payload["premarket_valid"],
                "structure_long_allowed": score_payload["structure_long_allowed"],
                "eligible": eligible,
                "eligibility_reasons": ",".join(reasons),
                "market_bias": market_context["bias"],
                "market_permission": market_context["permission"],
                "market_risk": market_context["risk"],
                "created_at_utc": now_utc(),
            }
        )

    plan = pd.DataFrame(rows)

    if plan.empty:
        return plan, pd.DataFrame(), pd.DataFrame(), {
            "created_at_utc": now_utc(),
            "error": "empty_plan",
            "market_context": market_context,
        }

    plan = plan.sort_values(["target_weight", "setup_score", "trade_confidence"], ascending=False).reset_index(drop=True)

    # Keep top N actionable names, but include LIQUIDATE rows if any exist.
    actionable = plan[plan["trade_action"].isin(["OPEN_LONG", "HOLD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION"])].head(MAX_OUTPUT_SYMBOLS)
    non_actionable_exits = plan[plan["trade_action"].isin(["LIQUIDATE"])]
    plan_out = pd.concat([actionable, non_actionable_exits], ignore_index=True)
    plan_out = normalize_plan_weights(plan_out, market_context)

    universe = actionable.head(MAX_OUTPUT_SYMBOLS).copy()
    actions = plan.copy()

    diagnostics = {
        "created_at_utc": now_utc(),
        "input_rows": int(len(df)),
        "plan_rows": int(len(plan_out)),
        "universe_rows": int(len(universe)),
        "actions_rows": int(len(actions)),
        "market_context": {
            k: v for k, v in market_context.items() if k != "raw"
        },
        "config": asdict(CONFIG),
        "premarket_valid_symbol_count": len(valid_premarket_symbols),
        "structure_symbol_count": len(structure_map),
        "top_symbols": universe["symbol"].head(20).tolist() if not universe.empty else [],
    }

    return plan_out, universe, actions, diagnostics


def main() -> dict[str, Any]:
    print("===== REGIME-AWARE PRODUCTION SIZING MODEL =====")

    df = load_data(INPUT_PATH)
    plan, universe, actions, diagnostics = build_regime_plan(df)

    plan.to_csv(REGIME_POSITION_PLAN_PATH, index=False)
    universe.to_csv(REGIME_UNIVERSE_PATH, index=False)
    actions.to_csv(REGIME_ACTIONS_PATH, index=False)
    write_json(DIAGNOSTICS_PATH, diagnostics)

    print(f"[saved] {REGIME_POSITION_PLAN_PATH} rows={len(plan)}")
    print(f"[saved] {REGIME_UNIVERSE_PATH} rows={len(universe)}")
    print(f"[saved] {REGIME_ACTIONS_PATH} rows={len(actions)}")
    print(f"[saved] {DIAGNOSTICS_PATH}")

    if not universe.empty:
        show_cols = ["symbol", "trade_action", "target_weight", "setup_score", "trade_confidence", "setup_type"]
        print("[top universe]")
        print(universe[[c for c in show_cols if c in universe.columns]].head(15).to_string(index=False))
    else:
        print("[top universe] empty")

    return {
        "regime_position_plan": str(REGIME_POSITION_PLAN_PATH),
        "regime_universe": str(REGIME_UNIVERSE_PATH),
        "regime_actions": str(REGIME_ACTIONS_PATH),
        "diagnostics": str(DIAGNOSTICS_PATH),
        "plan_rows": len(plan),
        "universe_rows": len(universe),
    }


if __name__ == "__main__":
    main()
