#!/usr/bin/env python3
"""
run.py

Clean 13K-account Alpaca paper executor with built-in momentum/reversal scanner.

Default behavior:
    python run.py
Reload systemd:
    systemctl daemon-reload

Enable auto-start on reboot:
    systemctl enable trade-bot

Start it:
    systemctl start trade-bot

Check status:
    systemctl status trade-bot
    
Restart after code changes:
    systemctl restart trade-bot
    
Live logs:
    tail -f /root/trade-bot/logs/run.log
    journalctl -u trade-bot -f

What it does:
    - loads Alpaca credentials from .env
    - refreshes llm_sentiment_universe_selector.py at the start of the trading day
    - starts the session-planning pipeline at/after 4:00 AM ET
    - builds market_context.json from market_context_builder.py
    - builds premarket_trade_plan.json from premarket_trade_planner.py
    - builds regime_position_plan.csv from production_regime_sizing_model.py
    - validates/loads the saved intraday RL timing model from rl_intraday_structure_trainer.py
    - gates entries/exits with the RL timing policy before Alpaca execution
    - scans the LLM-selected universe using Alpaca daily/intraday bars when regime plan is unavailable
    - uses market context + premarket trade plan as buy-side execution gates
    - writes rebalance/exit CSVs into system-data/
    - checks Alpaca market clock
    - applies small-account risk controls
    - sells before buys
    - enforces cash buffer, gross exposure, position size, order size, cooldown
    - applies daily loss kill switch and absolute equity floor
    - avoids duplicate open orders
    - logs plans, results, account snapshots, journal rows, and JSONL events

Designed for roughly a $13,000 paper account.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import inspect
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from io import StringIO
from pathlib import Path
from typing import Any, Optional
import subprocess
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    import pywt
except Exception:
    pywt = None
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

from execution_alpha_gate import add_execution_alpha_scores

try:
    from system_self_awareness import (
        run_self_assessment,
        load_self_assessment,
        self_assessment_size_multiplier,
        self_assessment_mode,
        self_assessment_raises_confidence,
    )
    _SA_AVAILABLE = True
except Exception:
    _SA_AVAILABLE = False
    def run_self_assessment(*_a, **_kw) -> dict:  # type: ignore[misc]
        return {}
    def load_self_assessment(*_a, **_kw) -> dict:  # type: ignore[misc]
        return {"execution_mode_recommendation": "NORMAL", "recommended_size_multiplier": 1.0,
                "disable_options": False, "disable_pyramiding": False,
                "raise_confidence_threshold_by": 0.0, "reasons": ["sa_unavailable"]}
    def self_assessment_size_multiplier(*_a, **_kw) -> float:  # type: ignore[misc]
        return 1.0
    def self_assessment_mode(*_a, **_kw) -> str:  # type: ignore[misc]
        return "NORMAL"
    def self_assessment_raises_confidence(*_a, **_kw) -> float:  # type: ignore[misc]
        return 0.0

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

try:
    from alpaca.trading.requests import TrailingStopOrderRequest
except Exception:
    TrailingStopOrderRequest = None

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception:
    DataFeed = None
    StockHistoricalDataClient = None
    OptionHistoricalDataClient = None
    StockBarsRequest = None
    OptionChainRequest = None
    TimeFrame = None


# ======================================================================# Paths / environment
# ======================================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", TODAY_TAG))

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

USE_TRADINGVIEW_SCREENER = os.getenv("USE_TRADINGVIEW_SCREENER", "1").lower() in {"1", "true", "yes"}

TRADINGVIEW_REQUIRE_RVOL = float(
    os.getenv("TRADINGVIEW_REQUIRE_RVOL", "1.2")
)

TRADINGVIEW_MAX_PLANNER_SYMBOLS = int(
    os.getenv("TRADINGVIEW_MAX_PLANNER_SYMBOLS", "20")
)
MAX_PLAN_AGE_MINUTES = int(os.getenv("MAX_PLAN_AGE_MINUTES", "15"))
REQUIRE_FRESH_PREMARKET_PLAN_FOR_BUYS = os.getenv("REQUIRE_FRESH_PREMARKET_PLAN_FOR_BUYS", "1").lower() in {"1", "true", "yes"}
REQUIRE_STRUCTURE_LONG_ALLOWED_FOR_BUYS = os.getenv("REQUIRE_STRUCTURE_LONG_ALLOWED_FOR_BUYS", "1").lower() in {"1", "true", "yes"}
SAFE_MODE_ON_PLANNER_FAILURE = os.getenv("SAFE_MODE_ON_PLANNER_FAILURE", "1").lower() in {"1", "true", "yes"}
SAFE_MODE_BUY_SIZE_MULTIPLIER = float(os.getenv("SAFE_MODE_BUY_SIZE_MULTIPLIER", "0.50"))
CAUTIOUS_MARKET_BUY_SIZE_MULTIPLIER = float(os.getenv("CAUTIOUS_MARKET_BUY_SIZE_MULTIPLIER", "0.50"))
CAUTIOUS_MARKET_CONFIDENCE_ADDON = float(os.getenv("CAUTIOUS_MARKET_CONFIDENCE_ADDON", "0.10"))
BEARISH_MARKET_DISABLE_NEW_LONGS = os.getenv("BEARISH_MARKET_DISABLE_NEW_LONGS", "1").lower() in {"1", "true", "yes"}
BEARISH_MARKET_MIN_RL_CONFIDENCE = float(os.getenv("BEARISH_MARKET_MIN_RL_CONFIDENCE", "0.85"))
MIXED_MARKET_BUY_SIZE_MULTIPLIER = float(os.getenv("MIXED_MARKET_BUY_SIZE_MULTIPLIER", "0.80"))

# =============================================================================
# FORECAST ENSEMBLE LAYER: ARIMA + LSTM + STACKED-LSTM + ATTENTION-LSTM
# =============================================================================
#
# This layer implements the paper's model stack as a production-friendly
# forecasting ensemble inside run.py:
#   - ARIMA-style baseline from recent adjusted/close prices.
#   - LSTM / Stacked-LSTM / Attention-LSTM signal ingestion from an optional
#     external forecast_engine.py.
#   - Local no-argument fallback when trained deep forecast files are missing.
#   - Final ensemble fields used by direction gating, Kelly sizing, and logs.
#
# No command-line arguments are required.

FORECAST_ENSEMBLE_ENABLED = os.getenv("FORECAST_ENSEMBLE_ENABLED", "1").lower() in {"1", "true", "yes"}
FORECAST_ENSEMBLE_REFRESH_ON_START = os.getenv("FORECAST_ENSEMBLE_REFRESH_ON_START", "0").lower() in {"1", "true", "yes"}
FORECAST_ENSEMBLE_SCRIPT_PATH = PROJECT_ROOT / os.getenv("FORECAST_ENSEMBLE_SCRIPT", "forecast_engine.py")

FORECAST_ENSEMBLE_PATH = OUT_DIR / f"{DATA_TAG}_forecast_ensemble_signals.json"
FORECAST_ENSEMBLE_CSV_PATH = OUT_DIR / f"{DATA_TAG}_forecast_ensemble_signals.csv"
FORECAST_ENSEMBLE_EVENTS_PATH = LOG_DIR / f"{DATA_TAG}_forecast_ensemble_events.jsonl"

FORECAST_LOOKBACK_BARS = int(os.getenv("FORECAST_LOOKBACK_BARS", "120"))
FORECAST_MIN_HISTORY_BARS = int(os.getenv("FORECAST_MIN_HISTORY_BARS", "40"))
FORECAST_BUY_THRESHOLD = float(os.getenv("FORECAST_BUY_THRESHOLD", "0.62"))
FORECAST_STRONG_BUY_THRESHOLD = float(os.getenv("FORECAST_STRONG_BUY_THRESHOLD", "0.72"))
FORECAST_MAX_DRAWDOWN_PROB_FOR_BUY = float(os.getenv("FORECAST_MAX_DRAWDOWN_PROB_FOR_BUY", "0.42"))
FORECAST_MIN_EXPECTED_RETURN_FOR_BUY = float(os.getenv("FORECAST_MIN_EXPECTED_RETURN_FOR_BUY", "0.0015"))
FORECAST_BLOCK_BUYS_WHEN_MISSING = os.getenv("FORECAST_BLOCK_BUYS_WHEN_MISSING", "0").lower() in {"1", "true", "yes"}
FORECAST_SIGNAL_MAX_AGE_SECONDS = int(os.getenv("FORECAST_SIGNAL_MAX_AGE_SECONDS", "900"))

FORECAST_WEIGHTS = {
    "arima": float(os.getenv("FORECAST_WEIGHT_ARIMA", "0.10")),
    "lstm": float(os.getenv("FORECAST_WEIGHT_LSTM", "0.20")),
    "stacked_lstm": float(os.getenv("FORECAST_WEIGHT_STACKED_LSTM", "0.15")),
    "attention_lstm": float(os.getenv("FORECAST_WEIGHT_ATTENTION_LSTM", "0.35")),
    "deeplob": float(os.getenv("FORECAST_WEIGHT_DEEPLOB", "0.10")),
    "rl": float(os.getenv("FORECAST_WEIGHT_RL", "0.10")),
}

# forecast_engine.py config — passed as env overrides when FORECAST_ENSEMBLE_REFRESH_ON_START=1.
FORECAST_ENGINE_ENABLED = os.getenv("FORECAST_ENGINE_ENABLED", "1")
FORECAST_ENGINE_TRAIN_ON_START = os.getenv("FORECAST_ENGINE_TRAIN_ON_START", "1")
FORECAST_ENGINE_MIN_ROWS = os.getenv("FORECAST_ENGINE_MIN_ROWS", "300")
FORECAST_ENGINE_EPOCHS = os.getenv("FORECAST_ENGINE_EPOCHS", "20")
FORECAST_ENGINE_LOOKBACK = os.getenv("FORECAST_ENGINE_LOOKBACK", "60")
FORECAST_ENGINE_MODEL_DIR = os.getenv("FORECAST_ENGINE_MODEL_DIR", str(PROJECT_ROOT / "models" / "forecast"))
FORECAST_ENGINE_USE_GPU = os.getenv("FORECAST_ENGINE_USE_GPU", "1")


# DWT preprocessing for forecasting.
# PyWavelets is optional. If unavailable, execution falls back to raw returns.
DWT_FORECAST_ENABLED = os.getenv("DWT_FORECAST_ENABLED", "1").lower() in {"1", "true", "yes"}
DWT_WAVELET = os.getenv("DWT_WAVELET", "db4")
DWT_LEVEL = int(os.getenv("DWT_LEVEL", "3"))
DWT_DENOISE_MODE = os.getenv("DWT_DENOISE_MODE", "soft")
DWT_THRESHOLD_SCALE = float(os.getenv("DWT_THRESHOLD_SCALE", "0.50"))


def _dwt_denoise_series(values: Any) -> np.ndarray:
    arr = pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).to_numpy()

    if not DWT_FORECAST_ENABLED or pywt is None or len(arr) < 16:
        return arr

    try:
        wavelet = pywt.Wavelet(DWT_WAVELET)
        max_level = pywt.dwt_max_level(data_len=len(arr), filter_len=wavelet.dec_len)
        level = max(1, min(DWT_LEVEL, max_level))
        coeffs = pywt.wavedec(arr, DWT_WAVELET, mode="symmetric", level=level)

        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if len(detail) else 0.0
        threshold = DWT_THRESHOLD_SCALE * sigma * np.sqrt(2.0 * np.log(max(len(arr), 2)))

        denoised_coeffs = [coeffs[0]]
        for coeff in coeffs[1:]:
            denoised_coeffs.append(pywt.threshold(coeff, threshold, mode=DWT_DENOISE_MODE))

        rec = pywt.waverec(denoised_coeffs, DWT_WAVELET, mode="symmetric")[: len(arr)]
        if len(rec) < len(arr):
            rec = np.pad(rec, (0, len(arr) - len(rec)), mode="edge")
        return rec.astype(float)
    except Exception as exc:
        try:
            _forecast_write_event("dwt_denoise_failed", {"error": str(exc), "wavelet": DWT_WAVELET, "level": DWT_LEVEL})
        except Exception:
            pass
        return arr


def _dwt_feature_pack(close: Any) -> dict[str, Any]:
    close_clean = pd.Series(close).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    if close_clean.empty:
        return {
            "dwt_enabled": bool(DWT_FORECAST_ENABLED and pywt is not None),
            "dwt_wavelet": DWT_WAVELET,
            "dwt_level": DWT_LEVEL,
            "dwt_trend_score": 0.50,
            "dwt_detail_energy_ratio": 0.0,
            "dwt_denoised_return_5": 0.0,
            "dwt_denoised_return_20": 0.0,
            "denoised_close": close_clean,
        }

    denoised = pd.Series(_dwt_denoise_series(close_clean), index=close_clean.index)
    den_ret = denoised.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    ret_5 = float(denoised.iloc[-1] / denoised.iloc[-min(len(denoised), 6)] - 1.0) if len(denoised) >= 6 else 0.0
    ret_20 = float(denoised.iloc[-1] / denoised.iloc[-min(len(denoised), 21)] - 1.0) if len(denoised) >= 21 else ret_5

    raw_ret = close_clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    raw_energy = float(np.sum(np.square(raw_ret.tail(80)))) if len(raw_ret) else 0.0
    den_energy = float(np.sum(np.square(den_ret.tail(80)))) if len(den_ret) else 0.0
    detail_energy_ratio = max(0.0, raw_energy - den_energy) / max(raw_energy, 1e-9)

    vol = float(den_ret.tail(20).std()) if len(den_ret) >= 3 else 1e-6
    vol = max(vol, 1e-6)
    trend_raw = (0.65 * ret_5 + 0.35 * ret_20) / vol
    trend_score = _forecast_sigmoid(trend_raw) if "_forecast_sigmoid" in globals() else 0.50

    return {
        "dwt_enabled": bool(DWT_FORECAST_ENABLED and pywt is not None),
        "dwt_wavelet": DWT_WAVELET,
        "dwt_level": DWT_LEVEL,
        "dwt_trend_score": float(trend_score),
        "dwt_detail_energy_ratio": float(detail_energy_ratio),
        "dwt_denoised_return_5": float(ret_5),
        "dwt_denoised_return_20": float(ret_20),
        "denoised_close": denoised,
    }




def _forecast_now_utc_iso() -> str:
    try:
        return now_utc().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _forecast_clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _forecast_sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, float(value)))
    return float(1.0 / (1.0 + np.exp(-value)))


def _forecast_write_event(event: str, payload: dict[str, Any]) -> None:
    try:
        row = {
            "timestamp_utc": _forecast_now_utc_iso(),
            "event": event,
            "payload": payload,
        }
        with FORECAST_ENSEMBLE_EVENTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=json_default) + "\n")
    except Exception:
        pass


def refresh_forecast_ensemble_if_available() -> None:
    """
    Optionally run an external model engine.

    Expected external output:
        system-data/YYYYMMDD/YYYYMMDD_forecast_ensemble_signals.json

    Expected JSON:
        {
          "created_at_utc": "...",
          "signals": [
            {
              "symbol": "NVDA",
              "ensemble_score": 0.74,
              "up_probability": 0.71,
              "expected_return": 0.006,
              "drawdown_probability": 0.25,
              "arima_score": 0.58,
              "lstm_score": 0.64,
              "stacked_lstm_score": 0.61,
              "attention_lstm_score": 0.77
            }
          ]
        }
    """
    if not FORECAST_ENSEMBLE_ENABLED or not FORECAST_ENSEMBLE_REFRESH_ON_START:
        return

    if not FORECAST_ENSEMBLE_SCRIPT_PATH.exists():
        print(f"[forecast] external script missing; using local fallback: {FORECAST_ENSEMBLE_SCRIPT_PATH}")
        return

    ok, output = run_python_script(
        FORECAST_ENSEMBLE_SCRIPT_PATH,
        args=[],
        timeout_seconds=int(os.getenv("FORECAST_ENSEMBLE_TIMEOUT_SECONDS", "300")),
        env_overrides={
            "RUN_DATA_DATE": DATA_TAG,
            "FORECAST_ENSEMBLE_OUTPUT": str(FORECAST_ENSEMBLE_PATH),
            "PSO_LSTM_BEST_PARAMS_PATH": str(PSO_LSTM_BEST_PARAMS_PATH),
            "DWT_FORECAST_ENABLED": "1" if DWT_FORECAST_ENABLED else "0",
            "DWT_WAVELET": DWT_WAVELET,
            "DWT_LEVEL": str(DWT_LEVEL),
            # forecast_engine.py config forwarded from this process's env or defaults above.
            "FORECAST_ENGINE_ENABLED": FORECAST_ENGINE_ENABLED,
            "FORECAST_ENGINE_TRAIN_ON_START": FORECAST_ENGINE_TRAIN_ON_START,
            "FORECAST_ENGINE_MIN_ROWS": FORECAST_ENGINE_MIN_ROWS,
            "FORECAST_ENGINE_EPOCHS": FORECAST_ENGINE_EPOCHS,
            "FORECAST_ENGINE_LOOKBACK": FORECAST_ENGINE_LOOKBACK,
            "FORECAST_ENGINE_MODEL_DIR": FORECAST_ENGINE_MODEL_DIR,
            "FORECAST_ENGINE_USE_GPU": FORECAST_ENGINE_USE_GPU,
            **{f"PSO_LSTM_{k.upper()}": str(v) for k, v in load_pso_lstm_best_params().items() if v is not None},
        },
    )
    print(f"[forecast] external refresh ok={ok}")
    if output:
        print(output[-4000:])
    _forecast_write_event("forecast_external_refresh", {"ok": ok, "output_tail": output[-4000:]})


def _forecast_score_from_returns(rets: pd.Series) -> dict[str, float]:
    """
    Local fallback forecast when trained ARIMA/LSTM files are not present.

    It approximates:
      - ARIMA drift using short/medium return autocorrelation and mean drift.
      - LSTM score using nonlinear momentum/volatility features.
      - Stacked-LSTM score using longer context / trend persistence.
      - Attention-LSTM score by emphasizing recent high-information bars.
    """
    r = pd.to_numeric(rets, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 10:
        return {
            "arima_score": 0.50,
            "lstm_score": 0.50,
            "stacked_lstm_score": 0.50,
            "attention_lstm_score": 0.50,
            "expected_return": 0.0,
            "drawdown_probability": 0.50,
            "forecast_source": "fallback_insufficient_history",
        }

    recent = r.tail(min(len(r), FORECAST_LOOKBACK_BARS))
    drift_5 = float(recent.tail(5).mean()) if len(recent) >= 5 else float(recent.mean())
    drift_20 = float(recent.tail(20).mean()) if len(recent) >= 20 else float(recent.mean())
    vol_20 = float(recent.tail(20).std()) if len(recent) >= 20 else float(recent.std())
    vol_20 = max(vol_20, 1e-6)

    autocorr_1 = float(recent.autocorr(lag=1)) if len(recent) > 5 else 0.0
    if not np.isfinite(autocorr_1):
        autocorr_1 = 0.0

    # ARIMA-style baseline: drift + autocorrelation persistence.
    arima_raw = (drift_20 / vol_20) + 0.35 * autocorr_1
    arima_score = _forecast_sigmoid(arima_raw)

    # Single LSTM approximation: short-term momentum normalized by volatility.
    lstm_raw = (0.65 * drift_5 + 0.35 * drift_20) / vol_20
    lstm_score = _forecast_sigmoid(lstm_raw)

    # Stacked-LSTM approximation: persistent multi-window structure.
    drift_60 = float(recent.tail(60).mean()) if len(recent) >= 60 else float(recent.mean())
    stacked_raw = (0.25 * drift_5 + 0.35 * drift_20 + 0.40 * drift_60) / vol_20
    stacked_lstm_score = _forecast_sigmoid(stacked_raw)

    # Attention-LSTM approximation: weight recent bars by absolute information.
    abs_r = recent.abs()
    weights = abs_r / max(float(abs_r.sum()), 1e-9)
    attention_return = float((recent * weights).sum())
    attention_raw = attention_return / vol_20
    attention_lstm_score = _forecast_sigmoid(attention_raw)

    expected_return = float(0.40 * drift_5 + 0.35 * drift_20 + 0.25 * attention_return)
    downside_hits = float((recent.tail(20) < -vol_20).mean()) if len(recent) >= 20 else float((recent < -vol_20).mean())
    drawdown_probability = _forecast_clip01(0.50 * downside_hits + 0.50 * (1.0 - attention_lstm_score))

    return {
        "arima_score": float(arima_score),
        "lstm_score": float(lstm_score),
        "stacked_lstm_score": float(stacked_lstm_score),
        "attention_lstm_score": float(attention_lstm_score),
        "expected_return": float(expected_return),
        "drawdown_probability": float(drawdown_probability),
        "forecast_source": "local_fallback_arima_lstm_attention_proxy",
    }


def _forecast_load_price_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load local price history already produced by the system.

    Preferred:
      - intraday 1m bars
      - daily features
      - daily_system_outputs/daily_features.csv
    """
    out: dict[str, pd.DataFrame] = {}
    symbols_set = {normalize_symbol(s) for s in symbols if normalize_symbol(s)}

    paths = [
        INTRADAY_BARS_1M_PATH,
        OUT_DIR / f"{DATA_TAG}_daily_features.csv",
        PROJECT_ROOT / "daily_system_outputs" / "daily_features.csv",
    ]

    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "symbol" not in df.columns:
                continue
            close_col = "close" if "close" in df.columns else "Close" if "Close" in df.columns else None
            if close_col is None:
                continue

            df = df.copy()
            df["symbol"] = df["symbol"].map(normalize_symbol)
            df = df[df["symbol"].isin(symbols_set)]
            if df.empty:
                continue

            time_col = "timestamp" if "timestamp" in df.columns else "date" if "date" in df.columns else None
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
                df = df.sort_values(["symbol", time_col])
            else:
                df = df.sort_values(["symbol"])

            for symbol, g in df.groupby("symbol", sort=False):
                x = g.copy()
                x["close"] = pd.to_numeric(x[close_col], errors="coerce")
                x = x.dropna(subset=["close"])
                if len(x) >= FORECAST_MIN_HISTORY_BARS and symbol not in out:
                    out[symbol] = x
        except Exception as exc:
            _forecast_write_event("forecast_price_history_load_error", {"path": str(path), "error": str(exc)})

    return out


def _forecast_load_external_signals() -> dict[str, dict[str, Any]]:
    if not FORECAST_ENSEMBLE_PATH.exists():
        return {}

    try:
        age_seconds = time.time() - FORECAST_ENSEMBLE_PATH.stat().st_mtime
        if age_seconds > FORECAST_SIGNAL_MAX_AGE_SECONDS:
            print(f"[forecast] stale external signals age_seconds={age_seconds:.1f}; rebuilding fallback")
            return {}

        payload = json.loads(FORECAST_ENSEMBLE_PATH.read_text(encoding="utf-8"))
        signals = payload.get("signals", [])
        out = {}
        for row in signals:
            symbol = normalize_symbol(row.get("symbol"))
            if symbol:
                out[symbol] = row
        return out
    except Exception as exc:
        _forecast_write_event("forecast_external_signal_load_error", {"error": str(exc)})
        return {}


def _forecast_merge_deeplob_signal(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    deeplob_path = OUT_DIR / f"{DATA_TAG}_deeplob_live_signals.json"
    if not deeplob_path.exists():
        payload.setdefault("deeplob_score", 0.50)
        return payload

    try:
        data = json.loads(deeplob_path.read_text(encoding="utf-8"))
        for row in data.get("signals", []):
            if normalize_symbol(row.get("symbol")) == symbol:
                payload["deeplob_score"] = float(row.get("prob_up", row.get("confidence", 0.50)) or 0.50)
                payload["deeplob_action"] = row.get("action")
                return payload
    except Exception:
        pass

    payload.setdefault("deeplob_score", 0.50)
    return payload


def _forecast_merge_rl_signal(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    # Prefer latest RL diagnostics if present.
    if RL_DIAGNOSTICS_PATH.exists():
        try:
            df = pd.read_csv(RL_DIAGNOSTICS_PATH)
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].map(normalize_symbol)
                match = df[df["symbol"] == symbol]
                if not match.empty:
                    row = match.iloc[-1]
                    for col in ["rl_confidence", "confidence", "enter_long_confidence", "action_confidence"]:
                        if col in row.index:
                            payload["rl_score"] = float(row.get(col) or 0.50)
                            return payload
        except Exception:
            pass

    payload.setdefault("rl_score", 0.50)
    return payload


def build_forecast_ensemble_for_symbols(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Build/load forecasts for requested symbols.

    Returns dict keyed by symbol with:
      ensemble_score, up_probability, expected_return, drawdown_probability,
      arima_score, lstm_score, stacked_lstm_score, attention_lstm_score.
    """
    if not FORECAST_ENSEMBLE_ENABLED:
        return {}

    symbols = sorted({normalize_symbol(s) for s in symbols if normalize_symbol(s)})
    if not symbols:
        return {}

    external = _forecast_load_external_signals()
    price_history = _forecast_load_price_history(symbols)

    results: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        if symbol in external:
            row = dict(external[symbol])
        else:
            hist = price_history.get(symbol)
            if hist is None or hist.empty or len(hist) < FORECAST_MIN_HISTORY_BARS:
                if FORECAST_BLOCK_BUYS_WHEN_MISSING:
                    row = {
                        "symbol": symbol,
                        "ensemble_score": 0.0,
                        "up_probability": 0.0,
                        "expected_return": 0.0,
                        "drawdown_probability": 1.0,
                        "forecast_source": "missing_history_blocked",
                    }
                else:
                    row = {
                        "symbol": symbol,
                        "ensemble_score": 0.50,
                        "up_probability": 0.50,
                        "expected_return": 0.0,
                        "drawdown_probability": 0.50,
                        "forecast_source": "missing_history_neutral",
                    }
            else:
                close = pd.to_numeric(hist["close"], errors="coerce").dropna()
                dwt_features = _dwt_feature_pack(close)
                denoised_close = dwt_features.pop("denoised_close")
                rets = pd.Series(denoised_close).pct_change().dropna()

                row = {"symbol": symbol, **_forecast_score_from_returns(rets), **dwt_features}

                dwt_score = float(row.get("dwt_trend_score", 0.50) or 0.50)
                row["lstm_score"] = float(0.70 * row.get("lstm_score", 0.50) + 0.30 * dwt_score)
                row["stacked_lstm_score"] = float(0.75 * row.get("stacked_lstm_score", 0.50) + 0.25 * dwt_score)
                row["attention_lstm_score"] = float(0.65 * row.get("attention_lstm_score", 0.50) + 0.35 * dwt_score)
                row["expected_return"] = float(
                    0.70 * row.get("expected_return", 0.0)
                    + 0.30 * row.get("dwt_denoised_return_5", 0.0)
                )

        row = _forecast_merge_deeplob_signal(symbol, row)
        row = _forecast_merge_rl_signal(symbol, row)

        # Normalize component names and compute weighted ensemble.
        arima = float(row.get("arima_score", row.get("arima_up_probability", 0.50)) or 0.50)
        lstm = float(row.get("lstm_score", row.get("lstm_up_probability", 0.50)) or 0.50)
        stacked = float(row.get("stacked_lstm_score", row.get("stacked_lstm_up_probability", 0.50)) or 0.50)
        attention = float(row.get("attention_lstm_score", row.get("attention_lstm_up_probability", 0.50)) or 0.50)
        deeplob = float(row.get("deeplob_score", 0.50) or 0.50)
        rl = float(row.get("rl_score", 0.50) or 0.50)

        weights = FORECAST_WEIGHTS
        weight_sum = max(sum(weights.values()), 1e-9)

        ensemble = (
            weights["arima"] * arima
            + weights["lstm"] * lstm
            + weights["stacked_lstm"] * stacked
            + weights["attention_lstm"] * attention
            + weights["deeplob"] * deeplob
            + weights["rl"] * rl
        ) / weight_sum

        expected_return = float(row.get("expected_return", row.get("forecast_expected_return", 0.0)) or 0.0)
        drawdown_probability = float(row.get("drawdown_probability", row.get("future_drawdown_probability", 0.50)) or 0.50)

        row.update(
            {
                "symbol": symbol,
                "arima_score": _forecast_clip01(arima),
                "lstm_score": _forecast_clip01(lstm),
                "stacked_lstm_score": _forecast_clip01(stacked),
                "attention_lstm_score": _forecast_clip01(attention),
                "deeplob_score": _forecast_clip01(deeplob),
                "rl_score": _forecast_clip01(rl),
                "ensemble_score": _forecast_clip01(float(ensemble)),
                "up_probability": _forecast_clip01(float(row.get("up_probability", ensemble) or ensemble)),
                "expected_return": expected_return,
                "drawdown_probability": _forecast_clip01(drawdown_probability),
                "forecast_buy_approved": bool(
                    ensemble >= FORECAST_BUY_THRESHOLD
                    and expected_return >= FORECAST_MIN_EXPECTED_RETURN_FOR_BUY
                    and drawdown_probability <= FORECAST_MAX_DRAWDOWN_PROB_FOR_BUY
                ),
                "forecast_strong_buy": bool(ensemble >= FORECAST_STRONG_BUY_THRESHOLD),
                "created_at_utc": _forecast_now_utc_iso(),
            }
        )

        row = apply_pso_params_to_forecast_row(row)
        results[symbol] = row

    payload = {
        "created_at_utc": _forecast_now_utc_iso(),
        "weights": FORECAST_WEIGHTS,
        "signals": list(results.values()),
    }
    try:
        FORECAST_ENSEMBLE_PATH.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
        pd.DataFrame(payload["signals"]).to_csv(FORECAST_ENSEMBLE_CSV_PATH, index=False)
        print(f"[forecast] saved {FORECAST_ENSEMBLE_PATH} rows={len(results)}")
    except Exception as exc:
        _forecast_write_event("forecast_save_error", {"error": str(exc)})

    return results


def apply_forecast_ensemble_to_order_plan(plan: pd.DataFrame, config: ScannerConfig | None = None) -> pd.DataFrame:
    """
    Attach forecast ensemble fields to the order plan and optionally gate BUYs.

    SELL rows are never blocked by the forecast layer.
    """
    if not FORECAST_ENSEMBLE_ENABLED or plan is None or plan.empty or "symbol" not in plan.columns:
        return plan

    out = plan.copy()
    symbols = out["symbol"].map(normalize_symbol).dropna().unique().tolist()
    forecasts = build_forecast_ensemble_for_symbols(symbols)

    if not forecasts:
        if FORECAST_BLOCK_BUYS_WHEN_MISSING and "side" in out.columns:
            buy_mask = out["side"].astype(str).str.lower() == "buy"
            out.loc[buy_mask, "forecast_reject_reason"] = "forecast_missing"
            out.loc[buy_mask, "forecast_buy_approved"] = False
        return out

    for idx, row in out.iterrows():
        symbol = normalize_symbol(row.get("symbol"))
        forecast = forecasts.get(symbol, {})
        if not forecast:
            continue

        for key in [
            "ensemble_score",
            "up_probability",
            "expected_return",
            "drawdown_probability",
            "arima_score",
            "lstm_score",
            "stacked_lstm_score",
            "attention_lstm_score",
            "dwt_trend_score",
            "dwt_detail_energy_ratio",
            "dwt_denoised_return_5",
            "dwt_denoised_return_20",
            "deeplob_score",
            "rl_score",
            "forecast_buy_approved",
            "forecast_strong_buy",
            "forecast_source",
        ]:
            if key in forecast:
                out.at[idx, key] = forecast[key]

        # Feed forecast probabilities into existing Kelly sizing.
        if "continuation_probability" not in out.columns:
            out["continuation_probability"] = np.nan
        if "expected_return" in forecast:
            out.at[idx, "forecast_expected_return"] = forecast.get("expected_return")
        out.at[idx, "continuation_probability"] = forecast.get("up_probability", forecast.get("ensemble_score", 0.50))

        side = str(row.get("side", "")).lower()
        if side == "buy":
            approved = bool(forecast.get("forecast_buy_approved", False))
            if not approved:
                out.at[idx, "forecast_reject_reason"] = (
                    f"forecast_weak:"
                    f"ensemble={float(forecast.get('ensemble_score', 0.0)):.2f},"
                    f"expected_return={float(forecast.get('expected_return', 0.0)):.4f},"
                    f"drawdown_prob={float(forecast.get('drawdown_probability', 1.0)):.2f}"
                )

            old_reason = str(row.get("reason", ""))
            if "forecast_ensemble" not in old_reason:
                out.at[idx, "reason"] = (
                    f"{old_reason}|forecast_ensemble:"
                    f"score={float(forecast.get('ensemble_score', 0.0)):.2f}:"
                    f"attn={float(forecast.get('attention_lstm_score', 0.0)):.2f}:"
                    f"up={float(forecast.get('up_probability', 0.0)):.2f}"
                )

    if "side" in out.columns:
        buy_mask = out["side"].astype(str).str.lower() == "buy"
        reject_mask = buy_mask & (out.get("forecast_buy_approved", pd.Series(True, index=out.index)).astype(str).str.lower().isin({"false", "0", "nan"}))
        if FORECAST_BLOCK_BUYS_WHEN_MISSING:
            if reject_mask.any():
                skipped = out.loc[reject_mask, ["symbol", "side", "ensemble_score", "expected_return", "drawdown_probability", "forecast_reject_reason"]].copy()
                print("[forecast] skipped BUY rows:")
                print(skipped.to_string(index=False))
                out = out.loc[~reject_mask].copy()
        else:
            # Soft mode: do not drop rows; lower confidence so existing gates/Kelly can size down.
            if reject_mask.any() and "confidence_used" in out.columns:
                out.loc[reject_mask, "confidence_used"] = pd.to_numeric(out.loc[reject_mask, "confidence_used"], errors="coerce").fillna(0.0) * 0.75

    return out.reset_index(drop=True)




# =============================================================================
# PSO-LSTM HYPERPARAMETER OPTIMIZER
# =============================================================================
#
# Particle Swarm Optimization layer for the forecasting ensemble.
#
# Purpose:
#   - tune LSTM / Stacked-LSTM / Attention-LSTM hyperparameters
#   - use no command-line arguments
#   - write best params to system-data/YYYYMMDD/YYYYMMDD_pso_lstm_best_params.json
#   - allow forecast_engine.py or local forecast logic to consume the optimized
#     lookback, hidden neurons, layers, epochs, batch size, and learning rate.
#
# Search dimensions:
#   0 = hidden neurons
#   1 = epochs / optimization iterations
#   2 = lookback bars
#   3 = LSTM layers
#   4 = learning rate scaled
#   5 = batch size scaled
#
# The PSO update follows:
#   v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
#   x = x + v

PSO_LSTM_ENABLED = os.getenv("PSO_LSTM_ENABLED", "1").lower() in {"1", "true", "yes"}
PSO_LSTM_OPTIMIZE_ON_START = os.getenv("PSO_LSTM_OPTIMIZE_ON_START", "0").lower() in {"1", "true", "yes"}
PSO_LSTM_USE_IN_FORECAST = os.getenv("PSO_LSTM_USE_IN_FORECAST", "1").lower() in {"1", "true", "yes"}

PSO_LSTM_BEST_PARAMS_PATH = OUT_DIR / f"{DATA_TAG}_pso_lstm_best_params.json"
PSO_LSTM_TRIALS_PATH = OUT_DIR / f"{DATA_TAG}_pso_lstm_trials.jsonl"
PSO_LSTM_EVENTS_PATH = LOG_DIR / f"{DATA_TAG}_pso_lstm_events.jsonl"

PSO_SWARM_SIZE = int(os.getenv("PSO_SWARM_SIZE", "20"))
PSO_MAX_ITER = int(os.getenv("PSO_MAX_ITER", "50"))
PSO_INERTIA_W = float(os.getenv("PSO_INERTIA_W", "0.80"))
PSO_C1 = float(os.getenv("PSO_C1", "1.50"))
PSO_C2 = float(os.getenv("PSO_C2", "1.50"))
PSO_VELOCITY_MIN = float(os.getenv("PSO_VELOCITY_MIN", "-2.0"))
PSO_VELOCITY_MAX = float(os.getenv("PSO_VELOCITY_MAX", "2.0"))

# Keep the paper's [0, 300] search idea, but map dimensions into sane
# production ranges for this trading system.
PSO_LSTM_BOUNDS = np.array(
    [
        [16, 300],       # hidden neurons
        [10, 100],       # epochs
        [20, 100],       # lookback bars; supports 20-day and 50-day retrospective settings
        [1, 3],          # hidden/lstm layers
        [1, 100],        # learning-rate bucket -> 1e-5 to 1e-3
        [16, 128],       # batch size
    ],
    dtype=float,
)

PSO_LSTM_DEFAULT_PARAMS = {
    "hidden_neurons": 128,
    "epochs": 50,
    "lookback": 50,
    "layers": 1,
    "learning_rate": 0.0007,
    "batch_size": 30,
    "fitness": None,
    "source": "default",
}


def _pso_lstm_event(event: str, payload: dict[str, Any]) -> None:
    try:
        row = {
            "timestamp_utc": _forecast_now_utc_iso() if "_forecast_now_utc_iso" in globals() else datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with PSO_LSTM_EVENTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=json_default) + "\n")
    except Exception:
        pass


def pso_position_to_lstm_params(position: np.ndarray) -> dict[str, Any]:
    """
    Convert a particle position into actual LSTM hyperparameters.
    """
    x = np.asarray(position, dtype=float)
    hidden = int(np.clip(round(x[0]), 16, 300))
    epochs = int(np.clip(round(x[1]), 10, 100))
    lookback = int(np.clip(round(x[2]), 20, 100))
    layers = int(np.clip(round(x[3]), 1, 3))

    # Map bucket [1,100] to [1e-5,1e-3] log-ish.
    lr_bucket = float(np.clip(x[4], 1, 100))
    learning_rate = float(10 ** np.interp(lr_bucket, [1, 100], [-5, -3]))

    batch_size_raw = int(np.clip(round(x[5]), 16, 128))
    # Favor mini-batch around 30 as described, but still allow PSO search.
    batch_choices = np.array([16, 24, 30, 32, 48, 64, 96, 128])
    batch_size = int(batch_choices[np.argmin(np.abs(batch_choices - batch_size_raw))])

    return {
        "hidden_neurons": hidden,
        "epochs": epochs,
        "lookback": lookback,
        "layers": layers,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }


def load_pso_lstm_best_params() -> dict[str, Any]:
    if not PSO_LSTM_BEST_PARAMS_PATH.exists():
        return dict(PSO_LSTM_DEFAULT_PARAMS)

    try:
        payload = json.loads(PSO_LSTM_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        if "best_params" in payload:
            params = payload["best_params"]
        else:
            params = payload
        out = dict(PSO_LSTM_DEFAULT_PARAMS)
        out.update(params)
        out["source"] = payload.get("source", out.get("source", "pso_lstm"))
        out["fitness"] = payload.get("best_fitness", out.get("fitness"))
        return out
    except Exception:
        return dict(PSO_LSTM_DEFAULT_PARAMS)


def save_pso_lstm_best_params(params: dict[str, Any], fitness: float, diagnostics: dict[str, Any] | None = None) -> None:
    payload = {
        "ok": True,
        "created_at_utc": _forecast_now_utc_iso() if "_forecast_now_utc_iso" in globals() else datetime.now(timezone.utc).isoformat(),
        "data_tag": DATA_TAG,
        "best_fitness": float(fitness),
        "best_params": params,
        "diagnostics": diagnostics or {},
        "source": "pso_lstm",
        "description": "PSO-optimized LSTM/Stacked-LSTM/Attention-LSTM hyperparameters",
    }
    PSO_LSTM_BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"[pso-lstm] saved best params={PSO_LSTM_BEST_PARAMS_PATH}")


def _pso_symbol_price_frame(symbols: list[str]) -> pd.DataFrame:
    """
    Build a compact multi-symbol close-price dataset from existing local files.
    """
    frames = []
    paths = [
        INTRADAY_BARS_1M_PATH if "INTRADAY_BARS_1M_PATH" in globals() else OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv",
        OUT_DIR / f"{DATA_TAG}_daily_features.csv",
        PROJECT_ROOT / "daily_system_outputs" / "daily_features.csv",
    ]

    wanted = {normalize_symbol(s) for s in symbols if normalize_symbol(s)}

    for path in paths:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
            if "symbol" not in df.columns:
                continue

            close_col = "close" if "close" in df.columns else "Close" if "Close" in df.columns else "adj_close" if "adj_close" in df.columns else None
            if close_col is None:
                continue

            time_col = "timestamp" if "timestamp" in df.columns else "date" if "date" in df.columns else None
            df = df.copy()
            df["symbol"] = df["symbol"].map(normalize_symbol)
            if wanted:
                df = df[df["symbol"].isin(wanted)]
            df["close"] = pd.to_numeric(df[close_col], errors="coerce")
            if time_col:
                df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            else:
                df["timestamp"] = pd.RangeIndex(len(df))

            df = df.dropna(subset=["symbol", "close"])
            if not df.empty:
                frames.append(df[["timestamp", "symbol", "close"]])
        except Exception as exc:
            _pso_lstm_event("pso_price_frame_load_error", {"path": str(path), "error": str(exc)})

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _pso_lstm_proxy_fitness(params: dict[str, Any], price_frame: pd.DataFrame) -> float:
    """
    Fast no-dependency proxy fitness.

    This keeps run.py lightweight. It estimates how well a given lookback can
    predict next-period return direction using a simple attention-weighted
    return model, while penalizing over-complex LSTM settings.

    If you wire an external forecast_engine.py, that engine can use these saved
    PSO params for real LSTM training. This proxy allows the optimizer itself to
    run safely inside run.py without expensive deep training every startup.
    """
    if price_frame.empty:
        return -9999.0

    lookback = int(params["lookback"])
    layers = int(params["layers"])
    hidden = int(params["hidden_neurons"])
    epochs = int(params["epochs"])
    lr = float(params["learning_rate"])
    batch = int(params["batch_size"])

    scores = []
    total_samples = 0

    for symbol, g in price_frame.groupby("symbol", sort=False):
        close = pd.to_numeric(g["close"], errors="coerce").dropna().astype(float)
        if len(close) < lookback + 25:
            continue

        dwt_pack = _dwt_feature_pack(close)
        denoised_close = pd.Series(dwt_pack.get("denoised_close", close))
        rets = denoised_close.pct_change().dropna().reset_index(drop=True)
        if len(rets) < lookback + 10:
            continue

        correct = 0
        count = 0
        pnl = 0.0
        losses = 0.0

        for i in range(lookback, len(rets) - 1):
            window = rets.iloc[i - lookback:i].to_numpy(dtype=float)
            target = float(rets.iloc[i + 1])

            # Soft-attention proxy: larger absolute bars get higher weight.
            abs_w = np.abs(window)
            weights = abs_w / max(abs_w.sum(), 1e-9)
            attention_return = float(np.sum(window * weights))

            # LSTM-ish memory proxy: recent drift plus attention return.
            recent_return = float(np.mean(window[-min(10, len(window)):]))
            long_return = float(np.mean(window))
            pred = 0.50 * attention_return + 0.35 * recent_return + 0.15 * long_return

            if np.sign(pred) == np.sign(target) or abs(target) < 1e-6:
                correct += 1

            if pred > 0:
                pnl += target
                if target < 0:
                    losses += abs(target)

            count += 1

        if count > 0:
            accuracy = correct / count
            avg_pnl = pnl / count
            downside = losses / max(count, 1)
            scores.append(accuracy * 2.0 + avg_pnl * 500.0 - downside * 300.0)
            total_samples += count

    if not scores:
        return -9999.0

    base = float(np.mean(scores))

    # Complexity penalty prevents PSO from always selecting max neurons/layers.
    complexity_penalty = (
        (hidden / 300.0) * 0.05
        + max(0, layers - 1) * 0.03
        + (epochs / 100.0) * 0.02
    )

    # Batch size around 30 is favored per the paper.
    batch_penalty = abs(batch - 30) / 128.0 * 0.04

    # LR penalty for extremes.
    lr_penalty = abs(np.log10(max(lr, 1e-9)) - np.log10(7e-4)) * 0.02

    fitness = base - complexity_penalty - batch_penalty - lr_penalty
    return float(fitness)


def run_pso_lstm_optimization(symbols: list[str] | None = None) -> dict[str, Any]:
    """
    Run PSO optimization and save best LSTM hyperparameters.

    This runs without arguments and uses current universe/order symbols.
    """
    if not PSO_LSTM_ENABLED:
        return {"ok": False, "reason": "disabled"}

    if symbols is None:
        try:
            symbols = get_es_symbols_from_orders()
        except Exception:
            symbols = []

    price_frame = _pso_symbol_price_frame(symbols or [])
    if price_frame.empty:
        params = dict(PSO_LSTM_DEFAULT_PARAMS)
        save_pso_lstm_best_params(params, -9999.0, {"reason": "missing_price_history"})
        return {"ok": False, "reason": "missing_price_history", "best_params": params}

    rng = np.random.default_rng(int(os.getenv("PSO_RANDOM_SEED", "42")))
    bounds = PSO_LSTM_BOUNDS
    dim = bounds.shape[0]

    positions = rng.uniform(bounds[:, 0], bounds[:, 1], size=(PSO_SWARM_SIZE, dim))
    velocities = rng.uniform(PSO_VELOCITY_MIN, PSO_VELOCITY_MAX, size=(PSO_SWARM_SIZE, dim))

    pbest_positions = positions.copy()
    pbest_fitness = np.full(PSO_SWARM_SIZE, np.inf, dtype=float)

    gbest_position = None
    gbest_fitness = np.inf

    for iteration in range(1, PSO_MAX_ITER + 1):
        for i in range(PSO_SWARM_SIZE):
            params = pso_position_to_lstm_params(positions[i])
            # PSO convention in the paper minimizes f(x), so use negative fitness.
            objective = -_pso_lstm_proxy_fitness(params, price_frame)

            trial_row = {
                "timestamp_utc": _forecast_now_utc_iso() if "_forecast_now_utc_iso" in globals() else datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "particle": i,
                "objective": float(objective),
                "fitness": float(-objective),
                "params": params,
            }
            with PSO_LSTM_TRIALS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(trial_row, default=json_default) + "\n")

            if objective < pbest_fitness[i]:
                pbest_fitness[i] = objective
                pbest_positions[i] = positions[i].copy()

            if objective < gbest_fitness:
                gbest_fitness = objective
                gbest_position = positions[i].copy()

        if gbest_position is None:
            break

        r1 = rng.random(size=(PSO_SWARM_SIZE, dim))
        r2 = rng.random(size=(PSO_SWARM_SIZE, dim))

        velocities = (
            PSO_INERTIA_W * velocities
            + PSO_C1 * r1 * (pbest_positions - positions)
            + PSO_C2 * r2 * (gbest_position - positions)
        )
        velocities = np.clip(velocities, PSO_VELOCITY_MIN, PSO_VELOCITY_MAX)

        positions = positions + velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

        best_params = pso_position_to_lstm_params(gbest_position)
        best_fitness = -float(gbest_fitness)
        print(f"[pso-lstm] iter={iteration}/{PSO_MAX_ITER} best_fitness={best_fitness:.6f} params={best_params}")

    if gbest_position is None:
        params = dict(PSO_LSTM_DEFAULT_PARAMS)
        best_fitness = -9999.0
    else:
        params = pso_position_to_lstm_params(gbest_position)
        best_fitness = -float(gbest_fitness)

    diagnostics = {
        "swarm_size": PSO_SWARM_SIZE,
        "max_iter": PSO_MAX_ITER,
        "inertia_w": PSO_INERTIA_W,
        "c1": PSO_C1,
        "c2": PSO_C2,
        "velocity_min": PSO_VELOCITY_MIN,
        "velocity_max": PSO_VELOCITY_MAX,
        "symbols": sorted(price_frame["symbol"].unique().tolist()) if "symbol" in price_frame.columns else [],
        "rows": int(len(price_frame)),
    }

    save_pso_lstm_best_params(params, best_fitness, diagnostics)
    _pso_lstm_event("pso_lstm_optimization_complete", {"fitness": best_fitness, "params": params, "diagnostics": diagnostics})

    return {
        "ok": True,
        "best_fitness": best_fitness,
        "best_params": params,
        "diagnostics": diagnostics,
    }


def maybe_run_pso_lstm_optimization() -> None:
    if not PSO_LSTM_ENABLED:
        return

    should_optimize = PSO_LSTM_OPTIMIZE_ON_START or not PSO_LSTM_BEST_PARAMS_PATH.exists()
    if not should_optimize:
        return

    try:
        print("[pso-lstm] optimizing LSTM hyperparameters")
        run_pso_lstm_optimization()
    except Exception as exc:
        print(f"[pso-lstm] optimization skipped/error: {exc}")
        _pso_lstm_event("pso_lstm_optimization_error", {"error": str(exc)})


def apply_pso_params_to_forecast_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Attach PSO-optimized LSTM hyperparameters to forecast rows.

    External LSTM/Attention-LSTM scripts can also read the same JSON directly.
    """
    if not PSO_LSTM_ENABLED or not PSO_LSTM_USE_IN_FORECAST:
        return row

    params = load_pso_lstm_best_params()
    row["pso_lstm_hidden_neurons"] = params.get("hidden_neurons")
    row["pso_lstm_epochs"] = params.get("epochs")
    row["pso_lstm_lookback"] = params.get("lookback")
    row["pso_lstm_layers"] = params.get("layers")
    row["pso_lstm_learning_rate"] = params.get("learning_rate")
    row["pso_lstm_batch_size"] = params.get("batch_size")
    row["pso_lstm_fitness"] = params.get("fitness")
    return row


def maybe_refresh_forecast_ensemble_layer() -> None:
    if not FORECAST_ENSEMBLE_ENABLED:
        return
    try:
        refresh_forecast_ensemble_if_available()
    except Exception as exc:
        print(f"[forecast] refresh skipped/error: {exc}")
        _forecast_write_event("forecast_refresh_error", {"error": str(exc)})



# =============================================================================
# EVOLUTION STRATEGY MAIN STRATEGY LAYER
# =============================================================================
#
# This is a cleaned, no-argument implementation of the user's Evolution Strategy
# agent. It trains automatically from available price history and emits long-only
# BUY / HOLD / SELL timing signals that can be merged into the main order plan.
#
# Important changes from the notebook-style snippet:
#   - no df_full dependency
#   - no pkg_resources requirement printing
#   - no close/global variable bug
#   - long-only: no shorts
#   - uses historical close data from daily/intraday bars already available
#   - writes dated signal/model artifacts
#   - can be called from run.py without command-line arguments

ES_LAYER_ENABLED = os.getenv("ES_LAYER_ENABLED", "1").lower() in {"1", "true", "yes"}
ES_TRAIN_ON_START = os.getenv("ES_TRAIN_ON_START", "0").lower() in {"1", "true", "yes"}
ES_USE_AS_ORDER_SOURCE = os.getenv("ES_USE_AS_ORDER_SOURCE", "1").lower() in {"1", "true", "yes"}
ES_LONG_ONLY = os.getenv("ES_LONG_ONLY", "1").lower() in {"1", "true", "yes"}

ES_WINDOW_SIZE = int(os.getenv("ES_WINDOW_SIZE", "30"))
ES_LAYER_SIZE = int(os.getenv("ES_LAYER_SIZE", "256"))
ES_POPULATION_SIZE = int(os.getenv("ES_POPULATION_SIZE", "15"))
ES_SIGMA = float(os.getenv("ES_SIGMA", "0.10"))
ES_LEARNING_RATE = float(os.getenv("ES_LEARNING_RATE", "0.03"))
ES_ITERATIONS = int(os.getenv("ES_ITERATIONS", "120"))
ES_CHECKPOINT = int(os.getenv("ES_CHECKPOINT", "10"))
ES_SKIP = int(os.getenv("ES_SKIP", "1"))
ES_INITIAL_MONEY = float(os.getenv("ES_INITIAL_MONEY", "10000"))
ES_MIN_SIGNAL_CONFIDENCE = float(os.getenv("ES_MIN_SIGNAL_CONFIDENCE", "0.34"))
ES_MAX_ORDER_DOLLARS = float(os.getenv("ES_MAX_ORDER_DOLLARS", "1500"))
ES_MIN_HISTORY_BARS = int(os.getenv("ES_MIN_HISTORY_BARS", "80"))
ES_SYMBOL_LIMIT = int(os.getenv("ES_SYMBOL_LIMIT", "30"))

ES_MODEL_PATH = OUT_DIR / f"{DATA_TAG}_evolution_strategy_models.json"
ES_SIGNALS_PATH = OUT_DIR / f"{DATA_TAG}_evolution_strategy_signals.json"
ES_SIGNALS_CSV_PATH = OUT_DIR / f"{DATA_TAG}_evolution_strategy_signals.csv"
ES_METRICS_PATH = OUT_DIR / f"{DATA_TAG}_evolution_strategy_metrics.json"


def _es_softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / max(float(e.sum()), 1e-12)


class EvolutionStrategyOptimizer:
    def __init__(
        self,
        weights: list[np.ndarray],
        reward_function: Any,
        population_size: int,
        sigma: float,
        learning_rate: float,
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = int(population_size)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)

    def _get_weight_from_population(self, weights: list[np.ndarray], population: list[np.ndarray]) -> list[np.ndarray]:
        return [weights[idx] + self.sigma * noise for idx, noise in enumerate(population)]

    def get_weights(self) -> list[np.ndarray]:
        return self.weights

    def train(self, epoch: int = 100, print_every: int = 10) -> list[float]:
        rewards_history: list[float] = []
        lasttime = time.time()

        for i in range(int(epoch)):
            population: list[list[np.ndarray]] = []
            rewards = np.zeros(self.population_size, dtype=float)

            for _ in range(self.population_size):
                population.append([np.random.randn(*w.shape) for w in self.weights])

            for k in range(self.population_size):
                candidate_weights = self._get_weight_from_population(self.weights, population[k])
                rewards[k] = self.reward_function(candidate_weights)

            rewards_norm = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

            for index, w in enumerate(self.weights):
                noise_matrix = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(noise_matrix.T, rewards_norm).T
                )

            current_reward = float(self.reward_function(self.weights))
            rewards_history.append(current_reward)

            if (i + 1) % max(1, int(print_every)) == 0:
                print(f"[es-layer] iter={i + 1} reward={current_reward:.4f}")

        print(f"[es-layer] training_time_seconds={time.time() - lasttime:.2f}")
        return rewards_history


class EvolutionStrategyModel:
    def __init__(self, input_size: int, layer_size: int, output_size: int, weights: list[np.ndarray] | None = None):
        if weights is not None:
            self.weights = weights
        else:
            scale_1 = 1.0 / max(np.sqrt(input_size), 1.0)
            scale_2 = 1.0 / max(np.sqrt(layer_size), 1.0)
            self.weights = [
                np.random.randn(input_size, layer_size) * scale_1,
                np.random.randn(layer_size, output_size) * scale_2,
                np.zeros((1, layer_size), dtype=float),
            ]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        feed = np.tanh(np.dot(inputs, self.weights[0]) + self.weights[-1])
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self) -> list[np.ndarray]:
        return self.weights

    def set_weights(self, weights: list[np.ndarray]) -> None:
        self.weights = weights


class EvolutionStrategyAgent:
    # Actions:
    #   0 = HOLD
    #   1 = BUY / ENTER_LONG
    #   2 = SELL / EXIT_LONG
    def __init__(
        self,
        model: EvolutionStrategyModel,
        window_size: int,
        trend: list[float],
        skip: int,
        initial_money: float,
        population_size: int = ES_POPULATION_SIZE,
        sigma: float = ES_SIGMA,
        learning_rate: float = ES_LEARNING_RATE,
    ):
        self.model = model
        self.window_size = int(window_size)
        self.trend = [float(x) for x in trend if float(x) > 0]
        self.skip = max(1, int(skip))
        self.initial_money = float(initial_money)
        self.population_size = int(population_size)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.es = EvolutionStrategyOptimizer(
            self.model.get_weights(),
            self.get_reward,
            self.population_size,
            self.sigma,
            self.learning_rate,
        )

    def act(self, sequence: np.ndarray) -> tuple[int, np.ndarray]:
        decision = self.model.predict(np.array(sequence, dtype=float))
        probs = _es_softmax(decision[0])
        return int(np.argmax(probs)), probs

    def get_state(self, t: int) -> np.ndarray:
        window_size = self.window_size + 1
        d = t - window_size + 1
        if d >= 0:
            block = self.trend[d:t + 1]
        else:
            block = [-d * [self.trend[0]][0]] * 0  # keeps lint quiet; replaced below
            block = (-d) * [self.trend[0]] + self.trend[0:t + 1]

        diffs = []
        for i in range(window_size - 1):
            prev_px = max(float(block[i]), 1e-9)
            next_px = float(block[i + 1])
            # Percent change is more stable across symbols than raw dollars.
            diffs.append((next_px / prev_px) - 1.0)
        return np.array([diffs], dtype=float)

    def get_reward(self, weights: list[np.ndarray]) -> float:
        if len(self.trend) < self.window_size + 5:
            return -100.0

        starting_cash = self.initial_money
        cash = starting_cash
        inventory: list[float] = []
        state = self.get_state(0)

        realized_pnl = 0.0
        trade_count = 0
        bad_trade_count = 0
        max_equity = starting_cash
        max_drawdown = 0.0

        for t in range(0, len(self.trend) - 1, self.skip):
            self.model.weights = weights
            action, _ = self.act(state)
            price = float(self.trend[t])
            next_state = self.get_state(t + 1)

            if action == 1 and cash >= price and not inventory:
                inventory.append(price)
                cash -= price
                trade_count += 1

            elif action == 2 and inventory:
                bought_price = inventory.pop(0)
                pnl = price - bought_price
                realized_pnl += pnl
                cash += price
                trade_count += 1
                if pnl <= 0:
                    bad_trade_count += 1

            mark_value = cash + (len(inventory) * price)
            max_equity = max(max_equity, mark_value)
            drawdown = (max_equity - mark_value) / max(max_equity, 1e-9)
            max_drawdown = max(max_drawdown, drawdown)

            state = next_state

        if inventory:
            final_price = float(self.trend[-1])
            bought_price = inventory.pop(0)
            realized_pnl += final_price - bought_price
            cash += final_price

        ret_pct = ((cash - starting_cash) / max(starting_cash, 1e-9)) * 100.0

        # Reward profitable strategies, penalize drawdown, bad trades, and no-action policies.
        reward = ret_pct
        reward -= max_drawdown * 100.0 * 0.75
        reward -= bad_trade_count * 0.15
        if trade_count == 0:
            reward -= 2.5
        return float(reward)

    def fit(self, iterations: int, checkpoint: int) -> list[float]:
        return self.es.train(iterations, print_every=checkpoint)

    def latest_signal(self) -> dict[str, Any]:
        state = self.get_state(len(self.trend) - 1)
        action, probs = self.act(state)
        price = float(self.trend[-1])
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "HOLD")
        confidence = float(np.max(probs))
        return {
            "action_id": int(action),
            "action": action_name,
            "confidence": confidence,
            "prob_hold": float(probs[0]),
            "prob_buy": float(probs[1]),
            "prob_sell": float(probs[2]),
            "price": price,
        }

    def backtest(self) -> dict[str, Any]:
        cash = self.initial_money
        starting_cash = cash
        inventory: list[float] = []
        buys: list[int] = []
        sells: list[int] = []
        state = self.get_state(0)

        for t in range(0, len(self.trend) - 1, self.skip):
            action, _ = self.act(state)
            price = float(self.trend[t])
            next_state = self.get_state(t + 1)

            if action == 1 and cash >= price and not inventory:
                inventory.append(price)
                cash -= price
                buys.append(t)

            elif action == 2 and inventory:
                inventory.pop(0)
                cash += price
                sells.append(t)

            state = next_state

        if inventory:
            cash += float(self.trend[-1])

        total_gains = cash - starting_cash
        invest_pct = (total_gains / max(starting_cash, 1e-9)) * 100.0
        return {
            "states_buy": buys,
            "states_sell": sells,
            "total_gains": float(total_gains),
            "invest_pct": float(invest_pct),
            "buy_count": len(buys),
            "sell_count": len(sells),
        }


def _es_weights_to_json(weights: list[np.ndarray]) -> list[Any]:
    return [w.tolist() for w in weights]


def _es_weights_from_json(payload: list[Any]) -> list[np.ndarray]:
    return [np.array(w, dtype=float) for w in payload]


def load_es_models() -> dict[str, Any]:
    if not ES_MODEL_PATH.exists():
        return {}
    try:
        return json.loads(ES_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_es_models(payload: dict[str, Any]) -> None:
    ES_MODEL_PATH.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")


def get_es_symbols_from_orders() -> list[str]:
    symbols: set[str] = set()

    candidate_paths = [
        OUT_DIR / f"{DATA_TAG}_daily_walkforward_emulated_rebalance_orders.csv",
        OUT_DIR / f"{DATA_TAG}_stream_confirmed_rebalance_orders.csv",
        PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt",
        PROJECT_ROOT / "locked_trading_universe.txt",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                if "symbol" in df.columns:
                    symbols.update(df["symbol"].map(normalize_symbol).dropna().tolist())
            else:
                raw = path.read_text(encoding="utf-8")
                symbols.update(normalize_symbol(x) for x in raw.replace("\n", ",").split(",") if normalize_symbol(x))
        except Exception:
            continue

    return sorted(symbols)[:ES_SYMBOL_LIMIT]


def load_es_price_history(symbols: list[str]) -> dict[str, list[float]]:
    """
    Load price history from local intraday/daily data first.
    Falls back to any order-plan price if no history exists.
    """
    out: dict[str, list[float]] = {}

    paths = [
        OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv",
        OUT_DIR / f"{DATA_TAG}_daily_features.csv",
        PROJECT_ROOT / "daily_system_outputs" / "daily_features.csv",
    ]

    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "symbol" not in df.columns:
                continue
            close_col = "close" if "close" in df.columns else "Close" if "Close" in df.columns else None
            if close_col is None:
                continue
            df["symbol"] = df["symbol"].map(normalize_symbol)
            for symbol, g in df[df["symbol"].isin(symbols)].groupby("symbol", sort=False):
                prices = pd.to_numeric(g[close_col], errors="coerce").dropna().astype(float).tolist()
                prices = [p for p in prices if p > 0]
                if len(prices) >= ES_MIN_HISTORY_BARS:
                    out[symbol] = prices
        except Exception:
            continue

    return out


def train_evolution_strategy_layer(symbols: list[str] | None = None) -> dict[str, Any]:
    if not ES_LAYER_ENABLED:
        return {"ok": False, "reason": "disabled"}

    symbols = symbols or get_es_symbols_from_orders()
    price_map = load_es_price_history(symbols)
    models = load_es_models()

    metrics: dict[str, Any] = {
        "ok": True,
        "created_at_utc": now_utc_iso() if "now_utc_iso" in globals() else datetime.utcnow().isoformat(),
        "symbols_requested": symbols,
        "symbols_trained": [],
        "symbols_skipped": [],
    }

    for symbol in symbols:
        trend = price_map.get(symbol, [])
        if len(trend) < max(ES_MIN_HISTORY_BARS, ES_WINDOW_SIZE + 5):
            metrics["symbols_skipped"].append({"symbol": symbol, "reason": f"not_enough_history:{len(trend)}"})
            continue

        print(f"[es-layer] training symbol={symbol} bars={len(trend)}")

        model = EvolutionStrategyModel(
            input_size=ES_WINDOW_SIZE,
            layer_size=ES_LAYER_SIZE,
            output_size=3,
        )
        agent = EvolutionStrategyAgent(
            model=model,
            window_size=ES_WINDOW_SIZE,
            trend=trend,
            skip=ES_SKIP,
            initial_money=ES_INITIAL_MONEY,
        )
        rewards = agent.fit(iterations=ES_ITERATIONS, checkpoint=ES_CHECKPOINT)
        backtest = agent.backtest()

        models[symbol] = {
            "symbol": symbol,
            "weights": _es_weights_to_json(agent.model.get_weights()),
            "window_size": ES_WINDOW_SIZE,
            "layer_size": ES_LAYER_SIZE,
            "output_size": 3,
            "trained_at_utc": metrics["created_at_utc"],
            "bars": len(trend),
            "last_reward": float(rewards[-1]) if rewards else None,
            "backtest": backtest,
        }
        metrics["symbols_trained"].append({
            "symbol": symbol,
            "bars": len(trend),
            "last_reward": float(rewards[-1]) if rewards else None,
            **backtest,
        })

    save_es_models(models)
    ES_METRICS_PATH.write_text(json.dumps(safe_json(metrics), indent=2) + "\n", encoding="utf-8")
    print(f"[es-layer] saved models={ES_MODEL_PATH}")
    print(f"[es-layer] saved metrics={ES_METRICS_PATH}")
    return metrics


def infer_evolution_strategy_signals(symbols: list[str] | None = None) -> list[dict[str, Any]]:
    if not ES_LAYER_ENABLED:
        return []

    symbols = symbols or get_es_symbols_from_orders()
    price_map = load_es_price_history(symbols)
    models_payload = load_es_models()
    signals: list[dict[str, Any]] = []

    for symbol in symbols:
        trend = price_map.get(symbol, [])
        model_data = models_payload.get(symbol)

        if not model_data or len(trend) < max(ES_MIN_HISTORY_BARS, ES_WINDOW_SIZE + 5):
            continue

        try:
            weights = _es_weights_from_json(model_data["weights"])
            model = EvolutionStrategyModel(
                input_size=int(model_data.get("window_size", ES_WINDOW_SIZE)),
                layer_size=int(model_data.get("layer_size", ES_LAYER_SIZE)),
                output_size=3,
                weights=weights,
            )
            agent = EvolutionStrategyAgent(
                model=model,
                window_size=int(model_data.get("window_size", ES_WINDOW_SIZE)),
                trend=trend,
                skip=ES_SKIP,
                initial_money=ES_INITIAL_MONEY,
            )
            signal = agent.latest_signal()
            action = signal["action"]

            if ES_LONG_ONLY and action not in {"HOLD", "BUY", "SELL"}:
                action = "HOLD"

            # Low confidence actions are treated as HOLD.
            if signal["confidence"] < ES_MIN_SIGNAL_CONFIDENCE:
                action = "HOLD"

            row = {
                "symbol": symbol,
                "action": action,
                "raw_action": signal["action"],
                "confidence": signal["confidence"],
                "prob_hold": signal["prob_hold"],
                "prob_buy": signal["prob_buy"],
                "prob_sell": signal["prob_sell"],
                "price": signal["price"],
                "source": "evolution_strategy_layer",
                "created_at_utc": now_utc_iso() if "now_utc_iso" in globals() else datetime.utcnow().isoformat(),
            }
            signals.append(row)
        except Exception as exc:
            print(f"[es-layer] inference failed symbol={symbol}: {exc}")

    payload = {
        "created_at_utc": now_utc_iso() if "now_utc_iso" in globals() else datetime.utcnow().isoformat(),
        "signals": signals,
    }
    ES_SIGNALS_PATH.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(signals).to_csv(ES_SIGNALS_CSV_PATH, index=False)

    if signals:
        print(f"[es-layer] saved signals={ES_SIGNALS_PATH} rows={len(signals)}")

    return signals


def load_evolution_strategy_orders() -> pd.DataFrame:
    """
    Convert ES BUY/SELL signals into executor-compatible order rows.
    SELL rows are exits and should always be allowed.
    BUY rows are small timing entries and still pass through the normal executor
    caps, Kelly sizing, cash buffer, PDT controls, and direction gates.
    """
    if not ES_LAYER_ENABLED or not ES_USE_AS_ORDER_SOURCE:
        return pd.DataFrame()

    if not ES_SIGNALS_PATH.exists():
        infer_evolution_strategy_signals()

    try:
        payload = json.loads(ES_SIGNALS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    signals = payload.get("signals", [])
    rows = []

    for sig in signals:
        symbol = normalize_symbol(sig.get("symbol"))
        action = str(sig.get("action", "HOLD")).upper()
        confidence = float(sig.get("confidence", 0.0) or 0.0)
        price = float(sig.get("price", 0.0) or 0.0)

        if not symbol or price <= 0 or action not in {"BUY", "SELL"}:
            continue

        side = "buy" if action == "BUY" else "sell"
        qty = max(1, int(ES_MAX_ORDER_DOLLARS // price)) if side == "buy" else 0

        rows.append({
            "symbol": symbol,
            "side": side,
            "order_qty": qty,
            "price": price,
            "abs_delta_dollars": qty * price if side == "buy" else 0.0,
            "capped_dollars": min(qty * price, ES_MAX_ORDER_DOLLARS) if side == "buy" else 0.0,
            "confidence_used": confidence,
            "confidence_multiplier": 1.0 + confidence,
            "reason": f"evolution_strategy_{action.lower()}:conf={confidence:.2f}:p_buy={sig.get('prob_buy', 0):.2f}:p_sell={sig.get('prob_sell', 0):.2f}",
            "source": "evolution_strategy_layer",
            "es_action": action,
            "es_prob_hold": sig.get("prob_hold"),
            "es_prob_buy": sig.get("prob_buy"),
            "es_prob_sell": sig.get("prob_sell"),
            "stream_confirmed": True if side == "buy" else False,
        })

    return pd.DataFrame(rows)


def merge_evolution_strategy_orders(plan: pd.DataFrame) -> pd.DataFrame:
    es_orders = load_evolution_strategy_orders()
    if es_orders.empty:
        return plan

    if plan is None or plan.empty:
        print(f"[es-layer] merged rows={len(es_orders)}")
        return es_orders.copy()

    out = plan.copy()
    for _, row in es_orders.iterrows():
        symbol = normalize_symbol(row.get("symbol"))
        side = str(row.get("side", "")).lower()

        if not symbol:
            continue

        same_symbol = out["symbol"].map(normalize_symbol) == symbol if "symbol" in out.columns else pd.Series(False, index=out.index)

        # Exits win. Entries do not duplicate existing rows.
        if side == "sell":
            out = out[~same_symbol].copy()
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
        elif side == "buy" and not same_symbol.any():
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    print(f"[es-layer] merged rows={len(es_orders)} total_rows={len(out)}")
    return out.reset_index(drop=True)


def maybe_refresh_evolution_strategy_layer() -> None:
    if not ES_LAYER_ENABLED:
        return

    should_train = ES_TRAIN_ON_START or not ES_MODEL_PATH.exists()
    if should_train:
        try:
            print("[es-layer] training/refreshing strategy models")
            train_evolution_strategy_layer()
        except Exception as exc:
            print(f"[es-layer] training skipped/error: {exc}")

    try:
        infer_evolution_strategy_signals()
    except Exception as exc:
        print(f"[es-layer] signal generation skipped/error: {exc}")



# =============================================================================
# SPX PRICE-ACTION / MULTIBAR RL LAYER INTEGRATION
# =============================================================================

SPX_PRICE_ACTION_LAYER_PATH = PROJECT_ROOT / "spx_price_action_layer.py"
SPX_LAYER_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "spx_price_action_layer.py",
    PROJECT_ROOT / "spx_price_action_layer_updated.py",
    PROJECT_ROOT / "spx_price_action_layer_2.py",
    PROJECT_ROOT / "spx_price_action_layer(2).py",
]

SPX_LAYER_ENABLED = os.getenv("SPX_LAYER_ENABLED", "1").lower() in {"1", "true", "yes"}
SPX_LAYER_INTERNAL_SUBPROCESS = os.getenv("SPX_LAYER_INTERNAL_SUBPROCESS", "1").lower() in {"1", "true", "yes"}
SPX_LAYER_MERGE_ORDERS = os.getenv("SPX_LAYER_MERGE_ORDERS", "1").lower() in {"1", "true", "yes"}
SPX_LAYER_TRAIN_ON_START_FROM_RUN = os.getenv("SPX_LAYER_TRAIN_ON_START_FROM_RUN", "0").lower() in {"1", "true", "yes"}
SPX_LAYER_EXECUTE_FROM_RUN = os.getenv("SPX_LAYER_EXECUTE_FROM_RUN", os.getenv("SPX_LAYER_EXECUTE", "0")).lower() in {"1", "true", "yes"}

SPX_TRADE_SYMBOL_FROM_RUN = os.getenv("SPX_TRADE_SYMBOL", "SPY").upper().strip()
SPX_CONFIRMED_ORDERS_PATH = OUT_DIR / f"{DATA_TAG}_spx_price_action_rebalance_orders.csv"
SPX_SIGNAL_PATH = OUT_DIR / f"{DATA_TAG}_spx_price_action_signal.json"
SPX_LAYER_LOG_PATH = LOG_DIR / f"{DATA_TAG}_spx_price_action_layer_from_run.log"

# Tuned SPX multibar RL defaults from the latest Optuna run.
SPX_RL_TUNED_ENV_DEFAULTS = {
    "SPX_LAYER_RL_TRAIN_LOOKBACK_DAYS": "59",
    "SPX_LAYER_RL_TIMESTEPS": "25000",
    "SPX_LAYER_RL_MAX_EPISODE_BARS": "180",
    "SPX_LAYER_RL_LEARNING_RATE": "0.0001465181270477539",
    "SPX_LAYER_RL_GAMMA": "0.9886884381471769",
    "SPX_LAYER_RL_N_STEPS": "1024",
    "SPX_LAYER_RL_BATCH_SIZE": "128",
    "SPX_LAYER_RL_ENT_COEF": "0.00793730366338695",
    "SPX_LAYER_RL_REWARD_SCALE": "123.98760355121524",
    "SPX_LAYER_RL_ENTRY_BONUS": "1.0736321062529641",
    "SPX_LAYER_RL_BAD_ENTRY_PENALTY": "2.78478083560159",
    "SPX_LAYER_RL_EARLY_MULTIBAR_BONUS_MULT": "201.93742669544676",
    "SPX_LAYER_RL_HOLD_WINNER_MULT": "55.993219878576085",
    "SPX_LAYER_RL_MFE_CAPTURE_MULT": "19.767240684961102",
    "SPX_LAYER_RL_GIVEBACK_PENALTY_MULT": "25.565381568951203",
    "SPX_LAYER_RL_MAE_PENALTY_MULT": "4.680926975116515",
    "SPX_LAYER_RL_MIN_QUALITY_HOLD_BARS": "11",
    "SPX_LAYER_RL_FLAT_HOLD_PENALTY": "0.0036799118887083098",
    "SPX_LAYER_RL_MIN_ENTRY_CONFIDENCE": "0.7164247276618194",
    "SPX_LAYER_RL_MIN_EXIT_CONFIDENCE": "0.4326416408435226",
}



def _spx_layer_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_DATA_DATE"] = DATA_TAG
    env["SPX_LAYER_DATA_DATE"] = DATA_TAG
    env["SPX_TRADE_SYMBOL"] = SPX_TRADE_SYMBOL_FROM_RUN
    env["SPX_LOGICAL_SYMBOL"] = os.getenv("SPX_LOGICAL_SYMBOL", "SPX")
    env["SPX_LAYER_EXECUTE"] = "1" if SPX_LAYER_EXECUTE_FROM_RUN else "0"
    env["SPX_LAYER_RL_ENABLED"] = os.getenv("SPX_LAYER_RL_ENABLED", "1")
    env["SPX_LAYER_RL_TRAIN_ON_START"] = (
        "1" if SPX_LAYER_TRAIN_ON_START_FROM_RUN else os.getenv("SPX_LAYER_RL_TRAIN_ON_START", "0")
    )

    # Use tuned Optuna defaults unless the shell explicitly overrides them.
    for key, value in SPX_RL_TUNED_ENV_DEFAULTS.items():
        env[key] = os.getenv(key, value)

    # Diagnostics for the updated SPX layer.
    env["SPX_LAYER_RL_FEATURES"] = "1m,5m,15m,1h,4h,RSI,SMA5_cross_up_SMA20,SMA20_below_SMA5,ATR,MACD"
    env["SPX_LAYER_ONE_TRADE_PER_SESSION"] = os.getenv("SPX_LAYER_ONE_TRADE_PER_SESSION", "1")
    env["SPX_LAYER_LEARN_MAX_RETURN_TRADE"] = os.getenv("SPX_LAYER_LEARN_MAX_RETURN_TRADE", "1")
    return env

def ensure_spx_price_action_layer_module() -> Path | None:
    """
    Resolve the dedicated SPX price-action layer module.

    Preferred project-root file:
        spx_price_action_layer.py

    Also supports:
        spx_price_action_layer_updated.py

    The updated layer owns:
      - one max-return trade per session
      - SPY proxy execution
      - tuned Optuna RL defaults
      - 1m/5m/15m/1h/4h RL multibar features
      - RSI / SMA5-SMA20 / ATR / MACD feature columns
      - SPX signal, state, candidate, model and metrics files
    """
    if not SPX_LAYER_ENABLED:
        return None

    for candidate in SPX_LAYER_SOURCE_CANDIDATES:
        if candidate.exists():
            print(f"[spx-layer] using module={candidate}")
            return candidate

    print("[spx-layer] module not found; expected spx_price_action_layer.py or spx_price_action_layer_updated.py in project root")
    return None

def start_spx_price_action_layer_background() -> None:
    """
    Launch the SPX layer as a separate process.

    Keeping it in a separate process avoids name collisions with run.py globals
    and prevents websocket / Alpaca polling failures from killing the main
    executor.
    """
    if not SPX_LAYER_ENABLED or not SPX_LAYER_INTERNAL_SUBPROCESS:
        return

    script = ensure_spx_price_action_layer_module()
    if script is None:
        return

    try:
        import subprocess

        args = [sys.executable, "-u", str(script)]

        # Train once and exit if explicitly requested from run.py.
        # Otherwise run live watcher loop.
        if SPX_LAYER_TRAIN_ON_START_FROM_RUN:
            args.extend(["--train-rl", "--once"])

        log_fh = SPX_LAYER_LOG_PATH.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            args,
            cwd=str(PROJECT_ROOT),
            env=_spx_layer_env(),
            stdout=log_fh,
            stderr=log_fh,
            text=True,
        )

        print(
            "[spx-layer] started "
            f"pid={proc.pid} script={script} symbol={SPX_TRADE_SYMBOL_FROM_RUN} "
            f"execute={SPX_LAYER_EXECUTE_FROM_RUN} log={SPX_LAYER_LOG_PATH}"
        )
        spx_env_snapshot = _spx_layer_env()
        print(
            "[spx-layer] tuned RL defaults "
            f"timesteps={spx_env_snapshot.get('SPX_LAYER_RL_TIMESTEPS')} "
            f"lr={spx_env_snapshot.get('SPX_LAYER_RL_LEARNING_RATE')} "
            f"entry_conf={spx_env_snapshot.get('SPX_LAYER_RL_MIN_ENTRY_CONFIDENCE')} "
            f"exit_conf={spx_env_snapshot.get('SPX_LAYER_RL_MIN_EXIT_CONFIDENCE')}"
        )

    except Exception as exc:
        print(f"[spx-layer] start failed: {exc}")
        try:
            write_event("spx_layer_start_failed", {"error": str(exc), "script": str(script)})
        except Exception:
            pass


def load_spx_layer_signal() -> dict[str, Any]:
    """
    Read the latest SPX layer signal if present.
    """
    path = OUT_DIR / f"{DATA_TAG}_spx_price_action_signal.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_spx_layer_orders() -> pd.DataFrame:
    """
    Convert the SPX layer's signal/output files into executor-compatible orders.

    The uploaded layer can submit orders itself when SPX_LAYER_EXECUTE=1. In the
    run.py integration, default SPX_LAYER_EXECUTE_FROM_RUN=0 means it only
    signals, and run.py merges BUY/SELL decisions into its normal execution
    path.
    """
    if not SPX_LAYER_ENABLED or not SPX_LAYER_MERGE_ORDERS:
        return pd.DataFrame()

    signal = load_spx_layer_signal()
    if not signal:
        return pd.DataFrame()

    action = str(signal.get("action", "")).upper().strip()
    symbol = str(signal.get("trade_symbol", SPX_TRADE_SYMBOL_FROM_RUN)).upper().strip()
    qty = int(float(signal.get("qty", 0) or 0))
    price = float(signal.get("price", 0.0) or 0.0)
    notional = float(signal.get("notional", 0.0) or 0.0)

    if symbol == "" or action not in {"BUY", "SELL"} or qty <= 0:
        return pd.DataFrame()

    side = "buy" if action == "BUY" else "sell"

    row = {
        "symbol": symbol,
        "side": side,
        "order_qty": qty,
        "price": price,
        "abs_delta_dollars": notional if notional > 0 else qty * price,
        "capped_dollars": notional if notional > 0 else qty * price,
        "confidence_used": float(signal.get("max_return_score", signal.get("setup_score", 0.0)) or 0.0),
        "confidence_multiplier": 1.0 + float(signal.get("max_return_score", signal.get("setup_score", 0.0)) or 0.0),
        "reason": "spx_price_action_layer:" + str(signal.get("reason", "")),
        "source": "spx_price_action_layer",
        "setup_type": signal.get("setup_type"),
        "entry_price": price,
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit"),
        "trailing_stop_enabled": True,
        "trail_percent": float(os.getenv("SPX_LAYER_TRAIL_PERCENT", "0.65")),
        "stream_confirmed": True,
        "spx_layer": True,
        "logical_symbol": signal.get("logical_symbol", "SPX"),
        "direction_score_1m": signal.get("direction_score_1m"),
        "direction_score_5m": signal.get("direction_score_5m"),
        "direction_score_15m": signal.get("direction_score_15m"),
        "max_return_score": signal.get("max_return_score"),
        "session_trade_used": signal.get("session_trade_used"),
        "rl_signal_path": str(OUT_DIR / f"{DATA_TAG}_spx_multibar_gain_rl_signal.json"),
        "rl_model_path": str(OUT_DIR / f"{DATA_TAG}_spx_multibar_gain_rl_model.zip"),
    }

    return pd.DataFrame([row])


def merge_spx_layer_orders(rebalance: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SPX layer orders into an existing rebalance order dataframe.

    Existing symbol rows win unless SPX emits an explicit SELL, because exits
    should always be allowed and should not be blocked by a same-symbol buy row.
    """
    spx_orders = load_spx_layer_orders()
    if spx_orders.empty:
        return rebalance

    if rebalance is None or rebalance.empty:
        print(f"[spx-layer] merged rows={len(spx_orders)}")
        return spx_orders.copy()

    out = rebalance.copy()
    for _, row in spx_orders.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        side = str(row.get("side", "")).lower()
        if not symbol:
            continue

        same = out["symbol"].astype(str).str.upper() == symbol if "symbol" in out.columns else pd.Series(False, index=out.index)

        if side == "sell":
            out = out[~same].copy()
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
        elif not same.any():
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    print(f"[spx-layer] merged rows={len(spx_orders)} total_rows={len(out)}")
    return out.reset_index(drop=True)


def spx_price_action_cycle(*args: Any, **kwargs: Any) -> None:
    """
    Compatibility wrapper for older run.py thread calls.

    If a prior edit still calls spx_price_action_cycle(), route it to the
    standalone module safely.
    """
    try:
        module_path = ensure_spx_price_action_layer_module()
        if module_path is None:
            return
        import importlib.util
        spec = importlib.util.spec_from_file_location("spx_price_action_layer", module_path)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "run_once"):
            mod.run_once()
    except Exception as exc:
        print(f"[spx-layer] cycle error: {exc}")


def spx_write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    try:
        write_event(event, payload or {})
    except Exception:
        pass


DEFAULT_REBALANCE_ORDERS = OUT_DIR / f"{DATA_TAG}_daily_walkforward_emulated_rebalance_orders.csv"
DEFAULT_EXIT_ORDERS = OUT_DIR / f"{DATA_TAG}_daily_walkforward_emulated_exit_orders.csv"

JOURNAL_PATH = LOG_DIR / f"{DATA_TAG}_executor_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / f"{DATA_TAG}_executor_equity_curve.csv"
EVENTS_JSONL = LOG_DIR / f"{DATA_TAG}_execution_events.jsonl"
KILL_SWITCH_PATH = LOG_DIR / f"{DATA_TAG}_kill_switch_state.json"
SCANNER_STATE_PATH = LOG_DIR / f"{DATA_TAG}_scanner_state.json"
SCANNER_SCORES_PATH = OUT_DIR / f"{DATA_TAG}_latest_scanner_scores.csv"
NO_REENTRY_STATE_PATH = LOG_DIR / f"{DATA_TAG}_no_reentry_state.json"
POSITION_MONITOR_PATH = LOG_DIR / f"{DATA_TAG}_active_position_monitor.csv"

LLM_SENTIMENT_UNIVERSE_PATH = OUT_DIR / f"{DATA_TAG}_latest_llm_sentiment_universe.json"
LLM_SENTIMENT_DISCOVERY_PATH = OUT_DIR / f"{DATA_TAG}_latest_llm_candidate_discovery.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"
LLM_SENTIMENT_SELECTOR_PATH = PROJECT_ROOT / "llm_sentiment_universe_selector.py"
MARKET_CONTEXT_BUILDER_PATH = PROJECT_ROOT / "market_context_builder.py"
PREMARKET_TRADE_PLANNER_PATH = PROJECT_ROOT / "premarket_trade_planner.py"
PRODUCTION_REGIME_SIZING_MODEL_PATH = PROJECT_ROOT / "production_regime_sizing_model.py"
RL_INTRADAY_STRUCTURE_TRAINER_PATH = PROJECT_ROOT / "rl_intraday_structure_trainer.py"
OPTIONS_BOOSTER_PATH = PROJECT_ROOT / "options_booster.py"
PREMARKET_DATA_COLLECTOR_PATH = PROJECT_ROOT / "premarket_data_collector.py"
PORTFOLIO_REBALANCER_PATH = PROJECT_ROOT / "portfolio_rebalancer.py"

MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"
PREMARKET_TRADE_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"
INTRADAY_STRUCTURE_MAP_PATH = OUT_DIR / f"{DATA_TAG}_intraday_structure_map.json"
INTRADAY_BARS_1M_PATH = OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv"
INTRADAY_TRAINING_DATASET_PATH = OUT_DIR / f"{DATA_TAG}_intraday_training_dataset.csv"
EXIT_TRAINING_DATASET_PATH = OUT_DIR / f"{DATA_TAG}_exit_training_dataset.csv"
MULTITIMEFRAME_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_multitimeframe_context.json"
FEATURE_STREAM_DIR = OUT_DIR / f"{DATA_TAG}_feature_stream"
FEATURE_STREAM_SYMBOL_STATE_PATH = FEATURE_STREAM_DIR / f"{DATA_TAG}_symbol_state.json"
FEATURE_STREAM_CSV_PATH = FEATURE_STREAM_DIR / f"{DATA_TAG}_intraday_feature_stream.csv"
FEATURE_STREAM_JSONL_PATH = FEATURE_STREAM_DIR / f"{DATA_TAG}_intraday_feature_stream.jsonl"
REGIME_POSITION_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_regime_position_plan.csv"
REGIME_UNIVERSE_PATH = OUT_DIR / f"{DATA_TAG}_regime_universe_30.csv"
REGIME_ACTIONS_PATH = OUT_DIR / f"{DATA_TAG}_regime_actions.csv"
RL_MODEL_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_structure_model.zip"
RL_METRICS_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_training_metrics.json"
RL_SIGNAL_LOG_PATH = LOG_DIR / f"{DATA_TAG}_rl_signal_log.csv"
RL_DIAGNOSTICS_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_signal_diagnostics.csv"
RL_SYMBOL_OVERLAP_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_symbol_overlap.csv"
LIVE_EXPERIENCE_JSONL_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_replay.jsonl"
LIVE_EXPERIENCE_CSV_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_replay.csv"
LIVE_EXPERIENCE_PARQUET_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_replay.parquet"
LIVE_EXPERIENCE_OUTCOME_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_outcomes.csv"
SESSION_PREP_STATE_PATH = LOG_DIR / f"{DATA_TAG}_session_planning_state.json"
OPTION_BOOSTER_CANDIDATES_PATH = OUT_DIR / f"{DATA_TAG}_latest_option_booster_candidates.csv"
DATED_OUTPUT_DIR = OUT_DIR / f"{DATA_TAG}_dated"
DATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_VALIDATION_STATE_PATH = OUT_DIR / f"{DATA_TAG}_data_validation_state.json"

ET = ZoneInfo("America/New_York")
EXECUTED_ORDERS_LOG = LOG_DIR / f"{DATA_TAG}_executed_orders.csv"
EXECUTED_ORDERS_JSONL = LOG_DIR / f"{DATA_TAG}_executed_orders.jsonl"

SPX_PRICE_ACTION_SIGNAL_PATH = OUT_DIR / f"{DATA_TAG}_spx_price_action_signal.json"
SPX_PRICE_ACTION_STATE_PATH = OUT_DIR / f"{DATA_TAG}_spx_price_action_state.json"
SPX_PRICE_ACTION_EVENTS_JSONL = LOG_DIR / f"{DATA_TAG}_spx_price_action_events.jsonl"
SPX_PRICE_ACTION_ORDERS_CSV = LOG_DIR / f"{DATA_TAG}_spx_price_action_orders.csv"
SPX_RL_PLAN_PATH = PROJECT_ROOT / "system-data" / f"{DATA_TAG}_SPX" / f"{DATA_TAG}_spx_intraday_breakout_plan.json"

# Stream-confirmed movement gate outputs.
STREAM_CONFIRM_SCRIPT_PATH = PROJECT_ROOT / "stream_confirmed_movement_gate.py"
PENDING_DIRECTION_WATCHLIST_PATH = LOG_DIR / "pending_direction_watchlist.csv"
STREAM_CONFIRMED_ORDERS_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_rebalance_orders.csv"
STREAM_CONFIRMED_TRADE_PLAN_CSV_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_trade_plan.csv"
STREAM_CONFIRMED_TRADE_PLAN_JSON_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_trade_plan.json"
STREAM_CONFIRM_STATE_PATH = LOG_DIR / "stream_confirmed_movement_state.json"
STREAM_CONFIRM_EVENTS_JSONL = LOG_DIR / "stream_confirmed_movement_events.jsonl"

# Support/resistance RL agent integration.
SR_RL_AGENT_PATH = PROJECT_ROOT / "sr_rl_agent.py"
SR_RL_MODEL_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_agent_ppo.zip"
SR_RL_SIGNALS_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_live_signals.json"
SR_RL_CONFIRMED_ORDERS_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_confirmed_rebalance_orders.csv"
SR_RL_EVENTS_JSONL = LOG_DIR / f"{DATA_TAG}_sr_rl_agent_events.jsonl"


# ======================================================================# Dynamic account-scaling defaults
# ======================================================================
POLL_SECONDS = 30
ORDER_WAIT_SECONDS = 20
COOLDOWN_SECONDS = 15
HEARTBEAT_SECONDS = 60

MIN_BUYING_POWER = 250.0
MIN_NOTIONAL_TO_TRADE = 75.0
MIN_WEIGHT_DELTA = 0.005

# Fallback defaults. Actual risk is dynamically scaled from account equity.
MIN_CASH_BUFFER = 1_500.0
DEFAULT_CASH_BUFFER_PCT = 0.12
MAX_GROSS_EXPOSURE = 0.72
MAX_POSITION_WEIGHT = 0.12
MAX_SINGLE_ORDER_NOTIONAL_PCT = 0.045
MAX_ORDER_DOLLARS = 1_500.0
MAX_SINGLE_ORDER_QTY = 25_000

DAILY_DRAWDOWN_LIMIT = -0.015
KILL_SWITCH_EQUITY_FLOOR = 11_700.0
MAX_DAILY_LOSS_PCT = 0.015
MAX_DAILY_LOSS_DOLLARS = 200.0

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DEFAULT_SCAN_UNIVERSE = [
    "QQQ", "TQQQ", "SQQQ", "SPY", "DIA", "IWM", "XLK", "SOXX", "SMH",
    "AMD", "NVDA", "AVGO", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
    "PLTR", "COIN", "MARA", "RIOT", "RGTI", "QBTS", "QUBT", "IONQ", "SOUN",
    "OKLO", "APLD", "ALAB", "ARM",
]


# ======================================================================# Dataclasses
# ======================================================================
@dataclass
class BrokerPosition:
    symbol: Optional[str] = None
    qty: float = 0.0
    market_value: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0


@dataclass
class AccountSnapshot:
    buying_power: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0


@dataclass
class AccountRiskProfile:
    band: str
    equity: float
    cash_buffer_pct: float
    min_cash_buffer: float
    max_gross_exposure: float
    max_position_weight: float
    max_single_order_notional_pct: float
    max_order_dollars: float
    max_daily_loss_pct: float
    max_daily_loss_dollars: float
    daily_drawdown_limit: float
    kill_switch_equity_floor: float
    min_notional_to_trade: float


@dataclass
class ScannerConfig:
    enabled: bool = True
    universe: Optional[list[str]] = None
    use_sp500_fallback: bool = False
    max_universe_symbols: int = 30
    lookback_days: int = 260
    data_feed: str = "iex"
    execution_alpha_min_score: float = float(os.getenv("EXECUTION_ALPHA_MIN_SCORE", "0.64"))
    execution_alpha_risk_off_size_mult: float = float(os.getenv("EXECUTION_ALPHA_RISK_OFF_SIZE_MULT", "0.50"))
    execution_alpha_risk_off_threshold_add: float = float(os.getenv("EXECUTION_ALPHA_RISK_OFF_THRESHOLD_ADD", "0.05"))

    # Forecast ensemble gates: ARIMA + LSTM + Stacked-LSTM + Attention-LSTM.
    forecast_ensemble_enabled: bool = True
    forecast_ensemble_soft_gate: bool = True
    forecast_min_ensemble_score: float = 0.62
    forecast_min_attention_lstm_score: float = 0.58
    forecast_min_expected_return: float = 0.0015
    forecast_max_drawdown_probability: float = 0.42

    # Broad-market mode. If Wikipedia retrieval fails, falls back to DEFAULT_SCAN_UNIVERSE.
    # Capped to avoid very large one-shot Alpaca requests.
    broad_market_scan: bool = True

    top_n: int = 30
    min_setup_score: float = 0.62
    min_buy_confidence: float = 0.58
    min_price: float = 2.0
    max_price: float = 500.0
    min_avg_dollar_volume: float = 5_000_000.0

    base_target_weight: float = 0.03
    max_target_weight: float = 0.10
    reversal_target_weight_mult: float = 0.75
    # Composite position-sizing weights: setup quality, entry confidence, liquidity.
    position_setup_score_weight: float = float(os.getenv("POSITION_SETUP_SCORE_WEIGHT", "0.50"))
    position_confidence_weight: float = float(os.getenv("POSITION_CONFIDENCE_WEIGHT", "0.35"))
    position_liquidity_weight: float = float(os.getenv("POSITION_LIQUIDITY_WEIGHT", "0.15"))

    stop_loss_pct: float = -0.025
    stale_position_score_exit: float = 0.45
    profit_lock_trigger_pct: float = 0.04
    profit_giveback_pct: float = 0.015

    require_positive_daily_move: bool = True
    require_positive_daily_move_for_breakouts: bool = os.getenv("REQUIRE_POSITIVE_DAILY_MOVE_FOR_BREAKOUTS", "1").lower() in {"1", "true", "yes"}
    allow_negative_daily_move_reversals: bool = os.getenv("ALLOW_NEGATIVE_DAILY_MOVE_REVERSALS", "1").lower() in {"1", "true", "yes"}
    reversal_min_reclaim_score: float = float(os.getenv("REVERSAL_MIN_RECLAIM_SCORE", "0.60"))
    avoid_bearish_regime_for_longs: bool = True
    allow_relaxed_fallback: bool = True
    fallback_min_setup_score: float = 0.52
    fallback_min_buy_confidence: float = 0.50
    max_overbought_rsi: float = 78.0
    min_regime_score_for_longs: float = 0.40
    

    # New-opportunity scanner: looks for stocks already green on the day
    # with positive momentum or confirmed breakout/reclaim behavior.
    opportunity_scan_enabled: bool = True
    min_daily_gain_pct: float = 0.005
    min_opportunity_score: float = 0.60
    min_breakout_confirmation_score: float = 0.55
    require_positive_momentum_or_breakout: bool = True

    # Intraday entry-analysis layer.
    analyze_intraday_entries: bool = True
    intraday_lookback_days: int = 5
    entry_near_support_buffer_pct: float = 0.003
    entry_breakout_buffer_pct: float = 0.002
    max_entry_chase_pct: float = 0.018

    # Pivot-point buying strategy.
    # This is the preferred intraday entry model: buy validated support pivots/zones
    # after price pulls into the zone and 1m/5m momentum confirms reversal.
    pivot_buy_enabled: bool = os.getenv("PIVOT_BUY_ENABLED", "1").lower() in {"1", "true", "yes"}
    pivot_scan_lookback_days: int = int(os.getenv("PIVOT_SCAN_LOOKBACK_DAYS", "3"))
    pivot_zone_tolerance_pct: float = float(os.getenv("PIVOT_ZONE_TOLERANCE_PCT", "0.003"))
    pivot_buy_zone_pct: float = float(os.getenv("PIVOT_BUY_ZONE_PCT", "0.004"))
    pivot_stop_buffer_pct: float = float(os.getenv("PIVOT_STOP_BUFFER_PCT", "0.004"))
    pivot_min_weight: float = float(os.getenv("PIVOT_MIN_WEIGHT", "5.0"))
    pivot_min_touch_count: int = int(os.getenv("PIVOT_MIN_TOUCH_COUNT", "2"))
    pivot_min_reversal_score: float = float(os.getenv("PIVOT_MIN_REVERSAL_SCORE", "0.60"))
    pivot_min_rel_volume: float = float(os.getenv("PIVOT_MIN_REL_VOLUME", "1.05"))
    pivot_min_rr: float = float(os.getenv("PIVOT_MIN_RR", "1.65"))
    pivot_target_rr: float = float(os.getenv("PIVOT_TARGET_RR", "2.0"))
    pivot_base_target_weight: float = float(os.getenv("PIVOT_BASE_TARGET_WEIGHT", "0.03"))
    pivot_max_target_weight: float = float(os.getenv("PIVOT_MAX_TARGET_WEIGHT", "0.07"))
    pivot_allow_market_entry_when_confirmed: bool = os.getenv("PIVOT_ALLOW_MARKET_ENTRY_WHEN_CONFIRMED", "1").lower() in {"1", "true", "yes"}
    pivot_block_bearish_regime: bool = os.getenv("PIVOT_BLOCK_BEARISH_REGIME", "1").lower() in {"1", "true", "yes"}

    # 8am 1H ORB / liquidity-sweep strategy.
    # Uses the 8:00-8:59 ET candle as the daily dealing range.
    # Long setup: sweep below 8am low -> reject/reclaim inside range -> bullish FVG -> enter pullback/retest.
    # Short setup: sweep above 8am high -> reject/back inside range -> bearish FVG -> optional short/exit.
    orb_8am_enabled: bool = os.getenv("ORB_8AM_ENABLED", "0").lower() in {"1", "true", "yes"}
    orb_8am_hour_et: int = int(os.getenv("ORB_8AM_HOUR_ET", "8"))
    orb_scan_lookback_days: int = int(os.getenv("ORB_SCAN_LOOKBACK_DAYS", "2"))
    orb_min_range_pct: float = float(os.getenv("ORB_MIN_RANGE_PCT", "0.0025"))
    orb_max_range_pct: float = float(os.getenv("ORB_MAX_RANGE_PCT", "0.14"))
    orb_sweep_buffer_pct: float = float(os.getenv("ORB_SWEEP_BUFFER_PCT", "0.0005"))
    orb_reclaim_buffer_pct: float = float(os.getenv("ORB_RECLAIM_BUFFER_PCT", "0.0002"))
    orb_fvg_min_gap_pct: float = float(os.getenv("ORB_FVG_MIN_GAP_PCT", "0.0004"))
    orb_retest_tolerance_pct: float = float(os.getenv("ORB_RETEST_TOLERANCE_PCT", "0.003"))
    orb_entry_max_age_minutes: int = int(os.getenv("ORB_ENTRY_MAX_AGE_MINUTES", "90"))
    orb_min_rr: float = float(os.getenv("ORB_MIN_RR", "2.0"))
    orb_target_rr: float = float(os.getenv("ORB_TARGET_RR", "2.0"))
    orb_base_target_weight: float = float(os.getenv("ORB_BASE_TARGET_WEIGHT", "0.025"))
    orb_max_target_weight: float = float(os.getenv("ORB_MAX_TARGET_WEIGHT", "0.06"))
    orb_allow_long_entries: bool = os.getenv("ORB_ALLOW_LONG_ENTRIES", "1").lower() in {"1", "true", "yes"}
    orb_allow_short_entries: bool = os.getenv("ORB_ALLOW_SHORT_ENTRIES", "0").lower() in {"1", "true", "yes"}
    orb_exit_longs_on_bearish_sweep: bool = os.getenv("ORB_EXIT_LONGS_ON_BEARISH_SWEEP", "1").lower() in {"1", "true", "yes"}
    orb_require_fvg_retest: bool = os.getenv("ORB_REQUIRE_FVG_RETEST", "1").lower() in {"1", "true", "yes"}
    orb_min_volume_rvol: float = float(os.getenv("ORB_MIN_VOLUME_RVOL", "1.05"))

    # Active-position monitoring. Every cycle, held symbols are checked on
    # 1m/5m/15m bars to ensure they are still moving in the right direction.
    active_position_monitor_enabled: bool = True
    active_monitor_lookback_days: int = 2
    # Momentum systems should prune quickly. These thresholds are intentionally
    # stricter than a swing-trading hold model.
    min_hold_direction_score: float = 0.55
    hard_exit_direction_score: float = 0.40
    intraday_trailing_stop_pct: float = 0.010
    failed_breakout_loss_pct: float = -0.004
    failed_breakout_buffer_pct: float = 0.002
    monitor_exit_on_vwap_loss: bool = True
    monitor_exit_on_lower_high_lower_low: bool = True
    monitor_exit_on_failed_breakout: bool = True

    # Multi-timeframe feature stream + exit-learning layer.
    # These consume files produced by build_intraday_training_dataset.py:
    #   system-data/feature_stream/YYYYMMDD_intraday_feature_stream.csv
    #   system-data/feature_stream/YYYYMMDD_symbol_state.json
    #   system-data/YYYYMMDD_multitimeframe_context.json
    feature_stream_enabled: bool = True
    feature_stream_max_age_minutes: int = int(os.getenv("FEATURE_STREAM_MAX_AGE_MINUTES", "5"))
    use_mtf_buy_gate: bool = True
    mtf_require_5m_or_15m_trend: bool = True
    mtf_require_price_above_vwap: bool = True
    mtf_block_buy_near_resistance_pct: float = 0.0025
    mtf_near_support_bonus_pct: float = 0.004

    exit_learning_enabled: bool = True
    exit_learning_min_hold_minutes: int = 15
    exit_learning_scale_out_fraction: float = 0.50
    exit_learning_take_profit_min_plpc: float = 0.006
    exit_learning_trail_stop_min_plpc: float = 0.003
    exit_learning_cut_loser_max_plpc: float = -0.004
    exit_learning_prob_giveback_threshold: float = 0.65
    exit_learning_prob_hold_threshold: float = 0.45
    exit_learning_reversal_threshold: float = 0.60
    exit_learning_negative_ev_threshold: float = 0.0

    # Churn control. Buys are blocked during the opening-noise window and
    # after a same-day exit, but sells/exits are always allowed.
    no_new_positions_first_minutes: int = 10
    symbol_reentry_cooldown_minutes: int = 30

    # Final pre-execution direction gate.
    # Buys must pass this immediately before order submission.
    # Sells/exits bypass this so risk can always be reduced.
    require_direction_confirmation_before_buy: bool = True
    min_pretrade_direction_score: float = 0.62
    min_pretrade_1m_score: float = 0.52
    min_pretrade_5m_score: float = 0.55
    min_pretrade_15m_score: float = 0.50
    require_price_above_vwap_for_buy: bool = True
    require_no_lower_low_for_buy: bool = True
    max_pretrade_pullback_pct: float = -0.006

    # ML next-bar direction prediction.
    ml_next_bar_enabled: bool = True
    ml_next_bar_timeframe_minutes: int = 5
    ml_next_bar_lookback_days: int = 20
    ml_next_bar_min_train_rows: int = 120
    ml_next_bar_min_buy_probability: float = 0.50
    ml_next_bar_min_pyramid_probability: float = 0.62
    ml_next_bar_max_exit_probability: float = 0.42

    # Pyramiding / loser scale-out behavior.
    pyramid_winners_enabled: bool = True
    pyramid_min_unrealized_plpc: float = 0.012
    pyramid_min_hold_direction_score: float = 0.62
    pyramid_add_weight: float = 0.025
    pyramid_max_adds_per_symbol_per_day: int = 2

    scale_out_losers_enabled: bool = True
    scale_out_loss_pct: float = -0.006
    scale_out_direction_score: float = 0.50
    scale_out_fraction: float = 0.50
    loser_full_exit_loss_pct: float = -0.015
    loser_full_exit_direction_score: float = 0.38

    # Universe source.
    # Default is LLM sentiment universe from llm_sentiment_universe_selector.py.
    use_llm_sentiment_universe: bool = True
    refresh_llm_sentiment_universe: bool = True
    min_llm_universe_confidence: float = 0.0

    # Morning orchestration / macro gate.
    market_context_enabled: bool = True
    refresh_market_context: bool = True
    halt_new_buys_on_market_context: bool = True
    allow_cautious_market_buys: bool = True

    # Premarket trade-plan gate.
    premarket_plan_enabled: bool = True
    refresh_premarket_trade_plan: bool = True
    require_valid_premarket_plan_for_buys: bool = False
    prefer_premarket_valid_symbols: bool = True
    max_plan_age_minutes: int = MAX_PLAN_AGE_MINUTES
    require_fresh_premarket_plan_for_buys: bool = REQUIRE_FRESH_PREMARKET_PLAN_FOR_BUYS
    require_structure_long_allowed_for_buys: bool = REQUIRE_STRUCTURE_LONG_ALLOWED_FOR_BUYS
    safe_mode_on_planner_failure: bool = SAFE_MODE_ON_PLANNER_FAILURE
    safe_mode_buy_size_multiplier: float = SAFE_MODE_BUY_SIZE_MULTIPLIER
    cautious_market_buy_size_multiplier: float = CAUTIOUS_MARKET_BUY_SIZE_MULTIPLIER
    cautious_market_confidence_addon: float = CAUTIOUS_MARKET_CONFIDENCE_ADDON
    mixed_market_buy_size_multiplier: float = MIXED_MARKET_BUY_SIZE_MULTIPLIER
    bearish_market_disable_new_longs: bool = BEARISH_MARKET_DISABLE_NEW_LONGS
    bearish_market_min_rl_confidence: float = BEARISH_MARKET_MIN_RL_CONFIDENCE
    use_tradingview_screener: bool = USE_TRADINGVIEW_SCREENER
    tradingview_require_rvol: float = TRADINGVIEW_REQUIRE_RVOL
    tradingview_max_planner_symbols: int = TRADINGVIEW_MAX_PLANNER_SYMBOLS

    # 4:00 AM continuous data collection + final session-planning pipeline.
    session_planning_enabled: bool = True
    session_planning_start_hour_et: int = 4
    session_planning_start_minute_et: int = 0
    session_planning_force_each_startup: bool = False

    # Between 4:00 AM and 9:30 AM ET, run.py can launch
    # premarket_data_collector.py and let it continuously refresh the live
    # planning files before normal market-open execution begins.
    premarket_data_collector_enabled: bool = True
    premarket_data_collector_block_until_market_open: bool = True
    premarket_data_collector_timeout_seconds: int = 21600
    premarket_data_collector_run_once_if_outside_window: bool = False

    # Validation/snapshots: keep latest files for the executor, but also write
    # dated copies so stale data can be detected and audited.
    dated_data_snapshots_enabled: bool = True
    require_today_data_for_market_open: bool = False
    max_premarket_file_age_minutes: int = 180

    run_market_context_builder: bool = True
    run_premarket_trade_planner: bool = True
    run_production_regime_sizing_model: bool = True

    # RL training is intentionally OFF by default. Train this model offline/on a
    # scheduled research job, then let run.py validate and load the saved model.
    run_rl_intraday_structure_trainer: bool = False
    allow_rl_training_from_run_py: bool = False
    session_planning_timeout_seconds: int = 900

    # Regime sizing + RL timing integration.
    use_regime_position_plan: bool = True
    prefer_regime_position_plan: bool = True
    use_rl_timing_gate: bool = True
    rl_min_enter_long_confidence: float = 0.65
    rl_min_exit_confidence: float = 0.55
    rl_min_add_long_confidence: float = 0.60
    rl_enable_add_long_gate: bool = True
    rl_min_enter_short_confidence: float = 0.70
    rl_gate_buys_only: bool = False

    # Fallback policy: when the RL model is missing/stale/invalid, run.py should
    # not hard-fail exits. Buys can be allowed to continue under regime/scanner
    # logic if rl_allow_hold_when_model_missing=True.
    rl_allow_hold_when_model_missing: bool = True
    rl_latest_bar_timeframe_minutes: int = 1
    rl_latest_bar_lookback_days: int = 5
    rl_position_aware_inference: bool = True

    # RL model validation gates. These guard against accidentally trading with a
    # stale, underfit, or failed training artifact.
    rl_validate_model_before_trading: bool = True
    rl_max_model_age_days: int = 14
    rl_metrics_max_age_days: int = 14
    rl_min_training_trades: int = 10
    rl_min_profit_factor: float = 0.75
    rl_min_win_rate: float = 0.30
    rl_require_symbol_overlap_file: bool = False
    rl_shadow_mode: bool = True
    rl_block_buys_when_model_invalid: bool = False


    live_experience_log_skips: bool = True
    live_experience_logging_enabled: bool = True
    live_experience_write_parquet: bool = True
    live_experience_flush_every_event: bool = True
    
    # Options booster handoff.
    options_booster_enabled: bool = True
    options_booster_confidence_threshold: float = 0.75
    options_booster_min_position_plpc: float = 0.0056
    options_booster_target_dte: int = 30
    options_booster_min_dte: int = 28
    options_booster_max_dte: int = 32
    options_booster_limit_price_source: str = "bid"

    # Alpaca options-data integration. options_booster.py should read option
    # chains from Alpaca's OptionHistoricalDataClient/OptionChainRequest instead
    # of Tradier/Yahoo. run.py passes these as environment variables so the
    # booster can run without CLI arguments and still know the intended data source.
    options_data_source: str = "alpaca"
    options_require_alpaca_chain: bool = True
    options_alpaca_chain_preflight_enabled: bool = True
    options_flow_scan_enabled: bool = True
    options_flow_min_volume: int = 10
    options_flow_min_open_interest: int = 25
    options_flow_min_volume_oi_ratio: float = 0.25
    options_flow_min_premium_dollars: float = 25_000.0
    options_flow_max_underlyings: int = 30
    options_flow_max_contracts_per_underlying: int = 500

    # ML-style call selection: rank option contracts using Greeks + liquidity + TA.
    # This is a deterministic model/ranker by default; if options_booster.py exposes
    # richer chain data, these fields let that module select the best contract.
    options_ml_contract_selection_enabled: bool = True
    options_target_delta: float = 0.52
    options_min_delta: float = 0.40
    options_max_delta: float = 0.65
    options_max_spread_pct: float = 0.15
    options_min_open_interest: int = 50
    options_min_volume: int = 5
    options_max_theta_abs: float = 0.12
    options_min_gamma: float = 0.005
    options_max_iv_rank: float = 0.85
    options_greek_score_weight: float = 0.35
    options_technical_score_weight: float = 0.35
    options_liquidity_score_weight: float = 0.20
    options_dte_score_weight: float = 0.10
    options_booster_require_technical_confirmation: bool = True
    options_booster_min_direction_score: float = 0.64
    options_booster_min_ml_next_bar_probability: float = 0.60
    options_booster_once_per_cycle: bool = True
    options_booster_timeout_seconds: int = 180


    # Dedicated SPX/SPY price-action execution layer.
    # Runs independently from the main equity scanner and watches SPX-style
    # price action through a tradable proxy, SPY by default.
    spx_price_action_layer_enabled: bool = True
    spx_price_action_trade_symbol: str = "SPY"
    spx_price_action_logical_symbol: str = "SPX"
    spx_price_action_execute_orders: bool = True
    spx_price_action_scan_seconds: int = 10
    spx_price_action_data_feed: str = "iex"
    spx_price_action_lookback_days: int = 8
    spx_price_action_sr_lookback_days: int = 20
    spx_price_action_orb_minutes: int = 15
    spx_price_action_min_setup_score: float = 0.68
    spx_price_action_min_breakout_score: float = 0.72
    spx_price_action_min_reclaim_score: float = 0.70
    spx_price_action_min_buying_power: float = 250.0
    spx_price_action_base_order_dollars: float = 1500.0
    spx_price_action_max_position_dollars: float = 3500.0
    spx_price_action_max_position_weight: float = 0.08
    spx_price_action_min_order_dollars: float = 100.0
    spx_price_action_stop_loss_pct: float = 0.006
    spx_price_action_take_profit_r: float = 2.0
    spx_price_action_trailing_giveback_pct: float = 0.0045
    spx_price_action_min_trail_profit_pct: float = 0.004
    spx_price_action_daily_loss_limit_pct: float = 0.012
    spx_price_action_daily_loss_limit_dollars: float = 750.0
    spx_price_action_avoid_first_minutes: int = 5
    spx_price_action_avoid_last_minutes: int = 10
    spx_price_action_allow_market_context_caution: bool = True
    spx_price_action_use_rl_plan_confirmation: bool = True
    spx_price_action_require_rl_for_entry: bool = False

    # Stream-confirmed movement gate.
    # If the final pretrade direction gate rejects a BUY, the symbol is not
    # discarded. It is added to pending_direction_watchlist.csv. The stream
    # confirmation layer watches all day symbols and writes confirmed trade
    # plans/rebalance rows once early multibar movement is detected.
    stream_confirmed_movement_enabled: bool = True
    stream_confirmed_movement_script: str = "stream_confirmed_movement_gate.py"
    stream_confirmed_use_polling: bool = True
    stream_confirmed_execute_now: bool = False
    stream_confirmed_reload_seconds: int = 10
    stream_confirmed_merge_orders: bool = True
    stream_confirmed_append_direction_rejects: bool = True
    stream_confirmed_require_trade_plan: bool = False
    stream_confirmed_trailing_stop_enabled: bool = True
    stream_confirmed_submit_trailing_stop_after_fill: bool = True
    stream_confirmed_trail_percent_default: float = 0.65
    stream_confirmed_take_profit_r_default: float = 2.0

    # Support/resistance RL agent.
    # sr_rl_agent.py learns 1H/4H support and resistance zones, then emits
    # live BUY/WAIT/SELL decisions into YYYYMMDD_sr_rl_live_signals.json.
    # run.py consumes BUY decisions as additional long-only order candidates.
    sr_rl_agent_enabled: bool = True
    sr_rl_agent_script: str = "sr_rl_agent.py"
    sr_rl_train_on_start: bool = False
    sr_rl_live_enabled: bool = True
    sr_rl_execute_now: bool = False
    sr_rl_merge_signals: bool = True
    sr_rl_require_model: bool = False
    sr_rl_scan_seconds: int = 10
    sr_rl_signal_max_age_seconds: int = 90
    sr_rl_min_qty: int = 1
    sr_rl_default_confidence: float = 0.66
    sr_rl_max_order_dollars: float = 6500.0
    sr_rl_submit_trailing_stop_after_fill: bool = True
    sr_rl_trail_percent_default: float = 0.75


@dataclass
class ExecutorConfig:
    paper: bool = True
    live: bool = False

    poll_once: bool = False
    poll_seconds: int = POLL_SECONDS

    rebalance_orders_path: Path = DEFAULT_REBALANCE_ORDERS
    exit_orders_path: Path = DEFAULT_EXIT_ORDERS

    require_market_open: bool = True
    wait_for_market_open: bool = False
    allow_closed_market: bool = False
    max_wait_for_open_minutes: int = 180

    cancel_open_orders_first: bool = False
    sell_first: bool = True

    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT
    min_cash_buffer: float = MIN_CASH_BUFFER
    min_buying_power: float = MIN_BUYING_POWER
    min_notional_to_trade: float = MIN_NOTIONAL_TO_TRADE
    min_weight_delta: float = MIN_WEIGHT_DELTA

    max_gross_exposure: float = MAX_GROSS_EXPOSURE
    max_position_weight: float = MAX_POSITION_WEIGHT
    max_single_order_notional_pct: float = MAX_SINGLE_ORDER_NOTIONAL_PCT
    max_order_dollars: float = MAX_ORDER_DOLLARS
    max_single_order_qty: int = MAX_SINGLE_ORDER_QTY

    cooldown_seconds: int = COOLDOWN_SECONDS

    enable_kill_switch: bool = True
    daily_drawdown_limit: float = DAILY_DRAWDOWN_LIMIT
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT
    max_daily_loss_dollars: float = MAX_DAILY_LOSS_DOLLARS
    kill_switch_equity_floor: float = KILL_SWITCH_EQUITY_FLOOR

    order_wait_seconds: int = ORDER_WAIT_SECONDS
    time_in_force: TimeInForce = TimeInForce.DAY
    order_tag_prefix: str = "qq_13k"

    scanner: ScannerConfig = field(default_factory=ScannerConfig)


# ======================================================================# Helpers
# ======================================================================
def seconds_until_market_open_plus_15() -> float:
    """
    Returns seconds until 9:45 AM ET.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))

    target = now_et.replace(
        hour=9,
        minute=45,
        second=0,
        microsecond=0,
    )

    # if already past 9:45 today, schedule tomorrow
    if now_et >= target:
        target = target.replace(day=target.day + 1)

    return max((target - now_et).total_seconds(), 0.0)


def launch_intraday_dataset_builder():
    """
    Wait until 15 minutes after market open,
    then launch the training dataset builder.
    """
    wait_seconds = seconds_until_market_open_plus_15()

    print(
        f"[scheduler] build_intraday_training_dataset.py "
        f"scheduled in {wait_seconds / 60:.2f} minutes"
    )

    time.sleep(wait_seconds)

    print("[scheduler] starting intraday dataset builder...")

    subprocess.Popen(
        [
            "python3",
            "build_intraday_training_dataset.py",
        ],
        cwd=str(PROJECT_ROOT),
    )
    
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return now_utc().strftime("%Y%m%d_%H%M%S")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def safe_symbol(value: Any) -> str:
    symbol = str(value or "").upper().strip().replace(".", "-")
    return re.sub(r"[^A-Z0-9\-]", "", symbol)


def normalize_symbol(symbol: Any) -> str:
    return safe_symbol(symbol)


def is_simple_alpaca_symbol(symbol: Any) -> bool:
    """
    Keep only simple Alpaca-friendly stock symbols.

    Excludes class-share / special symbols such as BF.B, BRK.B, BF-B,
    warrants, units, preferreds, and other symbols that commonly break
    Alpaca historical bar requests.
    """
    raw = str(symbol or "").upper().strip()
    normalized = safe_symbol(raw)
    if not normalized:
        return False
    if "." in raw or "/" in raw or "^" in raw or " " in raw:
        return False
    if "-" in normalized:
        return False
    return bool(re.fullmatch(r"[A-Z]{1,5}", normalized))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def safe_json(value: Any, default: Any = None) -> Any:
    """Return a JSON-serializable version of value.

    This is used by the ES/forecast layers before json.dumps(). It prevents
    failures from numpy, pandas, dataclasses, Path, datetime, enums, and other
    non-standard objects.
    """
    if default is None:
        default = {}
    try:
        if value is None:
            return default
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [safe_json(x, default=None) for x in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Series):
            return {str(k): safe_json(v, default=None) for k, v in value.to_dict().items()}
        if isinstance(value, pd.DataFrame):
            return [safe_json(row, default=None) for row in value.to_dict(orient="records")]
        if isinstance(value, dict):
            return {str(k): safe_json(v, default=None) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [safe_json(v, default=None) for v in value]
        if hasattr(value, "value"):
            return safe_json(value.value, default=default)
        if hasattr(value, "to_dict"):
            return safe_json(value.to_dict(), default=default)
        if hasattr(value, "__dict__"):
            return safe_json(vars(value), default=default)
        return str(value)
    except Exception:
        return default


def is_auth_error(exc: Any) -> bool:
    msg = str(exc).lower()
    return (
        "unauthorized" in msg
        or "401" in msg
        or "forbidden" in msg
        or "invalid api key" in msg
        or "authentication" in msg
    )


def alpaca_auth_help_message(paper: bool = True) -> str:
    mode = "paper" if paper else "live"
    base_url = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL") or ""
    key = os.getenv("APCA_API_KEY_ID") or ""
    return (
        f"Alpaca auth failed for {mode} trading. Check .env: "
        f"APCA_API_KEY_ID/APCA_API_SECRET_KEY must be a matching {mode} key pair; "
        f"paper={paper}; APCA_API_BASE_URL={base_url or 'not_set'}; "
        f"key_prefix={key[:4] + '...' if key else 'missing'}"
    )


def fallback_market_context(reason: str = "market_context_unavailable") -> dict[str, Any]:
    """Conservative context used when market_context_builder/OpenAI/Alpaca fails."""
    payload = {
        "created_at_utc": now_utc().isoformat() if "now_utc" in globals() else datetime.now(timezone.utc).isoformat(),
        "overall_market_bias": "mixed",
        "trade_permission": "cautious",
        "risk_level": "medium",
        "market_context_source": "run_py_fallback",
        "reason": str(reason),
        "notes": [
            "Fallback market context was used because the normal builder failed or timed out.",
            "Do not block exits. Keep new entries conservative until market context refresh succeeds.",
        ],
    }
    try:
        MARKET_CONTEXT_PATH.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    except Exception:
        pass
    return payload


def build_account_risk_profile(equity: float) -> AccountRiskProfile:
    """Scale risk controls from a small account to roughly $1M automatically."""
    equity = max(float(equity), 1.0)

    if equity < 25_000:
        band = "MICRO_UNDER_PDT"
        cash_buffer_pct = 0.12
        min_cash_buffer = max(1_500.0, equity * 0.10)
        max_gross_exposure = 0.72
        max_position_weight = 0.12
        max_single_order_notional_pct = 0.045
        max_order_dollars = min(1_500.0, equity * 0.12)
        max_daily_loss_pct = 0.015
        max_daily_loss_dollars = min(250.0, equity * 0.018)
        daily_drawdown_limit = -0.015
        kill_floor_drawdown = 0.10
        min_notional = 75.0
    elif equity < 100_000:
        band = "SMALL"
        cash_buffer_pct = 0.10
        min_cash_buffer = max(2_500.0, equity * 0.07)
        max_gross_exposure = 0.80
        max_position_weight = 0.14
        max_single_order_notional_pct = 0.055
        max_order_dollars = min(5_000.0, equity * 0.08)
        max_daily_loss_pct = 0.018
        max_daily_loss_dollars = min(1_250.0, equity * 0.018)
        daily_drawdown_limit = -0.018
        kill_floor_drawdown = 0.10
        min_notional = 100.0
    elif equity < 250_000:
        band = "GROWTH"
        cash_buffer_pct = 0.08
        min_cash_buffer = max(7_500.0, equity * 0.05)
        max_gross_exposure = 0.88
        max_position_weight = 0.16
        max_single_order_notional_pct = 0.06
        max_order_dollars = min(15_000.0, equity * 0.07)
        max_daily_loss_pct = 0.018
        max_daily_loss_dollars = min(3_000.0, equity * 0.018)
        daily_drawdown_limit = -0.018
        kill_floor_drawdown = 0.09
        min_notional = 250.0
    elif equity < 500_000:
        band = "PRO"
        cash_buffer_pct = 0.07
        min_cash_buffer = max(15_000.0, equity * 0.045)
        max_gross_exposure = 0.92
        max_position_weight = 0.16
        max_single_order_notional_pct = 0.055
        max_order_dollars = min(25_000.0, equity * 0.06)
        max_daily_loss_pct = 0.016
        max_daily_loss_dollars = min(5_000.0, equity * 0.016)
        daily_drawdown_limit = -0.016
        kill_floor_drawdown = 0.08
        min_notional = 500.0
    else:
        band = "INSTITUTIONAL_1M"
        cash_buffer_pct = 0.06
        min_cash_buffer = max(35_000.0, equity * 0.04)
        max_gross_exposure = 0.95
        max_position_weight = 0.14
        max_single_order_notional_pct = 0.04
        max_order_dollars = min(40_000.0, equity * 0.045)
        max_daily_loss_pct = 0.0125
        max_daily_loss_dollars = min(10_000.0, equity * 0.0125)
        daily_drawdown_limit = -0.0125
        kill_floor_drawdown = 0.07
        min_notional = 1_000.0

    return AccountRiskProfile(
        band=band,
        equity=equity,
        cash_buffer_pct=cash_buffer_pct,
        min_cash_buffer=min_cash_buffer,
        max_gross_exposure=max_gross_exposure,
        max_position_weight=max_position_weight,
        max_single_order_notional_pct=max_single_order_notional_pct,
        max_order_dollars=max_order_dollars,
        max_daily_loss_pct=max_daily_loss_pct,
        max_daily_loss_dollars=max_daily_loss_dollars,
        daily_drawdown_limit=daily_drawdown_limit,
        kill_switch_equity_floor=equity * (1.0 - kill_floor_drawdown),
        min_notional_to_trade=min_notional,
    )


def run_python_script(
    path: Path,
    args: Optional[list[str]] = None,
    timeout_seconds: int = 180,
    env_overrides: Optional[dict[str, str]] = None,
) -> tuple[bool, str]:
    if not path.exists():
        return False, f"script_not_found: {path}"

    cmd = [sys.executable, str(path)] + list(args or [])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"timeout_after_{timeout_seconds}_seconds"
    except Exception as exc:
        return False, str(exc)


def parse_hhmm_et(value: str) -> tuple[int, int]:
    hour, minute = str(value).strip().split(":")
    return int(hour), int(minute)


def premarket_collection_window_et() -> tuple[datetime, datetime]:
    """Return today's configured collector window in ET."""
    now = datetime.now(ET)
    start_raw = os.getenv("PREMARKET_COLLECTOR_START_ET", "04:00")
    end_raw = os.getenv("PREMARKET_COLLECTOR_END_ET", "09:30")
    sh, sm = parse_hhmm_et(start_raw)
    eh, em = parse_hhmm_et(end_raw)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start, end


def premarket_collection_window_is_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    start, end = premarket_collection_window_et()
    return start <= now < end


def wait_until_premarket_collection_window(config: ScannerConfig) -> None:
    """If run.py starts before 4 AM ET, wait for the premarket collector window."""
    if not config.premarket_data_collector_enabled:
        return

    now = datetime.now(ET)
    start, end = premarket_collection_window_et()

    if now.weekday() >= 5 or now >= end:
        return

    while datetime.now(ET) < start:
        seconds_left = max(1.0, (start - datetime.now(ET)).total_seconds())
        print(
            f"[premarket-collector] waiting until {start.strftime('%H:%M')} ET; "
            f"{int(seconds_left // 60)} minute(s) left"
        )
        time.sleep(min(300, max(5, seconds_left)))


def dated_snapshot_name(path: Path, date_et: str | None = None) -> Path:
    date_et = date_et or datetime.now(ET).date().isoformat()
    suffix = path.suffix
    stem = path.stem
    return DATED_OUTPUT_DIR / date_et / f"{stem}_{date_et}{suffix}"


def snapshot_data_file(path: Path, date_et: str | None = None) -> dict[str, Any]:
    """Copy a latest data file into system-data/dated/YYYY-MM-DD/ for audit/validation."""
    date_et = date_et or datetime.now(ET).date().isoformat()
    payload = {
        "path": str(path),
        "exists": path.exists(),
        "snapshot_path": "",
        "ok": False,
        "reason": "",
    }
    if not path.exists():
        payload["reason"] = "missing"
        return payload
    try:
        target = dated_snapshot_name(path, date_et)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        payload.update({"snapshot_path": str(target), "ok": True, "reason": "snapshotted"})
        return payload
    except Exception as exc:
        payload["reason"] = str(exc)
        return payload


def validate_data_file_freshness(path: Path, max_age_minutes: int = 180) -> dict[str, Any]:
    now = now_utc()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "mtime_utc": "",
        "age_minutes": None,
        "date_et": datetime.now(ET).date().isoformat(),
        "ok": False,
        "reason": "missing",
    }
    if not path.exists():
        return payload
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_minutes = (now - mtime).total_seconds() / 60.0
        mtime_et = mtime.astimezone(ET)
        today_et = datetime.now(ET).date()
        ok = mtime_et.date() == today_et and age_minutes <= max_age_minutes
        payload.update(
            {
                "mtime_utc": mtime.isoformat(),
                "mtime_et": mtime_et.isoformat(),
                "age_minutes": round(age_minutes, 3),
                "ok": bool(ok),
                "reason": "fresh_today" if ok else "stale_or_wrong_date",
            }
        )
    except Exception as exc:
        payload["reason"] = str(exc)
    return payload


def snapshot_and_validate_planning_files(config: ScannerConfig, stage: str) -> dict[str, Any]:
    """Date-stamp and validate all key planning/data files used by run.py."""
    date_et = datetime.now(ET).date().isoformat()
    files = [
        LLM_SENTIMENT_UNIVERSE_PATH,
        LLM_SENTIMENT_DISCOVERY_PATH,
        MARKET_CONTEXT_PATH,
        PREMARKET_TRADE_PLAN_PATH,
        INTRADAY_STRUCTURE_MAP_PATH,
        INTRADAY_BARS_1M_PATH,
        REGIME_POSITION_PLAN_PATH,
        REGIME_UNIVERSE_PATH,
        REGIME_ACTIONS_PATH,
        OPTION_BOOSTER_CANDIDATES_PATH,
        OUT_DIR / "latest_option_flow_candidates.csv",
        OUT_DIR / "latest_option_booster_ranked_contracts.csv",
    ]
    snapshots = []
    validations = []
    for path in files:
        validations.append(validate_data_file_freshness(path, config.max_premarket_file_age_minutes))
        if config.dated_data_snapshots_enabled:
            snapshots.append(snapshot_data_file(path, date_et))

    summary = {
        "stage": stage,
        "date_et": date_et,
        "created_at_utc": now_utc().isoformat(),
        "max_age_minutes": int(config.max_premarket_file_age_minutes),
        "validations": validations,
        "snapshots": snapshots,
        "all_required_fresh": all(v.get("ok") for v in validations if Path(v.get("path", "")).name in {
            "market_context.json",
            "premarket_trade_plan.json",
            "intraday_structure_map.json",
            "regime_position_plan.csv",
        }),
    }
    write_json_file(DATA_VALIDATION_STATE_PATH, summary)
    write_event("planning_data_validated", summary)
    print(
        f"[data-validation] stage={stage} "
        f"fresh_required={summary['all_required_fresh']} "
        f"snapshots={sum(1 for s in snapshots if s.get('ok'))}/{len(snapshots)}"
    )
    return summary


def run_premarket_data_collector(config: ScannerConfig) -> dict[str, Any]:
    """
    Start premarket_data_collector.py before the market-open wait.

    The collector blocks until 9:30 ET by default while it repeatedly refreshes
    the LLM universe, market context, premarket plan, intraday bars, and options
    flow. Its stdout streams into run.py/systemd logs.
    """
    payload: dict[str, Any] = {
        "enabled": bool(config.premarket_data_collector_enabled),
        "collector_path": str(PREMARKET_DATA_COLLECTOR_PATH),
        "started": False,
        "ok": False,
        "returncode": None,
        "reason": "",
    }

    if not config.premarket_data_collector_enabled:
        payload["reason"] = "premarket_data_collector_disabled"
        return payload

    if not PREMARKET_DATA_COLLECTOR_PATH.exists():
        payload["reason"] = f"collector_script_missing:{PREMARKET_DATA_COLLECTOR_PATH}"
        print(f"[premarket-collector] {payload['reason']}")
        write_event("premarket_collector_missing", payload)
        return payload

    wait_until_premarket_collection_window(config)

    if not premarket_collection_window_is_open() and not config.premarket_data_collector_run_once_if_outside_window:
        payload["reason"] = "outside_premarket_collection_window"
        print("[premarket-collector] outside 04:00-09:30 ET window; skipping continuous collector")
        write_event("premarket_collector_skipped", payload)
        snapshot_and_validate_planning_files(config, stage="collector_skipped")
        return payload

    start, end = premarket_collection_window_et()
    print(
        "[premarket-collector] starting continuous collection | "
        f"window={start.strftime('%H:%M')}-{end.strftime('%H:%M')} ET "
        f"path={PREMARKET_DATA_COLLECTOR_PATH}"
    )
    write_event(
        "premarket_collector_started",
        {
            "path": str(PREMARKET_DATA_COLLECTOR_PATH),
            "window_start_et": start.isoformat(),
            "window_end_et": end.isoformat(),
        },
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if config.premarket_data_collector_run_once_if_outside_window:
        env["PREMARKET_RUN_ONCE_IF_OUTSIDE_WINDOW"] = "1"

    cmd = [sys.executable, str(PREMARKET_DATA_COLLECTOR_PATH)]
    started_at = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        payload["started"] = True

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip())

        returncode = proc.wait(timeout=max(1, int(config.premarket_data_collector_timeout_seconds)))
        elapsed = time.time() - started_at
        payload.update(
            {
                "ok": returncode == 0,
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
                "reason": "completed" if returncode == 0 else "nonzero_returncode",
            }
        )
        print(f"[premarket-collector] completed returncode={returncode} elapsed={elapsed:.2f}s")
        write_event("premarket_collector_completed", payload)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - started_at
        try:
            proc.terminate()  # type: ignore[name-defined]
            proc.wait(timeout=20)  # type: ignore[name-defined]
        except Exception:
            try:
                proc.kill()  # type: ignore[name-defined]
            except Exception:
                pass
        payload.update(
            {
                "ok": False,
                "returncode": -1,
                "elapsed_seconds": round(elapsed, 3),
                "reason": f"timeout_after_{config.premarket_data_collector_timeout_seconds}s",
            }
        )
        print(f"[premarket-collector] timeout elapsed={elapsed:.2f}s")
        write_event("premarket_collector_timeout", payload)

    except Exception as exc:
        elapsed = time.time() - started_at
        payload.update(
            {
                "ok": False,
                "returncode": -2,
                "elapsed_seconds": round(elapsed, 3),
                "reason": str(exc),
            }
        )
        print(f"[premarket-collector] error: {exc}")
        write_event("premarket_collector_error", payload)

    snapshot_and_validate_planning_files(config, stage="after_premarket_collector")
    return payload


def alpaca_options_data_available(sample_symbol: str = "SPY") -> tuple[bool, str]:
    """Verify Alpaca option-chain access before triggering options_booster.py."""
    if OptionHistoricalDataClient is None or OptionChainRequest is None:
        return False, "alpaca_option_data_classes_unavailable"

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        return False, "missing_alpaca_credentials"

    sample_symbol = normalize_symbol(sample_symbol) or "SPY"
    try:
        client = OptionHistoricalDataClient(key, secret)
        chain = client.get_option_chain(OptionChainRequest(underlying_symbol=sample_symbol))
        if isinstance(chain, dict):
            count = len(chain)
        elif hasattr(chain, "data") and isinstance(chain.data, dict):
            count = len(chain.data)
        else:
            try:
                count = len(dict(chain)) if chain is not None else 0
            except Exception:
                count = 0
        if count <= 0:
            return False, f"empty_alpaca_option_chain:{sample_symbol}"
        return True, f"alpaca_option_chain_ok:{sample_symbol}:contracts={count}"
    except Exception as exc:
        return False, f"alpaca_option_chain_error:{sample_symbol}:{exc}"


def build_options_booster_env(config: ScannerConfig, candidate_symbols: list[str] | None = None) -> dict[str, str]:
    """Environment contract for the Alpaca-backed options_booster.py."""
    env = {
        "OPTIONS_BOOSTER_ENABLED": "1" if config.options_booster_enabled else "0",
        "OPTIONS_DATA_SOURCE": str(config.options_data_source or "alpaca").lower(),
        "OPTIONS_USE_ALPACA": "1",
        "OPTIONS_REQUIRE_ALPACA_CHAIN": "1" if config.options_require_alpaca_chain else "0",
        "OPTIONS_REQUIRE_OPEN_UNDERLYING_POSITION": "1",
        "OPTIONS_MIN_UNDERLYING_PLPC": str(config.options_booster_min_position_plpc),
        "OPTIONS_MIN_DTE": str(config.options_booster_min_dte),
        "OPTIONS_MAX_DTE": str(config.options_booster_max_dte),
        "OPTIONS_TARGET_DTE": str(config.options_booster_target_dte),
        "OPTIONS_IDEAL_DELTA": str(config.options_target_delta),
        "OPTIONS_MIN_DELTA": str(config.options_min_delta),
        "OPTIONS_MAX_DELTA": str(config.options_max_delta),
        "OPTIONS_MAX_SPREAD_PCT": str(config.options_max_spread_pct),
        "OPTIONS_MIN_OPEN_INTEREST": str(max(config.options_min_open_interest, config.options_flow_min_open_interest)),
        "OPTIONS_MIN_VOLUME": str(max(config.options_min_volume, config.options_flow_min_volume)),
        "OPTIONS_MAX_THETA_ABS": str(config.options_max_theta_abs),
        "OPTIONS_MIN_GAMMA": str(config.options_min_gamma),
        "OPTIONS_LIMIT_PRICE_SOURCE": str(config.options_booster_limit_price_source),
        "OPTIONS_FLOW_SCAN_ENABLED": "1" if config.options_flow_scan_enabled else "0",
        "OPTIONS_FLOW_MAX_UNDERLYINGS": str(config.options_flow_max_underlyings),
        "OPTIONS_FLOW_MAX_CONTRACTS_PER_UNDERLYING": str(config.options_flow_max_contracts_per_underlying),
        "OPTIONS_MIN_VOLUME_OI_RATIO": str(config.options_flow_min_volume_oi_ratio),
        "OPTIONS_MIN_PREMIUM_FLOW_DOLLARS": str(config.options_flow_min_premium_dollars),
        "APCA_PAPER": "1",
    }
    if candidate_symbols:
        env["OPTIONS_CANDIDATE_SYMBOLS"] = ",".join(candidate_symbols)
    if os.getenv("APCA_API_KEY_ID"):
        env["APCA_API_KEY_ID"] = os.getenv("APCA_API_KEY_ID", "")
    if os.getenv("APCA_API_SECRET_KEY"):
        env["APCA_API_SECRET_KEY"] = os.getenv("APCA_API_SECRET_KEY", "")
    return env


def refresh_llm_sentiment_universe(timeout_seconds: int = 180) -> None:
    """Refresh latest_llm_sentiment_universe.json and locked_trading_universe.txt."""
    try:
        from llm_sentiment_universe_selector import build_sentiment_universe, write_outputs

        payload = build_sentiment_universe()
        write_outputs(payload)
        print(f"[llm-universe] refreshed via import symbols={len(payload.get('symbols', []))}")
        write_event("llm_universe_refreshed", {"symbols": payload.get("symbols", [])})
        return
    except Exception as exc:
        print(f"[llm-universe] import refresh failed; trying subprocess: {exc}")

    ok, output = run_python_script(LLM_SENTIMENT_SELECTOR_PATH, timeout_seconds=timeout_seconds)
    if ok:
        print("[llm-universe] refreshed via subprocess")
        if output:
            print(output[-2000:])
        write_event("llm_universe_refreshed_subprocess", {"output_tail": output[-2000:]})
    else:
        print(f"[llm-universe] refresh failed: {output}")
        write_event("llm_universe_refresh_failed", {"error": output})



def refresh_market_context(timeout_seconds: int = 240) -> dict[str, Any]:
    """Build daily market_context.json using market_context_builder.py.

    The previous import path could hang inside OpenAI/httpx and block startup.
    This version runs the builder as a subprocess so timeout_seconds is enforced.
    If Alpaca/OpenAI fails, it writes and returns a conservative fallback context.
    """
    timeout_seconds = max(15, int(timeout_seconds or 240))

    env_overrides = {
        "OPENAI_TIMEOUT_SECONDS": os.getenv("OPENAI_TIMEOUT_SECONDS", "30"),
        "OPENAI_MAX_RETRIES": os.getenv("OPENAI_MAX_RETRIES", "1"),
        "MARKET_CONTEXT_FALLBACK_ON_ERROR": "1",
        "RUN_DATA_DATE": DATA_TAG,
    }

    ok, output = run_python_script(
        MARKET_CONTEXT_BUILDER_PATH,
        timeout_seconds=timeout_seconds,
        env_overrides=env_overrides,
    )

    if ok:
        print("[market-context] refreshed via subprocess")
        if output:
            print(output[-2000:])
        payload = read_json_file(MARKET_CONTEXT_PATH, {})
        write_event(
            "market_context_refreshed_subprocess",
            {
                "overall_market_bias": payload.get("overall_market_bias"),
                "trade_permission": payload.get("trade_permission"),
                "risk_level": payload.get("risk_level"),
                "output_tail": output[-2000:] if output else "",
            },
        )
        return payload or fallback_market_context("market_context_builder_returned_no_payload")

    reason = output or "market_context_builder_failed"
    if is_auth_error(reason):
        reason = f"{reason} | {alpaca_auth_help_message(paper=True)}"

    print(f"[market-context] refresh failed; using fallback: {reason}")
    write_event("market_context_refresh_failed_using_fallback", {"error": reason})
    return fallback_market_context(reason)


def refresh_premarket_trade_plan(timeout_seconds: int = 240) -> dict[str, Any]:
    """Build premarket_trade_plan.json using premarket_trade_planner.py."""
    try:
        from premarket_trade_planner import build_premarket_trade_plan

        payload = build_premarket_trade_plan()
        print(
            "[premarket-plan] refreshed via import "
            f"symbols={payload.get('symbol_count', 0)} valid={payload.get('valid_plan_count', 0)}"
        )
        write_event(
            "premarket_trade_plan_refreshed",
            {
                "symbol_count": payload.get("symbol_count"),
                "valid_plan_count": payload.get("valid_plan_count"),
                "valid_symbols": payload.get("valid_symbols", []),
            },
        )
        return payload
    except Exception as exc:
        print(f"[premarket-plan] import refresh failed; trying subprocess: {exc}")

    ok, output = run_python_script(PREMARKET_TRADE_PLANNER_PATH, timeout_seconds=timeout_seconds)
    if ok:
        print("[premarket-plan] refreshed via subprocess")
        if output:
            print(output[-2000:])
        write_event("premarket_trade_plan_refreshed_subprocess", {"output_tail": output[-2000:]})
    else:
        print(f"[premarket-plan] refresh failed: {output}")
        write_event("premarket_trade_plan_refresh_failed", {"error": output})

    return read_json_file(PREMARKET_TRADE_PLAN_PATH, {})


def load_market_context(refresh: bool = False) -> dict[str, Any]:
    if refresh or not MARKET_CONTEXT_PATH.exists():
        return refresh_market_context()
    return read_json_file(MARKET_CONTEXT_PATH, {})


def load_premarket_trade_plan(refresh: bool = False) -> dict[str, Any]:
    if refresh or not PREMARKET_TRADE_PLAN_PATH.exists():
        return refresh_premarket_trade_plan()
    return read_json_file(PREMARKET_TRADE_PLAN_PATH, {})


def load_intraday_structure_map() -> dict[str, Any]:
    return read_json_file(INTRADAY_STRUCTURE_MAP_PATH, {})


def parse_payload_time_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def payload_age_minutes(payload: dict[str, Any], path: Path | None = None) -> float | None:
    dt = parse_payload_time_utc(payload.get("created_at_utc") or payload.get("generated_at_utc") or payload.get("timestamp_utc"))
    if dt is not None:
        return max(0.0, (now_utc() - dt).total_seconds() / 60.0)
    if path is not None and path.exists():
        try:
            return max(0.0, (time.time() - path.stat().st_mtime) / 60.0)
        except Exception:
            return None
    return None


def structure_map_by_symbol(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("structures") or payload.get("symbols") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = safe_symbol(row.get("symbol"))
            if symbol:
                out[symbol] = row
    return out


def planner_health_snapshot(config: ScannerConfig | None = None) -> dict[str, Any]:
    cfg = config or ScannerConfig()
    plan_payload = read_json_file(PREMARKET_TRADE_PLAN_PATH, {})
    structure_payload = read_json_file(INTRADAY_STRUCTURE_MAP_PATH, {})
    plan_age = payload_age_minutes(plan_payload, PREMARKET_TRADE_PLAN_PATH)
    structure_age = payload_age_minutes(structure_payload, INTRADAY_STRUCTURE_MAP_PATH)

    status = str(plan_payload.get("status") or plan_payload.get("planner_status") or ("ok" if plan_payload else "missing")).lower()
    if status not in {"ok", "normal", "success"} and plan_payload:
        status = "degraded"

    long_allowed = set()
    raw_long = structure_payload.get("long_allowed_symbols") or plan_payload.get("long_allowed_symbols") or []
    if isinstance(raw_long, list):
        long_allowed = {safe_symbol(x) for x in raw_long if safe_symbol(x)}
    if not long_allowed:
        for symbol, row in structure_map_by_symbol(structure_payload).items():
            trade_structure = row.get("trade_structure", {}) if isinstance(row, dict) else {}
            if bool(trade_structure.get("long_allowed")):
                long_allowed.add(symbol)

    valid_symbols = set()
    raw_valid = plan_payload.get("valid_symbols") or []
    if isinstance(raw_valid, list):
        valid_symbols = {safe_symbol(x) for x in raw_valid if safe_symbol(x)}

    issues: list[str] = []
    if not plan_payload:
        issues.append("missing_premarket_trade_plan")
    if not structure_payload:
        issues.append("missing_intraday_structure_map")
    if plan_age is None:
        issues.append("unknown_plan_age")
    elif plan_age > cfg.max_plan_age_minutes:
        issues.append(f"stale_plan:{plan_age:.1f}m>{cfg.max_plan_age_minutes}m")
    if structure_age is None:
        issues.append("unknown_structure_age")
    elif structure_age > cfg.max_plan_age_minutes:
        issues.append(f"stale_structure:{structure_age:.1f}m>{cfg.max_plan_age_minutes}m")
    if status not in {"ok", "normal", "success"}:
        issues.append(f"planner_status:{status}")

    healthy = len(issues) == 0
    execution_mode = "NORMAL" if healthy else ("SAFE" if cfg.safe_mode_on_planner_failure else "BLOCK_NEW_BUYS")

    return {
        "healthy": healthy,
        "execution_mode": execution_mode,
        "issues": issues,
        "plan_status": status,
        "plan_age_minutes": plan_age,
        "structure_age_minutes": structure_age,
        "long_allowed_symbols": sorted(long_allowed),
        "valid_symbols": sorted(valid_symbols),
        "plan_payload": plan_payload,
        "structure_payload": structure_payload,
        "structure_by_symbol": structure_map_by_symbol(structure_payload),
    }




def session_planning_start_dt_et(hour: int = 4, minute: int = 0) -> datetime:
    now = datetime.now(ET)
    return now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)


def session_planning_window_is_open(config: ScannerConfig) -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return now >= session_planning_start_dt_et(
        config.session_planning_start_hour_et,
        config.session_planning_start_minute_et,
    )


def wait_until_session_planning_window(config: ScannerConfig) -> None:
    if not config.session_planning_enabled:
        return
    if session_planning_window_is_open(config):
        return

    start_dt = session_planning_start_dt_et(
        config.session_planning_start_hour_et,
        config.session_planning_start_minute_et,
    )

    while datetime.now(ET) < start_dt:
        seconds_left = max(1.0, (start_dt - datetime.now(ET)).total_seconds())
        print(f"[session-prep] waiting until {start_dt.strftime('%H:%M')} ET; {int(seconds_left // 60)} minute(s) left")
        time.sleep(min(300, max(5, seconds_left)))


def _read_session_prep_state() -> dict[str, Any]:
    return read_json_file(SESSION_PREP_STATE_PATH, {"date_et": "", "steps": {}})


def _write_session_prep_state(state: dict[str, Any]) -> None:
    write_json_file(SESSION_PREP_STATE_PATH, state)


def _session_step_already_done(state: dict[str, Any], date_et: str, step_name: str) -> bool:
    if state.get("date_et") != date_et:
        return False
    step = state.get("steps", {}).get(step_name, {})
    return bool(step.get("ok"))


def _record_session_step(state: dict[str, Any], date_et: str, step_name: str, ok: bool, output: str = "") -> None:
    state["date_et"] = date_et
    steps = state.setdefault("steps", {})
    steps[step_name] = {
        "ok": bool(ok),
        "completed_at_utc": now_utc().isoformat(),
        "output_tail": str(output or "")[-3000:],
    }
    _write_session_prep_state(state)
    write_event(
        "session_planning_step_completed" if ok else "session_planning_step_failed",
        {"step": step_name, "ok": bool(ok), "output_tail": str(output or "")[-1200:]},
    )


def refresh_production_regime_sizing_model(timeout_seconds: int = 900) -> dict[str, Any]:
    """Build regime_position_plan.csv using production_regime_sizing_model.py."""
    try:
        from production_regime_sizing_model import main as regime_main

        result = regime_main()
        print("[regime-model] refreshed via import")
        write_event("regime_model_refreshed", {"method": "import", "result": str(result)[:500]})
    except Exception as exc:
        print(f"[regime-model] import refresh failed; trying subprocess: {exc}")
        ok, output = run_python_script(PRODUCTION_REGIME_SIZING_MODEL_PATH, timeout_seconds=timeout_seconds)
        if ok:
            print("[regime-model] refreshed via subprocess")
            if output:
                print(output[-2000:])
            write_event("regime_model_refreshed_subprocess", {"output_tail": output[-2000:]})
        else:
            print(f"[regime-model] refresh failed: {output}")
            write_event("regime_model_refresh_failed", {"error": output})

    return {
        "regime_position_plan_exists": REGIME_POSITION_PLAN_PATH.exists(),
        "regime_universe_exists": REGIME_UNIVERSE_PATH.exists(),
        "regime_actions_exists": REGIME_ACTIONS_PATH.exists(),
    }



def validate_rl_intraday_structure_model(config: Optional[ScannerConfig] = None) -> dict[str, Any]:
    """
    Validate the saved RL timing model without retraining it.

    This is intentionally lightweight so run.py can safely call it during startup
    and before applying the RL timing gate. Training should happen offline unless
    allow_rl_training_from_run_py is explicitly enabled.
    """
    config = config or ScannerConfig()
    now = now_utc()
    issues: list[str] = []
    warnings: list[str] = []

    model_exists = RL_MODEL_PATH.exists()
    metrics_exists = RL_METRICS_PATH.exists()
    overlap_candidates = [
        RL_SYMBOL_OVERLAP_PATH,
        OUT_DIR / "rl_intraday_symbol_overlap.csv",
        OUT_DIR / f"{DATA_TAG}_rl_intraday_symbol_overlap.csv",
    ]
    overlap_exists = any(path.exists() for path in overlap_candidates)

    model_age_days: float | None = None
    metrics_age_days: float | None = None

    if not model_exists:
        issues.append("rl_model_missing")
    else:
        model_mtime = datetime.fromtimestamp(RL_MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        model_age_days = (now - model_mtime).total_seconds() / 86400.0
        if model_age_days > config.rl_max_model_age_days:
            issues.append(f"rl_model_stale:{model_age_days:.1f}d")

    metrics: dict[str, Any] = {}
    if not metrics_exists:
        issues.append("rl_metrics_missing")
    else:
        metrics_mtime = datetime.fromtimestamp(RL_METRICS_PATH.stat().st_mtime, tz=timezone.utc)
        metrics_age_days = (now - metrics_mtime).total_seconds() / 86400.0
        if metrics_age_days > config.rl_metrics_max_age_days:
            issues.append(f"rl_metrics_stale:{metrics_age_days:.1f}d")
        metrics = read_json_file(RL_METRICS_PATH, {})

    if config.rl_require_symbol_overlap_file and not overlap_exists:
        warnings.append("rl_symbol_overlap_file_missing")

    diagnostics = metrics.get("diagnostics", {}) if isinstance(metrics, dict) else {}
    trade_quality = diagnostics.get("trade_quality", {}) if isinstance(diagnostics, dict) else {}

    enter_long = to_int(diagnostics.get("enter_long_count", diagnostics.get("buy_count", 0)))
    enter_short = to_int(diagnostics.get("enter_short_count", 0))
    exit_long = to_int(diagnostics.get("exit_long_count", diagnostics.get("sell_count", 0)))
    exit_short = to_int(diagnostics.get("exit_short_count", 0))
    add_long = to_int(diagnostics.get("add_long_count", trade_quality.get("add_long_count", 0)))
    training_trades = enter_long + enter_short + exit_long + exit_short + add_long

    profit_factor = to_float(trade_quality.get("profit_factor", diagnostics.get("profit_factor", 0.0)), 0.0)
    win_rate = to_float(trade_quality.get("win_rate", diagnostics.get("win_rate", 0.0)), 0.0)

    if metrics_exists:
        if training_trades < config.rl_min_training_trades:
            issues.append(f"rl_training_trades_too_low:{training_trades}")
        if profit_factor and profit_factor < config.rl_min_profit_factor:
            warnings.append(f"rl_profit_factor_low:{profit_factor:.2f}")
        if win_rate and win_rate < config.rl_min_win_rate:
            warnings.append(f"rl_win_rate_low:{win_rate:.2f}")

    ok = model_exists and metrics_exists and not issues
    payload = {
        "ok": bool(ok),
        "model_exists": bool(model_exists),
        "metrics_exists": bool(metrics_exists),
        "overlap_exists": bool(overlap_exists),
        "model_path": str(RL_MODEL_PATH),
        "metrics_path": str(RL_METRICS_PATH),
        "model_age_days": model_age_days,
        "metrics_age_days": metrics_age_days,
        "training_trades": int(training_trades),
        "enter_long_count": int(enter_long),
        "enter_short_count": int(enter_short),
        "exit_long_count": int(exit_long),
        "exit_short_count": int(exit_short),
        "add_long_count": int(add_long),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "issues": issues,
        "warnings": warnings,
    }
    write_event("rl_model_validation", payload)
    return payload


def refresh_rl_intraday_structure_model(timeout_seconds: int = 900, config: Optional[ScannerConfig] = None, force_train: bool = False) -> dict[str, Any]:
    """
    Validate the saved RL model by default.

    Training from run.py is disabled unless both force_train=True and
    allow_rl_training_from_run_py=True. This prevents accidental expensive or
    unstable retraining every morning.
    """
    config = config or ScannerConfig()

    if not force_train or not config.allow_rl_training_from_run_py:
        payload = validate_rl_intraday_structure_model(config)
        print(
            "[rl-model] validation "
            f"ok={payload.get('ok')} model_exists={payload.get('model_exists')} "
            f"trades={payload.get('training_trades')} issues={payload.get('issues')}"
        )
        return payload

    ok, output = run_python_script(
        RL_INTRADAY_STRUCTURE_TRAINER_PATH,
        timeout_seconds=timeout_seconds,
        env_overrides={
            "RL_DATA_DATE": DATA_TAG,
            "RUN_DATA_DATE": DATA_TAG,
            "RL_DISABLE_SHORTS": "1",
            "RL_ALLOW_SHORTS": "0",
            "RL_ADD_LONG_ENABLED": "1",
        },
    )
    if ok:
        print("[rl-trainer] refreshed via subprocess")
        if output:
            print(output[-2500:])
        write_event("rl_intraday_model_refreshed", {"output_tail": output[-2500:]})
    else:
        print(f"[rl-trainer] refresh failed: {output}")
        write_event("rl_intraday_model_refresh_failed", {"error": output})

    validation = validate_rl_intraday_structure_model(config)
    validation.update({
        "trainer_ok": ok,
        "output_tail": output[-2000:] if output else "",
    })
    return validation

def run_session_planning_pipeline(config: ScannerConfig, force: bool = False) -> dict[str, Any]:
    """
    Runs once per ET trading day, starting no earlier than 4:00 AM ET by default.

    Order:
        1. llm_sentiment_universe_selector.py
        2. market_context_builder.py
        3. premarket_trade_planner.py
        4. production_regime_sizing_model.py
        5. RL model validation only by default; optional training only when explicitly enabled
    """
    if not config.session_planning_enabled:
        return {"skipped": True, "reason": "session_planning_disabled"}

    wait_until_session_planning_window(config)

    date_et = datetime.now(ET).date().isoformat()
    state = _read_session_prep_state()
    if state.get("date_et") != date_et:
        state = {"date_et": date_et, "started_at_utc": now_utc().isoformat(), "steps": {}}
        _write_session_prep_state(state)

    if not force and not config.session_planning_force_each_startup and state.get("completed") is True:
        print(f"[session-prep] already completed for {date_et}")
        return state

    print(f"[session-prep] starting daily planning pipeline for {date_et}")
    write_event("session_planning_started", {"date_et": date_et})

    if config.use_llm_sentiment_universe and config.refresh_llm_sentiment_universe:
        step = "llm_sentiment_universe_selector"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                refresh_llm_sentiment_universe(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, True, "refreshed")
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_market_context_builder and config.market_context_enabled:
        step = "market_context_builder"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_market_context(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, bool(payload), json.dumps(payload, default=json_default)[:3000])
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_premarket_trade_planner and config.premarket_plan_enabled:
        step = "premarket_trade_planner"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_premarket_trade_plan(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, bool(payload), json.dumps(payload, default=json_default)[:3000])
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_production_regime_sizing_model:
        step = "production_regime_sizing_model"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_production_regime_sizing_model(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, bool(payload.get("regime_position_plan_exists")), json.dumps(payload, default=json_default))
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    # Validate the saved RL model every session. Training remains offline by default.
    step = "rl_intraday_structure_model_validation"
    if force or not _session_step_already_done(state, date_et, step):
        try:
            payload = refresh_rl_intraday_structure_model(
                timeout_seconds=config.session_planning_timeout_seconds,
                config=config,
                force_train=bool(config.run_rl_intraday_structure_trainer and config.allow_rl_training_from_run_py),
            )
            ok = bool(payload.get("ok") or (payload.get("model_exists") and config.rl_allow_hold_when_model_missing))
            _record_session_step(state, date_et, step, ok, json.dumps(payload, default=json_default))
        except Exception as exc:
            _record_session_step(state, date_et, step, bool(RL_MODEL_PATH.exists()), str(exc))

    state = _read_session_prep_state()
    state["completed"] = True
    state["completed_at_utc"] = now_utc().isoformat()
    _write_session_prep_state(state)
    write_event("session_planning_completed", state)
    print(f"[session-prep] completed for {date_et}")
    return state


def load_regime_position_plan() -> pd.DataFrame:
    if not REGIME_POSITION_PLAN_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(REGIME_POSITION_PLAN_PATH)
    except Exception as exc:
        print(f"[regime-plan] failed to read {REGIME_POSITION_PLAN_PATH}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "symbol" not in df.columns:
        print(f"[regime-plan] missing symbol column: {REGIME_POSITION_PLAN_PATH}")
        return pd.DataFrame()
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df[df["symbol"] != ""].reset_index(drop=True)


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _regime_action_value(row: pd.Series) -> str:
    for col in ["trade_action", "action", "regime_action", "recommended_action", "signal"]:
        if col in row.index and pd.notna(row.get(col)):
            return str(row.get(col, "")).upper().strip()
    return ""


def _regime_target_weight(row: pd.Series, default: float = 0.0) -> float:
    for col in ["target_weight", "weight", "recommended_weight", "target_alloc", "allocation"]:
        if col in row.index and pd.notna(row.get(col)):
            val = to_float(row.get(col), default)
            if val > 1.0:
                val /= 100.0
            return float(np.clip(val, 0.0, 1.0))
    return default


def _regime_confidence(row: pd.Series, default: float = 0.55) -> float:
    for col in ["confidence", "trade_confidence", "confidence_score", "probability", "model_probability", "buy_probability"]:
        if col in row.index and pd.notna(row.get(col)):
            val = to_float(row.get(col), default)
            if val > 1.0:
                val /= 100.0
            return float(np.clip(val, 0.0, 1.0))
    return default


def _regime_price(row: pd.Series, fallback: float = 0.0) -> float:
    for col in ["price", "last_price", "close", "entry_price", "current_price"]:
        if col in row.index and pd.notna(row.get(col)):
            val = to_float(row.get(col), fallback)
            if val > 0:
                return val
    return fallback


def market_context_blocks_new_buys(context: dict[str, Any], allow_cautious: bool = True) -> tuple[bool, str]:
    if not context:
        return False, "market_context_missing"

    permission = str(context.get("trade_permission", "cautious")).lower().strip()
    bias = str(context.get("overall_market_bias", "mixed")).lower().strip()
    risk = str(context.get("risk_level", "medium")).lower().strip()

    if permission == "halt_new_buys":
        return True, f"market_context_halt_new_buys:bias={bias}:risk={risk}"
    if not allow_cautious and permission == "cautious":
        return True, f"market_context_cautious_block:bias={bias}:risk={risk}"
    if bias in {"risk_off", "bearish"} and risk == "high":
        return True, f"market_context_risk_off_high_risk:permission={permission}"

    return False, f"market_context_allows_buys:permission={permission}:bias={bias}:risk={risk}"


def valid_symbols_from_premarket_plan(payload: dict[str, Any]) -> set[str]:
    symbols = payload.get("valid_symbols", []) if isinstance(payload, dict) else []
    return {normalize_symbol(s) for s in symbols if normalize_symbol(s)}

def load_llm_sentiment_universe(refresh: bool = True, min_confidence: float = 0.0) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load symbols and per-symbol metadata from llm_sentiment_universe_selector.py output."""
    if refresh:
        refresh_llm_sentiment_universe()

    payload: dict[str, Any] = {}
    if LLM_SENTIMENT_UNIVERSE_PATH.exists():
        payload = read_json_file(LLM_SENTIMENT_UNIVERSE_PATH, {})
    else:
        print(f"[llm-universe] missing {LLM_SENTIMENT_UNIVERSE_PATH}; falling back to locked universe/default list")

    metadata: dict[str, dict[str, Any]] = {}
    symbols: list[str] = []

    selections = payload.get("selections", []) if isinstance(payload, dict) else []
    if isinstance(selections, list):
        for item in selections:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            confidence = to_float(item.get("confidence"), 0.0)
            if not symbol or confidence < min_confidence:
                continue
            metadata[symbol] = {
                "llm_rank": to_int(item.get("rank"), 999),
                "llm_confidence": confidence,
                "llm_catalyst_type": str(item.get("catalyst_type", "")),
                "llm_reason": str(item.get("reason", ""))[:700],
                "llm_market_tone": str(payload.get("market_tone", "")),
                "llm_summary": str(payload.get("summary", ""))[:700],
            }
            symbols.append(symbol)

    if not symbols and isinstance(payload.get("symbols"), list):
        symbols = [normalize_symbol(x) for x in payload.get("symbols", []) if normalize_symbol(x)]
        metadata = {s: {"llm_confidence": 0.50, "llm_rank": i + 1} for i, s in enumerate(symbols)}

    if not symbols and LOCKED_UNIVERSE_PATH.exists():
        symbols = [normalize_symbol(x) for x in LOCKED_UNIVERSE_PATH.read_text().splitlines() if normalize_symbol(x)]
        metadata = {s: {"llm_confidence": 0.50, "llm_rank": i + 1, "llm_source": "locked_trading_universe"} for i, s in enumerate(symbols)}

    symbols = [s for s in symbols if is_simple_alpaca_symbol(s)]
    symbols = list(dict.fromkeys(symbols))

    if symbols:
        print(f"[llm-universe] loaded symbols={len(symbols)} source={LLM_SENTIMENT_UNIVERSE_PATH}")
    else:
        print("[llm-universe] no valid LLM symbols loaded; using DEFAULT_SCAN_UNIVERSE")
        symbols = list(DEFAULT_SCAN_UNIVERSE)
        metadata = {s: {"llm_confidence": 0.50, "llm_rank": i + 1, "llm_source": "default_fallback"} for i, s in enumerate(symbols)}

    return symbols, metadata


def write_event(event: str, payload: Optional[dict[str, Any]] = None) -> None:
    row = {
        "timestamp_utc": now_utc().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=json_default) + "\n")


def _coerce_log_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return value


def _safe_dict_get_float(payload: dict[str, Any], key: str, default: float | None = None) -> float | None:
    if not isinstance(payload, dict):
        return default
    value = payload.get(key, default)
    if value is None or value == "":
        return default
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _executed_order_columns() -> list[str]:
    return [
        "timestamp_utc", "symbol", "side", "qty", "order_type",
        "submitted_at", "filled_at", "filled_avg_price", "status",
        "order_id", "client_order_id", "strategy", "regime", "confidence", "reason",
        "exit_reason", "exit_model_action",
        "exit_probability_hold", "exit_probability_giveback", "exit_probability_reversal",
        "expected_value_next_5m", "expected_value_next_15m", "expected_value_next_30m",
        "pnl_pct", "unrealized_pnl", "realized_pnl",
        "max_favorable_excursion", "max_adverse_excursion",
        "bars_held", "minutes_held",
        "distance_to_5m_support", "distance_to_5m_resistance",
        "distance_to_15m_support", "distance_to_15m_resistance",
        "distance_to_1h_support", "distance_to_1h_resistance",
        "distance_to_4h_support", "distance_to_4h_resistance",
        "5m_trend_stack", "15m_trend_stack", "1h_trend_stack", "4h_trend_stack",
        "above_vwap", "macd_hist", "rsi_14",
    ]


def log_executed_order(
    *,
    symbol: str,
    side: str,
    qty: float,
    order_type: str,
    submitted_at: str | None = None,
    filled_at: str | None = None,
    filled_avg_price: float | None = None,
    status: str | None = None,
    order_id: str | None = None,
    client_order_id: str | None = None,
    strategy: str | None = None,
    regime: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    exit_reason: str | None = None,
    exit_model_action: str | None = None,
    exit_probability_hold: float | None = None,
    exit_probability_giveback: float | None = None,
    exit_probability_reversal: float | None = None,
    expected_value_next_5m: float | None = None,
    expected_value_next_15m: float | None = None,
    expected_value_next_30m: float | None = None,
    pnl_pct: float | None = None,
    unrealized_pnl: float | None = None,
    realized_pnl: float | None = None,
    max_favorable_excursion: float | None = None,
    max_adverse_excursion: float | None = None,
    bars_held: int | None = None,
    minutes_held: float | None = None,
    distance_to_5m_support: float | None = None,
    distance_to_5m_resistance: float | None = None,
    distance_to_15m_support: float | None = None,
    distance_to_15m_resistance: float | None = None,
    distance_to_1h_support: float | None = None,
    distance_to_1h_resistance: float | None = None,
    distance_to_4h_support: float | None = None,
    distance_to_4h_resistance: float | None = None,
    trend_5m: int | None = None,
    trend_15m: int | None = None,
    trend_1h: int | None = None,
    trend_4h: int | None = None,
    above_vwap: int | None = None,
    macd_hist: float | None = None,
    rsi_14: float | None = None,
) -> None:
    row = {
        "timestamp_utc": now_utc().isoformat(),
        "symbol": safe_symbol(symbol),
        "side": str(side or "").lower(),
        "qty": qty,
        "order_type": order_type,
        "submitted_at": submitted_at,
        "filled_at": filled_at,
        "filled_avg_price": filled_avg_price,
        "status": status,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "strategy": strategy,
        "regime": regime,
        "confidence": confidence,
        "reason": reason,
        "exit_reason": exit_reason,
        "exit_model_action": exit_model_action,
        "exit_probability_hold": exit_probability_hold,
        "exit_probability_giveback": exit_probability_giveback,
        "exit_probability_reversal": exit_probability_reversal,
        "expected_value_next_5m": expected_value_next_5m,
        "expected_value_next_15m": expected_value_next_15m,
        "expected_value_next_30m": expected_value_next_30m,
        "pnl_pct": pnl_pct,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "max_favorable_excursion": max_favorable_excursion,
        "max_adverse_excursion": max_adverse_excursion,
        "bars_held": bars_held,
        "minutes_held": minutes_held,
        "distance_to_5m_support": distance_to_5m_support,
        "distance_to_5m_resistance": distance_to_5m_resistance,
        "distance_to_15m_support": distance_to_15m_support,
        "distance_to_15m_resistance": distance_to_15m_resistance,
        "distance_to_1h_support": distance_to_1h_support,
        "distance_to_1h_resistance": distance_to_1h_resistance,
        "distance_to_4h_support": distance_to_4h_support,
        "distance_to_4h_resistance": distance_to_4h_resistance,
        "5m_trend_stack": trend_5m,
        "15m_trend_stack": trend_15m,
        "1h_trend_stack": trend_1h,
        "4h_trend_stack": trend_4h,
        "above_vwap": above_vwap,
        "macd_hist": macd_hist,
        "rsi_14": rsi_14,
    }

    cols = _executed_order_columns()
    clean_row = {c: _coerce_log_value(row.get(c, "")) for c in cols}
    write_header = not EXECUTED_ORDERS_LOG.exists()

    EXECUTED_ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXECUTED_ORDERS_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)

    with EXECUTED_ORDERS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_row, default=json_default) + "\n")

    print(
        f"[execution-log] {clean_row['symbol']} {clean_row['side']} "
        f"qty={clean_row['qty']} fill={clean_row['filled_avg_price']} "
        f"status={clean_row['status']} exit_reason={clean_row['exit_reason']}"
    )



def _flatten_for_experience(prefix: str, payload: Any, max_items: int = 60) -> dict[str, Any]:
    """Flatten small dict-like payloads into one-level experience columns."""
    out: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out

    count = 0
    for key, value in payload.items():
        if count >= max_items:
            break
        clean_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_").lower()
        if not clean_key:
            continue
        col = f"{prefix}_{clean_key}" if prefix else clean_key
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[col] = value
            count += 1
        elif hasattr(value, "item"):
            try:
                out[col] = value.item()
                count += 1
            except Exception:
                pass
    return out


def _safe_row_dict(row: pd.Series | dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, pd.Series):
        data = row.to_dict()
    elif isinstance(row, dict):
        data = dict(row)
    else:
        return {}
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[str(key)] = value
        elif hasattr(value, "item"):
            try:
                clean[str(key)] = value.item()
            except Exception:
                clean[str(key)] = str(value)
        else:
            clean[str(key)] = str(value)
    return clean


def append_live_experience(row: dict[str, Any], write_parquet: bool = True) -> None:
    """
    Append a live decision/execution sample for offline ML training.

    This intentionally does NOT train inside run.py. It creates a replay store
    that an after-close trainer can label with future returns/PnL and validate
    before promoting any model.
    """
    row = dict(row)
    row.setdefault("logged_at_utc", now_utc().isoformat())
    row.setdefault("future_return_5m", "")
    row.setdefault("future_return_15m", "")
    row.setdefault("future_return_60m", "")
    row.setdefault("max_runup", "")
    row.setdefault("max_drawdown", "")
    row.setdefault("outcome_label", "")

    LIVE_EXPERIENCE_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_EXPERIENCE_JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=json_default) + "\n")

    df = pd.DataFrame([row])
    write_header = not LIVE_EXPERIENCE_CSV_PATH.exists()
    df.to_csv(LIVE_EXPERIENCE_CSV_PATH, mode="a", header=write_header, index=False)

    if write_parquet:
        try:
            if LIVE_EXPERIENCE_PARQUET_PATH.exists():
                old = pd.read_parquet(LIVE_EXPERIENCE_PARQUET_PATH)
                pd.concat([old, df], ignore_index=True).to_parquet(LIVE_EXPERIENCE_PARQUET_PATH, index=False)
            else:
                df.to_parquet(LIVE_EXPERIENCE_PARQUET_PATH, index=False)
        except Exception as exc:
            # Parquet is optional because pyarrow/fastparquet may not be installed.
            write_event("live_experience_parquet_failed", {"error": str(exc)})


def build_live_experience_row(
    *,
    cycle_id: str,
    stage: str,
    order_row: pd.Series | dict[str, Any] | None = None,
    acct_before: Optional[dict[str, Any]] = None,
    acct_after: Optional[dict[str, Any]] = None,
    result: Optional[dict[str, Any]] = None,
    position: Optional[BrokerPosition] = None,
    direction: Optional[dict[str, Any]] = None,
    message: str = "",
) -> dict[str, Any]:
    order = _safe_row_dict(order_row)
    result = dict(result or {})
    acct_before = dict(acct_before or {})
    acct_after = dict(acct_after or {})
    position = position or BrokerPosition(symbol=safe_symbol(order.get("symbol", result.get("symbol", ""))))

    symbol = safe_symbol(order.get("symbol", result.get("symbol", position.symbol or "")))
    side = str(order.get("side", result.get("side", ""))).lower()
    status = str(result.get("status", stage))

    row: dict[str, Any] = {
        "experience_id": f"{cycle_id}_{symbol}_{side}_{stage}_{int(time.time() * 1000)}",
        "cycle_id": cycle_id,
        "stage": stage,
        "timestamp_utc": now_utc().isoformat(),
        "symbol": symbol,
        "side": side,
        "action_taken": f"{side.upper()}" if side else "NONE",
        "executed": stage in {"submitted", "filled", "order"} or status.lower() in {"submitted", "filled", "partially_filled", "accepted", "new"},
        "status": status,
        "message": message or result.get("message", ""),
        "reason": order.get("reason", result.get("reason", "")),
        "qty_requested": order.get("order_qty", result.get("qty", "")),
        "qty_final": result.get("qty", order.get("order_qty", "")),
        "decision_price": order.get("price", result.get("price", "")),
        "estimated_notional": order.get("capped_dollars", order.get("abs_delta_dollars", result.get("estimated_notional", ""))),
        "target_weight": order.get("target_weight", ""),
        "current_weight": order.get("current_weight", ""),
        "delta_weight": order.get("delta_weight", ""),
        "confidence_used": order.get("confidence_used", order.get("confidence", "")),
        "confidence_multiplier": order.get("confidence_multiplier", ""),
        "regime_trade_action": order.get("regime_trade_action", order.get("trade_action", "")),
        "regime_confidence": order.get("regime_confidence", order.get("trade_confidence", "")),
        "regime_setup_score": order.get("setup_score", ""),
        "regime_setup_type": order.get("setup_type", ""),
        "rl_action": order.get("rl_action", ""),
        "rl_raw_action": order.get("rl_raw_action", ""),
        "rl_confidence": order.get("rl_confidence", ""),
        "rl_entropy": order.get("rl_entropy", ""),
        "rl_masked": order.get("rl_masked", ""),
        "rl_mask_reason": order.get("rl_mask_reason", ""),
        "premarket_valid": order.get("premarket_valid", order.get("premarket_valid_setup", "")),
        "market_bias": order.get("market_bias", order.get("market_context_bias", "")),
        "market_permission": order.get("market_permission", order.get("market_context_permission", "")),
        "market_risk": order.get("market_risk", order.get("market_context_risk_level", "")),
        "account_equity_before": acct_before.get("equity", ""),
        "account_cash_before": acct_before.get("cash", ""),
        "account_buying_power_before": acct_before.get("buying_power", ""),
        "gross_exposure_before": acct_before.get("gross_exposure", ""),
        "account_equity_after": acct_after.get("equity", ""),
        "account_cash_after": acct_after.get("cash", ""),
        "account_buying_power_after": acct_after.get("buying_power", ""),
        "gross_exposure_after": acct_after.get("gross_exposure", ""),
        "position_qty_before": position.qty,
        "position_market_value_before": position.market_value,
        "position_avg_entry_price_before": position.avg_entry_price,
        "position_current_price_before": position.current_price,
        "position_unrealized_pl_before": position.unrealized_pl,
        "position_unrealized_plpc_before": position.unrealized_plpc,
        "order_id": result.get("order_id", ""),
        "fill_status": result.get("fill_status", result.get("status", "")),
        "filled_qty": result.get("filled_qty", ""),
        "filled_avg_price": result.get("filled_avg_price", ""),
        "slippage_pct": result.get("slippage_pct", ""),
    }

    row.update(_flatten_for_experience("dir", direction or {}, max_items=40))
    return row


def save_df(df: Optional[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else pd.DataFrame()).to_csv(path, index=False)
    print(f"[saved] {path}")


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def local_trading_day() -> str:
    """Trading-day key used for same-day no-reentry state."""
    return datetime.now(ET).date().isoformat()


def read_json_file(path: Path, default: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_no_reentry_symbols() -> set[str]:
    """Symbols exited today that must not be re-entered until the next trading day."""
    today = local_trading_day()
    state = read_json_file(NO_REENTRY_STATE_PATH, {"date": today, "symbols": {}})
    if state.get("date") != today:
        state = {"date": today, "symbols": {}}
        write_json_file(NO_REENTRY_STATE_PATH, state)
    symbols = state.get("symbols", {})
    if isinstance(symbols, dict):
        return {safe_symbol(sym) for sym in symbols.keys() if safe_symbol(sym)}
    if isinstance(symbols, list):
        return {safe_symbol(sym) for sym in symbols if safe_symbol(sym)}
    return set()


def mark_no_reentry_symbol(symbol: str, reason: str = "exit") -> None:
    """Persist that a symbol was exited today and cannot be bought again today."""
    symbol = safe_symbol(symbol)
    if not symbol:
        return
    today = local_trading_day()
    state = read_json_file(NO_REENTRY_STATE_PATH, {"date": today, "symbols": {}})
    if state.get("date") != today:
        state = {"date": today, "symbols": {}}

    symbols = state.get("symbols", {})
    if not isinstance(symbols, dict):
        symbols = {}

    symbols[symbol] = {
        "symbol": symbol,
        "marked_at": now_utc().isoformat(),
        "reason": reason,
    }
    state["symbols"] = symbols
    write_json_file(NO_REENTRY_STATE_PATH, state)
    write_event("no_reentry_marked", {"symbol": symbol, "reason": reason, "date": today})


def no_reentry_cooldown_active(symbol: str, cooldown_minutes: int) -> tuple[bool, str]:
    """Return whether a symbol is blocked by same-day / timed re-entry cooldown."""
    symbol = safe_symbol(symbol)
    if not symbol:
        return False, ""

    today = local_trading_day()
    state = read_json_file(NO_REENTRY_STATE_PATH, {"date": today, "symbols": {}})
    if state.get("date") != today:
        return False, ""

    symbols = state.get("symbols", {})
    item = symbols.get(symbol) if isinstance(symbols, dict) else None
    if not isinstance(item, dict):
        return False, ""

    marked_at = str(item.get("marked_at", ""))
    if not marked_at:
        return True, "same_day_exit"

    try:
        ts = datetime.fromisoformat(marked_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_min = (now_utc() - ts.astimezone(timezone.utc)).total_seconds() / 60.0
    except Exception:
        age_min = 0.0

    if age_min < max(0, cooldown_minutes):
        return True, f"recent_exit_cooldown_{age_min:.1f}m"

    # Same-day no-reentry remains a stronger default than the timed cooldown.
    return True, "same_day_exit"


def minutes_since_regular_open() -> float:
    now_et = datetime.now(ET)
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    return (now_et - open_et).total_seconds() / 60.0


def load_pyramid_state() -> dict[str, Any]:
    today = local_trading_day()
    state = read_json_file(LOG_DIR / "pyramid_state.json", {"date": today, "symbols": {}})
    if state.get("date") != today:
        state = {"date": today, "symbols": {}}
        write_json_file(LOG_DIR / "pyramid_state.json", state)
    if not isinstance(state.get("symbols"), dict):
        state["symbols"] = {}
    return state


def save_pyramid_state(state: dict[str, Any]) -> None:
    write_json_file(LOG_DIR / "pyramid_state.json", state)


def pyramid_add_count(symbol: str) -> int:
    state = load_pyramid_state()
    return int(to_int(state.get("symbols", {}).get(safe_symbol(symbol), {}).get("adds", 0)))


def mark_pyramid_add(symbol: str, reason: str = "pyramid_add") -> None:
    symbol = safe_symbol(symbol)
    if not symbol:
        return
    state = load_pyramid_state()
    symbols = state.setdefault("symbols", {})
    current = symbols.get(symbol, {}) if isinstance(symbols.get(symbol, {}), dict) else {}
    current["adds"] = to_int(current.get("adds", 0)) + 1
    current["last_added_at"] = now_utc().isoformat()
    current["reason"] = reason
    symbols[symbol] = current
    state["symbols"] = symbols
    save_pyramid_state(state)
    write_event("pyramid_add_marked", {"symbol": symbol, "adds": current["adds"], "reason": reason})


# ======================================================================# CSV journal
# ======================================================================
class CsvJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = [
            "timestamp_utc", "cycle_id", "event", "market_open",
            "equity_before", "equity_after", "cash_before", "cash_after",
            "buying_power_before", "buying_power_after",
            "gross_exposure_before", "gross_exposure_after",
            "trade_symbol", "trade_side", "trade_qty", "trade_price",
            "estimated_notional", "target_weight", "current_weight", "delta_weight",
            "confidence", "cash_buffer", "max_gross_exposure", "max_position_weight",
            "max_order_notional", "order_id", "order_status", "filled_qty",
            "filled_avg_price", "slippage_pct", "message",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def append(self, row: dict[str, Any]) -> None:
        safe_row = {k: row.get(k, "") for k in self.columns}
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(safe_row)



# ======================================================================# Stream-confirmed movement helpers
# ======================================================================
def append_pending_direction_watch_from_order(
    order_row: dict[str, Any] | pd.Series,
    reject_reason: str,
    path: Path = PENDING_DIRECTION_WATCHLIST_PATH,
) -> None:
    """
    Add a skipped BUY to the live confirmation watchlist.

    This is called when run.py rejects an order for:
        price_direction_not_confirmed:...
    The stream confirmation layer can later promote the symbol into a confirmed
    rebalance order with entry/stop/take-profit/trailing-stop fields.
    """
    try:
        data = order_row.to_dict() if isinstance(order_row, pd.Series) else dict(order_row)
        symbol = safe_symbol(data.get("symbol"))
        side = str(data.get("side", "buy")).lower()
        if not symbol or side != "buy":
            return

        row = {
            "symbol": symbol,
            "side": "buy",
            "order_qty": to_int(data.get("order_qty", data.get("qty", 0))),
            "price": to_float(data.get("price")),
            "capped_dollars": to_float(data.get("capped_dollars", data.get("abs_delta_dollars", 0.0))),
            "confidence_used": to_float(data.get("confidence_used", data.get("trade_confidence", 0.0))),
            "reason": str(reject_reason),
            "rejected_at_utc": now_utc().isoformat(),
            "source": "run_py_direction_reject",
        }

        df = pd.DataFrame([row])
        header = not path.exists()
        df.to_csv(path, index=False, mode="a", header=header)
        write_event("pending_direction_watch_added", row)
        print(f"[stream-confirm] watchlist add {symbol}: {reject_reason}")
    except Exception as exc:
        print(f"[stream-confirm] failed to append watchlist row: {exc}")
        write_event("pending_direction_watch_append_error", {"error": str(exc)})


def load_stream_confirmed_orders() -> pd.DataFrame:
    """
    Load orders produced by stream_confirmed_movement_gate.py.

    Expected columns include:
        symbol, side, order_qty, price, entry_price, stop_loss, take_profit,
        trailing_stop_enabled, trail_percent, trail_price, reason
    """
    if not STREAM_CONFIRMED_ORDERS_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(STREAM_CONFIRMED_ORDERS_PATH)
    except Exception as exc:
        print(f"[stream-confirm] failed reading confirmed orders: {exc}")
        return pd.DataFrame()

    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["symbol"] = df["symbol"].map(safe_symbol)
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()
    else:
        df["side"] = "buy"
    df = df[(df["symbol"] != "") & (df["side"] == "buy")].copy()
    if df.empty:
        return pd.DataFrame()

    if "order_qty" not in df.columns and "qty" in df.columns:
        df["order_qty"] = df["qty"]
    if "price" not in df.columns and "entry_price" in df.columns:
        df["price"] = df["entry_price"]
    if "capped_dollars" not in df.columns and "abs_delta_dollars" in df.columns:
        df["capped_dollars"] = df["abs_delta_dollars"]
    if "abs_delta_dollars" not in df.columns:
        df["abs_delta_dollars"] = pd.to_numeric(df.get("order_qty", 0), errors="coerce").fillna(0) * pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)
    if "confidence_used" not in df.columns:
        df["confidence_used"] = pd.to_numeric(df.get("direction_score", 0.65), errors="coerce").fillna(0.65)
    if "confidence_multiplier" not in df.columns:
        df["confidence_multiplier"] = 1.0 + pd.to_numeric(df["confidence_used"], errors="coerce").fillna(0.65)
    if "reason" not in df.columns:
        df["reason"] = "stream_confirmed_movement"
    if "stream_confirmed" not in df.columns:
        df["stream_confirmed"] = True

    # Keep one latest confirmed order per symbol.
    sort_col = "stream_confirmed_at_utc" if "stream_confirmed_at_utc" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col)
    df = df.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    return df


def merge_stream_confirmed_orders(base: pd.DataFrame) -> pd.DataFrame:
    confirmed = load_stream_confirmed_orders()
    if confirmed.empty:
        return base if base is not None else pd.DataFrame()

    base = base.copy() if base is not None else pd.DataFrame()
    if not base.empty and "symbol" in base.columns:
        base["symbol"] = base["symbol"].map(safe_symbol)
        # Confirmed stream orders replace stale same-symbol buy rows.
        confirmed_symbols = set(confirmed["symbol"].map(safe_symbol))
        if "side" in base.columns:
            base_side = base["side"].astype(str).str.lower()
        else:
            base_side = pd.Series(["buy"] * len(base), index=base.index)
        base = base[~((base["symbol"].isin(confirmed_symbols)) & (base_side == "buy"))].copy()

    merged = pd.concat([base, confirmed], ignore_index=True, sort=False)
    print(f"[stream-confirm] merged confirmed_orders={len(confirmed)} total_rebalance_rows={len(merged)}")
    write_event(
        "stream_confirmed_orders_merged",
        {
            "confirmed_rows": int(len(confirmed)),
            "merged_rows": int(len(merged)),
            "path": str(STREAM_CONFIRMED_ORDERS_PATH),
        },
    )
    return merged




# ======================================================================# SR RL agent helpers
# ======================================================================
def sr_write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "timestamp_utc": now_utc().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    try:
        with SR_RL_EVENTS_JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=json_default) + "\n")
    except Exception:
        pass
    write_event(f"sr_rl_{event}", payload or {})


def load_sr_rl_signals(max_age_seconds: int = 90) -> list[dict[str, Any]]:
    """
    Load sr_rl_agent.py's live decisions.

    Expected file:
        system-data/YYYYMMDD/YYYYMMDD_sr_rl_live_signals.json

    Only fresh payloads are accepted. SELL decisions are treated as avoid/exit
    signals by sr_rl_agent.py, but run.py keeps this layer long-only and only
    promotes BUY decisions into the rebalance order set.
    """
    if not SR_RL_SIGNALS_PATH.exists():
        return []

    try:
        mtime = datetime.fromtimestamp(SR_RL_SIGNALS_PATH.stat().st_mtime, tz=timezone.utc)
        age = (now_utc() - mtime).total_seconds()
        if age > max_age_seconds:
            print(f"[sr-rl] stale signals age={age:.1f}s path={SR_RL_SIGNALS_PATH}")
            return []
        payload = json.loads(SR_RL_SIGNALS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[sr-rl] failed reading signals: {exc}")
        sr_write_event("signals_read_error", {"error": str(exc), "path": str(SR_RL_SIGNALS_PATH)})
        return []

    decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
    return [d for d in decisions if isinstance(d, dict)]


def load_sr_rl_orders(config: ScannerConfig | None = None) -> pd.DataFrame:
    """
    Convert SR-RL BUY decisions into executor-compatible rebalance rows.

    Output columns intentionally include delta_shares and delta_dollars because
    _build_order_plan normalizes from those fields.
    """
    cfg = config or ScannerConfig()
    decisions = load_sr_rl_signals(cfg.sr_rl_signal_max_age_seconds)
    if not decisions:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for d in decisions:
        symbol = safe_symbol(d.get("symbol"))
        action = str(d.get("action", "wait")).lower().strip()
        if not symbol or action != "buy":
            continue

        price = to_float(d.get("price"), 0.0)
        qty = to_int(d.get("qty"), 0)
        if price <= 0 or qty < int(cfg.sr_rl_min_qty):
            continue

        notional = qty * price
        if cfg.sr_rl_max_order_dollars > 0 and notional > cfg.sr_rl_max_order_dollars:
            qty = int(cfg.sr_rl_max_order_dollars // price)
            notional = qty * price
        if qty < int(cfg.sr_rl_min_qty) or notional <= 0:
            continue

        stop_loss = to_float(d.get("stop_loss"), 0.0)
        take_profit = to_float(d.get("take_profit"), 0.0)
        confidence = float(np.clip(cfg.sr_rl_default_confidence, 0.0, 1.0))
        reason = str(d.get("reason", "sr_rl_agent_buy"))

        rows.append(
            {
                "symbol": symbol,
                "side": "buy",
                "delta_shares": qty,
                "delta_dollars": notional,
                "price": price,
                "reason": f"sr_rl_agent:{reason}",
                "trade_confidence": confidence,
                "confidence": confidence,
                "confidence_score": confidence,
                "sr_rl_confirmed": True,
                "sr_rl_nearest_support": d.get("nearest_support"),
                "sr_rl_nearest_resistance": d.get("nearest_resistance"),
                "sr_rl_distance_to_support_pct": d.get("distance_to_support_pct"),
                "sr_rl_distance_to_resistance_pct": d.get("distance_to_resistance_pct"),
                "entry_price": price,
                "stop_loss": stop_loss if stop_loss > 0 else np.nan,
                "take_profit": take_profit if take_profit > 0 else np.nan,
                "trailing_stop_enabled": bool(cfg.sr_rl_submit_trailing_stop_after_fill),
                "trail_percent": float(cfg.sr_rl_trail_percent_default),
                "order_type": "sr_rl_market_entry_with_trailing_stop_exit",
                "planned_at": now_utc().isoformat(),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    try:
        out.to_csv(SR_RL_CONFIRMED_ORDERS_PATH, index=False)
    except Exception:
        pass
    return out


def merge_sr_rl_orders(base: pd.DataFrame, config: ScannerConfig | None = None) -> pd.DataFrame:
    cfg = config or ScannerConfig()
    if not cfg.sr_rl_merge_signals:
        return base if base is not None else pd.DataFrame()

    sr_orders = load_sr_rl_orders(cfg)
    if sr_orders.empty:
        return base if base is not None else pd.DataFrame()

    base = base.copy() if base is not None else pd.DataFrame()
    if not base.empty and "symbol" in base.columns:
        base["symbol"] = base["symbol"].map(safe_symbol)
        sr_symbols = set(sr_orders["symbol"].map(safe_symbol))
        base_side = base["side"].astype(str).str.lower() if "side" in base.columns else pd.Series(["buy"] * len(base), index=base.index)
        # SR-confirmed support entry replaces stale same-symbol buy rows.
        base = base[~((base["symbol"].isin(sr_symbols)) & (base_side == "buy"))].copy()

    merged = pd.concat([base, sr_orders], ignore_index=True, sort=False)
    print(f"[sr-rl] merged sr_orders={len(sr_orders)} total_rebalance_rows={len(merged)}")
    sr_write_event(
        "orders_merged",
        {
            "sr_rows": int(len(sr_orders)),
            "merged_rows": int(len(merged)),
            "signals_path": str(SR_RL_SIGNALS_PATH),
            "orders_path": str(SR_RL_CONFIRMED_ORDERS_PATH),
        },
    )
    return merged

def submit_trailing_stop_after_entry(
    broker: "AlpacaBroker",
    symbol: str,
    qty: int,
    row: pd.Series | dict[str, Any],
) -> None:
    """
    Submit a protective trailing stop after a buy fill.

    The stream-confirmed trade plan emits trailing_stop_enabled + trail_percent.
    This function is best-effort and will not block the executor if Alpaca
    rejects the trailing stop.
    """
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    enabled_raw = data.get("trailing_stop_enabled", data.get("trailing_stop", False))
    enabled = str(enabled_raw).lower() in {"1", "true", "yes"} or enabled_raw is True
    if not enabled or qty <= 0:
        return

    if TrailingStopOrderRequest is None:
        write_event("trailing_stop_unavailable", {"symbol": symbol, "reason": "TrailingStopOrderRequest missing"})
        print(f"[trailing-stop] unavailable for {symbol}: TrailingStopOrderRequest missing")
        return

    trail_percent = to_float(data.get("trail_percent"), 0.0)
    trail_price = to_float(data.get("trail_price"), 0.0)

    try:
        if trail_percent > 0:
            req = TrailingStopOrderRequest(
                symbol=safe_symbol(symbol),
                qty=int(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                trail_percent=str(round(trail_percent, 4)),
            )
        elif trail_price > 0:
            req = TrailingStopOrderRequest(
                symbol=safe_symbol(symbol),
                qty=int(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                trail_price=str(round(trail_price, 4)),
            )
        else:
            return

        order = broker.trading.submit_order(order_data=req)
        payload = {
            "symbol": symbol,
            "qty": int(qty),
            "trail_percent": trail_percent,
            "trail_price": trail_price,
            "order_id": str(getattr(order, "id", "")),
            "status": str(getattr(order, "status", "")),
        }
        print(f"[trailing-stop] submitted {symbol} qty={qty} trail_percent={trail_percent} trail_price={trail_price}")
        write_event("trailing_stop_submitted", payload)
    except Exception as exc:
        print(f"[trailing-stop] failed {symbol}: {exc}")
        write_event("trailing_stop_error", {"symbol": symbol, "qty": int(qty), "error": str(exc)})


# ======================================================================# Alpaca broker wrapper
# ======================================================================
class AlpacaBroker:
    def __init__(self, env_path: Path, paper: bool = True):
        load_dotenv(env_path, override=True)
        self.paper = bool(paper)
        self.api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise RuntimeError(f"Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in {env_path}")

        self.trading = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

    def get_account_snapshot(self) -> AccountSnapshot:
        try:
            acct = self.trading.get_account()
        except Exception as exc:
            if is_auth_error(exc):
                raise RuntimeError(alpaca_auth_help_message(self.paper)) from exc
            raise
        return AccountSnapshot(
            buying_power=to_float(getattr(acct, "buying_power", 0.0)),
            equity=to_float(getattr(acct, "equity", 0.0)),
            cash=to_float(getattr(acct, "cash", 0.0)),
            portfolio_value=to_float(getattr(acct, "portfolio_value", 0.0)),
        )

    def get_all_open_positions(self) -> list[BrokerPosition]:
        try:
            raw_positions = self.trading.get_all_positions()
        except Exception as exc:
            print(f"[broker] failed to fetch positions: {exc}")
            return []

        positions: list[BrokerPosition] = []
        for pos in raw_positions:
            qty = to_float(getattr(pos, "qty", 0.0))
            if qty <= 0:
                continue
            positions.append(
                BrokerPosition(
                    symbol=safe_symbol(getattr(pos, "symbol", "")),
                    qty=qty,
                    market_value=to_float(getattr(pos, "market_value", 0.0)),
                    avg_entry_price=to_float(getattr(pos, "avg_entry_price", 0.0)),
                    current_price=to_float(getattr(pos, "current_price", 0.0)),
                    unrealized_pl=to_float(getattr(pos, "unrealized_pl", 0.0)),
                    unrealized_plpc=to_float(getattr(pos, "unrealized_plpc", 0.0)),
                )
            )
        return positions

    def get_position(self, symbol: str) -> BrokerPosition:
        symbol = safe_symbol(symbol)
        for pos in self.get_all_open_positions():
            if pos.symbol == symbol:
                return pos
        return BrokerPosition(symbol=symbol)

    def list_open_orders(self) -> list[Any]:
        try:
            return self.trading.get_orders()
        except Exception:
            return []

    def has_open_order_for_symbol(self, symbol: str) -> bool:
        symbol = safe_symbol(symbol)
        open_statuses = {
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.ACCEPTED_FOR_BIDDING,
            OrderStatus.CALCULATED,
        }
        for order in self.list_open_orders():
            try:
                if safe_symbol(getattr(order, "symbol", "")) == symbol and order.status in open_statuses:
                    return True
            except Exception:
                continue
        return False

    def cancel_open_orders(self) -> None:
        self.trading.cancel_orders()

    def get_clock_info(self) -> dict[str, Any]:
        clock = self.trading.get_clock()
        return {
            "is_open": bool(getattr(clock, "is_open", False)),
            "timestamp": str(getattr(clock, "timestamp", "")),
            "next_open": str(getattr(clock, "next_open", "")),
            "next_close": str(getattr(clock, "next_close", "")),
        }

    def submit_market_order(self, symbol: str, side: str, qty: int, client_order_id: str, tif: TimeInForce):
        order = MarketOrderRequest(
            symbol=safe_symbol(symbol),
            qty=int(qty),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=tif,
            client_order_id=client_order_id,
        )
        return self.trading.submit_order(order_data=order)

    def submit_market_buy(self, symbol: str, qty: int, client_order_id: str, tif: TimeInForce):
        return self.submit_market_order(symbol, "buy", qty, client_order_id, tif)

    def submit_market_sell(self, symbol: str, qty: int, client_order_id: str, tif: TimeInForce):
        return self.submit_market_order(symbol, "sell", qty, client_order_id, tif)

    def get_order_by_id(self, order_id: str):
        return self.trading.get_order_by_id(order_id)


# ======================================================================# Scanner helpers
# ======================================================================
def _web_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }


def get_sp500_fallback_symbols() -> list[str]:
    try:
        resp = requests.get(WIKI_SP500_URL, headers=_web_headers(), timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables or "Symbol" not in tables[0].columns:
            raise RuntimeError("Could not parse S&P 500 symbols from Wikipedia.")
        symbols = []
        for raw in tables[0]["Symbol"].astype(str):
            if is_simple_alpaca_symbol(raw):
                symbols.append(normalize_symbol(raw))
        return sorted(set(symbols))
    except Exception as exc:
        print(f"[scanner] S&P 500 fallback failed: {exc}; using default universe")
        return list(DEFAULT_SCAN_UNIVERSE)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


class MomentumSetupScanner:
    def __init__(self, broker: AlpacaBroker, config: ScannerConfig, executor_config: ExecutorConfig):
        self.broker = broker
        self.config = config
        self.executor_config = executor_config

        if StockHistoricalDataClient is None:
            raise RuntimeError("alpaca-py data classes are unavailable. Install alpaca-py or disable scanner.")

        self.data = StockHistoricalDataClient(broker.api_key, broker.secret_key)
        self.llm_metadata_by_symbol: dict[str, dict[str, Any]] = {}
    def _trigger_options_booster_if_needed(
        self,
        candidate_rows: pd.DataFrame,
    ) -> None:
        """
        Optional options booster trigger.
        Safe no-op if booster unavailable.
        """

        try:
            if candidate_rows is None or candidate_rows.empty:
                return

            if not self.config.options_booster_enabled:
                return

            qualified = candidate_rows[
                candidate_rows["trade_confidence"]
                >= self.config.options_booster_confidence_threshold
            ]

            if qualified.empty:
                return

            symbols = (
                qualified["symbol"]
                .astype(str)
                .unique()
                .tolist()
            )

            print(
                "[options-booster] triggering "
                f"symbols={symbols}"
            )

            env = build_options_booster_env(
                self.config,
                candidate_symbols=symbols,
            )

            subprocess.Popen(
                [sys.executable, str(OPTIONS_BOOSTER_PATH)],
                cwd=str(PROJECT_ROOT),
                env={**os.environ, **env},
            )

        except Exception as exc:
            print(f"[options-booster] failed: {exc}")
        
    def _load_state(self) -> dict[str, Any]:
        if not SCANNER_STATE_PATH.exists():
            return {"high_plpc": {}}
        try:
            return json.loads(SCANNER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"high_plpc": {}}

    def _save_state(self, state: dict[str, Any]) -> None:
        SCANNER_STATE_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")

    def _universe(self) -> list[str]:
        if self.config.universe:
            raw_symbols = self.config.universe
            self.llm_metadata_by_symbol = {}
        elif self.config.use_llm_sentiment_universe:
            raw_symbols, self.llm_metadata_by_symbol = load_llm_sentiment_universe(
                refresh=self.config.refresh_llm_sentiment_universe,
                min_confidence=self.config.min_llm_universe_confidence,
            )
        elif self.config.use_sp500_fallback or self.config.broad_market_scan:
            raw_symbols = get_sp500_fallback_symbols()
            self.llm_metadata_by_symbol = {}
        else:
            raw_symbols = DEFAULT_SCAN_UNIVERSE
            self.llm_metadata_by_symbol = {}

        symbols: list[str] = []
        rejected: list[str] = []
        for raw in raw_symbols:
            if not is_simple_alpaca_symbol(raw):
                rejected.append(str(raw))
                continue
            symbol = normalize_symbol(raw)
            if symbol and symbol not in symbols:
                symbols.append(symbol)

        if rejected:
            print(f"[scanner] rejected non-simple Alpaca symbols={len(rejected)} examples={rejected[:8]}")
        return symbols[: self.config.max_universe_symbols]

    def _feed(self):
        if DataFeed is None:
            return None
        return DataFeed.IEX if str(self.config.data_feed).lower() == "iex" else DataFeed.SIP

    def _download_bars(self, symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        end = now_utc()
        start = end - timedelta(days=max(30, self.config.lookback_days))

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=self._feed(),
        )

        bars = self.data.get_stock_bars(req).df
        if bars is None or bars.empty:
            return pd.DataFrame()

        df = bars.reset_index().rename(columns={"level_0": "symbol", "level_1": "timestamp"})
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "symbol"})
        if "timestamp" not in df.columns and "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "timestamp"})

        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df.dropna(subset=["symbol", "timestamp", "close"]).sort_values(["symbol", "timestamp"])

    def _download_intraday_bars(self, symbols: list[str], minutes: int) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        if TimeFrameUnit is None:
            return pd.DataFrame()

        end = now_utc()
        start = end - timedelta(days=max(1, self.config.intraday_lookback_days))
        timeframe = TimeFrame(minutes, TimeFrameUnit.Minute)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=self._feed(),
        )

        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            print(f"[intraday] failed to download {minutes}Min bars: {exc}")
            return pd.DataFrame()

        if bars is None or bars.empty:
            return pd.DataFrame()

        df = bars.reset_index().rename(columns={"level_0": "symbol", "level_1": "timestamp"})
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "symbol"})
        if "timestamp" not in df.columns and "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "timestamp"})

        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timeframe_minutes"] = minutes
        return df.dropna(subset=["symbol", "timestamp", "close"]).sort_values(["symbol", "timestamp"])

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []

        for symbol, g in bars.groupby("symbol", sort=False):
            g = g.sort_values("timestamp").copy()
            c = g["close"].astype(float)
            v = g["volume"].astype(float)

            g["ret_1d"] = c.pct_change(1, fill_method=None)
            g["ret_3d"] = c.pct_change(3, fill_method=None)
            g["ret_5d"] = c.pct_change(5, fill_method=None)
            g["ret_10d"] = c.pct_change(10, fill_method=None)
            g["ret_20d"] = c.pct_change(20, fill_method=None)
            g["positive_daily_move"] = (g["ret_1d"] > 0).astype(float)
            g["sma5"] = c.rolling(5).mean()
            g["sma20"] = c.rolling(20).mean()
            g["sma50"] = c.rolling(50).mean()
            g["sma200"] = c.rolling(200).mean()
            g["sma5_cross_sma20"] = ((g["sma5"] > g["sma20"]) & (g["sma5"].shift(1) <= g["sma20"].shift(1))).astype(float)
            g["sma20_cross_sma50"] = ((g["sma20"] > g["sma50"]) & (g["sma20"].shift(1) <= g["sma50"].shift(1))).astype(float)
            g["sma50_cross_sma200"] = ((g["sma50"] > g["sma200"]) & (g["sma50"].shift(1) <= g["sma200"].shift(1))).astype(float)
            g["rsi14"] = _rsi(c, 14)

            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            g["macd"] = ema12 - ema26
            g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
            g["macd_hist"] = g["macd"] - g["macd_signal"]
            g["macd_hist_slope"] = g["macd_hist"].diff()
            g["macd_bullish"] = ((g["macd"] > g["macd_signal"]) & (g["macd_hist"] > 0)).astype(float)
            g["atr14"] = _atr(g, 14)
            g["atr_pct"] = g["atr14"] / c.replace(0, np.nan)
            g["rolling_std10"] = g["ret_1d"].rolling(10).std()
            g["rolling_std20"] = g["ret_1d"].rolling(20).std()
            g["vol20"] = v.rolling(20).mean()
            g["dollar_volume20"] = (c * v).rolling(20).mean()
            g["rel_volume"] = v / g["vol20"].replace(0, np.nan)

            mid = g["sma20"]
            sd = c.rolling(20).std()
            g["bb_low"] = mid - 2 * sd
            g["bb_high"] = mid + 2 * sd
            g["bb_pos"] = (c - g["bb_low"]) / (g["bb_high"] - g["bb_low"]).replace(0, np.nan)
            g["bb_breakout"] = ((c > g["bb_high"]) & (g["ret_1d"] > 0)).astype(float)
            g["bb_rejection"] = ((c < g["bb_high"]) & (c.shift(1) > g["bb_high"].shift(1)) & (g["ret_1d"] < 0)).astype(float)

            g["high20"] = c.rolling(20).max()
            g["drawdown_20h"] = c / g["high20"].replace(0, np.nan) - 1.0
            g["trend_stack"] = ((g["sma5"] > g["sma20"]) & (g["sma50"] > g["sma200"])).astype(float)
            g["above_sma20"] = (c > g["sma20"]).astype(float)
            g["above_prev_high"] = (c > g["high"].shift(1)).astype(float)
            g["lower_band_reclaim"] = ((c.shift(1) < g["bb_low"].shift(1)) & (c > g["bb_low"]) & (g["ret_1d"] > 0)).astype(float)
            g["overbought_rsi"] = (g["rsi14"] >= self.config.max_overbought_rsi).astype(float)
            g["weak_macd"] = ((g["macd_hist"] < 0) | (g["macd_hist_slope"] < 0)).astype(float)
            rows.append(g)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _pivot_levels(self, df: pd.DataFrame) -> tuple[list[float], list[float]]:
        """Extract simple pivot support/resistance levels from intraday bars."""
        if df.empty or len(df) < 10:
            return [], []

        g = df.sort_values("timestamp").copy()
        highs = g["high"].astype(float).reset_index(drop=True)
        lows = g["low"].astype(float).reset_index(drop=True)

        resistance: list[float] = []
        support: list[float] = []

        for i in range(2, len(g) - 2):
            h = highs.iloc[i]
            l = lows.iloc[i]
            if h >= highs.iloc[i - 2:i + 3].max():
                resistance.append(float(h))
            if l <= lows.iloc[i - 2:i + 3].min():
                support.append(float(l))

        # Include recent range extremes so the layer still works when pivots are sparse.
        support.extend([float(lows.tail(30).min()), float(lows.tail(60).min())])
        resistance.extend([float(highs.tail(30).max()), float(highs.tail(60).max())])

        return support, resistance

    def _cluster_price_levels(self, levels: list[float], tolerance_pct: float = 0.003) -> list[float]:
        """Cluster nearby price levels into cleaner S/R zones."""
        clean = sorted([float(x) for x in levels if np.isfinite(x) and x > 0])
        if not clean:
            return []

        clusters: list[list[float]] = []
        for level in clean:
            if not clusters:
                clusters.append([level])
                continue
            center = float(np.mean(clusters[-1]))
            if abs(level - center) / max(center, 1e-9) <= tolerance_pct:
                clusters[-1].append(level)
            else:
                clusters.append([level])

        return [float(np.mean(c)) for c in clusters]

    def _nearest_levels(self, current_price: float, support: list[float], resistance: list[float]) -> tuple[float, float]:
        supports_below = [x for x in support if x <= current_price]
        resistance_above = [x for x in resistance if x >= current_price]
        nearest_support = max(supports_below) if supports_below else (max(support) if support else 0.0)
        nearest_resistance = min(resistance_above) if resistance_above else (min(resistance) if resistance else 0.0)
        return float(nearest_support or 0.0), float(nearest_resistance or 0.0)

    def _entry_from_intraday_context(
        self,
        symbol: str,
        current_price: float,
        setup_type: str,
        is_breakout: bool,
        bars_1m: pd.DataFrame,
        bars_5m: pd.DataFrame,
        bars_15m: pd.DataFrame,
    ) -> dict[str, Any]:
        """Analyze 1m/5m/15m bars to estimate support, resistance, and a reasonable entry."""
        symbol = normalize_symbol(symbol)
        frames = {
            "1m": bars_1m[bars_1m["symbol"] == symbol].copy() if not bars_1m.empty else pd.DataFrame(),
            "5m": bars_5m[bars_5m["symbol"] == symbol].copy() if not bars_5m.empty else pd.DataFrame(),
            "15m": bars_15m[bars_15m["symbol"] == symbol].copy() if not bars_15m.empty else pd.DataFrame(),
        }

        all_support: list[float] = []
        all_resistance: list[float] = []
        timeframe_notes: dict[str, Any] = {}

        for label, frame in frames.items():
            if frame.empty:
                continue
            support, resistance = self._pivot_levels(frame)
            support = self._cluster_price_levels(support)
            resistance = self._cluster_price_levels(resistance)
            all_support.extend(support)
            all_resistance.extend(resistance)
            timeframe_notes[label] = {
                "bars": int(len(frame)),
                "support_levels": support[-5:],
                "resistance_levels": resistance[:5],
            }

        support_levels = self._cluster_price_levels(all_support)
        resistance_levels = self._cluster_price_levels(all_resistance)
        nearest_support, nearest_resistance = self._nearest_levels(current_price, support_levels, resistance_levels)

        if current_price <= 0:
            return {
                "entry_price": 0.0,
                "entry_type": "invalid_price",
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "risk_reward_ratio": 0.0,
                "stop_price": 0.0,
                "entry_notes": "invalid current price",
                "intraday_context": timeframe_notes,
            }

        support_entry = nearest_support * (1.0 + self.config.entry_near_support_buffer_pct) if nearest_support > 0 else current_price
        breakout_entry = nearest_resistance * (1.0 + self.config.entry_breakout_buffer_pct) if nearest_resistance > 0 else current_price
        max_chase_entry = current_price * (1.0 + self.config.max_entry_chase_pct)

        if is_breakout or setup_type == "momentum":
            entry_price = min(max(current_price, breakout_entry), max_chase_entry) if nearest_resistance > current_price else current_price
            entry_type = "breakout_confirmation_entry" if nearest_resistance > current_price else "market_momentum_entry"
        else:
            entry_price = min(current_price, support_entry) if nearest_support > 0 else current_price
            entry_type = "support_reclaim_entry"

        stop_price = nearest_support * 0.992 if nearest_support > 0 else current_price * 0.975
        target_price = nearest_resistance if nearest_resistance > entry_price else entry_price * 1.035
        risk = max(entry_price - stop_price, 0.01)
        reward = max(target_price - entry_price, 0.0)
        rr = reward / risk if risk > 0 else 0.0

        return {
            "entry_price": float(entry_price),
            "entry_type": entry_type,
            "nearest_support": float(nearest_support),
            "nearest_resistance": float(nearest_resistance),
            "support_levels": support_levels[-8:],
            "resistance_levels": resistance_levels[:8],
            "risk_reward_ratio": float(rr),
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "entry_notes": f"{entry_type}; support={nearest_support:.2f}; resistance={nearest_resistance:.2f}; rr={rr:.2f}",
            "intraday_context": timeframe_notes,
        }

    def _annotate_intraday_entries(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty or not self.config.analyze_intraday_entries:
            return candidates

        symbols = [normalize_symbol(x) for x in candidates["symbol"].tolist() if normalize_symbol(x)]
        if not symbols:
            return candidates

        print(f"[intraday] analyzing 1m/5m/15m support-resistance for candidates={len(symbols)}")
        bars_1m = self._download_intraday_bars(symbols, 1)
        bars_5m = self._download_intraday_bars(symbols, 5)
        bars_15m = self._download_intraday_bars(symbols, 15)

        rows: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            symbol = normalize_symbol(row["symbol"])
            current_price = to_float(row.get("close"))
            is_breakout = bool(to_float(row.get("confirmed_breakout_score")) >= self.config.min_breakout_confirmation_score)
            context = self._entry_from_intraday_context(
                symbol=symbol,
                current_price=current_price,
                setup_type=str(row.get("setup_type", "")),
                is_breakout=is_breakout,
                bars_1m=bars_1m,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
            )
            merged = row.to_dict()
            merged.update(context)
            rows.append(merged)

        out = pd.DataFrame(rows)
        if not out.empty:
            print("[intraday] entry analysis:")
            show = ["symbol", "close", "entry_price", "entry_type", "nearest_support", "nearest_resistance", "risk_reward_ratio"]
            print(out[[c for c in show if c in out.columns]].to_string(index=False))
        return out


    def _weighted_pivot_zones(self, df: pd.DataFrame, kind: str = "support") -> list[dict[str, Any]]:
        """Create weighted pivot zones from intraday bars.

        kind="support" clusters pivot lows. kind="resistance" clusters pivot highs.
        Weight is intentionally simple and transparent: pivot touches + nearby wick/body
        interactions + volume confirmation. The strategy buys support pivots, not exact
        prices, so every level is treated as a zone using pivot_zone_tolerance_pct.
        """
        if df is None or df.empty or len(df) < 20:
            return []

        g = df.copy()
        g["timestamp"] = pd.to_datetime(g.get("timestamp"), errors="coerce", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in g.columns:
                g[col] = 0.0
            g[col] = pd.to_numeric(g[col], errors="coerce")
        g = g.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
        if len(g) < 20:
            return []

        support, resistance = self._pivot_levels(g)
        raw_levels = support if kind == "support" else resistance
        raw_levels = [float(x) for x in raw_levels if np.isfinite(x) and x > 0]
        if not raw_levels:
            return []

        tol = float(self.config.pivot_zone_tolerance_pct)
        clustered = self._cluster_price_levels(raw_levels, tolerance_pct=tol)
        if not clustered:
            return []

        vol = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
        vol20 = vol.rolling(20).mean().replace(0, np.nan)
        zones: list[dict[str, Any]] = []
        for level in clustered:
            if kind == "support":
                wick_touches = (pd.to_numeric(g["low"], errors="coerce") - level).abs() / max(level, 1e-9) <= tol
                body_touches = (pd.to_numeric(g["close"], errors="coerce") - level).abs() / max(level, 1e-9) <= tol
            else:
                wick_touches = (pd.to_numeric(g["high"], errors="coerce") - level).abs() / max(level, 1e-9) <= tol
                body_touches = (pd.to_numeric(g["close"], errors="coerce") - level).abs() / max(level, 1e-9) <= tol
            touch_count = int(wick_touches.sum())
            body_count = int(body_touches.sum())
            touch_volume = float(vol[wick_touches | body_touches].sum())
            avg_bar_volume = float(vol.tail(100).mean()) if len(vol) else 0.0
            volume_bonus = min(5.0, touch_volume / max(avg_bar_volume * 10.0, 1.0)) if avg_bar_volume > 0 else 0.0
            recent_bonus = 0.0
            recent_slice = g.tail(60)
            if not recent_slice.empty:
                if kind == "support":
                    recent_touch = ((pd.to_numeric(recent_slice["low"], errors="coerce") - level).abs() / max(level, 1e-9) <= tol).any()
                else:
                    recent_touch = ((pd.to_numeric(recent_slice["high"], errors="coerce") - level).abs() / max(level, 1e-9) <= tol).any()
                recent_bonus = 2.0 if bool(recent_touch) else 0.0
            weight = float(touch_count * 1.5 + body_count * 0.75 + volume_bonus + recent_bonus)
            zones.append({
                "price": float(level),
                "zone_low": float(level * (1.0 - tol)),
                "zone_high": float(level * (1.0 + tol)),
                "touch_count": touch_count,
                "body_count": body_count,
                "weight": weight,
                "kind": kind,
            })

        zones = [z for z in zones if z["touch_count"] >= self.config.pivot_min_touch_count]
        return sorted(zones, key=lambda z: (z["price"], z["weight"]))

    def _pivot_reversal_metrics(self, frame_1m: pd.DataFrame, frame_5m: pd.DataFrame, support_zone: dict[str, Any]) -> dict[str, Any]:
        """Score whether price has reversed upward from a support pivot zone."""
        if frame_1m is None or frame_1m.empty:
            return {"reversal_score": 0.0, "reason": "no_1m_frame"}

        g1 = frame_1m.copy().sort_values("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            g1[col] = pd.to_numeric(g1.get(col), errors="coerce")
        g1 = g1.dropna(subset=["open", "high", "low", "close"])
        if len(g1) < 20:
            return {"reversal_score": 0.0, "reason": "not_enough_1m_bars"}

        close = g1["close"]
        high = g1["high"]
        low = g1["low"]
        vol = g1["volume"].fillna(0.0)
        current = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else current
        zone_price = float(support_zone.get("price", 0.0))
        zone_low = float(support_zone.get("zone_low", zone_price))
        zone_high = float(support_zone.get("zone_high", zone_price))
        buy_zone_high = zone_price * (1.0 + float(self.config.pivot_buy_zone_pct))
        buy_zone_low = zone_price * (1.0 - float(self.config.pivot_buy_zone_pct))

        typical = ((high + low + close) / 3.0).astype(float)
        vwap = float((typical * vol).tail(60).sum() / max(vol.tail(60).sum(), 1.0)) if float(vol.tail(60).sum()) > 0 else float(close.tail(20).mean())
        rel_volume = float(vol.iloc[-1] / max(vol.tail(20).mean(), 1.0)) if len(vol) >= 20 else 1.0
        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        macd_hist = float((ema_fast - ema_slow).iloc[-1] - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean().iloc[-1])
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100.0 - (100.0 / (1.0 + rs))).fillna(50.0).iloc[-1])

        recent = g1.tail(20)
        swept_or_tagged = bool((pd.to_numeric(recent["low"], errors="coerce") <= zone_high).any())
        reclaimed = bool(current > zone_price and current > prev_close)
        near_zone = bool(buy_zone_low <= current <= buy_zone_high)
        above_vwap = bool(current >= vwap)
        ema_turn = bool(ema_fast.iloc[-1] >= ema_slow.iloc[-1] or ema_fast.iloc[-1] > ema_fast.iloc[-2])
        bullish_candle = bool(close.iloc[-1] > g1["open"].iloc[-1] and close.iloc[-1] > prev_close)
        volume_ok = bool(rel_volume >= float(self.config.pivot_min_rel_volume))
        rsi_ok = bool(rsi >= 48.0)
        macd_ok = bool(macd_hist >= 0.0)

        score = 0.0
        score += 0.18 if near_zone else 0.0
        score += 0.16 if swept_or_tagged else 0.0
        score += 0.14 if reclaimed else 0.0
        score += 0.13 if above_vwap else 0.0
        score += 0.12 if ema_turn else 0.0
        score += 0.10 if bullish_candle else 0.0
        score += 0.09 if volume_ok else 0.0
        score += 0.04 if rsi_ok else 0.0
        score += 0.04 if macd_ok else 0.0

        # 5m confirmation prevents buying every tiny 1m bounce.
        five_min_confirmed = True
        if frame_5m is not None and not frame_5m.empty and len(frame_5m) >= 8:
            g5 = frame_5m.copy().sort_values("timestamp")
            g5_close = pd.to_numeric(g5["close"], errors="coerce").dropna()
            if len(g5_close) >= 8:
                five_min_confirmed = bool(g5_close.iloc[-1] >= g5_close.rolling(5).mean().iloc[-1] or g5_close.iloc[-1] > g5_close.iloc[-2])
                score += 0.08 if five_min_confirmed else -0.08

        return {
            "reversal_score": float(max(0.0, min(1.0, score))),
            "current_price": current,
            "vwap": vwap,
            "rel_volume": rel_volume,
            "rsi14": rsi,
            "macd_hist": macd_hist,
            "near_zone": near_zone,
            "swept_or_tagged": swept_or_tagged,
            "reclaimed": reclaimed,
            "above_vwap": above_vwap,
            "ema_turn": ema_turn,
            "bullish_candle": bullish_candle,
            "volume_ok": volume_ok,
            "five_min_confirmed": five_min_confirmed,
        }

    def _pivot_signal_for_symbol(self, symbol: str, bars_1m: pd.DataFrame, bars_5m: pd.DataFrame, bars_15m: pd.DataFrame) -> dict[str, Any]:
        """Find a buyable support-pivot pullback/reclaim setup for one symbol."""
        symbol = normalize_symbol(symbol)
        if not self.config.pivot_buy_enabled:
            return {"symbol": symbol, "available": False, "reason": "pivot_buy_disabled"}
        if bars_1m is None or bars_1m.empty:
            return {"symbol": symbol, "available": False, "reason": "no_1m_bars"}

        f1 = bars_1m[bars_1m["symbol"].map(normalize_symbol) == symbol].copy() if "symbol" in bars_1m.columns else bars_1m.copy()
        f5 = bars_5m[bars_5m["symbol"].map(normalize_symbol) == symbol].copy() if bars_5m is not None and not bars_5m.empty and "symbol" in bars_5m.columns else pd.DataFrame()
        f15 = bars_15m[bars_15m["symbol"].map(normalize_symbol) == symbol].copy() if bars_15m is not None and not bars_15m.empty and "symbol" in bars_15m.columns else pd.DataFrame()
        if f1.empty:
            return {"symbol": symbol, "available": False, "reason": "no_symbol_1m_bars"}

        support_zones = self._weighted_pivot_zones(f1, "support") + self._weighted_pivot_zones(f5, "support") + self._weighted_pivot_zones(f15, "support")
        resistance_zones = self._weighted_pivot_zones(f1, "resistance") + self._weighted_pivot_zones(f5, "resistance") + self._weighted_pivot_zones(f15, "resistance")
        support_zones = sorted(support_zones, key=lambda z: (z["price"], -z["weight"]))
        resistance_zones = sorted(resistance_zones, key=lambda z: (z["price"], -z["weight"]))
        if not support_zones:
            return {"symbol": symbol, "available": False, "reason": "no_support_pivots"}

        f1 = f1.sort_values("timestamp").copy()
        current = float(pd.to_numeric(f1["close"], errors="coerce").dropna().iloc[-1])
        candidates: list[dict[str, Any]] = []
        for zone in support_zones:
            level = float(zone.get("price", 0.0))
            if level <= 0:
                continue
            distance_pct = abs(current - level) / max(level, 1e-9)
            if current < level * (1.0 - self.config.pivot_buy_zone_pct) or distance_pct > self.config.pivot_buy_zone_pct:
                continue
            if float(zone.get("weight", 0.0)) < self.config.pivot_min_weight:
                continue
            metrics = self._pivot_reversal_metrics(f1, f5, zone)
            if float(metrics.get("reversal_score", 0.0)) < self.config.pivot_min_reversal_score:
                continue
            if not bool(metrics.get("near_zone", False)) or not bool(metrics.get("reclaimed", False)):
                continue

            recent_low = float(pd.to_numeric(f1["low"], errors="coerce").tail(20).min())
            stop = min(recent_low, level) * (1.0 - self.config.pivot_stop_buffer_pct)
            entry = current if self.config.pivot_allow_market_entry_when_confirmed else min(current, level * (1.0 + self.config.pivot_buy_zone_pct))
            risk = max(entry - stop, 0.01)
            rr_target = entry + risk * self.config.pivot_target_rr
            resistance_above = [float(r["price"]) for r in resistance_zones if float(r.get("price", 0.0)) > entry]
            next_resistance = min(resistance_above) if resistance_above else 0.0
            target = rr_target
            if next_resistance > entry and next_resistance >= entry + risk * self.config.pivot_min_rr:
                target = min(next_resistance, rr_target) if next_resistance >= rr_target else next_resistance
            rr = (target - entry) / risk if risk > 0 else 0.0
            if rr < self.config.pivot_min_rr:
                continue
            candidates.append({
                "symbol": symbol,
                "available": True,
                "direction": "long",
                "side": "buy",
                "setup_type": "support_pivot_pullback_reclaim",
                "reason": "buy_validated_support_pivot_reversal",
                "entry_price": float(entry),
                "market_price": float(current),
                "stop_price": float(stop),
                "target_price": float(target),
                "risk_reward_ratio": float(rr),
                "nearest_support": float(level),
                "nearest_resistance": float(next_resistance),
                "support_zone": zone,
                "support_levels": support_zones[-8:],
                "resistance_levels": resistance_zones[:8],
                "reversal_metrics": metrics,
                "setup_score": float(min(1.0, 0.45 + 0.35 * metrics.get("reversal_score", 0.0) + 0.20 * min(zone.get("weight", 0.0) / 12.0, 1.0))),
                "trade_confidence": float(min(0.90, 0.50 + 0.35 * metrics.get("reversal_score", 0.0) + 0.10 * min(zone.get("weight", 0.0) / 10.0, 1.0))),
            })

        if not candidates:
            return {"symbol": symbol, "available": False, "reason": "no_valid_pivot_pullback_reclaim", "current_price": current}
        return sorted(candidates, key=lambda x: (x["trade_confidence"], x["setup_score"], x["risk_reward_ratio"]), reverse=True)[0]

    def _build_pivot_point_orders(
        self,
        symbols: list[str],
        positions: dict[str, BrokerPosition],
        equity: float,
    ) -> pd.DataFrame:
        """Create BUY orders from validated support pivot pullbacks/reclaims."""
        if not self.config.pivot_buy_enabled or not symbols:
            return pd.DataFrame()
        symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
        if not symbols:
            return pd.DataFrame()

        print(f"[pivot-buy] scanning symbols={len(symbols)} for support pivot pullback/reclaim entries")
        old_lookback = self.config.intraday_lookback_days
        try:
            self.config.intraday_lookback_days = max(old_lookback, self.config.pivot_scan_lookback_days)
            bars_1m = self._download_intraday_bars(symbols, 1)
            bars_5m = self._download_intraday_bars(symbols, 5)
            bars_15m = self._download_intraday_bars(symbols, 15)
        finally:
            self.config.intraday_lookback_days = old_lookback

        if bars_1m.empty:
            print("[pivot-buy] no 1m bars available")
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for symbol in dict.fromkeys(symbols):
            sig = self._pivot_signal_for_symbol(symbol, bars_1m, bars_5m, bars_15m)
            if not sig.get("available"):
                continue

            price = to_float(sig.get("entry_price"), to_float(sig.get("market_price")))
            market_price = to_float(sig.get("market_price"), price)
            if price <= 0 or market_price <= 0:
                continue

            current_mv = float(getattr(positions.get(symbol), "market_value", 0.0)) if symbol in positions else 0.0
            current_weight = current_mv / max(equity, 1e-9)
            target_weight = min(
                self.config.pivot_max_target_weight,
                self.config.pivot_base_target_weight * min(1.75, max(0.75, to_float(sig.get("trade_confidence"), 0.65) / 0.60)),
                self.executor_config.max_position_weight,
            )
            delta_weight = target_weight - current_weight
            if delta_weight < self.executor_config.min_weight_delta:
                continue
            delta_dollars = delta_weight * equity
            qty = int(delta_dollars // price)
            if qty <= 0 or delta_dollars < self.executor_config.min_notional_to_trade:
                continue

            rows.append({
                "symbol": symbol,
                "side": "buy",
                "delta_shares": qty,
                "order_qty": qty,
                "delta_dollars": float(delta_dollars),
                "abs_delta_dollars": float(abs(delta_dollars)),
                "capped_dollars": float(abs(delta_dollars)),
                "price": float(price),
                "market_price": float(market_price),
                "entry_price": float(price),
                "entry_type": str(sig.get("setup_type", "support_pivot_pullback_reclaim")),
                "reason": str(sig.get("reason", "buy_validated_support_pivot_reversal")),
                "source": "pivot_point_buy_strategy",
                "target_weight": float(target_weight),
                "current_weight": float(current_weight),
                "delta_weight": float(delta_weight),
                "trade_confidence": to_float(sig.get("trade_confidence"), 0.66),
                "confidence_score": to_float(sig.get("trade_confidence"), 0.66),
                "continuation_probability": to_float(sig.get("trade_confidence"), 0.66),
                "setup_score": to_float(sig.get("setup_score"), 0.66),
                "risk_reward_ratio": to_float(sig.get("risk_reward_ratio"), 2.0),
                "stop_price": to_float(sig.get("stop_price")),
                "target_price": to_float(sig.get("target_price")),
                "nearest_support": to_float(sig.get("nearest_support")),
                "nearest_resistance": to_float(sig.get("nearest_resistance")),
                "support_levels": json.dumps(sig.get("support_levels", []), default=json_default),
                "resistance_levels": json.dumps(sig.get("resistance_levels", []), default=json_default),
                "entry_notes": json.dumps({"support_zone": sig.get("support_zone", {}), "reversal_metrics": sig.get("reversal_metrics", {})}, default=json_default),
                "stream_confirmed": True,
                "pivot_buy_signal": True,
                "rel_volume": to_float(sig.get("reversal_metrics", {}).get("rel_volume")),
                "rsi14": to_float(sig.get("reversal_metrics", {}).get("rsi14")),
                "macd_hist": to_float(sig.get("reversal_metrics", {}).get("macd_hist")),
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            print("[pivot-buy] generated orders:")
            show = ["symbol", "side", "entry_type", "price", "nearest_support", "stop_price", "target_price", "risk_reward_ratio", "trade_confidence"]
            print(out[[c for c in show if c in out.columns]].to_string(index=False))
            write_event("pivot_buy_orders_generated", {"rows": out.to_dict(orient="records")})
        else:
            print("[pivot-buy] no validated support pivot pullback/reclaim setups")
        return out


    def _today_8am_orb_range(self, frame_1m: pd.DataFrame) -> dict[str, Any]:
        """
        Build the 8:00-8:59 ET one-hour range from 1m bars.

        The returned high/low define the daily ORB dealing range used by the
        liquidity-sweep/FVG model.
        """
        if frame_1m is None or frame_1m.empty or "timestamp" not in frame_1m.columns:
            return {"available": False, "reason": "no_1m_bars"}

        g = frame_1m.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], errors="coerce", utc=True)
        g = g.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
        if g.empty:
            return {"available": False, "reason": "no_valid_1m_bars"}

        g["timestamp_et"] = g["timestamp"].dt.tz_convert(ET)
        today_et = datetime.now(ET).date()
        hour = int(self.config.orb_8am_hour_et)
        range_bars = g[
            (g["timestamp_et"].dt.date == today_et)
            & (g["timestamp_et"].dt.hour == hour)
        ].copy()

        if len(range_bars) < 20:
            return {"available": False, "reason": f"not_enough_{hour}am_bars", "bar_count": int(len(range_bars))}

        high = float(pd.to_numeric(range_bars["high"], errors="coerce").max())
        low = float(pd.to_numeric(range_bars["low"], errors="coerce").min())
        open_ = float(pd.to_numeric(range_bars["open"], errors="coerce").iloc[0])
        close = float(pd.to_numeric(range_bars["close"], errors="coerce").iloc[-1])
        mid = (high + low) / 2.0
        range_pct = (high - low) / max(mid, 1e-9)

        if not np.isfinite(high) or not np.isfinite(low) or high <= low or range_pct <= 0:
            return {"available": False, "reason": "invalid_8am_range", "bar_count": int(len(range_bars))}

        if range_pct < self.config.orb_min_range_pct:
            return {"available": False, "reason": "8am_range_too_tight", "high": high, "low": low, "range_pct": range_pct}
        if range_pct > self.config.orb_max_range_pct:
            return {"available": False, "reason": "8am_range_too_wide", "high": high, "low": low, "range_pct": range_pct}

        return {
            "available": True,
            "hour_et": hour,
            "bar_count": int(len(range_bars)),
            "start_et": range_bars["timestamp_et"].iloc[0].isoformat(),
            "end_et": range_bars["timestamp_et"].iloc[-1].isoformat(),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "mid": mid,
            "range_pct": range_pct,
        }

    def _find_recent_fvg(self, after_bars: pd.DataFrame, direction: str) -> dict[str, Any]:
        """
        Three-candle imbalance detector on 1m bars.

        bullish FVG: candle[i].low > candle[i-2].high
        bearish FVG: candle[i].high < candle[i-2].low

        Returns the latest valid gap and whether price has pulled back/retested it.
        """
        if after_bars is None or after_bars.empty or len(after_bars) < 5:
            return {"available": False, "reason": "not_enough_bars_for_fvg"}

        g = after_bars.dropna(subset=["high", "low", "close", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(g) < 5:
            return {"available": False, "reason": "not_enough_clean_bars_for_fvg"}

        gaps: list[dict[str, Any]] = []
        min_gap_pct = float(self.config.orb_fvg_min_gap_pct)

        for i in range(2, len(g)):
            b0 = g.iloc[i - 2]
            b2 = g.iloc[i]
            if direction == "long":
                lower = float(b0["high"])
                upper = float(b2["low"])
                gap_pct = (upper - lower) / max(float(b2["close"]), 1e-9)
                if upper > lower and gap_pct >= min_gap_pct:
                    gaps.append({"index": i, "timestamp": b2["timestamp"], "fvg_low": lower, "fvg_high": upper, "gap_pct": gap_pct, "kind": "bullish_fvg"})
            else:
                lower = float(b2["high"])
                upper = float(b0["low"])
                gap_pct = (upper - lower) / max(float(b2["close"]), 1e-9)
                if upper > lower and gap_pct >= min_gap_pct:
                    gaps.append({"index": i, "timestamp": b2["timestamp"], "fvg_low": lower, "fvg_high": upper, "gap_pct": gap_pct, "kind": "bearish_fvg"})

        if not gaps:
            return {"available": False, "reason": f"no_{direction}_fvg"}

        fvg = gaps[-1]
        recent = g.iloc[int(fvg["index"]) + 1:].copy()
        current_price = float(g["close"].iloc[-1])
        fvg_low = float(fvg["fvg_low"])
        fvg_high = float(fvg["fvg_high"])
        tol = float(self.config.orb_retest_tolerance_pct)

        if recent.empty:
            retested = False
        elif direction == "long":
            retested = bool((pd.to_numeric(recent["low"], errors="coerce") <= fvg_high * (1.0 + tol)).any())
        else:
            retested = bool((pd.to_numeric(recent["high"], errors="coerce") >= fvg_low * (1.0 - tol)).any())

        in_retest_zone = fvg_low * (1.0 - tol) <= current_price <= fvg_high * (1.0 + tol)
        fvg.update({
            "available": True,
            "current_price": current_price,
            "retested": bool(retested),
            "in_retest_zone": bool(in_retest_zone),
            "mid": (fvg_low + fvg_high) / 2.0,
        })
        return fvg

    def _orb_8am_signal_for_symbol(self, symbol: str, frame_1m: pd.DataFrame) -> dict[str, Any]:
        """
        Detect liquidity sweep + rejection + FVG pullback using the 8am 1H range.

        Long:
            sweep below 8am low, reclaim back inside range, bullish FVG, retest.

        Bearish/short:
            sweep above 8am high, reject back inside range, bearish FVG, retest.
            By default this is used to exit existing longs only unless
            ORB_ALLOW_SHORT_ENTRIES=1.
        """
        symbol = normalize_symbol(symbol)
        if not self.config.orb_8am_enabled:
            return {"symbol": symbol, "available": False, "reason": "orb_disabled"}

        if frame_1m is None or frame_1m.empty:
            return {"symbol": symbol, "available": False, "reason": "no_1m_bars"}

        g = frame_1m[frame_1m["symbol"].map(normalize_symbol) == symbol].copy() if "symbol" in frame_1m.columns else frame_1m.copy()
        if g.empty:
            return {"symbol": symbol, "available": False, "reason": "no_symbol_1m_bars"}

        g["timestamp"] = pd.to_datetime(g["timestamp"], errors="coerce", utc=True)
        g = g.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
        if g.empty:
            return {"symbol": symbol, "available": False, "reason": "no_clean_symbol_1m_bars"}

        orb = self._today_8am_orb_range(g)
        if not orb.get("available"):
            return {"symbol": symbol, "available": False, "reason": orb.get("reason", "orb_unavailable"), "orb": orb}

        hour = int(self.config.orb_8am_hour_et)
        g["timestamp_et"] = g["timestamp"].dt.tz_convert(ET)
        today_et = datetime.now(ET).date()
        after = g[(g["timestamp_et"].dt.date == today_et) & (g["timestamp_et"].dt.hour > hour)].copy()
        if len(after) < 8:
            return {"symbol": symbol, "available": False, "reason": "not_enough_post_8am_bars", "orb": orb}

        high = float(orb["high"])
        low = float(orb["low"])
        mid = float(orb["mid"])
        range_size = high - low
        current = float(after["close"].iloc[-1])
        sweep_buffer = float(self.config.orb_sweep_buffer_pct)
        reclaim_buffer = float(self.config.orb_reclaim_buffer_pct)

        above_sweep_mask = pd.to_numeric(after["high"], errors="coerce") > high * (1.0 + sweep_buffer)
        below_sweep_mask = pd.to_numeric(after["low"], errors="coerce") < low * (1.0 - sweep_buffer)

        signals: list[dict[str, Any]] = []

        if below_sweep_mask.any() and self.config.orb_allow_long_entries:
            sweep_idx = int(np.where(below_sweep_mask.to_numpy())[0][-1])
            sweep_bar = after.iloc[sweep_idx]
            after_sweep = after.iloc[sweep_idx:].copy()
            reclaimed = bool((pd.to_numeric(after_sweep["close"], errors="coerce") > low * (1.0 + reclaim_buffer)).any())
            if reclaimed:
                fvg = self._find_recent_fvg(after_sweep, "long")
                if fvg.get("available") and (fvg.get("retested") or fvg.get("in_retest_zone") or not self.config.orb_require_fvg_retest):
                    stop = float(sweep_bar["low"]) * (1.0 - max(0.0005, sweep_buffer))
                    entry = float(fvg.get("mid") or current)
                    if not np.isfinite(entry) or entry <= 0:
                        entry = current
                    risk = max(entry - stop, 0.01)
                    rr_target = entry + risk * float(self.config.orb_target_rr)
                    target = min(high, rr_target) if high > entry else rr_target
                    rr = (target - entry) / risk
                    age_min = (now_utc() - pd.Timestamp(fvg["timestamp"]).to_pydatetime()).total_seconds() / 60.0
                    signals.append({
                        "symbol": symbol,
                        "available": True,
                        "direction": "long",
                        "side": "buy",
                        "setup_type": "orb_8am_low_sweep_bullish_fvg_retest",
                        "reason": "orb_8am_low_sweep_reclaim_inside_range_bullish_fvg_retest",
                        "entry_price": entry,
                        "market_price": current,
                        "stop_price": stop,
                        "target_price": target,
                        "opposing_range_side": high,
                        "risk_reward_ratio": rr,
                        "age_minutes": age_min,
                        "sweep_wick": float(sweep_bar["low"]),
                        "swept_level": low,
                        "orb": orb,
                        "fvg": fvg,
                    })

        if above_sweep_mask.any():
            sweep_idx = int(np.where(above_sweep_mask.to_numpy())[0][-1])
            sweep_bar = after.iloc[sweep_idx]
            after_sweep = after.iloc[sweep_idx:].copy()
            rejected = bool((pd.to_numeric(after_sweep["close"], errors="coerce") < high * (1.0 - reclaim_buffer)).any())
            if rejected:
                fvg = self._find_recent_fvg(after_sweep, "short")
                if fvg.get("available") and (fvg.get("retested") or fvg.get("in_retest_zone") or not self.config.orb_require_fvg_retest):
                    stop = float(sweep_bar["high"]) * (1.0 + max(0.0005, sweep_buffer))
                    entry = float(fvg.get("mid") or current)
                    if not np.isfinite(entry) or entry <= 0:
                        entry = current
                    risk = max(stop - entry, 0.01)
                    rr_target = entry - risk * float(self.config.orb_target_rr)
                    target = max(low, rr_target) if low < entry else rr_target
                    rr = (entry - target) / risk
                    age_min = (now_utc() - pd.Timestamp(fvg["timestamp"]).to_pydatetime()).total_seconds() / 60.0
                    signals.append({
                        "symbol": symbol,
                        "available": True,
                        "direction": "short",
                        "side": "sell",
                        "setup_type": "orb_8am_high_sweep_bearish_fvg_retest",
                        "reason": "orb_8am_high_sweep_reject_back_inside_range_bearish_fvg_retest",
                        "entry_price": entry,
                        "market_price": current,
                        "stop_price": stop,
                        "target_price": target,
                        "opposing_range_side": low,
                        "risk_reward_ratio": rr,
                        "age_minutes": age_min,
                        "sweep_wick": float(sweep_bar["high"]),
                        "swept_level": high,
                        "orb": orb,
                        "fvg": fvg,
                    })

        if not signals:
            return {
                "symbol": symbol,
                "available": False,
                "reason": "no_completed_orb_sweep_fvg_retest",
                "orb": orb,
                "current_price": current,
                "swept_above": bool(above_sweep_mask.any()),
                "swept_below": bool(below_sweep_mask.any()),
            }

        # Prefer the freshest qualifying signal.
        signals = [s for s in signals if s["risk_reward_ratio"] >= self.config.orb_min_rr and s["age_minutes"] <= self.config.orb_entry_max_age_minutes]
        if not signals:
            return {"symbol": symbol, "available": False, "reason": "orb_signal_failed_rr_or_age_filter", "orb": orb, "current_price": current}
        return sorted(signals, key=lambda x: x["age_minutes"])[0]

    def _build_orb_8am_orders(
        self,
        symbols: list[str],
        positions: dict[str, BrokerPosition],
        equity: float,
    ) -> pd.DataFrame:
        """Create ORB orders from the 8am range liquidity-sweep strategy."""
        if not self.config.orb_8am_enabled or not symbols:
            return pd.DataFrame()

        symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
        if not symbols:
            return pd.DataFrame()

        print(f"[orb-8am] scanning symbols={len(symbols)} for 8am range sweeps/FVG retests")
        old_lookback = self.config.intraday_lookback_days
        try:
            self.config.intraday_lookback_days = max(old_lookback, self.config.orb_scan_lookback_days)
            bars_1m = self._download_intraday_bars(symbols, 1)
        finally:
            self.config.intraday_lookback_days = old_lookback

        if bars_1m.empty:
            print("[orb-8am] no 1m bars available")
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            sig = self._orb_8am_signal_for_symbol(symbol, bars_1m)
            if not sig.get("available"):
                continue

            direction = str(sig.get("direction", ""))
            side = str(sig.get("side", "buy")).lower()
            price = to_float(sig.get("entry_price"), to_float(sig.get("market_price")))
            market_price = to_float(sig.get("market_price"), price)
            if price <= 0 or market_price <= 0:
                continue

            current_mv = float(getattr(positions.get(symbol), "market_value", 0.0)) if symbol in positions else 0.0
            current_qty = int(float(getattr(positions.get(symbol), "qty", 0.0))) if symbol in positions else 0
            current_weight = current_mv / max(equity, 1e-9)

            if direction == "short":
                if current_qty > 0 and self.config.orb_exit_longs_on_bearish_sweep:
                    qty = current_qty
                    delta_dollars = qty * market_price
                    target_weight = 0.0
                    delta_weight = -current_weight
                    reason = str(sig.get("reason")) + "|exit_long_after_high_sweep_rejection"
                elif self.config.orb_allow_short_entries:
                    # Alpaca sell can open a short only if account/margin permissions allow it.
                    # Keep sizing small and make it opt-in through ORB_ALLOW_SHORT_ENTRIES=1.
                    delta_dollars = min(
                        equity * self.config.orb_base_target_weight,
                        self.executor_config.max_order_dollars,
                        self.config.orb_max_target_weight * equity,
                    )
                    qty = int(delta_dollars // market_price)
                    target_weight = -delta_dollars / max(equity, 1e-9)
                    delta_weight = target_weight - current_weight
                    reason = str(sig.get("reason")) + "|optional_short_entry"
                else:
                    write_event("orb_8am_short_signal_shadow_only", sig)
                    continue
            else:
                if not self.config.orb_allow_long_entries:
                    continue
                target_weight = min(
                    self.config.orb_max_target_weight,
                    self.config.orb_base_target_weight * min(1.50, max(1.0, to_float(sig.get("risk_reward_ratio"), 2.0) / 2.0)),
                    self.executor_config.max_position_weight,
                )
                delta_weight = target_weight - current_weight
                if delta_weight < self.executor_config.min_weight_delta:
                    continue
                delta_dollars = delta_weight * equity
                qty = int(delta_dollars // price)
                reason = str(sig.get("reason"))

            if qty <= 0 or abs(delta_dollars) < self.executor_config.min_notional_to_trade:
                continue

            rows.append({
                "symbol": symbol,
                "side": side,
                "delta_shares": qty,
                "order_qty": qty,
                "delta_dollars": float(abs(delta_dollars)),
                "abs_delta_dollars": float(abs(delta_dollars)),
                "capped_dollars": float(abs(delta_dollars)),
                "price": float(price),
                "market_price": float(market_price),
                "entry_price": float(price),
                "entry_type": str(sig.get("setup_type", "")),
                "reason": reason,
                "source": "orb_8am_liquidity_sweep_fvg",
                "target_weight": float(target_weight),
                "current_weight": float(current_weight),
                "delta_weight": float(delta_weight),
                "trade_confidence": 0.74 if direction == "long" else 0.70,
                "confidence_score": 0.74 if direction == "long" else 0.70,
                "continuation_probability": 0.66 if direction == "long" else 0.60,
                "setup_score": 0.72,
                "risk_reward_ratio": to_float(sig.get("risk_reward_ratio"), 2.0),
                "stop_price": to_float(sig.get("stop_price")),
                "target_price": to_float(sig.get("target_price")),
                "orb_8am_high": to_float(sig.get("orb", {}).get("high")),
                "orb_8am_low": to_float(sig.get("orb", {}).get("low")),
                "orb_8am_mid": to_float(sig.get("orb", {}).get("mid")),
                "orb_8am_range_pct": to_float(sig.get("orb", {}).get("range_pct")),
                "swept_level": to_float(sig.get("swept_level")),
                "sweep_wick": to_float(sig.get("sweep_wick")),
                "fvg_low": to_float(sig.get("fvg", {}).get("fvg_low")),
                "fvg_high": to_float(sig.get("fvg", {}).get("fvg_high")),
                "fvg_mid": to_float(sig.get("fvg", {}).get("mid")),
                "fvg_kind": str(sig.get("fvg", {}).get("kind", "")),
                "entry_notes": json.dumps({"orb": sig.get("orb", {}), "fvg": sig.get("fvg", {})}, default=json_default),
                "stream_confirmed": True,
                "orb_8am_signal": True,
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            print("[orb-8am] generated orders:")
            show = ["symbol", "side", "entry_type", "price", "stop_price", "target_price", "risk_reward_ratio", "reason"]
            print(out[[c for c in show if c in out.columns]].to_string(index=False))
            write_event("orb_8am_orders_generated", {"rows": out.to_dict(orient="records")})
        else:
            print("[orb-8am] no completed sweep/rejection/FVG-retest setups")
        return out

    def _intraday_direction_metrics(self, symbol: str, frame: pd.DataFrame, minutes: int, avg_entry_price: float) -> dict[str, Any]:
        if frame.empty or len(frame) < 5:
            return {
                f"{minutes}m_direction_score": 0.5,
                f"{minutes}m_ret": 0.0,
                f"{minutes}m_vwap": 0.0,
                f"{minutes}m_above_vwap": True,
                f"{minutes}m_lower_highs": False,
                f"{minutes}m_lower_lows": False,
                f"{minutes}m_below_fast_sma": False,
                f"{minutes}m_below_slow_sma": False,
                f"{minutes}m_failed_reclaim": False,
            }

        g = frame.sort_values("timestamp").copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float).replace(0, np.nan)

        last_close = float(close.iloc[-1])
        ret_n = last_close / max(float(close.iloc[0]), 1e-9) - 1.0
        ret_recent = last_close / max(float(close.iloc[-min(len(close), 6)]), 1e-9) - 1.0
        sma_fast = float(close.rolling(min(5, len(close))).mean().iloc[-1])
        sma_slow = float(close.rolling(min(15, len(close))).mean().iloc[-1])
        vwap = float((close * volume).sum() / max(volume.sum(), 1e-9)) if volume.notna().any() else last_close
        above_vwap = last_close >= vwap
        above_entry = last_close >= avg_entry_price if avg_entry_price > 0 else True
        below_fast_sma = last_close < sma_fast
        below_slow_sma = last_close < sma_slow

        recent_highs = high.tail(min(10, len(high))).reset_index(drop=True)
        recent_lows = low.tail(min(10, len(low))).reset_index(drop=True)
        lower_highs = bool(len(recent_highs) >= 4 and recent_highs.iloc[-1] < recent_highs.iloc[-3])
        lower_lows = bool(len(recent_lows) >= 4 and recent_lows.iloc[-1] < recent_lows.iloc[-3])
        failed_reclaim = bool(len(close) >= 4 and close.iloc[-1] < close.iloc[-2] and high.iloc[-1] <= high.iloc[-2])

        direction_score = (
            0.22 * np.clip((ret_n + 0.015) / 0.055, 0, 1)
            + 0.20 * np.clip((ret_recent + 0.006) / 0.030, 0, 1)
            + 0.18 * float(sma_fast >= sma_slow)
            + 0.14 * float(above_vwap)
            + 0.10 * float(above_entry)
            + 0.08 * float(not lower_lows)
            + 0.08 * float(not below_slow_sma)
            - 0.16 * float(lower_highs and lower_lows)
            - 0.10 * float(below_fast_sma and below_slow_sma)
            - 0.08 * float(failed_reclaim)
        )
        direction_score = float(np.clip(direction_score, 0, 1))

        return {
            f"{minutes}m_direction_score": direction_score,
            f"{minutes}m_ret": float(ret_n),
            f"{minutes}m_recent_ret": float(ret_recent),
            f"{minutes}m_vwap": vwap,
            f"{minutes}m_above_vwap": bool(above_vwap),
            f"{minutes}m_above_entry": bool(above_entry),
            f"{minutes}m_sma_fast": float(sma_fast),
            f"{minutes}m_sma_slow": float(sma_slow),
            f"{minutes}m_lower_highs": bool(lower_highs),
            f"{minutes}m_lower_lows": bool(lower_lows),
            f"{minutes}m_below_fast_sma": bool(below_fast_sma),
            f"{minutes}m_below_slow_sma": bool(below_slow_sma),
            f"{minutes}m_failed_reclaim": bool(failed_reclaim),
            f"{minutes}m_last_close": last_close,
        }

    def _build_next_bar_ml_dataset(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return pd.DataFrame()

        rows: list[pd.DataFrame] = []
        for symbol, g in bars.groupby("symbol", sort=False):
            g = g.sort_values("timestamp").copy()
            c = g["close"].astype(float)
            h = g["high"].astype(float)
            l = g["low"].astype(float)
            v = g["volume"].astype(float)

            g["ret_1"] = c.pct_change(1, fill_method=None)
            g["ret_2"] = c.pct_change(2, fill_method=None)
            g["ret_3"] = c.pct_change(3, fill_method=None)
            g["ret_6"] = c.pct_change(6, fill_method=None)
            g["range_pct"] = (h - l) / c.replace(0, np.nan)
            g["close_location"] = (c - l) / (h - l).replace(0, np.nan)
            g["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
            g["sma5_dist"] = c / c.rolling(5).mean().replace(0, np.nan) - 1.0
            g["sma15_dist"] = c / c.rolling(15).mean().replace(0, np.nan) - 1.0
            g["sma30_dist"] = c / c.rolling(30).mean().replace(0, np.nan) - 1.0
            g["rsi14"] = _rsi(c, 14)
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            g["macd"] = ema12 - ema26
            g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
            g["macd_hist"] = g["macd"] - g["macd_signal"]
            g["next_bar_up"] = (c.shift(-1) > c).astype(int)
            rows.append(g)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def predict_next_bar_direction(self, symbols: list[str]) -> pd.DataFrame:
        """Train a local intraday model and predict next-bar up probability per symbol."""
        symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
        if not symbols or not self.config.ml_next_bar_enabled:
            return pd.DataFrame({"symbol": symbols, "ml_next_bar_probability": 0.50, "ml_next_bar_confirmed": True})

        old_lookback = self.config.intraday_lookback_days
        self.config.intraday_lookback_days = max(old_lookback, self.config.ml_next_bar_lookback_days)
        try:
            bars = self._download_intraday_bars(symbols, self.config.ml_next_bar_timeframe_minutes)
        finally:
            self.config.intraday_lookback_days = old_lookback

        dataset = self._build_next_bar_ml_dataset(bars)
        if dataset.empty:
            return pd.DataFrame({"symbol": symbols, "ml_next_bar_probability": 0.50, "ml_next_bar_confirmed": False, "ml_next_bar_reason": "no_intraday_training_data"})

        feature_cols = [
            "ret_1", "ret_2", "ret_3", "ret_6", "range_pct", "close_location",
            "vol_ratio", "sma5_dist", "sma15_dist", "sma30_dist", "rsi14",
            "macd", "macd_signal", "macd_hist",
        ]

        train = dataset.dropna(subset=["next_bar_up"]).copy()
        train = train.iloc[:-len(symbols)] if len(train) > len(symbols) else train
        train = train.dropna(how="all", subset=feature_cols)

        if len(train) < self.config.ml_next_bar_min_train_rows or train["next_bar_up"].nunique() < 2:
            latest = dataset.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
            latest["ml_next_bar_probability"] = 0.50
            latest["ml_next_bar_confirmed"] = False
            latest["ml_next_bar_reason"] = "insufficient_training_rows"
            return latest[["symbol", "ml_next_bar_probability", "ml_next_bar_confirmed", "ml_next_bar_reason"]]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, random_state=42)),
            ]
        )
        model.fit(train[feature_cols], train["next_bar_up"].astype(int))

        latest = dataset.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
        probs = model.predict_proba(latest[feature_cols])[:, 1]
        latest["ml_next_bar_probability"] = probs
        latest["ml_next_bar_confirmed"] = latest["ml_next_bar_probability"] >= self.config.ml_next_bar_min_buy_probability
        latest["ml_next_bar_reason"] = np.where(latest["ml_next_bar_confirmed"], "next_bar_up_confirmed", "next_bar_probability_too_low")

        out = latest[["symbol", "ml_next_bar_probability", "ml_next_bar_confirmed", "ml_next_bar_reason"]].copy()
        print("[ml-next-bar] predictions:")
        print(out.to_string(index=False))
        return out

    def confirm_buy_direction(self, symbols: list[str]) -> pd.DataFrame:
        """Final directional confirmation before buy execution.

        Uses fresh 1m/5m/15m bars. A buy must show positive/constructive
        direction across timeframes, remain above VWAP when required, avoid
        lower-low deterioration, and avoid sharp immediate pullbacks.
        """
        symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
        if not symbols or not self.config.require_direction_confirmation_before_buy:
            return pd.DataFrame({"symbol": symbols, "direction_confirmed": True})

        bars_1m = self._download_intraday_bars(symbols, 1)
        bars_5m = self._download_intraday_bars(symbols, 5)
        bars_15m = self._download_intraday_bars(symbols, 15)

        ml_df = self.predict_next_bar_direction(symbols)
        ml_by_symbol = {safe_symbol(r["symbol"]): r for _, r in ml_df.iterrows()} if not ml_df.empty else {}

        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            frames = {
                1: bars_1m[bars_1m["symbol"] == symbol].copy() if not bars_1m.empty else pd.DataFrame(),
                5: bars_5m[bars_5m["symbol"] == symbol].copy() if not bars_5m.empty else pd.DataFrame(),
                15: bars_15m[bars_15m["symbol"] == symbol].copy() if not bars_15m.empty else pd.DataFrame(),
            }

            metrics: dict[str, Any] = {"symbol": symbol}
            for minutes, frame in frames.items():
                metrics.update(self._intraday_direction_metrics(symbol, frame, minutes, 0.0))

            combined = float(np.clip(
                0.35 * to_float(metrics.get("1m_direction_score"), 0.5)
                + 0.40 * to_float(metrics.get("5m_direction_score"), 0.5)
                + 0.25 * to_float(metrics.get("15m_direction_score"), 0.5),
                0,
                1,
            ))

            last_1m = to_float(metrics.get("1m_last_close"), 0.0)
            ret_1m = to_float(metrics.get("1m_recent_ret"), 0.0)
            above_vwap_ok = True
            lower_low_ok = True

            if self.config.require_price_above_vwap_for_buy:
                above_vwap_ok = bool(metrics.get("1m_above_vwap", True)) and bool(metrics.get("5m_above_vwap", True))

            if self.config.require_no_lower_low_for_buy:
                lower_low_ok = not bool(metrics.get("1m_lower_lows", False)) and not bool(metrics.get("5m_lower_lows", False))

            pullback_ok = ret_1m >= self.config.max_pretrade_pullback_pct

            ml_info = ml_by_symbol.get(symbol, {})
            ml_prob = to_float(ml_info.get("ml_next_bar_probability"), 0.50)
            ml_confirmed = bool(ml_info.get("ml_next_bar_confirmed", True))

            confirmed = (
                combined >= self.config.min_pretrade_direction_score
                and to_float(metrics.get("1m_direction_score"), 0.5) >= self.config.min_pretrade_1m_score
                and to_float(metrics.get("5m_direction_score"), 0.5) >= self.config.min_pretrade_5m_score
                and to_float(metrics.get("15m_direction_score"), 0.5) >= self.config.min_pretrade_15m_score
                and above_vwap_ok
                and lower_low_ok
                and pullback_ok
                and last_1m > 0
                and ml_confirmed
            )

            reason = "confirmed"
            if not confirmed:
                failed = []
                if combined < self.config.min_pretrade_direction_score:
                    failed.append("combined_direction_weak")
                if not above_vwap_ok:
                    failed.append("below_vwap")
                if not lower_low_ok:
                    failed.append("lower_low_structure")
                if not pullback_ok:
                    failed.append("active_pullback")
                if last_1m <= 0:
                    failed.append("missing_intraday_price")
                if not ml_confirmed:
                    failed.append(f"ml_next_bar_prob_low:{ml_prob:.2f}")
                reason = ",".join(failed) or "direction_not_confirmed"

            rows.append({
                **metrics,
                "direction_score": combined,
                "ml_next_bar_probability": ml_prob,
                "ml_next_bar_confirmed": ml_confirmed,
                "direction_confirmed": bool(confirmed),
                "direction_reject_reason": reason,
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            show = ["symbol", "direction_confirmed", "direction_score", "ml_next_bar_probability", "1m_direction_score", "5m_direction_score", "15m_direction_score", "direction_reject_reason"]
            print("[direction] pretrade confirmation:")
            print(out[[c for c in show if c in out.columns]].to_string(index=False))
        return out

    def monitor_active_positions(self, equity: float) -> pd.DataFrame:
        """Generate protective sell orders when active holdings lose intraday confirmation."""
        if not self.config.active_position_monitor_enabled:
            return pd.DataFrame()

        positions = self.broker.get_all_open_positions()
        if not positions:
            return pd.DataFrame()

        symbols = [normalize_symbol(p.symbol) for p in positions if p.symbol]
        print(f"[monitor] checking active positions with 1m/5m/15m bars symbols={symbols}")

        bars_1m = self._download_intraday_bars(symbols, 1)
        bars_5m = self._download_intraday_bars(symbols, 5)
        bars_15m = self._download_intraday_bars(symbols, 15)

        monitor_rows: list[dict[str, Any]] = []
        exit_rows: list[dict[str, Any]] = []

        for pos in positions:
            symbol = normalize_symbol(pos.symbol)
            avg_entry = float(pos.avg_entry_price or 0.0)
            current_price = float(pos.current_price or 0.0)
            unrealized_plpc = float(pos.unrealized_plpc or 0.0)

            frames = {
                1: bars_1m[bars_1m["symbol"] == symbol].copy() if not bars_1m.empty else pd.DataFrame(),
                5: bars_5m[bars_5m["symbol"] == symbol].copy() if not bars_5m.empty else pd.DataFrame(),
                15: bars_15m[bars_15m["symbol"] == symbol].copy() if not bars_15m.empty else pd.DataFrame(),
            }

            metrics: dict[str, Any] = {
                "timestamp_utc": now_utc().isoformat(),
                "symbol": symbol,
                "qty": pos.qty,
                "current_price": current_price,
                "avg_entry_price": avg_entry,
                "unrealized_plpc": unrealized_plpc,
                "market_value": pos.market_value,
            }

            for minutes, frame in frames.items():
                metrics.update(self._intraday_direction_metrics(symbol, frame, minutes, avg_entry))

            direction_score = float(np.clip(
                0.40 * to_float(metrics.get("1m_direction_score"), 0.5)
                + 0.35 * to_float(metrics.get("5m_direction_score"), 0.5)
                + 0.25 * to_float(metrics.get("15m_direction_score"), 0.5),
                0,
                1,
            ))
            metrics["hold_direction_score"] = direction_score

            lost_vwap = (
                self.config.monitor_exit_on_vwap_loss
                and not bool(metrics.get("1m_above_vwap", True))
                and not bool(metrics.get("5m_above_vwap", True))
            )
            structure_break = (
                self.config.monitor_exit_on_lower_high_lower_low
                and bool(metrics.get("5m_lower_highs", False))
                and bool(metrics.get("5m_lower_lows", False))
            )
            micro_structure_break = (
                bool(metrics.get("1m_lower_highs", False))
                and bool(metrics.get("1m_lower_lows", False))
                and bool(metrics.get("1m_below_slow_sma", False))
            )
            failed_reclaim = bool(metrics.get("1m_failed_reclaim", False)) and bool(metrics.get("5m_failed_reclaim", False))
            below_fast_and_slow = bool(metrics.get("1m_below_fast_sma", False)) and bool(metrics.get("5m_below_slow_sma", False))
            trailing_stop_hit = unrealized_plpc <= -abs(self.config.intraday_trailing_stop_pct)
            failed_breakout_loss = unrealized_plpc <= self.config.failed_breakout_loss_pct
            failed_breakout_price = avg_entry > 0 and current_price < avg_entry * (1.0 - self.config.failed_breakout_buffer_pct)

            exit_reason = ""
            if direction_score <= self.config.hard_exit_direction_score:
                exit_reason = "monitor_hard_direction_failure"
            elif self.config.monitor_exit_on_failed_breakout and failed_breakout_loss and failed_breakout_price:
                exit_reason = "monitor_failed_breakout_cut"
            elif trailing_stop_hit:
                exit_reason = "monitor_intraday_trailing_stop"
            elif lost_vwap and structure_break:
                exit_reason = "monitor_vwap_and_structure_break"
            elif direction_score < self.config.min_hold_direction_score and (lost_vwap or structure_break or micro_structure_break):
                exit_reason = "monitor_direction_breakdown"
            elif direction_score < self.config.min_hold_direction_score and failed_reclaim and below_fast_and_slow:
                exit_reason = "monitor_failed_reclaim"

            metrics["lost_vwap"] = bool(lost_vwap)
            metrics["structure_break"] = bool(structure_break)
            metrics["micro_structure_break"] = bool(micro_structure_break)
            metrics["failed_reclaim"] = bool(failed_reclaim)
            metrics["below_fast_and_slow"] = bool(below_fast_and_slow)
            metrics["failed_breakout_loss"] = bool(failed_breakout_loss)
            metrics["failed_breakout_price"] = bool(failed_breakout_price)
            metrics["exit_reason"] = exit_reason
            monitor_rows.append(metrics)

            if not exit_reason and self.config.scale_out_losers_enabled:
                if unrealized_plpc <= self.config.loser_full_exit_loss_pct and direction_score <= self.config.min_hold_direction_score:
                    exit_reason = "loser_full_exit_loss_and_weak_direction"
                elif unrealized_plpc <= self.config.scale_out_loss_pct and direction_score <= self.config.scale_out_direction_score:
                    exit_reason = "loser_scale_out_weak_direction"

            if exit_reason:
                if exit_reason.startswith("loser_scale_out"):
                    sell_qty = max(1, int(float(pos.qty) * self.config.scale_out_fraction))
                    target_weight = max(0.0, (float(pos.market_value) - sell_qty * current_price) / max(equity, 1e-9))
                else:
                    sell_qty = int(pos.qty)
                    target_weight = 0.0

                exit_rows.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "delta_shares": sell_qty,
                        "delta_dollars": float(sell_qty * current_price),
                        "price": current_price,
                        "reason": exit_reason,
                        "target_weight": target_weight,
                        "current_weight": float(pos.market_value) / max(equity, 1e-9),
                        "delta_weight": -float(sell_qty * current_price) / max(equity, 1e-9),
                        "trade_confidence": 1.0,
                        "hold_direction_score": direction_score,
                    }
                )

        monitor = pd.DataFrame(monitor_rows)
        if not monitor.empty:
            monitor.to_csv(POSITION_MONITOR_PATH, index=False)
            show = [
                "symbol", "unrealized_plpc", "hold_direction_score",
                "1m_direction_score", "5m_direction_score", "15m_direction_score",
                "lost_vwap", "structure_break", "micro_structure_break",
                "failed_reclaim", "failed_breakout_loss", "exit_reason",
            ]
            print("[monitor] active position direction check:")
            print(monitor[[c for c in show if c in monitor.columns]].to_string(index=False))

        return pd.DataFrame(exit_rows)

    def _score_latest(self, features: pd.DataFrame) -> pd.DataFrame:
        latest = features.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
        if latest.empty:
            return latest

        latest = latest[
            (latest["close"] >= self.config.min_price)
            & (latest["close"] <= self.config.max_price)
            & (latest["dollar_volume20"] >= self.config.min_avg_dollar_volume)
        ].copy()
        if latest.empty:
            return latest

        def clip01(s: pd.Series, lo: float, hi: float) -> pd.Series:
            return ((s - lo) / max(hi - lo, 1e-9)).clip(0, 1)

        # Market regime proxy from SPY/QQQ if present, otherwise neutral.
        benchmark = latest[latest["symbol"].isin(["SPY", "QQQ"])].copy()
        if benchmark.empty:
            regime_score = 0.55
            regime_label = "NEUTRAL_UNKNOWN"
        else:
            bench_momentum = (
                0.35 * clip01(benchmark["ret_5d"], -0.03, 0.06)
                + 0.25 * clip01(benchmark["ret_10d"], -0.04, 0.09)
                + 0.20 * benchmark["above_sma20"].fillna(0)
                + 0.20 * benchmark["trend_stack"].fillna(0)
            ).mean()
            regime_score = float(np.clip(bench_momentum, 0, 1))
            if regime_score >= 0.62:
                regime_label = "BULLISH"
            elif regime_score <= 0.38:
                regime_label = "BEARISH"
            else:
                regime_label = "NEUTRAL"

        latest["regime_score"] = regime_score
        latest["regime_label"] = regime_label

        trend_score = (
            0.24 * latest["trend_stack"].fillna(0)
            + 0.16 * latest["above_sma20"].fillna(0)
            + 0.12 * latest["above_prev_high"].fillna(0)
            + 0.12 * latest["sma5_cross_sma20"].fillna(0)
            + 0.10 * latest["sma20_cross_sma50"].fillna(0)
            + 0.08 * latest["sma50_cross_sma200"].fillna(0)
            + 0.10 * clip01(latest["bb_pos"], 0.45, 0.95)
            + 0.08 * latest["bb_breakout"].fillna(0)
        ).clip(0, 1)

        momentum_score = (
            0.18 * clip01(latest["ret_1d"], -0.01, 0.05)
            + 0.22 * clip01(latest["ret_5d"], -0.02, 0.10)
            + 0.18 * clip01(latest["ret_10d"], -0.03, 0.18)
            + 0.12 * clip01(latest["ret_20d"], -0.05, 0.25)
            + 0.12 * clip01(latest["rsi14"], 45, 72)
            + 0.10 * latest["macd_bullish"].fillna(0)
            + 0.08 * clip01(latest["macd_hist_slope"], -0.02, 0.05)
        ).clip(0, 1)

        liquidity_score = (
            0.55 * clip01(latest["dollar_volume20"], self.config.min_avg_dollar_volume, self.config.min_avg_dollar_volume * 20)
            + 0.45 * clip01(latest["rel_volume"], 0.8, 3.0)
        ).clip(0, 1)

        volatility_score = (
            0.50 * clip01(latest["atr_pct"], 0.01, 0.08)
            + 0.25 * (1 - clip01(latest["rolling_std10"], 0.015, 0.10))
            + 0.25 * (1 - clip01(latest["rolling_std20"], 0.015, 0.10))
        ).clip(0, 1)

        risk_penalty = (
            0.35 * latest["overbought_rsi"].fillna(0)
            + 0.25 * latest["bb_rejection"].fillna(0)
            + 0.25 * latest["weak_macd"].fillna(0)
            + 0.15 * (1 - latest["positive_daily_move"].fillna(0))
        ).clip(0, 1)

        reversal_score = (
            0.24 * (1 - clip01(latest["rsi14"], 30, 55))
            + 0.22 * latest["lower_band_reclaim"].fillna(0)
            + 0.18 * clip01(latest["ret_1d"], -0.01, 0.05)
            + 0.16 * clip01(latest["rel_volume"], 0.8, 2.8)
            + 0.10 * (1 - clip01(latest["bb_pos"], 0.15, 0.55))
            + 0.10 * (1 - clip01(latest["drawdown_20h"], -0.18, -0.03))
        ).clip(0, 1)

        # Continuation probability proxy. If a real ML model is later plugged in,
        # replace this column with model.predict_proba outputs.
        ml_probability = (
            0.30 * trend_score
            + 0.25 * momentum_score
            + 0.15 * liquidity_score
            + 0.10 * volatility_score
            + 0.15 * regime_score
            + 0.05 * latest["positive_daily_move"].fillna(0)
            - 0.22 * risk_penalty
        ).clip(0.01, 0.99)

        confidence_score = (
            0.35 * ml_probability
            + 0.20 * trend_score
            + 0.20 * momentum_score
            + 0.10 * liquidity_score
            + 0.10 * regime_score
            + 0.05 * volatility_score
            - 0.20 * risk_penalty
        ).clip(0, 1)

        latest["trend_score"] = trend_score
        latest["momentum_score"] = momentum_score
        latest["liquidity_score"] = liquidity_score
        latest["volatility_score"] = volatility_score
        latest["risk_penalty"] = risk_penalty
        latest["reversal_score"] = reversal_score
        latest["continuation_probability"] = ml_probability
        latest["ml_probability"] = ml_probability
        latest["confidence_score"] = confidence_score
        latest["setup_type"] = np.where(latest["momentum_score"] >= latest["reversal_score"], "momentum", "reversal")
        latest["setup_score"] = latest[["confidence_score", "momentum_score", "reversal_score"]].max(axis=1)
        latest["trade_confidence"] = latest["confidence_score"].clip(0, 1)

        # New-opportunity scanner layer.
        # This is intentionally rule-based so it can run without a trained model:
        # +P&L for the day, positive momentum, or confirmed breakout/reclaim.
        confirmed_breakout_score = (
            0.28 * latest["above_prev_high"].fillna(0)
            + 0.18 * latest["bb_breakout"].fillna(0)
            + 0.16 * latest["sma5_cross_sma20"].fillna(0)
            + 0.14 * latest["macd_bullish"].fillna(0)
            + 0.12 * clip01(latest["rel_volume"], 1.0, 3.0)
            + 0.12 * clip01(latest["ret_1d"], 0.0025, 0.06)
        ).clip(0, 1)

        positive_momentum_score = (
            0.24 * clip01(latest["ret_1d"], 0.0, 0.05)
            + 0.22 * clip01(latest["ret_5d"], 0.0, 0.12)
            + 0.16 * clip01(latest["ret_10d"], 0.0, 0.18)
            + 0.14 * latest["above_sma20"].fillna(0)
            + 0.12 * latest["macd_bullish"].fillna(0)
            + 0.12 * clip01(latest["rsi14"], 50, 72)
        ).clip(0, 1)

        opportunity_score = (
            0.35 * positive_momentum_score
            + 0.30 * confirmed_breakout_score
            + 0.15 * liquidity_score
            + 0.10 * regime_score
            + 0.10 * latest["positive_daily_move"].fillna(0)
            - 0.20 * risk_penalty
        ).clip(0, 1)

        latest["confirmed_breakout_score"] = confirmed_breakout_score
        latest["positive_momentum_score"] = positive_momentum_score
        latest["opportunity_score"] = opportunity_score
        latest["is_positive_day_opportunity"] = (
            (latest["ret_1d"] >= self.config.min_daily_gain_pct)
            & (
                (positive_momentum_score >= self.config.min_opportunity_score)
                | (confirmed_breakout_score >= self.config.min_breakout_confirmation_score)
            )
            & (risk_penalty <= 0.60)
            & (latest["dollar_volume20"] >= self.config.min_avg_dollar_volume)
        )

        return latest.sort_values(
            ["is_positive_day_opportunity", "opportunity_score", "trade_confidence", "setup_score", "continuation_probability", "dollar_volume20"],
            ascending=[False, False, False, False, False, False],
        ).reset_index(drop=True)

    def _update_high_water_and_exits(self, scores: pd.DataFrame, equity: float) -> pd.DataFrame:
        state = self._load_state()
        high_plpc: dict[str, float] = {k: to_float(v) for k, v in state.get("high_plpc", {}).items()}
        score_by_symbol = dict(zip(scores.get("symbol", []), scores.get("setup_score", [])))
        top_symbols = set(scores.head(self.config.top_n).get("symbol", []))

        rows: list[dict[str, Any]] = []
        for pos in self.broker.get_all_open_positions():
            symbol = normalize_symbol(pos.symbol)
            if pos.qty <= 0:
                continue

            old_high = high_plpc.get(symbol, pos.unrealized_plpc)
            new_high = max(old_high, pos.unrealized_plpc)
            high_plpc[symbol] = new_high

            reason = ""
            if pos.unrealized_plpc <= self.config.stop_loss_pct:
                reason = "scanner_stop_loss"
            elif new_high >= self.config.profit_lock_trigger_pct and pos.unrealized_plpc <= new_high - self.config.profit_giveback_pct:
                reason = "scanner_profit_giveback"
            elif symbol not in top_symbols and to_float(score_by_symbol.get(symbol), 0.0) < self.config.stale_position_score_exit:
                reason = "scanner_stale_position"

            if reason:
                rows.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "delta_shares": int(pos.qty),
                        "delta_dollars": float(pos.market_value),
                        "price": float(pos.current_price),
                        "reason": reason,
                        "target_weight": 0.0,
                        "current_weight": float(pos.market_value) / max(equity, 1e-9),
                        "delta_weight": -float(pos.market_value) / max(equity, 1e-9),
                        "trade_confidence": 1.0,
                    }
                )
                high_plpc.pop(symbol, None)

        state["high_plpc"] = high_plpc
        state["updated_at"] = now_utc().isoformat()
        self._save_state(state)
        return pd.DataFrame(rows)

    def build_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        symbols = self._universe()
        print(f"[scanner] scanning universe symbols={len(symbols)}")

        bars = self._download_bars(symbols)
        if bars.empty:
            print("[scanner] no bars downloaded")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        features = self._build_features(bars)
        scores = self._score_latest(features)
        if not scores.empty and self.llm_metadata_by_symbol:
            meta = pd.DataFrame(
                [{"symbol": sym, **data} for sym, data in self.llm_metadata_by_symbol.items()]
            )
            if not meta.empty:
                scores = scores.merge(meta, on="symbol", how="left")
                if "llm_confidence" in scores.columns:
                    scores["llm_confidence"] = pd.to_numeric(scores["llm_confidence"], errors="coerce").fillna(0.50)
                    scores["trade_confidence"] = (0.70 * scores["trade_confidence"] + 0.30 * scores["llm_confidence"]).clip(0, 1)
                    scores["confidence_score"] = (0.70 * scores["confidence_score"] + 0.30 * scores["llm_confidence"]).clip(0, 1)
                    scores["setup_score"] = scores[["setup_score", "trade_confidence", "confidence_score"]].max(axis=1)
                    scores = scores.sort_values(
                        ["trade_confidence", "setup_score", "continuation_probability", "dollar_volume20"],
                        ascending=[False, False, False, False],
                    ).reset_index(drop=True)
        if scores.empty:
            print("[scanner] no symbols passed scanner filters; trying ORB-8AM fallback scanner")
            acct = self.broker.get_account_snapshot()
            equity = max(float(acct.equity), 1.0)
            positions = {normalize_symbol(p.symbol): p for p in self.broker.get_all_open_positions()}
            orb_rebalance = self._build_orb_8am_orders(symbols, positions, equity)
            exits = self._update_high_water_and_exits(pd.DataFrame(), equity)
            orb_rebalance.to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
            exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
            return orb_rebalance, exits, pd.DataFrame()

        scores.to_csv(SCANNER_SCORES_PATH, index=False)
        print("[scanner] top setups:")
        show_cols = [
            "symbol", "close", "llm_confidence", "llm_catalyst_type", "regime_label", "setup_type", "is_positive_day_opportunity",
            "opportunity_score", "positive_momentum_score", "confirmed_breakout_score",
            "trade_confidence", "continuation_probability", "setup_score", "trend_score",
            "momentum_score", "liquidity_score", "risk_penalty", "rsi14", "ret_1d",
            "ret_5d", "rel_volume",
        ]
        print(scores[[c for c in show_cols if c in scores.columns]].head(self.config.top_n).to_string(index=False))

        acct = self.broker.get_account_snapshot()
        equity = max(float(acct.equity), 1.0)
        positions = {normalize_symbol(p.symbol): p for p in self.broker.get_all_open_positions()}

        strict_mask = (
            (scores["setup_score"] >= self.config.min_setup_score)
            & (scores["trade_confidence"] >= self.config.min_buy_confidence)
            & (scores["dollar_volume20"] >= self.config.min_avg_dollar_volume)
            & (scores["risk_penalty"] <= 0.55)
            & (scores["rsi14"] <= self.config.max_overbought_rsi)
        )

        if self.config.opportunity_scan_enabled:
            opportunity_mask = (
                scores["is_positive_day_opportunity"].fillna(False).astype(bool)
                & (scores["ret_1d"] >= self.config.min_daily_gain_pct)
                & (scores["opportunity_score"] >= self.config.min_opportunity_score)
            )
            if self.config.require_positive_momentum_or_breakout:
                opportunity_mask &= (
                    (scores["positive_momentum_score"] >= self.config.min_opportunity_score)
                    | (scores["confirmed_breakout_score"] >= self.config.min_breakout_confirmation_score)
                )
            strict_mask |= opportunity_mask

        if self.config.require_positive_daily_move:
            # Breakout/momentum setups require a positive daily move.
            # Reversal setups may pass with a negative daily move if they show
            # reclaim/recovery evidence and the gate is enabled.
            if self.config.allow_negative_daily_move_reversals and "setup_type" in scores.columns:
                is_reversal = scores["setup_type"].astype(str).str.lower().isin({"reversal", "reclaim", "gap_fill"})
                reclaim_col = None
                for _rc in ["reversal_score", "reclaim_score", "bullish_structure_score"]:
                    if _rc in scores.columns:
                        reclaim_col = _rc
                        break
                if reclaim_col:
                    strong_reclaim = pd.to_numeric(scores[reclaim_col], errors="coerce").fillna(0.0) >= self.config.reversal_min_reclaim_score
                else:
                    strong_reclaim = pd.Series(False, index=scores.index)
                # Gate condition: non-breakout or verified reversal with evidence
                vwap_ok = pd.Series(True, index=scores.index)
                if "dist_vwap" in scores.columns:
                    vwap_ok = pd.to_numeric(scores["dist_vwap"], errors="coerce").fillna(-1.0) >= -0.005
                elif "above_vwap" in scores.columns:
                    vwap_ok = pd.to_numeric(scores["above_vwap"], errors="coerce").fillna(0.0) > 0
                macd_ok = pd.Series(True, index=scores.index)
                if "macd_hist" in scores.columns:
                    macd_ok = pd.to_numeric(scores["macd_hist"], errors="coerce").fillna(-999.0) > -0.01
                reversal_exception = (
                    is_reversal
                    & strong_reclaim
                    & vwap_ok
                    & macd_ok
                    & (scores.get("regime_label", pd.Series("", index=scores.index)).astype(str) != "BEARISH")
                )
                # Flag reversals allowed despite negative day
                if reversal_exception.any() and not scores["positive_daily_move"].fillna(0).astype(bool).all():
                    neg_rev = scores[reversal_exception & ~scores["positive_daily_move"].fillna(0).astype(bool)]
                    if not neg_rev.empty:
                        print(f"[scanner] reversal_allowed_despite_negative_daily_move: {sorted(neg_rev['symbol'].map(safe_symbol).tolist())}")
                daily_move_ok = scores["positive_daily_move"].fillna(0).astype(bool) | reversal_exception
                strict_mask &= daily_move_ok
            else:
                strict_mask &= scores["positive_daily_move"].fillna(0).astype(bool)

        if self.config.avoid_bearish_regime_for_longs:
            strict_mask &= scores["regime_score"] >= self.config.min_regime_score_for_longs
            strict_mask &= scores["regime_label"] != "BEARISH"

        candidates = scores[strict_mask].head(self.config.top_n).copy()

        _scanner_dropped = scores[~strict_mask]
        if not _scanner_dropped.empty:
            _drop_parts = []
            for _, _dr in _scanner_dropped.sort_values("trade_confidence", ascending=False).head(8).iterrows():
                _why = []
                if to_float(_dr.get("setup_score")) < self.config.min_setup_score:
                    _why.append(f"setup={to_float(_dr.get('setup_score')):.2f}<{self.config.min_setup_score:.2f}")
                if to_float(_dr.get("trade_confidence")) < self.config.min_buy_confidence:
                    _why.append(f"conf={to_float(_dr.get('trade_confidence')):.2f}<{self.config.min_buy_confidence:.2f}")
                if to_float(_dr.get("dollar_volume20")) < self.config.min_avg_dollar_volume:
                    _why.append(f"dv20={to_float(_dr.get('dollar_volume20')):.0f}<{self.config.min_avg_dollar_volume:.0f}")
                if to_float(_dr.get("risk_penalty")) > 0.55:
                    _why.append(f"risk_penalty={to_float(_dr.get('risk_penalty')):.2f}")
                if to_float(_dr.get("rsi14")) > self.config.max_overbought_rsi:
                    _why.append(f"rsi14={to_float(_dr.get('rsi14')):.1f}>{self.config.max_overbought_rsi:.1f}")
                if self.config.avoid_bearish_regime_for_longs and (
                    to_float(_dr.get("regime_score")) < self.config.min_regime_score_for_longs
                    or str(_dr.get("regime_label", "")) == "BEARISH"
                ):
                    _why.append(f"regime={_dr.get('regime_label', '?')}:{to_float(_dr.get('regime_score')):.2f}<{self.config.min_regime_score_for_longs:.2f}")
                _drop_parts.append(f"{_dr.get('symbol', '?')}:[{','.join(_why or ['other'])}]")
            print(f"[scanner] strict filter: dropped {len(_scanner_dropped)} | top-confidence rejects: {' | '.join(_drop_parts)}")

        if candidates.empty and self.config.allow_relaxed_fallback:
            print("[scanner] strict filters returned no candidates; applying relaxed fallback filters")
            relaxed_mask = (
                (scores["setup_score"] >= self.config.fallback_min_setup_score)
                & (scores["trade_confidence"] >= self.config.fallback_min_buy_confidence)
                & (scores["dollar_volume20"] >= self.config.min_avg_dollar_volume)
                & (scores["risk_penalty"] <= 0.70)
                & (scores["rsi14"] <= min(85.0, self.config.max_overbought_rsi + 7.0))
                & (scores["regime_score"] >= 0.30)
            )
            candidates = scores[relaxed_mask].head(self.config.top_n).copy()

        candidates = self._annotate_intraday_entries(candidates)

        if not candidates.empty and self.config.ml_next_bar_enabled:
            ml_preds = self.predict_next_bar_direction(candidates["symbol"].tolist())
            if not ml_preds.empty:
                candidates = candidates.merge(ml_preds, on="symbol", how="left")
                candidates["ml_next_bar_probability"] = pd.to_numeric(candidates["ml_next_bar_probability"], errors="coerce").fillna(0.50)
                candidates["trade_confidence"] = (0.75 * candidates["trade_confidence"] + 0.25 * candidates["ml_next_bar_probability"]).clip(0, 1)
                candidates = candidates[candidates["ml_next_bar_probability"] >= self.config.ml_next_bar_min_buy_probability].copy()
                if candidates.empty:
                    print("[ml-next-bar] all candidates rejected by next-bar direction filter")

        no_reentry_symbols = load_no_reentry_symbols()
        if no_reentry_symbols:
            before = len(candidates)
            candidates = candidates[~candidates["symbol"].map(normalize_symbol).isin(no_reentry_symbols)].copy()
            skipped = before - len(candidates)
            if skipped:
                print(f"[scanner] skipped {skipped} same-day no-reentry candidate(s): {sorted(no_reentry_symbols)}")

        rows: list[dict[str, Any]] = []
        for _, r in candidates.iterrows():
            symbol = normalize_symbol(r["symbol"])
            price = to_float(r.get("entry_price"), to_float(r["close"]))
            market_price = to_float(r["close"])
            if price <= 0:
                print(f"[scanner] skip BUY {symbol}: no usable price (entry_price={r.get('entry_price')} close={r.get('close')})")
                continue

            current_mv = positions.get(symbol).market_value if symbol in positions else 0.0
            current_weight = float(current_mv) / equity
            type_mult = self.config.reversal_target_weight_mult if r["setup_type"] == "reversal" else 1.0

            # Composite quality score weights setup quality, entry confidence, and
            # liquidity so higher-conviction, more-liquid setups receive larger allocations.
            _wt_setup = self.config.position_setup_score_weight
            _wt_conf = self.config.position_confidence_weight
            _wt_liq = self.config.position_liquidity_weight
            composite_quality = (
                _wt_setup * float(r["setup_score"])
                + _wt_conf * float(r["trade_confidence"])
                + _wt_liq * min(float(r["liquidity_score"]), 1.0)
            )
            target_weight = min(
                self.config.max_target_weight,
                self.config.base_target_weight * (0.40 + 2.0 * composite_quality) * type_mult,
                self.executor_config.max_position_weight,
            )
            print(
                f"[sizing] {symbol}: setup={float(r['setup_score']):.2f} "
                f"conf={float(r['trade_confidence']):.2f} "
                f"liq={float(r['liquidity_score']):.2f} "
                f"composite={composite_quality:.3f} → target_weight={target_weight:.3%}"
            )
            delta_weight = target_weight - current_weight

            if delta_weight < self.executor_config.min_weight_delta:
                print(f"[scanner] skip BUY {symbol}: delta_weight={delta_weight:.4f} below min={self.executor_config.min_weight_delta:.4f} (current_weight={current_weight:.2%})")
                continue

            delta_dollars = delta_weight * equity
            if delta_dollars < self.executor_config.min_notional_to_trade:
                print(f"[scanner] skip BUY {symbol}: delta_dollars=${delta_dollars:.2f} below min_notional=${self.executor_config.min_notional_to_trade:.2f}")
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "delta_shares": int(delta_dollars // price),
                    "delta_dollars": float(delta_dollars),
                    "price": price,
                    "market_price": market_price,
                    "entry_price": to_float(r.get("entry_price"), price),
                    "entry_type": str(r.get("entry_type", "")),
                    "nearest_support": to_float(r.get("nearest_support")),
                    "nearest_resistance": to_float(r.get("nearest_resistance")),
                    "stop_price": to_float(r.get("stop_price")),
                    "target_price": to_float(r.get("target_price")),
                    "risk_reward_ratio": to_float(r.get("risk_reward_ratio")),
                    "entry_notes": str(r.get("entry_notes", "")),
                    "support_levels": json.dumps(r.get("support_levels", []), default=json_default),
                    "resistance_levels": json.dumps(r.get("resistance_levels", []), default=json_default),
                    "reason": f"scanner_{r['setup_type']}_candidate",
                    "target_weight": target_weight,
                    "current_weight": current_weight,
                    "delta_weight": delta_weight,
                    "trade_confidence": float(r["trade_confidence"]),
                    "confidence_score": float(r["confidence_score"]),
                    "ml_next_bar_probability": to_float(r.get("ml_next_bar_probability"), 0.50),
                    "ml_next_bar_confirmed": bool(r.get("ml_next_bar_confirmed", False)),
                    "llm_confidence": to_float(r.get("llm_confidence"), 0.0),
                    "llm_rank": to_int(r.get("llm_rank"), 999),
                    "llm_catalyst_type": str(r.get("llm_catalyst_type", "")),
                    "llm_reason": str(r.get("llm_reason", "")),
                    "llm_market_tone": str(r.get("llm_market_tone", "")),
                    "llm_summary": str(r.get("llm_summary", "")),
                    "continuation_probability": float(r["continuation_probability"]),
                    "ml_probability": float(r["ml_probability"]),
                    "setup_score": float(r["setup_score"]),
                    "trend_score": float(r["trend_score"]),
                    "momentum_score": float(r["momentum_score"]),
                    "liquidity_score": float(r["liquidity_score"]),
                    "volatility_score": float(r["volatility_score"]),
                    "opportunity_score": float(r["opportunity_score"]),
                    "positive_momentum_score": float(r["positive_momentum_score"]),
                    "confirmed_breakout_score": float(r["confirmed_breakout_score"]),
                    "is_positive_day_opportunity": bool(r["is_positive_day_opportunity"]),
                    "risk_penalty": float(r["risk_penalty"]),
                    "regime_score": float(r["regime_score"]),
                    "regime_label": str(r["regime_label"]),
                    "reversal_score": float(r["reversal_score"]),
                    "rsi14": to_float(r.get("rsi14")),
                    "ret_1d": to_float(r.get("ret_1d")),
                    "ret_5d": to_float(r.get("ret_5d")),
                    "ret_10d": to_float(r.get("ret_10d")),
                    "rel_volume": to_float(r.get("rel_volume")),
                    "macd": to_float(r.get("macd")),
                    "macd_signal": to_float(r.get("macd_signal")),
                    "macd_hist": to_float(r.get("macd_hist")),
                    "rolling_std10": to_float(r.get("rolling_std10")),
                    "rolling_std20": to_float(r.get("rolling_std20")),
                }
            )

        rebalance = pd.DataFrame(rows)

        # Merge support-pivot buy orders.
        # This is the preferred intraday path: buy strong support zones after
        # a pullback/tag/sweep and confirmed 1m/5m reversal. It avoids the old
        # rare sweep -> FVG -> retest dependency that left the bot sidelined.
        pivot_rebalance = self._build_pivot_point_orders(symbols, positions, equity)
        if not pivot_rebalance.empty:
            if rebalance.empty:
                rebalance = pivot_rebalance.copy()
            else:
                out = rebalance.copy()
                for _, _pivot_row in pivot_rebalance.iterrows():
                    _sym = normalize_symbol(_pivot_row.get("symbol"))
                    same = out["symbol"].map(normalize_symbol) == _sym if "symbol" in out.columns else pd.Series(False, index=out.index)
                    if not same.any():
                        out = pd.concat([out, pd.DataFrame([_pivot_row])], ignore_index=True)
                    else:
                        # If the existing scanner row is lower confidence, replace it with the cleaner pivot setup.
                        existing_conf = float(pd.to_numeric(out.loc[same, "trade_confidence"], errors="coerce").fillna(0.0).max()) if "trade_confidence" in out.columns else 0.0
                        pivot_conf = to_float(_pivot_row.get("trade_confidence"), 0.0)
                        if pivot_conf > existing_conf:
                            out = out[~same].copy()
                            out = pd.concat([out, pd.DataFrame([_pivot_row])], ignore_index=True)
                rebalance = out.reset_index(drop=True)

        # Optional legacy 8am ORB/liquidity-sweep/FVG scanner. Disabled by default.
        orb_rebalance = self._build_orb_8am_orders(symbols, positions, equity)
        if not orb_rebalance.empty:
            if rebalance.empty:
                rebalance = orb_rebalance.copy()
            else:
                out = rebalance.copy()
                for _, _orb_row in orb_rebalance.iterrows():
                    _sym = normalize_symbol(_orb_row.get("symbol"))
                    _side = str(_orb_row.get("side", "")).lower()
                    same = out["symbol"].map(normalize_symbol) == _sym if "symbol" in out.columns else pd.Series(False, index=out.index)
                    if _side == "sell":
                        out = out[~same].copy()
                        out = pd.concat([out, pd.DataFrame([_orb_row])], ignore_index=True)
                    elif not same.any():
                        out = pd.concat([out, pd.DataFrame([_orb_row])], ignore_index=True)
                rebalance = out.reset_index(drop=True)

        # ------------------------------------------------------------------
        # Execution-alpha gate:
        #   1. Rank candidates cross-sectionally vs today's universe.
        #   2. Confirm each symbol's own time-series direction.
        #   3. Block weak BUY orders before pyramids/options.
        #   4. Always leave exits untouched.
        # ------------------------------------------------------------------
        rebalance = self._add_execution_alpha_gate(rebalance, scores)

        rebalance = self._add_pyramid_orders(rebalance, scores, equity)

        # Run the gate again after pyramid orders, because pyramids are also BUYs.
        rebalance = self._add_execution_alpha_gate(rebalance, scores)

        self._trigger_options_booster_if_needed(rebalance)
        exits = self._update_high_water_and_exits(scores, equity)

        rebalance.to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
        exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
        print(f"[scanner] wrote rebalance_orders={DEFAULT_REBALANCE_ORDERS} rows={len(rebalance)}")
        print(f"[scanner] wrote exit_orders={DEFAULT_EXIT_ORDERS} rows={len(exits)}")
        return rebalance, exits, scores

    def _rank_pct(self, series: pd.Series) -> pd.Series:
        """Safe percentile rank used for cross-sectional execution ranking."""
        try:
            s = pd.to_numeric(series, errors="coerce").fillna(0.0)
            if s.nunique(dropna=True) <= 1:
                return pd.Series(0.5, index=series.index)
            return s.rank(pct=True).fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=series.index)

    def _add_execution_alpha_gate(self, rebalance: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Gate BUY orders using:
            - cross-sectional rank: is this symbol stronger than today's universe?
            - time-series score: is its own direction confirmed?
            - final execution_alpha_score

        This materially reduces premature entries and forced same-day exits.
        """
        if rebalance is None or rebalance.empty:
            return rebalance

        if "side" not in rebalance.columns or "symbol" not in rebalance.columns:
            return rebalance

        df = rebalance.copy()
        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["side"] = df["side"].astype(str).str.lower().str.strip()
        orb_mask = df.get("orb_8am_signal", pd.Series(False, index=df.index)).fillna(False).astype(bool) if "orb_8am_signal" in df.columns else pd.Series(False, index=df.index)

        if scores is None or scores.empty or "symbol" not in scores.columns:
            df["cross_sectional_score"] = 0.50
            df["time_series_score"] = 0.50
            df["execution_alpha_score"] = np.where(orb_mask, 0.72, 0.50)
            df["execution_alpha_pass"] = (df["side"] != "buy") | orb_mask
            blocked = df[(df["side"] == "buy") & (~df["execution_alpha_pass"])]
            if not blocked.empty:
                print(f"[execution-alpha] blocked {len(blocked)} BUY order(s): missing score context")
            return df[df["execution_alpha_pass"]].reset_index(drop=True)

        score_df = scores.copy()
        score_df["symbol"] = score_df["symbol"].map(normalize_symbol)

        for col in [
            "ret_1d",
            "ret_3d",
            "ret_5d",
            "momentum_score",
            "positive_momentum_score",
            "breakout_score",
            "setup_score",
            "trade_confidence",
            "confidence_score",
            "continuation_probability",
            "ml_next_bar_probability",
            "close",
            "entry_price",
            "vwap",
            "rsi14",
            "rsi_14",
        ]:
            if col not in score_df.columns:
                score_df[col] = 0.0

        # Cross-sectional score: relative strength vs the rest of today's universe.
        score_df["cs_return_rank"] = (
            0.25 * self._rank_pct(score_df["ret_1d"])
            + 0.25 * self._rank_pct(score_df["ret_3d"])
            + 0.20 * self._rank_pct(score_df["ret_5d"])
            + 0.30 * self._rank_pct(
                score_df["positive_momentum_score"].where(
                    pd.to_numeric(score_df["positive_momentum_score"], errors="coerce").fillna(0.0) > 0,
                    score_df["momentum_score"],
                )
            )
        )

        score_df["cs_quality_rank"] = (
            0.30 * self._rank_pct(score_df["breakout_score"])
            + 0.25 * self._rank_pct(score_df["trade_confidence"])
            + 0.25 * self._rank_pct(score_df["setup_score"])
            + 0.20 * self._rank_pct(score_df["continuation_probability"])
        )

        score_df["cross_sectional_score"] = (
            0.55 * score_df["cs_return_rank"]
            + 0.45 * score_df["cs_quality_rank"]
        ).clip(0.0, 1.0)

        def calc_time_series_score(row: pd.Series) -> float:
            close = to_float(row.get("close"), to_float(row.get("entry_price"), 0.0))
            entry = to_float(row.get("entry_price"), close)
            vwap = to_float(row.get("vwap"), close)
            rsi = to_float(row.get("rsi14"), to_float(row.get("rsi_14"), 50.0))

            ret_1d = to_float(row.get("ret_1d"), 0.0)
            ret_3d = to_float(row.get("ret_3d"), 0.0)
            ret_5d = to_float(row.get("ret_5d"), 0.0)

            momentum = to_float(
                row.get("positive_momentum_score"),
                to_float(row.get("momentum_score"), 0.0),
            )
            breakout = to_float(row.get("breakout_score"), 0.0)
            confidence = to_float(row.get("trade_confidence"), 0.0)
            continuation = to_float(row.get("continuation_probability"), confidence)
            ml_prob = to_float(row.get("ml_next_bar_probability"), 0.50)

            score = 0.0

            if close > 0 and vwap > 0 and close >= vwap:
                score += 0.16
            if close > 0 and entry > 0 and close >= entry * 0.998:
                score += 0.08
            if ret_1d > 0:
                score += 0.10
            if ret_3d > 0:
                score += 0.10
            if ret_5d > 0:
                score += 0.08
            if momentum >= 0.55:
                score += 0.15
            if breakout >= 0.50:
                score += 0.12
            if confidence >= self.config.min_buy_confidence:
                score += 0.10
            if continuation >= self.config.min_buy_confidence:
                score += 0.07
            if ml_prob >= self.config.ml_next_bar_min_buy_probability:
                score += 0.14
            if 50 <= rsi <= self.config.max_overbought_rsi:
                score += 0.10

            return float(np.clip(score, 0.0, 1.0))

        score_df["time_series_score"] = score_df.apply(calc_time_series_score, axis=1)

        # Blend based on market regime. Trend markets favor time-series continuation.
        regime_bias = ""
        try:
            market_context = read_json_file(MARKET_CONTEXT_PATH, default={})
            regime_bias = str(
                market_context.get("overall_market_bias")
                or market_context.get("market_bias")
                or market_context.get("regime")
                or ""
            ).lower()
        except Exception:
            market_context = {}

        trend_on = regime_bias in {
            "bullish",
            "risk_on",
            "trend_risk_on",
            "uptrend",
            "constructive",
        }
        risk_off = regime_bias in {
            "bearish",
            "risk_off",
            "defensive",
            "downtrend",
        }

        if trend_on:
            cs_weight = 0.40
            ts_weight = 0.60
        else:
            cs_weight = 0.55
            ts_weight = 0.45

        score_df["execution_alpha_score"] = (
            cs_weight * score_df["cross_sectional_score"]
            + ts_weight * score_df["time_series_score"]
        ).clip(0.0, 1.0)

        # risk_off: raise pass threshold slightly but do NOT haircut scores — a 0.70x
        # multiplier on scores with a 0.70 threshold made pass impossible (only perfect
        # scores could pass). Size is reduced later via risk_off_size_mult on accepted orders.
        risk_off_size_mult = 1.0
        if risk_off:
            risk_off_size_mult = float(getattr(self.config, "execution_alpha_risk_off_size_mult", 0.50))
            threshold_add = float(getattr(self.config, "execution_alpha_risk_off_threshold_add", 0.05))
            print(f"[alpha-gate] risk_off mode: size multiplier={risk_off_size_mult:.0%} applied after gate, score not haircutted")
        else:
            threshold_add = 0.0

        threshold = float(getattr(self.config, "execution_alpha_min_score", 0.70)) + threshold_add

        keep_cols = [
            "symbol",
            "cross_sectional_score",
            "time_series_score",
            "execution_alpha_score",
        ]

        df = df.merge(
            score_df[keep_cols].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )

        df["cross_sectional_score"] = pd.to_numeric(
            df["cross_sectional_score"], errors="coerce"
        ).fillna(0.50)
        df["time_series_score"] = pd.to_numeric(
            df["time_series_score"], errors="coerce"
        ).fillna(0.50)
        df["execution_alpha_score"] = pd.to_numeric(
            df["execution_alpha_score"], errors="coerce"
        ).fillna(0.50)

        # Sells/exits are always allowed. Only weak BUYs are blocked.
        df["execution_alpha_pass"] = (
            df["side"] != "buy"
        ) | (
            df["execution_alpha_score"] >= threshold
        )

        blocked = df[(df["side"] == "buy") & (~df["execution_alpha_pass"])].copy()
        if not blocked.empty:
            print(
                f"[execution-alpha] blocked {len(blocked)} weak BUY order(s) "
                f"threshold={threshold:.2f}"
            )
            show = [
                "symbol",
                "reason",
                "execution_alpha_score",
                "cross_sectional_score",
                "time_series_score",
                "trade_confidence",
                "ml_next_bar_probability",
                "delta_dollars",
            ]
            print(blocked[[c for c in show if c in blocked.columns]].to_string(index=False))

            for _, row in blocked.iterrows():
                write_event(
                    "execution_alpha_buy_blocked",
                    {
                        "symbol": row.get("symbol"),
                        "reason": row.get("reason"),
                        "execution_alpha_score": to_float(row.get("execution_alpha_score")),
                        "cross_sectional_score": to_float(row.get("cross_sectional_score")),
                        "time_series_score": to_float(row.get("time_series_score")),
                        "threshold": threshold,
                    },
                )

        passed = df[df["execution_alpha_pass"]].copy().reset_index(drop=True)

        # Apply risk_off size reduction to buy rows only — sells are always unaffected.
        if risk_off and risk_off_size_mult < 1.0 and "side" in passed.columns:
            ro_buy_mask = passed["side"] == "buy"
            if ro_buy_mask.any():
                for col in ["capped_dollars", "abs_delta_dollars", "delta_dollars"]:
                    if col in passed.columns:
                        passed.loc[ro_buy_mask, col] = pd.to_numeric(passed.loc[ro_buy_mask, col], errors="coerce").fillna(0.0) * risk_off_size_mult
                passed.loc[ro_buy_mask, "reason"] = passed.loc[ro_buy_mask, "reason"].astype(str) + f"|risk_off_size:{risk_off_size_mult:.0%}"

        buy_count = int((passed["side"] == "buy").sum()) if "side" in passed.columns else 0
        print(
            f"[execution-alpha] passed rows={len(passed)} buy_rows={buy_count} "
            f"blocked={len(blocked)} threshold={threshold:.2f}"
        )

        return passed

    def _add_pyramid_orders(self, rebalance: pd.DataFrame, scores: pd.DataFrame, equity: float) -> pd.DataFrame:
        """Add incremental buy orders for existing winners that are still confirmed."""
        if not self.config.pyramid_winners_enabled:
            return rebalance

        positions = self.broker.get_all_open_positions()
        if not positions or scores.empty:
            return rebalance

        score_by_symbol = scores.set_index("symbol").to_dict("index") if "symbol" in scores.columns else {}
        rows: list[dict[str, Any]] = []

        for pos in positions:
            symbol = normalize_symbol(pos.symbol)
            if not symbol or symbol not in score_by_symbol:
                continue
            if symbol in load_no_reentry_symbols():
                continue
            if pyramid_add_count(symbol) >= self.config.pyramid_max_adds_per_symbol_per_day:
                continue

            s = score_by_symbol[symbol]
            confidence = to_float(s.get("trade_confidence"), 0.0)
            direction_score = to_float(s.get("positive_momentum_score"), to_float(s.get("momentum_score"), 0.0))
            current_price = to_float(s.get("entry_price"), to_float(s.get("close"), pos.current_price))
            current_weight = float(pos.market_value) / max(equity, 1e-9)

            ml_prob = to_float(s.get("ml_next_bar_probability"), 0.50)

            if (
                pos.unrealized_plpc >= self.config.pyramid_min_unrealized_plpc
                and direction_score >= self.config.pyramid_min_hold_direction_score
                and ml_prob >= self.config.ml_next_bar_min_pyramid_probability
                and confidence >= self.config.min_buy_confidence
                and current_weight < min(self.config.max_target_weight, build_account_risk_profile(equity).max_position_weight)
            ):
                add_weight = min(
                    self.config.pyramid_add_weight,
                    min(self.config.max_target_weight, build_account_risk_profile(equity).max_position_weight) - current_weight,
                )
                delta_dollars = add_weight * equity
                if delta_dollars >= build_account_risk_profile(equity).min_notional_to_trade and current_price > 0:
                    rows.append({
                        "symbol": symbol,
                        "side": "buy",
                        "delta_shares": int(delta_dollars // current_price),
                        "delta_dollars": float(delta_dollars),
                        "price": current_price,
                        "market_price": to_float(s.get("close"), current_price),
                        "entry_price": current_price,
                        "reason": "pyramid_winner_confirmed_direction",
                        "target_weight": current_weight + add_weight,
                        "current_weight": current_weight,
                        "delta_weight": add_weight,
                        "trade_confidence": confidence,
                        "confidence_score": to_float(s.get("confidence_score"), confidence),
                        "continuation_probability": to_float(s.get("continuation_probability"), confidence),
                        "momentum_score": to_float(s.get("momentum_score"), direction_score),
                        "positive_momentum_score": direction_score,
                        "ml_next_bar_probability": ml_prob,
                        "unrealized_plpc": float(pos.unrealized_plpc),
                    })
                    mark_pyramid_add(symbol, reason="pyramid_winner_confirmed_direction")

        if rows:
            pyramid_df = pd.DataFrame(rows)
            print(f"[pyramid] adding {len(pyramid_df)} winner scale-in order(s)")
            show = ["symbol", "unrealized_plpc", "delta_dollars", "target_weight", "trade_confidence", "reason"]
            print(pyramid_df[[c for c in show if c in pyramid_df.columns]].to_string(index=False))
            return pd.concat([rebalance, pyramid_df], ignore_index=True)

        return rebalance
    
# ======================================================================# Executor
# ======================================================================
class ProductionOrderExecutor:
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.journal = CsvJournal(JOURNAL_PATH)
        self.broker = AlpacaBroker(ENV_PATH, paper=config.paper)
        self.scanner = MomentumSetupScanner(self.broker, self.config.scanner, self.config) if self.config.scanner.enabled else None
        self.last_order_time = 0.0
        self.last_heartbeat_time = 0.0
        self.account_history: list[dict[str, Any]] = []
        self._last_morning_refresh_date: str | None = None
        self._rl_validation_cache: dict[str, Any] | None = None
        self._planner_health_cache: dict[str, Any] | None = None
        self._sa_assessment: dict[str, Any] = {}
        self._sa_last_assessment_date: str | None = None

    def _planner_health(self, refresh: bool = False) -> dict[str, Any]:
        if refresh or self._planner_health_cache is None:
            self._planner_health_cache = planner_health_snapshot(self.config.scanner)
            print(
                "[planner-health] "
                f"healthy={self._planner_health_cache.get('healthy')} "
                f"mode={self._planner_health_cache.get('execution_mode')} "
                f"plan_age={self._planner_health_cache.get('plan_age_minutes')} "
                f"structure_age={self._planner_health_cache.get('structure_age_minutes')} "
                f"long_allowed={len(self._planner_health_cache.get('long_allowed_symbols', []))} "
                f"issues={self._planner_health_cache.get('issues', [])}"
            )
            write_event("planner_health_snapshot", {
                k: v for k, v in self._planner_health_cache.items()
                if k not in {"plan_payload", "structure_payload", "structure_by_symbol"}
            })
        return self._planner_health_cache

    def _current_market_context(self) -> dict[str, Any]:
        if not self.config.scanner.market_context_enabled:
            return {}
        return read_json_file(MARKET_CONTEXT_PATH, {})

    def _apply_planner_and_market_gates_to_plan(self, plan: pd.DataFrame) -> pd.DataFrame:
        if plan is None or plan.empty or self.config.scanner is None:
            return plan

        out = plan.copy()
        cfg = self.config.scanner
        health = self._planner_health(refresh=True) if cfg.premarket_plan_enabled else {"healthy": True, "execution_mode": "NORMAL", "long_allowed_symbols": []}
        market_context = self._current_market_context()

        market_bias = str(market_context.get("overall_market_bias", "")).lower()
        trade_permission = str(market_context.get("trade_permission", "")).lower()
        risk_signal_bias = str((market_context.get("key_etf_context_summary", {}) or {}).get("risk_signal_bias", "")).lower()

        buy_mask = out["side"].astype(str).str.lower() == "buy"
        _sell_count_before = int((out["side"].astype(str).str.lower() == "sell").sum())
        if not buy_mask.any():
            return out

        # Fresh planner / structure map gate. Exits are never blocked here.
        if cfg.premarket_plan_enabled:
            long_allowed = set(health.get("long_allowed_symbols", []) or [])
            valid_symbols = set(health.get("valid_symbols", []) or [])

            if cfg.require_fresh_premarket_plan_for_buys and not bool(health.get("healthy")):
                if health.get("execution_mode") == "BLOCK_NEW_BUYS":
                    blocked = int(buy_mask.sum())
                    print(f"[planner-gate] stale/missing plan → BLOCK_NEW_BUYS: suppressed {blocked} BUY(s) | {health.get('issues')} | exits unaffected: {_sell_count_before} SELL(s) remain open")
                    write_event("planner_gate_blocked_buys", {"count": blocked, "issues": health.get("issues", []), "sell_rows_preserved": _sell_count_before})
                    out = out.loc[~buy_mask].copy()
                    if out.empty:
                        return out
                    buy_mask = out["side"].astype(str).str.lower() == "buy"
                else:
                    _safe_syms = sorted(out.loc[buy_mask, "symbol"].map(safe_symbol).unique().tolist())
                    print(f"[planner-gate] stale/missing plan → SAFE mode: reducing {len(_safe_syms)} BUY(s) to {cfg.safe_mode_buy_size_multiplier:.0%} of size | symbols: {_safe_syms} | issues: {health.get('issues', [])} | exits unaffected: {_sell_count_before} SELL(s) remain open")
                    out.loc[buy_mask, "capped_dollars"] = pd.to_numeric(out.loc[buy_mask, "capped_dollars"], errors="coerce").fillna(0.0) * cfg.safe_mode_buy_size_multiplier
                    out.loc[buy_mask, "abs_delta_dollars"] = pd.to_numeric(out.loc[buy_mask, "abs_delta_dollars"], errors="coerce").fillna(0.0) * cfg.safe_mode_buy_size_multiplier
                    out.loc[buy_mask, "reason"] = out.loc[buy_mask, "reason"].astype(str) + f"|planner_safe_mode:{','.join(health.get('issues', []))}"

            if cfg.require_structure_long_allowed_for_buys and long_allowed:
                symbol_mask = out["symbol"].map(safe_symbol).isin(long_allowed)
                block_mask = buy_mask & ~symbol_mask
                if block_mask.any():
                    print(f"[structure-gate] blocked BUY symbols not long_allowed: {sorted(out.loc[block_mask, 'symbol'].map(safe_symbol).unique().tolist())}")
                    write_event("structure_gate_blocked_buys", {"symbols": sorted(out.loc[block_mask, "symbol"].map(safe_symbol).unique().tolist()), "long_allowed": sorted(long_allowed)})
                    out = out.loc[~block_mask].copy()
                    if out.empty:
                        return out
                    buy_mask = out["side"].astype(str).str.lower() == "buy"

            if cfg.require_valid_premarket_plan_for_buys and valid_symbols:
                valid_mask = out["symbol"].map(safe_symbol).isin(valid_symbols)
                block_mask = buy_mask & ~valid_mask
                if block_mask.any():
                    print(f"[premarket-valid-gate] blocked BUY symbols without valid setup: {sorted(out.loc[block_mask, 'symbol'].map(safe_symbol).unique().tolist())}")
                    out = out.loc[~block_mask].copy()
                    if out.empty:
                        return out
                    buy_mask = out["side"].astype(str).str.lower() == "buy"

        # Cautious / risk-off market degradation. Sells still pass through.
        if trade_permission == "cautious" or risk_signal_bias == "risk_off":
            out.loc[buy_mask, "capped_dollars"] = pd.to_numeric(out.loc[buy_mask, "capped_dollars"], errors="coerce").fillna(0.0) * cfg.cautious_market_buy_size_multiplier
            out.loc[buy_mask, "abs_delta_dollars"] = pd.to_numeric(out.loc[buy_mask, "abs_delta_dollars"], errors="coerce").fillna(0.0) * cfg.cautious_market_buy_size_multiplier
            out.loc[buy_mask, "confidence_used"] = pd.to_numeric(out.loc[buy_mask, "confidence_used"], errors="coerce").fillna(0.0) - cfg.cautious_market_confidence_addon
            out.loc[buy_mask, "reason"] = out.loc[buy_mask, "reason"].astype(str) + f"|market_cautious:{trade_permission or risk_signal_bias}"
            print(f"[market-gate] cautious/risk-off: reduced {int(buy_mask.sum())} BUY(s) to {cfg.cautious_market_buy_size_multiplier:.0%} of size | market={market_bias} permission={trade_permission} risk_signal={risk_signal_bias}")

        # Mixed market: lighter haircut than cautious. Skip when cautious/risk-off already applied.
        elif market_bias == "mixed" and trade_permission not in {"cautious", "halt_new_buys", "disabled"} and risk_signal_bias != "risk_off" and cfg.mixed_market_buy_size_multiplier < 1.0:
            out.loc[buy_mask, "capped_dollars"] = pd.to_numeric(out.loc[buy_mask, "capped_dollars"], errors="coerce").fillna(0.0) * cfg.mixed_market_buy_size_multiplier
            out.loc[buy_mask, "abs_delta_dollars"] = pd.to_numeric(out.loc[buy_mask, "abs_delta_dollars"], errors="coerce").fillna(0.0) * cfg.mixed_market_buy_size_multiplier
            out.loc[buy_mask, "reason"] = out.loc[buy_mask, "reason"].astype(str) + f"|market_mixed_haircut:{cfg.mixed_market_buy_size_multiplier:.0%}"
            print(f"[market-gate] mixed market: reduced {int(buy_mask.sum())} BUY(s) to {cfg.mixed_market_buy_size_multiplier:.0%} of size")

        if cfg.bearish_market_disable_new_longs and market_bias == "bearish":
            rl_col = None
            for candidate in ["rl_confidence", "rl_score", "enter_long_confidence", "action_confidence"]:
                if candidate in out.columns:
                    rl_col = candidate
                    break
            if rl_col:
                rl_ok = pd.to_numeric(out[rl_col], errors="coerce").fillna(0.0) >= cfg.bearish_market_min_rl_confidence
            else:
                rl_ok = pd.Series(False, index=out.index)
            block_mask = buy_mask & ~rl_ok
            if block_mask.any():
                print(f"[market-gate] bearish market blocked BUY symbols: {sorted(out.loc[block_mask, 'symbol'].map(safe_symbol).unique().tolist())}")
                write_event("bearish_market_gate_blocked_buys", {"symbols": sorted(out.loc[block_mask, "symbol"].map(safe_symbol).unique().tolist()), "min_rl_confidence": cfg.bearish_market_min_rl_confidence})
                out = out.loc[~block_mask].copy()
                if out.empty:
                    return out
                buy_mask = out["side"].astype(str).str.lower() == "buy"

        # Degrade buy sizing when account health, drawdown, or data quality indicate elevated risk.
        if _SA_AVAILABLE and self._sa_assessment:
            sa = self._sa_assessment
            sa_mode = self_assessment_mode(sa)
            if sa_mode == "HALT_NEW_BUYS" and buy_mask.any():
                sa_reasons = sa.get("reasons", [])
                blocked = int(buy_mask.sum())
                print(f"[sa-gate] HALT_NEW_BUYS: suppressed {blocked} BUY(s) | reasons={sa_reasons} | exits unaffected: {_sell_count_before} SELL(s)")
                write_event("sa_gate_halt_new_buys", {"count": blocked, "reasons": sa_reasons, "sell_rows_preserved": _sell_count_before})
                out = out.loc[~buy_mask].copy()
                if out.empty:
                    return out
            elif sa_mode in ("SAFE", "CAUTIOUS") and buy_mask.any():
                sa_mult = self_assessment_size_multiplier(sa)
                sa_conf_raise = self_assessment_raises_confidence(sa)
                sa_reasons = sa.get("reasons", [])
                out.loc[buy_mask, "capped_dollars"] = pd.to_numeric(out.loc[buy_mask, "capped_dollars"], errors="coerce").fillna(0.0) * sa_mult
                out.loc[buy_mask, "abs_delta_dollars"] = pd.to_numeric(out.loc[buy_mask, "abs_delta_dollars"], errors="coerce").fillna(0.0) * sa_mult
                if sa_conf_raise > 0 and "confidence_used" in out.columns:
                    out.loc[buy_mask, "confidence_used"] = pd.to_numeric(out.loc[buy_mask, "confidence_used"], errors="coerce").fillna(0.0) - sa_conf_raise
                out.loc[buy_mask, "reason"] = out.loc[buy_mask, "reason"].astype(str) + f"|sa_{sa_mode.lower()}:mult={sa_mult:.2f}"
                print(f"[sa-gate] {sa_mode}: reduced {int(buy_mask.sum())} BUY(s) to {sa_mult:.0%} | conf_raise={sa_conf_raise:.2f} | reasons={sa_reasons}")
                write_event("sa_gate_size_reduced", {"mode": sa_mode, "count": int(buy_mask.sum()), "multiplier": sa_mult, "conf_raise": sa_conf_raise, "reasons": sa_reasons, "sell_rows_preserved": _sell_count_before})

        # Recompute quantities after any sizing degradation.
        if not out.empty and "price" in out.columns and "capped_dollars" in out.columns:
            price = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
            needs_qty = buy_mask & (price > 0)
            out.loc[needs_qty, "order_qty"] = (pd.to_numeric(out.loc[needs_qty, "capped_dollars"], errors="coerce").fillna(0.0) / price.loc[needs_qty]).astype(int)
            out = out[(out["side"].astype(str).str.lower() == "sell") | (pd.to_numeric(out["order_qty"], errors="coerce").fillna(0).astype(int) > 0)].copy()

        # Invariant check: exit rows must always survive all gates.
        _sell_count_after = int((out["side"].astype(str).str.lower() == "sell").sum())
        if _sell_count_after != _sell_count_before:
            print(f"[planner-gate] BUG: sell row count changed {_sell_count_before}→{_sell_count_after}; exits should never be suppressed by market or planner gates")
        return out.reset_index(drop=True)

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_order_time) < self.config.cooldown_seconds

    def _mark_order(self) -> None:
        self.last_order_time = time.time()

    def _record_account_snapshot(self, label: str, cycle_id: str) -> dict[str, Any]:
        acct = self.broker.get_account_snapshot()
        snapshot = {
            "label": label,
            "cycle_id": cycle_id,
            "timestamp_utc": now_utc().isoformat(),
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
        }
        self.account_history.append(snapshot)
        return snapshot

    def _write_account_equity_curve(self) -> None:
        if self.account_history:
            pd.DataFrame(self.account_history).to_csv(EQUITY_CURVE_PATH, index=False)

    def _positions_by_symbol(self) -> dict[str, BrokerPosition]:
        return {safe_symbol(p.symbol): p for p in self.broker.get_all_open_positions() if p.symbol}

    def _log_live_experience(
        self,
        *,
        cycle_id: str,
        stage: str,
        order_row: pd.Series | dict[str, Any] | None = None,
        acct_before: Optional[dict[str, Any]] = None,
        acct_after: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        direction: Optional[dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        if not self.config.scanner.live_experience_logging_enabled:
            return

        if stage == "skipped" and not self.config.scanner.live_experience_log_skips:
            return
        if stage in {"submitted", "order", "filled"} and not self.config.scanner.live_experience_log_orders:
            return
        if stage in {"error", "api_error"} and not self.config.scanner.live_experience_log_errors:
            return

        symbol = safe_symbol(
            (order_row.to_dict() if isinstance(order_row, pd.Series) else dict(order_row or {})).get("symbol", "")
            or (result or {}).get("symbol", "")
        )
        position = self.broker.get_position(symbol) if symbol else BrokerPosition(symbol=symbol)
        row = build_live_experience_row(
            cycle_id=cycle_id,
            stage=stage,
            order_row=order_row,
            acct_before=acct_before,
            acct_after=acct_after,
            result=result,
            position=position,
            direction=direction,
            message=message,
        )
        append_live_experience(row, write_parquet=self.config.scanner.live_experience_write_parquet)

    def _gross_exposure(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        return sum(max(0.0, float(p.market_value)) for p in self.broker.get_all_open_positions()) / equity

    def _risk_profile(self, equity: float) -> AccountRiskProfile:
        return build_account_risk_profile(equity)

    def _cash_buffer(self, equity: float) -> float:
        profile = self._risk_profile(equity)
        return max(profile.min_cash_buffer, equity * profile.cash_buffer_pct)

    def _max_order_notional(self, equity: float) -> float:
        profile = self._risk_profile(equity)
        return min(profile.max_order_dollars, equity * profile.max_single_order_notional_pct)

    def _market_is_open_or_wait(self) -> bool:
        if self.config.allow_closed_market or not self.config.require_market_open:
            print("[clock] market-open guard disabled")
            return True

        info = self.broker.get_clock_info()
        print(
            "[clock] "
            f"is_open={info['is_open']} "
            f"timestamp={info['timestamp']} "
            f"next_open={info['next_open']} "
            f"next_close={info['next_close']}"
        )
        write_event("market_clock_checked", info)

        if info["is_open"]:
            return True

        if not self.config.wait_for_market_open:
            print("[clock] market is closed. Execution skipped")
            return False

        started = time.time()
        max_wait = self.config.max_wait_for_open_minutes * 60
        while time.time() - started < max_wait:
            time.sleep(max(5, self.config.poll_seconds))
            info = self.broker.get_clock_info()
            print(f"[clock] waiting | is_open={info['is_open']} next_open={info['next_open']}")
            write_event("market_clock_wait_poll", info)
            if info["is_open"]:
                return True

        print("[clock] market did not open within wait limit. Execution skipped")
        return False

    def _read_kill_state(self) -> dict[str, Any]:
        if not KILL_SWITCH_PATH.exists():
            return {}
        try:
            return json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_kill_state(self, state: dict[str, Any]) -> None:
        KILL_SWITCH_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")

    def _kill_switch_allows_execution(self, equity: float) -> tuple[bool, str, float]:
        if not self.config.enable_kill_switch:
            return True, "kill_switch_disabled", 0.0

        today = now_utc().date().isoformat()
        state = self._read_kill_state()

        if state.get("date") != today:
            state = {
                "date": today,
                "start_equity": equity,
                "kill_triggered": False,
                "triggered_at": "",
                "reason": "",
            }
            self._write_kill_state(state)

        start_equity = to_float(state.get("start_equity"), equity)
        profile = self._risk_profile(start_equity)

        # If an older run latched a kill switch while using stale settings,
        # clear it when the account is currently above the active dynamic floor.
        # Real daily drawdown/loss triggers remain latched for the trading day.
        if (
            state.get("kill_triggered")
            and str(state.get("reason", "")) == "kill_switch_equity_floor"
            and equity > profile.kill_switch_equity_floor
        ):
            state.update(
                {
                    "kill_triggered": False,
                    "triggered_at": "",
                    "reason": "",
                    "cleared_at": now_utc().isoformat(),
                    "clear_reason": "equity_above_dynamic_floor",
                }
            )
            self._write_kill_state(state)
            write_event("kill_switch_cleared", state)

        pnl = equity - start_equity
        dd_pct = pnl / max(start_equity, 1e-9)

        reason = "ok"
        blocked = False

        if state.get("kill_triggered"):
            blocked = True
            reason = str(state.get("reason", "kill_switch_already_triggered"))
        elif equity <= profile.kill_switch_equity_floor:
            blocked = True
            reason = "kill_switch_equity_floor"
        elif dd_pct <= profile.daily_drawdown_limit:
            blocked = True
            reason = "daily_drawdown_limit"
        elif pnl <= -abs(profile.max_daily_loss_dollars):
            blocked = True
            reason = "max_daily_loss_dollars"
        elif (-dd_pct) >= abs(profile.max_daily_loss_pct):
            blocked = True
            reason = "max_daily_loss_pct"

        print(
            f"[kill-switch] band={profile.band} start_equity=${start_equity:,.2f} equity=${equity:,.2f} "
            f"floor=${profile.kill_switch_equity_floor:,.2f} daily_pnl=${pnl:,.2f} "
            f"daily_return={dd_pct:.2%} status={reason}"
        )

        if blocked:
            state.update(
                {
                    "kill_triggered": True,
                    "triggered_at": now_utc().isoformat(),
                    "reason": reason,
                    "current_equity": equity,
                    "daily_pnl": pnl,
                    "daily_return": dd_pct,
                }
            )
            self._write_kill_state(state)
            write_event("kill_switch_triggered", state)
            return False, reason, dd_pct

        return True, reason, dd_pct

    def _safe_read_order_csv(self, path: Path, required: bool = False) -> pd.DataFrame:
        if not path.exists():
            if required:
                raise RuntimeError(f"Required order file not found: {path}. portfolio_rebalancer.py should generate this file before execution.")
            return pd.DataFrame()
        try:
            if path.stat().st_size == 0:
                return pd.DataFrame()
        except Exception:
            pass
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _run_portfolio_rebalancer(self) -> None:
        """Generate today's order CSVs through portfolio_rebalancer.py before file-based execution."""
        if not PORTFOLIO_REBALANCER_PATH.exists():
            raise RuntimeError(f"portfolio_rebalancer.py not found: {PORTFOLIO_REBALANCER_PATH}")

        print(f"[rebalancer] generating rebalance/exit orders via {PORTFOLIO_REBALANCER_PATH}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["RUN_DATA_DATE"] = DATA_TAG
        env["RL_DATA_DATE"] = DATA_TAG

        proc = subprocess.run(
            [sys.executable, str(PORTFOLIO_REBALANCER_PATH)],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"portfolio_rebalancer.py failed with code {proc.returncode}")

        if not self.config.rebalance_orders_path.exists():
            raise RuntimeError(
                f"Required order file not found after portfolio_rebalancer.py: "
                f"{self.config.rebalance_orders_path}"
            )

    def _load_orders(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.rebalance_orders_path.exists():
            self._run_portfolio_rebalancer()

        rebalance = self._safe_read_order_csv(self.config.rebalance_orders_path, required=True)
        exits = self._safe_read_order_csv(self.config.exit_orders_path, required=False)

        if self.config.scanner and self.config.scanner.stream_confirmed_merge_orders:
            rebalance = merge_stream_confirmed_orders(rebalance)
        if self.config.scanner and self.config.scanner.sr_rl_merge_signals:
            rebalance = merge_sr_rl_orders(rebalance, self.config.scanner)

        return rebalance, exits

    def _normalize_order_frame(self, orders: Optional[pd.DataFrame], default_reason: str = "") -> pd.DataFrame:
        cols = [
            "symbol", "side", "delta_shares", "delta_dollars", "price", "reason",
            "target_weight", "current_weight", "delta_weight", "trade_confidence",
            "confidence", "confidence_score", "continuation_prob", "continuation_probability",
            "explosive_alpha_probability",
        ]
        if orders is None or orders.empty:
            return pd.DataFrame(columns=cols)

        out = orders.copy()
        for col in cols:
            if col not in out.columns:
                out[col] = default_reason if col == "reason" else np.nan

        out["symbol"] = out["symbol"].map(safe_symbol)
        out["side"] = out["side"].astype(str).str.lower().str.strip()
        out["reason"] = out["reason"].fillna(default_reason).astype(str)

        numeric_cols = [c for c in cols if c not in {"symbol", "side", "reason"}]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        return out[(out["symbol"] != "") & out["side"].isin(["buy", "sell"])].reset_index(drop=True)

    def _confidence_for_row(self, row: pd.Series) -> float:
        for col in [
            "trade_confidence", "confidence", "confidence_score", "continuation_prob",
            "continuation_probability", "explosive_alpha_probability",
        ]:
            value = row.get(col, np.nan)
            if pd.notna(value):
                val = to_float(value, 0.5)
                if val > 1:
                    val /= 100.0
                return float(np.clip(val, 0.0, 1.0))
        return 0.5

    def _confidence_multiplier(self, confidence: float) -> float:
        return float(np.clip(0.50 + confidence, 0.35, 1.15))

    def _load_latest_feature_stream(self) -> pd.DataFrame:
        """Load latest 1m + 5m/15m/1h/4h features produced by build_intraday_training_dataset.py."""
        if not self.config.scanner.feature_stream_enabled:
            return pd.DataFrame()

        candidates = [FEATURE_STREAM_CSV_PATH]
        if FEATURE_STREAM_DIR.exists():
            candidates.extend(sorted(FEATURE_STREAM_DIR.glob("*_intraday_feature_stream.csv"), reverse=True))
        candidates.extend([INTRADAY_TRAINING_DATASET_PATH, EXIT_TRAINING_DATASET_PATH])

        for path in candidates:
            if not path or not Path(path).exists():
                continue
            try:
                age_min = (time.time() - Path(path).stat().st_mtime) / 60.0
                if path == FEATURE_STREAM_CSV_PATH or "feature_stream" in str(path):
                    max_age = self.config.scanner.feature_stream_max_age_minutes
                    if age_min > max_age:
                        print(f"[rl-gate] stale feature stream age={age_min:.1f}m max={max_age}m; trying fallback")
                        continue
                df = pd.read_csv(path)
                if df.empty or "symbol" not in df.columns:
                    continue
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                    df = df.sort_values(["symbol", "timestamp"])
                df["symbol"] = df["symbol"].map(safe_symbol)
                return df
            except Exception as exc:
                print(f"[feature-stream] failed loading {path}: {exc}")
        return pd.DataFrame()

    def _latest_feature_by_symbol(self) -> dict[str, dict[str, Any]]:
        df = self._load_latest_feature_stream()
        if df.empty:
            return {}
        latest: dict[str, dict[str, Any]] = {}
        for symbol, g in df.groupby("symbol"):
            if not symbol:
                continue
            row = g.sort_values("timestamp").iloc[-1].to_dict() if "timestamp" in g.columns else g.iloc[-1].to_dict()
            latest[safe_symbol(symbol)] = row
        return latest

    def _row_float(self, row: dict[str, Any], key: str, default: float = 0.0) -> float:
        return to_float(row.get(key, default), default)

    def _order_log_context(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        row: pd.Series | dict[str, Any],
        feat: dict[str, Any] | None,
        pos_before: BrokerPosition,
        fill_info: dict[str, Any],
        order_id: str,
        client_order_id: str,
        status: str,
    ) -> dict[str, Any]:
        """Build executed-order log kwargs, including exit reasoning and MTF context."""
        r = row.to_dict() if isinstance(row, pd.Series) else dict(row or {})
        feat = feat or {}
        side = str(side or "").lower()
        fill_price = to_float(fill_info.get("filled_avg_price"), to_float(r.get("price"), 0.0))
        avg_entry = to_float(pos_before.avg_entry_price, 0.0)
        pnl_pct = to_float(pos_before.unrealized_plpc, 0.0) if side == "sell" else None
        unrealized_pnl = to_float(pos_before.unrealized_pl, 0.0) if side == "sell" else None
        realized_pnl = None
        if side == "sell" and fill_price > 0 and avg_entry > 0:
            realized_pnl = (fill_price - avg_entry) * float(qty)

        exit_reason = r.get("exit_reason") or (r.get("reason") if side == "sell" else None)
        exit_model_action = r.get("exit_model_action") or ("EXIT" if side == "sell" else None)

        def row_or_feat(name: str, fallback: float | None = None) -> float | None:
            value = r.get(name, None)
            if value is not None and value != "":
                return to_float(value, fallback)
            return _safe_dict_get_float(feat, name, fallback)

        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": "market",
            "submitted_at": str(fill_info.get("submitted_at", "") or ""),
            "filled_at": str(fill_info.get("filled_at", "") or ""),
            "filled_avg_price": fill_price,
            "status": str(fill_info.get("fill_status", status)),
            "order_id": order_id,
            "client_order_id": client_order_id,
            "strategy": "regime_adaptive_intraday",
            "regime": str(r.get("market_regime", r.get("regime", ""))),
            "confidence": to_float(r.get("confidence_used", r.get("trade_confidence", r.get("confidence", 0.0))), None),
            "reason": str(r.get("reason", "")),
            "exit_reason": str(exit_reason or "") if side == "sell" else None,
            "exit_model_action": str(exit_model_action or "") if side == "sell" else None,
            "exit_probability_hold": row_or_feat("exit_probability_hold", row_or_feat("prob_hold_training_target", None)),
            "exit_probability_giveback": row_or_feat("exit_probability_giveback", row_or_feat("giveback_risk_training_target", None)),
            "exit_probability_reversal": row_or_feat("exit_probability_reversal", row_or_feat("reversal_risk_training_target", None)),
            "expected_value_next_5m": row_or_feat("expected_value_next_5m", row_or_feat("future_return_5m", None)),
            "expected_value_next_15m": row_or_feat("expected_value_next_15m", row_or_feat("future_return_15m", None)),
            "expected_value_next_30m": row_or_feat("expected_value_next_30m", row_or_feat("future_return_30m", None)),
            "pnl_pct": pnl_pct,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "max_favorable_excursion": row_or_feat("max_favorable_excursion", row_or_feat("max_favorable_excursion_15m", None)),
            "max_adverse_excursion": row_or_feat("max_adverse_excursion", row_or_feat("max_adverse_excursion_15m", None)),
            "bars_held": to_int(r.get("bars_held", 0), 0) if side == "sell" else None,
            "minutes_held": to_float(r.get("minutes_held", self._minutes_held_for_position(pos_before) if side == "sell" else None), None),
            "distance_to_5m_support": row_or_feat("distance_to_5m_support", row_or_feat("5m_dist_to_support", None)),
            "distance_to_5m_resistance": row_or_feat("distance_to_5m_resistance", row_or_feat("5m_dist_to_resistance", None)),
            "distance_to_15m_support": row_or_feat("distance_to_15m_support", row_or_feat("15m_dist_to_support", None)),
            "distance_to_15m_resistance": row_or_feat("distance_to_15m_resistance", row_or_feat("15m_dist_to_resistance", None)),
            "distance_to_1h_support": row_or_feat("distance_to_1h_support", row_or_feat("1h_dist_to_support", None)),
            "distance_to_1h_resistance": row_or_feat("distance_to_1h_resistance", row_or_feat("1h_dist_to_resistance", None)),
            "distance_to_4h_support": row_or_feat("distance_to_4h_support", row_or_feat("4h_dist_to_support", None)),
            "distance_to_4h_resistance": row_or_feat("distance_to_4h_resistance", row_or_feat("4h_dist_to_resistance", None)),
            "trend_5m": to_int(row_or_feat("5m_trend_stack", None), None),
            "trend_15m": to_int(row_or_feat("15m_trend_stack", None), None),
            "trend_1h": to_int(row_or_feat("1h_trend_stack", None), None),
            "trend_4h": to_int(row_or_feat("4h_trend_stack", None), None),
            "above_vwap": to_int(row_or_feat("above_vwap", None), None),
            "macd_hist": row_or_feat("macd_hist", None),
            "rsi_14": row_or_feat("rsi_14", None),
        }

    def _mtf_buy_gate_for_row(self, row: pd.Series, latest_features: dict[str, dict[str, Any]]) -> tuple[bool, str]:
        """Final BUY-only gate using 5m/15m/1h/4h context. Sells bypass this."""
        if not self.config.scanner.use_mtf_buy_gate:
            return True, "mtf_gate_disabled"

        symbol = safe_symbol(row.get("symbol"))
        feat = latest_features.get(symbol, {})
        if not feat:
            return True, "feature_stream_missing_allow"

        reasons: list[str] = []
        close = self._row_float(feat, "close", to_float(row.get("price"), 0.0))
        dist_vwap = self._row_float(feat, "dist_vwap", 0.0)
        trend_5m = self._row_float(feat, "5m_trend_stack", 0.0)
        trend_15m = self._row_float(feat, "15m_trend_stack", 0.0)
        trend_1h = self._row_float(feat, "1h_trend_stack", 0.0)
        brk_5m = self._row_float(feat, "5m_breakout_20", 0.0)
        brk_15m = self._row_float(feat, "15m_breakout_20", 0.0)
        dres_5m = self._row_float(feat, "5m_dist_to_resistance", 999.0)
        dres_15m = self._row_float(feat, "15m_dist_to_resistance", 999.0)
        dsup_5m = self._row_float(feat, "5m_dist_to_support", 999.0)
        dsup_15m = self._row_float(feat, "15m_dist_to_support", 999.0)

        if self.config.scanner.mtf_require_price_above_vwap and dist_vwap < 0:
            return False, f"mtf_block:below_1m_vwap:{dist_vwap:.3%}"

        if self.config.scanner.mtf_require_5m_or_15m_trend and max(trend_5m, trend_15m, brk_5m, brk_15m) <= 0:
            return False, "mtf_block:no_5m_15m_trend_or_breakout"

        near_resistance = min(dres_5m, dres_15m)
        if near_resistance <= self.config.scanner.mtf_block_buy_near_resistance_pct and max(brk_5m, brk_15m) <= 0:
            return False, f"mtf_block:near_resistance:{near_resistance:.3%}"

        if trend_1h > 0:
            reasons.append("1h_trend_aligned")
        if min(dsup_5m, dsup_15m) <= self.config.scanner.mtf_near_support_bonus_pct:
            reasons.append("near_support")
        if max(brk_5m, brk_15m) > 0:
            reasons.append("short_tf_breakout")
        if close > 0:
            reasons.append(f"close={close:.2f}")

        return True, "mtf_pass:" + ",".join(reasons or ["basic"])

    def _apply_mtf_buy_gate_to_plan(self, plan: pd.DataFrame) -> pd.DataFrame:
        if plan.empty or self.scanner is None or not self.config.scanner.use_mtf_buy_gate:
            return plan

        latest_features = self._latest_feature_by_symbol()
        if not latest_features:
            print("[mtf-gate] no fresh feature stream found; BUY rows allowed by fallback")
            return plan

        keep = []
        notes = []
        open_minutes = minutes_since_regular_open()
        for _, row in plan.iterrows():
            side = str(row.get("side", "")).lower()
            symbol = safe_symbol(row.get("symbol"))
            if side != "buy":
                keep.append(True)
                notes.append("sell_exit_always_allowed")
                continue

            if 0 <= open_minutes < self.config.scanner.no_new_positions_first_minutes:
                keep.append(False)
                notes.append(f"blocked_opening_noise_window_{open_minutes:.1f}m")
                print(f"[opening-gate] blocked BUY {symbol}; market open only {open_minutes:.1f}m ago")
                continue

            blocked, cooldown_reason = no_reentry_cooldown_active(symbol, self.config.scanner.symbol_reentry_cooldown_minutes)
            if blocked:
                keep.append(False)
                notes.append(cooldown_reason)
                print(f"[reentry-gate] blocked BUY {symbol}: {cooldown_reason}")
                continue

            allow, reason = self._mtf_buy_gate_for_row(row, latest_features)
            keep.append(bool(allow))
            notes.append(reason)
            if not allow:
                print(f"[mtf-gate] blocked BUY {symbol}: {reason}")
                write_event("mtf_gate_blocked_buy", {"symbol": symbol, "reason": reason})

        gated = plan.loc[keep].copy() if keep else plan.iloc[0:0].copy()
        if not gated.empty:
            gated["mtf_gate_reason"] = [n for k, n in zip(keep, notes) if k]
            buy_mask = gated["side"] == "buy"
            gated.loc[buy_mask, "reason"] = gated.loc[buy_mask, "reason"].astype(str) + ":" + gated.loc[buy_mask, "mtf_gate_reason"].astype(str)
        return gated

    def _minutes_held_for_position(self, pos: BrokerPosition) -> float:
        # BrokerPosition does not currently store entry timestamp, so use a safe high value.
        # Live experience replay can later add true entry_time for exact hold-time gating.
        return 999.0

    def _exit_learning_decision(self, symbol: str, pos: BrokerPosition, feat: dict[str, Any]) -> dict[str, Any]:
        plpc = to_float(pos.unrealized_plpc, 0.0)
        ev15 = self._row_float(feat, "expected_value_next_15m", self._row_float(feat, "future_return_15m", 0.0))
        ev30 = self._row_float(feat, "expected_value_next_30m", self._row_float(feat, "future_return_30m", 0.0))
        giveback_target = self._row_float(feat, "giveback_risk_training_target", 0.0)
        reversal_target = self._row_float(feat, "reversal_risk_training_target", 0.0)
        hold_target = self._row_float(feat, "prob_hold_training_target", 1.0)
        label = str(feat.get("exit_label", "HOLD")).upper()
        macd_hist = self._row_float(feat, "macd_hist", 0.0)
        dist_vwap = self._row_float(feat, "dist_vwap", 0.0)
        trend_5m = self._row_float(feat, "5m_trend_stack", 0.0)
        trend_15m = self._row_float(feat, "15m_trend_stack", 0.0)
        breakdown_5m = self._row_float(feat, "5m_breakdown_20", 0.0)
        breakdown_15m = self._row_float(feat, "15m_breakdown_20", 0.0)

        minutes_held = self._minutes_held_for_position(pos)
        min_hold = self.config.scanner.exit_learning_min_hold_minutes
        if minutes_held < min_hold and plpc > self.config.scanner.exit_learning_cut_loser_max_plpc:
            return {"action": "HOLD", "reason": f"min_hold_{minutes_held:.1f}m", "confidence": 0.0}

        bearish_structure = (dist_vwap < 0 and macd_hist < 0) or max(breakdown_5m, breakdown_15m) > 0
        trend_lost = max(trend_5m, trend_15m) <= 0

        if label == "CUT_LOSER_NOW" or (plpc <= self.config.scanner.exit_learning_cut_loser_max_plpc and bearish_structure):
            return {"action": "EXIT", "reason": f"exit_learning_cut_loser:plpc={plpc:.3%}:ev15={ev15:.3%}", "confidence": 0.85}

        if plpc >= self.config.scanner.exit_learning_take_profit_min_plpc:
            if label == "TAKE_PROFIT_NOW" or (ev15 <= self.config.scanner.exit_learning_negative_ev_threshold and hold_target < self.config.scanner.exit_learning_prob_hold_threshold):
                return {"action": "EXIT", "reason": f"exit_learning_take_profit:plpc={plpc:.3%}:ev15={ev15:.3%}", "confidence": 0.75}
            if label == "SCALE_OUT" or (giveback_target >= 1 and ev15 <= 0.002):
                return {"action": "SCALE_OUT", "reason": f"exit_learning_scale_out:plpc={plpc:.3%}:ev15={ev15:.3%}", "confidence": 0.68}

        if plpc >= self.config.scanner.exit_learning_trail_stop_min_plpc:
            if label == "TRAIL_STOP_NOW" or (giveback_target >= 1 and trend_lost):
                return {"action": "SCALE_OUT", "reason": f"exit_learning_giveback_risk:plpc={plpc:.3%}:trend_lost={trend_lost}", "confidence": 0.70}

        if reversal_target >= 1 and ev15 < 0 and ev30 < 0:
            return {"action": "EXIT", "reason": f"exit_learning_reversal_risk:ev15={ev15:.3%}:ev30={ev30:.3%}", "confidence": 0.72}

        return {"action": "HOLD", "reason": f"exit_learning_hold:plpc={plpc:.3%}:ev15={ev15:.3%}", "confidence": 0.50}

    def _augment_exit_orders_with_exit_learning(self, exit_orders: pd.DataFrame | None) -> pd.DataFrame:
        if self.scanner is None or not self.config.scanner.exit_learning_enabled:
            return exit_orders if exit_orders is not None else pd.DataFrame()

        exits = exit_orders.copy() if exit_orders is not None else pd.DataFrame()
        latest = self._latest_feature_by_symbol()
        if not latest:
            return exits

        existing = set(exits["symbol"].map(safe_symbol)) if not exits.empty and "symbol" in exits.columns else set()
        rows: list[dict[str, Any]] = []

        for pos in self.broker.get_all_open_positions():
            symbol = safe_symbol(pos.symbol)
            if not symbol or symbol in existing:
                continue
            feat = latest.get(symbol, {})
            if not feat:
                continue
            decision = self._exit_learning_decision(symbol, pos, feat)
            action = str(decision.get("action", "HOLD")).upper()
            if action not in {"EXIT", "SCALE_OUT"}:
                continue

            qty_total = int(max(0, to_float(pos.qty, 0.0)))
            if qty_total <= 0:
                continue
            if action == "SCALE_OUT":
                qty = max(1, int(qty_total * self.config.scanner.exit_learning_scale_out_fraction))
            else:
                qty = qty_total

            price = to_float(pos.current_price, to_float(pos.avg_entry_price, 0.0))
            if price <= 0:
                continue

            reason = str(decision.get("reason", "exit_learning"))
            rows.append({
                "symbol": symbol,
                "side": "sell",
                "delta_shares": qty,
                "delta_dollars": qty * price,
                "price": price,
                "reason": reason,
                "target_weight": 0.0,
                "current_weight": 0.0,
                "delta_weight": 0.0,
                "trade_confidence": to_float(decision.get("confidence"), 0.65),

                # Exit-reasoning payload preserved into the final executed-order log.
                "exit_reason": reason,
                "exit_model_action": action,
                "exit_probability_hold": self._row_float(feat, "prob_hold_training_target", None),
                "exit_probability_giveback": self._row_float(feat, "giveback_risk_training_target", None),
                "exit_probability_reversal": self._row_float(feat, "reversal_risk_training_target", None),
                "expected_value_next_5m": self._row_float(feat, "expected_value_next_5m", self._row_float(feat, "future_return_5m", None)),
                "expected_value_next_15m": self._row_float(feat, "expected_value_next_15m", self._row_float(feat, "future_return_15m", None)),
                "expected_value_next_30m": self._row_float(feat, "expected_value_next_30m", self._row_float(feat, "future_return_30m", None)),
                "max_favorable_excursion": self._row_float(feat, "max_favorable_excursion_15m", None),
                "max_adverse_excursion": self._row_float(feat, "max_adverse_excursion_15m", None),
                "distance_to_5m_support": self._row_float(feat, "5m_dist_to_support", None),
                "distance_to_5m_resistance": self._row_float(feat, "5m_dist_to_resistance", None),
                "distance_to_15m_support": self._row_float(feat, "15m_dist_to_support", None),
                "distance_to_15m_resistance": self._row_float(feat, "15m_dist_to_resistance", None),
                "distance_to_1h_support": self._row_float(feat, "1h_dist_to_support", None),
                "distance_to_1h_resistance": self._row_float(feat, "1h_dist_to_resistance", None),
                "distance_to_4h_support": self._row_float(feat, "4h_dist_to_support", None),
                "distance_to_4h_resistance": self._row_float(feat, "4h_dist_to_resistance", None),
                "5m_trend_stack": self._row_float(feat, "5m_trend_stack", None),
                "15m_trend_stack": self._row_float(feat, "15m_trend_stack", None),
                "1h_trend_stack": self._row_float(feat, "1h_trend_stack", None),
                "4h_trend_stack": self._row_float(feat, "4h_trend_stack", None),
                "above_vwap": self._row_float(feat, "above_vwap", None),
                "macd_hist": self._row_float(feat, "macd_hist", None),
                "rsi_14": self._row_float(feat, "rsi_14", None),
            })
            print(f"[exit-learning] {action} {symbol} qty={qty}/{qty_total}: {reason}")
            write_event("exit_learning_exit_added", {"symbol": symbol, "action": action, **decision})

        if rows:
            exits = pd.concat([exits, pd.DataFrame(rows)], ignore_index=True)
        return exits

    def _build_order_plan(self, rebalance: pd.DataFrame, exits: pd.DataFrame, equity: float) -> pd.DataFrame:
        rb = self._normalize_order_frame(rebalance)
        ex = self._normalize_order_frame(exits, default_reason="exit_signal")
        plan = pd.concat([ex, rb], ignore_index=True)

        if plan.empty:
            return pd.DataFrame()

        # Never re-enter a symbol on the same day it was exited.
        # This guard is BUY-ONLY. SELL/EXIT orders must always remain allowed.
        no_reentry_symbols = load_no_reentry_symbols()
        same_cycle_exit_symbols = set(plan.loc[plan["side"] == "sell", "symbol"].map(safe_symbol))
        blocked_buy_symbols = {sym for sym in no_reentry_symbols.union(same_cycle_exit_symbols) if sym}
        if blocked_buy_symbols:
            buy_block_mask = (plan["side"] == "buy") & plan["symbol"].map(safe_symbol).isin(blocked_buy_symbols)
            blocked_count = int(buy_block_mask.sum())
            if blocked_count:
                print(f"[no-reentry] removed {blocked_count} BUY order(s); EXIT orders remain allowed: {sorted(blocked_buy_symbols)}")
                plan = plan[~buy_block_mask].copy()
                if plan.empty:
                    return pd.DataFrame()

        plan["abs_delta_shares"] = plan["delta_shares"].abs().fillna(0)
        plan["abs_delta_dollars"] = plan["delta_dollars"].abs().fillna(0)
        _low_notional_buys = plan[(plan["side"] == "buy") & (plan["abs_delta_dollars"] < self.config.min_notional_to_trade)]
        if not _low_notional_buys.empty:
            for _, _r in _low_notional_buys.iterrows():
                print(f"[order-plan] skip BUY {_r.get('symbol', '?')}: notional=${to_float(_r.get('abs_delta_dollars')):.2f} < min=${self.config.min_notional_to_trade:.2f}")
        plan = plan[(plan["side"] == "sell") | (plan["abs_delta_dollars"] >= self.config.min_notional_to_trade)].copy()
        if plan.empty:
            return pd.DataFrame()

        plan["confidence_used"] = plan.apply(self._confidence_for_row, axis=1)
        plan["confidence_multiplier"] = plan["confidence_used"].map(self._confidence_multiplier)

        buy_mask = plan["side"] == "buy"
        plan.loc[buy_mask, "abs_delta_dollars"] *= plan.loc[buy_mask, "confidence_multiplier"]

        max_order_notional = self._max_order_notional(equity)
        plan.loc[buy_mask, "capped_dollars"] = plan.loc[buy_mask, "abs_delta_dollars"].clip(upper=max_order_notional)
        plan.loc[~buy_mask, "capped_dollars"] = plan.loc[~buy_mask, "abs_delta_dollars"]

        plan["order_qty"] = plan["abs_delta_shares"].fillna(0).astype(int)
        qty_from_dollars = (plan["order_qty"] <= 0) & (plan["price"] > 0)
        plan.loc[qty_from_dollars, "order_qty"] = (
            plan.loc[qty_from_dollars, "capped_dollars"] / plan.loc[qty_from_dollars, "price"]
        ).fillna(0).astype(int)

        _bad_qty_buys = plan[(plan["side"] == "buy") & ((plan["order_qty"] <= 0) | (plan["order_qty"] > self.config.max_single_order_qty))]
        if not _bad_qty_buys.empty:
            for _, _r in _bad_qty_buys.iterrows():
                _q = to_int(_r.get("order_qty"), 0)
                _reason = "zero_qty" if _q <= 0 else f"qty={_q}>{self.config.max_single_order_qty}_max"
                print(f"[order-plan] skip BUY {_r.get('symbol', '?')}: {_reason} price=${to_float(_r.get('price')):.2f} capped=${to_float(_r.get('capped_dollars')):.2f}")
        plan = plan[(plan["order_qty"] > 0) & (plan["order_qty"] <= self.config.max_single_order_qty)].copy()
        if plan.empty:
            return pd.DataFrame()

        side_rank = {"sell": 0, "buy": 1} if self.config.sell_first else {"buy": 0, "sell": 1}
        plan["side_rank"] = plan["side"].map(side_rank).fillna(9)
        plan = plan.sort_values(["side_rank", "capped_dollars"], ascending=[True, False]).reset_index(drop=True)

        stamp = run_id()
        plan["client_order_id"] = [
            f"{self.config.order_tag_prefix}_{stamp}_{i}_{row.symbol}_{row.side}"[:48]
            for i, row in plan.iterrows()
        ]
        plan["planned_at"] = now_utc().isoformat()
        return plan

    def _poll_fill(self, order_id: str, planned_price: float) -> dict[str, Any]:
        last_order = None
        for _ in range(6):
            try:
                last_order = self.broker.get_order_by_id(order_id)
            except Exception as exc:
                return {"fill_status": "fill_lookup_error", "filled_qty": 0.0, "filled_avg_price": 0.0, "slippage_pct": 0.0, "message": str(exc)}

            status = str(getattr(last_order, "status", "")).lower()
            if status in {"filled", "partially_filled", "canceled", "rejected", "expired"}:
                break
            time.sleep(2.0)

        if last_order is None:
            return {"fill_status": "unknown", "filled_qty": 0.0, "filled_avg_price": 0.0, "slippage_pct": 0.0}

        filled_qty = to_float(getattr(last_order, "filled_qty", 0.0))
        filled_avg_price = to_float(getattr(last_order, "filled_avg_price", 0.0))
        status = str(getattr(last_order, "status", ""))
        slippage_pct = filled_avg_price / planned_price - 1.0 if planned_price > 0 and filled_avg_price > 0 else 0.0

        return {
            "fill_status": status,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
            "slippage_pct": slippage_pct,
            "submitted_at": str(getattr(last_order, "submitted_at", "") or ""),
            "filled_at": str(getattr(last_order, "filled_at", "") or ""),
            "expired_at": str(getattr(last_order, "expired_at", "") or ""),
        }

    def _journal_order(
        self,
        cycle_id: str,
        event: str,
        order_row: pd.Series | dict[str, Any],
        acct_before: dict[str, Any],
        acct_after: dict[str, Any],
        message: str,
        order_id: str = "",
        order_status: str = "",
        fill_info: Optional[dict[str, Any]] = None,
    ) -> None:
        row = order_row.to_dict() if isinstance(order_row, pd.Series) else dict(order_row)
        fill_info = fill_info or {}
        self.journal.append(
            {
                "timestamp_utc": now_utc().isoformat(),
                "cycle_id": cycle_id,
                "event": event,
                "market_open": True,
                "equity_before": acct_before.get("equity", ""),
                "equity_after": acct_after.get("equity", ""),
                "cash_before": acct_before.get("cash", ""),
                "cash_after": acct_after.get("cash", ""),
                "buying_power_before": acct_before.get("buying_power", ""),
                "buying_power_after": acct_after.get("buying_power", ""),
                "gross_exposure_before": acct_before.get("gross_exposure", ""),
                "gross_exposure_after": acct_after.get("gross_exposure", ""),
                "trade_symbol": row.get("symbol", ""),
                "trade_side": str(row.get("side", "")).upper(),
                "trade_qty": row.get("order_qty", ""),
                "trade_price": row.get("price", ""),
                "estimated_notional": row.get("capped_dollars", row.get("abs_delta_dollars", "")),
                "target_weight": row.get("target_weight", ""),
                "current_weight": row.get("current_weight", ""),
                "delta_weight": row.get("delta_weight", ""),
                "confidence": row.get("confidence_used", ""),
                "cash_buffer": self._cash_buffer(float(acct_before.get("equity", 0.0))),
                "max_gross_exposure": self.config.max_gross_exposure,
                "max_position_weight": self.config.max_position_weight,
                "max_order_notional": self._max_order_notional(float(acct_before.get("equity", 0.0))),
                "order_id": order_id,
                "order_status": order_status,
                "filled_qty": fill_info.get("filled_qty", ""),
                "filled_avg_price": fill_info.get("filled_avg_price", ""),
                "slippage_pct": fill_info.get("slippage_pct", ""),
                "message": message,
            }
        )

    def _execute_order_plan(self, plan: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
        results: list[dict[str, Any]] = []
        if plan.empty:
            return pd.DataFrame()

        direction_by_symbol: dict[str, dict[str, Any]] = {}
        if self.scanner is not None and self.config.scanner.require_direction_confirmation_before_buy:
            buy_symbols = sorted(set(plan.loc[plan["side"] == "buy", "symbol"].map(safe_symbol)))
            if buy_symbols:
                direction_df = self.scanner.confirm_buy_direction(buy_symbols)
                if not direction_df.empty:
                    direction_by_symbol = {
                        safe_symbol(row["symbol"]): row.to_dict()
                        for _, row in direction_df.iterrows()
                    }

        # Load once per cycle so executed-order logs capture the same
        # multi-timeframe/exit-learning context that gated the trade.
        latest_feature_by_symbol = self._latest_feature_by_symbol()

        for _, row in plan.iterrows():
            symbol = safe_symbol(row.get("symbol"))
            side = str(row.get("side", "")).lower()
            qty = to_int(row.get("order_qty"))
            price = to_float(row.get("price"))
            est_notional = to_float(row.get("capped_dollars", row.get("abs_delta_dollars", 0.0)))
            reason = str(row.get("reason", ""))
            client_order_id = str(row.get("client_order_id"))

            acct_before = self._record_account_snapshot("before_order", cycle_id)
            equity = float(acct_before["equity"])
            cash = float(acct_before["cash"])
            buying_power = float(acct_before["buying_power"])
            gross_before = self._gross_exposure(equity)
            acct_before["gross_exposure"] = gross_before

            skip_message = ""
            direction = direction_by_symbol.get(symbol, {}) if direction_by_symbol else {}
            feature_context = latest_feature_by_symbol.get(symbol, {}) if latest_feature_by_symbol else {}
            position_before_order = self.broker.get_position(symbol)

            # Log the final executable decision state before broker submission/risk skips.
            self._log_live_experience(
                cycle_id=cycle_id,
                stage="decision",
                order_row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                acct_before=acct_before,
                direction=direction,
                message="pre_submit_decision",
            )

            if qty <= 0:
                skip_message = "qty <= 0"
            elif self.broker.has_open_order_for_symbol(symbol):
                skip_message = "existing open order for symbol"

            is_orb_order = bool(row.get("orb_8am_signal", False)) or str(row.get("source", "")) == "orb_8am_liquidity_sweep_fvg"
            if not skip_message and side == "buy" and direction_by_symbol and not is_orb_order:
                if not bool(direction.get("direction_confirmed", False)):
                    skip_message = f"price_direction_not_confirmed:{direction.get('direction_reject_reason', 'unknown')}"
            elif not skip_message and side == "buy" and direction_by_symbol and is_orb_order:
                # ORB entries already require sweep -> rejection/reclaim -> FVG -> retest.
                # Keep the direction metrics in logs, but do not let the generic trend gate
                # block this specific liquidity-reversal setup.
                if direction:
                    row = row.copy()
                    row["direction_gate_bypassed"] = "orb_8am_liquidity_sweep_fvg"

            if not skip_message and side == "buy":
                cash_buffer = self._cash_buffer(equity)
                max_order_notional = self._max_order_notional(equity)
                profile = self._risk_profile(equity)
                gross_room = max(0.0, (profile.max_gross_exposure - gross_before) * equity)

                if buying_power < self.config.min_buying_power:
                    skip_message = "buying_power_below_minimum"
                elif cash - est_notional < cash_buffer:
                    skip_message = f"cash_buffer_violation cash={cash:.2f} projected={cash - est_notional:.2f} required={cash_buffer:.2f}"
                else:
                    if est_notional > max_order_notional:
                        qty = int(max_order_notional // price) if price > 0 else 0
                        est_notional = qty * price
                        print(f"[clip] BUY {symbol}: clipped to qty={qty} by max_order_notional")
                    if est_notional > gross_room:
                        qty = int(gross_room // price) if price > 0 else 0
                        est_notional = qty * price
                        print(f"[clip] BUY {symbol}: clipped to qty={qty} by gross exposure room")
                    if qty <= 0 or est_notional < profile.min_notional_to_trade:
                        skip_message = "buy_clipped_below_min_notional"

            if skip_message:
                if (
                    side == "buy"
                    and self.config.scanner
                    and self.config.scanner.stream_confirmed_append_direction_rejects
                    and str(skip_message).startswith("price_direction_not_confirmed")
                ):
                    append_pending_direction_watch_from_order(
                        {**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                        skip_message,
                    )

                print(f"[skip] {side.upper()} {qty} {symbol}: {skip_message}")
                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "skipped",
                    "message": skip_message,
                }
                results.append(result)
                acct_after = self._record_account_snapshot("after_skip", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, skip_message)
                self._log_live_experience(
                    cycle_id=cycle_id,
                    stage="skipped",
                    order_row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before=acct_before,
                    acct_after=acct_after,
                    result=result,
                    direction=direction,
                    message=skip_message,
                )
                continue

            try:
                if side == "buy":
                    submitted = self.broker.submit_market_buy(symbol, qty, client_order_id, self.config.time_in_force)
                else:
                    submitted = self.broker.submit_market_sell(symbol, qty, client_order_id, self.config.time_in_force)

                order_id = str(getattr(submitted, "id", ""))
                status = str(getattr(submitted, "status", "submitted"))
                print(f"[submitted] {side.upper()} {qty} {symbol} status={status} id={order_id}")
                self._mark_order()

                fill_info = self._poll_fill(order_id, price)
                log_executed_order(
                    **self._order_log_context(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                        feat=feature_context,
                        pos_before=position_before_order,
                        fill_info=fill_info,
                        order_id=order_id,
                        client_order_id=client_order_id,
                        status=status,
                    )
                )
                acct_after = self._record_account_snapshot("after_order", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))

                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "estimated_notional": est_notional,
                    "price": price,
                    "reason": reason,
                    "status": status,
                    "order_id": order_id,
                    **fill_info,
                }
                results.append(result)
                self._journal_order(
                    cycle_id,
                    "ORDER",
                    {**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before,
                    acct_after,
                    reason or "submitted",
                    order_id,
                    status,
                    fill_info,
                )
                write_event("order_submitted", result)

                if (
                    side == "buy"
                    and self.config.scanner
                    and self.config.scanner.stream_confirmed_submit_trailing_stop_after_fill
                ):
                    filled_qty_for_trail = to_int(fill_info.get("filled_qty", qty), qty)
                    submit_trailing_stop_after_entry(
                        self.broker,
                        symbol,
                        filled_qty_for_trail,
                        {**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    )

                self._log_live_experience(
                    cycle_id=cycle_id,
                    stage="submitted",
                    order_row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before=acct_before,
                    acct_after=acct_after,
                    result=result,
                    direction=direction,
                    message=reason or "submitted",
                )
                if side == "sell":
                    fill_status = str(fill_info.get("fill_status", status)).lower()
                    if fill_status in {"filled", "partially_filled", "submitted", "accepted", "new"} or to_float(fill_info.get("filled_qty", 0.0)) > 0:
                        mark_no_reentry_symbol(symbol, reason=reason or "sell_order_submitted")
                time.sleep(0.35)

            except APIError as exc:
                message = str(exc)
                print(f"[error] APIError {side.upper()} {qty} {symbol}: {message}")
                acct_after = self._record_account_snapshot("after_error", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                result = {"timestamp_utc": now_utc().isoformat(), "symbol": symbol, "side": side, "qty": qty, "status": "api_error", "message": message, "order_id": ""}
                results.append(result)
                self._journal_order(cycle_id, "ERROR", row, acct_before, acct_after, message)
                write_event("order_api_error", result)
                self._log_live_experience(
                    cycle_id=cycle_id,
                    stage="api_error",
                    order_row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before=acct_before,
                    acct_after=acct_after,
                    result=result,
                    direction=direction,
                    message=message,
                )

            except Exception as exc:
                message = str(exc)
                print(f"[error] submit failed {side.upper()} {qty} {symbol}: {message}")
                acct_after = self._record_account_snapshot("after_error", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                result = {"timestamp_utc": now_utc().isoformat(), "symbol": symbol, "side": side, "qty": qty, "status": "error", "message": message, "order_id": ""}
                results.append(result)
                self._journal_order(cycle_id, "ERROR", row, acct_before, acct_after, message)
                write_event("order_error", result)
                self._log_live_experience(
                    cycle_id=cycle_id,
                    stage="error",
                    order_row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before=acct_before,
                    acct_after=acct_after,
                    result=result,
                    direction=direction,
                    message=message,
                )

        return pd.DataFrame(results)

    def run_once_from_frames(self, rebalance_orders: pd.DataFrame, exit_orders: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        cycle_id = run_id()
        acct_start = self._record_account_snapshot("cycle_start", cycle_id)
        equity = float(acct_start["equity"])
        cash = float(acct_start["cash"])
        gross = self._gross_exposure(equity)
        acct_start["gross_exposure"] = gross
        positions_before = pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])

        if not self._market_is_open_or_wait():
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_market_closed", "message": "market is closed"}])
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        allowed, kill_reason, daily_return = self._kill_switch_allows_execution(equity)
        if not allowed:
            print(f"[kill-switch] execution blocked: {kill_reason}")
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_kill_switch", "message": kill_reason, "daily_return": daily_return}])
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        if self.config.cancel_open_orders_first:
            print("[orders] canceling open orders")
            self.broker.cancel_open_orders()

        if self.config.scanner and self.config.scanner.stream_confirmed_merge_orders:
            rebalance_orders = merge_stream_confirmed_orders(rebalance_orders)
        if self.config.scanner and self.config.scanner.sr_rl_merge_signals:
            rebalance_orders = merge_sr_rl_orders(rebalance_orders, self.config.scanner)

        exit_orders = self._augment_exit_orders_with_exit_learning(exit_orders if exit_orders is not None else pd.DataFrame())
        plan = self._build_order_plan(rebalance_orders, exit_orders, equity)
        plan = self._apply_mtf_buy_gate_to_plan(plan)
        plan = self._apply_planner_and_market_gates_to_plan(plan)

        # Cooldown is buy-only. Exits are always allowed so risk can be reduced immediately.
        if self._in_cooldown() and not plan.empty:
            sells = plan[plan["side"] == "sell"].copy()
            buys = plan[plan["side"] == "buy"].copy()
            if not sells.empty:
                print(f"[cooldown] active; allowing {len(sells)} exit order(s), deferring {len(buys)} buy order(s)")
                plan = sells
            else:
                print("[cooldown] active; no exits to process, buys deferred")
                results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_cooldown", "message": "cooldown active; buys deferred"}])
                save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
                save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
                self._write_account_equity_curve()
                return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before
        if plan.empty:
            print("[plan] no valid orders")
            save_df(plan, LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(pd.DataFrame(), LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return plan, pd.DataFrame(), self.broker.get_account_snapshot(), positions_before

        plan = merge_evolution_strategy_orders(plan)

        print("\n===== ORDER PLAN =====")
        display_cols = [
            "symbol", "side", "order_qty", "price", "abs_delta_dollars", "capped_dollars",
            "ensemble_score", "up_probability", "expected_return", "drawdown_probability",
            "attention_lstm_score", "confidence_used", "confidence_multiplier", "reason",
        ]
        print(plan[[c for c in display_cols if c in plan.columns]].to_string(index=False))
        save_df(plan, LOG_DIR / f"order_plan_{cycle_id}.csv")

        results = self._execute_order_plan(plan, cycle_id)
        save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")

        positions_after = pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])
        save_df(positions_after, LOG_DIR / f"positions_after_{cycle_id}.csv")

        acct_after = self._record_account_snapshot("cycle_end", cycle_id)
        acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
        self._write_account_equity_curve()

        save_json(
            {
                "cycle_id": cycle_id,
                "timestamp_utc": now_utc().isoformat(),
                "paper": self.config.paper,
                "mode": "EXECUTION",
                "equity_before": equity,
                "cash_before": cash,
                "gross_exposure_before": gross,
                "equity_after": acct_after["equity"],
                "cash_after": acct_after["cash"],
                "gross_exposure_after": acct_after["gross_exposure"],
                "orders_planned": len(plan),
                "results_rows": len(results),
            },
            LOG_DIR / f"execution_summary_{cycle_id}.json",
        )

        return plan, results, self.broker.get_account_snapshot(), positions_before

    def _regime_plan_to_order_frames(self, regime_plan: pd.DataFrame, equity: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert regime_position_plan.csv into executor-compatible rebalance/exit frames."""
        if regime_plan is None or regime_plan.empty:
            return pd.DataFrame(), pd.DataFrame()

        positions = self._positions_by_symbol()
        buys: list[dict[str, Any]] = []
        exits: list[dict[str, Any]] = []

        for _, row in regime_plan.iterrows():
            symbol = safe_symbol(row.get("symbol"))
            if not symbol:
                continue

            action = _regime_action_value(row)
            target_weight = _regime_target_weight(row, 0.0)
            confidence = _regime_confidence(row, 0.55)
            pos = positions.get(symbol, BrokerPosition(symbol=symbol))
            current_value = max(0.0, float(pos.market_value or 0.0))
            current_weight = current_value / max(equity, 1.0)
            price = _regime_price(row, float(pos.current_price or pos.avg_entry_price or 0.0))
            if price <= 0:
                # The actual executor can still size from shares if supplied, but regime plans usually need a price.
                print(f"[regime-plan] skipping {symbol}: no usable price")
                continue

            reason = f"regime_model:{action or 'NO_ACTION'}"

            if action in {"LIQUIDATE", "EXIT", "EXIT_LONG", "SELL", "CLOSE", "CLOSE_LONG", "REDUCE_TO_ZERO"}:
                qty = int(max(0, float(pos.qty or 0)))
                if qty > 0:
                    exits.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "delta_shares": qty,
                            "delta_dollars": qty * price,
                            "price": price,
                            "reason": reason,
                            "target_weight": 0.0,
                            "current_weight": current_weight,
                            "delta_weight": -current_weight,
                            "trade_confidence": confidence,
                        }
                    )
                continue

            if action in {"OPEN_LONG", "BUY", "HOLD_LONG", "ADD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION", "REBALANCE_LONG"} and target_weight > 0:
                target_value = equity * target_weight
                delta_dollars = target_value - current_value
                if delta_dollars >= self.config.min_notional_to_trade:
                    buys.append(
                        {
                            "symbol": symbol,
                            "side": "buy",
                            "delta_shares": max(0, int(delta_dollars / max(price, 0.01))),
                            "delta_dollars": delta_dollars,
                            "price": price,
                            "reason": reason,
                            "target_weight": target_weight,
                            "current_weight": current_weight,
                            "delta_weight": target_weight - current_weight,
                            "trade_confidence": confidence,
                        }
                    )
                elif delta_dollars <= -self.config.min_notional_to_trade and float(pos.qty or 0) > 0:
                    reduce_qty = int(min(float(pos.qty), abs(delta_dollars) / max(price, 0.01)))
                    if reduce_qty > 0:
                        exits.append(
                            {
                                "symbol": symbol,
                                "side": "sell",
                                "delta_shares": reduce_qty,
                                "delta_dollars": reduce_qty * price,
                                "price": price,
                                "reason": reason + ":reduce_to_target",
                                "target_weight": target_weight,
                                "current_weight": current_weight,
                                "delta_weight": target_weight - current_weight,
                                "trade_confidence": confidence,
                            }
                        )

        rebalance = pd.DataFrame(buys)
        exits_df = pd.DataFrame(exits)
        if not rebalance.empty:
            save_df(rebalance, self.config.rebalance_orders_path)
        if not exits_df.empty:
            save_df(exits_df, self.config.exit_orders_path)
        print(f"[regime-plan] converted to buys={len(rebalance)} exits={len(exits_df)}")
        return rebalance, exits_df

    def _load_rl_structure_map(self) -> dict[str, Any]:
        if not INTRADAY_STRUCTURE_MAP_PATH.exists():
            return {}
        return read_json_file(INTRADAY_STRUCTURE_MAP_PATH, {})

    def _download_latest_bars_for_rl(self, symbols: list[str]) -> pd.DataFrame:
        symbols = [safe_symbol(s) for s in symbols if safe_symbol(s)]
        if not symbols or self.scanner is None:
            if INTRADAY_BARS_1M_PATH.exists():
                try:
                    return pd.read_csv(INTRADAY_BARS_1M_PATH)
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()

        old_lookback = self.scanner.config.intraday_lookback_days
        try:
            self.scanner.config.intraday_lookback_days = max(
                old_lookback,
                int(self.config.scanner.rl_latest_bar_lookback_days),
            )
            bars = self.scanner._download_intraday_bars(symbols, int(self.config.scanner.rl_latest_bar_timeframe_minutes))
            if not bars.empty:
                save_df(bars, INTRADAY_BARS_1M_PATH)
            return bars
        finally:
            self.scanner.config.intraday_lookback_days = old_lookback

    def _rl_validation(self) -> dict[str, Any]:
        if self._rl_validation_cache is None:
            self._rl_validation_cache = validate_rl_intraday_structure_model(self.config.scanner)
        return self._rl_validation_cache

    def _rl_position_state(self, symbol: str, positions_by_symbol: Optional[dict[str, BrokerPosition]] = None) -> dict[str, Any]:
        """Return current broker position state for position-aware RL inference.

        The current RL trainer may or may not expose position-aware keyword
        arguments yet. run.py still computes the state and uses it for a final
        sanity filter so impossible actions do not reach execution.
        """
        symbol = safe_symbol(symbol)
        positions_by_symbol = positions_by_symbol if positions_by_symbol is not None else self._positions_by_symbol()
        pos = positions_by_symbol.get(symbol)

        if not pos or to_float(pos.qty, 0.0) <= 0:
            return {
                "position_side": 0,
                "position_qty": 0.0,
                "entry_price": 0.0,
                "avg_entry_price": 0.0,
                "current_price": 0.0,
                "unrealized_plpc": 0.0,
                "is_position_aware": True,
            }

        # Alpaca long-only equity wrapper returns positive qty for long positions.
        # If later short positions are enabled, extend this to inspect side/asset_class.
        return {
            "position_side": 1,
            "position_qty": float(to_float(pos.qty, 0.0)),
            "entry_price": float(to_float(pos.avg_entry_price, 0.0)),
            "avg_entry_price": float(to_float(pos.avg_entry_price, 0.0)),
            "current_price": float(to_float(pos.current_price, 0.0)),
            "unrealized_plpc": float(to_float(pos.unrealized_plpc, 0.0)),
            "is_position_aware": True,
        }

    def _sanitize_rl_signal_for_position(self, signal: dict[str, Any], position_state: dict[str, Any]) -> dict[str, Any]:
        """Force impossible RL actions back to HOLD using live broker state."""
        signal = dict(signal or {})
        action = str(signal.get("action", "HOLD")).upper()
        side = int(to_int(position_state.get("position_side"), 0))

        blocked_reason: str | None = None

        if side == 1 and action in {"ENTER_LONG", "BUY"}:
            blocked_reason = "position_aware_block_enter_long_already_long"
        elif side == -1 and action in {"ENTER_SHORT", "SHORT"}:
            blocked_reason = "position_aware_block_enter_short_already_short"
        elif side == 0 and action in {"EXIT_LONG", "EXIT_SHORT", "SELL", "COVER"}:
            blocked_reason = "position_aware_block_exit_when_flat"
        elif side == 0 and action in {"ADD_LONG", "PYRAMID_LONG"}:
            blocked_reason = "position_aware_block_add_long_when_flat"
        elif side == 1 and action == "EXIT_SHORT":
            blocked_reason = "position_aware_block_exit_short_while_long"
        elif side == -1 and action == "EXIT_LONG":
            blocked_reason = "position_aware_block_exit_long_while_short"

        signal["position_side"] = side
        signal["position_qty"] = float(to_float(position_state.get("position_qty"), 0.0))
        signal["entry_price"] = float(to_float(position_state.get("entry_price"), 0.0))
        signal["avg_entry_price"] = float(to_float(position_state.get("avg_entry_price"), 0.0))
        signal["position_aware"] = True

        if blocked_reason:
            raw_action = signal.get("raw_action", action)
            signal["raw_position_unaware_action"] = raw_action
            signal["position_aware_blocked"] = True
            signal["position_aware_block_reason"] = blocked_reason
            signal["action"] = "HOLD"
            signal["action_id"] = 0
            signal["reason"] = f"{signal.get('reason', 'rl_signal')}:{blocked_reason}"
        else:
            signal["position_aware_blocked"] = False

        return signal

    def _call_predict_live_action_position_aware(
        self,
        predict_live_action: Any,
        symbol: str,
        latest_bars: pd.DataFrame,
        structure_map: dict[str, Any],
        position_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Call trainer inference with position kwargs when supported.

        This keeps run.py backward-compatible with older trainer files while
        immediately enabling position-aware inference when the trainer accepts
        position_side / position_qty / entry_price.
        """
        base_kwargs = {
            "model_path": RL_MODEL_PATH,
            "latest_bars": latest_bars,
            "structure_map": structure_map,
            "symbol": symbol,
        }
        position_kwargs = {
            "position_side": int(to_int(position_state.get("position_side"), 0)),
            "position_qty": float(to_float(position_state.get("position_qty"), 0.0)),
            "entry_price": float(to_float(position_state.get("entry_price"), 0.0)),
            "avg_entry_price": float(to_float(position_state.get("avg_entry_price"), 0.0)),
            "current_price": float(to_float(position_state.get("current_price"), 0.0)),
            "unrealized_plpc": float(to_float(position_state.get("unrealized_plpc"), 0.0)),
        }

        if self.config.scanner.rl_position_aware_inference:
            try:
                sig = inspect.signature(predict_live_action)
                supported = set(sig.parameters.keys())
                kwargs = dict(base_kwargs)
                kwargs.update({k: v for k, v in position_kwargs.items() if k in supported})
                return dict(predict_live_action(**kwargs))
            except (TypeError, ValueError):
                # Fall back to legacy signature below.
                pass

        return dict(predict_live_action(**base_kwargs))

    def _predict_rl_signal(
        self,
        symbol: str,
        latest_bars: pd.DataFrame,
        structure_map: dict[str, Any],
        positions_by_symbol: Optional[dict[str, BrokerPosition]] = None,
    ) -> dict[str, Any]:
        symbol = safe_symbol(symbol)
        if not self.config.scanner.use_rl_timing_gate:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_gate_disabled"}

        if self.config.scanner.rl_validate_model_before_trading:
            validation = self._rl_validation()
            if not validation.get("ok"):
                return {
                    "symbol": symbol,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": "rl_model_invalid",
                    "validation": validation,
                }

        if not RL_INTRADAY_STRUCTURE_TRAINER_PATH.exists():
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_trainer_missing"}
        if not RL_MODEL_PATH.exists():
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_model_missing"}
        if latest_bars.empty:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "no_latest_bars"}
        if not structure_map:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_structure_map_missing"}

        try:
            from rl_intraday_structure_trainer import predict_live_action

            position_state = self._rl_position_state(symbol, positions_by_symbol)
            signal = self._call_predict_live_action_position_aware(
                predict_live_action=predict_live_action,
                symbol=symbol,
                latest_bars=latest_bars,
                structure_map=structure_map,
                position_state=position_state,
            )
            signal["symbol"] = symbol
            return self._sanitize_rl_signal_for_position(signal, position_state)
        except Exception as exc:
            position_state = self._rl_position_state(symbol, positions_by_symbol)
            return self._sanitize_rl_signal_for_position(
                {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": f"rl_error:{exc}"},
                position_state,
            )

    def _apply_rl_timing_gate(self, rebalance: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Use rl_intraday_structure_trainer.py as timing confirmation.

        BUY rows require ENTER_LONG confidence.
        Existing exits always remain allowed, and RL EXIT signals can add protective exits.
        """
        if not self.config.scanner.use_rl_timing_gate:
            return rebalance, exits

        symbols = []
        if rebalance is not None and not rebalance.empty and "symbol" in rebalance.columns:
            symbols.extend(rebalance["symbol"].map(safe_symbol).tolist())
        positions = self.broker.get_all_open_positions()
        positions_by_symbol = {safe_symbol(p.symbol): p for p in positions if p.symbol}
        symbols.extend([safe_symbol(p.symbol) for p in positions if p.symbol])
        symbols = sorted(set(s for s in symbols if s))

        if not symbols:
            return rebalance, exits

        latest_bars = self._download_latest_bars_for_rl(symbols)
        structure_map = self._load_rl_structure_map()
        signals: list[dict[str, Any]] = []

        # Gate buys.
        gated_rebalance = rebalance.copy() if rebalance is not None else pd.DataFrame()
        if not gated_rebalance.empty:
            keep_mask = []
            for _, row in gated_rebalance.iterrows():
                symbol = safe_symbol(row.get("symbol"))
                signal = self._predict_rl_signal(symbol, latest_bars, structure_map, positions_by_symbol)
                signals.append(signal)
                action = str(signal.get("action", "HOLD")).upper()
                confidence = to_float(signal.get("confidence"), 0.0)
                has_position = bool(positions_by_symbol.get(symbol) and to_float(positions_by_symbol[symbol].qty, 0.0) > 0)
                if has_position and self.config.scanner.rl_enable_add_long_gate:
                    allow = action in {"ADD_LONG", "PYRAMID_LONG"} and confidence >= self.config.scanner.rl_min_add_long_confidence
                    required_action = "ADD_LONG"
                    required_confidence = self.config.scanner.rl_min_add_long_confidence
                else:
                    allow = action in {"ENTER_LONG", "BUY"} and confidence >= self.config.scanner.rl_min_enter_long_confidence
                    required_action = "ENTER_LONG"
                    required_confidence = self.config.scanner.rl_min_enter_long_confidence

                if self.config.scanner.rl_shadow_mode:
                    shadow = dict(signal)
                    shadow.update({
                        "regime_or_scanner_wanted_buy": True,
                        "raw_buy_row_reason": str(row.get("reason", "")),
                        "would_pass_rl_gate": bool(allow),
                        "required_action": required_action,
                        "required_confidence": float(required_confidence),
                    })
                    write_event("rl_shadow_buy_decision", shadow)

                if not allow and self.config.scanner.rl_allow_hold_when_model_missing:
                    reason = str(signal.get("reason", ""))
                    allow = reason in {
                        "model_not_found", "rl_model_missing", "rl_trainer_missing", "rl_inputs_missing",
                        "no_latest_bars", "rl_structure_map_missing", "rl_model_invalid"
                    } and not self.config.scanner.rl_block_buys_when_model_invalid

                keep_mask.append(bool(allow))
                if not allow:
                    print(
                        f"[rl-gate] blocked BUY {symbol}: action={action} confidence={confidence:.2f} "
                        f"required={required_action}>={required_confidence:.2f} "
                        f"reason={signal.get('reason')} mask={signal.get('mask_reason')}"
                    )
                    write_event("rl_gate_blocked_buy", signal)
                else:
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "reason"] = (
                        str(row.get("reason", "")) + f":rl_{action}_{confidence:.2f}"
                    )
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "rl_action"] = action
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "rl_confidence"] = confidence
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "rl_required_action"] = required_action
                    if action in {"ADD_LONG", "PYRAMID_LONG"}:
                        gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "reason"] = (
                            str(gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "reason"]) + ":rl_add_long_confirmed"
                        )

            gated_rebalance = gated_rebalance.loc[keep_mask].copy() if keep_mask else gated_rebalance.iloc[0:0].copy()

        # Add RL protective exits for held positions.
        rl_exit_rows: list[dict[str, Any]] = []
        existing_exit_symbols = set()
        if exits is not None and not exits.empty and "symbol" in exits.columns:
            existing_exit_symbols = set(exits["symbol"].map(safe_symbol))

        for pos in positions:
            symbol = safe_symbol(pos.symbol)
            if not symbol or symbol in existing_exit_symbols:
                continue
            signal = self._predict_rl_signal(symbol, latest_bars, structure_map, positions_by_symbol)
            signals.append(signal)
            action = str(signal.get("action", "HOLD")).upper()
            confidence = to_float(signal.get("confidence"), 0.0)
            if action in {"EXIT_LONG", "EXIT_SHORT", "SELL"} and confidence >= self.config.scanner.rl_min_exit_confidence:
                price = float(pos.current_price or pos.avg_entry_price or 0.0)
                qty = int(max(0, float(pos.qty or 0)))
                if price > 0 and qty > 0:
                    rl_exit_rows.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "delta_shares": qty,
                            "delta_dollars": qty * price,
                            "price": price,
                            "reason": f"rl_timing_exit:{action}:{confidence:.2f}",
                            "target_weight": 0.0,
                            "current_weight": 0.0,
                            "delta_weight": 0.0,
                            "trade_confidence": confidence,
                        }
                    )
                    print(f"[rl-gate] added protective exit {symbol}: action={action} confidence={confidence:.2f}")
                    write_event("rl_gate_added_exit", signal)

        if signals:
            signal_df = pd.DataFrame(signals)
            if RL_SIGNAL_LOG_PATH.exists():
                try:
                    prior = pd.read_csv(RL_SIGNAL_LOG_PATH)
                    signal_df = pd.concat([prior, signal_df], ignore_index=True)
                except Exception:
                    pass
            save_df(signal_df.tail(5000), RL_SIGNAL_LOG_PATH)

        if rl_exit_rows:
            exits = pd.concat([exits if exits is not None else pd.DataFrame(), pd.DataFrame(rl_exit_rows)], ignore_index=True)

        return gated_rebalance, exits if exits is not None else pd.DataFrame()


    def _prepare_morning_inputs_once(self) -> None:
        today = datetime.now(ET).date().isoformat()
        if self._last_morning_refresh_date == today:
            # Avoid rebuilding expensive OpenAI/Yahoo/SerpApi inputs every executor loop.
            self.config.scanner.refresh_llm_sentiment_universe = False
            self.config.scanner.refresh_market_context = False
            self.config.scanner.refresh_premarket_trade_plan = False
            return

        print(f"[morning] preparing daily inputs for {today}")
        run_session_planning_pipeline(self.config.scanner, force=False)

        self._last_morning_refresh_date = today
        self.config.scanner.refresh_llm_sentiment_universe = False
        self.config.scanner.refresh_market_context = False
        self.config.scanner.refresh_premarket_trade_plan = False

    def run_cycle_from_files(self) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        # Do not spend time refreshing/scanning before the market is executable.
        # Exits remain allowed once the market is open; buy-side gates are applied below.
        if not self._market_is_open_or_wait():
            cycle_id = run_id()
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_market_closed", "message": "market is closed"}])
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])

        if self.scanner is not None:
            try:
                self._prepare_morning_inputs_once()

                # Once-per-morning SA assessment. Degrades buy sizing when drawdown,
                # data quality, or execution risk metrics indicate elevated account risk.
                _sa_mode = "NORMAL"
                if _SA_AVAILABLE:
                    _today_sa = datetime.now(ET).date().isoformat()
                    if self._sa_last_assessment_date != _today_sa:
                        try:
                            self._sa_assessment = run_self_assessment()
                            self._sa_last_assessment_date = _today_sa
                        except Exception as _sa_exc:
                            print(f"[self-awareness] assessment failed (non-fatal): {_sa_exc}")
                            self._sa_assessment = {}
                    if self._sa_assessment:
                        _sa_mode = self_assessment_mode(self._sa_assessment)
                        sa_reasons = self._sa_assessment.get("reasons", [])
                        sa_mult = self_assessment_size_multiplier(self._sa_assessment)
                        print(f"[self-awareness] mode={_sa_mode} size_mult={sa_mult:.2f} reasons={sa_reasons}")
                        write_event("self_awareness_assessment", {
                            "mode": _sa_mode, "size_multiplier": sa_mult, "reasons": sa_reasons,
                            "disable_options": self._sa_assessment.get("disable_options"),
                            "disable_pyramiding": self._sa_assessment.get("disable_pyramiding"),
                        })

                # Save config flags before any SA override so they can be restored after build_orders().
                _orig_options_booster = self.config.scanner.options_booster_enabled
                _orig_allow_pyramiding = getattr(self.config.scanner, "allow_pyramiding", True)
                if _SA_AVAILABLE and self._sa_assessment:
                    if self._sa_assessment.get("disable_options"):
                        self.config.scanner.options_booster_enabled = False
                        print("[self-awareness] options booster disabled by SA assessment")
                    if self._sa_assessment.get("disable_pyramiding") and hasattr(self.config.scanner, "allow_pyramiding"):
                        self.config.scanner.allow_pyramiding = False
                        print("[self-awareness] pyramiding disabled by SA assessment")

                acct = self.broker.get_account_snapshot()
                equity = max(float(acct.equity), 1.0)

                market_context: dict[str, Any] = {}
                premarket_plan: dict[str, Any] = {}

                if self.config.scanner.market_context_enabled:
                    market_context = load_market_context(refresh=self.config.scanner.refresh_market_context)
                    blocked, reason = market_context_blocks_new_buys(
                        market_context,
                        allow_cautious=self.config.scanner.allow_cautious_market_buys,
                    )
                    print(f"[market-context] {reason}")
                    write_event("market_context_gate", {"blocked": blocked, "reason": reason})
                else:
                    blocked, reason = False, "market_context_disabled"

                if self.config.scanner.premarket_plan_enabled:
                    premarket_plan = load_premarket_trade_plan(refresh=self.config.scanner.refresh_premarket_trade_plan)

                # Run active-position monitor first so exits are generated before any new buys.
                monitor_exits = self.scanner.monitor_active_positions(equity)

                regime_plan = load_regime_position_plan() if self.config.scanner.use_regime_position_plan else pd.DataFrame()
                if self.config.scanner.prefer_regime_position_plan and not regime_plan.empty:
                    print(f"[regime-plan] using {REGIME_POSITION_PLAN_PATH} rows={len(regime_plan)}")
                    rebalance, scanner_exits = self._regime_plan_to_order_frames(regime_plan, equity)
                    _scores = pd.DataFrame()
                else:
                    rebalance, scanner_exits, _scores = self.scanner.build_orders()

                # Restore config flags that SA may have temporarily suppressed.
                self.config.scanner.options_booster_enabled = _orig_options_booster
                if hasattr(self.config.scanner, "allow_pyramiding"):
                    self.config.scanner.allow_pyramiding = _orig_allow_pyramiding

                exits = pd.concat([monitor_exits, scanner_exits], ignore_index=True)

                # Market-context risk gate is BUY-ONLY. Never block exits.
                if self.config.scanner.halt_new_buys_on_market_context and blocked and not rebalance.empty:
                    print(f"[market-context] removing {len(rebalance)} buy candidate(s): {reason}")
                    write_event("market_context_removed_buys", {"rows": len(rebalance), "reason": reason})
                    rebalance = rebalance.iloc[0:0].copy()

                # Premarket plan gate is also BUY-ONLY. It optionally restricts buys to valid premarket setups.
                valid_pm_symbols = valid_symbols_from_premarket_plan(premarket_plan)
                if self.config.scanner.premarket_plan_enabled and self.config.scanner.prefer_premarket_valid_symbols and valid_pm_symbols and not rebalance.empty:
                    before = len(rebalance)
                    _pm_removed_syms = sorted(rebalance.loc[~rebalance["symbol"].map(normalize_symbol).isin(valid_pm_symbols), "symbol"].map(normalize_symbol).unique().tolist()) if "symbol" in rebalance.columns else []
                    rebalance = rebalance[rebalance["symbol"].map(normalize_symbol).isin(valid_pm_symbols)].copy()
                    removed = before - len(rebalance)
                    if removed:
                        print(f"[premarket-plan] kept {len(rebalance)} valid premarket setup buy(s); removed {removed}: {_pm_removed_syms}")
                        write_event("premarket_plan_filtered_buys", {"kept": len(rebalance), "removed": removed, "removed_symbols": _pm_removed_syms, "valid_symbols": sorted(valid_pm_symbols)})
                elif self.config.scanner.premarket_plan_enabled and self.config.scanner.require_valid_premarket_plan_for_buys and not valid_pm_symbols and not rebalance.empty:
                    print("[premarket-plan] no valid premarket setups; removing all buy candidates")
                    write_event("premarket_plan_removed_all_buys", {"rows": len(rebalance), "reason": "no_valid_premarket_symbols"})
                    rebalance = rebalance.iloc[0:0].copy()

                # RL timing gate is applied after scanner/regime and buy-only context filters.
                # Exits remain allowed, and the RL layer may add additional protective exits.
                rebalance, exits = self._apply_rl_timing_gate(rebalance, exits)

                # SA HALT_NEW_BUYS: remove all buy candidates after RL gate. Exits survive.
                if _sa_mode == "HALT_NEW_BUYS" and not rebalance.empty:
                    print(f"[sa-gate] HALT_NEW_BUYS: clearing {len(rebalance)} rebalance candidate(s) | exits unaffected")
                    write_event("sa_halt_new_buys_rebalance", {"removed": len(rebalance), "reasons": self._sa_assessment.get("reasons", [])})
                    rebalance = rebalance.iloc[0:0].copy()

                if not monitor_exits.empty:
                    exited_today = sorted(set(monitor_exits["symbol"].map(safe_symbol)))
                    print(f"[monitor] generated protective exits; same-day re-entry blocked for {exited_today}")

                if not rebalance.empty or not exits.empty:
                    return self.run_once_from_frames(rebalance, exits)
            except Exception as exc:
                print(f"[scanner] cycle failed; falling back to existing order files: {exc}")
                write_event("scanner_cycle_error", {"error": str(exc)})

        self._run_portfolio_rebalancer()
        print(f"[executor] using rebalance_orders={self.config.rebalance_orders_path}")
        print(f"[executor] using exit_orders={self.config.exit_orders_path if self.config.exit_orders_path.exists() else 'NONE'}")
        rebalance, exits = self._load_orders()
        return self.run_once_from_frames(rebalance, exits)

    def _print_heartbeat(self) -> None:
        now_ts = time.time()
        if now_ts - self.last_heartbeat_time < HEARTBEAT_SECONDS:
            return
        acct = self.broker.get_account_snapshot()
        gross = self._gross_exposure(acct.equity)
        print(
            f"[executor] heartbeat | equity=${acct.equity:,.2f} | "
            f"cash=${acct.cash:,.2f} | buying_power=${acct.buying_power:,.2f} | gross={gross:.2%}"
        )
        self.last_heartbeat_time = now_ts


    def _start_spx_price_action_layer(self) -> None:
        if not self.config.scanner or not self.config.scanner.spx_price_action_layer_enabled:
            return
        if getattr(self, "_spx_price_action_thread_started", False):
            return

        self._spx_price_action_thread_started = True
        thread = threading.Thread(
            target=self._spx_price_action_loop,
            name="spx-price-action-layer",
            daemon=True,
        )
        thread.start()
        print(
            "[spx-layer] internal layer started "
            f"symbol={self.config.scanner.spx_price_action_trade_symbol} "
            f"scan_seconds={self.config.scanner.spx_price_action_scan_seconds} "
            f"execute={self.config.scanner.spx_price_action_execute_orders}"
        )

    def spx_write_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        write_event(event, payload or {})

    def spx_price_action_cycle(self) -> None:
        try:
            from spx_price_action_layer import run_once
            run_once()
        except ImportError as exc:
            print(f"[spx-layer] module import failed: {exc}")
            write_event("spx_layer_import_failed", {"error": str(exc)})
        except Exception as exc:
            print(f"[spx-layer] cycle error: {exc}")
            write_event("spx_cycle_error", {"error": str(exc)})

    def _spx_price_action_loop(self) -> None:
        while True:
            try:
                if self.config.scanner and self.config.scanner.spx_price_action_layer_enabled:
                    self.spx_price_action_cycle()
            except Exception as exc:
                print(f"[spx-layer] cycle error: {exc}")
                self.spx_write_event("spx_cycle_error", {"error": str(exc)})
            time.sleep(max(1, int(self.config.scanner.spx_price_action_scan_seconds if self.config.scanner else 10)))

    def run_spx_price_action_once(self) -> Optional[dict[str, Any]]:
        if not self.config.scanner:
            return None
        self.spx_price_action_cycle()
        return read_json_file(SPX_PRICE_ACTION_SIGNAL_PATH, {})


    def _start_sr_rl_agent(self) -> None:
        if not self.config.scanner or not self.config.scanner.sr_rl_agent_enabled:
            return
        if getattr(self, "_sr_rl_thread_started", False):
            return

        self._sr_rl_thread_started = True
        thread = threading.Thread(
            target=self._sr_rl_agent_loop,
            name="sr-rl-agent",
            daemon=True,
        )
        thread.start()
        print(
            "[sr-rl] agent watcher started "
            f"live={self.config.scanner.sr_rl_live_enabled} "
            f"execute_now={self.config.scanner.sr_rl_execute_now} "
            f"scan_seconds={self.config.scanner.sr_rl_scan_seconds}"
        )

    def _sr_rl_agent_loop(self) -> None:
        script_name = self.config.scanner.sr_rl_agent_script if self.config.scanner else "sr_rl_agent.py"
        script_path = PROJECT_ROOT / script_name
        if not script_path.exists():
            alt = PROJECT_ROOT / "sr_rl_agent.py"
            script_path = alt if alt.exists() else script_path

        if not script_path.exists():
            print(f"[sr-rl] script missing: {script_path}")
            sr_write_event("script_missing", {"path": str(script_path)})
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["RUN_DATA_DATE"] = DATA_TAG
        env["RL_DATA_DATE"] = DATA_TAG
        env["SR_RL_SCAN_SECONDS"] = str(self.config.scanner.sr_rl_scan_seconds)
        env["SR_RL_EXECUTE"] = "1" if self.config.scanner.sr_rl_execute_now else "0"

        if self.config.scanner.sr_rl_train_on_start:
            train_cmd = [sys.executable, str(script_path), "--train"]
            print(f"[sr-rl] training agent: {' '.join(train_cmd)}")
            sr_write_event("train_started", {"cmd": train_cmd, "model_path": str(SR_RL_MODEL_PATH)})
            proc = subprocess.Popen(
                train_cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line.rstrip())
            rc = proc.wait()
            sr_write_event("train_completed", {"returncode": rc})
            if rc != 0 and self.config.scanner.sr_rl_require_model:
                print(f"[sr-rl] train failed returncode={rc}; live disabled")
                return

        if self.config.scanner.sr_rl_require_model and not SR_RL_MODEL_PATH.exists():
            print(f"[sr-rl] model missing and required: {SR_RL_MODEL_PATH}")
            sr_write_event("model_missing_required", {"model_path": str(SR_RL_MODEL_PATH)})
            return

        if not self.config.scanner.sr_rl_live_enabled:
            return

        # If model doesn't exist and we haven't trained yet, attempt one training run now
        # so the live loop doesn't immediately crash-restart.
        if not SR_RL_MODEL_PATH.exists() and not self.config.scanner.sr_rl_train_on_start:
            print(f"[sr-rl] model missing: {SR_RL_MODEL_PATH}; attempting auto-train before live")
            sr_write_event("auto_train_started", {"model_path": str(SR_RL_MODEL_PATH)})
            train_cmd = [sys.executable, str(script_path), "--train"]
            try:
                proc = subprocess.Popen(
                    train_cmd, cwd=str(PROJECT_ROOT), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line.rstrip())
                train_rc = proc.wait()
                sr_write_event("auto_train_completed", {"returncode": train_rc, "model_exists": SR_RL_MODEL_PATH.exists()})
            except Exception as train_exc:
                print(f"[sr-rl] auto-train failed: {train_exc}")
                train_rc = -1
            if not SR_RL_MODEL_PATH.exists():
                print(f"[sr-rl] auto-train did not produce model (rc={train_rc}); live mode skipped")
                return

        cmd = [sys.executable, str(script_path), "--live"]
        # Keep sr_rl_agent long-only signal-only by default. run.py executes
        # merged rows after its own risk, direction, PDT, kill-switch and cash gates.
        if self.config.scanner.sr_rl_execute_now:
            cmd.append("--execute")

        while True:
            try:
                print(f"[sr-rl] launching: {' '.join(cmd)}")
                sr_write_event("live_started", {"cmd": cmd, "signals_path": str(SR_RL_SIGNALS_PATH)})
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line.rstrip())
                rc = proc.wait()
                sr_write_event("live_exited", {"returncode": rc})
                if rc != 0 and not SR_RL_MODEL_PATH.exists():
                    print(f"[sr-rl] model missing after exit (rc={rc}); waiting 300s before retry")
                    time.sleep(300)
                else:
                    print(f"[sr-rl] exited returncode={rc}; restarting in 20s")
                    time.sleep(20)
            except Exception as exc:
                print(f"[sr-rl] loop error: {exc}")
                sr_write_event("live_error", {"error": str(exc)})
                time.sleep(20)

    def _start_stream_confirmed_movement_gate(self) -> None:
        if not self.config.scanner or not self.config.scanner.stream_confirmed_movement_enabled:
            return
        if getattr(self, "_stream_confirm_thread_started", False):
            return

        self._stream_confirm_thread_started = True
        thread = threading.Thread(
            target=self._stream_confirmed_movement_loop,
            name="stream-confirmed-movement-gate",
            daemon=True,
        )
        thread.start()
        print(
            "[stream-confirm] internal watcher started "
            f"polling={self.config.scanner.stream_confirmed_use_polling} "
            f"execute_now={self.config.scanner.stream_confirmed_execute_now} "
            f"reload_seconds={self.config.scanner.stream_confirmed_reload_seconds}"
        )

    def _stream_confirmed_movement_loop(self) -> None:
        """
        Runs the stream-confirmation layer as a subprocess.

        The watcher monitors the full day's symbol universe, writes:
            - stream_confirmed_trade_plan.csv/json
            - stream_confirmed_rebalance_orders.csv

        run.py merges those confirmed orders into each execution cycle.
        """
        script_name = self.config.scanner.stream_confirmed_movement_script if self.config.scanner else "stream_confirmed_movement_gate.py"
        script_path = PROJECT_ROOT / script_name

        if not script_path.exists():
            # fallback to the specific generated filename if user has not renamed it
            alt = PROJECT_ROOT / "stream_confirmed_movement_gate_trade_plan_trailing.py"
            script_path = alt if alt.exists() else script_path

        if not script_path.exists():
            print(f"[stream-confirm] script missing: {script_path}")
            write_event("stream_confirm_script_missing", {"path": str(script_path)})
            return

        env = os.environ.copy()
        env["RUN_DATA_DATE"] = DATA_TAG
        env["STREAM_CONFIRM_DATA_DATE"] = DATA_TAG
        env["STREAM_CONFIRM_EXECUTE"] = "1" if self.config.scanner.stream_confirmed_execute_now else "0"
        env["STREAM_CONFIRM_USE_POLLING"] = "1" if self.config.scanner.stream_confirmed_use_polling else "0"
        env["STREAM_CONFIRM_RELOAD_SECONDS"] = str(self.config.scanner.stream_confirmed_reload_seconds)
        env["STREAM_CONFIRM_TRAILING_STOP_ENABLED"] = "1" if self.config.scanner.stream_confirmed_trailing_stop_enabled else "0"
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [sys.executable, str(script_path)]
        if self.config.scanner.stream_confirmed_use_polling:
            cmd.append("--poll")

        while True:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                write_event("stream_confirm_watcher_started", {"cmd": cmd, "path": str(script_path)})
                if proc.stdout is not None:
                    for line in proc.stdout:
                        print(line.rstrip())

                returncode = proc.wait()
                write_event("stream_confirm_watcher_exited", {"returncode": returncode})
                print(f"[stream-confirm] watcher exited returncode={returncode}; restarting in 10s")
                time.sleep(10)
            except Exception as exc:
                print(f"[stream-confirm] watcher error: {exc}; restarting in 10s")
                write_event("stream_confirm_watcher_error", {"error": str(exc)})
                time.sleep(10)

    def run(self) -> None:
        print("[executor] 13K account executor starting")
        print(f"[executor] journal={JOURNAL_PATH}")
        print(f"[executor] equity_curve={EQUITY_CURVE_PATH}")
        print(f"[executor] paper={self.config.paper}")
        if self.config.scanner:
            print(f"[executor] universe_source={'LLM_SENTIMENT' if self.config.scanner.use_llm_sentiment_universe else 'SCANNER_DEFAULT'}")
            print(f"[executor] options_booster_enabled={self.config.scanner.options_booster_enabled}")
            print(f"[executor] options_booster_threshold={self.config.scanner.options_booster_confidence_threshold:.0%}")
            print(f"[executor] options_data_source={self.config.scanner.options_data_source}")
            print(f"[executor] options_alpaca_chain_preflight={self.config.scanner.options_alpaca_chain_preflight_enabled}")
            print(f"[executor] market_context_enabled={self.config.scanner.market_context_enabled}")
            print(f"[executor] premarket_plan_enabled={self.config.scanner.premarket_plan_enabled}")
            print(f"[executor] session_planning_enabled={self.config.scanner.session_planning_enabled}")
            print(f"[executor] session_planning_start={self.config.scanner.session_planning_start_hour_et:02d}:{self.config.scanner.session_planning_start_minute_et:02d} ET")
            print(f"[executor] regime_plan_enabled={self.config.scanner.use_regime_position_plan}")
            print(f"[executor] rl_timing_gate_enabled={self.config.scanner.use_rl_timing_gate}")
            print(f"[executor] rl_model={RL_MODEL_PATH}")
            print(f"[executor] rl_metrics={RL_METRICS_PATH}")
            print(f"[executor] rl_add_long_gate={self.config.scanner.rl_enable_add_long_gate}")
            print(f"[executor] rl_training_from_run_py={self.config.scanner.allow_rl_training_from_run_py}")
            print(f"[executor] rl_shadow_mode={self.config.scanner.rl_shadow_mode}")
            print(f"[executor] live_experience_logging={self.config.scanner.live_experience_logging_enabled}")
            print(f"[executor] live_experience_jsonl={LIVE_EXPERIENCE_JSONL_PATH}")
            print(f"[executor] live_experience_csv={LIVE_EXPERIENCE_CSV_PATH}")
            print(f"[executor] premarket_data_collector_enabled={self.config.scanner.premarket_data_collector_enabled}")
            print(f"[executor] premarket_data_collector_path={PREMARKET_DATA_COLLECTOR_PATH}")
            print(f"[executor] dated_data_snapshots_enabled={self.config.scanner.dated_data_snapshots_enabled}")
            print(f"[executor] data_validation_state={DATA_VALIDATION_STATE_PATH}")
            print(f"[executor] spx_price_action_layer_enabled={self.config.scanner.spx_price_action_layer_enabled}")
            print(f"[executor] spx_price_action_trade_symbol={self.config.scanner.spx_price_action_trade_symbol}")
            print(f"[executor] spx_price_action_execute_orders={self.config.scanner.spx_price_action_execute_orders}")
            print(f"[executor] spx_price_action_signal={SPX_PRICE_ACTION_SIGNAL_PATH}")
            print(f"[executor] stream_confirmed_movement_enabled={self.config.scanner.stream_confirmed_movement_enabled}")
            print(f"[executor] stream_confirmed_orders={STREAM_CONFIRMED_ORDERS_PATH}")
            print(f"[executor] stream_confirmed_trade_plan_csv={STREAM_CONFIRMED_TRADE_PLAN_CSV_PATH}")
            print(f"[executor] sr_rl_agent_enabled={self.config.scanner.sr_rl_agent_enabled}")
            print(f"[executor] sr_rl_signals={SR_RL_SIGNALS_PATH}")
            print(f"[executor] sr_rl_model={SR_RL_MODEL_PATH}")
        try:
            acct = self.broker.get_account_snapshot()
            profile = self._risk_profile(acct.equity)
            print(f"[executor] account_band={profile.band}")
            print(f"[executor] equity=${acct.equity:,.2f}")
            print(f"[executor] cash_buffer_pct={profile.cash_buffer_pct:.2%}")
            print(f"[executor] min_cash_buffer=${profile.min_cash_buffer:,.2f}")
            print(f"[executor] max_gross_exposure={profile.max_gross_exposure:.2%}")
            print(f"[executor] max_position_weight={profile.max_position_weight:.2%}")
            print(f"[executor] max_order_dollars=${profile.max_order_dollars:,.2f}")
            print(f"[executor] max_daily_loss=${profile.max_daily_loss_dollars:,.2f}/{profile.max_daily_loss_pct:.2%}")
            print(f"[executor] dynamic_kill_floor=${profile.kill_switch_equity_floor:,.2f}")
        except Exception as exc:
            print(f"[executor] risk profile unavailable at startup: {exc}")

        # ------------------------------------------------------------------
        # IMPORTANT: run the continuous 4:00-9:30 ET premarket collector BEFORE
        # the market-open wait. Otherwise the executor can sit at the clock
        # loop while premarket volume, structure maps, intraday bars, and
        # options-flow files go stale.
        # ------------------------------------------------------------------
        if self.config.scanner and self.config.scanner.session_planning_enabled:
            today_et = datetime.now(ET).date().isoformat()
            print(
                f"[session-prep] startup gate active | "
                f"target={self.config.scanner.session_planning_start_hour_et:02d}:"
                f"{self.config.scanner.session_planning_start_minute_et:02d} ET | "
                f"today={today_et}"
            )

            try:
                collector_result = run_premarket_data_collector(self.config.scanner)
                print(
                    "[premarket-collector] startup result "
                    f"started={collector_result.get('started')} "
                    f"ok={collector_result.get('ok')} "
                    f"reason={collector_result.get('reason')}"
                )
            except Exception as exc:
                print(f"[premarket-collector] startup collector failed; continuing with one-shot fallback: {exc}")
                write_event("premarket_collector_startup_error", {"error": str(exc)})

            try:
                # Final handoff/validation pass after continuous collection.
                # This builds/refreshes the regime position plan and validates
                # the saved RL model, but does not retrain RL unless explicitly
                # enabled in config.
                run_session_planning_pipeline(
                    self.config.scanner,
                    force=bool(self.config.scanner.session_planning_force_each_startup or collector_result.get("started")),
                )
                validation = snapshot_and_validate_planning_files(self.config.scanner, stage="after_session_planning")
                if self.config.scanner.require_today_data_for_market_open and not validation.get("all_required_fresh"):
                    print("[data-validation] required files are stale/missing; market-open execution will still use risk fallbacks")
                    write_event("required_planning_data_stale", validation)

                # Prevent the first market-open cycle from immediately running
                # the expensive prep stack again.
                self._last_morning_refresh_date = today_et
                self.config.scanner.refresh_llm_sentiment_universe = False
                self.config.scanner.refresh_market_context = False
                self.config.scanner.refresh_premarket_trade_plan = False
                print(f"[session-prep] startup prep complete for {today_et}; entering execution loop")
            except Exception as exc:
                print(f"[session-prep] startup prep failed; executor will continue with fallbacks: {exc}")
                write_event("session_planning_startup_error", {"error": str(exc)})

        self._start_spx_price_action_layer()
        self._start_sr_rl_agent()
        self._start_stream_confirmed_movement_gate()

        if self.config.poll_once:
            self.run_cycle_from_files()
            self.run_spx_price_action_once()
            return

        while True:
            try:
                self.run_cycle_from_files()
                self._print_heartbeat()
            except KeyboardInterrupt:
                print("[executor] interrupted by user")
                raise
            except Exception as exc:
                print(f"[executor] cycle error: {exc}")
                write_event("cycle_error", {"error": str(exc)})
            time.sleep(max(1, int(self.config.poll_seconds)))


# ======================================================================# Backward-compatible module functions
# ======================================================================
def get_trading_client(config: Optional[ExecutorConfig] = None) -> TradingClient:
    config = config or ExecutorConfig()
    return AlpacaBroker(ENV_PATH, paper=config.paper).trading


def get_account_snapshot(client: TradingClient) -> AccountSnapshot:
    try:
        acct = client.get_account()
    except Exception as exc:
        if is_auth_error(exc):
            raise RuntimeError(alpaca_auth_help_message(True)) from exc
        raise
    return AccountSnapshot(
        buying_power=to_float(getattr(acct, "buying_power", 0.0)),
        equity=to_float(getattr(acct, "equity", 0.0)),
        cash=to_float(getattr(acct, "cash", 0.0)),
        portfolio_value=to_float(getattr(acct, "portfolio_value", 0.0)),
    )


def get_positions_df(client: TradingClient) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        positions = client.get_all_positions()
    except Exception:
        positions = []

    for p in positions:
        rows.append(
            {
                "symbol": safe_symbol(getattr(p, "symbol", "")),
                "qty": to_float(getattr(p, "qty", 0.0)),
                "avg_entry_price": to_float(getattr(p, "avg_entry_price", 0.0)),
                "current_price": to_float(getattr(p, "current_price", 0.0)),
                "market_value": to_float(getattr(p, "market_value", 0.0)),
                "cost_basis": to_float(getattr(p, "cost_basis", 0.0)),
                "unrealized_pl": to_float(getattr(p, "unrealized_pl", 0.0)),
                "unrealized_plpc": to_float(getattr(p, "unrealized_plpc", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def run_execution(
    rebalance_orders: pd.DataFrame,
    exit_orders: Optional[pd.DataFrame] = None,
    config: Optional[ExecutorConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
    config = config or ExecutorConfig(poll_once=True)
    config.poll_once = True
    executor = ProductionOrderExecutor(config)
    return executor.run_once_from_frames(rebalance_orders, exit_orders if exit_orders is not None else pd.DataFrame())


def save_execution_results(
    order_plan: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Optional[Path | str] = None,
    run_stamp: Optional[str] = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir) if output_dir is not None else LOG_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = run_stamp or run_id()
    order_plan_path = output_path / f"order_plan_{stamp}.csv"
    results_path = output_path / f"order_results_{stamp}.csv"
    save_df(order_plan if order_plan is not None else pd.DataFrame(), order_plan_path)
    save_df(results if results is not None else pd.DataFrame(), results_path)
    return order_plan_path, results_path


# ======================================================================# Zero-argument startup
# ======================================================================
def build_default_config() -> ExecutorConfig:
    """
    Default production-style paper configuration for a roughly $13K account.

    Run with:
        python run.py

    No command-line arguments are required.
    """
    scanner_config = ScannerConfig(
        enabled=True,
        universe=None,
        use_sp500_fallback=False,
        max_universe_symbols=500,
        top_n=30,
        min_setup_score=0.58,
        min_buy_confidence=0.54,
        min_avg_dollar_volume=5_000_000.0,
        use_llm_sentiment_universe=True,
        refresh_llm_sentiment_universe=True,
        min_llm_universe_confidence=0.0,
        max_plan_age_minutes=MAX_PLAN_AGE_MINUTES,
        require_fresh_premarket_plan_for_buys=False,
        require_structure_long_allowed_for_buys=False,
        safe_mode_on_planner_failure=SAFE_MODE_ON_PLANNER_FAILURE,
        safe_mode_buy_size_multiplier=SAFE_MODE_BUY_SIZE_MULTIPLIER,
        cautious_market_buy_size_multiplier=CAUTIOUS_MARKET_BUY_SIZE_MULTIPLIER,
        cautious_market_confidence_addon=CAUTIOUS_MARKET_CONFIDENCE_ADDON,
        bearish_market_disable_new_longs=BEARISH_MARKET_DISABLE_NEW_LONGS,
        bearish_market_min_rl_confidence=BEARISH_MARKET_MIN_RL_CONFIDENCE,
        use_tradingview_screener=USE_TRADINGVIEW_SCREENER,
        tradingview_require_rvol=TRADINGVIEW_REQUIRE_RVOL,
        tradingview_max_planner_symbols=TRADINGVIEW_MAX_PLANNER_SYMBOLS,
        options_booster_enabled=True,
        options_booster_confidence_threshold=0.75,
        opportunity_scan_enabled=True,
        min_daily_gain_pct=0.005,
        min_opportunity_score=0.60,
        min_breakout_confirmation_score=0.55,
        require_positive_momentum_or_breakout=True,
        base_target_weight=0.03,
        max_target_weight=0.10,
        broad_market_scan=False,
        stop_loss_pct=-0.025,
        profit_lock_trigger_pct=0.04,
        profit_giveback_pct=0.015,
        session_planning_enabled=True,
        session_planning_start_hour_et=4,
        session_planning_start_minute_et=0,
        run_market_context_builder=True,
        run_premarket_trade_planner=True,
        run_production_regime_sizing_model=True,
        run_rl_intraday_structure_trainer=False,
        allow_rl_training_from_run_py=False,
        use_regime_position_plan=True,
        prefer_regime_position_plan=True,
        use_rl_timing_gate=True,
        rl_min_enter_long_confidence=0.65,
        rl_min_exit_confidence=0.55,
        rl_min_enter_short_confidence=0.70,
    )

    return ExecutorConfig(
        paper=True,
        live=False,
        poll_once=False,
        poll_seconds=POLL_SECONDS,
        rebalance_orders_path=DEFAULT_REBALANCE_ORDERS,
        exit_orders_path=DEFAULT_EXIT_ORDERS,
        require_market_open=True,
        wait_for_market_open=True,
        allow_closed_market=False,
        cancel_open_orders_first=False,
        sell_first=True,
        cash_buffer_pct=DEFAULT_CASH_BUFFER_PCT,
        min_cash_buffer=MIN_CASH_BUFFER,
        min_buying_power=MIN_BUYING_POWER,
        min_notional_to_trade=MIN_NOTIONAL_TO_TRADE,
        min_weight_delta=MIN_WEIGHT_DELTA,
        max_gross_exposure=MAX_GROSS_EXPOSURE,
        max_position_weight=MAX_POSITION_WEIGHT,
        max_single_order_notional_pct=MAX_SINGLE_ORDER_NOTIONAL_PCT,
        max_order_dollars=MAX_ORDER_DOLLARS,
        max_single_order_qty=MAX_SINGLE_ORDER_QTY,
        cooldown_seconds=COOLDOWN_SECONDS,
        enable_kill_switch=True,
        daily_drawdown_limit=DAILY_DRAWDOWN_LIMIT,
        max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
        max_daily_loss_dollars=MAX_DAILY_LOSS_DOLLARS,
        kill_switch_equity_floor=KILL_SWITCH_EQUITY_FLOOR,
        order_wait_seconds=ORDER_WAIT_SECONDS,
        time_in_force=TimeInForce.DAY,
        order_tag_prefix="qq_13k",
        scanner=scanner_config,
    )


def main() -> None:
    try:
        maybe_run_pso_lstm_optimization()
    except Exception as exc:
        print(f"[pso-lstm] startup skipped: {exc}")

    try:
        maybe_refresh_forecast_ensemble_layer()
    except Exception as exc:
        print(f"[forecast] startup skipped: {exc}")

    try:
        maybe_refresh_evolution_strategy_layer()
    except Exception as exc:
        print(f"[es-layer] startup skipped: {exc}")

    try:
        start_spx_price_action_layer_background()
    except Exception as exc:
        print(f"[spx-layer] startup skipped: {exc}")

    config = build_default_config()
    threading.Thread(
    target=launch_intraday_dataset_builder,
    daemon=True,
).start()
    executor = ProductionOrderExecutor(config)
    executor.run()


if __name__ == "__main__":
    main()

"""ngrok http 5000"""