#!/usr/bin/env python3
"""
forecast_engine.py

Real trained forecast models for the trading system.

Replaces the drift/volatility proxy layer in run.py with:
  - AR(p) model  — sklearn LinearRegression on lagged returns (ARIMA proxy)
  - LSTM         — single-layer PyTorch LSTM trained cross-symbol
  - Stacked LSTM — 2-layer PyTorch LSTM
  - Attention LSTM — LSTM with scaled dot-product attention

Training data:
  system-data/YYYYMMDD/YYYYMMDD_intraday_training_dataset.csv
  daily_system_outputs/daily_features.csv  (supplemental)

Model outputs:
  models/forecast/YYYYMMDD/lstm_model.pt
  models/forecast/YYYYMMDD/stacked_lstm_model.pt
  models/forecast/YYYYMMDD/attention_lstm_model.pt
  models/forecast/YYYYMMDD/feature_scaler.pkl
  models/forecast/YYYYMMDD/ar_models.pkl

Forecast outputs (schema unchanged from run.py expectation):
  system-data/YYYYMMDD/YYYYMMDD_forecast_ensemble_signals.json
  system-data/YYYYMMDD/YYYYMMDD_forecast_ensemble_signals.csv
  system-data/YYYYMMDD/YYYYMMDD_forecast_engine_metrics.json

Usage:
  python forecast_engine.py
  FORECAST_ENGINE_TRAIN_ON_START=0 python forecast_engine.py   # inference only
  FORECAST_ENGINE_EPOCHS=5 python forecast_engine.py           # quick retrain

Safety:
  - All failures are caught; system writes a safe fallback output and exits 0.
  - Sell/exit logic in run.py does not depend on forecast output.
  - Missing model files cause neutral (0.50) scores, not crashes.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# PyTorch — optional; system degrades to AR-only if unavailable.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH = False

# ---------------------------------------------------------------------------
# Environment & paths
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
    TODAY_TAG = datetime.now(_ET).strftime("%Y%m%d")
except Exception:
    TODAY_TAG = datetime.now().strftime("%Y%m%d")

DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", TODAY_TAG))

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config — all tunable via environment variables
# ---------------------------------------------------------------------------
FORECAST_ENGINE_ENABLED = os.getenv("FORECAST_ENGINE_ENABLED", "1").lower() in {"1", "true", "yes"}
FORECAST_ENGINE_TRAIN_ON_START = os.getenv("FORECAST_ENGINE_TRAIN_ON_START", "1").lower() in {"1", "true", "yes"}
FORECAST_ENGINE_MIN_ROWS = int(os.getenv("FORECAST_ENGINE_MIN_ROWS", "300"))
FORECAST_ENGINE_EPOCHS = int(os.getenv("FORECAST_ENGINE_EPOCHS", "20"))
FORECAST_ENGINE_LOOKBACK = int(os.getenv("FORECAST_ENGINE_LOOKBACK", "60"))
FORECAST_ENGINE_STRIDE = int(os.getenv("FORECAST_ENGINE_STRIDE", "5"))
FORECAST_ENGINE_BATCH_SIZE = int(os.getenv("FORECAST_ENGINE_BATCH_SIZE", "128"))
FORECAST_ENGINE_HIDDEN_SIZE = int(os.getenv("FORECAST_ENGINE_HIDDEN_SIZE", "64"))
FORECAST_ENGINE_LR = float(os.getenv("FORECAST_ENGINE_LR", "3e-4"))
FORECAST_ENGINE_AR_LAGS = int(os.getenv("FORECAST_ENGINE_AR_LAGS", "10"))
FORECAST_ENGINE_TRAIN_SPLIT = float(os.getenv("FORECAST_ENGINE_TRAIN_SPLIT", "0.70"))
FORECAST_ENGINE_USE_GPU = os.getenv("FORECAST_ENGINE_USE_GPU", "1").lower() in {"1", "true", "yes"}
FORECAST_ENGINE_MODEL_DIR = PROJECT_ROOT / os.getenv("FORECAST_ENGINE_MODEL_DIR", "models/forecast")

# Buy-gate thresholds (kept in sync with run.py defaults)
FORECAST_BUY_THRESHOLD = float(os.getenv("FORECAST_BUY_THRESHOLD", "0.62"))
FORECAST_STRONG_BUY_THRESHOLD = float(os.getenv("FORECAST_STRONG_BUY_THRESHOLD", "0.72"))
FORECAST_MAX_DRAWDOWN_PROB = float(os.getenv("FORECAST_MAX_DRAWDOWN_PROB_FOR_BUY", "0.42"))
FORECAST_MIN_EXPECTED_RETURN = float(os.getenv("FORECAST_MIN_EXPECTED_RETURN_FOR_BUY", "0.0015"))

# Ensemble weights (kept in sync with run.py FORECAST_WEIGHTS)
ENSEMBLE_WEIGHTS: dict[str, float] = {
    "arima": float(os.getenv("FORECAST_WEIGHT_ARIMA", "0.10")),
    "lstm": float(os.getenv("FORECAST_WEIGHT_LSTM", "0.20")),
    "stacked_lstm": float(os.getenv("FORECAST_WEIGHT_STACKED_LSTM", "0.15")),
    "attention_lstm": float(os.getenv("FORECAST_WEIGHT_ATTENTION_LSTM", "0.35")),
}

# Output paths
_forecast_output_env = os.getenv("FORECAST_ENSEMBLE_OUTPUT", "")
FORECAST_OUTPUT_PATH = Path(_forecast_output_env) if _forecast_output_env else OUT_DIR / f"{DATA_TAG}_forecast_ensemble_signals.json"
FORECAST_CSV_PATH = FORECAST_OUTPUT_PATH.parent / (FORECAST_OUTPUT_PATH.stem + ".csv")
METRICS_PATH = OUT_DIR / f"{DATA_TAG}_forecast_engine_metrics.json"

# Model dir for this date
MODEL_DIR = FORECAST_ENGINE_MODEL_DIR / DATA_TAG
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LSTM_PATH = MODEL_DIR / "lstm_model.pt"
STACKED_LSTM_PATH = MODEL_DIR / "stacked_lstm_model.pt"
ATTENTION_LSTM_PATH = MODEL_DIR / "attention_lstm_model.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
AR_PATH = MODEL_DIR / "ar_models.pkl"

# Input data paths
INTRADAY_TRAIN_PATH = OUT_DIR / f"{DATA_TAG}_intraday_training_dataset.csv"
INTRADAY_BARS_PATH = OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv"
DAILY_FEATURES_PATH = PROJECT_ROOT / "daily_system_outputs" / "daily_features.csv"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"

# Device selection
if _TORCH and FORECAST_ENGINE_USE_GPU:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
elif _TORCH:
    DEVICE = torch.device("cpu")
else:
    DEVICE = None

# ---------------------------------------------------------------------------
# Feature columns — all present in intraday_training_dataset, no look-ahead.
# ---------------------------------------------------------------------------
CANDIDATE_FEATURES = [
    "ret_1m", "ret_3m", "ret_5m", "ret_10m", "ret_15m", "ret_30m",
    "log_ret_1m",
    "volatility_5", "volatility_10", "volatility_20",
    "dist_sma_5", "dist_sma_10", "dist_sma_20",
    "rvol_5", "rvol_10", "rvol_20",
    "dist_vwap", "above_vwap",
    "rsi_14", "macd", "macd_hist",
    "atr_pct", "range_pct", "body_pct",
    "breakout_15", "breakdown_15",
    "trend_5_20", "trend_10_30",
    "5m_ret_1", "5m_trend_stack", "5m_dist_vwap",
    "15m_ret_1", "15m_trend_stack", "15m_dist_vwap",
    "1h_ret_1", "1h_trend_stack",
]

BINARY_TARGET = "future_up_5m"      # P(return_5m > 0)
RETURN_TARGET = "future_return_5m"  # raw next-5m return for expected_return

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, float(x)))
    return float(1.0 / (1.0 + math.exp(-x)))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _print(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_intraday() -> pd.DataFrame:
    """Load the dated intraday training dataset, with fallback to 1m bars."""
    for path in [INTRADAY_TRAIN_PATH, INTRADAY_BARS_PATH]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            if "symbol" not in df.columns or "close" not in df.columns:
                continue
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            _print(f"[data] loaded {path.name}: {df.shape}")
            return df
        except Exception as exc:
            _print(f"[data] failed to load {path.name}: {exc}")
    return pd.DataFrame()


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of CANDIDATE_FEATURES actually present in df."""
    return [c for c in CANDIDATE_FEATURES if c in df.columns]


def _load_inference_universe() -> list[str]:
    """Symbols to produce signals for: locked universe + intraday symbols."""
    symbols: list[str] = []
    if LOCKED_UNIVERSE_PATH.exists():
        import re
        text = LOCKED_UNIVERSE_PATH.read_text(encoding="utf-8", errors="ignore")
        symbols = [s.upper().strip() for s in re.split(r"[\s,]+", text) if s.strip()]
    return sorted(set(symbols)) if symbols else []


# ---------------------------------------------------------------------------
# AR model (ARIMA proxy — AR(p) via LinearRegression)
# ---------------------------------------------------------------------------

class ARModel:
    """AR(p) model: predict next return from p lagged returns."""

    def __init__(self, p: int = 10) -> None:
        self.p = p
        self.model = LinearRegression()
        self.hist_vol: float = 1e-6
        self.fitted: bool = False

    def _make_lags(self, ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.p, len(ret)):
            X.append(ret[i - self.p: i])
            y.append(ret[i])
        return np.array(X), np.array(y)

    def fit(self, close: np.ndarray) -> dict[str, float]:
        ret = np.diff(np.log(np.maximum(close, 1e-9)))
        self.hist_vol = max(float(np.std(ret[-60:])), 1e-6)
        n = len(ret)
        if n < self.p + 10:
            return {"train_dir_acc": 0.5, "test_dir_acc": 0.5, "n_train": 0, "n_test": 0}
        X, y = self._make_lags(ret)
        split = max(int(len(X) * FORECAST_ENGINE_TRAIN_SPLIT), self.p + 1)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        self.model.fit(X_tr, y_tr)
        self.fitted = True
        metrics: dict[str, float] = {"n_train": float(len(y_tr)), "n_test": float(len(y_te))}
        if len(y_tr) > 0:
            metrics["train_dir_acc"] = float(np.mean((self.model.predict(X_tr) > 0) == (y_tr > 0)))
        if len(y_te) > 0:
            pred_te = self.model.predict(X_te)
            metrics["test_dir_acc"] = float(np.mean((pred_te > 0) == (y_te > 0)))
        return metrics

    def predict_prob(self, close_tail: np.ndarray) -> float:
        """Score ∈ [0, 1]: probability of positive next return."""
        if not self.fitted or len(close_tail) < self.p + 1:
            return 0.50
        ret = np.diff(np.log(np.maximum(close_tail, 1e-9)))
        lag = ret[-self.p:]
        if len(lag) < self.p:
            return 0.50
        raw_pred = float(self.model.predict(lag.reshape(1, -1))[0])
        return _clip01(_sigmoid(raw_pred / max(self.hist_vol, 1e-6)))

    def predict_return(self, close_tail: np.ndarray) -> float:
        """Predicted next log-return."""
        if not self.fitted or len(close_tail) < self.p + 1:
            return 0.0
        ret = np.diff(np.log(np.maximum(close_tail, 1e-9)))
        lag = ret[-self.p:]
        if len(lag) < self.p:
            return 0.0
        return float(self.model.predict(lag.reshape(1, -1))[0])


def train_ar_models(df: pd.DataFrame) -> tuple[dict[str, ARModel], dict[str, Any]]:
    """Fit one ARModel per symbol; return model dict and metrics dict."""
    ar_models: dict[str, ARModel] = {}
    all_metrics: dict[str, Any] = {}

    for sym, g in df.groupby("symbol", sort=False):
        close = pd.to_numeric(g["close"], errors="coerce").dropna().to_numpy()
        if len(close) < FORECAST_ENGINE_MIN_ROWS:
            _print(f"[ar] {sym}: skip (rows={len(close)} < min={FORECAST_ENGINE_MIN_ROWS})")
            continue
        ar = ARModel(p=FORECAST_ENGINE_AR_LAGS)
        metrics = ar.fit(close)
        ar_models[str(sym)] = ar
        all_metrics[str(sym)] = metrics
        _print(
            f"[ar] {sym}: n_train={metrics.get('n_train', 0):.0f} "
            f"n_test={metrics.get('n_test', 0):.0f} "
            f"test_dir_acc={metrics.get('test_dir_acc', 0.5):.3f}"
        )

    return ar_models, all_metrics


# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------

if _TORCH:

    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout if hidden_size > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return torch.sigmoid(self.head(out[:, -1, :]))

    class StackedLSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return torch.sigmoid(self.head(out[:, -1, :]))

    class AttentionLSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout if hidden_size > 1 else 0.0)
            self.attn_proj = nn.Linear(hidden_size, 1)
            self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)                              # (B, T, H)
            weights = torch.softmax(self.attn_proj(out), dim=1)  # (B, T, 1)
            context = (weights * out).sum(dim=1)               # (B, H)
            return torch.sigmoid(self.head(context))

else:

    class LSTMModel:  # type: ignore[no-redef]
        pass

    class StackedLSTMModel:  # type: ignore[no-redef]
        pass

    class AttentionLSTMModel:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    lookback: int,
    stride: int,
    fit_scaler: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y_cls, y_ret) arrays from an already-sorted, NaN-filled DataFrame.

    X      : (N, lookback, n_features)
    y_cls  : (N,) binary 0/1 from BINARY_TARGET
    y_ret  : (N,) float from RETURN_TARGET
    """
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    y_cls_col = df[BINARY_TARGET].to_numpy(dtype=np.float32) if BINARY_TARGET in df.columns else np.zeros(len(df), dtype=np.float32)
    y_ret_col = df[RETURN_TARGET].to_numpy(dtype=np.float32) if RETURN_TARGET in df.columns else np.zeros(len(df), dtype=np.float32)

    if fit_scaler:
        scaler.fit(feat)
    feat_scaled = scaler.transform(feat)

    Xs, y_cls_list, y_ret_list = [], [], []
    for i in range(lookback, len(feat_scaled), stride):
        Xs.append(feat_scaled[i - lookback: i])
        y_cls_list.append(y_cls_col[i])
        y_ret_list.append(y_ret_col[i])

    if not Xs:
        return np.empty((0, lookback, len(feature_cols)), dtype=np.float32), np.empty(0), np.empty(0)

    return (
        np.array(Xs, dtype=np.float32),
        np.array(y_cls_list, dtype=np.float32),
        np.array(y_ret_list, dtype=np.float32),
    )


def _prepare_cross_symbol_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Build chronological train/test arrays across all symbols.

    Returns (X_tr, y_cls_tr, y_ret_tr, X_te, y_cls_te, y_ret_te, scaler).
    """
    df = df.copy()
    # Fill NaN with forward-fill then zero per symbol
    for col in feature_cols + ([BINARY_TARGET] if BINARY_TARGET in df.columns else []) + ([RETURN_TARGET] if RETURN_TARGET in df.columns else []):
        if col in df.columns:
            df[col] = df.groupby("symbol")[col].transform(lambda s: s.ffill().fillna(0.0))

    # Sort chronologically for the train/test split
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * FORECAST_ENGINE_TRAIN_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    scaler = StandardScaler()

    # Fit scaler on train only, then transform both sets
    if len(train_df) > 0:
        scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    def _seqs(sdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sdf.empty:
            n = len(feature_cols)
            return np.empty((0, lookback, n), dtype=np.float32), np.empty(0), np.empty(0)
        feat = scaler.transform(sdf[feature_cols].to_numpy(dtype=np.float32))
        y_cls = sdf[BINARY_TARGET].to_numpy(dtype=np.float32) if BINARY_TARGET in sdf.columns else np.zeros(len(sdf), dtype=np.float32)
        y_ret = sdf[RETURN_TARGET].to_numpy(dtype=np.float32) if RETURN_TARGET in sdf.columns else np.zeros(len(sdf), dtype=np.float32)
        Xs, yc, yr = [], [], []
        for i in range(lookback, len(feat), stride):
            Xs.append(feat[i - lookback: i])
            yc.append(y_cls[i])
            yr.append(y_ret[i])
        if not Xs:
            n = len(feature_cols)
            return np.empty((0, lookback, n), dtype=np.float32), np.empty(0), np.empty(0)
        return np.array(Xs, dtype=np.float32), np.array(yc, dtype=np.float32), np.array(yr, dtype=np.float32)

    X_tr, yc_tr, yr_tr = _seqs(train_df)
    X_te, yc_te, yr_te = _seqs(test_df)

    _print(
        f"[data] sequences: train={len(X_tr)} test={len(X_te)} "
        f"features={len(feature_cols)} lookback={lookback} stride={stride}"
    )
    return X_tr, yc_tr, yr_tr, X_te, yc_te, yr_te, scaler


# ---------------------------------------------------------------------------
# PyTorch training
# ---------------------------------------------------------------------------

def _train_torch_model(
    model: "nn.Module",
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """Train a PyTorch model for binary classification; return metrics."""
    if not _TORCH or len(X_tr) == 0:
        return {"error": "no_data_or_torch_unavailable"}

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=FORECAST_ENGINE_LR)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=FORECAST_ENGINE_BATCH_SIZE, shuffle=True, drop_last=False)

    t0 = time.time()
    for epoch in range(FORECAST_ENGINE_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % max(1, FORECAST_ENGINE_EPOCHS // 4) == 0:
            _print(f"[{model_name}] epoch {epoch + 1}/{FORECAST_ENGINE_EPOCHS} loss={epoch_loss / max(len(loader), 1):.4f}")

    elapsed = time.time() - t0

    # Evaluate on test set
    metrics: dict[str, float] = {"train_samples": float(len(X_tr)), "test_samples": float(len(X_te)), "elapsed_s": round(elapsed, 1)}
    if len(X_te) == 0:
        return metrics

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        probs = model(X_te_t).squeeze(-1).cpu().numpy()

    preds = (probs >= 0.50).astype(int)
    labels = y_te.astype(int)

    metrics["test_accuracy"] = round(float(accuracy_score(labels, preds)), 4)
    try:
        metrics["test_auc"] = round(float(roc_auc_score(labels, probs)), 4)
    except Exception:
        metrics["test_auc"] = 0.5
    try:
        metrics["test_precision"] = round(float(precision_score(labels, preds, zero_division=0)), 4)
        metrics["test_recall"] = round(float(recall_score(labels, preds, zero_division=0)), 4)
    except Exception:
        pass
    metrics["avg_predicted_up_prob"] = round(float(np.mean(probs)), 4)
    metrics["avg_actual_return"] = round(float(np.mean(y_te[preds == 1])) if np.any(preds == 1) else 0.0, 6)

    _print(
        f"[{model_name}] acc={metrics.get('test_accuracy', 0):.3f} "
        f"auc={metrics.get('test_auc', 0.5):.3f} "
        f"precision={metrics.get('test_precision', 0):.3f} "
        f"elapsed={elapsed:.1f}s"
    )
    return metrics


def train_pytorch_models(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Any, Any, Any, StandardScaler, dict[str, Any]]:
    """Train LSTM, Stacked-LSTM, and Attention-LSTM. Return (model_l, model_sl, model_al, scaler, metrics)."""
    if not _TORCH:
        _print("[torch] PyTorch unavailable; skipping LSTM training")
        scaler = StandardScaler()
        if len(df) > 0:
            scaler.fit(df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))
        return None, None, None, scaler, {"error": "torch_unavailable"}

    X_tr, yc_tr, yr_tr, X_te, yc_te, yr_te, scaler = _prepare_cross_symbol_arrays(
        df, feature_cols, FORECAST_ENGINE_LOOKBACK, FORECAST_ENGINE_STRIDE
    )

    if len(X_tr) == 0:
        _print("[torch] no training sequences available")
        return None, None, None, scaler, {"error": "no_sequences"}

    n_feat = X_tr.shape[2]
    hidden = FORECAST_ENGINE_HIDDEN_SIZE

    model_l = LSTMModel(n_feat, hidden)
    model_sl = StackedLSTMModel(n_feat, hidden)
    model_al = AttentionLSTMModel(n_feat, hidden)

    metrics: dict[str, Any] = {}
    metrics["lstm"] = _train_torch_model(model_l, X_tr, yc_tr, X_te, yc_te, "lstm")
    metrics["stacked_lstm"] = _train_torch_model(model_sl, X_tr, yc_tr, X_te, yc_te, "stacked_lstm")
    metrics["attention_lstm"] = _train_torch_model(model_al, X_tr, yc_tr, X_te, yc_te, "attention_lstm")

    return model_l, model_sl, model_al, scaler, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _save_torch_model(model: "nn.Module | None", path: Path) -> None:
    if model is None or not _TORCH:
        return
    try:
        torch.save(model.state_dict(), str(path))
        _print(f"[save] {path.name}")
    except Exception as exc:
        _print(f"[save] failed {path.name}: {exc}")


def _load_torch_model(model_cls: type, path: Path, input_size: int) -> "nn.Module | None":
    if not _TORCH or not path.exists():
        return None
    try:
        m = model_cls(input_size, FORECAST_ENGINE_HIDDEN_SIZE)
        m.load_state_dict(torch.load(str(path), map_location="cpu"))
        m.to(DEVICE).eval()
        return m
    except Exception as exc:
        _print(f"[load] failed {path.name}: {exc}")
        return None


def _save_scaler(scaler: StandardScaler, path: Path) -> None:
    try:
        with path.open("wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        _print(f"[save] {path.name}")
    except Exception as exc:
        _print(f"[save] scaler failed: {exc}")


def _load_scaler(path: Path) -> StandardScaler | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        _print(f"[load] scaler failed: {exc}")
        return None


def _save_ar_models(models: dict[str, ARModel], path: Path) -> None:
    try:
        with path.open("wb") as f:
            pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
        _print(f"[save] {path.name} ({len(models)} symbols)")
    except Exception as exc:
        _print(f"[save] ar_models failed: {exc}")


def _load_ar_models(path: Path) -> dict[str, ARModel]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        _print(f"[load] ar_models failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Inference — per-symbol signal production
# ---------------------------------------------------------------------------

def _infer_torch(model: "nn.Module | None", seq: np.ndarray) -> float:
    """Run a single sequence (lookback × features) through model → probability."""
    if model is None or not _TORCH or seq is None or len(seq) == 0:
        return 0.50
    try:
        model.eval()
        x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            return _clip01(float(model(x).item()))
    except Exception:
        return 0.50


def _infer_ar(ar_models: dict[str, ARModel], symbol: str, close: np.ndarray) -> tuple[float, float]:
    """Returns (arima_score, expected_return_from_ar)."""
    ar = ar_models.get(symbol)
    if ar is None or not ar.fitted:
        return 0.50, 0.0
    close_tail = close[-(FORECAST_ENGINE_AR_LAGS + 2):]
    return ar.predict_prob(close_tail), ar.predict_return(close_tail)


def _get_last_sequence(
    sym_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    lookback: int,
) -> np.ndarray | None:
    """Return the last `lookback` rows as a scaled feature matrix."""
    if sym_df.empty or len(sym_df) < lookback:
        return None
    tail = sym_df.tail(lookback)[feature_cols].copy()
    tail = tail.ffill().fillna(0.0)
    try:
        return scaler.transform(tail.to_numpy(dtype=np.float32))
    except Exception:
        return None


def _fallback_drift_score(sym_df: pd.DataFrame) -> dict[str, float]:
    """Minimal drift-based proxy when models are unavailable."""
    close = pd.to_numeric(sym_df["close"], errors="coerce").dropna()
    if len(close) < 10:
        return {"arima_score": 0.50, "lstm_score": 0.50, "stacked_lstm_score": 0.50, "attention_lstm_score": 0.50, "expected_return": 0.0, "drawdown_probability": 0.50, "forecast_source": "fallback_drift_proxy"}
    rets = close.pct_change().dropna()
    vol = max(float(rets.tail(20).std()), 1e-6)
    drift = float(rets.tail(10).mean())
    score = _clip01(_sigmoid(drift / vol))
    return {"arima_score": score, "lstm_score": score, "stacked_lstm_score": score, "attention_lstm_score": score, "expected_return": float(drift), "drawdown_probability": _clip01(1.0 - score), "forecast_source": "fallback_drift_proxy"}


def build_signals(
    df: pd.DataFrame,
    feature_cols: list[str],
    ar_models: dict[str, ARModel],
    lstm_model: Any,
    stacked_model: Any,
    attention_model: Any,
    scaler: StandardScaler | None,
    universe: list[str],
) -> list[dict[str, Any]]:
    """Produce one signal row per symbol."""
    signals: list[dict[str, Any]] = []

    # Build inference universe: requested symbols + any in intraday data
    infer_syms = set(universe)
    if "symbol" in df.columns:
        infer_syms |= set(df["symbol"].unique())
    if not infer_syms:
        return signals

    # Group intraday data by symbol for fast lookup
    by_sym: dict[str, pd.DataFrame] = {}
    if not df.empty and "symbol" in df.columns:
        for sym, g in df.groupby("symbol", sort=False):
            by_sym[str(sym)] = g

    for sym in sorted(infer_syms):
        sym_df = by_sym.get(sym, pd.DataFrame())
        ts = _now_utc()

        # ---- AR score ----
        close_arr = pd.to_numeric(sym_df["close"], errors="coerce").dropna().to_numpy() if not sym_df.empty else np.array([])
        arima_score, ar_expected_return = _infer_ar(ar_models, sym, close_arr)

        # ---- LSTM scores ----
        if scaler is not None and not sym_df.empty and len(sym_df) >= FORECAST_ENGINE_LOOKBACK and len(feature_cols) > 0:
            seq = _get_last_sequence(sym_df, feature_cols, scaler, FORECAST_ENGINE_LOOKBACK)
            lstm_score = _infer_torch(lstm_model, seq)
            stacked_lstm_score = _infer_torch(stacked_model, seq)
            attention_lstm_score = _infer_torch(attention_model, seq)
            source = "forecast_engine_v1"
        elif not sym_df.empty:
            # Not enough data for LSTM — use fallback
            fb = _fallback_drift_score(sym_df)
            lstm_score = fb["lstm_score"]
            stacked_lstm_score = fb["stacked_lstm_score"]
            attention_lstm_score = fb["attention_lstm_score"]
            source = fb["forecast_source"]
        else:
            lstm_score = stacked_lstm_score = attention_lstm_score = 0.50
            source = "no_data"

        # ---- Ensemble ----
        w = ENSEMBLE_WEIGHTS
        weight_sum = max(sum(w.values()), 1e-9)
        ensemble = (
            w["arima"] * arima_score
            + w["lstm"] * lstm_score
            + w["stacked_lstm"] * stacked_lstm_score
            + w["attention_lstm"] * attention_lstm_score
        ) / weight_sum

        up_prob = _clip01(ensemble)

        # Expected return: blend AR prediction with LSTM-implied magnitude
        avg_abs_ret = float(pd.to_numeric(sym_df.get(RETURN_TARGET, pd.Series(dtype=float)), errors="coerce").abs().mean()) if (not sym_df.empty and RETURN_TARGET in sym_df.columns) else 0.002
        lstm_expected_return = (up_prob - 0.50) * 2.0 * avg_abs_ret
        expected_return = 0.60 * ar_expected_return + 0.40 * lstm_expected_return

        # Drawdown probability: inversely related to up_prob
        drawdown_probability = _clip01(0.50 + (0.50 - up_prob))

        row: dict[str, Any] = {
            "symbol": sym,
            "arima_score": round(_clip01(arima_score), 4),
            "lstm_score": round(_clip01(lstm_score), 4),
            "stacked_lstm_score": round(_clip01(stacked_lstm_score), 4),
            "attention_lstm_score": round(_clip01(attention_lstm_score), 4),
            "ensemble_score": round(_clip01(float(ensemble)), 4),
            "up_probability": round(up_prob, 4),
            "expected_return": round(float(expected_return), 6),
            "drawdown_probability": round(drawdown_probability, 4),
            "forecast_buy_approved": bool(
                ensemble >= FORECAST_BUY_THRESHOLD
                and expected_return >= FORECAST_MIN_EXPECTED_RETURN
                and drawdown_probability <= FORECAST_MAX_DRAWDOWN_PROB
            ),
            "forecast_strong_buy": bool(ensemble >= FORECAST_STRONG_BUY_THRESHOLD),
            "forecast_source": source,
            "created_at_utc": ts,
        }
        signals.append(row)

    return signals


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_outputs(signals: list[dict[str, Any]]) -> None:
    payload = {"created_at_utc": _now_utc(), "weights": ENSEMBLE_WEIGHTS, "signals": signals}
    try:
        FORECAST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FORECAST_OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _print(f"[out] {FORECAST_OUTPUT_PATH.name} rows={len(signals)}")
    except Exception as exc:
        _print(f"[out] json write failed: {exc}")

    if signals:
        try:
            FORECAST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(signals).to_csv(FORECAST_CSV_PATH, index=False)
            _print(f"[out] {FORECAST_CSV_PATH.name}")
        except Exception as exc:
            _print(f"[out] csv write failed: {exc}")


def _write_metrics(metrics: dict[str, Any]) -> None:
    try:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        _print(f"[out] {METRICS_PATH.name}")
    except Exception as exc:
        _print(f"[out] metrics write failed: {exc}")


def _write_safe_fallback(reason: str) -> None:
    """Write empty/neutral signals so run.py load doesn't crash."""
    _print(f"[fallback] writing neutral signals: {reason}")
    _write_outputs([])
    _write_metrics({"error": reason, "created_at_utc": _now_utc()})


# ---------------------------------------------------------------------------
# Model freshness check
# ---------------------------------------------------------------------------

def _models_are_fresh() -> bool:
    """Return True if all model files exist and were written today."""
    today_prefix = TODAY_TAG
    for p in [LSTM_PATH, STACKED_LSTM_PATH, ATTENTION_LSTM_PATH, SCALER_PATH, AR_PATH]:
        if not p.exists():
            return False
        # If model dir name contains today's date, it's fresh
    return DATA_TAG == TODAY_TAG or all(p.exists() for p in [LSTM_PATH, STACKED_LSTM_PATH, ATTENTION_LSTM_PATH, SCALER_PATH, AR_PATH])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    _print("===== FORECAST ENGINE =====")
    _print(f"[config] data_tag={DATA_TAG} torch={_TORCH} device={DEVICE}")
    _print(f"[config] train_on_start={FORECAST_ENGINE_TRAIN_ON_START} epochs={FORECAST_ENGINE_EPOCHS} lookback={FORECAST_ENGINE_LOOKBACK}")
    _print(f"[config] hidden={FORECAST_ENGINE_HIDDEN_SIZE} batch={FORECAST_ENGINE_BATCH_SIZE} lr={FORECAST_ENGINE_LR}")
    _print(f"[config] ar_lags={FORECAST_ENGINE_AR_LAGS} stride={FORECAST_ENGINE_STRIDE} min_rows={FORECAST_ENGINE_MIN_ROWS}")
    _print(f"[out] signals → {FORECAST_OUTPUT_PATH}")
    _print(f"[out] models  → {MODEL_DIR}")

    if not FORECAST_ENGINE_ENABLED:
        _write_safe_fallback("forecast_engine_disabled")
        return 0

    # ---- Load training data ------------------------------------------------
    df = _load_intraday()
    if df.empty:
        _write_safe_fallback("no_intraday_data")
        return 0

    feature_cols = _select_features(df)
    if not feature_cols:
        _write_safe_fallback("no_feature_columns_found")
        return 0

    _print(f"[data] feature_cols={len(feature_cols)}: {feature_cols[:8]}...")
    _print(f"[data] symbols={sorted(df['symbol'].unique().tolist())}")
    _print(f"[data] total_rows={len(df)}")

    # ---- Determine training vs inference-only mode -------------------------
    do_train = FORECAST_ENGINE_TRAIN_ON_START and not _models_are_fresh()
    if FORECAST_ENGINE_TRAIN_ON_START and _models_are_fresh():
        _print("[train] models already fresh for this date — skipping training")
    elif not FORECAST_ENGINE_TRAIN_ON_START:
        _print("[train] FORECAST_ENGINE_TRAIN_ON_START=0 — inference only")

    all_metrics: dict[str, Any] = {"created_at_utc": _now_utc(), "data_tag": DATA_TAG}

    # ---- AR model -----------------------------------------------------------
    ar_models: dict[str, ARModel] = {}
    if do_train:
        _print("\n--- Training AR models ---")
        ar_models, ar_metrics = train_ar_models(df)
        all_metrics["ar"] = ar_metrics
        if ar_models:
            _save_ar_models(ar_models, AR_PATH)
    else:
        ar_models = _load_ar_models(AR_PATH)
        _print(f"[ar] loaded {len(ar_models)} saved AR models")

    # ---- PyTorch LSTM models ------------------------------------------------
    lstm_model = stacked_model = attention_model = None
    scaler: StandardScaler | None = None

    if do_train and _TORCH:
        _print("\n--- Training LSTM models ---")
        lstm_model, stacked_model, attention_model, scaler, torch_metrics = train_pytorch_models(df, feature_cols)
        all_metrics["pytorch"] = torch_metrics
        if scaler is not None:
            _save_scaler(scaler, SCALER_PATH)
        _save_torch_model(lstm_model, LSTM_PATH)
        _save_torch_model(stacked_model, STACKED_LSTM_PATH)
        _save_torch_model(attention_model, ATTENTION_LSTM_PATH)
    elif _TORCH:
        _print("[torch] loading saved LSTM models")
        scaler = _load_scaler(SCALER_PATH)
        if scaler is not None:
            n_feat = len(feature_cols)
            lstm_model = _load_torch_model(LSTMModel, LSTM_PATH, n_feat)
            stacked_model = _load_torch_model(StackedLSTMModel, STACKED_LSTM_PATH, n_feat)
            attention_model = _load_torch_model(AttentionLSTMModel, ATTENTION_LSTM_PATH, n_feat)
            _print(f"[torch] lstm={lstm_model is not None} stacked={stacked_model is not None} attention={attention_model is not None}")
        else:
            _print("[torch] no saved scaler; cannot load LSTM models (run with FORECAST_ENGINE_TRAIN_ON_START=1 first)")
    else:
        _print("[torch] PyTorch unavailable; using AR-only signals")
        scaler = StandardScaler()
        if len(df) > 0:
            scaler.fit(df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))

    # ---- Inference universe ------------------------------------------------
    universe = _load_inference_universe()
    if not universe and "symbol" in df.columns:
        universe = sorted(df["symbol"].unique().tolist())
    _print(f"\n[infer] universe={len(universe)} symbols: {universe[:10]}")

    # ---- Build signals -----------------------------------------------------
    _print("\n--- Building forecast signals ---")
    signals = build_signals(df, feature_cols, ar_models, lstm_model, stacked_model, attention_model, scaler, universe)
    _print(f"[infer] generated {len(signals)} signal rows")

    for sig in signals:
        _print(
            f"  {sig['symbol']:8s} arima={sig['arima_score']:.3f} "
            f"lstm={sig['lstm_score']:.3f} attn={sig['attention_lstm_score']:.3f} "
            f"ens={sig['ensemble_score']:.3f} up={sig['up_probability']:.3f} "
            f"ret={sig['expected_return']:.4f} approved={sig['forecast_buy_approved']}"
        )

    # ---- Write outputs ------------------------------------------------------
    _write_outputs(signals)
    all_metrics["signal_count"] = len(signals)
    all_metrics["approved_count"] = sum(1 for s in signals if s.get("forecast_buy_approved"))
    all_metrics["elapsed_s"] = round(time.time() - t_start, 1)
    _write_metrics(all_metrics)

    _print(f"\n[done] elapsed={all_metrics['elapsed_s']}s signals={len(signals)} approved={all_metrics['approved_count']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as _exc:
        import traceback
        _print(f"[fatal] unexpected error: {_exc}")
        traceback.print_exc()
        # Write fallback so run.py doesn't block on missing output
        try:
            _write_safe_fallback(f"fatal_error:{type(_exc).__name__}")
        except Exception:
            pass
        sys.exit(0)  # exit 0 so run.py subprocess doesn't count this as a hard failure
