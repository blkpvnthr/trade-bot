#!/usr/bin/env python3
"""
deeplob_orderbook_direction_layer.py

DeepLOB-style limit-order-book direction layer.

Model idea
----------
This follows the common DeepLOB design pattern:

    1. CNN blocks extract local price/volume structure from the limit order book.
    2. Inception-style temporal convolutions extract multi-scale order-book patterns.
    3. LSTM captures temporal dependence across the most recent order-book states.
    4. Classifier predicts probabilities:
            0 = DOWN
            1 = FLAT / NO MOVE
            2 = UP

Input
-----
For each symbol, each timestep contains 10 bid levels and 10 ask levels:

    bid_price_1..10
    bid_size_1..10
    ask_price_1..10
    ask_size_1..10

That is 40 features per timestep.

With LOOKBACK=100, each training example is:

    100 x 40

The user's note mentioned "vector of size 400"; with 10 bid/ask levels and
prices+volumes, the standard DeepLOB input is 100 x 40 = 4000 scalar values.
This script uses the standard 100-by-40 tensor.

Labels
------
Labels are built from the difference between future and past moving averages:

    label_raw = future_mid_ma - past_mid_ma

Then quantized:

    label = UP   if label_raw >= threshold
    label = DOWN if label_raw <= -threshold
    label = FLAT otherwise

Threshold should be low enough to avoid extreme class imbalance, but high enough
to represent a meaningful move. It can be configured with:

    DEEPLOB_LABEL_THRESHOLD

Inputs
------
Expected historical order-book CSV:

    system-data/YYYYMMDD/YYYYMMDD_lob_snapshots.csv

Required columns:
    timestamp, symbol,
    bid_price_1..bid_price_10,
    bid_size_1..bid_size_10,
    ask_price_1..ask_price_10,
    ask_size_1..ask_size_10

Outputs
-------
    system-data/YYYYMMDD/YYYYMMDD_deeplob_model.pt
    system-data/YYYYMMDD/YYYYMMDD_deeplob_scaler.json
    system-data/YYYYMMDD/YYYYMMDD_deeplob_metrics.json
    system-data/YYYYMMDD/YYYYMMDD_deeplob_live_signals.json
    system-data/YYYYMMDD/YYYYMMDD_deeplob_live_signals.csv

Run
---
Train:
    python deeplob_orderbook_direction_layer.py --train

Run once:
    python deeplob_orderbook_direction_layer.py --once

Run live loop:
    python deeplob_orderbook_direction_layer.py --live

run.py integration:
    - start this layer in background
    - read *_deeplob_live_signals.json
    - use UP probability to approve entries/adds
    - use DOWN probability to accelerate exits
    - never short by default
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None

try:
    from alpaca.data.live.stock import StockDataStream
    from alpaca.trading.client import TradingClient
except Exception:
    StockDataStream = None
    TradingClient = None


# =============================================================================
# Paths/config
# =============================================================================

ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("DEEPLOB_DATA_DATE", TODAY_TAG))

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOB_SNAPSHOTS_PATH = OUT_DIR / f"{DATA_TAG}_lob_snapshots.csv"
MODEL_PATH = OUT_DIR / f"{DATA_TAG}_deeplob_model.pt"
SCALER_PATH = OUT_DIR / f"{DATA_TAG}_deeplob_scaler.json"
METRICS_PATH = OUT_DIR / f"{DATA_TAG}_deeplob_metrics.json"
SIGNALS_JSON_PATH = OUT_DIR / f"{DATA_TAG}_deeplob_live_signals.json"
SIGNALS_CSV_PATH = OUT_DIR / f"{DATA_TAG}_deeplob_live_signals.csv"
EVENTS_PATH = LOG_DIR / "deeplob_events.jsonl"

WATCHLIST_PATHS = [
    OUT_DIR / f"{DATA_TAG}_daily_walkforward_emulated_rebalance_orders.csv",
    OUT_DIR / f"{DATA_TAG}_stream_confirmed_rebalance_orders.csv",
    PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt",
    PROJECT_ROOT / "locked_trading_universe.txt",
]

LOOKBACK = int(os.getenv("DEEPLOB_LOOKBACK", "100"))
LEVELS = int(os.getenv("DEEPLOB_LEVELS", "10"))
FEATURES_PER_LEVEL = 4
N_FEATURES = LEVELS * FEATURES_PER_LEVEL

PAST_MA = int(os.getenv("DEEPLOB_PAST_MA", "20"))
FUTURE_MA = int(os.getenv("DEEPLOB_FUTURE_MA", "20"))
LABEL_HORIZON = int(os.getenv("DEEPLOB_LABEL_HORIZON", "30"))
LABEL_THRESHOLD = float(os.getenv("DEEPLOB_LABEL_THRESHOLD", "0.015"))

TRAIN_EPOCHS = int(os.getenv("DEEPLOB_EPOCHS", "12"))
BATCH_SIZE = int(os.getenv("DEEPLOB_BATCH_SIZE", "128"))
LR = float(os.getenv("DEEPLOB_LR", "0.0008"))
WEIGHT_DECAY = float(os.getenv("DEEPLOB_WEIGHT_DECAY", "0.0001"))
DROPOUT = float(os.getenv("DEEPLOB_DROPOUT", "0.15"))
LSTM_HIDDEN = int(os.getenv("DEEPLOB_LSTM_HIDDEN", "96"))
LSTM_LAYERS = int(os.getenv("DEEPLOB_LSTM_LAYERS", "1"))

UP_ENTRY_THRESHOLD = float(os.getenv("DEEPLOB_UP_ENTRY_THRESHOLD", "0.58"))
DOWN_EXIT_THRESHOLD = float(os.getenv("DEEPLOB_DOWN_EXIT_THRESHOLD", "0.55"))
NO_MOVE_BLOCK_THRESHOLD = float(os.getenv("DEEPLOB_NO_MOVE_BLOCK_THRESHOLD", "0.62"))

LONG_ONLY = os.getenv("DEEPLOB_LONG_ONLY", "1").lower() in {"1", "true", "yes"}
SCAN_SECONDS = int(os.getenv("DEEPLOB_SCAN_SECONDS", "5"))

DEVICE = (
    "mps"
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cuda"
    if torch is not None and torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# Helpers
# =============================================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(x: Any) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def safe_json(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: safe_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [safe_json(v) for v in x]
    return x


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {"timestamp_utc": now_utc_iso(), "event": event, "payload": safe_json(payload or {})}
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=safe_json) + "\n")


def get_credentials() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return key, secret


def feature_columns() -> list[str]:
    cols = []
    for i in range(1, LEVELS + 1):
        cols.append(f"bid_price_{i}")
    for i in range(1, LEVELS + 1):
        cols.append(f"bid_size_{i}")
    for i in range(1, LEVELS + 1):
        cols.append(f"ask_price_{i}")
    for i in range(1, LEVELS + 1):
        cols.append(f"ask_size_{i}")
    return cols


FEATURE_COLUMNS = feature_columns()


def validate_lob_columns(df: pd.DataFrame) -> None:
    missing = ["timestamp", "symbol"] + [c for c in FEATURE_COLUMNS if c not in df.columns]
    missing = [c for c in missing if c not in df.columns]
    if missing:
        raise ValueError(f"LOB snapshot file missing columns: {missing[:20]}")


def mid_price_from_lob(df: pd.DataFrame) -> pd.Series:
    return (df["bid_price_1"].astype(float) + df["ask_price_1"].astype(float)) / 2.0


def normalize_lob_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize order-book prices around mid price and sizes by total visible depth.
    This keeps the model focused on shape/imbalance rather than absolute price.
    """
    out = df.copy()
    mid = mid_price_from_lob(out).replace(0, np.nan)
    total_size = pd.Series(0.0, index=out.index)

    for i in range(1, LEVELS + 1):
        total_size += out[f"bid_size_{i}"].astype(float).abs()
        total_size += out[f"ask_size_{i}"].astype(float).abs()

    total_size = total_size.replace(0, np.nan)

    for i in range(1, LEVELS + 1):
        out[f"bid_price_{i}"] = out[f"bid_price_{i}"].astype(float) / mid - 1.0
        out[f"ask_price_{i}"] = out[f"ask_price_{i}"].astype(float) / mid - 1.0
        out[f"bid_size_{i}"] = np.log1p(out[f"bid_size_{i}"].astype(float).clip(lower=0.0)) / np.log1p(total_size)
        out[f"ask_size_{i}"] = np.log1p(out[f"ask_size_{i}"].astype(float).clip(lower=0.0)) / np.log1p(total_size)

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def load_lob_snapshots() -> pd.DataFrame:
    if not LOB_SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LOB_SNAPSHOTS_PATH}. Expected historical LOB snapshots with 10 bid/ask levels."
        )

    df = pd.read_csv(LOB_SNAPSHOTS_PATH)
    validate_lob_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df.dropna(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (3.0 * counts)
    return weights.astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================

class DeepLOBDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.samples: list[tuple[np.ndarray, int]] = []

        for symbol, g in df.groupby("symbol", sort=False):
            raw = g.sort_values("timestamp").reset_index(drop=True)
            if len(raw) < LOOKBACK + LABEL_HORIZON + FUTURE_MA + 5:
                continue

            mid = mid_price_from_lob(raw)
            norm = normalize_lob_frame(raw)
            xmat = norm[FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)

            past_ma = mid.rolling(PAST_MA).mean()
            future_ma = mid.shift(-LABEL_HORIZON).rolling(FUTURE_MA).mean()

            move = future_ma - past_ma

            for end_idx in range(LOOKBACK, len(raw) - LABEL_HORIZON - FUTURE_MA):
                m = f(move.iloc[end_idx], 0.0)
                if m >= LABEL_THRESHOLD:
                    label = 2
                elif m <= -LABEL_THRESHOLD:
                    label = 0
                else:
                    label = 1

                window = xmat[end_idx - LOOKBACK:end_idx, :]
                if window.shape != (LOOKBACK, N_FEATURES):
                    continue

                self.samples.append((window, int(label)))

        self.labels = np.array([y for _, y in self.samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =============================================================================
# Model
# =============================================================================

class DeepLOBModel(nn.Module):
    """
    DeepLOB-style CNN + inception + LSTM model.

    Input:
        B x T x F, where F=40 for 10 bid/ask price/size levels.

    Internally:
        B x 1 x T x F -> CNN feature maps -> sequence -> LSTM -> logits.
    """

    def __init__(self, input_features: int = N_FEATURES, hidden_size: int = LSTM_HIDDEN, dropout: float = DROPOUT):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(3, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.inception_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        self.inception_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )

        # Infer LSTM input dimension.
        with torch.no_grad():
            dummy = torch.zeros(2, 1, LOOKBACK, input_features)
            y = self._cnn_features(dummy)
            _, c, t, w = y.shape
            lstm_input_dim = c * w

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=dropout if LSTM_LAYERS > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def _cnn_features(self, x: Any) -> Any:
        x = self.conv_block(x)
        # Trim time dimension in case padding made it longer.
        x = x[:, :, :LOOKBACK, :]
        i1 = self.inception_1(x)
        i2 = self.inception_2(x)
        i3 = self.inception_3(x)
        x = torch.cat([i1, i2, i3], dim=1)
        return x

    def forward(self, x: Any) -> Any:
        # x: B x T x F
        x = x.unsqueeze(1)              # B x 1 x T x F
        x = self._cnn_features(x)       # B x C x T x W
        x = x.permute(0, 2, 1, 3)       # B x T x C x W
        x = x.reshape(x.shape[0], x.shape[1], -1)
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.head(last)


# =============================================================================
# Training
# =============================================================================

def train_model() -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required")

    df = load_lob_snapshots()
    ds = DeepLOBDataset(df)

    if len(ds) < 500:
        raise RuntimeError(f"Not enough LOB training samples: {len(ds)}")

    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    weights = torch.tensor(class_weights(ds.labels), dtype=torch.float32, device=DEVICE)
    model = DeepLOBModel().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += float(loss.detach().cpu())

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        pred_counts = np.zeros(3, dtype=int)

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.detach().cpu())

                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().cpu())
                total += len(yb)
                for c in pred.detach().cpu().numpy():
                    pred_counts[int(c)] += 1

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        acc = correct / max(total, 1)

        print(f"[train] epoch={epoch}/{TRAIN_EPOCHS} train_loss={train_loss:.5f} val_loss={val_loss:.5f} acc={acc:.3f} preds={pred_counts.tolist()}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": model.state_dict(),
            "lookback": LOOKBACK,
            "levels": LEVELS,
            "feature_columns": FEATURE_COLUMNS,
            "label_threshold": LABEL_THRESHOLD,
            "past_ma": PAST_MA,
            "future_ma": FUTURE_MA,
            "label_horizon": LABEL_HORIZON,
            "lstm_hidden": LSTM_HIDDEN,
            "lstm_layers": LSTM_LAYERS,
            "dropout": DROPOUT,
        },
        MODEL_PATH,
    )

    label_counts = np.bincount(ds.labels, minlength=3).astype(int).tolist()
    metrics = {
        "ok": True,
        "created_at_utc": now_utc_iso(),
        "model_path": str(MODEL_PATH),
        "rows": int(len(df)),
        "samples": int(len(ds)),
        "val_loss": float(best_val),
        "label_counts": {
            "down": label_counts[0],
            "flat": label_counts[1],
            "up": label_counts[2],
        },
        "label_threshold": LABEL_THRESHOLD,
        "lookback": LOOKBACK,
        "levels": LEVELS,
        "input_shape": [LOOKBACK, N_FEATURES],
        "classes": {"0": "DOWN", "1": "FLAT", "2": "UP"},
    }
    write_json(METRICS_PATH, metrics)
    write_json(SCALER_PATH, {"normalization": "mid_price_centered_and_depth_log_scaled", "feature_columns": FEATURE_COLUMNS})

    print(f"[saved] {MODEL_PATH}")
    print(f"[saved] {METRICS_PATH}")
    return metrics


def load_model() -> DeepLOBModel:
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Run: python deeplob_orderbook_direction_layer.py --train")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = DeepLOBModel(
        input_features=len(ckpt.get("feature_columns", FEATURE_COLUMNS)),
        hidden_size=int(ckpt.get("lstm_hidden", LSTM_HIDDEN)),
        dropout=float(ckpt.get("dropout", DROPOUT)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =============================================================================
# Live inference
# =============================================================================

@dataclass
class LOBSignal:
    symbol: str
    action: str
    prob_down: float
    prob_flat: float
    prob_up: float
    confidence: float
    mid_price: float
    reason: str
    created_at_utc: str


def load_watchlist() -> list[str]:
    symbols: set[str] = set()

    for path in WATCHLIST_PATHS:
        if not path.exists():
            continue

        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
                if "symbol" in df.columns:
                    symbols.update(df["symbol"].map(normalize_symbol).dropna().tolist())
            except Exception:
                pass
        else:
            try:
                raw = path.read_text(encoding="utf-8")
                for token in raw.replace("\n", ",").split(","):
                    sym = normalize_symbol(token)
                    if sym:
                        symbols.add(sym)
            except Exception:
                pass

    env_symbols = os.getenv("DEEPLOB_SYMBOLS", "")
    if env_symbols:
        for token in env_symbols.split(","):
            sym = normalize_symbol(token)
            if sym:
                symbols.add(sym)

    return sorted(s for s in symbols if s)


def latest_lob_windows_from_csv(symbols: list[str]) -> dict[str, pd.DataFrame]:
    if not LOB_SNAPSHOTS_PATH.exists():
        return {}

    df = load_lob_snapshots()
    if symbols:
        df = df[df["symbol"].isin(symbols)].copy()

    out = {}
    for symbol, g in df.groupby("symbol", sort=False):
        x = g.sort_values("timestamp").tail(LOOKBACK).copy()
        if len(x) >= LOOKBACK:
            out[symbol] = x
    return out


def predict_window(model: DeepLOBModel, symbol: str, window: pd.DataFrame) -> LOBSignal:
    norm = normalize_lob_frame(window)
    x = norm[FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)

    xb = torch.tensor(x[None, :, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    prob_down = float(probs[0])
    prob_flat = float(probs[1])
    prob_up = float(probs[2])
    mid = float(mid_price_from_lob(window).iloc[-1])

    if prob_up >= UP_ENTRY_THRESHOLD and prob_flat < NO_MOVE_BLOCK_THRESHOLD:
        action = "APPROVE_LONG_ENTRY"
        confidence = prob_up
        reason = f"deeplob_up_probability:{prob_up:.2f}"
    elif prob_down >= DOWN_EXIT_THRESHOLD:
        action = "EXIT_LONG"
        confidence = prob_down
        reason = f"deeplob_down_probability:{prob_down:.2f}"
    elif prob_flat >= NO_MOVE_BLOCK_THRESHOLD:
        action = "BLOCK_ENTRY"
        confidence = prob_flat
        reason = f"deeplob_no_move_probability:{prob_flat:.2f}"
    else:
        action = "HOLD"
        confidence = max(prob_down, prob_flat, prob_up)
        reason = f"deeplob_neutral:down={prob_down:.2f}:flat={prob_flat:.2f}:up={prob_up:.2f}"

    return LOBSignal(
        symbol=symbol,
        action=action,
        prob_down=prob_down,
        prob_flat=prob_flat,
        prob_up=prob_up,
        confidence=confidence,
        mid_price=mid,
        reason=reason,
        created_at_utc=now_utc_iso(),
    )


def infer_signals() -> list[LOBSignal]:
    symbols = load_watchlist()
    model = load_model()

    windows = latest_lob_windows_from_csv(symbols)
    signals = []

    for symbol, window in windows.items():
        try:
            signals.append(predict_window(model, symbol, window))
        except Exception as exc:
            write_event("deeplob_symbol_inference_error", {"symbol": symbol, "error": str(exc)})

    payload = {
        "created_at_utc": now_utc_iso(),
        "model_path": str(MODEL_PATH),
        "signals": [asdict(s) for s in signals],
    }
    write_json(SIGNALS_JSON_PATH, payload)
    pd.DataFrame([asdict(s) for s in signals]).to_csv(SIGNALS_CSV_PATH, index=False)

    print(f"[saved] {SIGNALS_JSON_PATH} signals={len(signals)}")
    print(f"[saved] {SIGNALS_CSV_PATH}")

    return signals


_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:
    global _STOP
    _STOP = True


def run_live() -> None:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print("===== DEEPLOB ORDERBOOK DIRECTION LAYER =====")
    print(f"[model] {MODEL_PATH}")
    print(f"[lob_snapshots] {LOB_SNAPSHOTS_PATH}")
    print(f"[signals] {SIGNALS_JSON_PATH}")
    print(f"[long_only] {LONG_ONLY}")
    print(f"[scan_seconds] {SCAN_SECONDS}")

    while not _STOP:
        try:
            signals = infer_signals()
            for s in signals:
                print(
                    f"[deeplob] {s.symbol} action={s.action} "
                    f"down={s.prob_down:.2f} flat={s.prob_flat:.2f} up={s.prob_up:.2f} "
                    f"reason={s.reason}"
                )
        except Exception as exc:
            print(f"[deeplob] error: {exc}")
            write_event("deeplob_live_error", {"error": str(exc)})

        time.sleep(max(1, SCAN_SECONDS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_model()
        return

    if args.once:
        infer_signals()
        return

    run_live()


if __name__ == "__main__":
    main()
