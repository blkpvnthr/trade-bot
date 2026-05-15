#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gymnasium import spaces
from stable_baselines3 import PPO
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


ET = ZoneInfo("America/New_York")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RL_DATA_DATE", YESTERDAY_TAG)

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "system-data" / f"{DATA_TAG}"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREMARKET_TRADE_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"
MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"
MODEL_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_agent_ppo.zip"
SIGNALS_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_live_signals.json"

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

SCAN_SECONDS = int(os.getenv("SR_RL_SCAN_SECONDS", "10"))
LOOKBACK_DAYS = int(os.getenv("SR_RL_LOOKBACK_DAYS", "90"))
LEVEL_WINDOW = int(os.getenv("SR_RL_LEVEL_WINDOW", "5"))
LEVEL_CLUSTER_PCT = float(os.getenv("SR_RL_LEVEL_CLUSTER_PCT", "0.003"))
ENTRY_DISTANCE_PCT = float(os.getenv("SR_RL_ENTRY_DISTANCE_PCT", "0.004"))
STOP_BEYOND_LEVEL_PCT = float(os.getenv("SR_RL_STOP_BEYOND_LEVEL_PCT", "0.01"))
TAKE_PROFIT_R_MULTIPLE = float(os.getenv("SR_RL_TAKE_PROFIT_R_MULTIPLE", "2.0"))
RISK_PER_TRADE_PCT = float(os.getenv("SR_RL_RISK_PER_TRADE_PCT", "0.005"))
MAX_POSITION_PCT = float(os.getenv("SR_RL_MAX_POSITION_PCT", "0.08"))

TRAIN_TIMESTEPS = int(os.getenv("SR_RL_TRAIN_TIMESTEPS", "50000"))

ACTION_WAIT = 0
ACTION_BUY = 1
ACTION_SELL = 2


@dataclass
class SRDecision:
    symbol: str
    action: str
    price: float
    nearest_support: float | None
    nearest_resistance: float | None
    distance_to_support_pct: float | None
    distance_to_resistance_pct: float | None
    stop_loss: float | None
    take_profit: float | None
    qty: int
    reason: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(x: Any) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_feed() -> DataFeed:
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
    )


def trading_client() -> TradingClient:
    return TradingClient(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        paper=True,
    )


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_symbols_from_plan() -> list[str]:
    plan = load_json(PREMARKET_TRADE_PLAN_PATH, {})
    symbols = []

    for row in plan.get("daily_trade_plans", []):
        symbol = normalize_symbol(row.get("symbol"))
        if symbol:
            symbols.append(symbol)

    return list(dict.fromkeys(symbols))


def market_allows_trading() -> tuple[bool, float, str]:
    ctx = load_json(MARKET_CONTEXT_PATH, {})
    permission = ctx.get("trade_permission", "cautious")
    bias = ctx.get("overall_market_bias", "mixed")

    if permission == "halt_new_buys":
        return False, 0.0, "market_context_halt_new_buys"

    if permission == "cautious" or bias in {"bearish", "risk_off"}:
        return True, 0.5, "market_context_cautious"

    return True, 1.0, "market_context_normal"


def fetch_bars(symbols: list[str], hours: int) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    client = data_client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    timeframe = TimeFrame(hours, TimeFrameUnit.Hour)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=get_feed(),
    )

    df = client.get_stock_bars(req).df
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"])


def detect_levels(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    if df.empty or len(df) < LEVEL_WINDOW * 2 + 1:
        return [], []

    highs = df["high"].astype(float).reset_index(drop=True)
    lows = df["low"].astype(float).reset_index(drop=True)

    supports = []
    resistances = []

    for i in range(LEVEL_WINDOW, len(df) - LEVEL_WINDOW):
        high_window = highs.iloc[i - LEVEL_WINDOW:i + LEVEL_WINDOW + 1]
        low_window = lows.iloc[i - LEVEL_WINDOW:i + LEVEL_WINDOW + 1]

        if highs.iloc[i] == high_window.max():
            resistances.append(float(highs.iloc[i]))

        if lows.iloc[i] == low_window.min():
            supports.append(float(lows.iloc[i]))

    return cluster_levels(supports), cluster_levels(resistances)


def cluster_levels(levels: list[float]) -> list[float]:
    clean = sorted([x for x in levels if x > 0])

    if not clean:
        return []

    clusters: list[list[float]] = []

    for level in clean:
        if not clusters:
            clusters.append([level])
            continue

        center = float(np.mean(clusters[-1]))

        if abs(level - center) / max(center, 1e-9) <= LEVEL_CLUSTER_PCT:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [round(float(np.mean(c)), 4) for c in clusters]


def combined_levels(symbol: str, bars_1h: pd.DataFrame, bars_4h: pd.DataFrame) -> tuple[list[float], list[float]]:
    s1 = bars_1h[bars_1h["symbol"] == symbol]
    s4 = bars_4h[bars_4h["symbol"] == symbol]

    sup1, res1 = detect_levels(s1)
    sup4, res4 = detect_levels(s4)

    return cluster_levels(sup1 + sup4), cluster_levels(res1 + res4)


def nearest_levels(price: float, supports: list[float], resistances: list[float]) -> tuple[float | None, float | None]:
    below = [x for x in supports if x <= price]
    above = [x for x in resistances if x >= price]

    support = max(below) if below else None
    resistance = min(above) if above else None

    return support, resistance


def latest_price(symbol: str, bars_1h: pd.DataFrame) -> float:
    sdf = bars_1h[bars_1h["symbol"] == symbol].sort_values("timestamp")
    if sdf.empty:
        return 0.0
    return float(sdf.iloc[-1]["close"])


def build_features(price: float, support: float | None, resistance: float | None, market_size_mult: float) -> np.ndarray:
    if price <= 0:
        return np.zeros(8, dtype=np.float32)

    ds = (price - support) / price if support else 1.0
    dr = (resistance - price) / price if resistance else 1.0
    range_pct = (resistance - support) / price if support and resistance else 0.0

    near_support = float(support is not None and abs(price - support) / price <= ENTRY_DISTANCE_PCT)
    near_resistance = float(resistance is not None and abs(price - resistance) / price <= ENTRY_DISTANCE_PCT)

    return np.array(
        [
            price,
            support or 0.0,
            resistance or 0.0,
            ds,
            dr,
            range_pct,
            near_support,
            near_resistance,
        ],
        dtype=np.float32,
    ) * np.array([0.001, 0.001, 0.001, 1, 1, 1, 1, 1], dtype=np.float32)


class SRTradingEnv(gym.Env):
    def __init__(self, rows: pd.DataFrame):
        super().__init__()
        self.rows = rows.reset_index(drop=True)
        self.i = 0
        self.position = 0
        self.entry = 0.0
        self.stop = 0.0
        self.tp = 0.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        row = self.rows.iloc[self.i]
        return build_features(
            price=to_float(row["close"]),
            support=to_float(row["support"]) or None,
            resistance=to_float(row["resistance"]) or None,
            market_size_mult=1.0,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.i = 0
        self.position = 0
        self.entry = 0.0
        self.stop = 0.0
        self.tp = 0.0
        return self._obs(), {}

    def step(self, action: int):
        row = self.rows.iloc[self.i]
        price = to_float(row["close"])
        support = to_float(row["support"])
        resistance = to_float(row["resistance"])

        reward = 0.0

        if self.position == 0:
            if action == ACTION_BUY and support > 0 and abs(price - support) / price <= ENTRY_DISTANCE_PCT:
                self.position = 1
                self.entry = price
                self.stop = support * (1.0 - STOP_BEYOND_LEVEL_PCT)
                risk = self.entry - self.stop
                self.tp = self.entry + risk * TAKE_PROFIT_R_MULTIPLE
                reward += 0.05
            elif action == ACTION_SELL and resistance > 0 and abs(price - resistance) / price <= ENTRY_DISTANCE_PCT:
                reward += 0.03
            elif action == ACTION_WAIT:
                reward += 0.01
            else:
                reward -= 0.02

        else:
            if price <= self.stop:
                reward -= 1.0
                self.position = 0
            elif price >= self.tp:
                reward += TAKE_PROFIT_R_MULTIPLE
                self.position = 0
            elif action == ACTION_SELL:
                reward += (price - self.entry) / max(self.entry - self.stop, 0.01)
                self.position = 0
            else:
                reward += (price - self.entry) / max(self.entry, 0.01) * 0.1

        self.i += 1
        terminated = self.i >= len(self.rows) - 1
        truncated = False

        return self._obs(), float(reward), terminated, truncated, {}


def build_training_rows(symbols: list[str]) -> pd.DataFrame:
    bars_1h = fetch_bars(symbols, 1)
    bars_4h = fetch_bars(symbols, 4)

    rows = []

    for symbol in symbols:
        sdf = bars_1h[bars_1h["symbol"] == symbol].sort_values("timestamp")
        supports, resistances = combined_levels(symbol, bars_1h, bars_4h)

        for _, row in sdf.iterrows():
            price = float(row["close"])
            support, resistance = nearest_levels(price, supports, resistances)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": str(row["timestamp"]),
                    "close": price,
                    "support": support or 0.0,
                    "resistance": resistance or 0.0,
                }
            )

    return pd.DataFrame(rows).dropna()


def train_agent() -> None:
    symbols = load_symbols_from_plan()

    if not symbols:
        raise RuntimeError("No symbols found in premarket_trade_plan.json")

    rows = build_training_rows(symbols)

    if len(rows) < 200:
        raise RuntimeError(f"Not enough training rows: {len(rows)}")

    env = SRTradingEnv(rows)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=64, gamma=0.98)
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    model.save(str(MODEL_PATH))

    print(f"[saved] {MODEL_PATH}")


def position_size(symbol: str, price: float, stop: float, size_mult: float) -> int:
    acct = trading_client().get_account()
    equity = to_float(getattr(acct, "equity", 0.0))

    risk_budget = equity * RISK_PER_TRADE_PCT * size_mult
    max_notional = equity * MAX_POSITION_PCT * size_mult
    risk_per_share = abs(price - stop)

    if price <= 0 or risk_per_share <= 0:
        return 0

    qty_by_risk = int(risk_budget / risk_per_share)
    qty_by_notional = int(max_notional / price)

    return max(0, min(qty_by_risk, qty_by_notional))


def submit_order(symbol: str, side: str, qty: int) -> None:
    if qty <= 0:
        return

    client = trading_client()

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )

    client.submit_order(order)


def live_cycle(model: PPO, execute: bool = False) -> list[dict[str, Any]]:
    can_trade, size_mult, market_reason = market_allows_trading()

    symbols = load_symbols_from_plan()
    bars_1h = fetch_bars(symbols, 1)
    bars_4h = fetch_bars(symbols, 4)

    decisions = []

    for symbol in symbols:
        price = latest_price(symbol, bars_1h)
        supports, resistances = combined_levels(symbol, bars_1h, bars_4h)
        support, resistance = nearest_levels(price, supports, resistances)

        obs = build_features(price, support, resistance, size_mult)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        dist_support = abs(price - support) / price if support and price > 0 else None
        dist_resistance = abs(price - resistance) / price if resistance and price > 0 else None

        stop = None
        tp = None
        qty = 0
        action_name = "wait"
        reason = market_reason

        if not can_trade:
            action_name = "wait"

        elif action == ACTION_BUY and support and dist_support is not None and dist_support <= ENTRY_DISTANCE_PCT:
            stop = support * (1.0 - STOP_BEYOND_LEVEL_PCT)
            risk = price - stop
            tp = price + risk * TAKE_PROFIT_R_MULTIPLE
            qty = position_size(symbol, price, stop, size_mult)
            action_name = "buy"
            reason = "price_near_support_rl_buy"

            if execute and qty > 0:
                submit_order(symbol, "buy", qty)

        elif action == ACTION_SELL and resistance and dist_resistance is not None and dist_resistance <= ENTRY_DISTANCE_PCT:
            stop = resistance * (1.0 + STOP_BEYOND_LEVEL_PCT)
            risk = stop - price
            tp = price - risk * TAKE_PROFIT_R_MULTIPLE
            action_name = "sell"
            reason = "price_near_resistance_rl_sell"

            # This layer treats SELL as exit/avoid unless you add shorting.
            # submit_order is intentionally not called for short selling.

        decisions.append(
            {
                "timestamp_utc": now_utc(),
                "symbol": symbol,
                "action": action_name,
                "price": round(price, 4),
                "nearest_support": round(support, 4) if support else None,
                "nearest_resistance": round(resistance, 4) if resistance else None,
                "distance_to_support_pct": round(dist_support, 5) if dist_support is not None else None,
                "distance_to_resistance_pct": round(dist_resistance, 5) if dist_resistance is not None else None,
                "stop_loss": round(stop, 4) if stop else None,
                "take_profit": round(tp, 4) if tp else None,
                "qty": qty,
                "reason": reason,
            }
        )

    payload = {
        "created_at_utc": now_utc(),
        "scan_seconds": SCAN_SECONDS,
        "execute_orders": execute,
        "decisions": decisions,
    }

    SIGNALS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[saved] {SIGNALS_PATH}")

    return decisions


def run_live(execute: bool = False) -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model: {MODEL_PATH}. Run: python sr_rl_agent.py --train")

    model = PPO.load(str(MODEL_PATH))

    while True:
        try:
            decisions = live_cycle(model, execute=execute)
            print(json.dumps(decisions[:5], indent=2))
        except Exception as exc:
            print(f"[sr-rl] cycle failed: {exc}")

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually submit BUY orders. Default only writes signals.")
    args = parser.parse_args()

    if args.train:
        train_agent()
    elif args.live:
        run_live(execute=args.execute)
    else:
        print("Use --train or --live")
        
"""
python sr_rl_agent.py --train
python sr_rl_agent.py --live

To actually submit BUY orders:
python sr_rl_agent.py --live --execute
"""