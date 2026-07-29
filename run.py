#!/usr/bin/env python3
"""
run.py

Opening-range liquidity sweep + retest + engulfing confirmation executor.

Strategy:
    - Read closed TradingView candle / volume-footprint rows from CSV.
    - Build the regular-session opening range from the 9:30 ET candle.
    - LONG setup:
          1) Price sweeps below the 9:30 candle low.
          2) Price reclaims back inside the opening range.
          3) A retest candle forms near the swept low.
          4) The next candle bullish-engulfs the retest candle.
          5) Enter at the CLOSE of the engulfing candle.
          6) Stop = sweep wick low.
          7) Take profit = 3R.
          8) Optional footprint confirmation: retest delta < 0 and engulfing delta > 0.
    - SHORT setup, optional:
          inverse of the long setup. Disabled by default.

Expected default CSV:
    daily_system_outputs/tradingview_volume_footprint.csv

Required columns:
    symbol, timestamp, open, high, low, close, delta

Accepted aliases:
    symbol: ticker, symbol_name, syminfo.ticker
    timestamp: time, datetime, bar_time, candle_time
    delta: volume_delta, footprint_delta, cvd_delta, net_delta, bar_delta

Useful flags:
    python run.py --once
    python run.py --live
    python run.py --footprint-file daily_system_outputs/tradingview_volume_footprint.csv
    python run.py --universe AAPL,NVDA,AMD
    python run.py --allow-shorts
    python run.py --take-profit-r-multiple 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FOOTPRINT_FILE = OUT_DIR / "tradingview_volume_footprint.csv"
JOURNAL_PATH = LOG_DIR / "orb_sweep_engulf_executor_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / "orb_sweep_engulf_equity_curve.csv"
EVENTS_JSONL = LOG_DIR / "orb_sweep_engulf_events.jsonl"
SIGNAL_STATE_PATH = LOG_DIR / "orb_sweep_engulf_signal_state.json"
KILL_SWITCH_PATH = LOG_DIR / "kill_switch_state.json"

DEFAULT_UNIVERSE = [
    "QQQ", "TQQQ", "SQQQ", "SPY", "DIA", "IWM", "XLK", "SOXX", "SMH",
    "AMD", "NVDA", "AVGO", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
    "PLTR", "COIN", "MARA", "RIOT", "RGTI", "QBTS", "QUBT", "IONQ", "SOUN",
    "OKLO", "APLD", "ALAB", "ARM",
]

POLL_SECONDS = 30
COOLDOWN_SECONDS = 8
ORDER_WAIT_SECONDS = 20
HEARTBEAT_SECONDS = 60

MIN_BUYING_POWER = 250.0
MIN_NOTIONAL_TO_TRADE = 100.0
MIN_CASH_BUFFER = 1_000.0
DEFAULT_CASH_BUFFER_PCT = 0.08

MAX_GROSS_EXPOSURE = 0.92
MAX_POSITION_WEIGHT = 0.18
MAX_SINGLE_ORDER_NOTIONAL_PCT = 0.08
MAX_ORDER_DOLLARS = 15_000.0
MAX_SINGLE_ORDER_QTY = 10_000

RISK_PER_TRADE_PCT = 0.005
TAKE_PROFIT_R_MULTIPLE = 3.0

MAX_DAILY_LOSS_PCT = 0.02
MAX_DAILY_LOSS_DOLLARS = 2_000.0
DAILY_DRAWDOWN_LIMIT = -0.025
# Default sized for the ~$13K paper account this system targets (see CLAUDE.md).
# Override per-deployment via the KILL_SWITCH_EQUITY_FLOOR env var (already defined
# in instances/*/.env* templates) rather than editing this constant directly.
KILL_SWITCH_EQUITY_FLOOR = float(os.getenv("KILL_SWITCH_EQUITY_FLOOR", "11700"))


@dataclass
class BrokerPosition:
    symbol: str = ""
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
class ORBSweepEngulfSignal:
    symbol: str
    side: Literal["long", "short"]
    trade_date: str
    timestamp: str
    opening_timestamp: str
    opening_high: float
    opening_low: float
    sweep_price: float
    retest_timestamp: str
    engulf_timestamp: str
    retest_delta: float
    engulf_delta: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_per_share: float
    reward_per_share: float
    retest_open: float
    retest_high: float
    retest_low: float
    retest_close: float
    engulf_open: float
    engulf_high: float
    engulf_low: float
    engulf_close: float


@dataclass
class ExecutorConfig:
    paper: bool = True
    live: bool = False
    poll_once: bool = False
    poll_seconds: int = POLL_SECONDS
    footprint_file: Path = DEFAULT_FOOTPRINT_FILE
    universe: list[str] | None = None

    session_timezone: str = "America/New_York"
    opening_candle_time: str = "09:30"
    max_opening_candle_minutes_late: int = 5
    retest_tolerance_pct: float = 0.0015
    require_delta_flip: bool = True
    allow_shorts: bool = False

    require_market_open: bool = True
    wait_for_market_open: bool = False
    allow_closed_market: bool = False
    max_wait_for_open_minutes: int = 180
    cancel_open_orders_first: bool = False

    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT
    min_cash_buffer: float = MIN_CASH_BUFFER
    min_buying_power: float = MIN_BUYING_POWER
    min_notional_to_trade: float = MIN_NOTIONAL_TO_TRADE
    max_gross_exposure: float = MAX_GROSS_EXPOSURE
    max_position_weight: float = MAX_POSITION_WEIGHT
    max_single_order_notional_pct: float = MAX_SINGLE_ORDER_NOTIONAL_PCT
    max_order_dollars: float = MAX_ORDER_DOLLARS
    max_single_order_qty: int = MAX_SINGLE_ORDER_QTY
    risk_per_trade_pct: float = RISK_PER_TRADE_PCT
    take_profit_r_multiple: float = TAKE_PROFIT_R_MULTIPLE
    cooldown_seconds: int = COOLDOWN_SECONDS

    enable_kill_switch: bool = True
    daily_drawdown_limit: float = DAILY_DRAWDOWN_LIMIT
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT
    max_daily_loss_dollars: float = MAX_DAILY_LOSS_DOLLARS
    kill_switch_equity_floor: float = KILL_SWITCH_EQUITY_FLOOR

    order_wait_seconds: int = ORDER_WAIT_SECONDS
    time_in_force: TimeInForce = TimeInForce.DAY
    order_tag_prefix: str = "orb_sweep_engulf_3r"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return now_utc().strftime("%Y%m%d_%H%M%S")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_symbol(value: Any) -> str:
    return str(value).upper().strip().replace(".", "-")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def round_price(value: float) -> float:
    return round(float(value), 2) if value >= 1 else round(float(value), 4)


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp_utc": now_utc().isoformat(), "event": event, "payload": payload or {}}, default=json_default) + "\n")


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


class CsvJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = [
            "timestamp_utc", "cycle_id", "event", "market_open", "equity_before", "equity_after",
            "cash_before", "cash_after", "buying_power_before", "buying_power_after",
            "gross_exposure_before", "gross_exposure_after", "symbol", "side", "qty",
            "opening_high", "opening_low", "sweep_price", "entry_price", "stop_loss", "take_profit",
            "risk_per_share", "reward_per_share", "risk_multiple", "retest_delta", "engulf_delta",
            "signal_timestamp", "estimated_notional", "order_id", "order_status", "filled_qty",
            "filled_avg_price", "message",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def append(self, row: dict[str, object]) -> None:
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow({k: row.get(k, "") for k in self.columns})


class AlpacaBroker:
    def __init__(self, env_path: Path, paper: bool = True):
        load_dotenv(env_path)
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not api_key or not secret_key:
            env_exists = env_path.exists()
            hint = (
                f"  .env exists at {env_path} but is missing the required keys."
                if env_exists else
                f"  No .env file found at {env_path}.\n"
                f"  Create one by copying an example: cp .env.bot01.example .env\n"
                f"  Then fill in APCA_API_KEY_ID and APCA_API_SECRET_KEY from https://alpaca.markets"
            )
            raise RuntimeError(
                f"Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY.\n{hint}"
            )
        self.trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)

    def get_account_snapshot(self) -> AccountSnapshot:
        acct = self.trading.get_account()
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
            if qty == 0:
                continue
            positions.append(
                BrokerPosition(
                    symbol=safe_symbol(getattr(pos, "symbol", "")),
                    qty=qty,
                    market_value=abs(to_float(getattr(pos, "market_value", 0.0))),
                    avg_entry_price=to_float(getattr(pos, "avg_entry_price", 0.0)),
                    current_price=to_float(getattr(pos, "current_price", 0.0)),
                    unrealized_pl=to_float(getattr(pos, "unrealized_pl", 0.0)),
                    unrealized_plpc=to_float(getattr(pos, "unrealized_plpc", 0.0)),
                )
            )
        return positions

    def list_open_orders(self) -> list:
        try:
            return self.trading.get_orders()
        except Exception:
            return []

    def has_open_order_for_symbol(self, symbol: str) -> bool:
        symbol = safe_symbol(symbol)
        open_statuses = {
            OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED, OrderStatus.ACCEPTED_FOR_BIDDING, OrderStatus.CALCULATED,
        }
        for order in self.list_open_orders():
            try:
                if safe_symbol(order.symbol) == symbol and order.status in open_statuses:
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

    def submit_market_bracket_order(
        self,
        symbol: str,
        side: Literal["long", "short"],
        qty: int,
        client_order_id: str,
        tif: TimeInForce,
        take_profit_price: float,
        stop_loss_price: float,
    ):
        alpaca_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=safe_symbol(symbol),
            qty=int(qty),
            side=alpaca_side,
            type=OrderType.MARKET,
            time_in_force=tif,
            client_order_id=client_order_id,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round_price(take_profit_price)),
            stop_loss=StopLossRequest(stop_price=round_price(stop_loss_price)),
        )
        return self.trading.submit_order(order_data=order)

    def get_order_by_id(self, order_id: str):
        return self.trading.get_order_by_id(order_id)


class ORBSweepEngulfSignalEngine:
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.tz = ZoneInfo(config.session_timezone)

    def _parse_opening_time(self) -> dtime:
        hh, mm = [int(x) for x in self.config.opening_candle_time.split(":")]
        return dtime(hh, mm)

    def _read_footprint(self) -> pd.DataFrame:
        path = self.config.footprint_file
        if not path.exists():
            print(f"[footprint] missing file: {path}")
            return pd.DataFrame()
        if path.stat().st_size == 0:
            print(f"[footprint] empty file: {path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df.columns = [str(c).strip().lower() for c in df.columns]
        rename_map: dict[str, str] = {}

        for candidate in ["ticker", "symbol_name", "syminfo.ticker"]:
            if candidate in df.columns and "symbol" not in df.columns:
                rename_map[candidate] = "symbol"
        for candidate in ["time", "datetime", "bar_time", "candle_time"]:
            if candidate in df.columns and "timestamp" not in df.columns:
                rename_map[candidate] = "timestamp"
        for candidate in ["volume_delta", "footprint_delta", "cvd_delta", "net_delta", "bar_delta"]:
            if candidate in df.columns and "delta" not in df.columns:
                rename_map[candidate] = "delta"

        df = df.rename(columns=rename_map)

        required = {"symbol", "timestamp", "open", "high", "low", "close", "delta"}
        missing = sorted(required - set(df.columns))
        if missing:
            print(f"[footprint] missing required columns={missing}; expected symbol,timestamp,open,high,low,close,delta")
            return pd.DataFrame()

        df["symbol"] = df["symbol"].map(safe_symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        for col in ["open", "high", "low", "close", "delta"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "delta"])
        df = df[(df["symbol"] != "") & (df["close"] > 0) & (df["high"] >= df["low"])]

        universe = {safe_symbol(x) for x in (self.config.universe or DEFAULT_UNIVERSE)}
        if universe:
            df = df[df["symbol"].isin(universe)].copy()

        if df.empty:
            return df

        df["timestamp_et"] = df["timestamp"].dt.tz_convert(self.config.session_timezone)
        df["trade_date"] = df["timestamp_et"].dt.date.astype(str)
        return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    def _signal_key(self, signal: ORBSweepEngulfSignal) -> str:
        return f"{signal.symbol}|{signal.side}|{signal.engulf_timestamp}"

    def load_signal_state(self) -> dict[str, Any]:
        if not SIGNAL_STATE_PATH.exists():
            return {"processed_signal_keys": []}
        try:
            return json.loads(SIGNAL_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"processed_signal_keys": []}

    def save_signal_state(self, state: dict[str, Any]) -> None:
        SIGNAL_STATE_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")

    def mark_processed(self, signal: ORBSweepEngulfSignal) -> None:
        state = self.load_signal_state()
        keys = list(state.get("processed_signal_keys", []))
        key = self._signal_key(signal)
        if key not in keys:
            keys.append(key)
        state["processed_signal_keys"] = keys[-2000:]
        state["updated_at"] = now_utc().isoformat()
        self.save_signal_state(state)

    def _opening_candle(self, g: pd.DataFrame) -> pd.Series | None:
        opening_time = self._parse_opening_time()
        opening_dt = pd.Timestamp.combine(pd.Timestamp(g["timestamp_et"].iloc[-1]).date(), opening_time).tz_localize(self.config.session_timezone)
        latest_allowed = opening_dt + pd.Timedelta(minutes=self.config.max_opening_candle_minutes_late)
        candidates = g[(g["timestamp_et"] >= opening_dt) & (g["timestamp_et"] < latest_allowed)]
        if candidates.empty:
            return None
        return candidates.iloc[0]

    def _make_long_signal(self, symbol: str, trade_date: str, opening: pd.Series, session: pd.DataFrame) -> ORBSweepEngulfSignal | None:
        if len(session) < 4:
            return None

        retest = session.iloc[-2]
        engulf = session.iloc[-1]
        prior = session.iloc[:-1]
        opening_low = float(opening["low"])
        opening_high = float(opening["high"])
        tolerance = opening_low * self.config.retest_tolerance_pct

        swept = prior[prior["low"] < opening_low]
        if swept.empty:
            return None

        sweep_idx = swept["low"].idxmin()
        sweep_price = float(swept.loc[sweep_idx, "low"])
        after_sweep = session.loc[sweep_idx:]
        if after_sweep[after_sweep["close"] > opening_low].empty:
            return None

        retest_near_swept_low = float(retest["low"]) <= opening_low + tolerance
        retest_accepted_inside = float(retest["close"]) >= opening_low
        bullish_engulf = float(engulf["close"]) > float(retest["high"]) and float(engulf["close"]) > float(engulf["open"])

        if not (retest_near_swept_low and retest_accepted_inside and bullish_engulf):
            return None

        if self.config.require_delta_flip and not (float(retest["delta"]) < 0 and float(engulf["delta"]) > 0):
            return None

        entry = float(engulf["close"])
        stop = sweep_price
        risk = entry - stop
        if risk <= 0:
            return None

        tp = entry + self.config.take_profit_r_multiple * risk

        return ORBSweepEngulfSignal(
            symbol=symbol, side="long", trade_date=trade_date,
            timestamp=pd.Timestamp(engulf["timestamp"]).isoformat(),
            opening_timestamp=pd.Timestamp(opening["timestamp"]).isoformat(),
            opening_high=round_price(opening_high), opening_low=round_price(opening_low),
            sweep_price=round_price(sweep_price),
            retest_timestamp=pd.Timestamp(retest["timestamp"]).isoformat(),
            engulf_timestamp=pd.Timestamp(engulf["timestamp"]).isoformat(),
            retest_delta=float(retest["delta"]), engulf_delta=float(engulf["delta"]),
            entry_price=round_price(entry), stop_loss=round_price(stop), take_profit=round_price(tp),
            risk_per_share=round_price(risk), reward_per_share=round_price(tp - entry),
            retest_open=round_price(retest["open"]), retest_high=round_price(retest["high"]),
            retest_low=round_price(retest["low"]), retest_close=round_price(retest["close"]),
            engulf_open=round_price(engulf["open"]), engulf_high=round_price(engulf["high"]),
            engulf_low=round_price(engulf["low"]), engulf_close=round_price(engulf["close"]),
        )

    def _make_short_signal(self, symbol: str, trade_date: str, opening: pd.Series, session: pd.DataFrame) -> ORBSweepEngulfSignal | None:
        if not self.config.allow_shorts or len(session) < 4:
            return None

        retest = session.iloc[-2]
        engulf = session.iloc[-1]
        prior = session.iloc[:-1]
        opening_low = float(opening["low"])
        opening_high = float(opening["high"])
        tolerance = opening_high * self.config.retest_tolerance_pct

        swept = prior[prior["high"] > opening_high]
        if swept.empty:
            return None

        sweep_idx = swept["high"].idxmax()
        sweep_price = float(swept.loc[sweep_idx, "high"])
        after_sweep = session.loc[sweep_idx:]
        if after_sweep[after_sweep["close"] < opening_high].empty:
            return None

        retest_near_swept_high = float(retest["high"]) >= opening_high - tolerance
        retest_accepted_inside = float(retest["close"]) <= opening_high
        bearish_engulf = float(engulf["close"]) < float(retest["low"]) and float(engulf["close"]) < float(engulf["open"])

        if not (retest_near_swept_high and retest_accepted_inside and bearish_engulf):
            return None

        if self.config.require_delta_flip and not (float(retest["delta"]) > 0 and float(engulf["delta"]) < 0):
            return None

        entry = float(engulf["close"])
        stop = sweep_price
        risk = stop - entry
        if risk <= 0:
            return None

        tp = entry - self.config.take_profit_r_multiple * risk

        return ORBSweepEngulfSignal(
            symbol=symbol, side="short", trade_date=trade_date,
            timestamp=pd.Timestamp(engulf["timestamp"]).isoformat(),
            opening_timestamp=pd.Timestamp(opening["timestamp"]).isoformat(),
            opening_high=round_price(opening_high), opening_low=round_price(opening_low),
            sweep_price=round_price(sweep_price),
            retest_timestamp=pd.Timestamp(retest["timestamp"]).isoformat(),
            engulf_timestamp=pd.Timestamp(engulf["timestamp"]).isoformat(),
            retest_delta=float(retest["delta"]), engulf_delta=float(engulf["delta"]),
            entry_price=round_price(entry), stop_loss=round_price(stop), take_profit=round_price(tp),
            risk_per_share=round_price(risk), reward_per_share=round_price(entry - tp),
            retest_open=round_price(retest["open"]), retest_high=round_price(retest["high"]),
            retest_low=round_price(retest["low"]), retest_close=round_price(retest["close"]),
            engulf_open=round_price(engulf["open"]), engulf_high=round_price(engulf["high"]),
            engulf_low=round_price(engulf["low"]), engulf_close=round_price(engulf["close"]),
        )

    def find_new_signals(self) -> list[ORBSweepEngulfSignal]:
        df = self._read_footprint()
        if df.empty:
            return []

        state = self.load_signal_state()
        processed = set(state.get("processed_signal_keys", []))
        signals: list[ORBSweepEngulfSignal] = []

        for (symbol, trade_date), g in df.groupby(["symbol", "trade_date"], sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            if trade_date != df["trade_date"].max():
                continue

            opening = self._opening_candle(g)
            if opening is None:
                continue

            session = g[g["timestamp"] > opening["timestamp"]].copy()
            if session.empty:
                continue

            long_signal = self._make_long_signal(symbol, trade_date, opening, session)
            short_signal = self._make_short_signal(symbol, trade_date, opening, session)

            for signal in [long_signal, short_signal]:
                if signal is not None and self._signal_key(signal) not in processed:
                    signals.append(signal)

        return signals


class ORBSweepEngulfExecutor:
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.journal = CsvJournal(JOURNAL_PATH)
        self.broker = AlpacaBroker(ENV_PATH, paper=config.paper)
        self.signal_engine = ORBSweepEngulfSignalEngine(config)
        self.last_order_time = 0.0
        self.last_heartbeat_time = 0.0
        self.account_history: list[dict[str, float | str]] = []
        # In-session dedup guard: symbols ordered this session (keyed by date).
        # Alpaca position API has latency; relying solely on it causes duplicate
        # orders within the same poll cycle (observed: 6x CVE/BITO buys on 2026-05-05).
        self._session_ordered: dict[str, set[str]] = {}  # date -> {symbol}

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_order_time) < self.config.cooldown_seconds

    def _mark_order(self) -> None:
        self.last_order_time = time.time()

    def _session_already_ordered(self, symbol: str) -> bool:
        today = now_utc().date().isoformat()
        return symbol in self._session_ordered.get(today, set())

    def _record_session_order(self, symbol: str) -> None:
        today = now_utc().date().isoformat()
        self._session_ordered.setdefault(today, set()).add(symbol)
        # Prune stale dates so this dict doesn't grow across midnight
        self._session_ordered = {k: v for k, v in self._session_ordered.items() if k >= today}

    def _record_account_snapshot(self, label: str, cycle_id: str) -> dict[str, float | str]:
        acct = self.broker.get_account_snapshot()
        snapshot = {
            "label": label, "cycle_id": cycle_id, "timestamp_utc": now_utc().isoformat(),
            "buying_power": float(acct.buying_power), "equity": float(acct.equity),
            "cash": float(acct.cash), "portfolio_value": float(acct.portfolio_value),
        }
        self.account_history.append(snapshot)
        return snapshot

    def _write_account_equity_curve(self) -> None:
        if self.account_history:
            pd.DataFrame(self.account_history).to_csv(EQUITY_CURVE_PATH, index=False)

    def _positions_by_symbol(self) -> dict[str, BrokerPosition]:
        return {safe_symbol(p.symbol): p for p in self.broker.get_all_open_positions() if p.symbol}

    def _gross_exposure(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        return sum(abs(float(p.market_value)) for p in self.broker.get_all_open_positions()) / equity

    def _cash_buffer(self, equity: float) -> float:
        return max(self.config.min_cash_buffer, equity * self.config.cash_buffer_pct)

    def _max_order_notional(self, equity: float) -> float:
        return min(self.config.max_order_dollars, equity * self.config.max_single_order_notional_pct)

    def _market_is_open_or_wait(self) -> bool:
        if self.config.allow_closed_market or not self.config.require_market_open:
            print("[clock] market-open guard disabled.")
            return True

        info = self.broker.get_clock_info()
        print(f"[clock] is_open={info['is_open']} timestamp={info['timestamp']} next_open={info['next_open']} next_close={info['next_close']}")
        write_event("market_clock_checked", info)

        if info["is_open"]:
            return True
        if not self.config.wait_for_market_open:
            print("[clock] market is closed. Execution skipped.")
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
        print("[clock] market did not open within wait limit. Execution skipped.")
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
            state = {"date": today, "start_equity": equity, "kill_triggered": False, "triggered_at": "", "reason": ""}
            self._write_kill_state(state)

        start_equity = to_float(state.get("start_equity"), equity)
        pnl = equity - start_equity
        dd_pct = pnl / max(start_equity, 1e-9)
        blocked = False
        reason = "ok"

        if state.get("kill_triggered"):
            blocked = True
            reason = str(state.get("reason", "kill_switch_already_triggered"))
        if equity <= self.config.kill_switch_equity_floor:
            blocked = True
            reason = "kill_switch_equity_floor"
        if dd_pct <= self.config.daily_drawdown_limit:
            blocked = True
            reason = "daily_drawdown_limit"
        if pnl <= -abs(self.config.max_daily_loss_dollars):
            blocked = True
            reason = "max_daily_loss_dollars"
        if (-dd_pct) >= abs(self.config.max_daily_loss_pct):
            blocked = True
            reason = "max_daily_loss_pct"

        print(f"[kill-switch] start=${start_equity:,.2f} equity=${equity:,.2f} pnl=${pnl:,.2f} return={dd_pct:.2%} status={reason}")

        if blocked:
            state.update({"kill_triggered": True, "triggered_at": now_utc().isoformat(), "reason": reason, "current_equity": equity, "daily_pnl": pnl, "daily_return": dd_pct})
            self._write_kill_state(state)
            write_event("kill_switch_triggered", state)
            return False, reason, dd_pct

        return True, reason, dd_pct

    def _position_weight(self, symbol: str, equity: float) -> float:
        pos = self._positions_by_symbol().get(safe_symbol(symbol))
        if not pos or equity <= 0:
            return 0.0
        return abs(float(pos.market_value)) / equity

    def _size_signal(self, signal: ORBSweepEngulfSignal, acct: dict[str, float | str]) -> tuple[int, float, str]:
        equity = float(acct["equity"])
        cash = float(acct["cash"])
        buying_power = float(acct["buying_power"])
        gross = self._gross_exposure(equity)

        if buying_power < self.config.min_buying_power:
            return 0, 0.0, "buying_power_below_minimum"
        if signal.risk_per_share <= 0:
            return 0, 0.0, "invalid_risk_per_share"

        existing_weight = self._position_weight(signal.symbol, equity)
        if existing_weight >= self.config.max_position_weight:
            return 0, 0.0, "position_weight_already_at_or_above_max"

        risk_budget = equity * self.config.risk_per_trade_pct
        risk_qty = int(risk_budget // signal.risk_per_share)
        max_order_notional = self._max_order_notional(equity)
        max_position_room = max(0.0, (self.config.max_position_weight - existing_weight) * equity)
        gross_room = max(0.0, (self.config.max_gross_exposure - gross) * equity)
        cash_room = max(0.0, cash - self._cash_buffer(equity))
        notional_cap = min(max_order_notional, max_position_room, gross_room, cash_room, buying_power)
        notional_qty = int(notional_cap // signal.entry_price)

        qty = min(risk_qty, notional_qty, self.config.max_single_order_qty)
        est_notional = qty * signal.entry_price

        if qty <= 0:
            return 0, 0.0, "sized_to_zero_after_risk_caps"
        if est_notional < self.config.min_notional_to_trade:
            return 0, est_notional, "below_min_notional_to_trade"
        return int(qty), float(est_notional), "ok"

    def _poll_fill(self, order_id: str) -> dict[str, Any]:
        last_order = None
        for _ in range(max(1, int(self.config.order_wait_seconds // 2))):
            try:
                last_order = self.broker.get_order_by_id(order_id)
            except Exception as exc:
                return {"fill_status": "fill_lookup_error", "filled_qty": 0.0, "filled_avg_price": 0.0, "message": str(exc)}
            status = str(getattr(last_order, "status", ""))
            if status.lower() in {"filled", "partially_filled", "canceled", "rejected", "expired"}:
                break
            time.sleep(2.0)

        if last_order is None:
            return {"fill_status": "unknown", "filled_qty": 0.0, "filled_avg_price": 0.0}

        return {
            "fill_status": str(getattr(last_order, "status", "")),
            "filled_qty": to_float(getattr(last_order, "filled_qty", 0.0)),
            "filled_avg_price": to_float(getattr(last_order, "filled_avg_price", 0.0)),
        }

    def _journal_signal(
        self,
        cycle_id: str,
        event: str,
        signal: ORBSweepEngulfSignal,
        qty: int,
        est_notional: float,
        acct_before: dict[str, float | str],
        acct_after: dict[str, float | str],
        message: str,
        order_id: str = "",
        order_status: str = "",
        fill_info: dict[str, Any] | None = None,
        market_open: bool = True,
    ) -> None:
        fill_info = fill_info or {}
        self.journal.append({
            "timestamp_utc": now_utc().isoformat(), "cycle_id": cycle_id, "event": event,
            "market_open": market_open, "equity_before": acct_before.get("equity", ""),
            "equity_after": acct_after.get("equity", ""), "cash_before": acct_before.get("cash", ""),
            "cash_after": acct_after.get("cash", ""), "buying_power_before": acct_before.get("buying_power", ""),
            "buying_power_after": acct_after.get("buying_power", ""), "gross_exposure_before": acct_before.get("gross_exposure", ""),
            "gross_exposure_after": acct_after.get("gross_exposure", ""), "symbol": signal.symbol,
            "side": signal.side, "qty": qty, "opening_high": signal.opening_high, "opening_low": signal.opening_low,
            "sweep_price": signal.sweep_price, "entry_price": signal.entry_price, "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit, "risk_per_share": signal.risk_per_share, "reward_per_share": signal.reward_per_share,
            "risk_multiple": self.config.take_profit_r_multiple, "retest_delta": signal.retest_delta,
            "engulf_delta": signal.engulf_delta, "signal_timestamp": signal.timestamp, "estimated_notional": est_notional,
            "order_id": order_id, "order_status": order_status, "filled_qty": fill_info.get("filled_qty", ""),
            "filled_avg_price": fill_info.get("filled_avg_price", ""), "message": message,
        })

    def _execute_signal(self, signal: ORBSweepEngulfSignal, cycle_id: str) -> dict[str, Any]:
        acct_before = self._record_account_snapshot("before_signal", cycle_id)
        acct_before["gross_exposure"] = self._gross_exposure(float(acct_before["equity"]))

        if self._in_cooldown():
            acct_after = self._record_account_snapshot("after_skip", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "SKIP", signal, 0, 0.0, acct_before, acct_after, "cooldown_active")
            return {"symbol": signal.symbol, "side": signal.side, "status": "skipped_cooldown", "message": "cooldown_active"}

        # In-session dedup: fastest guard, no API call required.
        if self._session_already_ordered(signal.symbol):
            acct_after = self._record_account_snapshot("after_skip", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "SKIP", signal, 0, 0.0, acct_before, acct_after, "session_already_ordered")
            self.signal_engine.mark_processed(signal)
            return {"symbol": signal.symbol, "side": signal.side, "status": "skipped_session_dedup", "message": "session_already_ordered"}

        positions = self._positions_by_symbol()
        if signal.symbol in positions and abs(positions[signal.symbol].qty) > 0:
            acct_after = self._record_account_snapshot("after_skip", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "SKIP", signal, 0, 0.0, acct_before, acct_after, "already_in_position")
            self.signal_engine.mark_processed(signal)
            return {"symbol": signal.symbol, "side": signal.side, "status": "skipped_position_exists", "message": "already_in_position"}

        if self.broker.has_open_order_for_symbol(signal.symbol):
            acct_after = self._record_account_snapshot("after_skip", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "SKIP", signal, 0, 0.0, acct_before, acct_after, "existing_open_order")
            self.signal_engine.mark_processed(signal)
            return {"symbol": signal.symbol, "side": signal.side, "status": "skipped_open_order", "message": "existing_open_order"}

        qty, est_notional, size_reason = self._size_signal(signal, acct_before)
        if qty <= 0:
            acct_after = self._record_account_snapshot("after_skip", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "SKIP", signal, qty, est_notional, acct_before, acct_after, size_reason)
            self.signal_engine.mark_processed(signal)
            return {"symbol": signal.symbol, "side": signal.side, "status": "skipped_size", "message": size_reason}

        client_order_id = f"{self.config.order_tag_prefix}_{signal.side}_{run_id()}_{signal.symbol}"[:48]

        try:
            submitted = self.broker.submit_market_bracket_order(
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                client_order_id=client_order_id,
                tif=self.config.time_in_force,
                take_profit_price=signal.take_profit,
                stop_loss_price=signal.stop_loss,
            )
            order_id = str(getattr(submitted, "id", ""))
            status = str(getattr(submitted, "status", "submitted"))
            print(f"[submitted] {signal.side.upper()} BRACKET {qty} {signal.symbol} entry_ref={signal.entry_price:.4f} stop={signal.stop_loss:.4f} tp={signal.take_profit:.4f} status={status} id={order_id}")

            self._mark_order()
            self._record_session_order(signal.symbol)
            self.signal_engine.mark_processed(signal)
            fill_info = self._poll_fill(order_id)

            acct_after = self._record_account_snapshot("after_order", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))

            result = {
                "timestamp_utc": now_utc().isoformat(), "cycle_id": cycle_id, "symbol": signal.symbol,
                "side": signal.side, "order_class": "bracket", "qty": qty, "entry_reference": signal.entry_price,
                "stop_loss": signal.stop_loss, "take_profit": signal.take_profit, "risk_per_share": signal.risk_per_share,
                "risk_multiple": self.config.take_profit_r_multiple, "estimated_notional": est_notional,
                "opening_high": signal.opening_high, "opening_low": signal.opening_low, "sweep_price": signal.sweep_price,
                "retest_delta": signal.retest_delta, "engulf_delta": signal.engulf_delta,
                "signal_timestamp": signal.timestamp, "status": status, "order_id": order_id, **fill_info,
            }

            self._journal_signal(cycle_id, "ORDER", signal, qty, est_notional, acct_before, acct_after, "orb_sweep_retest_engulf_entry_at_close_3r", order_id, status, fill_info)
            write_event("orb_sweep_engulf_bracket_order_submitted", result)
            return result

        except APIError as exc:
            message = str(exc)
            print(f"[error] APIError {signal.side.upper()} BRACKET {qty} {signal.symbol}: {message}")
            acct_after = self._record_account_snapshot("after_error", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "ERROR", signal, qty, est_notional, acct_before, acct_after, message)
            write_event("orb_sweep_engulf_order_api_error", {"symbol": signal.symbol, "side": signal.side, "error": message})
            return {"symbol": signal.symbol, "side": signal.side, "status": "api_error", "message": message}

        except Exception as exc:
            message = str(exc)
            print(f"[error] submit failed {signal.side.upper()} BRACKET {qty} {signal.symbol}: {message}")
            acct_after = self._record_account_snapshot("after_error", cycle_id)
            acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
            self._journal_signal(cycle_id, "ERROR", signal, qty, est_notional, acct_before, acct_after, message)
            write_event("orb_sweep_engulf_order_error", {"symbol": signal.symbol, "side": signal.side, "error": message})
            return {"symbol": signal.symbol, "side": signal.side, "status": "error", "message": message}

    def run_cycle(self) -> pd.DataFrame:
        cycle_id = run_id()
        acct_start = self._record_account_snapshot("cycle_start", cycle_id)
        equity = float(acct_start["equity"])
        acct_start["gross_exposure"] = self._gross_exposure(equity)

        market_open = self._market_is_open_or_wait()
        if not market_open:
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "cycle_id": cycle_id, "status": "skipped_market_closed", "message": "market is closed"}])
            save_df(results, LOG_DIR / f"orb_sweep_engulf_order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return results

        allowed, kill_reason, daily_return = self._kill_switch_allows_execution(equity)
        if not allowed:
            print(f"[kill-switch] execution blocked: {kill_reason}")
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "cycle_id": cycle_id, "status": "skipped_kill_switch", "message": kill_reason, "daily_return": daily_return}])
            save_df(results, LOG_DIR / f"orb_sweep_engulf_order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return results

        if self.config.cancel_open_orders_first:
            print("[orders] canceling open orders")
            self.broker.cancel_open_orders()

        signals = self.signal_engine.find_new_signals()
        if not signals:
            print("[signal] no new ORB sweep/retest/engulf signals")
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "cycle_id": cycle_id, "status": "no_signal", "message": "no new ORB sweep/retest/engulf signals"}])
            save_df(results, LOG_DIR / f"orb_sweep_engulf_order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return results

        print("\n===== ORB SWEEP / RETEST / ENGULF SIGNALS =====")
        signal_df = pd.DataFrame([s.__dict__ for s in signals])
        print(signal_df[["symbol", "side", "timestamp", "opening_high", "opening_low", "sweep_price", "entry_price", "stop_loss", "take_profit", "risk_per_share", "retest_delta", "engulf_delta"]].to_string(index=False))
        save_df(signal_df, LOG_DIR / f"orb_sweep_engulf_signals_{cycle_id}.csv")

        results = []
        for signal in signals:
            results.append(self._execute_signal(signal, cycle_id))
            time.sleep(0.35)

        results_df = pd.DataFrame(results)
        save_df(results_df, LOG_DIR / f"orb_sweep_engulf_order_results_{cycle_id}.csv")

        acct_after = self._record_account_snapshot("cycle_end", cycle_id)
        acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
        self._write_account_equity_curve()

        save_json({
            "cycle_id": cycle_id, "timestamp_utc": now_utc().isoformat(), "paper": self.config.paper,
            "mode": "ORB_SWEEP_RETEST_ENGULF_3R_BRACKET_EXECUTION",
            "footprint_file": str(self.config.footprint_file), "equity_before": equity,
            "equity_after": acct_after["equity"], "signals_found": len(signals), "results_rows": len(results_df),
            "entry": "close_of_engulfing_candle", "target": f"{self.config.take_profit_r_multiple}R",
        }, LOG_DIR / f"orb_sweep_engulf_execution_summary_{cycle_id}.json")

        return results_df

    def _print_heartbeat(self) -> None:
        now_ts = time.time()
        if now_ts - self.last_heartbeat_time < HEARTBEAT_SECONDS:
            return
        acct = self.broker.get_account_snapshot()
        gross = self._gross_exposure(acct.equity)
        print(f"[executor] heartbeat | equity=${acct.equity:,.2f} | cash=${acct.cash:,.2f} | buying_power=${acct.buying_power:,.2f} | gross={gross:.2%}")
        self.last_heartbeat_time = now_ts

    def run(self) -> None:
        print("[executor] ORB sweep/retest/engulf 3R bracket executor starting...")
        print(f"[executor] footprint_file={self.config.footprint_file}")
        print(f"[executor] journal={JOURNAL_PATH}")
        print(f"[executor] equity_curve={EQUITY_CURVE_PATH}")
        print(f"[executor] paper={self.config.paper}")
        print("[executor] mode=ORB_SWEEP_RETEST_ENGULF_3R_BRACKET_EXECUTION")
        print("[executor] opening_range=9:30 ET candle")
        print("[executor] entry=close_of_engulfing_candle")
        print("[executor] long=enabled short=" + ("enabled" if self.config.allow_shorts else "disabled"))
        print(f"[executor] take_profit={self.config.take_profit_r_multiple}R")

        if self.config.poll_once:
            self.run_cycle()
            return

        print("[executor] run_policy=FOREVER_UNTIL_INTERRUPTED")
        while True:
            try:
                self.run_cycle()
                self._print_heartbeat()
            except KeyboardInterrupt:
                print("[executor] interrupted by user")
                raise
            except Exception as exc:
                print(f"[executor] cycle error: {exc}")
                write_event("cycle_error", {"error": str(exc)})
            time.sleep(max(1, int(self.config.poll_seconds)))


def get_trading_client(config: ExecutorConfig | None = None) -> TradingClient:
    config = config or ExecutorConfig()
    return AlpacaBroker(ENV_PATH, paper=config.paper).trading


def get_account_snapshot(client: TradingClient) -> AccountSnapshot:
    acct = client.get_account()
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
        rows.append({
            "symbol": safe_symbol(getattr(p, "symbol", "")),
            "qty": to_float(getattr(p, "qty", 0.0)),
            "avg_entry_price": to_float(getattr(p, "avg_entry_price", 0.0)),
            "current_price": to_float(getattr(p, "current_price", 0.0)),
            "market_value": to_float(getattr(p, "market_value", 0.0)),
            "cost_basis": to_float(getattr(p, "cost_basis", 0.0)),
            "unrealized_pl": to_float(getattr(p, "unrealized_pl", 0.0)),
            "unrealized_plpc": to_float(getattr(p, "unrealized_plpc", 0.0)),
        })

    return pd.DataFrame(rows, columns=["symbol", "qty", "avg_entry_price", "current_price", "market_value", "cost_basis", "unrealized_pl", "unrealized_plpc"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORB sweep/retest/engulf 3R bracket executor.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit.")
    parser.add_argument("--loop", action="store_true", help="Compatibility flag. Default is continuous.")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--footprint-file", default=str(DEFAULT_FOOTPRINT_FILE))
    parser.add_argument("--universe", default="", help="Comma-separated symbols. Default uses built-in universe.")

    parser.add_argument("--session-timezone", default="America/New_York")
    parser.add_argument("--opening-candle-time", default="09:30")
    parser.add_argument("--max-opening-candle-minutes-late", type=int, default=5)
    parser.add_argument("--retest-tolerance-pct", type=float, default=0.0015)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--disable-delta-flip", action="store_true")

    parser.add_argument("--cash-buffer", type=float, default=DEFAULT_CASH_BUFFER_PCT)
    parser.add_argument("--max-gross-exposure", type=float, default=MAX_GROSS_EXPOSURE)
    parser.add_argument("--max-position-weight", type=float, default=MAX_POSITION_WEIGHT)
    parser.add_argument("--max-order-dollars", type=float, default=MAX_ORDER_DOLLARS)
    parser.add_argument("--max-single-order-qty", type=int, default=MAX_SINGLE_ORDER_QTY)
    parser.add_argument("--min-notional-to-trade", type=float, default=MIN_NOTIONAL_TO_TRADE)
    parser.add_argument("--risk-per-trade-pct", type=float, default=RISK_PER_TRADE_PCT)
    parser.add_argument("--take-profit-r-multiple", type=float, default=TAKE_PROFIT_R_MULTIPLE)

    parser.add_argument("--allow-closed-market", action="store_true")
    parser.add_argument("--wait-for-open", action="store_true")
    parser.add_argument("--cancel-open-orders-first", action="store_true")
    parser.add_argument("--disable-kill-switch", action="store_true")
    parser.add_argument("--max-daily-loss-pct", type=float, default=MAX_DAILY_LOSS_PCT)
    parser.add_argument("--max-daily-loss-dollars", type=float, default=MAX_DAILY_LOSS_DOLLARS)
    parser.add_argument("--kill-switch-equity-floor", type=float, default=KILL_SWITCH_EQUITY_FLOOR)
    parser.add_argument("--diagnostics", action="store_true", help="Print system health without trading and exit.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    universe = [safe_symbol(x) for x in str(args.universe).split(",") if safe_symbol(x)] or None
    return ExecutorConfig(
        paper=not args.live,
        live=args.live,
        poll_once=args.once,
        poll_seconds=args.poll_seconds,
        footprint_file=Path(args.footprint_file),
        universe=universe,
        session_timezone=args.session_timezone,
        opening_candle_time=args.opening_candle_time,
        max_opening_candle_minutes_late=args.max_opening_candle_minutes_late,
        retest_tolerance_pct=args.retest_tolerance_pct,
        require_delta_flip=not args.disable_delta_flip,
        allow_shorts=args.allow_shorts,
        cash_buffer_pct=args.cash_buffer,
        max_gross_exposure=args.max_gross_exposure,
        max_position_weight=args.max_position_weight,
        max_order_dollars=args.max_order_dollars,
        max_single_order_qty=args.max_single_order_qty,
        min_notional_to_trade=args.min_notional_to_trade,
        risk_per_trade_pct=args.risk_per_trade_pct,
        take_profit_r_multiple=args.take_profit_r_multiple,
        allow_closed_market=args.allow_closed_market,
        require_market_open=not args.allow_closed_market,
        wait_for_market_open=args.wait_for_open,
        cancel_open_orders_first=args.cancel_open_orders_first,
        enable_kill_switch=not args.disable_kill_switch,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_daily_loss_dollars=args.max_daily_loss_dollars,
        kill_switch_equity_floor=args.kill_switch_equity_floor,
    )


def _run_diagnostics(args: argparse.Namespace) -> None:
    """Print system health and signal-readiness without executing any trades."""
    print("=" * 60)
    print("[diagnostics] ORB Sweep/Retest/Engulf — system health check")
    print("=" * 60)

    # 1. Credentials
    load_dotenv(ENV_PATH)
    api_key = os.getenv("APCA_API_KEY_ID", "")
    secret_key = os.getenv("APCA_API_SECRET_KEY", "")
    env_ok = bool(api_key and secret_key)
    print(f"[diagnostics] .env present:  {ENV_PATH.exists()}")
    print(f"[diagnostics] credentials:   {'OK' if env_ok else 'MISSING — run cannot proceed'}")

    # 2. Footprint file
    fp = Path(args.footprint_file)
    if fp.exists() and fp.stat().st_size > 0:
        try:
            df = pd.read_csv(fp)
            print(f"[diagnostics] footprint_file: {fp}  rows={len(df)}  cols={list(df.columns)}")
        except Exception as exc:
            print(f"[diagnostics] footprint_file: {fp}  READ ERROR: {exc}")
    else:
        print(f"[diagnostics] footprint_file: {fp}  MISSING or EMPTY")
        print("[diagnostics]   → Signal generation will produce 0 signals.")
        print("[diagnostics]   → Export TradingView volume footprint CSV and place it at the path above.")

    # 3. Signal state
    if SIGNAL_STATE_PATH.exists():
        try:
            state = json.loads(SIGNAL_STATE_PATH.read_text())
            n_keys = len(state.get("processed_signal_keys", []))
            print(f"[diagnostics] signal_state:  {SIGNAL_STATE_PATH}  processed_keys={n_keys}")
        except Exception:
            print(f"[diagnostics] signal_state:  {SIGNAL_STATE_PATH}  CORRUPT")
    else:
        print(f"[diagnostics] signal_state:  {SIGNAL_STATE_PATH}  not yet created (normal on first run)")

    # 4. Kill switch
    if KILL_SWITCH_PATH.exists():
        try:
            ks = json.loads(KILL_SWITCH_PATH.read_text())
            triggered = ks.get("kill_triggered", False)
            print(f"[diagnostics] kill_switch:   triggered={triggered}  date={ks.get('date')}  reason={ks.get('reason')}")
        except Exception:
            print(f"[diagnostics] kill_switch:   CORRUPT — will be reset on next run")
    else:
        print("[diagnostics] kill_switch:   not active")

    # 5. Broker connectivity (only if credentials available)
    if env_ok:
        try:
            broker = AlpacaBroker(ENV_PATH, paper=not args.live)
            clock = broker.get_clock_info()
            acct = broker.get_account_snapshot()
            positions = broker.get_all_open_positions()
            print(f"[diagnostics] broker:        CONNECTED  paper={not args.live}")
            print(f"[diagnostics] market_open:   {clock['is_open']}  next_open={clock['next_open']}")
            print(f"[diagnostics] equity:        ${acct.equity:,.2f}")
            print(f"[diagnostics] cash:          ${acct.cash:,.2f}")
            print(f"[diagnostics] buying_power:  ${acct.buying_power:,.2f}")
            print(f"[diagnostics] open_positions:{len(positions)}")
            for pos in positions:
                print(f"[diagnostics]   {pos.symbol:8s} qty={pos.qty:8.0f}  mv=${pos.market_value:,.2f}  pnl={pos.unrealized_plpc:.2%}")
        except Exception as exc:
            print(f"[diagnostics] broker:        ERROR — {exc}")

    print("=" * 60)
    print("[diagnostics] done")


def main() -> None:
    args = parse_args()
    if args.diagnostics:
        _run_diagnostics(args)
        return
    config = build_config(args)
    trader = ORBSweepEngulfExecutor(config)
    trader.run()


if __name__ == "__main__":
    main()
