#!/usr/bin/env python3
from __future__ import annotations

"""
premarket_trade_planner.py

Builds a premarket breakout/consolidation trade plan and intraday structure map
using yfinance minute bars instead of Alpaca market data.

Default:
    python premarket_trade_planner.py

Inputs:
    system-data/YYYYMMDD/YYYYMMDD_latest_llm_sentiment_universe.json
    or YYYYMMDD_locked_trading_universe.txt

Outputs:
    system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan.json
    system-data/YYYYMMDD/YYYYMMDD_intraday_structure_map.json
    system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan_diagnostics.json

Notes:
    - Uses yfinance 1m bars with prepost=True for premarket data.
    - Uses yfinance 1m bars with prepost=True for intraday/structure data.
    - Does not place trades.
    - Structure maps are confirmation inputs for run.py / RL timing logic.
"""

import json
import math
import os
import time as time_module
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

try:
    from market_context_builder import TODAY_TAG
except Exception:
    TODAY_TAG = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


# =============================================================================
# Paths / environment
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed typo from older version: latest / premarket instead of llatest / lpremarket.
LLM_UNIVERSE_PATH = OUT_DIR / f"{TODAY_TAG}_latest_llm_sentiment_universe.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{TODAY_TAG}_locked_trading_universe.txt"
TRADE_PLAN_PATH = OUT_DIR / f"{TODAY_TAG}_premarket_trade_plan.json"
STRUCTURE_MAP_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_structure_map.json"
DIAGNOSTICS_PATH = OUT_DIR / f"{TODAY_TAG}_premarket_trade_plan_diagnostics.json"

ET = ZoneInfo("America/New_York")


# =============================================================================
# Configuration
# =============================================================================

YF_BATCH_SIZE = int(os.getenv("YF_BATCH_SIZE", "25"))
YF_INTRADAY_PERIOD = os.getenv("YF_INTRADAY_PERIOD", "5d")
YF_AUTO_ADJUST = os.getenv("YF_AUTO_ADJUST", "0").lower() in {"1", "true", "yes"}

MIN_PREMARKET_VOLUME = int(os.getenv("MIN_PREMARKET_VOLUME", "1000"))
PREMARKET_START_ET = os.getenv("PREMARKET_START_ET", "04:00")
PREMARKET_END_ET = os.getenv("PREMARKET_END_ET", "09:30")

MIN_INITIAL_MOVE_PCT = float(os.getenv("PREMARKET_MIN_INITIAL_MOVE_PCT", "2.0"))
MAX_CONSOLIDATION_RANGE_PCT = float(os.getenv("PREMARKET_MAX_CONSOLIDATION_RANGE_PCT", "2.75"))
MIN_CONSOLIDATION_BARS = int(os.getenv("PREMARKET_MIN_CONSOLIDATION_BARS", "8"))
MAX_ALLOWED_CONSOLIDATIONS = int(os.getenv("PREMARKET_MAX_ALLOWED_CONSOLIDATIONS", "3"))

DEFAULT_TARGET_GAIN_PCT = float(os.getenv("PREMARKET_TARGET_GAIN_PCT", "5.0"))
RISK_REWARD_RATIO = float(os.getenv("PREMARKET_RISK_REWARD_RATIO", "2.0"))
STOP_BUFFER_PCT = float(os.getenv("PREMARKET_STOP_BUFFER_PCT", "0.25"))

# Intraday structure map settings.
INTRADAY_LOOKBACK_DAYS = int(os.getenv("INTRADAY_STRUCTURE_LOOKBACK_DAYS", "5"))
ORB_MINUTES = int(os.getenv("ORB_MINUTES", "15"))
REGULAR_SESSION_START_ET = os.getenv("REGULAR_SESSION_START_ET", "09:30")
REGULAR_SESSION_END_ET = os.getenv("REGULAR_SESSION_END_ET", "16:00")

PIVOT_LEFT_RIGHT_BARS = int(os.getenv("SR_PIVOT_LEFT_RIGHT_BARS", "3"))
MAX_SR_LEVELS_EACH_SIDE = int(os.getenv("MAX_SR_LEVELS_EACH_SIDE", "5"))
SR_CLUSTER_PCT = float(os.getenv("SR_CLUSTER_PCT", "0.003"))
NEAR_LEVEL_PCT = float(os.getenv("NEAR_LEVEL_PCT", "0.004"))

VWAP_BAND_STD_MULT_1 = float(os.getenv("VWAP_BAND_STD_MULT_1", "1.0"))
VWAP_BAND_STD_MULT_2 = float(os.getenv("VWAP_BAND_STD_MULT_2", "2.0"))
MIN_INTRADAY_DOLLAR_VOLUME = float(os.getenv("MIN_INTRADAY_DOLLAR_VOLUME", "250000"))
MIN_INTRADAY_BAR_COUNT = int(os.getenv("MIN_INTRADAY_BAR_COUNT", "5"))
MIN_LIQUIDITY_SCORE_FOR_TRADE = float(os.getenv("MIN_LIQUIDITY_SCORE_FOR_TRADE", "0.35"))

WAIT_FOR_MARKET_OPEN = os.getenv("PREMARKET_WAIT_FOR_MARKET_OPEN", "1").lower() in {"1", "true", "yes"}
MARKET_OPEN_WAIT_TIME = os.getenv("MARKET_OPEN_WAIT_TIME", "09:30")


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class PremarketPlan:
    symbol: str
    valid: bool
    reason: str
    premarket_volume: int
    premarket_open: float | None
    premarket_high: float | None
    premarket_low: float | None
    premarket_close: float | None
    initial_move_pct: float | None
    consolidation_count: int
    consolidation_start: str | None
    consolidation_end: str | None
    support: float | None
    resistance: float | None
    entry_price: float | None
    stop_loss: float | None
    target_5pct: float | None
    target_risk_reward: float | None
    target_measured_move: float | None
    primary_target: float | None
    measured_range: float | None
    risk_per_share: float | None
    reward_per_share: float | None
    risk_reward_ratio: float | None


# =============================================================================
# Helpers
# =============================================================================

def empty_plan(symbol: str, reason: str) -> PremarketPlan:
    return PremarketPlan(
        symbol=symbol,
        valid=False,
        reason=reason,
        premarket_volume=0,
        premarket_open=None,
        premarket_high=None,
        premarket_low=None,
        premarket_close=None,
        initial_move_pct=None,
        consolidation_count=0,
        consolidation_start=None,
        consolidation_end=None,
        support=None,
        resistance=None,
        entry_price=None,
        stop_loss=None,
        target_5pct=None,
        target_risk_reward=None,
        target_measured_move=None,
        primary_target=None,
        measured_range=None,
        risk_per_share=None,
        reward_per_share=None,
        risk_reward_ratio=None,
    )


def parse_hhmm(value: str) -> time:
    h, m = value.strip().split(":")
    return time(int(h), int(m))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_et_date():
    return datetime.now(ET).date()


def round_price(x: float | None, ndigits: int = 2) -> float | None:
    if x is None or pd.isna(x):
        return None
    return round(float(x), ndigits)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").upper().strip().replace(".", "-")


def chunked(items: list[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json_value(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe_json_value(payload), indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def merge_diagnostics(label: str, diagnostics: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if DIAGNOSTICS_PATH.exists():
        try:
            existing = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[label] = diagnostics
    write_json(DIAGNOSTICS_PATH, existing)


def load_universe() -> list[str]:
    symbols: list[str] = []

    if LLM_UNIVERSE_PATH.exists():
        try:
            payload = json.loads(LLM_UNIVERSE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw = (
                    payload.get("symbols")
                    or payload.get("selected_symbols")
                    or payload.get("universe")
                    or payload.get("tickers")
                    or []
                )
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            symbols.append(normalize_symbol(item.get("symbol") or item.get("ticker")))
                        else:
                            symbols.append(normalize_symbol(item))
        except Exception as exc:
            print(f"[universe] failed to load {LLM_UNIVERSE_PATH}: {exc}")

    if not symbols and LOCKED_UNIVERSE_PATH.exists():
        symbols = [
            normalize_symbol(line)
            for line in LOCKED_UNIVERSE_PATH.read_text(encoding="utf-8").splitlines()
            if normalize_symbol(line)
        ]

    symbols = sorted(set(s for s in symbols if s))
    print(f"[universe] symbols={len(symbols)}")
    return symbols


# =============================================================================
# yfinance data ingestion
# =============================================================================

def flatten_yfinance_download(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """
    Normalize yfinance download output into:
        symbol, timestamp, open, high, low, close, volume, timestamp_et
    """
    if df is None or df.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    if isinstance(df.columns, pd.MultiIndex):
        # Typical shape with group_by="column": level 0 = OHLCV, level 1 = ticker.
        level0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        if any(x in {"open", "high", "low", "close", "volume"} for x in level0):
            ticker_level = 1
        else:
            ticker_level = 0

        tickers = sorted(set(str(x) for x in df.columns.get_level_values(ticker_level)))

        for ticker in tickers:
            symbol = normalize_symbol(ticker)
            try:
                sdf = df.xs(ticker, axis=1, level=ticker_level).copy()
            except Exception:
                continue
            sdf.columns = [str(c).lower().replace(" ", "_") for c in sdf.columns]
            sdf = sdf.reset_index()
            sdf["symbol"] = symbol
            frames.append(sdf)
    else:
        symbol = normalize_symbol(symbols[0]) if symbols else ""
        sdf = df.copy()
        sdf.columns = [str(c).lower().replace(" ", "_") for c in sdf.columns]
        sdf = sdf.reset_index()
        sdf["symbol"] = symbol
        frames.append(sdf)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    timestamp_col = None
    for col in ["datetime", "date", "timestamp"]:
        if col in out.columns:
            timestamp_col = col
            break
    if timestamp_col is None:
        timestamp_col = out.columns[0]

    out = out.rename(columns={timestamp_col: "timestamp"})

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out = out[out["symbol"] != ""].copy()
    out["timestamp_et"] = out["timestamp"].dt.tz_convert(ET)

    return out[["symbol", "timestamp", "timestamp_et", "open", "high", "low", "close", "volume"]]


def fetch_minute_bars(
    symbols: list[str],
    start_et: datetime,
    end_et: datetime,
    label: str,
) -> pd.DataFrame:
    """
    Fetch 1-minute bars with yfinance.

    yfinance only permits recent 1-minute history, so this uses a period-based
    download and then filters to the requested ET window.
    """
    if not symbols:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {
        "label": label,
        "source": "yfinance",
        "window_et": {"start": start_et.isoformat(), "end": end_et.isoformat()},
        "period": YF_INTRADAY_PERIOD,
        "interval": "1m",
        "prepost": True,
        "batches": [],
    }

    print(f"[{label}] source=yfinance period={YF_INTRADAY_PERIOD} interval=1m prepost=True")
    print(f"[{label}] window_et={start_et} -> {end_et}")
    print(f"[{label}] symbols={len(symbols)} batch_size={YF_BATCH_SIZE}")

    for batch in chunked(symbols, YF_BATCH_SIZE):
        batch_diag = {"symbols": batch, "row_count": 0, "counts": {}, "error": None}
        tickers = " ".join(batch)

        try:
            raw = yf.download(
                tickers=tickers,
                period=YF_INTRADAY_PERIOD,
                interval="1m",
                prepost=True,
                group_by="column",
                auto_adjust=YF_AUTO_ADJUST,
                progress=False,
                threads=True,
            )

            df = flatten_yfinance_download(raw, batch)
            if df.empty:
                print(f"[{label}] empty batch={batch}")
                diagnostics["batches"].append(batch_diag)
                continue

            df = df[(df["timestamp_et"] >= start_et) & (df["timestamp_et"] < end_et)].copy()

            counts = df.groupby("symbol").size().to_dict() if not df.empty else {}
            batch_diag["row_count"] = int(len(df))
            batch_diag["counts"] = {str(k): int(v) for k, v in counts.items()}

            if not df.empty:
                frames.append(df)

            print(f"[{label}] batch_counts={counts}")

        except Exception as exc:
            batch_diag["error"] = str(exc)
            print(f"[{label}] failed batch={batch}: {exc}")
        finally:
            diagnostics["batches"].append(batch_diag)

    if not frames:
        diagnostics["final_rows"] = 0
        diagnostics["symbols_with_bars"] = []
        merge_diagnostics(label, diagnostics)
        print(f"[{label}] no yfinance bars returned")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    diagnostics["final_rows"] = int(len(out))
    diagnostics["symbols_with_bars"] = sorted(out["symbol"].dropna().unique().tolist())
    merge_diagnostics(label, diagnostics)

    print(f"[{label}] final_rows={len(out)}")
    print(f"[{label}] symbols_with_bars={','.join(diagnostics['symbols_with_bars'])}")
    return out


def fetch_premarket_bars(symbols: list[str]) -> pd.DataFrame:
    day = today_et_date()
    start_et = datetime.combine(day, parse_hhmm(PREMARKET_START_ET), tzinfo=ET)
    end_et = datetime.combine(day, parse_hhmm(PREMARKET_END_ET), tzinfo=ET)
    return fetch_minute_bars(symbols, start_et, end_et, label="premarket_bars")


def fetch_intraday_bars(symbols: list[str]) -> pd.DataFrame:
    day = today_et_date()
    start_et = datetime.combine(day - timedelta(days=INTRADAY_LOOKBACK_DAYS), parse_hhmm(REGULAR_SESSION_START_ET), tzinfo=ET)
    end_et = datetime.now(ET)
    regular_end_today = datetime.combine(day, parse_hhmm(REGULAR_SESSION_END_ET), tzinfo=ET)
    end_et = min(end_et, regular_end_today)
    return fetch_minute_bars(symbols, start_et, end_et, label="intraday_bars")


# =============================================================================
# Support/resistance, VWAP, ORB, liquidity
# =============================================================================

def cluster_levels(levels: list[dict[str, Any]], cluster_pct: float = SR_CLUSTER_PCT) -> list[dict[str, Any]]:
    cleaned = [x for x in levels if x.get("price") is not None and safe_float(x.get("price")) > 0]
    if not cleaned:
        return []

    cleaned.sort(key=lambda x: safe_float(x["price"]))
    clusters: list[list[dict[str, Any]]] = []

    for item in cleaned:
        price = safe_float(item["price"])
        if not clusters:
            clusters.append([item])
            continue
        current_prices = [safe_float(x["price"]) for x in clusters[-1]]
        center = sum(current_prices) / max(len(current_prices), 1)
        if abs(price - center) / max(center, 1e-9) <= cluster_pct:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    out: list[dict[str, Any]] = []
    for cluster in clusters:
        weights = [max(1.0, safe_float(x.get("weight"), 1.0)) for x in cluster]
        prices = [safe_float(x["price"]) for x in cluster]
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / max(sum(weights), 1e-9)
        sources: list[str] = []
        for x in cluster:
            src = str(x.get("source", "unknown"))
            if src not in sources:
                sources.append(src)
        out.append(
            {
                "price": round_price(weighted_price),
                "touch_count": int(sum(safe_float(x.get("touch_count"), 1.0) for x in cluster)),
                "weight": round(float(sum(weights)), 4),
                "sources": sources,
            }
        )
    return out


def detect_pivot_levels(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or len(df) < PIVOT_LEFT_RIGHT_BARS * 2 + 1:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)
    levels: list[dict[str, Any]] = []
    n = PIVOT_LEFT_RIGHT_BARS

    avg_vol_series = df["volume"].rolling(20, min_periods=1).mean()

    for i in range(n, len(df) - n):
        high = safe_float(df.loc[i, "high"])
        low = safe_float(df.loc[i, "low"])
        left_high = df.loc[i - n:i - 1, "high"].max()
        right_high = df.loc[i + 1:i + n, "high"].max()
        left_low = df.loc[i - n:i - 1, "low"].min()
        right_low = df.loc[i + 1:i + n, "low"].min()
        vol = safe_float(df.loc[i, "volume"])
        avg_vol = safe_float(avg_vol_series.iloc[i], 1.0)
        vol_weight = max(1.0, min(5.0, vol / max(avg_vol, 1.0)))

        if high >= left_high and high >= right_high:
            levels.append({"price": high, "source": "pivot_high", "touch_count": 1, "weight": 1.5 * vol_weight})
        if low <= left_low and low <= right_low:
            levels.append({"price": low, "source": "pivot_low", "touch_count": 1, "weight": 1.5 * vol_weight})

    return levels


def add_vwap_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("timestamp").copy()

    for col in ["high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].clip(lower=0.0)
    pv = typical * volume

    cum_vol = volume.cumsum()
    cum_pv = pv.cumsum()

    # Avoid pd.NA because astype(float) can choke on NAType.
    df["vwap"] = (cum_pv / cum_vol.replace(0, float("nan"))).astype(float)
    df["vwap"] = df["vwap"].ffill().fillna(df["close"])

    df["vwap_dev"] = df["close"] - df["vwap"]
    df["vwap_dev_std"] = df["vwap_dev"].expanding(min_periods=5).std()
    df["vwap_dev_std"] = df["vwap_dev_std"].replace([float("inf"), -float("inf")], 0.0).fillna(0.0)

    df["vwap_upper_1"] = df["vwap"] + VWAP_BAND_STD_MULT_1 * df["vwap_dev_std"]
    df["vwap_lower_1"] = df["vwap"] - VWAP_BAND_STD_MULT_1 * df["vwap_dev_std"]
    df["vwap_upper_2"] = df["vwap"] + VWAP_BAND_STD_MULT_2 * df["vwap_dev_std"]
    df["vwap_lower_2"] = df["vwap"] - VWAP_BAND_STD_MULT_2 * df["vwap_dev_std"]

    return df


def build_orb_map(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "available": False,
            "reason": "no_intraday_bars",
            "minutes": ORB_MINUTES,
            "high": None,
            "low": None,
            "mid": None,
            "range_pct": None,
            "breakout_confirmed": False,
            "breakdown_confirmed": False,
        }

    day = today_et_date()
    start = datetime.combine(day, parse_hhmm(REGULAR_SESSION_START_ET), tzinfo=ET)
    end = start + timedelta(minutes=ORB_MINUTES)

    sdf = df.copy()
    if "timestamp_et" not in sdf.columns:
        sdf["timestamp_et"] = pd.to_datetime(sdf["timestamp"], utc=True).dt.tz_convert(ET)

    today = sdf[(sdf["timestamp_et"] >= start) & (sdf["timestamp_et"] < datetime.now(ET))].copy()
    orb = today[(today["timestamp_et"] >= start) & (today["timestamp_et"] < end)].copy()

    if len(orb) < max(1, min(ORB_MINUTES, 5)):
        return {
            "available": False,
            "reason": "not_enough_opening_range_bars",
            "minutes": ORB_MINUTES,
            "high": None,
            "low": None,
            "mid": None,
            "range_pct": None,
            "breakout_confirmed": False,
            "breakdown_confirmed": False,
        }

    high = safe_float(orb["high"].max())
    low = safe_float(orb["low"].min())
    mid = (high + low) / 2.0 if high > 0 and low > 0 else None
    last_close = safe_float(today.iloc[-1]["close"]) if not today.empty else None
    last_volume = safe_float(today.iloc[-1]["volume"]) if not today.empty else 0.0
    avg_volume = safe_float(today["volume"].rolling(20, min_periods=1).mean().iloc[-1]) if not today.empty else 1.0
    volume_confirmed = last_volume >= avg_volume

    return {
        "available": True,
        "reason": "ok",
        "minutes": ORB_MINUTES,
        "high": round_price(high),
        "low": round_price(low),
        "mid": round_price(mid),
        "range_pct": round(((high - low) / mid) * 100.0, 4) if mid and mid > 0 else None,
        "breakout_trigger": round_price(high * 1.001),
        "breakdown_trigger": round_price(low * 0.999),
        "breakout_confirmed": bool(last_close is not None and last_close > high and volume_confirmed),
        "breakdown_confirmed": bool(last_close is not None and last_close < low and volume_confirmed),
        "last_close": round_price(last_close),
        "volume_confirmed": bool(volume_confirmed),
    }


def build_liquidity_profile(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "bar_count": 0,
            "total_volume": 0,
            "total_dollar_volume": 0.0,
            "avg_bar_volume": 0.0,
            "avg_bar_dollar_volume": 0.0,
            "avg_range_pct": None,
            "liquidity_score": 0.0,
            "liquid_enough": False,
            "reason": "no_bars",
        }

    close = df["close"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0)
    dollar_volume = close * volume
    ranges = (df["high"].astype(float) - df["low"].astype(float)) / close.replace(0, pd.NA)

    bar_count = int(len(df))
    total_volume = int(volume.sum())
    total_dollar_volume = float(dollar_volume.sum())
    avg_bar_dollar_volume = float(dollar_volume.mean()) if len(dollar_volume) else 0.0
    avg_range_pct = float(ranges.dropna().mean()) if not ranges.dropna().empty else None

    bar_score = min(1.0, bar_count / max(MIN_INTRADAY_BAR_COUNT, 1))
    dollar_score = min(1.0, total_dollar_volume / max(MIN_INTRADAY_DOLLAR_VOLUME, 1.0))
    stability_score = 1.0
    if avg_range_pct is not None:
        stability_score = max(0.0, min(1.0, 1.0 - max(0.0, avg_range_pct - 0.03) / 0.08))

    liquidity_score = round(0.45 * dollar_score + 0.35 * bar_score + 0.20 * stability_score, 4)

    return {
        "bar_count": bar_count,
        "total_volume": total_volume,
        "total_dollar_volume": round(total_dollar_volume, 2),
        "avg_bar_volume": round(float(volume.mean()), 2) if len(volume) else 0.0,
        "avg_bar_dollar_volume": round(avg_bar_dollar_volume, 2),
        "avg_range_pct": round(avg_range_pct, 6) if avg_range_pct is not None else None,
        "liquidity_score": liquidity_score,
        "liquid_enough": bool(liquidity_score >= MIN_LIQUIDITY_SCORE_FOR_TRADE),
        "reason": "ok" if liquidity_score >= MIN_LIQUIDITY_SCORE_FOR_TRADE else "below_liquidity_threshold",
    }


def nearest_levels(current_price: float | None, levels: list[dict[str, Any]]) -> dict[str, Any]:
    if current_price is None or current_price <= 0:
        return {"nearest_support": None, "nearest_resistance": None, "near_any_level": False}

    supports = [x for x in levels if safe_float(x.get("price")) < current_price]
    resistances = [x for x in levels if safe_float(x.get("price")) > current_price]
    nearest_support = max(supports, key=lambda x: safe_float(x.get("price")), default=None)
    nearest_resistance = min(resistances, key=lambda x: safe_float(x.get("price")), default=None)

    def dist(level: dict[str, Any] | None) -> float | None:
        if not level:
            return None
        return abs(current_price - safe_float(level.get("price"))) / current_price

    ds = dist(nearest_support)
    dr = dist(nearest_resistance)
    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "distance_to_support_pct": round(ds * 100, 4) if ds is not None else None,
        "distance_to_resistance_pct": round(dr * 100, 4) if dr is not None else None,
        "near_any_level": bool((ds is not None and ds <= NEAR_LEVEL_PCT) or (dr is not None and dr <= NEAR_LEVEL_PCT)),
    }


def build_intraday_structure_map_for_symbol(
    symbol: str,
    premarket_bars: pd.DataFrame,
    intraday_bars: pd.DataFrame,
    plan: PremarketPlan,
) -> dict[str, Any]:
    pbar = premarket_bars[premarket_bars.get("symbol", pd.Series(dtype=str)).eq(symbol)].copy() if not premarket_bars.empty else pd.DataFrame()
    ibar = intraday_bars[intraday_bars.get("symbol", pd.Series(dtype=str)).eq(symbol)].copy() if not intraday_bars.empty else pd.DataFrame()

    if not ibar.empty:
        ibar = add_vwap_columns(ibar)

    level_candidates: list[dict[str, Any]] = []

    if plan.support is not None:
        level_candidates.append({"price": plan.support, "source": "premarket_consolidation_support", "touch_count": 2, "weight": 4.0})
    if plan.resistance is not None:
        level_candidates.append({"price": plan.resistance, "source": "premarket_consolidation_resistance", "touch_count": 2, "weight": 4.0})
    if plan.premarket_low is not None:
        level_candidates.append({"price": plan.premarket_low, "source": "premarket_low", "touch_count": 1, "weight": 2.0})
    if plan.premarket_high is not None:
        level_candidates.append({"price": plan.premarket_high, "source": "premarket_high", "touch_count": 1, "weight": 2.0})

    level_candidates.extend(detect_pivot_levels(ibar))

    orb = build_orb_map(ibar)
    if orb.get("high") is not None:
        level_candidates.append({"price": orb["high"], "source": "orb_high", "touch_count": 1, "weight": 5.0})
    if orb.get("low") is not None:
        level_candidates.append({"price": orb["low"], "source": "orb_low", "touch_count": 1, "weight": 5.0})

    if not ibar.empty and "vwap" in ibar.columns:
        last = ibar.iloc[-1]
        for col, weight in [
            ("vwap", 4.0),
            ("vwap_upper_1", 2.5),
            ("vwap_lower_1", 2.5),
            ("vwap_upper_2", 1.5),
            ("vwap_lower_2", 1.5),
        ]:
            value = safe_float(last.get(col), 0.0)
            if value > 0:
                level_candidates.append({"price": value, "source": col, "touch_count": 1, "weight": weight})

    clustered = cluster_levels(level_candidates)

    current_price = None
    if not ibar.empty:
        current_price = safe_float(ibar.iloc[-1]["close"], 0.0)
    if not current_price:
        current_price = plan.premarket_close or plan.premarket_high or plan.premarket_low

    supports = [x for x in clustered if current_price is not None and safe_float(x.get("price")) < current_price]
    resistances = [x for x in clustered if current_price is not None and safe_float(x.get("price")) > current_price]
    supports = sorted(supports, key=lambda x: safe_float(x.get("price")), reverse=True)[:MAX_SR_LEVELS_EACH_SIDE]
    resistances = sorted(resistances, key=lambda x: safe_float(x.get("price")))[:MAX_SR_LEVELS_EACH_SIDE]

    vwap_snapshot: dict[str, Any] = {"available": False}
    if not ibar.empty and "vwap" in ibar.columns:
        last = ibar.iloc[-1]
        vwap = safe_float(last.get("vwap"), 0.0)
        close = safe_float(last.get("close"), 0.0)
        vwap_snapshot = {
            "available": True,
            "vwap": round_price(vwap),
            "upper_1": round_price(safe_float(last.get("vwap_upper_1"), 0.0)),
            "lower_1": round_price(safe_float(last.get("vwap_lower_1"), 0.0)),
            "upper_2": round_price(safe_float(last.get("vwap_upper_2"), 0.0)),
            "lower_2": round_price(safe_float(last.get("vwap_lower_2"), 0.0)),
            "price_above_vwap": bool(close > vwap) if vwap > 0 and close > 0 else False,
            "distance_from_vwap_pct": round(((close - vwap) / close) * 100.0, 4) if close > 0 and vwap > 0 else None,
        }

    liquidity = build_liquidity_profile(ibar if not ibar.empty else pbar)
    near = nearest_levels(current_price, clustered)

    long_allowed = bool(
        liquidity.get("liquid_enough")
        and (
            plan.valid
            or orb.get("breakout_confirmed")
            or (vwap_snapshot.get("available") and vwap_snapshot.get("price_above_vwap"))
        )
    )
    short_or_put_allowed = bool(
        liquidity.get("liquid_enough")
        and (
            orb.get("breakdown_confirmed")
            or (vwap_snapshot.get("available") and not vwap_snapshot.get("price_above_vwap"))
        )
    )

    return {
        "symbol": symbol,
        "generated_at_utc": now_utc(),
        "current_price": round_price(current_price),
        "premarket_setup_valid": plan.valid,
        "premarket_reason": plan.reason,
        "support_levels": supports,
        "resistance_levels": resistances,
        "all_clustered_levels": clustered,
        "nearest_levels": near,
        "orb": orb,
        "vwap": vwap_snapshot,
        "liquidity": liquidity,
        "trade_structure": {
            "long_allowed": long_allowed,
            "short_or_put_allowed": short_or_put_allowed,
            "preferred_long_trigger": resistances[0]["price"] if resistances else None,
            "preferred_short_trigger": supports[0]["price"] if supports else None,
            "preferred_long_stop": supports[0]["price"] if supports else None,
            "preferred_short_stop": resistances[0]["price"] if resistances else None,
            "notes": [
                "Use this as a structure map, not a standalone order signal.",
                "Require ML/RL direction confirmation before entries.",
                "Always allow exits regardless of long_allowed/short_or_put_allowed.",
            ],
        },
    }


# =============================================================================
# Premarket pattern logic
# =============================================================================

def find_consolidation_windows(df: pd.DataFrame) -> list[pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    if df.empty or len(df) < MIN_CONSOLIDATION_BARS:
        return windows

    df = df.sort_values("timestamp").reset_index(drop=True)
    active_start: int | None = None
    active_end: int | None = None

    for i in range(0, len(df) - MIN_CONSOLIDATION_BARS + 1):
        window = df.iloc[i:i + MIN_CONSOLIDATION_BARS]
        high = float(window["high"].max())
        low = float(window["low"].min())
        mid = float(window["close"].mean())
        if mid <= 0:
            continue
        range_pct = ((high - low) / mid) * 100.0
        if range_pct <= MAX_CONSOLIDATION_RANGE_PCT:
            if active_start is None:
                active_start = i
                active_end = i + MIN_CONSOLIDATION_BARS
            elif active_end is not None and i <= active_end:
                active_end = i + MIN_CONSOLIDATION_BARS
            else:
                windows.append(df.iloc[active_start:active_end].copy())
                active_start = i
                active_end = i + MIN_CONSOLIDATION_BARS

    if active_start is not None and active_end is not None:
        windows.append(df.iloc[active_start:active_end].copy())
    return windows


def build_support_resistance_levels(
    support: float | None,
    resistance: float | None,
    premarket_low: float | None,
    premarket_high: float | None,
    current_price: float | None,
) -> dict[str, Any]:
    if current_price is None or current_price <= 0:
        return {"resistance_1": None, "resistance_2": None, "resistance_3": None, "support_1": None, "support_2": None, "support_3": None}

    resistance = resistance or premarket_high or current_price
    support = support or premarket_low or current_price
    range_size = max(float(resistance) - float(support), current_price * 0.005)

    return {
        "resistance_1": round_price(resistance),
        "resistance_2": round_price(resistance + range_size * 0.5),
        "resistance_3": round_price(resistance + range_size),
        "support_1": round_price(support),
        "support_2": round_price(support - range_size * 0.5),
        "support_3": round_price(support - range_size),
    }


def build_daily_trade_plan_json(plan: PremarketPlan, structure_map: dict[str, Any] | None = None) -> dict[str, Any]:
    current_price = plan.premarket_close or plan.premarket_high or plan.premarket_low
    levels = build_support_resistance_levels(plan.support, plan.resistance, plan.premarket_low, plan.premarket_high, current_price)

    if structure_map:
        resistances = structure_map.get("resistance_levels", []) or []
        supports = structure_map.get("support_levels", []) or []
        if resistances:
            levels["resistance_1"] = resistances[0].get("price")
        if len(resistances) > 1:
            levels["resistance_2"] = resistances[1].get("price")
        if len(resistances) > 2:
            levels["resistance_3"] = resistances[2].get("price")
        if supports:
            levels["support_1"] = supports[0].get("price")
        if len(supports) > 1:
            levels["support_2"] = supports[1].get("price")
        if len(supports) > 2:
            levels["support_3"] = supports[2].get("price")
        current_price = structure_map.get("current_price") or current_price

    call_entry = levels["resistance_1"]
    call_stop = levels["support_1"]
    put_entry = levels["support_1"]
    put_stop = levels["resistance_1"]

    measured_range = None
    if levels["resistance_1"] is not None and levels["support_1"] is not None:
        measured_range = levels["resistance_1"] - levels["support_1"]

    call_take_profit = round_price(call_entry + measured_range) if call_entry is not None and measured_range is not None else None
    put_take_profit = round_price(put_entry - measured_range) if put_entry is not None and measured_range is not None else None

    return {
        "symbol": plan.symbol,
        "date": datetime.now(ET).strftime("%B %d, %Y").replace(" 0", " "),
        "current_price": round_price(current_price),
        "valid_setup": plan.valid,
        "setup_reason": plan.reason,
        "premarket": {
            "volume": plan.premarket_volume,
            "open": round_price(plan.premarket_open),
            "high": round_price(plan.premarket_high),
            "low": round_price(plan.premarket_low),
            "close": round_price(plan.premarket_close),
            "initial_move_pct": plan.initial_move_pct,
            "consolidation_count": plan.consolidation_count,
            "consolidation_start": plan.consolidation_start,
            "consolidation_end": plan.consolidation_end,
        },
        "combined_support_and_resistance_levels": levels,
        "structure_summary": {
            "orb": structure_map.get("orb") if structure_map else None,
            "vwap": structure_map.get("vwap") if structure_map else None,
            "liquidity": structure_map.get("liquidity") if structure_map else None,
            "trade_structure": structure_map.get("trade_structure") if structure_map else None,
        },
        "trade_entries_and_exits": {
            "calls": {
                "entry": f"Above {call_entry}" if call_entry is not None else None,
                "entry_price": call_entry,
                "stop_loss": call_stop,
                "take_profit": call_take_profit,
                "target_5pct": round_price(call_entry * 1.05) if call_entry else None,
                "measured_move_target": call_take_profit,
            },
            "puts": {
                "entry": f"Below {put_entry}" if put_entry is not None else None,
                "entry_price": put_entry,
                "stop_loss": put_stop,
                "take_profit": put_take_profit,
                "target_5pct": round_price(put_entry * 0.95) if put_entry else None,
                "measured_move_target": put_take_profit,
            },
        },
    }


def analyze_symbol(symbol: str, bars: pd.DataFrame) -> PremarketPlan:
    if bars.empty or "symbol" not in bars.columns:
        return empty_plan(symbol, "No yfinance premarket bars found. No premarket breakout entry will be created.")

    sdf = bars[bars["symbol"] == symbol].copy()
    if sdf.empty:
        return empty_plan(symbol, "No yfinance premarket bars found. No premarket breakout entry will be created.")

    sdf = sdf.sort_values("timestamp").reset_index(drop=True)
    premarket_volume = int(pd.to_numeric(sdf["volume"], errors="coerce").fillna(0).sum())
    premarket_open = float(sdf.iloc[0]["open"])
    premarket_close = float(sdf.iloc[-1]["close"])
    premarket_high = float(sdf["high"].max())
    premarket_low = float(sdf["low"].min())

    def reject(reason: str, initial_move_pct: float | None = None, consolidation_count: int = 0) -> PremarketPlan:
        p = empty_plan(symbol, reason)
        p.premarket_volume = premarket_volume
        p.premarket_open = round(premarket_open, 4)
        p.premarket_high = round(premarket_high, 4)
        p.premarket_low = round(premarket_low, 4)
        p.premarket_close = round(premarket_close, 4)
        p.initial_move_pct = round(initial_move_pct, 4) if initial_move_pct is not None else None
        p.consolidation_count = consolidation_count
        return p

    if premarket_volume < MIN_PREMARKET_VOLUME:
        return reject("Premarket volume below threshold. Structure may still be mapped, but entry is not valid.")

    initial_high_idx = int(sdf["high"].idxmax())
    initial_high = float(sdf.loc[initial_high_idx, "high"])
    initial_move_pct = ((initial_high - premarket_open) / premarket_open) * 100.0 if premarket_open > 0 else None

    if initial_move_pct is None or initial_move_pct < MIN_INITIAL_MOVE_PCT:
        return reject("No strong initial premarket move up.", initial_move_pct)

    post_move = sdf.iloc[initial_high_idx + 1:].copy()
    if len(post_move) < MIN_CONSOLIDATION_BARS:
        return reject("Not enough bars after initial move to confirm consolidation.", initial_move_pct)

    consolidations = find_consolidation_windows(post_move)
    consolidation_count = len(consolidations)
    if consolidation_count < 1 or consolidation_count > MAX_ALLOWED_CONSOLIDATIONS:
        return reject(
            f"Expected 1-{MAX_ALLOWED_CONSOLIDATIONS} major consolidation(s), found {consolidation_count}.",
            initial_move_pct,
            consolidation_count,
        )

    consolidation = consolidations[-1]
    support = float(consolidation["low"].min())
    resistance = float(consolidation["high"].max())
    measured_range = resistance - support
    entry_price = resistance
    stop_loss = support * (1.0 - STOP_BUFFER_PCT / 100.0)
    risk_per_share = entry_price - stop_loss

    if risk_per_share <= 0:
        p = reject("Invalid risk range: entry is not above stop.", initial_move_pct, consolidation_count)
        p.consolidation_start = str(consolidation.iloc[0]["timestamp"])
        p.consolidation_end = str(consolidation.iloc[-1]["timestamp"])
        p.support = round(support, 4)
        p.resistance = round(resistance, 4)
        p.entry_price = round(entry_price, 4)
        p.stop_loss = round(stop_loss, 4)
        p.measured_range = round(measured_range, 4)
        p.risk_per_share = round(risk_per_share, 4)
        return p

    target_5pct = entry_price * (1.0 + DEFAULT_TARGET_GAIN_PCT / 100.0)
    target_risk_reward = entry_price + risk_per_share * RISK_REWARD_RATIO
    target_measured_move = resistance + measured_range
    primary_target = max(target_5pct, target_risk_reward, target_measured_move)
    reward_per_share = primary_target - entry_price
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else None

    return PremarketPlan(
        symbol=symbol,
        valid=True,
        reason="Valid setup: yfinance premarket volume, strong initial move, consolidation, support/resistance marked.",
        premarket_volume=premarket_volume,
        premarket_open=round(premarket_open, 4),
        premarket_high=round(premarket_high, 4),
        premarket_low=round(premarket_low, 4),
        premarket_close=round(premarket_close, 4),
        initial_move_pct=round(initial_move_pct, 4),
        consolidation_count=consolidation_count,
        consolidation_start=str(consolidation.iloc[0]["timestamp"]),
        consolidation_end=str(consolidation.iloc[-1]["timestamp"]),
        support=round(support, 4),
        resistance=round(resistance, 4),
        entry_price=round(entry_price, 4),
        stop_loss=round(stop_loss, 4),
        target_5pct=round(target_5pct, 4),
        target_risk_reward=round(target_risk_reward, 4),
        target_measured_move=round(target_measured_move, 4),
        primary_target=round(primary_target, 4),
        measured_range=round(measured_range, 4),
        risk_per_share=round(risk_per_share, 4),
        reward_per_share=round(reward_per_share, 4),
        risk_reward_ratio=round(rr, 4) if rr is not None else None,
    )


# =============================================================================
# Orchestration
# =============================================================================

def wait_until_market_open() -> None:
    if not WAIT_FOR_MARKET_OPEN:
        return

    market_open = parse_hhmm(MARKET_OPEN_WAIT_TIME)
    while True:
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:
            print("[wait] Weekend detected. Exiting wait loop.")
            return
        if now_et.time() >= market_open:
            print("[market-open] Running premarket/intraday structure planner...")
            return
        seconds_left = (datetime.combine(now_et.date(), market_open, tzinfo=ET) - now_et).total_seconds()
        print(f"[wait] Market opens in {int(seconds_left // 60)} minutes...")
        time_module.sleep(min(60, max(5, seconds_left)))


def build_premarket_trade_plan() -> dict[str, Any]:
    symbols = load_universe()
    if not symbols:
        payload = {
            "created_at_utc": now_utc(),
            "data_source": "yfinance",
            "valid_plan_count": 0,
            "plans": [],
            "error": "No symbols found in latest LLM universe or locked universe.",
        }
        write_json(TRADE_PLAN_PATH, payload)
        write_json(STRUCTURE_MAP_PATH, {"created_at_utc": now_utc(), "data_source": "yfinance", "structures": []})
        return payload

    premarket_bars = fetch_premarket_bars(symbols)
    intraday_bars = fetch_intraday_bars(symbols)

    plans: list[dict[str, Any]] = []
    daily_trade_plans: list[dict[str, Any]] = []
    structure_maps: list[dict[str, Any]] = []

    for symbol in symbols:
        plan = analyze_symbol(symbol, premarket_bars)
        structure = build_intraday_structure_map_for_symbol(symbol, premarket_bars, intraday_bars, plan)
        plans.append(asdict(plan))
        structure_maps.append(structure)
        daily_trade_plans.append(build_daily_trade_plan_json(plan, structure))

    valid = [p for p in plans if p["valid"]]
    liquid_structures = [s for s in structure_maps if s.get("liquidity", {}).get("liquid_enough")]
    long_allowed = [s["symbol"] for s in structure_maps if s.get("trade_structure", {}).get("long_allowed")]

    payload = {
        "created_at_utc": now_utc(),
        "data_source": "yfinance",
        "premarket_window_et": {"start": PREMARKET_START_ET, "end": PREMARKET_END_ET},
        "regular_session_window_et": {"start": REGULAR_SESSION_START_ET, "end": REGULAR_SESSION_END_ET},
        "symbol_count": len(symbols),
        "valid_plan_count": len(valid),
        "valid_symbols": [p["symbol"] for p in valid],
        "liquid_structure_count": len(liquid_structures),
        "long_allowed_symbols": long_allowed,
        "daily_trade_plans": daily_trade_plans,
        "raw_premarket_pattern_plans": plans,
    }

    structure_payload = {
        "created_at_utc": now_utc(),
        "data_source": "yfinance",
        "symbol_count": len(symbols),
        "long_allowed_symbols": long_allowed,
        "structures": structure_maps,
    }

    write_json(TRADE_PLAN_PATH, payload)
    write_json(STRUCTURE_MAP_PATH, structure_payload)

    print(f"[premarket-plan] symbols={len(symbols)} valid={len(valid)} liquid={len(liquid_structures)} long_allowed={len(long_allowed)}")
    print("[premarket-plan] valid_symbols=" + ",".join(payload["valid_symbols"]))
    print("[structure-map] long_allowed_symbols=" + ",".join(long_allowed))

    return payload


if __name__ == "__main__":
    wait_until_market_open()
    build_premarket_trade_plan()
