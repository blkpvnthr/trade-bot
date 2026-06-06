#!/usr/bin/env python3
from __future__ import annotations

"""
premarket_trade_planner.py

Builds a premarket breakout/consolidation trade plan and intraday structure map.

Data hierarchy:
    Universe   — tradingview-screener (discovery + TA enrichment only, no OHLCV)
    Premarket  — Alpaca BOATS/SIP feed  (04:00–09:30 ET extended-hours bars)
    Intraday   — Alpaca IEX feed        (09:30–16:00 ET regular-session bars)
    Fallback   — yfinance (only when ALLOW_YFINANCE_MARKET_DATA_FALLBACK=1)

Feed env vars:
    ALPACA_PREMARKET_FEED=boats|sip   (default: sip; boats requires overnight subscription)
    ALPACA_REGULAR_FEED=iex|sip       (default: iex)
    ALPACA_FAIL_ON_FEED_ERROR=0|1     (default: 0; set 1 to hard-fail instead of retrying)
    ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0|1  (default: 1; set 0 to disable yfinance fallback)

Default:
    python premarket_trade_planner.py

Inputs:
    system-data/YYYYMMDD/YYYYMMDD_latest_llm_sentiment_universe.json
    or YYYYMMDD_locked_trading_universe.txt

Outputs:
    system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan.json
    system-data/YYYYMMDD/YYYYMMDD_intraday_structure_map.json
    system-data/YYYYMMDD/YYYYMMDD_premarket_trade_plan_diagnostics.json

Diagnostics printed:
    [data-source] discovery=tradingview_screener
    [data-feed] premarket=boats|sip
    [data-feed] regular=iex|sip
    [data-feed] yfinance_fallback=true|false
    [data-source] alpaca_primary=true|false
    [data-source] symbols_from_alpaca=N
"""

import json
import math
import os
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from symbol_utils import normalize_for_yfinance
from config.env_loader import load_instance_env

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient as _AlpacaHistoricalClient
    from alpaca.data.requests import StockBarsRequest as _AlpacaBarsRequest
    from alpaca.data.timeframe import TimeFrame as _AlpacaTimeFrame, TimeFrameUnit as _AlpacaTimeFrameUnit
except Exception:
    DataFeed = None
    _AlpacaHistoricalClient = None
    _AlpacaBarsRequest = None
    _AlpacaTimeFrame = None
    _AlpacaTimeFrameUnit = None

try:
    from tradingview_screener import Query, col
except Exception:
    Query = None
    col = None

try:
    import rookiepy
except Exception:
    rookiepy = None

try:
    from market_context_builder import TODAY_TAG
except Exception:
    TODAY_TAG = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


# =============================================================================
# Paths / environment
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_instance_env()

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

# Alpaca primary data source config — feed separation.
# Premarket bars (04:00–09:30 ET) require extended-hours coverage.
#   boats = Alpaca overnight/premarket feed (paid subscription)
#   sip   = SIP consolidated tape (extended hours, paid subscription)
# Regular-session bars (09:30–16:00 ET) use IEX by default (free tier).
ALPACA_PREMARKET_FEED = os.getenv("ALPACA_PREMARKET_FEED", "sip").lower().strip()
ALPACA_REGULAR_FEED = os.getenv("ALPACA_REGULAR_FEED", "iex").lower().strip()
# When 1: raise immediately on feed errors instead of silent retry/fallback.
ALPACA_FAIL_ON_FEED_ERROR = os.getenv("ALPACA_FAIL_ON_FEED_ERROR", "0").lower() in {"1", "true", "yes"}
# When 0: never fall back to yfinance for market data.
ALLOW_YFINANCE_MARKET_DATA_FALLBACK = os.getenv("ALLOW_YFINANCE_MARKET_DATA_FALLBACK", "1").lower() in {"1", "true", "yes"}
ALPACA_BARS_TIMEOUT = int(os.getenv("ALPACA_BARS_TIMEOUT_SECONDS", "30"))
ALPACA_BATCH_SIZE = int(os.getenv("ALPACA_BATCH_SIZE", "100"))

# Legacy alias kept for backward compat with any callers that set ALPACA_DATA_FEED.
# Overrides both PREMARKET and REGULAR if explicitly set.
_LEGACY_FEED = os.getenv("ALPACA_DATA_FEED", "").lower().strip()
if _LEGACY_FEED:
    ALPACA_PREMARKET_FEED = _LEGACY_FEED
    ALPACA_REGULAR_FEED = _LEGACY_FEED

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

# IMPORTANT: this planner is called by the premarket collector between 04:00 and 09:30 ET.
# The old default waited until 09:30, which caused the collector to hit its 900s timeout.
WAIT_FOR_MARKET_OPEN = os.getenv("PREMARKET_WAIT_FOR_MARKET_OPEN", "0").lower() in {"1", "true", "yes"}
MARKET_OPEN_WAIT_TIME = os.getenv("MARKET_OPEN_WAIT_TIME", "09:30")

# Runtime protection. The collector may call this every few minutes, so this script must finish fast.
MAX_PREMARKET_PLANNER_SYMBOLS = int(os.getenv("MAX_PREMARKET_PLANNER_SYMBOLS", "20"))
MAX_STRUCTURE_WORKERS = int(os.getenv("MAX_STRUCTURE_WORKERS", "6"))
PLANNER_TOTAL_TIMEOUT_SECONDS = int(os.getenv("PLANNER_TOTAL_TIMEOUT_SECONDS", "240"))
PER_SYMBOL_TIMEOUT_SECONDS = int(os.getenv("PER_SYMBOL_TIMEOUT_SECONDS", "8"))
MAX_PLAN_AGE_MINUTES = int(os.getenv("MAX_PLAN_AGE_MINUTES", "15"))
MIN_STRUCTURE_ROWS_REQUIRED = int(os.getenv("MIN_STRUCTURE_ROWS_REQUIRED", "60"))
BREAKOUT_CONFIRMATION_BUFFER_PCT = float(os.getenv("BREAKOUT_CONFIRMATION_BUFFER_PCT", "0.002"))
MIN_TRAINING_ROWS_REQUIRED = int(os.getenv("MIN_TRAINING_ROWS_REQUIRED", "250"))
SAFE_MODE_ON_PLANNER_DEGRADATION = os.getenv("SAFE_MODE_ON_PLANNER_DEGRADATION", "1").lower() in {"1", "true", "yes"}
YF_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("YF_DOWNLOAD_TIMEOUT_SECONDS", "45"))

# Optional TradingView screener pre-rank layer. This is used only for lightweight candidate
# ranking and multi-timeframe context. yfinance remains the structure-bar fallback.
TRADINGVIEW_SCREENER_ENABLED = os.getenv("TRADINGVIEW_SCREENER_ENABLED", "1").lower() in {"1", "true", "yes"}
TRADINGVIEW_USE_BROWSER_COOKIES = os.getenv("TRADINGVIEW_USE_BROWSER_COOKIES", "0").lower() in {"1", "true", "yes"}
TRADINGVIEW_COOKIE_BROWSER = os.getenv("TRADINGVIEW_COOKIE_BROWSER", "chrome").lower().strip()
TRADINGVIEW_QUERY_LIMIT = int(os.getenv("TRADINGVIEW_QUERY_LIMIT", "10000"))
TRADINGVIEW_MIN_RVOL = float(os.getenv("TRADINGVIEW_MIN_RVOL", "0.0"))
TRADINGVIEW_REQUIRE_1M_MACD_BULLISH = os.getenv("TRADINGVIEW_REQUIRE_1M_MACD_BULLISH", "0").lower() in {"1", "true", "yes"}
TRADINGVIEW_RANK_WEIGHT = float(os.getenv("TRADINGVIEW_RANK_WEIGHT", "0.35"))

# Pivot-point signal planner settings (LuxAlgo-style).
PIVOT_PLANNER_ENABLED             = int(os.getenv("PIVOT_PLANNER_ENABLED", "1"))
PIVOT_LENGTH                      = int(os.getenv("PIVOT_LENGTH", "5"))
PIVOT_MIN_RR                      = float(os.getenv("PIVOT_MIN_RR", "2.0"))
PIVOT_MAX_STOP_DISTANCE_PCT       = float(os.getenv("PIVOT_MAX_STOP_DISTANCE_PCT", "0.04"))
PIVOT_MIN_TARGET_DISTANCE_PCT     = float(os.getenv("PIVOT_MIN_TARGET_DISTANCE_PCT", "0.02"))
PIVOT_REQUIRE_VWAP_HOLD           = int(os.getenv("PIVOT_REQUIRE_VWAP_HOLD", "1"))
PIVOT_BREAKOUT_BUFFER_PCT         = float(os.getenv("PIVOT_BREAKOUT_BUFFER_PCT", "0.001"))
PIVOT_RECLAIM_BUFFER_PCT          = float(os.getenv("PIVOT_RECLAIM_BUFFER_PCT", "0.001"))
PIVOT_MAX_DISTANCE_FROM_LEVEL_PCT = float(os.getenv("PIVOT_MAX_DISTANCE_FROM_LEVEL_PCT", "0.015"))

# Module-level cache: populated by _analyze_symbol_pivot, consumed by build_intraday_structure_map_for_symbol.
_pivot_signals_cache: dict[tuple[str, str], dict] = {}

# Timeframe weights for pivot setup scoring.
_TF_WEIGHTS: dict[str, float] = {"1m": 1.0, "5m": 1.5, "15m": 2.0, "1h": 3.0}


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

    # Fallback chain when today's dated file isn't written yet (planner runs before
    # run.py creates it): try yesterday → non-dated fallback. Safe: these files
    # represent the approved universe from the most recent prior session.
    if not symbols:
        try:
            from datetime import timedelta
            _yesterday_tag = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
            _yesterday_path = PROJECT_ROOT / f"{_yesterday_tag}_locked_trading_universe.txt"
            if _yesterday_path.exists():
                symbols = [
                    normalize_symbol(line)
                    for line in _yesterday_path.read_text(encoding="utf-8").splitlines()
                    if normalize_symbol(line)
                ]
                print(f"[universe] using yesterday's locked universe ({_yesterday_tag}): {len(symbols)} symbols")
        except Exception as _exc:
            print(f"[universe] yesterday fallback failed: {_exc}")

    if not symbols:
        _fallback_path = PROJECT_ROOT / "locked_trading_universe.txt"
        if _fallback_path.exists():
            symbols = [
                normalize_symbol(line)
                for line in _fallback_path.read_text(encoding="utf-8").splitlines()
                if normalize_symbol(line)
            ]
            print(f"[universe] using non-dated locked_trading_universe.txt: {len(symbols)} symbols")

    symbols = sorted(set(s for s in symbols if s))
    print(f"[universe] symbols={len(symbols)}")
    return symbols


# =============================================================================
# Data-source diagnostics tracker (populated by fetch functions, emitted in output)
# =============================================================================

_DATA_SOURCE_DIAGNOSTICS: dict[str, Any] = {
    "alpaca_primary": False,
    "fallback_used": None,
    "symbols_from_alpaca": 0,
}


# =============================================================================
# Alpaca data ingestion (primary)
# =============================================================================

def _get_alpaca_data_client() -> Any:
    if _AlpacaHistoricalClient is None:
        return None
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        return None
    try:
        return _AlpacaHistoricalClient(key, secret)
    except Exception as exc:
        print(f"[alpaca-client] init failed: {exc}")
        return None


def _resolve_feed(feed_name: str) -> Any:
    """Map a feed name string to the DataFeed enum value, or None if SDK unavailable."""
    if DataFeed is None:
        return None
    name = feed_name.lower().strip()
    if name == "sip":
        return DataFeed.SIP
    if name in {"boats", "overnight"}:
        # BOATS is exposed as DataFeed.BOATS in alpaca-py >= 0.20; fall back to SIP.
        return getattr(DataFeed, "BOATS", DataFeed.SIP)
    return DataFeed.IEX


def _get_alpaca_premarket_feed() -> Any:
    return _resolve_feed(ALPACA_PREMARKET_FEED)


def _get_alpaca_regular_feed() -> Any:
    return _resolve_feed(ALPACA_REGULAR_FEED)


def _alpaca_bars_to_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca get_stock_bars().df output to the shared OHLCV schema."""
    if bars_df is None or bars_df.empty:
        return pd.DataFrame()

    df = bars_df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    if "symbol" not in df.columns:
        # Single-symbol case: index was just timestamp
        df["symbol"] = ""

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"] != ""].copy()

    for col_name in ["open", "high", "low", "close", "volume"]:
        if col_name not in df.columns:
            df[col_name] = 0.0
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp_et"] = df["timestamp"].dt.tz_convert(ET)

    return df[["symbol", "timestamp", "timestamp_et", "open", "high", "low", "close", "volume"]].copy()


_SIP_SUBSCRIPTION_ERR = "subscription does not permit querying recent SIP data"


def _alpaca_request_bars(
    client: Any,
    req_kwargs: dict[str, Any],
    feed: Any,
    label: str,
) -> pd.DataFrame:
    """
    Execute one Alpaca StockBarsRequest.

    If ALPACA_FAIL_ON_FEED_ERROR=1: any feed error raises immediately — no retry.
    Otherwise: SIP subscription errors auto-retry with IEX (regular session only;
    premarket bars may still fail if the account lacks extended-hours access).
    """
    kwargs = dict(req_kwargs)
    if feed is not None:
        kwargs["feed"] = feed

    try:
        raw = client.get_stock_bars(_AlpacaBarsRequest(**kwargs)).df
        return _alpaca_bars_to_df(raw)
    except Exception as exc:
        if ALPACA_FAIL_ON_FEED_ERROR:
            raise RuntimeError(
                f"[alpaca-{label}] feed error (ALPACA_FAIL_ON_FEED_ERROR=1): {exc}"
            ) from exc
        err_str = str(exc)
        if _SIP_SUBSCRIPTION_ERR in err_str and DataFeed is not None and feed != DataFeed.IEX:
            print(f"[alpaca-{label}] SIP subscription error, retrying with IEX feed")
            kwargs["feed"] = DataFeed.IEX
            try:
                raw = client.get_stock_bars(_AlpacaBarsRequest(**kwargs)).df
                return _alpaca_bars_to_df(raw)
            except Exception as exc2:
                print(f"[alpaca-{label}] IEX retry also failed: {exc2}")
                return pd.DataFrame()
        raise


def fetch_alpaca_minute_bars(
    symbols: list[str],
    start_et: datetime,
    end_et: datetime,
    label: str,
    interval_minutes: int = 1,
    feed: Any = None,
) -> pd.DataFrame:
    """
    Fetch intraday bars from Alpaca. Returns normalized DataFrame or empty if unavailable.

    feed: explicit DataFeed value; if None, defaults to the regular-session feed (IEX).
    Call sites pass _get_alpaca_premarket_feed() for premarket windows and
    _get_alpaca_regular_feed() for intraday windows.
    """
    if not symbols or _AlpacaBarsRequest is None or _AlpacaTimeFrame is None:
        return pd.DataFrame()

    client = _get_alpaca_data_client()
    if client is None:
        return pd.DataFrame()

    start_utc = start_et.astimezone(timezone.utc)
    end_utc = end_et.astimezone(timezone.utc)

    if interval_minutes == 1:
        tf = _AlpacaTimeFrame.Minute
    elif interval_minutes == 60:
        tf = _AlpacaTimeFrame.Hour
    else:
        tf = _AlpacaTimeFrame(interval_minutes, _AlpacaTimeFrameUnit.Minute)

    if feed is None:
        feed = _get_alpaca_regular_feed()

    frames: list[pd.DataFrame] = []
    diag: dict[str, Any] = {
        "label": label,
        "source": "alpaca",
        "feed": str(feed),
        "interval_minutes": interval_minutes,
        "window_et": {"start": start_et.isoformat(), "end": end_et.isoformat()},
        "batches": [],
    }

    # Alpaca supports up to 1000 symbols per request but we batch conservatively.
    for batch in chunked(symbols, ALPACA_BATCH_SIZE):
        batch_diag: dict[str, Any] = {"symbols": batch, "row_count": 0, "error": None}
        try:
            req_kwargs: dict[str, Any] = {
                "symbol_or_symbols": batch,
                "timeframe": tf,
                "start": start_utc,
                "end": end_utc,
            }
            df = _alpaca_request_bars(client, req_kwargs, feed, label)

            if not df.empty:
                # Filter strictly to the requested ET window (Alpaca may return slightly outside)
                df = df[(df["timestamp_et"] >= start_et) & (df["timestamp_et"] < end_et)].copy()

            if not df.empty:
                frames.append(df)
                counts = df.groupby("symbol").size().to_dict()
                batch_diag["row_count"] = int(len(df))
                batch_diag["counts"] = {str(k): int(v) for k, v in counts.items()}
                print(f"[alpaca-{label}] batch_counts={counts}")
            else:
                print(f"[alpaca-{label}] empty batch={batch}")

        except Exception as exc:
            batch_diag["error"] = str(exc)
            print(f"[alpaca-{label}] batch failed: {exc}")
        finally:
            diag["batches"].append(batch_diag)

    if not frames:
        diag["final_rows"] = 0
        diag["symbols_with_bars"] = []
        merge_diagnostics(f"alpaca_{label}", diag)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    diag["final_rows"] = int(len(out))
    diag["symbols_with_bars"] = sorted(out["symbol"].dropna().unique().tolist())
    merge_diagnostics(f"alpaca_{label}", diag)

    print(f"[alpaca-{label}] final_rows={len(out)} symbols={out['symbol'].nunique()}")
    return out


def fetch_alpaca_multitimeframe_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for 1m/5m/15m/1h from Alpaca for pivot signal detection."""
    if not symbols or _AlpacaBarsRequest is None or _AlpacaTimeFrame is None:
        return {}

    client = _get_alpaca_data_client()
    if client is None:
        return {}

    now_et = datetime.now(ET)
    now_utc = datetime.now(timezone.utc)
    # MTF bars span regular-session history; use the regular-session feed.
    feed = _get_alpaca_regular_feed()
    print(f"[data-feed] mtf=regular feed={ALPACA_REGULAR_FEED}")

    tf_config = [
        ("1m",  _AlpacaTimeFrame.Minute,                                        1,  now_utc - timedelta(days=2)),
        ("5m",  _AlpacaTimeFrame(5,  _AlpacaTimeFrameUnit.Minute) if _AlpacaTimeFrameUnit else None, 5,  now_utc - timedelta(days=7)),
        ("15m", _AlpacaTimeFrame(15, _AlpacaTimeFrameUnit.Minute) if _AlpacaTimeFrameUnit else None, 15, now_utc - timedelta(days=35)),
        ("1h",  _AlpacaTimeFrame.Hour,                                           60, now_utc - timedelta(days=95)),
    ]

    result: dict[str, pd.DataFrame] = {}

    for interval_label, tf, _interval_mins, start_utc in tf_config:
        if tf is None:
            result[interval_label] = pd.DataFrame()
            continue

        frames: list[pd.DataFrame] = []
        diag: dict[str, Any] = {
            "interval": interval_label, "source": "alpaca", "feed": str(feed), "batches": []
        }

        for batch in chunked(symbols, ALPACA_BATCH_SIZE):
            batch_diag: dict[str, Any] = {"symbols": batch, "row_count": 0, "error": None}
            try:
                req_kwargs: dict[str, Any] = {
                    "symbol_or_symbols": batch,
                    "timeframe": tf,
                    "start": start_utc,
                    "end": now_utc,
                }
                df = _alpaca_request_bars(client, req_kwargs, feed, f"mtf-{interval_label}")
                if not df.empty:
                    frames.append(df)
                    batch_diag["row_count"] = int(len(df))
            except Exception as exc:
                batch_diag["error"] = str(exc)
                print(f"[alpaca-mtf-{interval_label}] batch failed: {exc}")
            finally:
                diag["batches"].append(batch_diag)

        if frames:
            combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            result[interval_label] = combined
            diag["total_rows"] = int(len(combined))
            print(f"[alpaca-mtf-{interval_label}] rows={len(combined)} symbols={combined['symbol'].nunique()}")
        else:
            result[interval_label] = pd.DataFrame()
            diag["total_rows"] = 0
            print(f"[alpaca-mtf-{interval_label}] no bars returned")

        merge_diagnostics(f"alpaca_mtf_{interval_label}", diag)

    return result


# =============================================================================
# yfinance data ingestion (fallback)
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


def _fetch_minute_bars_yfinance(
    symbols: list[str],
    start_et: datetime,
    end_et: datetime,
    label: str,
) -> pd.DataFrame:
    """Inner yfinance 1-minute bar fetcher. Called only when Alpaca is unavailable or returns empty."""
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
        tickers = " ".join(normalize_for_yfinance(s) for s in batch)

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
                timeout=YF_DOWNLOAD_TIMEOUT_SECONDS,
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


def fetch_minute_bars(
    symbols: list[str],
    start_et: datetime,
    end_et: datetime,
    label: str,
    alpaca_feed: Any = None,
) -> pd.DataFrame:
    """
    Fetch 1-minute bars. Alpaca is always attempted first.

    alpaca_feed: DataFeed to pass to Alpaca (premarket feed for extended-hours windows,
                 regular feed for intraday windows). Callers must supply the right feed.

    yfinance fallback is only used when ALLOW_YFINANCE_MARKET_DATA_FALLBACK=1 (default).
    When ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0 and Alpaca returns empty, returns empty DataFrame.
    """
    if not symbols:
        return pd.DataFrame()

    # --- Primary: Alpaca ---
    alpaca_df = fetch_alpaca_minute_bars(
        symbols, start_et, end_et, label, interval_minutes=1, feed=alpaca_feed
    )

    if not alpaca_df.empty:
        syms = sorted(alpaca_df["symbol"].dropna().unique().tolist())
        _DATA_SOURCE_DIAGNOSTICS["alpaca_primary"] = True
        _DATA_SOURCE_DIAGNOSTICS["symbols_from_alpaca"] = max(
            _DATA_SOURCE_DIAGNOSTICS.get("symbols_from_alpaca", 0), len(syms)
        )
        print(f"[data-source] alpaca_primary=true label={label} symbols_from_alpaca={len(syms)}")
        return alpaca_df

    # --- Fallback: yfinance (only when allowed) ---
    if not ALLOW_YFINANCE_MARKET_DATA_FALLBACK:
        print(
            f"[data-source] alpaca_empty label={label} "
            f"yfinance_fallback=false (ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0)"
        )
        _DATA_SOURCE_DIAGNOSTICS["fallback_used"] = None
        return pd.DataFrame()

    print(f"[data-source] fallback_used=yfinance label={label} reason=alpaca_empty_or_unavailable")
    _DATA_SOURCE_DIAGNOSTICS["fallback_used"] = "yfinance"
    return _fetch_minute_bars_yfinance(symbols, start_et, end_et, label)


def fetch_premarket_bars(symbols: list[str]) -> pd.DataFrame:
    day = today_et_date()
    start_et = datetime.combine(day, parse_hhmm(PREMARKET_START_ET), tzinfo=ET)
    end_et = datetime.combine(day, parse_hhmm(PREMARKET_END_ET), tzinfo=ET)
    feed = _get_alpaca_premarket_feed()
    print(f"[data-feed] premarket={ALPACA_PREMARKET_FEED} yfinance_fallback={ALLOW_YFINANCE_MARKET_DATA_FALLBACK}")
    return fetch_minute_bars(symbols, start_et, end_et, label="premarket_bars", alpaca_feed=feed)


def fetch_intraday_bars(symbols: list[str]) -> pd.DataFrame:
    day = today_et_date()
    start_et = datetime.combine(day - timedelta(days=INTRADAY_LOOKBACK_DAYS), parse_hhmm(REGULAR_SESSION_START_ET), tzinfo=ET)
    end_et = datetime.now(ET)
    regular_end_today = datetime.combine(day, parse_hhmm(REGULAR_SESSION_END_ET), tzinfo=ET)
    end_et = min(end_et, regular_end_today)
    feed = _get_alpaca_regular_feed()
    print(f"[data-feed] regular={ALPACA_REGULAR_FEED} yfinance_fallback={ALLOW_YFINANCE_MARKET_DATA_FALLBACK}")
    return fetch_minute_bars(symbols, start_et, end_et, label="intraday_bars", alpaca_feed=feed)


def _fetch_multitimeframe_bars_yfinance(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """yfinance fallback for multi-timeframe bar fetching."""
    tf_config = [
        ("1m",  "1d"),
        ("5m",  "5d"),
        ("15m", "1mo"),
        ("1h",  "3mo"),
    ]

    result: dict[str, pd.DataFrame] = {}

    for interval, period in tf_config:
        label = f"mtf_{interval}"
        frames: list[pd.DataFrame] = []
        diag: dict[str, Any] = {"interval": interval, "period": period, "source": "yfinance", "batches": []}

        for batch in chunked(symbols, YF_BATCH_SIZE):
            batch_diag: dict[str, Any] = {"symbols": batch, "row_count": 0, "error": None}
            tickers = " ".join(normalize_for_yfinance(s) for s in batch)
            try:
                raw = yf.download(
                    tickers=tickers,
                    period=period,
                    interval=interval,
                    prepost=False,
                    group_by="column",
                    auto_adjust=YF_AUTO_ADJUST,
                    progress=False,
                    threads=True,
                    timeout=YF_DOWNLOAD_TIMEOUT_SECONDS,
                )
                df = flatten_yfinance_download(raw, batch)
                if not df.empty:
                    frames.append(df)
                    batch_diag["row_count"] = int(len(df))
            except Exception as exc:
                batch_diag["error"] = str(exc)
                print(f"[{label}] batch failed: {exc}")
            finally:
                diag["batches"].append(batch_diag)

        if frames:
            combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            result[interval] = combined
            diag["total_rows"] = int(len(combined))
            print(f"[{label}] rows={len(combined)} symbols={combined['symbol'].nunique()}")
        else:
            result[interval] = pd.DataFrame()
            diag["total_rows"] = 0
            print(f"[{label}] no bars returned")

        merge_diagnostics(label, diag)

    return result


def fetch_multitimeframe_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for 1m/5m/15m/1h for pivot signal detection.
    Alpaca is the primary source using the regular-session feed (IEX by default).
    yfinance is used to fill empty timeframes only when ALLOW_YFINANCE_MARKET_DATA_FALLBACK=1.
    """
    if not symbols:
        return {}

    print(f"[data-feed] regular={ALPACA_REGULAR_FEED} yfinance_fallback={ALLOW_YFINANCE_MARKET_DATA_FALLBACK}")

    # --- Primary: Alpaca ---
    alpaca_result = fetch_alpaca_multitimeframe_bars(symbols)

    alpaca_has_data = any(not df.empty for df in alpaca_result.values())

    if alpaca_has_data:
        needs_yf_fallback = [tf for tf, df in alpaca_result.items() if df.empty]

        if needs_yf_fallback and ALLOW_YFINANCE_MARKET_DATA_FALLBACK:
            print(f"[data-source] alpaca_primary=true mtf_yfinance_fallback_tfs={needs_yf_fallback}")
            _DATA_SOURCE_DIAGNOSTICS["fallback_used"] = "yfinance_partial_mtf"
            yf_tf_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "3mo"}
            for tf in needs_yf_fallback:
                period = yf_tf_map.get(tf, "5d")
                label = f"mtf_{tf}_yf_fallback"
                frames: list[pd.DataFrame] = []
                for batch in chunked(symbols, YF_BATCH_SIZE):
                    try:
                        raw = yf.download(
                            tickers=" ".join(normalize_for_yfinance(s) for s in batch),
                            period=period,
                            interval=tf,
                            prepost=False,
                            group_by="column",
                            auto_adjust=YF_AUTO_ADJUST,
                            progress=False,
                            threads=True,
                            timeout=YF_DOWNLOAD_TIMEOUT_SECONDS,
                        )
                        df = flatten_yfinance_download(raw, batch)
                        if not df.empty:
                            frames.append(df)
                    except Exception as exc:
                        print(f"[{label}] batch failed: {exc}")
                if frames:
                    alpaca_result[tf] = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
                    print(f"[{label}] filled tf={tf} rows={len(alpaca_result[tf])}")
        elif needs_yf_fallback:
            print(
                f"[data-source] alpaca_primary=true mtf_empty_tfs={needs_yf_fallback} "
                f"yfinance_fallback=false (ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0)"
            )

        print(f"[data-source] alpaca_primary=true mtf tfs={list(alpaca_result.keys())}")
        return alpaca_result

    # --- Full fallback: yfinance for all timeframes ---
    if not ALLOW_YFINANCE_MARKET_DATA_FALLBACK:
        print(
            "[data-source] alpaca_empty mtf "
            "yfinance_fallback=false (ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0)"
        )
        _DATA_SOURCE_DIAGNOSTICS["fallback_used"] = None
        return {}

    print("[data-source] fallback_used=yfinance mtf reason=alpaca_empty_or_unavailable")
    _DATA_SOURCE_DIAGNOSTICS["fallback_used"] = "yfinance"
    return _fetch_multitimeframe_bars_yfinance(symbols)


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


def detect_pivot_signals(df: pd.DataFrame, length: int = PIVOT_LENGTH) -> dict[str, Any]:
    """
    LuxAlgo-style pivot detection with missed pivot (ghost) levels.

    A bar is a pivot high if its high is the unique maximum of [i-length .. i+length].
    A bar is a pivot low if its low is the unique minimum of [i-length .. i+length].

    Missed pivot highs (ghost resistance): the max high tracked between two consecutive
    pivot lows that exceeds the next confirmed pivot high — an unconfirmed swing high.

    Missed pivot lows (ghost support): the min low tracked between two consecutive
    pivot highs that falls below the next confirmed pivot low — an unconfirmed swing low.
    """
    if len(df) < 2 * length + 1:
        return {"valid": False}

    df = df.sort_values("timestamp").reset_index(drop=True)
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    timestamps = df["timestamp"].tolist()
    n = len(df)

    ph_events: list[tuple[int, float, Any]] = []
    pl_events: list[tuple[int, float, Any]] = []

    for i in range(length, n - length):
        h = highs[i]
        window_h = highs[i - length:i + length + 1]
        if h == max(window_h) and window_h.count(h) == 1:
            ph_events.append((i, h, timestamps[i]))

        lo = lows[i]
        window_l = lows[i - length:i + length + 1]
        if lo == min(window_l) and window_l.count(lo) == 1:
            pl_events.append((i, lo, timestamps[i]))

    # Merge and sort all pivot events by bar index
    all_events: list[tuple[int, str, float, Any]] = (
        [(b, "ph", p, t) for b, p, t in ph_events] +
        [(b, "pl", p, t) for b, p, t in pl_events]
    )
    all_events.sort(key=lambda x: x[0])

    os_state = 0  # 0=last confirmed pivot was a low, 1=last confirmed was a high
    last_ph: tuple | None = None
    last_pl: tuple | None = None
    tracked_max = float("-inf")
    tracked_min = float("inf")
    missed_highs: list[dict[str, Any]] = []
    missed_lows: list[dict[str, Any]] = []
    ghost_resistance: list[float] = []
    ghost_support: list[float] = []

    bar_cursor = 0
    for (bar_idx, etype, price, t) in all_events:
        while bar_cursor < bar_idx:
            tracked_max = max(tracked_max, highs[bar_cursor])
            tracked_min = min(tracked_min, lows[bar_cursor])
            bar_cursor += 1

        if etype == "ph":
            if os_state == 0 and last_pl is not None and tracked_max > price * 1.001:
                missed_highs.append({"price": round(tracked_max, 4), "bar": bar_idx})
                ghost_resistance.append(round(tracked_max, 4))
            os_state = 1
            last_ph = (bar_idx, price, t)
            tracked_min = float("inf")
        else:  # pl
            if os_state == 1 and last_ph is not None and tracked_min < price * 0.999:
                missed_lows.append({"price": round(tracked_min, 4), "bar": bar_idx})
                ghost_support.append(round(tracked_min, 4))
            os_state = 0
            last_pl = (bar_idx, price, t)
            tracked_max = float("-inf")

    active_range: dict[str, Any] | None = None
    if last_ph and last_pl:
        active_range = {
            "high": round(last_ph[1], 4),
            "low": round(last_pl[1], 4),
            "midpoint": round((last_ph[1] + last_pl[1]) / 2.0, 4),
        }

    return {
        "valid": True,
        "pivot_highs": [{"bar": i, "price": round(p, 4), "time": str(t)} for i, p, t in ph_events],
        "pivot_lows": [{"bar": i, "price": round(p, 4), "time": str(t)} for i, p, t in pl_events],
        "missed_highs": missed_highs,
        "missed_lows": missed_lows,
        "ghost_resistance": ghost_resistance,
        "ghost_support": ghost_support,
        "active_range": active_range,
        "os_state": os_state,
        "last_ph": {"bar": last_ph[0], "price": round(last_ph[1], 4)} if last_ph else None,
        "last_pl": {"bar": last_pl[0], "price": round(last_pl[1], 4)} if last_pl else None,
    }


def generate_pivot_trade_plan(
    symbol: str,
    mtf_signals: dict[str, dict[str, Any]],
    current_price: float,
    vwap: float | None,
) -> dict[str, Any] | None:
    """
    Score pivot setups across timeframes and return the best valid long plan, or None.

    Setup types:
      reclaim     — price just above a pivot low (support reclaim / bounce)
      breakout    — price just below a pivot high (imminent breakout)
      ghost_support  — price near a missed pivot low ghost level
      ghost_breakout — price near a missed pivot high ghost level
    """
    rejections: list[str] = []

    if PIVOT_REQUIRE_VWAP_HOLD and vwap is not None and vwap > 0 and current_price < vwap:
        rejections.append("pivot_rejected_below_vwap")
        merge_diagnostics(
            f"pivot_rejections_{symbol}",
            {"symbol": symbol, "rejections": rejections, "current_price": current_price, "vwap": vwap},
        )
        return None

    setups: list[dict[str, Any]] = []
    any_level_within_range = False

    for tf, sigs in mtf_signals.items():
        if not sigs.get("valid"):
            continue

        tf_weight = _TF_WEIGHTS.get(tf, 1.0)
        ph_prices = [x["price"] for x in sigs.get("pivot_highs", [])[-5:]]
        pl_prices = [x["price"] for x in sigs.get("pivot_lows", [])[-5:]]
        ghost_res = list(sigs.get("ghost_resistance", []))[-5:]
        ghost_sup = list(sigs.get("ghost_support", []))[-5:]
        all_res = sorted(set(ph_prices + ghost_res))
        all_pl = sorted(set(pl_prices + ghost_sup))

        # Reclaim and ghost support setups: price near or just above a support level
        for level_price in pl_prices + ghost_sup:
            if level_price <= 0:
                continue
            dist = (current_price - level_price) / level_price
            if abs(dist) > PIVOT_MAX_DISTANCE_FROM_LEVEL_PCT:
                continue
            any_level_within_range = True
            if current_price < level_price:
                continue  # Below support — not a reclaim

            entry = round(level_price * (1.0 + PIVOT_RECLAIM_BUFFER_PCT), 4)
            stop = round(level_price * (1.0 - 0.005), 4)
            if entry <= stop:
                continue
            stop_dist = (entry - stop) / max(entry, 1e-9)
            if stop_dist > PIVOT_MAX_STOP_DISTANCE_PCT:
                rejections.append("pivot_rejected_stop_too_wide")
                continue

            min_target = entry * (1.0 + PIVOT_MIN_TARGET_DISTANCE_PCT)
            next_res = [r for r in all_res if r >= min_target]
            target = min(next_res) if next_res else round(entry + 2.0 * (entry - stop), 4)
            rr = (target - entry) / max(entry - stop, 1e-9)
            if rr < PIVOT_MIN_RR:
                rejections.append("pivot_rejected_low_rr")
                continue

            src = "ghost_support" if level_price in ghost_sup else "reclaim"
            setups.append({
                "type": src, "timeframe": tf, "entry": entry, "stop": stop,
                "target": target, "rr": round(rr, 4), "pivot_level": round(level_price, 4),
                "setup_weight": round(tf_weight * rr, 4),
            })

        # Breakout and ghost breakout setups: price just below a resistance level
        for level_price in ph_prices + ghost_res:
            if level_price <= 0:
                continue
            dist = (current_price - level_price) / level_price
            if abs(dist) > PIVOT_MAX_DISTANCE_FROM_LEVEL_PCT:
                continue
            any_level_within_range = True
            if current_price >= level_price:
                continue  # Already above — not a pending breakout

            entry = round(level_price * (1.0 + PIVOT_BREAKOUT_BUFFER_PCT), 4)
            pl_below = [p for p in all_pl if p < entry]
            stop = round(max(pl_below) * 0.995, 4) if pl_below else round(entry * (1.0 - PIVOT_MAX_STOP_DISTANCE_PCT), 4)
            if entry <= stop:
                continue
            stop_dist = (entry - stop) / max(entry, 1e-9)
            if stop_dist > PIVOT_MAX_STOP_DISTANCE_PCT:
                rejections.append("pivot_rejected_stop_too_wide")
                continue

            min_target = entry * (1.0 + PIVOT_MIN_TARGET_DISTANCE_PCT)
            next_res = [r for r in all_res if r >= min_target and r > level_price * 1.005]
            target = min(next_res) if next_res else round(entry + 2.0 * (entry - stop), 4)
            rr = (target - entry) / max(entry - stop, 1e-9)
            if rr < PIVOT_MIN_RR:
                rejections.append("pivot_rejected_low_rr")
                continue

            src = "ghost_breakout" if level_price in ghost_res else "breakout"
            setups.append({
                "type": src, "timeframe": tf, "entry": entry, "stop": stop,
                "target": target, "rr": round(rr, 4), "pivot_level": round(level_price, 4),
                "setup_weight": round(tf_weight * rr, 4),
            })

    if not any_level_within_range:
        rejections.append("pivot_rejected_mid_range")

    if rejections:
        merge_diagnostics(
            f"pivot_rejections_{symbol}",
            {"symbol": symbol, "rejections": rejections, "setups_found": len(setups), "current_price": current_price, "vwap": vwap},
        )

    if not setups:
        return None

    return max(setups, key=lambda x: x["setup_weight"])


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

    # Ghost reversal levels from higher-timeframe pivot signal cache (15m and 1h only).
    for tf in ("1h", "15m"):
        cached = _pivot_signals_cache.get((symbol, tf), {})
        for gr in cached.get("ghost_resistance", []):
            if gr and gr > 0:
                level_candidates.append({"price": gr, "source": f"ghost_resistance_{tf}", "touch_count": 1, "weight": 2.5})
        for gs in cached.get("ghost_support", []):
            if gs and gs > 0:
                level_candidates.append({"price": gs, "source": f"ghost_support_{tf}", "touch_count": 1, "weight": 2.5})

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
            "raw_resistance": resistances[0]["price"] if resistances else None,
            "breakout_confirmation_buffer_pct": BREAKOUT_CONFIRMATION_BUFFER_PCT,
            "buffered_long_trigger": round(resistances[0]["price"] * (1.0 + BREAKOUT_CONFIRMATION_BUFFER_PCT), 4) if resistances and resistances[0]["price"] else None,
            "preferred_long_trigger": round(resistances[0]["price"] * (1.0 + BREAKOUT_CONFIRMATION_BUFFER_PCT), 4) if resistances and resistances[0]["price"] else None,
            "preferred_short_trigger": supports[0]["price"] if supports else None,
            "preferred_long_stop": supports[0]["price"] if supports else None,
            "preferred_short_stop": resistances[0]["price"] if resistances else None,
            "notes": [
                "Use this as a structure map, not a standalone order signal.",
                "Require ML/RL direction confirmation before entries.",
                "Always allow exits regardless of long_allowed/short_or_put_allowed.",
                f"preferred_long_trigger is {BREAKOUT_CONFIRMATION_BUFFER_PCT:.1%} above raw_resistance to reduce false breakouts.",
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


def _analyze_symbol_pivot(symbol: str, bars: pd.DataFrame, mtf_bars: dict[str, pd.DataFrame]) -> PremarketPlan:
    """
    Pivot-point-driven analysis: replaces consolidation pattern when PIVOT_PLANNER_ENABLED=1.
    Detects LuxAlgo-style pivot highs/lows and missed pivot ghost levels across
    1m/5m/15m/1h bars and generates a long trade plan with entry/stop/target/RR.
    """
    if bars.empty or "symbol" not in bars.columns:
        return empty_plan(symbol, "pivot_rejected_insufficient_bars")

    sdf = bars[bars["symbol"] == symbol].copy()
    if len(sdf) < 5:
        return empty_plan(symbol, "pivot_rejected_insufficient_bars")

    sdf = sdf.sort_values("timestamp").reset_index(drop=True)
    premarket_volume = int(pd.to_numeric(sdf["volume"], errors="coerce").fillna(0).sum())
    premarket_open   = safe_float(sdf.iloc[0].get("open", 0))
    premarket_close  = safe_float(sdf.iloc[-1].get("close", 0))
    premarket_high   = safe_float(sdf["high"].max())
    premarket_low    = safe_float(sdf["low"].min())
    current_price    = premarket_close or premarket_high or premarket_open

    if current_price <= 0:
        return empty_plan(symbol, "pivot_rejected_insufficient_bars")

    # VWAP from premarket 1m bars
    vwap: float | None = None
    if len(sdf) >= 5:
        sdf_v = add_vwap_columns(sdf)
        v_series = sdf_v.get("vwap")
        if v_series is not None and not v_series.empty:
            vwap = safe_float(v_series.iloc[-1], 0.0) or None

    # Run pivot detection on each timeframe; populate cache for structure map
    mtf_signals: dict[str, dict[str, Any]] = {}
    for tf, tf_df in mtf_bars.items():
        if tf_df is None or tf_df.empty:
            mtf_signals[tf] = {"valid": False}
            continue
        sym_df = tf_df[tf_df.get("symbol", pd.Series(dtype=str)).eq(symbol)].copy()
        if len(sym_df) < 2 * PIVOT_LENGTH + 1:
            mtf_signals[tf] = {"valid": False}
            continue
        sigs = detect_pivot_signals(sym_df, PIVOT_LENGTH)
        mtf_signals[tf] = sigs
        _pivot_signals_cache[(symbol, tf)] = sigs

    # Fall back to premarket bars for 1m if MTF fetch did not cover it
    if not mtf_signals.get("1m", {}).get("valid") and len(sdf) >= 2 * PIVOT_LENGTH + 1:
        pm_sigs = detect_pivot_signals(sdf, PIVOT_LENGTH)
        mtf_signals["1m"] = pm_sigs
        _pivot_signals_cache[(symbol, "1m")] = pm_sigs

    setup = generate_pivot_trade_plan(symbol, mtf_signals, current_price, vwap)

    if setup is None:
        p = empty_plan(symbol, "no_valid_pivot_setup")
        p.premarket_volume = premarket_volume
        p.premarket_open   = round(premarket_open, 4)
        p.premarket_high   = round(premarket_high, 4)
        p.premarket_low    = round(premarket_low, 4)
        p.premarket_close  = round(premarket_close, 4)
        return p

    entry          = setup["entry"]
    stop           = setup["stop"]
    target         = setup["target"]
    rr             = setup["rr"]
    risk_per_share = round(entry - stop, 4)
    reward_per_share = round(target - entry, 4)
    initial_move_pct = round(((premarket_high - premarket_open) / premarket_open) * 100.0, 4) if premarket_open > 0 else None

    return PremarketPlan(
        symbol=symbol,
        valid=True,
        reason=f"pivot_{setup['type']}_{setup['timeframe']}: rr={rr:.2f} level={setup['pivot_level']}",
        premarket_volume=premarket_volume,
        premarket_open=round(premarket_open, 4),
        premarket_high=round(premarket_high, 4),
        premarket_low=round(premarket_low, 4),
        premarket_close=round(premarket_close, 4),
        initial_move_pct=initial_move_pct,
        consolidation_count=0,
        consolidation_start=None,
        consolidation_end=None,
        support=stop,
        resistance=target,
        entry_price=entry,
        stop_loss=stop,
        target_5pct=round(entry * 1.05, 4),
        target_risk_reward=round(entry + risk_per_share * RISK_REWARD_RATIO, 4),
        target_measured_move=target,
        primary_target=target,
        measured_range=round(target - stop, 4),
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=round(rr, 4),
    )


def analyze_symbol(symbol: str, bars: pd.DataFrame, mtf_bars: dict[str, pd.DataFrame] | None = None) -> PremarketPlan:
    if PIVOT_PLANNER_ENABLED:
        return _analyze_symbol_pivot(symbol, bars, mtf_bars or {})

    if bars.empty or "symbol" not in bars.columns:
        return empty_plan(symbol, "No premarket bars found (Alpaca and yfinance both returned empty). No premarket breakout entry will be created.")

    sdf = bars[bars["symbol"] == symbol].copy()
    if sdf.empty:
        return empty_plan(symbol, "No premarket bars found (Alpaca and yfinance both returned empty). No premarket breakout entry will be created.")

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
        reason="Valid setup: premarket volume confirmed, strong initial move, consolidation, support/resistance marked.",
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
# Universe scoring / runtime guards
# =============================================================================

def seconds_remaining(start_monotonic: float, total_timeout: int = PLANNER_TOTAL_TIMEOUT_SECONDS) -> float:
    return max(0.0, float(total_timeout) - (time_module.monotonic() - start_monotonic))


def deadline_exceeded(start_monotonic: float, total_timeout: int = PLANNER_TOTAL_TIMEOUT_SECONDS) -> bool:
    return seconds_remaining(start_monotonic, total_timeout) <= 0


def build_minimal_structure(symbol: str, plan: PremarketPlan, reason: str) -> dict[str, Any]:
    current_price = plan.premarket_close or plan.premarket_high or plan.premarket_low
    return {
        "symbol": symbol,
        "generated_at_utc": now_utc(),
        "current_price": round_price(current_price),
        "premarket_setup_valid": plan.valid,
        "premarket_reason": plan.reason,
        "support_levels": [],
        "resistance_levels": [],
        "all_clustered_levels": [],
        "nearest_levels": {"nearest_support": None, "nearest_resistance": None, "near_any_level": False},
        "orb": {"available": False, "reason": reason, "minutes": ORB_MINUTES, "high": None, "low": None, "mid": None, "range_pct": None, "breakout_confirmed": False, "breakdown_confirmed": False},
        "vwap": {"available": False, "reason": reason},
        "liquidity": {"bar_count": 0, "total_volume": plan.premarket_volume, "total_dollar_volume": 0.0, "avg_bar_volume": 0.0, "avg_bar_dollar_volume": 0.0, "avg_range_pct": None, "liquidity_score": 0.0, "liquid_enough": False, "reason": reason},
        "trade_structure": {
            "long_allowed": False,
            "short_or_put_allowed": False,
            "preferred_long_trigger": None,
            "preferred_short_trigger": None,
            "preferred_long_stop": None,
            "preferred_short_stop": None,
            "notes": [
                "Planner degraded for this symbol; use exits only until a fresh structure map is available.",
                "Always allow exits regardless of planner status.",
            ],
        },
    }


def load_tradingview_cookies() -> Any:
    if not TRADINGVIEW_USE_BROWSER_COOKIES or rookiepy is None:
        return None
    try:
        loader = getattr(rookiepy, TRADINGVIEW_COOKIE_BROWSER, None)
        if loader is None:
            print(f"[tradingview] cookie browser unsupported={TRADINGVIEW_COOKIE_BROWSER}; continuing without cookies")
            return None
        return rookiepy.to_cookiejar(loader([".tradingview.com"]))
    except Exception as exc:
        print(f"[tradingview] cookie load failed: {exc}; continuing without cookies")
        return None


def fetch_tradingview_snapshot(symbols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("[data-source] discovery=tradingview_screener")
    diagnostics: dict[str, Any] = {
        "enabled": bool(TRADINGVIEW_SCREENER_ENABLED),
        "available": bool(Query is not None),
        "requested_symbols": symbols,
        "row_count": 0,
        "matched_symbols": [],
        "error": None,
    }
    if not TRADINGVIEW_SCREENER_ENABLED:
        diagnostics["error"] = "disabled_by_env"
        return pd.DataFrame(), diagnostics
    if Query is None:
        diagnostics["error"] = "tradingview_screener_not_installed"
        print("[tradingview] package missing; install with: pip install tradingview-screener")
        return pd.DataFrame(), diagnostics
    if not symbols:
        diagnostics["error"] = "no_symbols"
        return pd.DataFrame(), diagnostics

    columns = [
        "name",
        "close",
        "close|1",
        "close|5",
        "close|15",
        "close|60",
        "volume",
        "relative_volume_10d_calc",
        "MACD.macd|1",
        "MACD.signal|1",
        "MACD.macd|5",
        "MACD.signal|5",
        "market_cap_basic",
        "exchange",
        "update_mode",
    ]

    try:
        cookies = load_tradingview_cookies()
        query = Query().select(*columns).order_by("volume", ascending=False).limit(TRADINGVIEW_QUERY_LIMIT)

        # Some versions of tradingview-screener support SQL-like list filters, others do not.
        # Try a targeted query first; fall back to a broad high-volume snapshot and filter locally.
        try:
            if col is not None and hasattr(col("name"), "isin"):
                query = query.where(col("name").isin(symbols))
        except Exception:
            pass

        total, df = query.get_scanner_data(cookies=cookies) if cookies is not None else query.get_scanner_data()
        if df is None or df.empty:
            diagnostics["total_available"] = int(total) if total is not None else 0
            diagnostics["error"] = "empty_response"
            print("[tradingview] empty response")
            merge_diagnostics("tradingview_snapshot", diagnostics)
            return pd.DataFrame(), diagnostics

        out = df.copy()
        out.columns = [str(c) for c in out.columns]
        if "name" not in out.columns:
            diagnostics["error"] = "missing_name_column"
            merge_diagnostics("tradingview_snapshot", diagnostics)
            return pd.DataFrame(), diagnostics

        wanted = {normalize_symbol(s) for s in symbols}
        out["symbol"] = out["name"].map(normalize_symbol)
        out = out[out["symbol"].isin(wanted)].copy()

        # Normalize useful numeric fields, regardless of whether TradingView returned every column.
        for c in columns:
            if c not in out.columns:
                out[c] = None
        numeric_cols = ["close", "close|1", "close|5", "close|15", "close|60", "volume", "relative_volume_10d_calc", "MACD.macd|1", "MACD.signal|1", "MACD.macd|5", "MACD.signal|5", "market_cap_basic"]
        for c in numeric_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        diagnostics["total_available"] = int(total) if total is not None else None
        diagnostics["row_count"] = int(len(out))
        diagnostics["matched_symbols"] = sorted(out["symbol"].dropna().unique().tolist())
        diagnostics["using_browser_cookies"] = bool(cookies is not None)
        diagnostics["query_limit"] = TRADINGVIEW_QUERY_LIMIT
        print(f"[tradingview] matched={len(diagnostics['matched_symbols'])}/{len(symbols)} cookies={bool(cookies is not None)}")
        merge_diagnostics("tradingview_snapshot", diagnostics)
        return out, diagnostics
    except Exception as exc:
        diagnostics["error"] = str(exc)
        print(f"[tradingview] failed: {exc}; falling back to yfinance-only ranking")
        merge_diagnostics("tradingview_snapshot", diagnostics)
        return pd.DataFrame(), diagnostics


def tradingview_score_for_symbol(symbol: str, tv_snapshot: pd.DataFrame) -> dict[str, Any]:
    if tv_snapshot.empty or "symbol" not in tv_snapshot.columns:
        return {"tv_score": 0.0, "tv_reason": "no_tradingview_snapshot"}
    row_df = tv_snapshot[tv_snapshot["symbol"].eq(symbol)]
    if row_df.empty:
        return {"tv_score": 0.0, "tv_reason": "symbol_not_in_tradingview_snapshot"}
    row = row_df.iloc[0]

    close = safe_float(row.get("close"), 0.0)
    close_1m = safe_float(row.get("close|1"), close)
    close_5m = safe_float(row.get("close|5"), close)
    close_15m = safe_float(row.get("close|15"), close)
    close_60m = safe_float(row.get("close|60"), close)
    volume = safe_float(row.get("volume"), 0.0)
    rvol = safe_float(row.get("relative_volume_10d_calc"), 0.0)
    macd_1 = safe_float(row.get("MACD.macd|1"), 0.0)
    signal_1 = safe_float(row.get("MACD.signal|1"), 0.0)
    macd_5 = safe_float(row.get("MACD.macd|5"), 0.0)
    signal_5 = safe_float(row.get("MACD.signal|5"), 0.0)

    mtf_1m_pct = ((close_1m - close_5m) / close_5m) * 100.0 if close_5m > 0 else 0.0
    mtf_5m_pct = ((close_5m - close_15m) / close_15m) * 100.0 if close_15m > 0 else 0.0
    mtf_15m_pct = ((close_15m - close_60m) / close_60m) * 100.0 if close_60m > 0 else 0.0
    macd_1_bullish = macd_1 >= signal_1
    macd_5_bullish = macd_5 >= signal_5

    if TRADINGVIEW_REQUIRE_1M_MACD_BULLISH and not macd_1_bullish:
        penalty = 2.0
    else:
        penalty = 0.0

    rvol_score = min(5.0, max(0.0, rvol))
    volume_score = min(5.0, volume / 1_000_000.0)
    mtf_score = max(0.0, min(5.0, (mtf_1m_pct + mtf_5m_pct + mtf_15m_pct) / 1.5))
    macd_score = (1.25 if macd_1_bullish else 0.0) + (0.75 if macd_5_bullish else 0.0)
    tv_score = round(0.35 * rvol_score + 0.25 * volume_score + 0.25 * mtf_score + 0.15 * macd_score - penalty, 4)

    if rvol < TRADINGVIEW_MIN_RVOL:
        tv_score = min(tv_score, 0.0)

    return {
        "tv_score": tv_score,
        "tv_reason": "ok",
        "tv_close": round_price(close),
        "tv_close_1m": round_price(close_1m),
        "tv_close_5m": round_price(close_5m),
        "tv_close_15m": round_price(close_15m),
        "tv_close_60m": round_price(close_60m),
        "tv_volume": int(volume),
        "tv_relative_volume_10d": round(rvol, 4),
        "tv_macd_1m_bullish": bool(macd_1_bullish),
        "tv_macd_5m_bullish": bool(macd_5_bullish),
        "tv_mtf_1m_vs_5m_pct": round(mtf_1m_pct, 4),
        "tv_mtf_5m_vs_15m_pct": round(mtf_5m_pct, 4),
        "tv_mtf_15m_vs_60m_pct": round(mtf_15m_pct, 4),
        "tv_exchange": row.get("exchange"),
        "tv_update_mode": row.get("update_mode"),
    }


def score_symbol_from_premarket(symbol: str, premarket_bars: pd.DataFrame) -> dict[str, Any]:
    if premarket_bars.empty or "symbol" not in premarket_bars.columns:
        return {"symbol": symbol, "score": 0.0, "reason": "no_premarket_bars", "premarket_volume": 0, "dollar_volume": 0.0, "gap_pct": 0.0, "momentum_pct": 0.0, "bar_count": 0}

    sdf = premarket_bars[premarket_bars["symbol"].eq(symbol)].sort_values("timestamp").copy()
    if sdf.empty:
        return {"symbol": symbol, "score": 0.0, "reason": "no_symbol_bars", "premarket_volume": 0, "dollar_volume": 0.0, "gap_pct": 0.0, "momentum_pct": 0.0, "bar_count": 0}

    volume = pd.to_numeric(sdf["volume"], errors="coerce").fillna(0.0)
    close = pd.to_numeric(sdf["close"], errors="coerce").fillna(0.0)
    dollar_volume = float((volume * close).sum())
    premarket_volume = int(volume.sum())
    first_open = safe_float(sdf.iloc[0].get("open"), 0.0)
    last_close = safe_float(sdf.iloc[-1].get("close"), 0.0)
    high = safe_float(sdf["high"].max(), 0.0)
    low = safe_float(sdf["low"].min(), 0.0)
    gap_pct = ((last_close - first_open) / first_open) * 100.0 if first_open > 0 else 0.0
    range_pct = ((high - low) / max(last_close, 1e-9)) * 100.0 if last_close > 0 else 0.0
    last_n = sdf.tail(min(10, len(sdf)))
    momentum_pct = ((safe_float(last_n.iloc[-1].get("close"), 0.0) - safe_float(last_n.iloc[0].get("open"), 0.0)) / max(safe_float(last_n.iloc[0].get("open"), 0.0), 1e-9)) * 100.0 if len(last_n) >= 2 else 0.0

    volume_score = min(5.0, premarket_volume / max(MIN_PREMARKET_VOLUME, 1))
    dollar_score = min(5.0, dollar_volume / max(MIN_INTRADAY_DOLLAR_VOLUME, 1.0))
    gap_score = max(0.0, min(5.0, gap_pct / max(MIN_INITIAL_MOVE_PCT, 0.1)))
    momentum_score = max(0.0, min(5.0, momentum_pct / 1.0))
    range_penalty = max(0.0, min(2.0, (range_pct - 8.0) / 4.0))
    score = round(0.30 * volume_score + 0.30 * dollar_score + 0.25 * gap_score + 0.15 * momentum_score - range_penalty, 4)

    return {
        "symbol": symbol,
        "score": score,
        "reason": "ok",
        "premarket_volume": premarket_volume,
        "dollar_volume": round(dollar_volume, 2),
        "gap_pct": round(gap_pct, 4),
        "momentum_pct": round(momentum_pct, 4),
        "range_pct": round(range_pct, 4),
        "bar_count": int(len(sdf)),
    }


def rank_and_cap_symbols(symbols: list[str], premarket_bars: pd.DataFrame, tv_snapshot: pd.DataFrame | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    tv_snapshot = tv_snapshot if tv_snapshot is not None else pd.DataFrame()
    for symbol in symbols:
        base = score_symbol_from_premarket(symbol, premarket_bars)
        tv = tradingview_score_for_symbol(symbol, tv_snapshot)
        base_score = safe_float(base.get("score"), 0.0)
        tv_score = safe_float(tv.get("tv_score"), 0.0)
        if not tv_snapshot.empty:
            base["score_yfinance"] = base_score
            base["score_tradingview"] = tv_score
            base["score"] = round((1.0 - TRADINGVIEW_RANK_WEIGHT) * base_score + TRADINGVIEW_RANK_WEIGHT * tv_score, 4)
            base.update(tv)
        scored.append(base)

    # Keep symbols with real premarket participation first; TradingView adds RVOL/MACD/multi-timeframe context.
    scored = sorted(
        scored,
        key=lambda x: (
            safe_float(x.get("score")),
            safe_float(x.get("score_tradingview")),
            safe_float(x.get("dollar_volume")),
            safe_float(x.get("premarket_volume")),
            x.get("symbol", ""),
        ),
        reverse=True,
    )
    capped = [str(x["symbol"]) for x in scored[:max(1, MAX_PREMARKET_PLANNER_SYMBOLS)]]
    print(f"[universe-rank] input={len(symbols)} capped={len(capped)} max={MAX_PREMARKET_PLANNER_SYMBOLS} tv_enabled={TRADINGVIEW_SCREENER_ENABLED}")
    print("[universe-rank] selected=" + ",".join(capped))
    return capped, scored


def build_symbol_outputs(
    symbol: str,
    premarket_bars: pd.DataFrame,
    intraday_bars: pd.DataFrame,
    mtf_bars: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = analyze_symbol(symbol, premarket_bars, mtf_bars)
    ibar = intraday_bars[intraday_bars.get("symbol", pd.Series(dtype=str)).eq(symbol)].copy() if not intraday_bars.empty else pd.DataFrame()
    if not ibar.empty and len(ibar) < MIN_STRUCTURE_ROWS_REQUIRED:
        structure = build_minimal_structure(symbol, plan, f"insufficient_structure_rows_{len(ibar)}_lt_{MIN_STRUCTURE_ROWS_REQUIRED}")
    else:
        structure = build_intraday_structure_map_for_symbol(symbol, premarket_bars, intraday_bars, plan)
    return asdict(plan), structure, build_daily_trade_plan_json(plan, structure)
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


def _build_data_source_tag(tv_snapshot: pd.DataFrame) -> str:
    """Compute a human-readable data_source tag for JSON output payloads."""
    parts: list[str] = []
    if _DATA_SOURCE_DIAGNOSTICS.get("alpaca_primary"):
        parts.append("alpaca")
    fallback = _DATA_SOURCE_DIAGNOSTICS.get("fallback_used")
    if fallback:
        parts.append(str(fallback))
    if tv_snapshot is not None and not tv_snapshot.empty:
        parts.append("tradingview_screener")
    if not parts:
        parts.append("yfinance")
    return "+".join(parts)


def build_premarket_trade_plan() -> dict[str, Any]:
    run_started_monotonic = time_module.monotonic()

    # Reset per-run data-source diagnostics
    _DATA_SOURCE_DIAGNOSTICS["alpaca_primary"] = False
    _DATA_SOURCE_DIAGNOSTICS["fallback_used"] = None
    _DATA_SOURCE_DIAGNOSTICS["symbols_from_alpaca"] = 0

    symbols = load_universe()
    if not symbols:
        payload = {
            "created_at_utc": now_utc(),
            "data_source": "none",
            "planner_status": "disabled",
            "execution_mode": "NO_TRADE",
            "valid_plan_count": 0,
            "plans": [],
            "error": "No symbols found in latest LLM universe or locked universe.",
        }
        write_json(TRADE_PLAN_PATH, payload)
        write_json(STRUCTURE_MAP_PATH, {"created_at_utc": now_utc(), "data_source": "none", "planner_status": "disabled", "structures": []})
        return payload

    # Fetch premarket first for every candidate, then enrich/rank with optional TradingView screener context
    # before expensive structure work. TradingView is non-blocking: failures fall back to yfinance-only.
    premarket_bars = fetch_premarket_bars(symbols)
    tv_snapshot, tv_diagnostics = fetch_tradingview_snapshot(symbols)
    selected_symbols, symbol_scores = rank_and_cap_symbols(symbols, premarket_bars, tv_snapshot)

    if deadline_exceeded(run_started_monotonic):
        ds_tag = _build_data_source_tag(tv_snapshot)
        payload = {
            "created_at_utc": now_utc(),
            "data_source": ds_tag,
            "planner_status": "degraded_timeout_after_premarket_fetch",
            "execution_mode": "SAFE" if SAFE_MODE_ON_PLANNER_DEGRADATION else "NO_TRADE",
            "symbol_count": len(symbols),
            "selected_symbol_count": len(selected_symbols),
            "valid_plan_count": 0,
            "symbol_scores": symbol_scores,
            "tradingview_diagnostics": tv_diagnostics if "tv_diagnostics" in locals() else {},
            "daily_trade_plans": [],
            "raw_premarket_pattern_plans": [],
            "stale_after_minutes": MAX_PLAN_AGE_MINUTES,
            "notes": ["Planner timed out before structure build. Allow exits only or safe-mode reduced risk."],
        }
        write_json(TRADE_PLAN_PATH, payload)
        write_json(STRUCTURE_MAP_PATH, {"created_at_utc": now_utc(), "data_source": ds_tag, "planner_status": payload["planner_status"], "structures": []})
        return payload

    intraday_bars = fetch_intraday_bars(selected_symbols)
    mtf_bars = fetch_multitimeframe_bars(selected_symbols) if PIVOT_PLANNER_ENABLED else {}

    plans: list[dict[str, Any]] = []
    daily_trade_plans: list[dict[str, Any]] = []
    structure_maps: list[dict[str, Any]] = []
    symbol_errors: list[dict[str, Any]] = []

    max_workers = max(1, min(MAX_STRUCTURE_WORKERS, len(selected_symbols)))
    print(f"[planner] processing_symbols={len(selected_symbols)} workers={max_workers} total_timeout={PLANNER_TOTAL_TIMEOUT_SECONDS}s")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(build_symbol_outputs, symbol, premarket_bars, intraday_bars, mtf_bars): symbol for symbol in selected_symbols}
        for future in as_completed(future_map, timeout=max(1.0, seconds_remaining(run_started_monotonic))):
            symbol = future_map[future]
            if deadline_exceeded(run_started_monotonic):
                symbol_errors.append({"symbol": symbol, "error": "planner_deadline_exceeded"})
                break
            try:
                plan_dict, structure, daily_plan = future.result(timeout=PER_SYMBOL_TIMEOUT_SECONDS)
                plans.append(plan_dict)
                structure_maps.append(structure)
                daily_trade_plans.append(daily_plan)
            except TimeoutError:
                fallback_plan = empty_plan(symbol, "Per-symbol planner timeout; no entry allowed.")
                plans.append(asdict(fallback_plan))
                structure_maps.append(build_minimal_structure(symbol, fallback_plan, "per_symbol_timeout"))
                daily_trade_plans.append(build_daily_trade_plan_json(fallback_plan, structure_maps[-1]))
                symbol_errors.append({"symbol": symbol, "error": "per_symbol_timeout"})
            except Exception as exc:
                fallback_plan = empty_plan(symbol, f"Planner error: {exc}")
                plans.append(asdict(fallback_plan))
                structure_maps.append(build_minimal_structure(symbol, fallback_plan, "symbol_exception"))
                daily_trade_plans.append(build_daily_trade_plan_json(fallback_plan, structure_maps[-1]))
                symbol_errors.append({"symbol": symbol, "error": str(exc)})

    completed_symbols = {p.get("symbol") for p in plans}
    for symbol in selected_symbols:
        if symbol not in completed_symbols:
            fallback_plan = empty_plan(symbol, "Planner skipped symbol due to global timeout; no entry allowed.")
            plans.append(asdict(fallback_plan))
            structure_maps.append(build_minimal_structure(symbol, fallback_plan, "global_timeout_skipped"))
            daily_trade_plans.append(build_daily_trade_plan_json(fallback_plan, structure_maps[-1]))
            symbol_errors.append({"symbol": symbol, "error": "global_timeout_skipped"})

    valid = [p for p in plans if p.get("valid")]
    liquid_structures = [s for s in structure_maps if s.get("liquidity", {}).get("liquid_enough")]
    long_allowed = [s["symbol"] for s in structure_maps if s.get("trade_structure", {}).get("long_allowed")]
    elapsed = round(time_module.monotonic() - run_started_monotonic, 2)

    planner_status = "ok"
    if symbol_errors:
        planner_status = "degraded_symbol_errors"
    if elapsed >= PLANNER_TOTAL_TIMEOUT_SECONDS:
        planner_status = "degraded_global_timeout"

    execution_mode = "NORMAL"
    if planner_status != "ok":
        execution_mode = "SAFE" if SAFE_MODE_ON_PLANNER_DEGRADATION else "NO_TRADE"

    ds_tag = _build_data_source_tag(tv_snapshot)

    # Emit final [data-source] diagnostics line
    print(
        f"[data-source] alpaca_primary={_DATA_SOURCE_DIAGNOSTICS['alpaca_primary']}"
        f" fallback_used={_DATA_SOURCE_DIAGNOSTICS['fallback_used']}"
        f" symbols_from_alpaca={_DATA_SOURCE_DIAGNOSTICS['symbols_from_alpaca']}"
        f" data_source={ds_tag}"
    )
    merge_diagnostics("data_source_summary", {
        "alpaca_primary": _DATA_SOURCE_DIAGNOSTICS["alpaca_primary"],
        "fallback_used": _DATA_SOURCE_DIAGNOSTICS["fallback_used"],
        "symbols_from_alpaca": _DATA_SOURCE_DIAGNOSTICS["symbols_from_alpaca"],
        "data_source_tag": ds_tag,
    })

    payload = {
        "created_at_utc": now_utc(),
        "data_source": ds_tag,
        "planner_status": planner_status,
        "execution_mode": execution_mode,
        "elapsed_seconds": elapsed,
        "stale_after_minutes": MAX_PLAN_AGE_MINUTES,
        "premarket_window_et": {"start": PREMARKET_START_ET, "end": PREMARKET_END_ET},
        "regular_session_window_et": {"start": REGULAR_SESSION_START_ET, "end": REGULAR_SESSION_END_ET},
        "symbol_count": len(symbols),
        "selected_symbol_count": len(selected_symbols),
        "selected_symbols": selected_symbols,
        "valid_plan_count": len(valid),
        "valid_symbols": [p["symbol"] for p in valid],
        "liquid_structure_count": len(liquid_structures),
        "long_allowed_symbols": long_allowed,
        "symbol_scores": symbol_scores,
        "symbol_errors": symbol_errors,
        "risk_controls": {
            "max_symbols": MAX_PREMARKET_PLANNER_SYMBOLS,
            "total_timeout_seconds": PLANNER_TOTAL_TIMEOUT_SECONDS,
            "per_symbol_timeout_seconds": PER_SYMBOL_TIMEOUT_SECONDS,
            "min_structure_rows_required": MIN_STRUCTURE_ROWS_REQUIRED,
            "min_training_rows_required": MIN_TRAINING_ROWS_REQUIRED,
            "safe_mode_on_degradation": SAFE_MODE_ON_PLANNER_DEGRADATION,
            "always_allow_exits": True,
            "block_new_entries_when_stale": True,
        },
        "daily_trade_plans": daily_trade_plans,
        "raw_premarket_pattern_plans": plans,
    }

    structure_payload = {
        "created_at_utc": now_utc(),
        "data_source": ds_tag,
        "planner_status": planner_status,
        "execution_mode": execution_mode,
        "elapsed_seconds": elapsed,
        "stale_after_minutes": MAX_PLAN_AGE_MINUTES,
        "symbol_count": len(symbols),
        "selected_symbol_count": len(selected_symbols),
        "selected_symbols": selected_symbols,
        "long_allowed_symbols": long_allowed,
        "symbol_errors": symbol_errors,
        "structures": structure_maps,
    }

    write_json(TRADE_PLAN_PATH, payload)
    write_json(STRUCTURE_MAP_PATH, structure_payload)

    print(f"[premarket-plan] status={planner_status} mode={execution_mode} elapsed={elapsed}s symbols={len(symbols)} selected={len(selected_symbols)} valid={len(valid)} liquid={len(liquid_structures)} long_allowed={len(long_allowed)}")
    print("[premarket-plan] valid_symbols=" + ",".join(payload["valid_symbols"]))
    print("[structure-map] long_allowed_symbols=" + ",".join(long_allowed))

    return payload

if __name__ == "__main__":
    wait_until_market_open()
    build_premarket_trade_plan()
