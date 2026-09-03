#!/usr/bin/env python3
"""
options_booster.py

Self-contained unusual-options-flow scanner + EV-based long-call booster.

Default usage:
    python options_booster.py

What it does:
    1. Loads Alpaca credentials from .env.
    2. Builds an underlying universe automatically from:
        - system-data/regime_position_plan.csv
        - system-data/regime_universe_30.csv
        - system-data/latest_scanner_scores.csv
        - locked_trading_universe.txt
        - DEFAULT_UNIVERSE fallback
    3. Fetches option chains.
    4. Finds call flow using:
        - volume / open-interest ratio
        - option volume
        - bid/ask spread
        - delta fit
        - DTE fit
        - gamma efficiency
        - estimated premium flow
        - expected-value score
    5. Requires underlying stock position to already be open/profitable by default.
    6. Writes:
        system-data/latest_option_flow_candidates.csv
        system-data/latest_option_booster_ranked_contracts.csv
        system-data/latest_option_booster_candidates.csv
        system-data/daily_walkforward_emulated_rebalance_orders.csv
        system-data/daily_walkforward_emulated_exit_orders.csv
        system-data/latest_option_booster_summary.json

Safety:
    - Long calls only.
    - Does not place trades directly.
    - Does not short options.
    - Sizes by premium-at-risk.
    - Blocks near-expiration entries.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from config.env_loader import load_instance_env

from market_context_builder import TODAY_TAG

try:
    from alpaca.trading.client import TradingClient
except Exception as exc:
    raise RuntimeError("alpaca-py trading client is required. Install with: pip install alpaca-py") from exc

try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest
except Exception as exc:
    raise RuntimeError("alpaca-py option data client is required. Install with: pip install alpaca-py") from exc

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockLatestBarRequest,
        StockLatestQuoteRequest,
        StockLatestTradeRequest,
    )
    _STOCK_DATA_CLIENT_AVAILABLE = True
except Exception:
    StockHistoricalDataClient = None  # type: ignore[assignment,misc]
    StockLatestTradeRequest = None  # type: ignore[assignment]
    StockLatestQuoteRequest = None  # type: ignore[assignment]
    StockLatestBarRequest = None  # type: ignore[assignment]
    _STOCK_DATA_CLIENT_AVAILABLE = False


# =============================================================================
# Paths
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_instance_env()

OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"
LOG_DIR = PROJECT_ROOT / "logs" / "execution" / f"{TODAY_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

REGIME_POSITION_PLAN_PATH = OUT_DIR / f"{TODAY_TAG}_regime_position_plan.csv"
REGIME_UNIVERSE_PATH = OUT_DIR / f"{TODAY_TAG}_regime_universe_30.csv"
SCANNER_SCORES_PATH = OUT_DIR / f"{TODAY_TAG}_latest_scanner_scores.csv"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{TODAY_TAG}_locked_trading_universe.txt"
INTRADAY_STRUCTURE_MAP_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_structure_map.json"

REBALANCE_ORDERS_PATH = OUT_DIR / f"{TODAY_TAG}_daily_walkforward_emulated_rebalance_orders.csv"
EXIT_ORDERS_PATH = OUT_DIR / f"{TODAY_TAG}_daily_walkforward_emulated_exit_orders.csv"

OPTION_FLOW_CANDIDATES_PATH = OUT_DIR / f"{TODAY_TAG}_latest_option_flow_candidates.csv"
OPTION_BOOSTER_CANDIDATES_PATH = OUT_DIR / f"{TODAY_TAG}_latest_option_booster_candidates.csv"
RANKED_CONTRACTS_PATH = OUT_DIR / f"{TODAY_TAG}_latest_option_booster_ranked_contracts.csv"
SUMMARY_PATH = OUT_DIR / f"{TODAY_TAG}_latest_option_booster_summary.json"

STATE_PATH = LOG_DIR / f"{TODAY_TAG}_option_booster_state.json"
EVENTS_PATH = LOG_DIR / f"{TODAY_TAG}_option_booster_events.jsonl"


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", "SOXX",
    "NVDA", "AMD", "AVGO", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
    "PLTR", "COIN", "MARA", "RIOT", "RGTI", "QBTS", "QUBT", "IONQ", "SOUN",
    "OKLO", "APLD", "ALAB", "ARM",
]

OCC_OPTION_SYMBOL_RE = re.compile(r"^([A-Z][A-Z0-9-]*?)(\d{6})([CP])(\d{8})$")


# =============================================================================
# Config
# =============================================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return default


@dataclass
class OptionsBoosterConfig:
    enabled: bool = env_bool("OPTIONS_BOOSTER_ENABLED", True)
    paper: bool = env_bool("APCA_PAPER", True)

    # Universe / scan behavior.
    require_structure_long_allowed: bool = env_bool("OPTIONS_REQUIRE_STRUCTURE_LONG_ALLOWED", True)
    max_underlyings: int = env_int("OPTIONS_FLOW_MAX_UNDERLYINGS", 30)
    max_contracts_per_underlying_scan: int = env_int("OPTIONS_FLOW_MAX_CONTRACTS_PER_UNDERLYING", 500)
    max_output_candidates: int = env_int("OPTIONS_FLOW_MAX_OUTPUT_CANDIDATES", 50)
    top_orders: int = env_int("OPTIONS_BOOSTER_TOP_ORDERS", 3)

    # Underlying position gate.
    # For safest production behavior this defaults to true: options booster adds calls
    # only when the stock position is already open and profitable.
    require_open_underlying_position: bool = env_bool("OPTIONS_REQUIRE_OPEN_UNDERLYING_POSITION", True)
    min_underlying_unrealized_plpc: float = env_float("OPTIONS_MIN_UNDERLYING_PLPC", 0.0056)

    # Flow scan filters.
    min_dte: int = env_int("OPTIONS_MIN_DTE", 21)
    max_dte: int = env_int("OPTIONS_MAX_DTE", 45)
    target_dte: int = env_int("OPTIONS_TARGET_DTE", 30)

    target_delta_min: float = env_float("OPTIONS_MIN_DELTA", 0.35)
    target_delta_max: float = env_float("OPTIONS_MAX_DELTA", 0.70)
    ideal_delta: float = env_float("OPTIONS_IDEAL_DELTA", 0.52)

    max_spread_pct: float = env_float("OPTIONS_MAX_SPREAD_PCT", 0.18)
    min_open_interest: int = env_int("OPTIONS_MIN_OPEN_INTEREST", 25)
    min_volume: int = env_int("OPTIONS_MIN_VOLUME", 10)
    max_contract_mid_price: float = env_float("OPTIONS_MAX_CONTRACT_MID_PRICE", 7.50)

    # Unusual flow thresholds.
    min_rvolchg_pct: float = env_float("OPTIONS_MIN_RVOLCHG_PCT", 25.0)
    min_volume_oi_ratio: float = env_float("OPTIONS_MIN_VOLUME_OI_RATIO", 0.25)
    min_premium_flow_dollars: float = env_float("OPTIONS_MIN_PREMIUM_FLOW_DOLLARS", 25000.0)
    min_flow_score: float = env_float("OPTIONS_MIN_FLOW_SCORE", 0.45)

    # EV assumptions.
    base_expected_move_pct: float = env_float("OPTIONS_BASE_EXPECTED_MOVE_PCT", 0.012)
    max_expected_move_pct: float = env_float("OPTIONS_MAX_EXPECTED_MOVE_PCT", 0.080)
    # Scaled EV move params — override base/max when vol/signal data is available.
    expected_move_min_pct: float = env_float("OPTIONS_EXPECTED_MOVE_MIN_PCT", 0.012)
    expected_move_max_pct: float = env_float("OPTIONS_EXPECTED_MOVE_MAX_PCT", 0.060)
    expected_move_vol_mult: float = env_float("OPTIONS_EXPECTED_MOVE_VOL_MULT", 0.50)
    expected_move_signal_weight: float = env_float("OPTIONS_EXPECTED_MOVE_SIGNAL_WEIGHT", 0.65)
    loss_given_wrong_pct: float = env_float("OPTIONS_LOSS_GIVEN_WRONG_PCT", 0.55)
    min_probability_of_profit: float = env_float("OPTIONS_MIN_POP", 0.50)
    min_expected_value: float = env_float("OPTIONS_MIN_EXPECTED_VALUE", -5.0)

    # Greeks.
    min_gamma: float = env_float("OPTIONS_MIN_GAMMA", 0.0)
    max_theta_abs: float = env_float("OPTIONS_MAX_THETA_ABS", 999.0)
    max_iv: float = env_float("OPTIONS_MAX_IV", 999.0)

    # Final score weights.
    flow_score_weight: float = env_float("OPTIONS_FLOW_SCORE_WEIGHT", 0.40)
    ev_score_weight: float = env_float("OPTIONS_EV_SCORE_WEIGHT", 0.25)
    liquidity_score_weight: float = env_float("OPTIONS_LIQUIDITY_SCORE_WEIGHT", 0.12)
    spread_score_weight: float = env_float("OPTIONS_SPREAD_SCORE_WEIGHT", 0.10)
    delta_score_weight: float = env_float("OPTIONS_DELTA_SCORE_WEIGHT", 0.07)
    gamma_score_weight: float = env_float("OPTIONS_GAMMA_SCORE_WEIGHT", 0.04)
    dte_score_weight: float = env_float("OPTIONS_DTE_SCORE_WEIGHT", 0.02)

    # Risk sizing.
    max_risk_per_trade_pct: float = env_float("OPTIONS_MAX_RISK_PCT", 0.010)
    max_risk_per_trade_dollars: float = env_float("OPTIONS_MAX_RISK_DOLLARS", 250.0)
    max_contracts_per_underlying: int = env_int("OPTIONS_MAX_CONTRACTS_PER_UNDERLYING", 2)
    max_open_option_positions: int = env_int("OPTIONS_MAX_OPEN_POSITIONS", 4)

    # Expiration and exits.
    close_before_expiration_days: int = env_int("OPTIONS_CLOSE_BEFORE_EXPIRATION_DAYS", 3)
    block_new_entries_within_expiration_days: int = env_int("OPTIONS_BLOCK_NEW_ENTRIES_WITHIN_EXPIRATION_DAYS", 5)
    take_profit_pct: float = env_float("OPTIONS_TAKE_PROFIT_PCT", 0.60)
    hard_stop_pct: float = env_float("OPTIONS_HARD_STOP_PCT", -0.35)
    profit_lock_trigger_pct: float = env_float("OPTIONS_PROFIT_LOCK_TRIGGER_PCT", 0.40)
    profit_giveback_pct: float = env_float("OPTIONS_PROFIT_GIVEBACK_PCT", 0.25)

    # Order bridge.
    limit_price_source: str = os.getenv("OPTIONS_LIMIT_PRICE_SOURCE", "bid").lower().strip()
    append_to_existing_rebalance: bool = env_bool("OPTIONS_APPEND_REBALANCE", True)
    append_to_existing_exits: bool = env_bool("OPTIONS_APPEND_EXITS", True)


CONFIG = OptionsBoosterConfig()


# =============================================================================
# Advisory options sentiment (Alpha Vantage) — never execution-authoritative.
# Degrades to neutral (no behavior change) if the module/key/data is missing.
# =============================================================================

try:
    from options_sentiment_av import (
        OptionsSentimentClient,
        evaluate_setup_sentiment,
        neutral_sentiment,
    )
    _SENTIMENT_AVAILABLE = True
except Exception as _sentiment_import_err:  # pragma: no cover - optional dep
    OptionsSentimentClient = None  # type: ignore[assignment,misc]
    evaluate_setup_sentiment = None  # type: ignore[assignment]
    neutral_sentiment = None  # type: ignore[assignment]
    _SENTIMENT_AVAILABLE = False
    print(f"[sentiment] options_sentiment_av unavailable: {_sentiment_import_err}")

_SENTIMENT_CLIENT: Any = None


def get_sentiment_client() -> Any:
    """Lazily build the advisory sentiment client. Returns None if disabled/unavailable."""
    global _SENTIMENT_CLIENT
    if not _SENTIMENT_AVAILABLE or not env_bool("OPTIONS_SENTIMENT_AV_ENABLED", True):
        return None
    if _SENTIMENT_CLIENT is None:
        try:
            _SENTIMENT_CLIENT = OptionsSentimentClient()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[sentiment] client init failed: {exc}")
            _SENTIMENT_CLIENT = None
    return _SENTIMENT_CLIENT


# =============================================================================
# Helpers
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "timestamp_utc": now_utc().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=json_default) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def safe_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def clip01(value: Any) -> float:
    return float(np.clip(to_float(value, 0.0), 0.0, 1.0))


def obj_get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
    except Exception:
        pass
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_or_write_csv(path: Path, new_rows: pd.DataFrame, append: bool) -> None:
    if new_rows.empty:
        if not path.exists():
            new_rows.to_csv(path, index=False)
        return

    if append and path.exists() and path.stat().st_size > 0:
        existing = read_csv_safe(path)
        if not existing.empty:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(path, index=False)
            return

    new_rows.to_csv(path, index=False)


def parse_occ_option_symbol(symbol: Any) -> dict[str, Any] | None:
    raw = safe_symbol(symbol)
    match = OCC_OPTION_SYMBOL_RE.match(raw)
    if not match:
        return None

    underlying, yymmdd, cp, strike_raw = match.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
        strike = int(strike_raw) / 1000.0
    except Exception:
        return None

    return {
        "symbol": raw,
        "underlying": underlying,
        "expiration": expiration,
        "type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def option_quote_metrics(snapshot: Any) -> dict[str, float]:
    quote = obj_get(snapshot, "latest_quote", "latestQuote", default=None)
    greeks = obj_get(snapshot, "greeks", default=None)

    bid = to_float(obj_get(quote, "bid_price", "bp", "bid", default=0.0))
    ask = to_float(obj_get(quote, "ask_price", "ap", "ask", default=0.0))
    bid_size = to_float(obj_get(quote, "bid_size", "bs", default=0.0))
    ask_size = to_float(obj_get(quote, "ask_size", "as", default=0.0))

    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    spread_pct = (ask - bid) / mid if mid > 0 else 999.0

    delta = to_float(obj_get(greeks, "delta", default=np.nan), np.nan)
    gamma = to_float(obj_get(greeks, "gamma", default=np.nan), np.nan)
    theta = to_float(obj_get(greeks, "theta", default=np.nan), np.nan)
    vega = to_float(obj_get(greeks, "vega", default=np.nan), np.nan)

    iv = to_float(obj_get(snapshot, "implied_volatility", "iv", default=np.nan), np.nan)
    volume = to_float(obj_get(snapshot, "volume", default=0.0))
    open_interest = to_float(obj_get(snapshot, "open_interest", "openInterest", default=0.0))

    return {
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid": mid,
        "spread_pct": spread_pct,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "iv": iv,
        "volume": volume,
        "open_interest": open_interest,
    }


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")


# =============================================================================
# Alpaca
# =============================================================================

def get_api_keys() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY in .env")
    return key, secret


def build_trading_client() -> TradingClient:
    key, secret = get_api_keys()
    return TradingClient(api_key=key, secret_key=secret, paper=CONFIG.paper)


def build_option_data_client() -> OptionHistoricalDataClient:
    key, secret = get_api_keys()
    return OptionHistoricalDataClient(key, secret)


def build_stock_data_client() -> Any:
    if not _STOCK_DATA_CLIENT_AVAILABLE or StockHistoricalDataClient is None:
        return None
    key, secret = get_api_keys()
    return StockHistoricalDataClient(key, secret)


def resolve_underlying_price_alpaca(symbol: str, stock_client: Any) -> float:
    """Try latest trade → quote midpoint → bar close via Alpaca stock data API."""
    if stock_client is None:
        return 0.0

    try:
        resp = stock_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        trade = (resp or {}).get(symbol)
        if trade:
            price = to_float(obj_get(trade, "price", "p", default=0.0))
            if price > 0:
                return price
    except Exception:
        pass

    try:
        resp = stock_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        quote = (resp or {}).get(symbol)
        if quote:
            bid = to_float(obj_get(quote, "bid_price", "bp", default=0.0))
            ask = to_float(obj_get(quote, "ask_price", "ap", default=0.0))
            mid = (bid + ask) / 2.0
            if mid > 0:
                return mid
    except Exception:
        pass

    try:
        resp = stock_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=symbol))
        bar = (resp or {}).get(symbol)
        if bar:
            price = to_float(obj_get(bar, "close", "c", default=0.0))
            if price > 0:
                return price
    except Exception:
        pass

    return 0.0


_INTRADAY_PRICE_CACHE: dict[str, float] = {}
_INTRADAY_PRICE_CACHE_LOADED = False


def _load_intraday_price_cache() -> None:
    """Pre-load symbol→last-close from local intraday CSVs into a single dict."""
    global _INTRADAY_PRICE_CACHE, _INTRADAY_PRICE_CACHE_LOADED
    if _INTRADAY_PRICE_CACHE_LOADED:
        return
    _INTRADAY_PRICE_CACHE_LOADED = True
    for path in [
        OUT_DIR / f"{TODAY_TAG}_intraday_bars_1m.csv",
        OUT_DIR / f"{TODAY_TAG}_intraday_training_dataset.csv",
    ]:
        df = read_csv_safe(path)
        if df.empty:
            continue
        col_lower = {c.lower(): c for c in df.columns}
        sym_col = col_lower.get("symbol")
        if sym_col is None:
            continue
        for close_key in ("close", "c", "price"):
            actual = col_lower.get(close_key)
            if actual and actual in df.columns:
                for sym, grp in df.groupby(sym_col):
                    if str(sym).upper() in _INTRADAY_PRICE_CACHE:
                        continue
                    series = pd.to_numeric(grp[actual], errors="coerce").dropna()
                    if not series.empty:
                        val = to_float(series.iloc[-1])
                        if val > 0:
                            _INTRADAY_PRICE_CACHE[str(sym).upper()] = val
                break


def resolve_underlying_price_intraday(symbol: str) -> float:
    _load_intraday_price_cache()
    return _INTRADAY_PRICE_CACHE.get(symbol.upper(), 0.0)


_STRUCTURE_MAP_PAYLOAD: dict[str, Any] | None = None
_STRUCTURE_MAP_LOADED = False


def _load_structure_map_payload() -> dict[str, Any]:
    """Parse INTRADAY_STRUCTURE_MAP_PATH once and cache the result."""
    global _STRUCTURE_MAP_PAYLOAD, _STRUCTURE_MAP_LOADED
    if _STRUCTURE_MAP_LOADED:
        return _STRUCTURE_MAP_PAYLOAD or {}
    _STRUCTURE_MAP_LOADED = True
    if not INTRADAY_STRUCTURE_MAP_PATH.exists():
        return {}
    try:
        _STRUCTURE_MAP_PAYLOAD = json.loads(
            INTRADAY_STRUCTURE_MAP_PATH.read_text(encoding="utf-8", errors="ignore")
        )
    except Exception:
        _STRUCTURE_MAP_PAYLOAD = {}
    return _STRUCTURE_MAP_PAYLOAD or {}


def resolve_underlying_price_structure_map(symbol: str) -> float:
    payload = _load_structure_map_payload()
    for entry in payload.get("structures") or []:
        if not isinstance(entry, dict):
            continue
        if safe_symbol(entry.get("symbol")) == symbol:
            price = to_float(entry.get("current_price"), 0.0)
            if price > 0:
                return price
    return 0.0


def positions_by_symbol(client: TradingClient) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for pos in client.get_all_positions():
            symbol = safe_symbol(obj_get(pos, "symbol", default=""))
            qty = to_float(obj_get(pos, "qty", default=0.0))
            if symbol and qty > 0:
                out[symbol] = pos
    except Exception as exc:
        print(f"[options] failed to fetch positions: {exc}")
    return out


def account_equity(client: TradingClient) -> float:
    try:
        acct = client.get_account()
    except Exception as exc:
        # Network/auth failures should degrade the booster to safe no-op sizing,
        # not crash the collector subprocess and mask the broader runtime issue.
        print(f"[options] failed to fetch account equity: {exc}")
        return 1.0
    return max(to_float(obj_get(acct, "equity", default=0.0)), 1.0)


def open_option_underlyings(positions: dict[str, Any]) -> set[str]:
    out = set()
    for symbol, pos in positions.items():
        parsed = parse_occ_option_symbol(symbol)
        if parsed and parsed["type"] == "call" and to_float(obj_get(pos, "qty", default=0.0)) > 0:
            out.add(str(parsed["underlying"]))
    return out


def open_option_position_count(positions: dict[str, Any]) -> int:
    return sum(1 for symbol, pos in positions.items() if parse_occ_option_symbol(symbol))


# =============================================================================
# Universe/candidates
# =============================================================================

def load_locked_universe() -> list[str]:
    if not LOCKED_UNIVERSE_PATH.exists():
        return []
    text = LOCKED_UNIVERSE_PATH.read_text(encoding="utf-8", errors="ignore")
    symbols = []
    for token in re.split(r"[\s,]+", text):
        sym = safe_symbol(token)
        if sym and sym not in symbols:
            symbols.append(sym)
    return symbols


def load_structure_long_allowed() -> set[str]:
    """Return symbols where long_allowed=True from the intraday structure map."""
    payload = _load_structure_map_payload()
    if not payload:
        return set()
    raw = payload.get("long_allowed_symbols") or []
    if isinstance(raw, list) and raw:
        return {safe_symbol(s) for s in raw if safe_symbol(s)}
    out: set[str] = set()
    for entry in payload.get("structures") or []:
        if not isinstance(entry, dict):
            continue
        ts = entry.get("trade_structure") or {}
        if bool(ts.get("long_allowed")):
            sym = safe_symbol(entry.get("symbol"))
            if sym:
                out.add(sym)
    return out


def load_universe() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path, source in [
        (REGIME_POSITION_PLAN_PATH, "regime_position_plan"),
        (REGIME_UNIVERSE_PATH, "regime_universe"),
        (SCANNER_SCORES_PATH, "scanner_scores"),
    ]:
        df = read_csv_safe(path)
        if not df.empty and "symbol" in df.columns:
            tmp = df.copy()
            tmp["source"] = source
            frames.append(tmp)

    locked = load_locked_universe()
    if locked:
        frames.append(pd.DataFrame({"symbol": locked, "source": "locked_universe"}))

    if not frames:
        frames.append(pd.DataFrame({"symbol": DEFAULT_UNIVERSE, "source": "default_universe"}))

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["symbol"] = out["symbol"].map(safe_symbol)
    out = out[out["symbol"] != ""].copy()

    # Prefer rows with higher confidence/score if available.
    if "continuation_confidence" not in out.columns:
        possible = [c for c in ["trade_confidence", "confidence", "confidence_score", "setup_score", "momentum_score"] if c in out.columns]
        if possible:
            out["continuation_confidence"] = out[possible].apply(pd.to_numeric, errors="coerce").max(axis=1)
        else:
            out["continuation_confidence"] = 0.70

    out["continuation_confidence"] = pd.to_numeric(out["continuation_confidence"], errors="coerce").fillna(0.70)

    if "price" not in out.columns:
        for c in ["close", "current_price", "underlying_price"]:
            if c in out.columns:
                out["price"] = out[c]
                break
    if "price" not in out.columns:
        out["price"] = np.nan

    priority = {
        "regime_position_plan": 0,
        "regime_universe": 1,
        "scanner_scores": 2,
        "locked_universe": 3,
        "default_universe": 4,
    }
    out["source_priority"] = out["source"].map(priority).fillna(9)

    out = (
        out.sort_values(["source_priority", "continuation_confidence"], ascending=[True, False])
        .drop_duplicates("symbol", keep="first")
        .head(CONFIG.max_underlyings)
        .reset_index(drop=True)
    )

    return out


# =============================================================================
# Flow / EV scoring
# =============================================================================

def estimate_probability_of_profit(row: pd.Series, metrics: dict[str, float]) -> float:
    confidence = clip01(row.get("continuation_confidence", 0.70))

    delta = metrics.get("delta", np.nan)
    delta_component = 0.52 if math.isnan(delta) else float(np.clip(delta, 0.0, 1.0))

    pop = 0.70 * confidence + 0.30 * delta_component
    return float(np.clip(pop, 0.01, 0.99))


def estimate_expected_underlying_move_pct(row: pd.Series, flow_score: float, dte: int) -> float:
    """Estimate expected move scaled by flow, signal, volatility, and structure.

    Replaces the near-constant 1.2% prior with a range-aware calculation:
      - signal component: flow_score + continuation confidence + structure confidence
      - volatility component: realized vol proxy from ATR/rolling_std if available
      - blended with signal_weight / (1-signal_weight) ratio
      - DTE-scaled and clamped to [expected_move_min_pct, expected_move_max_pct]
    """
    # Signal component: combine flow score, continuation confidence, structure confidence.
    continuation_conf = clip01(to_float(row.get("continuation_confidence"), 0.0))
    structure_conf = clip01(to_float(row.get("structure_confidence", row.get("bullish_structure_score", 0.0)), 0.0))
    momentum_score = clip01(to_float(row.get("momentum_score", row.get("positive_momentum_score", 0.0)), 0.0))

    # Weighted signal: flow is most objective, rest are softer priors.
    raw_signal = (
        0.45 * flow_score
        + 0.25 * continuation_conf
        + 0.15 * structure_conf
        + 0.15 * momentum_score
    )
    signal_weight = CONFIG.expected_move_signal_weight
    signal_component = np.clip(raw_signal, 0.0, 1.0)

    # Volatility component: ATR pct or rolling_std acts as a realized-vol proxy.
    vol_component = 0.0
    for vol_col in ("atr_pct", "atr_ratio", "rolling_std10", "volatility_10", "rolling_std20"):
        raw_vol = to_float(row.get(vol_col), 0.0)
        if raw_vol > 0:
            vol_component = float(np.clip(raw_vol * CONFIG.expected_move_vol_mult, 0.0, CONFIG.expected_move_max_pct))
            break

    # Blend signal and vol components.
    if vol_component > 0:
        blended = signal_weight * signal_component * CONFIG.expected_move_max_pct + (1.0 - signal_weight) * vol_component
    else:
        # Fall back to pure signal scaling when no vol data.
        edge = max(signal_component - 0.50, 0.0) / 0.50
        blended = CONFIG.expected_move_min_pct + edge * (CONFIG.expected_move_max_pct - CONFIG.expected_move_min_pct)

    dte_scale = float(np.clip(math.sqrt(max(dte, 1) / max(CONFIG.target_dte, 1)), 0.75, 1.25))
    result = float(np.clip(blended * dte_scale, CONFIG.expected_move_min_pct, CONFIG.expected_move_max_pct))
    return result


def compute_flow_score(metrics: dict[str, float]) -> dict[str, float]:
    volume = max(metrics["volume"], 0.0)
    oi = max(metrics["open_interest"], 0.0)
    mid = metrics["mid"]
    spread_pct = metrics["spread_pct"]

    volume_oi_ratio = volume / max(oi, 1.0)
    rvolchg_pct = volume_oi_ratio * 100.0
    premium_flow_dollars = volume * mid * 100.0

    volume_score = min(volume / max(CONFIG.min_volume * 10.0, 1.0), 1.0)
    voi_score = min(volume_oi_ratio / max(CONFIG.min_volume_oi_ratio * 3.0, 1e-9), 1.0)
    premium_score = min(premium_flow_dollars / max(CONFIG.min_premium_flow_dollars * 4.0, 1.0), 1.0)
    spread_score = 1.0 - min(spread_pct / max(CONFIG.max_spread_pct, 1e-9), 1.0)

    bid_ask_size_bias = 0.5
    bid_size = metrics.get("bid_size", 0.0)
    ask_size = metrics.get("ask_size", 0.0)
    if bid_size > 0 or ask_size > 0:
        # Higher ask-side size is not a true buy/sell classification, but helps
        # avoid thin, one-sided quotes. Keep it low weight.
        bid_ask_size_bias = min(max(ask_size / max(bid_size + ask_size, 1e-9), 0.0), 1.0)

    flow_score = (
        0.30 * volume_score
        + 0.30 * voi_score
        + 0.25 * premium_score
        + 0.10 * spread_score
        + 0.05 * bid_ask_size_bias
    )

    return {
        "volume_oi_ratio": float(volume_oi_ratio),
        "rvolchg_pct": float(rvolchg_pct),
        "premium_flow_dollars": float(premium_flow_dollars),
        "volume_score": float(volume_score),
        "volume_oi_score": float(voi_score),
        "premium_flow_score": float(premium_score),
        "flow_spread_score": float(spread_score),
        "flow_score": float(np.clip(flow_score, 0.0, 1.0)),
    }


def contract_expected_value(
    row: pd.Series,
    underlying_price: float,
    strike: float,
    dte: int,
    metrics: dict[str, float],
    flow_score: float,
) -> dict[str, float]:
    mid = metrics["mid"]
    if mid <= 0:
        return {
            "expected_value_dollars": -999.0,
            "expected_value_pct": -999.0,
            "probability_of_profit": 0.0,
            "expected_underlying_move_pct": 0.0,
            "target_underlying_price": underlying_price,
            "expected_option_price": mid,
            "expected_profit_per_contract": -999.0,
            "premium_at_risk_per_contract": mid * 100.0,
        }

    delta = metrics.get("delta", np.nan)
    gamma = metrics.get("gamma", np.nan)
    theta = metrics.get("theta", np.nan)

    if math.isnan(delta):
        moneyness = strike / max(underlying_price, 1e-9)
        delta = float(np.clip(0.62 - abs(moneyness - 1.0) * 1.25, 0.25, 0.70))
    if math.isnan(gamma):
        gamma = 0.0
    if math.isnan(theta):
        theta = 0.0

    expected_move_pct = estimate_expected_underlying_move_pct(row, flow_score, dte)
    target_price = underlying_price * (1.0 + expected_move_pct)
    underlying_move = max(target_price - underlying_price, 0.0)

    delta_gain = delta * underlying_move
    gamma_gain = 0.5 * gamma * (underlying_move ** 2)
    theta_decay = abs(theta) * max(min(dte, CONFIG.target_dte), 1)

    expected_option_price = max(mid + delta_gain + gamma_gain - theta_decay, 0.01)
    expected_profit_per_contract = max(expected_option_price - mid, -mid) * 100.0
    premium_at_risk_per_contract = mid * 100.0 * CONFIG.loss_given_wrong_pct

    pop = estimate_probability_of_profit(row, metrics)
    ev_dollars = pop * expected_profit_per_contract - (1.0 - pop) * premium_at_risk_per_contract
    ev_pct = ev_dollars / max(mid * 100.0, 1e-9)

    return {
        "expected_value_dollars": float(ev_dollars),
        "expected_value_pct": float(ev_pct),
        "probability_of_profit": float(pop),
        "expected_underlying_move_pct": float(expected_move_pct),
        "target_underlying_price": float(target_price),
        "expected_option_price": float(expected_option_price),
        "expected_profit_per_contract": float(expected_profit_per_contract),
        "premium_at_risk_per_contract": float(premium_at_risk_per_contract),
    }


def _empty_sentiment_fields() -> dict[str, Any]:
    """Advisory sentiment columns with neutral defaults (no decision impact)."""
    return {
        "put_call_ratio_chain": None,
        "put_call_ratio_expiration": None,
        "volume_oi_ratio_chain": None,
        "volume_oi_ratio_expiration": None,
        "sentiment_directional_score": 0.0,
        "participation_score": 0.5,
        "contrarian_warning": False,
        "options_sentiment_confidence": 0.5,
        "options_sentiment_confidence_multiplier": 1.0,
        "options_sentiment_skip_low_confirmation": False,
        "options_sentiment_safety_warning": "",
        "options_sentiment_decision_impact": "neutral: sentiment unavailable",
    }


def apply_sentiment_to_contract(
    score: dict[str, Any],
    sentiment: dict[str, Any] | None,
    expiration_iso: str,
    technical_bullish: bool,
) -> dict[str, Any]:
    """Fold advisory Alpha Vantage sentiment into a contract score (long-call side).

    Advisory only: adjusts the composite option_score and attaches explanatory
    fields. It never touches sizing, exposure caps, or hard risk gates.
    """
    fields = _empty_sentiment_fields()
    if not sentiment or not sentiment.get("available") or evaluate_setup_sentiment is None:
        score.update(fields)
        return score

    ev = evaluate_setup_sentiment(
        sentiment,
        side="call",
        technical_bullish=technical_bullish,
        expiration=expiration_iso,
    )

    pcr_exp_map = sentiment.get("put_call_ratio_expiration") or {}
    voi_exp_map = sentiment.get("volume_oi_ratio_expiration") or {}

    fields.update(
        {
            "put_call_ratio_chain": sentiment.get("put_call_ratio_chain"),
            "put_call_ratio_expiration": pcr_exp_map.get(expiration_iso, sentiment.get("put_call_ratio_chain")),
            "volume_oi_ratio_chain": sentiment.get("volume_oi_ratio_chain"),
            "volume_oi_ratio_expiration": voi_exp_map.get(expiration_iso, sentiment.get("volume_oi_ratio_chain")),
            "sentiment_directional_score": float(sentiment.get("sentiment_directional_score", 0.0)),
            "participation_score": float(sentiment.get("participation_score", 0.5)),
            "contrarian_warning": bool(sentiment.get("contrarian_warning", False)),
            "options_sentiment_confidence": float(ev.get("final_options_sentiment_confidence", 0.5)),
            "options_sentiment_confidence_multiplier": float(ev.get("confidence_multiplier", 1.0)),
            "options_sentiment_skip_low_confirmation": bool(ev.get("skip_low_confirmation", False)),
            "options_sentiment_safety_warning": ev.get("safety_warning") or "",
            "options_sentiment_decision_impact": ev.get("decision_impact", ""),
        }
    )

    # Advisory re-weighting of the composite score (bounded by the multiplier caps).
    score["option_score"] = float(score.get("option_score", 0.0)) * fields["options_sentiment_confidence_multiplier"]
    score.update(fields)
    return score


def score_contract(
    row: pd.Series,
    underlying_price: float,
    parsed: dict[str, Any],
    dte: int,
    metrics: dict[str, float],
    sentiment: dict[str, Any] | None = None,
) -> dict[str, float]:
    flow = compute_flow_score(metrics)
    ev = contract_expected_value(
        row=row,
        underlying_price=underlying_price,
        strike=float(parsed["strike"]),
        dte=dte,
        metrics=metrics,
        flow_score=flow["flow_score"],
    )

    delta = metrics["delta"]
    gamma = metrics["gamma"]

    if math.isnan(delta):
        moneyness = parsed["strike"] / max(underlying_price, 1e-9)
        delta_score = 1.0 - min(abs(moneyness - 1.0) / 0.12, 1.0)
    else:
        half_width = max((CONFIG.target_delta_max - CONFIG.target_delta_min) / 2.0, 1e-9)
        delta_score = 1.0 - min(abs(delta - CONFIG.ideal_delta) / half_width, 1.0)

    dte_span = max((CONFIG.max_dte - CONFIG.min_dte) / 2.0, 1.0)
    dte_score = 1.0 - min(abs(dte - CONFIG.target_dte) / dte_span, 1.0)

    spread_score = 1.0 - min(metrics["spread_pct"] / max(CONFIG.max_spread_pct, 1e-9), 1.0)

    volume_score = min(metrics["volume"] / max(CONFIG.min_volume * 5.0, 1.0), 1.0)
    oi_score = min(metrics["open_interest"] / max(CONFIG.min_open_interest * 5.0, 1.0), 1.0)
    liquidity_score = 0.45 * volume_score + 0.55 * oi_score

    gamma_score = 0.5 if math.isnan(gamma) else float(np.clip(gamma / max(CONFIG.min_gamma * 3.0, 0.01), 0.0, 1.0))
    ev_score = float(np.clip((ev["expected_value_pct"] + 0.25) / 0.75, 0.0, 1.0))

    option_score = (
        CONFIG.flow_score_weight * flow["flow_score"]
        + CONFIG.ev_score_weight * ev_score
        + CONFIG.liquidity_score_weight * liquidity_score
        + CONFIG.spread_score_weight * spread_score
        + CONFIG.delta_score_weight * delta_score
        + CONFIG.gamma_score_weight * gamma_score
        + CONFIG.dte_score_weight * dte_score
    )

    if metrics["mid"] < 0.25:
        option_score *= 0.75

    score = {
        **flow,
        **ev,
        "option_score": float(option_score),
        "ev_score": float(ev_score),
        "liquidity_score": float(liquidity_score),
        "spread_score": float(spread_score),
        "delta_score": float(delta_score),
        "gamma_score": float(gamma_score),
        "dte_score": float(dte_score),
    }

    # Advisory sentiment overlay (long-call setup). Bullish premise: the booster
    # only adds calls when the underlying is in an allowed-long / profitable state.
    technical_bullish = (
        clip01(to_float(row.get("continuation_confidence"), 0.0)) >= 0.50
        or to_float(row.get("structure_confidence", row.get("bullish_structure_score", 0.0)), 0.0) > 0.0
        or to_float(row.get("momentum_score", row.get("positive_momentum_score", 0.0)), 0.0) > 0.0
    )
    score = apply_sentiment_to_contract(
        score,
        sentiment,
        expiration_iso=parsed["expiration"].isoformat(),
        technical_bullish=bool(technical_bullish),
    )
    return score


# =============================================================================
# Chain scan
# =============================================================================

def get_option_chain(option_client: OptionHistoricalDataClient, underlying: str) -> dict[str, Any]:
    try:
        req = OptionChainRequest(underlying_symbol=underlying)
        chain = option_client.get_option_chain(req)
        if isinstance(chain, dict):
            return chain
        if hasattr(chain, "data") and isinstance(chain.data, dict):
            return chain.data
        return dict(chain) if chain is not None else {}
    except Exception as exc:
        print(f"[flow] failed chain {underlying}: {exc}")
        write_event("option_chain_fetch_error", {"underlying": underlying, "error": str(exc)})
        return {}


def selected_limit_price(best: dict[str, Any]) -> float:
    source = CONFIG.limit_price_source
    if source == "ask":
        return to_float(best.get("ask"), 0.0)
    if source == "mid":
        return to_float(best.get("mid"), 0.0)
    return to_float(best.get("bid"), to_float(best.get("mid"), 0.0))


def scan_underlying(
    option_client: OptionHistoricalDataClient,
    row: pd.Series,
    underlying: str,
    underlying_price: float,
    sentiment: dict[str, Any] | None = None,
) -> pd.DataFrame:
    chain = get_option_chain(option_client, underlying)
    if not chain:
        return pd.DataFrame()

    today = now_utc().date()
    rows: list[dict[str, Any]] = []
    scanned = 0

    for contract_symbol, snapshot in chain.items():
        if scanned >= CONFIG.max_contracts_per_underlying_scan:
            break
        scanned += 1

        parsed = parse_occ_option_symbol(contract_symbol)
        if not parsed or parsed["type"] != "call":
            continue

        dte = (parsed["expiration"] - today).days
        min_entry_dte = max(CONFIG.min_dte, CONFIG.block_new_entries_within_expiration_days + 1)
        if dte < min_entry_dte or dte > CONFIG.max_dte:
            continue

        metrics = option_quote_metrics(snapshot)
        mid = metrics["mid"]

        if mid <= 0 or mid > CONFIG.max_contract_mid_price:
            continue
        if metrics["spread_pct"] > CONFIG.max_spread_pct:
            continue
        if metrics["volume"] < CONFIG.min_volume:
            continue
        if metrics["open_interest"] < CONFIG.min_open_interest:
            continue

        delta = metrics["delta"]
        if math.isnan(delta):
            moneyness = parsed["strike"] / max(underlying_price, 1e-9)
            if not (0.92 <= moneyness <= 1.12):
                continue
        else:
            if delta < CONFIG.target_delta_min or delta > CONFIG.target_delta_max:
                continue

        theta = metrics["theta"]
        if not math.isnan(theta) and abs(theta) > CONFIG.max_theta_abs:
            continue

        gamma = metrics["gamma"]
        if CONFIG.min_gamma > 0 and not math.isnan(gamma) and gamma < CONFIG.min_gamma:
            continue

        iv = metrics["iv"]
        if CONFIG.max_iv < 10 and not math.isnan(iv) and iv > CONFIG.max_iv:
            continue

        scores = score_contract(row, underlying_price, parsed, dte, metrics, sentiment=sentiment)

        if scores["rvolchg_pct"] < CONFIG.min_rvolchg_pct:
            continue
        if scores["premium_flow_dollars"] < CONFIG.min_premium_flow_dollars:
            continue
        if scores["flow_score"] < CONFIG.min_flow_score:
            continue
        if scores["probability_of_profit"] < CONFIG.min_probability_of_profit:
            continue
        # Advisory low-confirmation skip (only active when explicitly opted in via
        # OPTIONS_SENTIMENT_SKIP_LOW_CONFIRMATION; conservative — never forces a trade).
        if scores.get("options_sentiment_skip_low_confirmation"):
            write_event(
                "option_sentiment_low_confirmation_skip",
                {"underlying": underlying, "option_symbol": safe_symbol(contract_symbol),
                 "impact": scores.get("options_sentiment_decision_impact", "")},
            )
            continue

        rows.append(
            {
                "option_symbol": safe_symbol(contract_symbol),
                "underlying": underlying,
                "underlying_price": underlying_price,
                "expiration": parsed["expiration"].isoformat(),
                "dte": dte,
                "strike": parsed["strike"],
                "moneyness": parsed["strike"] / max(underlying_price, 1e-9),
                **metrics,
                **scores,
                "continuation_confidence": to_float(row.get("continuation_confidence"), 0.70),
                "source": row.get("source", ""),
                "created_at_utc": now_utc().isoformat(),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["option_score", "flow_score", "expected_value_dollars", "premium_flow_dollars"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def scan_unusual_options_flow(
    option_client: OptionHistoricalDataClient,
    universe: pd.DataFrame,
    stock_client: Any = None,
) -> pd.DataFrame:
    frames = []
    print(f"[flow] scanning underlyings={len(universe)}")
    sentiment_client = get_sentiment_client()
    if sentiment_client is not None:
        print(f"[sentiment] Alpha Vantage advisory sentiment enabled (api_key_present={bool(sentiment_client.api_key)})")

    for _, row in universe.iterrows():
        underlying = safe_symbol(row.get("symbol"))
        if not underlying:
            continue

        sentiment = None
        if sentiment_client is not None:
            try:
                sentiment = sentiment_client.get_symbol_sentiment(underlying)
                if sentiment.get("available"):
                    print(
                        f"[sentiment] {underlying}: PCR={sentiment.get('put_call_ratio_chain')} "
                        f"dir={sentiment.get('sentiment_directional_score'):+.2f} "
                        f"participation={sentiment.get('participation_score'):.2f} "
                        f"contrarian={sentiment.get('contrarian_warning')}"
                    )
            except Exception as exc:  # pragma: no cover - advisory must never break the scan
                print(f"[sentiment] {underlying}: fetch failed ({exc}); using neutral")
                sentiment = None

        # Fallback order: row price fields → Alpaca trade → quote mid → bar close → local intraday.
        underlying_price = to_float(row.get("price"), 0.0)
        if underlying_price <= 0:
            underlying_price = to_float(row.get("close"), 0.0)
        if underlying_price <= 0:
            underlying_price = to_float(row.get("current_price"), 0.0)
        if underlying_price <= 0:
            underlying_price = to_float(row.get("underlying_price"), 0.0)
        if underlying_price <= 0:
            underlying_price = to_float(row.get("stock_price"), 0.0)
        if underlying_price <= 0 and stock_client is not None:
            underlying_price = resolve_underlying_price_alpaca(underlying, stock_client)
            if underlying_price > 0:
                print(f"[flow] {underlying}: resolved price=${underlying_price:.2f} via Alpaca")
        if underlying_price <= 0:
            underlying_price = resolve_underlying_price_intraday(underlying)
            if underlying_price > 0:
                print(f"[flow] {underlying}: resolved price=${underlying_price:.2f} via local intraday")
        if underlying_price <= 0:
            underlying_price = resolve_underlying_price_structure_map(underlying)
            if underlying_price > 0:
                print(f"[flow] {underlying}: resolved price=${underlying_price:.2f} via structure map")
        if underlying_price <= 0:
            print(f"[flow] skip {underlying}: no underlying price from any source (tried row fields, Alpaca trade/quote/bar, local intraday, structure map)")
            continue

        ranked = scan_underlying(option_client, row, underlying, underlying_price, sentiment=sentiment)
        if not ranked.empty:
            frames.append(ranked)
            best = ranked.iloc[0]
            print(
                f"[flow] {underlying} best={best['option_symbol']} "
                f"score={best['option_score']:.3f} flow={best['flow_score']:.3f} "
                f"ev=${best['expected_value_dollars']:.2f} "
                f"Rvolchg%={best['rvolchg_pct']:.1f}"
            )
        else:
            print(f"[flow] {underlying}: no calls passed Rvolchg% filter")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["option_score", "flow_score", "premium_flow_dollars"], ascending=[False, False, False])
    return out.reset_index(drop=True)


# =============================================================================
# Orders / exits
# =============================================================================

def build_booster_orders(flow: pd.DataFrame, trading_client: TradingClient) -> pd.DataFrame:
    if flow.empty:
        return pd.DataFrame()

    positions = positions_by_symbol(trading_client)
    equity = account_equity(trading_client)
    existing_option_underlyings = open_option_underlyings(positions)

    if open_option_position_count(positions) >= CONFIG.max_open_option_positions:
        print("[options] max open option positions reached; no new calls")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for underlying, group in flow.groupby("underlying", sort=False):
        if len(rows) >= CONFIG.top_orders:
            break

        if underlying in existing_option_underlyings:
            print(f"[options] skip {underlying}: already has open call booster")
            continue

        underlying_pos = positions.get(underlying)

        if CONFIG.require_open_underlying_position:
            if underlying_pos is None or to_float(obj_get(underlying_pos, "qty", default=0.0)) <= 0:
                print(f"[options] skip {underlying}: no open underlying stock position")
                continue

            underlying_plpc = to_float(obj_get(underlying_pos, "unrealized_plpc", default=0.0))
            if underlying_plpc < CONFIG.min_underlying_unrealized_plpc:
                print(f"[options] skip {underlying}: underlying PLPC {underlying_plpc:.2%} below threshold")
                continue
        else:
            underlying_plpc = to_float(obj_get(underlying_pos, "unrealized_plpc", default=0.0)) if underlying_pos is not None else 0.0

        best = group.sort_values(["option_score", "flow_score", "expected_value_dollars"], ascending=[False, False, False]).iloc[0].to_dict()
        limit_price = selected_limit_price(best)
        if limit_price <= 0:
            limit_price = to_float(best.get("mid"), 0.0)

        contract_notional = limit_price * 100.0
        risk_budget = min(equity * CONFIG.max_risk_per_trade_pct, CONFIG.max_risk_per_trade_dollars)
        contracts = int(risk_budget // max(contract_notional, 1e-9))
        contracts = min(contracts, CONFIG.max_contracts_per_underlying)

        if contracts <= 0:
            print(f"[options] skip {underlying}: premium ${contract_notional:.2f} exceeds risk budget ${risk_budget:.2f}")
            continue

        rows.append(
            {
                "symbol": best["option_symbol"],
                "side": "buy",
                "delta_shares": contracts,
                "delta_dollars": contract_notional * contracts,
                "price": float(limit_price),
                "limit_price": float(limit_price),
                "reason": "option_booster_rvolchg_long_call",
                "target_weight": (contract_notional * contracts) / max(equity, 1e-9),
                "current_weight": 0.0,
                "delta_weight": (contract_notional * contracts) / max(equity, 1e-9),
                "trade_confidence": float(best.get("continuation_confidence", 0.70)),
                "confidence_score": float(best.get("option_score", 0.0)),
                "underlying_symbol": underlying,
                "underlying_unrealized_plpc": float(underlying_plpc),
                "underlying_price": float(best.get("underlying_price", 0.0)),
                "option_score": float(best.get("option_score", 0.0)),
                "option_flow_score": float(best.get("flow_score", 0.0)),
                "option_volume_oi_ratio": float(best.get("volume_oi_ratio", 0.0)),
                "option_premium_flow_dollars": float(best.get("premium_flow_dollars", 0.0)),
                "option_expected_value_dollars": float(best.get("expected_value_dollars", 0.0)),
                "option_expected_value_pct": float(best.get("expected_value_pct", 0.0)),
                "option_probability_of_profit": float(best.get("probability_of_profit", 0.0)),
                "option_delta": best.get("delta"),
                "option_gamma": best.get("gamma"),
                "option_theta": best.get("theta"),
                "option_iv": best.get("iv"),
                "option_dte": best.get("dte"),
                "option_strike": best.get("strike"),
                "option_spread_pct": best.get("spread_pct"),
                "option_volume": best.get("volume"),
                "option_open_interest": best.get("open_interest"),
                # Advisory Alpha Vantage options sentiment (never sizing-authoritative).
                "put_call_ratio_chain": best.get("put_call_ratio_chain"),
                "put_call_ratio_expiration": best.get("put_call_ratio_expiration"),
                "volume_oi_ratio_chain": best.get("volume_oi_ratio_chain"),
                "volume_oi_ratio_expiration": best.get("volume_oi_ratio_expiration"),
                "sentiment_directional_score": best.get("sentiment_directional_score"),
                "participation_score": best.get("participation_score"),
                "contrarian_warning": best.get("contrarian_warning"),
                "options_sentiment_confidence": best.get("options_sentiment_confidence"),
                "options_sentiment_confidence_multiplier": best.get("options_sentiment_confidence_multiplier"),
                "options_sentiment_safety_warning": best.get("options_sentiment_safety_warning"),
                "options_sentiment_decision_impact": best.get("options_sentiment_decision_impact"),
                "created_at_utc": now_utc().isoformat(),
            }
        )

    return pd.DataFrame(rows)


def build_exit_orders(trading_client: TradingClient) -> pd.DataFrame:
    positions = positions_by_symbol(trading_client)
    equity = account_equity(trading_client)
    state = load_state()
    high_plpc = {k: to_float(v) for k, v in state.get("option_high_plpc", {}).items()}

    rows: list[dict[str, Any]] = []

    for symbol, pos in positions.items():
        parsed = parse_occ_option_symbol(symbol)
        if not parsed or parsed["type"] != "call":
            continue

        qty = to_float(obj_get(pos, "qty", default=0.0))
        if qty <= 0:
            continue

        current_plpc = to_float(obj_get(pos, "unrealized_plpc", default=0.0))
        old_high = high_plpc.get(symbol, current_plpc)
        new_high = max(old_high, current_plpc)
        high_plpc[symbol] = new_high

        dte = (parsed["expiration"] - now_utc().date()).days
        reason = ""

        if dte <= CONFIG.close_before_expiration_days:
            reason = "option_booster_expiration_protection_exit"
        elif current_plpc <= CONFIG.hard_stop_pct:
            reason = "option_booster_hard_stop"
        elif current_plpc >= CONFIG.take_profit_pct:
            reason = "option_booster_take_profit"
        elif new_high >= CONFIG.profit_lock_trigger_pct and current_plpc <= new_high - CONFIG.profit_giveback_pct:
            reason = "option_booster_profit_giveback"

        if reason:
            market_value = to_float(obj_get(pos, "market_value", default=0.0))
            current_price = to_float(obj_get(pos, "current_price", default=0.0))
            rows.append(
                {
                    "symbol": symbol,
                    "side": "sell",
                    "delta_shares": int(qty),
                    "delta_dollars": float(market_value),
                    "price": float(current_price),
                    "reason": reason,
                    "target_weight": 0.0,
                    "current_weight": float(market_value) / max(equity, 1e-9),
                    "delta_weight": -float(market_value) / max(equity, 1e-9),
                    "trade_confidence": 1.0,
                    "underlying_symbol": parsed["underlying"],
                    "option_plpc": float(current_plpc),
                    "option_high_plpc": float(new_high),
                    "option_expiration": parsed["expiration"].isoformat(),
                    "option_dte": int(dte),
                    "created_at_utc": now_utc().isoformat(),
                }
            )
            high_plpc.pop(symbol, None)

    state["option_high_plpc"] = high_plpc
    state["updated_at_utc"] = now_utc().isoformat()
    save_state(state)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> dict[str, Any]:
    print("===== UNUSUAL FLOW OPTIONS BOOSTER =====")
    print(f"[config] enabled={CONFIG.enabled}")
    print(f"[config] paper={CONFIG.paper}")
    print(f"[config] universe_limit={CONFIG.max_underlyings}")
    print(f"[config] dte={CONFIG.min_dte}-{CONFIG.max_dte} target={CONFIG.target_dte}")
    print(f"[config] delta={CONFIG.target_delta_min:.2f}-{CONFIG.target_delta_max:.2f}")
    print(f"[config] min_volume={CONFIG.min_volume} min_oi={CONFIG.min_open_interest} min_rvolchg_pct={CONFIG.min_rvolchg_pct:.1f}")
    print(f"[config] min_premium_flow=${CONFIG.min_premium_flow_dollars:,.0f}")
    print(f"[config] require_open_underlying_position={CONFIG.require_open_underlying_position}")
    print(f"[config] options_sentiment_av_enabled={_SENTIMENT_AVAILABLE and env_bool('OPTIONS_SENTIMENT_AV_ENABLED', True)} (advisory only)")

    if not CONFIG.enabled:
        summary = {
            "created_at_utc": now_utc().isoformat(),
            "enabled": False,
            "message": "disabled",
        }
        save_json(SUMMARY_PATH, summary)
        return summary

    trading_client = build_trading_client()
    option_client = build_option_data_client()
    stock_client = build_stock_data_client()
    if stock_client is None:
        print("[config] stock data client unavailable; Alpaca price fallback disabled")

    universe = load_universe()
    print(f"[universe] loaded={len(universe)}")
    if not universe.empty:
        print("[universe] symbols=" + ",".join(universe["symbol"].head(CONFIG.max_underlyings).tolist()))

    # Filter universe to long_allowed_symbols from the structure map unless the
    # symbol already has an open stock position (booster can defend existing exposures).
    print(f"[config] require_structure_long_allowed={CONFIG.require_structure_long_allowed}")
    if CONFIG.require_structure_long_allowed and not universe.empty:
        structure_allowed = load_structure_long_allowed()
        if structure_allowed:
            open_stock_syms: set[str] = set()
            try:
                for sym in positions_by_symbol(trading_client):
                    if not parse_occ_option_symbol(sym):
                        open_stock_syms.add(safe_symbol(sym))
            except Exception as _pe:
                print(f"[structure-filter] could not fetch positions: {_pe}")
            allowed_set = structure_allowed | open_stock_syms
            before_count = len(universe)
            universe = universe[universe["symbol"].isin(allowed_set)].copy().reset_index(drop=True)
            removed = before_count - len(universe)
            print(
                f"[structure-filter] structure_allowed={len(structure_allowed)} "
                f"open_positions={len(open_stock_syms)} "
                f"universe_before={before_count} after={len(universe)} removed={removed}"
            )
            if not universe.empty:
                print("[structure-filter] filtered symbols=" + ",".join(universe["symbol"].tolist()))
        else:
            print("[structure-filter] no long_allowed symbols in structure map; skipping filter (map may be missing or pre-market)")
    elif not CONFIG.require_structure_long_allowed:
        print("[structure-filter] disabled via OPTIONS_REQUIRE_STRUCTURE_LONG_ALLOWED=0")

    flow = scan_unusual_options_flow(option_client, universe, stock_client=stock_client)

    if not flow.empty:
        flow.head(CONFIG.max_output_candidates).to_csv(OPTION_FLOW_CANDIDATES_PATH, index=False)
        flow.to_csv(RANKED_CONTRACTS_PATH, index=False)

        # Keep compatibility with run.py if it reads latest_option_booster_candidates.csv.
        flow_candidates = (
            flow.sort_values(["option_score", "flow_score"], ascending=[False, False])
            .groupby("underlying", as_index=False)
            .head(1)
            .rename(columns={"underlying": "symbol"})
        )
        flow_candidates.to_csv(OPTION_BOOSTER_CANDIDATES_PATH, index=False)

        print(f"[saved] flow_candidates={OPTION_FLOW_CANDIDATES_PATH} rows={min(len(flow), CONFIG.max_output_candidates)}")
        print(f"[saved] ranked_contracts={RANKED_CONTRACTS_PATH} rows={len(flow)}")
        print(f"[saved] booster_candidates={OPTION_BOOSTER_CANDIDATES_PATH} rows={len(flow_candidates)}")
    else:
        pd.DataFrame().to_csv(OPTION_FLOW_CANDIDATES_PATH, index=False)
        pd.DataFrame().to_csv(RANKED_CONTRACTS_PATH, index=False)
        pd.DataFrame().to_csv(OPTION_BOOSTER_CANDIDATES_PATH, index=False)
        print("[flow] no calls passed Rvolchg% filter")

    orders = build_booster_orders(flow, trading_client)
    exits = build_exit_orders(trading_client)

    append_or_write_csv(REBALANCE_ORDERS_PATH, orders, append=CONFIG.append_to_existing_rebalance)
    append_or_write_csv(EXIT_ORDERS_PATH, exits, append=CONFIG.append_to_existing_exits)

    print(f"[saved] rebalance_orders={REBALANCE_ORDERS_PATH} option_rows={len(orders)}")
    print(f"[saved] exit_orders={EXIT_ORDERS_PATH} option_exit_rows={len(exits)}")

    if not orders.empty:
        show_cols = [
            "underlying_symbol",
            "symbol",
            "delta_shares",
            "price",
            "option_score",
            "option_flow_score",
            "option_volume_oi_ratio",
            "option_premium_flow_dollars",
            "option_expected_value_dollars",
            "option_probability_of_profit",
            "option_delta",
            "option_dte",
            "option_spread_pct",
            "put_call_ratio_chain",
            "sentiment_directional_score",
            "participation_score",
            "options_sentiment_confidence_multiplier",
        ]
        print("[options] selected contracts:")
        print(orders[[c for c in show_cols if c in orders.columns]].to_string(index=False))
    else:
        print("[options] no option orders generated")

    summary = {
        "created_at_utc": now_utc().isoformat(),
        "enabled": CONFIG.enabled,
        "universe_count": int(len(universe)),
        "flow_candidates": int(len(flow)),
        "orders": int(len(orders)),
        "exits": int(len(exits)),
        "flow_candidates_path": str(OPTION_FLOW_CANDIDATES_PATH),
        "ranked_contracts_path": str(RANKED_CONTRACTS_PATH),
        "booster_candidates_path": str(OPTION_BOOSTER_CANDIDATES_PATH),
        "rebalance_orders_path": str(REBALANCE_ORDERS_PATH),
        "exit_orders_path": str(EXIT_ORDERS_PATH),
        "config": asdict(CONFIG),
    }

    save_json(SUMMARY_PATH, summary)
    write_event("unusual_flow_options_booster_completed", summary)

    return summary


if __name__ == "__main__":
    main()
