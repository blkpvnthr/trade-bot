#!/usr/bin/env python3
"""
backtest.py — Offline historical backtest for the options-first trading system.

Simulates the strategy from run.py using:
  - yfinance for historical stock/VIX prices (no Alpaca API calls)
  - Black-Scholes for option pricing and daily mark-to-market
  - Faithful re-implementation of all risk gates from run.py

IMPORTANT — $13K sizing constraint:
  risk_budget = equity × 1.5% = ~$195
  If 1 contract premium × 100 > $195, the trade is SKIPPED.
  No silent "floor to 1 contract" behaviour in this backtest.

Outputs:
  reports/backtests/options_13k_summary.md
  reports/backtests/options_13k_trades.csv
  reports/backtests/options_13k_equity_curve.csv

Usage:
  python backtest.py
  python backtest.py --start 2024-01-01 --end 2024-12-31 --equity 13000
  python backtest.py --force-min-1  # test variant that does allow min=1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

try:
    import requests
except Exception as e:
    print(f"[voi] requests import failed: {e}")
    raise

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")          # load ALPHA_VANTAGE_API_KEY and APCA_ keys

REPORTS_DIR = ROOT / "reports" / "backtests"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / ".backtest_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VOI_CACHE_DIR = ROOT / "logs" / "backtest_cache" / "volume_oi_ratio"
VOI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes (pure-Python, no scipy needed)
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float | None:
    if T <= 1e-6 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    if d1 is None:
        return max(S - K, 0.0)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    if d1 is None:
        return max(K - S, 0.0)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    if d1 is None:
        return 1.0 if S > K else 0.0
    return _ncdf(d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    if d1 is None:
        return -1.0 if S < K else 0.0
    return _ncdf(d1) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strike / expiry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strike_increment(price: float) -> float:
    if price < 5:
        return 0.50
    if price < 25:
        return 1.0
    if price < 100:
        return 2.50
    if price < 250:
        return 5.0
    if price < 500:
        return 10.0
    return 25.0


def strike_grid(price: float, width: float = 0.35) -> list[float]:
    """Round strikes around current price within ±width fraction."""
    inc = _strike_increment(price)
    lo = round(price * (1 - width) / inc) * inc
    hi = round(price * (1 + width) / inc) * inc
    strikes = []
    k = lo
    while k <= hi + 1e-6:
        strikes.append(round(k, 4))
        k += inc
    return strikes


def nearest_monthly_expiry(from_date: date, target_dte: int = 21) -> date:
    """Third Friday of the calendar month that puts DTE nearest to target_dte."""
    target = from_date + timedelta(days=target_dte)
    for month_offset in range(3):
        y = target.year + (target.month + month_offset - 1) // 12
        m = (target.month + month_offset - 1) % 12 + 1
        first = date(y, m, 1)
        # First Friday
        first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
        third_fri = first_fri + timedelta(weeks=2)
        dte = (third_fri - from_date).days
        if dte >= target_dte - 7:
            return third_fri
    # fallback
    first = date(target.year, target.month, 1)
    first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_fri + timedelta(weeks=2)


# ─────────────────────────────────────────────────────────────────────────────
# Scanner feature engineering (mirrors run.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _sym, g in bars.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        c = g["close"].astype(float)
        v = g["volume"].astype(float)
        g["ret_1d"]  = c.pct_change(1, fill_method=None)
        g["ret_5d"]  = c.pct_change(5, fill_method=None)
        g["ret_10d"] = c.pct_change(10, fill_method=None)
        g["sma5"]    = c.rolling(5).mean()
        g["sma20"]   = c.rolling(20).mean()
        g["sma50"]   = c.rolling(50).mean()
        g["sma200"]  = c.rolling(200).mean()
        g["rsi14"]   = _rsi(c, 14)
        g["atr14"]   = _atr(g, 14)
        g["atr_pct"] = g["atr14"] / c.replace(0, np.nan)
        g["vol20"]           = v.rolling(20).mean()
        g["dollar_volume20"] = (c * v).rolling(20).mean()
        g["rel_volume"]      = v / g["vol20"].replace(0, np.nan)
        sd = c.rolling(20).std()
        mid = g["sma20"]
        g["bb_low"]  = mid - 2 * sd
        g["bb_high"] = mid + 2 * sd
        g["bb_pos"]  = (c - g["bb_low"]) / (g["bb_high"] - g["bb_low"]).replace(0, np.nan)
        g["high20"]       = c.rolling(20).max()
        g["drawdown_20h"] = c / g["high20"].replace(0, np.nan) - 1.0
        g["trend_stack"]       = ((g["sma5"] > g["sma20"]) & (g["sma50"] > g["sma200"])).astype(float)
        g["above_sma20"]       = (c > g["sma20"]).astype(float)
        g["above_prev_high"]   = (c > g["high"].shift(1)).astype(float)
        g["lower_band_reclaim"] = (
            (c.shift(1) < g["bb_low"].shift(1)) & (c > g["bb_low"]) & (g["ret_1d"] > 0)
        ).astype(float)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score_latest(
    features: pd.DataFrame,
    min_price: float = 2.0,
    max_price: float = 1000.0,
    min_dollar_vol: float = 5_000_000.0,
) -> pd.DataFrame:
    latest = features.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()
    if latest.empty:
        return latest
    latest = latest[
        (latest["close"] >= min_price) &
        (latest["close"] <= max_price) &
        (latest["dollar_volume20"] >= min_dollar_vol)
    ].copy()
    if latest.empty:
        return latest

    def clip01(s: pd.Series, lo: float, hi: float) -> pd.Series:
        return ((s - lo) / max(hi - lo, 1e-9)).clip(0, 1)

    momentum = (
        0.20 * clip01(latest["ret_5d"], -0.02, 0.10)
        + 0.15 * clip01(latest["ret_10d"], -0.03, 0.18)
        + 0.15 * latest["trend_stack"].fillna(0)
        + 0.12 * latest["above_sma20"].fillna(0)
        + 0.10 * latest["above_prev_high"].fillna(0)
        + 0.10 * clip01(latest["rel_volume"], 0.8, 2.5)
        + 0.10 * clip01(latest["atr_pct"], 0.015, 0.09)
        + 0.08 * clip01(latest["bb_pos"], 0.45, 0.95)
    )
    reversal = (
        0.22 * (1 - clip01(latest["rsi14"], 30, 55))
        + 0.20 * latest["lower_band_reclaim"].fillna(0)
        + 0.16 * clip01(latest["ret_1d"], -0.01, 0.05)
        + 0.14 * clip01(latest["rel_volume"], 0.8, 2.8)
        + 0.14 * (1 - clip01(latest["bb_pos"], 0.15, 0.55))
        + 0.14 * (1 - clip01(latest["drawdown_20h"], -0.18, -0.03))
    )
    latest["momentum_score"] = momentum.clip(0, 1)
    latest["reversal_score"] = reversal.clip(0, 1)
    latest["setup_type"]     = np.where(latest["momentum_score"] >= latest["reversal_score"], "momentum", "reversal")
    latest["setup_score"]    = latest[["momentum_score", "reversal_score"]].max(axis=1)
    return latest.sort_values("setup_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    start_date: date        = date(2024, 1, 1)
    end_date: date          = date(2024, 12, 31)
    starting_equity: float  = 13_000.0
    risk_free_rate: float   = 0.053        # ~5.3% Fed Funds 2024

    enable_long_calls: bool = True
    enable_long_puts: bool  = True
    max_options_exposure: float   = 0.40
    risk_per_trade_pct: float     = 0.015  # 1.5%
    max_open_positions: int       = 4
    max_contracts_per_underlying: int = 2
    force_min_one_contract: bool  = False  # if True, allows 1 contract even if > risk budget

    target_dte: int      = 21
    min_dte: int         = 7
    max_dte: int         = 45
    preferred_dte_min: int = 14
    preferred_dte_max: int = 30

    call_delta_target: float = 0.50
    call_delta_min: float    = 0.35
    call_delta_max: float    = 0.65
    put_delta_target: float  = -0.50
    put_delta_min: float     = -0.65
    put_delta_max: float     = -0.35

    max_spread_pct: float     = 0.12   # 12% hard reject
    min_open_interest: int    = 100    # proxy: dollar_vol-based
    min_volume: int           = 20     # proxy

    take_profit_pct: float          = 0.60
    hard_stop_pct: float            = -0.35
    profit_lock_trigger_pct: float  = 0.40
    profit_giveback_pct: float      = 0.25
    close_before_expiration_days: int              = 3
    block_new_entries_within_expiration_days: int  = 5

    daily_drawdown_limit: float = -0.02  # kill switch threshold

    # Slippage model
    assumed_spread_pct: float       = 0.06   # 6% spread assumption for mid-market fills
    fill_slippage_half_spread: float = 0.25  # pay/receive 25% of half-spread beyond mid
    commission_per_contract: float  = 0.15   # regulatory fee per contract ($)

    # Sentiment proxy (replaces Alpha Vantage PCR in live system)
    vix_bullish_threshold: float  = 18.0
    vix_bearish_threshold: float  = 25.0
    vix_skip_threshold: float     = 35.0

    # IV model
    vol_risk_premium: float = 1.25   # realized_vol × this = IV estimate
    iv_floor: float         = 0.12
    iv_cap: float           = 2.50
    put_vol_skew: float     = 1.12   # puts trade at higher IV (skew premium)

    min_setup_score: float = 0.58
    scan_weekday: int      = 0       # 0=Monday

    # Dynamic risk budget (tiered by account size)
    use_dynamic_risk_budget: bool  = False
    max_single_position_pct: float = 0.08   # 8% of equity hard cap per position

    # Underlying liquidity filters (passed to score_latest)
    min_underlying_dollar_vol: float = 5_000_000.0
    min_underlying_price: float      = 2.0

    # Volume/OI scoring (Variant E)
    use_volume_oi_scoring: bool        = False
    volume_oi_min_ratio: float         = 0.20    # hard reject below this...
    volume_oi_oi_override: int         = 10_000  # ...unless proxy_oi > this
    volume_oi_spread_override: float   = 0.03    # ...AND spread < this

    universe: list[str] = field(default_factory=list)


@dataclass
class Position:
    underlying: str
    option_type: str    # "call" or "put"
    strike: float
    expiry: date
    entry_date: date
    entry_price: float  # premium per share
    contracts: int
    entry_cost: float   # total cash spent (premiums + commission)
    setup_score: float
    momentum_score: float
    high_plpc: float = 0.0
    dte_at_entry: int = 0
    current_price: float = 0.0        # set to entry_price on creation, updated daily
    vol_oi_ratio: float | None = None  # volume/OI ratio at entry date

    def current_value(self, premium: float) -> float:
        return premium * 100.0 * self.contracts

    def unrealized_plpc(self, premium: float) -> float:
        cv = self.current_value(premium)
        return (cv - self.entry_cost) / max(self.entry_cost, 1e-9)


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    underlying: str
    option_type: str
    strike: float
    expiry: str
    entry_dte: int
    exit_dte: int
    holding_days: int
    contracts: int
    entry_price: float
    exit_price: float
    entry_cost: float
    exit_proceeds: float
    gross_pl: float
    commission_total: float
    net_pl: float
    pl_pct: float
    exit_reason: str
    setup_score: float
    momentum_score: float
    vol_oi_ratio: float | None = None  # volume/OI ratio at entry
    skipped: bool = False
    skip_reason: str = ""


# Liquid options universe (Fix 1 from backtest report)
LIQUID_UNIVERSE_ONLY = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL",
    "TSLA", "AMD",
    "SOXX", "XLK", "XLE",
]


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Vantage Historical Volume / Open Interest Ratio
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalVolumeOIClient:
    """
    Fetches HISTORICAL_VOLUME_OPEN_INTEREST_RATIO from Alpha Vantage.
    Cache layout: logs/backtest_cache/volume_oi_ratio/{SYMBOL}/{YYYY-MM-DD}.json
    Never re-requests the same (symbol, date) pair.
    """

    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_PER_MINUTE = 5

    def __init__(self, api_key: str, cache_root: Path = VOI_CACHE_DIR, max_api_calls: int = 25):
        self.api_key      = api_key
        self.cache_root   = cache_root
        self.max_api_calls = max_api_calls   # per-session hard cap (free tier = 25/day)
        self._session_calls = 0
        self._call_times: list[float] = []

    def _cache_path(self, symbol: str, date_str: str) -> Path:
        return self.cache_root / symbol.upper() / f"{date_str}.json"

    def _rate_limit(self) -> None:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60.0]
        if len(self._call_times) >= self.RATE_LIMIT_PER_MINUTE:
            wait = 61.0 - (now - self._call_times[0])
            if wait > 0:
                print(f"[voi] rate limit: sleeping {wait:.1f}s")
                time.sleep(wait)
        self._call_times.append(time.time())

    def _fetch_api(self, symbol: str, date_str: str) -> dict | None:
        self._rate_limit()
        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "function": "HISTORICAL_VOLUME_OPEN_INTEREST_RATIO",
                    "symbol":   symbol,
                    "date":     date_str,
                    "apikey":   self.api_key,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for bad_key in ("Error Message", "Note", "Information"):
                if bad_key in data:
                    print(f"[voi] API blocked {symbol}/{date_str}: {str(data[bad_key])[:80]}")
                    return None
            return data
        except Exception as exc:
            print(f"[voi] fetch error {symbol}/{date_str}: {exc}")
            return None

    def _parse_ratio(self, data: dict | None, target_date: str) -> float | None:
        if not data:
            return None
        # Locate the list of records — try several key patterns
        records: list[dict] | None = None
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                records = val
                break
        if records is None:
            return None

        target_d = date.fromisoformat(target_date)
        best: dict | None = None
        best_diff = float("inf")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            d_str = rec.get("date", rec.get("timestamp", ""))
            if not d_str:
                continue
            try:
                d = date.fromisoformat(str(d_str)[:10])
                diff = abs((d - target_d).days)
                if diff < best_diff:
                    best_diff, best = diff, rec
            except Exception:
                continue

        if best is None or best_diff > 5:   # ignore data >5 trading days off target
            return None

        for k in ("volume_open_interest_ratio", "volumeOpenInterestRatio", "ratio", "value"):
            if k in best:
                try:
                    v = float(best[k])
                    return v if v >= 0 else None
                except (ValueError, TypeError):
                    continue
        return None

    def get_historical_volume_oi_ratio(self, symbol: str, date_str: str) -> float | None:
        """Return vol/OI ratio for symbol on date_str. Returns None on failure (neutral)."""
        symbol = symbol.upper()
        cache = self._cache_path(symbol, date_str)

        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8")).get("volume_open_interest_ratio")
            except Exception:
                pass

        if not self.api_key:
            return None

        # Respect per-session cap; return neutral on budget exhaustion
        if self._session_calls >= self.max_api_calls:
            return None

        raw   = self._fetch_api(symbol, date_str)
        self._session_calls += 1
        ratio = self._parse_ratio(raw, date_str)

        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache.write_text(json.dumps({
                "symbol":                    symbol,
                "date":                      date_str,
                "volume_open_interest_ratio": ratio,
                "raw_response":              raw,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

        return ratio


# ── Volume/OI scoring ─────────────────────────────────────────────────────────

def volume_oi_score_raw(ratio: float | None) -> float | None:
    """
    Converts a volume/OI ratio to a raw score per spec:
      ratio < 0.25         → -15  (stale, dead contracts)
      0.25 ≤ ratio < 0.75  →   0  (below-average activity)
      0.75 ≤ ratio < 1.50  → +10  (normal, slight positive)
      1.50 ≤ ratio < 3.00  → +20  (elevated participation)
      ratio ≥ 3.00         → +30  (strong new positioning)
    Returns None if ratio is None (→ neutral in ranking).
    """
    if ratio is None:
        return None
    if ratio < 0.25:
        return -15.0
    if ratio < 0.75:
        return 0.0
    if ratio < 1.50:
        return 10.0
    if ratio < 3.00:
        return 20.0
    return 30.0


def volume_oi_score_normalized(ratio: float | None) -> float:
    """
    Normalize raw score to [0, 1] for additive ranking with other [0,1] components.
    Raw range: [-15, +30] → span 45.  None → 0.50 (neutral).
    """
    raw = volume_oi_score_raw(ratio)
    if raw is None:
        return 0.50
    return (raw + 15.0) / 45.0


def _combined_contract_score(
    delta: float,
    delta_target: float,
    delta_half_range: float,
    proxy_oi: int,
    proxy_vol: int,
    momentum_score: float,
    vix_score: float,
    voi_ratio: float | None,
) -> float:
    """
    rank = technical_score + sentiment_score + liquidity_score + volume_oi_score
    Each component is in [0, 1]; result averaged to [0, 1].
    """
    delta_score     = max(0.0, 1.0 - abs(delta - delta_target) / max(delta_half_range, 1e-9))
    oi_score        = min(proxy_oi  / 10_000.0, 1.0)
    vol_score_c     = min(proxy_vol / 500.0,   1.0)
    liquidity_score = 0.55 * oi_score + 0.35 * vol_score_c + 0.10 * delta_score
    technical_score = float(np.clip(momentum_score, 0.0, 1.0))
    sentiment_score = float(np.clip(vix_score, 0.0, 1.0))
    voi_s           = volume_oi_score_normalized(voi_ratio)

    return (technical_score + sentiment_score + liquidity_score + voi_s) / 4.0


def _tiered_risk_pct(equity: float) -> float:
    """
    Dynamic risk % by account size.
    Larger accounts use lower % (diversified risk); smaller use higher %
    (minimum viable position sizing).
    """
    if equity >= 50_000:
        return 0.015   # 1.5% — standard
    elif equity >= 25_000:
        return 0.025   # 2.5%
    elif equity >= 10_000:
        return 0.040   # 4.0%
    else:
        return 0.050   # 5.0% — floor


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.prices: pd.DataFrame | None = None   # multi-symbol daily close
        self.ohlcv: pd.DataFrame | None = None    # full OHLCV for scanner
        self.vix: pd.Series | None = None         # daily VIX close
        self.vol30: pd.DataFrame | None = None    # 30-day annualised realised vol
        self.trading_days: list[date] = []

        # Volume/OI client — only active when use_volume_oi_scoring is enabled
        _av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.voi_client: HistoricalVolumeOIClient | None = (
            HistoricalVolumeOIClient(_av_key) if cfg.use_volume_oi_scoring and _av_key else None
        )
        if cfg.use_volume_oi_scoring and not _av_key:
            print("[voi] WARNING: use_volume_oi_scoring=True but ALPHA_VANTAGE_API_KEY not set. "
                  "Vol/OI scores will be neutral (0.50) for all contracts.")

    # ── Data ─────────────────────────────────────────────────────────────────

    def _cache_path(self, name: str) -> Path:
        return CACHE_DIR / f"{name}.csv.gz"

    def download_data(self) -> None:
        # Pull one extra year for indicator warmup (SMA200 needs 200 bars)
        warmup_start = date(self.cfg.start_date.year - 1, 1, 1)
        start_str = warmup_start.strftime("%Y-%m-%d")
        end_str   = (self.cfg.end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        universe = self.cfg.universe
        tickers  = list(set(universe + ["^VIX"]))

        cache_key = f"prices_{start_str}_{self.cfg.end_date.isoformat()}_{'_'.join(sorted(universe))[:40]}"
        cache_file = self._cache_path(cache_key)

        if cache_file.exists():
            print(f"[data] loading from cache {cache_file.name}")
            raw = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
        else:
            print(f"[data] downloading {len(tickers)} symbols from {start_str} to {end_str} ...")
            raw = yf.download(tickers, start=start_str, end=end_str,
                              auto_adjust=True, progress=False)
            raw.to_csv(cache_file)

        # VIX
        if ("Close", "^VIX") in raw.columns:
            self.vix = raw[("Close", "^VIX")].dropna()
        elif "Close" in raw.columns and isinstance(raw["Close"], pd.Series):
            self.vix = pd.Series(dtype=float)
        else:
            self.vix = pd.Series(dtype=float)

        # Stock prices
        rows = []
        for sym in universe:
            try:
                if ("Close", sym) in raw.columns:
                    sym_close  = raw[("Close",  sym)].rename("close")
                    sym_open   = raw[("Open",   sym)].rename("open")
                    sym_high   = raw[("High",   sym)].rename("high")
                    sym_low    = raw[("Low",    sym)].rename("low")
                    sym_volume = raw[("Volume", sym)].rename("volume")
                elif "Close" in raw.columns and isinstance(raw["Close"], pd.DataFrame):
                    sym_close  = raw["Close"][sym].rename("close")
                    sym_open   = raw["Open"][sym].rename("open")
                    sym_high   = raw["High"][sym].rename("high")
                    sym_low    = raw["Low"][sym].rename("low")
                    sym_volume = raw["Volume"][sym].rename("volume")
                else:
                    continue

                df = pd.concat([sym_open, sym_high, sym_low, sym_close, sym_volume], axis=1)
                df = df.dropna(subset=["close"])
                df.index = pd.to_datetime(df.index).date
                df.index.name = "date"
                df["symbol"] = sym
                df = df.reset_index()
                rows.append(df)
            except Exception as e:
                print(f"[data] skipping {sym}: {e}")

        if not rows:
            raise RuntimeError("No price data downloaded — check universe and date range.")

        self.ohlcv = pd.concat(rows, ignore_index=True)
        self.ohlcv["date"] = pd.to_datetime(self.ohlcv["date"]).dt.date

        # Pivot close prices for quick lookup
        self.prices = self.ohlcv.pivot(index="date", columns="symbol", values="close")

        # 30-day annualised realised vol
        log_returns = np.log(self.prices / self.prices.shift(1))
        self.vol30 = log_returns.rolling(30).std() * math.sqrt(252)

        # Trading days within backtest window
        all_dates = sorted(self.prices.index)
        self.trading_days = [d for d in all_dates
                             if self.cfg.start_date <= d <= self.cfg.end_date]
        print(f"[data] {len(self.trading_days)} trading days in backtest window")

    # ── IV estimate ──────────────────────────────────────────────────────────

    def implied_vol(self, symbol: str, as_of: date, option_type: str = "call") -> float:
        try:
            row_dates = [d for d in self.vol30.index if d <= as_of]
            if not row_dates:
                return 0.30
            rv = float(self.vol30.loc[row_dates[-1], symbol])
            if np.isnan(rv) or rv <= 0:
                return 0.30
            iv = rv * self.cfg.vol_risk_premium
            if option_type == "put":
                iv *= self.cfg.put_vol_skew
            return float(np.clip(iv, self.cfg.iv_floor, self.cfg.iv_cap))
        except Exception:
            return 0.30

    # ── Simulated OI / volume proxy ───────────────────────────────────────────

    def _proxy_oi_volume(self, symbol: str, as_of: date) -> tuple[int, int]:
        """Proxy OI and volume based on stock dollar volume (no real data)."""
        try:
            row_dates = [d for d in self.ohlcv["date"].unique() if d <= as_of]
            if not row_dates:
                return 0, 0
            last_date = max(row_dates)
            row = self.ohlcv[(self.ohlcv["symbol"] == symbol) & (self.ohlcv["date"] == last_date)]
            if row.empty:
                return 0, 0
            dv = float(row["close"].iloc[0]) * float(row["volume"].iloc[0])
            if dv >= 500_000_000:
                return 20000, 5000  # mega-cap / major ETF — easily clears any OI/vol threshold
            if dv >= 100_000_000:
                return 5000, 500
            if dv >= 20_000_000:
                return 800, 120
            if dv >= 5_000_000:
                return 150, 30
            return 30, 5            # too illiquid for options
        except Exception:
            return 0, 0

    # ── Contract selection ────────────────────────────────────────────────────

    def select_contract(
        self,
        underlying: str,
        spot: float,
        sigma: float,
        scan_date: date,
        option_type: str,
        momentum_score: float = 0.5,
        vix_score: float = 0.5,
    ) -> dict[str, Any] | None:
        """
        Find best contract using BS pricing.
        When use_volume_oi_scoring=True, ranks by combined score instead of pure
        delta proximity: rank = technical + sentiment + liquidity + volume_oi.
        Returns None if no contract passes hard filters.
        """
        cfg = self.cfg
        r = cfg.risk_free_rate

        expiry = nearest_monthly_expiry(scan_date, cfg.target_dte)
        dte = (expiry - scan_date).days

        # Expiration guard
        if dte < cfg.block_new_entries_within_expiration_days:
            return None
        if dte < cfg.min_dte or dte > cfg.max_dte:
            return None

        T = dte / 252.0

        # OI / volume proxy check
        proxy_oi, proxy_vol = self._proxy_oi_volume(underlying, scan_date)
        if proxy_oi < cfg.min_open_interest or proxy_vol < cfg.min_volume:
            return None

        # Fetch vol/OI ratio once for this underlying/date (used in scoring + filter)
        voi_ratio: float | None = None
        if cfg.use_volume_oi_scoring and self.voi_client is not None:
            voi_ratio = self.voi_client.get_historical_volume_oi_ratio(
                underlying, scan_date.isoformat()
            )
            # Hard reject: ratio < min_ratio unless override conditions met
            if (voi_ratio is not None
                    and voi_ratio < cfg.volume_oi_min_ratio
                    and not (proxy_oi > cfg.volume_oi_oi_override
                             and cfg.assumed_spread_pct < cfg.volume_oi_spread_override)):
                return None   # stale contract universe for this underlying

        if option_type == "call":
            delta_fn       = bs_call_delta
            price_fn       = bs_call
            delta_min      = cfg.call_delta_min
            delta_max      = cfg.call_delta_max
            delta_tgt      = cfg.call_delta_target
        else:
            delta_fn       = bs_put_delta
            price_fn       = bs_put
            delta_min      = cfg.put_delta_min
            delta_max      = cfg.put_delta_max
            delta_tgt      = cfg.put_delta_target

        delta_half_range = (delta_max - delta_min) / 2.0

        candidates: list[dict] = []

        for K in strike_grid(spot):
            d = delta_fn(spot, K, T, r, sigma)
            if not (delta_min <= d <= delta_max):
                continue
            premium = price_fn(spot, K, T, r, sigma)
            if premium <= 0:
                continue
            mid        = premium
            spread_pct = cfg.assumed_spread_pct
            if spread_pct > cfg.max_spread_pct:
                continue

            score = _combined_contract_score(
                delta            = d,
                delta_target     = delta_tgt,
                delta_half_range = abs(delta_half_range),
                proxy_oi         = proxy_oi,
                proxy_vol        = proxy_vol,
                momentum_score   = momentum_score,
                vix_score        = vix_score,
                voi_ratio        = voi_ratio if cfg.use_volume_oi_scoring else None,
            )
            candidates.append({
                "underlying":    underlying,
                "option_type":   option_type,
                "strike":        K,
                "expiry":        expiry,
                "dte":           dte,
                "delta":         d,
                "mid":           mid,
                "spread_pct":    spread_pct,
                "sigma":         sigma,
                "proxy_oi":      proxy_oi,
                "proxy_vol":     proxy_vol,
                "combined_score": score,
                "vol_oi_ratio":  voi_ratio,
            })

        if not candidates:
            return None

        # Pick highest combined score (or smallest delta diff if scoring disabled)
        return max(candidates, key=lambda c: c["combined_score"])

    # ── Option valuation (daily MTM) ─────────────────────────────────────────

    def value_option(
        self,
        pos: Position,
        today: date,
        sigma: float,
    ) -> float:
        """Return current BS premium per share for open position."""
        try:
            price_dates = [d for d in self.prices.index if d <= today]
            if not price_dates:
                return pos.entry_price
            last_date = max(price_dates)
            spot = float(self.prices.loc[last_date, pos.underlying])
            if np.isnan(spot) or spot <= 0:
                return pos.entry_price
        except Exception:
            return pos.entry_price

        dte = (pos.expiry - today).days
        if dte <= 0:
            # Intrinsic only at/past expiry
            if pos.option_type == "call":
                return max(spot - pos.strike, 0.0)
            else:
                return max(pos.strike - spot, 0.0)

        T = dte / 252.0
        r = self.cfg.risk_free_rate
        if pos.option_type == "call":
            return bs_call(spot, pos.strike, T, r, sigma)
        else:
            return bs_put(spot, pos.strike, T, r, sigma)

    # ── VIX sentiment proxy ───────────────────────────────────────────────────

    def vix_on(self, today: date) -> float:
        if self.vix is None or self.vix.empty:
            return 18.0  # neutral default
        try:
            dates = [d for d in self.vix.index if pd.Timestamp(d).date() <= today]
            if not dates:
                return 18.0
            return float(self.vix.loc[max(dates)])
        except Exception:
            return 18.0

    # ── Entry sizing ──────────────────────────────────────────────────────────

    def _size_entry(self, contract: dict, equity: float) -> tuple[int, float, str]:
        """
        Returns (n_contracts, entry_cost, skip_reason).
        skip_reason is "" if entry is approved, else a non-empty string.
        """
        cfg = self.cfg
        mid = contract["mid"]
        contract_cost = mid * 100.0  # premium for 1 contract (100 shares)

        # Dynamic or fixed risk budget
        if cfg.use_dynamic_risk_budget:
            risk_pct    = _tiered_risk_pct(equity)
            pos_cap     = equity * cfg.max_single_position_pct  # 8% hard cap
            risk_budget = min(equity * risk_pct, pos_cap)
        else:
            risk_pct    = cfg.risk_per_trade_pct
            risk_budget = equity * risk_pct

        if contract_cost > risk_budget:
            if cfg.force_min_one_contract:
                n = 1
            else:
                return 0, 0.0, (
                    f"contract_cost_exceeds_risk_budget "
                    f"cost=${contract_cost:.2f} budget=${risk_budget:.2f} "
                    f"pct={risk_pct*100:.1f}%"
                )
        else:
            n = int(risk_budget // contract_cost)

        n = max(1, min(n, cfg.max_contracts_per_underlying))

        # Slippage on entry: pay slightly above mid
        half_spread = mid * cfg.assumed_spread_pct / 2.0
        fill_price  = mid + half_spread * cfg.fill_slippage_half_spread
        commission  = cfg.commission_per_contract * n
        total_cost  = fill_price * 100.0 * n + commission

        return n, total_cost, ""

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(self) -> tuple[list[TradeRecord], list[dict], list[dict]]:
        """
        Returns: (closed_trades, equity_curve, skip_log)
        """
        cfg = self.cfg
        equity        = cfg.starting_equity
        cash          = cfg.starting_equity
        positions: list[Position] = []
        closed_trades: list[TradeRecord] = []
        equity_curve:  list[dict] = []
        skip_log:      list[dict] = []

        day_start_equity = equity
        kill_switch_active = False

        # Pre-build feature windows
        print("[backtest] pre-computing scanner features...")
        features = build_features(self.ohlcv)

        last_scan_date: date | None = None

        for idx, today in enumerate(self.trading_days):
            # ── Daily mark-to-market ─────────────────────────────────────────
            options_mv = 0.0
            for pos in positions:
                sigma = self.implied_vol(pos.underlying, today, pos.option_type)
                premium = self.value_option(pos, today, sigma)
                pos.current_price = premium
                mv = premium * 100.0 * pos.contracts
                options_mv += mv
                plpc = pos.unrealized_plpc(premium)
                pos.high_plpc = max(pos.high_plpc, plpc)

            equity = cash + options_mv

            # Reset daily drawdown tracking at start of each day
            if idx == 0 or today.month != self.trading_days[idx - 1].month:
                day_start_equity = equity

            # Record equity curve BEFORE exits so we see intraday value
            vix_now = self.vix_on(today)
            exposure_pct = options_mv / max(equity, 1e-9)
            equity_curve.append({
                "date":               today.isoformat(),
                "equity":             round(equity, 2),
                "cash":               round(cash, 2),
                "options_mv":         round(options_mv, 2),
                "options_exposure_pct": round(exposure_pct * 100, 2),
                "open_positions":     len(positions),
                "vix":                round(vix_now, 2),
                "kill_switch":        kill_switch_active,
            })

            # Daily drawdown kill switch (resets daily)
            daily_return = (equity - day_start_equity) / max(day_start_equity, 1e-9)
            if idx == 0:
                day_start_equity = equity
            if daily_return <= cfg.daily_drawdown_limit and not kill_switch_active:
                kill_switch_active = True
                print(f"[kill-switch] triggered on {today}: daily_return={daily_return:.2%}")

            # Reset kill switch at start of new day (next iteration resets it)
            # (In this simplified model we reset daily at start; keep active for rest of day)

            # ── EXIT LOGIC ───────────────────────────────────────────────────
            still_open = []
            for pos in positions:
                sigma   = self.implied_vol(pos.underlying, today, pos.option_type)
                premium = pos.current_price
                plpc    = pos.unrealized_plpc(premium)
                dte     = (pos.expiry - today).days

                exit_reason = ""
                if dte <= cfg.close_before_expiration_days:
                    exit_reason = f"expiration_protection (dte={dte})"
                elif plpc <= cfg.hard_stop_pct:
                    exit_reason = f"hard_stop ({plpc:.1%})"
                elif plpc >= cfg.take_profit_pct:
                    exit_reason = f"take_profit ({plpc:.1%})"
                elif (pos.high_plpc >= cfg.profit_lock_trigger_pct
                      and plpc <= pos.high_plpc - cfg.profit_giveback_pct):
                    exit_reason = f"profit_giveback (high={pos.high_plpc:.1%} cur={plpc:.1%})"

                if exit_reason:
                    # Slippage on exit: receive slightly below mid
                    half_spread = premium * cfg.assumed_spread_pct / 2.0
                    fill_price  = max(
                        premium - half_spread * cfg.fill_slippage_half_spread, 0.01
                    )
                    commission  = cfg.commission_per_contract * pos.contracts
                    proceeds    = fill_price * 100.0 * pos.contracts - commission
                    gross_pl    = proceeds - pos.entry_cost + commission * 2  # total commissions counted in entry
                    net_pl      = proceeds - pos.entry_cost
                    pl_pct      = net_pl / max(pos.entry_cost, 1e-9)

                    cash += proceeds

                    closed_trades.append(TradeRecord(
                        entry_date     = pos.entry_date.isoformat(),
                        exit_date      = today.isoformat(),
                        underlying     = pos.underlying,
                        option_type    = pos.option_type,
                        strike         = pos.strike,
                        expiry         = pos.expiry.isoformat(),
                        entry_dte      = pos.dte_at_entry,
                        exit_dte       = dte,
                        holding_days   = (today - pos.entry_date).days,
                        contracts      = pos.contracts,
                        entry_price    = pos.entry_price,
                        exit_price     = fill_price,
                        entry_cost     = pos.entry_cost,
                        exit_proceeds  = proceeds,
                        gross_pl       = net_pl,       # net of commissions
                        commission_total = commission * 2,
                        net_pl         = net_pl,
                        pl_pct         = pl_pct,
                        exit_reason    = exit_reason,
                        setup_score    = pos.setup_score,
                        momentum_score = pos.momentum_score,
                        vol_oi_ratio   = pos.vol_oi_ratio,
                    ))
                else:
                    still_open.append(pos)

            positions = still_open

            # ── ENTRY LOGIC (weekly scan on Mondays) ─────────────────────────
            is_scan_day = (today.weekday() == cfg.scan_weekday
                           or last_scan_date is None
                           or (today - last_scan_date).days >= 5)

            if (is_scan_day
                    and not kill_switch_active
                    and len(positions) < cfg.max_open_positions):

                last_scan_date = today
                kill_switch_active = False  # reset kill switch for new day's scan

                vix = self.vix_on(today)
                if vix >= cfg.vix_skip_threshold:
                    # High fear — skip all new entries
                    skip_log.append({"date": today.isoformat(), "reason": f"vix_too_high ({vix:.1f})"})
                else:
                    # Get scanner scores as of today (use all bars up to today)
                    today_features = features[
                        pd.to_datetime(features["date"]).dt.date <= today
                    ]
                    scores = score_latest(
                        today_features,
                        min_price=cfg.min_underlying_price,
                        min_dollar_vol=cfg.min_underlying_dollar_vol,
                    )

                    if not scores.empty:
                        open_underlyings = {p.underlying for p in positions}
                        candidates = scores[
                            (scores["setup_score"] >= cfg.min_setup_score)
                        ].head(cfg.max_open_positions * 2)

                        for _, row in candidates.iterrows():
                            if len(positions) >= cfg.max_open_positions:
                                break

                            sym   = str(row["symbol"])
                            spot  = float(row["close"])
                            if sym in open_underlyings:
                                continue
                            if spot <= 0:
                                continue

                            # Check options exposure room
                            recompute_mv = sum(
                                p.current_price * 100.0 * p.contracts for p in positions
                            )
                            recompute_equity = cash + recompute_mv
                            current_exposure = recompute_mv / max(recompute_equity, 1e-9)
                            if current_exposure >= cfg.max_options_exposure:
                                break

                            # Direction from VIX + momentum
                            opt_types = []
                            ms = float(row.get("momentum_score", 0))
                            rs = float(row.get("reversal_score", 0))
                            setup_type = str(row.get("setup_type", "momentum"))

                            if cfg.enable_long_calls:
                                # Calls when momentum or VIX is low-moderate
                                if vix < cfg.vix_bearish_threshold or setup_type == "momentum":
                                    opt_types.append("call")

                            if cfg.enable_long_puts:
                                # Puts when bearish VIX and reversal (oversold bounce)
                                if vix >= cfg.vix_bearish_threshold and setup_type == "reversal":
                                    opt_types.append("put")

                            for opt_type in opt_types:
                                if sym in {p.underlying for p in positions}:
                                    break

                                sigma = self.implied_vol(sym, today, opt_type)
                                vix_score_norm = max(0.0, 1.0 - vix / 50.0)
                                contract = self.select_contract(
                                    sym, spot, sigma, today, opt_type,
                                    momentum_score=ms,
                                    vix_score=vix_score_norm,
                                )

                                if contract is None:
                                    skip_log.append({
                                        "date":       today.isoformat(),
                                        "symbol":     sym,
                                        "opt_type":   opt_type,
                                        "reason":     "no_eligible_contract",
                                    })
                                    continue

                                n, total_cost, skip_reason = self._size_entry(
                                    contract, recompute_equity
                                )
                                if skip_reason:
                                    skip_log.append({
                                        "date":      today.isoformat(),
                                        "symbol":    sym,
                                        "opt_type":  opt_type,
                                        "reason":    skip_reason,
                                        "mid":       round(contract["mid"], 2),
                                        "risk_budget": round(recompute_equity * cfg.risk_per_trade_pct, 2),
                                    })
                                    continue

                                if total_cost > cash:
                                    skip_log.append({
                                        "date":    today.isoformat(),
                                        "symbol":  sym,
                                        "reason":  f"insufficient_cash total={total_cost:.2f} cash={cash:.2f}",
                                    })
                                    continue

                                cash -= total_cost
                                half_spread = contract["mid"] * cfg.assumed_spread_pct / 2.0
                                fill_entry  = contract["mid"] + half_spread * cfg.fill_slippage_half_spread

                                pos = Position(
                                    underlying     = sym,
                                    option_type    = opt_type,
                                    strike         = contract["strike"],
                                    expiry         = contract["expiry"],
                                    entry_date     = today,
                                    entry_price    = fill_entry,
                                    contracts      = n,
                                    entry_cost     = total_cost,
                                    setup_score    = float(row.get("setup_score", 0)),
                                    momentum_score = float(row.get("momentum_score", 0)),
                                    dte_at_entry   = contract["dte"],
                                    current_price  = fill_entry,
                                    vol_oi_ratio   = contract.get("vol_oi_ratio"),
                                )
                                positions.append(pos)
                                open_underlyings.add(sym)

                                print(
                                    f"  ENTRY {today} | {sym} {opt_type} "
                                    f"K={contract['strike']} exp={contract['expiry']} "
                                    f"DTE={contract['dte']} delta={contract['delta']:.2f} "
                                    f"mid={contract['mid']:.2f} n={n} "
                                    f"cost=${total_cost:.2f}"
                                )

            # Reset kill switch daily (only kills same-day entries)
            kill_switch_active = False

        # ── Force-close remaining open positions at end of backtest ──────────
        last_day = self.trading_days[-1]
        for pos in positions:
            sigma   = self.implied_vol(pos.underlying, last_day, pos.option_type)
            premium = self.value_option(pos, last_day, sigma)
            dte     = (pos.expiry - last_day).days
            half_spread = premium * cfg.assumed_spread_pct / 2.0
            fill_price  = max(premium - half_spread * cfg.fill_slippage_half_spread, 0.01)
            commission  = cfg.commission_per_contract * pos.contracts
            proceeds    = fill_price * 100.0 * pos.contracts - commission
            net_pl      = proceeds - pos.entry_cost
            pl_pct      = net_pl / max(pos.entry_cost, 1e-9)
            cash += proceeds
            closed_trades.append(TradeRecord(
                entry_date     = pos.entry_date.isoformat(),
                exit_date      = last_day.isoformat(),
                underlying     = pos.underlying,
                option_type    = pos.option_type,
                strike         = pos.strike,
                expiry         = pos.expiry.isoformat(),
                entry_dte      = pos.dte_at_entry,
                exit_dte       = dte,
                holding_days   = (last_day - pos.entry_date).days,
                contracts      = pos.contracts,
                entry_price    = pos.entry_price,
                exit_price     = fill_price,
                entry_cost     = pos.entry_cost,
                exit_proceeds  = proceeds,
                gross_pl       = net_pl,
                commission_total = commission * 2,
                net_pl         = net_pl,
                pl_pct         = pl_pct,
                exit_reason    = "end_of_backtest",
                setup_score    = pos.setup_score,
                momentum_score = pos.momentum_score,
                vol_oi_ratio   = pos.vol_oi_ratio,
            ))

        return closed_trades, equity_curve, skip_log

    # ── Performance metrics ───────────────────────────────────────────────────

    def compute_metrics(
        self,
        trades: list[TradeRecord],
        equity_curve: list[dict],
    ) -> dict:
        cfg = self.cfg
        starting = cfg.starting_equity
        ending   = float(equity_curve[-1]["equity"]) if equity_curve else starting

        eq = pd.Series(
            [float(r["equity"]) for r in equity_curve],
            index=[r["date"] for r in equity_curve],
        )
        daily_ret = eq.pct_change().dropna()

        # Max drawdown
        peak     = eq.cummax()
        drawdown = (eq - peak) / peak
        max_dd   = float(drawdown.min())

        # Sharpe
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            ann_ret    = (ending / starting) ** (252 / max(len(daily_ret), 1)) - 1
            ann_vol    = float(daily_ret.std() * math.sqrt(252))
            sharpe     = (ann_ret - cfg.risk_free_rate) / ann_vol
        else:
            sharpe  = 0.0
            ann_ret = 0.0
            ann_vol = 0.0

        completed = [t for t in trades if t.exit_reason != "end_of_backtest" or True]
        winners   = [t for t in completed if t.net_pl > 0]
        losers    = [t for t in completed if t.net_pl <= 0]
        calls     = [t for t in completed if t.option_type == "call"]
        puts      = [t for t in completed if t.option_type == "put"]

        gross_profit = sum(t.net_pl for t in winners)
        gross_loss   = abs(sum(t.net_pl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win  = gross_profit / len(winners) if winners else 0.0
        avg_loss = gross_loss   / len(losers)  if losers  else 0.0

        largest_winner = max((t.net_pl for t in completed), default=0.0)
        largest_loser  = min((t.net_pl for t in completed), default=0.0)
        avg_hold = (
            sum((t.holding_days for t in completed)) / len(completed)
            if completed else 0.0
        )
        avg_position_size = (
            sum(t.entry_cost for t in completed) / len(completed)
            if completed else 0.0
        )

        return {
            "starting_equity":    starting,
            "ending_equity":      round(ending, 2),
            "total_return_pct":   round((ending - starting) / starting * 100, 2),
            "annualised_return_pct": round(ann_ret * 100, 2),
            "annualised_vol_pct": round(ann_vol * 100, 2),
            "max_drawdown_pct":   round(max_dd * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "total_trades":       len(completed),
            "winning_trades":     len(winners),
            "losing_trades":      len(losers),
            "win_rate_pct":       round(len(winners) / max(len(completed), 1) * 100, 1),
            "profit_factor":      round(profit_factor, 3),
            "avg_win":            round(avg_win, 2),
            "avg_loss":           round(avg_loss, 2),
            "largest_winner":     round(largest_winner, 2),
            "largest_loser":      round(largest_loser, 2),
            "avg_holding_days":   round(avg_hold, 1),
            "avg_position_size":  round(avg_position_size, 2),
            "total_calls":        len(calls),
            "total_puts":         len(puts),
            "gross_profit":       round(gross_profit, 2),
            "gross_loss":         round(gross_loss, 2),
            "total_commissions":  round(sum(t.commission_total for t in completed), 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def save_reports(
    trades: list[TradeRecord],
    equity_curve: list[dict],
    skip_log: list[dict],
    metrics: dict,
    cfg: BacktestConfig,
) -> None:
    # ── Equity curve CSV ─────────────────────────────────────────────────────
    eq_path = REPORTS_DIR / "options_13k_equity_curve.csv"
    with open(eq_path, "w", newline="") as f:
        if equity_curve:
            w = csv.DictWriter(f, fieldnames=list(equity_curve[0].keys()))
            w.writeheader()
            w.writerows(equity_curve)
    print(f"[saved] {eq_path}")

    # ── Trades CSV ───────────────────────────────────────────────────────────
    tr_path = REPORTS_DIR / "options_13k_trades.csv"
    trade_dicts = [t.__dict__ for t in trades]
    # Add skip log entries flagged as skipped
    skip_entries_as_trades = [{
        "entry_date":     s.get("date", ""),
        "exit_date":      "",
        "underlying":     s.get("symbol", ""),
        "option_type":    s.get("opt_type", ""),
        "strike":         "",
        "expiry":         "",
        "entry_dte":      "",
        "exit_dte":       "",
        "holding_days":   0,
        "contracts":      0,
        "entry_price":    s.get("mid", ""),
        "exit_price":     "",
        "entry_cost":     "",
        "exit_proceeds":  "",
        "gross_pl":       "",
        "commission_total": "",
        "net_pl":         "",
        "pl_pct":         "",
        "exit_reason":    "",
        "setup_score":    "",
        "momentum_score": "",
        "skipped":        True,
        "skip_reason":    s.get("reason", ""),
    } for s in skip_log if "symbol" in s]

    all_rows = trade_dicts + skip_entries_as_trades
    if all_rows:
        with open(tr_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"[saved] {tr_path}")

    # ── Summary markdown ─────────────────────────────────────────────────────
    risk_skips = [s for s in skip_log if "contract_cost_exceeds_risk_budget" in s.get("reason", "")]
    no_contract_skips = [s for s in skip_log if s.get("reason") == "no_eligible_contract"]
    vix_skips  = [s for s in skip_log if "vix_too_high" in s.get("reason", "")]

    force_note = (
        "**Variant: force_min_one_contract=True** — minimum 1 contract even if cost > risk budget."
        if cfg.force_min_one_contract else
        "**Strict risk budget**: if 1 contract premium × 100 > risk_budget, trade is **SKIPPED**."
    )

    md = f"""# Options-First Backtest — $13K Account
## Configuration

| Parameter | Value |
|---|---|
| Start date | {cfg.start_date} |
| End date | {cfg.end_date} |
| Starting equity | ${cfg.starting_equity:,.0f} |
| Risk per trade | {cfg.risk_per_trade_pct*100:.1f}% (= ${cfg.starting_equity*cfg.risk_per_trade_pct:.0f} at start) |
| Max options exposure | {cfg.max_options_exposure*100:.0f}% |
| Max open positions | {cfg.max_open_positions} |
| Max contracts/underlying | {cfg.max_contracts_per_underlying} |
| Target DTE | {cfg.target_dte} days |
| Call delta range | [{cfg.call_delta_min}, {cfg.call_delta_max}] (target {cfg.call_delta_target}) |
| Put delta range | [{cfg.put_delta_min}, {cfg.put_delta_max}] (target {cfg.put_delta_target}) |
| Hard stop | {cfg.hard_stop_pct*100:.0f}% |
| Take profit | {cfg.take_profit_pct*100:.0f}% |
| Profit lock trigger | {cfg.profit_lock_trigger_pct*100:.0f}% |
| Profit giveback | {cfg.profit_giveback_pct*100:.0f}% |
| Close before expiry | {cfg.close_before_expiration_days} days |
| Block new entries | within {cfg.block_new_entries_within_expiration_days} days of expiry |
| Enable long calls | {cfg.enable_long_calls} |
| Enable long puts | {cfg.enable_long_puts} |
| Assumed spread | {cfg.assumed_spread_pct*100:.0f}% of mid |
| Commission | ${cfg.commission_per_contract}/contract |
| IV model | 30d realized vol × {cfg.vol_risk_premium} (puts ×{cfg.put_vol_skew}) |
| Sizing model | {force_note} |

## Performance Summary

| Metric | Value |
|---|---|
| Starting equity | ${metrics['starting_equity']:,.2f} |
| Ending equity | ${metrics['ending_equity']:,.2f} |
| **Total return** | **{metrics['total_return_pct']:+.2f}%** |
| Annualised return | {metrics['annualised_return_pct']:+.2f}% |
| Annualised volatility | {metrics['annualised_vol_pct']:.2f}% |
| **Max drawdown** | **{metrics['max_drawdown_pct']:.2f}%** |
| **Sharpe ratio** | **{metrics['sharpe_ratio']:.3f}** |
| Total trades | {metrics['total_trades']} |
| Win rate | {metrics['win_rate_pct']:.1f}% |
| Profit factor | {metrics['profit_factor']:.3f} |
| Avg win | ${metrics['avg_win']:.2f} |
| Avg loss | ${metrics['avg_loss']:.2f} |
| Largest winner | ${metrics['largest_winner']:.2f} |
| Largest loser | ${metrics['largest_loser']:.2f} |
| Avg holding period | {metrics['avg_holding_days']:.1f} days |
| Calls traded | {metrics['total_calls']} |
| Puts traded | {metrics['total_puts']} |
| Gross profit | ${metrics['gross_profit']:.2f} |
| Gross loss | ${metrics['gross_loss']:.2f} |
| Total commissions | ${metrics['total_commissions']:.2f} |

## Trade Breakdown

| Exit reason | Count |
|---|---|
"""
    if trades:
        exit_counts = {}
        for t in trades:
            r = t.exit_reason.split(" (")[0]
            exit_counts[r] = exit_counts.get(r, 0) + 1
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            md += f"| {reason} | {count} |\n"
    else:
        md += "| — | 0 |\n"

    md += f"""
## Skip Summary

| Skip reason | Count |
|---|---|
| contract_cost_exceeds_risk_budget | {len(risk_skips)} |
| no_eligible_contract | {len(no_contract_skips)} |
| vix_too_high | {len(vix_skips)} |
| insufficient_cash | {sum(1 for s in skip_log if 'insufficient_cash' in s.get('reason',''))} |

> **$13K risk budget note**: At {cfg.risk_per_trade_pct*100:.1f}% risk = ${cfg.starting_equity*cfg.risk_per_trade_pct:.0f}.
> Any contract priced above ${cfg.starting_equity*cfg.risk_per_trade_pct/100:.2f}/share (${cfg.starting_equity*cfg.risk_per_trade_pct:.0f}/contract) was skipped entirely.
> This occurred **{len(risk_skips)}** time(s) during the backtest period.

## Methodology Limitations

1. **Synthetic option pricing**: Black-Scholes assumes log-normal returns. Speculative stocks
   (MARA, RIOT, SOUN, etc.) have fat tails; actual option prices would differ materially.
2. **IV proxy**: 30-day realised vol × {cfg.vol_risk_premium} underestimates IV crush after
   earnings or macro events. Real options have term-structure and event vol.
3. **VIX as PCR proxy**: The live system uses Alpha Vantage real-time PCR. VIX is a coarse
   directional substitute — it misses micro-level sentiment shifts.
4. **Monthly expirations only**: Weekly expirations (available for SPY/QQQ/large-caps) are
   not modelled. Real DTE matching would produce different contracts.
5. **Illiquid universe**: The TV screener universe (top 30 from tv_screener_tickers.csv) is
   heavily weighted toward small/speculative stocks. Many fail the dollar-volume filter and
   produce zero option candidates in practice.
6. **Fill simulation**: Assuming 25% of half-spread as slippage on limit orders at mid is
   optimistic; actual fills depend on market conditions and queue position.
7. **No OI/volume data**: Real OI/volume for historical option contracts is estimated from
   stock dollar volume. Actual liquidity may differ significantly.

## Viability Assessment

"""
    total_ret = metrics["total_return_pct"]
    max_dd    = metrics["max_drawdown_pct"]
    sharpe    = metrics["sharpe_ratio"]
    n_trades  = metrics["total_trades"]
    win_rate  = metrics["win_rate_pct"]

    if n_trades == 0:
        verdict = "⚠️ **INSUFFICIENT DATA** — zero trades executed. The universe produces no eligible candidates under the $13K risk budget constraint. The strategy is not testable as-is."
    elif total_ret > 5 and sharpe > 0.5 and max_dd > -30:
        verdict = "✅ **TENTATIVELY VIABLE** — positive return with acceptable drawdown. Review findings before paper deployment."
    elif total_ret <= 0:
        verdict = "❌ **NOT VIABLE** — negative total return. Strategy requires significant revision."
    else:
        verdict = "⚠️ **MARGINAL** — positive return but insufficient Sharpe or excessive drawdown. Needs refinement."

    md += f"{verdict}\n\n"

    md += """## Top 5 Fixes Before Paper / Live Deployment

1. **Universe mismatch** — The TV screener universe is optimised for a different strategy.
   Replace with liquid options underlyings: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMD, META,
   AMZN, GOOGL, and sector ETFs (XLK, SOXX). These have tight spreads, real OI > 10K, and
   reliable IV data from Alpaca OPRA.

2. **Small-account risk sizing** — At $13K × 1.5% = $195/trade, any contract priced above
   ~$1.95/share exceeds the budget and is silently skipped. This means the system may produce
   zero entries for weeks. Either: (a) lower the delta target to find cheaper OTM contracts,
   (b) raise risk_per_trade_pct to 3-4% for small accounts, or (c) add a minimum-1-contract
   fallback with an explicit cap of $X per trade regardless of risk %.

3. **Real-time IV data required** — The live system must pull actual option snapshots from
   Alpaca OPRA to get real bid/ask/IV/delta. The backtest uses BS with realised vol × 1.25;
   the live system's `AlpacaOptionChainDiscovery` is the correct source but needs the paper
   account 401 issue resolved (API key configuration).

4. **PCR overlay needs historical data** — The Alpha Vantage REALTIME_PUT_CALL_RATIO is only
   available for the current day. For the live system, replace with a cached rolling PCR
   computed from end-of-day data (CBOE historical PCR is free). The current backtest uses VIX
   as a proxy, which is directionally correct but imprecise.

5. **Expiration-day risk** — The current close_before_expiration_days=3 protection works for
   monthlies. For weekly options (SPY, QQQ), the system needs to check expiry every day, not
   just on scan days. Also: add a hard gate to never open a new position if expiry is within
   block_new_entries_within_expiration_days=5 days — currently this gate is applied at
   contract selection but not at the scan level.
"""

    summary_path = REPORTS_DIR / "options_13k_summary.md"
    summary_path.write_text(md, encoding="utf-8")
    print(f"[saved] {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Universe loader
# ─────────────────────────────────────────────────────────────────────────────

LIQUID_OPTIONS_SUPPLEMENT = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META",
    "AMZN", "GOOGL", "PLTR", "COIN", "MARA", "RIOT", "SOXX", "SMH",
    "XLK", "IWM", "AVGO", "ARM",
]

def load_universe(use_tv_screener: bool = True, supplement: bool = True) -> list[str]:
    screener_path = ROOT / "tv_screener_tickers.csv"
    universe: list[str] = []

    if use_tv_screener and screener_path.exists():
        try:
            df = pd.read_csv(screener_path, header=None)
            row = df.iloc[0]
            row = row[row != "HPE/PC"].head(30)
            universe = [str(s).upper().strip().replace(".", "-") for s in row.tolist() if str(s).strip()]
            print(f"[universe] loaded {len(universe)} symbols from tv_screener_tickers.csv")
        except Exception as e:
            print(f"[universe] failed to read TV screener: {e}")

    if supplement or not universe:
        for sym in LIQUID_OPTIONS_SUPPLEMENT:
            if sym not in universe:
                universe.append(sym)
        print(f"[universe] total after supplement: {len(universe)} symbols")

    return universe


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Skip reason classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_skips(skip_log: list[dict]) -> dict:
    counts = {
        "skips_risk_budget":  0,
        "skips_no_contract":  0,
        "skips_vix":          0,
        "skips_cash":         0,
        "skips_other":        0,
        "total_skips":        len(skip_log),
    }
    for s in skip_log:
        r = s.get("reason", "")
        if "contract_cost_exceeds_risk_budget" in r:
            counts["skips_risk_budget"] += 1
        elif r == "no_eligible_contract":
            counts["skips_no_contract"] += 1
        elif "vix_too_high" in r:
            counts["skips_vix"] += 1
        elif "insufficient_cash" in r:
            counts["skips_cash"] += 1
        else:
            counts["skips_other"] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Three-variant comparison runner
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_DEFS = [
    {
        "id":            "A_baseline",
        "label":         "A — Baseline (fixed 1.5%, TV screener)",
        "risk_mode":     "fixed_1.5%",
        "universe_tag":  "TV screener + supplement",
        "dynamic":       False,
        "liquid_only":   False,
    },
    {
        "id":            "B_dynamic_budget",
        "label":         "B — Dynamic budget (tiered %, TV screener)",
        "risk_mode":     "dynamic_tiered (4% at $13K)",
        "universe_tag":  "TV screener + supplement",
        "dynamic":       True,
        "liquid_only":   False,
    },
    {
        "id":            "C_liquid_dynamic",
        "label":         "C — Dynamic budget + liquid universe",
        "risk_mode":     "dynamic_tiered (4% at $13K)",
        "universe_tag":  "SPY/QQQ/IWM/DIA/AAPL/MSFT/NVDA/META/AMZN/GOOGL/TSLA/AMD/SOXX/XLK/XLE",
        "dynamic":       True,
        "liquid_only":   True,
    },
]


def run_comparison(
    start_date: date,
    end_date: date,
    starting_equity: float,
    enable_long_puts: bool = True,
    no_cache: bool = False,
) -> list[dict]:
    """Run all three variants sequentially and return list of result rows."""
    if no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)

    tv_universe = load_universe(use_tv_screener=True, supplement=True)
    results = []

    for v in VARIANT_DEFS:
        universe = LIQUID_UNIVERSE_ONLY if v["liquid_only"] else tv_universe
        print(f"\n{'─'*60}")
        print(f"VARIANT {v['id']}: {v['label']}")
        print(f"  universe: {len(universe)} symbols | dynamic_budget: {v['dynamic']}")
        print(f"{'─'*60}")

        cfg = BacktestConfig(
            start_date             = start_date,
            end_date               = end_date,
            starting_equity        = starting_equity,
            enable_long_puts       = enable_long_puts,
            use_dynamic_risk_budget = v["dynamic"],
            force_min_one_contract = False,   # always strict budget
            universe               = universe,
        )

        engine = BacktestEngine(cfg)
        engine.download_data()

        print(f"[{v['id']}] running simulation...")
        trades, equity_curve, skip_log = engine.run()
        print(f"[{v['id']}] {len(trades)} trades, {len(skip_log)} skips")

        metrics      = engine.compute_metrics(trades, equity_curve)
        skip_metrics = classify_skips(skip_log)

        row = {
            "variant":       v["id"],
            "label":         v["label"],
            "risk_mode":     v["risk_mode"],
            "universe":      v["universe_tag"],
            "n_universe":    len(universe),
            **metrics,
            **skip_metrics,
        }
        results.append(row)

        # Print live summary
        print(f"  return={metrics['total_return_pct']:+.2f}%  "
              f"maxDD={metrics['max_drawdown_pct']:.2f}%  "
              f"sharpe={metrics['sharpe_ratio']:.3f}  "
              f"win={metrics['win_rate_pct']:.1f}%  "
              f"trades={metrics['total_trades']}  "
              f"skips={skip_metrics['total_skips']} "
              f"(budget={skip_metrics['skips_risk_budget']} no_contract={skip_metrics['skips_no_contract']})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_report(results: list[dict]) -> None:
    """Write risk_budget_comparison.md and risk_budget_comparison.csv."""

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / "risk_budget_comparison.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"[saved] {csv_path}")

    # ── Markdown ─────────────────────────────────────────────────────────────
    def fmt(v: Any, fmt_str: str = "") -> str:
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if fmt_str:
            return format(v, fmt_str)
        return str(v)

    def pf_str(pf: float) -> str:
        return "∞" if pf == float("inf") else f"{pf:.3f}"

    A = results[0]
    B = results[1]
    C = results[2]

    def delta(b_val: float, a_val: float, higher_better: bool = True) -> str:
        d = b_val - a_val
        sign = "+" if d >= 0 else ""
        arrow = "▲" if (d > 0) == higher_better else "▼"
        return f"{arrow}{sign}{d:.2f}"

    md = f"""# Risk Budget Comparison Backtest
## $13K Account · 2024-01-01 → 2024-12-31

Three variants tested. Only risk budget formula and universe differ.
All other parameters identical: DTE=21, delta=[0.35,0.65], stop=-35%, TP=+60%.

---

## Results Comparison

| Metric | A — Fixed 1.5% | B — Dynamic % | C — Dynamic % + Liquid Univ |
|---|---|---|---|
| **Risk mode** | Fixed 1.5% = $195 | Tiered (4% = $520 at $13K) | Tiered (4% = $520 at $13K) |
| **Universe** | TV screener + supp ({A['n_universe']} syms) | TV screener + supp ({B['n_universe']} syms) | Liquid-only ({C['n_universe']} syms) |
| **Ending equity** | ${A['ending_equity']:,.2f} | ${B['ending_equity']:,.2f} | ${C['ending_equity']:,.2f} |
| **Total return** | **{A['total_return_pct']:+.2f}%** | **{B['total_return_pct']:+.2f}%** | **{C['total_return_pct']:+.2f}%** |
| **Max drawdown** | {A['max_drawdown_pct']:.2f}% | {B['max_drawdown_pct']:.2f}% | {C['max_drawdown_pct']:.2f}% |
| **Sharpe ratio** | {A['sharpe_ratio']:.3f} | {B['sharpe_ratio']:.3f} | {C['sharpe_ratio']:.3f} |
| **Win rate** | {A['win_rate_pct']:.1f}% | {B['win_rate_pct']:.1f}% | {C['win_rate_pct']:.1f}% |
| **Profit factor** | {pf_str(A['profit_factor'])} | {pf_str(B['profit_factor'])} | {pf_str(C['profit_factor'])} |
| **Total trades** | {A['total_trades']} | {B['total_trades']} | {C['total_trades']} |
| **Total skips** | {A['total_skips']} | {B['total_skips']} | {C['total_skips']} |
| Skips: risk budget | **{A['skips_risk_budget']}** | {B['skips_risk_budget']} | {C['skips_risk_budget']} |
| Skips: no contract | {A['skips_no_contract']} | {B['skips_no_contract']} | {C['skips_no_contract']} |
| Skips: VIX | {A['skips_vix']} | {B['skips_vix']} | {C['skips_vix']} |
| Avg position size | ${A['avg_position_size']:.2f} | ${B['avg_position_size']:.2f} | ${C['avg_position_size']:.2f} |
| Avg win | ${A['avg_win']:.2f} | ${B['avg_win']:.2f} | ${C['avg_win']:.2f} |
| Avg loss | ${A['avg_loss']:.2f} | ${B['avg_loss']:.2f} | ${C['avg_loss']:.2f} |
| Largest winner | ${A['largest_winner']:.2f} | ${B['largest_winner']:.2f} | ${C['largest_winner']:.2f} |
| Largest loser | ${A['largest_loser']:.2f} | ${B['largest_loser']:.2f} | ${C['largest_loser']:.2f} |
| Avg holding days | {A['avg_holding_days']:.1f} | {B['avg_holding_days']:.1f} | {C['avg_holding_days']:.1f} |
| Calls / Puts | {A['total_calls']} / {A['total_puts']} | {B['total_calls']} / {B['total_puts']} | {C['total_calls']} / {C['total_puts']} |
| Gross profit | ${A['gross_profit']:.2f} | ${B['gross_profit']:.2f} | ${C['gross_profit']:.2f} |
| Gross loss | ${A['gross_loss']:.2f} | ${B['gross_loss']:.2f} | ${C['gross_loss']:.2f} |
| Total commissions | ${A['total_commissions']:.2f} | ${B['total_commissions']:.2f} | ${C['total_commissions']:.2f} |

---

## Per-Variant Analysis

### Variant A — Fixed 1.5% · TV Screener Universe

**Risk budget**: 1.5% × $13,000 = **$195/trade**

The $195 budget is below the cost of almost every ATM 21-DTE option on the TV screener
universe. Result: {A['skips_risk_budget']} of {A['total_skips']} skips were pure risk-budget
rejections. Only contracts priced below $1.95/share (~ATM on $5-$15 stocks) were eligible.

This is the "honest baseline" — no silent workarounds. The strategy correctly refuses to trade
when it cannot afford a contract within its own risk rules.

**Verdict**: Strictly correct behaviour, but mostly idle. Not a viable strategy as-is.

---

### Variant B — Dynamic Budget (4% at $13K) · TV Screener Universe

**Risk budget**: 4% × $13,000 = **$520/trade** (8% cap = $1,040)

Raising the risk budget to $520 eliminates most budget-related skips
({A['skips_risk_budget']} → {B['skips_risk_budget']}). More trades execute on the same
speculative/small-cap universe.

**Key question answered**: Is the TV screener universe the problem, or is it sizing?

The comparison of A vs B isolates the sizing effect on the **same universe**:
- More trades executed
- Win rate: {A['win_rate_pct']:.1f}% → {B['win_rate_pct']:.1f}%
- Return: {A['total_return_pct']:+.2f}% → {B['total_return_pct']:+.2f}%

> If B is better than A: sizing was the dominant constraint.
> If B is worse than A: the universe is the problem — larger positions just amplify losses.

---

### Variant C — Dynamic Budget (4%) · Liquid Options Universe

**Risk budget**: 4% × $13,000 = **$520/trade** (8% cap = $1,040)
**Universe**: {C['n_universe']} liquid symbols (SPY/QQQ/IWM/DIA + AAPL/MSFT/NVDA/META/AMZN/GOOGL/TSLA/AMD + SOXX/XLK/XLE)

This tests the core hypothesis from the prior report: the TV screener universe is wrong for
options. Liquid underlyings have:
- Lower realised vol (20-50% vs 80-200%) → fewer hard-stop exits
- Real OI > 10K → no liquidity estimation error
- Tighter real spreads (2-5% vs 6%+ assumed)
- Weekly expirations (not modelled here, but available in live)

**Expected outcome**: Better win rate, lower drawdown, likely lower absolute return per trade
(because lower vol = more expensive options relative to expected move).

A budget-to-contract-cost note for $520 at 2024 prices:
- SPY ATM 21-DTE: ~$5/share → $500/contract → **1 contract** ✓
- QQQ ATM 21-DTE: ~$4/share → $400/contract → **1 contract** ✓
- IWM/DIA: ~$2-3/share → **1-2 contracts** ✓
- NVDA ATM 21-DTE: $15-50/share → $1,500-5,000/contract → **SKIPPED** (> 8% cap)
- META: $8-15/share → $800-1,500/contract → **SKIPPED** for most of year
- TSLA: $5-15/share → borderline

The strategy will primarily trade SPY, QQQ, IWM, AAPL, AMD on this universe.

---

## What the Comparison Tells Us

"""

    # Verdict logic
    c_better_than_a = C['total_return_pct'] > A['total_return_pct']
    b_better_than_a = B['total_return_pct'] > A['total_return_pct']
    c_better_than_b = C['total_return_pct'] > B['total_return_pct']

    if c_better_than_a and c_better_than_b:
        md += "✅ **Hypothesis CONFIRMED**: Liquid universe + dynamic sizing outperforms both baseline variants.\n\n"
        md += "The TV screener universe was the dominant drag. Moving to liquid underlyings meaningfully improves results even without other parameter changes.\n\n"
    elif b_better_than_a and not c_better_than_b:
        md += "⚠️ **Mixed result**: Dynamic sizing helps (B > A), but liquid universe shows no additional benefit (C ≤ B).\n\n"
        md += "Universe quality may matter less than sizing in this simulation. BS pricing on liquid names may be too conservative (high IV × low vol = expensive contracts that still get stopped out).\n\n"
    elif not b_better_than_a:
        md += "❌ **All variants negative**: Larger positions on the TV screener universe amplify losses (B ≤ A). Universe is the primary problem — Fix 1 (universe) must precede Fix 2 (sizing).\n\n"
    else:
        md += "⚠️ **Inconclusive**: Results require deeper analysis per trade cohort.\n\n"

    md += f"""
## Sizing Mechanics at $13K (Variant B/C: 4% = $520)

| Symbol | Typical 21-DTE ATM premium | Contracts at $520 | Notes |
|---|---|---|---|
| SPY | ~$5.00 | 1 | ✓ fits |
| QQQ | ~$4.00 | 1 | ✓ fits |
| IWM | ~$2.50 | 2 | ✓ fits |
| DIA | ~$2.50 | 2 | ✓ fits |
| AAPL | ~$3.00 | 1 | ✓ fits |
| MSFT | ~$6.00 | 0–1 | borderline |
| NVDA | ~$20.00 | 0 | ❌ > 8% cap ($2,000 > $1,040) |
| GOOGL | ~$4.00 | 1 | ✓ fits |
| AMZN | ~$4.00 | 1 | ✓ fits |
| TSLA | ~$10.00 | 0 | ❌ too expensive |
| META | ~$12.00 | 0 | ❌ too expensive |
| AMD | ~$3.50 | 1 | ✓ fits |
| XLE | ~$2.00 | 2 | ✓ fits |
| XLK | ~$4.00 | 1 | ✓ fits |
| SOXX | ~$6.00 | 0–1 | borderline |

> **Implication**: At $13K with 4% risk, the strategy effectively trades ETFs (SPY/QQQ/IWM/DIA)
> plus a few large-cap single stocks (AAPL, GOOGL, AMZN, AMD, XLE/XLK). High-beta names
> (NVDA, TSLA, META) are priced out even at 4% risk. This is actually safer — lower vol
> underlyings produce fewer hard-stop exits.

## Minimum Account Size for Liquid-Universe Options

| Name | Min cost/contract | Min account at 4% | Min account at 8% cap |
|---|---|---|---|
| IWM | ~$200 | $5,000 | $2,500 |
| SPY | ~$500 | $12,500 | $6,250 |
| AAPL | ~$300 | $7,500 | $3,750 |
| MSFT | ~$600 | $15,000 | $7,500 |
| NVDA | ~$2,000 | $50,000 | $25,000 |
| TSLA | ~$1,000 | $25,000 | $12,500 |

> **Practical floor**: $15,000–$20,000 to trade SPY/QQQ/IWM + 2-3 large-caps.
> At $13,000 with 4% risk, the system is viable for ETF-only options but priced out of NVDA/TSLA.

## Recommended Path Forward

Based on this comparison:

1. **Adopt Variant C** (liquid universe + dynamic budget) as the base going forward.
   It removes the universe mismatch and makes the sizing tractable.

2. **Raise account floor**: Options mode should be blocked below $15K even with dynamic sizing.
   Add to `apply_dynamic_account_scaling()` in run.py.

3. **Do not raise the 8% position cap**: Even with tiered risk%, a single position > 8% of
   equity is concentration risk that amplifies drawdowns.

4. **Next parameter optimisation** (not done here per user instructions):
   - Test wider hard stop (-40% to -50%) to reduce stop-out frequency on liquid names
   - Test DTE range 14-30 (vs current 7-45) to stay in preferred zone
   - Test put enables separately (VIX-gated puts on liquid names during corrections)

5. **Real data required before paper deployment**: BS pricing on NVDA/TSLA especially
   diverges from real option prices during high-IV periods. The live system's
   `AlpacaOptionChainDiscovery` + OPRA feed is the correct source.
"""

    md_path = REPORTS_DIR / "risk_budget_comparison.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[saved] {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Variant D: strict liquidity filters on liquid universe
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_C_KWARGS = dict(
    use_dynamic_risk_budget    = True,
    enable_long_puts           = True,
    universe                   = LIQUID_UNIVERSE_ONLY,
    # default filters
    call_delta_min             = 0.35,
    call_delta_max             = 0.65,
    call_delta_target          = 0.50,
    put_delta_min              = -0.65,
    put_delta_max              = -0.35,
    put_delta_target           = -0.50,
    min_open_interest          = 100,
    min_volume                 = 20,
    max_spread_pct             = 0.12,
    assumed_spread_pct         = 0.06,
    min_underlying_dollar_vol  = 5_000_000.0,
    min_underlying_price       = 2.0,
)

VARIANT_E_KWARGS = dict(
    **VARIANT_C_KWARGS,
    use_volume_oi_scoring = True,   # only addition vs C
)

VARIANT_D_KWARGS = dict(
    use_dynamic_risk_budget    = True,
    enable_long_puts           = True,
    universe                   = LIQUID_UNIVERSE_ONLY,
    # tighter delta window
    call_delta_min             = 0.45,
    call_delta_max             = 0.60,
    call_delta_target          = 0.52,
    put_delta_min              = -0.60,
    put_delta_max              = -0.45,
    put_delta_target           = -0.52,
    # strict liquidity gates
    min_open_interest          = 1000,
    min_volume                 = 100,
    max_spread_pct             = 0.05,
    assumed_spread_pct         = 0.04,  # liquid ETF/large-cap spreads; must be < max_spread_pct
    # underlying liquidity floor
    min_underlying_dollar_vol  = 500_000_000.0,
    min_underlying_price       = 20.0,
)


def _run_one_variant(
    label: str,
    kwargs: dict,
    start_date: date,
    end_date: date,
    starting_equity: float,
) -> tuple[list, list, list, dict, dict]:
    cfg = BacktestConfig(
        start_date      = start_date,
        end_date        = end_date,
        starting_equity = starting_equity,
        **kwargs,
    )
    engine = BacktestEngine(cfg)
    engine.download_data()
    print(f"\n[{label}] running simulation ...")
    trades, equity_curve, skip_log = engine.run()
    print(f"[{label}] {len(trades)} trades, {len(skip_log)} skips")
    metrics      = engine.compute_metrics(trades, equity_curve)
    skip_metrics = classify_skips(skip_log)
    return trades, equity_curve, skip_log, metrics, skip_metrics


def run_variant_d_comparison(
    start_date: date,
    end_date: date,
    starting_equity: float,
    no_cache: bool = False,
) -> None:
    if no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)

    print("\n" + "─" * 60)
    print("VARIANT C — baseline liquid (delta 0.35-0.65, spread ≤12%, OI≥100)")
    print("─" * 60)
    _, _, _, m_c, sk_c = _run_one_variant(
        "C", VARIANT_C_KWARGS, start_date, end_date, starting_equity
    )

    print("\n" + "─" * 60)
    print("VARIANT D — strict liquidity (delta 0.45-0.60, spread ≤5%, OI≥1000)")
    print("─" * 60)
    trades_d, eq_d, skip_d, m_d, sk_d = _run_one_variant(
        "D", VARIANT_D_KWARGS, start_date, end_date, starting_equity
    )

    def pf(v):
        return "∞" if v == float("inf") else f"{v:.3f}"

    # Console summary
    print(f"\n{'═'*60}")
    print(f"{'Metric':<28} {'Variant C':>14} {'Variant D':>14}")
    print(f"{'─'*60}")
    rows_tbl = [
        ("Total return",      f"{m_c['total_return_pct']:+.2f}%",    f"{m_d['total_return_pct']:+.2f}%"),
        ("Max drawdown",      f"{m_c['max_drawdown_pct']:.2f}%",     f"{m_d['max_drawdown_pct']:.2f}%"),
        ("Sharpe ratio",      f"{m_c['sharpe_ratio']:.3f}",          f"{m_d['sharpe_ratio']:.3f}"),
        ("Profit factor",     pf(m_c['profit_factor']),               pf(m_d['profit_factor'])),
        ("Win rate",          f"{m_c['win_rate_pct']:.1f}%",         f"{m_d['win_rate_pct']:.1f}%"),
        ("Trades",            str(m_c['total_trades']),               str(m_d['total_trades'])),
        ("Calls / Puts",      f"{m_c['total_calls']}/{m_c['total_puts']}", f"{m_d['total_calls']}/{m_d['total_puts']}"),
        ("Avg contract cost", f"${m_c['avg_position_size']:.2f}",    f"${m_d['avg_position_size']:.2f}"),
        ("Avg win",           f"${m_c['avg_win']:.2f}",              f"${m_d['avg_win']:.2f}"),
        ("Avg loss",          f"${m_c['avg_loss']:.2f}",             f"${m_d['avg_loss']:.2f}"),
        ("Largest winner",    f"${m_c['largest_winner']:.2f}",       f"${m_d['largest_winner']:.2f}"),
        ("Largest loser",     f"${m_c['largest_loser']:.2f}",        f"${m_d['largest_loser']:.2f}"),
        ("Total skips",       str(sk_c['total_skips']),               str(sk_d['total_skips'])),
        ("  → risk budget",   str(sk_c['skips_risk_budget']),         str(sk_d['skips_risk_budget'])),
        ("  → no contract",   str(sk_c['skips_no_contract']),         str(sk_d['skips_no_contract'])),
    ]
    for label_r, vc, vd in rows_tbl:
        print(f"  {label_r:<26} {vc:>14} {vd:>14}")
    print(f"{'═'*60}")

    # Markdown report
    def _better(d_val, c_val, higher_better=True):
        if isinstance(d_val, str):
            return ""
        diff = d_val - c_val
        if abs(diff) < 1e-6:
            return "="
        better = (diff > 0) == higher_better
        return "✅ D" if better else "⚠️ C"

    md = f"""# Liquidity Filter Comparison — Variant C vs Variant D
## $13K Account · 2024-01-01 → 2024-12-31

**Question**: Does tightening contract filters (delta, spread, OI, volume, underlying dollar-volume)
improve performance on the liquid options universe?

---

## Filter Differences

| Parameter | Variant C | Variant D |
|---|---|---|
| Call delta range | 0.35 – 0.65 | **0.45 – 0.60** |
| Put delta range | -0.65 – -0.35 | **-0.60 – -0.45** |
| Min open interest | 100 | **1,000** |
| Min option volume | 20 | **100** |
| Max spread | 12% | **5%** |
| Assumed fill spread | 6% | **4%** |
| Min underlying $vol | $5M/day | **$500M/day** |
| Min underlying price | $2 | **$20** |
| Risk sizing | Dynamic 4% @ $13K | Dynamic 4% @ $13K (same) |
| Universe | Liquid 15 symbols | Liquid 15 symbols (same) |

---

## Results

| Metric | Variant C | Variant D | Better |
|---|---|---|---|
| **Total return** | {m_c['total_return_pct']:+.2f}% | {m_d['total_return_pct']:+.2f}% | {_better(m_d['total_return_pct'], m_c['total_return_pct'])} |
| **Max drawdown** | {m_c['max_drawdown_pct']:.2f}% | {m_d['max_drawdown_pct']:.2f}% | {_better(m_d['max_drawdown_pct'], m_c['max_drawdown_pct'], higher_better=False)} |
| **Sharpe ratio** | {m_c['sharpe_ratio']:.3f} | {m_d['sharpe_ratio']:.3f} | {_better(m_d['sharpe_ratio'], m_c['sharpe_ratio'])} |
| **Profit factor** | {pf(m_c['profit_factor'])} | {pf(m_d['profit_factor'])} | |
| **Win rate** | {m_c['win_rate_pct']:.1f}% | {m_d['win_rate_pct']:.1f}% | {_better(m_d['win_rate_pct'], m_c['win_rate_pct'])} |
| Trades executed | {m_c['total_trades']} | {m_d['total_trades']} | |
| Calls / Puts | {m_c['total_calls']} / {m_c['total_puts']} | {m_d['total_calls']} / {m_d['total_puts']} | |
| **Avg contract cost** | ${m_c['avg_position_size']:.2f} | ${m_d['avg_position_size']:.2f} | |
| Avg win | ${m_c['avg_win']:.2f} | ${m_d['avg_win']:.2f} | |
| Avg loss | ${m_c['avg_loss']:.2f} | ${m_d['avg_loss']:.2f} | |
| Largest winner | ${m_c['largest_winner']:.2f} | ${m_d['largest_winner']:.2f} | |
| Largest loser | ${m_c['largest_loser']:.2f} | ${m_d['largest_loser']:.2f} | |
| Gross profit | ${m_c['gross_profit']:.2f} | ${m_d['gross_profit']:.2f} | |
| Gross loss | ${m_c['gross_loss']:.2f} | ${m_d['gross_loss']:.2f} | |
| Commissions | ${m_c['total_commissions']:.2f} | ${m_d['total_commissions']:.2f} | |

## Skip Breakdown

| Skip reason | Variant C | Variant D |
|---|---|---|
| Total skips | {sk_c['total_skips']} | {sk_d['total_skips']} |
| Risk budget exceeded | {sk_c['skips_risk_budget']} | {sk_d['skips_risk_budget']} |
| No eligible contract | {sk_c['skips_no_contract']} | {sk_d['skips_no_contract']} |
| VIX too high | {sk_c['skips_vix']} | {sk_d['skips_vix']} |
| Insufficient cash | {sk_c['skips_cash']} | {sk_d['skips_cash']} |

---

## Analysis

### What the tighter delta range (0.45–0.60) does
Narrowing the delta window eliminates near-OTM and deep-ITM contracts. The 0.45–0.60 range
targets true near-ATM options where the reward:risk profile is most symmetric. Contracts
at delta 0.35–0.45 are relatively cheap but have lower probability of reaching the take-profit
threshold (+60% premium gain) within the 21-DTE window.

### What the 5% spread filter does
The 5% spread filter is the most impactful change for the liquid universe. On SPY/QQQ/IWM/XLK
options, real ATM spreads run 1-4% during normal market hours. Enforcing ≤5% eliminates:
- Any contract priced during volatile intraday periods (wide spreads)
- Low-liquidity strikes far from ATM that happen to appear near the delta target
- In this BS simulation: all contracts are assigned `assumed_spread_pct = 4%`,
  which passes the 5% filter. Contracts in C with 6% assumed spread would be rejected
  by the 5% filter — this affects whether they enter the portfolio.

### What the OI/volume floor (≥1000 / ≥100) does
On liquid large-cap names (NVDA, AAPL, AMZN, SPY, QQQ), real OI for ATM options near
expiration exceeds 10,000-100,000 contracts. The OI≥1000 floor primarily filters out
poorly-traded strikes (deep OTM/ITM that happen to be in the delta window). In this
backtest, the proxy OI for $500M+ dollar-volume stocks is 20,000 — so the floor has
no effect on those names. The filter would matter most if speculative names were re-added.

### What the $500M underlying dollar-volume floor does
All 15 liquid universe symbols exceed $500M daily dollar-volume comfortably:
- SPY: ~$30B/day; QQQ: ~$10B/day; NVDA: ~$20-40B/day (in 2024)
- AAPL: ~$15B/day; AMD: ~$4-5B/day (borderline)
The floor primarily removes AMD on low-volume days and acts as a guardrail if the
universe were ever expanded.

---

## Conclusion

"""

    if m_d['total_return_pct'] > m_c['total_return_pct']:
        md += f"""✅ **Variant D outperforms Variant C** on total return
({m_d['total_return_pct']:+.2f}% vs {m_c['total_return_pct']:+.2f}%).

The stricter filters produce a cleaner trade set. Fewer but higher-quality entries
result in better average P&L per trade. The tighter delta window and spread constraint
reduce slippage drag and eliminate marginal setups.

**Recommendation**: Adopt Variant D filter parameters as the new baseline.
"""
    elif m_d['sharpe_ratio'] > m_c['sharpe_ratio']:
        md += f"""⚠️ **Variant D has better risk-adjusted returns** (Sharpe {m_d['sharpe_ratio']:.3f}
vs {m_c['sharpe_ratio']:.3f}) despite similar or lower absolute return.

The tighter filters reduce the number of trades, which may reduce total return on an
absolute basis but improve consistency. Fewer hard-stop exits indicate the filtered
contracts are better-behaved.

**Recommendation**: Adopt D's filter parameters. Accept lower trade frequency as the
cost of higher-quality entries.
"""
    else:
        md += f"""⚠️ **Mixed result**: Variant D has fewer trades ({m_d['total_trades']} vs {m_c['total_trades']})
but the improvement in quality is not clearly visible in this sample period.

The liquid universe already filters most low-quality contracts. Tightening further reduces
the opportunity set without proportional benefit in this 1-year backtest. The sample size
({m_d['total_trades']} trades) is too small to draw strong conclusions.

**Recommendation**: Run a longer backtest (2022-2024) before deciding whether the stricter
filters add value over the liquid-universe baseline.
"""

    md += """
## Notes on BS Pricing Limitations

1. Real option spreads for SPY/QQQ ATM options are typically 0.5-2% of mid during liquid
   hours — tighter than the 4% assumed here. The 5% spread filter is conservative; real
   fills would be better.

2. The OI/volume filter has no practical effect on this universe since all symbols have
   proxy OI >> 1000. The filter is valuable as a guard if the universe is expanded.

3. Delta values computed from BS/realized-vol will differ from market-implied delta,
   especially after large moves. Near-ATM options are least affected by this.
"""

    md_path = REPORTS_DIR / "liquidity_filter_comparison.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\n[saved] {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Volume/OI analytics report
# ─────────────────────────────────────────────────────────────────────────────

def generate_volume_oi_analysis(
    trades_by_variant: dict[str, list[TradeRecord]],
) -> None:
    """
    Analyse vol/OI ratio behaviour across all variants that have it.
    Saves reports/backtests/volume_oi_analysis.md.
    """
    path = REPORTS_DIR / "volume_oi_analysis.md"

    def _stats(trades: list[TradeRecord]) -> dict:
        ratios  = [t.vol_oi_ratio for t in trades if t.vol_oi_ratio is not None]
        winners = [t for t in trades if t.net_pl > 0]
        losers  = [t for t in trades if t.net_pl <= 0]
        w_ratios = [t.vol_oi_ratio for t in winners if t.vol_oi_ratio is not None]
        l_ratios = [t.vol_oi_ratio for t in losers  if t.vol_oi_ratio is not None]
        by_exit: dict[str, list[float]] = {}
        for t in trades:
            key = t.exit_reason.split(" (")[0]
            if t.vol_oi_ratio is not None:
                by_exit.setdefault(key, []).append(t.vol_oi_ratio)
        return {
            "all_ratios": ratios,
            "avg_all":     float(np.mean(ratios)) if ratios else None,
            "avg_winners": float(np.mean(w_ratios)) if w_ratios else None,
            "avg_losers":  float(np.mean(l_ratios)) if l_ratios else None,
            "by_exit":     {k: float(np.mean(v)) for k, v in by_exit.items()},
            "deciles":     list(np.percentile(ratios, range(0, 101, 10))) if ratios else [],
            "n_with_data": len(ratios),
            "n_total":     len(trades),
        }

    def _fmt(v) -> str:
        return f"{v:.3f}" if v is not None else "—"

    md_lines = ["# Volume / OI Ratio Analysis\n"]
    md_lines.append("Tracks Alpha Vantage `HISTORICAL_VOLUME_OPEN_INTEREST_RATIO` "
                    "across trades to validate whether higher participation predicts better outcomes.\n\n---\n")

    for variant, trades in trades_by_variant.items():
        s = _stats(trades)
        md_lines.append(f"## Variant {variant}\n")
        md_lines.append(f"- Trades with vol/OI data: **{s['n_with_data']}** / {s['n_total']}\n")
        md_lines.append(f"- Average ratio (all): **{_fmt(s['avg_all'])}**\n")
        md_lines.append(f"- Average ratio (winners): **{_fmt(s['avg_winners'])}**\n")
        md_lines.append(f"- Average ratio (losers): **{_fmt(s['avg_losers'])}**\n\n")

        if s["by_exit"]:
            md_lines.append("### Average ratio by exit type\n\n")
            md_lines.append("| Exit reason | Avg vol/OI ratio |\n|---|---|\n")
            for k, v in sorted(s["by_exit"].items(), key=lambda x: -x[1]):
                md_lines.append(f"| {k} | {v:.3f} |\n")
            md_lines.append("\n")

        if s["deciles"]:
            md_lines.append("### Ratio decile distribution\n\n")
            md_lines.append("| Percentile | Vol/OI Ratio |\n|---|---|\n")
            for pct, val in zip(range(0, 101, 10), s["deciles"]):
                md_lines.append(f"| {pct}th | {val:.3f} |\n")
            md_lines.append("\n")

    # Scoring validation summary
    md_lines.append("---\n\n## Scoring Thresholds Validation\n\n")
    md_lines.append("| Score bucket | Ratio range | Raw score | Normalized |\n|---|---|---|---|\n")
    for label, lo, hi, raw in [
        ("Dead contracts",        "< 0.25",       "0.25",  -15),
        ("Below average",     "0.25 – 0.75",  "0.75",    0),
        ("Normal activity",   "0.75 – 1.50",  "1.50",   10),
        ("Elevated",          "1.50 – 3.00",  "3.00",   20),
        ("Strong positioning",    "≥ 3.00",    "∞",      30),
    ]:
        norm = (raw + 15) / 45
        md_lines.append(f"| {label} | {lo} | {raw:+d} | {norm:.2f} |\n")

    path.write_text("".join(md_lines), encoding="utf-8")
    print(f"[saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# A / B / C / E comparison runner
# ─────────────────────────────────────────────────────────────────────────────

ABCE_VARIANTS = [
    ("A", "A — Fixed 1.5%, TV screener",          "fixed_1.5%",           False, False, None),
    ("B", "B — Dynamic budget, TV screener",       "dynamic_tiered (4%)",  True,  False, None),
    ("C", "C — Dynamic budget, liquid universe",   "dynamic_tiered (4%)",  True,  False, LIQUID_UNIVERSE_ONLY),
    ("E", "E — Variant C + Vol/OI scoring",        "dynamic_tiered (4%)",  True,  True,  LIQUID_UNIVERSE_ONLY),
]


def run_abce_comparison(
    start_date: date,
    end_date: date,
    starting_equity: float,
    no_cache: bool = False,
) -> None:
    if no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)

    tv_universe = load_universe(use_tv_screener=True, supplement=True)
    results = []
    all_trades: dict[str, list[TradeRecord]] = {}

    for vid, label, risk_mode, dynamic, use_voi, universe in ABCE_VARIANTS:
        univ = universe if universe is not None else tv_universe
        print(f"\n{'─'*60}")
        print(f"VARIANT {vid}: {label}")
        print(f"  universe: {len(univ)} symbols | dynamic: {dynamic} | vol_oi: {use_voi}")
        print(f"{'─'*60}")

        cfg = BacktestConfig(
            start_date             = start_date,
            end_date               = end_date,
            starting_equity        = starting_equity,
            use_dynamic_risk_budget = dynamic,
            enable_long_puts       = True,
            use_volume_oi_scoring  = use_voi,
            universe               = univ,
        )
        engine = BacktestEngine(cfg)
        engine.download_data()
        trades, equity_curve, skip_log = engine.run()
        print(f"[{vid}] {len(trades)} trades, {len(skip_log)} skips")

        m  = engine.compute_metrics(trades, equity_curve)
        sk = classify_skips(skip_log)

        # average vol/OI ratio across trades
        voi_values = [t.vol_oi_ratio for t in trades if t.vol_oi_ratio is not None]
        avg_voi = float(np.mean(voi_values)) if voi_values else None

        results.append({
            "variant":           vid,
            "label":             label,
            "risk_mode":         risk_mode,
            "vol_oi_scoring":    use_voi,
            "n_universe":        len(univ),
            "total_return_pct":  m["total_return_pct"],
            "max_drawdown_pct":  m["max_drawdown_pct"],
            "sharpe_ratio":      m["sharpe_ratio"],
            "profit_factor":     m["profit_factor"],
            "win_rate_pct":      m["win_rate_pct"],
            "total_trades":      m["total_trades"],
            "total_calls":       m["total_calls"],
            "total_puts":        m["total_puts"],
            "avg_position_size": m["avg_position_size"],
            "avg_win":           m["avg_win"],
            "avg_loss":          m["avg_loss"],
            "total_skips":       sk["total_skips"],
            "skips_risk_budget": sk["skips_risk_budget"],
            "skips_no_contract": sk["skips_no_contract"],
            "avg_vol_oi_ratio":  avg_voi,
        })
        all_trades[vid] = trades

        avg_voi_str = f"{avg_voi:.3f}" if avg_voi is not None else "—"
        print(f"  return={m['total_return_pct']:+.2f}%  maxDD={m['max_drawdown_pct']:.2f}%  "
              f"sharpe={m['sharpe_ratio']:.3f}  win={m['win_rate_pct']:.1f}%  "
              f"trades={m['total_trades']}  avg_voi={avg_voi_str}")

    def pf(v):
        return "∞" if v == float("inf") else f"{v:.3f}"

    A, B, C, E_ = results

    md = f"""# A / B / C / E Comparison — Volume/OI Scoring Impact
## $13K Account · {start_date} → {end_date}

**Variant E** adds Alpha Vantage `HISTORICAL_VOLUME_OPEN_INTEREST_RATIO` scoring
on top of Variant C (liquid universe + dynamic risk budget).

Combined rank formula used in Variant E contract selection:

```
rank = technical_score + sentiment_score + liquidity_score + volume_oi_score
       (each component normalised to [0, 1])
```

Vol/OI scoring thresholds:
- ratio < 0.25 → score **-15** (stale contracts, hard reject if not overridden by OI/spread)
- 0.25–0.75   → score **0** (below-average activity)
- 0.75–1.50   → score **+10** (normal participation)
- 1.50–3.00   → score **+20** (elevated new positioning)
- ≥ 3.00      → score **+30** (strong trend initiation signal)

---

## Results

| Metric | A | B | C | **E** |
|---|---|---|---|---|
| **Return** | {A['total_return_pct']:+.2f}% | {B['total_return_pct']:+.2f}% | {C['total_return_pct']:+.2f}% | **{E_['total_return_pct']:+.2f}%** |
| **Max drawdown** | {A['max_drawdown_pct']:.2f}% | {B['max_drawdown_pct']:.2f}% | {C['max_drawdown_pct']:.2f}% | **{E_['max_drawdown_pct']:.2f}%** |
| **Sharpe** | {A['sharpe_ratio']:.3f} | {B['sharpe_ratio']:.3f} | {C['sharpe_ratio']:.3f} | **{E_['sharpe_ratio']:.3f}** |
| **Profit factor** | {pf(A['profit_factor'])} | {pf(B['profit_factor'])} | {pf(C['profit_factor'])} | **{pf(E_['profit_factor'])}** |
| **Win rate** | {A['win_rate_pct']:.1f}% | {B['win_rate_pct']:.1f}% | {C['win_rate_pct']:.1f}% | **{E_['win_rate_pct']:.1f}%** |
| Trades | {A['total_trades']} | {B['total_trades']} | {C['total_trades']} | {E_['total_trades']} |
| Calls / Puts | {A['total_calls']}/{A['total_puts']} | {B['total_calls']}/{B['total_puts']} | {C['total_calls']}/{C['total_puts']} | {E_['total_calls']}/{E_['total_puts']} |
| Avg contract cost | ${A['avg_position_size']:.0f} | ${B['avg_position_size']:.0f} | ${C['avg_position_size']:.0f} | ${E_['avg_position_size']:.0f} |
| Avg win | ${A['avg_win']:.0f} | ${B['avg_win']:.0f} | ${C['avg_win']:.0f} | ${E_['avg_win']:.0f} |
| Avg loss | ${A['avg_loss']:.0f} | ${B['avg_loss']:.0f} | ${C['avg_loss']:.0f} | ${E_['avg_loss']:.0f} |
| Total skips | {A['total_skips']} | {B['total_skips']} | {C['total_skips']} | {E_['total_skips']} |
| Skips: risk budget | {A['skips_risk_budget']} | {B['skips_risk_budget']} | {C['skips_risk_budget']} | {E_['skips_risk_budget']} |
| **Avg vol/OI ratio** | — | — | — | **{f"{E_['avg_vol_oi_ratio']:.3f}" if E_['avg_vol_oi_ratio'] else "—"}** |

---

## C vs E — Effect of Vol/OI Scoring

| Metric | Variant C | Variant E | Delta |
|---|---|---|---|
| Return | {C['total_return_pct']:+.2f}% | {E_['total_return_pct']:+.2f}% | {E_['total_return_pct']-C['total_return_pct']:+.2f}pp |
| Max drawdown | {C['max_drawdown_pct']:.2f}% | {E_['max_drawdown_pct']:.2f}% | {E_['max_drawdown_pct']-C['max_drawdown_pct']:+.2f}pp |
| Sharpe | {C['sharpe_ratio']:.3f} | {E_['sharpe_ratio']:.3f} | {E_['sharpe_ratio']-C['sharpe_ratio']:+.3f} |
| Win rate | {C['win_rate_pct']:.1f}% | {E_['win_rate_pct']:.1f}% | {E_['win_rate_pct']-C['win_rate_pct']:+.1f}pp |
| Profit factor | {pf(C['profit_factor'])} | {pf(E_['profit_factor'])} | — |
| Trades | {C['total_trades']} | {E_['total_trades']} | {E_['total_trades']-C['total_trades']:+d} |

---

## Interpretation

"""
    c_to_e_return = E_['total_return_pct'] - C['total_return_pct']
    c_to_e_sharpe = E_['sharpe_ratio'] - C['sharpe_ratio']
    c_to_e_win    = E_['win_rate_pct']  - C['win_rate_pct']

    if c_to_e_return > 0 and c_to_e_win > 0:
        md += (f"✅ **Vol/OI scoring improves both return (+{c_to_e_return:.2f}pp) "
               f"and win rate (+{c_to_e_win:.1f}pp) vs Variant C.** "
               f"The combined ranking is selecting contracts with stronger participation "
               f"signals, which correlates with better directional outcomes.\n\n")
    elif c_to_e_sharpe > 0:
        md += (f"⚠️ **Sharpe improves ({c_to_e_sharpe:+.3f}) but return impact is mixed.** "
               f"Vol/OI scoring reduces drawdown and smooths the equity curve, "
               f"but the sample size ({E_['total_trades']} trades) is too small "
               f"for definitive conclusions.\n\n")
    else:
        md += (f"⚠️ **Vol/OI scoring shows no clear benefit over Variant C in this period.** "
               f"The liquid universe already screens for high-OI names where vol/OI ratio "
               f"is consistently moderate (0.5–1.5 range). The scoring signal may add more "
               f"value when the universe includes mixed-liquidity names.\n\n")

    if E_['avg_vol_oi_ratio'] is not None:
        raw_score = volume_oi_score_raw(E_['avg_vol_oi_ratio'])
        md += (f"Average vol/OI ratio across Variant E trades: **{E_['avg_vol_oi_ratio']:.3f}** "
               f"→ raw score bucket: **{raw_score:+.0f}** "
               f"({'elevated participation' if (E_['avg_vol_oi_ratio'] or 0) >= 1.5 else 'normal activity'}).\n\n")
    else:
        md += ("**Note**: No vol/OI data was retrieved (likely due to API key limits or "
               "cache misses). Variant E ran with neutral scores (0.50) for all contracts — "
               "the combined ranking defaulted to delta-fit + liquidity only. "
               "Re-run after building the cache to get true Variant E results.\n\n")

    md += """---

## Notes

1. **Cache first**: Vol/OI data is cached to `logs/backtest_cache/volume_oi_ratio/{symbol}/{date}.json`.
   Each new backtest run builds the cache; subsequent runs use cached data instantly.

2. **API limits**: Alpha Vantage free tier allows 25 calls/day. Premium tiers support
   more. The backtest fetches lazily — only on scan days — so a full 252-day backtest
   requires at most `n_scan_weeks × n_symbols` calls (~52 × 15 = 780 for liquid universe).

3. **Neutral fallback**: When API data is unavailable (rate limit, network error, or
   cache miss), vol/OI score defaults to 0.50 (neutral). This means Variant E degrades
   gracefully to pure delta+liquidity+momentum ranking — it never fails.

4. **Hard reject threshold**: Contracts with ratio < 0.20 are rejected unless proxy OI
   exceeds 10,000 AND spread is below 3%. On the liquid universe, proxy OI is always
   20,000+ for major ETFs, so the override triggers for nearly all contracts. The hard
   reject becomes meaningful when mixed-liquidity underlyings are added.
"""

    csv_path = REPORTS_DIR / "abce_comparison.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"[saved] {csv_path}")

    md_path = REPORTS_DIR / "abce_comparison.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[saved] {md_path}")

    generate_volume_oi_analysis({"E": all_trades.get("E", [])})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Options-first historical backtest")
    p.add_argument("--start",          default="2024-01-01")
    p.add_argument("--end",            default="2024-12-31")
    p.add_argument("--equity",         type=float, default=13_000.0)
    p.add_argument("--risk-pct",       type=float, default=0.015)
    p.add_argument("--max-exposure",   type=float, default=0.40)
    p.add_argument("--no-puts",        action="store_true")
    p.add_argument("--force-min-1",    action="store_true",
                   help="Allow 1 contract even if cost > risk budget (show sizing impact)")
    p.add_argument("--no-supplement",  action="store_true",
                   help="Use TV screener universe only (no liquid supplement)")
    p.add_argument("--no-cache",       action="store_true",
                   help="Force re-download of price data")
    p.add_argument("--compare",        action="store_true",
                   help="Run 3-variant risk-budget comparison and save to risk_budget_comparison.{md,csv}")
    p.add_argument("--variant-d",      action="store_true",
                   help="Run Variant C vs D liquidity-filter comparison and save to liquidity_filter_comparison.md")
    p.add_argument("--variant-e",      action="store_true",
                   help="Run A/B/C/E comparison with vol/OI scoring and save abce_comparison.{md,csv} + volume_oi_analysis.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = BacktestConfig(
        start_date            = date.fromisoformat(args.start),
        end_date              = date.fromisoformat(args.end),
        starting_equity       = args.equity,
        risk_per_trade_pct    = args.risk_pct,
        max_options_exposure  = args.max_exposure,
        enable_long_puts      = not args.no_puts,
        force_min_one_contract = args.force_min_1,
        universe              = load_universe(
            use_tv_screener=True,
            supplement=not args.no_supplement,
        ),
    )

    print(f"\n{'='*60}")
    print(f"OPTIONS-FIRST BACKTEST")
    print(f"  period:  {cfg.start_date} → {cfg.end_date}")
    print(f"  equity:  ${cfg.starting_equity:,.0f}")
    print(f"  risk/trade: {cfg.risk_per_trade_pct*100:.1f}% = ${cfg.starting_equity*cfg.risk_per_trade_pct:.0f}")
    print(f"  force_min_1: {cfg.force_min_one_contract}")
    print(f"{'='*60}\n")

    if args.variant_e:
        run_abce_comparison(
            start_date      = date.fromisoformat(args.start),
            end_date        = date.fromisoformat(args.end),
            starting_equity = args.equity,
            no_cache        = args.no_cache,
        )
        print("\n[done] A/B/C/E comparison saved to reports/backtests/")
        return

    if args.variant_d:
        run_variant_d_comparison(
            start_date      = date.fromisoformat(args.start),
            end_date        = date.fromisoformat(args.end),
            starting_equity = args.equity,
            no_cache        = args.no_cache,
        )
        print("\n[done] liquidity filter comparison saved to reports/backtests/")
        return

    if args.compare:
        results = run_comparison(
            start_date      = date.fromisoformat(args.start),
            end_date        = date.fromisoformat(args.end),
            starting_equity = args.equity,
            enable_long_puts = not args.no_puts,
            no_cache        = args.no_cache,
        )
        save_comparison_report(results)
        print("\n[done] comparison saved to reports/backtests/")
        return

    if args.no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)

    engine = BacktestEngine(cfg)
    engine.download_data()

    print("\n[backtest] running simulation...")
    trades, equity_curve, skip_log = engine.run()

    print(f"\n[backtest] {len(trades)} trades closed, {len(skip_log)} skips logged")

    metrics = engine.compute_metrics(trades, equity_curve)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Starting equity: ${metrics['starting_equity']:,.2f}")
    print(f"  Ending equity:   ${metrics['ending_equity']:,.2f}")
    print(f"  Total return:    {metrics['total_return_pct']:+.2f}%")
    print(f"  Max drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:          {metrics['sharpe_ratio']:.3f}")
    print(f"  Win rate:        {metrics['win_rate_pct']:.1f}%  ({metrics['winning_trades']}/{metrics['total_trades']})")
    print(f"  Profit factor:   {metrics['profit_factor']:.3f}")
    print(f"  Calls/Puts:      {metrics['total_calls']}/{metrics['total_puts']}")
    print(f"  Avg hold:        {metrics['avg_holding_days']:.1f} days")
    print(f"{'='*60}\n")

    save_reports(trades, equity_curve, skip_log, metrics, cfg)
    print("\n[done] reports saved to reports/backtests/")


if __name__ == "__main__":
    main()
