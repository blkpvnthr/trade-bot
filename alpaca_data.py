#!/usr/bin/env python3
"""
Alpaca market-data adapter for the V2.9.8 live engine.
======================================================

Production decisions run on Alpaca daily bars. Two DISTINCT price notions are
kept separate on purpose:

* DECISION bars  -> ``adjustment=ALL`` (split + dividend adjusted). The research
  backtest uses yfinance ``auto_adjust=True`` total-return closes; Alpaca ``ALL``
  is the closest corporate-action-consistent equivalent. They are NOT guaranteed
  numerically identical (different vendors, different dividend timing) -- see
  LIVE_TRADING.md. Only ``ALL`` is used to compute weights.

* EXECUTABLE prices -> latest trade (raw, current). Order sizing/notional ALWAYS
  uses these real executable prices, never adjusted historical pseudo-prices.

Timing is backtest-faithful: only COMPLETED daily sessions feed the decision. An
in-progress (today's) bar is dropped while the market is open.

yfinance remains research-only; it is never imported here.
"""
from __future__ import annotations

import logging
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

log = logging.getLogger("v298.data")
ET = ZoneInfo("America/New_York")

_FEED_MAP = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}


class MarketDataError(RuntimeError):
    """Raised when market data is stale/missing/incomplete -- execution must fail closed."""


def get_credentials() -> tuple[str, str]:
    key = (os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
           or os.getenv("ALPACA_KEY"))
    sec = (os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
           or os.getenv("ALPACA_SECRET"))
    if not key or not sec:
        raise MarketDataError("Missing Alpaca credentials (APCA_API_KEY_ID / "
                              "APCA_API_SECRET_KEY).")
    return key, sec


def resolve_feed(configured: str | None = None) -> DataFeed:
    """Resolve and LOG the data feed. Never silently substitutes a feed (#11)."""
    name = (configured or os.getenv("ALPACA_DATA_FEED", "iex")).strip().lower()
    feed = _FEED_MAP.get(name)
    if feed is None:
        raise MarketDataError(f"Unsupported ALPACA_DATA_FEED={name!r} (use iex|sip)")
    log.info("v298 market-data feed = %s", feed.value)
    return feed


class AlpacaData:
    def __init__(self, feed: str | None = None):
        key, sec = get_credentials()
        self._data = StockHistoricalDataClient(key, sec)
        # TradingClient is used only for clock/calendar here; paper flag is
        # irrelevant for read-only clock/calendar but we default to paper so a
        # data-only construction never implies a live environment.
        self._trading = TradingClient(key, sec, paper=True)
        self.feed = resolve_feed(feed)

    # ---- clock / calendar ------------------------------------------------
    def get_market_clock(self) -> dict:
        c = self._trading.get_clock()
        ts = pd.Timestamp(c.timestamp)
        return {"is_open": bool(c.is_open),
                "timestamp": ts,
                "timestamp_et": ts.tz_convert(ET) if ts.tzinfo else ts,
                "next_open": pd.Timestamp(c.next_open),
                "next_close": pd.Timestamp(c.next_close)}

    def get_next_trading_session(self, after: pd.Timestamp | None = None) -> pd.Timestamp:
        """First trading session strictly after `after` (default: today ET)."""
        base = (pd.Timestamp(after) if after is not None
                else pd.Timestamp.now(tz=ET)).normalize()
        start = (base.tz_localize(None) if base.tzinfo else base).date()
        cal = self._trading.get_calendar(
            GetCalendarRequest(start=start, end=(pd.Timestamp(start) + pd.Timedelta(days=10)).date()))
        for day in cal:
            d = pd.Timestamp(day.date)
            if d.date() > start:
                return d.normalize()
        raise MarketDataError("no upcoming trading session found in calendar window")

    def last_completed_session(self, clock: dict | None = None) -> pd.Timestamp:
        """
        The most recent session whose close is in the past (ET). While the market
        is open, that is the PREVIOUS trading day; the in-progress bar is excluded.
        """
        clock = clock or self.get_market_clock()
        now_et = clock["timestamp_et"]
        today = pd.Timestamp(now_et).normalize().tz_localize(None)
        if clock["is_open"]:
            # today's session is in progress -> most recent completed is < today
            start = (today - pd.Timedelta(days=10)).date()
            cal = self._trading.get_calendar(
                GetCalendarRequest(start=start, end=today.date()))
            days = [pd.Timestamp(d.date).normalize() for d in cal
                    if pd.Timestamp(d.date).normalize() < today]
            if not days:
                raise MarketDataError("no completed session before today")
            return days[-1]
        # market closed: today's bar (if a trading day) is final; otherwise the
        # latest prior trading day. The actual value is pinned by the bar frame.
        return today

    # ---- daily bars (adjusted, for DECISIONS) ---------------------------
    def get_completed_daily_bars(self, symbols: list[str], lookback_sessions: int,
                                 end: pd.Timestamp | None = None,
                                 warmup_mult: float = 1.6) -> pd.DataFrame:
        """
        Adjusted (ALL) daily closes for `symbols`, dropping the in-progress bar.

        Returns a (date x symbol) DataFrame of adjusted closes, tz-naive date
        index, most recent row = last COMPLETED session. Fetches a generous
        warmup buffer (>= lookback_sessions) so covariance/SMA/IC windows fill.
        """
        clock = self.get_market_clock()
        end_ts = pd.Timestamp(end) if end is not None else clock["timestamp"]
        # calendar days needed for `lookback_sessions * warmup` trading days
        calendar_days = int(lookback_sessions * warmup_mult * (7.0 / 5.0)) + 20
        start_ts = (pd.Timestamp(end_ts).tz_localize(None) if pd.Timestamp(end_ts).tzinfo
                    else pd.Timestamp(end_ts)) - pd.Timedelta(days=calendar_days)

        req = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=start_ts.to_pydatetime(),
            end=(pd.Timestamp(end_ts).tz_localize(None).to_pydatetime()
                 if pd.Timestamp(end_ts).tzinfo else pd.Timestamp(end_ts).to_pydatetime()),
            adjustment=Adjustment.ALL,
            feed=self.feed,
        )
        bars = self._data.get_stock_bars(req)
        df = bars.df
        if df is None or len(df) == 0:
            raise MarketDataError("Alpaca returned no daily bars")

        # bars.df is a (symbol, timestamp) multiindex -> pivot to date x symbol closes
        close = (df.reset_index()[["symbol", "timestamp", "close"]]
                 .assign(date=lambda x: pd.to_datetime(x["timestamp"]).dt.tz_convert(ET)
                         .dt.normalize().dt.tz_localize(None))
                 .pivot_table(index="date", columns="symbol", values="close"))
        close = close.reindex(columns=list(symbols)).sort_index()

        # Drop the in-progress session while the market is open.
        if clock["is_open"]:
            today = pd.Timestamp(clock["timestamp_et"]).normalize().tz_localize(None)
            close = close.loc[close.index < today]

        return close

    # ---- executable prices (raw latest trade, for SIZING) ---------------
    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Latest executable trade price per symbol (raw, current) for order sizing.
        Tries the configured feed, then IEX (free real-time), so a SIP-only-
        historical paper subscription still gets a live price.
        """
        feeds = [self.feed]
        if DataFeed.IEX not in feeds:
            feeds.append(DataFeed.IEX)
        for feed in feeds:
            try:
                req = StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=feed)
                trades = self._data.get_stock_latest_trade(req)
                out = {}
                for sym in symbols:
                    t = trades.get(sym)
                    px = float(getattr(t, "price", 0.0) or 0.0) if t is not None else 0.0
                    if px > 0:
                        out[sym] = px
                if out:
                    return out
            except Exception as e:
                log.warning("latest-trade via %s failed (%s)", feed.value, e)
        return {}

    def get_recent_raw_closes(self, symbols: list[str]) -> dict[str, float]:
        """
        Most recent RAW (unadjusted, executable) daily close per symbol -- the
        sizing fallback when real-time quotes are not entitled. Uses only
        historical bars, so it works on a bars-only subscription.
        """
        end = self.get_market_clock()["timestamp"]
        start = (pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo
                 else pd.Timestamp(end)) - pd.Timedelta(days=12)
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day,
                               start=start.to_pydatetime(), adjustment=Adjustment.RAW,
                               feed=self.feed)
        try:
            df = self._data.get_stock_bars(req).df
        except Exception as e:
            log.warning("recent raw closes fetch failed (%s)", e)
            return {}
        if df is None or len(df) == 0:
            return {}
        d = df.reset_index()[["symbol", "close"]].groupby("symbol")["close"].last()
        return {s: float(d[s]) for s in symbols if s in d.index and d[s] > 0}


def validate_market_data(close: pd.DataFrame, symbols: list[str],
                         *, max_stale_sessions: int = 5,
                         max_missing_frac: float = 0.05,
                         expect_last_on_or_before: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Fail-closed validation of the decision price frame. Returns the decision date
    (the last completed session). Raises MarketDataError on any problem.
    """
    missing_cols = [s for s in symbols if s not in close.columns]
    if missing_cols:
        raise MarketDataError(f"missing symbols in bars: {missing_cols}")
    if len(close) < 260:
        raise MarketDataError(f"insufficient history: {len(close)} sessions "
                              f"(need >= 260 for the 200SMA / 252 windows)")

    idx = close.index
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise MarketDataError("bar timestamps not strictly increasing / have duplicates")

    last = idx[-1]
    if expect_last_on_or_before is not None and last > pd.Timestamp(expect_last_on_or_before):
        raise MarketDataError(f"last bar {last.date()} is after the expected "
                              f"completed session {pd.Timestamp(expect_last_on_or_before).date()}")

    last_row = close.loc[last, symbols].to_numpy(float)
    if not np.all(np.isfinite(last_row)) or np.any(last_row <= 0):
        bad = [s for s, v in zip(symbols, last_row) if not np.isfinite(v) or v <= 0]
        raise MarketDataError(f"non-positive/NaN close on the decision session for {bad}")

    # overall missingness per symbol over the window
    frac_missing = close[symbols].isna().mean()
    worst = frac_missing.max()
    if worst > max_missing_frac:
        raise MarketDataError(f"too much missing history (worst {worst:.1%} > "
                              f"{max_missing_frac:.1%})")

    # staleness vs 'now' would be checked by the caller with the clock; here we
    # only ensure the frame's own last row is the finalized decision session.
    return last
