"""
strategies/base.py

Abstract base class for all QuantumQuant signal-generation strategies.

CONTRACT
--------
* Strategies ONLY generate signals.
* Strategies NEVER place orders, call Alpaca, or read credentials.
* run.py owns execution, risk controls, and Alpaca interaction.
* All calculations must be free of lookahead bias:
    - Access row[t] using only bars.iloc[:i+1] or column.shift(n)
    - Never reference future bars
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

# Required columns in the signal DataFrame returned by generate_signals()
SIGNAL_COLUMNS = [
    "buy_signal",
    "sell_signal",
    "short_signal",
    "cover_signal",
    "entry_reason",
    "exit_reason",
    "stop_price",
    "target_price",
    "risk_reward",
    "strategy_id",
]


class BaseStrategy(ABC):
    """
    Base class for all signal strategies.

    Subclasses must implement generate_signals() and set strategy_id.
    """

    strategy_id: str = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = params or {}

    # ── Subclass contract ────────────────────────────────────────────────────

    @abstractmethod
    def generate_signals(
        self,
        bars: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV bars.

        Parameters
        ----------
        bars : pd.DataFrame
            OHLCV with columns: timestamp, open, high, low, close, volume.
            Must be sorted ascending by timestamp.
        context : dict | None
            Optional enrichment:
              - "mtf_bars": dict[str, pd.DataFrame]  — higher-TF bars
              - "structure_map": dict                 — premarket structure data
              - "vwap": float                         — session VWAP
              - "prev_day_high": float
              - "prev_day_low": float
              - "orb_high": float / "orb_low": float  — 30-min opening range

        Returns
        -------
        pd.DataFrame
            One row per input bar, with columns defined in SIGNAL_COLUMNS.
            All boolean signal columns default False, str columns default "".
        """

    # ── Helpers (shared across all strategies) ───────────────────────────────

    def _empty_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a zero-signal DataFrame aligned to bars."""
        n = len(bars)
        df = pd.DataFrame(
            {
                "buy_signal": False,
                "sell_signal": False,
                "short_signal": False,
                "cover_signal": False,
                "entry_reason": "",
                "exit_reason": "",
                "stop_price": np.nan,
                "target_price": np.nan,
                "risk_reward": np.nan,
                "strategy_id": self.strategy_id,
            },
            index=bars.index,
        )
        return df

    @staticmethod
    def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """True on bar where series_a crosses above series_b (no lookahead)."""
        return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))

    @staticmethod
    def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """True on bar where series_a crosses below series_b (no lookahead)."""
        return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))

    @staticmethod
    def resample_bars(
        bars: pd.DataFrame,
        timeframe: str,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Translate Pine security() into pandas resample.

        Resamples 1m OHLCV bars to a higher timeframe.
        Uses only completed bars (no partial candle lookahead).

        Parameters
        ----------
        bars : pd.DataFrame
            1-minute bars with timestamp, open, high, low, close, volume.
        timeframe : str
            pandas resample rule: "5min", "15min", "1h", "1D", etc.
        """
        if timestamp_col not in bars.columns:
            return pd.DataFrame()

        df = bars.copy()
        df = df.set_index(pd.DatetimeIndex(df[timestamp_col]))

        resampled = (
            df.resample(timeframe, closed="left", label="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "close"])
        )
        resampled.index.name = "timestamp"
        return resampled.reset_index()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range — no lookahead."""
        high = bars["high"]
        low = bars["low"]
        close = bars["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - close).abs(), (low - close).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def vwap(bars: pd.DataFrame) -> pd.Series:
        """Session VWAP — cumulative, resets each day."""
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_tpv = (typical * bars["volume"]).cumsum()
        cum_vol = bars["volume"].cumsum()
        return cum_tpv / cum_vol.replace(0, np.nan)

    def _validate_bars(self, bars: pd.DataFrame) -> bool:
        required = {"open", "high", "low", "close", "volume"}
        return required.issubset(bars.columns) and len(bars) > 0

    def _add_strategy_id(self, signals: pd.DataFrame) -> pd.DataFrame:
        signals["strategy_id"] = self.strategy_id
        return signals
