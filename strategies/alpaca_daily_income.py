"""
Long-only Alpaca paper strategy for disciplined intraday opportunity capture.

This module emits signals only.  ``run.py`` remains responsible for brokerage
access, hard risk gates, journaling, and order submission.
"""

from __future__ import annotations

from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

ET = ZoneInfo("America/New_York")


class AlpacaDailyIncomeStrategy(BaseStrategy):
    """ORB/VWAP/premarket breakout entries with conservative long-only exits."""

    strategy_id: str = "alpaca_daily_income"

    DEFAULT_PARAMS: dict[str, Any] = {
        "min_price": 5.0,
        "orb_minutes": 5,
        "no_entry_before": "09:35",
        "no_entry_after": "15:30",
        "fast_ema": 9,
        "slow_ema": 21,
        "volume_lookback": 20,
        "min_relative_volume": 1.5,
        "atr_period": 14,
        "atr_stop_buffer": 0.20,
        "minimum_stop_pct": 0.002,
        "maximum_stop_pct": 0.025,
        "min_risk_reward": 1.5,
        "target_risk_reward": 2.0,
        "max_entries_per_day": 3,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__({**self.DEFAULT_PARAMS, **(params or {})})

    def generate_signals(
        self,
        bars: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        p = self.params
        min_bars = max(p["volume_lookback"], p["slow_ema"], p["atr_period"]) + 2
        if not self._validate_bars(bars) or len(bars) < min_bars:
            return self._empty_signals(bars)

        frame = bars.copy()
        signals = self._empty_signals(bars)
        ts = self._timestamps_et(frame)
        if ts is None:
            return signals

        frame["_timestamp_et"] = ts
        frame["_date_et"] = ts.dt.date
        frame["_time_et"] = ts.dt.time
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        atr = self.atr(frame, p["atr_period"])
        fast = self.ema(close, p["fast_ema"])
        slow = self.ema(close, p["slow_ema"])
        vwap = self._session_vwap(frame)
        rel_volume = volume / volume.rolling(p["volume_lookback"]).mean().shift(1).replace(0, np.nan)
        orb = self._opening_ranges(frame, int(p["orb_minutes"]))
        ctx = context or {}
        pm_high = self._context_level(ctx, "premarket_high", "pre_market_high", "pm_high")

        earliest = self._parse_time(p["no_entry_before"])
        latest = self._parse_time(p["no_entry_after"])
        entries_by_day: dict[Any, int] = {}

        for i in range(min_bars, len(frame)):
            day = frame["_date_et"].iloc[i]
            bar_time = frame["_time_et"].iloc[i]
            price = float(close.iloc[i])
            prev_price = float(close.iloc[i - 1])
            if (
                not np.isfinite(price)
                or price < float(p["min_price"])
                or bar_time < earliest
                or bar_time >= latest
                or day not in orb
                or entries_by_day.get(day, 0) >= int(p["max_entries_per_day"])
            ):
                continue

            current_vwap = float(vwap.iloc[i]) if np.isfinite(vwap.iloc[i]) else 0.0
            trend_aligned = bool(price > current_vwap and fast.iloc[i] > slow.iloc[i])
            volume_expanding = bool(
                np.isfinite(rel_volume.iloc[i])
                and rel_volume.iloc[i] >= float(p["min_relative_volume"])
            )
            if not (trend_aligned and volume_expanding):
                self._mark_vwap_loss_exit(signals, frame, close, vwap, i)
                continue

            orb_high = float(orb[day]["high"])
            setup = ""
            if prev_price <= orb_high < price:
                setup = "opening_range_breakout"
            elif vwap.iloc[i - 1] >= prev_price and price > current_vwap:
                setup = "vwap_reclaim"
            elif pm_high > 0 and prev_price <= pm_high < price:
                setup = "premarket_high_breakout"

            if not setup:
                self._mark_vwap_loss_exit(signals, frame, close, vwap, i)
                continue

            atr_value = max(float(atr.iloc[i]), price * float(p["minimum_stop_pct"]))
            recent_low = float(frame["low"].iloc[max(0, i - 3): i + 1].min())
            structural_stop = min(recent_low, current_vwap)
            stop = structural_stop - atr_value * float(p["atr_stop_buffer"])
            risk = price - stop
            max_risk = price * float(p["maximum_stop_pct"])
            if risk <= 0 or risk > max_risk:
                continue

            rr = max(float(p["min_risk_reward"]), float(p["target_risk_reward"]))
            target = price + risk * rr
            signals.at[bars.index[i], "buy_signal"] = True
            signals.at[bars.index[i], "entry_reason"] = (
                f"alpaca_income:{setup}:above_vwap:ema_aligned:"
                f"rvol={rel_volume.iloc[i]:.2f}:rr={rr:.2f}"
            )
            signals.at[bars.index[i], "stop_price"] = round(stop, 2)
            signals.at[bars.index[i], "target_price"] = round(target, 2)
            signals.at[bars.index[i], "risk_reward"] = round(rr, 2)
            entries_by_day[day] = entries_by_day.get(day, 0) + 1

            self._mark_vwap_loss_exit(signals, frame, close, vwap, i)

        return self._add_strategy_id(signals)

    @staticmethod
    def _timestamps_et(bars: pd.DataFrame) -> pd.Series | None:
        if "timestamp_et" in bars.columns:
            ts = pd.to_datetime(bars["timestamp_et"], errors="coerce")
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(ET)
            return ts.dt.tz_convert(ET)
        if "timestamp" in bars.columns:
            return pd.to_datetime(bars["timestamp"], errors="coerce", utc=True).dt.tz_convert(ET)
        if isinstance(bars.index, pd.DatetimeIndex):
            idx = pd.to_datetime(bars.index, errors="coerce", utc=True).tz_convert(ET)
            return pd.Series(idx, index=bars.index)
        return None

    @staticmethod
    def _parse_time(value: str) -> time:
        return time.fromisoformat(str(value))

    @staticmethod
    def _context_level(context: dict[str, Any], *keys: str) -> float:
        for key in keys:
            try:
                value = float(context.get(key, 0.0))
                if np.isfinite(value) and value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _session_vwap(bars: pd.DataFrame) -> pd.Series:
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        tpv = typical * bars["volume"]
        cum_tpv = tpv.groupby(bars["_date_et"]).cumsum()
        cum_volume = bars["volume"].groupby(bars["_date_et"]).cumsum()
        return cum_tpv / cum_volume.replace(0, np.nan)

    @staticmethod
    def _opening_ranges(bars: pd.DataFrame, minutes: int) -> dict[Any, dict[str, float]]:
        ranges: dict[Any, dict[str, float]] = {}
        start_minutes = 9 * 60 + 30
        end_minutes = start_minutes + minutes
        total_minutes = (
            bars["_timestamp_et"].dt.hour * 60 + bars["_timestamp_et"].dt.minute
        )
        for day, group in bars.groupby("_date_et"):
            group_minutes = total_minutes.loc[group.index]
            opening = group[(group_minutes >= start_minutes) & (group_minutes < end_minutes)]
            if not opening.empty:
                ranges[day] = {
                    "high": float(opening["high"].max()),
                    "low": float(opening["low"].min()),
                }
        return ranges

    @staticmethod
    def _mark_vwap_loss_exit(
        signals: pd.DataFrame,
        bars: pd.DataFrame,
        close: pd.Series,
        vwap: pd.Series,
        i: int,
    ) -> None:
        if i > 0 and close.iloc[i - 1] >= vwap.iloc[i - 1] and close.iloc[i] < vwap.iloc[i]:
            signals.at[bars.index[i], "sell_signal"] = True
            signals.at[bars.index[i], "exit_reason"] = "alpaca_income:vwap_loss"
