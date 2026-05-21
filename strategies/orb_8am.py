"""
strategies/orb_8am.py

Opening Range Breakout — 8 AM / 30-minute variant
Python translation of the ORB Pine Script.

Logic
-----
1. Capture the opening range: first ORB_MINUTES bars starting at ORB_START_HOUR_ET.
2. ORB_HIGH = max(high) during range.
   ORB_LOW  = min(low)  during range.
3. BUY    — price breaks above ORB_HIGH with volume confirmation.
4. SHORT  — price breaks below ORB_LOW with volume confirmation.
5. Stops  — ATR-based, placed just inside the opposite range boundary.
6. Targets — measured-move: 1× range extension above/below breakout bar.

ORB_START_HOUR_ET=8 reflects a premarket/early-open setup.
Set ORB_START_HOUR_ET=9, ORB_START_MINUTE_ET=30 for standard equity open.

No lookahead: opening range is computed from bars where timestamp_et < range_end.
"""

from __future__ import annotations

from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

ET = ZoneInfo("America/New_York")


class ORB8amStrategy(BaseStrategy):
    """30-minute Opening Range Breakout starting at 8:00 AM ET."""

    strategy_id: str = "orb_8am"

    DEFAULT_PARAMS: dict[str, Any] = {
        "orb_start_hour": 8,
        "orb_start_minute": 0,
        "orb_minutes": 30,
        "atr_stop_mult": 0.5,
        "atr_target_mult": 1.0,    # 1× range = measured move default
        "target_range_mult": 1.0,  # target = breakout_price ± N × ORB_range
        "volume_confirm_mult": 1.3,
        "volume_lookback": 20,
        "allow_shorts": True,
        "max_entries_per_day": 2,
        "no_trade_after_hour": 14,  # ET — no new entries after 2pm
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signals(
        self,
        bars: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not self._validate_bars(bars) or len(bars) < 5:
            return self._empty_signals(bars)

        ctx = context or {}
        p = self.params
        signals = self._empty_signals(bars)

        # ── Localise timestamps ───────────────────────────────────────────────
        bars = bars.copy()
        if "timestamp_et" not in bars.columns:
            if "timestamp" in bars.columns:
                ts = pd.to_datetime(bars["timestamp"], utc=True)
                bars["timestamp_et"] = ts.dt.tz_convert(ET)
            else:
                return signals

        bars["_hour_et"] = bars["timestamp_et"].dt.hour
        bars["_minute_et"] = bars["timestamp_et"].dt.minute
        bars["_date_et"] = bars["timestamp_et"].dt.date

        volume = bars["volume"]
        vol_avg = volume.rolling(p["volume_lookback"]).mean()
        atr_vals = self.atr(bars, 14)

        # ── Compute ORB per day ───────────────────────────────────────────────
        # Use context values if provided (e.g. from run.py premarket planner)
        orb_high_ctx = ctx.get("orb_high")
        orb_low_ctx = ctx.get("orb_low")

        orb_data: dict[Any, dict[str, float]] = {}

        if orb_high_ctx is not None and orb_low_ctx is not None:
            # Use single context value for all dates in bars
            for d in bars["_date_et"].unique():
                orb_data[d] = {"high": float(orb_high_ctx), "low": float(orb_low_ctx)}
        else:
            orb_data = self._compute_orb_by_day(bars, p)

        # ── Signal generation ─────────────────────────────────────────────────
        entries_by_day: dict[Any, int] = {}

        for i in range(p["volume_lookback"] + 1, len(bars)):
            row = bars.iloc[i]
            date = row["_date_et"]
            hour = row["_hour_et"]
            price = bars["close"].iloc[i]
            prev_close = bars["close"].iloc[i - 1]

            # No new entries too late in session
            if hour >= p["no_trade_after_hour"]:
                continue

            if date not in orb_data:
                continue

            orb = orb_data[date]
            orb_h = orb["high"]
            orb_l = orb["low"]
            orb_range = orb_h - orb_l

            if orb_range <= 0:
                continue

            vol_ratio = (volume.iloc[i] / vol_avg.iloc[i]) if vol_avg.iloc[i] > 0 else 0.0
            vol_ok = vol_ratio >= p["volume_confirm_mult"]
            atr_val = atr_vals.iloc[i]
            day_entries = entries_by_day.get(date, 0)

            # ── ORB Breakout BUY ──────────────────────────────────────────────
            broke_high = prev_close <= orb_h < price
            if broke_high and vol_ok and day_entries < p["max_entries_per_day"]:
                stop = orb_h - atr_val * p["atr_stop_mult"]
                target = price + orb_range * p["target_range_mult"]
                rr = (target - price) / max(price - stop, 1e-8)
                signals.at[bars.index[i], "buy_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = (
                    f"orb:breakout_high:range={orb_range:.2f}:rr={rr:.1f}"
                )
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)
                entries_by_day[date] = day_entries + 1

            # ── ORB Breakdown SHORT ───────────────────────────────────────────
            broke_low = prev_close >= orb_l > price
            if p["allow_shorts"] and broke_low and vol_ok and day_entries < p["max_entries_per_day"]:
                stop = orb_l + atr_val * p["atr_stop_mult"]
                target = price - orb_range * p["target_range_mult"]
                rr = (price - target) / max(stop - price, 1e-8)
                signals.at[bars.index[i], "short_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = (
                    f"orb:breakdown_low:range={orb_range:.2f}:rr={rr:.1f}"
                )
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)
                entries_by_day[date] = day_entries + 1

            # ── Exit: price returns inside ORB ────────────────────────────────
            back_inside = orb_l <= price <= orb_h
            if back_inside:
                if bars["close"].iloc[i - 1] > orb_h:
                    signals.at[bars.index[i], "sell_signal"] = True
                    signals.at[bars.index[i], "exit_reason"] = "orb:failed_breakout"
                elif bars["close"].iloc[i - 1] < orb_l and p["allow_shorts"]:
                    signals.at[bars.index[i], "cover_signal"] = True
                    signals.at[bars.index[i], "exit_reason"] = "orb:failed_breakdown"

        return self._add_strategy_id(signals)

    def _compute_orb_by_day(
        self, bars: pd.DataFrame, p: dict[str, Any]
    ) -> dict[Any, dict[str, float]]:
        """Compute ORB high/low per trading day from 1m bars. No lookahead."""
        result: dict[Any, dict[str, float]] = {}
        orb_end_minute = p["orb_start_hour"] * 60 + p["orb_start_minute"] + p["orb_minutes"]

        for date, group in bars.groupby("_date_et"):
            total_minutes = group["_hour_et"] * 60 + group["_minute_et"]
            start_minutes = p["orb_start_hour"] * 60 + p["orb_start_minute"]
            mask = (total_minutes >= start_minutes) & (total_minutes < orb_end_minute)
            orb_bars = group[mask]
            if len(orb_bars) >= 2:
                result[date] = {
                    "high": float(orb_bars["high"].max()),
                    "low": float(orb_bars["low"].min()),
                }
        return result
