"""
strategies/flawless_victory.py

Flawless Victory Strategy — High-Conviction Multi-Condition Entry
Python translation of the Flawless Victory Pine Script.

Logic
-----
Requires ALL of the following conditions simultaneously:
  1. Price above VWAP (trend bias)
  2. Fast EMA > Slow EMA (short-term momentum)
  3. Price near a confirmed pivot support (institutional level)
  4. Volume spike above threshold (smart money participation)
  5. RSI in non-overbought range (40–68) for buy, non-oversold for short
  6. ATR expanding (volatility confirming breakout)

This strategy fires rarely but with high conviction — "flawless"
because every filter must pass before entry.

BUY  — all 6 conditions met on long side.
SHORT — all 6 conditions met on short side (inverted conditions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class FlawlessVictoryStrategy(BaseStrategy):
    """High-conviction multi-condition entry requiring all filters to align."""

    strategy_id: str = "flawless_victory"

    DEFAULT_PARAMS: dict[str, Any] = {
        "fast_ema": 9,
        "slow_ema": 21,
        "rsi_period": 14,
        "rsi_buy_min": 40,       # RSI must be above this for buy
        "rsi_buy_max": 68,       # RSI must be below this for buy
        "rsi_short_min": 32,     # RSI must be above this for short
        "rsi_short_max": 60,     # RSI must be below this for short
        "volume_spike_mult": 1.5,  # volume must be N× avg
        "volume_lookback": 20,
        "atr_expand_lookback": 5,  # ATR expanding = current > N-bar avg
        "pivot_proximity_pct": 0.004,  # within 0.4% of pivot level
        "pivot_lookback": 15,
        "atr_stop_mult": 1.0,
        "atr_target_mult": 2.5,
        "allow_shorts": True,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signals(
        self,
        bars: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        min_bars = max(self.params["slow_ema"], self.params["volume_lookback"], 30) + 5
        if not self._validate_bars(bars) or len(bars) < min_bars:
            return self._empty_signals(bars)

        ctx = context or {}
        p = self.params
        signals = self._empty_signals(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        fast_ema = self.ema(close, p["fast_ema"])
        slow_ema = self.ema(close, p["slow_ema"])
        rsi_vals = self.rsi(close, p["rsi_period"])
        atr_vals = self.atr(bars, 14)
        _vwap_ctx = ctx.get("vwap_series")
        vwap_series = _vwap_ctx if _vwap_ctx is not None else self.vwap(bars)

        vol_avg = volume.rolling(p["volume_lookback"]).mean()
        atr_avg = atr_vals.rolling(p["atr_expand_lookback"]).mean()

        # Pivot levels from context or derived
        pivot_highs = self._pivot_levels(high, p["pivot_lookback"])
        pivot_lows = self._pivot_levels(low, p["pivot_lookback"], use_min=True)

        start = max(p["slow_ema"], p["volume_lookback"], p["pivot_lookback"]) + 5

        for i in range(start, len(bars)):
            price = close.iloc[i]
            atr_val = atr_vals.iloc[i]

            # ── Condition checks ─────────────────────────────────────────────
            vwap_val = vwap_series.iloc[i] if hasattr(vwap_series, "iloc") else float(vwap_series)
            above_vwap = price > vwap_val
            fast_above_slow = fast_ema.iloc[i] > slow_ema.iloc[i]
            fast_below_slow = fast_ema.iloc[i] < slow_ema.iloc[i]

            rsi = rsi_vals.iloc[i]
            rsi_ok_buy = p["rsi_buy_min"] <= rsi <= p["rsi_buy_max"]
            rsi_ok_short = p["rsi_short_min"] <= rsi <= p["rsi_short_max"]

            vol_ratio = (volume.iloc[i] / vol_avg.iloc[i]) if vol_avg.iloc[i] > 0 else 0.0
            vol_spike = vol_ratio >= p["volume_spike_mult"]

            atr_expanding = atr_vals.iloc[i] > atr_avg.iloc[i] if not np.isnan(atr_avg.iloc[i]) else False

            near_pivot_support = self._near_level(price, pivot_lows.iloc[i], p["pivot_proximity_pct"])
            near_pivot_resistance = self._near_level(price, pivot_highs.iloc[i], p["pivot_proximity_pct"])

            # ── FLAWLESS VICTORY BUY — all 6 conditions ──────────────────────
            if (
                above_vwap
                and fast_above_slow
                and near_pivot_support
                and vol_spike
                and rsi_ok_buy
                and atr_expanding
            ):
                stop = price - atr_val * p["atr_stop_mult"]
                target = price + atr_val * p["atr_target_mult"]
                rr = (target - price) / max(price - stop, 1e-8)
                signals.at[bars.index[i], "buy_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = (
                    f"fv:all6:vwap+ema+pivot+vol+rsi+atr:rr={rr:.1f}"
                )
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

            # ── FLAWLESS VICTORY SHORT — all 6 conditions inverted ────────────
            if p["allow_shorts"] and (
                not above_vwap
                and fast_below_slow
                and near_pivot_resistance
                and vol_spike
                and rsi_ok_short
                and atr_expanding
            ):
                stop = price + atr_val * p["atr_stop_mult"]
                target = price - atr_val * p["atr_target_mult"]
                rr = (price - target) / max(stop - price, 1e-8)
                signals.at[bars.index[i], "short_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = (
                    f"fv:all6_short:vwap+ema+pivot+vol+rsi+atr:rr={rr:.1f}"
                )
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

            # ── EXIT: EMA cross against position ─────────────────────────────
            if fast_below_slow and fast_ema.iloc[i - 1] >= slow_ema.iloc[i - 1]:
                signals.at[bars.index[i], "sell_signal"] = True
                signals.at[bars.index[i], "exit_reason"] = "fv:ema_cross_down"
            if p["allow_shorts"] and fast_above_slow and fast_ema.iloc[i - 1] <= slow_ema.iloc[i - 1]:
                signals.at[bars.index[i], "cover_signal"] = True
                signals.at[bars.index[i], "exit_reason"] = "fv:ema_cross_up"

        return self._add_strategy_id(signals)

    @staticmethod
    def _pivot_levels(series: pd.Series, lb: int, use_min: bool = False) -> pd.Series:
        """Rolling pivot high or low, forward-filled so each bar has the last confirmed level."""
        result = pd.Series(np.nan, index=series.index)
        fn = min if use_min else max
        for i in range(lb, len(series) - lb):
            window = series.iloc[i - lb: i + lb + 1]
            pivot_val = fn(window)
            if series.iloc[i] == pivot_val:
                result.iloc[i + lb] = float(series.iloc[i])
        return result.ffill()

    @staticmethod
    def _near_level(price: float, level: float, pct: float) -> bool:
        if np.isnan(level) or level <= 0:
            return False
        return abs(price - level) / level <= pct
