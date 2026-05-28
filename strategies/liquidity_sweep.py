"""
strategies/liquidity_sweep.py

Liquidity Sweep Strategy — Smart Money Concepts
Python translation of the Liquidity Sweep Pine Script.

Logic
-----
Liquidity pools sit above recent swing highs and below recent swing lows.
Smart money engineers a "sweep" to fill their own orders before reversing.

1. Identify swing highs / swing lows using SWING_LOOKBACK bars.
2. A liquidity sweep occurs when:
   - Price spikes above a recent swing high (buyside liquidity taken),
     then closes back BELOW it on the same or next bar (rejection).
   - OR price spikes below a recent swing low (sellside liquidity taken),
     then closes back ABOVE it.
3. BUY  — sellside sweep (spike below swing low → close above).
4. SHORT — buyside sweep (spike above swing high → close below).
5. Stops placed at the wick extreme + ATR buffer.

Translated from Pine:
  - Pine ta.pivothigh() → pandas rolling argmax with shift
  - Pine ta.pivotlow()  → pandas rolling argmin with shift
  - No lookahead: sweep detection uses close of the current bar vs prior swing
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class LiquiditySweepStrategy(BaseStrategy):
    """Liquidity sweep / stop hunt reversal strategy."""

    strategy_id: str = "liquidity_sweep"

    DEFAULT_PARAMS: dict[str, Any] = {
        "swing_lookback": 10,      # bars left/right to identify pivot
        "sweep_wick_pct": 0.001,   # min wick size to count as a sweep
        "atr_stop_mult": 0.75,
        "atr_target_mult": 2.0,
        "volume_confirm_mult": 1.0,
        "volume_lookback": 20,
        "allow_shorts": True,
        "max_sweep_age_bars": 8,   # ignore stale pivots older than N bars
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signals(
        self,
        bars: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not self._validate_bars(bars) or len(bars) < self.params["swing_lookback"] * 2 + 5:
            return self._empty_signals(bars)

        ctx = context or {}
        p = self.params
        signals = self._empty_signals(bars)

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        volume = bars["volume"]
        vol_avg = volume.rolling(p["volume_lookback"]).mean()
        atr_vals = self.atr(bars, 14).values

        lb = p["swing_lookback"]

        # ── Pivot detection (no lookahead — shift by lb to align) ─────────────
        # A pivot high at bar i requires that bars[i] is the max of bars[i-lb:i+lb+1]
        # To avoid lookahead, we detect pivots with a lag of lb bars.
        swing_highs = self._pivot_highs(bars["high"], lb)  # Series, NaN where no pivot
        swing_lows = self._pivot_lows(bars["low"], lb)

        # Track the most recent active swing levels
        recent_swing_high: float | None = None
        recent_swing_high_idx: int = -1
        recent_swing_low: float | None = None
        recent_swing_low_idx: int = -1

        start = p["volume_lookback"] + lb + 2

        for i in range(start, len(bars)):
            # Update swing tracking (using confirmed pivots up to i)
            if not np.isnan(swing_highs.iloc[i]):
                recent_swing_high = float(swing_highs.iloc[i])
                recent_swing_high_idx = i
            if not np.isnan(swing_lows.iloc[i]):
                recent_swing_low = float(swing_lows.iloc[i])
                recent_swing_low_idx = i

            c = close[i]
            h = high[i]
            l = low[i]
            vol_ratio = (volume.iloc[i] / vol_avg.iloc[i]) if vol_avg.iloc[i] > 0 else 0.0
            atr_val = atr_vals[i]

            # ── BUYSIDE SWEEP: spike above swing high, close below it ──────────
            if (
                recent_swing_high is not None
                and (i - recent_swing_high_idx) <= p["max_sweep_age_bars"]
                and h > recent_swing_high                       # wick swept above
                and c < recent_swing_high                       # close rejected back below
                and (h - recent_swing_high) >= atr_val * p["sweep_wick_pct"]
            ):
                if p["allow_shorts"]:
                    stop = h + atr_val * p["atr_stop_mult"]
                    target = c - atr_val * p["atr_target_mult"]
                    rr = (c - target) / max(stop - c, 1e-8)
                    signals.at[bars.index[i], "short_signal"] = True
                    signals.at[bars.index[i], "entry_reason"] = (
                        f"liq_sweep:buyside_rejection:level={recent_swing_high:.2f}"
                    )
                    signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                    signals.at[bars.index[i], "target_price"] = round(target, 2)
                    signals.at[bars.index[i], "risk_reward"] = round(rr, 2)
                    recent_swing_high = None  # consumed

            # ── SELLSIDE SWEEP: spike below swing low, close above it ──────────
            if (
                recent_swing_low is not None
                and (i - recent_swing_low_idx) <= p["max_sweep_age_bars"]
                and l < recent_swing_low                        # wick swept below
                and c > recent_swing_low                        # close recovered above
                and (recent_swing_low - l) >= atr_val * p["sweep_wick_pct"]
                and vol_ratio >= p["volume_confirm_mult"]
            ):
                stop = l - atr_val * p["atr_stop_mult"]
                target = c + atr_val * p["atr_target_mult"]
                rr = (target - c) / max(c - stop, 1e-8)
                signals.at[bars.index[i], "buy_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = (
                    f"liq_sweep:sellside_reversal:level={recent_swing_low:.2f}"
                )
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)
                recent_swing_low = None  # consumed

            # ── Exits ─────────────────────────────────────────────────────────
            # Close long if price approaches next swing high (take profit)
            if recent_swing_high is not None:
                near_resistance = abs(c - recent_swing_high) / max(recent_swing_high, 1e-8) < 0.003
                if near_resistance:
                    signals.at[bars.index[i], "sell_signal"] = True
                    signals.at[bars.index[i], "exit_reason"] = "liq_sweep:near_resistance"

        return self._add_strategy_id(signals)

    @staticmethod
    def _pivot_highs(series: pd.Series, lb: int) -> pd.Series:
        """Identify pivot highs with lb-bar confirmation lag. No lookahead."""
        result = pd.Series(np.nan, index=series.index)
        for i in range(lb, len(series) - lb):
            window = series.iloc[i - lb: i + lb + 1]
            if series.iloc[i] == window.max():
                result.iloc[i + lb] = series.iloc[i]   # emit at i+lb (confirmed)
        return result

    @staticmethod
    def _pivot_lows(series: pd.Series, lb: int) -> pd.Series:
        result = pd.Series(np.nan, index=series.index)
        for i in range(lb, len(series) - lb):
            window = series.iloc[i - lb: i + lb + 1]
            if series.iloc[i] == window.min():
                result.iloc[i + lb] = series.iloc[i]
        return result
