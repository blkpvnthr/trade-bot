"""
strategies/tux_mtf.py

TUX Multi-Timeframe Strategy
Python translation of the TUX MTF Pine Script.

Logic
-----
1. Compute fast/slow EMAs on 1m, 5m, 15m, 1h bars.
2. Trend is bullish on a timeframe when fast EMA > slow EMA.
3. ENTRY: 1m fast crossover above slow, AND at least N_CONFIRM higher TFs agree.
4. EXIT: 1m fast crossunder below slow, OR stop hit.
5. Long-only variant (tux_mtf_long_only) sets allow_shorts=False.

Pine security() → pandas resample via BaseStrategy.resample_bars().
Pine crossover()  → BaseStrategy.crossover().

No lookahead: all resampled bars are left-closed / label-left.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class TuxMTF(BaseStrategy):
    """TUX Multi-Timeframe — original (long + short)."""

    strategy_id: str = "tux_mtf_original"

    # Default parameters — Claude can experiment with these via configs
    DEFAULT_PARAMS: dict[str, Any] = {
        "fast_ema": 9,
        "slow_ema": 21,
        "trend_ema": 50,        # higher-TF trend filter
        "n_confirm_tfs": 2,     # how many higher TFs must agree for entry
        "atr_stop_mult": 1.5,   # stop = entry - N * ATR
        "atr_target_mult": 3.0, # target = entry + N * ATR
        "min_volume_mult": 1.2, # entry bar volume >= N * rolling avg
        "allow_shorts": True,
        "volume_lookback": 20,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def generate_signals(
        self,
        bars: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not self._validate_bars(bars) or len(bars) < self.params["slow_ema"] + 2:
            return self._empty_signals(bars)

        ctx = context or {}
        p = self.params

        signals = self._empty_signals(bars)
        close = bars["close"]
        volume = bars["volume"]

        # ── 1m EMAs ──────────────────────────────────────────────────────────
        fast_1m = self.ema(close, p["fast_ema"])
        slow_1m = self.ema(close, p["slow_ema"])
        trend_1m = self.ema(close, p["trend_ema"])
        atr_1m = self.atr(bars, 14)

        bull_cross_1m = self.crossover(fast_1m, slow_1m)
        bear_cross_1m = self.crossunder(fast_1m, slow_1m)

        # Volume confirmation
        vol_avg = volume.rolling(p["volume_lookback"]).mean()
        vol_confirmed = volume >= (vol_avg * p["min_volume_mult"])

        # ── Higher-TF trend alignment ─────────────────────────────────────────
        mtf_bars: dict[str, pd.DataFrame] = ctx.get("mtf_bars", {})

        # Compute alignment score per bar using forward-fill from completed TF bars
        tf_alignment = self._compute_tf_alignment(bars, mtf_bars, p)

        # ── Signal generation ─────────────────────────────────────────────────
        for i in range(max(p["slow_ema"], p["volume_lookback"]) + 1, len(bars)):
            price = close.iloc[i]
            atr_val = atr_1m.iloc[i]
            aligned = tf_alignment.iloc[i] >= p["n_confirm_tfs"]
            above_trend = fast_1m.iloc[i] > trend_1m.iloc[i]

            # BUY: 1m bull cross + higher TF alignment + trend + volume
            if (
                bull_cross_1m.iloc[i]
                and aligned
                and above_trend
                and vol_confirmed.iloc[i]
            ):
                stop = price - atr_val * p["atr_stop_mult"]
                target = price + atr_val * p["atr_target_mult"]
                rr = (target - price) / max(price - stop, 1e-8)
                signals.at[bars.index[i], "buy_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = (
                    f"tux_mtf:bull_cross:aligned={tf_alignment.iloc[i]}"
                )
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

            # SELL: 1m bear cross — close long
            if bear_cross_1m.iloc[i]:
                signals.at[bars.index[i], "sell_signal"] = True
                signals.at[bars.index[i], "exit_reason"] = "tux_mtf:bear_cross"

            # SHORT: mirror of buy with inverted conditions
            if p["allow_shorts"]:
                below_trend = fast_1m.iloc[i] < trend_1m.iloc[i]
                if (
                    bear_cross_1m.iloc[i]
                    and (tf_alignment.iloc[i] <= -p["n_confirm_tfs"])
                    and below_trend
                    and vol_confirmed.iloc[i]
                ):
                    stop = price + atr_val * p["atr_stop_mult"]
                    target = price - atr_val * p["atr_target_mult"]
                    rr = (price - target) / max(stop - price, 1e-8)
                    signals.at[bars.index[i], "short_signal"] = True
                    signals.at[bars.index[i], "entry_reason"] = (
                        f"tux_mtf:bear_cross:aligned={tf_alignment.iloc[i]}"
                    )
                    signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                    signals.at[bars.index[i], "target_price"] = round(target, 2)
                    signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

                # COVER short: bull cross
                if bull_cross_1m.iloc[i]:
                    signals.at[bars.index[i], "cover_signal"] = True
                    signals.at[bars.index[i], "exit_reason"] = "tux_mtf:cover_bull_cross"

        return self._add_strategy_id(signals)

    def _compute_tf_alignment(
        self,
        bars_1m: pd.DataFrame,
        mtf_bars: dict[str, pd.DataFrame],
        p: dict[str, Any],
    ) -> pd.Series:
        """
        For each 1m bar, count how many TFs are bullish (fast > slow EMA).
        Bullish TF = +1, bearish TF = -1.
        Uses forward-fill so signal persists until next completed TF bar.
        """
        alignment = pd.Series(0, index=bars_1m.index, dtype=int)

        for tf_label, tf_rule in [("5m", "5min"), ("15m", "15min"), ("1h", "1h")]:
            tf_df = mtf_bars.get(tf_label)
            if tf_df is None or tf_df.empty:
                # Resample from 1m bars if not pre-provided
                tf_df = self.resample_bars(bars_1m, tf_rule)
                if tf_df.empty:
                    continue

            tf_close = tf_df["close"]
            tf_fast = self.ema(tf_close, p["fast_ema"])
            tf_slow = self.ema(tf_close, p["slow_ema"])
            tf_bull = (tf_fast > tf_slow).astype(int) * 2 - 1  # +1 bull, -1 bear

            # Map completed TF bars back to 1m timestamps via forward-fill
            if "timestamp" in tf_df.columns:
                ts_series = pd.Series(tf_bull.values, index=pd.DatetimeIndex(tf_df["timestamp"]))
            else:
                ts_series = tf_bull

            ts_1m = pd.DatetimeIndex(bars_1m["timestamp"]) if "timestamp" in bars_1m.columns else bars_1m.index
            reindexed = ts_series.reindex(ts_1m, method="ffill").fillna(0).values
            alignment += reindexed.astype(int)

        return alignment


class TuxMTFLongOnly(TuxMTF):
    """TUX Multi-Timeframe — long only variant (no shorts, no covers)."""

    strategy_id: str = "tux_mtf_long_only"

    DEFAULT_PARAMS: dict[str, Any] = {
        **TuxMTF.DEFAULT_PARAMS,
        "allow_shorts": False,
        "n_confirm_tfs": 1,     # slightly more permissive for long-only
        "atr_stop_mult": 1.2,
        "atr_target_mult": 2.5,
    }
