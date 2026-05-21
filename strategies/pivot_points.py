"""
strategies/pivot_points.py

Pivot Points Strategy
Python translation of the Pivot Points Pine Script.

Logic
-----
Uses prior-day OHLC to compute floor-trader pivot levels:
    PP  = (H_prev + L_prev + C_prev) / 3
    R1  = 2*PP - L_prev
    R2  = PP + (H_prev - L_prev)
    R3  = H_prev + 2*(PP - L_prev)
    S1  = 2*PP - H_prev
    S2  = PP - (H_prev - L_prev)
    S3  = L_prev - 2*(H_prev - PP)

Signals:
    BUY  — price bounces off S1/S2 with confirmation, or breaks above R1 with volume
    SELL — price reaches R1/R2 or breaks below S1 (close long)
    SHORT — price breaks below S1 with volume and below-PP trend
    COVER — price recovers above S2

Optionally uses Fibonacci pivots when use_fib=True.

No lookahead: prior-day levels are computed from the previous complete session.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class PivotPointsStrategy(BaseStrategy):
    """Floor-trader pivot point strategy."""

    strategy_id: str = "pivot_points"

    DEFAULT_PARAMS: dict[str, Any] = {
        "use_fib": False,            # True = Fibonacci pivots, False = classical
        "bounce_buffer_pct": 0.002,  # max % away from level to count as "at level"
        "breakout_confirm_pct": 0.001,
        "volume_confirm_mult": 1.1,
        "volume_lookback": 20,
        "atr_stop_mult": 1.0,
        "atr_target_mult": 2.0,
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
        if not self._validate_bars(bars) or len(bars) < 30:
            return self._empty_signals(bars)

        ctx = context or {}
        p = self.params
        signals = self._empty_signals(bars)

        # ── Compute daily pivot levels ─────────────────────────────────────
        # Use context if pre-computed, else derive from bars
        levels = self._get_pivot_levels(bars, ctx, p)
        if not levels:
            return signals

        close = bars["close"]
        volume = bars["volume"]
        vol_avg = volume.rolling(p["volume_lookback"]).mean()
        atr_vals = self.atr(bars, 14)
        vwap_series = ctx.get("vwap_series") or self.vwap(bars)

        pp = levels["PP"]
        r1, r2 = levels["R1"], levels["R2"]
        s1, s2 = levels["S1"], levels["S2"]
        buf = p["bounce_buffer_pct"]
        bo_buf = p["breakout_confirm_pct"]

        for i in range(p["volume_lookback"] + 1, len(bars)):
            price = close.iloc[i]
            prev = close.iloc[i - 1]
            vol_ratio = (volume.iloc[i] / vol_avg.iloc[i]) if vol_avg.iloc[i] > 0 else 0.0
            atr_val = atr_vals.iloc[i]
            above_pp = price > pp
            above_vwap = price > vwap_series.iloc[i] if hasattr(vwap_series, "iloc") else price > vwap_series

            # ── BUY signals ───────────────────────────────────────────────
            # Pattern 1: bounce off S1 with confirmation
            at_s1 = abs(price - s1) / max(s1, 1e-8) <= buf
            recovering = price > prev
            if at_s1 and recovering and above_pp and vol_ratio >= p["volume_confirm_mult"]:
                stop = s2 - atr_val * p["atr_stop_mult"]
                target = r1
                rr = (target - price) / max(price - stop, 1e-8)
                signals.at[bars.index[i], "buy_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = f"pivot:bounce_s1:rr={rr:.1f}"
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

            # Pattern 2: breakout above R1
            broke_r1 = prev < r1 and price >= r1 * (1 + bo_buf)
            if broke_r1 and above_vwap and vol_ratio >= p["volume_confirm_mult"]:
                stop = pp - atr_val * 0.5
                target = r2
                rr = (target - price) / max(price - stop, 1e-8)
                signals.at[bars.index[i], "buy_signal"] = True
                signals.at[bars.index[i], "entry_reason"] = f"pivot:breakout_r1:rr={rr:.1f}"
                signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                signals.at[bars.index[i], "target_price"] = round(target, 2)
                signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

            # ── SELL signals (close long) ──────────────────────────────────
            at_r1 = abs(price - r1) / max(r1, 1e-8) <= buf
            at_r2 = abs(price - r2) / max(r2, 1e-8) <= buf
            broke_s1 = prev > s1 and price < s1 * (1 - bo_buf)

            if at_r1 or at_r2 or broke_s1:
                reason = "pivot:r1_target" if at_r1 else ("pivot:r2_target" if at_r2 else "pivot:broke_s1")
                signals.at[bars.index[i], "sell_signal"] = True
                signals.at[bars.index[i], "exit_reason"] = reason

            # ── SHORT / COVER ─────────────────────────────────────────────
            if p["allow_shorts"]:
                at_r2_for_short = abs(price - r2) / max(r2, 1e-8) <= buf
                rejecting = price < prev
                if at_r2_for_short and rejecting and not above_pp and vol_ratio >= p["volume_confirm_mult"]:
                    stop = r2 + atr_val * p["atr_stop_mult"]
                    target = s1
                    rr = (price - target) / max(stop - price, 1e-8)
                    signals.at[bars.index[i], "short_signal"] = True
                    signals.at[bars.index[i], "entry_reason"] = f"pivot:short_r2:rr={rr:.1f}"
                    signals.at[bars.index[i], "stop_price"] = round(stop, 2)
                    signals.at[bars.index[i], "target_price"] = round(target, 2)
                    signals.at[bars.index[i], "risk_reward"] = round(rr, 2)

                at_s2 = abs(price - s2) / max(s2, 1e-8) <= buf
                if at_s2 and price > prev:
                    signals.at[bars.index[i], "cover_signal"] = True
                    signals.at[bars.index[i], "exit_reason"] = "pivot:cover_s2"

        return self._add_strategy_id(signals)

    def _get_pivot_levels(
        self,
        bars: pd.DataFrame,
        ctx: dict[str, Any],
        p: dict[str, Any],
    ) -> dict[str, float]:
        """Compute pivot levels from prior-day OHLC."""
        # Use context-provided levels if available
        if "pivot_pp" in ctx:
            return {
                "PP": ctx["pivot_pp"],
                "R1": ctx.get("pivot_r1", ctx["pivot_pp"]),
                "R2": ctx.get("pivot_r2", ctx["pivot_pp"]),
                "R3": ctx.get("pivot_r3", ctx["pivot_pp"]),
                "S1": ctx.get("pivot_s1", ctx["pivot_pp"]),
                "S2": ctx.get("pivot_s2", ctx["pivot_pp"]),
                "S3": ctx.get("pivot_s3", ctx["pivot_pp"]),
            }

        # Derive from daily bars (resample if needed)
        daily = self.resample_bars(bars, "1D")
        if len(daily) < 2:
            return {}

        # Use the second-to-last row (completed prior day)
        prev = daily.iloc[-2]
        H, L, C = prev["high"], prev["low"], prev["close"]
        PP = (H + L + C) / 3.0

        if p["use_fib"]:
            rng = H - L
            R1 = PP + 0.382 * rng
            R2 = PP + 0.618 * rng
            R3 = PP + 1.000 * rng
            S1 = PP - 0.382 * rng
            S2 = PP - 0.618 * rng
            S3 = PP - 1.000 * rng
        else:
            R1 = 2 * PP - L
            R2 = PP + (H - L)
            R3 = H + 2 * (PP - L)
            S1 = 2 * PP - H
            S2 = PP - (H - L)
            S3 = L - 2 * (H - PP)

        return {
            "PP": PP, "R1": R1, "R2": R2, "R3": R3,
            "S1": S1, "S2": S2, "S3": S3,
        }
