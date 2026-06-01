"""
tradingview_institutional_screener.py — QuantumQuant

Screens a watchlist for institutional dip-reclaim candidates.

Pattern scored:
    sweep -> reclaim -> retest/support proximity -> absorption/engulf -> entry quality

Input:
    bars_by_symbol = {
        "AAPL": [
            {
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": 100000,
                "volume_ma20": 80000,
                "vwap": 100.1,
                "ema9": 100.0,
                "ema20": 99.8,
            },
            ...
        ]
    }

Output:
    Ranked List[ScreenerResult]
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from pivot_reversal_scanner import PivotReversalScanner
except Exception:  # pragma: no cover
    PivotReversalScanner = None  # type: ignore


logger = logging.getLogger(__name__)


SCREENER_MIN_SCORE = float(os.environ.get("SCREENER_MIN_SCORE", 0.50))
SCREENER_MIN_VOL_MULT = float(os.environ.get("SCREENER_MIN_VOL_MULT", 1.30))
SCREENER_MAX_RESULTS = int(os.environ.get("SCREENER_MAX_RESULTS", 20))
SCREENER_MIN_UPSIDE_TO_RESISTANCE_PCT = float(
    os.environ.get("SCREENER_MIN_UPSIDE_TO_RESISTANCE_PCT", 0.01)
)


@dataclass
class ScreenerResult:
    symbol: str
    score: float = 0.0
    setup_type: str = "none"

    close: float = 0.0
    support: float = 0.0
    resistance: float = 0.0

    dist_to_support_pct: float = 0.0
    upside_to_resistance_pct: float = 0.0

    volume_spike_ratio: float = 0.0
    reclaim_vwap: bool = False
    reclaim_ema9: bool = False
    reclaim_ema20: bool = False

    lower_wick_ratio: float = 0.0
    absorption_score: float = 0.0

    liquidity_sweep: bool = False
    pivot_reclaim: bool = False
    support_break_failed: bool = False
    support_touch: bool = False
    support_bounce: bool = False

    tags: List[str] = field(default_factory=list)


class InstitutionalScreener:
    """
    Scores bars from an arbitrary provider such as Alpaca, Polygon, or TradingView-derived data.

    The screener is intentionally defensive:
    - bad symbols do not crash the entire scan
    - missing PivotReversalScanner label fields default safely
    - malformed bars are skipped
    """

    def __init__(
        self,
        min_score: float = SCREENER_MIN_SCORE,
        min_vol_mult: float = SCREENER_MIN_VOL_MULT,
        max_results: int = SCREENER_MAX_RESULTS,
        min_upside_to_resistance_pct: float = SCREENER_MIN_UPSIDE_TO_RESISTANCE_PCT,
    ) -> None:
        self.min_score = float(min_score)
        self.min_vol_mult = float(min_vol_mult)
        self.max_results = int(max_results)
        self.min_upside_to_resistance_pct = float(min_upside_to_resistance_pct)

    def screen(
        self,
        bars_by_symbol: Dict[str, List[Dict[str, Any]]],
    ) -> List[ScreenerResult]:
        """
        Screen symbols and return ranked qualifying setups.

        Args:
            bars_by_symbol:
                Dictionary mapping symbol to ordered bars.
                Latest bar must be last.

        Returns:
            Ranked list of ScreenerResult objects.
        """
        if PivotReversalScanner is None:
            logger.error("PivotReversalScanner could not be imported.")
            return []

        results: List[ScreenerResult] = []

        for symbol, bars in bars_by_symbol.items():
            try:
                if not bars or len(bars) < 15:
                    continue

                scanner = PivotReversalScanner(symbol=symbol)

                for bar in bars:
                    if not isinstance(bar, dict):
                        continue
                    scanner.push(bar)

                lbl = getattr(scanner, "latest_label", None)
                if lbl is None:
                    logger.debug("No latest_label available for %s", symbol)
                    continue

                latest_bar = bars[-1]
                scored = self._score_bar(symbol, latest_bar, lbl)

                if scored.score >= self.min_score:
                    results.append(scored)

            except Exception:
                logger.exception("Institutional screener failed for %s", symbol)
                continue

        results.sort(key=lambda result: result.score, reverse=True)
        return results[: self.max_results]

    def _score_bar(
        self,
        symbol: str,
        bar: Dict[str, Any],
        lbl: Any,
    ) -> ScreenerResult:
        close = self._bar_float(bar, "close", 0.0)
        open_ = self._bar_float(bar, "open", close)
        high = self._bar_float(bar, "high", close)
        low = self._bar_float(bar, "low", close)
        volume = self._bar_float(bar, "volume", 0.0)

        volume_ma20 = max(self._bar_float(bar, "volume_ma20", volume), 1e-8)
        vwap = self._bar_float(bar, "vwap", close)
        ema9 = self._bar_float(bar, "ema9", close)
        ema20 = self._bar_float(bar, "ema20", close)

        support = self._label_float(lbl, "nearest_support", low)
        resistance = self._label_float(lbl, "nearest_resistance", high)

        price_range = max(high - low, 1e-8)
        body = abs(close - open_)

        lower_wick_ratio = max(0.0, min(open_, close) - low) / price_range
        volume_spike_ratio = volume / volume_ma20

        absorption_score = max(
            0.0,
            (volume_spike_ratio - 1.0) * max(0.0, 1.0 - body / price_range),
        )

        dist_to_support_pct = self._distance_to_support_pct(close, support)
        upside_to_resistance_pct = self._upside_to_resistance_pct(close, resistance)

        score = 0.0
        tags: List[str] = []

        # 1. Volume confirmation
        if volume_spike_ratio >= self.min_vol_mult:
            score += 0.20
            tags.append("vol_spike")

        # 2. Reclaim confirmation
        reclaim_vwap = close > vwap
        reclaim_ema9 = close > ema9
        reclaim_ema20 = close > ema20

        if reclaim_vwap:
            score += 0.15
            tags.append("reclaim_vwap")

        if reclaim_ema9:
            score += 0.10
            tags.append("reclaim_ema9")

        if reclaim_ema20:
            score += 0.10
            tags.append("reclaim_ema20")

        # 3. Structural reversal signals
        liquidity_sweep = self._label_bool(lbl, "liquidity_sweep_detected")
        pivot_reclaim = self._label_bool(lbl, "pivot_low_reclaim")
        support_break_failed = self._label_bool(lbl, "support_break_failed")
        missed_reversal_reclaim = self._label_bool(
            lbl,
            "missed_reversal_level_reclaim",
        )
        support_touch = self._label_bool(lbl, "support_touch")
        support_bounce = self._label_bool(lbl, "support_bounce")

        if liquidity_sweep:
            score += 0.20
            tags.append("liq_sweep")

        if pivot_reclaim:
            score += 0.15
            tags.append("pivot_reclaim")

        if support_break_failed:
            score += 0.15
            tags.append("sup_break_failed")

        if missed_reversal_reclaim:
            score += 0.15
            tags.append("missed_rev")

        if support_touch:
            score += 0.10
            tags.append("sup_touch")

        if support_bounce:
            score += 0.10
            tags.append("sup_bounce")

        # 4. Candle quality
        if lower_wick_ratio > 0.40:
            score += 0.10
            tags.append("lower_wick")

        if absorption_score > 0.50:
            score += 0.10
            tags.append("absorption")

        # 5. Prefer entries close to support
        if dist_to_support_pct <= 0.01:
            score += 0.10
            tags.append("near_support")
        elif dist_to_support_pct <= 0.025:
            score += 0.05
            tags.append("support_area")

        # 6. Penalize trades with poor upside into resistance
        if resistance > 0 and upside_to_resistance_pct < self.min_upside_to_resistance_pct:
            score -= 0.20
            tags.append("near_resistance")

        score = float(np.clip(score, 0.0, 1.0))

        return ScreenerResult(
            symbol=symbol,
            score=score,
            setup_type=str(getattr(lbl, "setup_type", "none") or "none"),
            close=close,
            support=support,
            resistance=resistance,
            dist_to_support_pct=dist_to_support_pct,
            upside_to_resistance_pct=upside_to_resistance_pct,
            volume_spike_ratio=volume_spike_ratio,
            reclaim_vwap=reclaim_vwap,
            reclaim_ema9=reclaim_ema9,
            reclaim_ema20=reclaim_ema20,
            lower_wick_ratio=lower_wick_ratio,
            absorption_score=absorption_score,
            liquidity_sweep=liquidity_sweep,
            pivot_reclaim=pivot_reclaim,
            support_break_failed=support_break_failed,
            support_touch=support_touch,
            support_bounce=support_bounce,
            tags=tags,
        )

    @staticmethod
    def _bar_float(
        bar: Dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        try:
            value = bar.get(key, default)
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _label_float(
        label: Any,
        attr: str,
        default: float,
    ) -> float:
        try:
            value = getattr(label, attr, None)
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _label_bool(
        label: Any,
        attr: str,
        default: bool = False,
    ) -> bool:
        try:
            return bool(getattr(label, attr, default))
        except Exception:
            return bool(default)

    @staticmethod
    def _distance_to_support_pct(
        close: float,
        support: float,
    ) -> float:
        if close <= 0 or support <= 0:
            return 1.0
        return max(0.0, abs(close - support) / close)

    @staticmethod
    def _upside_to_resistance_pct(
        close: float,
        resistance: float,
    ) -> float:
        if close <= 0 or resistance <= 0:
            return 0.0
        return max(0.0, (resistance - close) / close)

    def print_results(
        self,
        results: List[ScreenerResult],
    ) -> None:
        if not results:
            logger.info("No qualifying setups found.")
            return

        logger.info("=" * 90)
        logger.info(
            "%-8s %-6s %-24s %-9s %-9s %-9s %-7s %s",
            "SYMBOL",
            "SCORE",
            "SETUP",
            "CLOSE",
            "SUPPORT",
            "RESIST",
            "VSPK",
            "TAGS",
        )

        for result in results:
            logger.info(
                "%-8s %-6.2f %-24s %-9.4f %-9.4f %-9.4f %-7.2f %s",
                result.symbol,
                result.score,
                result.setup_type,
                result.close,
                result.support,
                result.resistance,
                result.volume_spike_ratio,
                " ".join(result.tags[:6]),
            )

        logger.info("=" * 90)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    random.seed(3)

    watchlist: Dict[str, List[Dict[str, Any]]] = {
        "CRM": [],
        "TE": [],
        "RKLB": [],
        "PLUG": [],
    }

    for symbol in watchlist:
        price = 190.0 if symbol == "CRM" else 50.0

        for i in range(80):
            if i == 55:
                price *= 0.985
            elif i == 57:
                price *= 1.012
            else:
                price *= 1.0 + random.gauss(0, 0.0015)

            high = price + random.uniform(0.05, 0.40)
            low = price - random.uniform(0.05, 0.50)
            close = price + random.gauss(0, 0.12)
            close = min(max(close, low), high)

            bar = {
                "timestamp": f"bar_{i}",
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": random.uniform(3e4, 4e5) * (2.5 if i == 55 else 1.0),
                "volume_ma20": 1e5,
                "vwap": price * 0.998,
                "ema9": price * 0.997,
                "ema20": price * 0.995,
            }

            watchlist[symbol].append(bar)

    screener = InstitutionalScreener(min_score=0.30)
    screened_results = screener.screen(watchlist)
    screener.print_results(screened_results)

    logger.info("tradingview_institutional_screener.py — OK")