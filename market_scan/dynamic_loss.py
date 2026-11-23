"""Dynamic volatility-adjusted protective loss monitor -- decision core (Stage B).

Replaces the unconditional -0.5% liquidation with a per-symbol ATR-adjusted
threshold and a confirmation state machine, WITHOUT ever widening a losing
position's authorized risk:

  * ``-0.5%`` (``warning_pct``) is an early WARNING, not a liquidation trigger.
  * The adaptive soft threshold = ``clamp(atr_pct * atr_multiplier,
    adaptive_min_pct, adaptive_max_pct)`` -- bounded by the authorized ceiling,
    so higher volatility never pushes the exit past ``adaptive_max_pct``.
  * A discretionary protective exit fires only when the loss is beyond the
    adaptive threshold AND deterioration is confirmed across
    ``>= confirmation_seconds`` and ``>= min_observations`` distinct fresh
    observations. Recovery within the window cancels the pending exit.
  * An independent hard-stop breach exits IMMEDIATELY (no confirmation delay).
  * Missing/unreliable ATR falls back to the conservative floor (``warning_pct``)
    -- never a wider threshold.

Pure + deterministic: the caller passes ``now`` (and the sign-correct
``unrealized_plpc`` from the broker), so there is no clock or I/O here and it is
exhaustively unit-testable. State is per-symbol confirmation progress only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def adaptive_threshold(atr_pct: Optional[float], *, multiplier: float,
                       min_pct: float, max_pct: float) -> float:
    """Volatility-adjusted loss threshold magnitude (a positive fraction),
    clamped to [min_pct, max_pct]. Missing/invalid ATR -> the conservative floor
    (never a wider threshold)."""
    lo, hi = min(min_pct, max_pct), max(min_pct, max_pct)
    try:
        a = float(atr_pct)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(a) or a <= 0.0:
        return lo
    return max(lo, min(hi, a * float(multiplier)))


@dataclass
class LossDecision:
    action: str          # HOLD | WATCH | RECOVERING | EXIT_PENDING | EXIT | HARD_EXIT
    should_exit: bool
    threshold: float     # the adaptive threshold magnitude used this evaluation
    loss_pct: float      # the position's loss magnitude this evaluation (>=0 losing)
    reason: str = ""


class DynamicLossMonitor:
    def __init__(self, *, warning_pct: float = 0.005, adaptive_min_pct: float = 0.005,
                 adaptive_max_pct: float = 0.020, atr_multiplier: float = 1.5,
                 confirmation_seconds: float = 30.0, min_observations: int = 3,
                 hard_stop_pct: Optional[float] = None) -> None:
        self.warning_pct = float(warning_pct)
        self.adaptive_min_pct = float(adaptive_min_pct)
        self.adaptive_max_pct = float(adaptive_max_pct)
        self.atr_multiplier = float(atr_multiplier)
        self.confirmation_seconds = float(confirmation_seconds)
        self.min_observations = max(1, int(min_observations))
        self.hard_stop_pct = float(hard_stop_pct) if hard_stop_pct is not None else None
        # symbol -> {"first_ts": float, "last_ts": float, "obs": int}
        self._pending: dict[str, dict] = {}

    def reset(self, symbol: str) -> None:
        self._pending.pop(str(symbol).upper(), None)

    def observe(self, symbol: str, *, plpc: float, atr_pct: Optional[float],
                deteriorating: bool, now: float, side: str = "long") -> LossDecision:
        """Evaluate one fresh observation of a position and return the decision.

        ``plpc`` is the broker's sign-correct unrealized P/L fraction (negative ==
        losing) for both long and short positions, so the loss magnitude is
        ``-plpc`` regardless of ``side`` (accepted for API completeness)."""
        sym = str(symbol).upper()
        try:
            loss_pct = -float(plpc)
        except (TypeError, ValueError):
            loss_pct = 0.0
        threshold = adaptive_threshold(
            atr_pct, multiplier=self.atr_multiplier,
            min_pct=self.adaptive_min_pct, max_pct=self.adaptive_max_pct)

        # 1. Independent hard stop -> immediate exit, no confirmation.
        if self.hard_stop_pct is not None and loss_pct >= self.hard_stop_pct:
            self._pending.pop(sym, None)
            return LossDecision("HARD_EXIT", True, threshold, loss_pct,
                                reason=f"hard_stop {loss_pct:.4f}>={self.hard_stop_pct:.4f}")

        # 2. Healthy / above warning -> hold; cancel any pending confirmation.
        if loss_pct < self.warning_pct:
            had = self._pending.pop(sym, None)
            return LossDecision("RECOVERING" if had else "HOLD", False, threshold, loss_pct)

        # 3. Between warning and adaptive threshold -> WATCH (early warning only).
        if loss_pct < threshold:
            had = self._pending.pop(sym, None)
            return LossDecision("RECOVERING" if had else "WATCH", False, threshold, loss_pct,
                                reason="warning breached; within adaptive risk budget")

        # 4. Beyond the adaptive threshold.
        if not deteriorating:
            # Hold within the authorized risk budget while there is no deterioration
            # evidence; cancel any pending confirmation (conditions improved).
            self._pending.pop(sym, None)
            return LossDecision("WATCH", False, threshold, loss_pct,
                                reason="beyond adaptive threshold but not deteriorating")

        # 5. Beyond threshold AND deteriorating -> confirmation window.
        rec = self._pending.get(sym)
        if rec is None:
            self._pending[sym] = {"first_ts": now, "last_ts": now, "obs": 1}
            return LossDecision("EXIT_PENDING", False, threshold, loss_pct,
                                reason="confirmation window started")
        # Count only distinct fresh observations (strictly newer timestamps).
        if now > rec["last_ts"]:
            rec["obs"] += 1
            rec["last_ts"] = now
        elapsed = now - rec["first_ts"]
        if elapsed >= self.confirmation_seconds and rec["obs"] >= self.min_observations:
            self._pending.pop(sym, None)
            return LossDecision("EXIT", True, threshold, loss_pct,
                                reason=f"confirmed deterioration {elapsed:.0f}s/{rec['obs']}obs")
        return LossDecision("EXIT_PENDING", False, threshold, loss_pct,
                            reason=f"confirming {elapsed:.0f}s/{rec['obs']}obs")
