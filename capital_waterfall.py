#!/usr/bin/env python3
"""
capital_waterfall.py

Tracks realized options profits and allocates them across three buckets:

    50-70% → Buffett Accumulation Layer (buy-and-hold quality stocks)
    20-40% → Alpha reserve (replenish call buying power)
    10%    → Cash reserve

Outputs:
    daily_system_outputs/capital_waterfall.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)

WATERFALL_PATH = DAILY_OUTPUTS / "capital_waterfall.json"
WATERFALL_EVENTS_PATH = (
    PROJECT_ROOT / "logs" / "execution" / DATA_TAG / f"{DATA_TAG}_waterfall_events.jsonl"
)
WATERFALL_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class WaterfallConfig:
    bal_pct: float = float(os.getenv("OPTIONS_PROFIT_TO_BAL_PCT", "0.60"))
    alpha_pct: float = float(os.getenv("OPTIONS_PROFIT_TO_ALPHA_PCT", "0.30"))
    cash_pct: float = float(os.getenv("OPTIONS_PROFIT_TO_CASH_PCT", "0.10"))

    def __post_init__(self) -> None:
        total = self.bal_pct + self.alpha_pct + self.cash_pct
        if abs(total - 1.0) > 0.01:
            # Normalize so allocations always sum to 100%
            self.bal_pct /= total
            self.alpha_pct /= total
            self.cash_pct /= total


@dataclass
class WaterfallState:
    date_tag: str = DATA_TAG
    # Cumulative today
    realized_options_profit_today: float = 0.0
    allocated_to_bal_today: float = 0.0
    allocated_to_alpha_today: float = 0.0
    allocated_to_cash_today: float = 0.0
    # Running balances (carry across sessions)
    bal_reserve: float = 0.0       # available for BAL stock buying
    alpha_reserve: float = 0.0     # available for Alpha call buying
    cash_reserve: float = 0.0      # parked cash from profit allocation
    # History
    events: list[dict] = field(default_factory=list)
    updated_at_utc: str = ""


class CapitalWaterfall:
    """
    Receives realized profit notifications from closed call trades and
    allocates them across BAL, Alpha reserve, and cash.
    """

    def __init__(self, config: WaterfallConfig | None = None):
        self.config = config or WaterfallConfig()
        self._state = self._load()

    def _load(self) -> WaterfallState:
        if WATERFALL_PATH.exists():
            try:
                raw = json.loads(WATERFALL_PATH.read_text())
                state = WaterfallState(**{
                    k: raw[k] for k in WaterfallState.__dataclass_fields__ if k in raw
                })
                # Reset daily counters on new date
                if state.date_tag != DATA_TAG:
                    state.date_tag = DATA_TAG
                    state.realized_options_profit_today = 0.0
                    state.allocated_to_bal_today = 0.0
                    state.allocated_to_alpha_today = 0.0
                    state.allocated_to_cash_today = 0.0
                    # Reserves carry over
                return state
            except Exception:
                pass
        return WaterfallState()

    def _save(self) -> None:
        self._state.updated_at_utc = datetime.now(timezone.utc).isoformat()
        try:
            WATERFALL_PATH.write_text(
                json.dumps(asdict(self._state), indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    def _log_event(self, event: str, payload: dict) -> None:
        try:
            line = json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "payload": payload,
            })
            with open(WATERFALL_EVENTS_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def allocate(self, realized_profit: float, source: str = "alpha_call") -> dict:
        """
        Allocate a realized profit amount across the three buckets.
        Returns allocation breakdown.
        """
        if realized_profit <= 0:
            return {}

        to_bal = round(realized_profit * self.config.bal_pct, 2)
        to_alpha = round(realized_profit * self.config.alpha_pct, 2)
        to_cash = round(realized_profit * self.config.cash_pct, 2)

        self._state.realized_options_profit_today += realized_profit
        self._state.allocated_to_bal_today += to_bal
        self._state.allocated_to_alpha_today += to_alpha
        self._state.allocated_to_cash_today += to_cash
        self._state.bal_reserve += to_bal
        self._state.alpha_reserve += to_alpha
        self._state.cash_reserve += to_cash

        allocation = {
            "source": source,
            "realized_profit": realized_profit,
            "to_bal": to_bal,
            "to_alpha": to_alpha,
            "to_cash": to_cash,
            "bal_reserve_total": self._state.bal_reserve,
            "alpha_reserve_total": self._state.alpha_reserve,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._state.events.append(allocation)
        self._save()
        self._log_event("waterfall_allocation", allocation)
        return allocation

    def consume_bal(self, amount: float) -> float:
        """Consume from BAL reserve. Returns amount actually available."""
        available = min(amount, self._state.bal_reserve)
        if available > 0:
            self._state.bal_reserve -= available
            self._save()
            self._log_event("bal_reserve_consumed", {"amount": available, "remaining": self._state.bal_reserve})
        return available

    def consume_alpha(self, amount: float) -> float:
        """Consume from Alpha reserve. Returns amount actually available."""
        available = min(amount, self._state.alpha_reserve)
        if available > 0:
            self._state.alpha_reserve -= available
            self._save()
            self._log_event("alpha_reserve_consumed", {"amount": available, "remaining": self._state.alpha_reserve})
        return available

    @property
    def bal_reserve(self) -> float:
        return self._state.bal_reserve

    @property
    def alpha_reserve(self) -> float:
        return self._state.alpha_reserve

    @property
    def summary(self) -> dict:
        return {
            "date_tag": DATA_TAG,
            "realized_today": self._state.realized_options_profit_today,
            "bal_reserve": self._state.bal_reserve,
            "alpha_reserve": self._state.alpha_reserve,
            "cash_reserve": self._state.cash_reserve,
            "allocation_today": {
                "bal": self._state.allocated_to_bal_today,
                "alpha": self._state.allocated_to_alpha_today,
                "cash": self._state.allocated_to_cash_today,
            },
        }
