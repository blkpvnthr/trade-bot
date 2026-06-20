"""Shared data types for the RL strategy meta-agent.

Research-only. No broker types here on purpose.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Optional


class Tier(str, enum.Enum):
    A = "A"          # win_rate >= 60% AND profit_factor >= 2.0
    B = "B"          # win_rate >= 60% AND profit_factor < 2.0
    C = "C"          # win_rate < 60%  AND profit_factor >= 2.0
    REJECT = "REJECT"  # win_rate < 60% AND profit_factor < 2.0


class Status(str, enum.Enum):
    PENDING = "pending"      # parsed, not yet evaluated
    ELIGIBLE = "eligible"    # passed admission + OOS validation
    REJECTED = "rejected"    # failed admission or validation
    NOT_EXECUTABLE = "parsed_but_not_executable"  # no adapter; excluded from RL


@dataclass
class StrategyMeta:
    """Metadata extracted from a PineScript file."""
    strategy_id: str
    strategy_name: str
    version_hash: str
    date_added: str                       # ISO date
    source_file: str
    inputs: dict = field(default_factory=dict)
    entry_logic: list = field(default_factory=list)
    exit_logic: list = field(default_factory=list)
    stop_loss: list = field(default_factory=list)
    take_profit: list = field(default_factory=list)
    timeframe_hints: list = field(default_factory=list)
    indicators: list = field(default_factory=list)
    declares_strategy: bool = True   # False for indicator()/study scripts (not tradeable)
    status: str = Status.PENDING.value
    # Execution status (filled by the adapter layer + backtest pipeline).
    executable: bool = False
    adapter_type: Optional[str] = None
    unsupported_reason: Optional[str] = None
    tested_symbols: list = field(default_factory=list)
    tested_timeframe: Optional[str] = None
    # Source-of-truth tracking (filled by the backtest pipeline).
    evaluation_source: Optional[str] = None        # see backtest.EVAL_SOURCES
    tv_validated_symbols: list = field(default_factory=list)
    missing_tv_symbols: list = field(default_factory=list)
    strategy_confidence: float = 0.0               # 0..100 trustworthiness
    # Independence diagnostic (population-phase; see correlation.py).
    family_override: Optional[str] = None          # from a // family: X comment
    strategy_family: Optional[str] = None          # resolved family tag
    independence_score: float = 0.0                # 1 - avg|corr| to others
    avg_correlation: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeRecord:
    """One closed trade from a backtest.

    `pnl` is the EFFECTIVE dollar PnL used by all metrics. For imported
    TradingView trades it is the position-size-NORMALIZED PnL when normalization
    is on; `raw_pnl_usd` keeps the unscaled value for audit. `return_pct` is
    never scaled (size doesn't change percentage returns)."""
    entry_date: str       # ISO
    exit_date: str        # ISO
    entry_price: float
    exit_price: float
    pnl: float            # effective dollars (normalized for imported TV trades)
    return_pct: float     # fractional return -- NEVER scaled by normalization
    hold_days: float
    # Position-size normalization (imported TradingView trades only).
    raw_qty: float = 0.0
    normalized_qty: float = 0.0
    scale_factor: float = 1.0
    normalized: bool = False
    raw_pnl_usd: float = 0.0     # unscaled PnL, for audit/reporting only


@dataclass
class Metrics:
    """Lifetime (or windowed) performance metrics for one strategy."""
    trade_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0        # fractional, positive number (0.2 == 20%)
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_days: float = 0.0
    total_return: float = 0.0        # fractional
    cagr: float = 0.0                # fractional, 0 if span < ~1 month
    # Equity-curve audit (from a valid initial-equity-based curve).
    initial_equity: float = 0.0
    ending_equity: float = 0.0
    min_equity: float = 0.0
    blown_up: bool = False           # equity hit <=0 at some point -> ruin

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RollingMetrics:
    """Trailing-window metrics keyed by window length in days."""
    window_30: Metrics = field(default_factory=Metrics)
    window_60: Metrics = field(default_factory=Metrics)
    window_90: Metrics = field(default_factory=Metrics)

    def to_dict(self) -> dict:
        return {
            "30": self.window_30.to_dict(),
            "60": self.window_60.to_dict(),
            "90": self.window_90.to_dict(),
        }


@dataclass
class RankedStrategy:
    strategy_id: str
    strategy_name: str
    tier: str
    rank: int
    composite_score: float
    eligible: bool
    metrics: Metrics
    rolling: RollingMetrics
    reason: Optional[str] = None
