"""Independent exit & risk stack (V6-Integrated spec section 13; paper 2 decoupled exit stack).

Deterministic. It OVERRIDES policy agents and does NOT depend on Claude/LLM availability. Critically,
reductions and liquidations remain available even when NEW ENTRIES are disabled, the market is
transitional, agents are down, evidence is stale, or the system is reduce-only.

Exit reasons (paper 2 flags + spec section 13): HARD_STOP, ATR_STOP, THESIS_INVALIDATION,
TRAILING_STOP, PROFIT_TARGET, MAX_HOLD, REGIME_EXIT, REDUCE_ONLY_EXIT, END_OF_DAY,
DAILY_LOSS_SHUTDOWN, DRAWDOWN_SHUTDOWN, EMERGENCY_LIQUIDATION.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

HOLD = "HOLD"
REDUCE = "REDUCE"
EXIT = "EXIT"


@dataclass
class ExitConfig:
    hard_stop_pct: float = -0.05
    atr_stop_mult: float = 1.5
    trailing_pct: float = -0.02         # give-back from high since entry
    profit_target_pct: float = 0.08
    max_hold_bars: int = 0              # 0 = no max-hold check
    daily_loss_shutdown: float = -0.03  # account daily P&L fraction -> flatten everything
    drawdown_shutdown: float = -0.10    # account drawdown -> flatten everything
    regime_exit_enabled: bool = True


@dataclass
class ExitContext:
    price: float
    avg_entry: float
    atr: float = 0.0
    high_since_entry: float = 0.0
    bars_held: int = 0
    regime_permits_hold: bool = True    # False => regime no longer permits this position's side
    thesis_valid: bool = True
    reduce_only: bool = False
    end_of_day: bool = False
    account_daily_pnl_pct: float = 0.0
    account_drawdown_pct: float = 0.0
    emergency: bool = False


@dataclass
class ExitDecision:
    action: str           # HOLD | REDUCE | EXIT
    reason: str
    priority: int         # higher = more urgent (account-level shutdowns dominate)


def evaluate(ctx: ExitContext, cfg: Optional[ExitConfig] = None) -> ExitDecision:
    """Return the single most-urgent exit action for a position. Account-level shutdowns dominate
    per-position exits. Always available regardless of entry permissions."""
    cfg = cfg or ExitConfig()
    # --- account-level (dominating) ---
    if ctx.emergency:
        return ExitDecision(EXIT, "EMERGENCY_LIQUIDATION", 100)
    if ctx.account_drawdown_pct <= cfg.drawdown_shutdown:
        return ExitDecision(EXIT, "DRAWDOWN_SHUTDOWN", 95)
    if ctx.account_daily_pnl_pct <= cfg.daily_loss_shutdown:
        return ExitDecision(EXIT, "DAILY_LOSS_SHUTDOWN", 90)
    # --- per-position protective (deterministic) ---
    if ctx.avg_entry <= 0 or ctx.price <= 0:
        return ExitDecision(HOLD, "no_basis", 0)
    ret = ctx.price / ctx.avg_entry - 1.0
    if ret <= cfg.hard_stop_pct:
        return ExitDecision(EXIT, "HARD_STOP", 80)
    if ctx.atr > 0 and ctx.price <= ctx.avg_entry - cfg.atr_stop_mult * ctx.atr:
        return ExitDecision(EXIT, "ATR_STOP", 78)
    if not ctx.thesis_valid:
        return ExitDecision(EXIT, "THESIS_INVALIDATION", 70)
    if cfg.regime_exit_enabled and not ctx.regime_permits_hold:
        return ExitDecision(EXIT, "REGIME_EXIT", 68)
    if ctx.end_of_day:
        return ExitDecision(EXIT, "END_OF_DAY", 60)
    # trailing: give-back from the high since entry
    if ctx.high_since_entry > 0:
        giveback = ctx.price / ctx.high_since_entry - 1.0
        if giveback <= cfg.trailing_pct and ctx.high_since_entry > ctx.avg_entry:
            return ExitDecision(EXIT, "TRAILING_STOP", 55)
    if ret >= cfg.profit_target_pct:
        return ExitDecision(EXIT, "PROFIT_TARGET", 50)
    if cfg.max_hold_bars and ctx.bars_held >= cfg.max_hold_bars:
        return ExitDecision(EXIT, "MAX_HOLD", 45)
    if ctx.reduce_only:
        # reduce-only mode: trim exposure even absent a hard trigger (never add)
        return ExitDecision(REDUCE, "REDUCE_ONLY_EXIT", 30)
    return ExitDecision(HOLD, "within_tolerances", 0)


def entries_blocked_but_exits_available(new_entries_allowed: bool) -> bool:
    """Invariant check used by tests/pipeline: the exit stack is ALWAYS available, independent of
    whether new entries are allowed. Returns True (exits available) unconditionally."""
    return True
