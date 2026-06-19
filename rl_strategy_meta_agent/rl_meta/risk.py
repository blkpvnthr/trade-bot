"""Risk engine. The FINAL authority over the RL agent.

The RL agent proposes; the risk engine disposes. It may clamp or veto any
allocation but the agent can never override a hard limit. Research-only:
all checks operate on the simulated portfolio, never a broker.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    max_daily_loss: float = 0.02          # 2% of starting equity per day
    max_drawdown: float = 0.20            # 20% peak-to-trough -> halt
    max_position_pct: float = 0.25        # 25% of equity in one strategy
    max_open_positions: int = 4
    max_trades_per_day: int = 6
    min_liquidity: float = 0.0            # placeholder hooks for real data
    max_spread_pct: float = 0.01
    max_slippage_pct: float = 0.005
    max_stale_seconds: float = 900.0      # 15 min


@dataclass
class PortfolioState:
    equity: float
    start_equity: float
    peak_equity: float
    open_positions: int
    trades_today: int
    realized_today: float = 0.0


@dataclass
class OrderRequest:
    strategy_id: str
    target_pct: float                     # requested fraction of equity
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    data_age_seconds: float = 0.0
    liquidity: float = 1e9


@dataclass
class RiskDecision:
    approved: bool
    approved_pct: float                   # clamped allocation fraction
    halt: bool = False                    # kill-switch tripped
    violations: list = field(default_factory=list)


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def kill_switch(self, p: PortfolioState) -> tuple:
        """Hard halts that stop ALL new risk. Returns (halt, reasons)."""
        L, reasons = self.limits, []
        dd = (p.peak_equity - p.equity) / p.peak_equity if p.peak_equity > 0 else 0.0
        if dd >= L.max_drawdown:
            reasons.append(f"max_drawdown {dd:.3f}>={L.max_drawdown}")
        daily_loss = -p.realized_today / p.start_equity if p.start_equity > 0 else 0.0
        if daily_loss >= L.max_daily_loss:
            reasons.append(f"max_daily_loss {daily_loss:.3f}>={L.max_daily_loss}")
        return (len(reasons) > 0, reasons)

    def evaluate(self, order: OrderRequest, p: PortfolioState) -> RiskDecision:
        L = self.limits
        halt, reasons = self.kill_switch(p)
        if halt:
            return RiskDecision(False, 0.0, halt=True, violations=reasons)

        v = []
        # Market-quality vetoes (full reject, not clamp).
        if order.spread_pct > L.max_spread_pct:
            v.append("spread_check")
        if order.slippage_pct > L.max_slippage_pct:
            v.append("slippage_check")
        if order.data_age_seconds > L.max_stale_seconds:
            v.append("stale_data_check")
        if order.liquidity < L.min_liquidity:
            v.append("liquidity_check")
        if p.trades_today >= L.max_trades_per_day:
            v.append("max_trades_per_day")
        if p.open_positions >= L.max_open_positions and order.target_pct > 0:
            v.append("max_open_positions")
        if v:
            return RiskDecision(False, 0.0, violations=v)

        # Position-size clamp (never a hard veto, just capped).
        approved_pct = max(0.0, min(order.target_pct, L.max_position_pct))
        clamped = approved_pct < order.target_pct
        if clamped:
            v.append("max_position_size_clamped")
        return RiskDecision(True, approved_pct, violations=v)


def _demo():
    eng = RiskEngine()
    p = PortfolioState(equity=100_000, start_equity=100_000, peak_equity=100_000,
                       open_positions=0, trades_today=0)
    # Oversized request -> clamped to 25%.
    d = eng.evaluate(OrderRequest("x", target_pct=0.80), p)
    assert d.approved and abs(d.approved_pct - 0.25) < 1e-9
    assert "max_position_size_clamped" in d.violations
    # Stale data -> veto.
    d2 = eng.evaluate(OrderRequest("x", 0.1, data_age_seconds=9999), p)
    assert not d2.approved and "stale_data_check" in d2.violations
    # Drawdown breach -> halt.
    p2 = PortfolioState(80_000, 100_000, 100_000, 0, 0)
    d3 = eng.evaluate(OrderRequest("x", 0.1), p2)
    assert d3.halt and not d3.approved
    print("risk._demo OK")


if __name__ == "__main__":
    _demo()
