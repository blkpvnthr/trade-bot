from rl_meta.risk import (RiskEngine, RiskLimits, PortfolioState, OrderRequest)


def fresh_portfolio():
    return PortfolioState(equity=100_000, start_equity=100_000,
                          peak_equity=100_000, open_positions=0, trades_today=0)


def test_position_size_clamp():
    d = RiskEngine().evaluate(OrderRequest("x", target_pct=0.90), fresh_portfolio())
    assert d.approved
    assert d.approved_pct == RiskLimits().max_position_pct
    assert "max_position_size_clamped" in d.violations


def test_drawdown_kill_switch_halts():
    p = PortfolioState(78_000, 100_000, 100_000, 0, 0)
    d = RiskEngine().evaluate(OrderRequest("x", 0.1), p)
    assert d.halt and not d.approved and d.approved_pct == 0.0


def test_daily_loss_kill_switch():
    p = PortfolioState(99_000, 100_000, 100_000, 0, 0, realized_today=-2_500)
    d = RiskEngine().evaluate(OrderRequest("x", 0.1), p)
    assert d.halt


def test_stale_data_veto():
    d = RiskEngine().evaluate(OrderRequest("x", 0.1, data_age_seconds=10_000),
                              fresh_portfolio())
    assert not d.approved and "stale_data_check" in d.violations


def test_spread_and_slippage_veto():
    eng = RiskEngine()
    d1 = eng.evaluate(OrderRequest("x", 0.1, spread_pct=0.05), fresh_portfolio())
    d2 = eng.evaluate(OrderRequest("x", 0.1, slippage_pct=0.05), fresh_portfolio())
    assert "spread_check" in d1.violations
    assert "slippage_check" in d2.violations


def test_max_open_positions_blocks_new():
    p = PortfolioState(100_000, 100_000, 100_000, open_positions=4, trades_today=0)
    d = RiskEngine().evaluate(OrderRequest("x", 0.1), p)
    assert not d.approved and "max_open_positions" in d.violations


def test_max_trades_per_day():
    p = PortfolioState(100_000, 100_000, 100_000, open_positions=0, trades_today=6)
    d = RiskEngine().evaluate(OrderRequest("x", 0.1), p)
    assert not d.approved and "max_trades_per_day" in d.violations


def test_agent_cannot_override_hard_limit():
    # Even a 'valid' small order is refused once the kill switch trips.
    p = PortfolioState(75_000, 100_000, 100_000, 0, 0)
    d = RiskEngine().evaluate(OrderRequest("x", 0.01), p)
    assert d.halt and d.approved_pct == 0.0
