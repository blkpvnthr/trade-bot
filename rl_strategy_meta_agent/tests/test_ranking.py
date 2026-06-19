from rl_meta.ranking import rank_strategies, composite_score
from rl_meta.types import Metrics, RollingMetrics, StrategyMeta, Tier


def meta(sid):
    return StrategyMeta(sid, sid.title(), "h", "2023-01-01", f"{sid}.txt")


def good():
    return Metrics(trade_count=120, win_rate=0.68, profit_factor=2.6, expectancy=150,
                   max_drawdown=0.10, avg_hold_days=4)


def weak():
    return Metrics(trade_count=120, win_rate=0.45, profit_factor=1.1, expectancy=5,
                   max_drawdown=0.35, avg_hold_days=4)


def test_eligible_outrank_ineligible():
    ranked = rank_strategies([
        (meta("weak"), weak(), RollingMetrics()),
        (meta("good"), good(), RollingMetrics()),
    ])
    assert ranked[0].strategy_id == "good"
    assert ranked[0].rank == 1 and ranked[1].rank == 2
    assert ranked[0].eligible and not ranked[1].eligible


def test_score_monotonic_in_quality():
    better = composite_score(good(), RollingMetrics())
    worse = composite_score(weak(), RollingMetrics())
    assert better > worse


def test_low_drawdown_scores_higher():
    low = Metrics(trade_count=50, win_rate=0.65, profit_factor=2.2,
                  expectancy=100, max_drawdown=0.05, avg_hold_days=4)
    high = Metrics(trade_count=50, win_rate=0.65, profit_factor=2.2,
                   expectancy=100, max_drawdown=0.40, avg_hold_days=4)
    assert composite_score(low, RollingMetrics()) > composite_score(high, RollingMetrics())


def test_ranks_are_contiguous():
    rows = [(meta(f"s{i}"), good(), RollingMetrics()) for i in range(4)]
    ranked = rank_strategies(rows)
    assert [r.rank for r in ranked] == [1, 2, 3, 4]
