from rl_meta.performance import (compute_metrics, rolling_metrics, PerformanceDB)
from rl_meta.types import TradeRecord, RollingMetrics, Metrics


def trades():
    return [
        TradeRecord("2023-01-02", "2023-01-05", 100, 110, 1000.0, 0.10, 3),
        TradeRecord("2023-02-02", "2023-02-04", 110, 104, -600.0, -0.054, 2),
        TradeRecord("2023-03-02", "2023-03-09", 104, 115, 1100.0, 0.105, 7),
        TradeRecord("2023-03-20", "2023-03-22", 115, 112, -300.0, -0.026, 2),
    ]


def test_metrics_basic():
    m = compute_metrics(trades())
    assert m.trade_count == 4
    assert abs(m.win_rate - 0.5) < 1e-9
    assert abs(m.profit_factor - (2100 / 900)) < 1e-3
    assert m.expectancy == round((1000 - 600 + 1100 - 300) / 4, 2)
    assert m.max_drawdown >= 0


def test_empty_metrics():
    m = compute_metrics([])
    assert m.trade_count == 0 and m.profit_factor == 0.0


def test_profit_factor_no_losses():
    only_wins = [TradeRecord("2023-01-01", "2023-01-02", 100, 110, 500, 0.05, 1)]
    assert compute_metrics(only_wins).profit_factor == float("inf")


def test_rolling_window_filters_by_date():
    r = rolling_metrics(trades(), as_of="2023-03-22")
    # trailing 30d ending 2023-03-22 -> only the two March trades
    assert r.window_30.trade_count == 2
    assert r.window_90.trade_count >= r.window_30.trade_count


def test_db_record_and_latest(tmp_path):
    db = PerformanceDB(tmp_path / "perf.db")
    m = compute_metrics(trades())
    db.record("s1", "2023-04-01", m, RollingMetrics(), rank=1, tier="A",
              status="eligible")
    db.record("s1", "2023-04-02", m, RollingMetrics(), rank=2, tier="B",
              status="eligible")
    latest = db.latest("s1")
    assert latest["date"] == "2023-04-02" and latest["rank"] == 2
    assert latest["lifetime"]["trade_count"] == 4
    assert len(db.all_latest()) == 1
