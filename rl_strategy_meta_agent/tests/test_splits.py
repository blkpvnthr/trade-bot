from rl_meta.splits import split_of, split_trades, split_bar_ranges, SPLITS
from rl_meta.performance import compute_metrics
from rl_meta.types import TradeRecord


def mk(exit_date, pnl):
    return TradeRecord("2020-01-01", exit_date, 1.0, 1.0, pnl, pnl / 10000.0, 1)


def test_split_boundaries():
    assert split_of("2023-12-31") == "train"
    assert split_of("2024-01-01") == "validation"
    assert split_of("2024-12-31") == "validation"
    assert split_of("2025-01-01") == "oos"
    assert split_of("2026-06-01") == "oos"


def test_split_trades_partition():
    trades = [mk("2022-05-01", 100), mk("2024-06-01", -50),
              mk("2025-03-01", 200), mk("2025-09-01", -30)]
    b = split_trades(trades)
    assert [len(b[s]) for s in SPLITS] == [1, 1, 2]
    # Per-split metrics are computed independently.
    assert compute_metrics(b["train"]).trade_count == 1
    assert compute_metrics(b["oos"]).trade_count == 2
    assert compute_metrics(b["oos"]).expectancy == round((200 - 30) / 2, 2)


def test_split_bar_ranges_clip():
    r = split_bar_ranges("2018-01-01", "2025-06-01")
    assert r["train"] == ("2018-01-01", "2023-12-31")
    assert r["validation"] == ("2024-01-01", "2024-12-31")
    assert r["oos"] == ("2025-01-01", "2025-06-01")
