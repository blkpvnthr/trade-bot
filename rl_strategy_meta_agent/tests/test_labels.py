from rl_meta.labels import (actual_bucket, label_mismatch, label_correct,
                            bucket_accuracy, normalize_bucket)
from rl_meta.performance import PerformanceDB, compute_metrics
from rl_meta.types import TradeRecord


def test_actual_bucket_from_pnl():
    assert actual_bucket(50) == "FLAT"           # within flat band
    assert actual_bucket(9000) == "WINS"
    assert actual_bucket(-9000) == "LOSSES"


def test_mismatch_rules():
    assert label_mismatch("WINS", -300, 0.8)     # WINS but lost
    assert label_mismatch("LOSSES", 5000, 1.4)   # LOSSES but won
    assert label_mismatch("FLAT", 9000, 2.0)     # FLAT but big
    assert label_mismatch("WINS", 4000, 1.6) is None      # WINS and won -> ok
    assert label_mismatch("LOSSES", -4000, 0.7) is None   # LOSSES and lost -> ok


def test_unknown_never_mismatches():
    assert label_mismatch("UNKNOWN", -9999, 0.1) is None
    assert label_correct("UNKNOWN", -9999, 0.1) is None
    assert normalize_bucket(None) == "UNKNOWN"


def test_bucket_accuracy():
    rows = [
        {"human_bucket": "WINS", "net_pnl": 5000, "profit_factor": 2.0},   # ok
        {"human_bucket": "WINS", "net_pnl": -100, "profit_factor": 0.9},   # miss
        {"human_bucket": "LOSSES", "net_pnl": -3000, "profit_factor": 0.7},  # ok
        {"human_bucket": "UNKNOWN", "net_pnl": 1, "profit_factor": 1.0},   # ignored
    ]
    acc = bucket_accuracy(rows)
    assert acc["WINS"][1:] == (1, 2)             # 1 of 2 WINS correct
    assert acc["ALL"][1:] == (2, 3)              # UNKNOWN excluded


# -- the key rule: label is metadata, NEVER the outcome --------------------
def test_label_does_not_affect_metrics(tmp_path):
    db = PerformanceDB(tmp_path / "p.db")
    # A clearly WINNING trade stream, but the human filed it under LOSSES.
    trades = [TradeRecord("2025-03-01", "2025-03-02", 100, 110, 500.0, 0.05, 1),
              TradeRecord("2025-03-03", "2025-03-04", 110, 120, 400.0, 0.04, 1)]
    db.record_import("s", "AAPL", "15Min", "vh", "f.csv", "h1", trades, "2026-01-01")
    db.record_label("s", "AAPL", "15Min", "LOSSES", "f.csv", "2026-01-01")
    # label preserved as-is
    assert db.symbol_label("s", "AAPL", "15Min") == "LOSSES"
    assert db.labels("15Min") == {("s", "AAPL", "15Min"): "LOSSES"}
    # but the OUTCOME comes from PnL -> 100% win rate, PF inf, despite the label
    m = compute_metrics(db.imported_trades("s", "15Min", "vh")["AAPL"])
    assert m.win_rate == 1.0 and m.profit_factor == float("inf")
