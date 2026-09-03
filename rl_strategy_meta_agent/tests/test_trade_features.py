from pathlib import Path

import numpy as np
import pandas as pd

from rl_meta.trade_features import (read_trades_with_context, extract_pre_entry,
                                   extract_during, extract_exit_quality,
                                   assign_split, labels, build_row,
                                   FEATURE_GROUPS, PRE_ENTRY_FEATURES,
                                   DURING_TRADE_FEATURES, EXIT_FEATURES,
                                   CONTEXT_FEATURES, TARGET_LABELS, LOOKBACK)


def _bars(n=80):
    idx = pd.date_range("2025-03-03 09:30", periods=n, freq="15min", tz="UTC")
    c = 100 + np.cumsum(np.sin(np.linspace(0, 9, n))) + np.linspace(0, 3, n)
    return pd.DataFrame({"timestamp": idx, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": np.full(n, 1e5)})


def _trade(entry, exit_, qty=1000, pnl=500.0, ret=0.05, mfe=0.10, mae=-0.02, long=True):
    return {"entry_ts": pd.Timestamp(entry, tz="UTC"), "exit_ts": pd.Timestamp(exit_, tz="UTC"),
            "entry_price": 100.0, "exit_price": 105.0, "qty": qty, "raw_pnl": pnl,
            "return_pct": ret, "mfe_pct": mfe, "mae_pct": mae,
            "mfe_usd": 1000.0, "mae_usd": -200.0, "is_long": long}


CSV = ("Trade number,Type,Date and time,Price USD,Size (qty),Net PnL USD,Net PnL %,"
       "Favorable excursion USD,Favorable excursion %,Adverse excursion USD,Adverse excursion %\n"
       "1,Entry long,2023-06-01 10:00,100,1000,,,,,,\n"
       "1,Exit long,2023-06-01 12:00,110,1000,1000,10,1200,12,-300,-3\n"
       "2,Entry long,2025-02-01 10:00,100,1000,,,,,,\n"
       "2,Exit long,2025-02-01 11:00,95,1000,-500,-5,200,2,-700,-7\n")


# -- one row per trade + label from normalized PnL ------------------------
def test_one_row_per_trade(tmp_path):
    p = tmp_path / "x.csv"; p.write_text(CSV)
    trades = read_trades_with_context(p)
    assert len(trades) == 2
    rows = [build_row(t, None, {"strategy_id": "s", "symbol": "AAPL",
            "timeframe": "15Min"}, {}) for t in trades]
    assert len(rows) == 2


def test_labels_use_normalized_pnl():
    lab = labels(_trade("2025-03-03 10:00", "2025-03-03 11:00", qty=1000, pnl=500.0))
    assert lab["normalized_pnl"] == 5.0          # 500 * 10/1000
    assert lab["outcome_win"] == 1 and lab["outcome_loss"] == 0 and lab["outcome_flat"] == 0
    assert lab["raw_pnl_usd"] == 500.0           # raw preserved


def test_human_bucket_preserved_not_truth():
    # filed under LOSSES but actually a winning trade
    row = build_row(_trade("2025-03-03 10:00", "2025-03-03 11:00", pnl=500.0), None,
                    {"strategy_id": "s", "symbol": "AAPL", "timeframe": "15Min"},
                    {"human_bucket": "LOSSES"})
    assert row["human_bucket"] == "LOSSES"       # metadata kept
    assert row["outcome_win"] == 1               # outcome from PnL, not folder


# -- leakage prevention ---------------------------------------------------
def test_pre_features_ignore_post_entry_bars():
    bars = _bars(); entry = bars["timestamp"].iloc[60]
    f1 = extract_pre_entry(bars, entry)
    corrupt = bars.copy()
    corrupt.loc[corrupt["timestamp"] > entry,
                ["open", "high", "low", "close", "volume"]] = -1.0
    assert f1 == extract_pre_entry(corrupt, entry) and f1 is not None


def test_pre_features_insufficient_history():
    bars = _bars(); early = bars["timestamp"].iloc[LOOKBACK - 2]
    assert extract_pre_entry(bars, early) is None


# -- MFE/MAE + exit quality -----------------------------------------------
def test_mfe_mae_and_capture():
    d = extract_during(_trade("2025-03-03 10:00", "2025-03-03 12:30", mfe=0.10, mae=-0.02))
    assert d["mfe_pct"] == 0.10 and d["mae_pct"] == -0.02 and d["hold_bars"] == 10.0
    e = extract_exit_quality(_trade("a 10:00", "b 12:00", ret=0.05, mfe=0.10, mae=-0.02)
                             if False else _trade("2025-03-03 10:00", "2025-03-03 12:00",
                             ret=0.05, mfe=0.10, mae=-0.02))
    assert abs(e["profit_capture_ratio"] - 0.5) < 1e-9       # captured 0.05 of 0.10
    assert abs(e["giveback_from_mfe"] - 0.05) < 1e-9
    assert abs(e["exit_efficiency"] - (0.07 / 0.12)) < 1e-6  # (ret-mae)/(mfe-mae)


# -- split assignment ------------------------------------------------------
def test_split_assignment():
    assert assign_split(pd.Timestamp("2023-06-01", tz="UTC")) == "train"
    assert assign_split(pd.Timestamp("2024-06-01", tz="UTC")) == "validation"
    assert assign_split(pd.Timestamp("2025-06-01", tz="UTC")) == "oos"


# -- missing bars handled gracefully --------------------------------------
def test_missing_bars_nan_pre_label_present():
    row = build_row(_trade("2025-03-03 10:00", "2025-03-03 11:00"), None,
                    {"strategy_id": "s", "symbol": "X", "timeframe": "15Min"}, {})
    assert row["has_pre_features"] is False
    assert all(np.isnan(row[c]) if isinstance(row[c], float) else row[c] != row[c]
               or np.isnan(row[c]) for c in ["close_vs_sma20"])
    assert row["normalized_pnl"] == 5.0          # label still computed
    assert "dur_" not in "".join(PRE_ENTRY_FEATURES)


# -- feature groups + no broker path --------------------------------------
def test_feature_groups_structure():
    for g in ("pre_entry_features", "during_trade_features", "exit_features",
              "context_features", "target_labels"):
        assert g in FEATURE_GROUPS and FEATURE_GROUPS[g]
    assert set(PRE_ENTRY_FEATURES).isdisjoint(DURING_TRADE_FEATURES)
    assert set(PRE_ENTRY_FEATURES).isdisjoint(EXIT_FEATURES)
    assert set(PRE_ENTRY_FEATURES).isdisjoint(TARGET_LABELS)
    assert "human_bucket" in CONTEXT_FEATURES


def test_no_broker_path_in_new_files():
    root = Path(__file__).resolve().parent.parent
    for f in ("rl_meta/trade_features.py", "scripts/build_trade_outcome_dataset.py"):
        text = (root / f).read_text()
        for bad in ("submit_order", "TradingClient", "/v2/orders", "place_order"):
            assert bad not in text
