import json
from pathlib import Path

from rl_meta.export_organizer import (normalize_strategy_id, infer_symbol,
                                     infer_timeframe, summarize_csv, build_plan,
                                     apply_plan, write_manifest, dest_path)


def write_tv_csv(path: Path, trades):
    """trades: list of (entry_dt, exit_dt, entry_px, exit_px, pnl, pct)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Trade number,Type,Date and time,Price USD,Net PnL USD,Net PnL %"]
    for i, (edt, xdt, ep, xp, pnl, pct) in enumerate(trades, 1):
        lines.append(f"{i},Entry long,{edt},{ep},,")
        lines.append(f"{i},Exit long,{xdt},{xp},{pnl},{pct}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- normalization ---------------------------------------------------------
def test_strategy_id_normalization():
    assert normalize_strategy_id("Linear-Regression") == "linear_regression"
    assert normalize_strategy_id("MARTINGALE") == "martingale"
    assert normalize_strategy_id("Mean_Reversion") == "mean_reversion"
    assert normalize_strategy_id("Pin_Bar_Magic_v1") == "pin_bar_magic_v1"


def test_symbol_inference():
    assert infer_symbol("NASDAQ_AAPL") == "AAPL"
    assert infer_symbol("AAPL") == "AAPL"
    assert infer_symbol("NYSE_TSM") == "TSM"
    assert infer_symbol("WIN-Pin_Bar_Magic_v1_NASDAQ_GOOG_2026-06-19") == "GOOG"
    assert infer_symbol("LOSS-Linear-Regression_Strategy_NASDAQ_SMCI_2026-06-20") == "SMCI"
    assert infer_symbol("just_a_strategy_name") is None


def test_timeframe_inference_from_filename():
    assert infer_timeframe("foo_15Min_bar") == "15Min"
    assert infer_timeframe("foo_15m") == "15Min"
    assert infer_timeframe("foo_1h") == "1Hour"
    assert infer_timeframe("foo_1D") == "1Day"
    assert infer_timeframe("foo_daily") == "1Day"
    assert infer_timeframe("no_timeframe_here") == "unknown_timeframe"


def test_timeframe_inference_from_timestamps(tmp_path):
    p = tmp_path / "x.csv"
    write_tv_csv(p, [("2025-03-04 17:15", "2025-03-04 17:45", 100, 101, 1.0, 0.5),
                     ("2025-03-05 15:30", "2025-03-05 16:00", 101, 103, 2.0, 1.0)])
    assert summarize_csv(p)["ts_timeframe"] == "15Min"


def test_date_range_extraction(tmp_path):
    p = tmp_path / "x.csv"
    write_tv_csv(p, [("2025-03-04 17:15", "2025-03-06 17:45", 100, 110, 10.0, 10.0),
                     ("2025-06-19 10:00", "2025-06-20 11:00", 110, 120, 10.0, 9.0)])
    s = summarize_csv(p)
    assert s["date_range"] == "20250304_20250620"
    assert s["trade_count"] == 2 and s["total_net_pnl"] == 20.0


def _make_source(tmp_path):
    src = tmp_path / "trades"
    # A WIN file (positive pnl) and a file in LOSSES that is actually POSITIVE.
    write_tv_csv(src / "WINS" / "My_Strat" / "WIN-My_Strat_NASDAQ_AAPL_2025-06-01.csv",
                 [("2025-03-04 17:15", "2025-03-06 17:45", 100, 110, 50.0, 5.0)])
    write_tv_csv(src / "LOSSES" / "My_Strat" / "LOSS-My_Strat_NASDAQ_TSLA_2025-06-01.csv",
                 [("2025-03-04 17:15", "2025-03-06 17:45", 200, 220, 80.0, 4.0)])
    return src


def test_dry_run_does_not_write(tmp_path):
    src = _make_source(tmp_path)
    dest = tmp_path / "out"
    plan = build_plan(src, dest)            # build only -- the dry-run path
    assert len(plan) == 2
    assert not dest.exists() or not list(dest.rglob("*.csv"))   # nothing written


def test_apply_copies_and_preserves_originals(tmp_path):
    src = _make_source(tmp_path)
    dest = tmp_path / "out"
    plan = build_plan(src, dest)
    stats = apply_plan(plan, move=False)
    assert stats["organized"] == 2
    assert len(list(dest.rglob("*.csv"))) == 2
    assert len(list(src.rglob("*.csv"))) == 2        # originals preserved
    # canonical nesting: strategy_id/symbol/timeframe/file
    aapl = next(Path(r["dest_path"]) for r in plan if r["symbol"] == "AAPL")
    assert aapl.parts[-4:-1] == ("my_strat", "AAPL", "15Min")


def test_label_is_not_used_as_outcome(tmp_path):
    src = _make_source(tmp_path)
    dest = tmp_path / "out"
    plan = build_plan(src, dest)
    tsla = next(r for r in plan if r["symbol"] == "TSLA")
    # File lives under LOSSES/ but its Net PnL is POSITIVE -> outcome from CSV.
    assert tsla["source_folder_label"] == "LOSSES"
    assert tsla["total_net_pnl"] == 80.0           # > 0, NOT inferred from folder


def test_collision_handling(tmp_path):
    dest = tmp_path / "out"
    # two source files that map to the SAME canonical dest (same id/sym/tf/range)
    a = tmp_path / "a" / "WINS" / "S" / "WIN-S_NASDAQ_AAPL_2025-06-01.csv"
    b = tmp_path / "b" / "WINS" / "S" / "WIN-S_NASDAQ_AAPL_2025-06-02.csv"
    write_tv_csv(a, [("2025-03-04 17:15", "2025-03-06 17:45", 100, 110, 10.0, 5.0)])
    write_tv_csv(b, [("2025-03-04 17:15", "2025-03-06 17:45", 100, 120, 20.0, 9.0)])  # diff content
    plan_a = build_plan(a.parents[2], dest)
    plan_b = build_plan(b.parents[2], dest)
    apply_plan(plan_a)
    apply_plan(plan_b)                               # different hash -> suffixed
    assert len(list(dest.rglob("*.csv"))) == 2
    # identical re-apply -> skipped, no third file
    s = apply_plan(build_plan(a.parents[2], dest))
    assert s["skipped"] == 1
    assert len(list(dest.rglob("*.csv"))) == 2


def test_manifest_generation(tmp_path):
    src = _make_source(tmp_path)
    dest = tmp_path / "out"
    plan = build_plan(src, dest)
    apply_plan(plan)
    mpath = write_manifest(plan, dest)
    assert mpath.exists()
    data = json.loads(mpath.read_text())
    assert data["count"] == 2
    rec = data["records"][0]
    for k in ("source_path", "dest_path", "strategy_id", "strategy_name",
              "symbol", "timeframe", "date_range", "earliest_trade_time",
              "latest_trade_time", "row_count", "trade_count", "total_net_pnl",
              "total_net_pnl_pct", "source_folder_label", "copied_or_moved",
              "file_hash"):
        assert k in rec
