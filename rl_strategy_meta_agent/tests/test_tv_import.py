import numpy as np
import pandas as pd

from rl_meta.tv_import import parse_tv_csv, parse_filename, EXPORTS_DIR
from rl_meta.performance import PerformanceDB, compute_metrics
from rl_meta.backtest import evaluate_strategy
from rl_meta.registry import file_hash
from rl_meta.types import StrategyMeta, TradeRecord

# A minimal two-row-per-trade TradingView List-of-Trades export.
CSV = (
    "Trade #,Type,Date/Time,Signal,Price USD,Net P&L USD,Net P&L %\n"
    "1,Entry Long,2023-02-01,buy,100.00,,\n"
    "1,Exit Long,2023-02-10,sell,112.00,1200.00,12.0\n"
    "2,Entry Long,2023-06-01,buy,112.00,,\n"
    "2,Exit Long,2023-06-09,sell,106.40,-560.00,-5.0\n"
    "3,Entry Long,2025-02-01,buy,120.00,,\n"
    "3,Exit Long,2025-02-20,sell,138.00,1800.00,15.0\n"
)


def write_csv(tmp_path, text=CSV, name="strat__NVDA__1Day.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


# -- CSV parsing -----------------------------------------------------------
def test_parse_basic(tmp_path):
    trades = parse_tv_csv(write_csv(tmp_path))
    assert len(trades) == 3
    t0 = trades[0]
    assert t0.entry_date == "2023-02-01" and t0.exit_date == "2023-02-10"
    assert t0.entry_price == 100.0 and t0.exit_price == 112.0
    assert t0.pnl == 1200.0 and abs(t0.return_pct - 0.12) < 1e-9
    assert t0.hold_days == 9.0
    assert trades[1].pnl == -560.0


def test_parse_skips_open_trade(tmp_path):
    txt = CSV + "4,Entry Long,2025-05-01,buy,140.00,,\n"   # no exit row
    trades = parse_tv_csv(write_csv(tmp_path, txt))
    assert len(trades) == 3          # the dangling entry is ignored


def test_parse_handles_messy_numbers(tmp_path):
    messy = (
        "Trade #,Type,Date/Time,Price USD,Net P&L USD,Net P&L %\n"
        "1,Entry Long,2023-02-01,\"1,100.00\",,\n"
        "1,Exit Long,2023-02-10,\"1,210.00\",\"(50.00)\",-4.5\n"   # parenthesised loss
    )
    trades = parse_tv_csv(write_csv(tmp_path, messy))
    assert len(trades) == 1
    assert trades[0].entry_price == 1100.0 and trades[0].pnl == -50.0


def test_repo_sample_export_parses_and_scores():
    trades = parse_tv_csv(EXPORTS_DIR / "sample_strategy__NVDA__1Day.csv")
    assert len(trades) == 5
    m = compute_metrics(trades)
    assert m.trade_count == 5
    assert abs(m.win_rate - 0.8) < 1e-9          # 4 wins, 1 loss
    assert m.profit_factor > 1.0


# -- strategy matching -----------------------------------------------------
def test_filename_convention():
    assert parse_filename("sample_strategy__NVDA__1Day.csv") == \
        ("sample_strategy", "NVDA", "1Day")
    assert parse_filename("no_convention.csv") is None
    assert parse_filename("a__b.csv") is None    # needs exactly 3 parts


def test_version_pinning(tmp_path):
    db = PerformanceDB(tmp_path / "p.db")
    trades = parse_tv_csv(write_csv(tmp_path))
    db.record_import("strat", "NVDA", "1Day", "hashA", "f.csv", "c1", trades,
                     "2026-01-01")
    # Matching version returns trades; a different (stale) version returns none.
    assert "NVDA" in db.imported_trades("strat", "1Day", "hashA")
    assert db.imported_trades("strat", "1Day", "hashB") == {}


# -- duplicate import protection -------------------------------------------
def test_duplicate_protection(tmp_path):
    db = PerformanceDB(tmp_path / "p.db")
    trades = parse_tv_csv(write_csv(tmp_path))
    ch = file_hash(CSV)
    s1 = db.record_import("strat", "NVDA", "1Day", "h", "f.csv", ch, trades,
                          "2026-01-01")
    s2 = db.record_import("strat", "NVDA", "1Day", "h", "f.csv", ch, trades,
                          "2026-01-01")
    assert s1 == "imported" and s2 == "duplicate"
    # No double-counting: still exactly the imported trade count.
    got = db.imported_trades("strat", "1Day", "h")["NVDA"]
    assert len(got) == len(trades)
    # force=True re-imports.
    assert db.record_import("strat", "NVDA", "1Day", "h", "f.csv", ch, trades,
                            "2026-01-02", force=True) == "replaced"


# -- prefer imported over adapter ------------------------------------------
def make_bars(symbol, periods=300):
    idx = pd.to_datetime(pd.date_range("2023-01-01", periods=periods, freq="D"),
                         utc=True)
    x = np.linspace(0, periods / 6.0, periods)
    c = 100 + 10 * np.sin(x)
    return pd.DataFrame({"timestamp": idx, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": np.full(periods, 1e6),
                         "symbol": symbol})


def sma_meta():
    return StrategyMeta("sma_x", "SMA Crossover Sample", "h", "2023-01-01", "s.txt",
                        indicators=["sma", "crossover"],
                        inputs={"fast": "10", "slow": "30"})


def test_imported_overrides_adapter(tmp_path):
    imported = {"NVDA": parse_tv_csv(write_csv(tmp_path))}   # 3 TV trades
    res = evaluate_strategy(sma_meta(), {"NVDA": make_bars("NVDA")}, "1Day",
                            imported_by_symbol=imported)
    assert res.sources["NVDA"] == "tradingview"
    assert res.primary_source == "tradingview"
    # Uses the 3 imported trades, NOT the adapter's many crossover trades.
    assert res.per_symbol["NVDA"]["full"].trade_count == 3


def test_non_executable_strategy_evaluable_from_imports(tmp_path):
    junk = StrategyMeta("j", "ADX Mystery", "h", "2023-01-01", "j.txt",
                        indicators=["adx"])
    imported = {"NVDA": parse_tv_csv(write_csv(tmp_path))}
    res = evaluate_strategy(junk, {}, "1Day", imported_by_symbol=imported)
    assert not res.executable                       # no adapter...
    assert res.primary_source == "tradingview"      # ...but TV trades drive it
    assert res.aggregate_full.trade_count == 3
