import numpy as np
import pandas as pd

from rl_meta.backtest import evaluate_strategy, run_backtests, persist
from rl_meta.performance import PerformanceDB, aggregate_metrics
from rl_meta.types import StrategyMeta


def wave_bars(symbol, periods=2000, start="2018-01-01"):
    """Long sinusoid -> frequent SMA crossovers spanning all walk-forward splits."""
    idx = pd.to_datetime(pd.date_range(start, periods=periods, freq="D"), utc=True)
    x = np.linspace(0, periods / 12.0, periods)
    c = 100.0 + 20.0 * np.sin(x) + np.linspace(0, 10, periods)
    return pd.DataFrame({
        "timestamp": idx, "open": c, "high": c * 1.01, "low": c * 0.99,
        "close": c, "volume": np.full(periods, 1e6), "symbol": symbol,
    })


def sma_meta():
    return StrategyMeta("sma_x", "SMA Crossover Sample", "abc123", "2018-01-01",
                        "s.txt", indicators=["sma", "crossover"],
                        inputs={"fast": "10", "slow": "30"})


def test_multi_symbol_aggregation_pools_trades():
    bars = {"NVDA": wave_bars("NVDA"), "TSM": wave_bars("TSM")}
    res = evaluate_strategy(sma_meta(), bars, "1Day")
    per = sum(s["full"].trade_count for s in res.per_symbol.values())
    assert res.aggregate_full.trade_count == per
    assert res.aggregate_full.trade_count > 0
    assert set(res.tested_symbols) == {"NVDA", "TSM"}


def test_aggregate_metrics_helper_pools():
    from rl_meta.types import TradeRecord
    a = [TradeRecord("2023-01-01", "2023-01-02", 1, 1, 100, 0.01, 1)]
    b = [TradeRecord("2023-02-01", "2023-02-02", 1, 1, -50, -0.005, 1),
         TradeRecord("2023-03-01", "2023-03-02", 1, 1, 200, 0.02, 1)]
    m = aggregate_metrics({"X": a, "Y": b})
    assert m.trade_count == 3


def test_walk_forward_splits_present():
    bars = {"NVDA": wave_bars("NVDA")}
    res = evaluate_strategy(sma_meta(), bars, "1Day")
    for split in ("full", "train", "validation", "oos"):
        assert split in res.aggregate
    # train should hold the bulk of the 2018-2023 history.
    assert res.aggregate["train"].trade_count >= res.aggregate["oos"].trade_count


def test_trade_count_gate_blocks_thin_strategy():
    # Few bars -> few trades -> must be rejected by the >=100 gate.
    short = {"NVDA": wave_bars("NVDA", periods=120)}
    res = evaluate_strategy(sma_meta(), short, "1Day")
    assert not res.eligible
    assert "too few trades" in res.rejection_reason


def test_unsupported_strategy_not_executable():
    junk = StrategyMeta("j", "ADX Mystery", "h", "2018-01-01", "j.txt",
                        indicators=["adx"])
    res = evaluate_strategy(junk, {"NVDA": wave_bars("NVDA")}, "1Day")
    assert not res.executable and not res.eligible
    assert res.rejection_reason == "not executable"


def test_no_data_is_insufficient():
    res = evaluate_strategy(sma_meta(), {}, "1Day")
    assert not res.eligible and res.rejection_reason == "insufficient Alpaca data"


def test_oos_gate_flips_eligibility():
    # Clean-sine bars ending in 2023 (no 2025 OOS data). Six symbols pool to
    # >=150 winning trades so the FULL sample clears admission (and the overfit
    # guard), but the OOS limb must fail (zero OOS trades) and flip eligibility
    # off -- proving OOS is the deciding gate, not just full-sample.
    syms = ["NVDA", "TSM", "GOOGL", "ASML", "KLAC", "FIGS"]
    bars = {s: wave_bars(s, periods=2300) for s in syms}
    res = evaluate_strategy(sma_meta(), bars, "1Day")
    assert res.aggregate["oos"].trade_count == 0       # no 2025+ data
    assert res.aggregate_full.trade_count >= 150       # full sample is admissible
    assert res.aggregate_full.win_rate >= 0.60         # ...and clears the perf limb
    assert not res.eligible
    assert "OOS" in res.rejection_reason or "out-of-sample" in res.rejection_reason


def test_population_summary_counts():
    from types import SimpleNamespace
    from rl_meta.backtest import population_summary, StrategyResult

    metas = [
        StrategyMeta("a", "A", "h", "2023-01-01", "a.txt", declares_strategy=True),
        StrategyMeta("b", "B", "h", "2023-01-01", "b.txt", declares_strategy=True),
        StrategyMeta("regime", "Markov Regime", "h", "2023-01-01", "r.txt",
                     declares_strategy=False),   # indicator/study, not a strategy
    ]

    def res(sid, eligible, rl):
        r = StrategyResult(sid, sid, None, True, "1Day")
        r.eligible, r.rl_eligible = eligible, rl
        return r

    results = [res("a", True, True), res("b", True, False), res("regime", False, False)]
    report = SimpleNamespace(effective_breadth=1.4, n_strategies=2)

    s = population_summary(metas, results, report)
    assert s["discovered"] == 2            # the indicator file is NOT a strategy
    assert s["non_strategy_files"] == 1
    assert s["eligible"] == 2
    assert s["rl_eligible"] == 1
    assert s["effective_breadth"] == 1.4 and s["breadth_universe"] == 2


def test_persist_writes_cells(tmp_path):
    bars = {"NVDA": wave_bars("NVDA")}
    results = run_backtests([sma_meta()], bars, "1Day")
    db = PerformanceDB(tmp_path / "perf.db")
    persist(db, results, on_date="2026-01-01")
    cells = db.cells(strategy_id="sma_x")
    assert any(c["symbol"] == "AGGREGATE" and c["split"] == "full" for c in cells)
    assert any(c["symbol"] == "NVDA" for c in cells)
