"""Strict source separation: never mix TradingView and adapter trades."""
import numpy as np
import pandas as pd

from rl_meta import backtest as bt
from rl_meta.backtest import (evaluate_strategy, allow_adapter_screening_for_rl,
                              SRC_TV, SRC_ADAPTER, SRC_MISSING, SRC_NOT_EXEC)
from rl_meta.types import StrategyMeta, TradeRecord


def sma_meta():
    return StrategyMeta("sma_x", "SMA Crossover Sample", "h", "2023-01-01", "s.txt",
                        indicators=["sma", "crossover"],
                        inputs={"fast": "10", "slow": "30"})


def bars(symbol, periods=400):
    idx = pd.to_datetime(pd.date_range("2023-01-01", periods=periods, freq="D"),
                         utc=True)
    x = np.linspace(0, periods / 6.0, periods)
    c = 100 + 10 * np.sin(x)
    return pd.DataFrame({"timestamp": idx, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": np.full(periods, 1e6),
                         "symbol": symbol})


def tv_trades(n=3):
    out = []
    for i in range(n):
        out.append(TradeRecord(f"2023-0{i+1}-01", f"2023-0{i+1}-10",
                               100.0, 110.0, 1000.0, 0.10, 9))
    return out


def make_passing_trades(n=600):
    """600 trades, ~70% win, 300 in 2025 OOS, small +20/-10 PnL on 10 shares ->
    passes admission AND the hard RL risk gates (>=500 trades, >=250 OOS,
    PF>=1.25, DD<50%, not blown up)."""
    trades = []
    for i in range(n):
        year = 2025 if i >= n // 2 else 2019 + (i % 5)
        mo = (i % 12) + 1
        day = (i % 27) + 1
        win = (i % 10) < 7
        trades.append(TradeRecord(f"{year}-{mo:02d}-{day:02d}",
                                  f"{year}-{mo:02d}-{day:02d}",
                                  100.0, 102.0 if win else 99.0,
                                  20.0 if win else -10.0, 0.02 if win else -0.01, 5,
                                  raw_qty=10, normalized_qty=10,
                                  raw_pnl_usd=20.0 if win else -10.0))
    return trades


# 1. Any TV import disables adapter for the WHOLE strategy (all symbols).
def test_tv_import_disables_adapter_everywhere():
    res = evaluate_strategy(sma_meta(), {"NVDA": bars("NVDA"), "TSM": bars("TSM")},
                            "1Day", imported_by_symbol={"NVDA": tv_trades()},
                            universe=["NVDA", "TSM"])
    assert res.evaluation_source == SRC_TV
    assert res.sources["NVDA"] == "tradingview"
    # TSM has bars but NO import -> marked missing, NOT filled with adapter trades.
    assert res.sources["TSM"] == SRC_MISSING
    assert "TSM" not in res.per_symbol
    assert res.missing_tv_symbols == ["TSM"]
    assert res.tv_symbols == ["NVDA"]


# 2 & 3. Aggregate uses ONLY the TV trades -- never combines with adapter output.
def test_aggregate_is_single_source():
    res = evaluate_strategy(sma_meta(), {"NVDA": bars("NVDA"), "TSM": bars("TSM")},
                            "1Day", imported_by_symbol={"NVDA": tv_trades(3)},
                            universe=["NVDA", "TSM"])
    # Adapter on these bars would make many trades; aggregate must be exactly 3.
    assert res.aggregate_full.trade_count == 3


# 4. RL eligibility excludes adapter_screening unless the flag is set.
def test_rl_eligibility_source_rules(monkeypatch):
    # TradingView-verified passing strategy -> RL-eligible.
    tv = evaluate_strategy(sma_meta(), {}, "1Day",
                           imported_by_symbol={"NVDA": make_passing_trades()},
                           universe=["NVDA"])
    assert tv.evaluation_source == SRC_TV and tv.eligible and tv.rl_eligible

    # Adapter-screening passing strategy -> eligible but NOT RL-eligible by default.
    monkeypatch.setattr(bt, "backtest_executable",
                        lambda meta, df: make_passing_trades())
    monkeypatch.delenv("ALLOW_ADAPTER_SCREENING_FOR_RL", raising=False)
    ad = evaluate_strategy(sma_meta(), {"NVDA": bars("NVDA")}, "1Day")
    assert ad.evaluation_source == SRC_ADAPTER
    assert ad.eligible and not ad.rl_eligible
    assert "adapter_screening" in ad.rejection_reason

    # ...unless explicitly allowed.
    monkeypatch.setenv("ALLOW_ADAPTER_SCREENING_FOR_RL", "true")
    assert allow_adapter_screening_for_rl()
    ad2 = evaluate_strategy(sma_meta(), {"NVDA": bars("NVDA")}, "1Day")
    assert ad2.eligible and ad2.rl_eligible


# 5. Existing all-adapter workflow still works when there are no TV imports.
def test_all_adapter_workflow_without_imports():
    res = evaluate_strategy(sma_meta(), {"NVDA": bars("NVDA")}, "1Day")
    assert res.evaluation_source == SRC_ADAPTER
    assert res.sources.get("NVDA") == "adapter"
    assert res.per_symbol  # adapter produced metrics


def test_non_executable_no_imports_marks_not_executable():
    junk = StrategyMeta("j", "ADX Mystery", "h", "2023-01-01", "j.txt",
                        indicators=["adx"])
    res = evaluate_strategy(junk, {"NVDA": bars("NVDA")}, "1Day")
    assert res.evaluation_source == SRC_NOT_EXEC and not res.rl_eligible
