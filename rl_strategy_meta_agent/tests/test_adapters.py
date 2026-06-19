import numpy as np
import pandas as pd

from rl_meta.adapters import (detect_adapter, annotate_executability, ADAPTERS,
                              SUPPORTED)
from rl_meta.signals import backtest_executable, NotExecutable
from rl_meta.registry import load_strategies, PINE_DIR
from rl_meta.types import StrategyMeta, Status


def make_bars(closes, start="2023-01-02"):
    idx = pd.to_datetime(pd.date_range(start, periods=len(closes), freq="D"), utc=True)
    c = np.array(closes, dtype=float)
    return pd.DataFrame({
        "timestamp": idx, "open": c, "high": c * 1.01, "low": c * 0.99,
        "close": c, "volume": np.full(len(c), 1_000_000.0), "symbol": "T",
    })


def test_sample_strategy_is_executable():
    meta = load_strategies(PINE_DIR)[0]
    name, reason = detect_adapter(meta)
    assert name == "sma_crossover" and reason == ""
    annotate_executability(meta)
    assert meta.executable and meta.adapter_type == "sma_crossover"


def test_unsupported_is_rejected():
    junk = StrategyMeta("j", "Mystery ADX Thing", "h", "2023-01-01", "j.txt",
                        indicators=["adx", "stoch"])
    name, reason = detect_adapter(junk)
    assert name is None and reason
    annotate_executability(junk)
    assert not junk.executable
    assert junk.status == Status.NOT_EXECUTABLE.value
    assert junk.adapter_type is None


def test_all_supported_adapters_registered():
    assert set(ADAPTERS) == set(SUPPORTED)


def test_adapter_detection_by_indicator():
    cases = {
        "rsi_mean_reversion": ["rsi"],
        "vwap_reclaim": ["vwap"],
        "ema_crossover": ["ema", "crossover"],
        "sma_crossover": ["sma", "crossover"],
    }
    for expected, inds in cases.items():
        meta = StrategyMeta("x", "X", "h", "2023-01-01", "x.txt", indicators=inds)
        assert detect_adapter(meta)[0] == expected


def test_executable_produces_trades_on_real_bars():
    # Decline (fast below slow) then rise -> a real upward crossover -> entry,
    # then a decline -> exit. Pure up-ramps never cross (fast starts above slow).
    closes = (list(np.linspace(140, 100, 50)) + list(np.linspace(100, 170, 110))
              + list(np.linspace(170, 120, 60)))
    df = make_bars(closes)
    meta = StrategyMeta("sma", "SMA Crossover Sample", "h", "2023-01-01", "s.txt",
                        indicators=["sma", "crossover"],
                        inputs={"fast": "10", "slow": "30"})
    trades = backtest_executable(meta, df)
    assert len(trades) >= 1
    t = trades[0]
    # Entry happens at next-bar open (no look-ahead); dates come from real bars.
    assert t.entry_date <= t.exit_date
    assert t.entry_date >= "2023-01-02"


def test_entry_fills_at_next_bar_open():
    # Pin the no-look-ahead contract: entry price == the open of the bar AFTER
    # the entry signal, never the signal bar's close.
    from rl_meta.adapters import sma_crossover
    closes = (list(np.linspace(140, 100, 50)) + list(np.linspace(100, 170, 110)))
    df = make_bars(closes)
    meta = StrategyMeta("sma", "SMA Crossover Sample", "h", "2023-01-01", "s.txt",
                        indicators=["sma", "crossover"],
                        inputs={"fast": "10", "slow": "30"})
    entries, _ = sma_crossover(df, meta)
    first = int(np.argmax(entries))           # first True entry signal
    assert entries[first]
    trades = backtest_executable(meta, df)
    assert trades and trades[0].entry_price == round(float(df["open"].iloc[first + 1]), 4)


def test_not_executable_raises():
    junk = StrategyMeta("j", "Nope", "h", "2023-01-01", "j.txt", indicators=["adx"])
    try:
        backtest_executable(junk, make_bars([1, 2, 3]))
        assert False, "should have raised"
    except NotExecutable:
        pass
