"""Backtest / signal layer.

DEFAULT path: run a recognised strategy adapter (adapters.py) over REAL Alpaca
OHLCV bars and produce TradeRecords with next-bar-open fills (no look-ahead).

The synthetic SMA-crossover engine remains ONLY as an offline fallback so the
RL stack (env/check_env/tests) can run with no data feed or credentials. It is
never used for strategy admission -- only the real-bar path feeds the ranking
and RL eligibility pipeline.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .adapters import ADAPTERS, detect_adapter
from .types import StrategyMeta, TradeRecord

CAPITAL_PER_TRADE = 10_000.0


class NotExecutable(RuntimeError):
    """Raised when a strategy has no supported adapter."""


# -- real-bar execution engine --------------------------------------------
def _frac_input(meta: StrategyMeta, keys, default: float) -> float:
    for k in keys:
        raw = meta.inputs.get(k)
        if raw is None:
            continue
        try:
            v = float(raw)
            return v / 100.0 if v > 1.0 else v
        except ValueError:
            continue
    return default


def execute_on_bars(entries, exits, df: pd.DataFrame,
                    sl_pct: float, tp_pct: float) -> list:
    """Long-only loop. Enter at NEXT bar's open; exit on signal/SL/TP/last bar.

    SL/TP are checked intrabar on the held bar's low/high. Entering on the bar
    after the signal avoids same-bar look-ahead.
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ts = df["timestamp"].tolist()
    n = len(df)
    trades = []
    in_pos = False
    entry_i = 0
    entry_price = 0.0

    for i in range(n):
        if in_pos:
            ret_low = (low[i] - entry_price) / entry_price
            ret_high = (h[i] - entry_price) / entry_price
            exit_price = None
            if ret_low <= -sl_pct:
                exit_price = entry_price * (1.0 - sl_pct)
            elif ret_high >= tp_pct:
                exit_price = entry_price * (1.0 + tp_pct)
            elif bool(exits[i]):
                exit_price = c[i]
            elif i == n - 1:
                exit_price = c[i]
            if exit_price is not None:
                trades.append(_make_trade(ts[entry_i], ts[i], entry_price, exit_price))
                in_pos = False
        if not in_pos and i < n - 1 and bool(entries[i]):
            entry_i = i + 1
            entry_price = o[entry_i]
            in_pos = True
    return trades


def _make_trade(entry_ts, exit_ts, p_entry: float, p_exit: float) -> TradeRecord:
    ret = (p_exit - p_entry) / p_entry
    hold = max((exit_ts - entry_ts).total_seconds() / 86400.0, 0.0)
    return TradeRecord(
        entry_date=entry_ts.date().isoformat(),
        exit_date=exit_ts.date().isoformat(),
        entry_price=round(float(p_entry), 4),
        exit_price=round(float(p_exit), 4),
        pnl=round(ret * CAPITAL_PER_TRADE, 2),
        return_pct=float(ret),
        hold_days=round(hold, 3),
    )


def backtest_executable(meta: StrategyMeta, df: pd.DataFrame) -> list:
    """Backtest one strategy on one symbol's real bars. Raises NotExecutable."""
    name, _ = detect_adapter(meta)
    if name is None:
        raise NotExecutable(meta.strategy_id)
    if df is None or df.empty:
        return []
    entries, exits = ADAPTERS[name](df, meta)
    sl = _frac_input(meta, ("sl_pct", "stop_pct", "sl"), 0.05)
    tp = _frac_input(meta, ("tp_pct", "limit_pct", "tp"), 0.10)
    return execute_on_bars(entries, exits, df, sl, tp)


def backtest_universe(meta: StrategyMeta, bars_by_symbol: dict) -> dict:
    """{symbol: [TradeRecord]} across the universe for one executable strategy."""
    out = {}
    for sym, df in bars_by_symbol.items():
        out[sym] = backtest_executable(meta, df)
    return out


# -- synthetic fallback (offline only; NOT used for admission) --------------
TRADING_DAYS = 504
START_DATE = date(2023, 1, 2)


def _seed_from_hash(version_hash: str) -> int:
    return int(version_hash[:8], 16)


def synthetic_prices(seed: int, n: int = TRADING_DAYS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = rng.uniform(-0.0003, 0.0006)
    vol = rng.uniform(0.008, 0.02)
    rets = rng.normal(drift, vol, size=n)
    return 100.0 * np.exp(np.cumsum(rets))


def synthetic_bars(meta: StrategyMeta) -> pd.DataFrame:
    """Deterministic synthetic OHLCV for offline runs (fallback only)."""
    seed = _seed_from_hash(meta.version_hash)
    close = synthetic_prices(seed)
    idx = pd.to_datetime([START_DATE + timedelta(days=i) for i in range(len(close))],
                         utc=True)
    return pd.DataFrame({
        "timestamp": idx,
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": 1_000_000.0, "symbol": meta.strategy_id,
    })


def backtest_strategy(meta: StrategyMeta) -> list:
    """OFFLINE FALLBACK: backtest on deterministic synthetic bars. Reproducible.
    Not used for real admission -- see backtest_executable for the real path."""
    df = synthetic_bars(meta)
    try:
        return backtest_executable(meta, df)
    except NotExecutable:
        # default to a plain SMA crossover for unrecognised offline strategies
        f = df["close"].rolling(10).mean()
        s = df["close"].rolling(30).mean()
        from .adapters import _cross_up, _cross_dn
        return execute_on_bars(_cross_up(f, s), _cross_dn(f, s), df, 0.05, 0.10)


def _demo():
    from .registry import load_strategies, PINE_DIR
    meta = next(m for m in load_strategies(PINE_DIR) if m.strategy_id == "sample_strategy")
    df = synthetic_bars(meta)
    t1 = backtest_executable(meta, df)
    t2 = backtest_executable(meta, df)
    assert [x.pnl for x in t1] == [x.pnl for x in t2], "must be reproducible"
    assert all(t.exit_date >= t.entry_date for t in t1)
    print(f"signals._demo OK ({len(t1)} real-engine trades on synthetic bars)")


if __name__ == "__main__":
    _demo()
