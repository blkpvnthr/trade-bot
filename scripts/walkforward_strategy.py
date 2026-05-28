#!/usr/bin/env python3
"""
scripts/walkforward_strategy.py

Core walk-forward analysis engine for a single strategy variant.

For each rolling window:
  1. Grid-search parameters on train slice → select best on validation slice
  2. Evaluate selected params on test slice only (out-of-sample)
  3. Record all required metrics

Data source priority:
  1. User-supplied --bars <path>
  2. daily_system_outputs/dated/*/intraday_bars_1m_*.csv  (if ≥5 months)
  3. daily_system_outputs/daily_raw_bars.csv              (daily OHLCV, ~17 months)

Usage:
  python scripts/walkforward_strategy.py --strategy-id tux_mtf_original --initial-cash 100000
  python scripts/walkforward_strategy.py --strategy-id tux_mtf_original \\
    --bars daily_system_outputs/daily_raw_bars.csv --train-months 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY, get_strategy

OUTPUT_DIR = PROJECT_ROOT / "daily_system_outputs" / "walkforward"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFConfig:
    train_months: int   = 3
    val_months: int     = 1
    test_months: int    = 1
    step_months: int    = 1
    starting_cash: float       = 100_000.0
    max_position_weight: float = 0.12
    max_gross_exposure: float  = 0.72
    max_order_dollars: float   = 12_000.0
    cash_buffer_pct: float     = 0.08
    slippage_bps: float        = 5.0
    risk_per_trade_pct: float  = 0.0035
    max_daily_loss_pct: float  = 0.015
    daily_profit_target_dollars: float = 400.0
    max_open_positions: int    = 5
    min_bars_per_slice: int    = 30
    min_symbols_per_window: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Strategy defaults + parameter search spaces
# ─────────────────────────────────────────────────────────────────────────────

# Complete defaults ensure generate_signals() never throws KeyError
_STRATEGY_DEFAULTS: dict[str, dict[str, Any]] = {
    "tux_mtf_original": {
        "fast_ema": 9, "slow_ema": 21, "trend_ema": 50,
        "n_confirm_tfs": 2, "atr_stop_mult": 1.5, "atr_target_mult": 3.0,
        "min_volume_mult": 1.2, "allow_shorts": True, "volume_lookback": 20,
    },
    "tux_mtf_long_only": {
        "fast_ema": 9, "slow_ema": 21, "trend_ema": 50,
        "n_confirm_tfs": 1, "atr_stop_mult": 1.2, "atr_target_mult": 2.5,
        "min_volume_mult": 1.1, "allow_shorts": False, "volume_lookback": 20,
    },
    "pivot_points": {
        "use_fib": False, "bounce_buffer_pct": 0.002,
        "breakout_confirm_pct": 0.001, "volume_confirm_mult": 1.1,
        "atr_stop_mult": 1.0, "atr_target_mult": 2.0, "allow_shorts": True,
    },
    "orb_8am": {
        "orb_start_hour": 8, "orb_start_minute": 0, "orb_minutes": 30,
        "atr_stop_mult": 0.5, "atr_target_mult": 1.0, "target_range_mult": 1.0,
        "volume_confirm_mult": 1.3, "allow_shorts": True,
        "max_entries_per_day": 2, "no_trade_after_hour": 14,
    },
    "liquidity_sweep": {
        "swing_lookback": 10, "sweep_wick_pct": 0.001,
        "atr_stop_mult": 0.75, "atr_target_mult": 2.0,
        "volume_confirm_mult": 1.0, "volume_lookback": 20,
        "allow_shorts": True, "max_sweep_age_bars": 8,
    },
    "flawless_victory": {
        "fast_ema": 9, "slow_ema": 21, "rsi_period": 14,
        "rsi_buy_min": 40, "rsi_buy_max": 68,
        "rsi_short_min": 32, "rsi_short_max": 60,
        "volume_spike_mult": 1.5, "volume_lookback": 20,
        "atr_expand_lookback": 5, "pivot_proximity_pct": 0.004,
        "pivot_lookback": 15, "atr_stop_mult": 1.0, "atr_target_mult": 2.5,
        "allow_shorts": True,
    },
    "alpaca_daily_income": {
        "min_price": 5.0, "orb_minutes": 5,
        "no_entry_before": "09:35", "no_entry_after": "15:30",
        "fast_ema": 9, "slow_ema": 21, "volume_lookback": 20,
        "min_relative_volume": 1.5, "atr_period": 14,
        "atr_stop_buffer": 0.20, "minimum_stop_pct": 0.002,
        "maximum_stop_pct": 0.025, "min_risk_reward": 1.5,
        "target_risk_reward": 2.0, "max_entries_per_day": 3,
    },
}

# Search only 2–3 safe parameters per strategy (≤ 9 combinations)
_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "tux_mtf_original":  {"fast_ema": [8, 9, 12], "atr_stop_mult": [1.5, 2.0], "atr_target_mult": [2.5, 3.0]},
    "tux_mtf_long_only": {"fast_ema": [8, 9, 12], "atr_stop_mult": [1.2, 1.5], "atr_target_mult": [2.0, 2.5]},
    "pivot_points":      {"atr_stop_mult": [0.75, 1.0, 1.25], "atr_target_mult": [1.5, 2.0, 2.5]},
    "orb_8am":           {"orb_minutes": [15, 30, 45], "volume_confirm_mult": [1.3, 1.5, 2.0]},
    "liquidity_sweep":   {"swing_lookback": [8, 10, 14], "atr_stop_mult": [0.5, 0.75, 1.0]},
    "flawless_victory":  {"rsi_buy_min": [35, 40, 45], "volume_spike_mult": [1.3, 1.5, 2.0]},
    "alpaca_daily_income": {"min_relative_volume": [1.3, 1.5, 1.8], "target_risk_reward": [1.5, 2.0]},
}


def _get_param_grid(strategy_id: str, base_params: dict) -> list[dict]:
    """Bounded parameter grid. Baseline is always included and tried first."""
    defaults = {**_STRATEGY_DEFAULTS.get(strategy_id, {}), **base_params}
    space = _SEARCH_SPACES.get(strategy_id)
    if not space:
        return [defaults]

    keys = list(space.keys())
    grid: list[dict] = []
    seen: set[str] = set()
    for combo in product(*space.values()):
        p = {**defaults, **dict(zip(keys, combo))}
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            grid.append(p)

    # Ensure baseline is first
    base_key = json.dumps(defaults, sort_keys=True)
    if base_key not in seen:
        grid.insert(0, defaults)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_bars_file() -> Path | None:
    """Auto-discover the richest available bar CSV by data span."""
    # Dated intraday files
    intraday = sorted(
        PROJECT_ROOT.glob("daily_system_outputs/dated/*/intraday_bars_1m_*.csv"),
        reverse=True,
    )
    daily = PROJECT_ROOT / "daily_system_outputs" / "daily_raw_bars.csv"

    candidates = list(intraday) + [
        daily,
        PROJECT_ROOT / "daily_system_outputs" / "intraday_bars_1m.csv",
    ]
    return next((p for p in candidates if p.exists() and p.stat().st_size > 10_000), None)


def _load_bars(path: Path) -> tuple[pd.DataFrame | None, str]:
    """Load + validate an OHLCV CSV. Returns (df, error_msg)."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return None, f"read failed: {exc}"

    df.columns = [c.lower().strip() for c in df.columns]

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return None, f"missing column: {col}"

    if "volume" not in df.columns:
        df["volume"] = 1_000_000.0

    ts_col = next((c for c in df.columns if c in ("timestamp", "datetime", "date", "time")), None)
    if ts_col is None:
        return None, "no timestamp/date column found"

    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"

    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    )
    df = df.dropna()
    if df.empty:
        return None, "no valid rows after cleaning"
    return df, ""


def _split_by_symbol(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for sym, grp in df.groupby("symbol"):
        result[str(sym)] = grp.drop(columns=["symbol"]).set_index("timestamp").sort_index()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Window generation
# ─────────────────────────────────────────────────────────────────────────────

def _get_windows(first_ts: pd.Timestamp, last_ts: pd.Timestamp, config: WFConfig) -> list[dict]:
    """Generate rolling walk-forward windows using pandas DateOffset."""
    windows = []
    cursor = first_ts.normalize()

    while True:
        train_end  = cursor + pd.DateOffset(months=config.train_months) - pd.Timedelta(days=1)
        val_start  = train_end  + pd.Timedelta(days=1)
        val_end    = val_start  + pd.DateOffset(months=config.val_months)   - pd.Timedelta(days=1)
        test_start = val_end    + pd.Timedelta(days=1)
        test_end   = test_start + pd.DateOffset(months=config.test_months)  - pd.Timedelta(days=1)

        if test_end > last_ts:
            break

        windows.append({
            "train_start": cursor,     "train_end": train_end,
            "val_start":   val_start,  "val_end":   val_end,
            "test_start":  test_start, "test_end":  test_end,
        })
        cursor += pd.DateOffset(months=config.step_months)

    return windows


def _slice_bars(
    bars_by_sym: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_bars: int,
) -> dict[str, pd.DataFrame]:
    result = {}
    for sym, bars in bars_by_sym.items():
        sliced = bars[(bars.index >= start) & (bars.index <= end)]
        if len(sliced) >= min_bars:
            result[sym] = sliced
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Context builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(strategy_id: str, bars: pd.DataFrame) -> dict[str, Any]:
    """Build the minimal context dict required per strategy type."""
    if strategy_id != "pivot_points" or len(bars) < 5:
        return {}
    lb = bars.tail(min(10, len(bars)))
    h = float(lb["high"].max())
    l = float(lb["low"].min())
    c = float(lb["close"].iloc[-1])
    pp = (h + l + c) / 3
    r1, r2 = 2 * pp - l, pp + (h - l)
    s1, s2 = 2 * pp - h, pp - (h - l)
    return {
        "pivot_pp": pp, "pivot_r1": r1, "pivot_r2": r2,
        "pivot_r3": r2 + (r1 - pp), "pivot_s1": s1,
        "pivot_s2": s2, "pivot_s3": s2 - (pp - s1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio simulator
# ─────────────────────────────────────────────────────────────────────────────

class _Portfolio:
    """Bar-by-bar portfolio simulator. Signals on bar[T] execute at bar[T+1] open."""

    def __init__(self, starting_cash: float, config: WFConfig) -> None:
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.config = config
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.equity_by_day: dict[str, float] = {}
        self.skipped = 0
        self.rejected = 0
        self.daily_violations = 0
        self.daily_target_hits: set[str] = set()
        self.daily_loss_hits: set[str] = set()
        self._bars_with_position = 0
        self._total_bars = 0
        self._symbol_pnl: dict[str, float] = {}
        self._current_day: str | None = None
        self._daily_start_equity = starting_cash
        self._daily_blocked = False

    @property
    def equity(self) -> float:
        return self.cash + sum(p["shares"] * p["last_price"] for p in self.positions.values())

    def process_bar(
        self,
        ts: pd.Timestamp,
        sym: str,
        bar: pd.Series,
        prev_signal: pd.Series | None,
    ) -> None:
        self._total_bars += 1
        day_str = ts.strftime("%Y-%m-%d")

        if day_str != self._current_day:
            self._daily_start_equity = self.equity
            self._daily_blocked = False
            self._current_day = day_str

        if sym in self.positions:
            self.positions[sym]["last_price"] = float(bar["close"])

        if sym in self.positions:
            self._check_exit(ts, sym, bar, day_str)

        self._update_daily_locks(day_str)

        if prev_signal is not None and sym not in self.positions:
            self._check_entry(ts, sym, bar, prev_signal)

        if self.positions:
            self._bars_with_position += 1

        self.equity_by_day[day_str] = self.equity

    def _update_daily_locks(self, day_str: str) -> None:
        daily_pnl = self.equity - self._daily_start_equity
        if daily_pnl >= self.config.daily_profit_target_dollars:
            self._daily_blocked = True
            self.daily_target_hits.add(day_str)
        if daily_pnl <= -(self._daily_start_equity * self.config.max_daily_loss_pct):
            self._daily_blocked = True
            if day_str not in self.daily_loss_hits:
                self.daily_violations += 1
            self.daily_loss_hits.add(day_str)

    def _check_exit(self, ts: pd.Timestamp, sym: str, bar: pd.Series, _: str) -> None:
        pos = self.positions[sym]
        sl = self.config.slippage_bps / 10_000
        lo, hi = float(bar["low"]), float(bar["high"])
        exit_price: float | None = None
        exit_reason: str | None = None

        if pos["direction"] == "long":
            if lo <= pos["stop"]:
                exit_price, exit_reason = pos["stop"] * (1 - sl), "stop"
            elif hi >= pos["target"]:
                exit_price, exit_reason = pos["target"] * (1 - sl), "target"
        else:
            if hi >= pos["stop"]:
                exit_price, exit_reason = pos["stop"] * (1 + sl), "stop"
            elif lo <= pos["target"]:
                exit_price, exit_reason = pos["target"] * (1 + sl), "target"

        if exit_price is None:
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            return

        if pos["direction"] == "long":
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            self.cash += exit_price * pos["shares"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["shares"]
            self.cash += pos["cost"] + pnl

        self._record_trade(sym, pos, exit_price, exit_reason, ts, pnl)
        self._symbol_pnl[sym] = self._symbol_pnl.get(sym, 0.0) + pnl
        del self.positions[sym]

    def _check_entry(self, ts: pd.Timestamp, sym: str, bar: pd.Series, sig: pd.Series) -> None:
        if self._daily_blocked:
            self.skipped += 1
            return
        if len(self.positions) >= self.config.max_open_positions:
            self.rejected += 1
            return

        direction: str | None = None
        if bool(sig.get("buy_signal", False)):
            direction = "long"
        elif bool(sig.get("short_signal", False)):
            direction = "short"
        if direction is None:
            return

        try:
            stop   = float(sig["stop_price"])
            target = float(sig["target_price"])
        except (KeyError, TypeError, ValueError):
            self.skipped += 1
            return
        if math.isnan(stop) or math.isnan(target):
            self.skipped += 1
            return

        sl = self.config.slippage_bps / 10_000
        raw_open = float(bar["open"])
        entry_price = raw_open * (1 + sl) if direction == "long" else raw_open * (1 - sl)

        equity = self.equity
        max_pos = min(equity * self.config.max_position_weight, self.config.max_order_dollars)
        risk_per_share = abs(entry_price - stop)
        if risk_per_share <= 0:
            self.rejected += 1
            return
        risk_dollars = equity * self.config.risk_per_trade_pct
        shares = min(int(max_pos / entry_price), int(risk_dollars / risk_per_share))
        if shares <= 0:
            self.rejected += 1
            return
        cost = shares * entry_price

        if cost > self.cash * (1 - self.config.cash_buffer_pct):
            self.rejected += 1
            return

        gross = sum(p["shares"] * p["entry_price"] for p in self.positions.values())
        if equity > 0 and (gross + cost) / equity > self.config.max_gross_exposure:
            self.rejected += 1
            return

        self.cash -= cost
        self.positions[sym] = {
            "direction": direction,
            "entry_price": entry_price,
            "shares": shares,
            "cost": cost,
            "stop": stop,
            "target": target,
            "entry_ts": str(ts),
            "last_price": entry_price,
            "bars_held": 0,
            "initial_risk_dollars": risk_per_share * shares,
        }

    def _record_trade(
        self, sym: str, pos: dict, exit_price: float, reason: str,
        exit_ts: pd.Timestamp, pnl: float,
    ) -> None:
        self.trades.append({
            "symbol": sym,
            "direction": pos["direction"],
            "entry_ts": pos["entry_ts"],
            "exit_ts": str(exit_ts),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price":  round(exit_price, 4),
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / pos["cost"], 5) if pos["cost"] > 0 else 0.0,
            "r_multiple": round(pnl / pos["initial_risk_dollars"], 4) if pos.get("initial_risk_dollars", 0) > 0 else 0.0,
            "bars_held": pos.get("bars_held", 0),
            "exit_reason": reason,
        })

    def close_all(self, ts: pd.Timestamp, bars_by_sym: dict[str, pd.DataFrame]) -> None:
        sl = self.config.slippage_bps / 10_000
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            if sym in bars_by_sym and len(bars_by_sym[sym]) > 0:
                close_p = float(bars_by_sym[sym]["close"].iloc[-1])
            else:
                close_p = pos["last_price"]
            fill = close_p * (1 - sl) if pos["direction"] == "long" else close_p * (1 + sl)
            if pos["direction"] == "long":
                pnl = (fill - pos["entry_price"]) * pos["shares"]
                self.cash += fill * pos["shares"]
            else:
                pnl = (pos["entry_price"] - fill) * pos["shares"]
                self.cash += pos["cost"] + pnl
            self._record_trade(sym, pos, fill, "window_end", ts, pnl)
            self._symbol_pnl[sym] = self._symbol_pnl.get(sym, 0.0) + pnl
        self.positions.clear()

    @property
    def exposure_pct(self) -> float:
        return self._bars_with_position / self._total_bars if self._total_bars else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_portfolio(
    strategy_id: str,
    bars_by_sym: dict[str, pd.DataFrame],
    params: dict,
    config: WFConfig,
) -> _Portfolio:
    """Generate signals for all symbols, then simulate portfolio bar-by-bar."""
    try:
        strategy = get_strategy(strategy_id, deepcopy(params))
    except Exception:
        return _Portfolio(config.starting_cash, config)

    signals_by_sym: dict[str, pd.DataFrame] = {}
    for sym, bars in bars_by_sym.items():
        ctx = _build_context(strategy_id, bars)
        try:
            signals_by_sym[sym] = strategy.generate_signals(bars, ctx)
        except Exception:
            pass

    # Build unified event stream: (ts, sym, bar, prev_signal|None)
    events: list[tuple] = []
    for sym, bars in bars_by_sym.items():
        sigs = signals_by_sym.get(sym)
        for i, (ts, bar) in enumerate(bars.iterrows()):
            prev_sig: pd.Series | None = None
            if sigs is not None and i > 0:
                prev_ts = bars.index[i - 1]
                if prev_ts in sigs.index:
                    prev_sig = sigs.loc[prev_ts]
            events.append((ts, sym, bar, prev_sig))

    events.sort(key=lambda e: e[0])

    portfolio = _Portfolio(config.starting_cash, config)
    for ts, sym, bar, prev_sig in events:
        portfolio.process_bar(ts, sym, bar, prev_sig)

    portfolio.close_all(
        max(bars.index[-1] for bars in bars_by_sym.values()),
        bars_by_sym,
    )
    return portfolio


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(portfolio: _Portfolio) -> dict[str, Any]:
    starting = portfolio.starting_cash
    final_eq = portfolio.equity
    trades   = portfolio.trades

    net_pnl       = final_eq - starting
    total_ret_pct = net_pnl / starting * 100

    day_values = [v for _, v in sorted(portfolio.equity_by_day.items())]
    if len(day_values) < 2:
        day_values = [starting, final_eq]

    eq_arr = np.array(day_values, dtype=float)
    daily_rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-9)

    rf = 0.05 / 252
    excess = daily_rets - rf
    std_e  = excess.std()
    sharpe  = float(excess.mean() / std_e * np.sqrt(252)) if std_e > 1e-10 else 0.0

    down = excess[excess < 0]
    std_d = down.std() if len(down) > 1 else 0.0
    sortino = float(excess.mean() / std_d * np.sqrt(252)) if std_d > 1e-10 else (
        float(excess.mean() * np.sqrt(252)) if excess.mean() > 0 else 0.0
    )

    peak = np.maximum.accumulate(eq_arr)
    dd   = (eq_arr - peak) / np.maximum(peak, 1e-9)
    max_dd = float(-dd.min() * 100) if len(dd) else 0.0

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n      = len(trades)

    win_rate = len(wins) / n * 100 if n else 0.0
    pf_raw   = sum(wins) / (-sum(losses)) if losses else (99.0 if wins else 1.0)
    profit_factor = min(pf_raw, 99.0)

    expectancy    = sum(pnls) / n if n else 0.0
    avg_trade_ret = sum(t["return_pct"] for t in trades) / n * 100 if n else 0.0
    avg_hold      = sum(t["bars_held"] for t in trades) / n if n else 0.0
    avg_r          = sum(t.get("r_multiple", 0.0) for t in trades) / n if n else 0.0
    day_pnls: list[float] = []
    prior_equity = starting
    for day_equity in day_values:
        day_pnls.append(float(day_equity) - prior_equity)
        prior_equity = float(day_equity)
    trading_days = max(len(day_values), 1)

    sym_pnl  = portfolio._symbol_pnl
    best_sym  = max(sym_pnl, key=sym_pnl.get) if sym_pnl else "n/a"
    worst_sym = min(sym_pnl, key=sym_pnl.get) if sym_pnl else "n/a"

    return {
        "total_return_pct":    round(total_ret_pct, 4),
        "final_equity":        round(final_eq, 2),
        "net_pnl":             round(net_pnl, 2),
        "sharpe_ratio":        round(sharpe, 4),
        "sortino_ratio":       round(sortino, 4),
        "max_drawdown_pct":    round(max_dd, 4),
        "profit_factor":       round(profit_factor, 4),
        "win_rate_pct":        round(win_rate, 2),
        "expectancy_dollars":  round(expectancy, 2),
        "total_trades":        n,
        "avg_trade_return_pct":round(avg_trade_ret, 4),
        "avg_hold_bars":       round(avg_hold, 1),
        "average_r_per_trade": round(avg_r, 4),
        "average_daily_pnl":   round(float(np.mean(day_pnls)) if day_pnls else 0.0, 2),
        "trades_per_day":      round(n / trading_days, 2),
        "days_hitting_profit_target": len(portfolio.daily_target_hits),
        "days_hitting_max_loss": len(portfolio.daily_loss_hits),
        "exposure_pct":        round(portfolio.exposure_pct * 100, 2),
        "daily_loss_violations": portfolio.daily_violations,
        "skipped_signals":     portfolio.skipped,
        "rejected_orders":     portfolio.rejected,
        "best_symbol":         best_sym,
        "worst_symbol":        worst_sym,
    }


def _score_window(metrics: dict) -> float:
    """Validation-window scoring: risk-adjusted return. Penalizes < 2 trades."""
    if metrics.get("total_trades", 0) < 2:
        return -999.0
    ret = metrics.get("total_return_pct", 0.0)
    dd  = metrics.get("max_drawdown_pct", 1.0)
    return ret / (1.0 + dd / 100.0)


def _compute_param_stability(history: list[dict]) -> float:
    """
    Stability ∈ [0,1]. Measures consistency of selected params across windows.
    High CV across windows → low stability.
    """
    if len(history) < 2:
        return 1.0
    num_keys = [
        k for k in history[0]
        if isinstance(history[0][k], (int, float)) and not isinstance(history[0][k], bool)
    ]
    if not num_keys:
        return 1.0
    cvs = []
    for key in num_keys:
        vals = np.array([p[key] for p in history if key in p], dtype=float)
        if len(vals) < 2 or vals.mean() == 0:
            continue
        cvs.append(vals.std() / abs(vals.mean()))
    return float(1 / (1 + np.mean(cvs))) if cvs else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(run_windows: list[dict], params_history: list[dict]) -> dict[str, Any]:
    if not run_windows:
        return {"num_windows_run": 0, "num_profitable_windows": 0}

    rets      = [w["test_metrics"]["total_return_pct"] for w in run_windows]
    sharpes   = [w["test_metrics"]["sharpe_ratio"]      for w in run_windows]
    sortinos  = [w["test_metrics"]["sortino_ratio"]     for w in run_windows]
    dds       = [w["test_metrics"]["max_drawdown_pct"]  for w in run_windows]
    pfs       = [w["test_metrics"]["profit_factor"]     for w in run_windows if w["test_metrics"]["profit_factor"] < 90]

    compounded = 1.0
    for r in rets:
        compounded *= (1 + r / 100)
    compounded_pct = (compounded - 1) * 100

    profitable = sum(1 for r in rets if r > 0)

    return {
        "num_windows":              len(run_windows),
        "num_profitable_windows":   profitable,
        "num_skipped_windows":      0,   # filled by caller
        "compounded_oos_return_pct": round(compounded_pct, 4),
        "avg_monthly_return_pct":    round(float(np.mean(rets)), 4),
        "monthly_win_rate_pct":      round(profitable / len(run_windows) * 100, 2),
        "avg_sharpe":                round(float(np.mean(sharpes)), 4),
        "median_sharpe":             round(float(np.median(sharpes)), 4),
        "avg_sortino":               round(float(np.mean(sortinos)), 4),
        "avg_max_drawdown_pct":      round(float(np.mean(dds)), 4),
        "worst_window_drawdown_pct": round(float(max(dds)), 4),
        "avg_profit_factor":         round(float(np.mean(pfs)), 4) if pfs else 0.0,
        "parameter_stability":       round(_compute_param_stability(params_history), 4),
        "total_trades_all_windows":  sum(w["test_metrics"]["total_trades"] for w in run_windows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main walk-forward function
# ─────────────────────────────────────────────────────────────────────────────

def run_walkforward_strategy(
    strategy_id: str,
    bars_df: pd.DataFrame,
    config: WFConfig,
    base_params: dict | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Full walk-forward for one strategy. Returns JSON-serializable result dict.
    composite_score is 0.0 — filled later by walkforward_all_strategies.py.
    """
    base_params = base_params or {}

    # Intraday setup strategies require distinct timestamps within each day.
    if strategy_id in {"orb_8am", "alpaca_daily_income"}:
        unique_times_per_day = bars_df.groupby(bars_df["timestamp"].dt.date)["timestamp"].nunique().mean()
        if unique_times_per_day < 5:
            return {
                "strategy_id": strategy_id,
                "status": "skipped",
                "skip_reason": (
                    f"{strategy_id} requires intraday 1-minute bars with ET session timestamps. "
                    f"Available data averages {unique_times_per_day:.1f} bars/day — daily bars insufficient."
                ),
                "minimum_required": {
                    "bars_per_day": "≥ 30 intraday 1-min bars",
                    "data_files": "intraday_bars_1m_YYYYMMDD.csv with ≥5 months of 1-min OHLCV",
                },
                "windows": [], "aggregate": {}, "composite_score": 0.0,
            }

    bars_by_sym = _split_by_symbol(bars_df)
    global_first = bars_df["timestamp"].min()
    global_last  = bars_df["timestamp"].max()
    windows = _get_windows(global_first, global_last, config)

    if not windows:
        span = (global_last - global_first).days
        need  = (config.train_months + config.val_months + config.test_months) * 30
        return {
            "strategy_id": strategy_id,
            "status": "insufficient_data",
            "skip_reason": (
                f"data spans {span} days; need ≥{need} days for one window "
                f"(train={config.train_months}mo + val={config.val_months}mo + test={config.test_months}mo)"
            ),
            "data_available": {
                "first_date": str(global_first.date()),
                "last_date":  str(global_last.date()),
                "span_days":  span,
                "symbols":    len(bars_by_sym),
                "total_bars": len(bars_df),
            },
            "minimum_required": {
                "span_days":  need,
                "data_files": (
                    "daily_system_outputs/daily_raw_bars.csv (already has ~17 months of daily OHLCV) "
                    "or intraday CSVs with ≥5 months history"
                ),
            },
            "windows": [], "aggregate": {}, "composite_score": 0.0,
        }

    param_grid = _get_param_grid(strategy_id, base_params)

    if verbose:
        print(f"  [{strategy_id}] {len(windows)} windows | "
              f"{len(bars_by_sym)} symbols | {len(param_grid)} param combos")

    window_results: list[dict] = []
    params_history: list[dict] = []
    skipped_count = 0

    for wi, w in enumerate(windows):
        train_bars = _slice_bars(bars_by_sym, w["train_start"], w["train_end"], config.min_bars_per_slice)
        val_bars   = _slice_bars(bars_by_sym, w["val_start"],   w["val_end"],   config.min_bars_per_slice)
        test_bars  = _slice_bars(bars_by_sym, w["test_start"],  w["test_end"],  config.min_bars_per_slice)

        syms = set(train_bars) & set(val_bars) & set(test_bars)
        if len(syms) < config.min_symbols_per_window:
            window_results.append({
                "window_idx": wi, "skipped": True,
                "skip_reason": f"{len(syms)} symbols have full coverage (need {config.min_symbols_per_window})",
                "test_start": str(w["test_start"].date()), "test_end": str(w["test_end"].date()),
            })
            skipped_count += 1
            continue

        s_train = {s: train_bars[s] for s in syms}
        s_val   = {s: val_bars[s]   for s in syms}
        s_test  = {s: test_bars[s]  for s in syms}

        # ── Parameter search on validation slice ─────────────────────────────
        best_params   = param_grid[0]
        best_val_score = -999.0
        for params in param_grid:
            try:
                port = _simulate_portfolio(strategy_id, s_val, params, config)
                score = _score_window(_compute_metrics(port))
                if score > best_val_score:
                    best_val_score, best_params = score, params
            except Exception:
                pass
        params_history.append(deepcopy(best_params))

        # ── Test slice (out-of-sample) ────────────────────────────────────────
        try:
            test_port    = _simulate_portfolio(strategy_id, s_test, best_params, config)
            test_metrics = _compute_metrics(test_port)
            test_trades  = test_port.trades
            test_equity  = [round(v, 2) for v in test_port.equity_by_day.values()]
        except Exception as exc:
            window_results.append({
                "window_idx": wi, "skipped": True,
                "skip_reason": f"simulation error: {exc}",
                "test_start": str(w["test_start"].date()), "test_end": str(w["test_end"].date()),
            })
            skipped_count += 1
            continue

        window_results.append({
            "window_idx":    wi, "skipped": False,
            "train_start":   str(w["train_start"].date()), "train_end": str(w["train_end"].date()),
            "val_start":     str(w["val_start"].date()),   "val_end":   str(w["val_end"].date()),
            "test_start":    str(w["test_start"].date()),  "test_end":  str(w["test_end"].date()),
            "symbols_tested":list(syms),
            "selected_params":best_params,
            "val_score":     round(best_val_score, 4),
            "test_metrics":  test_metrics,
            "test_equity_curve": test_equity,
            "test_trades":   test_trades,
        })

        if verbose:
            m = test_metrics
            print(f"    W{wi:02d} {w['test_start'].date()}→{w['test_end'].date()} | "
                  f"ret={m['total_return_pct']:+.2f}% sharpe={m['sharpe_ratio']:.2f} "
                  f"trades={m['total_trades']} dd={m['max_drawdown_pct']:.1f}%")

    run_windows = [w for w in window_results if not w.get("skipped")]
    agg = _aggregate(run_windows, params_history)
    agg["num_skipped_windows"] = skipped_count

    return {
        "strategy_id":    strategy_id,
        "status":         "complete" if run_windows else "no_windows_ran",
        "config":         asdict(config),
        "data_source":    str(_find_best_bars_file() or "unknown"),
        "windows":        window_results,
        "aggregate":      agg,
        "composite_score":0.0,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward analysis for one strategy variant")
    parser.add_argument("--strategy-id", required=True, choices=list(STRATEGY_REGISTRY.keys()))
    parser.add_argument("--bars", default=None)
    parser.add_argument("--initial-cash",       type=float, default=100_000.0)
    parser.add_argument("--train-months",        type=int,   default=3)
    parser.add_argument("--validation-months",   type=int,   default=1)
    parser.add_argument("--test-months",         type=int,   default=1)
    parser.add_argument("--step-months",         type=int,   default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = WFConfig(
        train_months=args.train_months,
        val_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        starting_cash=args.initial_cash,
    )

    bars_path = Path(args.bars) if args.bars else _find_best_bars_file()
    if bars_path is None or not bars_path.exists():
        print("[walkforward] ERROR: no bar data found.\n"
              "  Ensure one of these exists:\n"
              "    daily_system_outputs/daily_raw_bars.csv\n"
              "    daily_system_outputs/dated/*/intraday_bars_1m_*.csv\n"
              "  Or pass --bars <path>")
        sys.exit(1)

    if not args.quiet:
        print(f"=== Walk-Forward: {args.strategy_id} ===")
        print(f"  data:  {bars_path}")
        print(f"  cash:  ${config.starting_cash:,.0f}")
        print(f"  windows: train={config.train_months}mo val={config.val_months}mo "
              f"test={config.test_months}mo step={config.step_months}mo\n")

    bars_df, err = _load_bars(bars_path)
    if bars_df is None:
        print(f"[walkforward] ERROR loading bars: {err}")
        sys.exit(1)

    # Load baseline params from YAML
    base_params: dict = {}
    try:
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "strategy_experiments.yaml").read_text())
        for vid, vcfg in cfg.get("variants", {}).items():
            if vid == args.strategy_id:
                base_params = vcfg.get("params", {})
                break
    except Exception:
        pass

    result = run_walkforward_strategy(
        args.strategy_id, bars_df, config, base_params, verbose=not args.quiet,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str  = datetime.now().strftime("%Y%m%d")
    out_path  = OUTPUT_DIR / f"{date_str}_walkforward_{args.strategy_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    if not args.quiet:
        status = result.get("status")
        if status in ("insufficient_data", "skipped"):
            print(f"\n[{status.upper()}] {result['skip_reason']}")
            di = result.get("data_available", {})
            mr = result.get("minimum_required", {})
            if di:
                print(f"  Available: {di.get('span_days','?')} days "
                      f"({di.get('first_date','?')} → {di.get('last_date','?')}), "
                      f"{di.get('symbols','?')} symbols, {di.get('total_bars','?')} bars")
            if mr:
                print(f"  Required:  {mr}")
        else:
            agg = result.get("aggregate", {})
            print(f"\n  OOS return:  {agg.get('compounded_oos_return_pct', 0):+.2f}%")
            print(f"  Avg Sharpe:  {agg.get('avg_sharpe', 0):.3f}")
            print(f"  Win rate:    {agg.get('monthly_win_rate_pct', 0):.0f}%")
            print(f"  Windows:     {agg.get('num_windows', 0)} run / "
                  f"{agg.get('num_skipped_windows', 0)} skipped")
        print(f"\n  Output: {out_path}")


if __name__ == "__main__":
    main()
