#!/usr/bin/env python3
"""
scripts/smoke_test_variants.py

Smoke-test every strategy variant:
  - synthetic bars  → generate_signals() → schema assertions
  - confirms SIGNAL_COLUMNS present
  - confirms stop/target non-NaN when signals fire
  - confirms strategy_id populated
  - confirms no exceptions

Exit code 0 = all pass. Non-zero = failures.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.base import SIGNAL_COLUMNS
from strategies.registry import STRATEGY_REGISTRY, get_strategy

ET = ZoneInfo("America/New_York")
N_BARS = 250   # generous — all strategies need fewer than this


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic bar builders
# ─────────────────────────────────────────────────────────────────────────────

def _base_bars(n: int, seed: int = 42, start_price: float = 100.0) -> pd.DataFrame:
    """Generic OHLCV bars, no timezone."""
    rng = np.random.default_rng(seed)
    close = start_price + np.cumsum(rng.normal(0, 0.15, n))
    close = np.maximum(close, 1.0)
    noise = rng.uniform(0.001, 0.005, n)
    return pd.DataFrame({
        "open":   close * (1 - noise / 2),
        "high":   close * (1 + noise),
        "low":    close * (1 - noise),
        "close":  close,
        "volume": rng.integers(5_000, 100_000, n).astype(float),
    })


def _make_generic_bars(n: int = N_BARS) -> pd.DataFrame:
    """Generic bars — no timezone, suitable for most strategies."""
    return _base_bars(n)


def _make_tux_bars(n: int = N_BARS) -> pd.DataFrame:
    """
    Bars designed to produce at least one bull cross (fast EMA crosses above slow EMA).
    First half trending down, second half trending up.
    """
    rng = np.random.default_rng(7)
    down = 100 + np.cumsum(rng.normal(-0.2, 0.1, n // 2))
    up   = down[-1] + np.cumsum(rng.normal(+0.3, 0.1, n - n // 2))
    close = np.concatenate([down, up])
    close = np.maximum(close, 1.0)
    noise = rng.uniform(0.001, 0.004, n)
    return pd.DataFrame({
        "open":   close * (1 - noise / 2),
        "high":   close * (1 + noise),
        "low":    close * (1 - noise),
        "close":  close,
        "volume": rng.integers(8_000, 80_000, n).astype(float),
    })


def _make_orb_bars(n: int = N_BARS) -> pd.DataFrame:
    """
    Bars with timezone-aware UTC timestamps starting at 8:00 ET.
    Includes an ORB window 8:00-8:30 ET and a breakout bar after 8:30 ET.
    """
    start_et = datetime(2026, 1, 2, 8, 0, tzinfo=ET)
    ts_et = [start_et + timedelta(minutes=i) for i in range(n)]
    ts_utc = pd.DatetimeIndex([t.astimezone(timezone.utc) for t in ts_et])

    rng = np.random.default_rng(13)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.maximum(close, 1.0)

    # Force bars 0-29 to be the ORB range (8:00-8:30 ET)
    orb_high = float(close[:30].max()) + 0.5
    orb_low  = float(close[:30].min()) - 0.5

    # Force bar 35 to break above ORB high with volume spike
    close[35] = orb_high + 0.3

    noise = rng.uniform(0.001, 0.004, n)
    high  = close * (1 + noise)
    high[35] = orb_high + 0.5   # clear breakout

    df = pd.DataFrame({
        "timestamp": ts_utc,
        "open":   close * (1 - noise / 2),
        "high":   high,
        "low":    close * (1 - noise),
        "close":  close,
        "volume": rng.integers(5_000, 80_000, n).astype(float),
    })
    # Volume spike on breakout bar
    df.loc[35, "volume"] = 150_000.0
    return df


def _make_pivot_bars(n: int = N_BARS) -> dict[str, Any]:
    """
    Bars + pre-computed context dict so pivot_points skips daily resampling.
    Price oscillates around PP=100, touching S1/R1.
    """
    rng = np.random.default_rng(99)
    pp, r1, r2, s1, s2 = 100.0, 101.5, 103.0, 98.5, 97.0

    # Oscillating close between s1 and r1
    t = np.linspace(0, 4 * np.pi, n)
    close = pp + 2.0 * np.sin(t) + rng.normal(0, 0.05, n)
    close = np.maximum(close, 1.0)
    noise = rng.uniform(0.001, 0.003, n)

    # Ensure at least some closes near S1 (buy zone) and near R1 (sell zone)
    close[40:45] = s1 + rng.uniform(-0.02, 0.02, 5)   # at S1, recovering
    close[80:85] = r1 + rng.uniform(-0.02, 0.02, 5)    # at R1

    bars = pd.DataFrame({
        "open":   close * (1 - noise / 2),
        "high":   close * (1 + noise),
        "low":    close * (1 - noise),
        "close":  close,
        "volume": rng.integers(8_000, 60_000, n).astype(float),
    })
    context = {
        "pivot_pp": pp, "pivot_r1": r1, "pivot_r2": r2,
        "pivot_r3": r2 + 1.5, "pivot_s1": s1, "pivot_s2": s2,
        "pivot_s3": s2 - 1.5,
    }
    return {"bars": bars, "context": context}


def _make_sweep_bars(n: int = N_BARS) -> pd.DataFrame:
    """
    Bars with a swing low followed by a wick sweep (low < swing_low, close > swing_low).
    swing_lookback=10 so we need at least 2*10+5=25 bars.
    """
    rng = np.random.default_rng(55)
    lb = 10
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.maximum(close, 1.0)
    noise = rng.uniform(0.001, 0.004, n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)

    # Create swing low at bar 30 (lb bars before and after are higher)
    swing_low_idx = 30
    close[swing_low_idx - lb: swing_low_idx + lb + 1] = 102.0
    close[swing_low_idx] = 98.0          # the actual swing low

    # Sweep bar: lb bars after detection (bar 30+lb = 40)
    sweep_idx = swing_low_idx + lb + 2    # 42 — after confirmation lag
    if sweep_idx < n:
        swing_low_val = 98.0
        low[sweep_idx]   = swing_low_val - 0.3   # wick below
        close[sweep_idx] = swing_low_val + 0.2   # closes above
        high[sweep_idx]  = swing_low_val + 0.5

    return pd.DataFrame({
        "open":   close * (1 - noise / 2),
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.integers(10_000, 70_000, n).astype(float),
    })


def _make_fv_bars(n: int = N_BARS) -> dict[str, Any]:
    """
    Bars designed to trigger flawless_victory (all 6 conditions).
    Also passes VWAP context so the VWAP check is deterministic.
    """
    rng = np.random.default_rng(77)
    # Create trending-up series so fast EMA > slow EMA after warmup
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.05, n))
    close = np.maximum(close, 1.0)
    noise = rng.uniform(0.001, 0.003, n)
    bars = pd.DataFrame({
        "open":   close * (1 - noise / 2),
        "high":   close * (1 + noise),
        "low":    close * (1 - noise),
        "close":  close,
        "volume": rng.integers(20_000, 100_000, n).astype(float),  # high volume
    })
    # Pass VWAP slightly below close so above_vwap=True
    vwap_series = pd.Series(close * 0.995, index=bars.index)
    context = {"vwap_series": vwap_series}
    return {"bars": bars, "context": context}


# ─────────────────────────────────────────────────────────────────────────────
# Assertions
# ─────────────────────────────────────────────────────────────────────────────

def _assert_schema(signals: pd.DataFrame, strategy_id: str) -> list[str]:
    """Return list of schema failures (empty = all pass)."""
    failures = []

    # All required columns present
    for col in SIGNAL_COLUMNS:
        if col not in signals.columns:
            failures.append(f"missing column: {col}")

    if failures:
        return failures  # stop early

    # strategy_id column is fully populated (no NaN)
    if signals["strategy_id"].isna().any():
        failures.append("strategy_id has NaN values")
    non_blank = signals["strategy_id"].dropna()
    if len(non_blank) and (non_blank != strategy_id).any():
        wrong = non_blank[non_blank != strategy_id].unique().tolist()
        failures.append(f"strategy_id has unexpected values: {wrong}")

    # Boolean signal columns contain only bool/int values
    for col in ["buy_signal", "sell_signal", "short_signal", "cover_signal"]:
        vals = signals[col].dropna().unique()
        if not all(v in (True, False, 0, 1, 0.0, 1.0) for v in vals):
            failures.append(f"{col} has non-boolean values: {vals[:5]}")

    # When buy_signal or short_signal is True, stop/target must be non-NaN
    for sig_col in ["buy_signal", "short_signal"]:
        active = signals[signals[sig_col].astype(bool)]
        if not active.empty:
            if active["stop_price"].isna().any():
                failures.append(f"{sig_col} rows have NaN stop_price")
            if active["target_price"].isna().any():
                failures.append(f"{sig_col} rows have NaN target_price")
            if active["risk_reward"].isna().any():
                failures.append(f"{sig_col} rows have NaN risk_reward")
            # stop/target must make sense directionally
            if sig_col == "buy_signal":
                bad = active[active["stop_price"] >= active["target_price"]]
                if not bad.empty:
                    failures.append(f"buy_signal: stop_price >= target_price on {len(bad)} row(s)")
            if sig_col == "short_signal":
                bad = active[active["stop_price"] <= active["target_price"]]
                if not bad.empty:
                    failures.append(f"short_signal: stop_price <= target_price on {len(bad)} row(s)")

    # risk_reward values must be positive when set
    rr_vals = signals["risk_reward"].dropna()
    neg_rr = rr_vals[rr_vals <= 0]
    if not neg_rr.empty:
        failures.append(f"risk_reward has {len(neg_rr)} non-positive value(s)")

    return failures


def _run_strategy_test(
    strategy_id: str,
    bars: pd.DataFrame,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run generate_signals and return a result dict."""
    result: dict[str, Any] = {
        "strategy_id": strategy_id,
        "bars_count": len(bars),
        "exception": None,
        "schema_failures": [],
        "signals_fired": {},
        "passed": False,
    }
    try:
        strategy = get_strategy(strategy_id)
        signals = strategy.generate_signals(bars, context)
        result["schema_failures"] = _assert_schema(signals, strategy_id)

        for col in ["buy_signal", "sell_signal", "short_signal", "cover_signal"]:
            result["signals_fired"][col] = int(signals[col].astype(bool).sum())

        result["passed"] = len(result["schema_failures"]) == 0
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test matrix
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_BAR_MAP = {
    "tux_mtf_original":  lambda: {"bars": _make_tux_bars(),    "context": {}},
    "tux_mtf_long_only": lambda: {"bars": _make_tux_bars(),    "context": {}},
    "pivot_points":      lambda: _make_pivot_bars(),
    "orb_8am":           lambda: {"bars": _make_orb_bars(),    "context": {}},
    "liquidity_sweep":   lambda: {"bars": _make_sweep_bars(),  "context": {}},
    "flawless_victory":  lambda: _make_fv_bars(),
}


def run_all_tests(verbose: bool = True) -> bool:
    """Run smoke tests for all strategies. Returns True if all pass."""
    results = []
    all_pass = True

    for strategy_id in STRATEGY_REGISTRY:
        bar_builder = STRATEGY_BAR_MAP.get(strategy_id)
        if bar_builder is None:
            print(f"[SKIP] {strategy_id}: no bar builder defined")
            continue

        spec = bar_builder()
        bars = spec["bars"]
        context = spec.get("context", {})
        result = _run_strategy_test(strategy_id, bars, context)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        if verbose:
            sigs = result["signals_fired"]
            sig_str = " ".join(f"{k}={v}" for k, v in sigs.items() if v > 0) or "no signals"
            print(f"[{status}] {strategy_id}: {sig_str}")
            if result["exception"]:
                print(f"       EXCEPTION: {result['exception']}")
                if result.get("traceback"):
                    for line in result["traceback"].strip().splitlines()[-4:]:
                        print(f"         {line}")
            for fail in result["schema_failures"]:
                print(f"       SCHEMA FAIL: {fail}")

        if not result["passed"]:
            all_pass = False

    # Also run a "generic bars" pass to check all strategies tolerate arbitrary input
    if verbose:
        print("\n--- Generic bars pass (no specialized bar construction) ---")
    generic = _make_generic_bars()
    for strategy_id in STRATEGY_REGISTRY:
        try:
            strategy = get_strategy(strategy_id)
            ctx: dict[str, Any] = {}
            if strategy_id == "pivot_points":
                ctx = {"pivot_pp": 100.0, "pivot_r1": 101.5, "pivot_r2": 103.0,
                       "pivot_s1": 98.5, "pivot_s2": 97.0}
            signals = strategy.generate_signals(generic, ctx)
            schema_fails = _assert_schema(signals, strategy_id)
            if schema_fails:
                if verbose:
                    print(f"[FAIL] {strategy_id} generic: {schema_fails}")
                all_pass = False
            elif verbose:
                print(f"[PASS] {strategy_id} generic")
        except Exception as exc:
            if verbose:
                print(f"[FAIL] {strategy_id} generic: {type(exc).__name__}: {exc}")
            all_pass = False

    return all_pass


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test all strategy variants")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=== QuantumQuant Strategy Smoke Tests ===\n")
    passed = run_all_tests(verbose=not args.quiet)
    print(f"\n{'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
