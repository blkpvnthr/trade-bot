"""
V2.9.8 strategy-parity, no-lookahead, and crisis tests.

These are the most important tests: they prove the live single-period decision
(`strat_decision.compute_target_portfolio`) reproduces the historical backtest
decision EXACTLY, that future bars cannot change today's decision, and that a
severe-bear/crisis state suppresses leverage.

Offline + deterministic (seeded synthetic prices, no network). Single-threaded
BLAS is pinned so the two-start SLSQP argmin is reproducible.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from strat_fixtures import make_synthetic_prices
from strat_allocator import AllocatorConfig
from strat import build_gate_schedule, momentum_view_panel, rolling_ic_panel, run_backtest
from strat_decision import compute_target_portfolio, DEFAULT_START, COLD_START_SESSIONS

# Documented tolerance: in-process, same BLAS/scipy, the reconstruction is exact
# to float noise. 1e-8 abs on weights is the contractual bar (see plan).
PARITY_ATOL = 1e-8


def _truncated_reference(prices, cfg, cutoff):
    """
    Run the reference backtest on prices truncated at `cutoff` and return the
    FINAL traced decision (prev_weights, sessions_since_rebalance, target, date).

    Truncating the reference is REQUIRED: rolling_ic_panel's ffill plateaus
    differently at the data edge, so only a truncated-at-d reference equals the
    live (production) decision at d.
    """
    px = prices.loc[:cutoff]
    gate = build_gate_schedule(px["SPY"], cfg)
    view = momentum_view_panel(px, cfg)
    ic = rolling_ic_panel(view, px)
    trace = []
    run_backtest(px, cfg, gate, view=view, ic=ic, start=DEFAULT_START,
                 label="ref", trace=trace)
    return trace[-1], px


# Build the parity cases once (each truncated backtest is a bit expensive).
_PRICES = make_synthetic_prices(end="2010-12-31")
_CFG = AllocatorConfig()
# Three cutoffs spanning BULL/DEFENSIVE, CRISIS, and RECOVERING (each runs its
# own truncated reference backtest, so keep the count modest).
_CUTOFFS = ["2007-09-14", "2008-11-20", "2010-05-14"]
_CASES = [_truncated_reference(_PRICES, _CFG, c) for c in _CUTOFFS]


def test_live_decision_matches_backtest_exactly():
    """compute_target_portfolio == run_backtest(truncated) final decision."""
    for rec, px in _CASES:
        d = rec["Date"]
        dec = compute_target_portfolio(
            px, prev_weights=rec["prev_weights"],
            sessions_since_rebalance=rec["sessions_since_rebalance"],
            cfg=_CFG, as_of_close=d, start=DEFAULT_START)
        got = dec.target_vector(_CFG)
        maxabs = float(np.max(np.abs(got - rec["target"])))
        assert maxabs <= PARITY_ATOL, f"parity mismatch @ {d.date()}: {maxabs:.2e}"
        assert dec.rebalanced == rec["rebalanced"], f"rebalanced flag @ {d.date()}"


def test_no_lookahead_future_bars_do_not_change_today():
    """Mutating bars AFTER as_of_close must not change the decision at as_of_close."""
    rec, px = _CASES[1]  # a mid-sample date with plenty of history and future room
    d = rec["Date"]
    prev = rec["prev_weights"]
    ssr = rec["sessions_since_rebalance"]

    base = compute_target_portfolio(_PRICES, prev_weights=prev,
                                    sessions_since_rebalance=ssr, cfg=_CFG,
                                    as_of_close=d, start=DEFAULT_START)

    # Corrupt everything strictly after d with wild values.
    mutated = _PRICES.copy()
    future = mutated.index > d
    for col in mutated.columns:
        if col == "CASH_RATE_3M":
            continue
        mutated.loc[future, col] = mutated.loc[future, col] * np.linspace(
            0.1, 5.0, int(future.sum()))
    after = compute_target_portfolio(mutated, prev_weights=prev,
                                     sessions_since_rebalance=ssr, cfg=_CFG,
                                     as_of_close=d, start=DEFAULT_START)

    assert np.array_equal(base.target_vector(_CFG), after.target_vector(_CFG))
    assert base.decision_hash == after.decision_hash


_GRID_MIN = pd.Timestamp("2007-06-01")  # comfortably inside the valid decision grid


def test_crisis_suppresses_leverage():
    """A severe-bear session forces leverage_cap == 0 (all-cash target)."""
    gate = build_gate_schedule(_PRICES["SPY"], _CFG)
    severe_dates = gate.index[gate["SevereBear"].to_numpy(bool)]
    severe_dates = severe_dates[severe_dates >= _GRID_MIN]
    assert len(severe_dates) > 0, "fixture must visit an in-grid severe-bear session"
    d = severe_dates[len(severe_dates) // 2]

    dec = compute_target_portfolio(
        _PRICES, prev_weights=np.full(len(_CFG.risk_assets), 0.05),
        sessions_since_rebalance=COLD_START_SESSIONS, cfg=_CFG,
        as_of_close=d, start=DEFAULT_START)

    assert dec.severe_bear is True
    assert dec.leverage_cap == 0.0
    assert dec.gross_exposure <= 1e-9, f"levered in crisis: gross={dec.gross_exposure}"


def test_crisis_state_caps_leverage_below_normal():
    """In CRISIS/DEFENSIVE (non-severe) leverage cap is crisis_max_leverage."""
    gate = build_gate_schedule(_PRICES["SPY"], _CFG)
    mask = (gate["RecoveryState"] == "CRISIS") & (~gate["SevereBear"].to_numpy(bool))
    crisis_dates = gate.index[mask.to_numpy(bool)]
    crisis_dates = crisis_dates[crisis_dates >= _GRID_MIN]
    if len(crisis_dates) == 0:
        return  # fixture-dependent; skip quietly
    d = crisis_dates[len(crisis_dates) // 2]
    dec = compute_target_portfolio(
        _PRICES, prev_weights=np.zeros(len(_CFG.risk_assets)),
        sessions_since_rebalance=COLD_START_SESSIONS, cfg=_CFG,
        as_of_close=d, start=DEFAULT_START)
    assert dec.leverage_cap <= _CFG.crisis_max_leverage + 1e-12
    assert dec.gross_exposure <= _CFG.crisis_max_leverage + 1e-6


def test_invalid_decision_date_rejected():
    """A date not on the strategy grid (e.g. weekend/pre-start) is rejected."""
    import pytest
    with pytest.raises(ValueError):
        compute_target_portfolio(
            _PRICES, prev_weights=np.zeros(len(_CFG.risk_assets)),
            sessions_since_rebalance=COLD_START_SESSIONS, cfg=_CFG,
            as_of_close="2004-01-01", start=DEFAULT_START)
