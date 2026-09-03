"""
tests/test_ddr_reward.py — data-free unit tests for the DDR reward mode (03-01).

No market data, no SB3 training, no env/rollout. Pure math on SharpeRewardCalculator.
(Importing rl_agent_core does pull torch/SB3 at module import — that is unavoidable
without extracting the class, and is acceptable: nothing here trains or needs data.)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from rl_agent_core import SharpeRewardCalculator


def _expected_sharpe_rolling(window_returns, step, sharpe_weight=0.7, risk_free=0.0):
    """Independent replica of the legacy sharpe_rolling formula (golden reference)."""
    if len(window_returns) < 2:
        return float(step)
    arr = np.array(window_returns, dtype=np.float64)
    excess = arr - risk_free
    sharpe = float(excess.mean() / max(excess.std(), 1e-9))
    return float(sharpe_weight * sharpe + (1.0 - sharpe_weight) * step)


def _deterministic_stream(n):
    """Index-derived pseudo-random-ish returns in ~[-0.06, 0.06]; no RNG, fully reproducible."""
    return [(((i * 37) % 13) - 6) / 100.0 for i in range(n)]


def test_default_mode_parity():
    """AC-3: mode='sharpe_rolling' reproduces the documented legacy formula exactly."""
    calc = SharpeRewardCalculator(mode="sharpe_rolling")
    mirror = deque(maxlen=60)
    for step in _deterministic_stream(80):
        mirror.append(step)
        expected = _expected_sharpe_rolling(mirror, step)
        got = calc.update(step)
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12), (got, expected)


def test_default_is_sharpe_rolling():
    """AC-2: constructing with no mode arg = legacy behavior."""
    assert SharpeRewardCalculator().mode == "sharpe_rolling"


def test_ddr_no_nan_inf():
    """AC-1/AC-3: DDR outputs finite across fixed + deterministic streams."""
    calc = SharpeRewardCalculator(mode="ddr")
    stream = [0.01, 0.02, -0.03, 0.0, 0.01] + _deterministic_stream(50)
    for step in stream:
        v = calc.update(step)
        assert math.isfinite(v), f"non-finite reward {v} at step {step}"


def test_ddr_warmup_raw_fallback():
    """AC-1: first step has no prior moments -> raw fallback, finite, no divide-by-zero."""
    calc = SharpeRewardCalculator(mode="ddr")
    first = calc.update(0.01)
    assert math.isfinite(first) and first == 0.01


def test_ddr_downside_sensitivity():
    """AC-1: from IDENTICAL prior state, a loss is penalised more than a gain.

    Correct form: feed both calcs an identical prefix (so A, DD^2, n match), then
    differ ONLY on the final step's sign. This isolates the current return's effect
    from the risk denominator (comparing across different prefixes would not).
    """
    prefix = [0.01, -0.02, 0.01, -0.01, 0.02]
    a = SharpeRewardCalculator(mode="ddr")
    b = SharpeRewardCalculator(mode="ddr")
    for x in prefix:
        a.update(x)
        b.update(x)
    r_gain = a.update(0.02)
    r_loss = b.update(-0.02)
    assert r_gain > r_loss, (r_gain, r_loss)
    assert r_loss < 0, r_loss  # a loss is a penalty


def test_ddr_determinism():
    """AC-3: identical input stream -> identical outputs."""
    seq = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]
    a = SharpeRewardCalculator(mode="ddr")
    b = SharpeRewardCalculator(mode="ddr")
    assert [a.update(x) for x in seq] == [b.update(x) for x in seq]


def test_reset_clears_ddr_state():
    """AC-2: reset() clears DDR state so a reused calc == a fresh calc."""
    seq = [0.01, -0.02, 0.03, -0.01]
    reused = SharpeRewardCalculator(mode="ddr")
    for x in seq:
        reused.update(x)
    reused.reset()
    assert reused._ddr_n == 0 and reused._A == 0.0 and reused._DD2 == 0.0
    fresh = SharpeRewardCalculator(mode="ddr")
    assert [reused.update(x) for x in seq] == [fresh.update(x) for x in seq]
