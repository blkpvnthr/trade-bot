from rl_meta.ranking import (tier_of, admit, validate, passes_perf,
                             MIN_TRADES, oos_pass)
from rl_meta.types import Metrics, Tier


def m(win_rate, pf, **kw):
    # Default trade_count clears the >=100 admission gate so tier/perf logic
    # is what gets exercised; override per-test to probe the count gate.
    base = dict(trade_count=120, avg_hold_days=4, expectancy=50, max_drawdown=0.1)
    base.update(kw)
    return Metrics(win_rate=win_rate, profit_factor=pf, **base)


def test_tier_a():
    assert tier_of(m(0.65, 2.5)) == Tier.A


def test_tier_b():
    assert tier_of(m(0.62, 1.5)) == Tier.B


def test_tier_c():
    assert tier_of(m(0.50, 2.2)) == Tier.C


def test_reject():
    assert tier_of(m(0.50, 1.2)) == Tier.REJECT


def test_admission_or_rule():
    assert admit(m(0.65, 1.0))      # win-rate path (>=100 trades)
    assert admit(m(0.40, 2.5))      # profit-factor path
    assert not admit(m(0.40, 1.2))  # neither limb


def test_admission_requires_min_trades():
    # The AND-gate: strong performance but too few trades -> not admitted.
    assert not admit(m(0.70, 3.0, trade_count=99))
    assert admit(m(0.70, 3.0, trade_count=100))
    assert MIN_TRADES == 100


def test_validation_min_trades():
    ok, reason = validate(m(0.65, 2.5, trade_count=5))
    assert not ok and "too few" in reason


def test_validation_overfit_guard():
    ok, reason = validate(m(0.99, 5.0, trade_count=120))
    assert not ok and "overfitting" in reason


def test_validation_not_executable():
    ok, reason = validate(m(0.65, 2.5), executable=False)
    assert not ok and reason == "not executable"


def test_validation_insufficient_data():
    ok, reason = validate(m(0.65, 2.5), has_data=False)
    assert not ok and "insufficient Alpaca data" in reason


def test_validation_pass():
    ok, reason = validate(m(0.65, 2.5))
    assert ok and reason == "eligible"


def test_oos_gate():
    assert oos_pass(m(0.65, 2.5, trade_count=40))[0]      # enough OOS + perf
    assert not oos_pass(m(0.65, 2.5, trade_count=10))[0]  # too few OOS trades
    assert not oos_pass(m(0.45, 1.1, trade_count=40))[0]  # fails perf


def test_inf_pf_needs_real_sample():
    # No-loss PF=inf only passes the perf limb on a real (>=100) sample.
    assert passes_perf(m(0.30, float("inf"), trade_count=120))
    assert not passes_perf(m(0.30, float("inf"), trade_count=10))
