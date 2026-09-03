from rl_meta.verification import (verification_status, metric_flags, pine_flags,
                                 VERIFIED, NEEDS_REVIEW, SUSPECTED_LOOKAHEAD)
from rl_meta.ranking import rl_risk_gate
from rl_meta.types import Metrics, StrategyMeta


def m(**kw):
    base = dict(trade_count=600, win_rate=0.62, profit_factor=1.8, max_drawdown=0.15,
                cagr=0.2)
    base.update(kw)
    return Metrics(**base)


# -- metric anomaly triggers ----------------------------------------------
def test_plausible_is_verified():
    assert verification_status(m())[0] == VERIFIED


def test_high_pf_low_dd_flags():
    assert verification_status(m(profit_factor=19.0, max_drawdown=0.007))[0] == NEEDS_REVIEW


def test_inf_pf_low_dd_flags():
    assert verification_status(m(profit_factor=float("inf"), max_drawdown=0.005))[0] == NEEDS_REVIEW


def test_high_winrate_many_trades_flags():
    assert verification_status(m(win_rate=0.82, trade_count=8000, profit_factor=2.0,
                                 max_drawdown=0.10))[0] == NEEDS_REVIEW


def test_extreme_cagr_tiny_dd_flags():
    assert verification_status(m(cagr=3.0, max_drawdown=0.01, profit_factor=2.0))[0] == NEEDS_REVIEW


def test_high_pf_but_normal_dd_is_ok():
    # PF high but drawdown realistic -> plausible, not flagged.
    assert verification_status(m(profit_factor=8.0, max_drawdown=0.20))[0] == VERIFIED


# -- pine-source repaint detection ----------------------------------------
def test_old_pine_security_flags_lookahead():
    meta = StrategyMeta("s", "Sniper", "h", "2026-01-01", "s.txt",
                        pine_version=2, uses_security=True, lookahead_off=False)
    assert pine_flags(meta)
    assert verification_status(m(), meta)[0] == SUSPECTED_LOOKAHEAD


def test_lookahead_off_is_safe():
    meta = StrategyMeta("s", "S", "h", "2026-01-01", "s.txt",
                        pine_version=2, uses_security=True, lookahead_off=True)
    assert not pine_flags(meta)


def test_modern_pine_security_not_flagged():
    meta = StrategyMeta("s", "S", "h", "2026-01-01", "s.txt",
                        pine_version=5, uses_security=True, lookahead_off=False)
    assert not pine_flags(meta)


def test_source_flag_dominates_metrics():
    meta = StrategyMeta("s", "S", "h", "2026-01-01", "s.txt",
                        pine_version=2, uses_security=True, lookahead_off=False)
    # both metric anomaly AND source flag -> the more specific source verdict wins
    status, reasons = verification_status(m(profit_factor=19.0, max_drawdown=0.007), meta)
    assert status == SUSPECTED_LOOKAHEAD and len(reasons) >= 2


# -- orthogonality: flagging does NOT reject ------------------------------
def test_verification_does_not_change_rl_eligibility():
    # A too-good strategy still PASSES the hard risk gates (verification only flags).
    too_good = m(trade_count=8000, win_rate=0.82, profit_factor=19.0, max_drawdown=0.007)
    oos = Metrics(trade_count=7000, profit_factor=19.0)
    ok, _ = rl_risk_gate(too_good, oos, "sniper_strategy", "unknown")
    assert ok                                            # risk gates pass it...
    assert verification_status(too_good)[0] == NEEDS_REVIEW   # ...but it's flagged
