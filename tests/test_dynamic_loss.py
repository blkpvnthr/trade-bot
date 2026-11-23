"""Stage B: dynamic volatility-adjusted protective loss monitor (decision core).

Pure + deterministic (clock injected). Pins:
  * ATR-adjusted threshold = clamp(atr_pct * mult, min_pct, max_pct)
  * -0.5% is a WARNING, not an unconditional liquidation
  * discretionary exit requires adaptive breach + confirmed deterioration over
    >= confirmation_seconds AND >= min_observations distinct fresh observations
  * recovery within the window cancels the pending exit
  * hard-risk breach => immediate exit (no confirmation delay)
  * long and short positions handled correctly
  * never widen a losing position's authorized risk (threshold only tightens vs max)
  * missing/unreliable data => conservative fallback to the fixed warning threshold
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.dynamic_loss import DynamicLossMonitor, adaptive_threshold


# ------------------------------------------------------------- threshold math
def test_adaptive_threshold_scales_with_atr():
    # atr_pct 0.008 * 1.5 = 0.012, within [0.005, 0.020]
    assert abs(adaptive_threshold(0.008, multiplier=1.5, min_pct=0.005, max_pct=0.020) - 0.012) < 1e-9


def test_adaptive_threshold_clamped_to_floor_and_ceiling():
    assert adaptive_threshold(0.0001, multiplier=1.5, min_pct=0.005, max_pct=0.020) == 0.005
    assert adaptive_threshold(0.5, multiplier=1.5, min_pct=0.005, max_pct=0.020) == 0.020


def test_adaptive_threshold_missing_atr_falls_back_to_floor():
    assert adaptive_threshold(None, multiplier=1.5, min_pct=0.005, max_pct=0.020) == 0.005
    assert adaptive_threshold(float("nan"), multiplier=1.5, min_pct=0.005, max_pct=0.020) == 0.005


def _mon(**kw):
    kw.setdefault("warning_pct", 0.005)
    kw.setdefault("adaptive_min_pct", 0.005)
    kw.setdefault("adaptive_max_pct", 0.020)
    kw.setdefault("atr_multiplier", 1.5)
    kw.setdefault("confirmation_seconds", 30.0)
    kw.setdefault("min_observations", 3)
    return DynamicLossMonitor(**kw)


# --------------------------------------------------------- warning vs exit
def test_small_loss_above_warning_is_hold():
    m = _mon()
    d = m.observe("A", plpc=-0.002, atr_pct=0.008, deteriorating=True, now=0.0)
    assert d.action == "HOLD"
    assert not d.should_exit


def test_warning_breach_enters_watch_not_exit():
    m = _mon()
    # -0.6% breaches the -0.5% warning but is within the adaptive threshold (1.2%).
    d = m.observe("A", plpc=-0.006, atr_pct=0.008, deteriorating=False, now=0.0)
    assert d.action == "WATCH"
    assert not d.should_exit


def test_adaptive_breach_requires_confirmation_window_and_observations():
    m = _mon()
    # Loss beyond the 1.2% adaptive threshold, deteriorating, but not yet confirmed.
    d0 = m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=0.0)
    assert d0.action == "EXIT_PENDING" and not d0.should_exit
    d1 = m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=10.0)
    assert not d1.should_exit                     # 10s < 30s, only 2 obs
    d2 = m.observe("A", plpc=-0.014, atr_pct=0.008, deteriorating=True, now=20.0)
    assert not d2.should_exit                     # 20s < 30s, 3 obs but time not met
    d3 = m.observe("A", plpc=-0.015, atr_pct=0.008, deteriorating=True, now=31.0)
    assert d3.should_exit and d3.action == "EXIT" # 31s >= 30s AND >= 3 obs


def test_recovery_cancels_pending_exit():
    m = _mon()
    m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=0.0)
    m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=10.0)
    # Recovers back above the adaptive threshold -> pending exit cancelled.
    rec = m.observe("A", plpc=-0.004, atr_pct=0.008, deteriorating=False, now=20.0)
    assert rec.action in ("HOLD", "RECOVERING") and not rec.should_exit
    # A later fresh breach must start a NEW confirmation window, not resume the old.
    again = m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=25.0)
    assert again.action == "EXIT_PENDING" and not again.should_exit


def test_no_exit_without_deterioration_even_past_window():
    m = _mon()
    for t in (0.0, 10.0, 20.0, 31.0, 61.0):
        d = m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=False, now=t)
    # Beyond the adaptive threshold but NOT deteriorating -> hold within risk budget.
    assert not d.should_exit


def test_hard_stop_exits_immediately_no_confirmation():
    m = _mon(hard_stop_pct=0.025)
    d = m.observe("A", plpc=-0.03, atr_pct=0.008, deteriorating=False, now=0.0)
    assert d.should_exit and d.action == "HARD_EXIT"   # bypasses the 30s/3obs gate


def test_short_position_handled_via_sign_correct_plpc():
    m = _mon()
    # Alpaca unrealized_plpc is sign-correct for a short too (negative == losing),
    # so the monitor treats a losing short exactly like a losing long.
    m.observe("S", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=0.0, side="short")
    m.observe("S", plpc=-0.014, atr_pct=0.008, deteriorating=True, now=15.0, side="short")
    d = m.observe("S", plpc=-0.015, atr_pct=0.008, deteriorating=True, now=31.0, side="short")
    assert d.should_exit


def test_missing_atr_uses_conservative_warning_threshold():
    # With no ATR, the adaptive threshold falls back to the floor (0.5%), so a
    # confirmed deteriorating loss past 0.5% can exit (conservative, never wider).
    m = _mon()
    m.observe("A", plpc=-0.006, atr_pct=None, deteriorating=True, now=0.0)
    m.observe("A", plpc=-0.006, atr_pct=None, deteriorating=True, now=15.0)
    d = m.observe("A", plpc=-0.007, atr_pct=None, deteriorating=True, now=31.0)
    assert d.should_exit
    assert d.threshold == 0.005


def test_threshold_never_exceeds_authorized_max():
    m = _mon(adaptive_max_pct=0.020)
    d = m.observe("A", plpc=-0.001, atr_pct=0.9, deteriorating=False, now=0.0)
    assert d.threshold <= 0.020          # huge ATR cannot widen beyond the ceiling


def test_reset_clears_symbol_state():
    m = _mon()
    m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=0.0)
    m.reset("A")
    d = m.observe("A", plpc=-0.013, atr_pct=0.008, deteriorating=True, now=31.0)
    assert d.action == "EXIT_PENDING" and not d.should_exit   # window restarted


# ----------------------------------------------------------- Stage B config (B1)
from market_scan.config import MarketScanConfig  # noqa: E402


def test_dynamic_loss_config_defaults_preserve_existing_behavior():
    cfg = MarketScanConfig()
    assert cfg.dynamic_loss_enabled is False       # OFF by default -> fixed -0.5% preserved
    assert cfg.loss_warning_pct == 0.005
    assert cfg.adaptive_loss_min_pct == 0.005
    assert cfg.adaptive_loss_max_pct == 0.020
    assert cfg.loss_atr_period == 14
    assert cfg.loss_atr_multiplier == 1.5
    assert cfg.loss_atr_timeframe == "5Min"
    assert cfg.exit_confirmation_seconds == 30.0
    assert cfg.exit_confirmation_min_observations == 3
    assert cfg.position_reentry_cooldown_seconds == 300.0
    assert cfg.replacement_min_score_improvement == 0.15


def test_dynamic_loss_config_from_env(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_DYNAMIC_LOSS", "1")
    monkeypatch.setenv("MARKET_SCAN_LOSS_ATR_MULT", "2.0")
    monkeypatch.setenv("MARKET_SCAN_ADAPTIVE_LOSS_MAX", "0.03")
    monkeypatch.setenv("MARKET_SCAN_EXIT_CONFIRM_SECONDS", "45")
    cfg = MarketScanConfig.from_env()
    assert cfg.dynamic_loss_enabled is True
    assert cfg.loss_atr_multiplier == 2.0
    assert cfg.adaptive_loss_max_pct == 0.03
    assert cfg.exit_confirmation_seconds == 45.0


def test_adaptive_loss_bounds_are_coerced_sane():
    # min must be <= max and both strictly positive magnitudes.
    cfg = MarketScanConfig(adaptive_loss_min_pct=0.03, adaptive_loss_max_pct=0.01)
    assert cfg.adaptive_loss_min_pct <= cfg.adaptive_loss_max_pct
