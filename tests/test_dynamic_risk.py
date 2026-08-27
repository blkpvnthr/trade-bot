"""
Dynamic portfolio-risk controller tests.
=========================================

OFF is a byte-identical no-op; ON changes the decision and stays bounded; the
controller is causal (no lookahead), deterministic, and reproduces exactly
between the backtest and the live decision path. Plus the refinement-specific
behaviors: the AND-gate does NOT collapse a broad bull on vol alone, volatility
acceleration distinguishes rising from falling vol, drawdown is EXCLUDED from the
default stress score, the deadband is asymmetric, and recovery re-risks without
requiring falling vol.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from strat_fixtures import make_synthetic_prices, RISK_ASSETS
from strat_allocator import AllocatorConfig
from strat import build_gate_schedule, momentum_view_panel, rolling_ic_panel, run_backtest
from strat_decision import compute_target_portfolio, COLD_START_SESSIONS
import dynamic_risk as dr

PRICES = make_synthetic_prices()
CFG = AllocatorConfig()
CFG_ON = replace(CFG, dynamic_risk_enabled=True)
GATE = build_gate_schedule(PRICES["SPY"], CFG)
VIEW = momentum_view_panel(PRICES, CFG)
IC = rolling_ic_panel(VIEW, PRICES)
START = "2007-01-02"
RISK_PX = PRICES[list(RISK_ASSETS)]
D = PRICES.index[-1]
PREV = np.zeros(len(RISK_ASSETS))


def _bt(cfg, prices=PRICES):
    return run_backtest(prices, cfg, GATE, VIEW, IC, START, "T")[0]


def _target(O, S, R, cfg):
    """Pure AND-gate + map: opportunity/stress/recovery -> target multiplier."""
    net = np.clip(O * (1 - S) - (1 - O) * S + cfg.dynamic_risk_stress_gamma_recovery * R,
                  -1.0, 1.0)
    return float(dr._map_net_to_mult(np.array([net]), cfg)[0])


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_config_defaults_valid_and_off():
    assert CFG.dynamic_risk_enabled is False
    # defaults satisfy the multiplier-bound and asymmetry invariants
    AllocatorConfig()  # __post_init__ does not raise


def test_config_bounds_rejected():
    with pytest.raises(ValueError):
        AllocatorConfig(dynamic_risk_max_mult=0.9)          # max < 1.0
    with pytest.raises(ValueError):
        AllocatorConfig(dynamic_risk_severe_floor=1.2)      # floor > 1.0
    with pytest.raises(ValueError):
        AllocatorConfig(dynamic_risk_deadband_down=0.4,
                        dynamic_risk_deadband_up=0.3)        # down > up (inverts asymmetry)


def test_cfg_signature_includes_dynamic_keys():
    from strat_decision import _cfg_signature
    base = _cfg_signature(CFG)
    changed = _cfg_signature(replace(CFG, dynamic_risk_enabled=True))
    assert base != changed
    changed2 = _cfg_signature(replace(CFG, dynamic_risk_max_mult=1.20))
    assert base != changed2


# ---------------------------------------------------------------------------
# the map: monotonic, bounded, asymmetric
# ---------------------------------------------------------------------------
def test_map_neutral_band_is_exactly_one():
    # inside BOTH deadbands -> exactly 1.0
    assert dr._map_net_to_mult(np.array([0.0]), CFG)[0] == 1.0
    assert dr._map_net_to_mult(np.array([CFG.dynamic_risk_deadband_down]), CFG)[0] == 1.0
    assert dr._map_net_to_mult(np.array([-CFG.dynamic_risk_deadband_down]), CFG)[0] == 1.0


def test_map_bounded_and_monotone():
    grid = np.linspace(-1, 1, 41)
    m = dr._map_net_to_mult(grid, CFG)
    assert np.all(m >= CFG.dynamic_risk_min_mult - 1e-9)
    assert np.all(m <= CFG.dynamic_risk_max_mult + 1e-9)
    assert np.all(np.diff(m) >= -1e-12)                     # non-decreasing in net


def test_map_asymmetric_harder_to_add_than_cut():
    # a net magnitude between the two deadbands cuts risk but does NOT add it.
    mid = 0.5 * (CFG.dynamic_risk_deadband_down + CFG.dynamic_risk_deadband_up)
    assert dr._map_net_to_mult(np.array([mid]), CFG)[0] == 1.0     # not enough to ADD
    assert dr._map_net_to_mult(np.array([-mid]), CFG)[0] < 1.0     # enough to CUT


# ---------------------------------------------------------------------------
# the AND-gate corners (refinement: no collapse of a broad bull on vol alone)
# ---------------------------------------------------------------------------
def test_and_gate_corners():
    assert _target(0.9, 0.1, 0.0, CFG) > 1.0        # strong + calm -> add risk
    assert _target(0.1, 0.9, 0.0, CFG) < 1.0        # weak + stressed -> cut hard
    # elevated stress AND strong opportunity cancel -> neutral (do NOT cut a bull on vol)
    assert _target(0.9, 0.9, 0.0, CFG) == 1.0
    # weak opportunity but calm also nets to neutral
    assert _target(0.1, 0.1, 0.0, CFG) == 1.0


def test_recovery_lifts_net():
    base = _target(0.5, 0.5, 0.0, CFG)
    withR = _target(0.5, 0.5, 1.0, CFG)
    assert withR >= base and withR > 1.0            # recovery pushes toward adding risk


# ---------------------------------------------------------------------------
# the stateful recurrence
# ---------------------------------------------------------------------------
def test_recurrence_cold_start_neutral():
    n = 200
    m = dr._reconstruct_risk_multiplier(np.ones(n), np.zeros(n, bool), np.zeros(n), CFG)
    assert np.allclose(m, 1.0)


def test_recurrence_severe_snaps_to_floor():
    n = 50
    sev = np.zeros(n, bool); sev[10] = True
    m = dr._reconstruct_risk_multiplier(np.ones(n), sev, np.zeros(n), CFG)
    assert m[10] == pytest.approx(CFG.dynamic_risk_severe_floor)


def test_recurrence_derisk_eager_fast():
    n = 40
    low = np.full(n, CFG.dynamic_risk_min_mult)
    m = dr._reconstruct_risk_multiplier(low, np.zeros(n, bool), np.zeros(n), CFG)
    assert m[1] < m[0]                                        # moves on day 1 (no confirmation)
    # each step is capped by max_down_change
    assert np.all(np.diff(m) >= -CFG.dynamic_risk_max_down_change - 1e-12)
    assert m[-1] == pytest.approx(CFG.dynamic_risk_min_mult)


def test_recurrence_rerisk_needs_confirmation_and_is_slow():
    n = 60
    k = CFG.dynamic_risk_confirm_days
    high = np.full(n, CFG.dynamic_risk_max_mult)
    m = dr._reconstruct_risk_multiplier(high, np.zeros(n, bool), np.zeros(n), CFG)
    assert m[k - 2] == 1.0                                    # not confirmed yet
    assert m[k - 1] > 1.0                                     # first step on confirmation
    up_steps = np.diff(m)
    assert np.all(up_steps <= CFG.dynamic_risk_max_up_change + 1e-12)


def test_recurrence_recovery_accelerates_rerisk():
    n = 60
    k = CFG.dynamic_risk_confirm_days
    high = np.full(n, CFG.dynamic_risk_max_mult)
    m0 = dr._reconstruct_risk_multiplier(high, np.zeros(n, bool), np.zeros(n), CFG)
    mR = dr._reconstruct_risk_multiplier(high, np.zeros(n, bool), np.ones(n), CFG)
    assert mR[k - 1] > m0[k - 1]                              # bigger first step with recovery


def test_recurrence_hysteresis_no_flip_on_small_oscillation():
    # a target that jitters within the deadbands around 1.0 must not move the mult.
    n = 100
    tgt = 1.0 + 0.001 * np.sin(np.arange(n))
    m = dr._reconstruct_risk_multiplier(tgt, np.zeros(n, bool), np.zeros(n), CFG)
    assert np.allclose(m, 1.0)


# ---------------------------------------------------------------------------
# trailing rank: strictly causal, neutral cold start
# ---------------------------------------------------------------------------
def test_trailing_rank_causal_and_neutral_coldstart():
    x = np.arange(100.0)
    r = dr._trailing_rank(x, lookback=504, min_periods=60)
    assert np.all(r[:59] == 0.5)                             # neutral until min_periods
    assert r[99] == pytest.approx(1.0)                       # last is the max in its window
    # mutating the FUTURE cannot change a past rank
    x2 = x.copy(); x2[80:] = -999.0
    r2 = dr._trailing_rank(x2, lookback=504, min_periods=60)
    assert np.array_equal(r[:80], r2[:80])


# ---------------------------------------------------------------------------
# frame: bounded, severe-floored, deterministic, no-lookahead
# ---------------------------------------------------------------------------
def test_frame_bounded_and_severe_floored():
    fr = dr.build_risk_frame(PRICES, CFG_ON)
    m = fr["risk_multiplier"].to_numpy(float)
    assert np.all(m >= CFG_ON.dynamic_risk_min_mult - 1e-9)
    assert np.all(m <= CFG_ON.dynamic_risk_max_mult + 1e-9)
    sev = fr["severe"].to_numpy(bool)
    assert np.allclose(m[sev], CFG_ON.dynamic_risk_severe_floor)


def test_frame_deterministic():
    a = dr.build_risk_frame(PRICES, CFG_ON)["risk_multiplier"].to_numpy(float)
    b = dr.build_risk_frame(PRICES, CFG_ON)["risk_multiplier"].to_numpy(float)
    assert np.array_equal(a, b)


def test_frame_no_lookahead():
    full = dr.build_risk_frame(PRICES, CFG_ON)
    d = full.index[-40]
    sliced = dr.build_risk_frame(PRICES.loc[:d], CFG_ON)
    # value at d computed on the full panel == value computed on the sliced panel
    assert sliced["risk_multiplier"].iloc[-1] == pytest.approx(
        float(full.loc[d, "risk_multiplier"]))
    # mutating strictly-after-d bars does not change the multiplier at d
    mutated = PRICES.copy()
    mutated.loc[mutated.index > d] *= 0.5
    mut = dr.build_risk_frame(mutated, CFG_ON)
    assert float(mut.loc[d, "risk_multiplier"]) == pytest.approx(
        float(full.loc[d, "risk_multiplier"]))


def test_frame_recovery_fires_during_recovering_regime():
    fr = dr.build_risk_frame(PRICES, CFG_ON)
    rec = fr[fr["regime"].between(0.44, 0.71)]               # BOTTOMING/RECOVERING states
    assert (rec["recovery"] > 0).any()                       # recovery term activates


# ---------------------------------------------------------------------------
# refinement 2: volatility acceleration distinguishes rising from falling vol
# ---------------------------------------------------------------------------
def test_vol_accel_distinguishes_rising_from_falling_vol():
    # vol_accel keys off the 20-day CHANGE in portfolio vol, so it separates
    # rising-vol days from falling-vol days far more than vol_level (which ranks
    # the vol LEVEL and is largely blind to the slope). That separation is the
    # whole point of refinement 2: high-but-falling vol should read calmer than
    # rapidly-rising vol.
    accel_cfg = replace(CFG_ON, dynamic_risk_stress_weights={"vol_accel": 1.0})
    level_cfg = replace(CFG_ON, dynamic_risk_stress_weights={"vol_level": 1.0})
    fa = dr.build_risk_frame(PRICES, accel_cfg)
    fl = dr.build_risk_frame(PRICES, level_cfg)
    d20 = fa["port_vol"] - fa["port_vol"].shift(20)
    rising, falling = d20 > 0, d20 < 0
    sep_accel = (fa.loc[rising, "stress_score"].mean()
                 - fa.loc[falling, "stress_score"].mean())
    sep_level = (fl.loc[rising, "stress_score"].mean()
                 - fl.loc[falling, "stress_score"].mean())
    assert sep_accel > 0                     # rising vol reads as more stressed
    assert sep_accel > sep_level             # accel separates slope better than level does


# ---------------------------------------------------------------------------
# refinement 1: drawdown is EXCLUDED from the default stress score
# ---------------------------------------------------------------------------
def test_drawdown_excluded_from_default_stress():
    fr = dr.build_risk_frame(PRICES, CFG_ON)
    # a config that uses ONLY drawdown stress reproduces clip(-Drawdown252/0.20).
    dd_cfg = replace(CFG_ON, dynamic_risk_stress_weights={},
                     dynamic_risk_dd_stress_weight=1.0)
    fdd = dr.build_risk_frame(PRICES, dd_cfg)
    dd252 = GATE.reindex(fr.index)["Drawdown252"].to_numpy(float)
    expected = np.clip(-np.where(np.isfinite(dd252), dd252, 0.0) / 0.20, 0.0, 1.0)
    assert np.allclose(fdd["stress_score"].to_numpy(float), expected)
    # the DEFAULT stress score is a different axis (driven by vol, not drawdown):
    assert not np.allclose(fr["stress_score"].to_numpy(float), expected)


def test_dd_stress_weight_is_reachable():
    a = dr.build_risk_frame(PRICES, CFG_ON)["stress_score"].to_numpy(float)
    b = dr.build_risk_frame(PRICES, replace(CFG_ON, dynamic_risk_dd_stress_weight=0.5)
                            )["stress_score"].to_numpy(float)
    assert not np.allclose(a, b)                             # dd term participates when > 0


# ---------------------------------------------------------------------------
# wiring: OFF byte-identical, ON active, backtest/live parity
# ---------------------------------------------------------------------------
def test_backtest_off_is_byte_identical():
    base = _bt(CFG)
    off = _bt(replace(CFG, dynamic_risk_enabled=False))
    assert np.array_equal(base["Equity"].to_numpy(), off["Equity"].to_numpy())
    assert "RiskMultiplier" not in base.columns


def test_backtest_on_changes_result_and_holds_constraints():
    base = _bt(CFG)
    on = _bt(CFG_ON)
    assert not np.array_equal(base["Equity"].to_numpy(), on["Equity"].to_numpy())
    assert "RiskMultiplier" in on.columns
    # run_backtest raises internally on any gate/leverage/vol breach; reaching here
    # means every hard constraint held with the dynamic budget applied.
    assert (on["Gross"] <= on["LeverageCap"] + 1e-6).all()
    assert (on["Growth"] <= on["GrowthCeiling"] + 1e-6).all()


def test_live_off_is_byte_identical():
    base = compute_target_portfolio(RISK_PX, PREV, COLD_START_SESSIONS, CFG, as_of_close=D)
    off = compute_target_portfolio(RISK_PX, PREV, COLD_START_SESSIONS,
                                   replace(CFG, dynamic_risk_enabled=False), as_of_close=D)
    assert np.array_equal(base.target_vector(CFG), off.target_vector(CFG))
    assert base.decision_hash == off.decision_hash
    assert off.portfolio_risk_multiplier == 1.0


def test_live_on_surfaces_multiplier():
    on = compute_target_portfolio(RISK_PX, PREV, COLD_START_SESSIONS, CFG_ON, as_of_close=D)
    assert on.forecast_source.endswith("+dynamic_risk")
    assert on.risk_state is not None
    assert CFG_ON.dynamic_risk_min_mult <= on.portfolio_risk_multiplier <= CFG_ON.dynamic_risk_max_mult


def test_backtest_live_parity_multiplier():
    trace = []
    h = run_backtest(PRICES, CFG_ON, GATE, VIEW, IC, START, "P", trace=trace)[0]
    worst = 0.0
    for k in (len(trace) - 1, len(trace) - 30, len(trace) - 250, len(trace) // 2):
        t = trace[k]
        dec = compute_target_portfolio(
            PRICES[list(RISK_ASSETS)], prev_weights=t["prev_weights"],
            sessions_since_rebalance=t["sessions_since_rebalance"],
            cfg=CFG_ON, as_of_close=t["Date"])
        worst = max(worst, float(np.max(np.abs(dec.target_vector(CFG_ON) - t["target"]))))
    assert worst < 1e-8, worst


def test_composes_with_adaptive_config_G():
    # G (adaptive + dynamic) constructs and runs without violating the __post_init__
    # exclusivity guard (that guard only bars the two NAIVE overlays, not dynamic).
    g = replace(CFG, adaptive_rotation_enabled=True, dynamic_risk_enabled=True)
    h = _bt(g)
    assert "RiskMultiplier" in h.columns
    assert h.attrs.get("dynamic_risk", {}).get("enabled") is True
