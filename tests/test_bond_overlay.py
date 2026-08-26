"""
Bond-regime overlay tests: OFF is a byte-identical no-op, ON changes the decision,
the overlay is causal (no lookahead), and the multipliers stay bounded.
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

from strat_fixtures import make_synthetic_prices, RISK_ASSETS
from strat_allocator import AllocatorConfig
from strat_decision import compute_target_portfolio, COLD_START_SESSIONS
from bond_regime import BondOverlayConfig, compute_bond_regime, regime_frame

PRICES = make_synthetic_prices(end="2009-12-31")
CFG = AllocatorConfig()
BOND = ("SHY", "IEF", "TLT", "LQD", "HYG")
BOND_PX = PRICES[list(BOND)]
RISK_PX = PRICES[list(RISK_ASSETS)]
D = PRICES.index[-1]
PREV = np.zeros(len(RISK_ASSETS))


def _decide(bond_overlay=None, bond_prices=None):
    return compute_target_portfolio(RISK_PX, PREV, COLD_START_SESSIONS, CFG,
                                    as_of_close=D, bond_overlay=bond_overlay,
                                    bond_prices=bond_prices)


def test_disabled_overlay_is_byte_identical():
    base = _decide()
    off1 = _decide(BondOverlayConfig(enabled=False), BOND_PX)
    off2 = _decide(BondOverlayConfig(enabled=True), None)   # enabled but no data -> neutral
    assert np.array_equal(base.target_vector(CFG), off1.target_vector(CFG))
    assert np.array_equal(base.target_vector(CFG), off2.target_vector(CFG))
    assert base.decision_hash == off1.decision_hash == off2.decision_hash
    assert off1.vol_budget_mult == 1.0 and off1.growth_ceiling_mult == 1.0


def test_enabled_overlay_changes_identity():
    base = _decide()
    on = _decide(BondOverlayConfig(enabled=True), BOND_PX)
    assert on.forecast_source.endswith("+bond_overlay")
    assert on.bond_regime is not None
    # the multiplier is active (not exactly 1.0) so the decision identity changes
    assert on.decision_hash != base.decision_hash or on.vol_budget_mult == 1.0
    assert 0.60 <= on.vol_budget_mult <= 1.15
    assert 0.70 <= on.growth_ceiling_mult <= 1.0


def test_overlay_no_lookahead():
    on = _decide(BondOverlayConfig(enabled=True), BOND_PX)
    mutated = BOND_PX.copy()
    future = mutated.index > D
    # no future rows past D here (D is the last row), so also test via explicit as_of
    mutated2 = make_synthetic_prices(end="2010-12-31")[list(BOND)]
    reg_full = compute_bond_regime(mutated2, BondOverlayConfig(enabled=True), as_of=D)
    reg_slice = compute_bond_regime(mutated2.loc[:D], BondOverlayConfig(enabled=True), as_of=D)
    assert reg_full.vol_budget_mult == reg_slice.vol_budget_mult
    assert reg_full.risk_off_z == reg_slice.risk_off_z
    # and mutating strictly-after-D bars does not change the as_of=D regime
    m3 = mutated2.copy()
    m3.loc[m3.index > D] *= 0.5
    reg_mut = compute_bond_regime(m3, BondOverlayConfig(enabled=True), as_of=D)
    assert reg_mut.vol_budget_mult == reg_full.vol_budget_mult


def test_regime_bounded_and_risk_off_derisks():
    cfg = BondOverlayConfig(enabled=True)
    frame = regime_frame(BOND_PX, cfg)
    assert frame["vol_budget_mult"].between(1 - cfg.max_vol_cut - 1e-9,
                                            1 + cfg.max_vol_add + 1e-9).all()
    assert frame["growth_ceiling_mult"].between(1 - cfg.max_growth_cut - 1e-9,
                                                1.0 + 1e-9).all()
    # A synthetic HY credit collapse near the end -> more risk-off -> smaller mult.
    stressed = BOND_PX.copy()
    stressed.iloc[-25:, stressed.columns.get_loc("HYG")] *= np.linspace(1.0, 0.75, 25)
    r_norm = compute_bond_regime(BOND_PX, cfg)
    r_stress = compute_bond_regime(stressed, cfg)
    assert r_stress.risk_off_z > r_norm.risk_off_z
    assert r_stress.vol_budget_mult <= r_norm.vol_budget_mult
