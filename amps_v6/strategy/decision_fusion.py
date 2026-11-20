"""Deterministic opportunity fusion (V6 spec section 17).

NOT agent voting. A candidate entry requires EVERY mandatory gate to pass (a conjunction). Any
single agent may VETO. Inverse-ETF and swing entries carry additional mandatory gates. Mandatory
gates are never weakened to raise trade frequency, and ordinary protective exits never depend on
entry-alpha approval (they are handled elsewhere, by the exit engine).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionInputs:
    valid_market_data: bool
    valid_setup: bool
    eligible_instrument: bool
    eligible_trading_horizon: bool
    alpha_confirmed: bool
    positive_expected_net_return: bool
    acceptable_execution_cost: bool
    adversarial_veto: bool = False
    security_veto: bool = False
    deterministic_risk_approval: bool = False
    # inverse-ETF specific
    is_inverse: bool = False
    bearish_underlying_confirmation: bool = True
    inverse_etf_strength: bool = True
    inverse_exposure_within_limits: bool = True
    no_unauthorized_conflicting_exposure: bool = True
    # swing specific
    is_swing: bool = False
    horizon_specific_alpha_confirmed: bool = True
    higher_timeframe_confirmation: bool = True
    overnight_policy_approval: bool = True
    overnight_risk_within_limits: bool = True
    instrument_swing_eligible: bool = True


@dataclass
class FusionResult:
    entry_allowed: bool
    failed_gates: list = field(default_factory=list)


def fuse(f: FusionInputs) -> FusionResult:
    failed = []
    base = {
        "valid_market_data": f.valid_market_data,
        "valid_setup": f.valid_setup,
        "eligible_instrument": f.eligible_instrument,
        "eligible_trading_horizon": f.eligible_trading_horizon,
        "alpha_confirmed": f.alpha_confirmed,
        "positive_expected_net_return": f.positive_expected_net_return,
        "acceptable_execution_cost": f.acceptable_execution_cost,
        "no_adversarial_veto": not f.adversarial_veto,
        "no_security_veto": not f.security_veto,
        "deterministic_risk_approval": f.deterministic_risk_approval,
    }
    for name, ok in base.items():
        if not ok:
            failed.append(name)
    if f.is_inverse:
        inv = {
            "bearish_underlying_confirmation": f.bearish_underlying_confirmation,
            "inverse_etf_strength": f.inverse_etf_strength,
            "inverse_exposure_within_limits": f.inverse_exposure_within_limits,
            "no_unauthorized_conflicting_exposure": f.no_unauthorized_conflicting_exposure,
        }
        failed += [n for n, ok in inv.items() if not ok]
    if f.is_swing:
        sw = {
            "horizon_specific_alpha_confirmed": f.horizon_specific_alpha_confirmed,
            "higher_timeframe_confirmation": f.higher_timeframe_confirmation,
            "overnight_policy_approval": f.overnight_policy_approval,
            "overnight_risk_within_limits": f.overnight_risk_within_limits,
            "instrument_swing_eligible": f.instrument_swing_eligible,
        }
        failed += [n for n, ok in sw.items() if not ok]
    return FusionResult(entry_allowed=(len(failed) == 0), failed_gates=failed)
