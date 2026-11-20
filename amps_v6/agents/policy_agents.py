"""Specialist policy agents (V6-Integrated spec section 7; paper 2 Trend/Breakout/Defensive branches).

Advisory only: each returns a typed Recommendation. They never size positions, override risk, or talk
to the broker. Deterministic by default (the paper's PPO branches become interpretable rule branches
here; a PPO-backed variant can subclass later behind a flag -- but per V5/V6 evidence, model complexity
has not added validated alpha, so defaults are transparent rules).

OrderFlowAgent returns a VETO/abstain with reason DATA_UNAVAILABLE when genuine ticks are absent --
it never manufactures Level I/II data (spec section 5).
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from amps_v6.agents.base import SpecialistAgent, Recommendation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exp(hours: int = 2) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _rec(agent_id: str, ctx: dict, action: str, horizon: str, confidence: float,
         reason_codes: list, target_weight: float = 0.0, evidence_id: Optional[str] = None) -> Recommendation:
    return Recommendation(
        agent_id=agent_id, recommendation_id=f"{agent_id}:{ctx.get('symbol','?')}:{_now()}",
        symbol=ctx.get("symbol", "?"), underlying=ctx.get("underlying", ctx.get("symbol", "?")),
        timestamp=_now(), data_version=ctx.get("data_version", "unknown"),
        strategy_version="amps_v6_integrated", feature_snapshot_id=ctx.get("feature_snapshot_id", "na"),
        recommended_action=action, trading_horizon=horizon, confidence=float(confidence),
        evidence_id=evidence_id, reason_codes=reason_codes,
        invalidation_conditions=ctx.get("invalidation", []), expiration_timestamp=_exp())


class TrendPolicyAgent(SpecialistAgent):
    """Persistent directional movement: MA structure, ADX/DI, VWAP slope, relative strength."""
    agent_id = "trend_policy_agent"

    def analyze(self, ctx: dict) -> Optional[Recommendation]:
        f = ctx.get("features", {})
        di_up = f.get("pdi", 0) > f.get("mdi", 0)
        adx_ok = f.get("adx", 0) >= 20
        above_vwap = f.get("vwap_dist", 0) > 0
        rs_ok = f.get("rel_strength", 0) >= 0
        score = sum([di_up, adx_ok, above_vwap, rs_ok]) / 4.0
        if di_up and adx_ok and above_vwap:
            return _rec(self.agent_id, ctx, "ENTER_LONG", ctx.get("horizon", "INTRADAY_MOMENTUM"),
                        score, ["trend_up", f"adx={f.get('adx')}"], evidence_id=ctx.get("evidence_id"))
        return _rec(self.agent_id, ctx, "NO_TRADE", "NONE", 1.0 - score, ["no_trend"])


class BreakoutPolicyAgent(SpecialistAgent):
    """Opening-range / prior-day / premarket breaks with volume confirmation + false-breakout guard."""
    agent_id = "breakout_policy_agent"

    def analyze(self, ctx: dict) -> Optional[Recommendation]:
        f = ctx.get("features", {})
        above_or = f.get("close", 0) > f.get("opening_range_high", float("inf"))
        above_pdh = f.get("close", 0) > f.get("prior_day_high", float("inf"))
        vol_ok = f.get("rel_volume", 0) >= 1.5
        false_bo = f.get("close", 0) < f.get("vwap", 0)   # back below vwap = suspect
        if (above_or or above_pdh) and vol_ok and not false_bo:
            return _rec(self.agent_id, ctx, "ENTER_LONG", ctx.get("horizon", "INTRADAY_MOMENTUM"),
                        0.7, ["breakout", f"relvol={f.get('rel_volume')}"], evidence_id=ctx.get("evidence_id"))
        return _rec(self.agent_id, ctx, "NO_TRADE", "NONE", 0.5, ["no_breakout"])


class DefensivePolicyAgent(SpecialistAgent):
    """Deteriorating conditions / inverse-ETF opportunity / exposure reduction / defensive transition."""
    agent_id = "defensive_policy_agent"

    def analyze(self, ctx: dict) -> Optional[Recommendation]:
        f = ctx.get("features", {})
        regime = ctx.get("regime", "INDETERMINATE")
        is_inverse = ctx.get("orientation") == "bear"
        underlying_bearish = f.get("underlying_score", 0) <= -30
        if regime in ("CRISIS", "TRANSITION"):
            return _rec(self.agent_id, ctx, "REDUCE", "NONE", 0.8, [f"regime={regime}", "defensive_reduce"])
        if regime == "BEAR" and is_inverse and underlying_bearish:
            return _rec(self.agent_id, ctx, "ENTER_LONG", ctx.get("horizon", "INTRADAY_MOMENTUM"),
                        0.65, ["inverse_opportunity", "underlying_bearish"], evidence_id=ctx.get("evidence_id"))
        return _rec(self.agent_id, ctx, "HOLD", "NONE", 0.5, ["no_defensive_action"])


class OrderFlowAgent(SpecialistAgent):
    """Immediate participation / absorption / imbalance / divergence -- ONLY when genuine ticks exist."""
    agent_id = "order_flow_agent"

    def analyze(self, ctx: dict) -> Optional[Recommendation]:
        of = ctx.get("order_flow")
        if not of or of.get("status") == "DATA_UNAVAILABLE":
            # never manufacture microstructure; abstain with an explicit reason
            return _rec(self.agent_id, ctx, "VETO", "NONE", 0.0, ["DATA_UNAVAILABLE"])
        buy_pressure = of.get("buy_pressure", 0.0)
        divergence = of.get("cvd_divergence", False)
        if buy_pressure > 0.6 and not divergence:
            return _rec(self.agent_id, ctx, "ENTER_LONG", ctx.get("horizon", "MICRO_SCALP"),
                        min(1.0, buy_pressure), ["buy_pressure", "no_divergence"], evidence_id=ctx.get("evidence_id"))
        if divergence:
            return _rec(self.agent_id, ctx, "VETO", "NONE", 0.6, ["bearish_divergence"])
        return _rec(self.agent_id, ctx, "HOLD", "NONE", 0.4, ["flow_neutral"])


class PositionManagementAgent(SpecialistAgent):
    """Continuation / deterioration / target progress / stop proximity / overnight eligibility."""
    agent_id = "position_management_agent"

    def analyze(self, ctx: dict) -> Optional[Recommendation]:
        pos = ctx.get("position", {})
        ret = pos.get("return_pct", 0.0)
        near_stop = pos.get("near_stop", False)
        target_progress = pos.get("target_progress", 0.0)
        if near_stop:
            return _rec(self.agent_id, ctx, "EXIT", "NONE", 0.85, ["near_stop"])
        if target_progress >= 1.0:
            return _rec(self.agent_id, ctx, "EXIT", "NONE", 0.8, ["target_reached"])
        if ret > 0 and target_progress >= 0.5:
            return _rec(self.agent_id, ctx, "REDUCE", "NONE", 0.6, ["partial_profit"])
        return _rec(self.agent_id, ctx, "HOLD", "NONE", 0.5, ["continuation"])


ALL_POLICY_AGENTS = [TrendPolicyAgent, BreakoutPolicyAgent, DefensivePolicyAgent,
                     OrderFlowAgent, PositionManagementAgent]
