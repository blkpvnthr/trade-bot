"""AMPS V6 -- multi-horizon, evidence-gated trading platform.

Design discipline (CLAUDE.md + V6 spec section 34): nothing here claims a trading edge.
The platform spine (registry, universe, features, risk gateway, execution harness, evidence
registry) is reusable infrastructure whose readiness (PLATFORM_LIVE_READY) is tracked SEPARATELY
from whether any individual strategy/horizon has validated entry alpha (STRATEGY_LIVE_ELIGIBLE).
Live activation (LIVE_ACTIVATION_APPROVED) always requires an explicit operator decision and is
false by default. Agents analyze/recommend/veto only; they never talk to the broker. Only the
deterministic execution gateway may submit/cancel/replace orders, and protective exits run
independently of any analytical agent or LLM.
"""
from __future__ import annotations

__version__ = "6.0.0-dev"
