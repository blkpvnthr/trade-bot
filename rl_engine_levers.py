#!/usr/bin/env python3
"""
rl_engine_levers.py — Gated wiring for the two validated strategy levers.
=========================================================================

Wires the two empirically-validated improvements into the live engines, but ONLY behind the
RL deployment gate. Both levers are SUPPRESSIVE / RISK-REDUCING and default OFF:

  1. REGIME GATE  — suppress NEW entries (alpha + Buffett) when SPY is not confirmed above its
     200-day MA (rl_regime_study.py: 2022-bear drawdown 37.9% → 9.7%). It can only PREVENT a
     new buy; it never forces a trade and never touches sells/exits.
  2. DYNAMIC EXIT — an asymmetric exit profile (take a partial at +1R, then trail the runner
     1R below its peak) for the alpha engine (rl_dynamic_exit.py: OOS profit-factor 1.5 → 2.3).
     It only adjusts the *partial/trail trigger levels*; the hard stop, 3R cap, kill switch,
     daily-loss halt, and RiskManager are untouched and always take priority.

A lever is ACTIVE only when ALL hold (fail-safe to inactive on any error):
  (a) its explicit ENABLE_* env flag is truthy           — default OFF
  (b) the process is in PAPER mode                        — never live
  (c) the RL deployment gate rates the relevant agent at  — "behind the deployment gate"
      least `paper_mode_only` (read from the latest
      rl_deployment_recommendation.json; missing/unknown → blocked)

This module decides ONLY whether a suppressive filter / exit-profile applies. It never submits
orders, never imports AlpacaGateway, and cannot weaken any risk control.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Optional

import rl_common as rc

# Recommendations that permit a lever to act (the deployment gate's "good enough for paper").
GATE_ALLOWED = {"paper_mode_only", "eligible_after_review"}

# Validated dynamic-exit profile (rl_dynamic_exit.py): partial at +1R, trail the runner 1R
# below the peak, trailing armed from +1R. R is the full premium paid (= entry_cost).
EXIT_PROFILE = {"partial_r": 1.0, "trail_activate_r": 1.0, "trail_offset_r": 1.0}


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def paper_mode() -> bool:
    """True only when the process is unambiguously in paper mode (conservative default)."""
    mode = os.environ.get("MODE", "paper").strip().lower()
    paper_only = os.environ.get("ALPACA_PAPER_ONLY", "true").strip().lower() != "false"
    return mode != "live" and paper_only


def deployment_status(agent: str) -> str:
    """Latest deployment recommendation for `agent` from rl_deployment_recommendation.json.
    Returns 'unknown' if no report exists or it can't be parsed (caller treats as blocked)."""
    try:
        files = sorted(glob.glob(str(rc.DAILY_OUTPUTS / "*" / "rl_deployment_recommendation.json")))
        if not files:
            return "unknown"
        doc = json.loads(open(files[-1], encoding="utf-8").read())
        agents = doc.get("agents", {}) if isinstance(doc, dict) else {}
        rec = (agents.get(agent) or {}).get("deployment_recommendation")
        return str(rec) if rec else "unknown"
    except Exception:
        return "unknown"


def _gate_ok(agent: str) -> bool:
    """Whether the deployment gate permits this agent's lever. Required by default; set
    RL_LEVER_REQUIRE_GATE=0 to bypass the gate (still needs flag + paper mode)."""
    if not _truthy("RL_LEVER_REQUIRE_GATE", "true"):
        return True
    return deployment_status(agent) in GATE_ALLOWED


def lever_active(flag_name: str, agent: str) -> bool:
    """Master gate: explicit flag ON + paper mode + deployment gate permits `agent`."""
    try:
        return _truthy(flag_name) and paper_mode() and _gate_ok(agent)
    except Exception:
        return False


# ── regime gate (suppressive entry filter) ───────────────────────────────────────────
def spy_confirmed_bull(market_state: Optional[dict], indicator_snapshot: Optional[dict]) -> bool:
    """True only when SPY is CONFIRMED above its 200-day MA. Prefer the explicit 200-DMA from
    the streaming snapshot; fall back to the market_state spy_trend label; otherwise (unknown)
    return False — matching the validated study's 'cash unless confirmed bull' rule and the
    project rule to never treat missing data as bullish."""
    snap = indicator_snapshot or {}
    spy = snap.get("SPY") or {}
    ind = spy.get("indicators") or {}
    sma200 = ind.get("sma_200")
    price = spy.get("price") or ind.get("close")
    if isinstance(sma200, (int, float)) and sma200 > 0 and isinstance(price, (int, float)) and price > 0:
        return float(price) >= float(sma200)
    trend = (market_state or {}).get("spy_trend")
    if trend:
        return str(trend).lower() == "up"
    return False


def regime_gate_active(agent: str) -> bool:
    return lever_active("ENABLE_RL_REGIME_GATE", agent)


def suppress_entry(agent: str, market_state: Optional[dict],
                   indicator_snapshot: Optional[dict]) -> bool:
    """True → suppress a NEW entry for `agent`: regime-gate lever active AND SPY not confirmed
    bullish. Suppressive only; callers must never let this affect exits/sells."""
    return regime_gate_active(agent) and not spy_confirmed_bull(market_state, indicator_snapshot)


# ── dynamic exit ──────────────────────────────────────────────────────────────────────
def dynamic_exit_active() -> bool:
    return lever_active("ENABLE_RL_DYNAMIC_EXIT", "alpha_exit")


def exit_profile() -> Dict[str, float]:
    """The validated partial/trail trigger profile (only used when dynamic_exit_active())."""
    return dict(EXIT_PROFILE)


# ── observability ─────────────────────────────────────────────────────────────────────
def status() -> Dict[str, Any]:
    """Snapshot of lever gating state (for logging / the daily summary). No side effects."""
    return {
        "paper_mode": paper_mode(),
        "gate_required": _truthy("RL_LEVER_REQUIRE_GATE", "true"),
        "regime_gate_flag": _truthy("ENABLE_RL_REGIME_GATE"),
        "dynamic_exit_flag": _truthy("ENABLE_RL_DYNAMIC_EXIT"),
        "deployment_status": {a: deployment_status(a) for a in
                              ("alpha_entry", "alpha_exit", "buffett_dca")},
        "regime_gate_active": {a: regime_gate_active(a) for a in ("alpha_entry", "buffett_dca")},
        "dynamic_exit_active": dynamic_exit_active(),
    }


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
