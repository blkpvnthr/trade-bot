#!/usr/bin/env python3
"""
rl_regime_gate_trial.py — Controlled, PAPER-ONLY trial of the RL regime gate.
=============================================================================

Markets are closed (Sunday 2026-05-31) and run.py has no bounded/dry-run mode, so a full
live loop would idle with no candidates to suppress. Instead this harness exercises the REAL
engine entry choke point (`AlphaCallEngine.evaluate_universe`) — the identical code path the
live loop uses — under three controlled scenarios, plus the exact gate decision the Buffett
engine calls (`rl_engine_levers.suppress_entry`). It SUBMITS NO ORDERS (engines only return
intents; nothing is routed to a gateway) and never calls any exit/sell logic.

Scenarios:
  A) regime gate ON  + SPY not confirmed >200DMA (bear)  → expect entries SUPPRESSED
  B) regime gate ON  + SPY confirmed >200DMA (bull)      → expect entries NOT suppressed
  C) regime gate OFF + SPY bear                          → expect entries NOT suppressed (baseline)

Hard rules honored: paper-only, dynamic-exit stays OFF, deployment gate NOT bypassed
(RL_LEVER_REQUIRE_GATE left at its default of required), no risk/exit/recommendation edits.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Defensive paper-only enforcement for this process (never live).
os.environ.setdefault("MODE", "paper")
os.environ.setdefault("ALPACA_PAPER_ONLY", "true")
os.environ["ENABLE_RL_DYNAMIC_EXIT"] = "0"          # stays OFF for the whole trial
os.environ.pop("RL_LEVER_REQUIRE_GATE", None)        # leave gate REQUIRED (do not bypass)

import rl_engine_levers as lv
import alpha_call_engine as ace

ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "daily_system_outputs" / "alpha_options_universe.json"
CANDIDATES = ["AAPL", "MSFT", "NVDA"]


class _DummyBroker:
    """No-network broker; chains are passed in explicitly so this is never called for I/O."""
    def get_options_chain(self, symbol):
        return []


def _read_universe_reasons():
    try:
        d = json.loads(UNIVERSE_PATH.read_text())
        rows = d.get("all_scanned") or []
        return ({r.get("symbol"): r.get("reject_reason") for r in rows},
                int(d.get("candidate_count", 0)), int(d.get("scanned", 0)),
                d.get("market_regime"))
    except Exception as exc:
        return ({"_error": str(exc)[:120]}, 0, 0, None)


def _run_alpha(engine, regime_flag: str, spy_trend: str) -> dict:
    os.environ["ENABLE_RL_REGIME_GATE"] = regime_flag
    mc = {"overall_market_bias": "bearish" if spy_trend == "down" else "bullish",
          "trade_permission": "normal", "regime": "neutral"}
    ms = {"spy_trend": spy_trend, "regime": "neutral", "confidence": 0.6}
    signals = engine.evaluate_universe(
        symbols=list(CANDIDATES), bars_by_symbol={}, market_context=mc,
        equity=100_000.0, options_exposure=0.0, waterfall_budget=5_000.0,
        chains_by_symbol={}, indicator_snapshot={}, premarket_snapshot={}, market_state=ms)
    reasons, cand, scanned, regime = _read_universe_reasons()
    suppressed = [s for s, r in reasons.items() if str(r).startswith("regime_gate")]
    return {
        "enable_rl_regime_gate": regime_flag, "spy_trend": spy_trend,
        "regime_gate_active(alpha_entry)": lv.regime_gate_active("alpha_entry"),
        "suppress_entry_decision": lv.suppress_entry("alpha_entry", ms, {}),
        "scanned": scanned, "candidate_count": cand,
        "signals_returned": len(signals),
        "regime_gate_suppressed_symbols": suppressed,
        "reject_reasons": reasons,
        "entries_suppressed_by_regime": len(suppressed) == len(CANDIDATES) and len(CANDIDATES) > 0,
    }


def _buffett_decision(regime_flag: str, spy_trend: str) -> dict:
    os.environ["ENABLE_RL_REGIME_GATE"] = regime_flag
    ms = {"spy_trend": spy_trend, "regime": "neutral", "confidence": 0.6}
    return {
        "enable_rl_regime_gate": regime_flag, "spy_trend": spy_trend,
        "regime_gate_active(buffett_dca)": lv.regime_gate_active("buffett_dca"),
        "suppress_entry_decision": lv.suppress_entry("buffett_dca", ms, {}),
    }


def main() -> dict:
    engine = ace.AlphaCallEngine(_DummyBroker())
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weekday": "Sunday", "market_open": False, "orders_submitted": 0,
        "exit_logic_invoked": False,
        "config": {
            "MODE": os.environ.get("MODE"), "ALPACA_PAPER_ONLY": os.environ.get("ALPACA_PAPER_ONLY"),
            "ENABLE_RL_DYNAMIC_EXIT": os.environ.get("ENABLE_RL_DYNAMIC_EXIT"),
            "RL_LEVER_REQUIRE_GATE(default)": os.environ.get("RL_LEVER_REQUIRE_GATE", "true(default)"),
            "paper_mode": lv.paper_mode(),
            "deployment_status": {a: lv.deployment_status(a)
                                  for a in ("alpha_entry", "buffett_dca", "alpha_exit")},
            "dynamic_exit_active": lv.dynamic_exit_active(),
        },
        "alpha": {
            "A_gate_on_bear": _run_alpha(engine, "1", "down"),
            "B_gate_on_bull": _run_alpha(engine, "1", "up"),
            "C_gate_off_bear": _run_alpha(engine, "0", "down"),
        },
        "buffett": {
            "A_gate_on_bear": _buffett_decision("1", "down"),
            "B_gate_on_bull": _buffett_decision("1", "up"),
            "C_gate_off_bear": _buffett_decision("0", "down"),
        },
        "dynamic_exit_active_after_trial": lv.dynamic_exit_active(),
    }
    os.environ["ENABLE_RL_REGIME_GATE"] = "0"      # leave the process flag OFF on exit
    return result


if __name__ == "__main__":
    res = main()
    out = ROOT / "reports" / "rl_regime_gate_paper_trial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n-> {out}")
