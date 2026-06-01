"""
rl_shadow_mode.py — Shadow-mode RL recommendations (observe-only, never executes).
==================================================================================

When ENABLE_RL_SHADOW_MODE=true, the trained RL agents produce recommendations that are
LOGGED ONLY. The live system ignores them entirely; the Research Lab can later compare
what would have happened. This module never imports AlpacaGateway and never submits orders.

Outputs (daily_system_outputs/YYYYMMDD/):
    rl_shadow_recommendations.jsonl
    rl_shadow_performance_report.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import rl_common as rc

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv("ENABLE_RL_SHADOW_MODE", "false").lower() in {"1", "true", "yes"}


def _shadow_path() -> Path:
    return rc.DAILY_OUTPUTS / rc.data_tag() / "rl_shadow_recommendations.jsonl"


def log_recommendation(agent: str, symbol: str, context: Dict[str, Any],
                       recommendation: Any) -> None:
    """Append ONE shadow recommendation. Live behavior is never affected. Fail-safe."""
    try:
        p = _shadow_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"timestamp_utc": rc.now_iso(), "agent": agent,
               "symbol": str(symbol or ""), "recommendation": recommendation,
               "context": {k: context.get(k) for k in list(context)[:20]},
               "live_action_unchanged": True, "shadow_only": True}
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:
        logger.debug("rl shadow log failed: %s", exc)


def _models() -> Dict[str, Any]:
    """Lazily load trained models (PPO) if present; else {} (surrogate used). Guarded."""
    out: Dict[str, Any] = {}
    if not rc.sb3_available():
        return out
    try:
        from stable_baselines3 import PPO
        for agent in ("alpha_entry", "alpha_exit", "buffett_dca", "capital_allocation"):
            p = rc.RL_MODEL_DIR / f"{agent}.zip"
            if p.exists():
                try:
                    out[agent] = PPO.load(str(p), device="cpu")
                except Exception:
                    pass
    except Exception:
        pass
    return out


def run_shadow_cycle() -> Dict[str, Any]:
    """
    Read the just-written daily signal files and log RL recommendations for each — WITHOUT
    touching live behavior. No-op (returns {enabled: False}) unless ENABLE_RL_SHADOW_MODE.
    """
    if not enabled():
        return {"enabled": False}
    import evaluate_rl_agents as ev
    models = _models()
    tag = rc.data_tag()
    out_dir = rc.DAILY_OUTPUTS / tag
    logged = {"alpha_entry": 0, "alpha_exit": 0, "buffett_dca": 0}

    def _rec(agent, row):
        kind = "ppo" if agent in models else "heuristic"
        return ev._rl_action(agent, kind, models.get(agent), row)

    # Alpha entry recommendations from options_alpha_signals.json
    try:
        sig_file = out_dir / "options_alpha_signals.json"
        if sig_file.exists():
            sigs = json.loads(sig_file.read_text()).get("signals", [])
            for s in sigs:
                log_recommendation("alpha_entry", s.get("symbol", ""), s, _rec("alpha_entry", s))
                logged["alpha_entry"] += 1
    except Exception as exc:
        logger.debug("shadow alpha entry skipped: %s", exc)

    # Buffett DCA recommendations from buffett_dca_signals.json
    try:
        dca_file = out_dir / "buffett_dca_signals.json"
        if dca_file.exists():
            for s in json.loads(dca_file.read_text()).get("signals", []):
                log_recommendation("buffett_dca", s.get("symbol", ""), s, _rec("buffett_dca", s))
                logged["buffett_dca"] += 1
    except Exception as exc:
        logger.debug("shadow bal dca skipped: %s", exc)

    write_shadow_performance_report()
    return {"enabled": True, "logged": logged}


def write_shadow_performance_report(out_dir: Path = None) -> Optional[Path]:
    """Aggregate the shadow jsonl into a performance report (counts per agent/recommendation)."""
    p = _shadow_path()
    target = Path(out_dir) if out_dir else p.parent
    counts: Dict[str, Dict[str, int]] = {}
    total = 0
    try:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                agent = r.get("agent", "?")
                rec = str(r.get("recommendation"))
                counts.setdefault(agent, {}).setdefault(rec, 0)
                counts[agent][rec] += 1
    except Exception as exc:
        logger.debug("shadow perf read failed: %s", exc)
    report = {"generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(),
              "total_recommendations": total, "by_agent": counts,
              "live_wired": False, "note": "shadow-only; live system ignored these recommendations"}
    rc.write_json(target / "rl_shadow_performance_report.json", report)
    return target / "rl_shadow_performance_report.json"
