#!/usr/bin/env python3
"""
evaluate_rl_agents.py — Walk-forward validation of the trained RL agents vs baselines.
======================================================================================

For each agent, runs an out-of-sample (walk-forward) comparison of the RL-assisted policy
against the CURRENT heuristic/mechanical baseline, computes the metric suite, and applies
the deployment gate. SAFETY: read-only/offline — no Alpaca, no orders, no live changes.
Emits explicit deployment recommendations only.

Writes (daily_system_outputs/YYYYMMDD/):
    rl_agent_comparison.json
    rl_walkforward_report.json
    rl_deployment_recommendation.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import rl_common as rc

logger = logging.getLogger(__name__)

AGENT_DATASETS = {
    "alpha_entry": "alpha_entry_dataset",
    "alpha_exit": "alpha_exit_dataset",
    "buffett_dca": "buffett_dca_dataset",
    "capital_allocation": "capital_allocation_dataset",
}


# ── model loading (trained → heuristic surrogate fallback) ──────────────────────────
def _load_model(agent: str):
    p = rc.RL_MODEL_DIR / f"{agent}.zip"
    if p.exists() and rc.sb3_available():
        # Try PPO, then DQN (Roy 2024 TDH-DQN models save to the same path). Either exposes
        # .predict(), and _rl_action only needs kind=="ppo" to call recommend(), so a DQN
        # model evaluates identically. The deployment gate is unchanged.
        for loader in ("PPO", "DQN"):
            try:
                import stable_baselines3 as sb3
                return "ppo", getattr(sb3, loader).load(str(p), device="cpu")
            except Exception:
                continue
        logger.warning("could not load %s model (PPO/DQN) — heuristic surrogate", agent)
    return "heuristic", None


# ── per-row effective outcome under a chosen action ─────────────────────────────────
def _entry_eff(action: str, row: dict) -> float:
    r = rc.to_f(row.get("realized_R"))
    return {"enter_now": r, "reduce_size": 0.5 * r, "skip": 0.0, "wait": 0.0}.get(action, r)


def _exit_eff(action: str, row: dict) -> float:
    r = rc.to_f(row.get("realized_R")); peak = rc.to_f(row.get("peak_R"))
    giveback = max(0.0, peak - r)
    overnight = bool(row.get("overnight_loss"))
    return {
        "full_exit": r,
        "partial_exit": 0.6 * r + 0.4 * peak - 0.3 * giveback,
        "tighten_trail": r - 0.3 * giveback,
        "hold": r - giveback,
        "no_overnight": r + (abs(r) if overnight else -0.05),
        "allow_overnight": r - (abs(r) if overnight else 0.0),
    }.get(action, r)


def _dca_eff(action: str, row: dict) -> float:
    fr = rc.to_f(row.get("fwd_return_60d"))
    scale = {"no_dca": 0.0, "defer": 0.0, "small_dca": 0.5, "normal_dca": 1.0, "aggressive_dca": 1.5}.get(action, 0.0)
    cur = rc.to_f(row.get("current_weight")); tgt = rc.to_f(row.get("optimized_target_weight"))
    overconc = 0.5 * scale if (tgt > 0 and cur + 0.02 * scale > tgt) else 0.0
    return scale * fr - overconc


def _alloc_port_return(weights: Dict[str, float], row: dict) -> float:
    br = {"alpha": rc.to_f(row.get("alpha_bucket_return")),
          "buffett": rc.to_f(row.get("buffett_bucket_return")),
          "cash": rc.to_f(row.get("cash_bucket_return"))}
    return sum(weights.get(b, 0.0) * br[b] for b in br) * 100.0   # scale to ~R units


# ── baseline + RL policies ──────────────────────────────────────────────────────────
def _baseline_action(agent: str, row: dict):
    if agent == "alpha_entry":
        return "enter_now"                       # current heuristic acts on every qualified setup
    if agent == "alpha_exit":
        return "full_exit"                       # current mechanical 3R/stop exit
    if agent == "buffett_dca":
        return "normal_dca"                      # current Composite DCA default tranche
    if agent == "capital_allocation":
        return {b: 1.0 / 3 for b in rc.ALLOC_BUCKETS}   # equal-weight baseline
    return None


def _rl_action(agent: str, kind: str, model, row: dict):
    mod = __import__({"alpha_entry": "rl_env_alpha_entry", "alpha_exit": "rl_env_alpha_exit",
                      "buffett_dca": "rl_env_buffett_dca",
                      "capital_allocation": "rl_env_capital_allocation"}[agent])
    if kind == "ppo" and model is not None:
        return mod.recommend(model, row)
    # heuristic surrogate (stands in for a trained policy when none is available)
    if agent == "alpha_entry":
        pop = rc.to_f(row.get("probability_of_profit"), 0.5)
        ss = rc.to_f(row.get("setup_score"))
        return "enter_now" if (pop >= 0.55 or ss >= 0.7) else ("reduce_size" if pop >= 0.45 else "skip")
    if agent == "alpha_exit":
        r = rc.to_f(row.get("unrealized_R", row.get("realized_R")))
        if bool(row.get("overnight_loss")):
            return "no_overnight"
        return "full_exit" if r >= 2.5 else ("tighten_trail" if r >= 1.0 else "hold")
    if agent == "buffett_dca":
        disc = rc.to_f(row.get("intrinsic_value_discount")); conv = rc.to_f(row.get("conviction_score"))
        if disc >= 0.3 and conv >= 0.7:
            return "aggressive_dca"
        return "normal_dca" if disc >= 0.15 else ("small_dca" if disc >= 0.05 else "no_dca")
    if agent == "capital_allocation":
        br = {"alpha": rc.to_f(row.get("alpha_bucket_return")),
              "buffett": rc.to_f(row.get("buffett_bucket_return")),
              "cash": rc.to_f(row.get("cash_bucket_return"))}
        best = max(br, key=br.get)
        return {b: (0.6 if b == best else 0.2) for b in rc.ALLOC_BUCKETS}
    return None


def _series(agent: str, rows: List[dict], policy) -> Tuple[List[float], Dict[str, int]]:
    """Return per-row effective-R series + entry-specific counters under `policy(row)`."""
    out: List[float] = []
    counters = {"trades_avoided": 0, "bad_trades_avoided": 0, "winners_missed": 0}
    for row in rows:
        act = policy(row)
        if agent == "alpha_entry":
            out.append(_entry_eff(act, row))
            r = rc.to_f(row.get("realized_R"))
            if act in ("skip", "wait"):
                counters["trades_avoided"] += 1
                counters["bad_trades_avoided"] += int(r < 0)
                counters["winners_missed"] += int(r > 0)
        elif agent == "alpha_exit":
            out.append(_exit_eff(act, row))
        elif agent == "buffett_dca":
            out.append(_dca_eff(act, row))
        else:
            out.append(_alloc_port_return(act, row))
    return out, counters


def evaluate_agent(agent: str, train_frac: float = 0.7) -> Dict[str, Any]:
    rows = rc.load_dataset(AGENT_DATASETS[agent])
    if not rows:
        # fall back to the env's synthetic rows so evaluation still runs
        mod = __import__({"alpha_entry": "rl_env_alpha_entry", "alpha_exit": "rl_env_alpha_exit",
                          "buffett_dca": "rl_env_buffett_dca",
                          "capital_allocation": "rl_env_capital_allocation"}[agent])
        cls = {"alpha_entry": "AlphaEntryEnv", "alpha_exit": "AlphaExitEnv",
               "buffett_dca": "BuffettDCAEnv", "capital_allocation": "CapitalAllocationEnv"}[agent]
        rows = getattr(mod, cls)().rows
    split = max(1, int(len(rows) * train_frac))
    test = rows[split:] or rows[-max(1, len(rows) // 3):]   # out-of-sample slice
    kind, model = _load_model(agent)

    base_series, _ = _series(agent, test, lambda r: _baseline_action(agent, r))
    rl_series, counters = _series(agent, test, lambda r: _rl_action(agent, kind, model, r))
    base_m = rc.compute_metrics(base_series)
    rl_m = rc.compute_metrics(rl_series)
    rec, notes = rc.deployment_recommendation(base_m, rl_m, len(test))

    improvement = {k: round(rl_m.get(k, 0) - base_m.get(k, 0), 4)
                   for k in ("expectancy", "win_rate", "sharpe", "sortino", "calmar")}
    degradation = {"max_drawdown_delta": round(rl_m.get("max_drawdown", 0) - base_m.get("max_drawdown", 0), 4)}
    return {
        "agent_name": agent,
        "model_path": str(rc.RL_MODEL_DIR / (f"{agent}.zip" if kind == "ppo" else f"{agent}_heuristic.json")),
        "model_kind": kind,
        "validation_samples": len(test),
        "baseline_metrics": base_m,
        "rl_metrics": rl_m,
        "improvement": improvement,
        "degradation": degradation,
        "entry_counters": counters if agent == "alpha_entry" else None,
        "overfit_warning": notes.get("overfit_warning", False),
        "deployment_recommendation": rec,
        "gate_notes": notes,
    }


def evaluate_all() -> Dict[str, Any]:
    results = {a: evaluate_agent(a) for a in AGENT_DATASETS}
    tag = rc.data_tag()
    out = rc.DAILY_OUTPUTS / tag
    comparison = {"generated_at_utc": rc.now_iso(), "date_tag": tag,
                  "baselines": {"alpha_entry": "always_enter_now", "alpha_exit": "mechanical_full_exit",
                                "buffett_dca": "composite_normal_dca", "capital_allocation": "equal_weight"},
                  "agents": {a: {"baseline_metrics": r["baseline_metrics"],
                                 "rl_metrics": r["rl_metrics"], "improvement": r["improvement"],
                                 "degradation": r["degradation"]} for a, r in results.items()}}
    walkforward = {"generated_at_utc": rc.now_iso(), "date_tag": tag, "method": "holdout_walk_forward",
                   "train_frac": 0.7, "agents": {a: {"validation_samples": r["validation_samples"],
                                                     "rl_metrics": r["rl_metrics"]} for a, r in results.items()}}
    deployment = {"generated_at_utc": rc.now_iso(), "date_tag": tag,
                  "agents": {a: {"deployment_recommendation": r["deployment_recommendation"],
                                 "overfit_warning": r["overfit_warning"], "model_path": r["model_path"],
                                 "reason": r["gate_notes"].get("reason")} for a, r in results.items()},
                  "live_wired": False,
                  "note": "Advisory only. No agent is wired into live trading. Deploy requires "
                          "out-of-sample proof and explicit approval."}
    rc.write_json(out / "rl_agent_comparison.json", comparison)
    rc.write_json(out / "rl_walkforward_report.json", walkforward)
    rc.write_json(out / "rl_deployment_recommendation.json", deployment)
    return {"results": results, "deployment": deployment}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = evaluate_all()
    print("RL walk-forward evaluation complete (offline):")
    for a, r in res["results"].items():
        print(f"  {a}: rec={r['deployment_recommendation']} "
              f"base_exp={r['baseline_metrics']['expectancy']} rl_exp={r['rl_metrics']['expectancy']} "
              f"samples={r['validation_samples']}")
