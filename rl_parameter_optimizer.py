#!/usr/bin/env python3
"""
rl_parameter_optimizer.py — Parameter optimization as objective-driven search.
==============================================================================

Implements the methodology of Tran, Pham-Hi & Bui, "Optimizing Automated Trading Systems
with Deep Reinforcement Learning" (Algorithms 2023, 16, 23) for THIS system's decision layer.

The paper reframes strategy parameter tuning as a search over a parameter response surface
(their MDP: state = dataset + history of evaluated (config, reward) pairs; action = the next
config to evaluate; reward = the strategy's objective on that config). Their two robust,
transferable findings drive this module:

  1. **Sharpe ratio is the winning objective function.** Their DDQN+Sharpe agent earned
     positive, *low-volatility* returns in every period, whereas the cumulative-return
     objective produced higher means but unstable, high-variance results. So we optimize for
     **risk-adjusted return (Sharpe)**, not raw return.
  2. **Walk-forward / anti-overfit.** They use 100 rolling periods, each split 80/20
     train/test. We tune the config on the IN-SAMPLE slice and *report* OUT-OF-SAMPLE, and
     flag configs whose OOS Sharpe collapses relative to IS (overfit).

The search itself is a dependency-free, deterministic **coordinate ascent** over a discretized
parameter grid (the paper's action space Λ) — the same greedy "pick argmax objective" move as
their ε-greedy(Q) with ε→0, and it TERMINATES when the same config is selected twice in a row
(the paper's stuck-prevention rule). No Optuna / torch needed; runs in seconds.

What it tunes: the live trade-selection / risk parameters of `rl_strategy_overlay` — including
the "Generating Alpha" downside gates (min_confluence, sent_gate_min) added alongside it:
    quality_percentile · base_risk · min_confluence · sent_gate_min · derisk_k

OFFLINE, read-only. Reuses the overlay's IS/OOS machinery. Never imports AlpacaGateway, never
submits an order, never changes a live gate — it only writes a ranked report + an OPTIONAL env
patch (same convention as run_optimizer_agent.py) for a human to review.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import rl_common as rc
import rl_strategy_overlay as ov
from oops_option_utility import summarize_returns

# ── the parameter response surface (the paper's action space Λ, discretized) ─────────
# We search the SELECTION levers — the parameters that change WHICH trades are taken, which
# is where Sharpe is the right objective (the paper tunes RSI entry/exit thresholds; our
# analog is the quality/confluence/sentiment gates). base_risk and derisk_k are deliberately
# NOT searched: Sharpe is scale-invariant to a flat risk multiplier (mean & std both scale),
# so "tuning" base_risk against Sharpe would be theater — it's a risk-budget decision left to
# the risk framework. They're held fixed (from env) and reported as context.
PARAM_GRID: Dict[str, List[Any]] = {
    "quality_pct":    [0.0, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9],
    "min_confluence": [1, 2, 3],
    "sent_gate_min":  [-1.01, -0.70, -0.40],     # -1.01 ≈ sentiment gate effectively off
}
# Deterministic starting config (current production-ish defaults).
START_CONFIG: Dict[str, Any] = {
    "quality_pct": 0.66, "min_confluence": int(rc.MIN_CONFLUENCE),
    "sent_gate_min": float(rc.SENT_GATE_MIN),
}
# Fixed risk-budget context (reported, not searched).
RISK_CONTEXT: Dict[str, Any] = {
    "base_risk": float(os.getenv("STRAT_BASE_RISK", "0.0025")),
    "derisk_k": float(os.getenv("STRAT_DERISK_K", "3.0")),
}
MIN_IS_TRADES = int(os.getenv("RL_PO_MIN_IS_TRADES", "200"))   # objective invalid below this
MIN_OOS_TRADES = int(os.getenv("RL_PO_MIN_OOS_TRADES", "50"))  # don't trust thin OOS
OVERFIT_SHARPE_GAP = float(os.getenv("RL_PO_OVERFIT_GAP", "0.10"))   # IS−OOS Sharpe gap → flag
CE_WEIGHT = float(os.getenv("RL_PO_CE_WEIGHT", "0.35"))
SHARPE_WEIGHT = float(os.getenv("RL_PO_SHARPE_WEIGHT", "0.65"))


def _params_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize an overlay params dict from a search config (gates on; Sharpe-managed)."""
    return {**ov.DEFAULT_PARAMS, **RISK_CONTEXT,
            "min_confluence": cfg["min_confluence"], "sent_gate_min": cfg["sent_gate_min"],
            "sentiment_gate": True, "regime_gate": True, "confluence_gate": True}


def _gated_full(trades: List[dict], thr: float, params: Dict[str, Any],
                sent_map: Dict[tuple, float]) -> List[float]:
    """Realized-R of every trade passing the quality threshold + gates, WITH NO per-day cap.
    This is the statistically meaningful selection surface (thousands of trades), the same
    basis the overlay uses for tune_quality_percentile / selection_edge."""
    return [rc.to_f(t.get("realized_R")) for t in trades
            if ov.quality_score(t) >= thr and ov.passes_gates(t, params, sent_map)]


def evaluate_config(is_t: List[dict], oos_t: List[dict], cfg: Dict[str, Any],
                    sent_map: Dict[tuple, float]) -> Dict[str, Any]:
    """Run one config: IS-tuned quality threshold, applied OOS (no leak). The selection
    OBJECTIVE is IS full-sample trade-Sharpe (Tran et al.); the per-day-capped account
    simulation is reported separately for realistic return/drawdown context."""
    params = _params_from_config(cfg)
    thr = ov.quality_threshold_for_percentile(is_t, cfg["quality_pct"])   # from IS only
    # Objective surface: full selection, no cap (thousands of trades).
    is_returns = _gated_full(is_t, thr, params, sent_map)
    oos_returns = _gated_full(oos_t, thr, params, sent_map)
    is_m = rc.compute_metrics(is_returns)
    oos_m = rc.compute_metrics(oos_returns)
    is_u = summarize_returns(is_returns)
    oos_u = summarize_returns(oos_returns)
    is_m.update({"certainty_equivalent": is_u["certainty_equivalent"], "skew": is_u["skew"], "kurtosis": is_u["kurtosis"]})
    oos_m.update({"certainty_equivalent": oos_u["certainty_equivalent"], "skew": oos_u["skew"], "kurtosis": oos_u["kurtosis"]})
    # Realistic account report: ≤max_trades_per_day cap + sizing + daily loss cap.
    acct_oos = ov.simulate(ov.select(oos_t, thr, params["max_trades_per_day"], params, sent_map), params)
    # Objective = IS trade-Sharpe (Tran et al.), invalid (−inf) below the trade floor so the
    # search can never "win" by concentrating onto a handful of lucky trades (their overfit case).
    utility_objective = SHARPE_WEIGHT * is_m["sharpe"] + CE_WEIGHT * is_m["certainty_equivalent"]
    objective = utility_objective if is_m["trades"] >= MIN_IS_TRADES else float("-inf")
    return {"config": dict(cfg), "objective_is_sharpe": is_m["sharpe"],
            "objective_is_certainty_equivalent": is_m["certainty_equivalent"],
            "quality_threshold": round(thr, 4), "objective": objective,
            "is": is_m, "oos": oos_m, "oos_account": acct_oos,
            "overfit": bool(is_m["sharpe"] - oos_m["sharpe"] > OVERFIT_SHARPE_GAP
                            or oos_m["trades"] < MIN_OOS_TRADES)}


def coordinate_ascent(is_t: List[dict], oos_t: List[dict], sent_map: Dict[tuple, float],
                      max_sweeps: int = 6) -> Dict[str, Any]:
    """Greedy coordinate ascent on IS Sharpe over PARAM_GRID. Terminates when a full sweep
    leaves the config unchanged (paper's 'same config twice → stop' rule) or max_sweeps hit."""
    cfg = dict(START_CONFIG)
    history: List[Dict[str, Any]] = []
    best = evaluate_config(is_t, oos_t, cfg, sent_map)
    history.append({"step": 0, "axis": "start", **{k: best[k] for k in ("config", "objective", "objective_is_sharpe")}})
    step = 0
    for sweep in range(max_sweeps):
        changed = False
        for axis, grid in PARAM_GRID.items():
            axis_best = best
            for val in grid:
                if val == cfg[axis]:
                    continue
                trial = {**cfg, axis: val}
                res = evaluate_config(is_t, oos_t, trial, sent_map)
                step += 1
                history.append({"step": step, "axis": axis, "config": dict(trial),
                                "objective": res["objective"], "objective_is_sharpe": res["objective_is_sharpe"]})
                if res["objective"] > axis_best["objective"] + 1e-9:
                    axis_best = res
            if axis_best["config"] != cfg:
                cfg = dict(axis_best["config"])           # accept the improving move on this axis
                best = axis_best
                changed = True
        if not changed:
            break                                          # converged: same config twice in a row
    return {"best": best, "sweeps": sweep + 1, "evaluations": step + 1, "history": history}


def _env_patch(best: Dict[str, Any]) -> List[str]:
    """Translate the winning SELECTION config into reviewable .env overrides for the overlay.
    base_risk / derisk_k are intentionally absent — they are risk-budget, not Sharpe levers."""
    c = best["config"]
    lines = [f"STRAT_MIN_CONFLUENCE={c['min_confluence']}"]
    if c["sent_gate_min"] <= -1.0:
        lines.append("STRAT_SENTIMENT_GATE=0   # optimizer found sentiment gate non-additive on this data")
    else:
        lines.append(f"STRAT_SENT_GATE_MIN={c['sent_gate_min']}")
    lines.append(f"# tuned quality threshold (apply via your selection layer): q>={best['quality_threshold']}")
    return lines


def optimize(train_frac: float = 0.8) -> Dict[str, Any]:
    """Full run: load enriched trades, 80/20 walk-forward split (paper §4.2), coordinate-ascent
    on IS Sharpe, report OOS. Also evaluates the START config as the honest baseline."""
    trades = ov._enriched_sorted()
    if not trades:
        return {"error": "no enriched trades — build/enrich the alpha_entry_dataset first"}
    k = max(1, int(len(trades) * train_frac))
    is_t, oos_t = trades[:k], trades[k:]
    sent_map = ov.load_sentiment_map()
    baseline = evaluate_config(is_t, oos_t, dict(START_CONFIG), sent_map)
    search = coordinate_ascent(is_t, oos_t, sent_map)
    best = search["best"]
    # Generalization check (Tran et al. §4.2 walk-forward): did the IS Sharpe gain actually
    # hold OUT-OF-SAMPLE vs the start config? This is a DIAGNOSTIC comparison (not used to
    # pick params), so it introduces no look-ahead. If it didn't generalize, say so loudly and
    # recommend NOT applying — maximizing IS Sharpe that collapses OOS is the paper's warning.
    generalizes = (best["oos"]["sharpe"] >= baseline["oos"]["sharpe"] - 1e-9 and
                   best["oos"].get("certainty_equivalent", -999.0) >= baseline["oos"].get("certainty_equivalent", -999.0) - 1e-9)
    recommendation = ("apply_after_review" if generalizes and not best["overfit"]
                      else "keep_current_config")
    return {
        "method": "coordinate_ascent_oops_utility_objective (Sharpe + CRRA certainty equivalent)",
        "total_enriched": len(trades), "is_trades": len(is_t), "oos_trades": len(oos_t),
        "train_frac": train_frac, "sentiment_keys_loaded": len(sent_map),
        "risk_context_fixed": RISK_CONTEXT,
        "evaluations": search["evaluations"], "sweeps": search["sweeps"],
        "baseline_start": {"config": baseline["config"], "is_sharpe": baseline["is"]["sharpe"],
                           "oos_sharpe": baseline["oos"]["sharpe"],
                           "oos_certainty_equivalent": baseline["oos"].get("certainty_equivalent"),
                           "oos_total_return_pct": baseline["oos_account"]["total_return_pct"],
                           "oos_max_drawdown_pct": baseline["oos_account"]["max_drawdown_pct"]},
        "best": {"config": best["config"], "quality_threshold": best["quality_threshold"],
                 "is_sharpe": best["is"]["sharpe"], "is_certainty_equivalent": best["is"].get("certainty_equivalent"),
                 "oos_sharpe": best["oos"]["sharpe"], "oos_certainty_equivalent": best["oos"].get("certainty_equivalent"),
                 "oos": best["oos"], "oos_account": best["oos_account"], "overfit_flag": best["overfit"]},
        "generalizes_oos": generalizes,
        "recommendation": recommendation,
        "env_patch": _env_patch(best) if recommendation == "apply_after_review" else
                     ["# optimizer recommends KEEPING the current config — the IS Sharpe gain did "
                      "NOT generalize out-of-sample (Tran et al. walk-forward guard)."],
        "history_tail": search["history"][-12:],
    }


def main() -> None:
    import json
    ap = argparse.ArgumentParser(description="Sharpe-objective walk-forward parameter optimizer (Tran et al. 2023)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--write-env-patch", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    res = optimize(args.train_frac)
    if "error" in res:
        print(res["error"]); raise SystemExit(0)

    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "rl_param_optimizer.json", res)
    if args.json_only:
        print(json.dumps(res, indent=2, default=str)); return

    b, base = res["best"], res["baseline_start"]
    print("=" * 78)
    print("  RL PARAMETER OPTIMIZER  —  Sharpe objective, 80/20 walk-forward (Tran et al. 2023)")
    print("=" * 78)
    print(f"  enriched={res['total_enriched']}  IS={res['is_trades']}  OOS={res['oos_trades']}  "
          f"evals={res['evaluations']}  sweeps={res['sweeps']}  sentiment_keys={res['sentiment_keys_loaded']}")
    print(f"\n  {'':14s}{'IS Sharpe':>12s}{'OOS Sharpe':>12s}{'OOS ret%':>12s}{'OOS maxDD%':>12s}")
    print(f"  {'baseline':14s}{base['is_sharpe']:>12}{base['oos_sharpe']:>12}"
          f"{base['oos_total_return_pct']:>12}{base['oos_max_drawdown_pct']:>12}")
    print(f"  {'optimized':14s}{b['is_sharpe']:>12}{b['oos_sharpe']:>12}"
          f"{b['oos_account']['total_return_pct']:>12}{b['oos_account']['max_drawdown_pct']:>12}")
    print(f"\n  Best config : {b['config']}")
    print(f"  Quality thr : q>={b['quality_threshold']}   overfit_flag={b['overfit_flag']}")
    gen = "✓ held OOS" if res["generalizes_oos"] else "✗ did NOT hold OOS"
    print(f"  Generalizes : {gen}   →  RECOMMENDATION: {res['recommendation'].upper()}")
    print(f"\n  .env overrides:")
    for line in res["env_patch"]:
        print(f"    {line}")
    if args.write_env_patch:
        out = rc.DAILY_OUTPUTS / rc.data_tag() / "rl_param_optimizer_env_patch.txt"
        rc.write_json  # noqa (keep import usage explicit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(res["env_patch"]) + "\n", encoding="utf-8")
        print(f"\n  env patch written → {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
