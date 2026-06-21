"""Paper-only sleeve for earnings_within_7d -- the first ACCEPTED deployable orthogonal
signal (extended-history validation: net Sharpe 1.11 @5bps, breakeven 56bps, bootstrap
+ placebo p=0, FDR-significant, 8/11 walk-forward years, holdout +0.32, vol_corr 0.03).

PAPER ONLY. This module is pure (numpy/pandas only) and imports NO broker / order /
execution / live-trading code. It produces target weights for a market-NEUTRAL
cross-sectional book (long names within ~7d of earnings, short the rest), capped, with a
hard fail-safe: if the earnings-date data is missing or stale, it BLOCKS (returns flat).
Live trading is out of scope; promotion still requires a shadow period.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SLEEVE = {
    "name": "earnings_within_7d",
    "signal": "earnings_within_7d",
    "description": "Long names within 7 calendar days of next scheduled earnings, short the "
                   "rest (pre-earnings drift). First ACCEPTED orthogonal signal.",
    "decision": "ACCEPT",
    "status": "PAPER_PROMOTED",
    "promotion_requires_shadow_period": True,
    "validated_metrics": {
        "net_sharpe_5bps": 1.11, "turnover_per_day": 0.24, "breakeven_bps": 56.0,
        "bootstrap_p": 0.0, "placebo_p": 0.0, "fdr_significant": True,
        "walk_forward_positive_years": "8/11", "holdout_net_sharpe": 0.32, "vol_corr": 0.03,
        "validation_script": "experiments/orthogonal_features/run_extended_validation.py",
    },
    "safety": {
        "paper_only": True, "max_allocation": 0.10, "max_name_weight": 0.04,
        "leverage_allowed": False, "options_allowed": False, "rebalance": "daily",
        "block_on_stale_earnings": True, "earnings_staleness_max_days": 2,
    },
}


def check_freshness(fetched_at, as_of, cfg=SLEEVE):
    """Fail-safe gate: block when earnings-date data is missing or stale."""
    s = cfg["safety"]
    if not s["block_on_stale_earnings"]:
        return True, "freshness_check_disabled"
    if fetched_at is None:
        return False, "earnings_data_missing"
    age = (pd.Timestamp(as_of) - pd.Timestamp(fetched_at)).total_seconds() / 86400.0
    if age < 0:
        return False, "earnings_data_timestamp_in_future"
    if age > s["earnings_staleness_max_days"]:
        return False, f"earnings_data_stale({age:.1f}d>{s['earnings_staleness_max_days']}d)"
    return True, "fresh"


def enforce_caps(weights: dict, cfg=SLEEVE) -> dict:
    """Per-name cap, then gross cap (no leverage). Options are not representable (equities
    only). Returns a new capped weight dict."""
    s = cfg["safety"]
    w = {k: float(np.clip(v, -s["max_name_weight"], s["max_name_weight"])) for k, v in weights.items()}
    gross = sum(abs(v) for v in w.values())
    if gross > s["max_allocation"] and gross > 0:
        w = {k: v * s["max_allocation"] / gross for k, v in w.items()}   # scale down, never up
    return w


def target_weights(signal: dict, fetched_at, as_of, cfg=SLEEVE):
    """Market-neutral cross-sectional weights from the earnings_within_7d signal, gross-capped.
    Returns (weights, diagnostics). Blocks (flat) on stale/missing earnings data."""
    ok, reason = check_freshness(fetched_at, as_of, cfg)
    if not ok:
        return {}, {"blocked": True, "reason": reason, "gross": 0.0}
    vals = {k: float(v) for k, v in signal.items() if v is not None and np.isfinite(v)}
    arr = np.array(list(vals.values()), float)
    if len(arr) < 2 or arr.std() < 1e-12:
        return {}, {"blocked": False, "reason": "no_dispersion_flat", "gross": 0.0, "n": 0}
    m, sd = arr.mean(), arr.std() + 1e-9
    z = {k: (v - m) / sd for k, v in vals.items()}                      # dollar-neutral (sum~0)
    g = sum(abs(v) for v in z.values()) or 1.0
    w = enforce_caps({k: v / g * cfg["safety"]["max_allocation"] for k, v in z.items()}, cfg)
    gross = sum(abs(v) for v in w.values())
    return w, {"blocked": False, "reason": "ok", "gross": round(gross, 4),
               "n_long": sum(v > 0 for v in w.values()), "n_short": sum(v < 0 for v in w.values())}


def registry_record(as_of=None, branch="research-orthogonal-features"):
    return {"alpha_id": SLEEVE["name"], "signal": SLEEVE["signal"],
            "decision": SLEEVE["decision"], "status": SLEEVE["status"],
            "promotion_requires_shadow_period": SLEEVE["promotion_requires_shadow_period"],
            "promoted_at": str(as_of) if as_of is not None else None, "branch": branch,
            "validated_metrics": SLEEVE["validated_metrics"], "safety": SLEEVE["safety"],
            "description": SLEEVE["description"]}


def export_registry(path, as_of=None):
    """Write the sleeve's promotion record as a JSONL row (the registry)."""
    rec = registry_record(as_of)
    Path(path).write_text(json.dumps(rec) + "\n")
    return rec


def monitor(daily_ledger: pd.DataFrame, cost_bps=5.0) -> dict:
    """Realized monitoring from a daily paper ledger. Expected columns: gross_pnl, turnover,
    n_wins, n_positions, slippage_bps, earnings_age_days, post_earnings_exits,
    post_earnings_eligible."""
    L = daily_ledger
    net = L["gross_pnl"].to_numpy() - cost_bps / 1e4 * L["turnover"].to_numpy()
    den_hit = max(int(L["n_positions"].sum()), 1)
    den_exit = max(int(L["post_earnings_eligible"].sum()), 1)
    return {
        "realized_net_sharpe": round(float(net.mean() / (net.std() + 1e-12) * np.sqrt(252)), 2),
        "realized_turnover_per_day": round(float(L["turnover"].mean()), 3),
        "avg_slippage_bps": round(float(L["slippage_bps"].mean()), 2),
        "hit_rate": round(float(L["n_wins"].sum() / den_hit), 3),
        "earnings_data_freshness_max_age_days": int(L["earnings_age_days"].max()),
        "post_earnings_exit_rate": round(float(L["post_earnings_exits"].sum() / den_exit), 3),
        "n_days": int(len(L)),
    }


def _demo():
    # market-neutral weights, gross capped, fail-safe on stale data
    sig = {f"S{i}": (1.0 if i < 3 else 0.0) for i in range(20)}      # 3 names within 7d of earnings
    w, diag = target_weights(sig, fetched_at="2026-06-17", as_of="2026-06-18")
    assert not diag["blocked"] and diag["gross"] <= SLEEVE["safety"]["max_allocation"] + 1e-9
    assert abs(sum(w.values())) < 1e-6                               # dollar-neutral
    assert max(abs(v) for v in w.values()) <= SLEEVE["safety"]["max_name_weight"] + 1e-9
    wb, db = target_weights(sig, fetched_at="2026-06-01", as_of="2026-06-18")  # stale
    assert db["blocked"] and wb == {}
    print(f"paper_sleeve._demo OK  gross={diag['gross']} long={diag['n_long']} short={diag['n_short']} "
          f"| stale-block reason: {db['reason']}")


if __name__ == "__main__":
    _demo()
