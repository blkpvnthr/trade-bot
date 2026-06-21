"""Paper-only sleeve for earnings_within_7d -- the first ACCEPTED deployable orthogonal
signal (extended-history validation: net Sharpe 1.11 @5bps, breakeven 56bps, bootstrap +
placebo p=0, FDR-significant, 8/11 walk-forward years, holdout +0.32, vol_corr 0.03).

Promotion preserves the EXACT statistical object that passed validation: a market-neutral
daily cross-sectional book -- long the strongest earnings candidates, short the weakest,
dollar-neutral, GROSS 100% (long +50% / short -50% / net 0%), equal-weight within legs.
Capital sizing (how much $ to risk on this object) is deferred to the Capital Allocation
Review gate AFTER the shadow period -- it is NOT applied to these weights.

PAPER ONLY and self-contained: pure numpy/pandas, NO RL-allocator / execution-agent /
broker / order / live-capital imports (AST-enforced by the test). The signal is FROZEN --
no optimizer, thresholds, feature search, parameter tuning, or RL overlays. The data
handling FAILS CLOSED: any doubt about the earnings feed -> flat portfolio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_SIGNAL = "earnings_within_7d"          # immutable; changing it requires a new research branch

SLEEVE = {
    "name": "earnings_within_7d",
    "signal": FROZEN_SIGNAL,
    "frozen": True,
    "description": "Market-neutral daily cross-sectional book: long names within ~7d of next "
                   "earnings, short the rest (pre-earnings drift). First ACCEPTED signal.",
    "decision": "ACCEPT",
    "status": "PAPER_PROMOTED",
    "promotion_requires_shadow_period": True,
    "construction": {
        "market_neutral": True, "gross_target": 1.0, "long_target": 0.5, "short_target": -0.5,
        "net_target": 0.0, "weighting": "equal_weight", "rebalance": "daily",
    },
    "validated_metrics": {
        "net_sharpe_5bps": 1.11, "turnover_per_day": 0.24, "breakeven_bps": 56.0,
        "bootstrap_p": 0.0, "placebo_p": 0.0, "fdr_significant": True,
        "walk_forward_positive_years": "8/11", "holdout_net_sharpe": 0.32, "vol_corr": 0.03,
        "validation_script": "experiments/orthogonal_features/run_extended_validation.py",
    },
    "safety": {
        "paper_only": True, "leverage_allowed": False, "options_allowed": False,
        "max_earnings_age_days": 3, "max_missing_symbols_ratio": 0.05,
        "fail_closed": True,
        # proposed at the Capital Allocation Review gate -- NOT applied to the sleeve weights:
        "capital_fraction_at_review": 0.10,
    },
}


def check_data(earnings_age_days, missing_symbols_ratio, feed_available, cfg=SLEEVE):
    """Fail-closed gate. Any doubt about the earnings feed -> block (flat portfolio)."""
    s = cfg["safety"]
    if not s["fail_closed"]:
        return True, "fail_closed_disabled"
    if not feed_available:
        return False, "earnings_feed_unavailable"
    if earnings_age_days is None:
        return False, "earnings_data_missing"
    if earnings_age_days < 0:
        return False, "earnings_data_timestamp_in_future"
    if earnings_age_days > s["max_earnings_age_days"]:
        return False, f"earnings_data_stale({earnings_age_days:.1f}d>{s['max_earnings_age_days']}d)"
    if missing_symbols_ratio is None or missing_symbols_ratio > s["max_missing_symbols_ratio"]:
        return False, f"too_many_missing_symbols({missing_symbols_ratio}>{s['max_missing_symbols_ratio']})"
    return True, "ok"


def target_weights(signal: dict, data_quality: dict, cfg=SLEEVE):
    """The exact validated object: market-neutral, gross 100%, equal-weight long the
    above-mean (earnings-imminent) names, short the below-mean names. Returns
    (weights, diagnostics). FAILS CLOSED (flat) on any data-quality doubt."""
    ok, reason = check_data(data_quality.get("earnings_age_days"),
                            data_quality.get("missing_symbols_ratio"),
                            data_quality.get("feed_available", False), cfg)
    if not ok:
        return {}, {"blocked": True, "reason": reason, "gross": 0.0, "net": 0.0}
    vals = {k: float(v) for k, v in signal.items() if v is not None and np.isfinite(v)}
    if len(vals) < 2:
        return {}, {"blocked": False, "reason": "insufficient_universe_flat", "gross": 0.0, "net": 0.0}
    m = float(np.mean(list(vals.values())))
    longs = [k for k, v in vals.items() if v > m]
    shorts = [k for k, v in vals.items() if v < m]
    if not longs or not shorts:                                     # no cross-sectional dispersion
        return {}, {"blocked": False, "reason": "no_dispersion_flat", "gross": 0.0, "net": 0.0}
    c = cfg["construction"]
    w = {k: c["long_target"] / len(longs) for k in longs}
    w.update({k: c["short_target"] / len(shorts) for k in shorts})
    gross = sum(abs(v) for v in w.values()); net = sum(w.values())
    return w, {"blocked": False, "reason": "ok", "gross": round(gross, 4), "net": round(net, 6),
               "n_long": len(longs), "n_short": len(shorts)}


def registry_record(as_of=None, branch="research-orthogonal-features", shadow=None):
    return {"alpha_id": SLEEVE["name"], "signal": SLEEVE["signal"], "frozen": SLEEVE["frozen"],
            "decision": SLEEVE["decision"], "status": SLEEVE["status"],
            "promotion_requires_shadow_period": SLEEVE["promotion_requires_shadow_period"],
            "promoted_at": str(as_of) if as_of is not None else None, "branch": branch,
            "construction": SLEEVE["construction"], "validated_metrics": SLEEVE["validated_metrics"],
            "safety": SLEEVE["safety"], "description": SLEEVE["description"],
            "shadow_phase": shadow}


def export_registry(path, as_of=None, shadow=None):
    rec = registry_record(as_of, shadow=shadow)
    Path(path).write_text(json.dumps(rec) + "\n")
    return rec


def monitor(daily_ledger: pd.DataFrame, cost_bps=5.0) -> dict:
    """Realized monitoring from a daily paper ledger (cols: gross_pnl, turnover, n_wins,
    n_positions, slippage_bps, earnings_age_days, post_earnings_exits, post_earnings_eligible)."""
    L = daily_ledger
    net = L["gross_pnl"].to_numpy() - cost_bps / 1e4 * L["turnover"].to_numpy()
    return {
        "realized_net_sharpe": round(float(net.mean() / (net.std() + 1e-12) * np.sqrt(252)), 2),
        "realized_turnover_per_day": round(float(L["turnover"].mean()), 3),
        "avg_slippage_bps": round(float(L["slippage_bps"].mean()), 2),
        "hit_rate": round(float(L["n_wins"].sum() / max(int(L["n_positions"].sum()), 1)), 3),
        "earnings_data_freshness_max_age_days": int(L["earnings_age_days"].max()),
        "post_earnings_exit_rate": round(float(L["post_earnings_exits"].sum()
                                               / max(int(L["post_earnings_eligible"].sum()), 1)), 3),
        "n_days": int(len(L)),
    }


def _demo():
    sig = {f"S{i}": (1.0 if i < 3 else 0.0) for i in range(20)}
    dq_ok = {"earnings_age_days": 1, "missing_symbols_ratio": 0.0, "feed_available": True}
    w, d = target_weights(sig, dq_ok)
    assert abs(d["net"]) < 1e-9 and abs(d["gross"] - 1.0) < 1e-9          # net 0, gross 100%
    assert abs(sum(v for v in w.values() if v > 0) - 0.5) < 1e-9          # long +50%
    assert d["n_long"] == 3 and d["n_short"] == 17
    for bad in ({"earnings_age_days": 5, "missing_symbols_ratio": 0.0, "feed_available": True},
                {"earnings_age_days": 1, "missing_symbols_ratio": 0.10, "feed_available": True},
                {"earnings_age_days": 1, "missing_symbols_ratio": 0.0, "feed_available": False}):
        wb, db = target_weights(sig, bad)
        assert db["blocked"] and wb == {}, bad
    print(f"paper_sleeve._demo OK  gross={d['gross']} net={d['net']} long={d['n_long']} short={d['n_short']} "
          f"| fail-closed on stale/missing/feed-down")


if __name__ == "__main__":
    _demo()
