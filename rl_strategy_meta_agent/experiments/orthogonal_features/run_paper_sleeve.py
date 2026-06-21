#!/usr/bin/env python3
"""Promote earnings_within_7d into the paper-only sleeve path: write the registry record,
run a PAPER backtest over the validated history to populate a monitoring ledger, and emit
the latest intended (paper) target book with the fail-safe freshness gate.

PAPER ONLY -- no broker, no order, no live execution. Promotion still requires a shadow
period before any consideration of live (which is out of scope here).

    python experiments/orthogonal_features/run_paper_sleeve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[1]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(HERE))

import feature_gate as G
import paper_sleeve as PS
from run_extended_validation import build_panel

REGISTRY = HERE / "sleeve_registry.jsonl"          # committed promotion record
SLIPPAGE_BPS = 5.0


def _weight_panel(d):
    """Vectorised market-neutral, gross-capped (10%), per-name-capped weight panel."""
    s = d["earnings_within_7d"]
    date = s.index.get_level_values(1)
    z = G._cs_z(s).fillna(0.0)
    gross = z.abs().groupby(date).transform("sum") + 1e-9
    cap = PS.SLEEVE["safety"]
    return (z / gross * cap["max_allocation"]).clip(-cap["max_name_weight"], cap["max_name_weight"])


def _paper_ledger(d, w):
    date = d.index.get_level_values(1)
    nm_pnl = w * d["fut_ret_1"]
    long_prev = w.groupby(level=0).shift(1) > 1e-6
    sig, sig_prev = d["earnings_within_7d"], d["earnings_within_7d"].groupby(level=0).shift(1)
    eligible = (sig_prev == 1) & (sig == 0)                 # earnings just passed
    exited = eligible & (w <= 1e-6) & long_prev             # long position closed after earnings
    L = pd.DataFrame({
        "gross_pnl": nm_pnl.groupby(date).sum(),
        "turnover": w.groupby(level=0).diff().abs().groupby(date).sum(),
        "n_positions": (w.abs() > 1e-6).groupby(date).sum(),
        "n_wins": ((nm_pnl > 0) & (w.abs() > 1e-6)).groupby(date).sum(),
        "slippage_bps": SLIPPAGE_BPS, "earnings_age_days": 0,    # historical replay = fresh
        "post_earnings_exits": exited.groupby(date).sum(),
        "post_earnings_eligible": eligible.groupby(date).sum(),
    }).dropna()
    return L[L["n_positions"] > 0]


def main():
    d = build_panel()
    w = _weight_panel(d)
    ledger = _paper_ledger(d, w)
    realized = PS.monitor(ledger, cost_bps=SLIPPAGE_BPS)

    # latest intended paper book (fail-safe freshness gate)
    last = d.index.get_level_values(1).max()
    sig_today = d.xs(last, level=1)["earnings_within_7d"].to_dict()
    weights, diag = PS.target_weights(sig_today, fetched_at=str(last.date()), as_of=str(last.date()))

    rec = PS.export_registry(REGISTRY, as_of=str(last.date()))
    out = HERE / "out"; out.mkdir(parents=True, exist_ok=True)
    status = {"sleeve": PS.SLEEVE["name"], "decision": rec["decision"], "status": rec["status"],
              "promotion_requires_shadow_period": rec["promotion_requires_shadow_period"],
              "validated_metrics": PS.SLEEVE["validated_metrics"], "safety": PS.SLEEVE["safety"],
              "paper_backtest_realized": realized, "latest_book_diag": diag,
              "latest_intended_weights": {k: round(v, 4) for k, v in sorted(
                  weights.items(), key=lambda x: -abs(x[1]))[:10]}}
    (out / "sleeve_status.json").write_text(json.dumps(status, indent=2, default=str))
    _print(rec, realized, diag, weights)
    print(f"\nregistry -> {REGISTRY}\nstatus   -> {out/'sleeve_status.json'}")


def _print(rec, realized, diag, weights):
    print(f"\n=== PAPER SLEEVE PROMOTION: {rec['alpha_id']} ===")
    print(f"decision={rec['decision']} status={rec['status']} "
          f"shadow_required={rec['promotion_requires_shadow_period']}")
    print(f"validated: net Sharpe {PS.SLEEVE['validated_metrics']['net_sharpe_5bps']} @5bps, "
          f"breakeven {PS.SLEEVE['validated_metrics']['breakeven_bps']}bps, vol_corr "
          f"{PS.SLEEVE['validated_metrics']['vol_corr']}")
    print(f"caps: max_alloc {PS.SLEEVE['safety']['max_allocation']}, paper_only "
          f"{PS.SLEEVE['safety']['paper_only']}, leverage {PS.SLEEVE['safety']['leverage_allowed']}, "
          f"options {PS.SLEEVE['safety']['options_allowed']}, rebalance {PS.SLEEVE['safety']['rebalance']}")
    print("\npaper-backtest realized monitoring:")
    for k, v in realized.items():
        print(f"  {k}: {v}")
    print(f"\nlatest intended paper book: {diag}")
    if weights:
        print("  top weights: " + ", ".join(f"{k} {v:+.3f}" for k, v in
              sorted(weights.items(), key=lambda x: -abs(x[1]))[:6]))


if __name__ == "__main__":
    main()
