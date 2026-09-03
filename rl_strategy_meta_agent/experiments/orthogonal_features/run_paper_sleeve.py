#!/usr/bin/env python3
"""Promote earnings_within_7d into the paper-only sleeve path, preserving the EXACT
validated object (market-neutral, gross 100%). Writes the registry record, runs a paper
backtest sanity check (realized ~ validated), and demonstrates the shadow-tracking +
promotion-gate framework on the most recent window as a shadow proxy.

PAPER ONLY -- no broker/order/live execution. The real shadow clock starts when a FORWARD
earnings calendar is wired (historical reportedDate has no future dates -> 'today' is
fail-safe flat). Live remains out of scope; Capital Allocation Review is the final gate.

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

import paper_sleeve as PS
import shadow as SH
from rl_meta.alphas import spearman_ic
from run_extended_validation import build_panel

REGISTRY = HERE / "sleeve_registry.jsonl"
SLIPPAGE_BPS = 5.0
SHADOW_WINDOW = 90


def _weight_panel(d):
    """Exact validated object: market-neutral, gross 100%, equal-weight per leg, per day."""
    s = d["earnings_within_7d"]
    date = s.index.get_level_values(1)
    m = s.groupby(date).transform("mean")
    longs, shorts = s > m, s < m
    nl = longs.groupby(date).transform("sum").replace(0, np.nan)
    ns = shorts.groupby(date).transform("sum").replace(0, np.nan)
    w = pd.Series(0.0, index=s.index)
    w[longs] = (0.5 / nl)[longs]
    w[shorts] = (-0.5 / ns)[shorts]
    return w.fillna(0.0)


def _paper_ledger(d, w):
    date = d.index.get_level_values(1)
    nm = w * d["fut_ret_1"]
    long_prev = w.groupby(level=0).shift(1) > 1e-9
    sig, sig_prev = d["earnings_within_7d"], d["earnings_within_7d"].groupby(level=0).shift(1)
    eligible = (sig_prev == 1) & (sig == 0)
    exited = eligible & (w <= 1e-9) & long_prev
    L = pd.DataFrame({"gross_pnl": nm.groupby(date).sum(),
                      "turnover": w.groupby(level=0).diff().abs().groupby(date).sum(),
                      "n_positions": (w.abs() > 1e-9).groupby(date).sum(),
                      "n_wins": ((nm > 0) & (w.abs() > 1e-9)).groupby(date).sum(),
                      "slippage_bps": SLIPPAGE_BPS, "earnings_age_days": 0,
                      "post_earnings_exits": exited.groupby(date).sum(),
                      "post_earnings_eligible": eligible.groupby(date).sum()}).dropna()
    return L[L["n_positions"] > 0]


def _shadow_replay(d, w):
    """Replay the last SHADOW_WINDOW trading days through ShadowTracker as a shadow proxy;
    research_ic = daily IC over the prior history. Demonstrates the promotion machinery."""
    dates = np.sort(pd.unique(d.index.get_level_values(1)))
    cut = dates[-SHADOW_WINDOW]
    lvl1 = d.index.get_level_values(1)
    research_ic = spearman_ic(d["earnings_within_7d"][lvl1 < cut], d["fut_ret_1"][lvl1 < cut])[0]
    log = HERE / "out" / "shadow_replay.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True); log.unlink(missing_ok=True)
    t = SH.ShadowTracker(log)
    for day in dates[-SHADOW_WINDOW:]:
        sl = d.xs(day, level=1)
        sig = sl["earnings_within_7d"].to_dict()
        ww = w[lvl1 == day]; ww.index = ww.index.get_level_values(0)
        weights = {k: v for k, v in ww.to_dict().items() if abs(v) > 1e-12}
        nr = sl["fut_ret_1"].to_dict()
        turn = float(ww.sub(w[d.index.get_level_values(1) == dates[list(dates).index(day) - 1]]
                            .pipe(lambda x: x.set_axis(x.index.get_level_values(0))), fill_value=0).abs().sum())
        diag = {"blocked": not weights, "reason": "ok" if weights else "no_dispersion_flat",
                "gross": sum(abs(v) for v in weights.values()), "net": sum(weights.values()),
                "n_long": sum(v > 0 for v in weights.values()),
                "n_short": sum(v < 0 for v in weights.values()), "turnover": turn}
        t.record_day(day, weights, diag, nr, sig)
    return SH.compare_to_research(t.read(), research_ic, PS.SLEEVE["validated_metrics"]["turnover_per_day"]), research_ic


def main():
    d = build_panel()
    w = _weight_panel(d)
    realized = PS.monitor(_paper_ledger(d, w), cost_bps=SLIPPAGE_BPS)
    shadow_report, research_ic = _shadow_replay(d, w)
    shadow_phase = {"status": "NOT_STARTED",
                    "required_days": "60-90 calendar (paper, zero capital)",
                    "start_trigger": "wire a FORWARD earnings calendar so 'today' produces a live book",
                    "framework_demo_on_recent_window": shadow_report}

    rec = PS.export_registry(REGISTRY, as_of=str(pd.Timestamp(d.index.get_level_values(1).max()).date()),
                             shadow=shadow_phase)
    out = HERE / "out"; out.mkdir(parents=True, exist_ok=True)
    (out / "sleeve_status.json").write_text(json.dumps(
        {"sleeve": PS.SLEEVE["name"], "decision": rec["decision"], "status": rec["status"],
         "construction": PS.SLEEVE["construction"], "validated_metrics": PS.SLEEVE["validated_metrics"],
         "safety": PS.SLEEVE["safety"], "paper_backtest_realized": realized,
         "shadow_phase": shadow_phase}, indent=2, default=str))
    _print(rec, realized, shadow_report, research_ic)
    print(f"\nregistry -> {REGISTRY}\nstatus   -> {out/'sleeve_status.json'}")


def _print(rec, realized, sr, research_ic):
    print(f"\n=== PAPER SLEEVE: {rec['alpha_id']} ===")
    print(f"decision={rec['decision']} status={rec['status']} frozen={rec['frozen']} "
          f"shadow_required={rec['promotion_requires_shadow_period']}")
    print(f"object: market-neutral gross {rec['construction']['gross_target']*100:.0f}% "
          f"(+{rec['construction']['long_target']*100:.0f}%/{rec['construction']['short_target']*100:.0f}%), "
          f"{rec['construction']['weighting']}, {rec['construction']['rebalance']} rebalance")
    print(f"caps: paper_only={rec['safety']['paper_only']} leverage={rec['safety']['leverage_allowed']} "
          f"options={rec['safety']['options_allowed']} | fail-closed: age>{rec['safety']['max_earnings_age_days']}d "
          f"or missing>{rec['safety']['max_missing_symbols_ratio']*100:.0f}% or feed-down -> FLAT")
    print(f"capital fraction PROPOSED at review (not applied now): {rec['safety']['capital_fraction_at_review']}")
    print("\npaper-backtest realized (exact validated object):")
    for k, v in realized.items():
        print(f"  {k}: {v}")
    print(f"\nshadow-framework demo (last {SHADOW_WINDOW} days as proxy): verdict={sr['verdict']}")
    print(f"  paper_ic {sr['paper_ic']} vs research_ic {SH._round(research_ic)} (match={sr['ic_match']}) | "
          f"turnover {sr['paper_turnover']} vs {sr['research_turnover']} (match={sr['turnover_match']}) | "
          f"operational_failures {sr['operational_failures']}")
    print(f"\n=== promotion path: Research -> Validated -> Paper Sleeve -> Shadow ({rec['shadow_phase']['status']}) "
          f"-> Capital Allocation Review (no direct jump to live) ===")


if __name__ == "__main__":
    main()
