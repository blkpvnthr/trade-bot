#!/usr/bin/env python3
"""Historical reconciliation validation for earnings_within_7d (Phase 2).

WHY THIS IS A PERTURBATION STUDY, NOT A DIRECT RECONCILIATION: there is NO historical archive of
forward-SCHEDULED earnings dates (the cache stores only the actual `reportedDate`). So the true
scheduled-vs-actual mismatch can only be measured GOING FORWARD from live shadow logs. Here we
instead PERTURB the reportedDate (simulating reschedule / announcement-time uncertainty) and ask:

  1. mismatch frequency      - how many flag cells move when dates shift by +-k days
  2. exit-timing differences - proxied by the flag-window shift the perturbation induces
  3. drift-capture difference- net@10bps Sharpe of the perturbed signal vs the reportedDate proxy
  4. reschedule frequency    - a MODELED assumption (unobservable historically), reported as such

Then two falsification checks:
  - the date-SHUFFLE placebo (edge must die) -> the proxy did not manufacture the edge
  - jitter robustness (edge must survive +-1d) -> the exact reportedDate is not load-bearing, so a
    slightly-off forward-scheduled date still captures the drift (live reproduces historical behavior)

Reuses run_extended_validation's panel + LV.streams cost model (canonical parity). Research only.

    python experiments/orthogonal_features/run_reconciliation_validation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_extended_validation as RX
from rl_meta.stream_validation import stream_metrics

OUT = Path(__file__).resolve().parent / "out" / "reconciliation_validation_report.json"
HORIZON = 7
COST_BPS = 10.0
# acceptance thresholds (pre-registered)
MIN_RETAINED_FRAC = 0.50     # +-1d jitter must retain >= 50% of proxy net Sharpe
MAX_PLACEBO_FRAC = 0.05      # date-shuffle placebo must reproduce the edge < 5% of the time


def _earn_dates():
    return RX.NF.load_earnings(RX.SYMS)


def _flag_from_dates(idx, earn, horizon=HORIZON):
    """earnings_within_7d flag for a MultiIndex, from a {sym: np.datetime64[] dates} mapping."""
    out = pd.Series(0.0, index=idx)
    sa = idx.get_level_values(0)
    dts = pd.to_datetime(idx.get_level_values(1)).to_numpy("datetime64[D]")
    for s in pd.unique(sa):
        ed = earn.get(s, np.array([], dtype="datetime64[D]"))
        if ed.size == 0:
            continue
        ed = np.asarray(ed, dtype="datetime64[D]")
        rows = np.where(sa == s)[0]
        for r in rows:
            fut = ed[ed >= dts[r]]
            if fut.size and (fut[0] - dts[r]) / np.timedelta64(1, "D") <= horizon:
                out.iloc[r] = 1.0
    return out


def _perturb_dates(earn, delta_days, frac, seed=0):
    """Shift `frac` of each symbol's events by +-delta_days (symmetric -> no systematic look-ahead)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, ed in earn.items():
        ed = np.asarray(ed, dtype="datetime64[D]")
        if ed.size == 0:
            out[s] = ed
            continue
        move = rng.random(ed.size) < frac
        sign = rng.choice(np.array([-1, 1]), size=ed.size)
        shift = (move * sign * delta_days).astype("timedelta64[D]")
        out[s] = ed + shift
    return out


def _net(panel, sig):
    sign = 1.0 if RX.spearman_ic(sig, panel["fut_ret_5"])[0] >= 0 else -1.0
    g, t = RX.LV.streams(panel, sig * sign, fwd="fut_ret_1")
    m = stream_metrics(g.values, t.values, cost_bps=COST_BPS)
    ic = float(RX.spearman_ic(sig * sign, panel["fut_ret_5"])[0])
    return {"net_sharpe_10bps": m["net_sharpe"], "net_max_drawdown_10bps": m["net_max_drawdown"],
            "turnover_day": m["mean_turnover"], "ic": round(ic, 4)}


def _shuffle_placebo(panel, proxy_flag, real_net, n=50, seed=0):
    flag = proxy_flag.to_numpy()
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n):
        sig = pd.Series(flag[rng.permutation(len(flag))], index=panel.index)
        if sig.sum() == 0:
            continue
        if _net(panel, sig)["net_sharpe_10bps"] >= real_net:
            ge += 1
    return round(ge / n, 4)


def main():
    panel = RX.build_panel()
    earn = {s: np.asarray(v, dtype="datetime64[D]") for s, v in _earn_dates().items()}
    proxy_flag = _flag_from_dates(panel.index, earn)
    proxy = _net(panel, proxy_flag)

    # perturbation sweep: worst-case all-events +-k, plus a modeled reschedule rate
    scenarios = {
        "jitter_all_1d": dict(delta=1, frac=1.0),
        "jitter_all_2d": dict(delta=2, frac=1.0),
        "reschedule_10pct_2d": dict(delta=2, frac=0.10),
        "reschedule_25pct_1d": dict(delta=1, frac=0.25),
    }
    results = {}
    n_cells = len(proxy_flag)
    for name, cfg in scenarios.items():
        pert = _perturb_dates(earn, cfg["delta"], cfg["frac"])
        pflag = _flag_from_dates(panel.index, pert)
        mismatch = float((pflag.to_numpy() != proxy_flag.to_numpy()).mean())
        m = _net(panel, pflag)
        retained = round(m["net_sharpe_10bps"] / proxy["net_sharpe_10bps"], 3) if proxy["net_sharpe_10bps"] else 0.0
        results[name] = {**cfg, "mismatch_frequency": round(mismatch, 4),
                         "flagged_cells_proxy": int(proxy_flag.sum()), "flagged_cells_pert": int(pflag.sum()),
                         **m, "retained_net_sharpe_frac": retained}

    placebo_frac = _shuffle_placebo(panel, proxy_flag, proxy["net_sharpe_10bps"])

    # acceptance
    j1 = results["jitter_all_1d"]
    why = []
    if not (j1["net_sharpe_10bps"] > 0 and j1["retained_net_sharpe_frac"] >= MIN_RETAINED_FRAC):
        why.append(f"+-1d jitter degrades edge below {MIN_RETAINED_FRAC} retained "
                   f"({j1['retained_net_sharpe_frac']})")
    if placebo_frac >= MAX_PLACEBO_FRAC:
        why.append(f"date-shuffle placebo reproduces edge too often ({placebo_frac})")
    if j1["ic"] * proxy["ic"] <= 0:
        why.append("jitter flips IC sign vs proxy (forward-calendar would not reproduce behavior)")
    passed = len(why) == 0

    report = {
        "limitation": ("No historical archive of forward-scheduled dates exists; this is a "
                       "date-PERTURBATION proxy for the scheduled-vs-actual mismatch, which can only "
                       "be measured forward from live shadow logs."),
        "cost_bps": COST_BPS,
        "reportedDate_proxy": proxy,
        "perturbation_scenarios": results,
        "date_shuffle_placebo_frac_ge_real": placebo_frac,
        "edge_not_created_by_proxy": placebo_frac < MAX_PLACEBO_FRAC,
        "mismatch_rate_documented": True,
        "degradation_vs_proxy_acceptable": bool(j1["retained_net_sharpe_frac"] >= MIN_RETAINED_FRAC
                                                and j1["net_sharpe_10bps"] > 0),
        "forward_calendar_reproduces_behavior": bool(j1["ic"] * proxy["ic"] > 0 and j1["net_sharpe_10bps"] > 0),
        "methodology_validation_passed": passed,
        "why_not_passed": why,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    _print(report)
    return 0


def _print(r):
    print("\n=== earnings_within_7d HISTORICAL reconciliation validation (perturbation proxy) ===\n")
    print(f"reportedDate proxy: net@10bps {r['reportedDate_proxy']['net_sharpe_10bps']}, "
          f"ic {r['reportedDate_proxy']['ic']}")
    print("scenario | mismatch | net@10 | retained_frac | ic")
    for k, v in r["perturbation_scenarios"].items():
        print(f"  {k} | {v['mismatch_frequency']} | {v['net_sharpe_10bps']} | "
              f"{v['retained_net_sharpe_frac']} | {v['ic']}")
    print(f"\ndate-shuffle placebo frac>=real: {r['date_shuffle_placebo_frac_ge_real']} "
          f"(edge_not_created_by_proxy={r['edge_not_created_by_proxy']})")
    print(f"METHODOLOGY_VALIDATION_PASSED={r['methodology_validation_passed']} "
          f"{r['why_not_passed'] or ''}\n")


if __name__ == "__main__":
    raise SystemExit(main())
