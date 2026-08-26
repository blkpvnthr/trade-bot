#!/usr/bin/env python3
"""
Falsification harness: rotation configs A/B/C/D + adaptive diagnostics.
=======================================================================

Runs the EXACT allocator on one price path in four configurations with identical
universe / dates / capital / fees / forecasts / risk params:

    A  base                  no rotation
    B  fixed loss -0.5%      loss_replacement, threshold -0.005
    C  fixed loss -2.0%      loss_replacement, threshold -0.020
    D  adaptive              opportunity-gated rotation (deterministic)
    (E adaptive + ML         DEFERRED this pass -- see LIVE_TRADING.md)

Reports net-of-cost excess-return metrics + rotation accounting + adaptive-only
diagnostics: rotation alpha (replacement - exited forward return, net of cost),
turnover / drawdown efficiency, and regime-by-regime (year-window) stability. Ends
with a PROMOTE / REJECT recommendation for config D per the acceptance gate.

    python backtest_rotation_overlays.py --synthetic [--sensitivity]
    python backtest_rotation_overlays.py            # real yfinance data
    python backtest_rotation_overlays.py --prices px.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from strat_allocator import AllocatorConfig, TRADING_DAYS, annualized_metrics
from strat import build_gate_schedule, momentum_view_panel, rolling_ic_panel, run_backtest

START = "2007-01-02"
UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ"]
ROUND_TRIP = 2 * 0.0005      # cfg.transaction_cost round trip
WINDOWS = [("2006-2009", "2006-01-01", "2009-12-31"), ("2010-2014", "2010-01-01", "2014-12-31"),
           ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2022", "2020-01-01", "2022-12-31"),
           ("2023-2026", "2023-01-01", "2026-12-31")]


def _load(args) -> pd.DataFrame:
    if args.prices:
        return pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index().dropna()
    if args.synthetic:
        sys.path.insert(0, "tests")
        from strat_fixtures import make_synthetic_prices
        return make_synthetic_prices()
    import yfinance as yf                          # research-only; isolated behind _load
    px = yf.download(UNIVERSE, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    cash = yf.download("^IRX", start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    cash = cash.iloc[:, 0] if isinstance(cash, pd.DataFrame) else cash
    out = px[UNIVERSE].copy()
    out["CASH_RATE_3M"] = cash.reindex(out.index).ffill()
    return out.dropna()


def _extra(h, cfg):
    """Ulcer index, avg drawdown, cost drag."""
    eq = h["Equity"]
    dd = eq / eq.cummax() - 1.0
    return {"ulcer": float(np.sqrt((dd ** 2).mean())), "avg_dd": float(dd[dd < 0].mean() if (dd < 0).any() else 0.0),
            "cost_drag": float((h["Turnover"] * cfg.transaction_cost * 2.0).sum())}


def _counts(h):
    lr = h.attrs.get("loss_replacement", {}); ad = h.attrs.get("adaptive_rotation", {})
    reasons = h.get("RebalanceReason")
    normal = int(((reasons == "scheduled") & h["Rebalanced"]).sum()) if reasons is not None else int(h["Rebalanced"].sum())
    return {"rebalances": int(h["Rebalanced"].sum()), "normal": normal,
            "loss_exits": int(lr.get("exits", 0)),
            "ad_sources": int(ad.get("sources", 0)), "ad_rotations": int(ad.get("rotations", 0)),
            "ad_cash": int(ad.get("cash_rotations", 0)), "ad_held": int(ad.get("held", 0))}


def _rotation_alpha(prices, h, horizons=(5, 10, 20)):
    """Rotation alpha (replacement - exited fwd return, net of cost), split risky vs cash."""
    log = h.attrs.get("adaptive_exit_log", [])
    if not log:
        return None
    px = prices
    rf = prices["CASH_RATE_3M"].astype(float) / 100.0 / TRADING_DAYS
    buckets = {"risky": {hh: [] for hh in horizons}, "cash": {hh: [] for hh in horizons}}
    for r in log:
        d, s, c = r["Date"], r["symbol"], r["replacement_asset"]
        if s not in px.columns or d not in px.index:
            continue
        fut = px.loc[px.index > d]
        b = "cash" if c is None else "risky"
        for hh in horizons:
            seg = fut.iloc[:hh]
            if len(seg) < 1:
                continue
            ex_ret = float(seg[s].iloc[-1]) / float(px.loc[d, s]) - 1.0
            if c is None:                           # rotated to cash -> earn rf (avoidance)
                rp_ret = float(rf.loc[fut.index[:hh]].sum()) if len(fut) else 0.0
            elif c in px.columns:
                rp_ret = float(seg[c].iloc[-1]) / float(px.loc[d, c]) - 1.0
            else:
                continue
            buckets[b][hh].append(rp_ret - ex_ret)
    out = {}
    for b, data in buckets.items():
        stats = {}
        for hh in horizons:
            arr = np.array(data[hh], float)
            if arr.size == 0:
                continue
            stats[hh] = {"n": arr.size, "mean": float(arr.mean()), "median": float(np.median(arr)),
                         "hit": float((arr > 0).mean()), "net_hit": float((arr - ROUND_TRIP > 0).mean())}
        out[b] = stats
    return out


def _funnel(h):
    """643 -> ... -> executed funnel from the per-source stage counts."""
    st = h.attrs.get("adaptive_stages", {})
    ad = h.attrs.get("adaptive_rotation", {})
    if not st:
        return None
    def c(*names):
        return sum(st.get(n, 0) for n in names)
    total = sum(st.values())
    return {
        "sources": total,
        "passed_hysteresis": total - c("hysteresis"),
        "passed_min_hold": total - c("hysteresis", "min_hold"),
        "candidate_available": total - c("hysteresis", "min_hold", "no_candidate"),
        "passed_quality": c("quality", "edge", "executed"),
        "passed_min_edge": c("edge", "executed"),
        "passed_cost_gate": c("executed"),
        "executed": c("executed"),
        "edge_blocked": ad.get("edge_blocked", 0),   # min_edge was the binding gate
    }


def _run(prices, cfg, gate, view, ic, start, label):
    return run_backtest(prices, cfg, gate, view=view, ic=ic, start=start, label=label)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()

    prices = _load(args)
    miss = [c for c in UNIVERSE + ["CASH_RATE_3M"] if c not in prices.columns]
    if miss:
        print(f"ERROR: prices missing columns: {miss}", file=sys.stderr)
        return 1

    base = AllocatorConfig()
    cfgs = {
        "A base":   base,
        "B -0.5%":  replace(base, loss_replacement_enabled=True, loss_replacement_threshold=-0.005),
        "C -2.0%":  replace(base, loss_replacement_enabled=True, loss_replacement_threshold=-0.020),
        "D adaptv": replace(base, adaptive_rotation_enabled=True),
    }
    rf = prices["CASH_RATE_3M"].astype(float) / 100.0 / TRADING_DAYS
    gate = build_gate_schedule(prices["SPY"], base)
    view = momentum_view_panel(prices, base)
    ic = rolling_ic_panel(view, prices)
    hs = {k: _run(prices, c, gate, view, ic, args.start, k) for k, c in cfgs.items()}
    ms = {k: annualized_metrics(h["Equity"], rf, h["Turnover"]) for k, h in hs.items()}

    print("=" * 100)
    print("ROTATION CONFIG COMPARISON  A/B/C/D  (excess-return, net of cost)")
    print("=" * 100)
    tag = ("SYNTHETIC (mechanism check, NOT validation)" if args.synthetic
           else f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Data: {tag}")
    print(f"\n{'Config':<10}{'FinalVal':>11}{'CAGR':>8}{'MaxDD':>8}{'Ulcer':>7}{'Sharpe':>8}"
          f"{'Sortino':>8}{'Calmar':>7}{'AnnTurn':>8}{'Cost':>7}")
    for k in cfgs:
        m, e = ms[k], _extra(hs[k], cfgs[k])
        print(f"{k:<10}{m['final_equity']:>11,.0f}{m['cagr']:>8.2%}{m['max_drawdown']:>8.2%}"
              f"{e['ulcer']:>7.3f}{m['sharpe']:>8.2f}{m['sortino']:>8.2f}"
              f"{(m['calmar'] if np.isfinite(m['calmar']) else 0):>7.2f}"
              f"{m['annual_turnover']:>8.1%}{e['cost_drag']:>7.2%}")

    print(f"\n{'Config':<10}{'Rebal':>7}{'Normal':>7}{'LossX':>7}{'AdSrc':>7}{'AdRot':>7}"
          f"{'AdCash':>7}{'AdHold':>7}")
    for k in cfgs:
        c = _counts(hs[k])
        print(f"{k:<10}{c['rebalances']:>7}{c['normal']:>7}{c['loss_exits']:>7}"
              f"{c['ad_sources']:>7}{c['ad_rotations']:>7}{c['ad_cash']:>7}{c['ad_held']:>7}")

    # efficiency vs baseline
    A = ms["A base"]
    print("\nEfficiency vs A (baseline):")
    for k in ("B -0.5%", "C -2.0%", "D adaptv"):
        m = ms[k]
        dturn = m["annual_turnover"] - A["annual_turnover"]
        dd_impr = A["max_drawdown"] - m["max_drawdown"]   # >0 = shallower (D better)
        cagr_sac = A["cagr"] - m["cagr"]                  # >0 = D gave up return
        te_cagr = (m["cagr"] - A["cagr"]) / dturn if abs(dturn) > 1e-9 else float("nan")
        te_shrp = (m["sharpe"] - A["sharpe"]) / dturn if abs(dturn) > 1e-9 else float("nan")
        dd_eff = (cagr_sac / dd_impr) if abs(dd_impr) > 1e-9 else float("nan")
        print(f"  {k:<9} dSharpe {m['sharpe']-A['sharpe']:+.3f}  dCAGR {m['cagr']-A['cagr']:+.2%}  "
              f"dMaxDD {m['max_drawdown']-A['max_drawdown']:+.2%}  dTurn {dturn:+.1%}  "
              f"turn-eff(Sharpe/turn) {te_shrp:+.3f}  DDeff(CAGRsac/DDimpr) {dd_eff:+.2f}")

    # adaptive decision funnel (are the intended gates causing the abstention?)
    fn = _funnel(hs["D adaptv"])
    if fn:
        print("\n[D adaptive] decision funnel (deterioration -> executed):")
        print(f"    {fn['sources']:>5}  deterioration sources")
        print(f"    {fn['passed_hysteresis']:>5}  passed hysteresis (hold_score <= floor)")
        print(f"    {fn['passed_min_hold']:>5}  passed min-hold (or damage-bypass)")
        print(f"    {fn['candidate_available']:>5}  had an eligible candidate")
        print(f"    {fn['passed_quality']:>5}  candidate passed quality floor")
        print(f"    {fn['passed_min_edge']:>5}  passed minimum-edge gate")
        print(f"    {fn['passed_cost_gate']:>5}  passed cost gate")
        print(f"    {fn['executed']:>5}  EXECUTED   "
              f"(abstention {1 - fn['executed'] / max(fn['sources'], 1):.1%})")
        print(f"    min_edge binding on {fn['edge_blocked']} source(s) "
              f"(> 0 confirms adaptive_minimum_rotation_edge participates)")

    # rotation alpha, split risky->risky vs risky->cash (different economics)
    ra = _rotation_alpha(prices, hs["D adaptv"])
    print("\n[D adaptive] rotation alpha (replacement - exited fwd return; EVALUATION ONLY):")
    if not ra:
        print("    no rotations executed (opportunity gate held everything)")
    else:
        for b, label in (("risky", "risky->risky (selection)"), ("cash", "risky->cash (avoidance)")):
            stats = ra.get(b, {})
            if not stats:
                print(f"    {label:<26} none")
                continue
            for hh, r in stats.items():
                print(f"    {label if hh == list(stats)[0] else '':<26} {hh:>2}D  n={r['n']:<3} "
                      f"mean {r['mean']:+.2%}  median {r['median']:+.2%}  "
                      f"hit>0 {r['hit']:.0%}  net-of-cost hit {r['net_hit']:.0%}")

    # regime-by-regime (year windows)
    print("\nRegime stability (Sharpe / MaxDD per window):")
    print(f"  {'window':<11}" + "".join(f"{k:>16}" for k in cfgs))
    for name, s, e in WINDOWS:
        row = f"  {name:<11}"
        any_data = False
        for k in cfgs:
            h = hs[k]; sub = h.loc[(h.index >= s) & (h.index <= e), "Equity"]
            if len(sub) > 20:
                mm = annualized_metrics(sub, rf.loc[sub.index], hs[k].loc[sub.index, "Turnover"])
                row += f"  {mm['sharpe']:>6.2f}/{mm['max_drawdown']:>6.1%}"
                any_data = True
            else:
                row += f"  {'--':>13}"
        if any_data:
            print(row)

    # verdict
    print("\n" + "=" * 100)
    D, Bm, Cm = ms["D adaptv"], ms["B -0.5%"], ms["C -2.0%"]
    dS = D["sharpe"] - A["sharpe"]; dC = D["cagr"] - A["cagr"]
    ddI = A["max_drawdown"] - D["max_drawdown"]
    risky_stats = (ra or {}).get("risky", {})
    # convincing only if a MAJORITY of horizons are net-positive AND better than a coin flip.
    net_alpha_pos = (sum(1 for r in risky_stats.values()
                         if r["mean"] - ROUND_TRIP > 0 and r["hit"] >= 0.5) >= 2)
    # architecture gate: does D dominate the fixed stops (better Sharpe AND lower turnover)?
    beats_fixed = (D["sharpe"] >= Bm["sharpe"] - 1e-9 and D["annual_turnover"] < Bm["annual_turnover"]
                   and D["sharpe"] >= Cm["sharpe"] - 1e-9 and D["annual_turnover"] < Cm["annual_turnover"])
    alpha_gates = {
        "Sharpe up vs A @ comparable MaxDD": dS > 0.02 and abs(D["max_drawdown"] - A["max_drawdown"]) < 0.02,
        "CAGR up vs A @ comparable MaxDD": dC > 0.005 and abs(D["max_drawdown"] - A["max_drawdown"]) < 0.02,
        "MaxDD materially shallower vs A, modest CAGR sacrifice": ddI > 0.03 and dC > -0.02,
        "net RISKY rotation alpha > 0 after cost": net_alpha_pos,
    }
    print("Acceptance gate:")
    print(f"  [{'PASS' if beats_fixed else '  - '}] ARCHITECTURE: D dominates B/C "
          f"(>= Sharpe AND < turnover)")
    for g, ok in alpha_gates.items():
        print(f"  [{'PASS' if ok else '  - '}] ALPHA vs A: {g}")
    print(f"\nVERDICT:")
    print(f"  {'PROMOTE D over B/C as the rotation architecture.' if beats_fixed else 'D does NOT clearly beat the fixed stops -- REVISE.'}")
    if not any(alpha_gates.values()):
        print(f"  KEEP D OFF by default vs A (no convincing alpha: dSharpe {dS:+.3f}, "
              f"dCAGR {dC:+.2%}) -- pending true holdout validation.")
    else:
        print(f"  D shows possible alpha vs A -- still confirm on true holdout before enabling.")
    print(f"{'SYNTHETIC -- not evidence for deployment.' if args.synthetic else 'Real data; still OWES walk-forward/holdout/placebo/bootstrap before deploy.'}")

    if args.sensitivity:
        print("\n" + "=" * 100)
        print("ADAPTIVE PARAMETER SENSITIVITY (config D; coarse grid, development-period only)")
        print("=" * 100)
        print(f"{'min_edge':>9}{'dmgx':>6}{'hyst':>6}{'AdRot':>7}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}{'AnnTurn':>9}")
        for edge in (0.10, 0.15, 0.20):
            for dmgx in (0.75, 1.0, 1.25):
                for hyst in (0.30, 0.40, 0.50):
                    thr = {kk: vv * dmgx for kk, vv in base.adaptive_damage_thresholds.items()}
                    cfg = replace(base, adaptive_rotation_enabled=True,
                                  adaptive_minimum_rotation_edge=edge,
                                  adaptive_damage_thresholds=thr,
                                  adaptive_hold_score_floor=hyst)
                    h = _run(prices, cfg, gate, view, ic, args.start, "S")
                    m = annualized_metrics(h["Equity"], rf, h["Turnover"])
                    rot = _counts(h)["ad_rotations"] + _counts(h)["ad_cash"]
                    print(f"{edge:>9.2f}{dmgx:>6.2f}{hyst:>6.2f}{rot:>7}{m['cagr']:>9.2%}"
                          f"{m['max_drawdown']:>9.2%}{m['sharpe']:>8.2f}{m['annual_turnover']:>9.1%}")
        print("Do NOT pick the best row on the evaluation period; report stability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
