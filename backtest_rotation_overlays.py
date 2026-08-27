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
    F  dynamic risk          portfolio-risk controller (vol-budget multiplier)
    G  D + dynamic risk      adaptive selection AND dynamic risk budgeting
    (E adaptive + ML         DEFERRED this pass -- see LIVE_TRADING.md)

Reports net-of-cost excess-return metrics + rotation accounting + adaptive-only
diagnostics (rotation alpha, turnover/drawdown efficiency, regime stability) and,
for the dynamic-risk controller, multiplier distribution, up/down-market capture
vs SPY, deepest-drawdown attribution, risk-timing contribution, and a FROZEN
chronological dev/holdout split (holdout evaluated once, never used to pick
params). Ends with a PROMOTE / REVISE / REJECT recommendation for D and for the
dynamic-risk controller (F/G). The --ablation ladder (F1-F5) establishes whether
the controller adds information beyond the existing market-regime gate.

    python backtest_rotation_overlays.py --synthetic [--sensitivity] [--ablation]
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


def _capture(h, spy_ret):
    """Up/down-market capture vs SPY. down_capture < 1 = falls less; ratio > 1 = good."""
    r = h["Equity"].pct_change()
    common = r.index.intersection(spy_ret.index)
    r, s = r.reindex(common), spy_ret.reindex(common)
    up, dn = s > 0, s < 0
    su, sd = float(s[up].sum()), float(s[dn].sum())
    upc = float(r[up].sum()) / su if abs(su) > 1e-12 else float("nan")
    dnc = float(r[dn].sum()) / sd if abs(sd) > 1e-12 else float("nan")
    ratio = upc / abs(dnc) if np.isfinite(dnc) and abs(dnc) > 1e-9 else float("nan")
    return {"up": upc, "down": dnc, "ratio": ratio}


def _dd_attribution(h):
    """Multiplier behavior through the deepest drawdown: did it de-risk BEFORE the bottom?"""
    if "RiskMultiplier" not in h.columns:
        return None
    eq = h["Equity"]
    dd = eq / eq.cummax() - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    win = h.loc[peak:trough, "RiskMultiplier"]
    if len(win) < 2:
        return None
    # days between the lowest-multiplier day and the price trough (>0 = led the bottom).
    lead = int(len(win) - 1 - win.to_numpy().argmin())
    after = h.loc[trough:, "RiskMultiplier"]
    back_1x = after[after >= 1.0]
    recov = int((back_1x.index[0] - trough).days) if len(back_1x) else -1
    return {"peak": peak.date(), "trough": trough.date(), "depth": float(dd.min()),
            "mult_at_peak": float(win.iloc[0]), "mult_at_trough": float(win.iloc[-1]),
            "mult_avg": float(win.mean()), "mult_min": float(win.min()),
            "derisk_lead_days": lead, "recovery_days_to_1x": recov}


def _subperiod(h, rf, s, e):
    sub = h.loc[(h.index >= s) & (h.index <= e), "Equity"]
    if len(sub) < 20:
        return None
    return annualized_metrics(sub, rf.loc[sub.index], h.loc[sub.index, "Turnover"])


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


def _ablation_ladder(prices, base, gate, view, ic, start, rf, dev):
    """
    F1-F5: does the controller add information BEYOND the market-regime gate?

    Each rung wires a strict superset of the previous one; everything else is
    neutral. SevereBear snap-to-floor is common to every rung (it is the
    independent hard-risk layer, not part of the controller's opportunity/stress
    logic). Read the marginal Sharpe/CAGR/MaxDD steps F1->F5; the controller only
    earns its place if F4/F5 beat F1 (regime only) materially.
    """
    BR = {"view_breadth": 0.30, "mom_breadth": 0.20, "trend_breadth": 0.25}   # breadth, no regime
    ALL = dict(base.dynamic_risk_opp_weights)
    SW = dict(base.dynamic_risk_stress_weights)

    def _cfg(opp, stress, gamma):
        return replace(base, dynamic_risk_enabled=True, dynamic_risk_opp_weights=opp,
                       dynamic_risk_stress_weights=stress, dynamic_risk_stress_gamma_recovery=gamma)

    ladder = {
        "F1 regime":  _cfg({"regime": 1.0}, {}, 0.0),
        "F2 breadth": _cfg(BR, {}, 0.0),
        "F3 br+strs": _cfg(BR, SW, 0.0),
        "F4 +regime": _cfg(ALL, SW, 0.0),
        "F5 +recov":  _cfg(ALL, SW, 0.5),
    }
    print("\n" + "=" * 100)
    print("COMPONENT ABLATION LADDER F1-F5  (development period only)")
    print("=" * 100)
    print(f"  {'rung':<12}{'devCAGR':>9}{'devMaxDD':>10}{'devShp':>8}{'devClmr':>9}{'AnnTurn':>9}")
    res = {}
    for name, cfg in ladder.items():
        h = _run(prices, cfg, gate, view, ic, start, name)
        m = _subperiod(h, rf, *dev) or annualized_metrics(h["Equity"], rf, h["Turnover"])
        res[name] = m
        print(f"  {name:<12}{m['cagr']:>9.2%}{m['max_drawdown']:>10.2%}{m['sharpe']:>8.2f}"
              f"{(m['calmar'] if np.isfinite(m['calmar']) else 0):>9.2f}{m['annual_turnover']:>9.1%}")
    f1, f4, f5 = res["F1 regime"], res["F4 +regime"], res["F5 +recov"]
    incr = (f4["sharpe"] > f1["sharpe"] + 0.03) or (f5["sharpe"] > f1["sharpe"] + 0.03)
    print(f"\n  Incremental info beyond regime (F4/F5 Sharpe > F1 + 0.03): "
          f"{'YES' if incr else 'NO -- controller is a slower regime filter; REJECT'}")
    print("  (SevereBear de-risk is common to all rungs; the ladder isolates the "
          "opportunity/stress/recovery signal only.)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="run the F1-F5 component ablation ladder (dev period only)")
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
        "F dynrsk": replace(base, dynamic_risk_enabled=True),
        "G D+dynr": replace(base, adaptive_rotation_enabled=True, dynamic_risk_enabled=True),
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

    # up/down-market capture vs SPY (does the controller keep upside while cutting downside?)
    spy_ret = prices["SPY"].astype(float).pct_change()
    print("\nUp/down-market capture vs SPY (down < 1 = falls less; ratio > 1 = favorable):")
    for k in cfgs:
        cap = _capture(hs[k], spy_ret)
        print(f"  {k:<10} up {cap['up']:>6.2f}  down {cap['down']:>6.2f}  ratio {cap['ratio']:>6.2f}")

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

    # -------------------------------------------------------------------------
    # DYNAMIC PORTFOLIO-RISK CONTROLLER (configs F / G)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DYNAMIC PORTFOLIO-RISK CONTROLLER  (F = dynamic only, G = D + dynamic)")
    print("=" * 100)

    print("\nMultiplier distribution (per-day vol-budget scaler):")
    for k in ("F dynrsk", "G D+dynr"):
        d = hs[k].attrs.get("dynamic_risk")
        if d:
            print(f"  {k:<10} mean {d['mult_mean']:.3f}  min {d['mult_min']:.3f}  "
                  f"max {d['mult_max']:.3f}  std {d['mult_std']:.3f}  "
                  f"de-risked {d['frac_derisked']:.0%} of days  added-risk {d['frac_added']:.0%}")

    print("\nDeepest-drawdown attribution (did the controller de-risk BEFORE the bottom?):")
    for k in ("F dynrsk", "G D+dynr"):
        at = _dd_attribution(hs[k])
        if at:
            print(f"  {k:<10} {at['peak']}->{at['trough']} depth {at['depth']:+.1%}  "
                  f"mult peak {at['mult_at_peak']:.2f} -> trough {at['mult_at_trough']:.2f} "
                  f"(avg {at['mult_avg']:.2f}, min {at['mult_min']:.2f})  "
                  f"de-risk lead {at['derisk_lead_days']}d  back-to-1x {at['recovery_days_to_1x']}d")

    # risk-timing contribution: F-A isolates exposure timing (no selection change);
    # G-D isolates the same on top of the adaptive selection.
    Fm, Gm, Dm = ms["F dynrsk"], ms["G D+dynr"], ms["D adaptv"]
    print("\nRisk-timing contribution (exposure timing only, selection held fixed):")
    print(f"  F - A : dCAGR {Fm['cagr']-A['cagr']:+.2%}  dMaxDD {Fm['max_drawdown']-A['max_drawdown']:+.2%}  "
          f"dSharpe {Fm['sharpe']-A['sharpe']:+.3f}  dCalmar {(Fm['calmar'] if np.isfinite(Fm['calmar']) else 0)-(A['calmar'] if np.isfinite(A['calmar']) else 0):+.2f}")
    print(f"  G - D : dCAGR {Gm['cagr']-Dm['cagr']:+.2%}  dMaxDD {Gm['max_drawdown']-Dm['max_drawdown']:+.2%}  "
          f"dSharpe {Gm['sharpe']-Dm['sharpe']:+.3f}  dCalmar {(Gm['calmar'] if np.isfinite(Gm['calmar']) else 0)-(Dm['calmar'] if np.isfinite(Dm['calmar']) else 0):+.2f}")

    # frozen chronological dev/holdout split. Params are frozen (defaults); the
    # holdout is evaluated ONCE and never used to pick parameters (refinement 6).
    DEV = ("2006-01-01", "2018-12-31"); HOLD = ("2019-01-01", "2026-12-31")
    print(f"\nDev/holdout split (dev {DEV[0][:4]}-{DEV[1][:4]}, HELD-OUT {HOLD[0][:4]}-{HOLD[1][:4]}; "
          f"holdout evaluated once, params frozen):")
    print(f"  {'config':<10}{'dev CAGR':>10}{'dev MaxDD':>11}{'dev Shp':>9}   "
          f"{'HOLD CAGR':>10}{'HOLD MaxDD':>11}{'HOLD Shp':>9}{'HOLD Clmr':>10}")
    hold_metrics = {}
    for k in ("A base", "D adaptv", "F dynrsk", "G D+dynr"):
        dv = _subperiod(hs[k], rf, *DEV); ho = _subperiod(hs[k], rf, *HOLD)
        hold_metrics[k] = ho
        if dv and ho:
            print(f"  {k:<10}{dv['cagr']:>10.2%}{dv['max_drawdown']:>11.2%}{dv['sharpe']:>9.2f}   "
                  f"{ho['cagr']:>10.2%}{ho['max_drawdown']:>11.2%}{ho['sharpe']:>9.2f}"
                  f"{(ho['calmar'] if np.isfinite(ho['calmar']) else 0):>10.2f}")
        else:
            print(f"  {k:<10}  (insufficient data in one or both sub-periods -- synthetic?)")

    # F/G acceptance on the HELD-OUT split (primary criterion, refinement 7).
    print("\n" + "-" * 100)
    hA, hF, hG = hold_metrics.get("A base"), hold_metrics.get("F dynrsk"), hold_metrics.get("G D+dynr")
    if hA and hF and hG:
        def _verdict(name, cand, baseline, base_name):
            cagr_up = cand["cagr"] > baseline["cagr"]
            dd_better = (baseline["max_drawdown"] - cand["max_drawdown"]) > 0.02   # materially shallower
            risk_adj_up = (cand["sharpe"] > baseline["sharpe"] + 1e-9
                           and (cand["calmar"] if np.isfinite(cand["calmar"]) else 0)
                               >= (baseline["calmar"] if np.isfinite(baseline["calmar"]) else 0))
            primary = cagr_up and dd_better and risk_adj_up
            secondary = (dd_better and cand["cagr"] > baseline["cagr"] - 0.02
                         and cand["sharpe"] > baseline["sharpe"] + 1e-9)
            tag = "PROMOTE" if primary else ("REVISE" if secondary else "REJECT")
            print(f"  [{tag:>7}] {name} vs {base_name} (held-out): "
                  f"CAGR {cand['cagr']:+.2%} vs {baseline['cagr']:+.2%} | "
                  f"MaxDD {cand['max_drawdown']:+.2%} vs {baseline['max_drawdown']:+.2%} | "
                  f"Sharpe {cand['sharpe']:.2f} vs {baseline['sharpe']:.2f}")
            return tag
        vF = _verdict("F", hF, hA, "A")
        vG = _verdict("G", hG, hA, "A")
        _verdict("G", hG, hold_metrics["D adaptv"], "D")   # complementarity read
        print("\nVERDICT (DYNAMIC RISK CONTROLLER):")
        best = "PROMOTE" if "PROMOTE" in (vF, vG) else ("REVISE" if "REVISE" in (vF, vG) else "REJECT")
        print(f"  {best} DYNAMIC RISK CONTROLLER  (F={vF}, G={vG} vs A on the held-out split).")
        print("  Reminder: also require F4/F5 > F1 (regime-only) in the --ablation ladder before")
        print("  crediting the controller with information beyond the existing market-regime gate.")
        print("  Do NOT promote a config that only rescues 2020/2022 while hurting other regimes.")
    else:
        print("  (dev/holdout verdict needs real multi-year data; run without --synthetic.)")

    if args.ablation:
        _ablation_ladder(prices, base, gate, view, ic, args.start, rf, DEV)

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
