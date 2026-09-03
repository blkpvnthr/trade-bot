#!/usr/bin/env python3
"""Net-of-cost validation of the two surviving orthogonal leads: idio_reversal and
earnings_within_7d. Passing the gate earns the right to this test; it does NOT mean
deployable. The real question (esp. for the 15-min reversal): does it survive realistic
trading costs?

Per lead: cross-sectional dollar-neutral long-short -> daily gross return + turnover ->
  * cost curve (net Sharpe / net cum-return vs cost in bps) + breakeven cost
  * block-bootstrap significance of net returns at a realistic cost
  * capacity proxy (1%-of-bar-volume participation)
Plus an earnings event study (event-time return path) for earnings_within_7d.

Research only: no broker/order/live path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_meta.strategy_streams import _cs_z
from rl_meta.stream_validation import stream_metrics
from rl_meta.multiple_testing import wild_bootstrap_pvalue
from rl_meta.alphas import spearman_ic
import news_features as NF
from run_feature_study import build_candidates

COST_GRID = [0, 1, 2, 3, 5, 7, 10, 15, 20]
REALISTIC_BPS = 5.0          # round-trip for liquid mega-caps (optimistic-to-fair)
BARS_PER_DAY = 26


def _ny_day(idx):
    return pd.to_datetime(idx, utc=True).tz_convert("America/New_York").normalize()


def streams(panel, signal, fwd="fut_ret_1"):
    """Dollar-neutral cross-sectional portfolio -> (daily gross return, daily turnover)."""
    w = _cs_z(signal).fillna(0.0)
    ts = w.index.get_level_values(1)
    wn = w / (w.abs().groupby(ts).transform("sum") + 1e-12)     # sum|w| = 1 per timestamp
    gross_ts = (wn * panel[fwd]).groupby(ts).sum()
    dwn = wn.groupby(level=0).diff()                            # weight change per name over time
    dwn = dwn.where(dwn.notna(), wn).abs()                      # first appearance = full entry
    turn_ts = dwn.groupby(dwn.index.get_level_values(1)).sum()
    g = gross_ts.groupby(_ny_day(gross_ts.index)).sum()
    t = turn_ts.groupby(_ny_day(turn_ts.index)).sum()
    j = g.index.intersection(t.index)
    return g.loc[j], t.loc[j]


def cost_curve(g, t):
    out = []
    for c in COST_GRID:
        m = stream_metrics(g.values, t.values, c)
        out.append({"cost_bps": c, "net_mean_daily": m["net_mean_daily"],
                    "net_cum_return": m["net_cum_return"], "net_sharpe": m["net_sharpe"]})
    return out


def capacity_usd(panel, signal, part_cap=0.01):
    """$ AUM at which a typical trade would exceed `part_cap` of that bar's dollar volume."""
    w = _cs_z(signal).fillna(0.0)
    ts = w.index.get_level_values(1)
    wn = w / (w.abs().groupby(ts).transform("sum") + 1e-12)
    dwn = wn.groupby(level=0).diff().abs()
    dvol = (panel["close"] * panel["volume"]).reindex(dwn.index)
    active = dwn > 1e-4
    cap = part_cap * dvol[active] / dwn[active]                 # $AUM s.t. trade = part_cap*dollar_vol
    cap = cap.replace([np.inf, -np.inf], np.nan).dropna()
    return {"median_usd": float(cap.median()), "p10_usd": float(cap.quantile(0.10))}


def earnings_event_study(panel, syms, window=5):
    """Average event-time daily-return path around earnings (d-w .. d+w)."""
    earn = NF.load_earnings(syms)
    daily = (panel["close"].groupby([panel.index.get_level_values(0), _ny_day(panel.index.get_level_values(1))])
             .last())
    paths = []
    for s in syms:
        if s not in daily.index.get_level_values(0):
            continue
        px = daily.loc[s]
        ret = px.pct_change()
        dts = px.index.to_numpy("datetime64[D]")
        for ed in earn.get(s, np.array([], dtype="datetime64[ns]")):
            k = np.searchsorted(dts, np.datetime64(ed, "D"))
            if window <= k < len(dts) - window:
                paths.append(ret.iloc[k - window:k + window + 1].to_numpy())
    if not paths:
        return None
    P = np.array(paths)
    mean = np.nanmean(P, axis=0)
    return {"n_events": len(paths), "rel_days": list(range(-window, window + 1)),
            "mean_daily_ret": [round(float(x), 5) for x in mean],
            "cum_into_earnings": round(float(np.nansum(mean[:window + 1])), 5),
            "cum_post_earnings": round(float(np.nansum(mean[window + 1:])), 5)}


def validate(name, panel, signal, oo_mask):
    ic_full = spearman_ic(signal, panel["fut_ret_1"])[0]
    sign = 1.0 if ic_full >= 0 else -1.0                       # trade the profitable direction
    sig = signal * sign
    g, t = streams(panel, sig)
    go, to = streams(panel.iloc[oo_mask], sig.iloc[oo_mask])
    be = stream_metrics(g.values, t.values)["breakeven_cost_bps"]
    net5 = g.values - REALISTIC_BPS / 1e4 * t.values
    sh = lambda x: round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(252)), 3)
    return {"signal": name, "ic_full": round(float(ic_full), 4), "traded_sign": sign,
            "mean_turnover_daily": round(float(t.mean()), 3),
            "gross_sharpe_ann": sh(g.values), "net_sharpe_ann_5bps": sh(net5),
            "breakeven_cost_bps": be, "cost_curve": cost_curve(g, t),
            "oos_breakeven_bps": stream_metrics(go.values, to.values)["breakeven_cost_bps"],
            "oos_net_sharpe_5bps": sh(go.values - REALISTIC_BPS / 1e4 * to.values),
            "bootstrap_p_net5bps": round(wild_bootstrap_pvalue(net5, n_boot=2000, block=1), 4),
            "capacity": capacity_usd(panel, sig),
            "survives_5bps": bool(be > REALISTIC_BPS and sh(net5) > 0.5)}


def main():
    panel = pd.read_parquet(RL_ROOT / "data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    syms = sorted(panel.index.get_level_values(0).unique())
    ts = panel.index.get_level_values(1)
    cut = pd.Series(ts).quantile(0.7)
    oo = np.where(np.asarray(ts >= cut))[0]

    px = build_candidates(panel)
    earn = NF.align(panel.index, NF.load_news(syms), NF.load_earnings(syms), {"window_h": 24.0})
    leads = {"idio_reversal": px["idio_reversal"], "earnings_within_7d": earn["earnings_within_7d"]}

    rep = {"realistic_cost_bps": REALISTIC_BPS, "leads": {}}
    for name, sig in leads.items():
        rep["leads"][name] = validate(name, panel, sig.fillna(0.0), oo)
    rep["earnings_event_study"] = earnings_event_study(panel, syms)

    out = Path(__file__).resolve().parent / "out"; out.mkdir(parents=True, exist_ok=True)
    (out / "lead_validation.json").write_text(json.dumps(rep, indent=2, default=str))
    rows = []
    for n, r in rep["leads"].items():
        rows.append({k: r[k] for k in ("signal", "ic_full", "mean_turnover_daily", "gross_sharpe_ann",
                     "net_sharpe_ann_5bps", "breakeven_cost_bps", "oos_breakeven_bps",
                     "bootstrap_p_net5bps", "survives_5bps")})
    pd.DataFrame(rows).to_csv(out / "lead_validation_summary.csv", index=False)
    _print(rep)
    print(f"\nartifacts -> {out}/ (lead_validation.json, lead_validation_summary.csv)")


def _print(rep):
    print(f"\n=== net-of-cost validation of the surviving leads (realistic cost {rep['realistic_cost_bps']}bps) ===\n")
    for n, r in rep["leads"].items():
        print(f"  {n}:")
        print(f"    IC {r['ic_full']:+.4f} | turnover/day {r['mean_turnover_daily']} | "
              f"gross Sharpe {r['gross_sharpe_ann']} -> net@5bps {r['net_sharpe_ann_5bps']}")
        print(f"    breakeven {r['breakeven_cost_bps']} bps (OOS {r['oos_breakeven_bps']}) | "
              f"bootstrap p(net@5bps) {r['bootstrap_p_net5bps']} | "
              f"capacity ~${r['capacity']['median_usd']:,.0f} (p10 ${r['capacity']['p10_usd']:,.0f})")
        print(f"    cost curve (bps -> net Sharpe): " +
              ", ".join(f"{c['cost_bps']}:{c['net_sharpe']}" for c in r['cost_curve']))
        print(f"    SURVIVES 5bps: {r['survives_5bps']}\n")
    es = rep.get("earnings_event_study")
    if es:
        print(f"  earnings event study ({es['n_events']} events): "
              f"cum return into earnings (d-5..d0) {es['cum_into_earnings']:+.4f}, "
              f"post (d+1..d+5) {es['cum_post_earnings']:+.4f}")


if __name__ == "__main__":
    main()
