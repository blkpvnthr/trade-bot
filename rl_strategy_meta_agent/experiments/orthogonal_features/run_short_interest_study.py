#!/usr/bin/env python3
"""Short-interest / short-volume through the gate + net-of-cost funnel, DAILY horizon.

Does short/borrow-pressure information add deployable orthogonal alpha beyond the
accepted baseline -- volatility_factor, idio_reversal, earnings_within_7d? Daily panel
(avoid 15m churn), horizons 1/3/5/10/20d, both channels (bi-weekly short interest +
daily short volume), interaction tests, then the full net-of-cost funnel. No new search
engine -- reuses feature_gate + the lead-validation cost machinery. Research only.

    python experiments/orthogonal_features/run_short_interest_study.py
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

import feature_gate as G
import short_interest_features as SI
import news_features as NF
import run_lead_validation as LV

HORIZONS = [1, 3, 5, 10, 20]
COST_BPS = 5.0
LOOKBACK = "2025-07-15"


def build_daily_panel(p15: pd.DataFrame, syms) -> pd.DataFrame:
    """(symbol, date) daily panel with multi-horizon fwd returns + the baseline features."""
    day = pd.to_datetime(p15.index.get_level_values(1)).normalize()
    g = p15.groupby([p15.index.get_level_values(0), day])
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()})
    d.index = d.index.set_names(["symbol", "timestamp"])
    sym = d.index.get_level_values(0)
    bysym = lambda c: d[c].groupby(level=0)
    cs = lambda s: s.groupby(s.index.get_level_values(1))
    ret = bysym("close").pct_change()
    d["ret"] = ret
    for h in HORIZONS:
        d[f"fut_ret_{h}"] = bysym("close").transform(lambda s: s.shift(-h) / s - 1.0)
    d["atr_pct"] = d["ret"].groupby(level=0).transform(lambda s: s.rolling(20).std())   # realized vol
    d["vol_factor"] = d["atr_pct"]
    d["idio_reversal"] = -(ret - cs(ret).transform("mean"))
    d["gap"] = d["open"] / bysym("close").shift(1) - 1.0
    d["abnormal_volume"] = d["volume"] / bysym("volume").transform(lambda s: s.rolling(20).mean())
    d["earnings_within_7d"] = _earnings_flag(d.index, syms, 7)
    return d


def _earnings_flag(idx, syms, horizon):
    earn = NF.load_earnings(syms)
    out = pd.Series(0.0, index=idx)
    s_arr = idx.get_level_values(0)
    dts = pd.to_datetime(idx.get_level_values(1)).to_numpy("datetime64[D]")
    for s in pd.unique(s_arr):
        ed = earn.get(s, np.array([], dtype="datetime64[ns]"))
        if ed.size == 0:
            continue
        ed = ed.astype("datetime64[D]")
        rows = np.where(s_arr == s)[0]
        for r in rows:
            fut = ed[ed >= dts[r]]
            if fut.size and (fut[0] - dts[r]) / np.timedelta64(1, "D") <= horizon:
                out.iloc[r] = 1.0
    return out


def main():
    p15 = pd.read_parquet(RL_ROOT / "data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    syms = sorted(p15.index.get_level_values(0).unique())
    d = build_daily_panel(p15, syms)
    print(f"daily panel {d.shape} | {len(syms)} symbols, {d.index.get_level_values(1).nunique()} days")

    sv = SI.fetch_short_volume(syms, LOOKBACK, "2026-06-18")
    si = SI.fetch_short_interest(syms, "2025-06-01", "2026-06-18")
    sf = SI.align(d.index, sv, si)
    # interaction tests (standalone candidates judged on their own merit)
    z = G._cs_z
    sf["si_dtc_x_earnings7"] = (z(sf["si_days_to_cover"]) * d["earnings_within_7d"]).fillna(0.0)
    sf["sv_ratio_x_vol"] = (z(sf["sv_ratio"]) * z(d["vol_factor"])).fillna(0.0)
    sf["sv_ratio_x_abnvol"] = (z(sf["sv_ratio"]) * z(d["abnormal_volume"])).fillna(0.0)
    sf["si_dtc_x_gap"] = (z(sf["si_days_to_cover"]) * z(d["gap"])).fillna(0.0)
    cand = sf[[c for c in sf.columns if sf[c].std() > 1e-9]]
    extra = d[["vol_factor", "idio_reversal", "earnings_within_7d"]]
    print(f"{cand.shape[1]} short candidates vs accepted leads [vol_factor, idio_reversal, earnings_within_7d]")

    # --- gate across horizons ---
    best = {}   # feature -> (horizon, record)
    per_h = {}
    for h in HORIZONS:
        rep = G.study(d, cand, accepted_nodes=None, accepted_extra=extra, cfg={"label": f"fut_ret_{h}"})
        per_h[h] = rep
        for r in rep["features"]:
            cur = best.get(r["feature"])
            if cur is None or abs(r["oos_ic"]) > abs(cur[1]["oos_ic"]):
                best[r["feature"]] = (h, r)

    # --- net-of-cost at each feature's best horizon ---
    verdict = []
    for feat, (h, r) in best.items():
        sig = (sf[feat] * r["sign"]).fillna(0.0)
        # net-of-cost P&L is ALWAYS daily mark-to-market (fut_ret_1); the horizon h only
        # says which forward window the signal best predicts, not the holding P&L. Using
        # fut_ret_{h} here would assign overlapping h-day returns to each day and inflate
        # Sharpe ~sqrt(h)x -- a methodology error, not alpha.
        g, t = LV.streams(d, sig, fwd="fut_ret_1")
        m = __import__("rl_meta.stream_validation", fromlist=["stream_metrics"]).stream_metrics(g.values, t.values)
        net5 = g.values - COST_BPS / 1e4 * t.values
        sh = lambda x: round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(252)), 2)
        from rl_meta.multiple_testing import wild_bootstrap_pvalue
        cap = LV.capacity_usd(d, sig)
        row = {"feature": feat, "best_horizon_d": h, "oos_ic": r["oos_ic"], "fdr": r["fdr_significant"],
               "vol_corr": r["vol_corr"], "rank_contrib": r["rank_contribution"],
               "gross_sharpe": sh(g.values), "net_sharpe_5bps": sh(net5),
               "turnover_day": round(float(t.mean()), 3), "breakeven_bps": m["breakeven_cost_bps"],
               "bootstrap_p": round(wild_bootstrap_pvalue(net5, n_boot=2000, block=1), 4),
               "corr_to_accepted": round(1 - r["rank_contribution"], 3),
               "capacity_usd": round(cap["median_usd"], 0)}
        row["verdict"] = _verdict(row)
        verdict.append(row)
    verdict.sort(key=lambda x: -x["net_sharpe_5bps"])

    out = Path(__file__).resolve().parent / "out"; out.mkdir(parents=True, exist_ok=True)
    rep_all = {"horizons": HORIZONS, "cost_bps": COST_BPS, "accepted_leads": list(extra.columns),
               "verdict_table": verdict,
               "data_note": "True short interest (bi-weekly days_to_cover, SI change) + daily short volume "
                            "via Massive. Borrow fee / utilization unavailable (proprietary lending data).",
               "per_horizon": {h: per_h[h]["features"] for h in HORIZONS}}
    pd.DataFrame(verdict).to_csv(out / "short_interest_gate_report.csv", index=False)
    (out / "short_interest_gate_report.json").write_text(json.dumps(rep_all, indent=2, default=str))
    acc = [v for v in verdict if v["verdict"] == "ACCEPT"]
    pd.DataFrame(acc).to_csv(out / "accepted_short_features.csv", index=False)
    pd.DataFrame([v for v in verdict if v["verdict"] != "ACCEPT"]).to_csv(out / "rejected_short_features.csv", index=False)
    _print(verdict, acc)
    print(f"\nartifacts -> {out}/ (short_interest_gate_report.csv/.json, accepted/rejected_short_features.csv)")


def _verdict(r):
    if not (r["oos_ic"] > 0 and r["fdr"]):
        return "REJECT"                                   # not OOS-positive / not significant
    if r["vol_corr"] >= 0.7 or r["rank_contrib"] < 0.5:
        return "REJECT"                                   # volatility/baseline in disguise
    if r["net_sharpe_5bps"] <= 0 or r["breakeven_bps"] <= COST_BPS:
        return "REJECT"                                   # dies on cost
    if r["bootstrap_p"] > 0.05:
        return "QUARANTINE"                               # net edge not significant
    if r["capacity_usd"] < 1_000_000:
        return "QUARANTINE"                               # too small to deploy
    return "ACCEPT"


def _print(verdict, acc):
    print(f"\n=== short interest / short volume -- daily, net of {COST_BPS}bps, vs the 3 validated leads ===\n")
    cols = ["feature", "best_horizon_d", "oos_ic", "fdr", "vol_corr", "rank_contrib",
            "gross_sharpe", "net_sharpe_5bps", "turnover_day", "breakeven_bps", "bootstrap_p",
            "capacity_usd", "verdict"]
    print("  " + " | ".join(c[:13] for c in cols))
    for v in verdict:
        print("  " + " | ".join(str(v[c])[:13].ljust(13) for c in cols))
    if acc:
        print(f"\n=== VERDICT: short-pressure ADDS deployable orthogonal alpha: "
              f"{', '.join(a['feature'] for a in acc)} ===")
    else:
        print(f"\n=== VERDICT: NO short-pressure feature is deployable beyond the validated leads "
              f"(net-of-cost, on this universe/horizon) ===")


if __name__ == "__main__":
    main()
