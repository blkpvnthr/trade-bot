#!/usr/bin/env python3
"""Extended-history validation of the quarantined signals on a multi-year DAILY panel.

Resolves the three quarantined signals (earnings_within_7d, sv_ratio, sv_ratio x vol)
plus the idio_reversal baseline on a decade of daily data (Alpaca 2016+; short-volume
2024-02+). Focused funnel, no new channels: walk-forward by year, symbol holdout,
bootstrap, placebo labels, FDR, net-of-cost daily mark-to-market P&L, breakeven, capacity.

Decision: ACCEPT only if bootstrap + FDR pass net-of-cost; QUARANTINE if positive but
underpowered; REJECT if OOS decays or costs kill it. Research only.

    python experiments/orthogonal_features/run_extended_validation.py
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

from rl_meta.alphas import spearman_ic
from rl_meta.multiple_testing import wild_bootstrap_pvalue, fdr_control
import feature_gate as G
import daily_bars as DB
import short_interest_features as SI
import news_features as NF
import run_lead_validation as LV

SYMS = ['AAPL','AMSC','AMZN','ARM','AVGO','CE','ENSG','GOOG','GOOGL','GWRE','IREN','LULU','META',
        'MSTR','NVDA','ORCL','QBTS','RYAN','SMCI','TSLA','TSM','UNH','WGS','WING']
COST_BPS = 5.0
HOR = {"idio_reversal": 1, "earnings_within_7d": 5, "sv_ratio": 20, "sv_ratio_x_vol": 5}


def build_panel():
    b = DB.fetch(SYMS, "2016-01-01", "2026-06-18")
    sym = b.index.get_level_values(0)
    g = lambda c: b[c].groupby(level=0)
    cs = lambda s: s.groupby(s.index.get_level_values(1))
    ret = g("close").pct_change()
    d = b.copy()
    d["ret"] = ret
    for h in (1, 5, 20):
        d[f"fut_ret_{h}"] = g("close").transform(lambda s: s.shift(-h) / s - 1.0)
    d["atr_pct"] = ret.groupby(level=0).transform(lambda s: s.rolling(20).std())
    d["vol_factor"] = d["atr_pct"]
    d["idio_reversal"] = -(ret - cs(ret).transform("mean"))
    d["abnormal_volume"] = d["volume"] / g("volume").transform(lambda s: s.rolling(20).mean())
    d["earnings_within_7d"] = _earn_flag(d.index)
    return d


def _earn_flag(idx, horizon=7):
    earn = NF.load_earnings(SYMS)
    out = pd.Series(0.0, index=idx)
    sa = idx.get_level_values(0); dts = pd.to_datetime(idx.get_level_values(1)).to_numpy("datetime64[D]")
    for s in pd.unique(sa):
        ed = earn.get(s, np.array([], dtype="datetime64[ns]"))
        if ed.size == 0:
            continue
        ed = ed.astype("datetime64[D]"); rows = np.where(sa == s)[0]
        for r in rows:
            fut = ed[ed >= dts[r]]
            if fut.size and (fut[0] - dts[r]) / np.timedelta64(1, "D") <= horizon:
                out.iloc[r] = 1.0
    return out


def _net_sharpe(g, t, cost=COST_BPS):
    net = g.values - cost / 1e4 * t.values
    return round(float(np.mean(net) / (np.std(net) + 1e-12) * np.sqrt(252)), 2)


def _walk_forward(panel, sig, mask):
    """Per-year net Sharpe of the daily stream; fraction of positive years."""
    sub = panel[mask]; s = sig[mask]
    yrs = pd.to_datetime(sub.index.get_level_values(1)).year
    out = {}
    for y in sorted(np.unique(yrs)):
        m = yrs == y
        if m.sum() < 200:
            continue
        gg, tt = LV.streams(sub[m], s[m], fwd="fut_ret_1")
        if len(gg) > 20:
            out[int(y)] = _net_sharpe(gg, tt)
    pos = sum(v > 0 for v in out.values()) / max(len(out), 1)
    return out, round(pos, 2)


def _symbol_holdout(panel, sig, mask, nat_h):
    """Direction fit on symbol-half A; net Sharpe measured on held-out half B."""
    sub = panel[mask]; s = sig[mask]
    syms = np.array(sorted(sub.index.get_level_values(0).unique()))
    A, B = set(syms[::2]), set(syms[1::2])
    sa = sub.index.get_level_values(0)
    ma, mb = np.isin(sa, list(A)), np.isin(sa, list(B))
    sign = np.sign(spearman_ic(s[ma], sub[f"fut_ret_{nat_h}"][ma])[0]) or 1.0
    gg, tt = LV.streams(sub[mb], s[mb] * sign, fwd="fut_ret_1")
    return _net_sharpe(gg, tt), round(float(spearman_ic((s[mb] * sign), sub["fut_ret_1"][mb])[0]), 4)


def _placebo_p(panel, sig, mask, real_gross_sharpe, n=120):
    """Shuffle forward returns within each day; fraction of placebo gross Sharpe >= real."""
    sub = panel[mask]; w = G._cs_z(sig[mask]).fillna(0.0)
    ts = w.index.get_level_values(1)
    wn = (w / (w.abs().groupby(ts).transform("sum") + 1e-12))
    r = sub["fut_ret_1"].fillna(0.0)
    rng = np.random.default_rng(0); ge = 0
    for _ in range(n):
        rp = r.groupby(ts).transform(lambda x: rng.permutation(x.values))
        daily = (wn * rp).groupby(ts).sum()
        sh = float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(252))
        if sh >= real_gross_sharpe:
            ge += 1
    return round(ge / n, 4)


def evaluate(panel, name, sig):
    nat_h = HOR[name]
    mask = (sig.abs() > 1e-12).to_numpy() if name.startswith("sv_") else np.ones(len(sig), bool)
    sub = panel[mask]
    sign = 1.0 if spearman_ic(sig[mask], sub[f"fut_ret_{nat_h}"])[0] >= 0 else -1.0
    s = (sig * sign)
    ic = spearman_ic(s[mask], sub[f"fut_ret_{nat_h}"])[0]
    pic = wild_bootstrap_pvalue(_ics(s[mask], sub[f"fut_ret_{nat_h}"]), 2000, block=8)
    g, t = LV.streams(sub, s[mask], fwd="fut_ret_1")
    gross = round(float(g.mean() / (g.std() + 1e-12) * np.sqrt(252)), 2)
    net = _net_sharpe(g, t)
    be = __import__("rl_meta.stream_validation", fromlist=["stream_metrics"]).stream_metrics(g.values, t.values)["breakeven_cost_bps"]
    boot = round(wild_bootstrap_pvalue(g.values - COST_BPS / 1e4 * t.values, 2000, block=1), 4)
    wf, pos = _walk_forward(panel, s, mask)
    ho_sh, ho_ic = _symbol_holdout(panel, s, mask, nat_h)
    plac = _placebo_p(panel, s, mask, gross)
    vc = abs(np.corrcoef(G._cs_z(s[mask]).fillna(0).values, G._cs_z(sub["vol_factor"]).fillna(0).values)[0, 1])
    return {"signal": name, "n_obs": int(mask.sum()), "natural_horizon_d": nat_h,
            "ic": round(float(ic), 4), "ic_bootstrap_p": round(pic, 4),
            "vol_corr": round(float(vc), 3), "gross_sharpe": gross, "net_sharpe_5bps": net,
            "turnover_day": round(float(t.mean()), 3), "breakeven_bps": be,
            "net_bootstrap_p": boot, "placebo_p": plac,
            "wf_positive_year_frac": pos, "wf_by_year": wf,
            "holdout_net_sharpe": ho_sh, "holdout_ic": ho_ic}


def _ics(a, b):
    from rl_meta.alphas import ic_series
    return ic_series(a, b)


def decide(r, fdr_ok):
    if r["net_sharpe_5bps"] <= 0 or r["breakeven_bps"] <= COST_BPS:
        return "REJECT"                                          # costs kill it
    if r["vol_corr"] >= 0.7:
        return "REJECT"                                          # volatility in disguise
    strong = (r["net_bootstrap_p"] < 0.05 and fdr_ok and r["placebo_p"] < 0.05
              and r["holdout_net_sharpe"] > 0 and r["wf_positive_year_frac"] >= 0.6)
    if strong:
        return "ACCEPT"
    if r["net_sharpe_5bps"] > 0 and r["placebo_p"] < 0.2:
        return "QUARANTINE"                                      # positive but underpowered/unstable
    return "REJECT"


def main():
    d = build_panel()
    print(f"daily panel {d.shape} | {len(SYMS)} symbols | "
          f"{pd.to_datetime(d.index.get_level_values(1)).year.min()}-"
          f"{pd.to_datetime(d.index.get_level_values(1)).year.max()}")
    sv = SI.fetch_short_volume(SYMS, "2024-01-15", "2026-06-18")
    si = SI.fetch_short_interest(SYMS, "2017-01-01", "2026-06-18")
    sf = SI.align(d.index, sv, si)
    sigs = {"idio_reversal": d["idio_reversal"], "earnings_within_7d": d["earnings_within_7d"],
            "sv_ratio": sf["sv_ratio"], "sv_ratio_x_vol": (G._cs_z(sf["sv_ratio"]) * G._cs_z(d["vol_factor"]))}

    recs = [evaluate(d, n, s.fillna(0.0)) for n, s in sigs.items()]
    fdr = fdr_control([r["net_bootstrap_p"] for r in recs], q=0.10, n_tests=len(recs))
    for r, ok in zip(recs, fdr["discoveries"]):
        r["fdr_significant"] = bool(ok); r["decision"] = decide(r, bool(ok))

    out = Path(__file__).resolve().parent / "out"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "wf_by_year"} for r in recs]).to_csv(
        out / "extended_validation.csv", index=False)
    (out / "extended_validation.json").write_text(json.dumps(recs, indent=2, default=str))
    _print(recs)
    print(f"\nartifacts -> {out}/ (extended_validation.csv/.json)")


def _print(recs):
    print(f"\n=== EXTENDED-HISTORY validation (daily, net {COST_BPS}bps) ===\n")
    cols = ["signal", "n_obs", "ic", "vol_corr", "gross_sharpe", "net_sharpe_5bps", "turnover_day",
            "breakeven_bps", "net_bootstrap_p", "placebo_p", "fdr_significant",
            "wf_positive_year_frac", "holdout_net_sharpe", "decision"]
    print("  " + " | ".join(c[:11] for c in cols))
    for r in recs:
        print("  " + " | ".join(str(r[c])[:11].ljust(11) for c in cols))
    for r in recs:
        print(f"\n  {r['signal']} walk-forward by year: {r['wf_by_year']}")
    acc = [r["signal"] for r in recs if r["decision"] == "ACCEPT"]
    print(f"\n=== VERDICT: {'DEPLOYABLE: '+', '.join(acc) if acc else 'still none ACCEPT '+ '('+', '.join(r['signal']+':'+r['decision'] for r in recs)+')'} ===")


if __name__ == "__main__":
    main()
