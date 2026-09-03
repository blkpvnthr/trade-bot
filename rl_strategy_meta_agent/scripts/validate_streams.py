#!/usr/bin/env python3
"""Validation sprint for the new-stack return streams.

Answers the decisive question: does each stream (alpha_mined_ensemble, ORB_relvol,
giveback_exit_model) produce a statistically significant, COST-ADJUSTED edge that
survives OUT-OF-TIME? Derives per-stream turnover + train/OOS IC, then promotes a
stream only if it clears every gate (net expectancy>0, net Sharpe>=thr, breakeven
cost>=10bps, OOS expectancy>0, IC decay acceptable). Emits explicit fail reasons.

Research-only: no broker/order/allocator-execution/live path.

    python scripts/validate_streams.py --cost-bps 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import strategy_streams as ss, stream_validation as sv
from rl_meta.alphas import eval_alpha, spearman_ic
from rl_meta.bar_coverage import cached_bars


def _to_node(x):
    return tuple(_to_node(e) for e in x) if isinstance(x, list) else x


def _et_day(idx):
    return pd.to_datetime(idx, utc=True).tz_convert("America/New_York").normalize()


def _weight_turnover(w):
    """Per-bar cross-sectional turnover of weight series w, summed to ET day."""
    dw = w.groupby(level=0).diff().abs()
    per_ts = dw.groupby(level=1).sum() / (w.abs().groupby(level=1).sum() + 1e-12)
    return per_ts.groupby(_et_day(per_ts.index)).sum()


def _spy_regime(dates, bars_dir):
    spy = cached_bars("SPY", "15Min", bars_dir)
    if spy.empty:
        return None
    s = spy.set_index(pd.to_datetime(spy["timestamp"], utc=True))["close"].resample("1D").last().dropna()
    s.index = s.index.tz_convert("America/New_York").normalize()
    reg = s.pct_change(20).reindex(dates, method="ffill")
    return np.where(reg > 0.01, "bull", np.where(reg < -0.01, "bear", "chop"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    ap.add_argument("--nodes", default="data/trade_outcomes/accepted_alpha_nodes.json")
    ap.add_argument("--streams", default="data/trade_outcomes/strategy_streams.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--min-net-sharpe", type=float, default=0.5)
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()
    cfg = {"cost_bps": args.cost_bps, "min_net_sharpe": args.min_net_sharpe}

    panel = pd.read_parquet(args.panel)
    nodes = [_to_node(d["node"]) for d in json.loads(Path(args.nodes).read_text())]
    streams = pd.read_parquet(args.streams)
    s = ss.composite_signal(panel, [eval_alpha(n, panel) for n in nodes])   # giveback signal
    fut = panel["fut_ret_1"]
    ts = panel.index.get_level_values(1)
    cut = pd.Series(ts).quantile(0.7)                   # keep same tz-awareness as ts
    tr, oo = ts < cut, ts >= cut
    cut_utc = cut.tz_localize("UTC") if cut.tzinfo is None else cut
    cut_day = cut_utc.tz_convert("America/New_York").normalize()
    print(f"panel {panel.shape} | nodes {len(nodes)} | streams {list(streams.columns)} | "
          f"train/OOS cut {cut_day.date()}")

    # per-stream daily turnover + IC of the driving signal
    w_ls = ss._cs_z(-s)                                  # alpha_mined_ensemble weights
    w_ex = 1.0 - ss._cs_rank(s)                          # giveback exposure weights
    turn = {
        "alpha_mined_ensemble": _weight_turnover(w_ls),
        "giveback_exit_model": _weight_turnover(w_ex),
        "ORB_relvol": pd.Series(2.0, index=streams["ORB_relvol"].dropna().index),  # enter+exit
    }
    ic = {}
    for name in ("alpha_mined_ensemble", "giveback_exit_model"):
        ic[name] = {"train": spearman_ic(-s[tr], fut[tr])[0], "oos": spearman_ic(-s[oo], fut[oo])[0]}

    regime = _spy_regime(streams.index, Path(args.bars_dir))
    rep = {"cost_bps": args.cost_bps, "train_oos_cut": str(cut_day.date()), "streams": {}}
    for name in ("alpha_mined_ensemble", "ORB_relvol", "giveback_exit_model"):
        ret = streams[name].dropna()
        tn = turn[name].reindex(ret.index).fillna(turn[name].mean())
        oos_mask = ret.index >= cut_day
        tic = ic.get(name, {}).get("train"); oic = ic.get(name, {}).get("oos")
        m = sv.validate_stream(ret.values, tn.values, train_ic=tic, oos_ic=oic,
                               oos_ret=ret[oos_mask].values, oos_turn=tn[oos_mask].values, cfg=cfg)
        if regime is not None:
            net = ret.values - args.cost_bps / 1e4 * tn.values
            m["by_regime_net_mean"] = {r: round(float(net[regime == r].mean()), 5)
                                       for r in ("bull", "bear", "chop") if (regime == r).any()}
        rep["streams"][name] = m

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "stream_validation_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _write_md(rep, out / "stream_validation.md")
    _print(rep)
    print(f"\nartifacts -> {out}/ (stream_validation_report.json, stream_validation.md)")


def _print(rep):
    print(f"\n=== stream validation sprint (cost {rep['cost_bps']}bps) ===")
    h = f"{'stream':<22}{'grossW':>8}{'netW':>8}{'netShrp':>8}{'Sortino':>8}{'maxDD':>7}{'turn':>7}{'beBps':>7}{'PROMO':>7}"
    print(h); print("-" * len(h))
    for name, m in rep["streams"].items():
        be = m["breakeven_cost_bps"]
        print(f"{name:<22}{m['gross_cum_return']*100:>7.0f}%{m['net_cum_return']*100:>7.0f}%"
              f"{m['net_sharpe']:>8.2f}{m['net_sortino']:>8.2f}{m['net_max_drawdown']:>7.2f}"
              f"{m['mean_turnover']:>7.2f}{be if be != float('inf') else 999:>7.1f}"
              f"{'  Y' if m['promoted'] else '  -':>7}")
    print()
    for name, m in rep["streams"].items():
        tag = "PROMOTED" if m["promoted"] else "REJECTED: " + "; ".join(m["reasons"])
        ictxt = ""
        if m.get("train_ic") is not None:
            ictxt = f" | IC train {m['train_ic']:+.3f} oos {m['oos_ic']:+.3f}"
        print(f"  {name}: {tag}{ictxt}")


def _write_md(rep, path):
    lines = [f"# Stream validation sprint (cost {rep['cost_bps']}bps, OOS cut {rep['train_oos_cut']})", "",
             "| stream | grossCum | netCum | netSharpe | Sortino | maxDD | turnover | breakevenBps | promoted |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in rep["streams"].items():
        be = m["breakeven_cost_bps"]
        lines.append(f"| {name} | {m['gross_cum_return']*100:.0f}% | {m['net_cum_return']*100:.0f}% | "
                     f"{m['net_sharpe']} | {m['net_sortino']} | {m['net_max_drawdown']} | "
                     f"{m['mean_turnover']} | {be if be != float('inf') else 'inf'} | "
                     f"{'YES' if m['promoted'] else 'no'} |")
    lines += ["", "## Verdict per stream"]
    for name, m in rep["streams"].items():
        v = "PROMOTED" if m["promoted"] else "REJECTED -- " + "; ".join(m["reasons"])
        lines.append(f"- **{name}**: {v}")
    lines += ["", "## Read", "A stream is only a deployable allocator input if it clears every gate "
              "net of costs and out-of-time. Caveat: for the alpha-derived streams the window "
              "overlaps the alpha-mining period, so a clean OOS test requires walk-forward mining "
              "(mine earlier, test on never-seen-future). Research-only; not a deployment decision."]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
