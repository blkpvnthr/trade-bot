#!/usr/bin/env python3
"""
tearsheet_insider.py — full performance tearsheet for the strategy run.py deploys:
vol-gated insider-buy SHARES (the validated live edge), plus the breadth>=2 amplifier overlay.

Builds a realistic portfolio equity curve (risk-based sizing: 1% equity risk / 10% stop = ~10% per
name, max 3 concurrent; 1% round-trip cost), then emits a metrics table + a multi-panel tearsheet PNG
(equity curve, underwater drawdown, monthly-returns heatmap, yearly bars, trade-return distribution,
rolling 63d Sharpe).
"""
from __future__ import annotations

import warnings
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

HOLD = 3
COST = 0.010
ALLOC = 0.10            # per-position fraction (1% risk / 10% stop)
MAX_OPEN = 3
TRAIN_END = _date(2024, 1, 1)
START_EQUITY = 10_000.0


def build_trades(breadth_min=1):
    ins = vg._collect(hold=HOLD, min_adv=1_000_000)
    vthr = np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80)
    trades = [r for r in ins if r["rv20"] >= vthr]
    if breadth_min >= 2:
        import json
        import study_insider_amplifiers as A
        rich = A._load_rich()
        out = []
        for r in trades:
            f = A._features(rich, r["sym"], r["date"])
            if f and f["breadth"] >= breadth_min:
                out.append(r)
        trades = out
    return sorted(trades, key=lambda r: r["date"])


def equity_curve(trades):
    """Risk-sized, max-3-concurrent portfolio. Returns a daily equity Series + per-trade net returns."""
    daily = {}                       # date -> portfolio return that day
    open_until = []                  # list of exit dates of open positions
    pt = []                          # per-trade net returns (taken)
    for r in trades:
        d0 = r["date"]
        open_until = [x for x in open_until if x > d0]      # free finished slots
        if len(open_until) >= MAX_OPEN:
            continue                                         # capacity full → skip signal
        net = r["gross"] - COST
        pt.append(net)
        per_day = (1 + net) ** (1 / HOLD) - 1
        for k in range(HOLD):
            dd = d0 + timedelta(days=k)
            daily[dd] = daily.get(dd, 0.0) + ALLOC * per_day
        open_until.append(d0 + timedelta(days=HOLD))
    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full, fill_value=0.0)
    eq = START_EQUITY * (1 + s).cumprod()
    return eq, s, np.array(pt)


def metrics(eq, ret, pt):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    act = ret[ret != 0]                              # active (deployed) days
    sharpe = (act.mean() / act.std() * np.sqrt(252)) if len(act) and act.std() > 0 else 0
    downside = act[act < 0]
    sortino = (act.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0
    w, l = pt[pt > 0], pt[pt < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else float("inf")
    return {
        "Period": f"{eq.index[0].date()} → {eq.index[-1].date()} ({yrs:.1f}y)",
        "Total return": f"{tot*100:+.1f}%",
        "CAGR": f"{cagr*100:+.1f}%",
        "Sharpe (ann, active)": f"{sharpe:.2f}",
        "Sortino (ann, active)": f"{sortino:.2f}",
        "Max drawdown": f"{maxdd*100:.1f}%",
        "Calmar": f"{(cagr/abs(maxdd)):.2f}" if maxdd < 0 else "—",
        "Trades taken": f"{len(pt)}",
        "Win rate": f"{(pt>0).mean()*100:.0f}%",
        "Profit factor": ("∞" if pf == float('inf') else f"{pf:.2f}"),
        "Avg trade (net)": f"{pt.mean()*100:+.2f}%",
        "Avg win / Avg loss": f"{w.mean()*100:+.2f}% / {l.mean()*100:+.2f}%" if len(l) else "—",
        "Best / Worst trade": f"{pt.max()*100:+.1f}% / {pt.min()*100:+.1f}%",
        "Capital deployed": f"≤{ALLOC*MAX_OPEN*100:.0f}% ({MAX_OPEN}×{ALLOC*100:.0f}%)",
    }


def monthly_table(ret):
    m = (1 + ret).resample("ME").prod() - 1
    df = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.values})
    return df.pivot_table(index="y", columns="m", values="r")


def main():
    base_t = build_trades(1)
    br_t = build_trades(2)
    eqB, retB, ptB = equity_curve(base_t)
    eqR, retR, ptR = equity_curve(br_t)
    mB, mR = metrics(eqB, retB, ptB), metrics(eqR, retR, ptR)

    # ---- metrics table (markdown + stdout) ----
    L = ["# Tearsheet — Vol-Gated Insider Shares (the run.py edge)", "",
         f"Risk-sized portfolio (≤{MAX_OPEN}×{ALLOC*100:.0f}%), {COST*1e4:.0f}bps round-trip, {HOLD}d hold. "
         "Two variants: base vol-gate (live) vs +breadth≥2 amplifier.", "",
         "| Metric | Vol-gated (base) | + Breadth≥2 |", "|---|---|---|"]
    for k in mB:
        L.append(f"| {k} | {mB[k]} | {mR[k]} |")
    out_md = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "tearsheet_insider.md").write_text(out_md)
    print(out_md)

    # ---- plots ----
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.38, wspace=0.22)
    fig.suptitle("Vol-Gated Insider Shares — Tearsheet (base vs +breadth≥2)", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(gs[0, :])
    ax.plot(eqB.index, eqB.values, label=f"Vol-gated  (CAGR {mB['CAGR']}, Sharpe {mB['Sharpe (ann, active)']})", lw=1.6)
    ax.plot(eqR.index, eqR.values, label=f"+Breadth≥2 (CAGR {mR['CAGR']}, Sharpe {mR['Sortino (ann, active)']})", lw=1.6, color="darkgreen")
    ax.set_yscale("log"); ax.set_title("Equity curve (log)"); ax.legend(loc="upper left"); ax.set_ylabel("$")

    ax = fig.add_subplot(gs[1, 0])
    ddB = (eqB / eqB.cummax() - 1) * 100
    ax.fill_between(ddB.index, ddB.values, 0, color="crimson", alpha=0.4)
    ax.set_title(f"Underwater / drawdown (base, max {mB['Max drawdown']})"); ax.set_ylabel("%")

    ax = fig.add_subplot(gs[1, 1])
    yB = (1 + retB).resample("YE").prod() - 1
    yR = (1 + retR).resample("YE").prod() - 1
    x = np.arange(len(yB))
    ax.bar(x - 0.2, yB.values * 100, 0.4, label="base")
    ax.bar(x + 0.2, yR.reindex(yB.index).values * 100, 0.4, label="+breadth", color="darkgreen")
    ax.set_xticks(x); ax.set_xticklabels([d.year for d in yB.index]); ax.set_title("Yearly return %"); ax.legend()

    ax = fig.add_subplot(gs[2, :])
    mt = monthly_table(retB) * 100
    sns.heatmap(mt, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "%"})
    ax.set_title("Monthly returns % (base)"); ax.set_xlabel("month"); ax.set_ylabel("")

    ax = fig.add_subplot(gs[3, 0])
    ax.hist(ptB * 100, bins=40, alpha=0.7, label="base", color="steelblue")
    ax.axvline(0, color="k", lw=0.8); ax.axvline(ptB.mean() * 100, color="blue", ls="--", label=f"mean {ptB.mean()*100:+.1f}%")
    ax.set_title("Per-trade net return distribution"); ax.set_xlabel("%"); ax.legend()

    ax = fig.add_subplot(gs[3, 1])
    rs = retB[retB != 0]
    roll = rs.rolling(63).apply(lambda v: v.mean() / v.std() * np.sqrt(252) if v.std() > 0 else 0)
    ax.plot(roll.index, roll.values, color="purple", lw=1.2)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="g", ls=":", lw=0.8)
    ax.set_title("Rolling Sharpe (63 active-day)"); ax.set_ylabel("Sharpe")

    out_png = base.PROJECT_ROOT / "backtest_outputs" / "tearsheet_insider.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nTearsheet PNG → {out_png}")


if __name__ == "__main__":
    main()
