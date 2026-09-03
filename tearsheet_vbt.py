#!/usr/bin/env python3
"""
tearsheet_vbt.py — vectorbt tearsheet for the run.py edge (vol-gated insider SHARES).

Builds a multi-asset close panel + entry/exit signal panels from the validated vol-gated insider
trades, runs vbt.Portfolio.from_signals (shared cash, 10%/name, 1% fees+slippage, 3-bar hold), and
emits vectorbt's native pf.stats() table + the pf.plot() tearsheet PNG. Optional breadth>=2 variant.
"""
from __future__ import annotations

import warnings
from datetime import date as _date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import vectorbt as vbt

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

HOLD = 3
TRAIN_END = _date(2024, 1, 1)


def gated_trades(breadth_min=1):
    ins = vg._collect(hold=HOLD, min_adv=1_000_000)
    vthr = np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80)
    tr = [r for r in ins if r["rv20"] >= vthr]
    if breadth_min >= 2:
        import study_insider_amplifiers as A
        rich = A._load_rich()
        tr = [r for r in tr if (f := A._features(rich, r["sym"], r["date"])) and f["breadth"] >= breadth_min]
    return tr


def build_panels(trades):
    syms = sorted({r["sym"] for r in trades})
    closes, opens = {}, {}
    for s in syms:
        try:
            b = pd.read_parquet(base.BARS_DIR / f"{s}_day.parquet")
        except Exception:
            continue
        b.index = pd.to_datetime([d.date() if hasattr(d, "date") else d for d in b.index])
        closes[s] = b["close"].astype(float)
        opens[s] = b["open"].astype(float)
    close = pd.DataFrame(closes).sort_index()
    open_ = pd.DataFrame(opens).reindex(close.index)
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    price = close.copy()                       # exits execute at close…
    for r in trades:
        s = r["sym"]
        if s not in close.columns:
            continue
        ed = pd.Timestamp(r["date"])
        if ed not in close.index:
            continue
        pos = close.index.get_loc(ed)
        if pos + HOLD >= len(close.index):
            continue
        ci = close.columns.get_loc(s)
        entries.iloc[pos, ci] = True
        exits.iloc[pos + HOLD, ci] = True
        price.iloc[pos, ci] = open_.iloc[pos, ci]   # …entries execute at the OPEN (validated timing)
    return close, entries, exits, price


def run(trades, label):
    close, entries, exits, price = build_panels(trades)
    pf = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits, price=price,
        size=0.10, size_type="percent",
        fees=0.005, slippage=0.0,              # 0.5%/side = 1% round-trip (project-validated cost)
        init_cash=10_000, cash_sharing=True, group_by=True, freq="1D",
        call_seq="auto",
    )
    return pf


def main():
    base_tr = gated_trades(1)
    print(f"vol-gated trades: {len(base_tr)}")
    pf = run(base_tr, "vol-gated")
    st = pf.stats()
    print("\n=== vectorbt Portfolio.stats() — vol-gated insider shares ===")
    print(st.to_string())
    (base.PROJECT_ROOT / "backtest_outputs" / "tearsheet_vbt_stats.md").write_text(
        "# vectorbt stats — vol-gated insider shares (run.py edge)\n\n```\n" + st.to_string() + "\n```\n")

    # breadth>=2 variant
    br = gated_trades(2)
    pfb = run(br, "breadth>=2")
    sb = pfb.stats()

    # ---- tearsheet PNG from vbt-computed series (matplotlib; robust) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    eqB, eqR = pf.value(), pfb.value()
    ddB = (eqB / eqB.cummax() - 1) * 100
    trB = np.asarray(pf.trades.returns.values) * 100
    fig = plt.figure(figsize=(15, 13)); gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.2)
    fig.suptitle("vectorbt — Vol-Gated Insider Shares (run.py edge)", fontsize=15, fontweight="bold")
    ax = fig.add_subplot(gs[0, :])
    ax.plot(eqB.index, eqB.values, lw=1.5, label=f"vol-gated (ret {st['Total Return [%]']:.1f}%, Sharpe {st['Sharpe Ratio']:.2f})")
    ax.plot(eqR.index, eqR.values, lw=1.5, color="darkgreen", label=f"+breadth≥2 (ret {sb['Total Return [%]']:.1f}%, Sharpe {sb['Sharpe Ratio']:.2f})")
    ax.set_title("Portfolio value (vbt cash-sharing, 10%/name, 1% round-trip)"); ax.legend(loc="upper left"); ax.set_ylabel("$")
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(ddB.index, ddB.values, 0, color="crimson", alpha=0.4)
    ax.set_title(f"Drawdown (max {st['Max Drawdown [%]']:.1f}%)"); ax.set_ylabel("%")
    ax = fig.add_subplot(gs[1, 1])
    retsB = pf.returns(); yr = (1 + retsB).resample("YE").prod() - 1
    ax.bar([d.year for d in yr.index], yr.values * 100, color="steelblue")
    ax.axhline(0, color="k", lw=0.6); ax.set_title("Yearly return %")
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(trB, bins=40, color="steelblue", alpha=0.75); ax.axvline(0, color="k", lw=0.8)
    ax.axvline(trB.mean(), color="blue", ls="--", label=f"mean {trB.mean():+.1f}%")
    ax.set_title("Per-trade return distribution"); ax.set_xlabel("%"); ax.legend()
    ax = fig.add_subplot(gs[2, 1])
    roll = retsB.rolling(63).apply(lambda v: v.mean() / v.std() * np.sqrt(252) if v.std() > 0 else 0)
    ax.plot(roll.index, roll.values, color="purple", lw=1.1); ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="g", ls=":", lw=0.8)
    ax.set_title("Rolling Sharpe (63d)"); ax.set_ylabel("Sharpe")
    out_png = base.PROJECT_ROOT / "backtest_outputs" / "tearsheet_vbt.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nTearsheet PNG → {out_png}")

    print(f"\n=== base vs breadth>=2 (vbt, 1% round-trip, open-entry) ===")
    for k in ["Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio", "Sortino Ratio",
              "Win Rate [%]", "Profit Factor", "Calmar Ratio", "Expectancy"]:
        if k in st.index and k in sb.index:
            print(f"  {k:20}  base={float(st[k]):>8.2f}   breadth>=2={float(sb[k]):>8.2f}")


if __name__ == "__main__":
    main()
