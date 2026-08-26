#!/usr/bin/env python3
"""
Falsification harness: V2.9.8 bond-regime overlay ON vs OFF.
============================================================

The overlay was wired in by explicit user override, BUT the standing research
policy demands it prove itself. This script is the (minimum) evidence: it runs the
exact allocator with the bond-regime multipliers applied per day, and without
them, on the SAME price path, and compares net-of-cost, excess-return metrics.

    python backtest_bond_overlay.py --synthetic         # fast, offline, deterministic
    python backtest_bond_overlay.py --prices bonds.csv  # real cached data (10+SHY/LQD/HYG+CASH_RATE_3M)

A SYNTHETIC run is a mechanism/directional check only -- it is NOT validation.
Real-data ACCEPT requires this to also survive walk-forward, holdout, placebo and
bootstrap per event_study.py / feature_gate.py. The overlay stays OFF by default
until then.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from strat_allocator import AllocatorConfig, TRADING_DAYS, annualized_metrics
from strat import build_gate_schedule, momentum_view_panel, rolling_ic_panel, run_backtest
from bond_regime import BondOverlayConfig, regime_frame

START = "2007-01-02"
BOND_ASSETS = ("SHY", "IEF", "TLT", "LQD", "HYG")


def _load(args) -> pd.DataFrame:
    if args.prices:
        return pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index().dropna()
    if args.synthetic:
        sys.path.insert(0, "tests")
        from strat_fixtures import make_synthetic_prices
        return make_synthetic_prices()
    # real data via yfinance (research-only dependency)
    import yfinance as yf
    uni = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ",
           "SHY", "LQD", "HYG"]
    px = yf.download(uni, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    cash = yf.download("^IRX", start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    cash = cash.iloc[:, 0] if isinstance(cash, pd.DataFrame) else cash
    out = px[uni].copy()
    out["CASH_RATE_3M"] = cash.reindex(out.index).ffill()
    return out.dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    prices = _load(args)
    cfg = AllocatorConfig()
    bcfg = BondOverlayConfig(enabled=True)
    missing = [a for a in BOND_ASSETS if a not in prices.columns]
    if missing:
        print(f"ERROR: bond ETFs missing from prices: {missing}", file=sys.stderr)
        return 1

    rf = prices["CASH_RATE_3M"].astype(float) / 100.0 / TRADING_DAYS
    gate = build_gate_schedule(prices["SPY"], cfg)
    view = momentum_view_panel(prices, cfg)
    ic = rolling_ic_panel(view, prices)

    frame = regime_frame(prices[list(BOND_ASSETS)], bcfg)
    vol_mult = frame["vol_budget_mult"]
    growth_mult = frame["growth_ceiling_mult"]

    base_h, _ = run_backtest(prices, cfg, gate, view=view, ic=ic, start=args.start,
                             label="OFF")
    ov_h, _ = run_backtest(prices, cfg, gate, view=view, ic=ic, start=args.start,
                           label="ON", vol_mult=vol_mult, growth_mult=growth_mult)

    mb = annualized_metrics(base_h["Equity"], rf, base_h["Turnover"])
    mo = annualized_metrics(ov_h["Equity"], rf, ov_h["Turnover"])

    print("=" * 78)
    print("BOND-REGIME OVERLAY FALSIFICATION  (excess-return, net of cost)")
    print("=" * 78)
    print(f"Data            : {'SYNTHETIC (mechanism check, NOT validation)' if args.synthetic else prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Overlay mult    : vol in [{vol_mult.min():.2f},{vol_mult.max():.2f}]  "
          f"growth in [{growth_mult.min():.2f},{growth_mult.max():.2f}]")
    print()
    hdr = f"{'Variant':<10}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'Calmar':>9}{'AnnTurn':>9}"
    print(hdr)
    for name, m in (("OVERLAY OFF", mb), ("OVERLAY ON", mo)):
        print(f"{name:<10}{m['cagr']:>9.2%}{m['vol']:>9.2%}{m['sharpe']:>9.2f}"
              f"{m['sortino']:>9.2f}{m['max_drawdown']:>9.2%}{m['calmar']:>9.2f}"
              f"{m['annual_turnover']:>9.1%}")
    print()
    dS = mo["sharpe"] - mb["sharpe"]
    dC = mo["cagr"] - mb["cagr"]
    dD = mo["max_drawdown"] - mb["max_drawdown"]
    print(f"delta (ON-OFF)  : Sharpe {dS:+.3f}   CAGR {dC:+.2%}   MaxDD {dD:+.2%}")
    verdict = ("IMPROVES net Sharpe" if dS > 0.02 else
               "NO Sharpe improvement" if dS <= 0.0 else "marginal")
    print(f"VERDICT (this path): {verdict}. "
          f"{'Still OWES walk-forward/holdout/placebo/bootstrap before deploy.' if not args.synthetic else 'SYNTHETIC -- not evidence for deployment.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
