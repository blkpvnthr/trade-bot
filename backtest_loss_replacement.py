#!/usr/bin/env python3
"""
Falsification harness: V2.9.8 loss-replacement / momentum-rotation overlay ON vs OFF.
=====================================================================================

Runs the EXACT allocator on one price path twice -- overlay OFF (the frozen
strategy) and overlay ON (exit any holding > loss_replacement_threshold below its
cost basis, rotate its allocation into the strongest positive-momentum eligible
asset) -- and compares net-of-cost, excess-return metrics on identical universe /
dates / capital / fees. The purpose is to determine whether a -0.50% replacement
threshold IMPROVES the strategy or merely increases churn.

    python backtest_loss_replacement.py --synthetic          # fast, offline, deterministic
    python backtest_loss_replacement.py --prices px.csv       # cached (10 ETFs + CASH_RATE_3M)
    python backtest_loss_replacement.py                        # real data via yfinance

The threshold is NOT tuned on the evaluation period. A synthetic run is a
mechanism check only, NOT validation. Like every overlay in this repo, it ships
OFF (AllocatorConfig default) until it clears real-data walk-forward / holdout /
placebo / bootstrap. This script reports the ON-vs-OFF evidence; it does not
claim the overlay is validated.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import pandas as pd

from strat_allocator import AllocatorConfig, TRADING_DAYS, annualized_metrics
from strat import build_gate_schedule, momentum_view_panel, rolling_ic_panel, run_backtest

START = "2007-01-02"
UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ"]


def _load(args) -> pd.DataFrame:
    if args.prices:
        return pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index().dropna()
    if args.synthetic:
        sys.path.insert(0, "tests")
        from strat_fixtures import make_synthetic_prices
        return make_synthetic_prices()
    import yfinance as yf   # research-only dependency
    px = yf.download(UNIVERSE, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    cash = yf.download("^IRX", start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    cash = cash.iloc[:, 0] if isinstance(cash, pd.DataFrame) else cash
    out = px[UNIVERSE].copy()
    out["CASH_RATE_3M"] = cash.reindex(out.index).ffill()
    return out.dropna()


def _cost_drag(h, cfg) -> float:
    """Cumulative transaction-cost fraction implied by the daily turnover series."""
    return float((h["Turnover"] * cfg.transaction_cost * 2.0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override loss_replacement_threshold (default -0.005)")
    ap.add_argument("--lookback", type=int, default=None,
                    help="override replacement_momentum_lookback (default 20)")
    ap.add_argument("--cooldown", type=int, default=None,
                    help="override replacement_cooldown_days (default 5)")
    args = ap.parse_args()

    prices = _load(args)
    missing = [a for a in UNIVERSE if a not in prices.columns] + (
        ["CASH_RATE_3M"] if "CASH_RATE_3M" not in prices.columns else [])
    if missing:
        print(f"ERROR: prices missing columns: {missing}", file=sys.stderr)
        return 1

    cfg_off = AllocatorConfig()
    on_kw = {"loss_replacement_enabled": True}
    if args.threshold is not None:
        on_kw["loss_replacement_threshold"] = args.threshold
    if args.lookback is not None:
        on_kw["replacement_momentum_lookback"] = args.lookback
    if args.cooldown is not None:
        on_kw["replacement_cooldown_days"] = args.cooldown
    cfg_on = replace(cfg_off, **on_kw)

    rf = prices["CASH_RATE_3M"].astype(float) / 100.0 / TRADING_DAYS
    # gate/view/ic depend only on cfg fields shared by both configs -> compute once.
    gate = build_gate_schedule(prices["SPY"], cfg_off)
    view = momentum_view_panel(prices, cfg_off)
    ic = rolling_ic_panel(view, prices)

    off_h, _ = run_backtest(prices, cfg_off, gate, view=view, ic=ic, start=args.start,
                            label="OFF")
    on_h, _ = run_backtest(prices, cfg_on, gate, view=view, ic=ic, start=args.start,
                           label="ON")

    mo_off = annualized_metrics(off_h["Equity"], rf, off_h["Turnover"])
    mo_on = annualized_metrics(on_h["Equity"], rf, on_h["Turnover"])

    lr = on_h.attrs.get("loss_replacement", {})
    reason = on_h.get("RebalanceReason")
    normal_reb_off = int(off_h["Rebalanced"].sum())
    normal_reb_on = int(((reason == "scheduled") & on_h["Rebalanced"]).sum())
    loss_reb_on = int((reason == "loss_replacement").sum())

    print("=" * 82)
    print("LOSS-REPLACEMENT OVERLAY FALSIFICATION  (excess-return, net of cost)")
    print("=" * 82)
    tag = ("SYNTHETIC (mechanism check, NOT validation)" if args.synthetic
           else f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Data            : {tag}")
    print(f"Overlay params  : threshold {cfg_on.loss_replacement_threshold:.3%}  "
          f"lookback {cfg_on.replacement_momentum_lookback}  "
          f"cooldown {cfg_on.replacement_cooldown_days}d")
    print()
    hdr = (f"{'Variant':<8}{'FinalVal':>13}{'TotRet':>9}{'CAGR':>8}{'MaxDD':>8}"
           f"{'Sharpe':>8}{'Sortino':>8}{'AnnTurn':>9}{'CostDrag':>9}")
    print(hdr)
    for name, m, h in (("OFF", mo_off, off_h), ("ON", mo_on, on_h)):
        cfg = cfg_off if name == "OFF" else cfg_on
        print(f"{name:<8}{m['final_equity']:>13,.0f}{m['return']:>9.1%}{m['cagr']:>8.2%}"
              f"{m['max_drawdown']:>8.2%}{m['sharpe']:>8.2f}{m['sortino']:>8.2f}"
              f"{m['annual_turnover']:>9.1%}{_cost_drag(h, cfg):>9.2%}")
    print()
    print(f"Rebalances      : OFF {normal_reb_off} scheduled | "
          f"ON {normal_reb_on} scheduled + {loss_reb_on} loss-triggered")
    print(f"Overlay actions : exits {lr.get('exits', 0)}  "
          f"replacements {lr.get('replacements', 0)}  "
          f"no-candidate fallbacks {lr.get('no_candidate', 0)}")
    print()
    dS = mo_on["sharpe"] - mo_off["sharpe"]
    dC = mo_on["cagr"] - mo_off["cagr"]
    dD = mo_on["max_drawdown"] - mo_off["max_drawdown"]
    dT = mo_on["annual_turnover"] - mo_off["annual_turnover"]
    print(f"delta (ON-OFF)  : Sharpe {dS:+.3f}   CAGR {dC:+.2%}   "
          f"MaxDD {dD:+.2%}   AnnTurn {dT:+.1%}")
    verdict = ("IMPROVES net Sharpe" if dS > 0.02 else
               "NO Sharpe improvement (churn without edge)" if dS <= 0.0 else "marginal")
    tail = ("SYNTHETIC -- not evidence for deployment."
            if args.synthetic else
            "Still OWES walk-forward/holdout/placebo/bootstrap before deploy.")
    print(f"VERDICT (this path): {verdict}. {tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
