#!/usr/bin/env python3
"""
rl_account_simulation.py — Offline simulated walk-forward over $25K / $50K / $100K.
===================================================================================

Replays the alpha engine's out-of-sample (chronological last-30%) trade outcomes through a
%-risk position-sizing model at three account sizes and reports the account metrics. This is
an OFFLINE backtest of the CURRENT (baseline) tradeable system — no RL agent is promoted, so
the simulated policy is the mechanical baseline (RL agents are advisory and gated).

Methodology / assumptions (stated explicitly so the numbers are interpretable):
  * One row of alpha_exit_dataset = one candidate trade with a realized R-multiple.
  * The replay holds MANY overlapping decision-points per symbol per day, which is NOT a
    sequential single-account trade stream. To make a realistic account sim we cap to
    MAX_TRADES_PER_DAY (default 5) per date_tag (first-K by timestamp → no look-ahead) and
    compound DAY BY DAY.
  * Each trade risks RISK_PCT (default 0.5%) of CURRENT equity; trade P&L = realized_R ×
    dollar_risk.  → %-returns are account-size invariant under %-risk sizing; only the dollar
    figures scale. Both are reported.
  * Chronological OOS slice (last 30% by date_tag) = the held-out walk-forward window.
  * Alpha trades are intraday (holding_period_days ≈ 0); hold time reported as such.
  * This is a STYLIZED backtest on replay decision-points, not a tick-accurate fills
    simulation (no commissions / slippage / partial fills) — directional, not a P&L promise.

SAFETY: read-only. No orders, no live wiring, no gate/risk changes.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import rl_common as rc

ACCOUNTS = [25_000.0, 50_000.0, 100_000.0]
RISK_PCT = float(os.getenv("SIM_RISK_PCT", "0.005"))
TEST_FRAC = float(os.getenv("SIM_TEST_FRAC", "0.30"))
MAX_TRADES_PER_DAY = int(os.getenv("SIM_MAX_TRADES_PER_DAY", "5"))


def _oos_trades(dataset: str = "alpha_exit_dataset") -> List[dict]:
    rows = rc.load_dataset(dataset)
    rows = sorted(rows, key=lambda r: (str(r.get("date_tag", "")), str(r.get("timestamp", ""))))
    if not rows:
        return []
    k = max(1, int(len(rows) * (1.0 - TEST_FRAC)))
    oos = rows[k:]
    # Cap to a realistic number of trades per day (first-K by timestamp; no look-ahead).
    by_day: Dict[str, int] = {}
    capped: List[dict] = []
    for r in oos:
        d = str(r.get("date_tag", ""))
        if by_day.get(d, 0) >= MAX_TRADES_PER_DAY:
            continue
        by_day[d] = by_day.get(d, 0) + 1
        capped.append(r)
    return capped


def simulate(account: float, trades: List[dict], risk_pct: float = RISK_PCT) -> Dict[str, Any]:
    equity = account
    peak = account
    max_dd = 0.0
    wins = losses = 0
    gross_win = gross_loss = 0.0
    daily: Dict[str, float] = {}
    holds: List[float] = []
    for t in trades:
        r = rc.to_f(t.get("realized_R", t.get("r_multiple")))
        dollar_risk = equity * risk_pct
        pnl = r * dollar_risk
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        if r > 0:
            wins += 1; gross_win += pnl
        elif r < 0:
            losses += 1; gross_loss += abs(pnl)
        daily[str(t.get("date_tag", ""))] = daily.get(str(t.get("date_tag", "")), 0.0) + pnl
        holds.append(rc.to_f(t.get("holding_period_days")))
    n = len(trades)
    day_pnls = list(daily.values())
    return {
        "account_start": round(account, 2),
        "account_end": round(equity, 2),
        "total_return_pct": round((equity / account - 1.0) * 100.0, 4),
        "total_pnl": round(equity - account, 2),
        "daily_avg_pnl": round(sum(day_pnls) / len(day_pnls), 2) if day_pnls else 0.0,
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "win_rate": round(wins / n, 4) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0),
        "trade_count": n,
        "avg_hold_days": round(sum(holds) / len(holds), 4) if holds else 0.0,
        "trading_days": len(day_pnls),
        "worst_day": round(min(day_pnls), 2) if day_pnls else 0.0,
        "best_day": round(max(day_pnls), 2) if day_pnls else 0.0,
    }


def run() -> Dict[str, Any]:
    trades = _oos_trades()
    results = {"risk_pct": RISK_PCT, "test_frac": TEST_FRAC, "oos_trades": len(trades),
               "max_trades_per_day": MAX_TRADES_PER_DAY,
               "policy": "baseline_mechanical (no RL promoted; advisory agents gated)",
               "accounts": {f"${int(a/1000)}K": simulate(a, trades) for a in ACCOUNTS}}
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "rl_account_simulation.json", results)
    return results


def _print(results: Dict[str, Any]) -> None:
    print(f"Simulated walk-forward (OOS trades={results['oos_trades']}, risk/trade={results['risk_pct']:.2%}, "
          f"policy={results['policy']})")
    cols = ["total_return_pct", "daily_avg_pnl", "max_drawdown_pct", "win_rate",
            "profit_factor", "trade_count", "avg_hold_days", "worst_day", "best_day"]
    hdr = f"{'account':9s} " + " ".join(f"{c:>15s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for acct, m in results["accounts"].items():
        print(f"{acct:9s} " + " ".join(f"{m[c]:>15}" for c in cols))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    res = run()
    _print(res)
