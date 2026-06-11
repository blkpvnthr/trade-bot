#!/usr/bin/env python3
"""
book_skew_backtest.py — HISTORICAL backtest of the Databento book-skew liquidity-taking strategy.

Faithful port of the Databento tutorial strategy (top-of-book imbalance skew = log10(bid_sz) -
log10(ask_sz); buy 1 lot when skew > k, sell when skew < -k, capped at +/- POSITION_MAX), BUT
replayed over HISTORICAL MBP-1 (db.Historical) instead of db.Live — because:
  • our Databento key has NO live data license (db.Live raises), and
  • book-skew liquidity-taking is a true HFT strategy needing colocation / sub-ms latency / DMA;
    on a retail Alpaca path (~100ms+ latency, no queue priority) it is NOT deployable.

HONESTY: the tutorial assumes ZERO latency and fills at the live BBO. We keep that (buys lift the
ask, sells hit the bid — so the spread IS paid), which is the *optimistic* bound. Real fills are
worse (latency + alpha decay + queue position). Treat any positive net PnL here as an upper bound,
not a tradeable edge. Adds the tutorial's spoofing upper-limit L and a per-tick cooldown knob.

Usage:
  python book_skew_backtest.py --symbol AAPL --dataset XNAS.ITCH --date 2026-06-08 \
      --start 13:30 --end 20:00 --k 1.7 --max-pos 10 --fee-per-side 0.0 --point-value 1
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import databento as db

ROOT = Path(__file__).resolve().parent


def _key():
    for line in open(ROOT / ".env"):
        if line.startswith(("DATABENTO_API=", "DATABENTO_API_KEY=")):
            return line.split("=", 1)[1].strip()
    return ""


def run(symbol, dataset, date, start, end, k, L, max_pos, fee_per_side, point_value, cooldown):
    c = db.Historical(_key())
    df = c.timeseries.get_range(dataset=dataset, schema="mbp-1", symbols=symbol,
                                stype_in="raw_symbol",
                                start=f"{date}T{start}", end=f"{date}T{end}").to_df()
    if df.empty:
        print("no MBP-1 data for window"); return None

    pos = 0
    buy_qty = sell_qty = 0
    real_buy = real_sell = 0.0     # realized $ paid/received (per share * point_value applied at PnL)
    fees = 0.0
    last_trade_i = -10**9
    eq_curve = []                  # mark-to-market PnL path for Sharpe
    spread_paid = 0.0

    bp = df["bid_px_00"].to_numpy(); ap = df["ask_px_00"].to_numpy()
    bs = df["bid_sz_00"].to_numpy(); asz = df["ask_sz_00"].to_numpy()
    n = len(df)
    for i in range(n):
        b, a, vb, va = bp[i], ap[i], bs[i], asz[i]
        if not (b > 0 and a > 0 and vb > 0 and va > 0):
            continue
        skew = math.log10(vb) - math.log10(va)
        mid = (a + b) / 2.0
        can_trade = (i - last_trade_i) >= cooldown
        if can_trade and skew > k and abs(skew) < L and pos < max_pos:
            pos += 1; buy_qty += 1; real_buy += a; fees += fee_per_side
            spread_paid += (a - mid); last_trade_i = i
        elif can_trade and skew < -k and abs(skew) < L and pos > -max_pos:
            pos -= 1; sell_qty += 1; real_sell += b; fees += fee_per_side
            spread_paid += (mid - b); last_trade_i = i

        # mark-to-market theo PnL (liquidate at current BBO)
        theo_sell = b * pos if pos > 0 else 0.0
        theo_buy = a * (-pos) if pos < 0 else 0.0
        pnl = point_value * (real_sell + theo_sell - real_buy - theo_buy) - fees
        eq_curve.append(pnl)

    trades = buy_qty + sell_qty
    final_pnl = eq_curve[-1] if eq_curve else 0.0
    # PnL-increment Sharpe (per-event), annualization-agnostic — just a stability read
    import numpy as np
    d = np.diff(np.array(eq_curve)) if len(eq_curve) > 2 else np.array([0.0])
    sharpe = float(d.mean() / d.std()) if d.std() > 0 else 0.0
    return {"symbol": symbol, "date": date, "events": n, "trades": trades,
            "final_pos": pos, "fees": fees, "spread_paid": point_value * spread_paid,
            "final_pnl": final_pnl, "gross_pnl": final_pnl + fees + point_value * spread_paid,
            "sharpe": sharpe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--dataset", default="XNAS.ITCH")
    ap.add_argument("--date", default="2026-06-08")
    ap.add_argument("--start", default="13:30")
    ap.add_argument("--end", default="20:00")
    ap.add_argument("--k", type=float, default=1.7, help="skew alpha threshold")
    ap.add_argument("--L", type=float, default=4.0, help="spoofing upper limit (|skew|<L)")
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--fee-per-side", type=float, default=0.0, help="$ commission per side (equities ~0)")
    ap.add_argument("--point-value", type=float, default=1.0, help="$ per price point (ES=50, equity=1)")
    ap.add_argument("--cooldown", type=int, default=0, help="min MBP-1 events between trades (0=toy)")
    args = ap.parse_args()

    r = run(args.symbol, args.dataset, args.date, args.start, args.end, args.k, args.L,
            args.max_pos, args.fee_per_side, args.point_value, args.cooldown)
    if not r:
        return
    print(f"\n# Book-skew backtest — {r['symbol']} {r['date']} (historical MBP-1, k={args.k} L={args.L})")
    print(f"  MBP-1 events: {r['events']:,}   trades: {r['trades']:,}   final position: {r['final_pos']}")
    print(f"  gross PnL (mid-to-mid): ${r['gross_pnl']:,.2f}")
    print(f"  spread paid (taking liquidity): -${r['spread_paid']:,.2f}")
    print(f"  fees: -${r['fees']:,.2f}")
    print(f"  NET PnL: ${r['final_pnl']:,.2f}   (PnL-increment Sharpe {r['sharpe']:+.3f})")
    print("\n  CAVEAT: zero-latency upper bound. Real fills (latency + queue + alpha decay) are worse.")
    print("  NOT deployable here — no Databento live license + retail latency. Research only.")


if __name__ == "__main__":
    main()
