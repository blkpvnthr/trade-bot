#!/usr/bin/env python3
"""live_trend_engine.py — MULTI-ASSET, VOL-TARGETED trend overlay (the validated upgrade).

Upgraded from equity-only (Sharpe ~0.67) to the multi-asset vol-targeted book that passed the
robustness sweep (Sharpe ~0.89, maxDD 21.7→16.7%, Calmar 0.39→0.54; robust across params,
no single-asset cliff; edge driven by genuine bonds+gold diversification):

  Per asset (SPY/QQQ/IWM/TLT/GLD/DBC/UUP), independently:
     LONG if  price > 200DMA × (1+BAND)  AND  200DMA slope > 0   else  flat
  Among the LONG ("active") assets:
     inverse-volatility raw weights (risk parity)
     scale total exposure (leverage) to TARGET_VOL using the trailing-window covariance,
     capped at MAX_LEVERAGE (and at account buying power).
  No options. Plain equity/ETF orders. Alpaca positions ARE the state.

SAFETY (unchanged):
  • Keys are PAPER (live endpoint 401s) → no real-money risk.
  • Default = DRY-RUN (reads account, prints intended rebalance, submits nothing).
  • --submit places PAPER orders; system_self_awareness CRITICAL health aborts submission.
  • Leverage is capped at MAX_LEVERAGE and never exceeds 95% of account buying power.
"""
from __future__ import annotations
import os, sys, json, math, argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path("/Users/blkpvnthr/Desktop/trader"); sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import edge_scan_spreads as E

ET = ZoneInfo("America/New_York")
UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DBC", "UUP"]  # equities + bonds + gold + commodities + $
BAND = 0.02
SLOPE_LOOKBACK = 20
JOURNAL = ROOT / "logs" / "live_trend_journal.jsonl"


def sma_and_slope(closes):
    days = sorted(closes)
    if len(days) < 200 + SLOPE_LOOKBACK:
        return None, None, None
    sma = sum(closes[d] for d in days[-200:]) / 200
    sma_prev = sum(closes[d] for d in days[-200 - SLOPE_LOOKBACK:-SLOPE_LOOKBACK]) / 200
    return closes[days[-1]], sma, (sma - sma_prev)


def log_returns(closes, n):
    days = sorted(closes)
    if len(days) < n + 1:
        return None
    return [math.log(closes[days[i]] / closes[days[i - 1]]) for i in range(len(days) - n, len(days))]


def ann_vol(closes, n):
    r = log_returns(closes, n)
    return float(np.std(r) * math.sqrt(252)) if r else None


def compute_allocation(closes, equity, target_vol=0.10, max_leverage=1.25, window=60, band=BAND, buying_power=None):
    """SINGLE SOURCE OF TRUTH for the allocation — used by both the live engine and the backtest.
    `closes` = {symbol: {date: close}} (full history up to the decision date).
    Returns active list, weights (fraction of equity), integer target shares, leverage, raw port vol."""
    spx = {}; active = []; detail = {}
    for s in closes:
        px, sma, slope = sma_and_slope(closes[s])
        if sma is None:
            detail[s] = "no-data"; continue
        spx[s] = px
        v = ann_vol(closes[s], window)
        on = (px > sma * (1 + band)) and (slope > 0) and bool(v)
        detail[s] = dict(px=round(px, 2), sma200=round(sma, 2), slope=round(slope, 2),
                         vol=round(v, 3) if v else None, active=on)
        if on:
            active.append(s)
    weights = {}; lev = 0.0; pv = 0.0
    if active:
        vols = {s: ann_vol(closes[s], window) for s in active}
        iv = {s: 1.0 / vols[s] for s in active}; tot = sum(iv.values())
        wraw = {s: iv[s] / tot for s in active}
        if len(active) >= 2:
            rets = {s: log_returns(closes[s], window) for s in active}
            ml = min(len(rets[s]) for s in active)
            Rm = np.array([rets[s][-ml:] for s in active])
            cov = np.cov(Rm) * 252
            wv = np.array([wraw[s] for s in active])
            pv = float(math.sqrt(max(wv @ cov @ wv, 1e-9)))
        else:
            pv = vols[active[0]]
        lev = min(target_vol / pv, max_leverage) if pv > 0 else 0.0
        if buying_power and equity > 0:
            lev = min(lev, (buying_power / equity) * 0.95)        # never exceed margin
        weights = {s: wraw[s] * lev for s in active}
    target = {s: int((weights[s] * equity) // spx[s]) for s in active if spx.get(s)}
    return dict(active=active, weights=weights, target=target, lev=lev, pv=pv, spx=spx, detail=detail)


def health_check():
    try:
        import system_self_awareness as SA
        h = SA.check_system_health()
        return h.get("status", "unknown"), h
    except Exception as e:
        return "unavailable", {"error": str(e)[:120]}


def journal(rec, dry):
    rec = {**rec, "ts": datetime.now(ET).isoformat(), "dry_run": dry}
    if not dry:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="place PAPER orders (else dry-run preview)")
    ap.add_argument("--target-vol", type=float, default=0.10, help="annualized portfolio vol target")
    ap.add_argument("--max-leverage", type=float, default=1.25, help="cap on gross exposure")
    ap.add_argument("--vol-window", type=int, default=60, help="trailing days for vol/cov")
    ap.add_argument("--band", type=float, default=BAND, help="price must clear 200DMA by this fraction")
    args = ap.parse_args()
    dry = not args.submit
    W = args.vol_window

    print("=" * 74)
    print(f"LIVE TREND ENGINE (multi-asset, vol-targeted)  {datetime.now(ET):%Y-%m-%d %H:%M %Z}")
    print(f"mode = {'DRY-RUN (no orders)' if dry else 'SUBMIT (paper)'}  "
          f"target_vol={args.target_vol:.0%}  max_lev={args.max_leverage}x  window={W}d")
    print("=" * 74)

    status, h = health_check()
    print(f"self-awareness health: {status}")
    if args.submit and status == "critical":
        print("ABORT: system health CRITICAL — refusing to submit."); return

    from alpaca.trading.client import TradingClient
    paper = str(os.getenv("ALPACA_PAPER_ONLY", "true")).lower() in {"1", "true", "yes", "paper"}
    client = TradingClient(E.AK, E.AS, paper=True)
    acct = client.get_account()
    equity = float(acct.equity); buying_power = float(acct.buying_power)
    print(f"account: {acct.account_number}  equity=${equity:,.2f}  buying_power=${buying_power:,.2f}")

    held = {}
    for p in client.get_all_positions():
        ac = p.asset_class.value if hasattr(p.asset_class, "value") else str(p.asset_class)
        if ac == "us_equity":
            held[p.symbol] = int(float(p.qty))
    held = {s: held.get(s, 0) for s in set(list(held) + UNIVERSE)}

    # ---- allocation (single source of truth — the SAME function the backtest calls) ----
    alloc = compute_allocation({s: E.get_closes(s) for s in UNIVERSE}, equity,
                               args.target_vol, args.max_leverage, W, args.band, buying_power)
    active, weights, target = alloc["active"], alloc["weights"], alloc["target"]
    spx, detail, lev, pv = alloc["spx"], alloc["detail"], alloc["lev"], alloc["pv"]

    print(f"\nportfolio vol(raw)={pv:.1%}  ->  leverage={lev:.2f}x  (gross exposure ${lev*equity:,.0f})")
    print(f"  {'asset':<6}{'px':>9}{'200DMA':>9}{'slope':>8}{'vol':>7}{'weight':>8}{'tgt$':>10}")
    for s in UNIVERSE:
        d = detail.get(s)
        if not isinstance(d, dict):
            print(f"  {s:<6}{'(no data)':>9}"); continue
        w = weights.get(s, 0.0); tdollar = w * equity
        flag = "LONG" if d["active"] else "flat"
        print(f"  {s:<6}{d['px']:>9}{d['sma200']:>9}{d['slope']:>+8.1f}{(d['vol'] or 0):>7.2f}"
              f"{w:>8.2f}{tdollar:>10,.0f}  {flag}")

    # ---- diff -> orders ----
    actions = []
    for s in set(list(held) + list(target)):
        d = target.get(s, 0) - held.get(s, 0)
        if d > 0: actions.append(("BUY", s, d))
        elif d < 0: actions.append(("SELL", s, -d))

    print(f"\nPLANNED REBALANCE ({len(actions)}):")
    if not actions:
        print("  (already aligned — no orders)")
    for side, s, qty in actions:
        print(f"  {side:<4} {qty:>6} {s}")

    # ---- expected snapshot (execution harness compares vs actual fills) ----
    today = datetime.now(ET).date()
    ref = {s: spx.get(s, 0.0) for s in UNIVERSE}
    expected = dict(
        date=today.isoformat(), ts=datetime.now(ET).isoformat(),
        leverage=round(lev, 3), port_vol_raw=round(pv, 4), active=active,
        equity=round(equity, 2),
        reference_price={s: round(ref[s], 4) for s in UNIVERSE},
        target_shares={s: target.get(s, 0) for s in UNIVERSE},
        target_alloc_pct={s: round(target.get(s, 0) * ref[s] / equity * 100, 2) if ref[s] else 0 for s in UNIVERSE},
        intended_orders=[dict(side=a[0], symbol=a[1], qty=a[2]) for a in actions],
        held_before={s: held.get(s, 0) for s in UNIVERSE}, submit=bool(args.submit))
    exp_dir = ROOT / "logs" / "execution"; exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / f"expected_{today:%Y%m%d}.json").write_text(json.dumps(expected, indent=2))
    (exp_dir / "expected_latest.json").write_text(json.dumps(expected, indent=2))

    if args.submit:
        # idempotency: clear any stale/pending orders so re-runs (e.g. pre-open) don't double-submit
        try:
            canceled = client.cancel_orders()
            print(f"  canceled {len(canceled) if canceled else 0} open order(s) before rebalance")
        except Exception as ex:
            print(f"  cancel_orders warning: {str(ex)[:80]}")

    if args.submit and actions:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        # sells first (free up buying power), then buys
        for side, s, qty in sorted(actions, key=lambda a: 0 if a[0] == "SELL" else 1):
            req = MarketOrderRequest(symbol=s, qty=qty,
                                     side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                                     time_in_force=TimeInForce.DAY)
            try:
                r = client.submit_order(req)
                print(f"  SUBMITTED {side} {qty} {s}  id={getattr(r,'id','?')}")
                journal({"event": "order", "side": side, "symbol": s, "qty": qty, "leverage": round(lev, 2)}, dry)
            except Exception as ex:
                print(f"  ORDER FAILED {side} {s}: {str(ex)[:120]}")

    print("\nNOTE: dry-run makes NO changes." if dry else "\nSUBMIT mode: paper orders sent above.")
    print("LIVE_TREND_DONE")


if __name__ == "__main__":
    main()
