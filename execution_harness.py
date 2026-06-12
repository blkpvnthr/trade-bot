#!/usr/bin/env python3
"""execution_harness.py — expected-vs-actual execution audit for the shares-only trend engine.

Proves the live (paper) implementation behaves like the backtest. Compares the engine's
EXPECTED snapshot (logs/execution/expected_latest.json) against ACTUAL Alpaca fills/positions
and reports slippage, turnover, allocation drift, and state-recovery.

Run AFTER live_trend_engine.py (--submit) each day:
    python execution_harness.py            # audit today's run
    python execution_harness.py --date 20260608

Daily report → logs/execution/exec_report_<date>.{txt,json}
"""
from __future__ import annotations
import os, sys, json, argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/blkpvnthr/Desktop/trader"); sys.path.insert(0, str(ROOT))
ET = ZoneInfo("America/New_York")
EXEC_DIR = ROOT / "logs" / "execution"


def _parse_expected_since(expected: dict) -> datetime:
    ts = expected.get("ts")
    if ts:
        try:
            return datetime.fromisoformat(str(ts))
        except ValueError:
            pass
    return datetime.strptime(expected["date"], "%Y-%m-%d").replace(tzinfo=ET)


def _order_matches_expected_universe(order_symbol: str, expected_symbols: set[str]) -> bool:
    return str(order_symbol or "").strip().upper() in expected_symbols


def _collect_expected_fills(orders, ref: dict, expected_symbols: set[str]) -> tuple[dict, list[str]]:
    fills: dict[str, dict[str, float]] = {}
    ignored_symbols: set[str] = set()
    for o in orders:
        status = str(getattr(o, "status", ""))
        if status not in ("OrderStatus.FILLED", "filled") and not getattr(o, "filled_avg_price", None):
            continue
        fq = float(getattr(o, "filled_qty", 0) or 0)
        fp = float(getattr(o, "filled_avg_price", 0) or 0)
        if fq <= 0 or fp <= 0:
            continue
        symbol = str(getattr(o, "symbol", "") or "").upper()
        if not _order_matches_expected_universe(symbol, expected_symbols):
            if symbol:
                ignored_symbols.add(symbol)
            continue
        side = getattr(o, "side", "")
        side = side.value if hasattr(side, "value") else str(side)
        fill = fills.setdefault(symbol, dict(buy_qty=0.0, sell_qty=0.0, notional=0.0, slip_notional=0.0))
        fill["notional"] += fq * fp
        ref_price = ref.get(symbol, fp)
        slip = ((fp - ref_price) / ref_price) if "buy" in side.lower() else ((ref_price - fp) / ref_price)
        fill["slip_notional"] += slip * fq * fp
        fill["buy_qty" if "buy" in side.lower() else "sell_qty"] += fq
    return fills, sorted(ignored_symbols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYYMMDD (default: latest expected snapshot)")
    args = ap.parse_args()

    exp_path = (EXEC_DIR / f"expected_{args.date}.json") if args.date else (EXEC_DIR / "expected_latest.json")
    if not exp_path.exists():
        print(f"No expected snapshot at {exp_path} — run live_trend_engine.py first."); return
    exp = json.loads(exp_path.read_text())
    uni = list(exp["target_shares"].keys())
    equity = exp["equity"]; ref = exp["reference_price"]; target = exp["target_shares"]
    target_pct = exp["target_alloc_pct"]; day = exp["date"]

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import edge_scan_spreads as E
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    client = TradingClient(E.AK, E.AS, paper=True)
    acct = client.get_account(); live_equity = float(acct.equity)

    # actual positions (equities)
    pos = {}
    for p in client.get_all_positions():
        ac = p.asset_class.value if hasattr(p.asset_class, "value") else str(p.asset_class)
        if ac == "us_equity":
            pos[p.symbol] = dict(qty=int(float(p.qty)), mv=float(p.market_value or 0), price=float(p.current_price or 0))

    # today's fills
    since = _parse_expected_since(exp)
    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=since, limit=200))
    except Exception as e:
        orders = []; print(f"[warn] get_orders failed: {str(e)[:80]}")
    fills, ignored_fill_symbols = _collect_expected_fills(orders, ref, {s.upper() for s in uni})

    # metrics
    traded_notional = sum(f["notional"] for f in fills.values())
    turnover = 100 * traded_notional / live_equity if live_equity else 0
    slip_w = (sum(f["slip_notional"] for f in fills.values()) / traded_notional * 100) if traded_notional else 0.0
    actual_pct = {s: round(pos.get(s, {}).get("mv", 0) / live_equity * 100, 2) for s in uni}
    drift = sum(abs(actual_pct[s] - target_pct.get(s, 0)) for s in uni)
    # state recovery: after fills, holdings should equal target (engine is idempotent)
    recov = all(pos.get(s, {}).get("qty", 0) == target.get(s, 0) for s in uni)

    mode = exp.get("regime") or f"lev {exp.get('leverage','?')}x, {len(exp.get('active', []))} active"
    rep = dict(date=day, regime=mode, expected_equity=equity, live_equity=round(live_equity, 2),
               target_shares=target, actual_shares={s: pos.get(s, {}).get("qty", 0) for s in uni},
               target_alloc_pct=target_pct, actual_alloc_pct=actual_pct,
               turnover_pct=round(turnover, 2), slippage_pct=round(slip_w, 3),
               rebalance_drift_pct=round(drift, 2), state_recovery="PASS" if recov else "FAIL",
               intended_orders=exp["intended_orders"],
               fills={s: {k: round(v, 2) for k, v in f.items()} for s, f in fills.items()},
               ignored_fill_symbols=ignored_fill_symbols)

    # text report
    lines = [f"\n{'='*56}", f"EXECUTION AUDIT  {day}  ({mode})", "=" * 56,
             f"  equity: expected ${equity:,.2f}  live ${live_equity:,.2f}",
             f"  {'sym':<5}{'tgt%':>7}{'act%':>7}{'tgtSh':>7}{'actSh':>7}"]
    for s in uni:
        lines.append(f"  {s:<5}{target_pct.get(s,0):>7.1f}{actual_pct[s]:>7.1f}{target.get(s,0):>7}{pos.get(s,{}).get('qty',0):>7}")
    lines += [f"  slippage   : {slip_w:+.3f}%   (notional-weighted, + = adverse)",
              f"  turnover   : {turnover:.2f}%",
              f"  drift      : {drift:.2f}%   (|actual - target| alloc, summed)",
              f"  state recov: {'PASS' if recov else 'FAIL'}", "=" * 56]
    report = "\n".join(lines); print(report)

    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    dtag = day.replace("-", "")
    (EXEC_DIR / f"exec_report_{dtag}.json").write_text(json.dumps(rep, indent=2))
    (EXEC_DIR / f"exec_report_{dtag}.txt").write_text(report)
    print(f"\nsaved → logs/execution/exec_report_{dtag}.json")
    print("EXEC_HARNESS_DONE")


if __name__ == "__main__":
    main()
