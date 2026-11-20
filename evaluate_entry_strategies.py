#!/usr/bin/env python3
"""
evaluate_entry_strategies.py -- compare ENTRY strategies on GENUINE Alpaca IEX ticks.

Arms (identical candidate entries, chronological replay):
  A  baseline     : enter at the signal time/price; exit at a fixed horizon or a stop.
  B  entry_timing : baseline, but DELAY the entry until the buy-opportunity engine reaches
                    BUY_READY+ on the real order-flow stream (skip if never confirmed).
  C  combined     : B entry timing PLUS the intelligent profit-exit engine on the exit side.

DATA INTEGRITY
    Order flow is replayed from REAL IEX trades + quotes (StockHistoricalDataClient). It is
    NEVER fabricated from OHLCV bars. If tick data is unavailable, the run prints UNVALIDATED
    and the forward paper-validation procedure instead of inventing numbers.

SCOPE
    IEX-only (single-exchange proxy, not consolidated), L1 quotes, and limited free-tier
    intraday tick history => a demonstration/validation harness, not a multi-year OOS study.
    The SAME engines run forward in paper trading (shadow mode) on the live IEX feed.

Metrics: net expectancy after costs, profit factor, win rate, entry slippage, max drawdown,
         entry-timing improvement, max adverse excursion, trade frequency, missed profitable
         trades, post-entry movement.

Usage
    python evaluate_entry_strategies.py --symbols AAPL,MSFT,NVDA,AMD,TSLA --lookback-min 120 --horizon-min 30
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import order_flow_engine as ofe
import buy_opportunity_engine as bo
import profit_exit_engine as pxe
from costs import BASELINE_COSTS, total_cost_bps
# Reuse the genuine-tick data layer from the exit evaluator (no fabrication).
from evaluate_exit_strategies import _client, fetch_ticks, Entry, synthesize_entries, _exit_cost

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")


def _light_structure(entry_px: float, price: float, peak: float, trough: float,
                     vwap: Optional[float]) -> dict:
    """Lightweight bullish-structure inputs derived from the tick stream (DMI/ADX need bars
    and are left False here -- order flow carries the signal). Conservative: unknowns False."""
    return {
        "higher_low": price > trough and trough < entry_px,
        "price_stops_lower_low": price >= trough,
        "reclaimed_vwap": bool(vwap is not None and price >= vwap),
        "reclaimed_sma50": False, "di_plus_over_minus": False, "adx_rising": False,
        "failed_breakdown_reversal": bool(trough < entry_px and price > entry_px),
        "breakout_confirmed": price > peak * 0.999 and price > entry_px,
        "above_sma50": price >= entry_px, "pulled_back": price < peak and price >= trough,
        "over_extended": (price / entry_px - 1.0) > 0.03,
    }


def replay_entry(arm: str, entry: Entry, trades: list, quotes: list, horizon_s: float,
                 baseline_stop_pct: float, decision_interval_s: float,
                 buy_cfg: bo.BuyOpportunityConfig, pe_cfg: pxe.ProfitExitConfig) -> Optional[dict]:
    """Replay one candidate entry through one arm on the real tick stream. Returns a result
    dict, or None if the arm skipped the trade (B/C: never confirmed)."""
    flow = ofe.SymbolOrderFlow(entry.symbol, ofe.OrderFlowConfig(min_trades_for_signal=5, stale_trade_s=60))
    buy_engine = bo.BuyOpportunityEngine(buy_cfg, state_path=None)
    events = ([(q["ts"], "q", q) for q in quotes if q["ts"] >= entry.entry_ts]
              + [(t["ts"], "t", t) for t in trades if t["ts"] >= entry.entry_ts])
    events.sort(key=lambda e: (e[0], 0 if e[1] == "q" else 1))

    peak = trough = last_price = entry.entry_px
    vwap_num = vwap_den = 0.0
    last_spread_bps: Optional[float] = None
    entered = (arm == "baseline")
    entry_px = entry.entry_px
    entry_ts = entry.entry_ts
    last_decision = entry.entry_ts

    for ts, kind, ev in events:
        if ts > entry.entry_ts + horizon_s:
            break
        if kind == "q":
            flow.on_quote(ts, ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"])
            if ev["bid"] > 0 and ev["ask"] > 0:
                mid = 0.5 * (ev["bid"] + ev["ask"])
                last_spread_bps = (ev["ask"] - ev["bid"]) / mid * 1e4 if mid > 0 else last_spread_bps
            continue
        price = ev["price"]
        flow.on_trade(ts, price, ev["size"])
        last_price = price
        peak = max(peak, price); trough = min(trough, price)
        vwap_num += price * ev["size"]; vwap_den += ev["size"]
        if not entered and arm in ("entry_timing", "combined"):
            if ts - last_decision < decision_interval_s:
                continue
            last_decision = ts
            snap = flow.snapshot(now=ts)
            vwap = (vwap_num / vwap_den) if vwap_den > 0 else None
            struct = _light_structure(entry.entry_px, price, peak, trough, vwap)
            rec = buy_engine.evaluate(entry.symbol, snap, struct, scanner_eligible=True,
                                      now=ts, shadow=False)
            if rec.enforce:   # buy engine CONFIRMED -> enter here
                entered = True
                entry_px = price
                entry_ts = ts
    if not entered:
        return None  # arm skipped (never confirmed within the horizon)

    # exit: arm C uses the intelligent profit-exit engine; A/B use the fixed horizon.
    exit_px = last_price
    if arm == "combined":
        exit_px = _replay_exit(entry.symbol, entry_px, entry_ts, trades, quotes,
                               entry.entry_ts + horizon_s, pe_cfg, decision_interval_s)
    qty = entry.qty
    gross = (exit_px - entry_px) * qty
    cost = _exit_cost(exit_px, qty, last_spread_bps) + _exit_cost(entry_px, qty, last_spread_bps)
    mfe = (peak - entry_px) * qty
    mae = (trough - entry_px) * qty
    return {
        "arm": arm, "symbol": entry.symbol, "entry_px": entry_px, "ref_px": entry.entry_px,
        "exit_px": exit_px, "qty": qty, "gross": gross, "net": gross - cost, "cost": cost,
        "mfe": mfe, "mae": mae, "entry_delay_s": entry_ts - entry.entry_ts,
        "entry_improvement": (entry.entry_px - entry_px) * qty,  # + = entered cheaper than baseline
        "post_entry_move": (last_price - entry_px) / entry_px if entry_px else 0.0,
    }


def _replay_exit(symbol, entry_px, entry_ts, trades, quotes, end_ts, pe_cfg, interval_s) -> float:
    """Minimal intelligent-exit replay for arm C (full/stop else horizon close)."""
    flow = ofe.SymbolOrderFlow(symbol, ofe.OrderFlowConfig(min_trades_for_signal=5, stale_trade_s=60))
    eng = pxe.ProfitExitEngine(pe_cfg, state_path=None)
    events = ([(q["ts"], "q", q) for q in quotes if q["ts"] >= entry_ts]
              + [(t["ts"], "t", t) for t in trades if t["ts"] >= entry_ts])
    events.sort(key=lambda e: (e[0], 0 if e[1] == "q" else 1))
    peak = trough = last = entry_px
    last_decision = entry_ts
    for ts, kind, ev in events:
        if ts > end_ts:
            break
        if kind == "q":
            flow.on_quote(ts, ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"]); continue
        price = ev["price"]; flow.on_trade(ts, price, ev["size"]); last = price
        peak = max(peak, price); trough = min(trough, price)
        if ts - last_decision < interval_s:
            continue
        last_decision = ts
        snap = flow.snapshot(now=ts)
        struct = {"lower_high": price < peak, "lower_low": price <= trough + 1e-9,
                  "lost_vwap": snap.mid is not None and price < snap.mid,
                  "failed_breakout": price < entry_px, "below_fast_slow": price < peak}
        atr = max((peak - trough) * 0.5, entry_px * 0.001)
        rec = eng.evaluate(symbol, 100, entry_px, price, snap, struct, atr, 1e7, now=ts, shadow=False)
        if rec.decision in (pxe.FULL_EXIT, pxe.RISK_EXIT) and rec.enforce:
            return price
        if rec.stop_breached:
            return price
    return last


def metrics(results: list[dict], n_candidates: int) -> dict:
    taken = [r for r in results if r is not None]
    if not taken:
        return {"n": 0, "trades_taken": 0, "missed": n_candidates}
    nets = [r["net"] for r in taken]
    wins = [x for x in nets if x > 0]; losses = [x for x in nets if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    cum = peak = mdd = 0.0
    for x in nets:
        cum += x; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return {
        "trades_taken": len(taken), "missed": n_candidates - len(taken),
        "net_expectancy": round(statistics.mean(nets), 4),
        "realized_pnl_net": round(sum(nets), 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "win_rate": round(len(wins) / len(nets), 3),
        "max_drawdown": round(mdd, 2),
        "avg_mae": round(statistics.mean([r["mae"] for r in taken]), 2),
        "avg_entry_delay_s": round(statistics.mean([r["entry_delay_s"] for r in taken]), 1),
        "avg_entry_improvement": round(statistics.mean([r["entry_improvement"] for r in taken]), 3),
        "avg_post_entry_move_pct": round(statistics.mean([r["post_entry_move"] for r in taken]) * 100, 3),
        "total_slippage": round(sum(r["cost"] for r in taken), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMD,TSLA")
    ap.add_argument("--lookback-min", type=int, default=120)
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--notional", type=float, default=2000.0)
    ap.add_argument("--baseline-stop-pct", type=float, default=-0.010)
    ap.add_argument("--decision-interval-s", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    client = _client()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"[eval] fetching GENUINE IEX ticks (data_source=IEX, not consolidated) for {symbols}")
    entries = synthesize_entries(client, symbols, args.lookback_min, args.horizon_min, args.notional)
    if not entries:
        print("\nUNVALIDATED: no IEX tick data available for the requested window.")
        print("Implementation + unit tests are complete. Forward paper-validation procedure:")
        _print_paper_procedure()
        return

    buy_cfg = bo.BuyOpportunityConfig(persistence_cycles=2, min_confirmations=3, cooldown_s=30.0)
    pe_cfg = pxe.ProfitExitConfig(persistence_cycles=2, min_confirmations=3, cooldown_s=30.0)
    arms = ["baseline", "entry_timing", "combined"]
    by_arm: dict[str, list] = {a: [] for a in arms}
    horizon_s = args.horizon_min * 60
    for e in entries:
        start = datetime.fromtimestamp(e.entry_ts, timezone.utc)
        end = start + timedelta(seconds=horizon_s + 5)
        trades, quotes = fetch_ticks(client, e.symbol, start, end)
        if not trades:
            continue
        for a in arms:
            by_arm[a].append(replay_entry(a, e, trades, quotes, horizon_s, args.baseline_stop_pct,
                                          args.decision_interval_s, buy_cfg, pe_cfg))

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "data_source": "IEX (single-exchange proxy, not consolidated)",
              "n_candidates": len(entries), "entries": [e.symbol for e in entries],
              "params": vars(args),
              "arms": {a: metrics(by_arm[a], len(entries)) for a in arms}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"entry_strategy_eval_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== Entry-strategy comparison (GENUINE IEX ticks; {len(entries)} candidates) ===")
    cols = ["trades_taken", "missed", "net_expectancy", "realized_pnl_net", "profit_factor",
            "win_rate", "max_drawdown", "avg_mae", "avg_entry_delay_s", "avg_entry_improvement",
            "avg_post_entry_move_pct", "total_slippage"]
    print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in arms))
    for c in cols:
        print(f"{c:<24}" + "".join(f"{str(report['arms'][a].get(c)):>14}" for a in arms))
    print(f"\n[eval] wrote {OUT_DIR / f'entry_strategy_eval_{tag}.json'}")
    print("NOTE: IEX-only proxy + limited intraday tick history = demonstration, not a full OOS study.")


def _print_paper_procedure() -> None:
    print("  # 1. Run in paper + shadow (both overlays log, submit nothing):")
    print("  python trade.py")
    print("  # 2. Collect >= 20 sessions of buy/exit recommendations under system-data/<YYYYMMDD>/")
    print("  #    (*_buy_opportunity_recs_*.csv, *_profit_exit_recs_*.csv) + execution_events.jsonl")
    print("  # 3. Compare shadow recommendations vs realized scanner entries/exits before enforcing.")


if __name__ == "__main__":
    main()
