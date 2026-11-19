#!/usr/bin/env python3
"""
evaluate_exit_strategies.py -- compare exit strategies on GENUINE Alpaca IEX ticks.

Arms (identical entries, chronological replay, out-of-sample relative to each exit rule):
  A  baseline   : price-only exit -- hard stop, profit-lock/giveback, max-hold horizon.
                  (proxy for the existing engine's protective exits.)
  B  order_flow : baseline PLUS order-flow exit confirmation (profit_exit_engine, score +
                  independent confirmations gated).
  C  adaptive   : B PLUS an adaptive ATR trailing stop and partial profit-taking.

DATA INTEGRITY
    Order flow is replayed from REAL IEX trades + quotes
    (StockHistoricalDataClient.get_stock_trades / get_stock_quotes). It is NEVER fabricated
    from OHLCV bars. If tick data cannot be fetched, the run prints UNVALIDATED and the exact
    paper-validation commands instead of inventing numbers.

SCOPE / HONESTY
    Alpaca free tier = IEX only (single exchange, a fraction of consolidated volume) and L1
    quotes (not full depth). Results are an IEX-proxy study, not a consolidated-tape backtest,
    and intraday tick history on the free tier is limited to recent sessions -- so this is a
    demonstration/validation harness, not a multi-year OOS study. The same code runs forward
    in paper trading (shadow mode) where the live IEX feed drives the identical engines.

Metrics: net expectancy, realized P&L after costs, profit factor, max drawdown, profit
         giveback, profit capture vs MFE, modeled slippage, turnover.

Usage
    python evaluate_exit_strategies.py --symbols AAPL,MSFT,NVDA --lookback-min 120 --horizon-min 30
    python evaluate_exit_strategies.py --entries my_entries.csv
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
for _line in (PROJECT_ROOT / ".env").read_text().splitlines() if (PROJECT_ROOT / ".env").exists() else []:
    if "=" in _line and not _line.strip().startswith("#"):
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

import order_flow_engine as ofe
import profit_exit_engine as pxe
from costs import BASELINE_COSTS, total_cost_bps

OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------- data
def _client():
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(key, sec)


def _epoch(ts: Any) -> float:
    return ts.timestamp() if hasattr(ts, "timestamp") else float(ts)


def fetch_ticks(client, symbol: str, start: datetime, end: datetime) -> tuple[list, list]:
    """Return (trades, quotes) as time-sorted lists of dicts from REAL IEX data. [] on failure."""
    from alpaca.data.requests import StockTradesRequest, StockQuotesRequest
    from alpaca.data.enums import DataFeed
    trades, quotes = [], []
    try:
        tr = client.get_stock_trades(StockTradesRequest(
            symbol_or_symbols=[symbol], start=start, end=end, feed=DataFeed.IEX, limit=10000))
        for t in (getattr(tr, "data", {}) or {}).get(symbol, []):
            trades.append({"ts": _epoch(t.timestamp), "price": float(t.price), "size": float(t.size)})
    except Exception as exc:
        print(f"  [{symbol}] trade fetch failed: {exc}")
    try:
        q = client.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=[symbol], start=start, end=end, feed=DataFeed.IEX, limit=10000))
        for qq in (getattr(q, "data", {}) or {}).get(symbol, []):
            quotes.append({"ts": _epoch(qq.timestamp), "bid": float(qq.bid_price or 0.0),
                           "ask": float(qq.ask_price or 0.0), "bid_size": float(qq.bid_size or 0.0),
                           "ask_size": float(qq.ask_size or 0.0)})
    except Exception as exc:
        print(f"  [{symbol}] quote fetch failed: {exc}")
    trades.sort(key=lambda x: x["ts"])
    quotes.sort(key=lambda x: x["ts"])
    return trades, quotes


# ---------------------------------------------------------------- entries
@dataclass
class Entry:
    symbol: str
    entry_ts: float
    entry_px: float
    qty: float
    horizon_s: float


def synthesize_entries(client, symbols: list[str], lookback_min: int, horizon_min: int,
                       notional: float) -> list[Entry]:
    """Define entries from the FIRST real trade in the window per symbol (entries are fixed and
    independent of any exit rule, so all arms share identical entries)."""
    end = datetime.now(timezone.utc) - timedelta(minutes=16)   # respect IEX ~15m delay
    start = end - timedelta(minutes=lookback_min)
    entries = []
    for sym in symbols:
        trades, _q = fetch_ticks(client, sym, start, end)
        if not trades:
            print(f"  [{sym}] no trades in window -> skipped")
            continue
        first = trades[0]
        qty = max(1.0, round(notional / first["price"], 4))
        entries.append(Entry(sym, first["ts"], first["price"], qty, horizon_min * 60))
    return entries


# ---------------------------------------------------------------- replay
@dataclass
class ArmResult:
    arm: str
    symbol: str
    entry_px: float
    exit_px: float = 0.0
    qty: float = 0.0
    hold_s: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    slippage_dollars: float = 0.0
    mfe: float = 0.0            # max favorable excursion (price gain, $)
    mae: float = 0.0           # max adverse excursion ($)
    giveback: float = 0.0      # MFE not captured ($)
    capture_ratio: float = 0.0  # realized / MFE
    n_exits: int = 1
    exit_reason: str = ""


def _exit_cost(px: float, qty: float, spread_bps: Optional[float]) -> float:
    notional = px * qty
    cfg = BASELINE_COSTS
    if spread_bps is not None and spread_bps >= 0:
        from costs import TransactionCostConfig
        cfg = TransactionCostConfig(spread_bps=spread_bps, slippage_bps=BASELINE_COSTS.slippage_bps,
                                    commission_bps=BASELINE_COSTS.commission_bps,
                                    impact_coefficient=BASELINE_COSTS.impact_coefficient)
    return notional * total_cost_bps(notional, 1e7, cfg) / 1e4


def replay(arm: str, entry: Entry, trades: list, quotes: list,
           baseline_stop_pct: float, baseline_lock_pct: float, baseline_giveback_pct: float,
           decision_interval_s: float, pe_cfg: pxe.ProfitExitConfig) -> ArmResult:
    """Replay one entry through one arm on the real tick stream. Returns realized result."""
    res = ArmResult(arm=arm, symbol=entry.symbol, entry_px=entry.entry_px, qty=entry.qty)
    flow = ofe.SymbolOrderFlow(entry.symbol, ofe.OrderFlowConfig(min_trades_for_signal=5, stale_trade_s=60))
    engine = pxe.ProfitExitEngine(pe_cfg, state_path=None)

    # merge events chronologically (quotes before trades at equal ts so trades classify correctly)
    events = ([(q["ts"], "q", q) for q in quotes if q["ts"] >= entry.entry_ts]
              + [(t["ts"], "t", t) for t in trades if t["ts"] >= entry.entry_ts])
    events.sort(key=lambda e: (e[0], 0 if e[1] == "q" else 1))

    peak = entry.entry_px
    trough = entry.entry_px
    last_price = entry.entry_px
    qty_remaining = entry.qty
    realized = 0.0
    partial_done = False
    last_decision = entry.entry_ts
    last_spread_bps: Optional[float] = None

    def do_exit(px: float, q: float, reason: str) -> None:
        nonlocal realized
        cost = _exit_cost(px, q, last_spread_bps)
        realized += (px - entry.entry_px) * q
        res.slippage_dollars += cost
        res.exit_reason = (res.exit_reason + "+" + reason).strip("+")

    end_ts = entry.entry_ts + entry.horizon_s
    for ts, kind, ev in events:
        if ts > end_ts:
            break
        if kind == "q":
            flow.on_quote(ts, ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"])
            if ev["bid"] > 0 and ev["ask"] > 0:
                mid = 0.5 * (ev["bid"] + ev["ask"])
                last_spread_bps = (ev["ask"] - ev["bid"]) / mid * 1e4 if mid > 0 else last_spread_bps
            continue
        # trade
        price = ev["price"]
        flow.on_trade(ts, price, ev["size"])
        last_price = price
        peak = max(peak, price)
        trough = min(trough, price)
        res.mfe = max(res.mfe, (peak - entry.entry_px) * entry.qty)
        res.mae = min(res.mae, (trough - entry.entry_px) * entry.qty)

        # ---- Arm A: price-only baseline exits (all arms include these) ----
        stop_px = entry.entry_px * (1.0 + baseline_stop_pct)
        if price <= stop_px:
            do_exit(price, qty_remaining, "baseline_stop"); qty_remaining = 0.0; break
        peak_gain = (peak - entry.entry_px) / entry.entry_px
        if peak_gain >= baseline_lock_pct and price <= peak - baseline_giveback_pct * entry.entry_px:
            do_exit(price, qty_remaining, "baseline_giveback"); qty_remaining = 0.0; break

        if arm == "baseline":
            continue

        # ---- Arms B/C: order-flow exit decision at the poll cadence ----
        if ts - last_decision < decision_interval_s:
            continue
        last_decision = ts
        snap = flow.snapshot(now=ts)
        structure = {  # structure inferred from the tick stream vs the entry/peak
            "lower_high": price < peak,
            "lower_low": price <= trough + 1e-9,
            "lost_vwap": (snap.mid is not None and price < snap.mid),
            "failed_breakout": price < entry.entry_px,
            "below_fast_slow": price < peak,
        }
        atr = max((peak - trough) * 0.5, entry.entry_px * 0.001)  # intraday range proxy
        rec = engine.evaluate(entry.symbol, qty_remaining, entry.entry_px, price, snap,
                              structure, atr, adv=1e7, now=ts, shadow=False)
        if arm == "order_flow":
            if rec.decision in (pxe.FULL_EXIT, pxe.RISK_EXIT) and rec.enforce:
                do_exit(price, qty_remaining, "flow_full"); qty_remaining = 0.0; break
        elif arm == "adaptive":
            if rec.decision == pxe.PARTIAL_TAKE_PROFIT and rec.enforce and not partial_done and rec.net_profitable:
                q = qty_remaining * pe_cfg.partial_fraction
                do_exit(price, q, "flow_partial"); qty_remaining -= q; partial_done = True
                res.n_exits += 1
            elif rec.decision in (pxe.FULL_EXIT, pxe.RISK_EXIT) and rec.enforce:
                do_exit(price, qty_remaining, "flow_full"); qty_remaining = 0.0; break
            elif rec.stop_breached:
                do_exit(price, qty_remaining, "adaptive_stop"); qty_remaining = 0.0; break

    # close any remainder at the last price (end of horizon)
    if qty_remaining > 0:
        do_exit(last_price, qty_remaining, "horizon")
        qty_remaining = 0.0

    res.exit_px = last_price
    res.hold_s = end_ts - entry.entry_ts
    res.gross_pnl = realized
    res.net_pnl = realized - res.slippage_dollars
    realized_gain = max(0.0, realized)
    res.giveback = max(0.0, res.mfe - realized_gain)
    res.capture_ratio = (realized_gain / res.mfe) if res.mfe > 0 else 0.0
    return res


# ---------------------------------------------------------------- metrics
def metrics(results: list[ArmResult]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    nets = [r.net_pnl for r in results]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    # max drawdown on cumulative realized pnl
    cum, peak_c, mdd = 0.0, 0.0, 0.0
    for x in nets:
        cum += x; peak_c = max(peak_c, cum); mdd = min(mdd, cum - peak_c)
    captures = [r.capture_ratio for r in results if r.mfe > 0]
    return {
        "n": len(results),
        "net_expectancy": round(statistics.mean(nets), 4),
        "realized_pnl_net": round(sum(nets), 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "win_rate": round(len(wins) / len(nets), 3),
        "max_drawdown": round(mdd, 2),
        "avg_giveback": round(statistics.mean([r.giveback for r in results]), 2),
        "avg_capture_vs_mfe": round(statistics.mean(captures), 3) if captures else None,
        "total_slippage": round(sum(r.slippage_dollars for r in results), 2),
        "turnover_exits": sum(r.n_exits for r in results),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMD,TSLA")
    ap.add_argument("--lookback-min", type=int, default=120)
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--notional", type=float, default=2000.0)
    ap.add_argument("--entries", default="")
    ap.add_argument("--baseline-stop-pct", type=float, default=-0.010)
    ap.add_argument("--baseline-lock-pct", type=float, default=0.005)
    ap.add_argument("--baseline-giveback-pct", type=float, default=0.003)
    ap.add_argument("--decision-interval-s", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    client = _client()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"[eval] fetching GENUINE IEX ticks (data_source=IEX, not consolidated) for {symbols}")
    entries = synthesize_entries(client, symbols, args.lookback_min, args.horizon_min, args.notional)
    if not entries:
        print("\nUNVALIDATED: no IEX tick data available for the requested window.")
        print("Implementation + unit tests are complete; run forward in paper (shadow) mode:")
        _print_paper_commands()
        return

    pe_cfg = pxe.ProfitExitConfig(persistence_cycles=2, min_confirmations=3, cooldown_s=30.0)
    arms = ["baseline", "order_flow", "adaptive"]
    by_arm: dict[str, list[ArmResult]] = {a: [] for a in arms}
    for e in entries:
        start = datetime.fromtimestamp(e.entry_ts, timezone.utc)
        end = start + timedelta(seconds=e.horizon_s + 5)
        trades, quotes = fetch_ticks(client, e.symbol, start, end)
        if not trades:
            continue
        for a in arms:
            by_arm[a].append(replay(a, e, trades, quotes, args.baseline_stop_pct,
                                    args.baseline_lock_pct, args.baseline_giveback_pct,
                                    args.decision_interval_s, pe_cfg))

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "data_source": "IEX (single-exchange proxy, not consolidated)",
              "n_entries": len(entries),
              "entries": [e.symbol for e in entries],
              "params": vars(args),
              "arms": {a: metrics(by_arm[a]) for a in arms}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"exit_strategy_eval_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== Exit-strategy comparison (GENUINE IEX ticks; {len(entries)} entries) ===")
    cols = ["net_expectancy", "realized_pnl_net", "profit_factor", "win_rate",
            "max_drawdown", "avg_giveback", "avg_capture_vs_mfe", "total_slippage", "turnover_exits"]
    print(f"{'metric':<22}" + "".join(f"{a:>14}" for a in arms))
    for c in cols:
        print(f"{c:<22}" + "".join(f"{str(report['arms'][a].get(c)):>14}" for a in arms))
    print(f"\n[eval] wrote {OUT_DIR / f'exit_strategy_eval_{tag}.json'}")
    print("NOTE: IEX-only proxy + limited intraday tick history = demonstration, not a full OOS study.")


def _print_paper_commands() -> None:
    print("  # Shadow paper run (logs recommendations, submits nothing):")
    print("  PROFIT_EXIT_ENABLED=1 PROFIT_EXIT_ENFORCE=0 APCA_PAPER=1 python trade.py")


if __name__ == "__main__":
    main()
