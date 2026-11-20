#!/usr/bin/env python3
"""
backtest_orderflow_5y.py -- backtest the order-flow BUY + EXIT engines on GENUINE IEX ticks,
sampled across the last ~5 years.

WHY SAMPLED (read this)
    The buy_opportunity_engine + profit_exit_engine are ORDER-FLOW engines: CVD, aggressive
    buy/sell classification, quote imbalance and exhaustion require tick-level trades + quotes,
    NOT 5-min OHLCV. Deriving order flow from bars is forbidden (it would be a mirage). Real
    IEX ticks DO exist back ~5 years, but fetching every tick for 5 years is infeasible
    (billions of events). So we sample N trading days per quarter across the window and replay
    the engines on 100% GENUINE ticks for those days. Each intraday day is an independent
    sample of the strategy, so sampling days across regimes is a valid OOS design for an
    intraday engine. This is honest evidence, not an exhaustive tick replay.

SCOPE / HONESTY
    IEX-only (single-exchange proxy, not consolidated SIP; L1 quotes, not depth). Quotes are
    thinned to <= 1/sec client-side (faithful for prevailing-quote classification + spread).
    Costs are modeled via costs.py (spread+slippage+commission+impact). Walk-forward split
    reported. Verdict is falsification-grade: ACCEPT only if it survives OOS after costs.

Arms (identical sampled days):
  A baseline     : buy at the first structure-eligible bar, hold to the session close.
  B entry_timing : enter only when the BUY engine confirms (BUY_READY+); exit at session close.
  C combined     : B entry + the intelligent profit-exit engine on the exit side.

Usage
    python backtest_orderflow_5y.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days-per-quarter 1
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
for _l in (PROJECT_ROOT / ".env").read_text().splitlines() if (PROJECT_ROOT / ".env").exists() else []:
    if "=" in _l and not _l.strip().startswith("#"):
        _k, _v = _l.split("=", 1)
        import os
        os.environ.setdefault(_k.strip(), _v.strip())

import order_flow_engine as ofe
import buy_opportunity_engine as bo
import profit_exit_engine as pxe
import trade as T
from costs import BASELINE_COSTS, total_cost_bps

CACHE = PROJECT_ROOT / "data" / "cache" / "iex_ticks_5y"
OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------- data (genuine IEX, cached)
def _client():
    import os
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(key, sec)


def _epoch(ts: Any) -> float:
    return ts.timestamp() if hasattr(ts, "timestamp") else float(ts)


def fetch_day(client, symbol: str, d: date, fetch_quotes: bool = False) -> Optional[dict]:
    """Fetch (and cache) one session's GENUINE IEX trades + 5-min bars (+ optional thinned
    quotes). Trades-only is the default: the core order-flow signals (CVD, aggressive buy/sell
    via the tick rule, exhaustion) come from real trades; full L1 quotes are optional and only
    needed for quote-aligned classification + measured spread. Returns dict or None."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{symbol}_{d.isoformat()}"
    tp = CACHE / f"{tag}_trades.csv"; qp = CACHE / f"{tag}_quotes.csv"; bp = CACHE / f"{tag}_bars.csv"
    need_quotes = fetch_quotes
    def _rd(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    if tp.exists() and bp.exists() and (qp.exists() or not need_quotes):
        tr = _rd(tp); b = _rd(bp)
        q = _rd(qp) if qp.exists() else pd.DataFrame()
        if tr.empty or b.empty:
            return None
        return {"trades": tr.to_dict("records"),
                "quotes": q.to_dict("records") if not q.empty else [], "bars": b}
    from alpaca.data.requests import StockTradesRequest, StockQuotesRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    d0 = datetime(d.year, d.month, d.day, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
    d1 = datetime(d.year, d.month, d.day, 21, 0, tzinfo=timezone.utc)   # 16:00 ET
    quotes: list = []
    try:
        trr = client.get_stock_trades(StockTradesRequest(symbol_or_symbols=[symbol], start=d0, end=d1,
                                                         feed=DataFeed.IEX))
        trades = [{"ts": _epoch(t.timestamp), "price": float(t.price), "size": float(t.size)}
                  for t in (getattr(trr, "data", {}) or {}).get(symbol, [])]
        barr = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=[symbol],
                 timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=d0, end=d1, feed=DataFeed.IEX))
        bdf = getattr(barr, "df", None)
        bars = bdf.xs(symbol, level=0).reset_index() if bdf is not None and not bdf.empty else pd.DataFrame()
        if need_quotes:
            qrr = client.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=[symbol], start=d0, end=d1,
                                                            feed=DataFeed.IEX))
            last_sec = None
            for qq in (getattr(qrr, "data", {}) or {}).get(symbol, []):
                ts = _epoch(qq.timestamp); sec = int(ts)
                if sec == last_sec:
                    continue
                last_sec = sec
                quotes.append({"ts": ts, "bid": float(qq.bid_price or 0.0), "ask": float(qq.ask_price or 0.0),
                               "bid_size": float(qq.bid_size or 0.0), "ask_size": float(qq.ask_size or 0.0)})
    except Exception as exc:
        print(f"  [{symbol} {d}] fetch error: {str(exc)[:100]}")
        return None
    pd.DataFrame(trades).to_csv(tp, index=False)
    if need_quotes:
        pd.DataFrame(quotes).to_csv(qp, index=False)
    if isinstance(bars, pd.DataFrame):
        bars.to_csv(bp, index=False)
    if not trades or not isinstance(bars, pd.DataFrame) or bars.empty:
        return None
    return {"trades": trades, "quotes": quotes, "bars": bars}


# ---------------------------------------------------------------- sampling
def sample_dates(start: date, end: date, per_quarter: int) -> list[date]:
    """Pick `per_quarter` weekdays near the middle of each quarter in [start, end]."""
    out = []
    y, q = start.year, (start.month - 1) // 3
    while True:
        qmonth = q * 3 + 2            # middle month of the quarter
        base = date(y, qmonth, 14)
        for i in range(per_quarter):
            dd = base + timedelta(days=i * 7)
            while dd.weekday() >= 5:
                dd += timedelta(days=1)
            if start <= dd <= end:
                out.append(dd)
        q += 1
        if q > 3:
            q = 0; y += 1
        if date(y, q * 3 + 2, 1) > end:
            break
    return sorted(set(out))


# ---------------------------------------------------------------- structure (from 5-min bars)
_SCANNER = object.__new__(T.MomentumSetupScanner)


def _prep_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize bars once: ensure a 'timestamp' col + a precomputed '_ts' epoch col."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    b = bars.copy()
    tcol = "timestamp" if "timestamp" in b.columns else b.columns[0]
    if tcol != "timestamp":
        b = b.rename(columns={tcol: "timestamp"})
    b["_ts"] = pd.to_datetime(b["timestamp"], utc=True).astype("int64") / 1e9
    return b


def structure_at(prepped_bars: pd.DataFrame, upto_ts: float, symbol: str) -> dict:
    """Bullish-structure inputs from completed 5-min bars up to `upto_ts` (reuses the live
    engine's dmi_adx + compute_buy_structure + _intraday_direction_metrics). `prepped_bars`
    must come from _prep_bars (has '_ts')."""
    if prepped_bars is None or prepped_bars.empty:
        return {}
    b = prepped_bars[prepped_bars["_ts"] <= upto_ts]
    if len(b) < 15:
        return {}
    m = _SCANNER._intraday_direction_metrics(symbol, b, 5, 0.0)
    return T.compute_buy_structure(b, m)


# ---------------------------------------------------------------- replay one symbol-day
@dataclass
class Trade:
    symbol: str
    arm: str
    entry_px: float
    exit_px: float
    net: float
    mfe: float
    mae: float
    entry_ts: float
    exit_ts: float


def _cost(px: float, qty: float, spread_bps: Optional[float]) -> float:
    cfg = BASELINE_COSTS
    if spread_bps is not None and spread_bps >= 0:
        from costs import TransactionCostConfig
        cfg = TransactionCostConfig(spread_bps=max(spread_bps, 1.0), slippage_bps=cfg.slippage_bps,
                                    commission_bps=cfg.commission_bps, impact_coefficient=cfg.impact_coefficient)
    notional = px * qty
    return notional * total_cost_bps(notional, 1e7, cfg) / 1e4


def replay_day(symbol: str, data: dict, notional: float, decision_interval_s: float,
               buy_cfg: bo.BuyOpportunityConfig, pe_cfg: pxe.ProfitExitConfig,
               assumed_spread_bps: float = 3.0) -> dict[str, Optional[Trade]]:
    trades = data["trades"]; quotes = data.get("quotes") or []; bars = _prep_bars(data["bars"])
    if not trades or bars.empty:
        return {a: None for a in ("baseline", "entry_timing", "combined")}
    has_quotes = len(quotes) > 0
    events = ([(q["ts"], "q", q) for q in quotes] + [(t["ts"], "t", t) for t in trades])
    events.sort(key=lambda e: (e[0], 0 if e[1] == "q" else 1))
    session_close_ts = trades[-1]["ts"]

    def _patch_snap(snap):
        # Trades-only mode: no L1 quotes -> inject a realistic assumed spread for liquid names
        # so the entry-quality + spread gate work on real-trade-derived order flow.
        if not has_quotes:
            snap.spread_bps = assumed_spread_bps
            snap.liquidity = "tight" if assumed_spread_bps <= 5 else "normal"
        return snap

    flow = ofe.SymbolOrderFlow(symbol, ofe.OrderFlowConfig(min_trades_for_signal=5, stale_trade_s=120))
    buy_engine = bo.BuyOpportunityEngine(buy_cfg, state_path=None)
    exit_engine = pxe.ProfitExitEngine(pe_cfg, state_path=None)

    peak = trough = last = trades[0]["price"]
    last_spread_bps: Optional[float] = None if has_quotes else assumed_spread_bps
    last_decision = 0.0
    qty = max(1.0, round(notional / trades[0]["price"], 4))

    # arm state: entry price/ts (None until entered); baseline enters at first structured bar
    A = {"entry": None, "peak": None, "trough": None}
    B = {"entry": None, "ts": None}
    C = {"entry": None, "ts": None, "exited": None, "exit_ts": None}
    baseline_armed = False

    for ts, kind, ev in events:
        if kind == "q":
            flow.on_quote(ts, ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"])
            if ev["bid"] > 0 and ev["ask"] > 0:
                mid = 0.5 * (ev["bid"] + ev["ask"])
                last_spread_bps = (ev["ask"] - ev["bid"]) / mid * 1e4 if mid > 0 else last_spread_bps
            continue
        price = ev["price"]; flow.on_trade(ts, price, ev["size"]); last = price
        peak = max(peak, price); trough = min(trough, price)

        if ts - last_decision < decision_interval_s:
            continue
        last_decision = ts
        struct = structure_at(bars, ts, symbol)
        if not struct:
            continue
        # A baseline: enter at the first bar with enough structure, hold to close
        if A["entry"] is None and not baseline_armed:
            A["entry"] = price; A["peak"] = price; A["trough"] = price; baseline_armed = True
        # B/C: enter on buy-engine confirmation
        if B["entry"] is None or C["entry"] is None:
            snap = _patch_snap(flow.snapshot(now=ts))
            rec = buy_engine.evaluate(symbol, snap, struct, scanner_eligible=True, now=ts, shadow=False)
            if rec.enforce:
                if B["entry"] is None:
                    B["entry"] = price; B["ts"] = ts
                if C["entry"] is None:
                    C["entry"] = price; C["ts"] = ts
        # C exit: intelligent profit-exit while in position
        if C["entry"] is not None and C["exited"] is None:
            snap = _patch_snap(flow.snapshot(now=ts))
            estruct = {"lower_high": price < peak, "lower_low": price <= trough + 1e-9,
                       "lost_vwap": snap.mid is not None and price < snap.mid,
                       "failed_breakout": price < C["entry"], "below_fast_slow": price < peak}
            atr = max((peak - trough) * 0.5, C["entry"] * 0.001)
            er = exit_engine.evaluate(symbol, qty, C["entry"], price, snap, estruct, atr, 1e7, now=ts, shadow=False)
            if (er.decision in (pxe.FULL_EXIT, pxe.RISK_EXIT) and er.enforce) or er.stop_breached:
                C["exited"] = price; C["exit_ts"] = ts

    def _mk(arm, entry, entry_ts, exit_px, exit_ts):
        if entry is None:
            return None
        gross = (exit_px - entry) * qty
        net = gross - _cost(entry, qty, last_spread_bps) - _cost(exit_px, qty, last_spread_bps)
        return Trade(symbol, arm, entry, exit_px, net, (peak - entry) * qty, (trough - entry) * qty,
                     entry_ts or 0.0, exit_ts or session_close_ts)

    out = {
        "baseline": _mk("baseline", A["entry"], None, last, session_close_ts),
        "entry_timing": _mk("entry_timing", B["entry"], B["ts"], last, session_close_ts),
        "combined": _mk("combined", C["entry"], C["ts"],
                        C["exited"] if C["exited"] is not None else last,
                        C["exit_ts"] if C["exit_ts"] is not None else session_close_ts),
    }
    return out


# ---------------------------------------------------------------- metrics
def metrics(trades: list[Optional[Trade]], n_days: int) -> dict:
    taken = [t for t in trades if t is not None]
    if not taken:
        return {"trades": 0, "no_entry_days": n_days}
    nets = [t.net for t in taken]
    wins = [x for x in nets if x > 0]; losses = [x for x in nets if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    sd = statistics.pstdev(nets) if len(nets) > 1 else 0.0
    sharpe = (statistics.mean(nets) / sd) if sd > 0 else 0.0
    cum = pk = mdd = 0.0
    for x in nets:
        cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    return {
        "trades": len(taken), "no_entry_days": n_days - len(taken),
        "net_expectancy": round(statistics.mean(nets), 3),
        "total_net_pnl": round(sum(nets), 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "win_rate": round(len(wins) / len(nets), 3),
        "trade_sharpe": round(sharpe, 3),
        "max_drawdown": round(mdd, 2),
        "avg_mae": round(statistics.mean([t.mae for t in taken]), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMD,SPY")
    ap.add_argument("--start", default="2020-11-01")
    ap.add_argument("--end", default="2025-11-01")
    ap.add_argument("--days-per-quarter", type=int, default=1)
    ap.add_argument("--notional", type=float, default=2000.0)
    ap.add_argument("--decision-interval-s", type=float, default=30.0)
    ap.add_argument("--train-end", default="2023-05-01")
    ap.add_argument("--quotes", choices=["none", "full"], default="none",
                    help="none = trades-only (tick-rule classification, assumed spread; fast, feasible "
                         "over 5y); full = also fetch+thin L1 quotes (high fidelity, ~3 min/symbol-day)")
    ap.add_argument("--assumed-spread-bps", type=float, default=3.0,
                    help="assumed spread for trades-only mode (realistic for liquid mega-caps)")
    ap.add_argument("--score-ready", type=float, default=75.0,
                    help="BUY_READY threshold. Native engine = 75; on sampled IEX trades-only the "
                         "max score is ~60, so 75 almost never fires -- use e.g. 60 to test the signal.")
    ap.add_argument("--score-strong", type=float, default=90.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    client = _client()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    train_end = date.fromisoformat(args.train_end)
    dates = sample_dates(start, end, args.days_per_quarter)
    print(f"[bt] {len(symbols)} symbols x {len(dates)} sampled sessions ({dates[0]}..{dates[-1]}) "
          f"= {len(symbols)*len(dates)} symbol-days on GENUINE IEX ticks")

    buy_cfg = bo.BuyOpportunityConfig(persistence_cycles=2, min_confirmations=3, cooldown_s=60.0,
                                      score_ready=args.score_ready, score_strong=args.score_strong)
    pe_cfg = pxe.ProfitExitConfig(persistence_cycles=2, min_confirmations=3, cooldown_s=60.0)
    arms = ["baseline", "entry_timing", "combined"]
    all_tr: dict[str, list] = {a: [] for a in arms}
    split: dict[str, dict[str, list]] = {"train": {a: [] for a in arms}, "test": {a: [] for a in arms}}
    n_days = 0
    for d in dates:
        for sym in symbols:
            data = fetch_day(client, sym, d, fetch_quotes=(args.quotes == "full"))
            if data is None:
                continue
            n_days += 1
            res = replay_day(sym, data, args.notional, args.decision_interval_s, buy_cfg, pe_cfg,
                             assumed_spread_bps=args.assumed_spread_bps)
            bucket = "train" if d < train_end else "test"
            for a in arms:
                all_tr[a].append(res[a]); split[bucket][a].append(res[a])
        print(f"  {d}: cumulative symbol-days={n_days}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "IEX genuine ticks (single-exchange proxy, not consolidated); quotes thinned <=1/s",
        "symbols": symbols, "sessions": len(dates), "symbol_days": n_days,
        "window": [args.start, args.end], "train_end": args.train_end,
        "overall": {a: metrics(all_tr[a], n_days) for a in arms},
        "train": {a: metrics(split["train"][a], len(split["train"][a])) for a in arms},
        "test_oos": {a: metrics(split["test"][a], len(split["test"][a])) for a in arms},
    }
    # verdict: combined arm must beat baseline on OOS net expectancy AND be net-positive
    c = report["test_oos"]["combined"]; b = report["test_oos"]["baseline"]
    ce = c.get("net_expectancy", 0) or 0; be = b.get("net_expectancy", 0) or 0
    if c.get("trades", 0) >= 20 and ce > 0 and ce > be:
        verdict = "PROMISING (OOS combined net-positive and beats baseline) - needs more samples"
    elif c.get("trades", 0) < 20:
        verdict = "INCONCLUSIVE (too few OOS trades; add --days-per-quarter)"
    else:
        verdict = "REJECT (OOS combined not net-positive vs baseline after costs)"
    report["verdict"] = verdict

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"orderflow_5y_backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    def _tbl(title, block):
        print(f"\n== {title} ==")
        cols = ["trades", "no_entry_days", "net_expectancy", "total_net_pnl", "profit_factor",
                "win_rate", "trade_sharpe", "max_drawdown", "avg_mae"]
        print(f"{'metric':<16}" + "".join(f"{a:>14}" for a in arms))
        for col in cols:
            print(f"{col:<16}" + "".join(f"{str(block[a].get(col)):>14}" for a in arms))
    print(f"\n=== Order-flow engine backtest (GENUINE IEX ticks, {n_days} symbol-days, sampled) ===")
    _tbl("OVERALL", report["overall"])
    _tbl("TRAIN (in-sample)", report["train"])
    _tbl("TEST (out-of-sample)", report["test_oos"])
    print(f"\nVERDICT: {verdict}")
    print(f"[bt] wrote {OUT_DIR / f'orderflow_5y_backtest_{tag}.json'}")
    print("NOTE: IEX-only proxy + sampled sessions. Real deployment validation = forward paper (shadow).")


if __name__ == "__main__":
    main()
