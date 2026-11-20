#!/usr/bin/env python3
"""
backtest_di_bb_E.py -- variant E: DI/Bollinger entries & exits WITH order-flow intelligence
for ranking + profit protection. Evaluated on GENUINE IEX tick days (never fabricated from
bars), comparing:
    D : DI/BB entry (5-min completed candle) + DI/BB exit.
    E : D, but an entry is TAKEN only if the order-flow buy score confirms (ranking), and the
        position is protected by the intelligent profit-exit engine (tick-level) in addition
        to the DI/BB exit.

Reuses the sampled genuine-tick cache + engines from backtest_orderflow_5y. This is a smaller
real-tick sample (order flow requires ticks); the broad A-D comparison is bar-based in
backtest_di_bb.py. No look-ahead: a completed-bar signal fills at the next bar's open.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
import order_flow_engine as ofe
import profit_exit_engine as pxe
import buy_opportunity_engine as bo
from backtest_orderflow_5y import fetch_day, sample_dates, structure_at, _prep_bars, _cost
from backtest_dmi_adx import indicators

OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")
SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]


def _signals_5m(bars: pd.DataFrame) -> pd.DataFrame:
    d = indicators(bars, slope_lb=3)
    d["di_bb_entry"] = (d["pdi"] > d["mdi"]) & (d["close"] > d["mid"])
    d["di_bb_exit"] = (d["mdi"] > d["pdi"]) & (d["close"] < d["mid"])
    d["_ts"] = d["timestamp"].astype("int64") / 1e9
    return d


def replay_E(symbol: str, data: dict, of_gate_score: float, assumed_spread_bps: float = 3.0):
    """Return (D_trades, E_trades): lists of net-of-cost returns for the two variants on one day."""
    trades = data["trades"]
    if not trades:
        return [], []
    bars = _signals_5m(_prep_bars(data["bars"]).rename(columns={}))
    if len(bars) < 25:
        return [], []
    o = bars["open"].astype(float).values; ts = bars["_ts"].values
    ent = bars["di_bb_entry"].values; ext = bars["di_bb_exit"].values
    n = len(bars)

    flow = ofe.SymbolOrderFlow(symbol, ofe.OrderFlowConfig(min_trades_for_signal=5, stale_trade_s=120))
    buy_engine = bo.BuyOpportunityEngine(bo.BuyOpportunityConfig(persistence_cycles=1, score_ready=of_gate_score))
    exit_engine = pxe.ProfitExitEngine(pxe.ProfitExitConfig(persistence_cycles=1), state_path=None)
    ti = 0

    def feed_to(t_epoch):
        nonlocal ti
        while ti < len(trades) and trades[ti]["ts"] <= t_epoch:
            tr = trades[ti]; flow.on_trade(tr["ts"], tr["price"], tr["size"]); ti += 1

    def of_confirms(bar_i) -> bool:
        feed_to(ts[bar_i])
        snap = flow.snapshot(now=ts[bar_i]); snap.spread_bps = assumed_spread_bps; snap.liquidity = "tight"
        struct = structure_at(bars, ts[bar_i], symbol)
        rec = buy_engine.evaluate(symbol, snap, struct, scanner_eligible=True, now=ts[bar_i], shadow=False)
        return rec.enforce or rec.buy_score >= of_gate_score

    def run(use_of_gate: bool, use_of_protect: bool):
        out = []
        i = 0
        rt = 3.0 / 1e4  # ~3bps/side cost (liquid names)
        while i < n - 1:
            if not ent[i]:
                i += 1; continue
            if use_of_gate and not of_confirms(i):
                i += 1; continue
            ei = i + 1
            entry_px = o[ei] * (1 + rt)
            peak = o[ei]
            j = ei
            xi = None
            while j < n - 1:
                peak = max(peak, float(bars["high"].iloc[j]))
                if ext[j]:
                    xi = j + 1; break
                if use_of_protect:
                    feed_to(ts[j]); snap = flow.snapshot(now=ts[j])
                    snap.spread_bps = assumed_spread_bps
                    es = {"lower_high": o[j] < peak, "lower_low": False, "lost_vwap": False,
                          "failed_breakout": o[j] < o[ei], "below_fast_slow": False}
                    atr = max((peak - o[ei]) * 0.5, o[ei] * 0.001)
                    er = exit_engine.evaluate(symbol, 100, o[ei], float(bars["close"].iloc[j]), snap,
                                              es, atr, 1e7, now=ts[j], shadow=False)
                    if (er.decision in (pxe.FULL_EXIT, pxe.RISK_EXIT) and er.enforce) or er.stop_breached:
                        xi = j + 1; break
                j += 1
            if xi is None:
                xi = n - 1
            exit_px = o[xi] * (1 - rt)
            out.append(exit_px / entry_px - 1.0)
            i = xi + 1
        return out

    d_trades = run(use_of_gate=False, use_of_protect=False)
    e_trades = run(use_of_gate=True, use_of_protect=True)
    return d_trades, e_trades


def metrics(rets: list, notional=2000.0) -> dict:
    if not rets:
        return {"trades": 0}
    pnls = [r * notional for r in rets]
    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    return {"trades": len(pnls), "net_pnl": round(sum(pnls), 2),
            "net_expectancy": round(statistics.mean(pnls), 3),
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "win_rate": round(len(wins) / len(pnls), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2020-11-01")
    ap.add_argument("--end", default="2025-11-01")
    ap.add_argument("--days-per-quarter", type=int, default=3)
    ap.add_argument("--train-end", default="2023-05-01")
    ap.add_argument("--of-gate-score", type=float, default=55.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    train_end = date.fromisoformat(args.train_end)
    dates = sample_dates(start, end, args.days_per_quarter)
    print(f"[di-bb-E] {len(SYMBOLS)} symbols x {len(dates)} sampled tick days (GENUINE IEX ticks)")

    D = {"is": [], "oos": []}; E = {"is": [], "oos": []}
    nd = 0
    for d in dates:
        for sym in SYMBOLS:
            data = fetch_day(None, sym, d, fetch_quotes=False)  # cache-only (no client)
            if data is None:
                continue
            nd += 1
            dt, et = replay_E(sym, data, args.of_gate_score)
            b = "is" if d < train_end else "oos"
            D[b].extend(dt); E[b].extend(et)
        print(f"  {d}: symbol-days={nd}", flush=True)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "data_source": "GENUINE IEX ticks (sampled); order flow never fabricated",
              "symbol_days": nd, "of_gate_score": args.of_gate_score,
              "D_out_of_sample": metrics(D["oos"]), "E_out_of_sample": metrics(E["oos"]),
              "D_in_sample": metrics(D["is"]), "E_in_sample": metrics(E["is"])}
    eo = report["E_out_of_sample"]; do = report["D_out_of_sample"]
    ee = eo.get("net_expectancy", 0) or 0; de = do.get("net_expectancy", 0) or 0
    if eo.get("trades", 0) < 30:
        v = "INCONCLUSIVE (too few OOS trades)"
    elif ee > 0 and ee > de:
        v = "ACCEPT (order-flow ranking/protection improves DI/BB OOS, net-positive)"
    elif ee > de:
        v = "INCONCLUSIVE (order-flow helps vs D but still not net-positive)"
    else:
        v = "REJECT (order-flow ranking/protection does not improve DI/BB OOS)"
    report["verdict_E_vs_D"] = v

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"di_bb_E_backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n=== Variant E (DI/BB + order-flow) vs D, OOS, genuine ticks ({nd} symbol-days) ===")
    for k in ("trades", "net_expectancy", "profit_factor", "win_rate", "net_pnl"):
        print(f"{k:<16}{str(do.get(k)):>14}{str(eo.get(k)):>14}   (D / E)")
    print(f"\nVERDICT (E vs D): {v}")
    print(f"[di-bb-E] wrote {OUT_DIR / f'di_bb_E_backtest_{tag}.json'}")


if __name__ == "__main__":
    main()
