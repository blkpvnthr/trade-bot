#!/usr/bin/env python3
"""trade_log.py — Kelly-based options trade log (expected vs actual probabilities).

Implements the book's logging discipline: record expected PoP + IV + Kelly exit levels at
ENTRY, then log the exit + realized PoP, so you can compare expected vs actual over time and
refine your Kelly profit/loss exits.

Kelly break-even (from the book):
    R_be = (1 - W) / W          # break-even ratio = avgWinner/avgLoser, W = prob of profit
    given a profit-take fraction f (of credit), break-even loss fraction L = f / R_be
    take losses with `edge` room before break-even to be positive-expectancy.

CSV: logs/trade_log.csv (the 13 fields). Use as a library or CLI (entry / exit / report).
"""
from __future__ import annotations
import os, csv, argparse
from datetime import date, datetime
from pathlib import Path

ROOT = Path("/Users/blkpvnthr/Desktop/trader")
LOG = ROOT / "logs" / "trade_log.csv"
FIELDS = ["entry_date","symbol","strategy","risk_type","underlying_px","strikes","contracts",
          "pop","atm_iv","credit","profit_take_pct","loss_take_pct","tp_price","sl_price",
          "exit_date","days_in_trade","exit_price","realized_pnl","outcome","actual_vs_expected"]


def kelly_levels(pop, profit_take_pct, edge=0.10):
    """Return break-even R, break-even loss fraction, and edge-adjusted loss fraction.
    pop=W (0-1), profit_take_pct=f (fraction of credit taken as profit, e.g. 0.50)."""
    W = pop
    R_be = (1 - W) / W if W > 0 else 9.99
    loss_be = profit_take_pct / R_be if R_be > 0 else 9.99      # break-even loss (fraction of credit)
    loss_edge = loss_be * (1 - edge)                            # take losses earlier for positive edge
    return dict(break_even_R=round(R_be, 3), break_even_loss_pct=round(loss_be, 3),
                edge_loss_pct=round(loss_edge, 3))


def exit_prices(credit, profit_take_pct, loss_take_pct):
    """Buy-back prices for TP and SL given a short credit."""
    return dict(tp_price=round(credit * (1 - profit_take_pct), 2),
                sl_price=round(credit * (1 + loss_take_pct), 2))


def _read():
    if not LOG.exists(): return []
    return list(csv.DictReader(open(LOG)))


def _write(rows):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def log_entry(symbol, strategy, risk_type, underlying_px, strikes, contracts, pop, atm_iv,
              credit, profit_take_pct=0.50, edge=0.10, entry_date=None):
    k = kelly_levels(pop, profit_take_pct, edge)
    loss_take = k["edge_loss_pct"]
    px = exit_prices(credit, profit_take_pct, loss_take)
    row = {f: "" for f in FIELDS}
    row.update(entry_date=entry_date or date.today().isoformat(), symbol=symbol, strategy=strategy,
               risk_type=risk_type, underlying_px=underlying_px, strikes=strikes, contracts=contracts,
               pop=round(pop, 4), atm_iv=round(atm_iv, 4), credit=round(credit, 2),
               profit_take_pct=profit_take_pct, loss_take_pct=round(loss_take, 3),
               tp_price=px["tp_price"], sl_price=px["sl_price"])
    rows = _read(); rows.append(row); _write(rows)
    print(f"ENTRY logged: {symbol} {strategy} credit=${credit:.2f}  PoP={pop*100:.0f}%  "
          f"-> TP buy-back ${px['tp_price']} / SL buy-back ${px['sl_price']}  "
          f"(Kelly BE-R={k['break_even_R']}, BE-loss={k['break_even_loss_pct']*100:.0f}%)")
    return row


def log_exit(symbol, exit_price, exit_date=None, entry_date=None):
    rows = _read()
    # match most recent open row for symbol
    for r in reversed(rows):
        if r["symbol"] == symbol and not r["exit_date"] and (not entry_date or r["entry_date"] == entry_date):
            ed = exit_date or date.today().isoformat()
            r["exit_date"] = ed; r["exit_price"] = round(exit_price, 2)
            d0 = datetime.fromisoformat(r["entry_date"]).date(); d1 = datetime.fromisoformat(ed).date()
            r["days_in_trade"] = (d1 - d0).days
            credit = float(r["credit"])
            pnl = (credit - exit_price) * 100 * int(r["contracts"])   # short: credit - buyback
            r["realized_pnl"] = round(pnl, 2)
            r["outcome"] = "win" if pnl > 0 else "loss"
            _write(rows)
            print(f"EXIT logged: {symbol} buyback ${exit_price:.2f}  pnl=${pnl:.0f}  "
                  f"days={r['days_in_trade']}  ({r['outcome']})")
            return r
    print(f"no open trade found for {symbol}")


def report():
    rows = [r for r in _read() if r["exit_date"]]
    if not rows:
        print("no closed trades logged yet"); return
    def stats(subset, label):
        n = len(subset)
        if not n: return
        wins = [r for r in subset if r["outcome"] == "win"]
        W = len(wins) / n
        pops = [float(r["pop"]) for r in subset]; exp_W = sum(pops) / n
        pnls = [float(r["realized_pnl"]) for r in subset]
        aw = sum(p for p in pnls if p > 0) / max(len(wins), 1)
        al = abs(sum(p for p in pnls if p <= 0) / max(n - len(wins), 1))
        R = aw / al if al > 0 else 9.99; R_be = (1 - W) / W if W > 0 else 9.99
        days = [int(r["days_in_trade"]) for r in subset if r["days_in_trade"]]
        print(f"  {label:<22} n={n} actualW={W*100:.0f}% expectedPoP={exp_W*100:.0f}% "
              f"R={R:.2f}(BE={R_be:.2f}) total=${sum(pnls):.0f} avgDays={sum(days)//len(days) if days else 0} "
              f"-> {'POSITIVE' if R>R_be else 'NEGATIVE'}-exp")
    print("TRADE LOG — expected vs actual (closed trades)")
    stats(rows, "ALL")
    for strat in sorted(set(r["strategy"] for r in rows)):
        stats([r for r in rows if r["strategy"] == strat], strat)
    for rt in sorted(set(r["risk_type"] for r in rows)):
        stats([r for r in rows if r["risk_type"] == rt], rt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("entry");
    for a in ["symbol","strategy","risk_type"]: e.add_argument(a)
    e.add_argument("--underlying-px", type=float, default=0); e.add_argument("--strikes", default="")
    e.add_argument("--contracts", type=int, default=1); e.add_argument("--pop", type=float, required=True)
    e.add_argument("--iv", type=float, default=0); e.add_argument("--credit", type=float, required=True)
    e.add_argument("--tp", type=float, default=0.50); e.add_argument("--edge", type=float, default=0.10)
    x = sub.add_parser("exit"); x.add_argument("symbol"); x.add_argument("--price", type=float, required=True)
    sub.add_parser("report")
    sub.add_parser("kelly").add_argument("--pop", type=float, required=True)
    args = ap.parse_args()
    if args.cmd == "entry":
        log_entry(args.symbol, args.strategy, args.risk_type, args.underlying_px, args.strikes,
                  args.contracts, args.pop, args.iv, args.credit, args.tp, args.edge)
    elif args.cmd == "exit":
        log_exit(args.symbol, args.price)
    elif args.cmd == "report":
        report()
    elif args.cmd == "kelly":
        for tp in (0.25, 0.50, 0.80):
            k = kelly_levels(args.pop, tp)
            print(f"PoP={args.pop*100:.0f}% TP={tp*100:.0f}%: break-even loss={k['break_even_loss_pct']*100:.0f}% "
                  f"of credit (R_be={k['break_even_R']}), take-loss-with-edge={k['edge_loss_pct']*100:.0f}%")
