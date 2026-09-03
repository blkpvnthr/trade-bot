#!/usr/bin/env python3
"""premarket_levels.py — precompute structure levels + pre-trade gate for the watchlist.

Run before each session (7:00 premarket) and again after the ORB forms (~10:05). Writes:
  logs/structure/levels_<YYYYMMDD>.json   (full levels per symbol)
  logs/structure/levels_latest.json       (pointer)
  logs/structure/gate_<YYYYMMDD>.json      (GO/NO-GO decisions)

Watchlist resolution (first that applies):
  positional SYMBOLS  >  --tickers-file FILE  >  <script_dir>/options_watchlist.txt  >  QQQ,SPY,IWM
Index trio (QQQ,SPY,IWM) is always included for context. Files may be one-per-line or
comma/quote separated (e.g.  "AAPL","AMD",...). Throttled for the IEX rate limit.

NOTE on timing: 7:00 premarket run = PRIOR session's intraday (ORB/today/VWAP/POC) + all
daily levels = correct premarket reference. The 10:05 run captures TODAY's opening range.

Usage:  python premarket_levels.py                      # default options_watchlist.txt
        python premarket_levels.py --tickers-file tickers.txt
        python premarket_levels.py QQQ SPY TSLA --min-rr 1.5
"""
from __future__ import annotations
import os, sys, json, re, time, argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import structure_levels as SL
ET = ZoneInfo("America/New_York")
OUT = HERE / "logs" / "structure"
TRIO = ["QQQ", "SPY", "IWM"]


def read_tickers(path):
    """Parse one-per-line OR comma/quote-separated ticker files; drop preferred (/) + dupes."""
    toks = re.split(r"[\s,]+", Path(path).read_text())
    syms = []
    for t in toks:
        s = t.strip().strip('"').strip("'").upper()
        if not s or "/" in s:               # skip blanks + preferred-share tickers Alpaca lacks
            continue
        if all(c.isalnum() or c in ".-" for c in s):
            syms.append(s)
    return list(dict.fromkeys(syms))


def screener_watchlist(limit=100, rating_col="Recommend.Other"):
    """Dynamic watchlist from the TradingView premarket-momentum screen (tv-screener.py)."""
    import importlib.util
    p = HERE / "tv-screener.py"
    spec = importlib.util.spec_from_file_location("tv_screener_mod", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.scan_premarket_momentum(limit=limit, rating_col=rating_col, symbols_only=True, write=False)


def resolve_watchlist(args):
    syms = None
    if getattr(args, "screener", False):
        try:
            syms = screener_watchlist(args.screener_limit, args.rating_col)
            if not syms:
                print("screener returned 0 names — falling back to file watchlist")
        except Exception as e:
            print(f"screener failed ({str(e)[:80]}) — falling back to file watchlist")
            syms = None
    if not syms:
        if args.symbols:
            syms = [s.upper() for s in args.symbols]
        elif args.tickers_file and Path(args.tickers_file).exists():
            syms = read_tickers(args.tickers_file)
        elif (HERE / "options_watchlist.txt").exists():
            syms = read_tickers(HERE / "options_watchlist.txt")
        else:
            syms = list(TRIO)
    # always include the index trio for context, trio first
    syms = TRIO + [s for s in syms if s not in TRIO]
    if args.max:
        syms = syms[: args.max]
    return syms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--screener", action="store_true", help="watchlist from TradingView premarket-momentum screen")
    ap.add_argument("--screener-limit", type=int, default=100)
    ap.add_argument("--rating-col", default="Recommend.All", help="Recommend.All=Summary; Recommend.Other=Oscillators")
    ap.add_argument("--min-rr", type=float, default=1.5)
    ap.add_argument("--max", type=int, default=None, help="cap watchlist size")
    ap.add_argument("--sleep", type=float, default=0.25, help="throttle between symbols (IEX rate limit)")
    a = ap.parse_args()
    syms = resolve_watchlist(a)
    now = datetime.now(ET); tag = now.strftime("%Y%m%d")
    OUT.mkdir(parents=True, exist_ok=True)

    all_levels = []; gates = []
    for i, s in enumerate(syms):
        try:
            L = SL.levels(s)
        except Exception as e:
            L = {"symbol": s, "error": str(e)[:120]}
        all_levels.append(L); gates.append(SL.gate(L, a.min_rr))
        if a.sleep and i < len(syms) - 1:
            time.sleep(a.sleep)

    payload = {"generated_at": now.isoformat(), "min_rr": a.min_rr, "count": len(syms),
               "levels": all_levels, "gates": gates}
    (OUT / f"levels_{tag}.json").write_text(json.dumps(payload, indent=2))
    (OUT / "levels_latest.json").write_text(json.dumps(payload, indent=2))
    (OUT / f"gate_{tag}.json").write_text(json.dumps(gates, indent=2))

    go = [d for d in gates if d.get("go")]
    err = [d for d in gates if d.get("reasons") and "error" in str(d["reasons"][0]).lower()]
    print(f"PREMARKET LEVELS  {now:%Y-%m-%d %H:%M %Z}  watchlist={len(syms)}  GO={len(go)}  err={len(err)}  (min R:R {a.min_rr})")
    for d in go:
        print(f"  [GO]  {d['symbol']:<6} ${d.get('price')}  {d['bias']} entry ${d['entry']} TP ${d['tp']} stop ${d['stop']} R:R {d['rr']}")
    if not go:
        print("  (no GO setups — all mixed/insufficient R:R)")
    print(f"saved -> logs/structure/levels_{tag}.json  ({len(syms)} symbols)")
    print("PREMARKET_LEVELS_DONE")


if __name__ == "__main__":
    main()
