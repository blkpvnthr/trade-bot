#!/usr/bin/env python3
"""Backtest for the intraday reversal engine on cached 1-minute bars.

Replays the full live decision path per symbol-day at the afternoon scan time
(default 1:30 PM ET):
  screener-style universe row  ->  enrich with the intraday bars (derive VWAP/ORB/
  sweep/RSI/EMA20/ATR/PDH via intraday_reversal_enrichment)  ->  score_candidate
  ->  if eligible, simulate the trade: enter at the scan close, target the Previous
  Day High, trail the underlying (default 0.75%), and carry overnight to sell into
  the next day's high when the position is up at the close (the engine's EOD rule).

Universe gates that ARE computable from bars are enforced (price>5, vol>3M,
chg<0 = red today, perf_1w<0). The analyst/oscillator "strong buy" gate is
NEUTRALIZED — the cache has no historical ratings — so this measures the PRICE
setup + PDH-target trade management, not the fundamental overlay. Gross returns
(no slippage/fees/borrow); stops fill at the level, not the gap.

Usage:
  python backtest_reversal_engine.py                     # all cached symbols
  python backtest_reversal_engine.py --max-symbols 40 --config-sweep
  python backtest_reversal_engine.py --decision-et 13:30 --no-overnight
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backtest_run_v2 import load_symbol_bars, perf, ET, RTH_OPEN_ET, RTH_CLOSE_ET
from intraday_reversal_engine import ReversalInput, score_candidate, EngineConfig
from intraday_reversal_enrichment import enrich_input


def _daily_series(et_bars: pd.DataFrame) -> pd.DataFrame:
    """Per-RTH-day OHLCV aggregates (index = ET date)."""
    rth = et_bars.between_time(RTH_OPEN_ET, RTH_CLOSE_ET)
    g = rth.groupby(rth.index.date)
    return pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })


def simulate_trade(post: pd.DataFrame, next_post: Optional[pd.DataFrame], entry: float,
                   pdh: float, trail_pct: float, next_day_trail_pct: float,
                   carry_overnight: bool) -> tuple[float, str]:
    """Walk the post-decision bars: exit at the PDH target, the high-water trailing stop,
    or — if still up at the close and carrying overnight — sell into the next day's high
    (trail off the running high). Returns (return_fraction, exit_reason)."""
    peak = entry
    for h, l in zip(post["high"], post["low"]):
        peak = max(peak, float(h))
        if float(h) >= pdh:
            return (pdh - entry) / entry, "target_pdh"
        stop = peak * (1.0 - trail_pct)
        if float(l) <= stop:
            return (stop - entry) / entry, "trail_stop"
    close_px = float(post["close"].iloc[-1])
    if not (carry_overnight and next_post is not None and len(next_post) and close_px > entry):
        return (close_px - entry) / entry, "eod_close"
    # Overnight carry: sell into the next day's high (PDH target still active).
    npeak = close_px
    for h, l in zip(next_post["high"], next_post["low"]):
        npeak = max(npeak, float(h))
        if float(h) >= pdh:
            return (pdh - entry) / entry, "target_pdh_d2"
        nstop = npeak * (1.0 - next_day_trail_pct)
        if float(l) <= nstop:
            return (nstop - entry) / entry, "next_day_trail"
    return (float(next_post["close"].iloc[-1]) - entry) / entry, "d2_close"


def run_symbol(symbol: str, df: pd.DataFrame, dec_t: time, cfg: EngineConfig,
               carry_overnight: bool) -> list[dict[str, Any]]:
    et = df.tz_convert(ET)
    daily = _daily_series(et)
    dates = list(daily.index)
    by_date = {d: g for d, g in et.groupby(et.index.date)}
    trades: list[dict[str, Any]] = []
    for i in range(5, len(dates)):
        d, prev = dates[i], dates[i - 1]
        rth = by_date[d].between_time(RTH_OPEN_ET, RTH_CLOSE_ET)
        pre = rth.between_time(RTH_OPEN_ET, dec_t)
        post = rth.between_time(dec_t, RTH_CLOSE_ET)
        if len(pre) < 20 or len(post) < 5:
            continue
        price_t = float(pre["close"].iloc[-1])
        prev_close = float(daily.loc[prev, "close"])
        pdh = float(daily.loc[prev, "high"])
        wk_close = float(daily.iloc[i - 5]["close"])
        if prev_close <= 0 or wk_close <= 0:
            continue
        chg = (price_t - prev_close) / prev_close * 100.0
        perf_1w = (price_t - wk_close) / wk_close * 100.0
        cum_vol = float(pre["volume"].sum())
        avg_vol = float(daily.iloc[max(0, i - 10):i]["volume"].mean())

        inp = ReversalInput(
            symbol=symbol, price=price_t, volume=cum_vol, avg_volume=avg_vol,
            chg_pct=chg, perf_1w=perf_1w, analyst_rating="strong buy", os_rating="strong buy",
            vol_5m=float(pre["volume"].tail(5).sum()), prev_day_high=pdh)
        daily_hist = daily.iloc[max(0, i - 20):i][["high", "low", "close"]].reset_index(drop=True)
        enrich_input(inp, pre, daily_hist, orb_minutes=cfg_orb(), bars_per_min=1.0)
        c = score_candidate(inp, cfg)
        if not c.eligible:
            continue
        next_post = (by_date[dates[i + 1]].between_time(dec_t, RTH_CLOSE_ET)
                     if i + 1 < len(dates) else None)
        ret, reason = simulate_trade(post, next_post, price_t, pdh, cfg.trailing_stop_pct,
                                     cfg.next_day_high_trail_pct, carry_overnight)
        trades.append({"symbol": symbol, "date": str(d), "score": c.score, "ret": ret,
                       "exit": reason, "chg": chg, "perf_1w": perf_1w,
                       "rel_vol_5m": c.rel_vol_5m})
    return trades


def cap_top_n_per_day(trades: list[dict], n: Optional[int]) -> list[dict]:
    """Keep only the top-N most-relatively-active (rel_vol_5m desc) eligible names per day,
    mirroring the live universe sort + per-day entry cap. None == uncapped."""
    if not n:
        return trades
    by_day: dict[str, list[dict]] = {}
    for t in trades:
        by_day.setdefault(t["date"], []).append(t)
    out: list[dict] = []
    for ts in by_day.values():
        ts.sort(key=lambda t: -(t.get("rel_vol_5m") or 0.0))
        out.extend(ts[:n])
    return out


_ORB_MIN = 15


def cfg_orb() -> int:
    return _ORB_MIN


def evaluate(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {"n": 0}
    r = pd.Series([t["ret"] for t in trades])
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["exit"]] = reasons.get(t["exit"], 0) + 1
    out = perf(r)
    out["exit_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    out["pct_hit_pdh"] = round(sum(1 for t in trades if t["exit"].startswith("target_pdh"))
                               / len(trades), 4)
    return out


def render_report(base: dict, sweep: Optional[list], n_symbols: int, n_days: int,
                  avg_elig: float, cfg: EngineConfig, decision_et: str, carry: bool,
                  top_n: Optional[int]) -> str:
    L = ["# Intraday reversal engine backtest\n",
         f"- Generated: {datetime.now(ET).isoformat()}",
         f"- Universe: {n_symbols} cached symbols | {n_days} trading days | "
         f"avg {avg_elig} eligible/day",
         f"- Scan @ {decision_et} ET | target = Previous Day High | trail {cfg.trailing_stop_pct*100:.2f}%"
         f" | overnight carry: {carry} | **rel-vol top-N/day: {top_n if top_n else 'uncapped'}**",
         "- Gates from bars: price>5, vol>3M, chg>0 (GREEN today), perf_1w<0. Analyst/OS "
         "'strong buy' NEUTRALIZED (no historical ratings in cache) — this tests the price "
         "setup + PDH-target management, not the fundamental overlay.\n",
         "## Realized trade performance (rel-vol top-N names per day)\n",
         "| metric | value |", "|---|---|"]
    for k in ("n", "hit_rate", "avg_ret_bps", "median_bps", "profit_factor", "t_stat",
              "avg_win_bps", "avg_loss_bps", "pct_hit_pdh"):
        L.append(f"| {k} | {base.get(k)} |")
    L.append("")
    L.append(f"- **Exit reasons:** {base.get('exit_reasons')}")
    L.append("- target_pdh = hit the Previous Day High; trail_stop = trailing stop; eod_close = "
             "flat-or-down at the close; *_d2 / next_day_trail = overnight carry sold into the "
             "next day's high.\n")
    if sweep:
        L.append("## rel-vol top-N-per-day sensitivity\n")
        L.append("Keep only the N most-relatively-active (rel_vol_5m desc) eligible names each day.\n")
        L.append("| top_N/day | n | avg bps | PF | t-stat | hit-rate | % hit PDH |")
        L.append("|---|---|---|---|---|---|---|")
        for s in sweep:
            e = s["eval"]
            L.append(f"| {s['top_n'] if s['top_n'] else 'uncapped'} | {e.get('n')} | "
                     f"{e.get('avg_ret_bps')} | {e.get('profit_factor')} | {e.get('t_stat')} | "
                     f"{e.get('hit_rate')} | {e.get('pct_hit_pdh')} |")
        L.append("")
    L.append("## Caveats\n")
    L.append("- Gross returns (no slippage/fees/borrow); trailing stops fill AT the level, not "
             "through overnight/intrabar gaps (optimistic on gap-downs).")
    L.append("- The analyst/OS fundamental gate is not modeled; live, it narrows the universe to "
             "fundamentally-liked names, which can change the realized edge materially.")
    L.append("- Entry at the decision-bar close (the live engine uses a marketable limit).")
    return "\n".join(L)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backtest the intraday reversal engine on cached bars")
    p.add_argument("--bars-dir", default="backtest_cache")
    p.add_argument("--decision-et", default="13:30", help="afternoon scan time (ET)")
    p.add_argument("--orb-minutes", type=int, default=15)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--no-overnight", action="store_true", help="disable the overnight carry")
    p.add_argument("--top-n", type=int, default=3,
                   help="keep only the top-N most-relatively-active names per day (0 = uncapped)")
    p.add_argument("--config-sweep", action="store_true")
    p.add_argument("--out-dir", default="backtest_outputs")
    args = p.parse_args(argv)
    global _ORB_MIN
    _ORB_MIN = args.orb_minutes
    carry = not args.no_overnight
    dec_t = time.fromisoformat(args.decision_et)

    print(f"[1/3] loading bars from {args.bars_dir} ...")
    bars = load_symbol_bars(args.bars_dir, args.max_symbols)
    print(f"      {len(bars)} symbols")
    if not bars:
        print("no bars — nothing to backtest")
        return 2

    print(f"[2/3] replaying reversal engine (scan @ {args.decision_et} ET) ...")
    base_cfg = EngineConfig()
    all_trades: list[dict] = []
    for sym, df in bars.items():
        all_trades.extend(run_symbol(sym, df, dec_t, base_cfg, carry))
    per_day: dict[str, int] = {}
    for t in all_trades:
        per_day[t["date"]] = per_day.get(t["date"], 0) + 1
    avg_elig = round(len(all_trades) / max(1, len(per_day)), 2)

    top_n = args.top_n or None
    base_eval = evaluate(cap_top_n_per_day(all_trades, top_n))

    print("[3/3] rel-vol top-N-per-day sweep ...")
    sweep = [{"top_n": n, "eval": evaluate(cap_top_n_per_day(all_trades, n))}
             for n in (1, 2, 3, 5, 10, None)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": {"min_score": base_cfg.min_score, "trail": base_cfg.trailing_stop_pct,
                          "decision_et": args.decision_et, "orb_minutes": args.orb_minutes,
                          "overnight": carry, "top_n": top_n},
               "n_symbols": len(bars), "n_trading_days": len(per_day),
               "avg_eligible_per_day": avg_elig, "uncapped_trades": len(all_trades),
               "evaluation": base_eval, "rel_vol_top_n_sweep": sweep}
    (out_dir / "reversal_backtest.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "reversal_backtest.md").write_text(
        render_report(base_eval, sweep, len(bars), len(per_day), avg_elig, base_cfg,
                      args.decision_et, carry, top_n), encoding="utf-8")

    print("\n" + "=" * 70)
    e = base_eval
    print(f"rel-vol top-N={top_n}  (avg {avg_elig} eligible/day over {len(per_day)} days)")
    print(f"trades={e.get('n')}  hit={e.get('hit_rate')}  avg={e.get('avg_ret_bps')}bps  "
          f"PF={e.get('profit_factor')}  t={e.get('t_stat')}  %hitPDH={e.get('pct_hit_pdh')}")
    print(f"exits: {e.get('exit_reasons')}")
    print(f"report: {out_dir / 'reversal_backtest.md'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
