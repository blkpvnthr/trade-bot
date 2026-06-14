#!/usr/bin/env python3
"""
tjl_backtest_report.py — run the TJL backtest over a universe and write
backtest_outputs/tjl_backtest.json for the dashboard /dash/backtest page.

Uses the SAME deterministic engine as tjl_signal/tjl_backtest (no LLM). Per-symbol
stats + an aggregate (trades, win rate, total return %, profit factor) + a no-lookahead
flag. Needs Alpaca data keys in the environment (paper keys are fine — read-only bars).

Usage: python tjl_backtest_report.py [SYM,SYM,...] [days]
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import tjl_signal as T
import tjl_backtest as B

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "backtest_outputs" / os.environ.get("TJL_OUT_NAME", "tjl_backtest.json")

UNIVERSE = (sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1]
            else ["AMD", "NVDA", "MU", "AAPL", "MSFT", "TSLA", "AMZN", "META", "AVGO", "GOOGL"])
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def main():
    cfg = T.TJLConfig.from_env()
    results = []   # (stat_dict, [ret_pct, ...], source)
    for sym in UNIVERSE:
        try:
            trades, st = B.backtest(sym, cfg, DAYS)
            sd = {k: st[k] for k in (
                "symbol", "source", "entry_bars", "trades", "wins",
                "win_rate", "total_ret_pct", "profit_factor", "lookahead_ok")}
            sd["win_rate"] = round(sd["win_rate"], 1)
            sd["total_ret_pct"] = round(sd["total_ret_pct"], 2)
            pf = sd["profit_factor"]   # inf (all-win) OR 0-trade => None (renders ∞ / dropped)
            sd["profit_factor"] = (None if sd["trades"] == 0 or pf is None or pf == float("inf")
                                   else round(pf, 2))
            results.append((sd, [(tr.exit_t, tr.ret_pct) for tr in trades], st.get("source")))
        except Exception as e:  # one bad symbol never sinks the report
            results.append(({"symbol": sym, "source": "error", "entry_bars": 0, "trades": 0,
                             "wins": 0, "win_rate": 0.0, "total_ret_pct": 0.0,
                             "profit_factor": 0.0, "lookahead_ok": True, "error": str(e)[:160]},
                            [], "error"))
    # VIABLE = names where Alpaca returned real bars; drop synthetic-fallback / errors
    # (not backtestable on real data, so excluded from the report + aggregate).
    viable = [(sd, rets) for sd, rets, src in results if src == "alpaca"]
    nonviable = [sd["symbol"] for sd, rets, src in results if src != "alpaca"]
    no_setups = [sd["symbol"] for sd, rets in viable if sd["trades"] == 0]   # viable data, 0 signals
    # PF floor (TJL_MIN_PF, default 0 = keep all): drop names with profit factor below it.
    # None PF = all-win / inf -> always kept (it clears any floor).
    min_pf = float(os.environ.get("TJL_MIN_PF", "0") or 0)
    def _keep(sd):
        pf = sd.get("profit_factor")
        return sd["trades"] > 0 and (pf is None or pf >= min_pf)
    dropped_pf = [sd["symbol"] for sd, rets in viable if sd["trades"] > 0 and not _keep(sd)]
    traded = [sd for sd, rets in viable if _keep(sd)]
    per = traded
    all_rets = [r for sd, rets in viable if _keep(sd) for (_t, r) in rets]

    # portfolio equity curve = cumulative SUM of per-trade returns across kept names,
    # ordered by exit time (equal-weight / additive, NOT compounded; ends at total_ret_pct).
    _tp = sorted((p for sd, rets in viable if _keep(sd) for p in rets), key=lambda x: x[0])
    equity_curve = [{"x": 0.0, "v": 0.0}]
    eq_final = eq_maxdd = 0.0
    eq_start = eq_end = None
    if _tp:
        t0, t1 = _tp[0][0], _tp[-1][0]
        span = (t1 - t0).total_seconds() or 1.0
        cum = peak = 0.0
        for tt, r in _tp:
            cum += r
            peak = max(peak, cum)
            eq_maxdd = max(eq_maxdd, peak - cum)
            equity_curve.append({"x": round((tt - t0).total_seconds() / span, 4), "v": round(cum, 2)})
        eq_final = round(cum, 2)
        eq_start = t0.astimezone(ET).date().isoformat()
        eq_end = t1.astimezone(ET).date().isoformat()

    wins = sum(1 for r in all_rets if r > 0)
    gw = sum(r for r in all_rets if r > 0)
    gl = -sum(r for r in all_rets if r <= 0)
    agg = {
        "symbols": len(per),
        "symbols_traded": len(traded),
        "trades": len(all_rets),
        "wins": wins,
        "win_rate": round(wins / len(all_rets) * 100, 1) if all_rets else 0.0,
        "total_ret_pct": round(sum(all_rets), 2),
        "avg_ret_pct": round(sum(all_rets) / len(all_rets), 3) if all_rets else 0.0,
        "profit_factor": (round(gw / gl, 2) if gl > 0 else None),  # None => inf, rendered as ∞
        "no_lookahead": all(p.get("lookahead_ok", True) for p in per),
    }

    _exits = (f"{cfg.stop_pct*100:.1f}% initial stop, {cfg.trail_gap_pct*100:.1f}% trailing stop "
              f"{'armed after a close above the prior-day high' if cfg.trail_require_pdh else 'from entry'}"
              f"{', hold overnight' if cfg.swing_mode else ''}"
              if cfg.use_trail
              else f"{cfg.stop_pct*100:.1f}% stop / {cfg.tp_pct*100:.1f}% TP"
                   f"{'' if cfg.swing_mode else ' / EOD-flat'}")
    _caveats = [
        "Long-only momentum-into-trend, non-repainting (HTF filters use last COMPLETED bar).",
        f"Exits: {_exits}. Entries gated to RTH (09:30-16:00 ET).",
        "Backtest fills: stop at bar low, TP at bar high (stop assumed first when both touch).",
        "Data = Alpaca IEX bars; this is a behavior snapshot, NOT a validated edge or live-P&L claim.",
        "Bar-for-bar parity vs the TradingView Pine source was NOT fully confirmed (see TJL notes).",
    ]
    if cfg.swing_mode:
        _caveats.append("Overnight holds carry GAP risk — a % stop does not protect against a gap.")
    if min_pf > 0 and dropped_pf:
        _caveats.append(f"Dropped {len(dropped_pf)} name(s) with PF < {min_pf:g} "
                        f"({', '.join(dropped_pf)}) — ex-post winner selection, NOT tradeable as shown.")

    doc = {
        "strategy": "Trend Join Long (TJL)",
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "days": DAYS,
        "data_source": "alpaca",   # per_symbol/aggregate are viable (real-bar) names only
        "config": {
            "trend_tf": cfg.trend_tf, "trend_sma_len": cfg.trend_sma_len,
            "momo_tf": cfg.momo_tf, "stoch": f"{cfg.stoch_len}/{cfg.smooth_k}/{cfg.smooth_d}<{cfg.stoch_ob}",
            "ema": f"{cfg.fast_ema}/{cfg.slow_ema}", "entry_tf": "5m",
            "stop_pct": round(cfg.stop_pct * 100, 2), "tp_pct": round(cfg.tp_pct * 100, 2),
            "eod_flat": cfg.eod_flat, "swing_mode": cfg.swing_mode, "use_trail": cfg.use_trail,
            "trail_gap_pct": round(cfg.trail_gap_pct * 100, 2),
            "trail_require_pdh": cfg.trail_require_pdh,
            "exit_mode": ((("swing+trail" + ("(PDH)" if cfg.trail_require_pdh else ""))
                           if cfg.swing_mode and cfg.use_trail
                           else "swing" if cfg.swing_mode
                           else "trail" if cfg.use_trail else "intraday+TP")),
        },
        "universe_requested": UNIVERSE,
        "universe": [p["symbol"] for p in per],   # viable (real Alpaca data) only
        "non_viable": nonviable,                  # dropped: no real bars (synthetic/error)
        "no_setups": no_setups,                   # viable data but 0 TJL signals in window
        "dropped_pf": dropped_pf,                 # traded but dropped by the PF floor (TJL_MIN_PF)
        "min_pf": min_pf,
        "equity_curve": equity_curve,             # [{x:0..1 along time, v:cum sum of trade %}]
        "equity": {"final_pct": eq_final, "max_drawdown_pct": round(eq_maxdd, 2),
                   "start_date": eq_start, "end_date": eq_end, "points": len(equity_curve)},
        "aggregate": agg,
        "per_symbol": sorted(traded, key=lambda p: (
            -(p["profit_factor"] if p["profit_factor"] is not None else 1e9), -p["trades"])),
        "caveats": _caveats,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT}")
    print(f"  {agg['symbols']} symbols · {agg['trades']} trades · win {agg['win_rate']}% · "
          f"total {agg['total_ret_pct']}% · PF {agg['profit_factor']} · no_lookahead={agg['no_lookahead']}")


if __name__ == "__main__":
    main()
