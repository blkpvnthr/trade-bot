#!/usr/bin/env python3
"""
signal_fidelity_report.py — is the LIVE bot producing the same signals as the backtest?
=======================================================================================

The meaningful validation isn't whether live P&L matches the backtest curve — it's whether
run.py *fires the same signals in real time*, and where it diverges. This replays the
structured accept/reject ledgers run.py writes each cycle:

    logs/execution/<YYYYMMDD>/<YYYYMMDD>_accepted_trades.jsonl   (every EXECUTED order)
    logs/execution/<YYYYMMDD>/<YYYYMMDD>_skipped_trades.jsonl    (every rejection + reason)

and classifies each rejection so you can tell signal divergence (a coding/config bug) from
execution-reality divergence (regime/risk/liquidity blocking a real signal):

    STRATEGY_FILTER  — gates that ALSO exist in the backtest (vol, ADV, freshness, trend,
                       setup quality). These legitimately reduce signals the same way the
                       backtest does — NOT a missed signal.
    LIVE_BLOCK       — live-only constraints absent from the idealized backtest (daily-loss
                       halt, kill switch, exposure cap, cash buffer, regime halt, order size).
                       A real signal the strategy WANTED but live reality stopped.

  intended signals = executed + LIVE_BLOCK     (what the strategy tried to do)
  execution rate   = executed / intended signals

If intended-signal COUNTS line up with the backtest but P&L differs, the gap is execution &
market reality, not a bug. If the counts DON'T line up, the implementation has drifted.

Realized win-rate / return-per-trade come from broker round-trips (run on the droplet);
max-DD / Sharpe come from the executor equity curves. Backtest column is read from
backtest_reference.json (written by tearsheet_insider_blend.py). Pure stdlib; fail-safe.

Usage:  python signal_fidelity_report.py [--days 21] [--engine insider_copy]
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXEC_DIR = ROOT / "logs" / "execution"

# Gates that ALSO constrain the backtest's signal set — rejections here are NOT missed signals.
STRATEGY_FILTER = {
    "rejected_by_setup_quality", "rejected_by_momentum", "rejected_by_valuation",
    "rejected_by_fundamentals", "rejected_by_macro", "rejected_by_macro_cycle",
    "rejected_by_analyst_downside", "rejected_no_trend_crossover", "rejected_not_9ema_dip",
    "rejected_by_below_sma50", "rejected_by_below_9ema", "vol_gate_short_history",
    "rejected_market_regime_chop_or_reversal", "rejected_by_liq_sweep_no_buy",
}
# Live-only constraints absent from the idealized backtest — a blocked REAL signal.
LIVE_BLOCK = {
    "rejected_by_market_regime", "rejected_by_regime", "halt_new_buys",
    "rejected_by_daily_loss", "rejected_by_kill_switch", "rejected_by_drawdown",
    "rejected_by_equity_floor", "rejected_by_exposure_limit", "rejected_by_order_size",
    "rejected_by_cash_buffer", "rejected_by_options_buying_power",
    "rejected_by_options_liquidity", "rejected_by_option_liquidity",
    "rejected_by_session", "rejected_by_cycle", "rejected_by_data_unavailable",
    "throttled_by_self_assessment",
}


def _base(reason: str) -> str:
    return str(reason or "").split(":", 1)[0]


def _engine_of_skip(rec: dict) -> str:
    detail = rec.get("detail", {}) or {}
    phase = str(detail.get("phase", "")).lower()
    if "insider" in phase or str(rec.get("symbol", "")).upper() == "INSIDER_COPIER":
        return "insider_copy"
    if "buffett" in phase or "dca" in phase:
        return "buffett_dca"
    r = str(rec.get("reason", "")).lower()
    if "dca" in r or "buffett" in r:
        return "buffett_dca"
    return "other"


def _load(days: int):
    start_tag = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    accepted, skipped, equity_pts = [], [], []
    if not EXEC_DIR.exists():
        return accepted, skipped, equity_pts, start_tag
    for d in sorted(EXEC_DIR.glob("*")):
        if not (d.is_dir() and d.name.isdigit() and d.name >= start_tag):
            continue
        for f in d.glob("*_accepted_trades.jsonl"):
            for line in f.read_text(errors="replace").splitlines():
                try:
                    accepted.append(json.loads(line))
                except Exception:
                    pass
        for f in d.glob("*_skipped_trades.jsonl"):
            for line in f.read_text(errors="replace").splitlines():
                try:
                    skipped.append(json.loads(line))
                except Exception:
                    pass
        for f in d.glob("*_executor_equity_curve.csv"):
            for ln in f.read_text(errors="replace").splitlines()[1:]:
                parts = ln.split(",")
                for p in parts:
                    try:
                        v = float(p)
                        if v > 1.0:
                            equity_pts.append(v)
                            break
                    except Exception:
                        continue
    return accepted, skipped, equity_pts, start_tag


def _max_dd_sharpe(equity: list[float]):
    if len(equity) < 3:
        return None, None
    peak = equity[0]
    mdd = 0.0
    rets = []
    for i, v in enumerate(equity):
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
        if i > 0 and equity[i - 1] > 0:
            rets.append(v / equity[i - 1] - 1.0)
    if len(rets) < 2:
        return mdd, None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = var ** 0.5
    sharpe = (mean / sd * (252 ** 0.5)) if sd > 0 else None
    return mdd, sharpe


def _realized_round_trips(days: int):
    """win-rate + avg return/trade from broker FIFO round-trips (best-effort; needs creds)."""
    try:
        import os
        import order_emailer
        from alpaca.trading.client import TradingClient
        for ln in (ROOT / ".env").read_text().splitlines():
            s = ln.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        tc = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=False)
        rts = order_emailer.round_trips(order_emailer.fetch_fills(tc, days=max(days, 30)))
        if not rts:
            return 0, None, None
        wins = [r for r in rts if float(r.get("pnl", r.get("realized_pl", 0)) or 0) > 0]
        rets = [float(r.get("pnl_pct", r.get("return_pct", 0)) or 0)
                for r in rts if (r.get("pnl_pct") is not None or r.get("return_pct") is not None)]
        wr = len(wins) / len(rts) if rts else None
        avg = sum(rets) / len(rets) / 100.0 if rets else None  # pct -> fraction
        return len(rts), wr, avg
    except Exception as e:
        return None, None, str(e)[:60]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--engine", default=None, help="filter: insider_copy | buffett_dca | trump_copy")
    args = ap.parse_args()

    accepted, skipped, equity, start_tag = _load(args.days)
    if args.engine:
        accepted = [a for a in accepted if a.get("engine") == args.engine]
        skipped = [s for s in skipped if _engine_of_skip(s) == args.engine]

    # Meta-symbols are COPIER-LEVEL gates logged EVERY cycle (not per-candidate signals) — exclude
    # them and dedupe by (symbol, day) so a name evaluated N cycles/day counts as ONE signal/day.
    META = {"INSIDER_COPIER", "UNIVERSE", "ALPHA_ENGINE", "TRUMP_TRACKER", "BAL"}

    def _dedup(records):
        seen = {}
        for r in records:
            sym = str(r.get("symbol", "")).upper()
            if sym in META:
                continue
            seen[(sym, r.get("date_tag", ""))] = r
        return list(seen.values())

    buys = [a for a in accepted if str(a.get("side")) == "buy"]
    cand_buys = _dedup(buys)
    cand_skipped = _dedup(skipped)
    by_engine_exec = Counter(a.get("engine", "other") for a in cand_buys)

    reason_counts = Counter(_base(s.get("reason", "")) for s in cand_skipped)
    live_block = {r: c for r, c in reason_counts.items() if r in LIVE_BLOCK}
    strat_filter = {r: c for r, c in reason_counts.items() if r in STRATEGY_FILTER}
    other = {r: c for r, c in reason_counts.items() if r not in LIVE_BLOCK and r not in STRATEGY_FILTER}

    # Distinct DAYS the whole copier was halted on regime before it could evaluate candidates.
    halt_days = sorted({s.get("date_tag") for s in skipped
                        if str(s.get("symbol", "")).upper() in META
                        and "regime" in str(s.get("reason", "")).lower()})

    n_exec = len(cand_buys)
    n_live_block = sum(live_block.values())
    n_intended = n_exec + n_live_block
    exec_rate = (n_exec / n_intended) if n_intended else None

    mdd, sharpe = _max_dd_sharpe(equity)
    n_rt, win_rate, avg_ret = _realized_round_trips(args.days)

    ref = {}
    rp = ROOT / "backtest_reference.json"
    if rp.exists():
        try:
            ref = json.loads(rp.read_text())
        except Exception:
            ref = {}

    def cell(v, pct=False, dol=False):
        if v is None:
            return "—"
        if isinstance(v, str):
            return v
        if pct:
            return f"{v:.1%}"
        if dol:
            return f"${v:,.0f}"
        return f"{v:g}"

    print(f"\n===== SIGNAL-FIDELITY: LIVE vs BACKTEST (last {args.days}d"
          f"{', engine='+args.engine if args.engine else ''}) =====")
    print(f"window: {start_tag} → today   |   ledger dirs: {EXEC_DIR}")
    print(f"\n{'Metric':<26}{'Backtest':>16}{'Live':>16}")
    print("-" * 58)
    print(f"{'Intended signals':<26}{cell(ref.get('signals')):>16}{cell(n_intended):>16}")
    print(f"{'  └ executed':<26}{'—':>16}{cell(n_exec):>16}")
    print(f"{'  └ blocked (live)':<26}{'—':>16}{cell(n_live_block):>16}")
    print(f"{'Execution rate':<26}{'100%':>16}{cell(exec_rate, pct=True):>16}")
    _slip_bt = f"{ref.get('slippage_bps_per_side','?')} bps" if ref else "—"
    print(f"{'Avg entry delay':<26}{'N/A':>16}{'pending fills':>16}")
    print(f"{'Avg slippage':<26}{_slip_bt:>16}{'pending fills':>16}")
    print(f"{'Win rate':<26}{'historical':>16}{cell(win_rate, pct=True):>16}")
    print(f"{'Avg return / trade':<26}{'historical':>16}{cell(avg_ret, pct=True):>16}")
    print(f"{'Max drawdown':<26}{cell(ref.get('max_dd'), pct=True):>16}{cell(mdd, pct=True):>16}")
    print(f"{'Sharpe':<26}{cell(ref.get('sharpe')):>16}{cell(sharpe):>16}")
    print(f"{'CAGR':<26}{cell(ref.get('cagr'), pct=True):>16}{'(too short)':>16}")

    if halt_days:
        print(f"\nNOTE: the whole copier was REGIME-HALTED (couldn't evaluate candidates) on "
              f"{len(halt_days)} day(s): {', '.join(halt_days[-7:])}{' …' if len(halt_days) > 7 else ''}")
    print(f"signals are DISTINCT (symbol, day) — per-candidate, meta-symbols excluded.")
    print("\n--- WHY intended signals didn't execute (LIVE_BLOCK reasons) ---")
    if live_block:
        for r, c in sorted(live_block.items(), key=lambda x: -x[1]):
            print(f"  {c:>5}  {r}")
    else:
        print("  (none)")
    print("\n--- strategy-filter rejections (expected; also constrain the backtest) ---")
    for r, c in sorted(strat_filter.items(), key=lambda x: -x[1])[:8]:
        print(f"  {c:>5}  {r}")
    if other:
        print("\n--- uncategorized reasons (review) ---")
        for r, c in sorted(other.items(), key=lambda x: -x[1])[:8]:
            print(f"  {c:>5}  {r}")
    print(f"\nexecuted by engine: {dict(by_engine_exec)}")
    if isinstance(avg_ret, str) or n_rt is None:
        print(f"note: realized win-rate/return need broker creds (run on the droplet).")
    print("\nINTERPRETATION: if intended-signal counts track the backtest's signal rate but P&L\n"
          "differs, the gap is execution/market reality. If counts diverge, the implementation\n"
          "has drifted from the backtested logic — inspect the LIVE_BLOCK reasons above.\n")


if __name__ == "__main__":
    main()
