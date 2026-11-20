#!/usr/bin/env python3
"""
backtest_green_momentum.py -- backtest GREEN_MOMENTUM_945 variants V0..V5 on actual 5-min bars.

Variants (same universe/period/costs/execution):
    V0 Baseline          : green + VWAP                              + fixed $0.50 trail
    V1 Momentum          : green + VWAP + rising-5m                  + fixed $0.50 trail
    V2 Breakout          : green + VWAP + opening-range breakout     + fixed $0.50 trail
    V3 Combined          : green + VWAP + momentum + breakout        + fixed $0.50 trail
    V4 Adaptive          : green + VWAP                              + 1x ATR(14) trail
    V5 Combined Adaptive : green + VWAP + momentum + breakout        + 1x ATR(14) trail

Reuses green_momentum_945.py (the SAME logic trade.py runs). Actual Alpaca IEX 5-min bars
(single-exchange proxy). No look-ahead: a completed-bar signal fills at the NEXT bar's open;
ATR/VWAP/opening-range use only completed, in-session information. Intrabar stop ambiguity is
modeled CONSERVATIVELY (a bar's low is tested against the stop BEFORE that bar's high can raise
the trail; gap-through fills at the bar open). Costs via costs.py buckets + slippage, both sides.
Walk-forward: train < --train-end < test (OOS). Reports per-ticker, per-entry-hour, per
price-tier, stop-outs, MFE/MAE, and filter-rejection counts.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone, date, time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
import green_momentum_945 as gm
from backtest_dmi_adx import bucket_of, COST_BPS

BARS_CACHE = PROJECT_ROOT / "data" / "cache" / "bars_multi_tf"
OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")
ET_TZ = "America/New_York"
SLIPPAGE_BPS = 2.0

VARIANTS = {
    "V0": dict(require_momentum_confirmation=False, require_opening_range_breakout=False, stop_mode="fixed"),
    "V1": dict(require_momentum_confirmation=True, require_opening_range_breakout=False, stop_mode="fixed"),
    "V2": dict(require_momentum_confirmation=False, require_opening_range_breakout=True, stop_mode="fixed"),
    "V3": dict(require_momentum_confirmation=True, require_opening_range_breakout=True, stop_mode="fixed"),
    "V4": dict(require_momentum_confirmation=False, require_opening_range_breakout=False, stop_mode="atr"),
    "V5": dict(require_momentum_confirmation=True, require_opening_range_breakout=True, stop_mode="atr"),
}


def price_tier(px: float) -> str:
    if px < 20:
        return "low(<$20)"
    if px <= 100:
        return "mid($20-100)"
    return "high(>$100)"


def load_5m(symbol: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(BARS_CACHE / f"{symbol}_5m_*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["et"] = df["ts"].dt.tz_convert(ET_TZ)
    df["date"] = df["et"].dt.date
    df["et_time"] = df["et"].dt.time
    df = df[(df["et_time"] >= time(9, 30)) & (df["et_time"] < time(16, 0))].copy()  # RTH
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.sort_values("ts").reset_index(drop=True)


@dataclass
class GTrade:
    symbol: str
    session: Any
    entry_time: str
    entry_px: float
    exit_px: float
    net_ret: float
    gross_ret: float
    hold_min: float
    reason: str
    mfe: float
    mae: float
    tier: str
    stop_mode: str


def _session_frames(df: pd.DataFrame):
    """Yield (session_date, session_df, prev_close). prev_close = prior session's last RTH close."""
    prev_close = None
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        yield d, g, prev_close
        prev_close = float(g["close"].iloc[-1])


def _session_indicators(g: pd.DataFrame, cfg: gm.GreenMomentumConfig) -> pd.DataFrame:
    g = g.copy()
    typ = (g["high"] + g["low"] + g["close"]) / 3.0
    g["cum_vol"] = g["volume"].cumsum()
    g["vwap"] = (typ * g["volume"]).cumsum() / g["cum_vol"].replace(0, pd.NA)
    g["or_high"] = gm.opening_range_high(g, cfg.opening_range_start, cfg.opening_range_end)
    # Vectorized per-bar Wilder ATR (atr[i] uses bars 0..i) -- same result as atr_from_bars.
    h, l, c = g["high"], g["low"], g["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atr"] = tr.ewm(alpha=1.0 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    return g


def simulate_session(symbol: str, g: pd.DataFrame, prev_close: Optional[float],
                     cfg: gm.GreenMomentumConfig, cost_bps: float,
                     reject: dict) -> Optional[GTrade]:
    """One symbol, one session. Returns a GTrade or None (no entry). Updates reject counters."""
    if prev_close is None or prev_close <= 0 or len(g) < 6:
        return None
    g = _session_indicators(g, cfg)
    n = len(g)
    o = g["open"].values; h = g["high"].values; l = g["low"].values; c = g["close"].values
    vwap = g["vwap"].values; ettime = g["et_time"].values; atr = g["atr"].values
    dates = g["date"].values
    or_high = float(g["or_high"].iloc[-1]) if g["or_high"].notna().any() else float("nan")

    # ---- find the entry bar (first bar >= 09:45 whose COMPLETED-candle signal is eligible) ----
    entry_i = None
    for i in range(n - 1):            # need bar i+1 to exist for next-open fill
        now_et = datetime.combine(dates[i], ettime[i])
        if ettime[i] < time(9, 45):
            continue
        if ettime[i] >= time(15, 0):
            break
        latest5 = float(c[i]); prev5 = float(c[i - 1]) if i >= 1 else None
        diag = gm.evaluate_entry(cfg, now_et, float(c[i]), prev_close, float(vwap[i]) if vwap[i] == vwap[i] else float("nan"),
                                 latest5, prev5, or_high, already_positioned=False, pending_entry=False,
                                 entered_today=False, risk_ok=True, data_valid=True)
        if diag["eligible"]:
            entry_i = i
            break
        # reject diagnostics (count the first blocking filter)
        if not diag["within_window"]:
            pass
        elif not diag["green_on_day"]:
            reject["green"] += 1
        elif not diag["above_vwap"]:
            reject["vwap"] += 1
        elif not diag["momentum_confirmed"]:
            reject["momentum"] += 1
        elif not diag["opening_range_breakout"]:
            reject["breakout"] += 1
    if entry_i is None:
        return None

    ei = entry_i + 1                   # next-observation execution (no look-ahead)
    rt = (cost_bps + SLIPPAGE_BPS) / 1e4
    entry_px = o[ei] * (1 + rt)
    atrv = float(atr[entry_i]) if atr[entry_i] == atr[entry_i] else None
    pos = gm.new_position(symbol, entry_px, str(dates[ei]), 1.0, cfg, atrv)
    peak = entry_px; dip = entry_px
    exit_px = c[n - 1]; reason = gm.END_OF_DAY_LIQUIDATION; exit_i = n - 1
    for j in range(ei, n):
        now_et = datetime.combine(dates[j], ettime[j])
        # Conservative intrabar: test the stop against this bar's LOW before its HIGH can raise it.
        if l[j] <= pos.effective_stop and not gm.past_force_exit(now_et, cfg.force_exit):
            fill = min(pos.effective_stop, o[j])   # gap-through fills at the open
            exit_px = fill
            reason = (gm.INITIAL_PROTECTIVE_STOP if pos.effective_stop <= pos.initial_stop + 1e-9
                      else (gm.ATR_TRAILING_STOP if cfg.stop_mode == "atr" else gm.FIXED_TRAILING_STOP))
            exit_i = j; break
        # advance high-water + trailing stop with this bar's high
        atrj = float(atr[j]) if atr[j] == atr[j] else None
        ex, r, pos = gm.exit_decision(pos, float(h[j]), now_et, cfg, atrj)
        peak = max(peak, float(h[j])); dip = min(dip, float(l[j]))
        if ex and r == gm.END_OF_DAY_LIQUIDATION:
            exit_px = o[j] if j > ei else c[j]; reason = r; exit_i = j; break
    exit_px_net = exit_px * (1 - rt)
    gross = exit_px / entry_px - 1.0
    net = exit_px_net / entry_px - 1.0
    hold_min = (exit_i - ei) * 5.0
    return GTrade(symbol, g["date"].iloc[ei], str(ettime[entry_i]), entry_px, exit_px_net, net, gross,
                  hold_min, reason, peak / entry_px - 1.0, dip / entry_px - 1.0,
                  price_tier(entry_px), cfg.stop_mode)


# ---------------------------------------------------------------- metrics
def metrics(trades: list[GTrade], notional: float = 2000.0) -> dict:
    if not trades:
        return {"trades": 0}
    nets = [t.net_ret for t in trades]; gross = [t.gross_ret for t in trades]
    pnls = [r * notional for r in nets]
    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    sd = statistics.pstdev(nets) if len(nets) > 1 else 0.0
    sharpe = (statistics.mean(nets) / sd * math.sqrt(252)) if sd > 0 else 0.0
    cum = pk = mdd = 0.0
    for x in pnls:
        cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    stops = sum(1 for t in trades if t.reason in (gm.INITIAL_PROTECTIVE_STOP, gm.FIXED_TRAILING_STOP, gm.ATR_TRAILING_STOP))
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(pnls), 3),
        "avg_gross_pnl": round(statistics.mean(gross) * notional, 3),
        "avg_net_pnl": round(statistics.mean(pnls), 3),
        "net_profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "max_drawdown": round(mdd, 2),
        "net_sharpe": round(sharpe, 3),
        "avg_hold_min": round(statistics.mean([t.hold_min for t in trades]), 1),
        "stop_out_freq": round(stops / len(trades), 3),
        "avg_mfe_pct": round(statistics.mean([t.mfe for t in trades]) * 100, 3),
        "avg_mae_pct": round(statistics.mean([t.mae for t in trades]) * 100, 3),
        "total_net_pnl": round(sum(pnls), 2),
    }


def _group_metrics(trades: list[GTrade], key) -> dict:
    out = {}
    buckets: dict[Any, list] = {}
    for t in trades:
        buckets.setdefault(key(t), []).append(t)
    for k, ts in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        m = metrics(ts)
        out[str(k)] = {"trades": m["trades"], "avg_net_pnl": m["avg_net_pnl"],
                       "win_rate": m["win_rate"], "net_profit_factor": m["net_profit_factor"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,META,GOOGL")
    ap.add_argument("--train-end", default="2023-05-01")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    train_end = date.fromisoformat(args.train_end)

    # preload bars
    bars = {s: load_5m(s) for s in symbols}
    bars = {s: d for s, d in bars.items() if not d.empty}
    print(f"[gm-bt] {len(bars)} symbols with 5m bars; variants {list(VARIANTS)}; train<{train_end}")

    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                              "data_source": "Alpaca IEX 5m bars (single-exchange)",
                              "symbols": list(bars), "train_end": args.train_end,
                              "max_positions": args.max_positions, "variants": {}}
    base_cfg = gm.GreenMomentumConfig.from_dict(gm.GREEN_MOMENTUM_CONFIG)
    for vk, over in VARIANTS.items():
        cfg = gm.GreenMomentumConfig.from_dict({**base_cfg.as_dict(), **over})
        reject = {"green": 0, "vwap": 0, "momentum": 0, "breakout": 0}
        # collect per-symbol session trades, then apply the portfolio max-positions cap per day
        by_day: dict[Any, list[GTrade]] = {}
        for sym, df in bars.items():
            cost = COST_BPS[bucket_of(sym)]
            for d, g, prev_close in _session_frames(df):
                tr = simulate_session(sym, g, prev_close, cfg, cost, reject)
                if tr is not None:
                    by_day.setdefault(d, []).append(tr)
        trades: list[GTrade] = []
        capped_out = 0
        for d, ts in by_day.items():
            ts.sort(key=lambda t: (t.entry_time, ))      # earliest signals first
            kept = ts[: args.max_positions]
            capped_out += len(ts) - len(kept)
            trades.extend(kept)
        is_tr = [t for t in trades if t.session < train_end]
        oos_tr = [t for t in trades if t.session >= train_end]
        report["variants"][vk] = {
            "config": {k: over.get(k) for k in ("require_momentum_confirmation", "require_opening_range_breakout", "stop_mode")},
            "overall": metrics(trades), "in_sample": metrics(is_tr), "out_of_sample": metrics(oos_tr),
            "oos_by_price_tier": _group_metrics(oos_tr, lambda t: t.tier),
            "oos_by_ticker": _group_metrics(oos_tr, lambda t: t.symbol),
            "oos_by_entry_hour": _group_metrics(oos_tr, lambda t: t.entry_time[:2] + "h"),
            "filter_rejections": reject, "trades_capped_by_max_positions": capped_out,
        }
        print(f"  {vk}: trades={len(trades)} (OOS {len(oos_tr)})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"green_momentum_backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    cols = ["trades", "win_rate", "avg_net_pnl", "net_profit_factor", "net_sharpe",
            "max_drawdown", "avg_hold_min", "stop_out_freq", "avg_mfe_pct", "avg_mae_pct"]
    print("\n================= OUT-OF-SAMPLE (net of costs) =================")
    print(f"{'metric':<18}" + "".join(f"{v:>13}" for v in VARIANTS))
    for col in cols:
        print(f"{col:<18}" + "".join(f"{str(report['variants'][v]['out_of_sample'].get(col)):>13}" for v in VARIANTS))
    print("\n-- OOS avg_net_pnl by price tier --")
    for v in VARIANTS:
        t = report["variants"][v]["oos_by_price_tier"]
        s = " | ".join(f"{k}:{t[k]['avg_net_pnl']}(n={t[k]['trades']})" for k in t)
        print(f"  {v}: {s}")
    print(f"\n[gm-bt] wrote {OUT_DIR / f'green_momentum_backtest_{tag}.json'}")


if __name__ == "__main__":
    main()
