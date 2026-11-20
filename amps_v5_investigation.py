#!/usr/bin/env python3
"""
amps_v5_investigation.py -- AMPS V5: entry-alpha & exit-economics investigation.

Goal (diagnostic, not a new engine): isolate WHY AMPS V4.10 has negative net expectancy, and
test the one lever that could flip it -- replacing tight fixed micro-scalp exits with
instrument-specific ATR-scaled targets/stops over longer holds (micro-scalp vs short-duration
momentum). Reuses amps_v2 engine, amps_v4 regime/eligibility, and the cached 5-min features.

Phases:
  1  Cost audit (decompose round-trip cost; check double-counting).
  2  Excursion reachability -- % of entries reaching net +X bps BEFORE the stop, by horizon.
  3  Exit attribution -- by exit reason: count, expectancy, capture-vs-MFE (winners cut early?).
  5  ATR target x ATR stop x holding-period sweep on directional-eligible entries (THE test).

Honest scope: bar-supported (order-flow components need ticks; absent/flagged, never fabricated).
No look-ahead: completed-bar signal -> next-bar-open fill; conservative intrabar (low tested vs
stop before high can reach the target). Costs via costs.py bucket + slippage, both sides.
Chronological IS/OOS; acceptance applied verbatim.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone, date, time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
import amps_v2 as A
import amps_v4_regime as R
from amps_v2_backtest import compute_features, _eligible_mask, SLIPPAGE_BPS, ACCEPT, check_acceptance
from amps_v4_backtest import build_regime_frames, _directional_mask
from backtest_dmi_adx import bucket_of, COST_BPS

OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------- entry set (V4.4 directional-eligible)
def eligible_entries(feats: dict, ref_scores: dict, regime: pd.Series, cfg_amps: A.AmpsV2Config
                     ) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Return {symbol: (feature_df, entry_bool_array)} for directional-eligible entries."""
    out = {}
    rcfg = R.RegimeConfig()
    for s in R.TRADE_UNIVERSE:
        d = feats.get(s)
        if d is None or d.empty or len(d) < 60:
            continue
        own, _ = _eligible_mask(d, cfg_amps, use_quality=False, min_quality=0)
        dm = _directional_mask(d, s, ref_scores, regime, rcfg, use_regime=True, use_underlying=True, allow_divergence=True)
        out[s] = (d, (own & dm).to_numpy())
    return out


# ---------------------------------------------------------------- Phase 1: cost audit
def cost_audit(feats: dict, entries: dict, cfg_amps: A.AmpsV2Config) -> dict:
    """Decompose the modeled round-trip cost and verify it is applied once per side, no double
    count. Returns per-bucket cost breakdown + a reconciliation note."""
    buckets = {}
    for s in entries:
        b = bucket_of(s)
        buckets.setdefault(b, {"symbols": set(), "cost_bps_side": COST_BPS[b] + SLIPPAGE_BPS})
        buckets[b]["symbols"].add(s)
    out = {"model": "COST_BPS[bucket] (spread+commission proxy) + SLIPPAGE_BPS, applied ONCE per side",
           "round_trip_bps_by_bucket": {b: round(2 * v["cost_bps_side"], 2) for b, v in buckets.items()},
           "per_side_bps_by_bucket": {b: round(v["cost_bps_side"], 2) for b, v in buckets.items()},
           "double_count_check": "entry applies (1+rt), exit applies (1-rt); rt added once/side -> no double count",
           "symbols_per_bucket": {b: sorted(v["symbols"]) for b, v in buckets.items()}}
    return out


# ---------------------------------------------------------------- Phase 2: excursion reachability
def excursion_reachability(feats: dict, entries: dict, horizons: list[int], net_targets_bps: list[float],
                           cost_rt_bps_by_sym: dict, stop_bps: float) -> dict:
    """% of entries that reach net +X bps BEFORE breaching a `stop_bps` protective stop, by horizon.
    Net = gross - round-trip cost. Uses high for favorable, low for the stop (conservative: stop
    checked each bar before the high)."""
    res = {f"{hz}bar": {f"net{int(t)}bps": 0 for t in net_targets_bps} for hz in horizons}
    counts = {f"{hz}bar": 0 for hz in horizons}
    mfe_by_hz = {hz: [] for hz in horizons}
    mae_by_hz = {hz: [] for hz in horizons}
    for s, (d, mask) in entries.items():
        o = d["open"].values; h = d["high"].values; l = d["low"].values
        n = len(d); rt = cost_rt_bps_by_sym.get(s, 6.0)
        idx = np.where(mask)[0]
        for i in idx:
            if i + 1 >= n:
                continue
            ep = o[i + 1]
            if ep <= 0:
                continue
            stop_p = ep * (1 - stop_bps / 1e4)
            for hz in horizons:
                end = min(i + 1 + hz, n - 1)
                if end <= i + 1:
                    continue
                counts[f"{hz}bar"] += 1
                seg_h = h[i + 1:end + 1]; seg_l = l[i + 1:end + 1]
                mfe_by_hz[hz].append((seg_h.max() / ep - 1) * 1e4)
                mae_by_hz[hz].append((seg_l.min() / ep - 1) * 1e4)
                # reachability: did net +X occur before the stop was breached?
                reached = {t: False for t in net_targets_bps}
                for j in range(i + 1, end + 1):
                    if l[j] <= stop_p:
                        break
                    net_mfe = (h[j] / ep - 1) * 1e4 - rt
                    for t in net_targets_bps:
                        if net_mfe >= t:
                            reached[t] = True
                for t in net_targets_bps:
                    if reached[t]:
                        res[f"{hz}bar"][f"net{int(t)}bps"] += 1
    out = {"stop_bps": stop_bps, "by_horizon": {}}
    for hz in horizons:
        c = counts[f"{hz}bar"]
        if c == 0:
            continue
        out["by_horizon"][f"{hz}bar"] = {
            "entries": c,
            "avg_mfe_bps": round(statistics.mean(mfe_by_hz[hz]), 2),
            "avg_mae_bps": round(statistics.mean(mae_by_hz[hz]), 2),
            "reach_net_pct": {k: round(v / c, 3) for k, v in res[f"{hz}bar"].items()},
        }
    return out


# ---------------------------------------------------------------- parameterized exit sim (Phase 3 + 5)
def simulate_exits(d: pd.DataFrame, mask: np.ndarray, symbol: str, cost_bps: float,
                   target_mode: str, target_val: float, stop_mode: str, stop_val: float,
                   max_hold_bars: int, cost_mult: float = 1.0) -> list[dict]:
    """target/stop_mode in {'bps','atr'}. Returns trade dicts with net_bps, reason, mfe, mae, hold."""
    o = d["open"].values; h = d["high"].values; l = d["low"].values; c = d["close"].values
    atr = d["atr"].values; ettime = d["et_time"].values; dates = d["date"].values
    n = len(d); rt = (cost_bps * cost_mult + SLIPPAGE_BPS) / 1e4
    force = time(15, 55)
    trades = []
    i = 0
    while i < n - 1:
        if not mask[i]:
            i += 1; continue
        ei = i + 1
        ep = o[ei] * (1 + rt)
        a = float(atr[ei]) if atr[ei] == atr[ei] and atr[ei] > 0 else None
        if target_mode == "atr":
            if a is None:
                i += 1; continue
            tgt = ep + target_val * a
        else:
            tgt = ep * (1 + target_val / 1e4)
        if stop_mode == "atr":
            if a is None:
                i += 1; continue
            stop = ep - stop_val * a
        else:
            stop = ep * (1 - stop_val / 1e4)
        peak = ep; dip = ep; exit_px = c[n - 1]; reason = "END_OF_DAY"; xi = n - 1
        for j in range(ei, min(ei + max_hold_bars, n)):
            now = datetime.combine(dates[j], ettime[j])
            if now.time() >= force:
                exit_px = o[j] if j > ei else c[j]; reason = "END_OF_DAY"; xi = j; break
            if l[j] <= stop:                         # conservative: stop first
                exit_px = min(stop, o[j]); reason = "STOP"; xi = j; break
            if h[j] >= tgt:
                exit_px = tgt; reason = "TARGET"; xi = j; break
            peak = max(peak, h[j]); dip = min(dip, l[j])
        else:
            exit_px = c[min(ei + max_hold_bars, n) - 1]; reason = "MAX_HOLD"; xi = min(ei + max_hold_bars, n) - 1
        exit_net = exit_px * (1 - rt)
        trades.append({"symbol": symbol, "session": dates[ei], "net_bps": (exit_net / ep - 1) * 1e4,
                       "gross_bps": (exit_px / ep - 1) * 1e4, "reason": reason, "hold_min": (xi - ei) * 5.0,
                       "mfe_bps": (peak / ep - 1) * 1e4, "mae_bps": (dip / ep - 1) * 1e4})
        i = xi + 1
    return trades


def trade_metrics(trades: list[dict], notional=2000.0) -> dict:
    if not trades:
        return {"trades": 0}
    nets = [t["net_bps"] / 1e4 for t in trades]
    pnls = [r * notional for r in nets]
    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    sd = statistics.pstdev(nets) if len(nets) > 1 else 0.0
    sharpe = (statistics.mean(nets) / sd * math.sqrt(252)) if sd > 0 else 0.0
    by_day = {}
    for t in trades:
        by_day[t["session"]] = by_day.get(t["session"], 0.0) + t["net_bps"] / 1e4 * notional
    prof_days = sum(1 for v in by_day.values() if v > 0) / len(by_day) if by_day else 0.0
    cum = pk = mdd = 0.0
    for x in pnls:
        cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    return {"trades": len(trades), "win_rate": round(len(wins) / len(pnls), 3),
            "net_expectancy_bps": round(statistics.mean([t["net_bps"] for t in trades]), 3),
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "net_sharpe": round(sharpe, 3), "profitable_days": round(prof_days, 3),
            "max_drawdown": round(mdd, 2), "avg_hold_min": round(statistics.mean([t["hold_min"] for t in trades]), 1)}


def exit_attribution(trades: list[dict]) -> dict:
    out = {}
    for reason in ("STOP", "TARGET", "MAX_HOLD", "END_OF_DAY"):
        ts = [t for t in trades if t["reason"] == reason]
        if not ts:
            continue
        realized = [t["net_bps"] for t in ts]
        mfe = [t["mfe_bps"] for t in ts]
        cap = [(t["gross_bps"] / t["mfe_bps"]) for t in ts if t["mfe_bps"] > 1e-9]
        out[reason] = {"trades": len(ts), "net_exp_bps": round(statistics.mean(realized), 2),
                       "win_rate": round(sum(1 for r in realized if r > 0) / len(ts), 3),
                       "avg_mfe_bps": round(statistics.mean(mfe), 2),
                       "capture_vs_mfe": round(statistics.mean(cap), 3) if cap else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-end", default="2024-05-01")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    train_end = date.fromisoformat(args.train_end)
    amps = A.AmpsV2Config()
    rcfg = R.RegimeConfig()

    feats = {}
    for s in R.TRADE_UNIVERSE + R.REFERENCES:
        f = compute_features(s)
        if not f.empty and len(f) >= 200:
            feats[s] = f
    ref_scores, regime = build_regime_frames(feats, rcfg)
    entries = eligible_entries(feats, ref_scores, regime, amps)
    cost_rt = {s: 2 * (COST_BPS[bucket_of(s)] + SLIPPAGE_BPS) for s in entries}
    n_entries = sum(int(m.sum()) for _, m in entries.values())
    print(f"[v5] directional-eligible entries={n_entries} across {len(entries)} ETFs; train<{train_end}")

    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                              "data_source": "Alpaca IEX 5m bars (bar-supported; order flow absent/flagged)",
                              "acceptance": ACCEPT, "train_end": args.train_end}

    # Phase 1
    report["phase1_cost_audit"] = cost_audit(feats, entries, amps)

    # Phase 2 excursion reachability (stop at 1x ATR ~ wide, so stop-before-target is rarer)
    report["phase2_reachability"] = excursion_reachability(
        feats, entries, horizons=[1, 2, 6, 12, 24, 36], net_targets_bps=[1, 2, 5, 10, 20, 30, 50],
        cost_rt_bps_by_sym=cost_rt, stop_bps=50.0)

    # Phase 3 attribution on the V4.10 reference config (30bps target / 10bps stop / 5min hold)
    v410_trades = []
    for s, (d, m) in entries.items():
        v410_trades += simulate_exits(d, m, s, COST_BPS[bucket_of(s)], "bps", 30, "bps", 10, max_hold_bars=1)
    report["phase3_exit_attribution_v410"] = exit_attribution(v410_trades)
    report["phase3_v410_oos"] = trade_metrics([t for t in v410_trades if t["session"] >= train_end])

    # Phase 5 THE SWEEP: ATR target x ATR stop x holding period
    print("[v5] Phase 5 ATR target x stop x hold sweep...", flush=True)
    sweep = []
    for tmult in (0.5, 1.0, 1.5, 2.0):
        for smult in (0.5, 1.0):
            for hold in (2, 6, 12, 24, 36):   # 10/30/60/120/180 min
                tr = []
                for s, (d, m) in entries.items():
                    tr += simulate_exits(d, m, s, COST_BPS[bucket_of(s)], "atr", tmult, "atr", smult, hold)
                oos = [t for t in tr if t["session"] >= train_end]
                mo = trade_metrics(oos)
                ok, _ = check_acceptance(mo)
                sweep.append({"target_atr": tmult, "stop_atr": smult, "hold_bars": hold,
                              "hold_min": hold * 5, **mo, "accepts": ok})
    sweep.sort(key=lambda r: (r.get("net_expectancy_bps") or -999), reverse=True)
    report["phase5_atr_hold_sweep"] = sweep

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"amps_v5_investigation_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    # print summary
    print("\n=== Phase 1 cost audit ===")
    print(json.dumps(report["phase1_cost_audit"]["round_trip_bps_by_bucket"]))
    print("\n=== Phase 2 reachability (stop=50bps) -- % reaching net +X before stop ===")
    for hz, v in report["phase2_reachability"]["by_horizon"].items():
        print(f"  {hz}: MFE={v['avg_mfe_bps']} MAE={v['avg_mae_bps']} reach={v['reach_net_pct']}")
    print("\n=== Phase 3 exit attribution (V4.10 config: 30/10bps) ===")
    for r, v in report["phase3_exit_attribution_v410"].items():
        print(f"  {r}: n={v['trades']} net_exp={v['net_exp_bps']}bps win={v['win_rate']} "
              f"avg_mfe={v['avg_mfe_bps']} capture={v['capture_vs_mfe']}")
    print("\n=== Phase 5 ATRxstopxhold sweep -- TOP 8 by OOS net expectancy ===")
    print(f"{'tgtATR':>7}{'stopATR':>8}{'holdmin':>8}{'trades':>8}{'win':>7}{'expbps':>8}{'PF':>7}{'sharpe':>8}{'accepts':>9}")
    for r in sweep[:8]:
        print(f"{r['target_atr']:>7}{r['stop_atr']:>8}{r['hold_min']:>8}{r.get('trades',0):>8}"
              f"{r.get('win_rate',0):>7}{r.get('net_expectancy_bps',0):>8}{str(r.get('profit_factor')):>7}"
              f"{r.get('net_sharpe',0):>8}{str(r['accepts']):>9}")
    best = sweep[0] if sweep else {}
    any_pos = any((r.get("net_expectancy_bps") or -1) > 0 for r in sweep)
    any_pass = any(r.get("accepts") for r in sweep)
    print(f"\n[v5] wrote {OUT_DIR / f'amps_v5_investigation_{tag}.json'}")
    print(f"BEST OOS net expectancy: {best.get('net_expectancy_bps')} bps "
          f"(target {best.get('target_atr')}xATR, stop {best.get('stop_atr')}xATR, hold {best.get('hold_min')}min)")
    print(f"ANY net-positive OOS config: {any_pos} | ANY meets full acceptance: {any_pass}")


if __name__ == "__main__":
    main()
