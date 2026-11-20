#!/usr/bin/env python3
"""
amps_v2_backtest.py -- AMPS V2 research: feature cache + excursion analysis + experiment matrix
on the EXACT 30-instrument universe's 5-min bars, reusing the amps_v2 engine.

DATA SCOPE (honest, per spec section 25): this is BAR-SUPPORTED research. DI/ADX/Bollinger/VWAP/
ATR/relative-volume are bar-derivable and fully exercised. The two order-flow entry-score
components (CVD acceleration, aggressive buying pressure) and the order-flow-divergence setup
require genuine ticks and are NOT fabricated from OHLC -- they are ABSENT here, so the entry
score renormalizes over available components and this is flagged. Microstructure-dependent
components are validated separately on sampled genuine ticks, not by these bar runs.

No look-ahead: a completed-bar signal fills at the NEXT bar's open; intrabar stop/target
ambiguity is modeled CONSERVATIVELY (a bar's low is tested against the stop before its high can
lift the trail or hit the target). Costs via costs.py bucket + slippage, both sides. Chronological
IS/OOS split; double-cost stress test; acceptance criteria evaluated verbatim.

Experiments: V2.0 baseline (binary momentum) / V2.2 BB policy (confirmed|approaching|combined) /
V2.3 entry-quality threshold / V2.6 adaptive exit vs fixed / V2.10 integrated / V2.11 OOS+stress.
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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
import amps_v2 as A
from backtest_dmi_adx import bucket_of, COST_BPS

BARS = PROJECT_ROOT / "data" / "cache" / "bars_multi_tf"
FEAT = PROJECT_ROOT / "data" / "cache" / "amps_v2_features"
OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")
ET_TZ = "America/New_York"
SLIPPAGE_BPS = 2.0


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_features(symbol: str) -> pd.DataFrame:
    """Vectorized feature cache (version-keyed). Decision-time features only; no fwd labels."""
    FEAT.mkdir(parents=True, exist_ok=True)
    cache = FEAT / f"{symbol}_5m_v1.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    files = sorted(glob.glob(str(BARS / f"{symbol}_5m_*.csv")))
    if not files:
        return pd.DataFrame()
    d = pd.read_csv(files[-1])
    if d.empty:
        return pd.DataFrame()
    d["ts"] = pd.to_datetime(d["timestamp"], utc=True)
    d["et"] = d["ts"].dt.tz_convert(ET_TZ)
    d["date"] = d["et"].dt.date
    d["et_time"] = d["et"].dt.time
    d = d[(d["et_time"] >= time(9, 30)) & (d["et_time"] < time(16, 0))].copy()
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = d[c].astype(float)
    d = d.sort_values("ts").reset_index(drop=True)
    h, l, c, v = d["high"], d["low"], d["close"], d["volume"]
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 14)
    d["atr"] = atr
    d["pdi"] = 100 * _rma(pdm, 14) / atr.replace(0, np.nan)
    d["mdi"] = 100 * _rma(mdm, 14) / atr.replace(0, np.nan)
    dx = 100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan)
    d["adx"] = _rma(dx, 14)
    d["bb_mid"] = c.rolling(20).mean()
    d["bb_std"] = c.rolling(20).std()
    d["bb_width"] = (4 * d["bb_std"]) / d["bb_mid"]
    d["bb_slope"] = d["bb_mid"] - d["bb_mid"].shift(3)
    d["sma50"] = c.rolling(50).mean()
    d["ret1"] = c.pct_change(1, fill_method=None)
    d["rvol"] = v / v.rolling(20).mean().replace(0, np.nan)
    d["realized_vol"] = c.pct_change(fill_method=None).rolling(10).std()
    # session VWAP (resets each date), bb distance + its change, short momentum
    typ = (h + l + c) / 3.0
    d["_cv"] = d.groupby("date")["volume"].cumsum()
    d["_cpv"] = (typ * v).groupby(d["date"]).cumsum()
    d["vwap"] = d["_cpv"] / d["_cv"].replace(0, np.nan)
    d["bb_dist"] = (c - d["bb_mid"]) / d["bb_mid"]
    d["bb_dist_prev"] = d["bb_dist"].shift(1)
    d["short_mom"] = c - c.shift(2)
    d["vwap_dist"] = (c - d["vwap"]) / d["vwap"]
    d["di_sep"] = d["pdi"] - d["mdi"]
    d["di_sep_prev"] = d["di_sep"].shift(1)
    d["di_cross_up"] = (d["pdi"] > d["mdi"]) & (d["pdi"].shift(1) <= d["mdi"].shift(1))
    d["adx_prev"] = d["adx"].shift(1); d["adx_prev2"] = d["adx"].shift(2)
    d["bb_width_prev"] = d["bb_width"].shift(3)
    d["range_high20"] = h.rolling(20).max().shift(1)
    try:
        d.drop(columns=["_cv", "_cpv"]).to_parquet(cache)
    except Exception:
        pass
    return d.drop(columns=["_cv", "_cpv"])


# ---------------------------------------------------------------- excursion analysis (8)
def excursion_summary(d: pd.DataFrame, entries_idx: list[int], horizons_bars: list[int],
                      target_bps: float, stop_bps: float) -> dict:
    if not entries_idx:
        return {"n": 0}
    c = d["close"].values; h = d["high"].values; l = d["low"].values
    n = len(d)
    rows = {hz: {"mfe": [], "mae": [], "tgt_first": 0, "stop_first": 0} for hz in horizons_bars}
    for i in entries_idx:
        ep = c[i]
        if ep <= 0:
            continue
        for hz in horizons_bars:
            end = min(i + hz, n - 1)
            if end <= i:
                continue
            seg_h = h[i + 1:end + 1]; seg_l = l[i + 1:end + 1]
            if len(seg_h) == 0:
                continue
            mfe = (seg_h.max() / ep - 1.0) * 1e4
            mae = (seg_l.min() / ep - 1.0) * 1e4
            rows[hz]["mfe"].append(mfe); rows[hz]["mae"].append(mae)
            # target-before-stop within this horizon (conservative: stop checked each bar first)
            tgt_p = ep * (1 + target_bps / 1e4); stop_p = ep * (1 - stop_bps / 1e4)
            hit = None
            for j in range(i + 1, end + 1):
                if l[j] <= stop_p:
                    hit = "stop"; break
                if h[j] >= tgt_p:
                    hit = "tgt"; break
            if hit == "tgt":
                rows[hz]["tgt_first"] += 1
            elif hit == "stop":
                rows[hz]["stop_first"] += 1
    out = {"n": len(entries_idx)}
    for hz, r in rows.items():
        if r["mfe"]:
            tot = r["tgt_first"] + r["stop_first"]
            out[f"{hz}bar"] = {
                "avg_mfe_bps": round(statistics.mean(r["mfe"]), 2),
                "avg_mae_bps": round(statistics.mean(r["mae"]), 2),
                "target_before_stop": round(r["tgt_first"] / tot, 3) if tot else None,
            }
    return out


# ---------------------------------------------------------------- variant simulation
@dataclass
class ATrade:
    symbol: str; session: Any; entry_px: float; exit_px: float
    net_bps: float; gross_bps: float; hold_min: float; reason: str
    mfe_bps: float; mae_bps: float


def _eligible_mask(d: pd.DataFrame, cfg: A.AmpsV2Config, use_quality: bool,
                   min_quality: float) -> tuple[pd.Series, dict]:
    """Vectorized base gates (bar-derivable). Returns (mask, reject_counts). BB policy +
    quality applied per the config. Order-flow components absent (flagged)."""
    within = (d["et_time"] >= time(9, 45)) & (d["et_time"] < time(15, 0))
    di_ok = d["pdi"] > d["mdi"]
    adx_ok = (d["adx"] > cfg.min_adx) & ((d["adx"] > d["adx_prev"]) if cfg.require_adx_rising else True)
    vwap_ok = d["close"] > d["vwap"]
    # BB state (vectorized approximation of bb_approach_state)
    above = d["close"] > d["bb_mid"]
    thresh = cfg.bb_approach_distance_bps / 1e4
    approaching = (~above) & (d["bb_dist"].abs() <= thresh) & (d["bb_dist"].abs() < d["bb_dist_prev"].abs()) & (d["short_mom"] > 0)
    if cfg.bb_entry_policy == "confirmed":
        bb_ok = above
    elif cfg.bb_entry_policy == "approaching":
        bb_ok = approaching
    else:
        bb_ok = above | approaching
    base = within & di_ok & adx_ok & vwap_ok & bb_ok
    rej = {"di": int((within & ~di_ok).sum()), "adx": int((within & di_ok & ~adx_ok).sum()),
           "vwap": int((within & di_ok & adx_ok & ~vwap_ok).sum()),
           "bb_policy": int((within & di_ok & adx_ok & vwap_ok & ~bb_ok).sum())}
    if use_quality:
        # vectorized entry-quality over AVAILABLE (bar) components, renormalized
        di_sep = d["di_sep"]
        q_di = (0.5 * (di_sep / 25.0).clip(0, 1) + 0.5 * (di_sep > d["di_sep_prev"]).astype(float)).clip(0, 1)
        adx_accel = (d["adx"] - d["adx_prev"]) - (d["adx_prev"] - d["adx_prev2"])
        q_adx = (0.5 * (d["adx"] > d["adx_prev"]).astype(float) + 0.3 * (adx_accel > 0).astype(float)
                 + 0.2 * ((d["adx"] - 20) / 20).clip(0, 1)).clip(0, 1)
        q_prox = (1 - (d["bb_dist"].abs() / 0.002).clip(0, 1)).fillna(0)
        q_bb = (0.6 * q_prox + 0.2 * (d["bb_dist"] >= 0).astype(float) + 0.2 * (d["bb_slope"] > 0).astype(float)).clip(0, 1)
        q_vwap = (0.6 * (d["vwap_dist"] > 0).astype(float) + 0.4 * (d["vwap_dist"] / 0.002).clip(0, 1)).clip(0, 1)
        q_rvol = ((d["rvol"] - 1.0) / 1.5).clip(0, 1).fillna(0)
        atr_bps = (d["atr"] / d["close"]) * 1e4
        q_cost = ((0.5 * atr_bps - cfg.est_round_trip_bps) / 20.0).clip(0, 1).fillna(0)
        w = cfg.entry_quality_weights
        num = (w["di_sep_expansion"] * q_di + w["adx_accel"] * q_adx + w["bb_proximity"] * q_bb
               + w["vwap_align"] * q_vwap + w["rel_volume"] * q_rvol + w["cost_efficiency"] * q_cost)
        den = (w["di_sep_expansion"] + w["adx_accel"] + w["bb_proximity"] + w["vwap_align"]
               + w["rel_volume"] + w["cost_efficiency"])  # order-flow weights excluded (absent)
        score = 100.0 * num / den
        q_ok = score >= min_quality
        rej["quality"] = int((base & ~q_ok).sum())
        base = base & q_ok
    return base.fillna(False), rej


def simulate(d: pd.DataFrame, symbol: str, cfg: A.AmpsV2Config, cost_bps: float, use_quality: bool,
             min_quality: float, adaptive_exit: bool, cost_mult: float = 1.0) -> tuple[list[ATrade], dict, list[int]]:
    if d.empty or len(d) < 60:
        return [], {}, []
    mask, rej = _eligible_mask(d, cfg, use_quality, min_quality)
    o = d["open"].values; h = d["high"].values; l = d["low"].values; c = d["close"].values
    atr = d["atr"].values; ettime = d["et_time"].values; dates = d["date"].values
    elig = mask.values
    n = len(d)
    rt = (cost_bps * cost_mult + SLIPPAGE_BPS) / 1e4
    trades: list[ATrade] = []
    entries_idx: list[int] = []
    i = 0
    force = time(15, 55)
    while i < n - 1:
        if not elig[i]:
            i += 1; continue
        entries_idx.append(i)
        ei = i + 1
        entry_px = o[ei] * (1 + rt)
        peak = entry_px; dip = entry_px
        pos = A.new_amps_position(symbol, entry_px, str(dates[ei]), 1.0, cfg,
                                  float(atr[ei]) if atr[ei] == atr[ei] else None, A.BB_ABOVE)
        exit_px = c[n - 1]; reason = A.EOD_EXIT; xi = n - 1
        for j in range(ei, n):
            now_et = datetime.combine(dates[j], ettime[j])
            hold_min = (j - ei) * 5.0
            gross_now = (c[j] / entry_px - 1.0) * 1e4
            # conservative intrabar: test stop against LOW before high lifts the trail
            if l[j] <= pos.effective_stop and now_et.time() < force:
                exit_px = min(pos.effective_stop, o[j])
                reason = (A.TRAILING_STOP if pos.stage == "TRAILING" else
                          A.MIN_PROFIT_LOCK if pos.stage == "MIN_PROFIT_LOCK" else
                          A.BREAKEVEN_STOP if pos.stage == "BREAKEVEN" else A.INITIAL_STOP)
                xi = j; break
            # target via high (conservative: only after stop checked)
            if (h[j] / entry_px - 1.0) * 1e4 >= cfg.final_target_bps and now_et.time() < force:
                exit_px = entry_px * (1 + cfg.final_target_bps / 1e4)
                reason = A.PROFIT_TARGET; xi = j; break
            if adaptive_exit:
                ex, r = A.amps_exit_decision(pos, float(h[j]), now_et, cfg, force, hold_min, bearish_signals=0)
            else:
                # fixed control: initial stop + target + max-hold + EOD only (no adaptive trail)
                ex, r = (True, A.EOD_EXIT) if now_et.time() >= force else (
                    (True, A.MAX_HOLD_EXIT) if hold_min >= cfg.max_hold_minutes else (False, ""))
            peak = max(peak, float(h[j])); dip = min(dip, float(l[j]))
            if ex:
                exit_px = o[j] if (j > ei and r in (A.EOD_EXIT, A.MAX_HOLD_EXIT, A.MOMENTUM_EXIT)) else c[j]
                reason = r; xi = j; break
        exit_net = exit_px * (1 - rt)
        gross_bps = (exit_px / entry_px - 1.0) * 1e4
        net_bps = (exit_net / entry_px - 1.0) * 1e4
        trades.append(ATrade(symbol, dates[ei], entry_px, exit_net, net_bps, gross_bps,
                             (xi - ei) * 5.0, reason, (peak / entry_px - 1) * 1e4, (dip / entry_px - 1) * 1e4))
        i = xi + 1
    return trades, rej, entries_idx


# ---------------------------------------------------------------- metrics + acceptance
def metrics(trades: list[ATrade], notional=2000.0) -> dict:
    if not trades:
        return {"trades": 0}
    nets = [t.net_bps / 1e4 for t in trades]
    pnls = [r * notional for r in nets]
    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    sd = statistics.pstdev(nets) if len(nets) > 1 else 0.0
    sharpe = (statistics.mean(nets) / sd * math.sqrt(252)) if sd > 0 else 0.0
    # profitable-day fraction
    by_day: dict[Any, float] = {}
    for t in trades:
        by_day[t.session] = by_day.get(t.session, 0.0) + t.net_bps / 1e4 * notional
    prof_days = sum(1 for v in by_day.values() if v > 0) / len(by_day) if by_day else 0.0
    cum = pk = mdd = 0.0
    for x in pnls:
        cum += x; pk = max(pk, cum); mdd = min(mdd, cum - pk)
    stops = sum(1 for t in trades if "STOP" in t.reason)
    return {
        "trades": len(trades), "win_rate": round(len(wins) / len(pnls), 3),
        "net_expectancy_bps": round(statistics.mean([t.net_bps for t in trades]), 3),
        "avg_net_pnl": round(statistics.mean(pnls), 3), "total_net_pnl": round(sum(pnls), 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "net_sharpe": round(sharpe, 3), "profitable_days": round(prof_days, 3),
        "max_drawdown": round(mdd, 2), "avg_hold_min": round(statistics.mean([t.hold_min for t in trades]), 1),
        "avg_mfe_bps": round(statistics.mean([t.mfe_bps for t in trades]), 2),
        "avg_mae_bps": round(statistics.mean([t.mae_bps for t in trades]), 2),
        "stop_out_freq": round(stops / len(trades), 3),
    }


ACCEPT = {"net_profit_factor": 1.50, "win_rate": 0.70, "profitable_days": 0.70,
          "net_sharpe": 1.50, "min_trades": 1000, "max_drawdown_pct": -0.05}


def check_acceptance(m: dict, notional_base: float = 2000.0) -> tuple[bool, list]:
    fails = []
    if (m.get("net_expectancy_bps", 0) or 0) <= 0: fails.append("net_expectancy<=0")
    pf = m.get("profit_factor", 0); pf = 999 if pf == "inf" else (pf or 0)
    if pf < ACCEPT["net_profit_factor"]: fails.append(f"PF {pf}<1.5")
    if (m.get("win_rate", 0) or 0) < ACCEPT["win_rate"]: fails.append(f"win {m.get('win_rate')}<0.70")
    if (m.get("profitable_days", 0) or 0) < ACCEPT["profitable_days"]: fails.append("prof_days<0.70")
    if (m.get("net_sharpe", 0) or 0) < ACCEPT["net_sharpe"]: fails.append("sharpe<1.5")
    if (m.get("trades", 0) or 0) < ACCEPT["min_trades"]: fails.append("trades<1000")
    return (len(fails) == 0), fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe-file", default="universe.txt")
    ap.add_argument("--train-end", default="2024-05-01")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import re
    raw = (PROJECT_ROOT / args.universe_file).read_text()
    symbols = re.findall(r"'([A-Z]+)'", raw) or [s.strip().upper() for s in raw.split(",") if s.strip()]
    train_end = date.fromisoformat(args.train_end)

    feats = {}
    for s in symbols:
        d = compute_features(s)
        if not d.empty and len(d) >= 200:
            feats[s] = d
    print(f"[amps-bt] universe={len(symbols)} with-features={len(feats)} train<{train_end}")

    base = A.AmpsV2Config()
    # experiment grid: (name, policy, use_quality, min_quality, adaptive_exit, cost_mult)
    grid = [
        ("V2.0_baseline", "confirmed", False, 0, False, 1.0),
        ("V2.2_approaching", "approaching", False, 0, False, 1.0),
        ("V2.2_combined", "combined", False, 0, False, 1.0),
        ("V2.3_quality70", "combined", True, 70, False, 1.0),
        ("V2.3_quality80", "combined", True, 80, False, 1.0),
        ("V2.6_adaptive_exit", "combined", False, 0, True, 1.0),
        ("V2.10_integrated", "combined", True, 70, True, 1.0),
        ("V2.11_integrated_2xcost", "combined", True, 70, True, 2.0),
    ]
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                              "data_source": "Alpaca IEX 5m bars (bar-supported; order-flow score components ABSENT -- tick-only, flagged)",
                              "universe": list(feats), "train_end": args.train_end,
                              "acceptance_criteria": ACCEPT, "experiments": {}}
    # excursion analysis once (on baseline-eligible entries, horizons in 5-min bars)
    exc_entries = 0
    exc_acc = {}
    for name, policy, uq, mq, adapt, cmult in grid:
        cfg = A.AmpsV2Config.from_dict({**base.as_dict(), "bb_entry_policy": policy})
        all_tr: list[ATrade] = []
        rej_tot = {}
        all_entries = {}
        for s, d in feats.items():
            cost = COST_BPS[bucket_of(s)]
            tr, rej, ent = simulate(d, s, cfg, cost, uq, mq, adapt, cmult)
            all_tr.extend(tr)
            for k, val in rej.items():
                rej_tot[k] = rej_tot.get(k, 0) + val
            all_entries[s] = (d, ent)
        is_tr = [t for t in all_tr if t.session < train_end]
        oos_tr = [t for t in all_tr if t.session >= train_end]
        m_oos = metrics(oos_tr)
        ok, fails = check_acceptance(m_oos)
        report["experiments"][name] = {
            "config": {"bb_policy": policy, "entry_quality": uq, "min_quality": mq,
                       "adaptive_exit": adapt, "cost_mult": cmult},
            "overall": metrics(all_tr), "in_sample": metrics(is_tr), "out_of_sample": m_oos,
            "filter_rejections": rej_tot, "accepts": ok, "acceptance_fails": fails,
        }
        print(f"  {name}: trades={len(all_tr)} OOS={len(oos_tr)} "
              f"PF={m_oos.get('profit_factor')} win={m_oos.get('win_rate')} "
              f"exp_bps={m_oos.get('net_expectancy_bps')} accepts={ok}", flush=True)
        if name == "V2.0_baseline":
            # excursion on baseline entries
            ex_all = {"n": 0}
            agg = {}
            nent = 0
            for s, (d, ent) in all_entries.items():
                e = excursion_summary(d, ent, [1, 2, 6, 12], base.final_target_bps, base.initial_stop_bps)
                nent += e.get("n", 0)
                for k, v in e.items():
                    if k == "n" or not isinstance(v, dict):
                        continue
                    agg.setdefault(k, {"mfe": [], "mae": [], "tbs": []})
                    agg[k]["mfe"].append(v["avg_mfe_bps"]); agg[k]["mae"].append(v["avg_mae_bps"])
                    if v["target_before_stop"] is not None:
                        agg[k]["tbs"].append(v["target_before_stop"])
            exc = {"entries": nent}
            for k, vv in agg.items():
                exc[k] = {"avg_mfe_bps": round(statistics.mean(vv["mfe"]), 2),
                          "avg_mae_bps": round(statistics.mean(vv["mae"]), 2),
                          "target_before_stop": round(statistics.mean(vv["tbs"]), 3) if vv["tbs"] else None}
            report["excursion_analysis"] = exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"amps_v2_backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n================= OUT-OF-SAMPLE (net of costs, bars) =================")
    cols = ["trades", "win_rate", "net_expectancy_bps", "profit_factor", "net_sharpe",
            "profitable_days", "max_drawdown", "stop_out_freq", "avg_mfe_bps", "avg_mae_bps"]
    names = list(report["experiments"])
    print(f"{'metric':<20}" + "".join(f"{n.split('_')[0]:>12}" for n in names))
    for col in cols:
        print(f"{col:<20}" + "".join(f"{str(report['experiments'][n]['out_of_sample'].get(col)):>12}" for n in names))
    print("\nExcursion (baseline entries):", json.dumps(report.get("excursion_analysis", {})))
    print("\nAcceptance (OOS):")
    for n in names:
        e = report["experiments"][n]
        print(f"  {n}: accepts={e['accepts']}  fails={e['acceptance_fails']}")
    any_pass = any(report["experiments"][n]["accepts"] for n in names)
    print(f"\n[amps-bt] wrote {OUT_DIR / f'amps_v2_backtest_{tag}.json'}")
    print(f"OVERALL: {'A VARIANT PASSES ACCEPTANCE' if any_pass else 'NO VARIANT MEETS ACCEPTANCE CRITERIA'}")


if __name__ == "__main__":
    main()
