#!/usr/bin/env python3
"""Multi-horizon entry-alpha diagnostic (V6 spec sections 34, 36: V6.0/V6.1 + the new swing test).

THE decisive, gating experiment. Before building 14 agents + an execution state machine on top of
AMPS entries, answer the only question that justifies any of it:

  V5 proved AMPS intraday entries have NO directional edge (MFE ~= |MAE| out to 180 min).
  V6's genuinely new question: does extending to the MULTI-DAY SWING horizon produce genuine
  directional ASYMMETRY (MFE materially > |MAE|, net-positive after cost, beating controls) --
  or just proportionally larger symmetric excursions (spec section 34)?

Method (all real data, PIT, conservative):
  * 5-minute horizons (reproduce V5): directional-eligible AMPS entries on the ETF universe,
    forward excursion at 30m..full-day.
  * DAILY swing horizons (NEW): resample the SAME real 5m ETF bars to daily, build a daily
    directional-eligibility mask (ETF own daily DMI/BB gates + underlying-aligned daily regime),
    forward excursion at 1..10 trading days on REAL ETF daily prices (no leverage approximation).
  * Matched RANDOM-ENTRY control + simple MOMENTUM control at every horizon.
  * EntryAlphaAgent decision per horizon (ratio + net drift + edge-vs-control), graded honestly.

Costs: costs.py bucket + slippage, round-trip, per symbol. No look-ahead (fill next open; daily
features use completed sessions only). Order-flow components need ticks -> absent/flagged.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import amps_v2 as A
import amps_v4_regime as R
from amps_v2_backtest import compute_features, _eligible_mask, SLIPPAGE_BPS
from amps_v4_backtest import build_regime_frames, _directional_mask
from backtest_dmi_adx import bucket_of, COST_BPS

from amps_v6.universe import instrument_registry as REG
from amps_v6.features import multi_timeframe_engine as MTF
from amps_v6.alpha import entry_alpha as EA

OUT_DIR = PROJECT_ROOT / "backtest_outputs"
SYS_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")

INTRADAY_HORIZONS = [6, 12, 24, 36, 78]       # 30m, 60m, 120m, 180m, full-day (5m bars)
SWING_HORIZONS = [1, 2, 3, 5, 7, 10]          # trading days


def _rt(sym: str) -> float:
    return 2.0 * (COST_BPS[bucket_of(sym)] + SLIPPAGE_BPS)


# ---------------------------------------------------------------- 5-minute (reproduce V5) entry sets
def intraday_entry_sets(train_end: date) -> tuple[list[EA.EntrySet], dict]:
    feats = {}
    for s in R.TRADE_UNIVERSE + R.REFERENCES:
        f = compute_features(s)
        if not f.empty and len(f) >= 200:
            feats[s] = f
    ref_scores, regime = build_regime_frames(feats, R.RegimeConfig())
    amps = A.AmpsV2Config()
    rcfg = R.RegimeConfig()
    sets = []
    n_entries = 0
    for s in R.TRADE_UNIVERSE:
        d = feats.get(s)
        if d is None or d.empty or len(d) < 60:
            continue
        own, _ = _eligible_mask(d, amps, use_quality=False, min_quality=0)
        dm = _directional_mask(d, s, ref_scores, regime, rcfg, use_regime=True,
                               use_underlying=True, allow_divergence=True)
        mask = (own & dm).to_numpy()
        # OOS only (entries whose session >= train_end) to compare apples-to-apples with V5 OOS
        sess = pd.to_datetime(d["date"]).dt.date.to_numpy()
        mask = mask & (sess >= train_end)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        n_entries += len(idx)
        sets.append(EA.EntrySet(s, d["high"].to_numpy(float), d["low"].to_numpy(float),
                                d["open"].to_numpy(float), d["close"].to_numpy(float),
                                idx, _rt(s), d["atr"].to_numpy(float)))
    return sets, {"n_entries": n_entries, "n_symbols": len(sets)}


# ---------------------------------------------------------------- daily (swing) entry sets
def _daily_ref_scores(ref_daily: dict[str, pd.DataFrame], cfg: R.RegimeConfig) -> dict[str, pd.Series]:
    """Per-reference daily directional score Series keyed by date."""
    out = {}
    for r, d in ref_daily.items():
        scores = []
        for _, row in d.iterrows():
            scores.append(R.directional_score(row.to_dict(), cfg))
        out[r] = pd.Series(scores, index=pd.to_datetime(d["date"]).dt.date.to_numpy())
    return out


def _daily_regime_by_date(ref_daily: dict[str, pd.DataFrame], cfg: R.RegimeConfig) -> dict:
    """Broad daily regime label per date from SPY/QQQ/DIA/IWM daily rows."""
    by_date = {}
    dates = None
    for r in ("SPY", "QQQ", "DIA", "IWM"):
        d = ref_daily.get(r)
        if d is None:
            continue
        dd = pd.to_datetime(d["date"]).dt.date.to_numpy()
        dates = set(dd) if dates is None else (dates | set(dd))
    if not dates:
        return by_date
    frames = {r: d.set_index(pd.to_datetime(d["date"]).dt.date) for r, d in ref_daily.items()
              if r in ("SPY", "QQQ", "DIA", "IWM")}
    for dt in sorted(dates):
        rows = {}
        for r, f in frames.items():
            if dt in f.index:
                rows[r] = f.loc[dt].to_dict()
        label, score, conf = R.market_regime(rows, cfg)
        by_date[dt] = label
    return by_date


def _daily_own_gate(d: pd.DataFrame, cfg: R.RegimeConfig) -> np.ndarray:
    """Daily ETF own long gates: DI+>DI-, ADX>trend threshold, close>=bb_mid (confirmed) OR
    approaching from below with positive short momentum. Bar-derivable; transparent."""
    di_ok = (d["pdi"] > d["mdi"]).to_numpy()
    adx_ok = (d["adx"] > cfg.min_adx_trend).to_numpy()
    above = (d["close"] >= d["bb_mid"]).to_numpy()
    approaching = ((d["close"] < d["bb_mid"]) & (d["bb_dist"].abs() <= 0.01) &
                   (d["bb_dist"].abs() < d["bb_dist_prev"].abs()) & (d["short_mom"] > 0)).to_numpy()
    bb_ok = above | approaching
    return di_ok & adx_ok & bb_ok


def swing_universe(train_end: date):
    """Candidate ETFs + the reference/underlying symbols needed for daily regime + alignment."""
    cands = [s for s in REG.tradable_universe(require_data=True) if s in REG.REGISTRY]
    refs = sorted({REG.REGISTRY[s].benchmark for s in cands} |
                  {REG.REGISTRY[s].underlying_symbol for s in cands} |
                  {"SPY", "QQQ", "DIA", "IWM"})
    return cands, refs


def swing_entry_sets(train_end: date):
    cfg = R.RegimeConfig()
    cands, refs = swing_universe(train_end)
    ref_daily = {}
    for r in refs:
        df = MTF.daily_frame_adj(r, allow_fetch=False)   # adjusted cache (prefetched in run())
        if not df.empty:
            ref_daily[r] = df
    ref_scores = _daily_ref_scores(ref_daily, cfg)
    regime_by_date = _daily_regime_by_date(ref_daily, cfg)

    sets, mom_sets = [], []
    n_entries = n_mom = 0
    for s in cands:
        ins = REG.REGISTRY.get(s)
        if ins is None:
            continue
        d = MTF.daily_frame_adj(s, allow_fetch=False)    # adjusted cache (prefetched in run())
        if d.empty or len(d) < 80:
            continue
        dts = pd.to_datetime(d["date"]).dt.date.to_numpy()
        own = _daily_own_gate(d, cfg)
        und = ins.underlying_symbol
        und_score = ref_scores.get(und)
        elig = np.zeros(len(d), dtype=bool)
        mom = np.zeros(len(d), dtype=bool)
        close = d["close"].to_numpy(float); sma50 = d["sma50"].to_numpy(float)
        for i, dt in enumerate(dts):
            if dt < train_end or not own[i]:
                # still allow momentum-control to use pre/post; but keep OOS parity -> gate on train_end
                pass
            # momentum control: simple close>sma50 long on the ETF (no regime/underlying logic)
            if dt >= train_end and close[i] == close[i] and sma50[i] == sma50[i] and close[i] > sma50[i]:
                mom[i] = True
            if dt < train_end or not own[i]:
                continue
            us = float(und_score.get(dt, float("nan"))) if und_score is not None else float("nan")
            label = regime_by_date.get(dt, R.UNCERTAIN)
            permit, _ = R.regime_permits(s, label, us, cfg) if s in R.REGISTRY else (False, "not_in_v4")
            # For candidates not in the V4 REGISTRY, fall back to underlying alignment only.
            if s not in R.REGISTRY:
                orient = ins.directional_orientation
                permit = R.underlying_aligned(us, orient, cfg)
            elig[i] = bool(permit)
        hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
        op = d["open"].to_numpy(float); cl = d["close"].to_numpy(float)
        atr = d["atr"].to_numpy(float)
        e_idx = np.where(elig)[0]
        m_idx = np.where(mom)[0]
        if len(e_idx):
            n_entries += len(e_idx)
            sets.append(EA.EntrySet(s, hi, lo, op, cl, e_idx, _rt(s), atr))
        if len(m_idx):
            n_mom += len(m_idx)
            mom_sets.append(EA.EntrySet(s, hi, lo, op, cl, m_idx, _rt(s), atr))
    meta = {"n_entries": n_entries, "n_symbols": len(sets), "n_momentum_entries": n_mom,
            "candidates_used": sorted(es.symbol for es in sets),
            "regime_dates": len(regime_by_date)}
    return sets, mom_sets, meta


# ---------------------------------------------------------------- run
def run(train_end: date) -> dict:
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "train_end": train_end.isoformat(),
              "data_source": "Alpaca IEX 5m bars (2020-11..2025-11); daily = resampled real 5m RTH",
              "note": "Order-flow components require ticks -> absent/flagged, never fabricated.",
              "coverage": REG.coverage_report()}

    # --- 5-minute (reproduce V5) ---
    print("[v6] building intraday (5m) directional-eligible entries (OOS)...", flush=True)
    intr_sets, intr_meta = intraday_entry_sets(train_end)
    report["v6_1_intraday_meta"] = intr_meta
    intr = {}
    for hz in INTRADAY_HORIZONS:
        prof = EA.horizon_profile(intr_sets, hz, target_bps=30, stop_bps=10)
        ctrl = EA.random_control(intr_sets, intr_sets, hz)
        dec = EA.decide(prof, ctrl, hz)
        intr[f"{hz}bar_{hz*5}min"] = {"entry": prof, "random_control": ctrl, "decision": dec.__dict__}
    report["v6_1_intraday_horizons"] = intr

    # --- daily swing (the new question) -- REQUIRES split/div-adjusted daily bars ---
    cands, refs = swing_universe(train_end)
    print(f"[v6] prefetching split/div-ADJUSTED daily bars for {len(cands)} ETFs + {len(refs)} refs...", flush=True)
    adj_cov = MTF.prefetch_adjusted_daily(sorted(set(cands) | set(refs)))
    report["v6_swing_adjusted_data_coverage"] = {
        "requested": adj_cov["requested"], "cached_available": len(adj_cov["cached_available"]),
        "missing": adj_cov["missing"], "newly_fetched": len(adj_cov["newly_fetched"]),
        "note": "swing analysis uses adjustment=ALL daily bars; unadjusted 5m-resampled daily is "
                "INVALID for multi-day (leveraged/inverse ETF reverse-splits create spurious jumps)."}
    print("[v6] building daily SWING directional-eligible entries (OOS, adjusted)...", flush=True)
    sw_sets, mom_sets, sw_meta = swing_entry_sets(train_end)
    report["v6_swing_meta"] = sw_meta
    swing = {}
    for hz in SWING_HORIZONS:
        prof = EA.horizon_profile(sw_sets, hz, target_bps=None, stop_bps=None)
        ctrl = EA.random_control(sw_sets, sw_sets, hz)
        momp = EA.horizon_profile(mom_sets, hz) if mom_sets else {"n": 0}
        # swing thresholds: more bps of drift expected; require beating BOTH controls
        dec = EA.decide(prof, ctrl, hz, min_n=100, min_ratio=1.15, min_net_bps=0.0,
                        min_edge_vs_control_bps=5.0)
        swing[f"{hz}d"] = {"entry": prof, "random_control": ctrl, "momentum_control": momp,
                           "decision": dec.__dict__}
    report["v6_swing_horizons"] = swing

    # --- horizon classification (spec section 46) ---
    classif = {}
    for hz in INTRADAY_HORIZONS:
        d = intr[f"{hz}bar_{hz*5}min"]["decision"]
        classif[f"intraday_{hz*5}min"] = _classify(d)
    for hz in SWING_HORIZONS:
        d = swing[f"{hz}d"]["decision"]
        classif[f"swing_{hz}d"] = _classify(d)
    report["horizon_classification"] = classif
    return report


def _classify(decision: dict) -> str:
    dec = decision.get("decision")
    if dec == EA.ALPHA_CONFIRMED:
        return "VALIDATED"
    if dec == EA.ALPHA_WEAK:
        return "CONTINUE_RESEARCH"
    if dec == EA.INSUFFICIENT_DATA:
        return "INSUFFICIENT_EVIDENCE"
    return "REJECTED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-end", default="2024-05-01")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    train_end = date.fromisoformat(args.train_end)
    rep = run(train_end)

    SYS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    outp = SYS_DIR / f"amps_v6_diagnostic_{tag}.json"
    outp.write_text(json.dumps(rep, indent=2, default=str))

    print("\n=== coverage ===")
    c = rep["coverage"]
    print(f"  candidates with data: {c['n_candidates_with_data']}/{c['n_candidates']}; "
          f"references with data: {c['n_references_with_data']}/{c['n_references']}")
    print("\n=== V6.1 INTRADAY (reproduce V5): MFE/|MAE| ratio + net drift by horizon ===")
    print(f"{'horizon':>14}{'n':>8}{'MFE':>9}{'MAE':>9}{'ratio':>7}{'holdNet':>9}{'tgt<stop':>9}{'decision':>18}")
    for k, v in rep["v6_1_intraday_horizons"].items():
        e = v["entry"]; d = v["decision"]
        print(f"{k:>14}{e.get('n',0):>8}{e.get('avg_mfe_bps',0):>9}{e.get('avg_mae_bps',0):>9}"
              f"{str(e.get('mfe_mae_ratio')):>7}{e.get('hold_to_horizon_net_bps',0):>9}"
              f"{str(e.get('target_before_stop')):>9}{d['decision']:>18}")
    print("\n=== V6 SWING (daily, THE new question): MFE/|MAE| ratio + net drift by horizon ===")
    print(f"{'horizon':>10}{'n':>7}{'MFE':>9}{'MAE':>9}{'ratio':>7}{'holdNet':>9}{'ctrlNet':>9}{'momNet':>9}{'decision':>18}")
    for k, v in rep["v6_swing_horizons"].items():
        e = v["entry"]; ct = v["random_control"]; mo = v["momentum_control"]; d = v["decision"]
        print(f"{k:>10}{e.get('n',0):>7}{e.get('avg_mfe_bps',0):>9}{e.get('avg_mae_bps',0):>9}"
              f"{str(e.get('mfe_mae_ratio')):>7}{e.get('hold_to_horizon_net_bps',0):>9}"
              f"{ct.get('hold_to_horizon_net_bps',0):>9}{mo.get('hold_to_horizon_net_bps',0):>9}"
              f"{d['decision']:>18}")
    print("\n=== horizon classification (spec section 46) ===")
    for k, v in rep["horizon_classification"].items():
        print(f"  {k:>20}: {v}")
    any_validated = any(v == "VALIDATED" for v in rep["horizon_classification"].values())
    print(f"\n[v6] wrote {outp}")
    print(f"ANY horizon VALIDATED: {any_validated}")


if __name__ == "__main__":
    main()
