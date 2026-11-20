#!/usr/bin/env python3
"""
amps_v4_backtest.py -- AMPS V4: regime + directional-eligibility experiment sequence on
Universe V4 (20 leveraged/inverse ETFs), actual IEX 5-min bars, reusing the amps_v2 engine +
feature cache and the amps_v4_regime engine.

The central question: does requiring the ETF's UNDERLYING momentum to align with the ETF's
orientation (go long SQQQ only when QQQ is falling; long TQQQ only when QQQ is rising) improve
net expectancy vs the unfiltered AMPS baseline?

Experiments: V4.0 baseline (ETF own AMPS gates only) / V4.1 broad-regime filter / V4.2
underlying-directional filter / V4.3 combined / V4.4 combined + divergence overrides /
V4.10 combined + max-5-positions + opposing-pair prevention.

Honest scope (sec 16/18): bar-supported research (DI/ADX/Bollinger/VWAP/relvol are bar-derivable;
order-flow score components require ticks and are absent/flagged -- never fabricated). No
look-ahead: completed-bar signal -> next-bar-open fill; conservative intrabar stop/target.
Chronological IS/OOS; double-cost stress; acceptance criteria applied verbatim. Reference
instruments are used ONLY for signals, never traded.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone, date, time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
import amps_v2 as A
import amps_v4_regime as R
from amps_v2_backtest import (compute_features, _eligible_mask, metrics, check_acceptance,
                              ATrade, SLIPPAGE_BPS)
from backtest_dmi_adx import bucket_of, COST_BPS

OUT_DIR = PROJECT_ROOT / "system-data" / datetime.now().strftime("%Y%m%d")


def _directional_score_series(feat: pd.DataFrame, cfg: R.RegimeConfig) -> pd.Series:
    """Vectorized directional score [-100,100] per bar for a reference feature frame."""
    pdi, mdi, adx = feat["pdi"], feat["mdi"], feat["adx"]
    di_dir = ((pdi - mdi) / 25.0).clip(-1, 1)
    adx_conf = (((adx - cfg.min_adx_trend) / 15.0)).clip(0, 1)
    w = cfg.weights
    vwap_pos = np.sign(feat["vwap_dist"].fillna(0))
    vslope = np.sign(feat.get("bb_slope", pd.Series(0, index=feat.index)).fillna(0))  # proxy for vwap slope direction
    bb_align = np.sign(feat["bb_dist"].fillna(0))
    mom = np.sign(feat["short_mom"].fillna(0))
    score = 100.0 * (w["vwap_pos"] * vwap_pos + w["di_sep"] * di_dir + w["adx_strength"] * (di_dir * adx_conf)
                     + w["short_mom"] * mom + w["vwap_slope"] * vslope + w["bb_align"] * bb_align)
    return score.clip(-100, 100)


def build_regime_frames(feats: dict[str, pd.DataFrame], cfg: R.RegimeConfig) -> tuple[dict, pd.Series]:
    """Return (per-reference ts->score dict, broad-regime ts->label Series)."""
    ref_scores: dict[str, pd.Series] = {}
    for ref in R.REFERENCES:
        f = feats.get(ref)
        if f is None or f.empty:
            continue
        s = _directional_score_series(f, cfg)
        s.index = pd.to_datetime(f["ts"])
        ref_scores[ref] = s
    # broad regime from SPY/QQQ/DIA/IWM aligned on ts
    broad_parts = [ref_scores[r].rename(r) for r in ("SPY", "QQQ", "DIA", "IWM") if r in ref_scores]
    if not broad_parts:
        return ref_scores, pd.Series(dtype=object)
    bdf = pd.concat(broad_parts, axis=1).sort_index().ffill()
    prim = bdf[[c for c in ("SPY", "QQQ") if c in bdf.columns]].mean(axis=1)
    blended = 0.7 * prim + 0.3 * bdf.mean(axis=1)
    labels = pd.Series(R.SIDEWAYS, index=bdf.index)
    labels[blended >= cfg.confirmed_thresh] = R.CONFIRMED_BULLISH
    labels[(blended >= cfg.emerging_thresh) & (blended < cfg.confirmed_thresh)] = R.EMERGING_BULLISH
    labels[blended <= -cfg.confirmed_thresh] = R.CONFIRMED_BEARISH
    labels[(blended <= -cfg.emerging_thresh) & (blended > -cfg.confirmed_thresh)] = R.EMERGING_BEARISH
    return ref_scores, labels


def _directional_mask(feat: pd.DataFrame, symbol: str, ref_scores: dict, regime: pd.Series,
                      cfg: R.RegimeConfig, use_regime: bool, use_underlying: bool,
                      allow_divergence: bool) -> pd.Series:
    """Vectorized directional-eligibility mask aligned to `feat` bars."""
    meta = R.REGISTRY[symbol]
    orient = meta["dir"]; ref = meta["ref"]
    idx = pd.to_datetime(feat["ts"])
    u = ref_scores.get(ref)
    if u is None:
        return pd.Series(False, index=feat.index)
    uscore = u.reindex(idx, method="ffill").to_numpy()
    reg = regime.reindex(idx, method="ffill").to_numpy() if not regime.empty else np.array([R.UNCERTAIN] * len(feat))
    aligned = (uscore >= cfg.underlying_align_thresh) if orient == "bull" else (uscore <= -cfg.underlying_align_thresh)
    out = np.ones(len(feat), dtype=bool)
    if use_underlying:
        out &= aligned
    if use_regime:
        bull_reg = np.isin(reg, [R.CONFIRMED_BULLISH, R.EMERGING_BULLISH])
        bear_reg = np.isin(reg, [R.CONFIRMED_BEARISH, R.EMERGING_BEARISH])
        reg_ok = bull_reg if orient == "bull" else bear_reg
        if allow_divergence:
            diverg = aligned & (np.abs(uscore) >= cfg.divergence_thresh)
            reg_ok = reg_ok | diverg
        # block uncertain/high-vol handled implicitly (not bull/bear regime -> reg_ok False unless divergence)
        out &= reg_ok
    return pd.Series(out, index=feat.index).fillna(False)


def simulate_v4(feat: pd.DataFrame, symbol: str, cfg_amps: A.AmpsV2Config, cost_bps: float,
                dir_mask: pd.Series, cost_mult: float = 1.0) -> list[ATrade]:
    """AMPS own-gate mask (quality off for V4 base) intersected with the directional mask; then
    the amps_v2 adaptive exit with conservative intrabar fills."""
    if feat.empty or len(feat) < 60:
        return []
    own, _rej = _eligible_mask(feat, cfg_amps, use_quality=False, min_quality=0)
    elig = (own & dir_mask).to_numpy()
    o = feat["open"].values; h = feat["high"].values; l = feat["low"].values; c = feat["close"].values
    atr = feat["atr"].values; ettime = feat["et_time"].values; dates = feat["date"].values
    n = len(feat); rt = (cost_bps * cost_mult + SLIPPAGE_BPS) / 1e4
    trades: list[ATrade] = []
    i = 0; force = time(15, 55)
    while i < n - 1:
        if not elig[i]:
            i += 1; continue
        ei = i + 1
        entry_px = o[ei] * (1 + rt)
        peak = entry_px; dip = entry_px
        pos = A.new_amps_position(symbol, entry_px, str(dates[ei]), 1.0, cfg_amps,
                                  float(atr[ei]) if atr[ei] == atr[ei] else None, A.BB_ABOVE)
        exit_px = c[n - 1]; reason = A.EOD_EXIT; xi = n - 1
        for j in range(ei, n):
            now_et = datetime.combine(dates[j], ettime[j]); hold = (j - ei) * 5.0
            if l[j] <= pos.effective_stop and now_et.time() < force:
                exit_px = min(pos.effective_stop, o[j])
                reason = (A.TRAILING_STOP if pos.stage == "TRAILING" else A.MIN_PROFIT_LOCK if pos.stage == "MIN_PROFIT_LOCK"
                          else A.BREAKEVEN_STOP if pos.stage == "BREAKEVEN" else A.INITIAL_STOP); xi = j; break
            if (h[j] / entry_px - 1) * 1e4 >= cfg_amps.final_target_bps and now_et.time() < force:
                exit_px = entry_px * (1 + cfg_amps.final_target_bps / 1e4); reason = A.PROFIT_TARGET; xi = j; break
            ex, r = A.amps_exit_decision(pos, float(h[j]), now_et, cfg_amps, force, hold, bearish_signals=0)
            peak = max(peak, float(h[j])); dip = min(dip, float(l[j]))
            if ex:
                exit_px = o[j] if (j > ei and r in (A.EOD_EXIT, A.MAX_HOLD_EXIT)) else c[j]; reason = r; xi = j; break
        exit_net = exit_px * (1 - rt)
        trades.append(ATrade(symbol, dates[ei], entry_px, exit_net, (exit_net / entry_px - 1) * 1e4,
                             (exit_px / entry_px - 1) * 1e4, (xi - ei) * 5.0, reason,
                             (peak / entry_px - 1) * 1e4, (dip / entry_px - 1) * 1e4))
        i = xi + 1
    return trades


def _apply_portfolio_cap(trades: list[ATrade], max_positions: int) -> list[ATrade]:
    """Cross-ETF max-concurrent + opposing-pair prevention per day (earliest entries kept)."""
    by_day: dict[Any, list[ATrade]] = {}
    for t in trades:
        by_day.setdefault(t.session, []).append(t)
    kept = []
    for d, ts in by_day.items():
        ts.sort(key=lambda t: t.entry_px)  # stable-ish; no intraday entry time stored -> approx
        chosen = []; used_pairs = set()
        for t in ts:
            pair = R.REGISTRY[t.symbol]["pair"]
            if pair in used_pairs:            # one directional exposure per underlying pair
                continue
            if len(chosen) >= max_positions:
                continue
            chosen.append(t); used_pairs.add(pair)
        kept.extend(chosen)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-end", default="2024-05-01")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    train_end = date.fromisoformat(args.train_end)
    rcfg = R.RegimeConfig()
    amps = A.AmpsV2Config()

    allsym = R.TRADE_UNIVERSE + R.REFERENCES
    feats = {}
    for s in allsym:
        f = compute_features(s)
        if not f.empty and len(f) >= 200:
            feats[s] = f
    ref_scores, regime = build_regime_frames(feats, rcfg)
    etfs = [s for s in R.TRADE_UNIVERSE if s in feats]
    print(f"[amps-v4] ETFs with features={len(etfs)}/20 refs={len([r for r in R.REFERENCES if r in feats])} "
          f"regime_bars={len(regime)} train<{train_end}")

    # experiment grid: (name, use_regime, use_underlying, allow_divergence, portfolio_cap, cost_mult)
    grid = [
        ("V4.0_baseline", False, False, False, False, 1.0),
        ("V4.1_regime", True, False, False, False, 1.0),
        ("V4.2_underlying", False, True, False, False, 1.0),
        ("V4.3_combined", True, True, False, False, 1.0),
        ("V4.4_divergence", True, True, True, False, 1.0),
        ("V4.10_combined_portfolio", True, True, True, True, 1.0),
        ("V4.11_combined_2xcost", True, True, True, True, 2.0),
    ]
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                              "data_source": "Alpaca IEX 5m bars (bar-supported; order-flow absent/flagged)",
                              "universe": etfs, "references": [r for r in R.REFERENCES if r in feats],
                              "train_end": args.train_end, "acceptance": {}, "experiments": {}}
    from amps_v2_backtest import ACCEPT
    report["acceptance"] = ACCEPT
    for name, u_reg, u_und, div, cap, cmult in grid:
        all_tr: list[ATrade] = []
        by_orient = {"bull": [], "bear": []}
        for s in etfs:
            dm = _directional_mask(feats[s], s, ref_scores, regime, rcfg, u_reg, u_und, div)
            tr = simulate_v4(feats[s], s, amps, COST_BPS[bucket_of(s)], dm, cmult)
            all_tr.extend(tr)
            by_orient[R.REGISTRY[s]["dir"]].extend(tr)
        if cap:
            all_tr = _apply_portfolio_cap(all_tr, args.max_positions)
        is_tr = [t for t in all_tr if t.session < train_end]
        oos_tr = [t for t in all_tr if t.session >= train_end]
        m_oos = metrics(oos_tr)
        ok, fails = check_acceptance(m_oos)
        report["experiments"][name] = {
            "config": {"regime": u_reg, "underlying": u_und, "divergence": div, "portfolio_cap": cap, "cost_mult": cmult},
            "overall": metrics(all_tr), "in_sample": metrics(is_tr), "out_of_sample": m_oos,
            "oos_bull_etfs": metrics([t for t in oos_tr if R.REGISTRY[t.symbol]["dir"] == "bull"]),
            "oos_bear_etfs": metrics([t for t in oos_tr if R.REGISTRY[t.symbol]["dir"] == "bear"]),
            "accepts": ok, "acceptance_fails": fails,
        }
        print(f"  {name}: OOS trades={len(oos_tr)} PF={m_oos.get('profit_factor')} win={m_oos.get('win_rate')} "
              f"exp_bps={m_oos.get('net_expectancy_bps')} accepts={ok}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%H%M%S")
    (OUT_DIR / f"amps_v4_backtest_{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    cols = ["trades", "win_rate", "net_expectancy_bps", "profit_factor", "net_sharpe", "profitable_days", "max_drawdown"]
    names = list(report["experiments"])
    print("\n============ OUT-OF-SAMPLE (net of costs, bars) ============")
    print(f"{'metric':<18}" + "".join(f"{n.split('_')[0]:>11}" for n in names))
    for col in cols:
        print(f"{col:<18}" + "".join(f"{str(report['experiments'][n]['out_of_sample'].get(col)):>11}" for n in names))
    print("\n-- OOS net expectancy (bps): bull ETFs | bear ETFs --")
    for n in names:
        e = report["experiments"][n]
        print(f"  {n}: bull={e['oos_bull_etfs'].get('net_expectancy_bps')}(n={e['oos_bull_etfs'].get('trades')}) "
              f"bear={e['oos_bear_etfs'].get('net_expectancy_bps')}(n={e['oos_bear_etfs'].get('trades')})")
    any_pass = any(report["experiments"][n]["accepts"] for n in names)
    print(f"\n[amps-v4] wrote {OUT_DIR / f'amps_v4_backtest_{tag}.json'}")
    print(f"OVERALL: {'A VARIANT PASSES ACCEPTANCE' if any_pass else 'NO VARIANT MEETS ACCEPTANCE CRITERIA'}")


if __name__ == "__main__":
    main()
