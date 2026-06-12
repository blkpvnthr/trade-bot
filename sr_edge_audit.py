#!/usr/bin/env python3
"""sr_edge_audit.py — does the intraday SR setup have a REAL, leak-free, OOS edge?

"Audit edge first" gate before any live wiring of sr_rl_agent. This is read-only research:
it never trades and never modifies a model.

Pipeline (all causal — features as-of signal time, target from FUTURE bars only):
  1. Fetch historical 1m bars (Alpaca IEX) for a small liquid universe.
  2. Per symbol/day, stream the regular session through PivotReversalScanner + detect_sr_setup
     (the exact labeling sr_rl_agent uses), recording as-of features at each setup signal.
  3. Build a forward-only target: simulate the SR env's trade from the NEXT bar (entry=close,
     stop=max(session_low*0.999, close*0.99), target=entry+risk*RR, max 60 bars / EOD), after
     a round-trip cost. realized_R is derived ONLY from future bars — never from a feature.
  4. LEAKAGE CHECK — reuse leakage_audit.py's univariate + temporal-CV baselines on the
     (features, realized_R) matrix. Any near-deterministic feature or a model that nails the
     future target OOS ⇒ leak (the alpha_entry failure mode).
  5. EDGE CHECK — temporal OOS holdout (last 30% by time): expectancy / profit-factor / win-rate
     per setup type AND overall, vs a RANDOM-ENTRY control on the same bars+sim. Edge exists only
     if setups beat the random control OOS after cost.

Verdict: ACTIVATE-WORTHY only if CLEAN (no leak) AND setups beat random control OOS with PF>1.
Writes docs/sr_edge_audit.md + data/rl_training/sr_setup_dataset.parquet.

CLI:  python sr_edge_audit.py [--symbols QQQ,SPY,IWM,AAPL,NVDA,AMD,MSFT,TSLA] [--days 365] [--rr 3.0]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from pivot_reversal_scanner import PivotReversalScanner
from sr_rl_agent import detect_sr_setup, SRSetupType
import leakage_audit as la

ET = ZoneInfo("America/New_York")

# As-of-signal feature columns fed to the leak harness. NONE of these encode the future.
FEATURES = [
    "support_distance_pct", "resistance_distance_pct", "support_strength", "resistance_strength",
    "rvol", "vwap_distance", "ema9_distance", "ema20_distance",
    "lower_wick_ratio", "upper_wick_ratio", "body_to_range_ratio",
    "red_candle_streak", "green_candle_streak", "minutes_since_open",
    "liquidity_sweep_detected", "support_touch", "pivot_low_reclaim",
    "support_break_failed", "missed_reversal_level_reclaim", "breakout_retest",
]
TARGET = "realized_R"
MAX_BARS = 60
DATASET_PATH = ROOT / "data" / "rl_training" / "sr_setup_dataset.parquet"
REPORT_PATH = ROOT / "docs" / "sr_edge_audit.md"


def fetch_bars(symbols: list[str], days: int) -> pd.DataFrame:
    from build_intraday_training_dataset import get_client, get_feed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    try:
        from alpaca.data.enums import DataFeed
    except Exception:
        DataFeed = None
    client = get_client()
    end = datetime.now(timezone.utc); start = end - timedelta(days=days)
    frames = []
    for sym in symbols:
        kw = dict(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=start, end=end)
        if DataFeed is not None:
            kw["feed"] = DataFeed.IEX   # free historical 1m
        try:
            bars = client.get_stock_bars(StockBarsRequest(**kw))
            if bars is None or bars.df is None or bars.df.empty:
                print(f"  [data] {sym}: empty"); continue
            d = bars.df.reset_index()
            d = d[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
            frames.append(d)
            print(f"  [data] {sym}: {len(d)} 1m bars")
        except Exception as e:
            print(f"  [data] {sym}: FAILED {e}")
    if not frames:
        raise RuntimeError("No bars fetched.")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values(["symbol", "timestamp"])


def session_frames(df: pd.DataFrame):
    """Yield (symbol, day, regular-session DataFrame) ordered, 09:30<=t<16:00 ET."""
    df = df.copy()
    df["et"] = df["timestamp"].dt.tz_convert(ET)
    df["day"] = df["et"].dt.date
    mins = df["et"].dt.hour * 60 + df["et"].dt.minute
    df = df[(mins >= 570) & (mins < 960)]   # 9:30 .. 16:00
    df["minutes_since_open"] = mins - 570
    for (sym, day), g in df.groupby(["symbol", "day"], sort=True):
        if len(g) >= 40:
            yield sym, day, g.sort_values("timestamp").reset_index(drop=True)


def build_session_features(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (g[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(g)
    typ = (h + l + c) / 3.0
    cum_v = np.cumsum(v); cum_pv = np.cumsum(typ * v)
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), c)
    ema9 = pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    vma20 = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    sess_low = np.minimum.accumulate(l)
    pr = np.maximum(h - l, 1e-9); body = np.abs(c - o)
    red = (c < o); grn = (c >= o)
    red_streak = np.zeros(n); grn_streak = np.zeros(n)
    rs = gs = 0
    for i in range(n):
        rs = rs + 1 if red[i] else 0; gs = gs + 1 if grn[i] else 0
        red_streak[i] = rs; grn_streak[i] = gs
    g = g.copy()
    g["vwap"], g["ema9"], g["ema20"], g["volume_ma20"] = vwap, ema9, ema20, vma20
    g["session_low"] = sess_low
    g["rvol"] = v / np.maximum(vma20, 1e-9)
    g["vwap_distance"] = (c - vwap) / np.maximum(vwap, 1e-9)
    g["ema9_distance"] = (c - ema9) / np.maximum(ema9, 1e-9)
    g["ema20_distance"] = (c - ema20) / np.maximum(ema20, 1e-9)
    g["lower_wick_ratio"] = (np.minimum(o, c) - l) / pr
    g["upper_wick_ratio"] = (h - np.maximum(o, c)) / pr
    g["body_to_range_ratio"] = body / pr
    g["red_candle_streak"], g["green_candle_streak"] = red_streak, grn_streak
    return g


def simulate_R(entry: float, stop: float, target: float, hi, lo, cl, cost_R: float) -> float:
    """Forward R from FUTURE bars only. hi/lo/cl are arrays for bars AFTER entry (chronological)."""
    risk = max(entry - stop, 1e-9)
    for k in range(len(hi)):
        if lo[k] <= stop:
            return -1.0 - cost_R
        if hi[k] >= target:
            return (target - entry) / risk - cost_R
    exit_px = cl[-1] if len(cl) else entry
    return (exit_px - entry) / risk - cost_R


def build_dataset(df: pd.DataFrame, rr: float, cost_pct: float) -> pd.DataFrame:
    rows = []
    for sym, day, g in session_frames(df):
        g = build_session_features(g)
        scanner = PivotReversalScanner(symbol=sym)
        prev_bar = None
        H_arr = g["high"].to_numpy(float); L_arr = g["low"].to_numpy(float); C_arr = g["close"].to_numpy(float)
        for i in range(len(g)):
            r = g.iloc[i]
            bar = dict(open=float(r.open), high=float(r.high), low=float(r.low),
                       close=float(r.close), volume=float(r.volume))
            lbl = scanner.push(bar)
            # enrich bar with causal label + features for detect_sr_setup
            ns = lbl.nearest_support or float(r.low) * 0.99
            nr = lbl.nearest_resistance or float(r.high) * 1.01
            bar.update(vwap=float(r.vwap), support_zone=ns, resistance_zone=nr,
                       liquidity_sweep_detected=lbl.liquidity_sweep_detected,
                       pivot_low_reclaim=lbl.pivot_low_reclaim,
                       support_break_failed=lbl.support_break_failed,
                       missed_reversal_level_reclaim=lbl.missed_reversal_level_reclaim,
                       breakout_retest=lbl.breakout_retest)
            setup = detect_sr_setup(bar, prev_bar)
            prev_bar = bar
            if setup == SRSetupType.NONE or i >= len(g) - 2:
                continue
            cl0 = float(r.close); slow = float(r.session_low)
            stop = max(slow * 0.999, cl0 * 0.99); risk = max(cl0 - stop, cl0 * 0.005)
            stop = cl0 - risk; target = cl0 + risk * rr
            j = min(i + 1 + MAX_BARS, len(g))
            cost_R = (cost_pct * cl0) / risk
            realized = simulate_R(cl0, stop, target, H_arr[i + 1:j], L_arr[i + 1:j], C_arr[i + 1:j], cost_R)
            rows.append(dict(
                timestamp=r.timestamp, symbol=sym, setup_type=setup.value,
                support_distance_pct=(cl0 - ns) / max(cl0, 1e-9),
                resistance_distance_pct=(nr - cl0) / max(cl0, 1e-9),
                support_strength=float(lbl.support_strength), resistance_strength=float(lbl.resistance_strength),
                rvol=float(r.rvol), vwap_distance=float(r.vwap_distance),
                ema9_distance=float(r.ema9_distance), ema20_distance=float(r.ema20_distance),
                lower_wick_ratio=float(r.lower_wick_ratio), upper_wick_ratio=float(r.upper_wick_ratio),
                body_to_range_ratio=float(r.body_to_range_ratio),
                red_candle_streak=float(r.red_candle_streak), green_candle_streak=float(r.green_candle_streak),
                minutes_since_open=float(r.minutes_since_open),
                liquidity_sweep_detected=int(lbl.liquidity_sweep_detected), support_touch=int(lbl.support_touch),
                pivot_low_reclaim=int(lbl.pivot_low_reclaim), support_break_failed=int(lbl.support_break_failed),
                missed_reversal_level_reclaim=int(lbl.missed_reversal_level_reclaim),
                breakout_retest=int(lbl.breakout_retest), realized_R=float(realized),
            ))
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def random_control(df: pd.DataFrame, n_signals: int, rr: float, cost_pct: float, oos_start_ts) -> dict:
    """Same forward-sim from RANDOM session bars (OOS window) — the no-skill benchmark."""
    rng = np.random.RandomState(0)
    Rs = []
    sess = list(session_frames(df))
    rng.shuffle(sess)
    for sym, day, g in sess:
        g = build_session_features(g)
        if g["timestamp"].iloc[0] < oos_start_ts:
            continue
        H = g["high"].to_numpy(float); L = g["low"].to_numpy(float); C = g["close"].to_numpy(float)
        for _ in range(max(1, n_signals // max(len(sess), 1) + 1)):
            i = rng.randint(0, len(g) - 2)
            cl0 = float(C[i]); slow = float(g["session_low"].iloc[i])
            stop = max(slow * 0.999, cl0 * 0.99); risk = max(cl0 - stop, cl0 * 0.005)
            stop = cl0 - risk; target = cl0 + risk * rr
            j = min(i + 1 + MAX_BARS, len(g)); cost_R = (cost_pct * cl0) / risk
            Rs.append(simulate_R(cl0, stop, target, H[i + 1:j], L[i + 1:j], C[i + 1:j], cost_R))
        if len(Rs) >= n_signals:
            break
    Rs = np.array(Rs[:n_signals]) if Rs else np.array([0.0])
    return stats_block(Rs)


def stats_block(R: np.ndarray) -> dict:
    R = np.asarray(R, float)
    wins = R[R > 0]; losses = R[R <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    return dict(n=int(len(R)), win_rate=round(float((R > 0).mean()), 4),
                expectancy_R=round(float(R.mean()), 4), profit_factor=round(float(pf), 3),
                median_R=round(float(np.median(R)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="QQQ,SPY,IWM,AAPL,NVDA,AMD,MSFT,TSLA")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--cost-pct", type=float, default=0.0005, help="round-trip cost as frac of price")
    ap.add_argument("--oos-frac", type=float, default=0.30)
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"[1/5] fetching {args.days}d of 1m bars for {syms} ...")
    bars = fetch_bars(syms, args.days)
    print(f"[2/5] building causal SR dataset (RR={args.rr}, cost={args.cost_pct}) ...")
    ds = build_dataset(bars, args.rr, args.cost_pct)
    if ds.empty:
        print("No SR signals produced — abort."); return 1
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(DATASET_PATH)
    print(f"      {len(ds)} SR signals over {ds['symbol'].nunique()} symbols -> {DATASET_PATH}")

    # ---- leakage check (reuse repo harness) ----
    print("[3/5] leakage check (univariate + temporal-CV baselines) ...")
    sub = ds[FEATURES + [TARGET]].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = sub[FEATURES].to_numpy(); y = sub[TARGET].to_numpy()
    uni = la._univariate(X, y, FEATURES)
    base = la._baselines(X, y)
    flagged = [u for u in uni if u["flag"]]
    macro = ((base.get("logreg_oos_auc") or {}).get("mean", 0) or 0) > la.AUC_FLAG or \
            ((base.get("random_forest_oos_auc") or {}).get("mean", 0) or 0) > la.AUC_FLAG or \
            ((base.get("rf_regressor_oos_r2") or {}).get("mean", 0) or 0) > la.R2_FLAG
    leak_verdict = "LEAKAGE DETECTED" if (flagged or macro) else "CLEAN"

    # ---- OOS edge check vs random control ----
    print("[4/5] OOS edge check vs random-entry control ...")
    cut = ds["timestamp"].quantile(1 - args.oos_frac)
    oos = ds[ds["timestamp"] >= cut]
    overall = stats_block(oos[TARGET].to_numpy())
    by_setup = {st: stats_block(grp[TARGET].to_numpy()) for st, grp in oos.groupby("setup_type")}
    ctrl = random_control(bars, max(overall["n"], 200), args.rr, args.cost_pct, cut)

    edge = (leak_verdict == "CLEAN" and overall["profit_factor"] > 1.0
            and overall["expectancy_R"] > ctrl["expectancy_R"] and overall["n"] >= 30)
    final = "EDGE — activation-worthy" if edge else (
        "NO EDGE (clean but no OOS edge over control)" if leak_verdict == "CLEAN" else "LEAK — do not use")

    # ---- report ----
    print("[5/5] writing report ...")
    L = [f"# SR setup edge audit", "",
         f"- universe: `{','.join(syms)}`  | lookback: {args.days}d  | RR: {args.rr}  | cost: {args.cost_pct}",
         f"- signals: {len(ds)} (OOS holdout last {int(args.oos_frac*100)}% = {len(oos)})",
         f"- **Leakage verdict: {leak_verdict}**", f"- **Edge verdict: {final}**", "",
         "## Leakage — baselines (temporal expanding-window CV, OOS)",
         f"- LogReg OOS AUC: {base.get('logreg_oos_auc')}",
         f"- RandomForest OOS AUC: {base.get('random_forest_oos_auc')}",
         f"- RF-Regressor OOS R²: {base.get('rf_regressor_oos_r2')}  (macro-leak={macro})", "",
         f"### Flagged features ({len(flagged)})"]
    if flagged:
        L += ["| feature | AUC | R² | reason |", "|---|--:|--:|---|"]
        L += [f"| `{u['feature']}` | {u['auc']} | {u['r2']} | {u.get('reason','')} |" for u in flagged]
    else:
        L.append("_None — no feature near-determines realized_R._")
    L += ["", "## Top univariate signal (all features)", "| feature | AUC | R² | Spearman² |", "|---|--:|--:|--:|"]
    L += [f"| `{u['feature']}` | {u['auc']} | {u['r2']} | {u['spearman2']} |" for u in uni[:10]]
    L += ["", "## OOS edge (last holdout) vs random-entry control", "",
          f"| group | n | win% | expectancy_R | PF | median_R |", "|---|--:|--:|--:|--:|--:|",
          f"| **setups (all)** | {overall['n']} | {overall['win_rate']} | {overall['expectancy_R']} | {overall['profit_factor']} | {overall['median_R']} |",
          f"| random control | {ctrl['n']} | {ctrl['win_rate']} | {ctrl['expectancy_R']} | {ctrl['profit_factor']} | {ctrl['median_R']} |"]
    for st, s in sorted(by_setup.items()):
        L.append(f"| {st} | {s['n']} | {s['win_rate']} | {s['expectancy_R']} | {s['profit_factor']} | {s['median_R']} |")
    L += ["", "## Interpretation",
          ("**Activation-worthy.** No leakage AND setups beat the random-entry control OOS with PF>1 after cost. "
           "Proceed to train sr_rl_agent and wire as a SECONDARY buy-side filter (paper only)."
           if edge else
           "**Not activation-worthy.** " + ("A feature/model predicts the future target OOS — target leakage; do not use. "
            if leak_verdict != "CLEAN" else
            "Setups do not beat random entries OOS after cost — the apparent profit is stop/target geometry, not signal. "
            "Keep sr_rl_agent frozen; this matches the repo's prior RL-edge findings.")), ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"LEAKAGE: {leak_verdict}   EDGE: {final}")
    print(f"  setups OOS : n={overall['n']} exp={overall['expectancy_R']}R PF={overall['profit_factor']} win={overall['win_rate']}")
    print(f"  random ctrl: n={ctrl['n']} exp={ctrl['expectancy_R']}R PF={ctrl['profit_factor']} win={ctrl['win_rate']}")
    print(f"  report -> {REPORT_PATH}")
    print("=" * 64)
    print("SR_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
