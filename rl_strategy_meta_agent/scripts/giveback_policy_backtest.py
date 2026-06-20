#!/usr/bin/env python3
"""Trade-management policy backtest driven by the engineered p_severe_giveback.

Trains the engineered severe-giveback model (pre + run + market-context features
at each decision bar) on TRAIN symbols, then on HELD-OUT symbols scores every bar
of every trade and simulates five management policies along the realized path:

  1 baseline           original strategy exits
  2 full_exit_p065     close fully when p_giveback >= 0.65
  3 partial_50_p060    sell 50% when p_giveback >= 0.60, ride the rest
  4 tighten_stop_p055  ride until p>=0.55 then a tight trailing stop
  5 de_risk_p060       trim 50% + protective trail when p>=0.60 (no new adds)

Reports total/avg/median return, profit factor, win rate, max drawdown, avoided
severe givebacks, missed profitable recoveries, trade count, and a symbol-level
breakdown -- all on held-out symbols. Research only: no broker/order/risk path.

    python scripts/giveback_policy_backtest.py --sample-trades 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.exit_quality import (return_path, running_features, decision_targets,
                                  sample_decision_ks, RUN_FEATURES)
from rl_meta.market_context import (build_context_frame, CTX_NUMERIC, REF_SYMBOLS,
                                    SECTOR_ETF)
from rl_meta.management_policies import (POLICIES, severe_giveback_happened, EPS,
                                         action_labels)
from rl_meta.bar_coverage import cached_bars, fetch_with_coverage
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from rl_meta.trade_quality import make_models, proba


def _load_refs(start, end, bars_dir):
    refs = {}
    for s in REF_SYMBOLS:
        df = cached_bars(s, "15Min", bars_dir)
        if df.empty:
            df, _ = fetch_with_coverage(s, "15Min", start, end, bars_dir)
        refs[s] = df
    return refs


def _frames(bars, refs, sym):
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    ts_ns = pd.to_datetime(bars["timestamp"], utc=True).astype("int64").to_numpy()
    ctx = build_context_frame(bars, refs.get("SPY"), refs.get("QQQ"),
                              refs.get(SECTOR_ETF.get(sym)))
    return bars, ts_ns, ctx


def _path_and_idx(t, ts_ns, close):
    lo = int(np.searchsorted(ts_ns, t.entry_time.value, "left"))
    r = return_path(ts_ns, close, t.entry_time.value, t.exit_time.value,
                    t.entry_price, t.direction == "long")
    return lo, r


def _label_fn(target: str):
    if target == "severe_giveback":
        return lambda r, k: int(decision_targets(r, k)["severe_giveback"])
    return lambda r, k: int(action_labels(r, k)[target])   # e.g. tighten_stop_better


def train_model(df, pre_cols, bars_dir, refs, target: str = "severe_giveback"):
    """Engineered model on sampled decision bars (train symbols), for any decision
    target ('severe_giveback' or an action label like 'tighten_stop_better')."""
    label = _label_fn(target)
    feat = pre_cols + list(RUN_FEATURES) + list(CTX_NUMERIC)
    rows, ys = [], []
    cols = pre_cols + ["entry_time", "exit_time", "entry_price", "direction"]
    for sym, g in df.groupby("symbol"):
        bars = cached_bars(sym, "15Min", bars_dir)
        if bars.empty:
            continue
        bars, ts_ns, ctx = _frames(bars, refs, sym)
        close = bars["close"].to_numpy(float)
        for t in g[cols].itertuples(index=False):
            lo, r = _path_and_idx(t, ts_ns, close)
            if r is None:
                continue
            pre = [getattr(t, c) for c in pre_cols]
            for k in sample_decision_ks(len(r) - 1):
                run = running_features(r, k)
                crow = ctx.iloc[lo + k]
                rows.append(pre + [run[f] for f in RUN_FEATURES] +
                            [crow[c] for c in CTX_NUMERIC])
                ys.append(label(r, k))
    X = pd.DataFrame(rows, columns=feat)
    y = np.asarray(ys)
    model = make_models().get("xgboost") or make_models()["random_forest"]
    model.fit(X, y)
    return model, feat


def collect_paths(df, pre_cols, feat, model, bars_dir, refs):
    """Per held-out trade: the realized return path r and the per-bar p_giveback.
    Returned once so a threshold sweep can re-apply policies in-memory."""
    out = []
    cols = pre_cols + ["entry_time", "exit_time", "entry_price", "direction"]
    for sym, g in df.groupby("symbol"):
        bars = cached_bars(sym, "15Min", bars_dir)
        if bars.empty:
            continue
        bars, ts_ns, ctx = _frames(bars, refs, sym)
        close = bars["close"].to_numpy(float)
        for t in g[cols].itertuples(index=False):
            lo, r = _path_and_idx(t, ts_ns, close)
            if r is None:
                continue
            pre = [getattr(t, c) for c in pre_cols]
            feats = [pre + [running_features(r, k)[f] for f in RUN_FEATURES] +
                     [ctx.iloc[lo + k][c] for c in CTX_NUMERIC]
                     for k in range(1, len(r))]
            p = proba(model, pd.DataFrame(feats, columns=feat))
            out.append({"symbol": sym, "entry_time": t.entry_time, "exit_time": t.exit_time,
                        "r": r.tolist(), "p": np.concatenate([[0.0], p]).tolist()})
    return out


def simulate(df, pre_cols, feat, model, bars_dir, refs):
    """Score every bar of every held-out trade and apply each policy."""
    recs = []
    cols = pre_cols + ["entry_time", "exit_time", "entry_price", "direction"]
    for sym, g in df.groupby("symbol"):
        bars = cached_bars(sym, "15Min", bars_dir)
        if bars.empty:
            continue
        bars, ts_ns, ctx = _frames(bars, refs, sym)
        close = bars["close"].to_numpy(float)
        for t in g[cols].itertuples(index=False):
            lo, r = _path_and_idx(t, ts_ns, close)
            if r is None:
                continue
            pre = [getattr(t, c) for c in pre_cols]
            feats = [pre + [running_features(r, k)[f] for f in RUN_FEATURES] +
                     [ctx.iloc[lo + k][c] for c in CTX_NUMERIC]
                     for k in range(1, len(r))]
            p = proba(model, pd.DataFrame(feats, columns=feat))   # P(severe) per bar k>=1
            p_full = np.concatenate([[0.0], p])                   # align to r index
            rec = {"symbol": sym, "exit_time": t.exit_time,
                   "severe_gb": bool(severe_giveback_happened(r))}
            for name, fn in POLICIES.items():
                rec[name] = float(fn(r, p_full))
            recs.append(rec)
    return pd.DataFrame(recs)


def _maxdd(returns):
    eq = np.cumsum(returns)
    peak = np.maximum.accumulate(eq)
    return round(float(np.max(peak - eq)), 4) if len(eq) else 0.0


def _profit_factor(x):
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    return round(float(up / dn), 3) if dn > 0 else float("inf")


def metrics(sim):
    sim = sim.sort_values("exit_time")
    base = sim["baseline"].to_numpy()
    out = {}
    for name in POLICIES:
        x = sim[name].to_numpy()
        avoided = int(((sim["severe_gb"]) & (x > base + EPS)).sum())
        missed = int((base > x + EPS).sum())
        out[name] = {
            "trades": int(len(x)),
            "total_return": round(float(x.sum()), 4),
            "avg_trade_return": round(float(x.mean()), 5),
            "median_trade_return": round(float(np.median(x)), 5),
            "profit_factor": _profit_factor(x),
            "win_rate": round(float((x > 0).mean()), 3),
            "max_drawdown": _maxdd(x),
            "avoided_severe_givebacks": avoided,
            "missed_profitable_recoveries": missed,
        }
    return out


def symbol_breakdown(sim, min_n=20):
    out = {}
    for sym, g in sim.groupby("symbol"):
        if len(g) < min_n:
            continue
        out[sym] = {"trades": int(len(g)),
                    **{name: round(float(g[name].sum()), 4) for name in POLICIES}}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["trades"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--sample-trades", type=int, default=8000)
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    if args.sample_trades and len(df) > args.sample_trades:
        df = df.sample(args.sample_trades, random_state=0)

    start = df["entry_time"].min().date().isoformat()
    end = (df["exit_time"].max() + pd.Timedelta(days=1)).date().isoformat()
    refs = _load_refs(start, end, Path(args.bars_dir))

    from sklearn.model_selection import GroupShuffleSplit
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0)
                  .split(df, groups=df["symbol"]))
    train_df, test_df = df.iloc[gi], df.iloc[ge]
    print(f"train symbols {train_df['symbol'].nunique()} ({len(train_df):,} trades) | "
          f"held-out symbols {test_df['symbol'].nunique()} ({len(test_df):,} trades)")

    print("training engineered p_severe_giveback model...")
    model, feat = train_model(train_df, pre_cols, Path(args.bars_dir), refs)
    print("simulating policies on held-out symbols...")
    sim = simulate(test_df, pre_cols, feat, model, Path(args.bars_dir), refs)
    if sim.empty:
        print("no held-out trades with usable paths."); return

    rep = {"train_symbols": int(train_df["symbol"].nunique()),
           "test_symbols": int(test_df["symbol"].nunique()),
           "held_out_trades": int(len(sim)),
           "severe_giveback_rate": round(float(sim["severe_gb"].mean()), 3),
           "policies": metrics(sim), "symbol_breakdown": symbol_breakdown(sim)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sim.to_csv(out / "giveback_policy_trades.csv", index=False)
    (out / "giveback_policy_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nreport -> {out / 'giveback_policy_report.json'}")


def _print(rep):
    print(f"\nHeld-out trades: {rep['held_out_trades']:,} | severe-giveback rate "
          f"{rep['severe_giveback_rate']:.2f}\n")
    h = f"{'policy':<20}{'total':>8}{'avg':>8}{'PF':>7}{'win':>7}{'maxDD':>8}{'avoid':>7}{'miss':>7}"
    print(h); print("-" * len(h))
    for name, m in rep["policies"].items():
        print(f"{name:<20}{m['total_return']:>8.3f}{m['avg_trade_return']*100:>7.2f}%"
              f"{m['profit_factor']:>7}{m['win_rate']:>7.2f}{m['max_drawdown']:>8.3f}"
              f"{m['avoided_severe_givebacks']:>7}{m['missed_profitable_recoveries']:>7}")
    base = rep["policies"]["baseline"]["total_return"]
    best = max(rep["policies"], key=lambda k: rep["policies"][k]["total_return"])
    bt = rep["policies"][best]["total_return"]
    print(f"\nbaseline total {base:.3f} | best policy '{best}' total {bt:.3f} "
          f"({'+' if bt >= base else ''}{bt - base:.3f} vs baseline)")


if __name__ == "__main__":
    main()
