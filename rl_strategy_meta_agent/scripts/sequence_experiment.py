#!/usr/bin/env python3
"""Sequence-learning experiment: can a model learn chart behavior directly?

Two targets, same machinery:

  --target giveback  (default, the high-value one)
      At each mid-trade DECISION bar k, feed the model the raw candle window
      ending at k and ask: is a severe giveback (>=1% surrendered from the peak
      ahead) coming? The engineered pre+run+context features already hit AUC
      ~0.79 here -- the question is whether raw sequences match or beat that.

  --target outcome
      Pre-entry candle window -> winner/loser. Entry prediction is ~0.53 with
      engineered features, so this is the lower-probability experiment.

On the SAME rows and the SAME group-by-symbol split we fit the engineered
baseline and the CNN / LSTM / Transformer, so the comparison is apples-to-apples
and honest (no trade leaks across held-out symbols).

Research-only: only a cached historical-bars read. No allocator/broker/order path.

    python scripts/sequence_experiment.py --target giveback --sample-trades 8000 --window 240
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.sequences import (candle_feature_frame, extract_window, N_FEATURES,
                               SEQ_FEATURES, MIN_VALID)
from rl_meta.market_context import build_context_frame, CTX_NUMERIC
from rl_meta.patterns import detect_patterns, FLAG_COLS
from rl_meta.exit_quality import (return_path, running_features, decision_targets,
                                  sample_decision_ks, RUN_FEATURES)
from rl_meta.bar_coverage import cached_bars
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from rl_meta.trade_quality import make_models, classification_metrics, proba


def _frames(bars):
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    ts_ns = pd.to_datetime(bars["timestamp"], utc=True).astype("int64").to_numpy()
    ctx = build_context_frame(bars)
    return bars, ts_ns, ctx, candle_feature_frame(bars, ctx)


def build_outcome(df, pre_cols, bars_dir, tf, window):
    """Pre-entry window -> outcome_win. Engineered baseline = indicators + patterns."""
    X, y, sym, eng = [], [], [], []
    eng_cols = pre_cols + [c for c in FLAG_COLS if c not in pre_cols]
    skipped, syms = 0, df["symbol"].nunique()
    for i, (s, g) in enumerate(df.groupby("symbol"), 1):
        bars = cached_bars(s, tf, bars_dir)
        if bars.empty:
            skipped += len(g); continue
        bars, ts_ns, ctx, frame = _frames(bars)
        for t in g[pre_cols + ["entry_time", "outcome_win"]].itertuples(index=False):
            e = int(np.searchsorted(ts_ns, t.entry_time.value, "right")) - 1
            win, nv = extract_window(frame, e, window)
            if win is None or nv < MIN_VALID:
                skipped += 1; continue
            pat = detect_patterns(bars, e, ctx)
            X.append(win); y.append(float(t.outcome_win)); sym.append(s)
            eng.append([getattr(t, c) for c in pre_cols] +
                       [pat.get(c, np.nan) for c in eng_cols[len(pre_cols):]])
        if i % 10 == 0 or i == syms:
            print(f"  built {i}/{syms} symbols (kept {len(X):,}, skipped {skipped:,})")
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(sym), \
        pd.DataFrame(eng, columns=eng_cols), skipped


def build_giveback(df, pre_cols, bars_dir, tf, window):
    """Decision-bar window -> severe_giveback. Engineered baseline = pre+run+ctx."""
    X, y, sym, eng = [], [], [], []
    eng_cols = pre_cols + list(RUN_FEATURES) + list(CTX_NUMERIC)
    skipped, syms = 0, df["symbol"].nunique()
    for i, (s, g) in enumerate(df.groupby("symbol"), 1):
        bars = cached_bars(s, tf, bars_dir)
        if bars.empty:
            skipped += len(g); continue
        bars, ts_ns, ctx, frame = _frames(bars)
        close = bars["close"].to_numpy(float)
        cols = pre_cols + ["entry_time", "exit_time", "entry_price", "direction"]
        for t in g[cols].itertuples(index=False):
            lo = int(np.searchsorted(ts_ns, t.entry_time.value, "left"))
            r = return_path(ts_ns, close, t.entry_time.value, t.exit_time.value,
                            t.entry_price, t.direction == "long")
            if r is None:
                continue
            pre_vec = [getattr(t, c) for c in pre_cols]
            for k in sample_decision_ks(len(r) - 1):
                abs_bar = lo + k
                win, nv = extract_window(frame, abs_bar, window)
                if win is None or nv < MIN_VALID:
                    skipped += 1; continue
                run = running_features(r, k)
                crow = ctx.iloc[abs_bar]
                X.append(win)
                y.append(float(decision_targets(r, k)["severe_giveback"]))
                sym.append(s)
                eng.append(pre_vec + [run[f] for f in RUN_FEATURES] +
                           [crow[c] for c in CTX_NUMERIC])
        if i % 10 == 0 or i == syms:
            print(f"  built {i}/{syms} symbols (kept {len(X):,}, skipped {skipped:,})")
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(sym), \
        pd.DataFrame(eng, columns=eng_cols), skipped


def _split(symbols, seed=0):
    from sklearn.model_selection import GroupShuffleSplit
    return next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
               .split(symbols, symbols, groups=symbols))


def _sk_auc(Xdf, y, tr, te):
    best = -1
    for _, m in make_models().items():
        m.fit(Xdf.iloc[tr], y[tr])
        best = max(best, classification_metrics(y[te], proba(m, Xdf.iloc[te]))["auc"])
    return round(best, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["giveback", "outcome"], default="giveback")
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--timeframe", default="15Min")
    ap.add_argument("--sample-trades", type=int, default=8000)
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda|auto")
    ap.add_argument("--phase", choices=["all", "engineered", "sequence"], default="all",
                    help="internal: 'all' runs the two phases as separate processes")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / f"seq_{args.target}_{args.sample_trades}_{args.window}_f{N_FEATURES}.npz"

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"] & df["outcome_win"].notna()].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    if args.sample_trades and len(df) > args.sample_trades:
        df = df.sample(args.sample_trades, random_state=0)

    if cache.exists() and not args.rebuild:
        d = np.load(cache, allow_pickle=True)
        X, y, sym = d["X"].astype(np.float32), d["y"], d["sym"]
        eng = pd.DataFrame(d["eng"], columns=list(d["eng_cols"]))
        skipped = int(d["skipped"])
        print(f"loaded cache {cache.name}: X {X.shape}")
    else:
        builder = build_giveback if args.target == "giveback" else build_outcome
        print(f"Building '{args.target}' sequences from {len(df):,} trades, "
              f"window {args.window}, {N_FEATURES} features/candle...")
        X, y, sym, eng, skipped = builder(df, pre_cols, Path(args.bars_dir),
                                          args.timeframe, args.window)
        np.savez_compressed(cache, X=X.astype(np.float16), y=y, sym=sym,
                            eng=eng.to_numpy(), eng_cols=np.array(eng.columns),
                            skipped=skipped)
        print(f"built X {X.shape}, cached -> {cache.name}")

    tr, te = _split(sym)
    print(f"\ntarget {args.target} | rows {len(X):,} | window {args.window} | "
          f"features {N_FEATURES} | pos-rate {y.mean():.3f} | "
          f"train {len(tr):,} / test {len(te):,} (held-out symbols)\n")

    # IMPORTANT: xgboost (libgomp) and torch (libomp) DEADLOCK if both run their
    # OpenMP pools in one process on macOS. So the engineered baseline and the
    # sequence models run as SEPARATE process phases; "all" orchestrates them.
    eng_path = out / f"_phase_eng_{args.target}.json"
    seq_path = out / f"_phase_seq_{args.target}.json"
    if args.phase == "engineered":
        auc = _sk_auc(eng, y, tr, te)
        eng_path.write_text(json.dumps({"engineered_auc": auc}))
        print(f"  engineered features ({eng.shape[1]} cols)  AUC {auc}")
        return
    if args.phase == "sequence":
        from rl_meta.seq_models import fit_eval
        seq = {}
        for mdl in ("tcn", "lstm", "transformer"):
            seq[mdl] = round(fit_eval(X, y, tr, te, mdl, epochs=args.epochs,
                                      device=args.device), 3)
            print(f"  sequence:{mdl:<12}        AUC {seq[mdl]}")
        seq_path.write_text(json.dumps({"sequence_models": seq}))
        return

    import subprocess
    argv = [sys.executable, __file__, "--target", args.target, "--dataset", args.dataset,
            "--bars-dir", args.bars_dir, "--timeframe", args.timeframe,
            "--sample-trades", str(args.sample_trades), "--window", str(args.window),
            "--epochs", str(args.epochs), "--out", args.out, "--device", args.device]
    for ph in ("engineered", "sequence"):
        print(f"--- phase: {ph} (separate process) ---", flush=True)
        subprocess.run(argv + ["--phase", ph], check=True)

    rep = {"target": args.target, "rows": int(len(X)), "window": args.window,
           "features": SEQ_FEATURES, "skipped": int(skipped),
           "pos_rate": round(float(y.mean()), 3),
           **json.loads(eng_path.read_text()), **json.loads(seq_path.read_text())}
    name = f"sequence_experiment_{args.target}.json"
    (out / name).write_text(json.dumps(rep, indent=2, default=str))
    _verdict(rep)
    print(f"\nreport -> {out / name}")


def _verdict(rep):
    base = rep["engineered_auc"]
    seq = max(rep["sequence_models"].values())
    best_m = max(rep["sequence_models"], key=rep["sequence_models"].get)
    print(f"\n=== Verdict ({rep['target']}): raw candle sequences vs engineered features "
          "(held-out symbols) ===")
    print(f"engineered features AUC: {base:.3f}")
    print(f"best sequence model:     {seq:.3f}  ({best_m})")
    if seq >= base + 0.03:
        print("-> Sequences BEAT engineered features: first evidence the system is "
              "learning chart behavior directly, not just consuming indicators.")
    elif seq >= base - 0.02:
        print("-> Sequences ~TIE engineered features: no extra signal captured by the "
              "raw window here; the engineered recipe already encodes it.")
    else:
        print("-> Sequences underperform engineered features at this horizon/size.")
    if rep["target"] == "outcome":
        print("NOTE: entry prediction is ~random; the giveback target is the higher-"
              "probability experiment (run with --target giveback).")


if __name__ == "__main__":
    main()
