#!/usr/bin/env python3
"""
backtest_catalyst_stack.py — does an earnings-CATALYST composite rank stocks that go up?

Backtestable PIT signals (no lookahead, all known at entry = T+1 after the reported date):
  - transcript LLM sentiment (av_catalyst.transcript_sentiment, mgmt-weighted)
  - earnings surprise % (av_catalyst.earnings reportedEPS vs estimate)
  - news sentiment in [reportedDate-3d, reportedDate+1d] (av_catalyst.news_window, ticker-specific)
(REALTIME put/call & vol/OI and estimate-revisions are EXCLUDED — no usable PIT history.)

Discipline against overfitting:
  - Composite = equal-weight z-scores (z fit on TRAIN ONLY), no per-feature weight tuning.
  - Train/test split by date; ALL reported metrics are OOS (entry >= --train-end).
  - Forward return = entry-day OPEN -> +H-day CLOSE, minus round-trip slippage; excess vs SPY.
  - Reports quintile spread (Q5-Q1) + top-quintile vs SPY: the honest "does it rank winners" test.

Usage:
  python backtest_catalyst_stack.py --quarters 12 --horizon 5 --train-end 2024-01-01 \
      --slip-bps 20 --max-events 400
"""
from __future__ import annotations

import argparse
from datetime import date as _date, datetime, timedelta
from statistics import mean, pstdev
from typing import List, Optional

import numpy as np

import backtest_alpha_momentum_options as mom
import av_catalyst as cat

DEFAULT_UNIVERSE = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AMD AVGO CRM ORCL ADBE NFLX INTC CSCO "
                    "QCOM TXN IBM NOW UBER PYPL SHOP COIN PLTR SNOW DDOG NET CRWD ABNB MU "
                    "BA CAT GE JPM BAC WMT TGT DIS NKE PFE").split()


def _fiscal_quarter(fiscal_date_ending: str) -> Optional[str]:
    try:
        d = datetime.strptime(fiscal_date_ending[:10], "%Y-%m-%d").date()
        return f"{d.year}Q{(d.month - 1) // 3 + 1}"
    except Exception:
        return None


def _spy_window(spy, entry_idx, horizon):
    try:
        o = float(spy["open"].iloc[entry_idx]); c = float(spy["close"].iloc[entry_idx + horizon])
        return c / o - 1.0 if o > 0 else None
    except Exception:
        return None


def build_events(universe, quarters, horizon, slip, max_events, feats=None):
    """Assemble PIT catalyst events: features known at T+1 after reportedDate + forward return.
    Only fetches the data a requested feature needs (e.g. call-flow-only skips transcript/news)."""
    feats = set(feats or ["f_sent", "f_surp", "f_news", "f_pcr", "f_voi"])
    need_sent, need_news = "f_sent" in feats, "f_news" in feats
    need_pcr, need_voi = "f_pcr" in feats, "f_voi" in feats
    spy = mom._bars("SPY")
    cutoff_recent = quarters  # most-recent N quarters per symbol
    events = []
    for sym in universe:
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 60:
            continue
        bdates = [d.date() if hasattr(d, "date") else d for d in bars.index]
        op = bars["open"].tolist(); cl = bars["close"].tolist()
        erows = cat.earnings(sym)
        if not erows:
            continue
        # most-recent `quarters` reports with a reported date
        erows = [e for e in erows if e.get("reportedDate")][:cutoff_recent]
        for e in erows:
            try:
                rep = datetime.strptime(e["reportedDate"][:10], "%Y-%m-%d").date()
            except Exception:
                continue
            # entry = first trading day STRICTLY AFTER the reported date (transcript is public by then)
            entry_idx = next((i for i, d in enumerate(bdates) if d > rep), None)
            if entry_idx is None or entry_idx + horizon >= len(bdates):
                continue
            ts = None
            if need_sent:
                q = _fiscal_quarter(e.get("fiscalDateEnding") or "")
                ts = cat.transcript_sentiment(sym, q) if q else None
                if not ts:
                    continue
            nw = None
            if need_news:
                t_from = (rep - timedelta(days=3)).strftime("%Y%m%dT0000")
                t_to = (rep + timedelta(days=1)).strftime("%Y%m%dT2359")
                nw = cat.news_window(sym, t_from, t_to)
            # options-positioning as-of the reported session (PIT: known before T+1 entry)
            pcr = cat.put_call_ratio(sym, rep.isoformat()) if need_pcr else None
            voi = cat.volume_oi_call_share(sym, rep.isoformat()) if need_voi else None
            entry_px = float(op[entry_idx]); exit_px = float(cl[entry_idx + horizon])
            if entry_px <= 0 or exit_px <= 0:
                continue
            fwd = exit_px / entry_px - 1.0 - slip
            spy_ret = _spy_window(spy, next((i for i, d in enumerate(
                [dd.date() if hasattr(dd, "date") else dd for dd in spy.index]) if d >= bdates[entry_idx]), 0), horizon) if spy is not None else None
            events.append({
                "sym": sym, "date": bdates[entry_idx], "fwd": fwd,
                "excess": (fwd - spy_ret) if spy_ret is not None else None,
                "f_sent": float(ts["mgmt_mean"]) if ts else 0.0,
                "f_surp": float(e.get("surprisePct") or 0.0),
                "f_news": float(nw["mean_score"]) if nw else 0.0,
                "f_pcr": pcr, "f_voi": voi})        # directional (put/call, call-share); may be None
            if cat.quota_hit():
                print(f"  AV quota hit after {len(events)} events; stopping fetch.")
                return events
            if len(events) >= max_events:
                return events
    return events


def _z(vals, mu, sd):
    return [(v - mu) / sd if sd > 0 else 0.0 for v in vals]


def _stat(arr):
    if len(arr) == 0:
        return None
    v = np.array(arr, float)
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    return {"n": len(v), "mean": v.mean(), "win": (v > 0).mean(),
            "sharpe": v.mean() / sd if sd > 0 else 0.0, "pf": pf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarters", type=int, default=12, help="most-recent N reports per symbol")
    ap.add_argument("--horizon", type=int, default=5, help="hold in trading days (T+1 open -> +H close)")
    ap.add_argument("--train-end", default="2024-01-01")
    ap.add_argument("--slip-bps", type=float, default=20.0)
    ap.add_argument("--max-events", type=int, default=400)
    ap.add_argument("--symbols", default="", help="comma list (default: built-in liquid universe)")
    ap.add_argument("--features", default="f_sent,f_surp,f_news,f_pcr,f_voi",
                    help="comma list of features in the composite (isolate call-flow with f_pcr,f_voi)")
    args = ap.parse_args()

    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or DEFAULT_UNIVERSE
    slip = args.slip_bps / 10000.0
    print(f"Building catalyst events: {len(universe)} symbols, last {args.quarters}q, "
          f"horizon {args.horizon}d (fetching transcripts/earnings/news, cached)...")
    ev = build_events(universe, args.quarters, args.horizon, slip, args.max_events)
    if len(ev) < 40:
        print(f"Only {len(ev)} events — insufficient. (quota? coverage?)"); return
    te = _date.fromisoformat(args.train_end)
    train = [e for e in ev if e["date"] < te]
    test = [e for e in ev if e["date"] >= te]
    if len(train) < 20 or len(test) < 20:
        print(f"train={len(train)} test={len(test)} — need >=20 each; widen quarters/universe."); return

    # Features incl. directional options-positioning. Each is z-scored on TRAIN and ORIENTED by the
    # sign of its TRAIN correlation with forward return — so the data (not convention) decides whether
    # e.g. a LOW put/call is bullish or contrarian. None values impute to the train mean (z=0).
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    stats, orient = {}, {}
    fwd_tr = np.array([e["fwd"] for e in train], float)
    for f in feats:
        tv = [e[f] for e in train if e.get(f) is not None]
        mu = mean(tv) if tv else 0.0
        sd = (pstdev(tv) if len(tv) > 1 else 1.0) or 1.0
        stats[f] = (mu, sd)
        fv = np.array([(e[f] if e.get(f) is not None else mu) for e in train], float)
        orient[f] = (1.0 if np.corrcoef(fv, fwd_tr)[0, 1] >= 0 else -1.0) if fv.std() > 0 else 0.0
    for e in ev:
        e["score"] = sum(orient[f] * (((e[f] if e.get(f) is not None else stats[f][0]) - stats[f][0]) / stats[f][1])
                         for f in feats)
    orient_lbl = ", ".join(f"{f.replace('f_','')}{'+' if orient[f] >= 0 else '−'}" for f in feats)

    test.sort(key=lambda e: e["score"])
    n = len(test); qn = n // 5
    quints = [test[i * qn:(i + 1) * qn] for i in range(4)] + [test[4 * qn:]]
    L = [f"# Earnings-Catalyst Composite — OOS quintiles (horizon {args.horizon}d, {args.slip_bps:.0f}bps slip)", "",
         f"Universe {len(universe)} liquid names, last {args.quarters} reports each. PIT entry = T+1 after "
         f"reportedDate. Composite = equal-weight z(transcript mgmt-sentiment, surprise%, news sentiment), "
         f"z fit on TRAIN (<{args.train_end}) only. Metrics below are OOS (entry ≥ train-end).", "",
         f"Events: {len(ev)} total / train {len(train)} / **test {len(test)}**.", "",
         f"Train-learned feature orientation (sign of corr w/ fwd ret): **{orient_lbl}**. "
         f"Q5 = signal says BUY (long/calls); Q1 = signal says SELL/avoid (short/puts).", "",
         "| Quintile (low→high score) | n | Mean fwd | Win% | Sharpe | PF | Mean excess vs SPY |",
         "|---|---|---|---|---|---|---|"]
    for i, qd in enumerate(quints, 1):
        s = _stat([e["fwd"] for e in qd])
        ex = [e["excess"] for e in qd if e["excess"] is not None]
        exm = f"{mean(ex)*100:+.2f}%" if ex else "—"
        pf = "∞" if s and s["pf"] == float("inf") else (f"{s['pf']:.2f}" if s else "—")
        tag = " (top)" if i == 5 else (" (bottom)" if i == 1 else "")
        L.append(f"| Q{i}{tag} | {s['n']} | {s['mean']*100:+.2f}% | {s['win']*100:.0f}% | "
                 f"{s['sharpe']:+.3f} | {pf} | {exm} |")
    q1, q5 = _stat([e["fwd"] for e in quints[0]]), _stat([e["fwd"] for e in quints[-1]])
    spread = (q5["mean"] - q1["mean"]) * 100 if (q1 and q5) else 0.0
    if q5:
        q5_pf = "∞" if q5["pf"] == float("inf") else f"{q5['pf']:.2f}"
        L += ["", f"_**Q5−Q1 spread: {spread:+.2f}%** over {args.horizon}d. Top-quintile mean "
              f"{q5['mean']*100:+.2f}% (Sharpe {q5['sharpe']:+.3f}, PF {q5_pf})._"]
    verdict = ("SIGNAL: top-quintile beats bottom AND is positive net of slippage — worth a larger OOS run."
               if (q5 and q1 and q5["mean"] > q1["mean"] and q5["mean"] > 0) else
               "NO clear signal: composite does not rank winners on this OOS sample. Don't deploy.")
    L += ["", f"**VERDICT: {verdict}**",
          "_Caveat: liquid-name pilot, single horizon. PEAD is strongest in small caps; widen universe "
          "+ test multiple horizons/slippage before trusting. Live confirmation (put/call, vol/OI) not in this test._"]
    out = "\n".join(L)
    from pathlib import Path as _Path
    (_Path(__file__).resolve().parent / "backtest_outputs" / "catalyst_stack_oos.md").write_text(out)
    print("\n" + out)


if __name__ == "__main__":
    main()
