#!/usr/bin/env python3
"""
study_insider_news.py — does AV NEWS_SENTIMENT AMPLIFY the validated vol-gated insider edge?

Leverages the one win: for each vol-gated insider-buy entry, fetch the relevance-weighted news
sentiment for that ticker in the 3 days BEFORE entry (PIT-clean: known at entry). Split entries by
news sentiment and ask whether positive-news insider buys outperform — per-year stable, not overfit.
Does NOT wire anything. AV news history starts ~2022, so this is a 2022-2024 sub-study.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

CACHE = base.PROJECT_ROOT / "data" / "cache" / "av_news"
_CTX = ssl.create_default_context()
try:
    _CTX.load_default_certs()
except Exception:
    pass
_last = 0.0


def _key():
    for line in (base.PROJECT_ROOT / ".env").read_text().splitlines():
        if line.startswith("ALPHA_VANTAGE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("ALPHA_VANTAGE_API_KEY", "")


def _alpaca_news_sentiment(sym: str, day) -> Optional[float]:
    """Relevance-weighted mean FinBERT sentiment for `sym` over [day-3, day] via Alpaca news.
    None if no scored news. NOTE: Alpaca news history coverage differs from AV's ~2022 start."""
    try:
        from datetime import datetime as _dt, timezone as _tz
        from data_sources.alpaca_news_client import get_default_alpaca_news_client
        start = _dt(day.year, day.month, day.day, tzinfo=_tz.utc) - timedelta(days=3)
        end = _dt(day.year, day.month, day.day, tzinfo=_tz.utc)
        rows = get_default_alpaca_news_client().window_news(sym, start, end, limit=200)
    except Exception:
        return None
    num = den = 0.0
    for r in rows:
        s = r.get("ticker_sentiment_score")
        rel = r.get("relevance_score") or 0.0
        if s is not None and rel > 0:
            num += rel * s
            den += rel
    return (num / den) if den > 0 else None


def news_sentiment(sym: str, day) -> Optional[float]:
    """Relevance-weighted mean ticker_sentiment_score for `sym` over [day-3, day]. None if no data."""
    global _last
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{sym}_{day.isoformat()}.json"
    if cf.exists():
        try:
            obj = json.loads(cf.read_text())
            return obj.get("sent")
        except Exception:
            pass
    t_from = (day - timedelta(days=3)).strftime("%Y%m%dT0000")
    t_to = day.strftime("%Y%m%dT0000")
    # NEWS_SOURCE=alpaca → relevance-weighted FinBERT sentiment over [day-3, day] (Alpaca news).
    if os.getenv("NEWS_SOURCE", "av").strip().lower() == "alpaca":
        sent = _alpaca_news_sentiment(sym, day)
        try:
            cf.write_text(json.dumps({"sent": sent, "n_articles": -1, "source": "alpaca"}))
        except Exception:
            pass
        return sent
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(
        {"function": "NEWS_SENTIMENT", "tickers": sym, "time_from": t_from, "time_to": t_to,
         "limit": "200", "sort": "RELEVANCE", "apikey": _key()})
    w = 0.9 - (time.time() - _last)
    if w > 0:
        time.sleep(w)
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "QQ/1.0"}),
                                               timeout=30, context=_CTX).read())
        _last = time.time()
    except Exception:
        _last = time.time()
        return None
    if "Information" in d or "Note" in d:
        return None
    num = den = 0.0
    for art in d.get("feed", []):
        for ts in art.get("ticker_sentiment", []):
            if ts.get("ticker") == sym:
                try:
                    r = float(ts.get("relevance_score", 0)); s = float(ts.get("ticker_sentiment_score", 0))
                    num += r * s; den += r
                except Exception:
                    pass
    sent = (num / den) if den > 0 else None
    try:
        cf.write_text(json.dumps({"sent": sent, "n_articles": len(d.get("feed", []))}))
    except Exception:
        pass
    return sent


def _pf(v):
    w, l = v[v > 0], v[v < 0]
    return (w.sum() / abs(l.sum())) if len(l) else float("inf")


def _stat(v):
    if len(v) < 5:
        return None
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)),
            "win": (v > 0).mean(), "pf": _pf(v)}


def _row(name, v):
    s = _stat(v)
    if not s:
        return f"| {name} | {len(v)} |—|—|—|—|"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"| {name} | {s['n']} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | {s['win']*100:.0f}% | {pf} |"


def main():
    ins = vg._collect(hold=3, min_adv=1_000_000)
    gated = [r for r in ins if r["rv20"] >= np.percentile([x["rv20"] for x in ins if x["date"].year < 2024], 80)]
    gated = [r for r in gated if r["date"].year >= 2022]      # AV news history starts ~2022
    print(f"Fetching news sentiment for {len(gated)} vol-gated insider entries (2022+)...")
    rows = []
    for k, r in enumerate(gated):
        if k % 50 == 0:
            print(f"  ...{k}/{len(gated)}", flush=True)
        sent = news_sentiment(r["sym"], r["date"])
        rows.append({**r, "sent": sent})
    have = [r for r in rows if r["sent"] is not None]
    cov = len(have) / max(len(rows), 1)
    pos = np.array([r["gross"] for r in have if r["sent"] > 0.15])    # bullish news
    neu = np.array([r["gross"] for r in have if -0.15 <= r["sent"] <= 0.15])
    neg = np.array([r["gross"] for r in have if r["sent"] < -0.15])
    allg = np.array([r["gross"] for r in rows])
    L = ["# Insider × News-Sentiment Amplifier (does AV news improve the insider edge?)", "",
         f"Vol-gated insider entries 2022+, 3d hold (gross, no slippage for the relative test). "
         f"News = relevance-weighted ticker_sentiment over [entry-3d, entry]. Coverage: {cov*100:.0f}% "
         f"({len(have)}/{len(rows)} had news).", "",
         "| Cohort | n | Mean | Median | Win% | PF |", "|---|---|---|---|---|---|",
         _row("ALL gated insider (baseline)", allg),
         _row("+ bullish news (sent>0.15)", pos),
         _row("+ neutral news", neu),
         _row("+ bearish news (sent<-0.15)", neg), ""]
    sb = _stat(pos); ba = _stat(allg)
    amp = bool(sb and ba and sb["mean"] > ba["mean"] and sb["pf"] >= ba["pf"] and sb["n"] >= 30)
    # per-year stability of the bullish-news cohort
    L += ["Per-year (bullish-news insider cohort):", "", "| Year | n | Mean | PF |", "|---|---|---|---|"]
    yr_ok = True; n_yr = 0
    for yr in (2022, 2023, 2024):
        yv = np.array([r["gross"] for r in have if r["sent"] > 0.15 and r["date"].year == yr])
        if len(yv) >= 10:
            n_yr += 1; ypf = _pf(yv)
            if ypf < 1.2:
                yr_ok = False
            L.append(f"| {yr} | {len(yv)} | {yv.mean()*100:+.2f}% | {ypf:.2f} |")
    L += ["", f"**VERDICT: {'AMPLIFIES (validate further)' if amp and yr_ok and n_yr>=2 else 'NO AMPLIFICATION'}** — "
          f"bullish-news cohort {'beats' if amp else 'does NOT beat'} baseline on mean+PF"
          f"{'' if (yr_ok and n_yr>=2) else '; not regime-stable / too few years'}. "
          "(Refinement of the insider edge, not an independent signal.)"]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "insider_news_amplifier.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
