#!/usr/bin/env python3
"""
av_catalyst.py — cached fetchers for the earnings-CATALYST stack (Alpha Vantage premium):

  transcript_sentiment(symbol, quarter) -> aggregate LLM call sentiment (mean / mgmt / Q&A)
  earnings(symbol)                       -> [{fiscalDateEnding, reportedDate, reportedEPS,
                                              estimatedEPS, surprise, surprisePct}, ...] (PIT report dates)
  news_window(symbol, t_from, t_to)      -> aggregate ticker news sentiment over a window

Mirrors av_historical_options.py: reads ALPHA_VANTAGE_API_KEY from env/.env, rate-limited,
immutable file cache, fail-safe (returns None/[] on quota/error, never raises). Used ONLY by
the catalyst backtest — credential-free reuse of cached JSON elsewhere is fine.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache" / "av_catalyst"
_RATE = float(os.environ.get("AV_CATALYST_RATE_SEC", "0.9"))
_last_req = 0.0
_quota_hit = False


def _api_key() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if key:
        return key
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("ALPHA_VANTAGE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _get(params: dict, cache_key: str) -> Optional[dict]:
    """Cached AV GET → JSON dict. None on quota/error. Caches successful payloads forever."""
    global _last_req, _quota_hit
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{cache_key}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    if _quota_hit:
        return None
    key = _api_key()
    if not key:
        return None
    wait = _RATE - (time.time() - _last_req)
    if wait > 0:
        time.sleep(wait)
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode({**params, "apikey": key})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuantumQuant/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        _last_req = time.time()
    except Exception:
        _last_req = time.time()
        return None
    if payload.get("Note") or payload.get("Information"):
        _quota_hit = True
        return None
    try:
        cf.write_text(json.dumps(payload))
    except Exception:
        pass
    return payload


def transcript_sentiment(symbol: str, quarter: str) -> Optional[dict]:
    """Aggregate LLM segment sentiment for one earnings call. Returns
    {n, mean, mgmt_mean, qa_mean, pos_frac} or None when no transcript."""
    symbol = symbol.upper().strip()
    p = _get({"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": symbol, "quarter": quarter},
             f"transcript_{symbol}_{quarter}")
    if not p:
        return None
    seg = p.get("transcript") or []
    vals, mgmt, qa = [], [], []
    for s in seg:
        try:
            v = float(s.get("sentiment"))
        except (TypeError, ValueError):
            continue
        vals.append(v)
        title = str(s.get("title") or "").lower()
        # crude management vs analyst/Q&A split by speaker title
        if any(t in title for t in ("chief", "ceo", "cfo", "officer", "president", "founder")):
            mgmt.append(v)
        elif "analyst" in title:
            qa.append(v)
    if not vals:
        return None
    return {"n": len(vals), "mean": mean(vals),
            "mgmt_mean": mean(mgmt) if mgmt else mean(vals),
            "qa_mean": mean(qa) if qa else mean(vals),
            "pos_frac": sum(1 for v in vals if v >= 0.5) / len(vals)}


def earnings(symbol: str) -> List[dict]:
    """Historical quarterly earnings with PIT report dates + surprise. [] on failure."""
    symbol = symbol.upper().strip()
    p = _get({"function": "EARNINGS", "symbol": symbol}, f"earnings_{symbol}")
    if not p:
        return []
    out = []
    for q in (p.get("quarterlyEarnings") or []):
        try:
            def f(k):
                v = q.get(k)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            out.append({"fiscalDateEnding": q.get("fiscalDateEnding"),
                        "reportedDate": q.get("reportedDate"),
                        "reportedEPS": f("reportedEPS"), "estimatedEPS": f("estimatedEPS"),
                        "surprise": f("surprise"), "surprisePct": f("surprisePercentage")})
        except Exception:
            continue
    return out


def news_window(symbol: str, t_from: str, t_to: str, limit: int = 200) -> Optional[dict]:
    """Aggregate ticker news sentiment over [t_from, t_to] (AV YYYYMMDDTHHMM). Returns
    {n, mean_score, mean_relevance, pos_frac} or None. PIT-safe when t_to <= entry date."""
    symbol = symbol.upper().strip()
    p = _get({"function": "NEWS_SENTIMENT", "tickers": symbol, "time_from": t_from,
              "time_to": t_to, "sort": "RELEVANCE", "limit": str(limit)},
             f"news_{symbol}_{t_from}_{t_to}")
    if not p:
        return None
    feed = p.get("feed") or []
    scores, rels = [], []
    for it in feed:
        for ts in (it.get("ticker_sentiment") or []):
            if str(ts.get("ticker")).upper() != symbol:
                continue
            try:
                scores.append(float(ts.get("ticker_sentiment_score")))
                rels.append(float(ts.get("relevance_score") or 0.0))
            except (TypeError, ValueError):
                continue
    if not scores:
        return {"n": 0, "mean_score": 0.0, "mean_relevance": 0.0, "pos_frac": 0.0}
    return {"n": len(scores), "mean_score": mean(scores), "mean_relevance": mean(rels),
            "pos_frac": sum(1 for v in scores if v > 0.15) / len(scores)}


def put_call_ratio(symbol: str, date: str) -> Optional[float]:
    """Full-chain historical put/call ratio for (symbol, YYYY-MM-DD). None on failure.
    PIT: pass a date <= entry. Lower ratio (≤0.6) ≈ bullish; extremes can be contrarian."""
    symbol = symbol.upper().strip()
    p = _get({"function": "HISTORICAL_PUT_CALL_RATIO", "symbol": symbol, "date": date},
             f"pcr_{symbol}_{date}")
    if not p:
        return None
    try:
        return float(p.get("put_call_ratio_full_chain"))
    except (TypeError, ValueError):
        return None


def volume_oi_directional(symbol: str, date: str) -> Optional[dict]:
    """Aggregate the per-contract historical vol/OI chain into a DIRECTIONAL read for (symbol, date).
    Returns {call_voi, put_voi, call_share} where call_share = call activity / (call+put) in [0,1]
    (>0.5 = fresh trading skewed to CALLS → bullish lean). None on failure. PIT: date <= entry."""
    symbol = symbol.upper().strip()
    p = _get({"function": "HISTORICAL_VOLUME_OPEN_INTEREST_RATIO", "symbol": symbol, "date": date},
             f"voi_{symbol}_{date}")
    if not p:
        return None
    call_voi = put_voi = 0.0
    for row in (p.get("data") or []):
        try:
            r = float(row.get("volume_open_interest_ratio") or 0.0)
        except (TypeError, ValueError):
            continue
        if str(row.get("type")).lower() == "call":
            call_voi += r
        elif str(row.get("type")).lower() == "put":
            put_voi += r
    tot = call_voi + put_voi
    if tot <= 0:
        return None
    return {"call_voi": call_voi, "put_voi": put_voi, "call_share": call_voi / tot}


def volume_oi_call_share(symbol: str, date: str) -> Optional[float]:
    """Scalar convenience: fraction of vol/OI activity on the call side (>0.5 = bullish lean)."""
    d = volume_oi_directional(symbol, date)
    return d["call_share"] if d else None


def quota_hit() -> bool:
    return _quota_hit


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "IBM"
    q = sys.argv[2] if len(sys.argv) > 2 else "2024Q1"
    print("transcript:", transcript_sentiment(sym, q))
    e = earnings(sym)
    print(f"earnings rows: {len(e)}; latest:", e[0] if e else None)
