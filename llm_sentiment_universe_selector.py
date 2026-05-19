#!/usr/bin/env python3
"""
llm_sentiment_universe_selector.py

LLM sentiment and financials REPORTING layer.

This module does NOT select tickers by default.
Universe selection is handled exclusively by tradingview_institutional_screener.py.

Default flow:
  1. Ensure TradingView universe exists (calls build_tradingview_institutional_universe()).
  2. Load TradingView-selected symbols.
  3. Enrich with yfinance financials.
  4. Optionally call OpenAI for sentiment/catalyst analysis.
  5. Write sentiment, catalyst, and financials reports.
  6. Re-write compatibility universe files UNCHANGED (TV selection is preserved).

To allow LLM to override universe ranking (not recommended):
  LLM_ALLOW_RERANK_UNIVERSE=1   (default: 0)

Report outputs:
  system-data/YYYYMMDD/YYYYMMDD_llm_universe_sentiment_report.json
  system-data/YYYYMMDD/YYYYMMDD_llm_universe_sentiment_report.csv
  system-data/YYYYMMDD/YYYYMMDD_llm_universe_financials_report.json

Compatibility outputs (always preserved from TV screener):
  system-data/YYYYMMDD/YYYYMMDD_latest_llm_sentiment_universe.json
  system-data/YYYYMMDD/YYYYMMDD_latest_llm_candidate_discovery.json
  YYYYMMDD_locked_trading_universe.txt

Standalone usage:
  python llm_sentiment_universe_selector.py

Env vars:
  LLM_SENTIMENT_REPORT_ENABLED=1
  LLM_ALLOW_RERANK_UNIVERSE=0
  LLM_SENTIMENT_REPORT_MODEL=gpt-4.1-mini
  LLM_SENTIMENT_REPORT_MAX_SYMBOLS=30
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

ET = ZoneInfo("America/New_York")

try:
    from market_context_builder import TODAY_TAG as _MCB_TODAY
except Exception:
    _MCB_TODAY = datetime.now(ET).strftime("%Y%m%d")

DATA_TAG: str = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", _MCB_TODAY))
TODAY_TAG: str = DATA_TAG  # alias for backward compatibility

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config via env vars
# ---------------------------------------------------------------------------

LLM_SENTIMENT_REPORT_ENABLED: bool = os.getenv("LLM_SENTIMENT_REPORT_ENABLED", "1").lower() in {"1", "true", "yes"}
LLM_ALLOW_RERANK_UNIVERSE: bool = os.getenv("LLM_ALLOW_RERANK_UNIVERSE", "0").lower() in {"1", "true", "yes"}
LLM_SENTIMENT_REPORT_MODEL: str = os.getenv(
    "LLM_SENTIMENT_REPORT_MODEL",
    os.getenv("OPENAI_SENTIMENT_UNIVERSE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
)
LLM_SENTIMENT_REPORT_MAX_SYMBOLS: int = int(os.getenv("LLM_SENTIMENT_REPORT_MAX_SYMBOLS", "30"))

# Legacy compat env vars (still respected for yfinance settings)
TARGET_COUNT: int = int(os.getenv("TARGET_UNIVERSE_COUNT", str(LLM_SENTIMENT_REPORT_MAX_SYMBOLS)))
MODEL: str = LLM_SENTIMENT_REPORT_MODEL

OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
YF_BATCH_SIZE: int = int(os.getenv("YF_BATCH_SIZE", "25"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TV_UNIVERSE_JSON = OUT_DIR / f"{DATA_TAG}_latest_llm_sentiment_universe.json"
TV_ACCUMULATION_JSON = OUT_DIR / f"{DATA_TAG}_tradingview_institutional_accumulation.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"

# Report outputs
SENTIMENT_REPORT_JSON = OUT_DIR / f"{DATA_TAG}_llm_universe_sentiment_report.json"
SENTIMENT_REPORT_CSV = OUT_DIR / f"{DATA_TAG}_llm_universe_sentiment_report.csv"
FINANCIALS_REPORT_JSON = OUT_DIR / f"{DATA_TAG}_llm_universe_financials_report.json"

# Compatibility paths (re-exported for run.py import)
JSON_PATH = TV_UNIVERSE_JSON
DISCOVERY_PATH = OUT_DIR / f"{DATA_TAG}_latest_llm_candidate_discovery.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(val: Any) -> str:
    return str(val or "").upper().strip().replace(".", "-")


def is_valid_symbol(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", s))


def to_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _safe_json(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    raise TypeError(f"Not serializable: {type(o)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_safe_json) + "\n", encoding="utf-8")
    print(f"[llm-report] wrote {path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean[k] = None
                elif isinstance(v, list):
                    clean[k] = "|".join(str(x) for x in v)
                else:
                    clean[k] = v
            writer.writerow(clean)
    print(f"[llm-report] wrote {path}")


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------


def _load_fallback_symbols() -> list[str]:
    for candidate in [
        LOCKED_UNIVERSE_PATH,
        PROJECT_ROOT / "locked_trading_universe.txt",
        PROJECT_ROOT / f"{(datetime.now(ET) - timedelta(days=1)).strftime('%Y%m%d')}_locked_trading_universe.txt",
    ]:
        if candidate.exists():
            syms = [s.strip() for s in candidate.read_text().splitlines() if s.strip()]
            syms = [normalize_symbol(s) for s in syms if is_valid_symbol(normalize_symbol(s))]
            if syms:
                return syms
    return []


def _ensure_tv_universe() -> dict[str, Any]:
    """Ensure TradingView universe exists. Build if missing or empty."""
    if TV_UNIVERSE_JSON.exists():
        try:
            payload = json.loads(TV_UNIVERSE_JSON.read_text())
            if payload.get("symbols"):
                return payload
        except Exception as exc:
            print(f"[llm-report] existing TV universe unreadable: {exc}")

    print("[llm-report] TradingView universe missing or empty; running screener")
    try:
        from tradingview_institutional_screener import build_tradingview_institutional_universe
        return build_tradingview_institutional_universe()
    except Exception as exc:
        print(f"[llm-report] tradingview_institutional_screener failed: {exc}")
        return {}


def load_tradingview_universe() -> tuple[list[str], dict[str, Any]]:
    payload = _ensure_tv_universe()
    symbols = [normalize_symbol(s) for s in payload.get("symbols", []) if s]
    symbols = [s for s in symbols if is_valid_symbol(s)]

    if not symbols:
        print("[llm-report] no symbols in TV universe; trying locked universe fallback")
        symbols = _load_fallback_symbols()

    symbols = symbols[:LLM_SENTIMENT_REPORT_MAX_SYMBOLS]
    return symbols, payload


# ---------------------------------------------------------------------------
# yfinance financial enrichment
# ---------------------------------------------------------------------------


def enrich_company_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info: dict[str, Any] = {}
            try:
                info = ticker.get_info() or {}
            except Exception:
                pass
            fast_info: dict[str, Any] = {}
            try:
                fast_info = dict(ticker.fast_info or {})
            except Exception:
                pass
            merged = {**fast_info, **info}
            enriched[symbol] = {
                "company_name": merged.get("shortName") or merged.get("longName") or symbol,
                "sector": merged.get("sector"),
                "industry": merged.get("industry"),
                "market_cap": merged.get("marketCap") or fast_info.get("market_cap"),
                "enterprise_value": merged.get("enterpriseValue"),
                "trailing_pe": merged.get("trailingPE"),
                "forward_pe": merged.get("forwardPE"),
                "revenue_growth": merged.get("revenueGrowth"),
                "earnings_growth": merged.get("earningsGrowth"),
                "profit_margins": merged.get("profitMargins"),
                "gross_margins": merged.get("grossMargins"),
                "operating_margins": merged.get("operatingMargins"),
                "beta": merged.get("beta"),
                "float_shares": merged.get("floatShares"),
                "shares_outstanding": merged.get("sharesOutstanding"),
                "short_ratio": merged.get("shortRatio"),
                "short_percent_of_float": merged.get("shortPercentOfFloat"),
                "average_volume": merged.get("averageVolume") or fast_info.get("three_month_average_volume"),
                "current_price": merged.get("currentPrice") or fast_info.get("last_price"),
                "target_mean_price": merged.get("targetMeanPrice"),
                "target_high_price": merged.get("targetHighPrice"),
                "analyst_recommendation": merged.get("recommendationKey"),
                "number_of_analyst_opinions": merged.get("numberOfAnalystOpinions"),
            }
        except Exception as exc:
            print(f"[llm-report] yf_info failed for {symbol}: {exc}")
            enriched[symbol] = {"company_name": symbol}
    return enriched


# ---------------------------------------------------------------------------
# Deterministic per-symbol report
# ---------------------------------------------------------------------------


def _compute_financial_quality_score(info: dict[str, Any]) -> float:
    score = 0.5
    rev_g = to_float(info.get("revenue_growth"))
    earn_g = to_float(info.get("earnings_growth"))
    margins = to_float(info.get("profit_margins"))
    beta = to_float(info.get("beta"), 1.0)

    if rev_g > 0.15:
        score += 0.15
    elif rev_g > 0:
        score += 0.05
    if earn_g > 0.15:
        score += 0.15
    elif earn_g > 0:
        score += 0.05
    if margins > 0.15:
        score += 0.10
    elif margins > 0:
        score += 0.03
    if 0.5 <= beta <= 1.5:
        score += 0.05
    elif beta > 2.5:
        score -= 0.05
    return max(0.0, min(1.0, score))


def _compute_risk_score(info: dict[str, Any]) -> float:
    score = 0.3
    short_pct = to_float(info.get("short_percent_of_float"))
    beta = to_float(info.get("beta"), 1.0)
    mc = to_float(info.get("market_cap"), 0.0)

    if short_pct > 0.20:
        score += 0.25
    elif short_pct > 0.10:
        score += 0.10
    if beta > 2.0:
        score += 0.20
    elif beta > 1.5:
        score += 0.10
    if mc < 100_000_000:
        score += 0.20
    elif mc < 500_000_000:
        score += 0.10
    return max(0.0, min(1.0, score))


def _build_symbol_report(
    symbol: str, yf_info: dict[str, Any], tv_data: dict[str, Any]
) -> dict[str, Any]:
    fq = _compute_financial_quality_score(yf_info)
    risk = _compute_risk_score(yf_info)
    acc_score = to_float(tv_data.get("institutional_accumulation_score") or tv_data.get("confidence"), 0.0)
    tags: list[str] = tv_data.get("accumulation_tags") or []
    rv = to_float(tv_data.get("relative_volume"), 0.0)
    chg = to_float(tv_data.get("change_pct"), 0.0)
    rec = str(yf_info.get("analyst_recommendation") or "").lower()

    bullish: list[str] = []
    bearish: list[str] = []
    catalysts: list[str] = []
    notes: list[str] = []

    if rv >= 2.0:
        bullish.append(f"Relative volume {rv:.1f}x above 10-day average")
    if chg > 1.0:
        bullish.append(f"Positive daily momentum +{chg:.1f}%")
    if to_float(yf_info.get("revenue_growth")) > 0.10:
        bullish.append(f"Revenue growth {to_float(yf_info.get('revenue_growth')):.0%} YoY")
    if to_float(yf_info.get("earnings_growth")) > 0.10:
        bullish.append(f"Earnings growth {to_float(yf_info.get('earnings_growth')):.0%} YoY")
    if to_float(yf_info.get("profit_margins")) > 0.10:
        bullish.append(f"Healthy profit margins {to_float(yf_info.get('profit_margins')):.0%}")
    if rec in {"buy", "strong_buy"}:
        bullish.append(f"Analyst consensus: {rec}")
    if "MTF_BULLISH_ALIGNMENT" in tags:
        bullish.append("Multi-timeframe bullish alignment")
    if "MACD_BULLISH" in tags:
        bullish.append("MACD bullish crossover across timeframes")

    if chg < -2.0:
        bearish.append(f"Negative daily move {chg:.1f}%")
    if to_float(yf_info.get("short_percent_of_float")) > 0.15:
        bearish.append(f"High short interest {to_float(yf_info.get('short_percent_of_float')):.0%} of float")
    if to_float(yf_info.get("profit_margins")) < 0:
        bearish.append("Negative profit margins")

    if "INSTITUTIONAL_ACCUMULATION" in tags:
        catalysts.append("Strong institutional accumulation signal")
    if "POTENTIAL_BREAKOUT_ACCUMULATION" in tags:
        catalysts.append("Potential breakout with volume confirmation")
    if rv >= 1.5:
        catalysts.append(f"Elevated relative volume ({rv:.1f}x)")

    if acc_score >= 0.75:
        notes.append("Top-tier institutional accumulation score; monitor for breakout entry")
    elif acc_score >= 0.60:
        notes.append("Strong accumulation signal; watch volume confirmation")
    else:
        notes.append("Moderate accumulation; confirm with intraday structure")

    sentiment_score = min(1.0, acc_score * 0.6 + fq * 0.3 + max(0, chg / 10.0) * 0.1)
    catalyst_score = min(1.0, (len(catalysts) / 3) * 0.5 + acc_score * 0.5)

    return {
        "symbol": symbol,
        "company_name": yf_info.get("company_name") or symbol,
        "sector": yf_info.get("sector"),
        "industry": yf_info.get("industry"),
        "market_cap": yf_info.get("market_cap"),
        "current_price": yf_info.get("current_price") or tv_data.get("price"),
        "revenue_growth": yf_info.get("revenue_growth"),
        "earnings_growth": yf_info.get("earnings_growth"),
        "profit_margins": yf_info.get("profit_margins"),
        "gross_margins": yf_info.get("gross_margins"),
        "operating_margins": yf_info.get("operating_margins"),
        "short_percent_of_float": yf_info.get("short_percent_of_float"),
        "analyst_recommendation": yf_info.get("analyst_recommendation"),
        "institutional_accumulation_score": acc_score,
        "accumulation_tags": tags,
        "sentiment_score": round(sentiment_score, 4),
        "catalyst_score": round(catalyst_score, 4),
        "financial_quality_score": round(fq, 4),
        "risk_score": round(risk, 4),
        "llm_summary": None,
        "bullish_factors": bullish,
        "bearish_factors": bearish,
        "key_catalysts": catalysts,
        "watch_notes": notes,
    }


# ---------------------------------------------------------------------------
# OpenAI sentiment enrichment (optional)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """
You are an institutional equity analyst providing pre-market sentiment analysis.

For each symbol, provide:
- A 1-2 sentence bullish/bearish assessment
- Key near-term catalysts
- Risk factors

Return ONLY valid JSON:
{
  "analyses": [
    {
      "symbol": "TICKER",
      "llm_summary": "One-sentence assessment.",
      "sentiment": "bullish | bearish | mixed | neutral",
      "catalyst_highlights": ["..."],
      "risk_highlights": ["..."]
    }
  ]
}

Rules:
- No markdown. Valid JSON only.
- Use "symbol" not "ticker". Uppercase symbols.
- Max 2 sentences per summary.
""".strip()


def _call_llm_for_sentiment(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[llm-report] OPENAI_API_KEY missing; skipping LLM enrichment")
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        print("[llm-report] openai package not installed; skipping LLM enrichment")
        return {}

    compact: list[dict[str, Any]] = []
    for r in reports:
        compact.append({
            "symbol": r["symbol"],
            "sector": r.get("sector"),
            "change_pct": r.get("change_pct"),
            "relative_volume": r.get("relative_volume"),
            "institutional_accumulation_score": r.get("institutional_accumulation_score"),
            "accumulation_tags": r.get("accumulation_tags"),
            "revenue_growth": r.get("revenue_growth"),
            "earnings_growth": r.get("earnings_growth"),
            "profit_margins": r.get("profit_margins"),
            "analyst_recommendation": r.get("analyst_recommendation"),
            "bullish_factors": r.get("bullish_factors"),
            "bearish_factors": r.get("bearish_factors"),
        })

    prompt = (
        f"Today is {datetime.now(ET).strftime('%Y-%m-%d')}.\n"
        f"Analyze these {len(compact)} pre-selected institutional accumulation candidates:\n"
        + json.dumps(compact, indent=2, default=str)
        + "\n\nReturn only the required JSON."
    )

    try:
        client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=LLM_SENTIMENT_REPORT_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content.strip())
        result: dict[str, dict[str, Any]] = {}
        for a in data.get("analyses", []):
            sym = normalize_symbol(a.get("symbol", ""))
            if sym:
                result[sym] = a
        print(f"[llm-report] OpenAI enriched {len(result)} symbols")
        return result
    except Exception as exc:
        print(f"[llm-report] OpenAI call failed: {exc}; using deterministic reports only")
        return {}


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def build_sentiment_universe() -> dict[str, Any]:
    """
    Main entry point. Compatible with run.py import:
      from llm_sentiment_universe_selector import build_sentiment_universe, write_outputs

    Flow:
      1. Ensure TradingView universe exists.
      2. Load TV-selected symbols (authoritative).
      3. Enrich with yfinance financials.
      4. Optionally call OpenAI for sentiment.
      5. Write sentiment/financials reports.
      6. Preserve TV universe as locked universe (no reranking unless explicitly enabled).

    Returns the TV universe payload.
    """
    print(f"[llm-report] starting data_tag={DATA_TAG}")
    print("[llm-report] generating sentiment/financial reports only")
    if LLM_ALLOW_RERANK_UNIVERSE:
        print("[llm-report] WARNING: LLM_ALLOW_RERANK_UNIVERSE=1 (rerank enabled, but TV universe preserved)")
    else:
        print("[llm-report] rerank disabled")

    symbols, tv_payload = load_tradingview_universe()
    if not symbols:
        print("[llm-report] no symbols available; nothing to report")
        return tv_payload or {}

    print(f"[llm-report] loaded TradingView universe size={len(symbols)}")

    # Build TV data lookup
    tv_by_symbol: dict[str, dict[str, Any]] = {
        normalize_symbol(s.get("symbol", "")): s
        for s in tv_payload.get("selections", [])
        if s.get("symbol")
    }

    # yfinance enrichment
    print(f"[llm-report] fetching yfinance financials for {len(symbols)} symbols")
    yf_data = enrich_company_details(symbols)

    # Build per-symbol reports
    reports: list[dict[str, Any]] = []
    for symbol in symbols:
        info = yf_data.get(symbol, {"company_name": symbol})
        tv_data = tv_by_symbol.get(symbol, {})
        report = _build_symbol_report(symbol, info, tv_data)
        reports.append(report)

    # Optional LLM enrichment
    llm_enrichments: dict[str, dict[str, Any]] = {}
    if LLM_SENTIMENT_REPORT_ENABLED:
        try:
            llm_enrichments = _call_llm_for_sentiment(reports)
        except Exception as exc:
            print(f"[llm-report] LLM enrichment error: {exc}")

    # Merge LLM data into reports
    for report in reports:
        sym = report["symbol"]
        llm = llm_enrichments.get(sym, {})
        if llm:
            report["llm_summary"] = llm.get("llm_summary")
            report["sentiment"] = llm.get("sentiment", "mixed")
            llm_cats = llm.get("catalyst_highlights") or []
            llm_risks = llm.get("risk_highlights") or []
            report["key_catalysts"] = list(dict.fromkeys(report.get("key_catalysts", []) + llm_cats))
            report["bearish_factors"] = list(dict.fromkeys(report.get("bearish_factors", []) + llm_risks))
        else:
            report.setdefault("sentiment", "mixed")

    # Write sentiment report
    write_json(SENTIMENT_REPORT_JSON, {
        "created_at_utc": now_utc(),
        "data_tag": DATA_TAG,
        "source": "llm_sentiment_universe_selector",
        "universe_source": tv_payload.get("source", "tradingview_institutional_screener"),
        "symbol_count": len(reports),
        "llm_enriched": len(llm_enrichments) > 0,
        "llm_model": LLM_SENTIMENT_REPORT_MODEL if llm_enrichments else None,
        "rerank_enabled": LLM_ALLOW_RERANK_UNIVERSE,
        "reports": reports,
    })
    _write_csv(SENTIMENT_REPORT_CSV, reports)

    # Write financials report
    write_json(FINANCIALS_REPORT_JSON, {
        "created_at_utc": now_utc(),
        "data_tag": DATA_TAG,
        "source": "llm_sentiment_universe_selector",
        "symbol_count": len(symbols),
        "financials": {sym: yf_data.get(sym, {}) for sym in symbols},
    })

    # Preserve TV compatibility outputs unchanged
    # The TV screener already wrote the canonical universe files.
    # We re-confirm the locked universe txt exists (safety net only).
    if not LOCKED_UNIVERSE_PATH.exists() and symbols:
        LOCKED_UNIVERSE_PATH.write_text("\n".join(symbols) + "\n", encoding="utf-8")
        print(f"[llm-report] wrote fallback {LOCKED_UNIVERSE_PATH}")
    else:
        print(f"[llm-report] compatibility outputs preserved (source={tv_payload.get('source', 'unknown')})")

    print(f"[llm-report] complete: {len(reports)} symbol reports written")
    return tv_payload


def write_outputs(payload: dict[str, Any]) -> None:
    """
    Compatibility shim called by run.py:
      from llm_sentiment_universe_selector import build_sentiment_universe, write_outputs

    The TV screener already wrote all canonical outputs.
    This is a no-op to preserve the import interface.
    """
    pass


if __name__ == "__main__":
    build_sentiment_universe()
