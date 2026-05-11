#!/usr/bin/env python3
"""
portfolio_sentiment_monitor.py

OpenAI-powered portfolio constituent sentiment monitor.

Purpose:
    - Monitor current portfolio holdings, not the trading universe.
    - Pull recent headlines for currently held symbols.
    - Use OpenAI Structured Outputs to classify sentiment/risk.
    - Save latest_position_sentiment.csv for executor/optimizer overlays.
    - Optionally apply a conservative risk overlay to rebalance orders.

Default behavior:
    python portfolio_sentiment_monitor.py

Environment:
    OPENAI_API_KEY=...
    OPENAI_SENTIMENT_MODEL=gpt-5.4-mini      # optional

Outputs:
    daily_system_outputs/latest_position_sentiment.csv
    logs/execution/position_sentiment_events.jsonl

Important design rule:
    This module does not create new entries and does not add symbols to the
    universe. It can only classify current positions and recommend HOLD,
    WATCH, REDUCE, or EXIT pressure for downstream risk controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None

try:
    from alpaca.trading.client import TradingClient
except Exception:  # pragma: no cover - standalone CSV mode can still work
    TradingClient = None


# =============================================================================
# Paths / environment
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

if not ENV_PATH.exists():
    # Allows running from /mnt/data during testing, while still working when copied
    # into the project root.
    for parent in THIS_FILE.parents:
        maybe = parent / ".env"
        if maybe.exists():
            ENV_PATH = maybe
            PROJECT_ROOT = maybe.parent
            break

load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LATEST_SENTIMENT_PATH = OUT_DIR / "latest_position_sentiment.csv"
SENTIMENT_HISTORY_PATH = OUT_DIR / "position_sentiment_history.csv"
EVENTS_JSONL = LOG_DIR / "position_sentiment_events.jsonl"

DEFAULT_POSITIONS_CSV = OUT_DIR / "current_positions.csv"
DEFAULT_REBALANCE_ORDERS = OUT_DIR / "daily_walkforward_emulated_rebalance_orders.csv"
DEFAULT_SENTIMENT_ADJUSTED_ORDERS = OUT_DIR / "sentiment_adjusted_rebalance_orders.csv"

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"


# =============================================================================
# Config / models
# =============================================================================

@dataclass(frozen=True)
class SentimentMonitorConfig:
    model: str = os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-5.4-mini")
    max_headlines_per_symbol: int = 8
    request_timeout_sec: int = 20
    min_seconds_between_openai_calls: float = 0.25
    news_user_agent: str = "Mozilla/5.0 portfolio-sentiment-monitor/1.0"

    # Risk thresholds. These should be conservative because sentiment is noisy.
    reduce_score_threshold: float = -0.55
    exit_score_threshold: float = -0.75
    min_confidence_for_action: float = 0.70
    bullish_hold_threshold: float = 0.50

    # Overlay multipliers applied only to existing rebalance orders.
    watch_multiplier: float = 0.85
    reduce_multiplier: float = 0.50
    exit_multiplier: float = 0.00
    bullish_hold_multiplier: float = 1.05
    max_bullish_multiplier: float = 1.10

    # Source quality weighting is applied after LLM output.
    min_source_quality_for_boost: float = 0.55


@dataclass
class HeadlineItem:
    symbol: str
    title: str
    source: str = "Yahoo Finance RSS"
    published: str = ""
    link: str = ""


@dataclass
class PositionSentiment:
    timestamp_utc: str
    symbol: str
    sentiment_score: float
    sentiment_confidence: float
    source_quality: float
    event_type: str
    exit_pressure: str
    hold_confidence: float
    options_boost_allowed: bool
    bullish_catalyst: bool
    bearish_catalyst: bool
    risk_flag: bool
    risk_reason: str
    summary: str
    headline_count: int
    raw_headlines: str


SENTIMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentiment_score": {
            "type": "number",
            "description": "Overall current sentiment from -1.0 bearish to +1.0 bullish.",
        },
        "sentiment_confidence": {
            "type": "number",
            "description": "Confidence in the sentiment classification from 0.0 to 1.0.",
        },
        "source_quality": {
            "type": "number",
            "description": "Quality/relevance of the available headlines from 0.0 to 1.0.",
        },
        "event_type": {
            "type": "string",
            "enum": [
                "earnings", "guidance", "analyst", "product", "macro", "sector",
                "legal_regulatory", "financing", "ma", "insider", "technical",
                "mixed", "none", "other",
            ],
        },
        "exit_pressure": {
            "type": "string",
            "enum": ["hold", "watch", "reduce", "exit"],
        },
        "hold_confidence": {
            "type": "number",
            "description": "Confidence that sentiment supports continuing to hold, 0.0 to 1.0.",
        },
        "options_boost_allowed": {
            "type": "boolean",
            "description": "True only when sentiment clearly supports long-call booster confirmation.",
        },
        "bullish_catalyst": {"type": "boolean"},
        "bearish_catalyst": {"type": "boolean"},
        "risk_flag": {"type": "boolean"},
        "risk_reason": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": [
        "sentiment_score", "sentiment_confidence", "source_quality", "event_type",
        "exit_pressure", "hold_confidence", "options_boost_allowed", "bullish_catalyst",
        "bearish_catalyst", "risk_flag", "risk_reason", "summary",
    ],
}


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return utc_now().isoformat()


def safe_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {"timestamp_utc": now_iso(), "event": event, "payload": payload or {}}
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def append_history(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


# =============================================================================
# Position loading
# =============================================================================

def get_api_credentials() -> tuple[str, str]:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Expected APCA_API_KEY_ID and "
            f"APCA_API_SECRET_KEY in {ENV_PATH} or the process environment."
        )
    return api_key, secret_key


def get_trading_client(paper: bool = True):
    if TradingClient is None:
        raise RuntimeError("alpaca-py is not installed. Install alpaca-py or use --positions-csv.")
    api_key, secret_key = get_api_credentials()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)


def get_positions_df_from_alpaca(paper: bool = True) -> pd.DataFrame:
    client = get_trading_client(paper=paper)
    try:
        raw_positions = client.get_all_positions()
    except Exception as exc:
        write_event("alpaca_positions_error", {"error": str(exc)})
        raise

    rows: list[dict[str, Any]] = []
    for p in raw_positions:
        qty = to_float(getattr(p, "qty", 0.0))
        if qty <= 0:
            continue
        rows.append({
            "symbol": safe_symbol(getattr(p, "symbol", "")),
            "qty": qty,
            "avg_entry_price": to_float(getattr(p, "avg_entry_price", 0.0)),
            "current_price": to_float(getattr(p, "current_price", 0.0)),
            "market_value": to_float(getattr(p, "market_value", 0.0)),
            "cost_basis": to_float(getattr(p, "cost_basis", 0.0)),
            "unrealized_pl": to_float(getattr(p, "unrealized_pl", 0.0)),
            "unrealized_plpc": to_float(getattr(p, "unrealized_plpc", 0.0)),
        })

    return pd.DataFrame(rows)


def load_positions(positions_csv: Path | None = None, paper: bool = True) -> pd.DataFrame:
    if positions_csv is not None and positions_csv.exists():
        df = pd.read_csv(positions_csv)
    else:
        df = get_positions_df_from_alpaca(paper=paper)

    if df.empty:
        return pd.DataFrame(columns=["symbol", "qty", "market_value", "unrealized_plpc"])

    if "symbol" not in df.columns:
        raise ValueError("positions dataframe must contain a symbol column")

    df = df.copy()
    df["symbol"] = df["symbol"].map(safe_symbol)
    df = df[df["symbol"].astype(bool)].copy()
    if "qty" in df.columns:
        df = df[pd.to_numeric(df["qty"], errors="coerce").fillna(0.0) > 0].copy()
    return df.reset_index(drop=True)


# =============================================================================
# News/headline collection
# =============================================================================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:500]


def fetch_yahoo_headlines(symbol: str, config: SentimentMonitorConfig) -> list[HeadlineItem]:
    symbol = safe_symbol(symbol)
    url = YAHOO_RSS_URL.format(symbol=quote_plus(symbol))
    headers = {"User-Agent": config.news_user_agent}

    try:
        resp = requests.get(url, headers=headers, timeout=config.request_timeout_sec)
        resp.raise_for_status()
    except Exception as exc:
        write_event("headline_fetch_error", {"symbol": symbol, "error": str(exc)})
        return []

    try:
        root = ET.fromstring(resp.text)
    except Exception as exc:
        write_event("headline_parse_error", {"symbol": symbol, "error": str(exc)})
        return []

    items: list[HeadlineItem] = []
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title", ""))
        link = clean_text(item.findtext("link", ""))
        published = clean_text(item.findtext("pubDate", ""))
        if not title:
            continue
        items.append(HeadlineItem(symbol=symbol, title=title, published=published, link=link))
        if len(items) >= config.max_headlines_per_symbol:
            break

    return items


def headlines_to_prompt(symbol: str, position: dict[str, Any], headlines: list[HeadlineItem]) -> str:
    if headlines:
        headline_lines = "\n".join(
            f"- {h.title} | published={h.published or 'unknown'} | source={h.source}"
            for h in headlines
        )
    else:
        headline_lines = "- No recent headlines were found from the configured source."

    position_json = json.dumps(position, default=str, sort_keys=True)
    return f"""
Analyze current news sentiment for this existing portfolio holding only.
Do not recommend opening new positions. Do not add any new tickers.
Your job is to decide whether sentiment supports HOLD, WATCH, REDUCE, or EXIT pressure.

Symbol: {symbol}
Current position data: {position_json}

Recent headlines:
{headline_lines}

Scoring rules:
- sentiment_score must be from -1.0 to +1.0.
- sentiment_confidence must be from 0.0 to 1.0.
- source_quality should be low if headlines are missing, stale, generic, or irrelevant.
- exit_pressure should be conservative. Use exit only for clearly severe negative catalysts.
- options_boost_allowed should be true only for clearly bullish, high-confidence, relevant catalysts.
- This is a risk-monitoring signal, not a trade-entry signal.
""".strip()


# =============================================================================
# OpenAI structured sentiment
# =============================================================================

def get_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed. Run: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env or environment.")
    return OpenAI()


def extract_response_text(response: Any) -> str:
    # Newer OpenAI SDKs expose output_text. Keep fallbacks for compatibility.
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    try:
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                val = getattr(content, "text", None)
                if val:
                    chunks.append(str(val))
        if chunks:
            return "\n".join(chunks)
    except Exception:
        pass

    return str(response)


def call_openai_sentiment(
    client: Any,
    symbol: str,
    position: dict[str, Any],
    headlines: list[HeadlineItem],
    config: SentimentMonitorConfig,
) -> dict[str, Any]:
    prompt = headlines_to_prompt(symbol, position, headlines)

    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a conservative portfolio risk sentiment classifier. "
                    "Return only valid structured data. Do not provide investment advice. "
                    "Never create entry recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "position_sentiment",
                "schema": SENTIMENT_SCHEMA,
                "strict": True,
            }
        },
    )

    raw_text = extract_response_text(response)
    try:
        data = json.loads(raw_text)
    except Exception as exc:
        write_event("openai_json_parse_error", {"symbol": symbol, "error": str(exc), "raw": raw_text[:1000]})
        raise

    return normalize_sentiment_payload(data, config=config)


def fallback_sentiment(symbol: str, headlines: list[HeadlineItem], reason: str) -> dict[str, Any]:
    return {
        "sentiment_score": 0.0,
        "sentiment_confidence": 0.0,
        "source_quality": 0.0 if not headlines else 0.25,
        "event_type": "none",
        "exit_pressure": "hold",
        "hold_confidence": 0.50,
        "options_boost_allowed": False,
        "bullish_catalyst": False,
        "bearish_catalyst": False,
        "risk_flag": True,
        "risk_reason": f"sentiment unavailable: {reason}",
        "summary": f"No reliable sentiment score generated for {symbol}.",
    }


def normalize_sentiment_payload(data: dict[str, Any], config: SentimentMonitorConfig) -> dict[str, Any]:
    out = dict(data)
    out["sentiment_score"] = clamp(to_float(out.get("sentiment_score")), -1.0, 1.0)
    out["sentiment_confidence"] = clamp(to_float(out.get("sentiment_confidence")), 0.0, 1.0)
    out["source_quality"] = clamp(to_float(out.get("source_quality")), 0.0, 1.0)
    out["hold_confidence"] = clamp(to_float(out.get("hold_confidence"), 0.5), 0.0, 1.0)
    out["summary"] = clean_text(out.get("summary", ""))[:400]
    out["risk_reason"] = clean_text(out.get("risk_reason", ""))[:300]

    score = out["sentiment_score"]
    conf = out["sentiment_confidence"]

    # Enforce conservative action thresholds regardless of what the LLM chose.
    if conf < config.min_confidence_for_action:
        if score <= config.exit_score_threshold:
            out["exit_pressure"] = "reduce"
        elif score <= config.reduce_score_threshold:
            out["exit_pressure"] = "watch"
        elif str(out.get("exit_pressure", "hold")).lower() == "exit":
            out["exit_pressure"] = "watch"
    else:
        if score <= config.exit_score_threshold:
            out["exit_pressure"] = "exit"
        elif score <= config.reduce_score_threshold:
            out["exit_pressure"] = "reduce"
        elif score < 0:
            out["exit_pressure"] = "watch"
        else:
            out["exit_pressure"] = str(out.get("exit_pressure", "hold")).lower()
            if out["exit_pressure"] not in {"hold", "watch", "reduce", "exit"}:
                out["exit_pressure"] = "hold"

    # Options confirmation must be strict and cannot survive weak source quality.
    out["options_boost_allowed"] = bool(
        out.get("options_boost_allowed")
        and score >= config.bullish_hold_threshold
        and conf >= config.min_confidence_for_action
        and out["source_quality"] >= config.min_source_quality_for_boost
        and out.get("bullish_catalyst") is True
        and out.get("bearish_catalyst") is False
        and out.get("risk_flag") is False
    )

    return out


# =============================================================================
# Main monitor
# =============================================================================

def build_position_context(row: pd.Series) -> dict[str, Any]:
    fields = [
        "symbol", "qty", "avg_entry_price", "current_price", "market_value",
        "cost_basis", "unrealized_pl", "unrealized_plpc",
    ]
    return {k: row.get(k, None) for k in fields if k in row.index}


def analyze_positions(
    positions_df: pd.DataFrame,
    config: SentimentMonitorConfig | None = None,
    use_openai: bool = True,
) -> pd.DataFrame:
    config = config or SentimentMonitorConfig()
    if positions_df.empty:
        empty = pd.DataFrame(columns=[field for field in PositionSentiment.__dataclass_fields__])
        save_df(empty, LATEST_SENTIMENT_PATH)
        return empty

    client = get_openai_client() if use_openai else None
    rows: list[dict[str, Any]] = []

    for _, pos in positions_df.iterrows():
        symbol = safe_symbol(pos.get("symbol", ""))
        if not symbol:
            continue

        headlines = fetch_yahoo_headlines(symbol, config=config)
        position_context = build_position_context(pos)

        try:
            if client is None:
                payload = fallback_sentiment(symbol, headlines, "OpenAI disabled")
            else:
                payload = call_openai_sentiment(client, symbol, position_context, headlines, config)
                time.sleep(max(0.0, config.min_seconds_between_openai_calls))
        except Exception as exc:
            write_event("sentiment_score_error", {"symbol": symbol, "error": str(exc)})
            payload = fallback_sentiment(symbol, headlines, str(exc))

        raw_headlines = " | ".join(h.title for h in headlines)[:2000]
        row = PositionSentiment(
            timestamp_utc=now_iso(),
            symbol=symbol,
            sentiment_score=float(payload["sentiment_score"]),
            sentiment_confidence=float(payload["sentiment_confidence"]),
            source_quality=float(payload["source_quality"]),
            event_type=str(payload["event_type"]),
            exit_pressure=str(payload["exit_pressure"]),
            hold_confidence=float(payload["hold_confidence"]),
            options_boost_allowed=bool(payload["options_boost_allowed"]),
            bullish_catalyst=bool(payload["bullish_catalyst"]),
            bearish_catalyst=bool(payload["bearish_catalyst"]),
            risk_flag=bool(payload["risk_flag"]),
            risk_reason=str(payload["risk_reason"]),
            summary=str(payload["summary"]),
            headline_count=len(headlines),
            raw_headlines=raw_headlines,
        )
        rows.append(asdict(row))
        print(
            f"[sentiment] {symbol} score={row.sentiment_score:+.2f} "
            f"conf={row.sentiment_confidence:.2f} pressure={row.exit_pressure} "
            f"options_boost={row.options_boost_allowed}"
        )

    out = pd.DataFrame(rows)
    save_df(out, LATEST_SENTIMENT_PATH)
    append_history(out, SENTIMENT_HISTORY_PATH)
    write_event("sentiment_cycle_complete", {"positions": len(out)})
    return out


def run_portfolio_sentiment_monitor(
    positions_df: pd.DataFrame | None = None,
    positions_csv: Path | None = None,
    paper: bool = True,
    config: SentimentMonitorConfig | None = None,
    use_openai: bool = True,
) -> pd.DataFrame:
    if positions_df is None:
        positions_df = load_positions(positions_csv=positions_csv, paper=paper)
    return analyze_positions(positions_df=positions_df, config=config, use_openai=use_openai)


# =============================================================================
# Risk overlay helpers for executor / optimizer integration
# =============================================================================

def _overlay_multiplier(row: pd.Series, config: SentimentMonitorConfig) -> float:
    pressure = str(row.get("exit_pressure", "hold")).lower()
    score = to_float(row.get("sentiment_score", 0.0))
    conf = to_float(row.get("sentiment_confidence", 0.0))
    source_quality = to_float(row.get("source_quality", 0.0))

    if pressure == "exit":
        return config.exit_multiplier
    if pressure == "reduce":
        return config.reduce_multiplier
    if pressure == "watch":
        return config.watch_multiplier

    if (
        score >= config.bullish_hold_threshold
        and conf >= config.min_confidence_for_action
        and source_quality >= config.min_source_quality_for_boost
    ):
        return min(config.max_bullish_multiplier, config.bullish_hold_multiplier)

    return 1.0


def apply_sentiment_risk_overlay(
    rebalance_orders: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    config: SentimentMonitorConfig | None = None,
) -> pd.DataFrame:
    """Adjust existing orders using portfolio sentiment.

    This never creates new symbols or new buy orders. It only scales existing
    target_weight/target_dollars/recommended_shares when those columns exist.
    If exit_pressure == exit, downstream execution should interpret the target
    as zero for that symbol.
    """
    config = config or SentimentMonitorConfig()

    if rebalance_orders is None or rebalance_orders.empty or sentiment_df is None or sentiment_df.empty:
        return rebalance_orders.copy() if rebalance_orders is not None else pd.DataFrame()

    orders = rebalance_orders.copy()
    if "symbol" not in orders.columns:
        return orders

    sentiment = sentiment_df.copy()
    sentiment["symbol"] = sentiment["symbol"].map(safe_symbol)
    sentiment = sentiment.drop_duplicates("symbol", keep="last")
    sentiment["sentiment_overlay_mult"] = sentiment.apply(lambda r: _overlay_multiplier(r, config), axis=1)

    cols = [
        "symbol", "sentiment_score", "sentiment_confidence", "source_quality",
        "exit_pressure", "options_boost_allowed", "sentiment_overlay_mult", "summary",
    ]
    cols = [c for c in cols if c in sentiment.columns]

    orders["symbol"] = orders["symbol"].map(safe_symbol)
    orders = orders.merge(sentiment[cols], on="symbol", how="left")
    orders["sentiment_overlay_mult"] = pd.to_numeric(
        orders.get("sentiment_overlay_mult", 1.0), errors="coerce"
    ).fillna(1.0)

    for col in ["target_weight", "target_dollars", "recommended_shares"]:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce").fillna(0.0) * orders["sentiment_overlay_mult"]

    if "sentiment_exit_flag" not in orders.columns:
        orders["sentiment_exit_flag"] = orders.get("exit_pressure", "").astype(str).str.lower().eq("exit")

    if "sizing_reason" in orders.columns:
        orders["sizing_reason"] = orders["sizing_reason"].fillna("").astype(str) + orders.apply(
            lambda r: f" | sentiment={r.get('exit_pressure', 'none')} mult={to_float(r.get('sentiment_overlay_mult', 1.0)):.2f}",
            axis=1,
        )

    return orders


def apply_overlay_to_file(
    rebalance_orders_path: Path = DEFAULT_REBALANCE_ORDERS,
    sentiment_path: Path = LATEST_SENTIMENT_PATH,
    output_path: Path = DEFAULT_SENTIMENT_ADJUSTED_ORDERS,
    config: SentimentMonitorConfig | None = None,
) -> pd.DataFrame:
    if not rebalance_orders_path.exists():
        raise FileNotFoundError(f"missing rebalance orders: {rebalance_orders_path}")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"missing sentiment file: {sentiment_path}")

    orders = pd.read_csv(rebalance_orders_path)
    sentiment = pd.read_csv(sentiment_path)
    adjusted = apply_sentiment_risk_overlay(orders, sentiment, config=config)
    save_df(adjusted, output_path)
    return adjusted


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI portfolio constituent sentiment monitor.")
    parser.add_argument("--positions-csv", default="", help="Optional positions CSV. If omitted, pulls current Alpaca positions.")
    parser.add_argument("--paper", action="store_true", default=True, help="Use Alpaca paper account. Default true.")
    parser.add_argument("--live", action="store_true", help="Use live Alpaca account instead of paper.")
    parser.add_argument("--model", default=os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--max-headlines", type=int, default=8)
    parser.add_argument("--no-openai", action="store_true", help="Fetch headlines but write neutral fallback sentiment.")
    parser.add_argument("--apply-overlay", action="store_true", help="Also adjust rebalance order CSV using latest sentiment.")
    parser.add_argument("--rebalance-orders", default=str(DEFAULT_REBALANCE_ORDERS))
    parser.add_argument("--adjusted-orders", default=str(DEFAULT_SENTIMENT_ADJUSTED_ORDERS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SentimentMonitorConfig(model=args.model, max_headlines_per_symbol=args.max_headlines)
    positions_csv = Path(args.positions_csv) if args.positions_csv else None

    print("===== PORTFOLIO SENTIMENT MONITOR =====")
    print(f"[env] {ENV_PATH if ENV_PATH.exists() else 'process environment only'}")
    print(f"[model] {config.model}")
    print(f"[mode] {'live' if args.live else 'paper'}")

    sentiment = run_portfolio_sentiment_monitor(
        positions_csv=positions_csv,
        paper=not args.live,
        config=config,
        use_openai=not args.no_openai,
    )

    if args.apply_overlay:
        apply_overlay_to_file(
            rebalance_orders_path=Path(args.rebalance_orders),
            sentiment_path=LATEST_SENTIMENT_PATH,
            output_path=Path(args.adjusted_orders),
            config=config,
        )

    if sentiment.empty:
        print("[sentiment] no open long positions found")


if __name__ == "__main__":
    main()
