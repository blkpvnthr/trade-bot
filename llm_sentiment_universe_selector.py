#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUT_DIR / "latest_llm_sentiment_universe.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / "locked_trading_universe.txt"

MODEL = os.getenv("OPENAI_SENTIMENT_UNIVERSE_MODEL", "gpt-4.1-mini")
TARGET_COUNT = 30

CANDIDATES = [
    "SPY", "QQQ", "DIA", "IWM", "XLK", "SOXX", "SMH", "ARKK", "TQQQ",
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ARM", "AMZN", "META",
    "GOOGL", "GOOG", "TSLA", "NFLX", "ORCL", "CRM", "ADBE", "INTC",
    "MU", "QCOM", "PLTR", "SOUN", "AI", "BBAI", "PATH", "SNOW",
    "DDOG", "NET", "CRWD", "PANW", "MDB", "COIN", "MARA", "RIOT",
    "CLSK", "HUT", "BTBT", "IREN", "CIFR", "WULF", "BITF", "HIVE",
    "IONQ", "RGTI", "QBTS", "QUBT", "ARQQ", "QMCO", "LAES", "OKLO",
    "SMR", "CEG", "VST", "GEV", "ETN", "PWR", "APLD", "ALAB",
    "LUNR", "RKLB", "ASTS", "JOBY", "ACHR",
]

FALLBACK = [
    "NVDA", "PLTR", "AMD", "AVGO", "ARM", "SMCI", "TSLA", "META",
    "AMZN", "GOOGL", "MSFT", "COIN", "MARA", "RIOT", "CLSK", "IREN",
    "IONQ", "RGTI", "QBTS", "QUBT", "SOUN", "AI", "BBAI", "OKLO",
    "SMR", "CEG", "VST", "SOXX", "QQQ", "SPY",
]

SYSTEM_PROMPT = """
You are a premarket institutional sentiment and catalyst engine.

Select EXACTLY 30 liquid US stocks or ETFs that are most relevant for today's trading session.

Focus on:
- premarket tone
- overnight news
- earnings reactions
- analyst upgrades/downgrades
- sector momentum
- macro-sensitive names
- AI, semiconductors, quantum, crypto-beta, space, energy infrastructure, and high-growth leadership
- liquidity and tradability
- likely intraday continuation potential

Return ONLY valid JSON.

Schema:
{
  "market_tone": "bullish | bearish | mixed | risk_off | risk_on",
  "summary": "One concise sentence describing what kind of trading day this may be.",
  "selections": [
    {
      "symbol": "NVDA",
      "reason": "One concise sentence explaining why this symbol was selected."
    }
  ]
}

Rules:
- EXACTLY 30 selections.
- Use "symbol", not "ticker".
- Symbols must be uppercase.
- No duplicate symbols.
- Reasons must be concise.
- No markdown.
- No extra commentary.
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value).upper().strip().replace(".", "-")


def is_valid_symbol(symbol: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", symbol))


def clean_reason(reason: Any) -> str:
    text = str(reason or "").replace("\n", " ").strip()
    return text[:300] or "Selected by the premarket LLM sentiment universe engine."


def build_prompt() -> str:
    return f"""
Candidate universe:
{",".join(CANDIDATES)}

Return the 30 best premarket symbols with reasons using the required JSON schema.
""".strip()


def call_llm() -> dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.15,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt()},
        ],
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_selections = payload.get("selections", [])

    selections: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in raw_selections:
        symbol = normalize_symbol(row.get("symbol") or row.get("ticker") or "")

        if not is_valid_symbol(symbol):
            continue

        if symbol in seen:
            continue

        seen.add(symbol)
        selections.append(
            {
                "symbol": symbol,
                "reason": clean_reason(row.get("reason")),
            }
        )

        if len(selections) >= TARGET_COUNT:
            break

    for symbol in FALLBACK:
        symbol = normalize_symbol(symbol)

        if len(selections) >= TARGET_COUNT:
            break

        if symbol not in seen and is_valid_symbol(symbol):
            seen.add(symbol)
            selections.append(
                {
                    "symbol": symbol,
                    "reason": "Fallback high-liquidity momentum candidate added because the LLM returned fewer than 30 valid symbols.",
                }
            )

    return {
        "created_at_utc": now_utc(),
        "model": MODEL,
        "target_count": TARGET_COUNT,
        "market_tone": str(payload.get("market_tone", "mixed")).strip() or "mixed",
        "summary": str(payload.get("summary", "Premarket tone is mixed.")).strip(),
        "symbols": [row["symbol"] for row in selections],
        "quoted_csv": ",".join(f'"{row["symbol"]}"' for row in selections),
        "selections": selections,
    }


def write_outputs(payload: dict[str, Any]) -> None:
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Backward-compatible output for your current run.py universe loader.
    LOCKED_UNIVERSE_PATH.write_text(
        "\n".join(payload["symbols"]) + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {JSON_PATH}")
    print(f"[saved] {LOCKED_UNIVERSE_PATH}")
    print(payload["quoted_csv"])


def build_sentiment_universe() -> dict[str, Any]:
    try:
        raw = call_llm()
    except Exception as exc:
        print(f"[llm-universe] LLM failed, using fallback: {exc}")
        raw = {
            "market_tone": "mixed",
            "summary": "LLM unavailable; using fallback high-liquidity momentum universe.",
            "selections": [{"symbol": s, "reason": "Fallback high-liquidity momentum candidate."} for s in FALLBACK],
        }

    payload = normalize_payload(raw)
    write_outputs(payload)
    return payload


if __name__ == "__main__":
    build_sentiment_universe()