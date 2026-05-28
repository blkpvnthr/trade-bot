#!/usr/bin/env python3
"""
tradingview_institutional_screener.py

Institutional accumulation universe discovery using TradingView's screener API.

This is the AUTHORITATIVE universe generator for the trading system.
It runs BEFORE llm_sentiment_universe_selector.py, which is now a
reporting/enrichment layer only.

Outputs:
  system-data/YYYYMMDD/YYYYMMDD_tradingview_institutional_accumulation.csv
  system-data/YYYYMMDD/YYYYMMDD_tradingview_institutional_accumulation.json
  system-data/YYYYMMDD/YYYYMMDD_tradingview_institutional_accumulation_diagnostics.json

Compatibility outputs (same filenames used by run.py/premarket_trade_planner.py):
  system-data/YYYYMMDD/YYYYMMDD_latest_llm_sentiment_universe.json
  system-data/YYYYMMDD/YYYYMMDD_latest_llm_candidate_discovery.json
  YYYYMMDD_locked_trading_universe.txt

Standalone usage:
  python tradingview_institutional_screener.py

Required env (optional - all have defaults):
  TV_INST_SCREENER_ENABLED=1
  TV_INST_LIMIT=250
  TV_INST_TARGET_UNIVERSE_SIZE=30
  TV_INST_MAX_SYMBOLS=30
  TV_INST_MIN_PRICE=2
  TV_INST_MAX_PRICE=300
  TV_INST_MIN_MARKET_CAP=50000000
  TV_INST_MIN_DOLLAR_VOLUME=10000000
  TV_INST_MIN_REL_VOLUME=1.5
  TV_INST_EXCLUDE_OTC=1
  TV_INST_INCLUDE_ETFS=0
  TV_INST_MIN_SCORE=0.60
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from config.env_loader import load_instance_env

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
load_instance_env()

ET = ZoneInfo("America/New_York")

try:
    from market_context_builder import TODAY_TAG as _MCB_TODAY
except Exception:
    _MCB_TODAY = datetime.now(ET).strftime("%Y%m%d")

DATA_TAG: str = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", _MCB_TODAY))

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config via env vars
# ---------------------------------------------------------------------------

TV_INST_SCREENER_ENABLED: bool = os.getenv("TV_INST_SCREENER_ENABLED", "1").lower() in {"1", "true", "yes"}
TV_INST_LIMIT: int = int(os.getenv("TV_INST_LIMIT", "250"))
TV_INST_TARGET_UNIVERSE_SIZE: int = int(os.getenv("TV_INST_TARGET_UNIVERSE_SIZE", "30"))
TV_INST_MAX_SYMBOLS: int = int(os.getenv("TV_INST_MAX_SYMBOLS", "30"))
TV_INST_MIN_PRICE: float = float(os.getenv("TV_INST_MIN_PRICE", "2"))
TV_INST_MAX_PRICE: float = float(os.getenv("TV_INST_MAX_PRICE", "300"))
TV_INST_MIN_MARKET_CAP: float = float(os.getenv("TV_INST_MIN_MARKET_CAP", "50000000"))
TV_INST_MIN_DOLLAR_VOLUME: float = float(os.getenv("TV_INST_MIN_DOLLAR_VOLUME", "10000000"))
TV_INST_MIN_REL_VOLUME: float = float(os.getenv("TV_INST_MIN_REL_VOLUME", "1.5"))
TV_INST_EXCLUDE_OTC: bool = os.getenv("TV_INST_EXCLUDE_OTC", "1").lower() in {"1", "true", "yes"}
TV_INST_INCLUDE_ETFS: bool = os.getenv("TV_INST_INCLUDE_ETFS", "0").lower() in {"1", "true", "yes"}
TV_INST_MIN_SCORE: float = float(os.getenv("TV_INST_MIN_SCORE", "0.60"))

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

ACC_CSV_PATH = OUT_DIR / f"{DATA_TAG}_tradingview_institutional_accumulation.csv"
ACC_JSON_PATH = OUT_DIR / f"{DATA_TAG}_tradingview_institutional_accumulation.json"
DIAG_JSON_PATH = OUT_DIR / f"{DATA_TAG}_tradingview_institutional_accumulation_diagnostics.json"

COMPAT_UNIVERSE_JSON = OUT_DIR / f"{DATA_TAG}_latest_llm_sentiment_universe.json"
COMPAT_DISCOVERY_JSON = OUT_DIR / f"{DATA_TAG}_latest_llm_candidate_discovery.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"

# ---------------------------------------------------------------------------
# TradingView field lists
# ---------------------------------------------------------------------------

TV_FIELDS_FULL: list[str] = [
    "name",
    "description",
    "exchange",
    "type",
    "subtype",
    "close",
    "change",
    "volume",
    "relative_volume_10d_calc",
    "market_cap_basic",
    "close|1",
    "close|5",
    "close|15",
    "close|60",
    "volume|1",
    "volume|5",
    "volume|15",
    "MACD.macd|1",
    "MACD.signal|1",
    "MACD.macd|5",
    "MACD.signal|5",
    "MACD.macd|15",
    "MACD.signal|15",
    "Recommend.All",
    "Recommend.All|5",
    "Recommend.All|15",
    "Recommend.All|60",
    "RSI",
    "RSI|5",
    "RSI|15",
]

TV_FIELDS_MINIMAL: list[str] = [
    "name",
    "description",
    "exchange",
    "type",
    "close",
    "change",
    "volume",
    "relative_volume_10d_calc",
    "market_cap_basic",
    "Recommend.All",
    "Recommend.All|5",
    "Recommend.All|15",
    "RSI",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize_symbol(val: Any) -> str:
    s = str(val or "").upper().strip().replace(".", "-")
    # TradingView returns "EXCHANGE:SYMBOL" — strip the exchange prefix.
    if ":" in s:
        s = s.split(":")[-1]
    return s


def _is_valid_symbol(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", s))


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"[tv-inst] wrote {path}")


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Sanitize for CSV
            clean = {}
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean[k] = None
                elif isinstance(v, list):
                    clean[k] = "|".join(str(x) for x in v)
                else:
                    clean[k] = v
            writer.writerow(clean)
    print(f"[tv-inst] wrote {path}")


# ---------------------------------------------------------------------------
# Percentile-rank scorer
# ---------------------------------------------------------------------------


def _prank(values: list[float]) -> list[float]:
    """Return 0-1 percentile rank for each value. Ties share their average rank."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    result = [0.0] * n
    for rank, (orig_idx, _) in enumerate(indexed):
        result[orig_idx] = rank / (n - 1)
    return result


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


def _score_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute institutional_accumulation_score and component scores for each row.

    institutional_accumulation_score =
        relative_volume_score     * 0.20
      + dollar_volume_score       * 0.15
      + volume_acceleration_score * 0.15
      + price_strength_score      * 0.15
      + macd_alignment_score      * 0.10
      + mtf_alignment_score       * 0.10
      + recommendation_score      * 0.10
      + liquidity_quality_score   * 0.05
    """
    n = len(rows)
    if n == 0:
        return rows

    # --- Raw feature extraction ---
    rel_vols = [_to_float(r.get("relative_volume_10d_calc"), 0.0) for r in rows]

    dollar_vols: list[float] = [
        _to_float(r.get("volume"), 0.0) * _to_float(r.get("close"), 0.0) for r in rows
    ]

    vol_accels: list[float] = []
    for r in rows:
        v_now = _to_float(r.get("volume"), 0.0)
        v5 = _to_float(r.get("volume|5"), 0.0)
        v15 = _to_float(r.get("volume|15"), 0.0)
        avg_prev = (v5 + v15) / 2.0 if (v5 + v15) > 0 else 0.0
        vol_accels.append(v_now / avg_prev if avg_prev > 0 else 1.0)

    price_strengths: list[float] = []
    for r in rows:
        chg = _to_float(r.get("change"), 0.0)
        close = _to_float(r.get("close"), 0.0)
        c5 = _to_float(r.get("close|5"), 0.0)
        c15 = _to_float(r.get("close|15"), 0.0)
        c60 = _to_float(r.get("close|60"), 0.0)
        ps = max(chg, 0.0) / 10.0
        if close > 0:
            if c5 > 0 and close > c5:
                ps += 0.20
            if c15 > 0 and close > c15:
                ps += 0.20
            if c60 > 0 and close > c60:
                ps += 0.20
        price_strengths.append(min(ps, 1.0))

    macd_alignments: list[float] = []
    for r in rows:
        bull = 0.0
        total = 0.0
        for tf in ("1", "5", "15"):
            m = r.get(f"MACD.macd|{tf}")
            s = r.get(f"MACD.signal|{tf}")
            if m is not None and s is not None:
                total += 1.0
                if _to_float(m) > _to_float(s):
                    bull += 1.0
        macd_alignments.append(bull / total if total > 0 else 0.5)

    mtf_alignments: list[float] = []
    for r in rows:
        bull = 0.0
        total = 0.0
        for tf_key in ("Recommend.All", "Recommend.All|5", "Recommend.All|15", "Recommend.All|60"):
            val = r.get(tf_key)
            if val is not None:
                total += 1.0
                if _to_float(val, -1.0) > 0:
                    bull += 1.0
        mtf_alignments.append(bull / max(total, 1.0))

    rec_raw: list[float] = [max(_to_float(r.get("Recommend.All"), 0.0), 0.0) for r in rows]

    liq_scores: list[float] = []
    for r in rows:
        mc = _to_float(r.get("market_cap_basic"), 0.0)
        if mc >= 10_000_000_000:
            liq_scores.append(1.0)
        elif mc >= 1_000_000_000:
            liq_scores.append(0.80)
        elif mc >= 500_000_000:
            liq_scores.append(0.65)
        elif mc >= 100_000_000:
            liq_scores.append(0.45)
        elif mc >= 50_000_000:
            liq_scores.append(0.25)
        else:
            liq_scores.append(0.10)

    # --- Normalize continuous columns via percentile rank ---
    rv_scores = _prank(rel_vols)
    dv_scores = _prank(dollar_vols)
    va_scores = _prank(vol_accels)
    rec_scores = _prank(rec_raw)

    # --- Compose final score ---
    for i, r in enumerate(rows):
        rv = _clip01(rv_scores[i])
        dv = _clip01(dv_scores[i])
        va = _clip01(va_scores[i])
        ps = _clip01(price_strengths[i])
        macd = _clip01(macd_alignments[i])
        mtf = _clip01(mtf_alignments[i])
        rec = _clip01(rec_scores[i])
        liq = _clip01(liq_scores[i])

        composite = (
            rv   * 0.20
            + dv * 0.15
            + va * 0.15
            + ps * 0.15
            + macd * 0.10
            + mtf * 0.10
            + rec * 0.10
            + liq * 0.05
        )

        r["_rv_score"] = round(rv, 4)
        r["_dv_score"] = round(dv, 4)
        r["_va_score"] = round(va, 4)
        r["_ps_score"] = round(ps, 4)
        r["_macd_score"] = round(macd, 4)
        r["_mtf_score"] = round(mtf, 4)
        r["_rec_score"] = round(rec, 4)
        r["_liq_score"] = round(liq, 4)
        r["_dollar_volume"] = round(dollar_vols[i], 0)
        r["_vol_accel"] = round(vol_accels[i], 4)
        r["institutional_accumulation_score"] = round(_clip01(composite), 4)

    return rows


def _assign_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    rv = _to_float(row.get("relative_volume_10d_calc"), 0.0)
    dv = _to_float(row.get("_dollar_volume"), 0.0)
    va = _to_float(row.get("_vol_accel"), 0.0)
    macd = _to_float(row.get("_macd_score"), 0.0)
    mtf = _to_float(row.get("_mtf_score"), 0.0)
    ps = _to_float(row.get("_ps_score"), 0.0)
    score = _to_float(row.get("institutional_accumulation_score"), 0.0)

    if rv >= TV_INST_MIN_REL_VOLUME * 1.5:
        tags.append("AGGRESSIVE_RVOL")
    if dv >= TV_INST_MIN_DOLLAR_VOLUME * 2:
        tags.append("HIGH_DOLLAR_VOLUME")
    if va >= 1.5:
        tags.append("VOLUME_ACCELERATION")
    if mtf >= 0.75:
        tags.append("MTF_BULLISH_ALIGNMENT")
    if macd >= 0.67:
        tags.append("MACD_BULLISH")
    if ps >= 0.70:
        tags.append("RELATIVE_STRENGTH")
    if score >= 0.75:
        tags.append("INSTITUTIONAL_ACCUMULATION")
    if score >= 0.60 and rv >= TV_INST_MIN_REL_VOLUME:
        tags.append("POTENTIAL_BREAKOUT_ACCUMULATION")

    return tags


# ---------------------------------------------------------------------------
# TradingView query
# ---------------------------------------------------------------------------


def _run_tv_query_full() -> pd.DataFrame | None:
    try:
        from tradingview_screener import Query, col as tv_col
    except ImportError:
        print("[tv-inst] tradingview-screener not installed. Run: pip install tradingview-screener")
        return None

    try:
        q = (
            Query()
            .select(*TV_FIELDS_FULL)
            .where(
                tv_col("close") >= TV_INST_MIN_PRICE,
                tv_col("close") <= TV_INST_MAX_PRICE,
                tv_col("market_cap_basic") >= TV_INST_MIN_MARKET_CAP,
                tv_col("relative_volume_10d_calc") >= TV_INST_MIN_REL_VOLUME,
                tv_col("MACD.macd|5") >= tv_col("MACD.signal|5"),
                tv_col("Recommend.All|5") > 0,
                tv_col("Recommend.All|15") > 0,
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(TV_INST_LIMIT)
        )
        count, df = q.get_scanner_data()
        print(f"[tv-inst] full query: {count} candidates matched ({len(df)} fetched)")
        return df
    except Exception as exc:
        print(f"[tv-inst] full query failed: {exc}")
        return None


def _run_tv_query_minimal() -> pd.DataFrame | None:
    try:
        from tradingview_screener import Query, col as tv_col
    except ImportError:
        return None

    try:
        q = (
            Query()
            .select(*TV_FIELDS_MINIMAL)
            .where(
                tv_col("close") >= TV_INST_MIN_PRICE,
                tv_col("close") <= TV_INST_MAX_PRICE,
                tv_col("market_cap_basic") >= TV_INST_MIN_MARKET_CAP,
                tv_col("relative_volume_10d_calc") >= TV_INST_MIN_REL_VOLUME,
                tv_col("Recommend.All|5") > 0,
                tv_col("Recommend.All|15") > 0,
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(TV_INST_LIMIT)
        )
        count, df = q.get_scanner_data()
        print(f"[tv-inst] minimal query: {count} candidates matched ({len(df)} fetched)")
        return df
    except Exception as exc:
        print(f"[tv-inst] minimal query failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Fallback universe loader
# ---------------------------------------------------------------------------


def _load_fallback_symbols() -> list[str]:
    yesterday = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
    for candidate in [
        PROJECT_ROOT / f"{yesterday}_locked_trading_universe.txt",
        PROJECT_ROOT / "locked_trading_universe.txt",
    ]:
        if candidate.exists():
            syms = [s.strip() for s in candidate.read_text().splitlines() if s.strip()]
            syms = [_normalize_symbol(s) for s in syms if _is_valid_symbol(_normalize_symbol(s))]
            if syms:
                print(f"[tv-inst] fallback: {len(syms)} symbols from {candidate}")
                return syms

    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        csv_files = sorted(data_dir.glob("bullish_*.csv"))
        if csv_files:
            try:
                df = pd.read_csv(csv_files[-1])
                syms = [
                    _normalize_symbol(s)
                    for s in df.get("Symbol", pd.Series()).dropna().tolist()
                ]
                syms = [s for s in syms if _is_valid_symbol(s)]
                if syms:
                    print(f"[tv-inst] fallback: {len(syms)} symbols from {csv_files[-1]}")
                    return syms[:TV_INST_TARGET_UNIVERSE_SIZE]
            except Exception:
                pass
    return []


# ---------------------------------------------------------------------------
# DataFrame parsing and post-filtering
# ---------------------------------------------------------------------------


def _parse_tv_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        # tradingview_screener exposes the ticker in a 'ticker' column (added by library)
        # or falls back to 'name'
        symbol = _normalize_symbol(
            row.get("ticker") or row.get("name") or ""
        )
        if not _is_valid_symbol(symbol):
            continue

        d: dict[str, Any] = {"symbol": symbol}
        for col_name in df.columns:
            val = row.get(col_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                val = None
            d[col_name] = val

        d.setdefault("close", 0.0)
        d.setdefault("change", 0.0)
        d.setdefault("volume", 0)
        d.setdefault("relative_volume_10d_calc", 0.0)
        d.setdefault("market_cap_basic", 0.0)
        rows.append(d)
    return rows


def _apply_post_filters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for r in rows:
        symbol = r.get("symbol", "")
        close = _to_float(r.get("close"), 0.0)
        volume = _to_float(r.get("volume"), 0.0)
        dollar_vol = close * volume
        exchange = str(r.get("exchange") or "").upper()
        asset_type = str(r.get("type") or "").lower()

        if TV_INST_MIN_DOLLAR_VOLUME > 0 and dollar_vol < TV_INST_MIN_DOLLAR_VOLUME:
            continue
        if TV_INST_EXCLUDE_OTC and exchange in {"OTC", "OTCMKTS", "OTCBB", "PINK"}:
            continue
        if not TV_INST_INCLUDE_ETFS and asset_type in {"etf", "fund", "dr"}:
            continue
        if close <= 0 or not _is_valid_symbol(symbol):
            continue

        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# Core screener logic
# ---------------------------------------------------------------------------


def fetch_institutional_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diag: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "data_tag": DATA_TAG,
        "query_strategy": "full",
        "raw_count": 0,
        "post_filter_count": 0,
        "selected_count": 0,
        "fallback_used": False,
        "errors": [],
    }

    df = _run_tv_query_full()

    if df is None or df.empty:
        print("[tv-inst] full field list failed; retrying with minimal fields")
        diag["query_strategy"] = "minimal_retry"
        df = _run_tv_query_minimal()

    rows: list[dict[str, Any]] = []
    if df is not None and not df.empty:
        rows = _parse_tv_dataframe(df)
        diag["raw_count"] = len(rows)
        rows = _apply_post_filters(rows)
        diag["post_filter_count"] = len(rows)

    if len(rows) < 5:
        print(f"[tv-inst] only {len(rows)} candidates after filters; loading fallback universe")
        diag["fallback_used"] = True
        fallback_syms = _load_fallback_symbols()
        existing_syms = {r.get("symbol") for r in rows}
        for sym in fallback_syms:
            if sym not in existing_syms:
                rows.append({
                    "symbol": sym,
                    "close": 0.0,
                    "change": 0.0,
                    "volume": 0,
                    "relative_volume_10d_calc": 0.0,
                    "market_cap_basic": 0.0,
                    "source": "fallback_locked_universe",
                })

    return rows, diag


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def _build_compat_payload(
    selected: list[dict[str, Any]], diag: dict[str, Any]
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    for rank, r in enumerate(selected, 1):
        symbol = r.get("symbol", "")
        score = _to_float(r.get("institutional_accumulation_score"), 0.0)
        tags = r.get("accumulation_tags") or []
        rv = _to_float(r.get("relative_volume_10d_calc"), 0.0)
        close = _to_float(r.get("close"), 0.0)
        vol = _to_float(r.get("volume"), 0.0)
        chg = _to_float(r.get("change"), 0.0)

        if "INSTITUTIONAL_ACCUMULATION" in tags or "POTENTIAL_BREAKOUT_ACCUMULATION" in tags:
            catalyst = "liquidity_momentum"
        else:
            catalyst = "technical_gainer"

        parts = [f"{symbol} selected via TradingView institutional screener"]
        if rv > 0:
            parts.append(f"relative volume {rv:.2f}x")
        if close > 0 and vol > 0:
            parts.append(f"dollar volume ${close * vol / 1_000_000:.1f}M")
        if chg != 0:
            parts.append(f"change {chg:+.2f}%")
        if tags:
            parts.append(f"tags: {', '.join(tags)}")

        selections.append({
            "rank": rank,
            "symbol": symbol,
            "catalyst_type": catalyst,
            "reason": ". ".join(parts) + ".",
            "confidence": round(score, 4),
            "price": close,
            "change_pct": chg,
            "volume": int(vol),
            "relative_volume": rv,
            "market_cap": _to_float(r.get("market_cap_basic"), 0.0),
            "institutional_accumulation_score": score,
            "accumulation_tags": tags,
            "source": "tradingview_institutional_screener",
        })

    symbols = [s["symbol"] for s in selections]
    return {
        "created_at_utc": _now_utc(),
        "model": "tradingview_institutional_screener",
        "source": "tradingview_institutional_screener",
        "target_count": TV_INST_TARGET_UNIVERSE_SIZE,
        "market_tone": "mixed",
        "summary": (
            f"TradingView institutional screener selected {len(symbols)} symbols "
            f"with strong accumulation signals from {diag.get('raw_count', 0)} scanned candidates."
        ),
        "symbols": symbols,
        "quoted_csv": ",".join(f'"{s}"' for s in symbols),
        "selections": selections,
        "discovery": {
            "source": "tradingview_institutional_screener",
            "candidate_count": diag.get("raw_count", 0),
            "post_filter_count": diag.get("post_filter_count", 0),
            "selected_count": len(symbols),
            "fallback_used": diag.get("fallback_used", False),
        },
    }


def _write_all_outputs(
    payload: dict[str, Any],
    selected: list[dict[str, Any]],
    diag: dict[str, Any],
) -> None:
    # Primary accumulation CSV
    _write_csv_rows(ACC_CSV_PATH, selected)

    # Primary accumulation JSON
    _write_json(ACC_JSON_PATH, {
        "created_at_utc": _now_utc(),
        "source": "tradingview_institutional_screener",
        "data_tag": DATA_TAG,
        "symbol_count": len(selected),
        "diagnostics": diag,
        "candidates": selected,
    })

    # Diagnostics
    _write_json(DIAG_JSON_PATH, diag)

    # Compatibility: llm_sentiment_universe.json
    _write_json(COMPAT_UNIVERSE_JSON, payload)

    # Compatibility: llm_candidate_discovery.json
    _write_json(COMPAT_DISCOVERY_JSON, {
        "created_at_utc": _now_utc(),
        "source": "tradingview_institutional_screener",
        "data_tag": DATA_TAG,
        "candidate_count": len(selected),
        "selected_count": diag.get("selected_count", 0),
        "filters": {
            "min_price": TV_INST_MIN_PRICE,
            "max_price": TV_INST_MAX_PRICE,
            "min_market_cap": TV_INST_MIN_MARKET_CAP,
            "min_dollar_volume": TV_INST_MIN_DOLLAR_VOLUME,
            "min_rel_volume": TV_INST_MIN_REL_VOLUME,
            "min_score": TV_INST_MIN_SCORE,
            "target_universe_size": TV_INST_TARGET_UNIVERSE_SIZE,
            "exclude_otc": TV_INST_EXCLUDE_OTC,
            "include_etfs": TV_INST_INCLUDE_ETFS,
        },
        "candidates": selected,
    })

    # Compatibility: locked universe text file
    symbols = payload.get("symbols", [])
    LOCKED_UNIVERSE_PATH.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(f"[tv-inst] locked universe written → {LOCKED_UNIVERSE_PATH}")
    if symbols:
        print(f"[tv-inst] universe ({len(symbols)}): {', '.join(symbols[:10])}" + (" ..." if len(symbols) > 10 else ""))


def _write_empty_outputs(diag: dict[str, Any]) -> None:
    empty = {
        "created_at_utc": _now_utc(),
        "model": "tradingview_institutional_screener",
        "source": "tradingview_institutional_screener",
        "target_count": TV_INST_TARGET_UNIVERSE_SIZE,
        "market_tone": "mixed",
        "summary": "TradingView screener returned no qualifying candidates.",
        "symbols": [],
        "quoted_csv": "",
        "selections": [],
        "discovery": {"source": "tradingview_institutional_screener", "candidate_count": 0},
    }
    _write_json(COMPAT_UNIVERSE_JSON, empty)
    _write_json(DIAG_JSON_PATH, diag)
    LOCKED_UNIVERSE_PATH.write_text("", encoding="utf-8")
    print("[tv-inst] wrote empty fallback outputs")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_tradingview_institutional_universe() -> dict[str, Any]:
    """
    Discover, score, select, and write all outputs.

    Returns the universe payload (compatible with llm_sentiment_universe.json schema).
    Never raises — falls back gracefully.
    """
    print(f"[tv-inst] discovering institutional accumulation universe data_tag={DATA_TAG}")

    if not TV_INST_SCREENER_ENABLED:
        print("[tv-inst] disabled via TV_INST_SCREENER_ENABLED=0")
        if COMPAT_UNIVERSE_JSON.exists():
            try:
                return json.loads(COMPAT_UNIVERSE_JSON.read_text())
            except Exception:
                pass
        return {}

    try:
        rows, diag = fetch_institutional_candidates()
    except Exception as exc:
        print(f"[tv-inst] fetch failed: {exc}")
        diag = {
            "created_at_utc": _now_utc(), "data_tag": DATA_TAG,
            "fallback_used": True, "errors": [str(exc)],
        }
        _write_empty_outputs(diag)
        return {}

    if not rows:
        print("[tv-inst] no candidates found; writing empty fallback outputs")
        _write_empty_outputs(diag)
        return {}

    try:
        rows = _score_components(rows)
    except Exception as exc:
        print(f"[tv-inst] scoring failed: {exc}; using unsorted candidates")
        for r in rows:
            r.setdefault("institutional_accumulation_score", 0.0)

    rows = sorted(
        rows,
        key=lambda r: _to_float(r.get("institutional_accumulation_score"), 0.0),
        reverse=True,
    )

    # Apply minimum score filter
    qualified = [
        r for r in rows
        if _to_float(r.get("institutional_accumulation_score"), 0.0) >= TV_INST_MIN_SCORE
    ]
    if not qualified:
        print(f"[tv-inst] no candidates >= min_score={TV_INST_MIN_SCORE}; using top {len(rows)} without score filter")
        qualified = rows

    universe_size = min(TV_INST_TARGET_UNIVERSE_SIZE, TV_INST_MAX_SYMBOLS, len(qualified))
    selected = qualified[:universe_size]

    if len(selected) < TV_INST_TARGET_UNIVERSE_SIZE:
        print(f"[tv-inst] only {len(selected)} candidates passed filters")
    else:
        print(f"[tv-inst] scanned={diag.get('post_filter_count', 0)} selected={len(selected)}")

    for r in selected:
        r["accumulation_tags"] = _assign_tags(r)

    diag["selected_count"] = len(selected)
    diag.update({
        "tv_inst_limit": TV_INST_LIMIT,
        "tv_inst_min_price": TV_INST_MIN_PRICE,
        "tv_inst_max_price": TV_INST_MAX_PRICE,
        "tv_inst_min_market_cap": TV_INST_MIN_MARKET_CAP,
        "tv_inst_min_dollar_volume": TV_INST_MIN_DOLLAR_VOLUME,
        "tv_inst_min_rel_volume": TV_INST_MIN_REL_VOLUME,
        "tv_inst_min_score": TV_INST_MIN_SCORE,
        "tv_inst_target_size": TV_INST_TARGET_UNIVERSE_SIZE,
    })

    payload = _build_compat_payload(selected, diag)

    try:
        _write_all_outputs(payload, selected, diag)
    except Exception as exc:
        print(f"[tv-inst] output write error: {exc}")

    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    result = build_tradingview_institutional_universe()
    n = len(result.get("symbols", []))
    print(f"[tv-inst] complete: {n} symbols selected")
    sys.exit(0)
