#!/usr/bin/env python3
"""
buffett_accumulation_screener.py — Buffett Accumulation Screener.

Finds businesses worth OWNING for the long term, on a universe that is INDEPENDENT
of the Alpha options universe. It reuses the Buffett Accumulation Layer's scoring
(buffett_accumulation_layer.score_symbol / get_fundamentals) — it does not duplicate
fundamental logic — and produces a ranked ownership-quality universe.

Output:
    daily_system_outputs/buffett_accumulation_universe.json

Pipeline:
    run.py
      → BuffettAccumulationScreener.screen(market_context, ...)
      → candidate symbols → BuffettAccumulationLayer.scan(...)
      → long-term stock accumulation

The screener selects on BUSINESS QUALITY + VALUATION (timing-independent). WHEN to
accumulate (macro/cycle/technical timing) remains the BAL layer's job downstream.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from buffett_accumulation_layer import (
    BALConfig, score_symbol, get_fundamentals,
    multi_model_intrinsic_value, compute_conviction_score, conviction_target_allocation,
    compute_insider_buy_score, _earnings_quality_score,
    _dca_valuation_discount_score, _dca_macro_cycle_opportunity_score,
)

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)
BUFFETT_UNIVERSE_PATH = DAILY_OUTPUTS / "buffett_accumulation_universe.json"
OWNERSHIP_UNIVERSE_PATH = DAILY_OUTPUTS / "ownership_universe.json"

# ── config ───────────────────────────────────────────────────────────────────
ENABLE = os.getenv("ENABLE_BUFFETT_ACCUMULATION_SCREENER", "true").lower() in {"1", "true"}
UNIVERSE_SIZE = int(os.getenv("BUFFETT_UNIVERSE_SIZE", "100"))
MIN_SCREENER_SCORE = float(os.getenv("BUFFETT_MIN_SCREENER_SCORE", "70"))
DISCOVERY_SOURCE = "SEC_FORM4_DISCOVERY"

# Curated S&P 500 / Nasdaq 100 quality compounders (seed universe).
STATIC_QUALITY = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ADBE", "CRM", "ORCL",
    "CSCO", "ACN", "TXN", "QCOM", "AMD", "INTU", "NOW", "ISRG", "AMAT", "LRCX", "KLAC",
    "PANW", "V", "MA", "AXP", "JPM", "BAC", "GS", "MS", "BLK", "SPGI", "MCO",
    "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP", "PM",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "REGN", "VRTX",
    "LIN", "CAT", "DE", "HON", "UNP", "GE", "BRK.B",
]


def _watchlist() -> list[str]:
    raw = os.getenv("BUFFETT_WATCHLIST", "")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _ownership_symbols() -> list[str]:
    if OWNERSHIP_UNIVERSE_PATH.exists():
        try:
            return [str(k).upper() for k in json.loads(OWNERSHIP_UNIVERSE_PATH.read_text()).keys()]
        except Exception:
            pass
    return []


def _sec_form4_discovered_symbols() -> list[str]:
    """SEC Form 4 insider-buy discoveries to merge into the review universe (fail-safe)."""
    try:
        import sec_form4_feed
        return [str(s).upper() for s in sec_form4_feed.discovered_symbols()]
    except Exception:
        return []


def _sec_insider_candidates() -> dict:
    """Symbol-keyed SEC insider candidate map (fail-safe, empty when unavailable)."""
    try:
        import sec_form4_feed
        return {str(k).upper(): v for k, v in sec_form4_feed.load_insider_candidates().items()}
    except Exception:
        return {}


def build_buffett_universe(extra_symbols: list[str] | None = None,
                           size: int | None = None) -> list[str]:
    """
    Quality seed + existing ownership + watchlist + SEC Form 4 discoveries (+ caller
    extras), deduped & capped. SEC Form 4 only EXPANDS the review universe — every
    symbol still runs the full Buffett pipeline downstream.
    """
    size = size or UNIVERSE_SIZE
    syms = list(dict.fromkeys(
        STATIC_QUALITY + _ownership_symbols() + _watchlist() + _sec_form4_discovered_symbols() +
        [s.strip().upper() for s in (extra_symbols or []) if s and s.strip()]
    ))
    return syms[:size]


# ── sub-scores (normalised 0-100) ─────────────────────────────────────────────
def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _profitability_score(fd) -> float:
    """0-100 profitability quality from ROE / ROIC / margins."""
    roe = _clip(float(getattr(fd, "roe", 0) or 0) / 0.20, 0, 1)
    roic = _clip(float(getattr(fd, "roic", 0) or 0) / 0.15, 0, 1)
    net_m = _clip(float(getattr(fd, "net_margin", 0) or 0) / 0.20, 0, 1)
    op_m = _clip(float(getattr(fd, "operating_margin", 0) or 0) / 0.25, 0, 1)
    return round(100.0 * (0.35 * roe + 0.30 * roic + 0.20 * net_m + 0.15 * op_m), 1)


def _norm(v, vmax):
    return _clip(float(v or 0) / vmax, 0.0, 1.0) if vmax else 0.0


def screen(
    market_context: dict,
    symbols: list[str] | None = None,
    config: BALConfig | None = None,
    bars_ctx_by_symbol: dict[str, dict] | None = None,
    write: bool = True,
) -> dict:
    """
    Score the ownership universe and return ranked accumulation candidates.

    Returns {"candidates": [symbols], "detail": [rows], "funnel": {...}, "output_path": str}.
    Never raises on a single-symbol failure — that symbol is bucketed as a rejection.
    """
    cfg = config or BALConfig()
    bars_ctx_by_symbol = bars_ctx_by_symbol or {}
    universe = symbols if symbols is not None else build_buffett_universe()
    sec_map = _sec_insider_candidates()   # SEC Form 4 discoveries (fail-safe)

    rows: list[dict] = []
    funnel = {
        "scanned": len(universe), "passed": 0,
        "rejected_by_fundamentals": 0, "rejected_by_valuation": 0,
        "rejected_by_macro_cycle": 0,
    }

    for symbol in universe:
        try:
            card = score_symbol(symbol, cfg, market_context, bars_ctx_by_symbol.get(symbol, {}))
            fd = get_fundamentals(symbol, cfg)
        except Exception as exc:
            rows.append({"symbol": symbol, "buffett_screener_score": 0.0, "BAL_score": 0.0,
                         "accumulation_action": "no_action",
                         "reject_reason": f"rejected_by_fundamentals:error:{str(exc)[:80]}",
                         "is_candidate": False})
            funnel["rejected_by_fundamentals"] += 1
            continue

        moat_n = _norm(card.moat_score, 15.0)
        fin_n = _norm(card.financial_score, 20.0)
        mgmt_n = _norm(card.management_score, 10.0)
        val_n = _norm(card.valuation_score, 20.0)
        macro_n = _norm(card.macro_score, 10.0)
        cycle_n = _norm(card.cycle_score, 10.0)
        tech_n = _norm(card.technical_score, 5.0)
        prof = _profitability_score(fd)

        screener_score = round(100.0 * (
            0.20 * moat_n + 0.25 * fin_n + 0.20 * (prof / 100.0) +
            0.15 * mgmt_n + 0.20 * val_n
        ), 1)

        data_bad = bool(getattr(fd, "fetch_error", "")) or float(getattr(fd, "current_price", 0) or 0) <= 0
        macro_cycle_n = 0.5 * macro_n + 0.5 * cycle_n
        reject = ""
        if data_bad:
            reject = "rejected_by_fundamentals:data_unavailable"
        elif screener_score < MIN_SCREENER_SCORE:
            # attribute to the weakest dimension
            quality_min = min(moat_n, fin_n, prof / 100.0, mgmt_n)
            if val_n < quality_min:
                reject = f"rejected_by_valuation:score={screener_score:.0f}<{MIN_SCREENER_SCORE:.0f}"
            else:
                reject = f"rejected_by_fundamentals:score={screener_score:.0f}<{MIN_SCREENER_SCORE:.0f}"
        elif macro_cycle_n < 0.20:
            reject = f"rejected_by_macro_cycle:macro_cycle={macro_cycle_n:.2f}"

        if reject.startswith("rejected_by_valuation"):
            funnel["rejected_by_valuation"] += 1
        elif reject.startswith("rejected_by_macro_cycle"):
            funnel["rejected_by_macro_cycle"] += 1
        elif reject.startswith("rejected_by_fundamentals"):
            funnel["rejected_by_fundamentals"] += 1
        else:
            funnel["passed"] += 1

        # ── multi-model valuation + conviction (insider defaults to none here) ──
        price = float(getattr(fd, "current_price", 0) or 0)
        val_detail = multi_model_intrinsic_value(fd, market_context)
        iv = float(val_detail.get("intrinsic_value", 0.0) or 0.0)
        p2iv = round(price / iv, 4) if (iv > 0 and price > 0) else 0.0
        val_discount_s = _dca_valuation_discount_score(p2iv)
        macro_cycle_s = _dca_macro_cycle_opportunity_score(card, market_context)
        eq_s = _earnings_quality_score(fd)
        # SEC Form 4 insider discovery (discovery/attention only — never a buy signal).
        sec_rec = sec_map.get(str(symbol).upper(), {})
        sec_discovered = bool(sec_rec)
        insider = compute_insider_buy_score(sec_rec or None, fd)
        # Insider score only modifies conviction once quality + valuation already pass.
        bal_ok = float(card.bal_score) >= cfg.min_score_to_buy
        undervalued = 0 < p2iv <= 1.0
        insider_used = bool(insider["detected"]) and bal_ok and undervalued
        insider_for_conviction = insider["score"] if insider_used else 0.0
        conviction = compute_conviction_score(
            card, fd, val_discount_s, macro_cycle_s, insider_for_conviction, eq_s, prof)["conviction_score"]
        target_alloc_pct, _max_alloc_pct, _alloc_reason = conviction_target_allocation(conviction, cfg=None)

        rows.append({
            "symbol": symbol,
            "moat_score": round(moat_n * 100, 1),
            "financial_strength_score": round(fin_n * 100, 1),
            "profitability_score": prof,
            "management_score": round(mgmt_n * 100, 1),
            "valuation_score": round(val_n * 100, 1),
            "macro_score": round(macro_n * 100, 1),
            "cycle_score": round(cycle_n * 100, 1),
            "technical_entry_score": round(tech_n * 100, 1),
            "buffett_screener_score": screener_score,
            "BAL_score": round(float(card.bal_score), 1),
            "accumulation_action": card.action,
            # ── valuation + conviction (upgrade 5 fields) ──
            "intrinsic_value": iv,
            "price_to_intrinsic_value": p2iv,
            "valuation_model_values": val_detail.get("valuation_model_values", {}),
            "valuation_model_count": int(val_detail.get("valuation_model_count", 0)),
            "valuation_confidence": float(val_detail.get("valuation_confidence", 0.0)),
            "valuation_dispersion": float(val_detail.get("valuation_dispersion", 0.0)),
            "primary_valuation_method": str(val_detail.get("primary_valuation_method", "none")),
            "conviction_score": conviction,
            "target_allocation_pct": round(target_alloc_pct, 4),
            # ── SEC Form 4 insider discovery fields ──
            "sec_form4_discovered": sec_discovered,
            "discovery_source": DISCOVERY_SOURCE if sec_discovered else "",
            "insider_buy_detected": bool(insider["detected"]),
            "insider_buy_score": float(insider["score"]),
            "insider_buy_amount": float(sec_rec.get("insider_buy_amount", 0.0) or 0.0),
            "insider_count": int(sec_rec.get("insider_count", 0) or 0),
            "insider_roles": sec_rec.get("insider_roles", []),
            "insider_buy_source": sec_rec.get("source", "") if sec_discovered else "",
            "insider_buy_used_in_conviction": insider_used,
            "latest_insider_buy_date": sec_rec.get("latest_insider_buy_date", ""),
            "sec_filing_urls": sec_rec.get("sec_filing_urls", []),
            "reject_reason": reject,
            "is_candidate": reject == "",
        })

    candidates = sorted([r for r in rows if r["is_candidate"]],
                        key=lambda r: r["buffett_screener_score"], reverse=True)

    if write:
        try:
            BUFFETT_UNIVERSE_PATH.write_text(json.dumps({
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "market_regime": str(market_context.get("overall_market_bias")),
                "screener": {"min_score": MIN_SCREENER_SCORE, "universe_size": UNIVERSE_SIZE},
                "scanned": len(rows),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "all_scanned": rows,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    return {
        "candidates": [r["symbol"] for r in candidates],
        "detail": rows,
        "funnel": funnel,
        "output_path": str(BUFFETT_UNIVERSE_PATH),
    }


if __name__ == "__main__":
    import sys
    ctx = {"overall_market_bias": "bullish", "rate_direction": "down",
           "inflation_trend": "down", "recession_risk": "low", "cycle_phase": "expansion"}
    res = screen(ctx, symbols=build_buffett_universe(size=int(os.getenv("BUFFETT_UNIVERSE_SIZE", "8"))))
    print(json.dumps({"candidates": res["candidates"], "funnel": res["funnel"]}, indent=2))
    sys.exit(0)
