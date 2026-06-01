#!/usr/bin/env python3
"""
funnel_report.py — execution-funnel visibility (read-only / observational).
============================================================================

Builds the per-cycle execution funnel from the artifacts the engines ALREADY write — it
changes no thresholds, gates, or decisions, and submits nothing. Stages:

    Universe → Tradable → Signals → Setups Qualified → Approved → Submitted → Filled

and categorises every dropped symbol into the rejection buckets:

    buffett_score | options_score | liquidity | market_regime | rl_gate
    risk_manager | exposure_limits | position_limits | confidence | already_held | other

Sources (all optional; missing → that stage reads 0, never raises):
  * daily_system_outputs/alpha_options_universe.json   (alpha: scanned / candidates / all_scanned)
  * daily_system_outputs/buffett_dca_signals.json      (buffett: per-symbol signals + reasons)
  * logs/execution/{tag}/{tag}_executor_journal.csv    (orders submitted / filled)
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "daily_system_outputs"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _alpha_bucket(reason: str) -> str:
    r = (reason or "").lower()
    # Missing data first — "no_bars" etc. is a DATA gap, not a real score rejection (premarket).
    if any(k in r for k in ("no_bars", "not_enough_bars", "no_data", "missing", "data_unavailable",
                            "no_symbol_bars", "no_premarket_bars")):
        return "missing_data"
    if "regime_gate" in r or "halt_new_buys" in r or "regime" in r:
        return "market_regime"
    if "rl_gate" in r or "rl_reject" in r or "rl_timing" in r:
        return "rl_gate"
    if "exposure" in r:
        return "exposure_limits"
    if "positions_full" in r or "max_positions" in r:
        return "position_limits"
    if "already_in_position" in r:
        return "already_held"
    if "entries_disabled" in r:
        return "entries_disabled"
    if any(k in r for k in ("liquid", "spread", "open_interest", "oi", "delta", "no_qualified",
                            "contract", "overpriced", "theta", "mispric", "iv_")):
        return "liquidity"
    if "setup_quality" in r or "setup_score" in r or "min_setup" in r:
        return "options_score"
    if any(k in r for k in ("no_setup", "confirm", "reclaim", "sweep", "continuation",
                            "momentum", "structure")):
        return "technical_confirmation"
    if not r:
        return "(passed/ready)"
    return "other"


def _buffett_bucket(reason: str) -> str:
    r = (reason or "").lower()
    if any(k in r for k in ("data_unavailable", "no_data", "missing")):
        return "missing_data"
    if "regime_gate" in r:
        return "market_regime"
    if "rl_gate" in r or "rl_reject" in r:
        return "rl_gate"
    if "bal_score" in r or "thesis_broken" in r or "quality" in r:
        return "buffett_score"
    if "low_conviction" in r or "conviction" in r:
        return "confidence"
    if "score_too_low" in r or "composite" in r:
        return "buffett_score"          # composite DCA timing score (Buffett-side)
    if "not_undervalued" in r or "overvalued" in r or "valuation" in r:
        return "valuation"
    if "risk" in r:
        return "risk_manager"
    if not r:
        return "(approved)"
    return "other"


def build_funnel(data_tag: str = None) -> Dict[str, Any]:
    data_tag = data_tag or os.getenv("RUN_DATA_DATE") or ""
    out: Dict[str, Any] = {"data_tag": data_tag, "stages": {}, "rejections": {},
                           "alpha": {}, "buffett": {}, "orders": {}, "notes": []}

    # ── Alpha funnel ──────────────────────────────────────────────────────────────
    a = _load_json(DAILY / "alpha_options_universe.json") or {}
    a_scanned = int(a.get("scanned", 0) or 0)
    a_ready = int(a.get("candidate_count", 0) or 0)
    a_rows = a.get("all_scanned") or []
    # Date-strict: only count the alpha artifact if it is TODAY's. A stale artifact (e.g. left
    # from a prior session or an offline harness) must not be reported as the current funnel.
    a_stale = bool(data_tag and a.get("date_tag") and str(a.get("date_tag")) != str(data_tag))
    if a_stale:
        out["notes"].append(f"alpha entry pass has NOT run today — artifact dated {a.get('date_tag')} "
                            f"(not {data_tag}); alpha funnel counted as 0 for today (premarket/closed)")
        a_scanned, a_ready, a_rows = 0, 0, []
    a_rej: Dict[str, int] = {}
    a_top: List[dict] = []
    for r in a_rows:
        if not isinstance(r, dict):
            continue
        b = _alpha_bucket(r.get("reject_reason", ""))
        a_rej[b] = a_rej.get(b, 0) + 1
        a_top.append({"symbol": r.get("symbol"), "score": r.get("setup_score") or r.get("alpha_screener_score"),
                      "rejection_reason": r.get("reject_reason", ""), "bucket": b, "engine": "alpha"})
    out["alpha"] = {"scanned": a_scanned, "ready_signals": a_ready, "market_regime": a.get("market_regime"),
                    "rejections": a_rej, "artifact_date": a.get("date_tag")}

    # ── Buffett funnel ────────────────────────────────────────────────────────────
    bf = _load_json(DAILY / "buffett_dca_signals.json") or {}
    b_rows = bf.get("signals") or []
    b_rej: Dict[str, int] = {}
    b_buys = 0
    b_top: List[dict] = []
    for r in b_rows:
        if not isinstance(r, dict):
            continue
        act = str(r.get("DCA_action") or r.get("dca_action") or "")
        reason = r.get("journal_reason") or r.get("reject_reason") or r.get("dca_reason") or ""
        bucket = _buffett_bucket(reason)
        b_rej[bucket] = b_rej.get(bucket, 0) + 1
        if act in ("small_dca", "normal_dca", "aggressive_dca") and r.get("risk_limits_passed"):
            b_buys += 1
        b_top.append({"symbol": r.get("symbol"),
                      "score": r.get("composite_dca_score") or r.get("composite") or r.get("bal_score"),
                      "bal_score": r.get("bal_score") or r.get("BAL_score"),
                      "conviction": r.get("conviction_score") or r.get("conviction"),
                      "action": act, "rejection_reason": str(reason), "bucket": bucket, "engine": "buffett"})
    out["buffett"] = {"evaluated": len(b_rows), "approved_buys": b_buys, "rejections": b_rej,
                      "artifact_date": bf.get("date_tag"), "count_field": bf.get("count")}

    # Per-symbol rejection rows (symbol / score / rejection_reason / bucket) across both engines.
    out["per_symbol_rejections"] = [
        {"symbol": r.get("symbol"), "score": r.get("score"), "engine": r.get("engine"),
         "rejection_reason": r.get("rejection_reason"), "bucket": r.get("bucket")}
        for r in (a_top + b_top)
        if r.get("bucket") not in ("(passed/ready)", "(approved)")]

    # ── Orders submitted / filled (executor journal) ───────────────────────────────
    submitted = filled = 0
    # Date-strict: ONLY today's executor journal. No glob fallback — a prior session's journal
    # must never be reported as today's order flow.
    jpath = (ROOT / "logs" / "execution" / data_tag / f"{data_tag}_executor_journal.csv") if data_tag else None
    if jpath and jpath.exists():
        try:
            with open(jpath, newline="") as f:
                for row in csv.DictReader(f):
                    submitted += 1
                    st = str(row.get("status") or row.get("order_status") or "").lower()
                    if "fill" in st:
                        filled += 1
        except Exception:
            pass
    else:
        out["notes"].append("no executor journal for today → 0 orders submitted this session")
    out["orders"] = {"journal": str(jpath) if (jpath and jpath.exists()) else None,
                     "submitted": submitted, "filled": filled}

    # ── unified rejection buckets + stage counts ────────────────────────────────────
    rej: Dict[str, int] = {}
    for src in (a_rej, b_rej):
        for k, v in src.items():
            if k.startswith("(") or k == "entries_disabled":
                continue
            rej[k] = rej.get(k, 0) + v
    out["rejections"] = rej

    universe = max(a_scanned, len(b_rows), 0)
    out["stages"] = {
        "universe": universe,
        "tradable": a_scanned + len(b_rows),
        "signals_generated": a_scanned + len(b_rows),
        "setups_qualified": a_ready + b_buys,
        "approved": a_ready + b_buys,
        "submitted": submitted,
        "filled": filled,
    }
    out["top_candidates"] = (sorted(b_top, key=lambda x: float(x.get("score") or 0), reverse=True)[:7]
                             + a_top[:3])[:10]
    return out


def format_funnel(f: Dict[str, Any]) -> str:
    s = f["stages"]
    line = (f"FUNNEL universe={s['universe']} → tradable={s['tradable']} → signals={s['signals_generated']} "
            f"→ qualified={s['setups_qualified']} → approved={s['approved']} → submitted={s['submitted']} "
            f"→ filled={s['filled']}")
    rej = " | ".join(f"{k}={v}" for k, v in sorted(f["rejections"].items(), key=lambda kv: -kv[1])) or "none"
    return line + f"  [rejections: {rej}]"


def log_funnel(logger, data_tag: str = None, max_rejects: int = 20) -> None:
    """Emit the per-cycle FUNNEL line + one line per rejected symbol (symbol / score /
    rejection_reason / bucket). Observational; never raises (safe in the live loop)."""
    try:
        f = build_funnel(data_tag)
        logger.info(format_funnel(f))
        for row in (f.get("per_symbol_rejections") or [])[:max_rejects]:
            logger.info("[funnel-reject] %s score=%s reason=%s gate=%s engine=%s",
                        row.get("symbol"), row.get("score"), row.get("rejection_reason"),
                        row.get("bucket"), row.get("engine"))
    except Exception as exc:  # pragma: no cover
        try:
            logger.debug("funnel_report skipped: %s", exc)
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    f = build_funnel(tag)
    print(format_funnel(f))
    for n in f.get("notes", []):
        print("  NOTE:", n)
    print("\nAlpha :", json.dumps(f["alpha"], default=str))
    print("Buffett:", json.dumps(f["buffett"], default=str))
    print("Orders :", json.dumps(f["orders"], default=str))
    print("\nTop candidates examined (symbol / scores / pass-fail):")
    for c in f["top_candidates"]:
        print("  ", json.dumps(c, default=str))
    pr = f.get("per_symbol_rejections") or []
    print(f"\nPer-symbol rejections ({len(pr)}):")
    for row in pr[:20]:
        print(f"  {str(row.get('symbol')):8} score={row.get('score')} "
              f"gate={row.get('bucket')} reason={row.get('rejection_reason')} [{row.get('engine')}]")
