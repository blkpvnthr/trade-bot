#!/usr/bin/env python3
"""
smoke_check_market_data_packages.py — Live API Smoke Checks (manual only)
========================================================================

Validates that the OPTIONAL live market-data packages still match the API
assumptions baked into the trading system BEFORE relying on them in live
premarket/RTH operation:

    1. streaming-indicators  → protects streaming_indicator_engine.py
                               (_stream_value, the SuperTrend block, fallback)
    2. yliveticker           → protects PremarketLiveFeed.on_message
                               (callback shape on_ticker(ws, msg); msg is a dict;
                                symbol arrives under "id"/"symbol")

This is a MANUAL smoke check. It is intentionally NOT a unit test and must never
run inside `python -m unittest discover -s tests -p "test_*.py"` (it lives in
scripts/, not tests/, and requires live network/package/runtime behaviour).

Safety:
    * Never places orders. Never imports AlpacaGateway / execution.
    * Never mutates strategy configs.
    * Safe to run manually; degrades gracefully when packages/quotes are absent.

CLI:
    python scripts/smoke_check_market_data_packages.py --allow-missing
    python scripts/smoke_check_market_data_packages.py --require-tick
    python scripts/smoke_check_market_data_packages.py --symbols AAPL SPY QQQ
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

# Expected streaming-indicators classes (what streaming_indicator_engine.py uses).
EXPECTED_SI_CLASSES = ["RSI", "SMA", "EMA", "VWAP", "ATR"]
# Expected yliveticker constructor params, and params that MUST NOT be present.
EXPECTED_YLT_PARAMS = ["self", "on_ticker", "ticker_names", "reconnect", "ping_interval", "ping_timeout"]
FORBIDDEN_YLT_PARAMS = ["on_open", "on_error", "on_close"]
# Message keys the parser may see (PremarketLiveFeed.on_message uses .get() for all).
KNOWN_MSG_KEYS = ["id", "symbol", "price", "previousClose", "dayVolume", "dayHigh",
                  "dayLow", "change", "changePercent", "bid", "ask", "marketHours"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ───────────────────────── streaming-indicators check ─────────────────────────
def run_streaming_indicators_check(si_module: Any) -> Dict[str, Any]:
    """Validate the streaming-indicators API. `si_module` is the imported module or None."""
    errors: List[str] = []
    if si_module is None:
        return {"available": False, "status": "missing", "errors": []}

    for cls_name in EXPECTED_SI_CLASSES:
        cls = getattr(si_module, cls_name, None)
        if cls is None:
            errors.append(f"missing class {cls_name} (streaming_indicator_engine expects it)")
            continue
        # update/add method name check (engine feeds via .update(...))
        if not (hasattr(cls, "update") or hasattr(cls, "add")):
            errors.append(f"{cls_name} has neither .update() nor .add() — engine feeds via .update()")

    # SuperTrend behaviour (optional): engine reads .super_trend / .final_band, or a tuple return.
    st_cls = getattr(si_module, "SuperTrend", None)
    if st_cls is None:
        errors.append("SuperTrend not available — engine SuperTrend block will use static fallback (non-fatal)")

    # Returned values must convert to float safely once warm.
    try:
        sma = getattr(si_module, "SMA", None)
        if sma is not None:
            obj = sma(3)
            last = None
            for v in (1.0, 2.0, 3.0, 4.0):
                last = obj.update(v) if hasattr(obj, "update") else (obj.add(v) if hasattr(obj, "add") else None)
            val = last if last is not None else getattr(obj, "value", None)
            if val is not None:
                float(val)   # must not raise
    except Exception as exc:
        errors.append(f"SMA value not float-convertible: {exc}")

    status = "ok" if not errors else "mismatch"
    return {"available": True, "status": status, "errors": errors}


# ───────────────────────────── yliveticker check ──────────────────────────────
def check_yliveticker_signature(ylt_module: Any) -> Dict[str, Any]:
    """Inspect YLiveTicker.__init__ params WITHOUT constructing it (it blocks forever)."""
    out: Dict[str, Any] = {"signature": [], "signature_ok": False, "errors": []}
    try:
        params = list(inspect.signature(ylt_module.YLiveTicker.__init__).parameters)
        out["signature"] = params
        missing = [p for p in EXPECTED_YLT_PARAMS if p not in params]
        forbidden = [p for p in FORBIDDEN_YLT_PARAMS if p in params]
        if missing:
            out["errors"].append(f"missing expected params: {missing}")
        if forbidden:
            out["errors"].append(
                f"unexpected params {forbidden} present — PremarketLiveFeed removed on_open/"
                f"on_error/on_close assumptions; re-check premarket_live_feed.py")
        out["signature_ok"] = not missing and not forbidden
    except Exception as exc:
        out["errors"].append(f"could not inspect YLiveTicker signature: {exc}")
    return out


def run_yliveticker_check(ylt_module: Any, symbols: List[str],
                          live_seconds: float = 8.0, require_tick: bool = False) -> Dict[str, Any]:
    """
    Validate the yliveticker API. `ylt_module` is the imported module or None.
    Briefly streams in a DAEMON thread (YLiveTicker blocks), closing the socket on the
    first tick. Zero ticks outside market hours is NOT fatal unless require_tick=True.
    """
    if ylt_module is None:
        return {"available": False, "signature": [], "signature_ok": False,
                "callback_ok": False, "ticks_received": 0, "message_keys": [], "errors": []}

    sig = check_yliveticker_signature(ylt_module)
    seen: List[dict] = []
    keys: set = set()
    callback_ok = False
    errors: List[str] = list(sig["errors"])
    symbol_key_ok = True

    def on_ticker(ws, msg):
        nonlocal callback_ok, symbol_key_ok
        try:
            if isinstance(msg, dict):
                seen.append(msg)
                keys.update(msg.keys())
                callback_ok = True
                if not (msg.get("id") or msg.get("symbol")):
                    symbol_key_ok = False
            try:
                ws.close()
            except Exception:
                pass
        except Exception as exc:
            errors.append(f"on_ticker handler error: {exc}")

    if live_seconds > 0:
        def _run():
            try:
                ylt_module.YLiveTicker(on_ticker=on_ticker, ticker_names=list(symbols))
            except Exception as exc:
                errors.append(f"YLiveTicker run error: {exc}")
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout=live_seconds)

    if seen and not isinstance(seen[0], dict):
        errors.append("message is not a dict — PremarketLiveFeed.on_message expects a dict")
    if seen and not symbol_key_ok:
        errors.append("symbol not under 'id'/'symbol' — PremarketLiveFeed.on_message will not "
                      "populate quotes; re-check the symbol key")
    if require_tick and not seen:
        errors.append("no ticks received with --require-tick set")

    return {"available": True, "signature": sig["signature"], "signature_ok": sig["signature_ok"],
            "callback_ok": callback_ok, "ticks_received": len(seen),
            "message_keys": sorted(keys), "errors": errors}


# ───────────────────────────── report + main ──────────────────────────────────
def _recommendations(si: Dict[str, Any], ylt: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    if not si["available"]:
        recs.append("streaming-indicators not installed → StreamingIndicatorEngine uses static_fallback.")
    elif si["status"] != "ok":
        recs.append("streaming-indicators API mismatch → review streaming_indicator_engine.py "
                    "_stream_value / SuperTrend block before relying on streaming values.")
    if not ylt["available"]:
        recs.append("yliveticker not installed → PremarketLiveFeed uses yfinance/Alpaca fallback.")
    else:
        if not ylt["signature_ok"]:
            recs.append("yliveticker constructor signature changed → re-check PremarketLiveFeed._run_stream.")
        if ylt["ticks_received"] == 0:
            recs.append("0 ticks (often just no live quotes outside market hours). Re-run during "
                        "premarket/RTH; use --require-tick to enforce.")
        if ylt["message_keys"] and not ({"id", "symbol"} & set(ylt["message_keys"])):
            recs.append("yliveticker messages lack 'id'/'symbol' → PremarketLiveFeed.on_message "
                        "will not populate quotes.")
    if not recs:
        recs.append("All checked package APIs match system assumptions.")
    return recs


def build_report(si: Dict[str, Any], ylt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "streaming_indicators_available": si["available"],
        "streaming_indicators_status": si["status"],
        "streaming_indicators_errors": si["errors"],
        "yliveticker_available": ylt["available"],
        "yliveticker_signature": ylt["signature"],
        "yliveticker_signature_ok": ylt["signature_ok"],
        "yliveticker_callback_ok": ylt["callback_ok"],
        "yliveticker_ticks_received": ylt["ticks_received"],
        "yliveticker_message_keys": ylt["message_keys"],
        "yliveticker_errors": ylt["errors"],
        "recommendations": _recommendations(si, ylt),
    }


def write_report(report: Dict[str, Any], out_dir: Optional[Path] = None) -> Optional[Path]:
    tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = Path(out_dir) if out_dir else (DAILY_OUTPUTS / tag)
    try:
        target.mkdir(parents=True, exist_ok=True)
        p = target / "market_data_package_smoke_check.json"
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return p
    except Exception as exc:
        print(f"WARN: could not write report: {exc}", file=sys.stderr)
        return None


def _import_optional(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live smoke checks for optional market-data packages.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="exit 0 even if an optional package is absent")
    ap.add_argument("--require-tick", action="store_true",
                    help="fail if yliveticker receives zero ticks")
    ap.add_argument("--symbols", nargs="+", default=["AAPL", "SPY"],
                    help="symbols to subscribe for the yliveticker check")
    ap.add_argument("--live-seconds", type=float, default=8.0,
                    help="seconds to stream yliveticker before stopping")
    args = ap.parse_args(argv)

    si_module = _import_optional("streaming_indicators")
    ylt_module = _import_optional("yliveticker")

    if si_module is None:
        print("STREAMING_INDICATORS_AVAILABLE=false")
    if ylt_module is None:
        print("YLIVETICKER_AVAILABLE=false")

    si = run_streaming_indicators_check(si_module)
    ylt = run_yliveticker_check(ylt_module, args.symbols,
                                live_seconds=args.live_seconds, require_tick=args.require_tick)
    report = build_report(si, ylt)
    path = write_report(report)
    print(json.dumps(report, indent=2, default=str))
    if path:
        print(f"\nReport: {path}")

    # Exit code policy
    missing = (not si["available"]) or (not ylt["available"])
    if missing and not args.allow_missing:
        return 1
    if args.require_tick and ylt["available"] and ylt["ticks_received"] == 0:
        return 1
    if si["available"] and si["status"] == "mismatch":
        return 1
    if ylt["available"] and not ylt["signature_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
