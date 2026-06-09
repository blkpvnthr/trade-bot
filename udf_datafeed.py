#!/usr/bin/env python3
"""
udf_datafeed.py — UDF-compatible datafeed backend for the TradingView Advanced Charts library.

Serves the bot's OWN data — the strategy equity curve (QQEQUITY), insider buy/sell marks, and
cached daily bars — to a self-hosted TradingView Charting Library over the HTTP UDF protocol.
Credential-free (reads files, never live Alpaca) so it never touches the broker throttle.

HONEST SCOPE: this feeds YOUR data into TradingView's chart UI. It does NOT (and cannot) pull
TradingView's purchased OPRA/CME/Cboe feeds — those entitlements live inside tradingview.com.

Wire-up:
    from udf_datafeed import udf_bp
    app.register_blueprint(udf_bp, url_prefix="/dash/udf")
    # JS:  new Datafeeds.UDFCompatibleDatafeed("/dash/udf")
"""
from __future__ import annotations

import csv as _csv
import glob
import logging
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger("udf_datafeed")
ROOT = Path(__file__).resolve().parent
BARS_DIR = ROOT / "data" / "cache" / "alpaca_bars"
EXEC_DIR = ROOT / "logs" / "execution"

udf_bp = Blueprint("udf", __name__)
EQUITY_SYMBOL = "QQEQUITY"
_RES = ["1", "5", "15", "30", "60", "1D", "1W", "1M"]


@udf_bp.route("/config")
def config():
    return jsonify({
        "supported_resolutions": _RES,
        "supports_search": True, "supports_group_request": False,
        "supports_marks": True, "supports_timescale_marks": False, "supports_time": True,
        "exchanges": [{"value": "", "name": "All", "desc": ""},
                      {"value": "QuantumQuant", "name": "QuantumQuant", "desc": "Strategy"}],
        "symbols_types": [{"name": "All", "value": ""},
                          {"name": "Strategy", "value": "index"}, {"name": "Stock", "value": "stock"}],
    })


@udf_bp.route("/time")
def srv_time():
    return str(int(time.time()))


def _symbol_info(sym: str) -> dict:
    sym = (sym or "").upper().split(":")[-1]
    is_eq = sym == EQUITY_SYMBOL
    return {
        "name": sym, "ticker": sym,
        "description": "QuantumQuant strategy equity" if is_eq else sym,
        "type": "index" if is_eq else "stock",
        "session": "24x7" if is_eq else "0930-1600", "timezone": "America/New_York",
        "exchange": "QuantumQuant" if is_eq else "", "listed_exchange": "",
        "minmov": 1, "pricescale": 100, "has_intraday": True, "has_daily": True,
        "has_weekly_and_monthly": True, "supported_resolutions": _RES,
        "volume_precision": 0, "data_status": "streaming" if is_eq else "endofday",
        "currency_code": "USD",
    }


@udf_bp.route("/symbols")
def symbols():
    return jsonify(_symbol_info(request.args.get("symbol", "")))


def _cached_symbols():
    try:
        return sorted({p.stem.replace("_day", "") for p in BARS_DIR.glob("*_day.parquet")})
    except Exception:
        return []


@udf_bp.route("/search")
def search():
    q = (request.args.get("query") or "").upper()
    limit = int(request.args.get("limit") or 30)
    cands = [EQUITY_SYMBOL] + _cached_symbols()
    out = [{"symbol": s, "full_name": s,
            "description": ("Strategy equity" if s == EQUITY_SYMBOL else s),
            "exchange": "QuantumQuant" if s == EQUITY_SYMBOL else "",
            "ticker": s, "type": "index" if s == EQUITY_SYMBOL else "stock"}
           for s in cands if q in s][:limit]
    return jsonify(out)


@udf_bp.route("/history")
def history():
    sym = (request.args.get("symbol") or "").upper().split(":")[-1]
    frm = int(float(request.args.get("from") or 0))
    to = int(float(request.args.get("to") or time.time()))
    try:
        bars = _equity_bars() if sym == EQUITY_SYMBOL else _cached_bars(sym)
        if not bars:
            return jsonify({"s": "no_data"})
        win = [b for b in bars if frm <= b[0] <= to]
        bars = win or bars[-300:]
        return jsonify({"s": "ok",
                        "t": [b[0] for b in bars], "o": [b[1] for b in bars],
                        "h": [b[2] for b in bars], "l": [b[3] for b in bars],
                        "c": [b[4] for b in bars], "v": [b[5] for b in bars]})
    except Exception as e:
        logger.warning("history(%s) failed: %s", sym, e)
        return jsonify({"s": "error", "errmsg": str(e)[:120]})


def _cached_bars(sym: str):
    """Daily bars from cached parquet → [(unix, o, h, l, c, v)]. Fail-safe."""
    try:
        import pandas as pd
        p = BARS_DIR / f"{sym}_day.parquet"
        if not p.exists():
            return []
        df = pd.read_parquet(p)
        out = []
        for ts, row in df.iterrows():
            try:
                t = int(pd.Timestamp(ts).timestamp())
                c = float(row.get("close"))
                out.append((t, float(row.get("open", c)), float(row.get("high", c)),
                            float(row.get("low", c)), c, float(row.get("volume", 0) or 0)))
            except Exception:
                continue
        return sorted(out)
    except Exception as e:
        logger.debug("cached_bars(%s): %s", sym, e)
        return []


def _equity_bars():
    """Daily strategy equity from logs/execution/*/*_executor_equity_curve.csv. Fail-safe."""
    try:
        import pandas as pd
        series = {}   # day_unix -> last equity of the day
        for f in sorted(glob.glob(str(EXEC_DIR / "*" / "*_executor_equity_curve.csv"))):
            try:
                with open(f) as fh:
                    for row in _csv.DictReader(fh):
                        eq = next((float(row[k]) for k in
                                   ("equity", "account_equity", "total_equity", "value")
                                   if row.get(k)), None)
                        ts = row.get("timestamp") or row.get("time") or row.get("ts")
                        if eq is None or not ts:
                            continue
                        series[int(pd.Timestamp(ts).normalize().timestamp())] = eq
            except Exception:
                continue
        return [(d, series[d], series[d], series[d], series[d], 0) for d in sorted(series)]
    except Exception as e:
        logger.debug("equity_bars: %s", e)
        return []


@udf_bp.route("/marks")
def marks():
    """Insider buy/sell marks on the strategy equity symbol. Fail-safe → empty."""
    sym = (request.args.get("symbol") or "").upper().split(":")[-1]
    if sym != EQUITY_SYMBOL:
        return jsonify({"id": [], "time": [], "color": [], "text": [], "label": [],
                        "labelFontColor": [], "minSize": []})
    ev = _trade_marks()
    return jsonify({
        "id": [m["id"] for m in ev], "time": [m["time"] for m in ev],
        "color": [m["color"] for m in ev], "text": [m["text"] for m in ev],
        "label": [m["label"] for m in ev], "labelFontColor": [m["color"] for m in ev],
        "minSize": [16 for _ in ev],
    })


def _trade_marks():
    """Buy/sell events from the executor journal. Fail-safe → []."""
    out = []
    try:
        import pandas as pd
        for f in sorted(glob.glob(str(EXEC_DIR / "*" / "*_executor_journal.csv")))[-30:]:
            try:
                with open(f) as fh:
                    for i, row in enumerate(_csv.DictReader(fh)):
                        side = (row.get("side") or "").lower()
                        ts = row.get("timestamp") or row.get("time")
                        if side not in ("buy", "sell") or not ts:
                            continue
                        out.append({"id": f"{Path(f).stem}_{i}",
                                    "time": int(pd.Timestamp(ts).normalize().timestamp()),
                                    "color": "#26a69a" if side == "buy" else "#ef5350",
                                    "text": f"{side.upper()} {row.get('symbol', '')}",
                                    "label": side[0].upper()})
            except Exception:
                continue
    except Exception as e:
        logger.debug("trade_marks: %s", e)
    return out[-500:]
