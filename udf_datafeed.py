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

RUN_LOG = ROOT / "logs" / "run.log"
import re as _re
_EQ_RE = _re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*Account cycle=\d+: equity=\$([\d.]+)")
_ORD_RE = _re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*ORDER (buy|sell) ([A-Z0-9.]+)")
_LCACHE = {"eq": None, "eq_t": 0.0, "mk": None, "mk_t": 0.0}

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
        "description": "QuantumQuant account equity" if is_eq else sym,
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
            "description": ("Account equity" if s == EQUITY_SYMBOL else s),
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
        res = request.args.get("resolution") or "1D"
        bars = _equity_bars(res) if sym == EQUITY_SYMBOL else _cached_bars(sym)
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


def _equity_points():
    """All (unix_ts, equity) account-equity readings from run.log Account-cycle lines
    (full timestamp — the reliable source). Cached 60s. Fail-safe → []."""
    import datetime
    now = time.time()
    if _LCACHE["eq"] is not None and now - _LCACHE["eq_t"] < 60:
        return _LCACHE["eq"]
    pts = []
    try:
        with open(RUN_LOG, errors="replace") as fh:
            for line in fh:
                m = _EQ_RE.match(line)
                if not m:
                    continue
                # Drop uninitialized / paper-account snapshots (status=n/a). After a restart the
                # bot can momentarily log a fresh paper account (equity=$100000, status=n/a) before
                # the live account loads; those readings would otherwise spike the equity curve.
                if "status=n/a" in line:
                    continue
                t = int(datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=datetime.timezone.utc).timestamp())
                pts.append((t, float(m.group(3))))
    except Exception as e:
        logger.debug("equity_points: %s", e)
    pts.sort()
    _LCACHE["eq"], _LCACHE["eq_t"] = pts, now
    return pts


def _equity_bars(resolution: str = "1D"):
    """Account-equity bars from run.log. Intraday resolution → one bar per cycle reading
    (the full equity curve); daily/weekly/monthly → OHLC aggregated per UTC day. Fail-safe → []."""
    import datetime
    pts = _equity_points()
    if not pts:
        return []
    intraday = str(resolution).upper() not in ("1D", "1W", "1M", "D", "W", "M")
    if intraday:
        seen = {}   # collapse duplicate timestamps, keep last reading
        for t, eq in pts:
            seen[t] = eq
        return [(t, seen[t], seen[t], seen[t], seen[t], 0) for t in sorted(seen)]
    days = {}   # day_unix -> [open, high, low, close]
    for t, eq in pts:
        d = int(datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        if d not in days:
            days[d] = [eq, eq, eq, eq]
        else:
            days[d][1] = max(days[d][1], eq); days[d][2] = min(days[d][2], eq); days[d][3] = eq
    return [(d, v[0], v[1], v[2], v[3], 0) for d, v in sorted(days.items())]


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
    """Buy/sell marks parsed from run.log ORDER lines (the reliable source). Fail-safe → []."""
    import datetime
    now = time.time()
    if _LCACHE["mk"] is not None and now - _LCACHE["mk_t"] < 60:
        return _LCACHE["mk"]
    out = []
    try:
        with open(RUN_LOG, errors="replace") as fh:
            for i, line in enumerate(fh):
                m = _ORD_RE.match(line)
                if not m:
                    continue
                t = int(datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=datetime.timezone.utc).timestamp())
                side, sym = m.group(3), m.group(4)
                out.append({"id": f"ord_{i}", "time": t,
                            "color": "#26a69a" if side == "buy" else "#ef5350",
                            "text": f"{side.upper()} {sym}", "label": side[0].upper()})
    except Exception as e:
        logger.debug("trade_marks: %s", e)
    out = out[-500:]
    _LCACHE["mk"], _LCACHE["mk_t"] = out, now
    return out
