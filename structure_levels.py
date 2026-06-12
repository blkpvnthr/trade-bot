#!/usr/bin/env python3
"""structure_levels.py — pre-trade market-structure levels for the underlying.

Computes, for a symbol, everything the system should know BEFORE an options entry:
  30m opening range (ORB) hi/lo · supply/demand zones · prev-day hi/lo · 6m hi/lo ·
  30D hi/lo · today hi/lo · POC/VAH/VAL (volume profile) · session VWAP ·
  price>SMA200 · price>VWAP  -> then derives underlying entry / TP / stop.

Standalone (Alpaca data, IEX feed). Usage:
  python structure_levels.py QQQ SPY IWM            # print levels + entry/TP plan
  python structure_levels.py QQQ --json             # JSON for the trade planner
"""
from __future__ import annotations
import os, sys, json, argparse
from datetime import datetime, date, timedelta, time as dtime, timezone
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
from dotenv import load_dotenv
load_dotenv("/Users/blkpvnthr/Desktop/trader/.env")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York"); UTC = timezone.utc
AK = os.getenv("APCA_API_KEY_ID"); AS = os.getenv("APCA_API_SECRET_KEY")


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(AK, AS)


def _minute_bars(symbol, days=6):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    end = datetime.now(UTC); start = end - timedelta(days=days)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed=DataFeed.IEX)
    raw = _client().get_stock_bars(req).df
    if raw is None or len(raw) == 0: return pd.DataFrame()
    df = raw.reset_index()
    df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_convert(ET)
    return df.between_time("09:30", "16:00")


def _daily_bars(symbol, days=400):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    end = datetime.now(UTC); start = end - timedelta(days=days)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=start, end=end, feed=DataFeed.IEX)
    raw = _client().get_stock_bars(req).df
    if raw is None or len(raw) == 0: return pd.DataFrame()
    df = raw.reset_index()
    df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def _vp(bars):
    """POC/VAH/VAL via the MarketProfile library (proper TPO/volume profile); fallback to histogram."""
    if len(bars) == 0:
        return None, None, None, None
    try:
        from market_profile import MarketProfile
        d = bars.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]
        mp = MarketProfile(d)
        sl = mp[d.index.min():d.index.max()]
        val, vah = sl.value_area
        bt = getattr(sl, "balanced_target", None)
        return float(sl.poc_price), float(vah), float(val), (float(bt) if bt else None)
    except Exception:
        poc, vah, val = _volume_profile(bars)
        return poc, vah, val, None


def _volume_profile(bars, bins=60):
    """Contiguous 70% value area around POC from minute bars. Returns (poc, vah, val)."""
    if len(bars) == 0: return None, None, None
    lo, hi = float(bars["low"].min()), float(bars["high"].max())
    if hi <= lo: return lo, hi, lo
    import numpy as np
    edges = np.linspace(lo, hi, bins + 1); centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    idx = np.clip(np.searchsorted(edges, tp, side="right") - 1, 0, bins - 1)
    for i, v in zip(idx, bars["volume"].values): vol[i] += v
    total = vol.sum()
    if total <= 0: return float(centers[len(centers)//2]), hi, lo
    poc_i = int(vol.argmax()); lo_i = hi_i = poc_i; acc = vol[poc_i]
    while acc < 0.70 * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down: hi_i += 1; acc += vol[hi_i]
        else: lo_i -= 1; acc += vol[lo_i]
    return float(centers[poc_i]), float(edges[hi_i + 1]), float(edges[lo_i])


def _session_vwap(today_bars):
    if len(today_bars) == 0: return None
    tp = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
    return float((tp * today_bars["volume"]).sum() / max(today_bars["volume"].sum(), 1e-9))


def _swing_zones(daily, window=5, n=3):
    """Pivot highs (supply) / lows (demand) via simple fractals; nearest n each side."""
    h = daily["high"].values; l = daily["low"].values; sup = []; dem = []
    for i in range(window, len(daily) - window):
        if h[i] == max(h[i - window:i + window + 1]): sup.append(round(float(h[i]), 2))
        if l[i] == min(l[i - window:i + window + 1]): dem.append(round(float(l[i]), 2))
    return sorted(set(sup))[-12:], sorted(set(dem))[:12]


def levels(symbol):
    mins = _minute_bars(symbol); daily = _daily_bars(symbol)
    if len(daily) == 0: return {"symbol": symbol, "error": "no daily data"}
    sessions = sorted(set(mins.index.date)) if len(mins) else []
    today = sessions[-1] if sessions else None
    tb = mins[mins.index.date == today] if today else mins.iloc[0:0]
    orb = tb.between_time("09:30", "10:00")
    px = float(daily["close"].iloc[-1])
    if len(tb): px = float(tb["close"].iloc[-1])
    sma200 = float(daily["close"].tail(200).mean()) if len(daily) >= 200 else None
    poc, vah, val, balanced_target = _vp(tb if len(tb) else mins)
    vwap = _session_vwap(tb)
    sup, dem = _swing_zones(daily)
    d = daily.tail(126); d30 = daily.tail(30)
    L = {
        "symbol": symbol, "session": str(today), "price": round(px, 2),
        "orb30_high": round(float(orb["high"].max()), 2) if len(orb) else None,
        "orb30_low": round(float(orb["low"].min()), 2) if len(orb) else None,
        "today_high": round(float(tb["high"].max()), 2) if len(tb) else None,
        "today_low": round(float(tb["low"].min()), 2) if len(tb) else None,
        "prev_day_high": round(float(daily["high"].iloc[-2]), 2) if len(daily) >= 2 else None,
        "prev_day_low": round(float(daily["low"].iloc[-2]), 2) if len(daily) >= 2 else None,
        "high_30d": round(float(d30["high"].max()), 2), "low_30d": round(float(d30["low"].min()), 2),
        "high_6m": round(float(d["high"].max()), 2), "low_6m": round(float(d["low"].min()), 2),
        "poc": round(poc, 2) if poc else None, "vah": round(vah, 2) if vah else None,
        "val": round(val, 2) if val else None, "vwap": round(vwap, 2) if vwap else None,
        "balanced_target": round(balanced_target, 2) if balanced_target else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "price_gt_sma200": (px > sma200) if sma200 else None,
        "price_gt_vwap": (px > vwap) if vwap else None,
        "supply_zones": sup, "demand_zones": dem,
    }
    L["plan"] = _entry_tp(L)
    return L


def _entry_tp(L):
    """Derive underlying entry / TP / stop from the structure (long-bias if trend up)."""
    px = L["price"]
    up = bool(L.get("price_gt_sma200")) and bool(L.get("price_gt_vwap"))
    dn = (L.get("price_gt_sma200") is False) and (L.get("price_gt_vwap") is False)
    def above(*vals): return sorted(v for v in vals if v and v > px)
    def below(*vals): return sorted((v for v in vals if v and v < px), reverse=True)
    res = above(L["orb30_high"], L["prev_day_high"], L["today_high"], L["vah"], L["high_30d"], L["high_6m"], *L["supply_zones"])
    sup = below(L["orb30_low"], L["prev_day_low"], L["val"], L["vwap"], L["low_30d"], *L["demand_zones"])
    if up:
        bias = "long (call)"
        entry = res[0] if res else px            # breakout over nearest resistance (ORB/PDH)
        tp = res[1] if len(res) > 1 else (L["high_6m"] or round(entry * 1.03, 2))
        stop = (L["vwap"] or (sup[0] if sup else round(entry * 0.985, 2)))
    elif dn:
        bias = "short (put)"
        entry = sup[0] if sup else px            # breakdown under nearest support
        tp = sup[1] if len(sup) > 1 else (L["low_6m"] or round(entry * 0.97, 2))
        stop = (L["vwap"] or (res[0] if res else round(entry * 1.015, 2)))
    else:
        return {"bias": "no-trade (mixed: price not aligned vs SMA200 & VWAP)",
                "nearest_resistance": res[:3], "nearest_support": sup[:3]}
    risk = abs(entry - stop); reward = abs(tp - entry)
    return {"bias": bias, "entry": round(entry, 2), "tp": round(tp, 2), "stop": round(stop, 2),
            "rr": round(reward / risk, 2) if risk > 0 else None,
            "nearest_resistance": res[:3], "nearest_support": sup[:3]}


def gate(symbol_or_L, min_rr=1.5):
    """Pre-trade gate: GO only if structure aligns (price vs SMA200 & VWAP) AND R:R >= min_rr.
    Returns a clean decision dict with entry/TP/stop and explicit reasons."""
    L = symbol_or_L if isinstance(symbol_or_L, dict) else levels(symbol_or_L)
    if L.get("error"):
        return {"symbol": L.get("symbol"), "go": False, "reasons": [L["error"]]}
    p = L["plan"]; reasons = []
    aligned = "entry" in p
    if not aligned:
        reasons.append(p["bias"])
    rr = p.get("rr")
    rr_ok = aligned and rr is not None and rr >= min_rr
    if aligned and not rr_ok:
        reasons.append(f"R:R {rr} < {min_rr} min")
    go = bool(aligned and rr_ok)
    if go:
        reasons.append(f"{p['bias']} aligned (price>SMA200 & >VWAP), R:R {rr}")
    return {"symbol": L["symbol"], "session": L["session"], "price": L["price"], "go": go,
            "bias": p["bias"], "entry": p.get("entry"), "tp": p.get("tp"), "stop": p.get("stop"),
            "rr": rr, "min_rr": min_rr, "reasons": reasons}


def _print(L):
    if L.get("error"): print(f"{L['symbol']}: {L['error']}"); return
    print(f"\n=== {L['symbol']}  (session {L['session']})  price ${L['price']} ===")
    rows = [("ORB30", L["orb30_high"], L["orb30_low"]), ("today", L["today_high"], L["today_low"]),
            ("prev day", L["prev_day_high"], L["prev_day_low"]), ("30D", L["high_30d"], L["low_30d"]),
            ("6M", L["high_6m"], L["low_6m"])]
    for nm, hi, lo in rows: print(f"  {nm:<9} high {hi}   low {lo}")
    print(f"  VWAP {L['vwap']}   POC {L['poc']}   VAH {L['vah']}   VAL {L['val']}   balanced_target {L['balanced_target']}")
    print(f"  SMA200 {L['sma200']}   price>SMA200 {L['price_gt_sma200']}   price>VWAP {L['price_gt_vwap']}")
    print(f"  supply(res) {L['supply_zones'][-4:]}   demand(sup) {L['demand_zones'][:4]}")
    p = L["plan"]
    print(f"  PLAN: {p['bias']}")
    if "entry" in p:
        print(f"        entry ${p['entry']}  TP ${p['tp']}  stop ${p['stop']}  R:R {p['rr']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--gate", action="store_true", help="print GO/NO-GO pre-trade gate")
    ap.add_argument("--min-rr", type=float, default=1.5)
    a = ap.parse_args()
    out = [levels(s) for s in a.symbols]
    if a.gate:
        decs = [gate(L, a.min_rr) for L in out]
        if a.json:
            print(json.dumps(decs, indent=2))
        else:
            for d in decs:
                tag = "✅ GO  " if d["go"] else "⛔ NO-GO"
                line = f"{tag} {d['symbol']:<5} ${d.get('price')}"
                if d.get("go"):
                    line += f"  {d['bias']}  entry ${d['entry']} TP ${d['tp']} stop ${d['stop']} R:R {d['rr']}"
                line += f"  — {'; '.join(d['reasons'])}"
                print(line)
    elif a.json:
        print(json.dumps(out, indent=2))
    else:
        for L in out: _print(L)
