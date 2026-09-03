#!/usr/bin/env python3
"""
alpha_discovery/price_feature_builder.py — realtime price/quote features.

Transforms normalized Alpha Vantage realtime intraday bars + global quote
(from data_sources.alpha_vantage_client) into schema-controlled feature columns
for the alpha discovery layer and RL state. Pure functions; neutral on missing
data; never raises.

Feature family: realtime_price
    realtime_last_price
    realtime_quote_change_pct
    realtime_quote_momentum                  (short EMA vs long EMA of closes)
    realtime_intraday_vwap_distance          (last close vs rolling VWAP, %)
    realtime_intraday_volume_acceleration    (recent vs prior volume rate)
    realtime_intraday_range_pct
    realtime_price_available
"""

from __future__ import annotations

from typing import Any, Optional

_FAMILY = "realtime_price"


def neutral_price_features() -> dict[str, Any]:
    return {
        "realtime_last_price": None,
        "realtime_quote_change_pct": 0.0,
        "realtime_quote_momentum": 0.0,
        "realtime_intraday_vwap_distance": 0.0,
        "realtime_intraday_volume_acceleration": 0.0,
        "realtime_intraday_range_pct": 0.0,
        "realtime_price_available": 0,
    }


def _ema(values: list[float], span: int) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    alpha = 2.0 / (span + 1.0)
    ema = vals[0]
    for v in vals[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def build_price_features(bars: list[dict[str, Any]],
                         quote: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build realtime price features from intraday bars (oldest-first) + quote."""
    bars = [b for b in (bars or []) if b.get("close") is not None]
    if not bars and not quote:
        return neutral_price_features()

    feats = neutral_price_features()
    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars if b.get("high") is not None]
    lows = [float(b["low"]) for b in bars if b.get("low") is not None]
    vols = [float(b["volume"]) for b in bars if b.get("volume") is not None]

    last = None
    if quote and quote.get("price") is not None:
        last = float(quote["price"])
    elif closes:
        last = closes[-1]
    feats["realtime_last_price"] = last
    feats["realtime_price_available"] = 1 if (closes or last is not None) else 0

    if quote and quote.get("change_percent") is not None:
        feats["realtime_quote_change_pct"] = round(float(quote["change_percent"]), 4)
    elif len(closes) >= 2 and closes[-2] != 0:
        feats["realtime_quote_change_pct"] = round((closes[-1] / closes[-2] - 1.0) * 100.0, 4)

    # Momentum: short EMA vs long EMA of closes, normalized.
    if len(closes) >= 5:
        ema_fast = _ema(closes, 5)
        ema_slow = _ema(closes, min(len(closes), 20))
        if ema_fast is not None and ema_slow not in (None, 0):
            feats["realtime_quote_momentum"] = round((ema_fast / ema_slow - 1.0), 5)

    # Rolling VWAP distance.
    if closes and vols and len(vols) == len(closes):
        tp = []
        for b in bars:
            h = b.get("high"); l = b.get("low"); c = b.get("close")
            if None in (h, l, c):
                tp.append(float(c))
            else:
                tp.append((float(h) + float(l) + float(c)) / 3.0)
        denom = sum(vols)
        if denom > 0:
            vwap = sum(p * v for p, v in zip(tp, vols)) / denom
            ref = last if last is not None else closes[-1]
            if vwap > 0:
                feats["realtime_intraday_vwap_distance"] = round((ref / vwap - 1.0) * 100.0, 4)

    # Volume acceleration: last 3 bars vs prior 3 bars.
    if len(vols) >= 6:
        recent = sum(vols[-3:]) / 3.0
        prior = sum(vols[-6:-3]) / 3.0
        if prior > 0:
            feats["realtime_intraday_volume_acceleration"] = round((recent / prior - 1.0), 4)

    # Intraday range.
    if highs and lows and last:
        hi = max(highs); lo = min(lows)
        if last > 0:
            feats["realtime_intraday_range_pct"] = round((hi - lo) / last * 100.0, 4)

    return feats
