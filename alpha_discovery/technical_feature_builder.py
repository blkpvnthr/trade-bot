#!/usr/bin/env python3
"""
alpha_discovery/technical_feature_builder.py — realtime technical features.

Normalizes Alpha Vantage realtime technical-indicator rows (SMA, EMA, RSI, MACD,
VWAP, BBANDS, ...) into feature columns and a single composite alignment score.

Each indicator's latest value is extracted and exposed under a stable column
name. The composite `realtime_technical_alignment_score` in [-1, 1] aggregates
the directional read of the available indicators (trend, momentum, mean-reversion
position) so the RL state gets one normalized signal even when only a subset of
indicators is available.

Pure functions; neutral on missing data; never raises.
"""

from __future__ import annotations

from typing import Any, Optional

_FAMILY = "realtime_technical"


def neutral_technical_features() -> dict[str, Any]:
    return {
        "realtime_sma": None,
        "realtime_ema": None,
        "realtime_rsi": None,
        "realtime_macd": None,
        "realtime_macd_signal": None,
        "realtime_macd_hist": None,
        "realtime_vwap": None,
        "realtime_bbands_upper": None,
        "realtime_bbands_lower": None,
        "realtime_bbands_middle": None,
        "realtime_bbands_pctb": None,
        "realtime_technical_alignment_score": 0.0,
        "realtime_technical_available": 0,
    }


def _latest(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    rows = rows or []
    return rows[-1] if rows else None


def _first_numeric(row: dict[str, Any], *substrings: str) -> Optional[float]:
    """Find the first numeric column whose name contains all given substrings."""
    for k, v in row.items():
        if k in ("symbol", "timestamp", "source", "entitlement"):
            continue
        kl = k.lower()
        if all(s in kl for s in substrings) and isinstance(v, (int, float)):
            return float(v)
    return None


def build_technical_features(
    indicators: dict[str, list[dict[str, Any]]],
    last_price: Optional[float] = None,
) -> dict[str, Any]:
    """Build realtime technical features.

    ``indicators`` maps indicator name (e.g. "SMA", "RSI", "MACD", "BBANDS",
    "VWAP", "EMA") to its normalized row list (oldest-first), as returned by
    AlphaVantageClient.get_technical_indicator.
    """
    feats = neutral_technical_features()
    if not indicators:
        return feats

    available = False
    components: list[float] = []  # each in [-1, 1]

    def latest_val(name: str, *subs: str) -> Optional[float]:
        rows = indicators.get(name) or indicators.get(name.upper()) or indicators.get(name.lower())
        row = _latest(rows or [])
        if not row:
            return None
        return _first_numeric(row, *subs)

    sma = latest_val("SMA", "sma")
    ema = latest_val("EMA", "ema")
    rsi = latest_val("RSI", "rsi")
    macd = latest_val("MACD", "macd")
    macd_signal = latest_val("MACD", "signal")
    macd_hist = latest_val("MACD", "hist")
    vwap = latest_val("VWAP", "vwap")
    bb_upper = latest_val("BBANDS", "upper")
    bb_lower = latest_val("BBANDS", "lower")
    bb_middle = latest_val("BBANDS", "middle")

    feats.update({
        "realtime_sma": sma, "realtime_ema": ema, "realtime_rsi": rsi,
        "realtime_macd": macd, "realtime_macd_signal": macd_signal, "realtime_macd_hist": macd_hist,
        "realtime_vwap": vwap, "realtime_bbands_upper": bb_upper,
        "realtime_bbands_lower": bb_lower, "realtime_bbands_middle": bb_middle,
    })

    ref = last_price if last_price is not None else (ema or sma)

    # RSI: map 30..70 band to a [-1, 1] momentum read.
    if rsi is not None:
        available = True
        components.append(max(-1.0, min(1.0, (rsi - 50.0) / 20.0)))

    # Trend: price vs SMA/EMA.
    if ref is not None and sma not in (None, 0):
        available = True
        components.append(max(-1.0, min(1.0, (ref / sma - 1.0) * 20.0)))
    if ema not in (None, 0) and sma not in (None, 0):
        available = True
        components.append(max(-1.0, min(1.0, (ema / sma - 1.0) * 20.0)))

    # MACD histogram sign.
    if macd_hist is not None:
        available = True
        components.append(max(-1.0, min(1.0, macd_hist)))
    elif macd is not None and macd_signal is not None:
        available = True
        components.append(max(-1.0, min(1.0, (macd - macd_signal))))

    # VWAP distance.
    if ref is not None and vwap not in (None, 0):
        available = True
        components.append(max(-1.0, min(1.0, (ref / vwap - 1.0) * 20.0)))

    # Bollinger %B -> position read (centered).
    if bb_upper is not None and bb_lower is not None and ref is not None and bb_upper > bb_lower:
        available = True
        pctb = (ref - bb_lower) / (bb_upper - bb_lower)
        feats["realtime_bbands_pctb"] = round(pctb, 4)
        components.append(max(-1.0, min(1.0, (pctb - 0.5) * 2.0)))

    feats["realtime_technical_available"] = 1 if available else 0
    if components:
        feats["realtime_technical_alignment_score"] = round(sum(components) / len(components), 4)
    return feats
