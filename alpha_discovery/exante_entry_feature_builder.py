#!/usr/bin/env python3
"""
alpha_discovery/exante_entry_feature_builder.py — compute the VALIDATED ex-ante
intraday mean-reversion features at decision time, from a recent 1-minute bar window.

These are the features that survived the leakage-resistant validation in
exante_feature_research.py (per-symbol-day OOS IC vs forward returns, FDR-corrected,
leakage-tripwire-screened): a coherent SHORT-HORIZON MEAN-REVERSION complex whose OOS
ICs are ~-0.03..-0.05 at the 5–15 minute horizon (strongest at 5m, decaying by 30m).

Leak-free BY CONSTRUCTION: every value is computed only from bars at or before the
decision bar t (no .shift(-k), no resampled timeframe that could contain future minutes).

Validated core (negative IC = extension predicts reversal):
    dist_sma_5, dist_sma_10, dist_sma_15   (price vs short moving averages)
    dist_vwap                              (price vs session VWAP)
    macd_hist                              (momentum exhaustion)
    ret_1m, ret_3m, ret_5m, ret_10m, log_ret_1m   (short-horizon trailing reversal)
    breakout_15                            (fade false 15-bar breakouts)

Composite:
    exante_reversion_score in [-1, 1] — high = oversold/extended-down (bounce expected),
    low = overbought/extended-up (pullback expected). For a LONG entry, prefer HIGH.

HORIZON NOTE: these predict 5–15 min UNDERLYING moves. They fit intraday / scalping entry
agents. They are NOT validated for multi-day options direction (alpha_entry) — see
reports/exante_entry_features.md. Advisory; neutral on insufficient data; never raises.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

EXANTE_CORE_FEATURES = [
    "dist_sma_5", "dist_sma_10", "dist_sma_15", "dist_vwap", "macd_hist",
    "ret_1m", "ret_3m", "ret_5m", "ret_10m", "log_ret_1m", "breakout_15",
]
# Extension features whose NEGATIVE z-score = oversold = bullish-reversion (for the composite).
_EXTENSION_FOR_COMPOSITE = ["dist_sma_5", "dist_sma_10", "dist_sma_15", "dist_vwap",
                            "macd_hist", "ret_5m", "ret_10m"]


def neutral_exante_features() -> dict[str, Any]:
    out = {f: 0.0 for f in EXANTE_CORE_FEATURES}
    out["exante_reversion_score"] = 0.0
    out["exante_available"] = 0
    return out


def _sma(closes: Sequence[float], n: int) -> Optional[float]:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _ema(vals: Sequence[float], span: int) -> Optional[float]:
    if not vals:
        return None
    k = 2.0 / (span + 1.0)
    e = vals[0]
    for v in vals[1:]:
        e = k * v + (1 - k) * e
    return e


def _macd_hist(closes: Sequence[float]) -> Optional[float]:
    if len(closes) < 35:
        return None
    ema12 = _ema(closes, 12); ema26 = _ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None
    # signal = EMA9 of the MACD line; approximate with the MACD line series tail
    macd_line = []
    for i in range(26, len(closes) + 1):
        w = closes[:i]
        e12 = _ema(w, 12); e26 = _ema(w, 26)
        if e12 is not None and e26 is not None:
            macd_line.append(e12 - e26)
    if len(macd_line) < 9:
        return None
    signal = _ema(macd_line, 9)
    return float(macd_line[-1] - signal) if signal is not None else None


def build_exante_features(bars: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute the validated ex-ante features from a window of 1m bars (oldest→newest).

    ``bars``: list of dicts with at least close, high, low, volume (and the LAST bar is the
    decision bar t). Needs ~35+ bars for the full set; degrades to neutral otherwise.
    """
    bars = [b for b in (bars or []) if b.get("close") is not None]
    if len(bars) < 16:
        return neutral_exante_features()
    closes = [float(b["close"]) for b in bars]
    highs = [float(b.get("high", b["close"])) for b in bars]
    lows = [float(b.get("low", b["close"])) for b in bars]
    vols = [float(b.get("volume", 0.0)) for b in bars]
    c = closes[-1]
    out = neutral_exante_features()

    def dist_sma(n):
        s = _sma(closes, n)
        return (c / s - 1.0) if s and s > 0 else None
    for n in (5, 10, 15):
        v = dist_sma(n)
        if v is not None:
            out[f"dist_sma_{n}"] = round(v, 6)

    # session VWAP (typical price * volume) over the available window
    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes))]
    vsum = sum(vols)
    if vsum > 0:
        vwap = sum(tp[i] * vols[i] for i in range(len(closes))) / vsum
        if vwap > 0:
            out["dist_vwap"] = round(c / vwap - 1.0, 6)

    out["ret_1m"] = round(c / closes[-2] - 1.0, 6) if closes[-2] else 0.0
    out["log_ret_1m"] = round(math.log(c / closes[-2]), 6) if closes[-2] and c > 0 else 0.0
    if len(closes) > 4 and closes[-4]:
        out["ret_3m"] = round(c / closes[-4] - 1.0, 6)
    if len(closes) > 6 and closes[-6]:
        out["ret_5m"] = round(c / closes[-6] - 1.0, 6)
    if len(closes) > 11 and closes[-11]:
        out["ret_10m"] = round(c / closes[-11] - 1.0, 6)

    mh = _macd_hist(closes)
    if mh is not None:
        out["macd_hist"] = round(mh, 6)

    prior_high_15 = max(highs[-16:-1]) if len(highs) >= 16 else None
    out["breakout_15"] = int(c > prior_high_15) if prior_high_15 is not None else 0

    out["exante_reversion_score"] = round(_reversion_composite(out), 6)
    out["exante_available"] = 1
    return out


def _reversion_composite(feats: dict[str, Any]) -> float:
    """Mean of the negated extension features, squashed to [-1,1]. High = oversold (bullish)."""
    vals = [feats.get(f, 0.0) for f in _EXTENSION_FOR_COMPOSITE]
    if not any(vals):
        return 0.0
    # scale each extension into a comparable band, negate (extension -> reversion), average, tanh
    scaled = []
    for f, v in zip(_EXTENSION_FOR_COMPOSITE, vals):
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        scaled.append(-x * (5.0 if f.startswith("dist_") or f.startswith("ret_") else 1.0))
    if not scaled:
        return 0.0
    return math.tanh(sum(scaled) / len(scaled))
