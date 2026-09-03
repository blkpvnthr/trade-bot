#!/usr/bin/env python3
"""
options_sentiment_av.py

Advisory options-sentiment decision layer over Alpha Vantage realtime data.

This module is the *decision overlay* for the options booster. The raw network
fetching, API-key handling, realtime entitlement, request logging, caching and
retry/timeout all live in the unified client
``data_sources/alpha_vantage_client.AlphaVantageClient`` — this module delegates
to it and adds only what is specific to options decision-making:

    * normalize_put_call_ratio()  — PCR -> directional read + contrarian flags
    * normalize_volume_oi_ratio() — volume/OI -> participation amplifier
    * get_symbol_sentiment()      — the advisory bundle the booster consumes
    * evaluate_setup_sentiment()  — side-aware adjustment for call/put/naked_call

Relationship to siblings
-------------------------
    data_sources/alpha_vantage_client.py  -> unified raw AV client (PCR/VOI/etc.)
    alpha_discovery/options_feature_builder.py -> PCR/VOI features for the RL
                                                  pipeline (gated, default-off)
    options_sentiment_av.py (this file)   -> LIVE advisory overlay that re-ranks
                                             option contracts in options_booster.

Hard rule: these signals may raise or lower confidence and re-rank contracts,
but are NEVER execution-authoritative on their own and NEVER override hard risk
gates (max exposure, position size, stop loss, drawdown limits, liquidity/spread
checks). See docs/options_sentiment_decisioning.md.

Alpha Vantage endpoints (fetched via the unified client)
-------------------------------------------------------
    REALTIME_PUT_CALL_RATIO              -> chain + per-expiration PCR
    REALTIME_VOLUME_OPEN_INTEREST_RATIO  -> chain + per-side (call/put) volume/OI

Configuration
-------------
API key + entitlement + HTTP/cache config are owned by AlphaVantageClient
(ALPHA_VANTAGE_API_KEY, with ALPHA_VANTAGE_API accepted as an alias; never
hardcoded). This module adds the interpretation thresholds below.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from config.env_loader import load_instance_env

    load_instance_env()
except Exception:  # pragma: no cover - env loader optional in isolated tests
    pass

from data_sources.alpha_vantage_client import (
    AlphaVantageClient,
    get_default_av_client,
)

logger = logging.getLogger(__name__)

# Endpoint names (kept for callers/tests that reference them symbolically).
PCR_FUNCTION = "REALTIME_PUT_CALL_RATIO"
VOI_FUNCTION = "REALTIME_VOLUME_OPEN_INTEREST_RATIO"


# =============================================================================
# Small helpers
# =============================================================================

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_float(value: Any) -> Optional[float]:
    """Lenient float parse. Returns None for missing / NaN / non-numeric values."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Config (interpretation thresholds only — HTTP/cache config is AlphaVantageConfig)
# =============================================================================

@dataclass
class SentimentConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("OPTIONS_SENTIMENT_AV_ENABLED", True))

    # Put-call ratio interpretation bands.
    pcr_bullish_max: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_PCR_BULLISH_MAX", 0.60))
    pcr_neutral_max: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_PCR_NEUTRAL_MAX", 1.00))
    pcr_extreme_low: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_PCR_EXTREME_LOW", 0.30))
    pcr_extreme_high: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_PCR_EXTREME_HIGH", 1.80))

    # Volume/open-interest participation bands.
    voi_weak: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_VOI_WEAK", 0.10))
    voi_strong: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_VOI_STRONG", 0.50))

    # Advisory adjustment weights (bounded — sentiment can only nudge, never dominate).
    direction_weight: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_DIRECTION_WEIGHT", 0.25))
    participation_swing: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_PARTICIPATION_SWING", 0.12))
    contrarian_damp: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_CONTRARIAN_DAMP", 0.50))
    multiplier_floor: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_MULT_FLOOR", 0.60))
    multiplier_ceiling: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_MULT_CEILING", 1.40))

    # Low-confirmation handling (advisory; default off so behavior is unchanged unless opted-in).
    skip_low_confirmation: bool = field(default_factory=lambda: _env_bool("OPTIONS_SENTIMENT_SKIP_LOW_CONFIRMATION", False))
    low_participation_threshold: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_LOW_PARTICIPATION", 0.20))
    naked_call_min_participation: float = field(default_factory=lambda: _env_float("OPTIONS_SENTIMENT_NAKED_MIN_PARTICIPATION", 0.45))


# =============================================================================
# Neutral defaults — returned whenever data is unavailable.
# =============================================================================

def neutral_sentiment(symbol: str = "", reason: str = "unavailable") -> dict[str, Any]:
    return {
        "symbol": (symbol or "").upper(),
        "available": False,
        "source": "neutral_fallback",
        "reason": reason,
        "fetched_at_utc": now_utc().isoformat(),
        "put_call_ratio_chain": None,
        "put_call_ratio_expiration": {},
        "volume_oi_ratio_chain": None,
        "volume_oi_ratio_call": None,
        "volume_oi_ratio_put": None,
        "volume_oi_ratio_expiration": {},
        # Normalized advisory fields.
        "put_call_ratio": None,
        "sentiment_directional_score": 0.0,
        "contrarian_warning": False,
        "put_call_confidence_modifier": 1.0,
        "volume_oi_ratio": None,
        "participation_score": 0.5,
        "volume_oi_confidence_modifier": 1.0,
        "options_sentiment_confidence": 0.5,
    }


# =============================================================================
# Normalizers (pure functions; safe to unit test in isolation)
# =============================================================================

def normalize_put_call_ratio(value: Any, config: Optional[SentimentConfig] = None) -> dict[str, Any]:
    """Map a put-call ratio to an advisory directional read.

    Returns:
        directional_score   : float in [-1, 1] (+ = bullish call demand, - = bearish put demand)
        contrarian_warning  : bool (extreme readings flagged as possible reversal)
        confidence_modifier : float ~[0.85, 1.15] conviction scalar (dampened on contrarian)
        band                : human-readable band label
    """
    cfg = config or SentimentConfig()
    pcr = to_float(value)
    if pcr is None or pcr < 0:
        return {
            "directional_score": 0.0,
            "contrarian_warning": False,
            "confidence_modifier": 1.0,
            "band": "unavailable",
        }

    extreme_low = cfg.pcr_extreme_low
    bullish_max = cfg.pcr_bullish_max
    neutral_max = cfg.pcr_neutral_max
    extreme_high = cfg.pcr_extreme_high

    contrarian_warning = False

    if pcr <= extreme_low:
        # Extremely low PCR: heavy call skew can mark crowded/contrarian-bearish risk.
        contrarian_warning = True
        directional = -0.25
        band = "extreme_low_contrarian_bearish"
    elif pcr <= bullish_max:
        # Bullish call demand: lower ratio -> stronger bullish read.
        frac = (pcr - extreme_low) / max(bullish_max - extreme_low, 1e-9)
        directional = 1.0 - 0.7 * frac  # +1.0 .. +0.3
        band = "bullish_call_demand"
    elif pcr <= neutral_max:
        # Neutral / mixed.
        frac = (pcr - bullish_max) / max(neutral_max - bullish_max, 1e-9)
        directional = 0.2 - 0.4 * frac  # +0.2 .. -0.2
        band = "neutral_mixed"
    elif pcr < extreme_high:
        # Bearish put demand: higher ratio -> stronger bearish read.
        frac = (pcr - neutral_max) / max(extreme_high - neutral_max, 1e-9)
        directional = -0.3 - 0.7 * frac  # -0.3 .. -1.0
        band = "bearish_put_demand"
    else:
        # Extremely high PCR: capitulation / possible contrarian-bullish.
        contrarian_warning = True
        directional = 0.25
        band = "extreme_high_contrarian_bullish"

    directional = _clip(directional, -1.0, 1.0)
    # Conviction scalar from magnitude, dampened when the read is contrarian/uncertain.
    confidence_modifier = 1.0 + 0.15 * abs(directional)
    if contrarian_warning:
        confidence_modifier = 1.0 + (confidence_modifier - 1.0) * cfg.contrarian_damp * 0.5
        confidence_modifier = min(confidence_modifier, 0.95)

    return {
        "directional_score": float(directional),
        "contrarian_warning": bool(contrarian_warning),
        "confidence_modifier": float(_clip(confidence_modifier, 0.80, 1.20)),
        "band": band,
    }


def normalize_volume_oi_ratio(value: Any, config: Optional[SentimentConfig] = None) -> dict[str, Any]:
    """Map a volume/open-interest ratio to a participation amplifier.

    High volume/OI = strong fresh short-term participation. This is a
    liquidity/speculation/confirmation amplifier, NOT a directional signal.

    Returns:
        participation_score : float in [0, 1] (0.5 neutral when unavailable)
        confidence_modifier : float ~[0.85, 1.15]
        band                : human-readable label
    """
    cfg = config or SentimentConfig()
    voi = to_float(value)
    if voi is None or voi < 0:
        return {
            "participation_score": 0.5,
            "confidence_modifier": 1.0,
            "band": "unavailable",
        }

    weak = cfg.voi_weak
    strong = cfg.voi_strong
    participation = _clip((voi - weak) / max(strong - weak, 1e-9), 0.0, 1.0)

    if participation >= 0.75:
        band = "strong_participation"
    elif participation <= 0.25:
        band = "weak_participation"
    else:
        band = "moderate_participation"

    confidence_modifier = 0.85 + 0.30 * participation  # 0.85 .. 1.15
    return {
        "participation_score": float(participation),
        "confidence_modifier": float(_clip(confidence_modifier, 0.80, 1.20)),
        "band": band,
    }


# =============================================================================
# Client — thin decision-layer wrapper around the unified AlphaVantageClient.
# =============================================================================

class OptionsSentimentClient:
    """Builds the advisory sentiment bundle the options booster consumes.

    Delegates all network/cache/entitlement work to AlphaVantageClient and adds
    the option-decision normalization on top. Advisory-safe: any failure
    degrades to neutral values (no behavior change), never raises.
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        av_client: Optional[AlphaVantageClient] = None,
    ) -> None:
        self.config = config or SentimentConfig()
        self.av = av_client if av_client is not None else get_default_av_client()

    @property
    def api_key(self) -> str:
        return getattr(self.av, "api_key", "") or ""

    def get_symbol_sentiment(self, symbol: str) -> dict[str, Any]:
        """Return the full advisory sentiment bundle for an underlying symbol.

        Always returns a dict; degrades to neutral_sentiment() on any failure.
        """
        symbol = (symbol or "").upper().strip()
        if not symbol:
            return neutral_sentiment(symbol, reason="empty_symbol")
        if not self.config.enabled:
            return neutral_sentiment(symbol, reason="disabled")

        try:
            pcr = self.av.get_put_call_ratio(symbol)
            voi = self.av.get_volume_oi_ratio(symbol)
        except Exception as exc:  # pragma: no cover - unified client is advisory-safe
            logger.warning("AV sentiment fetch failed for %s: %s", symbol, exc)
            return neutral_sentiment(symbol, reason="fetch_error")

        pcr_available = bool(pcr.get("available"))
        voi_available = bool(voi.get("available"))
        if not pcr_available and not voi_available:
            return neutral_sentiment(symbol, reason="no_data")

        pcr_chain = pcr.get("put_call_ratio_chain")
        pcr_by_exp = pcr.get("put_call_ratio_expiration") or {}
        voi_chain = voi.get("volume_oi_ratio_chain")
        voi_call = voi.get("volume_oi_ratio_call")
        voi_put = voi.get("volume_oi_ratio_put")

        pcr_norm = normalize_put_call_ratio(pcr_chain, self.config)
        voi_norm = normalize_volume_oi_ratio(voi_chain, self.config)

        # Overall confidence in the sentiment read itself (0..1, 0.5 = neutral).
        directional = pcr_norm["directional_score"]
        participation = voi_norm["participation_score"]
        base_conf = 0.5 + 0.4 * directional
        base_conf *= (0.75 + 0.5 * participation)
        if pcr_norm["contrarian_warning"]:
            base_conf = 0.5 + (base_conf - 0.5) * self.config.contrarian_damp
        options_sentiment_confidence = float(_clip(base_conf, 0.0, 1.0))

        return {
            "symbol": symbol,
            "available": True,
            "source": "alpha_vantage",
            "reason": "ok",
            "fetched_at_utc": now_utc().isoformat(),
            "put_call_ratio_chain": pcr_chain,
            "put_call_ratio_expiration": pcr_by_exp,
            "volume_oi_ratio_chain": voi_chain,
            "volume_oi_ratio_call": voi_call,
            "volume_oi_ratio_put": voi_put,
            # Per-expiration VOI is not provided by the unified client; callers
            # fall back to chain-level participation when a key is absent.
            "volume_oi_ratio_expiration": {},
            # Normalized advisory fields.
            "put_call_ratio": pcr_chain,
            "sentiment_directional_score": directional,
            "contrarian_warning": pcr_norm["contrarian_warning"],
            "put_call_band": pcr_norm["band"],
            "put_call_confidence_modifier": pcr_norm["confidence_modifier"],
            "volume_oi_ratio": voi_chain,
            "participation_score": participation,
            "volume_oi_band": voi_norm["band"],
            "volume_oi_confidence_modifier": voi_norm["confidence_modifier"],
            "options_sentiment_confidence": options_sentiment_confidence,
        }


# =============================================================================
# Side-aware advisory evaluation — used by the options decision path.
# =============================================================================

def evaluate_setup_sentiment(
    sentiment: dict[str, Any],
    side: str = "call",
    technical_bullish: Optional[bool] = None,
    expiration: Optional[str] = None,
    config: Optional[SentimentConfig] = None,
) -> dict[str, Any]:
    """Translate a sentiment bundle into a side-aware advisory adjustment.

    side: "call" | "put" | "naked_call".

    Returns a dict with:
        confidence_multiplier : bounded scalar applied to a contract score
        ranking_delta         : additive ranking nudge (small)
        skip_low_confirmation : advisory flag (never a hard risk gate)
        safety_warning        : str | None (e.g. sentiment vs technical conflict)
        final_options_sentiment_confidence : 0..1
        decision_impact       : human-readable explanation for journals
    """
    cfg = config or SentimentConfig()
    side = (side or "call").lower().strip()
    is_call_like = side in ("call", "naked_call")

    if not sentiment or not sentiment.get("available"):
        return {
            "confidence_multiplier": 1.0,
            "ranking_delta": 0.0,
            "skip_low_confirmation": False,
            "safety_warning": None,
            "final_options_sentiment_confidence": 0.5,
            "decision_impact": "neutral: options sentiment unavailable (no adjustment)",
        }

    directional = float(sentiment.get("sentiment_directional_score", 0.0))
    contrarian = bool(sentiment.get("contrarian_warning", False))
    participation = float(sentiment.get("participation_score", 0.5))

    # Prefer the per-expiration PCR when available, else the chain aggregate.
    pcr_used = sentiment.get("put_call_ratio_chain")
    if expiration:
        exp_map = sentiment.get("put_call_ratio_expiration") or {}
        if expiration in exp_map:
            exp_norm = normalize_put_call_ratio(exp_map[expiration], cfg)
            directional = float(exp_norm["directional_score"])
            contrarian = bool(exp_norm["contrarian_warning"])
            pcr_used = exp_map[expiration]
        voi_exp_map = sentiment.get("volume_oi_ratio_expiration") or {}
        if expiration in voi_exp_map:
            participation = float(normalize_volume_oi_ratio(voi_exp_map[expiration], cfg)["participation_score"])

    # Directional alignment with the trade side (+ favors the position, - opposes).
    aligned = directional if is_call_like else -directional

    direction_mult = 1.0 + cfg.direction_weight * aligned
    participation_mult = 1.0 + cfg.participation_swing * (2.0 * participation - 1.0)
    confidence_multiplier = direction_mult * participation_mult

    if contrarian:
        # Pull any boost/penalty toward neutral — extreme readings are uncertain.
        confidence_multiplier = 1.0 + (confidence_multiplier - 1.0) * cfg.contrarian_damp

    confidence_multiplier = float(_clip(confidence_multiplier, cfg.multiplier_floor, cfg.multiplier_ceiling))

    safety_warning: Optional[str] = None
    skip_low_confirmation = False

    # Sentiment-vs-technical conflict warnings.
    if technical_bullish is True and is_call_like and aligned <= -0.5 and not contrarian:
        safety_warning = (
            f"options sentiment bearish (PCR={pcr_used}) conflicts with bullish technical trend"
        )
    elif technical_bullish is False and side == "put" and aligned <= -0.5 and not contrarian:
        safety_warning = (
            f"options sentiment bullish (PCR={pcr_used}) conflicts with bearish technical trend"
        )

    # Low-participation = weak/illiquid confirmation; the move may lack new flow.
    if participation < cfg.low_participation_threshold and aligned <= 0.0:
        if cfg.skip_low_confirmation:
            skip_low_confirmation = True

    # Naked calls demand stronger participation and a non-hostile read; sentiment
    # ALONE can never justify a naked call (technical confirmation required).
    if side == "naked_call":
        if participation < cfg.naked_call_min_participation and cfg.skip_low_confirmation:
            skip_low_confirmation = True
        if technical_bullish is False or aligned < 0.0:
            warn = "naked_call: options sentiment / participation does not support bullish premise"
            safety_warning = f"{safety_warning}; {warn}" if safety_warning else warn

    final_conf = float(sentiment.get("options_sentiment_confidence", 0.5))
    if not is_call_like:
        # For puts the favorable confidence is the inverse read.
        final_conf = float(_clip(1.0 - final_conf, 0.0, 1.0)) if directional != 0 else 0.5

    ranking_delta = float(_clip((confidence_multiplier - 1.0) * 0.10, -0.05, 0.05))

    impact = (
        f"side={side} pcr={pcr_used} dir_score={directional:+.2f} "
        f"participation={participation:.2f} contrarian={contrarian} "
        f"-> conf_mult={confidence_multiplier:.3f}"
    )
    if safety_warning:
        impact += f" | WARNING: {safety_warning}"
    if skip_low_confirmation:
        impact += " | skip_low_confirmation"

    return {
        "confidence_multiplier": confidence_multiplier,
        "ranking_delta": ranking_delta,
        "skip_low_confirmation": bool(skip_low_confirmation),
        "safety_warning": safety_warning,
        "final_options_sentiment_confidence": final_conf,
        "decision_impact": impact,
    }


# =============================================================================
# Module-level singleton convenience
# =============================================================================

_DEFAULT_CLIENT: Optional[OptionsSentimentClient] = None


def get_default_client() -> OptionsSentimentClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OptionsSentimentClient()
    return _DEFAULT_CLIENT


def get_symbol_sentiment(symbol: str) -> dict[str, Any]:
    """Convenience wrapper using the process-wide default client."""
    return get_default_client().get_symbol_sentiment(symbol)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["IBM", "NVDA"]
    client = OptionsSentimentClient()
    print(f"API key present: {bool(client.api_key)} | enabled: {client.config.enabled}")
    for s in syms:
        bundle = client.get_symbol_sentiment(s)
        call_eval = evaluate_setup_sentiment(bundle, side="call", technical_bullish=True)
        print(json.dumps({"symbol": s, "sentiment": bundle, "call_eval": call_eval}, indent=2, default=str))
