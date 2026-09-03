#!/usr/bin/env python3
"""
alpha_discovery/options_feature_builder.py — realtime options chain / flow /
sentiment features + realtime-data-driven contract selection.

Inputs (all from data_sources.alpha_vantage_client):
    chain      : normalized REALTIME_OPTIONS list (calls + puts)
    pcr        : get_put_call_ratio() dict
    voi        : get_volume_oi_ratio() dict

Feature families
----------------
realtime_options_sentiment:
    realtime_put_call_ratio_total, realtime_put_call_ratio_near_expiry,
    realtime_put_call_ratio_30d, put_call_ratio_signal,
    put_call_ratio_contrarian_signal, realtime_put_call_sentiment_score
realtime_options_flow:
    realtime_volume_oi_ratio_total, realtime_volume_oi_ratio_call,
    realtime_volume_oi_ratio_put, unusual_options_activity_score,
    options_participation_score, realtime_volume_oi_unusual_activity_score
realtime_options_chain (hidden alpha):
    realtime_option_iv_skew, realtime_option_delta_exposure,
    realtime_option_gamma_exposure, realtime_options_liquidity_score

Plus contract selection: score_contract / select_best_contract use the realtime
chain (spread, liquidity, volume, OI, IV, greeks, expiration, moneyness, vol/OI).

Pure functions; neutral on missing data; never raises.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def neutral_options_features() -> dict[str, Any]:
    return {
        # sentiment
        "realtime_put_call_ratio_total": None,
        "realtime_put_call_ratio_near_expiry": None,
        "realtime_put_call_ratio_30d": None,
        "put_call_ratio_signal": 0.0,
        "put_call_ratio_contrarian_signal": 0.0,
        "realtime_put_call_sentiment_score": 0.0,
        # flow
        "realtime_volume_oi_ratio_total": None,
        "realtime_volume_oi_ratio_call": None,
        "realtime_volume_oi_ratio_put": None,
        "unusual_options_activity_score": 0.0,
        "options_participation_score": 0.5,
        "realtime_volume_oi_unusual_activity_score": 0.0,
        # chain / hidden alpha
        "realtime_option_iv_skew": 0.0,
        "realtime_option_delta_exposure": 0.0,
        "realtime_option_gamma_exposure": 0.0,
        "realtime_options_liquidity_score": 0.0,
        "realtime_options_available": 0,
        # options_sentiment research family (neutral defaults; folded in so the
        # empty build still equals the neutral dict).
        **neutral_options_sentiment_features(),
    }


# =============================================================================
# Put-call ratio features
# =============================================================================

def _nearest_expiration_pcr(by_exp: dict[str, float]) -> Optional[float]:
    if not by_exp:
        return None
    try:
        items = sorted(by_exp.items(), key=lambda kv: kv[0])
        return items[0][1]
    except Exception:
        return None


def _pcr_30d(by_exp: dict[str, float], as_of: Optional[date] = None) -> Optional[float]:
    if not by_exp:
        return None
    ref = as_of or date.today()
    vals = []
    for exp, v in by_exp.items():
        try:
            d = date.fromisoformat(str(exp)[:10])
            if 0 <= (d - ref).days <= 30:
                vals.append(v)
        except Exception:
            continue
    return (sum(vals) / len(vals)) if vals else None


def build_pcr_features(pcr: Optional[dict[str, Any]], as_of: Optional[date] = None) -> dict[str, Any]:
    feats: dict[str, Any] = {
        "realtime_put_call_ratio_total": None,
        "realtime_put_call_ratio_near_expiry": None,
        "realtime_put_call_ratio_30d": None,
        "put_call_ratio_signal": 0.0,
        "put_call_ratio_contrarian_signal": 0.0,
        "realtime_put_call_sentiment_score": 0.0,
    }
    if not pcr or not pcr.get("available"):
        return feats
    total = pcr.get("put_call_ratio_chain")
    by_exp = pcr.get("put_call_ratio_expiration") or {}
    feats["realtime_put_call_ratio_total"] = total
    feats["realtime_put_call_ratio_near_expiry"] = _nearest_expiration_pcr(by_exp)
    feats["realtime_put_call_ratio_30d"] = _pcr_30d(by_exp, as_of)

    if total is not None:
        # signal: low PCR (call demand) -> bullish (+), high PCR -> bearish (-).
        feats["put_call_ratio_signal"] = round(_clip((0.8 - total) / 0.8, -1.0, 1.0), 4)
        # contrarian: extreme readings flip.
        if total <= 0.30:
            feats["put_call_ratio_contrarian_signal"] = -0.5
        elif total >= 1.80:
            feats["put_call_ratio_contrarian_signal"] = 0.5
        feats["realtime_put_call_sentiment_score"] = round(
            _clip(0.7 * feats["put_call_ratio_signal"] + 0.3 * feats["put_call_ratio_contrarian_signal"], -1.0, 1.0), 4)
    return feats


# =============================================================================
# Volume / open-interest flow features
# =============================================================================

def build_voi_features(voi: Optional[dict[str, Any]]) -> dict[str, Any]:
    feats: dict[str, Any] = {
        "realtime_volume_oi_ratio_total": None,
        "realtime_volume_oi_ratio_call": None,
        "realtime_volume_oi_ratio_put": None,
        "unusual_options_activity_score": 0.0,
        "options_participation_score": 0.5,
        "realtime_volume_oi_unusual_activity_score": 0.0,
    }
    if not voi or not voi.get("available"):
        return feats
    total = voi.get("volume_oi_ratio_chain")
    call_v = voi.get("volume_oi_ratio_call")
    put_v = voi.get("volume_oi_ratio_put")
    feats["realtime_volume_oi_ratio_total"] = total
    feats["realtime_volume_oi_ratio_call"] = call_v
    feats["realtime_volume_oi_ratio_put"] = put_v
    if total is not None:
        # Participation: vol/OI of 0.1 weak .. 0.5 strong -> [0,1].
        participation = _clip((total - 0.10) / 0.40, 0.0, 1.0)
        feats["options_participation_score"] = round(participation, 4)
        # Unusual activity: vol/OI well above 1.0 = fresh speculative flow.
        unusual = _clip((total - 0.5) / 1.5, 0.0, 1.0)
        feats["unusual_options_activity_score"] = round(unusual, 4)
        feats["realtime_volume_oi_unusual_activity_score"] = round(unusual, 4)
    return feats


# =============================================================================
# Chain-derived hidden-alpha features
# =============================================================================

def build_chain_features(chain: list[dict[str, Any]], spot: Optional[float] = None) -> dict[str, Any]:
    feats = {
        "realtime_option_iv_skew": 0.0,
        "realtime_option_delta_exposure": 0.0,
        "realtime_option_gamma_exposure": 0.0,
        "realtime_options_liquidity_score": 0.0,
    }
    chain = [c for c in (chain or []) if isinstance(c, dict)]
    if not chain:
        return feats

    calls = [c for c in chain if c.get("type") == "call"]
    puts = [c for c in chain if c.get("type") == "put"]

    # IV skew: mean put IV - mean call IV (positive = downside protection demand).
    call_ivs = [c["implied_volatility"] for c in calls if c.get("implied_volatility") is not None]
    put_ivs = [p["implied_volatility"] for p in puts if p.get("implied_volatility") is not None]
    if call_ivs and put_ivs:
        feats["realtime_option_iv_skew"] = round(
            (sum(put_ivs) / len(put_ivs)) - (sum(call_ivs) / len(call_ivs)), 5)

    # Net delta / gamma exposure weighted by open interest (dealer-flow proxy).
    delta_exp = 0.0
    gamma_exp = 0.0
    for c in chain:
        oi = c.get("open_interest") or 0
        if c.get("delta") is not None:
            delta_exp += c["delta"] * oi
        if c.get("gamma") is not None:
            gamma_exp += c["gamma"] * oi * (1 if c.get("type") == "call" else -1)
    # Normalize to a bounded read via tanh-like squash.
    feats["realtime_option_delta_exposure"] = round(_squash(delta_exp / 1e5), 5)
    feats["realtime_option_gamma_exposure"] = round(_squash(gamma_exp / 1e5), 5)

    # Liquidity score: mean over chain of (tight spread * volume depth * OI depth).
    liq_scores = [_contract_liquidity(c) for c in chain]
    liq_scores = [s for s in liq_scores if s is not None]
    if liq_scores:
        feats["realtime_options_liquidity_score"] = round(sum(liq_scores) / len(liq_scores), 4)
    return feats


def _squash(x: float) -> float:
    # cheap tanh without importing math at call sites everywhere
    import math
    return math.tanh(x)


def _contract_liquidity(c: dict[str, Any]) -> Optional[float]:
    bid = c.get("bid"); ask = c.get("ask")
    vol = c.get("volume") or 0
    oi = c.get("open_interest") or 0
    if bid is None or ask is None or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    spread_pct = (ask - bid) / mid
    spread_score = _clip(1.0 - spread_pct / 0.10, 0.0, 1.0)   # <=0% tight, >=10% wide
    vol_score = _clip(vol / 500.0, 0.0, 1.0)
    oi_score = _clip(oi / 2000.0, 0.0, 1.0)
    return 0.5 * spread_score + 0.25 * vol_score + 0.25 * oi_score


# =============================================================================
# options_sentiment family — hidden-alpha candidate factors + interactions
# =============================================================================
#
# A research-layer feature family derived from the same realtime options inputs
# (PCR / volume-OI / chain greeks+IV). These named factors are normalized,
# bounded composites intended for the alpha-discovery / latent-compression /
# RL-training pipeline — NOT for the live booster overlay (which stays
# independent in options_sentiment_av.py). Pure functions; neutral on missing
# data; never raises.

# Directional / pressure / participation factors.
OPTIONS_SENTIMENT_FACTORS = [
    "put_call_pressure_factor",        # net call(+) vs put(-) pressure, signed [-1,1]
    "call_pressure_factor",            # call-demand strength [0,1]
    "put_pressure_factor",             # put-demand strength [0,1]
    "volume_oi_activity_factor",       # vol/OI participation [0,1]
    "unusual_options_activity_factor", # fresh speculative flow [0,1]
    "gamma_pressure_factor",           # OI-weighted net gamma exposure [-1,1]
    "delta_pressure_factor",           # OI-weighted net delta exposure [-1,1]
    "iv_skew_factor",                  # mean put IV - mean call IV
    "iv_term_structure_factor",        # near-term vs far-term IV slope [-1,1]
    "option_sentiment_factor",         # bounded composite sentiment [-1,1]
]

# Cross-signal interaction factors (require a `context` dict; 0.0 when absent).
OPTIONS_SENTIMENT_INTERACTIONS = [
    "options_sentiment_x_price_momentum",
    "options_sentiment_x_news_sentiment",
    "options_sentiment_x_relative_volume",
    "gamma_pressure_x_breakout_strength",
    "put_call_ratio_x_trend_strength",
    "volume_oi_ratio_x_volatility_expansion",
]

# Latent factors produced downstream by latent_feature_discovery (declared here
# so the family registry is the single source of truth).
OPTIONS_SENTIMENT_LATENT = [
    "latent_options_factor_1",
    "latent_options_factor_2",
    "latent_options_factor_3",
    "latent_options_sentiment_state",
    "latent_gamma_state",
]


def neutral_options_sentiment_features() -> dict[str, Any]:
    feats = {k: 0.0 for k in OPTIONS_SENTIMENT_FACTORS}
    feats.update({k: 0.0 for k in OPTIONS_SENTIMENT_INTERACTIONS})
    feats["options_sentiment_available"] = 0
    return feats


def _iv_term_structure(chain: Optional[list[dict[str, Any]]], as_of: Optional[date] = None) -> float:
    """Near-term (≤30d) mean IV minus far-term (>30d) mean IV, scaled to [-1,1].

    Positive = inverted/backwardated term structure (near IV richer → event/stress
    premium); negative = upward-sloping (calm front, richer back). 0.0 when one
    side is empty (no slope inferable)."""
    near: list[float] = []
    far: list[float] = []
    for c in chain or []:
        if not isinstance(c, dict):
            continue
        iv = c.get("implied_volatility")
        if iv is None:
            continue
        d = _dte(c.get("expiration"), as_of)
        if d is None:
            continue
        (near if d <= 30 else far).append(float(iv))
    if not near or not far:
        return 0.0
    slope = (sum(near) / len(near)) - (sum(far) / len(far))
    return round(_clip(slope / 0.10, -1.0, 1.0), 5)   # 10 vol-pts → saturates


def build_options_sentiment_features(
    chain: Optional[list[dict[str, Any]]] = None,
    pcr: Optional[dict[str, Any]] = None,
    voi: Optional[dict[str, Any]] = None,
    spot: Optional[float] = None,
    as_of: Optional[date] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the `options_sentiment` candidate-alpha family (factors + interactions).

    Reuses build_pcr_features / build_voi_features / build_chain_features so the
    raw normalization stays in one place, then derives the named research factors.
    `context` may carry price_momentum, news_sentiment, relative_volume,
    breakout_strength, trend_strength, volatility_expansion to populate the
    interaction factors (all default 0.0 when absent)."""
    feats = neutral_options_sentiment_features()

    pf = build_pcr_features(pcr, as_of) if pcr else {}
    vf = build_voi_features(voi) if voi else {}
    cf = build_chain_features(chain, spot) if chain else {}
    any_data = bool(pf or vf or cf) and (
        bool(pcr and pcr.get("available")) or bool(voi and voi.get("available")) or bool(chain)
    )

    signal = float(pf.get("put_call_ratio_signal", 0.0) or 0.0)
    pcr_sentiment = float(pf.get("realtime_put_call_sentiment_score", 0.0) or 0.0)
    participation = float(vf.get("options_participation_score", 0.0) or 0.0)
    unusual = float(vf.get("unusual_options_activity_score", 0.0) or 0.0)
    gamma_pressure = float(cf.get("realtime_option_gamma_exposure", 0.0) or 0.0)
    delta_pressure = float(cf.get("realtime_option_delta_exposure", 0.0) or 0.0)
    iv_skew = float(cf.get("realtime_option_iv_skew", 0.0) or 0.0)
    iv_term = _iv_term_structure(chain, as_of)

    # Pressure / participation factors.
    feats["put_call_pressure_factor"] = round(_clip(signal, -1.0, 1.0), 5)
    feats["call_pressure_factor"] = round(max(0.0, signal), 5)
    feats["put_pressure_factor"] = round(max(0.0, -signal), 5)
    feats["volume_oi_activity_factor"] = round(_clip(participation, 0.0, 1.0), 5)
    feats["unusual_options_activity_factor"] = round(_clip(unusual, 0.0, 1.0), 5)
    feats["gamma_pressure_factor"] = round(_clip(gamma_pressure, -1.0, 1.0), 5)
    feats["delta_pressure_factor"] = round(_clip(delta_pressure, -1.0, 1.0), 5)
    feats["iv_skew_factor"] = round(iv_skew, 5)
    feats["iv_term_structure_factor"] = iv_term

    # Composite sentiment: directional PCR + dealer-flow pressure − put skew.
    skew_scaled = _clip(iv_skew / 0.20, -1.0, 1.0)
    option_sentiment = _clip(
        0.50 * pcr_sentiment + 0.20 * delta_pressure + 0.15 * gamma_pressure - 0.15 * skew_scaled,
        -1.0, 1.0,
    )
    feats["option_sentiment_factor"] = round(option_sentiment, 5)

    # Cross-signal interactions (neutral 0.0 when context absent).
    ctx = context or {}
    pm = float(ctx.get("price_momentum", 0.0) or 0.0)
    ns = float(ctx.get("news_sentiment", 0.0) or 0.0)
    rv = float(ctx.get("relative_volume", 0.0) or 0.0)
    bs = float(ctx.get("breakout_strength", 0.0) or 0.0)
    ts = float(ctx.get("trend_strength", 0.0) or 0.0)
    ve = float(ctx.get("volatility_expansion", 0.0) or 0.0)
    feats["options_sentiment_x_price_momentum"] = round(option_sentiment * pm, 5)
    feats["options_sentiment_x_news_sentiment"] = round(option_sentiment * ns, 5)
    feats["options_sentiment_x_relative_volume"] = round(option_sentiment * rv, 5)
    feats["gamma_pressure_x_breakout_strength"] = round(feats["gamma_pressure_factor"] * bs, 5)
    feats["put_call_ratio_x_trend_strength"] = round(feats["put_call_pressure_factor"] * ts, 5)
    feats["volume_oi_ratio_x_volatility_expansion"] = round(feats["volume_oi_activity_factor"] * ve, 5)

    feats["options_sentiment_available"] = 1 if any_data else 0
    return feats


# =============================================================================
# Aggregate builder
# =============================================================================

def build_options_features(
    chain: Optional[list[dict[str, Any]]] = None,
    pcr: Optional[dict[str, Any]] = None,
    voi: Optional[dict[str, Any]] = None,
    spot: Optional[float] = None,
    as_of: Optional[date] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the full realtime options feature dict (sentiment + flow + chain +
    the options_sentiment research family)."""
    feats = neutral_options_features()
    any_data = False
    if pcr:
        feats.update(build_pcr_features(pcr, as_of))
        any_data = any_data or bool(pcr.get("available"))
    if voi:
        feats.update(build_voi_features(voi))
        any_data = any_data or bool(voi.get("available"))
    if chain:
        feats.update(build_chain_features(chain, spot))
        any_data = any_data or len(chain) > 0
    # options_sentiment research family (additive; existing keys above unchanged).
    feats.update(build_options_sentiment_features(chain, pcr, voi, spot, as_of, context))
    feats["realtime_options_available"] = 1 if any_data else 0
    return feats


# =============================================================================
# Contract selection driven by realtime chain data
# =============================================================================

def _dte(expiration: Any, as_of: Optional[date] = None) -> Optional[int]:
    try:
        d = date.fromisoformat(str(expiration)[:10])
        return (d - (as_of or date.today())).days
    except Exception:
        return None


def score_contract(
    c: dict[str, Any],
    spot: Optional[float] = None,
    side: str = "call",
    target_delta: float = 0.55,
    min_dte: int = 21,
    max_dte: int = 60,
    as_of: Optional[date] = None,
) -> dict[str, Any]:
    """Score a single realtime option contract for selection.

    Returns {score (0-100), passes_gates (bool), reasons (list), dte, moneyness}.
    Prefers tight spreads, sufficient volume + OI, clean greeks, reasonable IV,
    expiration in the configured DTE band and delta near target.
    """
    reasons: list[str] = []
    bid = c.get("bid"); ask = c.get("ask")
    vol = c.get("open_interest") and (c.get("volume") or 0)
    vol = c.get("volume") or 0
    oi = c.get("open_interest") or 0
    iv = c.get("implied_volatility")
    delta = c.get("delta")
    strike = c.get("strike")
    dte = _dte(c.get("expiration"), as_of)

    # Hard gates.
    passes = True
    if dte is None or not (min_dte <= dte <= max_dte):
        passes = False; reasons.append(f"dte_out_of_band({dte})")
    if bid is None or ask is None or ask <= 0:
        passes = False; reasons.append("no_quote")
    spread_pct = None
    if bid is not None and ask is not None and ask > 0:
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
        if spread_pct > 0.10:
            passes = False; reasons.append(f"wide_spread({spread_pct:.2%})")
    if oi < 200:
        passes = False; reasons.append(f"low_oi({oi})")
    if vol < 10:
        reasons.append(f"low_volume({vol})")

    # Scoring (0-100) — only meaningful when basic data present.
    liq = _contract_liquidity(c) or 0.0
    spread_score = _clip(1.0 - (spread_pct or 0.10) / 0.10, 0.0, 1.0) if spread_pct is not None else 0.0
    delta_score = 0.0
    if delta is not None:
        delta_score = _clip(1.0 - abs(abs(delta) - target_delta) / 0.30, 0.0, 1.0)
    iv_score = 0.0
    if iv is not None:
        # reasonable IV band 0.15..0.80; penalize extremes.
        iv_score = _clip(1.0 - abs(iv - 0.40) / 0.40, 0.0, 1.0)
    greek_clean = 1.0 if (delta is not None and c.get("gamma") is not None and c.get("theta") is not None) else 0.5

    score = 100.0 * (0.30 * liq + 0.25 * delta_score + 0.20 * spread_score
                     + 0.15 * iv_score + 0.10 * greek_clean)
    return {
        "score": round(score, 2),
        "passes_gates": passes,
        "reasons": reasons,
        "dte": dte,
        "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "liquidity": round(liq, 4),
        "contract_id": c.get("contract_id"),
        "strike": strike,
        "type": c.get("type"),
        "expiration": c.get("expiration"),
    }


def select_best_contract(
    chain: list[dict[str, Any]],
    spot: Optional[float] = None,
    side: str = "call",
    target_delta: float = 0.55,
    min_dte: int = 21,
    max_dte: int = 60,
    as_of: Optional[date] = None,
) -> Optional[dict[str, Any]]:
    """Pick the highest-scoring contract of ``side`` that passes hard gates.

    Returns the original contract dict augmented with a `selection` sub-dict, or
    None when no contract passes (caller should fall back to its existing logic).
    """
    side = (side or "call").lower()
    candidates = [c for c in (chain or []) if c.get("type") == side]
    scored = []
    for c in candidates:
        s = score_contract(c, spot, side, target_delta, min_dte, max_dte, as_of)
        scored.append((s, c))
    passing = [(s, c) for s, c in scored if s["passes_gates"]]
    pool = passing or []
    if not pool:
        return None
    pool.sort(key=lambda sc: sc[0]["score"], reverse=True)
    best_score, best_contract = pool[0]
    out = dict(best_contract)
    out["selection"] = best_score
    return out
