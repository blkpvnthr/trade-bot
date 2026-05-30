#!/usr/bin/env python3
"""
alpha_probability_model.py — Probability-Weighted Alpha Management Layer.

Improves realized win rate and expectancy by adapting SIZE and MANAGEMENT to an
estimated probability of profit — NOT by blocking trades.

Estimates per signal:
    probability_of_profit, expected_R, confidence, historical_sample_size

and maps the probability to MANAGEMENT outputs (never a reject):
    size_multiplier, management_style, stop_adjustment, partial_profit_trigger,
    trailing_tightness, overnight_allowed, exploration_tag

Hard rules:
  1. No hard probability rejection — low-probability trades still execute (smaller, tighter).
  2. Probability never overrides RiskManager (it only emits sizing/management metadata).
  3. Probability adjustments may ONLY reduce risk or improve profit protection
     (size multiplier ≤ 1.0; stop/trail tightenings are ≤ 1.0 multipliers; partials move
     earlier). Protective exits are never weakened.

Probability bands (0-100 = probability_of_profit × 100):
    80-100 → size 1.00, high_conviction, normal stop, runner + overnight (if BS agrees)
    70-79  → size 0.75, standard,        normal stop, partial ~+45%, normal trail
    60-69  → size 0.50, cautious,        tighter stop, partial ~+35%, tighter trail, overnight only if BS exceptional
    50-59  → size 0.25, exploratory,     tighter stop, quick partial ~+27.5%, tight trail, no overnight
    <50    → size 0.10, micro_exploration, tight stop, partial ~+25%, very tight trail, no overnight
"""
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class ProbabilityConfig:
    enabled: bool = os.getenv("ENABLE_ALPHA_PROBABILITY_WEIGHTING", "true").lower() in {"1", "true"}
    target_win_rate: float = float(os.getenv("ALPHA_TARGET_WIN_RATE", "0.80"))
    low_prob_size_multiplier: float = float(os.getenv("ALPHA_LOW_PROB_SIZE_MULTIPLIER", "0.25"))
    micro_prob_size_multiplier: float = float(os.getenv("ALPHA_MICRO_PROB_SIZE_MULTIPLIER", "0.10"))
    exploration_rate: float = float(os.getenv("ALPHA_PROBABILITY_EXPLORATION_RATE", "0.05"))
    min_size_multiplier: float = float(os.getenv("ALPHA_MIN_SIZE_MULTIPLIER", "0.10"))
    max_size_multiplier: float = float(os.getenv("ALPHA_MAX_SIZE_MULTIPLIER", "1.00"))
    exploration_enabled: bool = os.getenv("ALPHA_PROBABILITY_EXPLORATION_ENABLED", "true").lower() in {"1", "true"}
    # Evidence blending
    min_sample: int = int(os.getenv("ALPHA_PROB_MIN_SAMPLE", "10"))   # trades for full empirical confidence
    # Engine management defaults (mirror AlphaCallConfig) — the layer only tightens these.
    base_stop_loss_pct: float = float(os.getenv("OPTIONS_SL_PCT", os.getenv("ALPHA_STOP_PCT", "0.28")))
    base_partial_profit_pct: float = float(os.getenv("ALPHA_PARTIAL_PROFIT_PCT", "0.5"))
    base_trail_offset: float = float(os.getenv("ALPHA_R_TRAIL_OFFSET", "0.8"))
    # Overnight eligibility thresholds (Black-Scholes must agree).
    overnight_bs_min: float = float(os.getenv("ALPHA_OVERNIGHT_BS_MIN", "60"))
    overnight_bs_exceptional: float = float(os.getenv("ALPHA_OVERNIGHT_BS_EXCEPTIONAL", "80"))
    overnight_max_iv_crush: float = float(os.getenv("ALPHA_OVERNIGHT_MAX_IV_CRUSH", "40"))


# ── probability estimation (pure) ─────────────────────────────────────────────────
def load_setup_history(date_tag: str | None = None) -> dict:
    """Empirical per-setup win rate / avg R / trades from the research lab (fail-safe)."""
    d = DAILY_OUTPUTS / (date_tag or DATA_TAG)
    for name, key in (("research_lab.json", "alpha"), ("alpha_expectancy_report.json", "alpha")):
        try:
            p = d / name
            if p.exists():
                data = json.loads(p.read_text())
                alpha = data.get(key, {}) if name == "research_lab.json" else data.get("alpha", {})
                bs = alpha.get("by_setup_type", {})
                if bs:
                    return bs
        except Exception:
            continue
    return {}


def estimate_probability(setup_type: str, setup_score: float, grade: str,
                         expected_r_bs: float = 1.0, setup_history: dict | None = None,
                         config: ProbabilityConfig | None = None) -> dict:
    """
    Estimate probability_of_profit (0-1), expected_R, confidence (0-1), and the
    historical sample size. Blends the empirical per-setup win rate (weighted by sample
    confidence) with a setup-score prior, then nudges by the Black-Scholes expected-R
    edge. Never rejects — purely an estimate.
    """
    cfg = config or ProbabilityConfig()
    hist = (setup_history or {}).get(setup_type, {})
    n = int(hist.get("trades", 0) or 0)
    emp_wr = _clamp(float(hist.get("win_rate", 0.0) or 0.0), 0.0, 1.0)
    emp_r = float(hist.get("avg_r_multiple", 0.0) or 0.0)

    # Setup-score prior: 0→0.40, 100→0.85 (A+ setups imply higher base hit rate).
    score_prob = _clamp(0.40 + (float(setup_score or 0.0) / 100.0) * 0.45, 0.30, 0.90)

    # Confidence in the empirical rate from sample size.
    conf = min(n / max(cfg.min_sample, 1), 1.0) if n > 0 else 0.0
    prob = conf * emp_wr + (1.0 - conf) * score_prob

    # Forward edge nudge from Black-Scholes expected R (small, bounded).
    if expected_r_bs:
        prob += _clamp((float(expected_r_bs) - 1.0) * 0.02, -0.05, 0.05)
    prob = _clamp(prob, 0.05, 0.95)

    expected_R = conf * emp_r + (1.0 - conf) * float(expected_r_bs or 1.0)
    grade_bonus = 0.15 if str(grade).upper() in ("A+", "A") else 0.0
    confidence = round(_clamp(0.30 + 0.55 * conf + grade_bonus, 0.0, 1.0), 3)
    source = ("empirical" if conf >= 0.5 else "setup_score") + ("+bs" if expected_r_bs else "")

    return {
        "probability_of_profit": round(prob, 4),
        "expected_R": round(expected_R, 3),
        "confidence": confidence,
        "historical_sample_size": n,
        "source": source,
    }


def probability_band(prob_100: float) -> str:
    if prob_100 >= 80:
        return "80-100"
    if prob_100 >= 70:
        return "70-79"
    if prob_100 >= 60:
        return "60-69"
    if prob_100 >= 50:
        return "50-59"
    return "below-50"


def management_for_probability(prob_of_profit: float, config: ProbabilityConfig | None = None,
                               bs_score: float = 0.0, iv_crush_risk: float = 0.0) -> dict:
    """
    Map probability_of_profit (0-1) to management directives. Every adjustment is in the
    PROTECTIVE direction only: size_multiplier ≤ 1.0, stop/trail tightness ≤ 1.0, partials
    move earlier as probability falls. This NEVER returns a reject.
    """
    cfg = config or ProbabilityConfig()
    p100 = prob_of_profit * 100.0
    band = probability_band(p100)
    bs_ok = bs_score >= cfg.overnight_bs_min and iv_crush_risk < cfg.overnight_max_iv_crush
    bs_exceptional = bs_score >= cfg.overnight_bs_exceptional and iv_crush_risk < cfg.overnight_max_iv_crush

    if band == "80-100":
        out = {"size_multiplier": 1.00, "management_style": "high_conviction",
               "stop_adjustment": "normal", "stop_tightness": 1.00,
               "partial_profit_trigger": cfg.base_partial_profit_pct,
               "trailing_tightness": "normal", "trail_tightness_mult": 1.00,
               "allow_runner": True, "overnight_allowed": bool(bs_ok), "exploration_tag": False}
    elif band == "70-79":
        out = {"size_multiplier": 0.75, "management_style": "standard",
               "stop_adjustment": "normal", "stop_tightness": 1.00,
               "partial_profit_trigger": 0.45,
               "trailing_tightness": "normal", "trail_tightness_mult": 1.00,
               "allow_runner": True, "overnight_allowed": False, "exploration_tag": False}
    elif band == "60-69":
        out = {"size_multiplier": 0.50, "management_style": "cautious",
               "stop_adjustment": "tighter", "stop_tightness": 0.90,
               "partial_profit_trigger": 0.35,
               "trailing_tightness": "tighter", "trail_tightness_mult": 0.70,
               "allow_runner": False, "overnight_allowed": bool(bs_exceptional), "exploration_tag": False}
    elif band == "50-59":
        out = {"size_multiplier": cfg.low_prob_size_multiplier, "management_style": "exploratory",
               "stop_adjustment": "tight", "stop_tightness": 0.80,
               "partial_profit_trigger": 0.275,
               "trailing_tightness": "tight", "trail_tightness_mult": 0.50,
               "allow_runner": False, "overnight_allowed": False, "exploration_tag": True}
    else:  # below-50
        out = {"size_multiplier": cfg.micro_prob_size_multiplier, "management_style": "micro_exploration",
               "stop_adjustment": "very_tight", "stop_tightness": 0.70,
               "partial_profit_trigger": 0.25,
               "trailing_tightness": "very_tight", "trail_tightness_mult": 0.40,
               "allow_runner": False, "overnight_allowed": False, "exploration_tag": True}
        if not cfg.exploration_enabled:
            # Still NEVER rejects — execute at the minimum size and note it.
            out["size_multiplier"] = cfg.min_size_multiplier
            out["exploration_note"] = "exploration_disabled_min_size"

    # Enforce the size-multiplier bounds (never increase beyond max, never below min).
    out["size_multiplier"] = round(_clamp(out["size_multiplier"], cfg.min_size_multiplier,
                                          cfg.max_size_multiplier), 4)
    out["probability_band"] = band
    return out


def evaluate(setup_type: str, setup_score: float, grade: str, expected_r_bs: float = 1.0,
             bs_score: float = 0.0, iv_crush_risk: float = 0.0,
             setup_history: dict | None = None, config: ProbabilityConfig | None = None) -> dict:
    """Convenience: probability estimate + management directives in one call."""
    cfg = config or ProbabilityConfig()
    est = estimate_probability(setup_type, setup_score, grade, expected_r_bs, setup_history, cfg)
    mgmt = management_for_probability(est["probability_of_profit"], cfg, bs_score, iv_crush_risk)
    return {**est, **mgmt}
