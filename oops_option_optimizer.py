#!/usr/bin/env python3
"""
oops_option_optimizer.py — OOPS-inspired Options Utility Optimization Layer.

Inspired by "Optimal Option Portfolio Strategies": instead of ranking call contracts by
Black-Scholes score / static 3R targets / liquidity alone, simulate each candidate's
payoff distribution and rank by EXPECTED UTILITY (CRRA) of the resulting portfolio
wealth. Options have non-normal returns (skew, kurtosis, convexity, high-probability
small losses with rare large gains, bid/ask drag), so mean-variance is not enough.

This layer is ADVISORY inside the Alpha Call Engine. It never places orders; RiskManager
and AlpacaGateway remain the only execution path, and OOPS can never override the hard
stop, kill switch, daily-loss halt, liquidity collapse, regime flip, or RiskManager.

Final rule:
    Black-Scholes estimates fair value. OOPS estimates payoff-distribution utility.
    Alpha setup determines direction. OOPS chooses the best option expression.
    RiskManager decides whether the trade is allowed.
"""
from __future__ import annotations

import json
import math
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
OOPS_REPORT = "oops_option_optimizer_report.json"


def _research_dir(date_tag: str | None = None) -> Path:
    d = DAILY_OUTPUTS / (date_tag or DATA_TAG)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class OOPSConfig:
    enabled: bool = os.getenv("ENABLE_OOPS_OPTIMIZER", "true").lower() in {"1", "true"}
    risk_aversion_gamma: float = float(os.getenv("OOPS_RISK_AVERSION_GAMMA", "5"))
    monte_carlo_paths: int = int(os.getenv("OOPS_MONTE_CARLO_PATHS", "5000"))
    use_empirical_bootstrap: bool = os.getenv("OOPS_USE_EMPIRICAL_BOOTSTRAP", "true").lower() in {"1", "true"}
    transaction_cost_bps: float = float(os.getenv("OOPS_TRANSACTION_COST_BPS", "5"))
    slippage_bps: float = float(os.getenv("OOPS_SLIPPAGE_BPS", "2"))
    min_expected_utility_edge: float = float(os.getenv("OOPS_MIN_EXPECTED_UTILITY_EDGE", "0.0"))
    # Portfolio weight the option occupies — utility is over TOTAL wealth, so a total
    # option loss is finite (W = 1 - weight > 0). Mirrors options risk-per-trade.
    portfolio_weight: float = float(os.getenv("OOPS_PORTFOLIO_WEIGHT", os.getenv("OPTIONS_RISK_PER_TRADE_PCT", "0.02")))
    holding_days: int = int(os.getenv("OOPS_HOLDING_DAYS", "0"))   # 0 = hold to expiry (terminal payoff)
    random_seed: int = int(os.getenv("OOPS_RANDOM_SEED", "12345"))  # deterministic / reproducible
    risk_free_rate: float = float(os.getenv("OPTIONS_RISK_FREE_RATE", "0.043"))
    drift: float = float(os.getenv("OOPS_DRIFT", "0.0"))
    combined_bs_weight: float = float(os.getenv("OOPS_COMBINED_BS_WEIGHT", "0.40"))
    combined_oops_weight: float = float(os.getenv("OOPS_COMBINED_OOPS_WEIGHT", "0.40"))
    combined_liq_weight: float = float(os.getenv("OOPS_COMBINED_LIQ_WEIGHT", "0.20"))
    # Liquidity gates re-checked defensively so OOPS can never select an illiquid name.
    min_open_interest: int = int(os.getenv("ALPHA_MIN_OI", "200"))
    min_volume: int = int(os.getenv("ALPHA_MIN_VOLUME", "10"))
    max_spread_pct: float = float(os.getenv("ALPHA_MAX_SPREAD_PCT", "0.05"))
    exit_min_dte: int = int(os.getenv("OOPS_EXIT_MIN_DTE", "10"))


def _rng(cfg: OOPSConfig):
    return np.random.default_rng(cfg.random_seed)


# ── simulation ────────────────────────────────────────────────────────────────────
def simulate_underlying_returns(current_price: float, dte: int, vol_annual: float,
                                cfg: OOPSConfig | None = None,
                                hist_daily_returns=None, rng=None,
                                regime_returns=None) -> np.ndarray:
    """
    Simulate terminal underlying prices over the holding horizon (default = to expiry).

    Methods (in order of preference):
      * empirical bootstrap of daily returns (regime-conditioned when provided),
        volatility-scaled to the current vol,
      * lognormal Monte-Carlo fallback (drift, vol_annual).
    Returns an array of terminal prices (length monte_carlo_paths).
    """
    cfg = cfg or OOPSConfig()
    rng = rng or _rng(cfg)
    n = max(int(cfg.monte_carlo_paths), 1)
    horizon = dte if cfg.holding_days <= 0 else min(int(cfg.holding_days), int(dte))
    horizon = max(int(horizon), 1)
    if current_price <= 0:
        return np.full(n, max(current_price, 0.0))

    src = regime_returns if (regime_returns is not None and len(regime_returns) >= 20) else hist_daily_returns
    if cfg.use_empirical_bootstrap and src is not None and len(src) >= 20:
        daily = np.asarray(src, dtype=float)
        daily = daily[np.isfinite(daily)]
        if len(daily) >= 20:
            # Volatility-scale the empirical daily returns to the target annual vol.
            emp_daily_vol = float(np.std(daily)) or 1e-9
            target_daily_vol = (vol_annual / math.sqrt(252.0)) if vol_annual > 0 else emp_daily_vol
            scale = target_daily_vol / emp_daily_vol if emp_daily_vol > 0 else 1.0
            draws = rng.choice(daily, size=(n, horizon), replace=True) * scale
            cum_log = np.sum(draws, axis=1)
            return current_price * np.exp(cum_log)

    # Lognormal Monte-Carlo fallback.
    if vol_annual <= 0:
        vol_annual = 0.30
    T = horizon / 365.0
    z = rng.standard_normal(n)
    terminal = current_price * np.exp((cfg.drift - 0.5 * vol_annual ** 2) * T
                                      + vol_annual * math.sqrt(T) * z)
    return terminal


def simulate_option_payoffs(terminal_prices, strike: float) -> np.ndarray:
    """Terminal call payoff per share: max(S_T - K, 0)."""
    s = np.asarray(terminal_prices, dtype=float)
    return np.maximum(s - float(strike), 0.0)


def compute_option_return_distribution(payoffs, entry_mid: float, spread_pct: float,
                                       cfg: OOPSConfig | None = None) -> dict:
    """
    Convert per-share payoffs to net option-return outcomes, charging the bid/ask
    half-spread, transaction cost, and slippage on BOTH entry and exit. Returns the
    distribution and its moments.
    """
    cfg = cfg or OOPSConfig()
    payoffs = np.asarray(payoffs, dtype=float)
    fee = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10000.0
    half_spread = max(float(spread_pct), 0.0) / 2.0
    entry_cost = max(float(entry_mid), 1e-9) * (1.0 + half_spread + fee)   # paid to get in
    exit_value = payoffs * (1.0 - half_spread - fee)                       # received on the way out
    returns = exit_value / entry_cost - 1.0

    n = len(returns) or 1
    mean = float(np.mean(returns))
    std = float(np.std(returns)) or 1e-12
    median = float(np.median(returns))
    prob_profit = float(np.mean(returns > 0.0))
    prob_total_loss = float(np.mean(payoffs <= 0.0))            # expires worthless
    # Sample skew / excess kurtosis (manual — no scipy dependency).
    z = (returns - mean) / std
    skew = float(np.mean(z ** 3))
    kurtosis = float(np.mean(z ** 4) - 3.0)
    # Downside tail risk = expected shortfall at 5% (mean of worst 5% returns).
    k = max(1, int(0.05 * n))
    worst = np.sort(returns)[:k]
    downside_tail_risk = float(np.mean(worst))
    return {
        "returns": returns, "entry_cost": round(entry_cost, 6),
        "expected_return": round(mean, 4), "median_return": round(median, 4),
        "probability_profit": round(prob_profit, 4), "probability_total_loss": round(prob_total_loss, 4),
        "skew": round(skew, 4), "kurtosis": round(kurtosis, 4),
        "downside_tail_risk": round(downside_tail_risk, 4),
    }


# ── utility ─────────────────────────────────────────────────────────────────────
def expected_utility(returns, cfg: OOPSConfig | None = None) -> dict:
    """
    CRRA / power utility of TOTAL portfolio wealth from allocating `portfolio_weight`
    to this option:  W = 1 + weight·option_return  (>0 even on total loss).
        U(W) = W^(1-γ)/(1-γ)   (log utility when γ == 1)
    Returns expected_utility, certainty_equivalent (wealth), and the CE option-return.
    """
    cfg = cfg or OOPSConfig()
    g = float(cfg.risk_aversion_gamma)
    w = float(cfg.portfolio_weight)
    r = np.asarray(returns, dtype=float)
    wealth = np.clip(1.0 + w * r, 1e-9, None)
    if abs(g - 1.0) < 1e-9:
        u = np.log(wealth)
        eu = float(np.mean(u))
        ce_wealth = math.exp(eu)
    else:
        u = np.power(wealth, 1.0 - g) / (1.0 - g)
        eu = float(np.mean(u))
        ce_wealth = (eu * (1.0 - g)) ** (1.0 / (1.0 - g))
    ce_return = (ce_wealth - 1.0) / w if w > 0 else 0.0
    return {"expected_utility": eu, "certainty_equivalent": round(ce_wealth, 8),
            "certainty_equivalent_return": round(ce_return, 4)}


# ── contract scoring ──────────────────────────────────────────────────────────────
def _liquidity_score(contract: dict, cfg: OOPSConfig) -> float:
    oi = int(contract.get("open_interest") or 0)
    vol = int(contract.get("volume") or 0)
    spread = float(contract.get("spread_pct") or 1.0)
    oi_s = _clamp(oi / 1000.0, 0.0, 1.0)
    vol_s = _clamp(vol / 100.0, 0.0, 1.0)
    spread_s = _clamp(1.0 - spread / max(cfg.max_spread_pct, 1e-6), 0.0, 1.0)
    return round(100.0 * (0.45 * oi_s + 0.25 * vol_s + 0.30 * spread_s), 2)


def score_oops_contract(dist: dict, util: dict, contract: dict, cfg: OOPSConfig | None = None,
                        iv_crush_risk: float = 0.0) -> dict:
    """
    0-100 OOPS score from the certainty-equivalent return plus explicit penalties for
    total-loss probability, negative skew, high kurtosis, wide spread, theta decay, and
    IV-crush risk. Higher = a better option EXPRESSION of the setup.
    """
    cfg = cfg or OOPSConfig()
    ce_ret = float(util.get("certainty_equivalent_return", 0.0))
    prob_profit = float(dist.get("probability_profit", 0.0))
    prob_total_loss = float(dist.get("probability_total_loss", 0.0))
    skew = float(dist.get("skew", 0.0))
    kurt = float(dist.get("kurtosis", 0.0))
    spread = float(contract.get("spread_pct") or 0.0)
    theta_pct = abs(float(contract.get("theta_pct_per_day", 0.0) or 0.0))

    score = 50.0
    score += 60.0 * _clamp(ce_ret, -0.5, 0.5)          # certainty-equivalent return (±30)
    score += 20.0 * (prob_profit - 0.5)                # profit probability (±10)
    score -= 40.0 * prob_total_loss                    # high worthless prob is heavily penalised
    score += 5.0 * _clamp(skew, -2.0, 2.0)             # positive skew rewarded
    score -= 2.0 * _clamp(kurt, 0.0, 10.0)             # fat tails penalised
    score -= 100.0 * _clamp(spread, 0.0, 0.10)         # wide spread penalised
    score -= 50.0 * _clamp(theta_pct, 0.0, 0.08)       # theta decay penalised
    score -= 0.20 * _clamp(iv_crush_risk, 0.0, 100.0)  # IV-crush penalty (0-20)
    return {
        "oops_score": round(_clamp(score, 0.0, 100.0), 2),
        "utility_score": round(_clamp(50.0 + 100.0 * ce_ret, 0.0, 100.0), 2),
        "expected_return": dist["expected_return"], "median_return": dist["median_return"],
        "probability_profit": dist["probability_profit"],
        "probability_total_loss": dist["probability_total_loss"],
        "expected_utility": round(util["expected_utility"], 8),
        "certainty_equivalent": util["certainty_equivalent"],
        "certainty_equivalent_return": util["certainty_equivalent_return"],
        "skew": dist["skew"], "kurtosis": dist["kurtosis"],
        "downside_tail_risk": dist["downside_tail_risk"],
    }


def _passes_liquidity(contract: dict, cfg: OOPSConfig) -> bool:
    if int(contract.get("open_interest") or 0) < cfg.min_open_interest:
        return False
    if int(contract.get("volume") or 0) < cfg.min_volume:
        return False
    if float(contract.get("spread_pct") or 1.0) > cfg.max_spread_pct:
        return False
    return True


def optimize_option_candidates(candidates: list[dict], current_price: float,
                               vol_annual_default: float = 0.30,
                               hist_daily_returns=None, cfg: OOPSConfig | None = None,
                               iv_crush_risk: float = 0.0, regime_returns=None) -> list[dict]:
    """
    Rank candidate calls by OOPS score (highest = best expression). Re-applies liquidity
    gates defensively so an illiquid contract can never be selected. Returns the input
    contracts enriched with OOPS fields, sorted by oops_score desc. Never raises — on a
    simulation failure a candidate is skipped (caller falls back to Black-Scholes).
    """
    cfg = cfg or OOPSConfig()
    rng = _rng(cfg)
    ranked: list[dict] = []
    for c in candidates or []:
        try:
            if not _passes_liquidity(c, cfg):
                continue
            strike = float(c.get("strike") or c.get("strike_price") or 0.0)
            mid = float(c.get("mid") or 0.0)
            dte = int(c.get("dte") or 0)
            iv = float(c.get("implied_volatility") or 0.0) or vol_annual_default
            if strike <= 0 or mid <= 0 or dte <= 0:
                continue
            terminal = simulate_underlying_returns(current_price, dte, iv, cfg,
                                                   hist_daily_returns=hist_daily_returns,
                                                   rng=rng, regime_returns=regime_returns)
            payoffs = simulate_option_payoffs(terminal, strike)
            dist = compute_option_return_distribution(payoffs, mid, float(c.get("spread_pct") or 0.0), cfg)
            util = expected_utility(dist["returns"], cfg)
            scored = score_oops_contract(dist, util, c, cfg, iv_crush_risk)
            row = dict(c)
            row.update(scored)
            row["liquidity_score_oops"] = _liquidity_score(c, cfg)
            ranked.append(row)
        except Exception as exc:
            logger.debug("OOPS candidate skipped (%s): %s", c.get("contract_symbol"), exc)
            continue
    ranked.sort(key=lambda r: r.get("oops_score", 0.0), reverse=True)
    return ranked


# ── exit assessment (advisory; never overrides hard rules) ───────────────────────
def oops_exit_recommendation(current_value: float, entry_cost: float, dte_remaining: int,
                             partial_taken: bool = False, cfg: OOPSConfig | None = None) -> dict:
    """
    Advisory exit guidance based on the deteriorating/strengthening payoff outlook. The
    caller checks the hard stop / 3R / kill switch / RiskManager FIRST — this only runs
    on a position that already survived them, and it can only move in the protective
    direction (cut early / tighten) or flag a runner. It NEVER loosens protection.

    Returns {"action": "hold"|"tighten"|"reduce"|"exit"|"runner", "rationale", "trail_tightness_mult"}.
    """
    cfg = cfg or OOPSConfig()
    if entry_cost <= 0:
        return {"action": "hold", "rationale": "no_entry_cost", "trail_tightness_mult": 1.0}
    ratio = current_value / entry_cost
    # Near expiry + not profitable: convexity gone, expected utility decays → cut.
    if dte_remaining <= cfg.exit_min_dte and ratio < 1.0:
        return {"action": "exit", "rationale": f"oops_eu_decay:dte={dte_remaining}<= {cfg.exit_min_dte},ratio={ratio:.2f}",
                "trail_tightness_mult": 0.3}
    # Underwater and giving back: tighten protection.
    if ratio < 0.85:
        return {"action": "tighten", "rationale": f"oops_eu_weak:ratio={ratio:.2f}", "trail_tightness_mult": 0.5}
    # Strong, with time left: let the convex winner run (advisory only).
    if ratio >= 1.7 and dte_remaining > cfg.exit_min_dte:
        return {"action": "runner", "rationale": f"oops_eu_strong:ratio={ratio:.2f}", "trail_tightness_mult": 1.0}
    return {"action": "hold", "rationale": f"oops_eu_stable:ratio={ratio:.2f}", "trail_tightness_mult": 1.0}


# ── diagnostic output ─────────────────────────────────────────────────────────────
def write_oops_diagnostic(rows: list[dict], market_context: dict | None = None,
                          date_tag: str | None = None) -> None:
    """Write daily_system_outputs/<YYYYMMDD>/oops_option_optimizer_report.json."""
    try:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date_tag": date_tag or DATA_TAG,
            "market_regime": str((market_context or {}).get("overall_market_bias") or ""),
            "config": {"gamma": OOPSConfig().risk_aversion_gamma,
                       "paths": OOPSConfig().monte_carlo_paths,
                       "portfolio_weight": OOPSConfig().portfolio_weight},
            "count": len(rows),
            "contracts": rows,
        }
        (_research_dir(date_tag) / OOPS_REPORT).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("write_oops_diagnostic failed: %s", exc)
