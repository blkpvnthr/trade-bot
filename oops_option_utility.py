
#!/usr/bin/env python3
"""
oops_option_utility.py — OOPS-style option utility scoring for offline RL.

Implements the practical part of Faias & Santa-Clara, "Optimal Option Portfolio
Strategies": simulate underlying return paths, map those paths into option payoffs,
then score contracts/option candidates by expected utility instead of mean/variance
alone.  This module is intentionally offline-only and dependency-light so it can be
used by dataset builders, reward functions, tuners, and parameter optimizers without
importing broker/execution code.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

OOPS_FIELDS: List[str] = [
    "oops_expected_utility",
    "oops_certainty_equivalent",
    "oops_sim_sharpe",
    "oops_skew",
    "oops_kurtosis",
    "oops_bid_ask_penalty",
    "oops_elasticity",
    "oops_delta_neutrality_score",
    "oops_liquidity_score",
    "oops_score",
]


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip(x: float, lo: float = -10.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, float(x)))


def crra_utility(wealth: np.ndarray, gamma: float = 3.0) -> np.ndarray:
    """Power utility. wealth is clipped positive to avoid undefined utility."""
    w = np.maximum(np.asarray(wealth, dtype=float), 1e-9)
    if abs(gamma - 1.0) < 1e-9:
        return np.log(w)
    return (np.power(w, 1.0 - gamma) - 1.0) / (1.0 - gamma)


def certainty_equivalent(expected_utility: float, gamma: float = 3.0) -> float:
    """Invert CRRA utility into a wealth-equivalent score."""
    if abs(gamma - 1.0) < 1e-9:
        return float(math.exp(expected_utility) - 1.0)
    val = expected_utility * (1.0 - gamma) + 1.0
    if val <= 0:
        return -1.0
    return float(val ** (1.0 / (1.0 - gamma)) - 1.0)


def simulate_underlying_paths(
    spot: float,
    historical_log_returns: Sequence[float] | None = None,
    n: int = 2048,
    horizon_bars: int = 1,
    conditional_vol: float | None = None,
    seed: int = 17,
) -> np.ndarray:
    """Bootstrap/parametric simulation of S_T under the empirical distribution."""
    spot = max(_f(spot, 1.0), 1e-9)
    rng = np.random.default_rng(seed)
    hist = np.array([_f(r) for r in (historical_log_returns or []) if math.isfinite(_f(r))], dtype=float)
    if hist.size >= 20:
        draws = rng.choice(hist, size=(n, max(1, horizon_bars)), replace=True)
    else:
        sigma = max(_f(conditional_vol, 0.02), 1e-4)
        mu = float(np.mean(hist)) if hist.size else 0.0
        draws = rng.normal(mu, sigma, size=(n, max(1, horizon_bars)))
    log_ret = draws.sum(axis=1)
    return spot * np.exp(log_ret)


def option_payoff_returns(
    terminal_spot: np.ndarray,
    strike: float,
    option_price: float,
    option_type: str = "call",
    side: str = "long",
) -> np.ndarray:
    """Convert simulated terminal underlying prices into long/short option returns."""
    strike = max(_f(strike, 0.0), 0.0)
    option_price = max(_f(option_price, 0.01), 1e-6)
    typ = str(option_type or "call").lower()
    payoff = np.maximum(terminal_spot - strike, 0.0) if typ.startswith("c") else np.maximum(strike - terminal_spot, 0.0)
    long_r = payoff / option_price - 1.0
    return -long_r if str(side or "long").lower().startswith("short") else long_r


def summarize_returns(returns: Sequence[float], gamma: float = 3.0) -> Dict[str, float]:
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"expected_utility": -1.0, "certainty_equivalent": -1.0, "sharpe": 0.0, "skew": 0.0, "kurtosis": 0.0}
    mean = float(np.mean(r)); std = float(np.std(r) + 1e-9)
    z = (r - mean) / std
    eu = float(np.mean(crra_utility(1.0 + np.clip(r, -0.99, 10.0), gamma=gamma)))
    return {
        "expected_utility": eu,
        "certainty_equivalent": certainty_equivalent(eu, gamma=gamma),
        "sharpe": mean / std,
        "skew": float(np.mean(z ** 3)),
        "kurtosis": float(np.mean(z ** 4)),
    }


def score_option_candidate(candidate: Dict[str, Any], historical_log_returns: Sequence[float] | None = None) -> Dict[str, float]:
    """Return OOPS features for one option candidate/event row."""
    spot = _f(candidate.get("underlying_price", candidate.get("spot", candidate.get("close", 100.0))), 100.0)
    strike = _f(candidate.get("strike", candidate.get("option_strike", spot)), spot)
    bid = _f(candidate.get("bid", candidate.get("option_bid", 0.0)), 0.0)
    ask = _f(candidate.get("ask", candidate.get("option_ask", 0.0)), 0.0)
    mid = _f(candidate.get("mid", candidate.get("option_mid", (bid + ask) / 2.0 if bid and ask else 0.0)), 0.0)
    price = max(mid or ask or bid or _f(candidate.get("premium", 1.0), 1.0), 0.01)
    spread = max(0.0, ask - bid) if ask and bid else _f(candidate.get("bid_ask_spread", 0.0), 0.0)
    rel_spread = spread / max(price, 1e-6)
    opt_type = str(candidate.get("option_type", candidate.get("right", "call"))).lower()
    side = str(candidate.get("side", "long")).lower()
    vol = _f(candidate.get("realized_vol", candidate.get("iv", candidate.get("implied_volatility", 0.02))), 0.02)
    terminal = simulate_underlying_paths(spot, historical_log_returns, n=1024, conditional_vol=vol)
    ret = option_payoff_returns(terminal, strike, price, opt_type, side)
    stats = summarize_returns(ret, gamma=_f(candidate.get("risk_aversion", 3.0), 3.0))
    delta = abs(_f(candidate.get("delta", 0.5), 0.5))
    elasticity = abs(delta * spot / max(price, 1e-6))
    oi = _f(candidate.get("open_interest", candidate.get("oi", 0.0)), 0.0)
    volume = _f(candidate.get("volume", candidate.get("option_volume", 0.0)), 0.0)
    liquidity = _clip(math.log1p(oi + volume) / 10.0 - rel_spread, 0.0, 1.0)
    delta_neutral = _clip(1.0 - abs(_f(candidate.get("portfolio_delta", delta), delta)), 0.0, 1.0)
    # Utility-first composite. Positive skew helps; kurtosis, spread, and leverage hurt.
    raw = (
        0.35 * stats["certainty_equivalent"]
        + 0.25 * stats["sharpe"]
        + 0.10 * max(stats["skew"], -2.0)
        - 0.06 * max(stats["kurtosis"] - 3.0, 0.0)
        - 0.25 * rel_spread
        - 0.01 * max(elasticity - 5.0, 0.0)
        + 0.15 * liquidity
        + 0.10 * delta_neutral
    )
    return {
        "oops_expected_utility": round(_clip(stats["expected_utility"], -10, 10), 6),
        "oops_certainty_equivalent": round(_clip(stats["certainty_equivalent"], -10, 10), 6),
        "oops_sim_sharpe": round(_clip(stats["sharpe"], -10, 10), 6),
        "oops_skew": round(_clip(stats["skew"], -10, 10), 6),
        "oops_kurtosis": round(_clip(stats["kurtosis"], 0, 50), 6),
        "oops_bid_ask_penalty": round(_clip(rel_spread, 0, 5), 6),
        "oops_elasticity": round(_clip(elasticity, 0, 100), 6),
        "oops_delta_neutrality_score": round(delta_neutral, 6),
        "oops_liquidity_score": round(liquidity, 6),
        "oops_score": round(_clip(raw, -10, 10), 6),
    }


def enrich_event_with_oops(event: Dict[str, Any], historical_log_returns: Sequence[float] | None = None) -> Dict[str, float]:
    """Safe wrapper used by data builders; always returns every OOPS field."""
    try:
        scored = score_option_candidate(event, historical_log_returns=historical_log_returns)
    except Exception:
        scored = {}
    return {field: float(scored.get(field, 0.0)) for field in OOPS_FIELDS}


def utility_adjusted_reward(action: str, realized_r: float, row: Dict[str, Any]) -> float:
    """Reward overlay for alpha-entry RL: entering requires realized edge AND utility quality."""
    action = str(action or "").lower()
    r = _f(realized_r)
    score = _f(row.get("oops_score"))
    ce = _f(row.get("oops_certainty_equivalent"))
    sharpe = _f(row.get("oops_sim_sharpe"))
    spread_pen = _f(row.get("oops_bid_ask_penalty"))
    kurt_pen = max(_f(row.get("oops_kurtosis"), 3.0) - 3.0, 0.0) * 0.02
    utility_bonus = 0.25 * score + 0.20 * ce + 0.10 * sharpe - 0.35 * spread_pen - kurt_pen
    if action == "enter_now":
        return r + utility_bonus if score >= 0.0 and ce >= -0.05 else r - abs(utility_bonus) - 0.25
    if action == "reduce_size":
        return 0.45 * r + 0.20 * utility_bonus - 0.05
    if action == "skip":
        return 0.15 if (r < 0 or score < -0.25) else -0.15 * max(r, 0.0)
    # wait
    return 0.05 * utility_bonus - 0.02 * max(r, 0.0)
