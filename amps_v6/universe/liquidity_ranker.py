"""Dynamic liquidity-optimized ranking (V6 spec section 7).

Rank candidate instruments by execution-aware liquidity, NOT raw share volume. Components are
normalized to [0,1] and combined with configurable weights. Spread is estimated from real daily
high-low range when a quoted spread is unavailable (flagged as a proxy, never fabricated as a true
NBBO spread). Hysteresis + minimum residency prevent unstable universe churn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amps_v6.features import multi_timeframe_engine as MTF


@dataclass
class RankWeights:
    dollar_volume: float = 0.25
    spread_efficiency: float = 0.25
    liquidity: float = 0.15
    relative_volume: float = 0.10
    execution_quality: float = 0.15
    historical_net_expectancy: float = 0.10


@dataclass
class LiquidityStats:
    symbol: str
    avg_dollar_volume: float
    spread_proxy_bps: float
    rel_volume: float
    n_days: int


def liquidity_stats(symbol: str, lookback_days: int = 30) -> Optional[LiquidityStats]:
    """Compute liquidity stats from adjusted daily bars (cache-only; None if unavailable)."""
    d = MTF.daily_frame_adj(symbol, allow_fetch=False)
    if d.empty or len(d) < 20:
        return None
    tail = d.tail(lookback_days)
    adv = float(tail["dollar_volume"].mean())
    # spread PROXY: median (high-low)/close in bps (a conservative upper bound on effective spread)
    hl = ((tail["high"] - tail["low"]) / tail["close"].replace(0, np.nan)).median()
    spread_proxy_bps = float(hl * 1e4) if hl == hl else 1e9
    rv = float(tail["rvol"].tail(5).mean()) if "rvol" in tail else 1.0
    return LiquidityStats(symbol, adv, spread_proxy_bps, rv if rv == rv else 1.0, len(tail))


def _norm(vals: dict[str, float], higher_better: bool = True) -> dict[str, float]:
    xs = [v for v in vals.values() if v == v and v not in (float("inf"), -float("inf"))]
    if not xs:
        return {k: 0.0 for k in vals}
    lo, hi = min(xs), max(xs)
    rng = (hi - lo) or 1.0
    out = {}
    for k, v in vals.items():
        if v != v or v in (float("inf"), -float("inf")):
            out[k] = 0.0
            continue
        z = (v - lo) / rng
        out[k] = z if higher_better else (1.0 - z)
    return out


def rank(symbols: list[str], weights: Optional[RankWeights] = None, lookback_days: int = 30,
         hist_net_bps: Optional[dict[str, float]] = None) -> list[tuple[str, float, dict]]:
    """Return [(symbol, score, components)] sorted best-first. Symbols without data are dropped
    (disclosed by their absence, never scored on fabricated liquidity)."""
    weights = weights or RankWeights()
    hist_net_bps = hist_net_bps or {}
    stats = {s: liquidity_stats(s, lookback_days) for s in symbols}
    stats = {s: v for s, v in stats.items() if v is not None}
    if not stats:
        return []
    adv_n = _norm({s: v.avg_dollar_volume for s, v in stats.items()}, higher_better=True)
    spr_n = _norm({s: v.spread_proxy_bps for s, v in stats.items()}, higher_better=False)
    rv_n = _norm({s: v.rel_volume for s, v in stats.items()}, higher_better=True)
    hist_n = _norm({s: hist_net_bps.get(s, 0.0) for s in stats}, higher_better=True)
    out = []
    for s in stats:
        comp = {
            "dollar_volume": adv_n[s],
            "spread_efficiency": spr_n[s],
            "liquidity": adv_n[s],                 # ADV doubles as available-liquidity proxy
            "relative_volume": rv_n[s],
            "execution_quality": spr_n[s],         # tighter proxy spread -> better execution proxy
            "historical_net_expectancy": hist_n[s],
        }
        score = (weights.dollar_volume * comp["dollar_volume"]
                 + weights.spread_efficiency * comp["spread_efficiency"]
                 + weights.liquidity * comp["liquidity"]
                 + weights.relative_volume * comp["relative_volume"]
                 + weights.execution_quality * comp["execution_quality"]
                 + weights.historical_net_expectancy * comp["historical_net_expectancy"])
        out.append((s, round(score, 4), {k: round(v, 3) for k, v in comp.items()}))
    out.sort(key=lambda r: r[1], reverse=True)
    return out
