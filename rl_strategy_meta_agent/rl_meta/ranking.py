"""Admission, tiering, composite scoring and the ranking engine."""
from __future__ import annotations

import math

from .types import Metrics, RollingMetrics, RankedStrategy, StrategyMeta, Tier, Status

# Admission thresholds (from the spec).
WIN_RATE_GATE = 0.60
PROFIT_FACTOR_GATE = 2.0

# Validation gates beyond raw admission.
MIN_TRADES = 100           # full-sample minimum for trustworthy admission
MIN_OOS_TRADES = 30        # minimum out-of-sample trades to validate eligibility
OVERFIT_MAX_TRADES = 150   # near-perfect win rate below this many trades -> overfit

# Composite-score weights. Tunable; documented, not magic.
WEIGHTS = {
    "win_rate": 0.20,
    "profit_factor": 0.25,
    "expectancy": 0.15,
    "drawdown": 0.15,       # rewarded for LOW drawdown
    "consistency": 0.10,    # lifetime vs rolling agreement
    "rolling": 0.15,        # recent (30d) strength
}


def tier_of(m: Metrics) -> Tier:
    wr_ok = m.win_rate >= WIN_RATE_GATE
    pf_ok = (m.profit_factor >= PROFIT_FACTOR_GATE) if math.isfinite(m.profit_factor) else True
    if wr_ok and pf_ok:
        return Tier.A
    if wr_ok and not pf_ok:
        return Tier.B
    if not wr_ok and pf_ok:
        return Tier.C
    return Tier.REJECT


def _pf_pass(m: Metrics) -> bool:
    # A non-finite PF (no losing trades) only counts as a pass on a real sample.
    if math.isfinite(m.profit_factor):
        return m.profit_factor >= PROFIT_FACTOR_GATE
    return m.trade_count >= MIN_TRADES


def passes_perf(m: Metrics) -> bool:
    """Performance limb of admission: win_rate>=60% OR profit_factor>=2.0."""
    return m.win_rate >= WIN_RATE_GATE or _pf_pass(m)


def admit(m: Metrics) -> bool:
    """Admission: trade_count>=100 AND (win_rate>=60% OR profit_factor>=2.0)."""
    return m.trade_count >= MIN_TRADES and passes_perf(m)


def validate(m: Metrics, executable: bool = True, has_data: bool = True) -> tuple:
    """Return (eligible, reason) for the full-sample metrics."""
    if not executable:
        return False, "not executable"
    if not has_data:
        return False, "insufficient Alpaca data"
    if m.trade_count < MIN_TRADES:
        return False, f"too few trades ({m.trade_count} < {MIN_TRADES})"
    # Cheap overfitting guard: implausibly perfect record on a thin sample.
    if m.win_rate >= 0.95 and m.trade_count < OVERFIT_MAX_TRADES:
        return False, "obvious overfitting (near-perfect win rate, low n)"
    if not passes_perf(m):
        return False, "failed admission (win_rate<60% and profit_factor<2.0)"
    return True, "eligible"


def oos_pass(oos: Metrics) -> tuple:
    """Out-of-sample gate. Only OOS-passing strategies are RL-eligible."""
    if oos.trade_count < MIN_OOS_TRADES:
        return False, f"too few OOS trades ({oos.trade_count} < {MIN_OOS_TRADES})"
    if not passes_perf(oos):
        return False, "failed out-of-sample validation"
    return True, "oos validated"


def _norm_pf(pf: float) -> float:
    if not math.isfinite(pf):
        return 1.0
    return max(0.0, min(1.0, pf / 4.0))     # PF 4+ saturates


def _consistency(life: Metrics, roll: RollingMetrics) -> float:
    """1.0 when 90d rolling PF tracks lifetime PF; lower when diverging."""
    r = roll.window_90
    if r.trade_count == 0 or life.profit_factor == 0:
        return 0.5
    lp = min(life.profit_factor, 4.0) if math.isfinite(life.profit_factor) else 4.0
    rp = min(r.profit_factor, 4.0) if math.isfinite(r.profit_factor) else 4.0
    return max(0.0, 1.0 - abs(lp - rp) / 4.0)


def composite_score(life: Metrics, roll: RollingMetrics) -> float:
    """0..1 composite. Higher is better. Drawdown rewarded for being low."""
    w = WEIGHTS
    s = (
        w["win_rate"] * min(life.win_rate, 1.0)
        + w["profit_factor"] * _norm_pf(life.profit_factor)
        + w["expectancy"] * (1.0 if life.expectancy > 0 else 0.0)
        + w["drawdown"] * (1.0 - min(life.max_drawdown, 1.0))
        + w["consistency"] * _consistency(life, roll)
        + w["rolling"] * min(roll.window_30.win_rate, 1.0)
    )
    return round(s, 4)


def rank_strategies(rows: list) -> list:
    """rows: list of (StrategyMeta, lifetime Metrics, RollingMetrics).

    Returns RankedStrategy list sorted best-first. Ineligible strategies are
    ranked last (composite score forced low) but still reported.
    """
    scored = []
    for meta, life, roll in rows:
        tier = tier_of(life)
        eligible, reason = validate(life)
        score = composite_score(life, roll) if eligible else 0.0
        scored.append((meta, life, roll, tier, eligible, reason, score))

    scored.sort(key=lambda x: (x[4], x[6]), reverse=True)  # eligible first, then score

    ranked = []
    for i, (meta, life, roll, tier, eligible, reason, score) in enumerate(scored, start=1):
        ranked.append(RankedStrategy(
            strategy_id=meta.strategy_id,
            strategy_name=meta.strategy_name,
            tier=tier.value,
            rank=i,
            composite_score=score,
            eligible=eligible,
            metrics=life,
            rolling=roll,
            reason=reason,
        ))
    return ranked


def rank_results(results: list) -> list:
    """Rank pre-evaluated StrategyResults (eligibility/metrics already decided).

    `results` items must expose: strategy_id, strategy_name, tier, eligible,
    rejection_reason, aggregate_full (Metrics), aggregate_rolling (RollingMetrics).
    """
    scored = []
    for r in results:
        score = composite_score(r.aggregate_full, r.aggregate_rolling) if r.eligible else 0.0
        is_tv = getattr(r, "evaluation_source", "") == "tradingview_verified"
        scored.append((r, score, is_tv))
    # eligible first, then TradingView-verified, then score.
    scored.sort(key=lambda x: (x[0].eligible, x[2], x[1]), reverse=True)

    ranked = []
    for i, (r, score, _is_tv) in enumerate(scored, start=1):
        ranked.append(RankedStrategy(
            strategy_id=r.strategy_id, strategy_name=r.strategy_name,
            tier=r.tier, rank=i, composite_score=score, eligible=r.eligible,
            metrics=r.aggregate_full, rolling=r.aggregate_rolling,
            reason=r.rejection_reason or ("eligible" if r.eligible else None)))
    return ranked


def status_for(eligible: bool) -> str:
    return Status.ELIGIBLE.value if eligible else Status.REJECTED.value


def _demo():
    good = Metrics(trade_count=120, win_rate=0.65, profit_factor=2.4, expectancy=120,
                   max_drawdown=0.12, avg_hold_days=4)
    meh = Metrics(trade_count=120, win_rate=0.50, profit_factor=1.2, expectancy=10,
                  max_drawdown=0.30, avg_hold_days=4)
    assert tier_of(good) == Tier.A
    assert tier_of(meh) == Tier.REJECT
    assert validate(good)[0] is True
    assert validate(meh)[0] is False
    ranked = rank_strategies([
        (StrategyMeta("g", "Good", "h", "2023-01-01", "g.txt"), good, RollingMetrics()),
        (StrategyMeta("m", "Meh", "h", "2023-01-01", "m.txt"), meh, RollingMetrics()),
    ])
    assert ranked[0].strategy_id == "g" and ranked[0].rank == 1
    assert ranked[0].composite_score > ranked[1].composite_score
    print("ranking._demo OK", ranked[0].composite_score)


if __name__ == "__main__":
    _demo()
