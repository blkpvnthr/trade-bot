"""strategy_confidence: how TRUSTWORTHY is the evidence for a strategy?

This is a deliberately separate axis from `composite_score` (which answers "how
GOOD does the strategy look?"). A strategy can look excellent yet rest on thin
evidence -- 120 trades on one symbol with no TradingView verification. That is
high quality but LOW confidence, and the RL allocator should see both.

Pure function of an already-evaluated StrategyResult. No I/O, no side effects.
Source strings mirror backtest.EVAL_SOURCES (kept as literals to avoid an import
cycle: backtest imports confidence, not the other way round).
"""
from __future__ import annotations

import math

SRC_TV = "tradingview_verified"
SRC_ADAPTER = "adapter_screening"
SRC_NOT_EXEC = "parsed_but_not_executable"

# Caps: the most evidence each source can ever be worth.
SOURCE_CAP = {SRC_TV: 1.0, SRC_ADAPTER: 0.30, SRC_NOT_EXEC: 0.0}

TRADE_TARGET = 500.0      # trades for full trade_score
YEARS_TARGET = 5.0        # years for full years_score
OOS_TARGET = 30.0         # OOS trades for full stability weight


def _pf_cap(pf: float) -> float:
    if not math.isfinite(pf):
        return 4.0
    return max(0.0, min(pf, 4.0))


def _oos_stability(result) -> float:
    """0..1 agreement between full-sample and OOS metrics, scaled down when the
    OOS sample is thin. inf profit factors are handled via _pf_cap."""
    agg = getattr(result, "aggregate", None) or {}
    full = result.aggregate_full
    oos = agg.get("oos")
    if oos is None or oos.trade_count == 0:
        return 0.0
    pf_term = 1.0 - min(abs(_pf_cap(full.profit_factor) - _pf_cap(oos.profit_factor))
                        / 4.0, 1.0)
    wr_term = 1.0 - min(abs(full.win_rate - oos.win_rate), 1.0)
    base = 0.5 * pf_term + 0.5 * wr_term
    n_factor = min(oos.trade_count / OOS_TARGET, 1.0)   # thin OOS -> less trust
    return base * n_factor


def strategy_confidence(result, universe=None) -> float:
    """Return a 0..100 trustworthiness score for an evaluated StrategyResult."""
    src = getattr(result, "evaluation_source", SRC_NOT_EXEC)
    cap = SOURCE_CAP.get(src, 0.0)
    if cap == 0.0:
        return 0.0
    if result.aggregate_full.trade_count == 0:
        return 0.0

    # Universe size for coverage ratios (fall back to observed symbols).
    n_uni = len(universe) if universe else max(
        len(result.tested_symbols) + len(result.missing_tv_symbols), 1)

    tv_coverage = (min(len(result.tv_symbols) / n_uni, 1.0)
                   if src == SRC_TV else 0.0)
    symbol_coverage = min(len(result.tested_symbols) / n_uni, 1.0)
    trade_score = min(result.aggregate_full.trade_count / TRADE_TARGET, 1.0)
    years_score = min(getattr(result, "years_tested", 0.0) / YEARS_TARGET, 1.0)
    oos_stability = _oos_stability(result)

    confidence_raw = (0.25 * tv_coverage
                      + 0.20 * symbol_coverage
                      + 0.20 * trade_score
                      + 0.15 * years_score
                      + 0.20 * oos_stability)
    return round(max(0.0, min(100.0, 100.0 * confidence_raw * cap)), 1)


class _FakeResult:
    """Minimal duck-typed result for the self-check."""
    def __init__(self, **kw):
        self.evaluation_source = kw.get("src", SRC_TV)
        self.tv_symbols = kw.get("tv_symbols", [])
        self.tested_symbols = kw.get("tested_symbols", [])
        self.missing_tv_symbols = kw.get("missing_tv_symbols", [])
        self.years_tested = kw.get("years_tested", 0.0)
        self.aggregate_full = kw["aggregate_full"]
        self.aggregate = kw.get("aggregate", {})


def _demo():
    from .types import Metrics
    uni = list("ABCDEFGH")
    strong_full = Metrics(trade_count=2500, win_rate=0.66, profit_factor=2.6)
    strong_oos = Metrics(trade_count=400, win_rate=0.65, profit_factor=2.5)
    strong = _FakeResult(src=SRC_TV, tv_symbols=uni, tested_symbols=uni,
                         years_tested=5.0, aggregate_full=strong_full,
                         aggregate={"oos": strong_oos})
    thin = _FakeResult(src=SRC_TV, tv_symbols=["A"], tested_symbols=["A"],
                       missing_tv_symbols=uni[1:], years_tested=1.0,
                       aggregate_full=Metrics(trade_count=120, win_rate=0.9,
                                              profit_factor=float("inf")),
                       aggregate={"oos": Metrics()})
    adapter = _FakeResult(src=SRC_ADAPTER, tested_symbols=uni, years_tested=5.0,
                          aggregate_full=strong_full, aggregate={"oos": strong_oos})

    cs = strategy_confidence(strong, uni)
    ct = strategy_confidence(thin, uni)
    ca = strategy_confidence(adapter, uni)
    assert cs >= 70, cs
    assert ct < cs, (ct, cs)              # thin evidence -> lower
    assert ca <= 30.0, ca                 # adapter capped
    assert strategy_confidence(_FakeResult(src=SRC_NOT_EXEC,
            aggregate_full=Metrics()), uni) == 0.0
    print(f"confidence._demo OK strong={cs} thin={ct} adapter={ca}")


if __name__ == "__main__":
    _demo()
