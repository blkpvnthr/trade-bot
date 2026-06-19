"""Multi-symbol, walk-forward backtest orchestration.

For one executable strategy: run its adapter across every symbol's real bars,
pool trades for the aggregate, partition by walk-forward split, compute metrics
everywhere, and decide eligibility. Eligibility for RL allocation requires BOTH
the full-sample admission gate AND out-of-sample validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

from .adapters import annotate_executability
from .confidence import strategy_confidence
from .performance import compute_metrics, rolling_metrics
from .ranking import tier_of, validate, oos_pass
from .signals import backtest_executable, NotExecutable
from .splits import split_trades, SPLITS
from .types import Metrics, RollingMetrics, StrategyMeta, Status, Tier

# Per-strategy evaluation source (the single source of truth used for metrics).
SRC_TV = "tradingview_verified"
SRC_ADAPTER = "adapter_screening"
SRC_MISSING = "missing_tv_validation"        # per-symbol marker, not a strategy source
SRC_NOT_EXEC = "parsed_but_not_executable"
EVAL_SOURCES = (SRC_TV, SRC_ADAPTER, SRC_MISSING, SRC_NOT_EXEC)


def allow_adapter_screening_for_rl() -> bool:
    """Config flag: may adapter_screening strategies receive RL eligibility?"""
    return os.environ.get("ALLOW_ADAPTER_SCREENING_FOR_RL", "false").lower() in \
        ("1", "true", "yes", "on")


@dataclass
class StrategyResult:
    strategy_id: str
    strategy_name: str
    adapter_type: str | None
    executable: bool
    timeframe: str
    tested_symbols: list = field(default_factory=list)
    per_symbol: dict = field(default_factory=dict)      # symbol -> {split|full: Metrics}
    aggregate: dict = field(default_factory=dict)        # split|full -> Metrics
    aggregate_full: Metrics = field(default_factory=Metrics)
    aggregate_rolling: RollingMetrics = field(default_factory=RollingMetrics)
    tier: str = Tier.REJECT.value
    eligible: bool = False                              # metrics gate (source-agnostic)
    rl_eligible: bool = False                           # eligible AND allowed for RL capital
    rejection_reason: str = "not evaluated"
    evaluation_source: str = SRC_NOT_EXEC              # the ONE source used for metrics
    sources: dict = field(default_factory=dict)        # symbol -> tradingview|adapter|missing_tv_validation
    tv_symbols: list = field(default_factory=list)     # symbols with version-matched TV imports
    missing_tv_symbols: list = field(default_factory=list)  # universe symbols lacking TV import
    primary_source: str = "adapter"                    # 'tradingview' when TV-verified (back-compat)
    years_tested: float = 0.0                          # calendar span of pooled trades
    strategy_confidence: float = 0.0                   # 0..100 trustworthiness (confidence.py)


def _split_metric_set(trades: list) -> dict:
    by_split = split_trades(trades)
    out = {"full": compute_metrics(trades)}
    for s in SPLITS:
        out[s] = compute_metrics(by_split[s])
    return out


def evaluate_strategy(meta: StrategyMeta, bars_by_symbol: dict, timeframe: str,
                      imported_by_symbol: dict | None = None,
                      universe: list | None = None) -> StrategyResult:
    """Evaluate one strategy from a SINGLE source of truth -- never a mix.

    If any version-matched TradingView import exists for the strategy, the whole
    strategy is `tradingview_verified`: adapter fallback is disabled for ALL
    symbols, only imported trades are used, and universe symbols without an
    import are recorded as `missing_tv_validation` (not filled with adapter
    results). Otherwise an executable strategy is `adapter_screening`; a
    non-executable one with no imports is `parsed_but_not_executable`.
    """
    annotate_executability(meta)
    res = StrategyResult(meta.strategy_id, meta.strategy_name,
                         meta.adapter_type, meta.executable, timeframe)

    bars_by_symbol = bars_by_symbol or {}
    imported = {s: t for s, t in (imported_by_symbol or {}).items() if t}
    have_imported = bool(imported)

    # Reference universe for completeness of TV coverage.
    if universe is None:
        universe = sorted(set(bars_by_symbol) | set(imported))

    if have_imported:
        # SOURCE OF TRUTH: TradingView. No adapter anywhere for this strategy.
        res.evaluation_source = SRC_TV
        res.primary_source = "tradingview"
        res.tv_symbols = sorted(imported)
        res.missing_tv_symbols = sorted(s for s in universe if s not in imported)
        for sym in res.missing_tv_symbols:
            res.sources[sym] = SRC_MISSING
        trades_by_symbol = {}
        for sym in res.tv_symbols:
            trades = imported[sym]
            trades_by_symbol[sym] = trades
            res.per_symbol[sym] = _split_metric_set(trades)
            res.sources[sym] = "tradingview"
    elif meta.executable:
        # SCREENING: Python adapter on real bars.
        res.evaluation_source = SRC_ADAPTER
        res.primary_source = "adapter"
        trades_by_symbol = {}
        for sym in sorted(bars_by_symbol):
            try:
                trades = backtest_executable(meta, bars_by_symbol[sym])
            except NotExecutable:
                continue
            if trades:
                trades_by_symbol[sym] = trades
                res.per_symbol[sym] = _split_metric_set(trades)
                res.sources[sym] = "adapter"
    else:
        res.evaluation_source = SRC_NOT_EXEC
        res.rejection_reason = "not executable"
        return res

    if not trades_by_symbol:
        res.rejection_reason = ("no TradingView trades imported"
                                if res.evaluation_source == SRC_TV
                                else "insufficient Alpaca data")
        return res

    res.tested_symbols = sorted(trades_by_symbol.keys())
    meta.tested_symbols = res.tested_symbols
    meta.tested_timeframe = timeframe

    pooled = [t for ts in trades_by_symbol.values() for t in ts]
    if not pooled:
        res.rejection_reason = "insufficient Alpaca data"
        return res

    res.aggregate = _split_metric_set(pooled)
    res.aggregate_full = res.aggregate["full"]
    res.aggregate_rolling = rolling_metrics(pooled)
    res.tier = tier_of(res.aggregate_full).value

    # Calendar span + trustworthiness score (available even if later rejected).
    entries = [date.fromisoformat(t.entry_date) for t in pooled]
    exits = [date.fromisoformat(t.exit_date) for t in pooled]
    res.years_tested = max((max(exits) - min(entries)).days, 0) / 365.25
    res.strategy_confidence = strategy_confidence(res, universe)

    ok_full, reason_full = validate(res.aggregate_full, executable=True,
                                    has_data=True)
    if not ok_full:
        res.rejection_reason = reason_full
        meta.status = Status.REJECTED.value
        return res

    ok_oos, reason_oos = oos_pass(res.aggregate["oos"])
    res.eligible = ok_oos
    # RL (real-capital) eligibility: TV-verified always; adapter_screening only
    # when explicitly allowed via ALLOW_ADAPTER_SCREENING_FOR_RL.
    rl_ok = res.evaluation_source == SRC_TV or allow_adapter_screening_for_rl()
    res.rl_eligible = ok_oos and rl_ok
    if ok_oos and not rl_ok:
        res.rejection_reason = "adapter_screening not RL-eligible " \
            "(set ALLOW_ADAPTER_SCREENING_FOR_RL=true to allow)"
    else:
        res.rejection_reason = "eligible" if ok_oos else reason_oos
    meta.status = Status.ELIGIBLE.value if ok_oos else Status.REJECTED.value
    return res


def run_backtests(metas: list, bars_by_symbol: dict, timeframe: str,
                  db=None, universe: list | None = None) -> list:
    """Evaluate all strategies. If a PerformanceDB is given, version-matched
    imported TradingView trades make the WHOLE strategy tradingview_verified
    (adapter disabled). Annotates each meta with its evaluation source."""
    out = []
    for m in metas:
        imported = (db.imported_trades(m.strategy_id, timeframe, m.version_hash)
                    if db is not None else None)
        res = evaluate_strategy(m, bars_by_symbol, timeframe, imported, universe)
        m.evaluation_source = res.evaluation_source
        m.tv_validated_symbols = res.tv_symbols
        m.missing_tv_symbols = res.missing_tv_symbols
        m.strategy_confidence = res.strategy_confidence
        out.append(res)
    return out


def persist(db, results: list, on_date: str | None = None):
    """Write per-symbol-per-split and AGGREGATE rows, tagged with the source."""
    on_date = on_date or date.today().isoformat()
    for r in results:
        tf = r.timeframe
        for sym, splits in r.per_symbol.items():
            for split, m in splits.items():
                db.record_cell(r.strategy_id, sym, tf, split, on_date, m,
                               tier=tier_of(m).value, source=r.sources.get(sym, ""))
        # Record symbols missing TV validation (zero-metric markers).
        for sym in r.missing_tv_symbols:
            db.record_cell(r.strategy_id, sym, tf, "full", on_date, Metrics(),
                           source=SRC_MISSING, rejection_reason=SRC_MISSING)
        for split, m in (r.aggregate or {}).items():
            roll = r.aggregate_rolling if split == "full" else RollingMetrics()
            db.record_cell(r.strategy_id, "AGGREGATE", tf, split, on_date, m,
                           rolling=roll, tier=r.tier, eligible=r.rl_eligible,
                           rejection_reason=r.rejection_reason,
                           source=r.evaluation_source,
                           confidence=r.strategy_confidence)


def _demo():
    from .registry import load_strategies, PINE_DIR
    from .signals import synthetic_bars
    meta = load_strategies(PINE_DIR)[0]
    bars = {"SYNTH_A": synthetic_bars(meta), "SYNTH_B": synthetic_bars(meta)}
    res = evaluate_strategy(meta, bars, "1Day")
    # Pooled aggregate trade_count == sum of per-symbol counts.
    per = sum(s["full"].trade_count for s in res.per_symbol.values())
    assert res.aggregate_full.trade_count == per, (res.aggregate_full.trade_count, per)
    assert res.executable and res.adapter_type == "sma_crossover"
    print("backtest._demo OK trades", res.aggregate_full.trade_count,
          "eligible", res.eligible, "/", res.rejection_reason)


if __name__ == "__main__":
    _demo()
