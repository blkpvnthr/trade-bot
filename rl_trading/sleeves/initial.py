"""Initial sleeves — all scaffolded, NONE enabled for trading.

These exist so the framework has concrete shapes to carry, and so the dashboard tells the true
story: the best thing we have (the MLP trend ranker) is scanner-only because it does not beat SPY,
and every other sleeve is disabled until a source clears the gate.
"""
from __future__ import annotations

from .core import RiskLimits, Sleeve, ValidationRecord
from .promote import from_extended_validation
from .registry import SleeveRegistry

# Conservative default caps for any future paper sleeve.
DEFAULT_LIMITS = RiskLimits(
    max_notional=10_000.0,
    per_order_notional=2_000.0,
    max_position_pct=0.10,
    max_gross_exposure=1.0,
    max_concurrent=10,
    max_daily_loss=500.0,
)

# The 24-name liquid universe the earnings_within_7d signal was validated on
# (run_extended_validation.py SYMS). Trade only what was validated.
EARNINGS_UNIVERSE = (
    "AAPL", "AMSC", "AMZN", "ARM", "AVGO", "CE", "ENSG", "GOOG", "GOOGL", "GWRE", "IREN", "LULU",
    "META", "MSTR", "NVDA", "ORCL", "QBTS", "RYAN", "SMCI", "TSLA", "TSM", "UNH", "WGS", "WING",
)

# earnings_within_7d caps: long-only, gross <= 50% equity, <= 10 names, <= 10% per name.
EARNINGS_LIMITS = RiskLimits(
    max_notional=10_000.0,
    per_order_notional=2_000.0,
    max_position_pct=0.10,
    max_gross_exposure=0.50,
    max_concurrent=10,
    max_daily_loss=500.0,
)


def trend_ranker_sleeve() -> Sleeve:
    """Current champion MLP trend ranker. SCANNER-ONLY: it does not beat SPY.

    The validation record is honest: it has out-of-sample results but NO positive bootstrap CI on
    excess-vs-SPY, and OOD/placebo/attribution/concentration were not established as passing. The
    gate therefore keeps it scanner-only — which is the correct, truthful outcome.
    """
    record = ValidationRecord(
        source="mlp_trend_ranker",
        ledger_id="supervised_ranker_mlp",
        bootstrap_ci=None,            # no positive excess-vs-SPY CI -> fails gate
        placebo_passed=None,
        oos_passed=True,              # walk-forward OOS was run (Sharpe ~0.90)
        ood_passed=None,
        attribution_passed=None,
        concentration_passed=None,
        metric="excess_return_vs_spy",
        value=-0.27,                  # ~0.90 sharpe vs SPY ~1.17 -> negative excess
        notes="Best research model (Sharpe 0.90) but < passive SPY (1.17). No SPY-beating edge.",
    )
    return Sleeve(
        name="trend_ranker_sleeve",
        signal_source="mlp_trend_ranker",
        native_execution_type="trend_breakout",
        eligible_symbols=("SPY", "QQQ", "AAPL", "MSFT", "NVDA"),  # research universe placeholder
        entry_rules={"rank_top_k": 5, "native_horizon_days": 20, "min_score": 0.0},
        exit_rules={"horizon_days": 20, "stop_r": 1.0, "target_r": 2.0},
        sizing_rules={"scheme": "equal_risk", "risk_per_trade_pct": 0.005},
        risk_limits=DEFAULT_LIMITS,
        required_validation=record,
        paper_enabled=False,
    )


def event_hold_sleeve() -> Sleeve:
    """Generic fixed-horizon event sleeve. DISABLED until a source passes the event-study gate."""
    return Sleeve(
        name="event_hold_sleeve",
        signal_source="<unassigned_event_source>",
        native_execution_type="event_hold",
        eligible_symbols=(),
        entry_rules={"enter": "first_bar_after_public_event", "native_horizon_days": 20},
        exit_rules={"horizon_days": 20, "no_stop": True},
        sizing_rules={"scheme": "equal_weight"},
        risk_limits=DEFAULT_LIMITS,
        required_validation=None,  # no source attached -> DISABLED
        paper_enabled=False,
    )


def earnings_within_7d_sleeve(paper_enabled: bool = False) -> Sleeve:
    """Long names with a SCHEDULED earnings report within the next 7 calendar days; exit at the
    close of the trading day BEFORE the print. The FIRST source to clear the gate.

    Validation is minted mechanically from the canonical extended-history daily-rebalance artifact
    (net Sharpe 1.11@5bps, breakeven 56bps, turnover 0.24/day, bootstrap+placebo p~0, ACCEPT).
    Ships SCANNER-ONLY: paper_enabled defaults False, so mode() == SCANNER_ONLY even though the gate
    passes. Live is impossible. Trades the validated 24-name universe only.
    """
    record = from_extended_validation("earnings_within_7d")
    return Sleeve(
        name="earnings_within_7d_sleeve",
        signal_source="earnings_within_7d",
        native_execution_type="event_hold",
        eligible_symbols=EARNINGS_UNIVERSE,
        entry_rules={
            "enter": "first_day_flagged_earnings_within_7_calendar_days",
            "lookahead_calendar_days": 7,
            "native_horizon_days": 5,
            "forward_scheduled_dates_only": True,  # never estimate/overview snapshots
            "long_only": True,
            "equal_weight": True,
        },
        exit_rules={
            "exit": "close_of_trading_day_before_scheduled_earnings",
            "exit_days_before": 1,
            "never_hold_through_print": True,
            "horizon_days": 5,
        },
        sizing_rules={
            "scheme": "equal_weight_long_only",
            "max_gross_pct": 0.50,
            "max_position_pct": 0.10,
            "cash_buffer_pct": 0.08,
            "fractional": True,
        },
        risk_limits=EARNINGS_LIMITS,
        required_validation=record,
        paper_enabled=paper_enabled,
    )


def macro_timing_sleeve() -> Sleeve:
    """Placeholder for a future macro/breadth timing overlay. DISABLED."""
    return Sleeve(
        name="macro_timing_sleeve",
        signal_source="<future_macro_breadth>",
        native_execution_type="timing_overlay",
        eligible_symbols=("SPY",),
        entry_rules={"overlay": True, "native_horizon_days": None},
        exit_rules={"overlay": True},
        sizing_rules={"scheme": "exposure_scale"},
        risk_limits=DEFAULT_LIMITS,
        required_validation=None,
        paper_enabled=False,
    )


def news_event_sleeve() -> Sleeve:
    """Placeholder for a future News/FinBERT source. DISABLED (deferred research program)."""
    return Sleeve(
        name="news_event_sleeve",
        signal_source="<future_news_finbert>",
        native_execution_type="event_hold",
        eligible_symbols=(),
        entry_rules={"enter": "first_bar_after_news", "native_horizon_days": 5},
        exit_rules={"horizon_days": 5, "no_stop": True},
        sizing_rules={"scheme": "equal_weight"},
        risk_limits=DEFAULT_LIMITS,
        required_validation=None,
        paper_enabled=False,
    )


def legacy_momentum_sleeve() -> Sleeve:
    """Faithful re-emulation of run.py's scanner (legacy_momentum_sleeve.py). SCANNER-ONLY.

    Replay 2018-2026 on the curated universe: Sharpe 0.90 (beats SPY 0.82, ~QQQ 0.91) BUT loses to
    equal-weight of its own universe (1.23) and decays under drop-top-winners (0.87 -> 0.79) =>
    survivorship-dependent, no selection alpha. Gate refuses it.
    """
    record = ValidationRecord(
        source="legacy_run_py_momentum_scanner",
        ledger_id="legacy_momentum_replay_2018_2026",
        bootstrap_ci=None,
        placebo_passed=None, oos_passed=True, ood_passed=False,  # fails drop-top-winners
        attribution_passed=None, concentration_passed=False,
        metric="excess_return_vs_eqw",
        value=-0.33,  # Sharpe 0.90 vs equal-weight 1.23
        notes="Beats SPY but loses to equal-weight of own picks; survivorship-dependent.",
    )
    return Sleeve(
        name="legacy_momentum_sleeve", signal_source="legacy_run_py_momentum_scanner",
        native_execution_type="momentum_reversal_scanner",
        eligible_symbols=("SPY", "QQQ", "AAPL", "MSFT", "NVDA"),
        entry_rules={"min_setup_score": 0.62, "top_n": 8, "native_horizon_days": 20},
        exit_rules={"stop_loss_pct": -0.035, "profit_lock": 0.06, "giveback": 0.025},
        sizing_rules={"scheme": "regime_scaled_target_weight"},
        risk_limits=DEFAULT_LIMITS, required_validation=record, paper_enabled=False,
    )


def legacy_explosive_alpha_sleeve() -> Sleeve:
    """Faithful re-emulation of the explosive-continuation walkforward (Transformer). SCANNER-ONLY.

    OOS on the only 2 windows the legacy data supports: net-negative (Sharpe ~ -1), AUC 0.64-0.70 but
    LOSES to equal-weight of its own picks (1.47) and to SPY/QQQ. Retired for poor chop/downside
    execution (corroborated by per-regime attribution). Chop/downside guards help directionally; a
    fixed take-profit hurts convexity. No deployable edge on available data. See experiments/
    legacy_explosive/REPORT.md.
    """
    record = ValidationRecord(
        source="legacy_explosive_alpha_walkforward",
        ledger_id="legacy_explosive_walkforward_2024_2026",
        bootstrap_ci=None,            # ADR block-CI includes zero (negative)
        placebo_passed=None, oos_passed=False,  # net-negative OOS
        ood_passed=None, attribution_passed=None, concentration_passed=None,
        metric="excess_return_vs_eqw",
        value=-2.0,  # Sharpe ~ -1 vs equal-weight 1.47
        notes="Net-negative OOS; underperforms equal-weight of own picks; 2-window/84-day sample.",
    )
    return Sleeve(
        name="legacy_explosive_alpha_sleeve", signal_source="legacy_explosive_alpha_walkforward",
        native_execution_type="explosive_continuation_walkforward",
        eligible_symbols=(),  # universe is data-driven from daily_features
        entry_rules={"transformer_seq_len": 20, "min_probability": 0.50, "native_horizon_days": 35},
        exit_rules={"hard_stop": -0.085, "chop_giveback_lock": True, "edge_decay": True,
                    "take_profit": "hurts_convexity_disabled"},
        sizing_rules={"scheme": "EV_kelly_convex_vol_regime"},
        risk_limits=DEFAULT_LIMITS, required_validation=record, paper_enabled=False,
    )


def legacy_explosive_signal_sleeve() -> Sleeve:
    """Simplest expression of the legacy Transformer signal (top-5 by probability, fixed 5d hold).
    REJECTED for paper after the robustness/diagnosis gate (experiments/legacy_explosive/DIAGNOSIS.md).

    Verdict: no tradeable edge once survivorship/universe removed — random-5 matched/beat it (83% of
    random draws won), top5 spreads vs random/momentum/eqw all non-significant, 63% of the rank IC is
    static cross-symbol survivorship (rho +0.27), dynamic IC only +0.015 (CI [-0.003,+0.033], p=0.049,
    spans zero), date-shuffle placebo did not collapse. Tested-and-failed -> rejected bucket.
    """
    record = ValidationRecord(
        source="legacy_explosive_transformer_top5",
        ledger_id="legacy_explosive_signal_diagnosis_2026_06_16",
        bootstrap_ci=None,
        placebo_passed=False,        # random-5 matched/beat; real-minus-placebo CI spans zero
        oos_passed=False,            # no tradeable spread over simple baselines
        ood_passed=False,            # collapses under drop-top-2-years
        attribution_passed=False,    # 63% static survivorship, not dynamic prediction
        concentration_passed=False,
        metric="dynamic_rank_ic / top5_minus_random",
        value=0.0008,                # top5-minus-random ADR (non-significant)
        notes="REJECTED: survivorship artifact, no tradeable incremental info. See DIAGNOSIS.md.",
    )
    return Sleeve(
        name="legacy_explosive_signal_sleeve", signal_source="legacy_explosive_transformer_top5",
        native_execution_type="top5_prob_fixed_hold",
        eligible_symbols=(),
        entry_rules={"rank_top_k": 5, "native_horizon_days": 5, "rank_by": "explosive_alpha_probability"},
        exit_rules={"horizon_days": 5, "no_stop": True},
        sizing_rules={"scheme": "equal_weight_or_voltarget"},
        risk_limits=DEFAULT_LIMITS, required_validation=record, paper_enabled=False,
    )


def build_default_registry() -> SleeveRegistry:
    reg = SleeveRegistry(champion="trend_ranker_sleeve")
    reg.add(trend_ranker_sleeve())
    reg.add(earnings_within_7d_sleeve())  # gate PASSES; scanner-only until paper explicitly enabled
    reg.add(event_hold_sleeve())
    reg.add(macro_timing_sleeve())
    reg.add(news_event_sleeve())
    reg.add(legacy_momentum_sleeve())
    reg.add(legacy_explosive_alpha_sleeve())
    reg.add(legacy_explosive_signal_sleeve())
    reg.research_findings = [
        "legacy_explosive_signal: REJECTED (2026-06-16) — 63% of rank IC is static survivorship; "
        "dynamic IC +0.015 CI[-0.003,+0.033] spans zero; random-5 beat top-5 in 83% of draws; no "
        "tradeable spread vs random/momentum/eqw. See experiments/legacy_explosive/DIAGNOSIS.md.",
        "kronos: Stage-0 audit FEASIBLE but PAUSED — information-gate experiment only, not a "
        "deployment candidate. See experiments/kronos/AUDIT.md.",
    ]
    return reg
