"""
research_platform — a reproducible champion/challenger research machine.

Turns every trading decision into a permanently reproducible research observation:
  registry  : immutable, content-addressed experiment specs (config + git + data snapshot + preregistered criteria)
  events    : append-only decision ledger with PIT feature snapshots + realized/modeled fill tags
  fills     : ONE shared fill/slippage model (so champion vs challenger is apples-to-apples)
  portfolios: champion/challenger shadow portfolios as filter+weight over a shared candidate pool
  promotion : evaluate challengers against PREREGISTERED criteria (significance + per-regime + burn-in)
  report    : nightly markdown ranking challengers with promote/reject/continue verdicts

Design rule: this layer NEVER generates signals. It consumes candidate decisions and evaluates them.
"""
from .registry import ExperimentSpec, Registry, data_snapshot_hash, git_commit
from .events import DecisionEvent, EventLog, make_event_id
from .fills import FillModel
from .portfolios import PortfolioConfig, run_portfolio, PortfolioResult, port_stats, daily_series
from .promotion import PromotionCriteria, evaluate
from .report import nightly_report

__all__ = [
    "ExperimentSpec", "Registry", "data_snapshot_hash", "git_commit",
    "DecisionEvent", "EventLog", "make_event_id", "FillModel",
    "PortfolioConfig", "run_portfolio", "PortfolioResult", "port_stats", "daily_series",
    "PromotionCriteria", "evaluate", "nightly_report",
]
