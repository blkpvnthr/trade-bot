"""Nested, purged, embargoed cross-validation infrastructure for Objective 2.

Outer loop = Objective 1's purged/embargoed walk-forward windows (unchanged
-- see walkforward.make_windows). Inner loop = an additional purged/
embargoed split WITHIN each outer window's training period, used to select
among a small, pre-specified set of candidate configurations before the
outer model is fit on the full outer-train period and scored exactly once
on the outer-test fold. Outer-test data must never influence a selection
made here.

This module also hosts the trial ledger: every materially different
strategy configuration whose outer-test performance gets inspected is
recorded here, on disk, so deflated-Sharpe calculations across a whole
Objective 2 session use the actual cumulative trial count rather than an
understated n_trials=1 for each one in isolation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def split_inner_fold(
    dates: Sequence[pd.Timestamp],
    val_fraction: float = 0.2,
    embargo_days: int = 5,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Purge an embargo gap between an inner-train prefix and inner-val
    suffix of `dates`. `dates` must already be just the OUTER window's
    training dates -- this never sees the outer-test dates, so whatever is
    selected using inner_val can't have leaked from the outer test fold.
    """
    dates = list(dates)
    n = len(dates)
    if n < 10:
        raise ValueError(f"Not enough dates ({n}) to carve an inner fold.")

    val_days = max(1, int(round(n * val_fraction)))
    val_start_idx = n - val_days
    embargo_start_idx = max(0, val_start_idx - embargo_days)

    inner_train = dates[:embargo_start_idx]
    inner_val = dates[val_start_idx:]

    if not inner_train or not inner_val:
        raise ValueError(f"Inner split produced an empty fold (n={n}, val_days={val_days}).")

    return inner_train, inner_val


def select_best_config(
    candidate_configs: Sequence[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
) -> tuple[dict[str, Any], float, list[tuple[dict[str, Any], float]]]:
    """Evaluate score_fn against each candidate using inner-val data only
    (score_fn is the caller's closure over whatever inner-val predictions
    it has already computed) and return the winner plus the full score
    table, so the selection is auditable rather than a black box.
    """
    scored = [(cfg, float(score_fn(cfg))) for cfg in candidate_configs]
    best_cfg, best_score = max(scored, key=lambda x: x[1])
    return best_cfg, best_score, scored


# =============================================================================
# Trial ledger
# =============================================================================

@dataclass
class Trial:
    trial_id: str
    description: str
    config: dict[str, Any]
    outer_window_net_sharpes: list[float] = field(default_factory=list)
    holdout_net_sharpe: float | None = None
    n_outer_windows: int = 0
    pooled_net_sharpe: float | None = None
    notes: str = ""


@dataclass
class TrialLedger:
    trials: list[Trial] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        """Cumulative count of materially different strategy configurations
        whose outer-test performance has been inspected. This is the
        n_trials input to metrics.deflated_sharpe_ratio."""
        return len(self.trials)

    def record(
        self,
        trial_id: str,
        description: str,
        config: dict[str, Any],
        outer_window_net_sharpes: list[float],
        holdout_net_sharpe: float | None = None,
        pooled_net_sharpe: float | None = None,
        notes: str = "",
    ) -> Trial:
        if any(t.trial_id == trial_id for t in self.trials):
            raise ValueError(f"trial_id {trial_id!r} already recorded -- use a new id or load/reuse the existing trial.")
        trial = Trial(
            trial_id=trial_id,
            description=description,
            config=config,
            outer_window_net_sharpes=list(outer_window_net_sharpes),
            holdout_net_sharpe=holdout_net_sharpe,
            n_outer_windows=len(outer_window_net_sharpes),
            pooled_net_sharpe=pooled_net_sharpe,
            notes=notes,
        )
        self.trials.append(trial)
        return trial

    def to_dict(self) -> dict[str, Any]:
        return {"n_trials": self.n_trials, "trials": [asdict(t) for t in self.trials]}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "TrialLedger":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        ledger = cls()
        for t in data.get("trials", []):
            ledger.trials.append(Trial(**t))
        return ledger
