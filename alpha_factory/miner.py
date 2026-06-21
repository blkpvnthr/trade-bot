"""Alpha Miner - generates candidate signal panels from a price/volume panel.

v0 ships a handful of Alpha101-style formulaic candidates plus mutation/combination of accepted
alphas, so the Scientist always has something to chew and the loop runs end-to-end.

ponytail: v0 emits ~dozens of candidates, not the aspirational 5000/day. Scale up = add formulas
here or wire rl_strategy_meta_agent's AlphaEvolve/gplearn miners as candidate sources. The point of
v0 is the *validation* gate; the generator is deliberately thin until the gate is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Candidate:
    name: str
    source: str
    formula: str
    signals: pd.DataFrame
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1D"
    created_at: str = ""

    def __post_init__(self):
        if not self.symbols:
            self.symbols = list(self.signals.columns)


def _z(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


# --- Alpha101-style formulaic base signals --------------------------------

def base_signals(closes: pd.DataFrame, vol: pd.DataFrame, created_at: str = "") -> List[Candidate]:
    ret1 = closes.pct_change()
    formulas = {
        "rev_1d":      ("-1 * return_1d (short-term reversal)", -ret1),
        "rev_5d":      ("-1 * return_5d", -(closes.pct_change(5))),
        "mom_20d":     ("close/close[-20] - 1 (momentum)", closes.pct_change(20)),
        "mom_60d":     ("close/close[-60] - 1", closes.pct_change(60)),
        "vol_5d":      ("-1 * std(return,5) (low-vol)", -ret1.rolling(5).std()),
        "volu_z":      ("-1 * zscore(volume) (illiquidity)", -_z(vol)),
        "ts_rank_ret": ("-1 * ts_rank(return_1d,10)", -ret1.rolling(10).apply(
                            lambda x: pd.Series(x).rank().iloc[-1], raw=False)),
        "dist_sma20":  ("-1 * (close/sma20 - 1) (mean-revert)", -(closes / closes.rolling(20).mean() - 1)),
    }
    return [Candidate(name=k, source="alpha101", formula=f, signals=s, created_at=created_at)
            for k, (f, s) in formulas.items()]


def mutate(cand: Candidate, created_at: str = "") -> Candidate:
    """v0 mutation = rank-then-square-sign (a monotone-PRESERVING tail emphasis). Cheap, real."""
    s = cand.signals
    mutated = np.sign(s) * (s.rank(axis=1, pct=True) - 0.5).abs()
    return Candidate(name=f"{cand.name}_mut", source="mutation",
                     formula=f"mutate({cand.formula})", signals=mutated, created_at=created_at)


def combine(a: Candidate, b: Candidate, created_at: str = "") -> Candidate:
    """Equal-weight z-score blend of two base signals (independence-seeking)."""
    s = (_z(a.signals) + _z(b.signals)) / 2
    return Candidate(name=f"{a.name}+{b.name}", source="combination",
                     formula=f"z({a.name}) + z({b.name})", signals=s, created_at=created_at)


def planted(factor: pd.DataFrame, created_at: str = "") -> Candidate:
    """The synthetic panel's true factor - a guaranteed-real alpha (oracle). Test/demo only."""
    return Candidate(name="planted_factor", source="manual",
                     formula="hidden factor f_t (oracle)", signals=factor, created_at=created_at)


def discover(closes: pd.DataFrame, vol: pd.DataFrame, created_at: str = "",
             factor: Optional[pd.DataFrame] = None) -> List[Candidate]:
    cands = base_signals(closes, vol, created_at)
    cands += [mutate(c, created_at) for c in cands[:3]]
    cands.append(combine(cands[0], cands[2], created_at))   # rev_1d + mom_20d
    if factor is not None:
        cands.append(planted(factor, created_at))
    return cands
