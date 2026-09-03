"""
rl_env_capital_allocation.py — Offline Gymnasium env for Agent D (capital allocation).
Continuous action = raw weights over [alpha, buffett, cash] (softmax-normalised). Reward =
portfolio log-return + Sharpe/Sortino components − drawdown − turnover. ADVISORY ONLY — the
Adaptive Capital Allocator remains the execution-facing allocator. OFFLINE only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rl_common as rc

FEATURES = rc.ALLOC_FEATURES
BUCKETS = rc.ALLOC_BUCKETS


def _synth_rows(n: int = 128) -> List[dict]:
    rows = []
    for i in range(n):
        a = ((i % 7) - 3) * 0.004        # alpha bucket return swings sign
        b = 0.001 + (i % 5) * 0.0005
        rows.append({f: 0.0 for f in FEATURES} | {
            "alpha_expectancy": a * 50, "alpha_perf": a * 100, "buffett_quality": b * 100,
            "market_state_confidence": 0.5, "drawdown": abs(min(0.0, a)) * 5,
            "volatility": 0.01 + (i % 4) * 0.003, "waterfall_ratio": 0.5,
            "optimizer_perf": b * 100,
            "alpha_bucket_return": round(a, 6), "buffett_bucket_return": round(b, 6),
            "cash_bucket_return": 0.0001})
    return rows


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) - np.max(x)
    e = np.exp(np.clip(x, -20, 20))
    s = e.sum()
    return (e / s) if s > 0 else np.ones_like(e) / len(e)


class CapitalAllocationEnv(rc.EnvBase):
    metadata = {"render_modes": []}

    def __init__(self, rows: Optional[List[dict]] = None, episode_len: int = 32):
        self.rows = rows if rows is not None else rc.load_dataset("capital_allocation_dataset")
        if not self.rows or len(self.rows) < 4:
            self.rows = (self.rows or []) + _synth_rows()
        self.episode_len = max(1, min(episode_len, len(self.rows)))
        self.observation_space = rc.spaces.Box(low=-10.0, high=10.0,
                                               shape=(len(FEATURES),), dtype=np.float32)
        # raw logits over the 3 buckets → softmax to weights
        self.action_space = rc.spaces.Box(low=-5.0, high=5.0, shape=(len(BUCKETS),), dtype=np.float32)
        self._i = 0
        self._step_n = 0
        self._prev_w = np.ones(len(BUCKETS)) / len(BUCKETS)

    def _obs(self) -> np.ndarray:
        row = self.rows[self._i % len(self.rows)]
        return np.array(rc.normalize_obs([rc.to_f(row.get(f)) for f in FEATURES]), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self._step_n = 0
        self._i = (self._i + 1) % len(self.rows)
        self._prev_w = np.ones(len(BUCKETS)) / len(BUCKETS)
        return self._obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        row = self.rows[self._i % len(self.rows)]
        w = _softmax(np.asarray(action, dtype=np.float64).reshape(-1)[:len(BUCKETS)])
        bucket_returns = [rc.to_f(row.get("alpha_bucket_return")),
                          rc.to_f(row.get("buffett_bucket_return")),
                          rc.to_f(row.get("cash_bucket_return"))]
        turnover = float(np.abs(w - self._prev_w).sum())
        reward = rc.capital_alloc_reward(list(w), bucket_returns,
                                         drawdown=rc.to_f(row.get("drawdown")), turnover=turnover)
        self._prev_w = w
        self._i += 1
        self._step_n += 1
        truncated = self._step_n >= self.episode_len
        return self._obs(), float(reward), False, truncated, {
            "weights": {BUCKETS[k]: round(float(w[k]), 4) for k in range(len(BUCKETS))}}

    def close(self):
        pass


def recommend(model: Any, obs_row: Dict[str, Any]) -> Dict[str, float]:
    """Return advisory bucket weights {alpha, buffett, cash} from a trained model."""
    vec = np.array(rc.normalize_obs([rc.to_f(obs_row.get(f)) for f in FEATURES]), dtype=np.float32)
    try:
        a, _ = model.predict(vec, deterministic=True)
        w = _softmax(np.asarray(a, dtype=np.float64).reshape(-1)[:len(BUCKETS)])
        return {BUCKETS[k]: round(float(w[k]), 4) for k in range(len(BUCKETS))}
    except Exception:
        return {b: round(1.0 / len(BUCKETS), 4) for b in BUCKETS}
