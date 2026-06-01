"""
rl_env_buffett_dca.py — Offline Gymnasium env for Agent C (Buffett DCA timing).
Actions: no_dca | small_dca | normal_dca | aggressive_dca | defer.
RL cannot bypass the BAL quality gate, valuation gate, thesis-broken check, the optimized
target-weight cap, or RiskManager — it only re-times accumulation. OFFLINE only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rl_common as rc

FEATURES = rc.BUFFETT_DCA_FEATURES
ACTIONS = rc.DCA_ACTIONS


def _synth_rows(n: int = 256) -> List[dict]:
    rows = []
    for i in range(n):
        discount = (i % 10) / 10.0
        conv = 0.4 + 0.05 * (i % 10)
        base = 0.5 * discount + 0.3 * conv
        row = {f: 0.0 for f in FEATURES}
        row.update({"bal_score": 0.6 + 0.03 * (i % 10), "conviction_score": conv,
                    "intrinsic_value_discount": discount, "composite_dca_score": 0.5 + 0.04 * (i % 10),
                    "optimized_target_weight": 0.08, "current_weight": 0.02 + 0.005 * (i % 8),
                    "market_state_confidence": 0.5,
                    "fwd_return_60d": round(base, 4), "valuation_improvement": round(discount, 4)})
        rows.append(row)
    return rows


class BuffettDCAEnv(rc.EnvBase):
    metadata = {"render_modes": []}

    def __init__(self, rows: Optional[List[dict]] = None, episode_len: int = 64,
                 horizon: str = "fwd_return_60d"):
        self.rows = rows if rows is not None else rc.load_dataset("buffett_dca_dataset")
        if not self.rows:
            self.rows = _synth_rows()
        self.horizon = horizon
        self.episode_len = max(1, min(episode_len, len(self.rows)))
        self.observation_space = rc.spaces.Box(low=-10.0, high=10.0,
                                               shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = rc.spaces.Discrete(len(ACTIONS))
        self._i = 0
        self._step_n = 0

    def _obs(self) -> np.ndarray:
        row = self.rows[self._i % len(self.rows)]
        return np.array(rc.normalize_obs([rc.to_f(row.get(f)) for f in FEATURES]), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self._step_n = 0
        self._i = (self._i + 3) % len(self.rows)
        return self._obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        row = self.rows[self._i % len(self.rows)]
        act = ACTIONS[int(action) % len(ACTIONS)]
        reward = rc.buffett_dca_reward(
            act, rc.to_f(row.get(self.horizon, row.get("fwd_return_60d"))),
            rc.to_f(row.get("valuation_improvement")),
            rc.to_f(row.get("current_weight")), rc.to_f(row.get("optimized_target_weight")))
        self._i += 1
        self._step_n += 1
        truncated = self._step_n >= self.episode_len
        return self._obs(), float(reward), False, truncated, {"action": act}

    def close(self):
        pass


def recommend(model: Any, obs_row: Dict[str, Any]) -> str:
    vec = np.array(rc.normalize_obs([rc.to_f(obs_row.get(f)) for f in FEATURES]), dtype=np.float32)
    try:
        a, _ = model.predict(vec, deterministic=True)
        return ACTIONS[int(a) % len(ACTIONS)]
    except Exception:
        return "no_dca"
