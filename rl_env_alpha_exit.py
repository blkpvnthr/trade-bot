"""
rl_env_alpha_exit.py — Offline Gymnasium env for Agent B (Alpha exit management).
Actions: hold | tighten_trail | partial_exit | full_exit | no_overnight | allow_overnight.
Hard exits (stop / kill switch / daily-loss halt / liquidity collapse / regime flip) ALWAYS
override RL — this env only learns timing within those guardrails. OFFLINE only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rl_common as rc

FEATURES = rc.ALPHA_EXIT_FEATURES
ACTIONS = rc.EXIT_ACTIONS


def _synth_rows(n: int = 256) -> List[dict]:
    rows = []
    for i in range(n):
        r = round(((i % 11) - 4) * 0.6, 4)            # spread of realized R (-2.4 .. 3.6)
        peak = round(max(r, r * 1.3, 0.0), 4)
        row = {f: 0.0 for f in FEATURES}
        row.update({"unrealized_R": r, "premium_ratio": 1.0 + 0.1 * r,
                    "black_scholes_score": 0.5, "oops_score": 0.5,
                    "probability_band": 0.6, "market_state_confidence": 0.5,
                    "time_to_close": (i % 8) / 8.0, "overnight_confidence": (i % 5) / 5.0,
                    "realized_R": r, "peak_R": peak, "overnight_loss": (r < 0 and i % 2 == 0)})
        rows.append(row)
    return rows


class AlphaExitEnv(rc.EnvBase):
    metadata = {"render_modes": []}

    def __init__(self, rows: Optional[List[dict]] = None, episode_len: int = 64):
        self.rows = rows if rows is not None else rc.load_dataset("alpha_exit_dataset")
        if not self.rows:
            self.rows = _synth_rows()
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
        self._i = (self._i + 5) % len(self.rows)
        return self._obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        row = self.rows[self._i % len(self.rows)]
        act = ACTIONS[int(action) % len(ACTIONS)]
        reward = rc.alpha_exit_reward(act, rc.to_f(row.get("realized_R")),
                                      rc.to_f(row.get("peak_R")),
                                      overnight_loss=bool(row.get("overnight_loss")))
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
        return "hold"
