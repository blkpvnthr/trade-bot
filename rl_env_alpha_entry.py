"""
rl_env_alpha_entry.py — Offline Gymnasium env for Agent A (Alpha entry timing).
Actions: enter_now | wait | skip | reduce_size  (advisory only — never submits orders).
OFFLINE/SIMULATED: does not import AlpacaGateway and never places an order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rl_common as rc
from oops_option_utility import OOPS_FIELDS, utility_adjusted_reward

FEATURES = list(dict.fromkeys(list(rc.ALPHA_ENTRY_FEATURES) + OOPS_FIELDS))
ACTIONS = rc.ENTRY_ACTIONS


def _synth_rows(n: int = 256) -> List[dict]:
    rows = []
    for i in range(n):
        pop = (i % 10) / 10.0
        realized_R = round(pop * 3.0 - (1.0 - pop) * 1.0, 4)
        row = {f: 0.0 for f in FEATURES}
        row.update({"setup_score": 0.4 + 0.06 * (i % 10), "probability_of_profit": pop,
                    "black_scholes_score": pop, "oops_score": pop, "optionlab_score": pop,
                    "expected_R": realized_R, "market_state_confidence": 0.5 + 0.05 * (i % 5),
                    "time_of_day": (i % 8) / 8.0, "realized_R": realized_R})
        rows.append(row)
    return rows


class AlphaEntryEnv(rc.EnvBase):
    """One historical Alpha entry decision per step (contextual-bandit episodes)."""

    metadata = {"render_modes": []}

    def __init__(self, rows: Optional[List[dict]] = None, episode_len: int = 64):
        super().__init__() if hasattr(rc.EnvBase, "__init__") else None
        self.rows = rows if rows is not None else rc.load_dataset("alpha_entry_dataset")
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
        self._i = (self._i + 7) % len(self.rows)   # deterministic rotation (no RNG)
        return self._obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        row = self.rows[self._i % len(self.rows)]
        act = ACTIONS[int(action) % len(ACTIONS)]
        base_reward = rc.alpha_entry_reward(act, rc.to_f(row.get("realized_R")))
        # OOPS overlay: do not teach the entry agent to buy contracts whose simulated
        # utility is poor after bid/ask, skew/kurtosis, and leverage penalties.
        reward = 0.50 * float(base_reward) + 0.50 * utility_adjusted_reward(act, rc.to_f(row.get("realized_R")), row)
        self._i += 1
        self._step_n += 1
        terminated = False
        truncated = self._step_n >= self.episode_len
        return self._obs(), float(reward), terminated, truncated, {"action": act}

    def close(self):
        pass


def recommend(model: Any, obs_row: Dict[str, Any]) -> str:
    """Map a feature dict → recommended action via a trained model (advisory)."""
    vec = np.array(rc.normalize_obs([rc.to_f(obs_row.get(f)) for f in FEATURES]), dtype=np.float32)
    # Guard against a stale model whose observation width no longer matches the env's
    # FEATURES (e.g. after a feature family is added without retraining). Previously this was
    # silently swallowed and returned "wait" for EVERY row → the policy looked like a total
    # no-trade collapse (rl_exp=0.0) when it was really a width mismatch. Surface it loudly.
    try:
        exp_w = getattr(getattr(model, "observation_space", None), "shape", (None,))[0]
        if exp_w is not None and exp_w != vec.shape[0]:
            import logging
            logging.getLogger(__name__).error(
                "alpha_entry model/env width mismatch: model=%s env=%s — RETRAIN required; "
                "falling back to 'wait'", exp_w, vec.shape[0])
            return "wait"
        a, _ = model.predict(vec, deterministic=True)
        return ACTIONS[int(a) % len(ACTIONS)]
    except Exception as exc:  # pragma: no cover - defensive; never break the eval path
        import logging
        logging.getLogger(__name__).warning("alpha_entry recommend() failed: %s", exc)
        return "wait"
