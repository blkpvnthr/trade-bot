"""Baseline allocation policies + a policy-comparison harness.

Answers the validation question "did the RL allocator beat naive baselines?" by
running the trained PPO policy and several rule-based baselines through the SAME
env episodes (identical seeds -> identical simulated market + per-strategy
returns), so the only difference is the allocation decision.

Read-only over the env: no observation/reward/risk changes. A policy is a
callable (env, obs, t) -> discrete action.
"""
from __future__ import annotations

import numpy as np

from .env import (A_CASH, A_A_SMALL, A_A_MED, A_B_SMALL, A_C_SMALL, A_REDUCE)


# -- rule-based baselines --------------------------------------------------
def p_cash(env, obs, t):
    return A_CASH                                   # never allocate


def p_fund_top(env, obs, t):
    return A_A_MED                                  # always medium into top-ranked (composite)


def p_fund_small_top(env, obs, t):
    return A_A_SMALL                                # always small into top-ranked


def p_equal_weight(env, obs, t):
    return (A_A_SMALL, A_B_SMALL, A_C_SMALL)[t % 3]  # spread across tiers over time


def p_reduce_bias(env, obs, t):
    return A_A_SMALL if t % 2 == 0 else A_REDUCE     # conservative on/off


def make_random_policy(seed: int = 123):
    rng = np.random.default_rng(seed)
    return lambda env, obs, t: int(rng.integers(0, env.action_space.n))


def make_rl_policy(model):
    def pol(env, obs, t):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return pol


# -- evaluation ------------------------------------------------------------
def evaluate_policy(env, policy, episodes: int = 50, base_seed: int = 0) -> dict:
    """Run `policy` for N episodes (seeds base_seed..base_seed+N-1). Returns
    risk/return stats aggregated across episodes."""
    rew, rets, dds, blow = [], [], [], 0
    for e in range(episodes):
        obs, _ = env.reset(seed=base_seed + e)
        eqs = [env.equity]
        total, done, t = 0.0, False, 0
        while not done:
            obs, r, term, trunc, info = env.step(policy(env, obs, t))
            total += r
            eqs.append(info["equity"])
            t += 1
            done = term or trunc
        eqs = np.asarray(eqs, float)
        peak = np.maximum.accumulate(eqs)
        dds.append(float(((peak - eqs) / peak).max()))
        rew.append(total)
        rets.append(eqs[-1] / env.start_equity - 1.0)
        blow += int(term)                            # terminated == drawdown blowup
    mret, mdd = float(np.mean(rets)), float(np.mean(dds))
    return {
        "mean_reward": round(float(np.mean(rew)), 3),
        "mean_return": round(mret, 4),
        "mean_maxdd": round(mdd, 4),
        "blowup_rate": round(blow / episodes, 3),
        "ret_per_dd": round(mret / (mdd + 1e-9), 3),
    }


def compare_policies(env, rl_model=None, episodes: int = 50,
                     base_seed: int = 0) -> dict:
    """Evaluate every baseline (and the RL policy, if given) on the same env."""
    out = {
        "cash": evaluate_policy(env, p_cash, episodes, base_seed),
        "random": evaluate_policy(env, make_random_policy(), episodes, base_seed),
        "fund_top (composite)": evaluate_policy(env, p_fund_top, episodes, base_seed),
        "fund_small_top": evaluate_policy(env, p_fund_small_top, episodes, base_seed),
        "equal_weight_rr": evaluate_policy(env, p_equal_weight, episodes, base_seed),
    }
    if rl_model is not None:
        out["RL (PPO)"] = evaluate_policy(env, make_rl_policy(rl_model), episodes,
                                          base_seed)
    return out


def _demo():
    from .env import StrategyMetaEnv, make_demo_views
    env = StrategyMetaEnv(make_demo_views(), horizon=40)
    res = compare_policies(env, rl_model=None, episodes=8)
    assert "cash" in res and "fund_top (composite)" in res
    assert all(np.isfinite(v["mean_reward"]) for v in res.values())
    print("policies._demo OK cash_ret", res["cash"]["mean_return"],
          "fund_top_ret", res["fund_top (composite)"]["mean_return"])


if __name__ == "__main__":
    _demo()
