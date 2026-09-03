import numpy as np
import pytest

from rl_meta.env import (StrategyMetaEnv, make_demo_views, A_CASH, A_A_MED,
                         A_EXIT, MARKET_FEATURES, PORTFOLIO_FEATURES, STRAT_FEATURES)


def env(k=5):
    return StrategyMetaEnv(make_demo_views(k), max_strategies=k, horizon=64)


def test_observation_dim():
    k = 5
    e = env(k)
    expected = MARKET_FEATURES + PORTFOLIO_FEATURES + k * STRAT_FEATURES
    assert e.observation_space.shape == (expected,)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (expected,)
    assert e.observation_space.contains(obs)


def test_strategy_confidence_added_dimension():
    # strategy_confidence is a distinct 14th per-strategy feature; obs 77 -> 82 at K=5.
    assert STRAT_FEATURES == 14
    assert MARKET_FEATURES + PORTFOLIO_FEATURES + 5 * STRAT_FEATURES == 82


def test_strategy_confidence_is_distinct_from_composite():
    import dataclasses
    from rl_meta.env import StrategyView
    base = make_demo_views(1)[0]
    lo = dataclasses.replace(base, strategy_confidence=0.1)
    hi = dataclasses.replace(base, strategy_confidence=0.9)
    o_lo, _ = StrategyMetaEnv([lo], max_strategies=1).reset(seed=0)
    o_hi, _ = StrategyMetaEnv([hi], max_strategies=1).reset(seed=0)
    # Same seed, identical except the trust axis -> observations must differ...
    assert not np.allclose(o_lo, o_hi)
    # ...while the quality 'confidence' feature is untouched.
    assert lo.confidence == hi.confidence


def test_action_space():
    assert env().action_space.n == 7


def test_reset_is_deterministic():
    o1, _ = env().reset(seed=42)
    o2, _ = env().reset(seed=42)
    assert np.allclose(o1, o2)


def test_step_returns_valid_obs_and_finite_reward():
    e = env()
    e.reset(seed=1)
    obs, r, term, trunc, info = e.step(A_A_MED)
    assert e.observation_space.contains(obs)
    assert np.isfinite(r)
    assert "equity" in info


def test_cash_action_holds_no_position():
    e = env()
    e.reset(seed=1)
    e.step(A_A_MED)
    e.step(A_EXIT)
    assert e.cur_strat == -1 and e.cur_pct == 0.0


def test_episode_truncates_at_horizon():
    e = env()
    e.reset(seed=2)
    steps = 0
    done = False
    while not done and steps < 1000:
        _, _, term, trunc, _ = e.step(A_CASH)
        done = term or trunc
        steps += 1
    assert done and steps <= e.horizon + 1


def test_risk_engine_caps_exposure():
    e = env()
    e.reset(seed=3)
    for _ in range(20):
        e.step(A_A_MED)
        assert e.cur_pct <= 0.25 + 1e-9  # risk engine max_position_pct


def test_sb3_check_env():
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    check_env(env(), warn=True)
