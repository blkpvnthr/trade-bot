import numpy as np

from rl_meta.env import StrategyMetaEnv, make_demo_views, A_CASH
from rl_meta.policies import (p_cash, evaluate_policy, compare_policies,
                             make_rl_policy)


def env():
    return StrategyMetaEnv(make_demo_views(), horizon=40)


def test_cash_policy_constant():
    assert p_cash(None, None, 5) == A_CASH


def test_evaluate_policy_returns_finite_stats():
    s = evaluate_policy(env(), p_cash, episodes=5, base_seed=0)
    for k in ("mean_reward", "mean_return", "mean_maxdd", "blowup_rate", "ret_per_dd"):
        assert k in s and np.isfinite(s[k])
    assert 0.0 <= s["blowup_rate"] <= 1.0


def test_cash_holds_no_drawdown_from_allocation():
    # Cash never allocates -> equity never moves -> ~zero drawdown.
    s = evaluate_policy(env(), p_cash, episodes=5)
    assert s["mean_maxdd"] < 1e-9


def test_compare_includes_baselines_and_rl():
    class FakeModel:
        def predict(self, obs, deterministic=True):
            return np.array(A_CASH), None
    table = compare_policies(env(), rl_model=FakeModel(), episodes=4)
    for name in ("cash", "random", "fund_top (composite)", "equal_weight_rr",
                 "RL (PPO)"):
        assert name in table
    # The RL-as-cash stub should match the cash baseline.
    assert table["RL (PPO)"]["mean_return"] == table["cash"]["mean_return"]


def test_deterministic_under_fixed_seed():
    e = env()
    a = evaluate_policy(e, p_cash, episodes=5, base_seed=7)
    b = evaluate_policy(e, p_cash, episodes=5, base_seed=7)
    assert a == b
