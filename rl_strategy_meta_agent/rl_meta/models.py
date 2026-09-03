"""PPO policy factory for the meta-agent.

Small MLP -- the observation is low-dimensional engineered features, not
pixels, so a big network would just overfit. ponytail: one factory, no
custom policy class until a custom policy is actually needed.
"""
from __future__ import annotations

# Small two-layer net for both policy and value heads.
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))


def make_ppo(env, seed: int = 0, **overrides):
    """Build a PPO agent on `env`. Import is lazy so the rest of the package
    works without stable-baselines3 installed."""
    from stable_baselines3 import PPO

    params = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=POLICY_KWARGS,
        seed=seed,
        verbose=0,
    )
    params.update(overrides)
    return PPO(**params)
