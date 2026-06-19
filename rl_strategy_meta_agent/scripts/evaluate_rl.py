#!/usr/bin/env python3
"""Run the full RL allocation pipeline and compare the agent against baselines.

Research-only. The RL agent is an ADAPTIVE STRATEGY ALLOCATOR, not a standalone
trading strategy: it learns which validated strategies deserve capital, when to
sit in cash, and how to manage exposure -- it does NOT discover raw price
signals. This script:

  1. Builds the allocation env from the strategy pipeline (real eligible
     strategies if available, else synthetic demo views -- reported clearly).
  2. Trains PPO.
  3. Compares the trained policy against cash / random / fund-top / equal-weight
     baselines on identical episodes.
  4. Prints an honest verdict.

    python scripts/evaluate_rl.py --timesteps 20000 --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.env import StrategyMetaEnv
from rl_meta.train import build_views_from_pipeline
from rl_meta.models import make_ppo
from rl_meta.policies import compare_policies


def _is_demo(views) -> bool:
    return all(v.strategy_id.startswith("strat_") for v in views)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-strategies", type=int, default=5)
    args = ap.parse_args()

    views = build_views_from_pipeline(args.max_strategies)
    demo = _is_demo(views)

    print("=== Data context ===")
    if demo:
        print("RL is training on SYNTHETIC DEMO views -- the real strategy")
        print("population has no RL-eligible (TradingView-verified, OOS-passed)")
        print("strategies yet. Numbers below validate the HARNESS, not real edge.")
    else:
        print(f"RL is training on {len(views)} real RL-eligible strategies:")
        for v in views:
            print(f"  {v.strategy_id:<24} tier={v.tier} conf={v.strategy_confidence:.2f}")
    print(f"Strategies in env: {len(views)}\n")

    env = StrategyMetaEnv(views, max_strategies=args.max_strategies)
    print(f"Training PPO for {args.timesteps} timesteps "
          f"(obs dim {env.observation_space.shape[0]})...")
    model = make_ppo(env)
    model.learn(total_timesteps=args.timesteps)

    print(f"\n=== Policy comparison ({args.episodes} episodes, identical seeds) ===")
    table = compare_policies(env, rl_model=model, episodes=args.episodes)
    _print_table(table)
    _verdict(table)


def _print_table(table):
    print(f"{'policy':<22} {'reward':>8} {'return%':>8} {'maxDD%':>7} "
          f"{'blowup%':>8} {'ret/DD':>7}")
    print("-" * 64)
    for name, s in table.items():
        print(f"{name:<22} {s['mean_reward']:>8.2f} {s['mean_return']*100:>7.2f}% "
              f"{s['mean_maxdd']*100:>6.2f}% {s['blowup_rate']*100:>7.1f}% "
              f"{s['ret_per_dd']:>7.2f}")


def _verdict(table):
    rl = table.get("RL (PPO)")
    if not rl:
        return
    baselines = {k: v for k, v in table.items() if k != "RL (PPO)"}
    best_rdd = max(baselines.items(), key=lambda kv: kv[1]["ret_per_dd"])
    best_ret = max(baselines.items(), key=lambda kv: kv[1]["mean_return"])
    print("\n=== Verdict ===")
    print(f"RL ret/DD {rl['ret_per_dd']} vs best baseline "
          f"{best_rdd[0]} {best_rdd[1]['ret_per_dd']}")
    print(f"RL return {rl['mean_return']*100:.2f}% vs best baseline "
          f"{best_ret[0]} {best_ret[1]['mean_return']*100:.2f}%")
    won = rl["ret_per_dd"] >= best_rdd[1]["ret_per_dd"]
    print("RL improved risk-adjusted allocation on this env."
          if won else
          "RL did NOT beat the best naive baseline here -- expected with a thin /"
          " synthetic population; revisit once pinescripts/ holds diverse,"
          " TV-validated strategies.")


if __name__ == "__main__":
    main()
