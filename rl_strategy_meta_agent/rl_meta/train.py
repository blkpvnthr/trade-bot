"""Build the env from the live pipeline, validate it, and run a PPO smoke train.

Usage:
    python -m rl_meta.train --check          # check_env only
    python -m rl_meta.train --timesteps 5000 # smoke training run
"""
from __future__ import annotations

import argparse

from .env import StrategyMetaEnv, StrategyView, make_demo_views
from .ranking import rank_results
from .registry import load_strategies
from .data import load_universe_bars, read_universe
from .backtest import run_backtests


def build_views_from_pipeline(max_strategies: int = 5, timeframe: str = "1Day",
                              start: str = "2018-01-01") -> list:
    """Discover PineScripts -> backtest on REAL cached Alpaca bars -> rank ->
    StrategyViews, keeping only OOS-eligible executable strategies.

    Reads bars OFFLINE (cache-only) so the RL stack needs no credentials. If no
    cached bars exist (run scripts/fetch_alpaca_bars.py first), falls back to
    synthetic demo views so check_env / training always run.
    """
    from .performance import PerformanceDB
    metas = load_strategies()
    bars = load_universe_bars(read_universe(), timeframe, start, offline=True) \
        if metas else {}
    db = PerformanceDB() if metas else None
    any_imports = bool(metas) and any(
        db.imported_trades(m.strategy_id, timeframe, m.version_hash) for m in metas)
    if not metas or (not bars and not any_imports):
        return make_demo_views(max_strategies)

    results = run_backtests(metas, bars, timeframe, db=db, universe=read_universe())
    # RL receives only RL-eligible strategies: TradingView-verified by default,
    # adapter_screening only if ALLOW_ADAPTER_SCREENING_FOR_RL is set.
    res_by_id = {r.strategy_id: r for r in results}
    ranked = rank_results([r for r in results if r.rl_eligible])
    if not ranked:
        return make_demo_views(max_strategies)

    views = []
    for r in ranked[:max_strategies]:
        life, roll = r.metrics, r.rolling
        sr = res_by_id[r.strategy_id]
        views.append(StrategyView(
            strategy_id=r.strategy_id, tier=r.tier, rank=r.rank,
            confidence=r.composite_score,                       # quality axis
            win_rate=life.win_rate, roll_win_rate=roll.window_30.win_rate,
            profit_factor=life.profit_factor,
            roll_profit_factor=roll.window_30.profit_factor,
            expectancy=life.expectancy, roll_expectancy=roll.window_30.expectancy,
            drawdown=life.max_drawdown, roll_drawdown=roll.window_30.max_drawdown,
            avg_loss=life.avg_loss, avg_hold_days=life.avg_hold_days or 4.0,
            strategy_confidence=sr.strategy_confidence / 100.0))   # trust axis (0..1)
    return views or make_demo_views(max_strategies)


def make_env(max_strategies: int = 5) -> StrategyMetaEnv:
    return StrategyMetaEnv(build_views_from_pipeline(max_strategies),
                           max_strategies=max_strategies)


def run_check():
    from stable_baselines3.common.env_checker import check_env
    env = make_env()
    check_env(env, warn=True)
    print("check_env OK -- observation_space", env.observation_space.shape)


def run_train(timesteps: int):
    from .models import make_ppo
    env = make_env()
    model = make_ppo(env)
    model.learn(total_timesteps=timesteps)
    print(f"PPO smoke training OK -- {timesteps} timesteps")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run SB3 check_env and exit")
    ap.add_argument("--timesteps", type=int, default=2000,
                    help="PPO smoke-train timesteps (0 to skip)")
    args = ap.parse_args()

    run_check()
    if not args.check and args.timesteps > 0:
        run_train(args.timesteps)


if __name__ == "__main__":
    main()
