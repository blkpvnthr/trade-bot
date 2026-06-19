"""Gymnasium environment: the RL strategy-allocation meta-agent.

The agent is NOT a candle trader. Each step it chooses HOW to allocate capital
across pre-ranked strategies (or stay cash). Strategy step-returns are
simulated from each strategy's metrics + the market regime, so the learnable
optimum is: fund high-expectancy Tier-A strategies, avoid deteriorating ones,
and sit in cash during bad regimes / drawdowns. The risk engine clamps every
action -- the agent can never breach a hard limit.

Single-allocation baseline (one strategy at a time). The observation/action
layout leaves room for multi-strategy weight vectors later (see README).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .risk import RiskEngine, RiskLimits, PortfolioState, OrderRequest
from .types import Tier

MARKET_FEATURES = 7
PORTFOLIO_FEATURES = 5
STRAT_FEATURES = 14   # 13 metric/quality features + strategy_confidence (trust)

# Action ids (Discrete(7)).
A_CASH, A_A_SMALL, A_A_MED, A_B_SMALL, A_C_SMALL, A_REDUCE, A_EXIT = range(7)
SMALL, MEDIUM = 0.10, 0.20

# Reward penalty weights. Tunable; documented.
P_DRAWDOWN = 0.50
P_OVERTRADE = 0.02
P_SLIPPAGE = 0.005      # per unit turnover
P_CONCENTRATION = 0.05
P_RULE = 0.10


@dataclass
class StrategyView:
    """What the env needs to know about one ranked strategy."""
    strategy_id: str
    tier: str
    rank: int
    confidence: float        # 0..1 composite score (how GOOD it looks)
    win_rate: float
    roll_win_rate: float
    profit_factor: float
    roll_profit_factor: float
    expectancy: float        # dollars per trade
    roll_expectancy: float
    drawdown: float
    roll_drawdown: float
    avg_loss: float          # dollars (<=0)
    avg_hold_days: float
    strategy_confidence: float = 0.0   # 0..1 trustworthiness (how TRUSTED the evidence)

    def mu_sigma(self) -> tuple:
        """Per-step (daily) expected return and vol implied by metrics."""
        from .signals import CAPITAL_PER_TRADE
        hold = max(self.avg_hold_days, 1.0)
        mu = (self.expectancy / CAPITAL_PER_TRADE) / hold
        sigma = max(abs(self.avg_loss) / CAPITAL_PER_TRADE, 0.005)
        return mu, sigma


def _tier_scalar(t: str) -> float:
    return {Tier.A.value: 1.0, Tier.B.value: 0.66,
            Tier.C.value: 0.33}.get(t, 0.0)


class StrategyMetaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, views: list, max_strategies: int = 5,
                 horizon: int = 252, start_equity: float = 100_000.0,
                 risk_limits: RiskLimits | None = None):
        super().__init__()
        if not views:
            views = make_demo_views(max_strategies)
        self.k = max_strategies
        self.views = list(views)[:self.k]
        self.horizon = horizon
        self.start_equity = start_equity
        self.risk = RiskEngine(risk_limits)

        obs_dim = MARKET_FEATURES + PORTFOLIO_FEATURES + self.k * STRAT_FEATURES
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)

        # Pre-index strategies by tier (best rank first).
        self._by_tier = {t: [i for i, v in enumerate(self.views) if v.tier == t]
                         for t in (Tier.A.value, Tier.B.value, Tier.C.value)}
        self._reset_state()

    # -- lifecycle ---------------------------------------------------------
    def _reset_state(self):
        self.t = 0
        self.equity = self.start_equity
        self.peak = self.start_equity
        self.cur_strat = -1          # -1 == cash
        self.cur_pct = 0.0
        self.trades_today = 0
        self.realized_today = 0.0
        self.last_step_ret = np.zeros(self.k, dtype=float)
        self.active = np.zeros(self.k, dtype=float)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self._advance_market(initial=True)
        self._roll_signals()
        return self._obs(), {}

    # -- market & strategy simulation -------------------------------------
    def _advance_market(self, initial: bool = False):
        rng = self.np_random
        if initial:
            self.regime = rng.choice([0.0, 0.5, 1.0], p=[0.25, 0.35, 0.40])
        else:
            # Sticky regime with small transition probability.
            if rng.random() < 0.05:
                self.regime = rng.choice([0.0, 0.5, 1.0])
        self.spy_trend = float(np.clip(rng.normal(self.regime - 0.5, 0.3), -1, 1))
        self.qqq_trend = float(np.clip(rng.normal(self.regime - 0.5, 0.35), -1, 1))
        self.vix = float(np.clip(rng.normal(0.6 - 0.4 * self.regime, 0.1), 0, 1))
        self.atr = float(np.clip(rng.normal(0.4 + 0.3 * self.vix, 0.1), 0, 1))
        self.vol_regime = 1.0 if self.vix > 0.55 else 0.0
        self.tod = float((np.sin(self.t / 8.0) + 1) / 2)   # placeholder cyclical

    def _roll_signals(self):
        """Draw this step's strategy returns + active flags, regime-modulated."""
        rng = self.np_random
        regime_factor = 0.4 + 1.2 * self.regime   # bear shrinks edge
        for i, v in enumerate(self.views):
            mu, sigma = v.mu_sigma()
            self.last_step_ret[i] = rng.normal(mu * regime_factor, sigma)
            self.active[i] = 1.0 if rng.random() < v.win_rate else 0.0

    # -- step --------------------------------------------------------------
    def _resolve_action(self, action: int) -> tuple:
        """Map action -> (target_strategy_index, target_pct)."""
        if action == A_CASH or action == A_EXIT:
            return -1, 0.0
        if action == A_REDUCE:
            return self.cur_strat, self.cur_pct * 0.5
        tier_map = {A_A_SMALL: (Tier.A.value, SMALL), A_A_MED: (Tier.A.value, MEDIUM),
                    A_B_SMALL: (Tier.B.value, SMALL), A_C_SMALL: (Tier.C.value, SMALL)}
        tier, pct = tier_map[action]
        idx_list = self._by_tier.get(tier, [])
        if not idx_list:
            return -1, 0.0          # no such strategy -> cash
        return idx_list[0], pct     # best-ranked of that tier

    def step(self, action: int):
        action = int(action)
        target_idx, target_pct = self._resolve_action(action)

        p = PortfolioState(self.equity, self.start_equity, self.peak,
                           open_positions=(1 if self.cur_strat >= 0 else 0),
                           trades_today=self.trades_today,
                           realized_today=self.realized_today)
        order = OrderRequest(
            strategy_id=self.views[target_idx].strategy_id if target_idx >= 0 else "cash",
            target_pct=target_pct,
            # Simulated market-quality inputs (the risk engine is the authority).
            spread_pct=0.002, slippage_pct=0.001, data_age_seconds=1.0)
        decision = self.risk.evaluate(order, p)

        rule_violation = (not decision.approved and target_pct > 0) or decision.halt
        if decision.halt:
            target_idx, new_pct = -1, 0.0          # forced flat
        elif decision.approved:
            new_pct = decision.approved_pct
        else:
            target_idx, new_pct = self.cur_strat, self.cur_pct  # veto: hold

        turnover = self._apply_allocation(target_idx, new_pct)

        # Realized PnL from the held position over this step.
        realized = 0.0
        if self.cur_strat >= 0 and self.cur_pct > 0:
            realized = self.equity * self.cur_pct * self.last_step_ret[self.cur_strat]
        self.equity += realized
        self.realized_today += realized
        self.peak = max(self.peak, self.equity)

        reward = self._reward(realized, turnover, new_pct, rule_violation)

        # Advance world.
        self.t += 1
        self._advance_market()
        self._roll_signals()
        if self.t % 5 == 0:                 # crude "new day" reset of daily counters
            self.trades_today = 0
            self.realized_today = 0.0

        terminated = self.equity <= self.start_equity * 0.7   # blew the dd limit
        truncated = self.t >= self.horizon
        info = {"equity": self.equity, "realized": realized,
                "violations": decision.violations, "halt": decision.halt}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _apply_allocation(self, target_idx: int, target_pct: float) -> float:
        prev_idx, prev_pct = self.cur_strat, self.cur_pct
        # Turnover = capital moved (switching strategy = full out + full in).
        if target_idx != prev_idx:
            turnover = prev_pct + target_pct
            if target_pct > 0:
                self.trades_today += 1
        else:
            turnover = abs(target_pct - prev_pct)
        self.cur_strat = target_idx if target_pct > 0 else -1
        self.cur_pct = target_pct if target_pct > 0 else 0.0
        return turnover

    def _reward(self, realized: float, turnover: float, target_pct: float,
                rule_violation: bool) -> float:
        dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        r = (realized / self.start_equity) * 100.0
        r -= P_DRAWDOWN * dd
        r -= P_OVERTRADE * (1.0 if turnover > 0 else 0.0)
        r -= P_SLIPPAGE * turnover
        r -= P_CONCENTRATION * max(0.0, target_pct - SMALL)
        r -= P_RULE * (1.0 if rule_violation else 0.0)
        return r

    # -- observation -------------------------------------------------------
    def _obs(self) -> np.ndarray:
        dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        market = [self.regime, self.spy_trend, self.qqq_trend, self.vix,
                  self.atr, self.vol_regime, self.tod]
        portfolio = [
            1.0 - self.cur_pct,                       # cash fraction
            self.equity / self.start_equity,
            dd,
            self.cur_pct,                             # exposure
            (1.0 if self.cur_strat >= 0 else 0.0),    # open positions (0/1)
        ]
        strat = []
        for i, v in enumerate(self.views):
            strat.extend([
                self.active[i],
                v.confidence,
                v.win_rate,
                v.roll_win_rate,
                _pf_norm(v.profit_factor),
                _pf_norm(v.roll_profit_factor),
                np.tanh(v.expectancy / 500.0),
                np.tanh(v.roll_expectancy / 500.0),
                v.drawdown,
                v.roll_drawdown,
                np.tanh(self.last_step_ret[i] * 50.0),
                1.0 - (v.rank / max(self.k, 1)),
                _tier_scalar(v.tier),
                v.strategy_confidence,            # distinct trust axis (0..1)
            ])
        obs = np.array(market + portfolio + strat, dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)


def _pf_norm(pf: float) -> float:
    if not np.isfinite(pf):
        return 1.0
    return float(min(max(pf, 0.0), 4.0) / 4.0)


def make_demo_views(k: int = 5, seed: int = 7) -> list:
    """Synthetic ranked strategies so the env runs standalone (tests, check_env)."""
    rng = np.random.default_rng(seed)
    tiers = [Tier.A.value, Tier.A.value, Tier.B.value, Tier.C.value, Tier.REJECT.value]
    views = []
    for i in range(k):
        tier = tiers[i % len(tiers)]
        edge = {Tier.A.value: 1.0, Tier.B.value: 0.4,
                Tier.C.value: 0.2}.get(tier, -0.2)
        wr = 0.62 if tier in (Tier.A.value, Tier.B.value) else 0.5
        views.append(StrategyView(
            strategy_id=f"strat_{i}", tier=tier, rank=i + 1,
            confidence=float(np.clip(0.7 * edge + 0.3, 0, 1)),
            win_rate=wr, roll_win_rate=wr - 0.02,
            profit_factor=2.4 if tier == Tier.A.value else 1.5,
            roll_profit_factor=2.2 if tier == Tier.A.value else 1.4,
            expectancy=200 * edge, roll_expectancy=180 * edge,
            drawdown=float(rng.uniform(0.08, 0.2)),
            roll_drawdown=float(rng.uniform(0.05, 0.18)),
            avg_loss=-300.0, avg_hold_days=4.0,
            # Trust axis, deliberately NOT equal to confidence (quality).
            strategy_confidence=float(np.clip(0.5 * edge + 0.35, 0, 1))))
    return views


def _demo():
    env = StrategyMetaEnv(make_demo_views())
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    total = 0.0
    for a in [A_A_MED] * 50:
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    assert np.isfinite(total)
    print("env._demo OK obs_dim", obs.shape[0], "ret", round(total, 3))


if __name__ == "__main__":
    _demo()
