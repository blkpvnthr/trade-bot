#!/usr/bin/env python3
"""
rl_agent_core.py  —  Shared architectural primitives for all RL agents.

Research basis
--------------
1. Nevmyvaka, Feng & Kearns (ICML 2006): Separate private state variables
   (inventory, time remaining) from market state variables (spread, order-book
   imbalance, volume) to exploit approximate independence and reduce overfitting.
   Use a backwards-induction / Q-learning hybrid so the algorithm is
   data-efficient regardless of how many market features are added.

2. Kong & So (Applied Sciences 2023): Careful design of actions is more
   important than choosing which RL algorithm to use.  Conservative strategies
   (lower hmax, non-zero buy/sell thresholds, integer share counts) outperform
   aggressive ones in range-bound / non-trending markets.  Market-specific
   tuning beats a universal strategy.

3. Tran, Pham-Hi & Bui (Algorithms 2023): Double Deep Q-Network (DDQN) with
   Sharpe ratio as the reward function is the best Q-learning trading system.
   CNN layers over time-series windows outperform flat MLPs. The ADAM optimizer
   with Leaky-ReLU activations gives fast, stable convergence.

Architecture
------------
* CNNFeatureExtractor  — 1-D convolutions over the recent price/indicator window,
  outputting a compact latent vector per timestep.  Replaces the flat MLP that
  stable-baselines3 uses by default.

* PrivateMarketStateBuilder  — implements the Nevmyvaka et al. state-space
  decomposition.  Private variables (time_remaining, inventory_fraction,
  unrealized_r) are kept separate from market variables so that the algorithm
  can enumerate every private-state combination without re-sweeping market data.

* SharpeRewardCalculator  — computes a rolling Sharpe ratio from the per-step
  PnL history and uses it as the primary reward signal (Tran et al.).  A small
  market-structure bonus/penalty is added on top so the agent still learns
  entry/exit timing.

* EnsembleAgentSelector  — replicates the Kong & So ensemble: after each
  validation window it picks the highest-Sharpe agent among several candidates
  (PPO, DDQN-style via QR-DQN wrapper, baseline buy-and-hold) and routes live
  decisions through it.

* OOPS utility overlay — option candidates can be scored with simulated payoff
  distributions, CRRA certainty equivalent, skew/kurtosis, bid/ask costs, and
  elasticity penalties before the RL entry/selection layer is rewarded.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# Shared constants — observation and action dimensions
# ---------------------------------------------------------------------------

# Number of feature channels per timestep in the CNN window.
# 20 = 12 original OHLCV/indicator features + 8 dip-reclaim features.
N_CHANNELS: int = int(os.getenv("RL_N_CHANNELS", "20"))

# How many bars the CNN looks back (window size).
CNN_LOOKBACK: int = int(os.getenv("RL_CNN_LOOKBACK", "20"))

# Discrete action indices shared across all RL environments.
ACTION_HOLD:        int = 0
ACTION_ENTER_LONG:  int = 1
ACTION_EXIT_LONG:   int = 2
ACTION_ADD_LONG:    int = 3   # optional; some envs use only 0-2


def make_observation_space(n_channels: int = N_CHANNELS, lookback: int = CNN_LOOKBACK):
    """Return a gymnasium Box observation space for a CNN-windowed observation."""
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces  # type: ignore[no-redef]
    return spaces.Box(
        low=-10.0,
        high=10.0,
        shape=(n_channels * lookback,),
        dtype=np.float32,
    )


def make_action_space(n_actions: int = 3):
    """Return a Discrete action space."""
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces  # type: ignore[no-redef]
    return spaces.Discrete(n_actions)


def get_policy_kwargs(
    n_channels: int = N_CHANNELS,
    lookback: int = CNN_LOOKBACK,
    features_dim: int = 256,
) -> Dict[str, Any]:
    """
    Return SB3 policy_kwargs that wire the CNN feature extractor.
    Pass as `policy_kwargs=get_policy_kwargs()` when constructing PPO.
    """
    return {
        "features_extractor_class": CNNFeatureExtractor,
        "features_extractor_kwargs": {
            "n_features": n_channels,
            "window_size": lookback,
            "features_dim": features_dim,
        },
    }

# ---------------------------------------------------------------------------
# 1.  CNN feature extractor  (Tran et al., Section 3.2)
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Two-layer 1-D CNN over a sliding window of OHLCV + indicator features.

    The observation space is expected to be shaped as
    (n_features * window_size,) — a flat vector that we internally reshape to
    (n_features, window_size) before passing through the conv layers.

    Why CNN instead of MLP?
    -------------------------
    Tran et al. show that CNNs extract meaningful local patterns from
    multivariate price time-series more efficiently than flat MLPs, especially
    when the input contains multiple moving-average or momentum features whose
    relative ordering carries information (e.g. EMA cross-overs, MACD).
    """

    def __init__(
        self,
        observation_space,
        n_features: int = 12,          # features per timestep
        window_size: int = 20,         # lookback bars baked into obs
        n_filters_1: int = 64,
        n_filters_2: int = 128,
        kernel_size: int = 3,
        features_dim: int = 256,
    ):
        # features_dim is the output size that SB3 will use as policy input
        super().__init__(observation_space, features_dim=features_dim)

        self.n_features = n_features
        self.window_size = window_size

        # Two conv blocks with Leaky-ReLU (Tran et al. found this beats sigmoid)
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_filters_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_filters_1,
            out_channels=n_filters_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.AdaptiveAvgPool1d(1)   # global average pool → (batch, n_filters_2)

        # Dense head maps pooled features to features_dim
        self.fc = nn.Sequential(
            nn.Linear(n_filters_2, features_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, n_features * window_size)
        batch = observations.shape[0]
        # Reshape to (batch, n_features, window_size) for Conv1d
        x = observations.view(batch, self.n_features, self.window_size)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x).squeeze(-1)   # (batch, n_filters_2)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 2.  Private / market state decomposition  (Nevmyvaka et al., Section 3)
# ---------------------------------------------------------------------------

class PrivateMarketStateBuilder:
    """
    Dual-mode state builder:

    *Streaming mode* (new — used by dip-reclaim trainers):
        builder = PrivateMarketStateBuilder(lookback=20)
        builder.push(bar_dict)       # ingest one bar
        obs = builder.build()        # returns flat float32 array length N_CHANNELS*lookback
        builder.reset()              # clear window at episode start

    *Static mode* (legacy — original behaviour):
        obs = PrivateMarketStateBuilder.build(time_remaining=…, …)   # classmethod
    """

    def __init__(self, lookback: int = CNN_LOOKBACK, n_channels: int = N_CHANNELS):
        self._lookback   = lookback
        self._n_channels = n_channels
        self._window: deque = deque(maxlen=lookback)
        # Feature extraction order must match _BAR_FEATURES
        self._BAR_FEATURES = [
            "close", "vwap", "ema9", "ema20",
            "lower_wick_ratio", "upper_wick_ratio",
            "volume_spike_ratio", "absorption_score",
            "liquidity_sweep_detected", "support_touch",
            "pivot_low_reclaim", "support_break_failed",
            "reclaim_vwap", "unrealised_pnl",
            "in_position", "bars_in_trade",
            "red_candle_streak", "green_candle_streak",
            "support_distance_pct", "resistance_distance_pct",
        ]
        assert len(self._BAR_FEATURES) == n_channels, (
            f"BAR_FEATURES length {len(self._BAR_FEATURES)} != n_channels {n_channels}"
        )

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def push(self, bar: Dict) -> None:
        price = float(bar.get("close", 1.0))
        sup   = float(bar.get("support_zone",   price * 0.99))
        res   = float(bar.get("resistance_zone", price * 1.01))
        vwap  = float(bar.get("vwap", price))

        vec = np.array([
            price / max(float(bar.get("day_open", price)), 1e-9) - 1.0,  # intraday return from open
            price / max(vwap, 1e-9) - 1.0,                               # dist to vwap
            price / max(float(bar.get("ema9",  price)), 1e-9) - 1.0,    # dist to ema9
            price / max(float(bar.get("ema20", price)), 1e-9) - 1.0,    # dist to ema20
            float(np.clip(bar.get("lower_wick_ratio",  0.0), 0, 1)),
            float(np.clip(bar.get("upper_wick_ratio",  0.0), 0, 1)),
            float(np.clip(bar.get("volume_spike_ratio", 1.0) / 5.0, 0, 1)),
            float(np.clip(bar.get("absorption_score", 0.0), 0, 1)),
            float(bar.get("liquidity_sweep_detected", 0)),
            float(bar.get("support_touch",            0)),
            float(bar.get("pivot_low_reclaim",        0)),
            float(bar.get("support_break_failed",     0)),
            float(bar.get("reclaim_vwap",             0)),
            float(np.clip(bar.get("unrealised_pnl", 0.0), -0.10, 0.10)),
            float(bar.get("in_position",              0)),
            float(np.clip(bar.get("bars_in_trade",   0) / 60.0, 0, 1)),
            float(np.clip(bar.get("red_candle_streak",   0) / 10.0, 0, 1)),
            float(np.clip(bar.get("green_candle_streak", 0) / 10.0, 0, 1)),
            float(np.clip((price - sup) / max(price, 1e-9), -0.05, 0.05)),
            float(np.clip((res - price) / max(price, 1e-9), -0.05, 0.05)),
        ], dtype=np.float32)

        self._window.append(vec)

    def build(self) -> np.ndarray:
        """Return flat observation: (n_channels × lookback,) float32."""
        n = self._n_channels
        lookback = self._lookback
        obs = np.zeros((lookback, n), dtype=np.float32)
        for i, vec in enumerate(self._window):
            obs[lookback - len(self._window) + i] = vec[:n]
        return obs.flatten()

    def reset(self) -> None:
        self._window.clear()

    # ------------------------------------------------------------------
    # Legacy static interface
    # ------------------------------------------------------------------

    # (legacy class preserved below for backward compat)
    N_PRIVATE = 3
    N_MARKET  = 7
    N_TOTAL   = 10
    """
    Constructs the state vector used in the RL environment following the
    Nevmyvaka et al. decomposition.

    Private variables
    -----------------
    These are only known to the execution agent, not other market participants:
      * time_remaining  — fraction of the episode horizon left  [0, 1]
      * inventory_frac  — remaining inventory as fraction of target  [0, 1]
      * unrealized_r    — current unrealized gain / initial-risk ratio  [-∞, +∞]

    Market variables
    ----------------
    These describe the current state of the limit order book and recent flow.
    Unlike private variables they are shared across all episodes, so the
    algorithm can re-use the same market observation for every private-state
    combination (Nevmyvaka key insight — makes data-use O(T·I·L) not O(episodes)):
      * bid_ask_spread_norm  — (ask−bid)/mid, normalised
      * market_order_cost    — estimated market-impact if we go to market now
      * volume_imbalance     — (bid_vol − ask_vol) / (bid_vol + ask_vol)
      * rsi_norm             — RSI / 100
      * macd_norm            — MACD histogram / price
      * vwap_dist            — (price − VWAP) / price
      * vol_ratio            — current volume / rolling average (capped)

    The full observation is the concatenation of private + market features,
    but the two sub-vectors are tracked separately so callers can zero-out
    market features during debugging or policy analysis.
    """

    N_PRIVATE = 3
    N_MARKET  = 7
    N_TOTAL   = N_PRIVATE + N_MARKET  # 10

    @staticmethod
    def build_state_vec(
        *,
        time_remaining: float,
        inventory_frac: float,
        unrealized_r: float,
        bid_ask_spread_norm: float = 0.0,
        market_order_cost: float   = 0.0,
        volume_imbalance: float    = 0.0,
        rsi: float                 = 50.0,
        macd_hist: float           = 0.0,
        price: float               = 1.0,
        vwap_dist: float           = 0.0,
        vol_ratio: float           = 1.0,
        vol_ratio_cap: float       = 5.0,
    ) -> np.ndarray:
        """Legacy static helper — returns a float32 array of length N_TOTAL."""
        private = np.array([
            float(np.clip(time_remaining, 0.0, 1.0)),
            float(np.clip(inventory_frac, 0.0, 1.0)),
            float(np.clip(unrealized_r, -5.0, 5.0)),
        ], dtype=np.float32)

        market = np.array([
            float(np.clip(bid_ask_spread_norm, 0.0, 0.05)),
            float(np.clip(market_order_cost,   0.0, 0.05)),
            float(np.clip(volume_imbalance,   -1.0, 1.0)),
            float(rsi / 100.0),
            float(np.clip(macd_hist / max(abs(price), 1e-9), -0.05, 0.05)),
            float(np.clip(vwap_dist, -0.05, 0.05)),
            float(np.clip(vol_ratio / vol_ratio_cap, 0.0, 1.0)),
        ], dtype=np.float32)

        return np.concatenate([private, market])

    @staticmethod
    def split(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a combined observation into (private, market) sub-vectors."""
        return obs[:3], obs[3:]


# ---------------------------------------------------------------------------
# 3.  Sharpe-ratio reward calculator  (Tran et al., Section 3.4)
# ---------------------------------------------------------------------------

class SharpeRewardCalculator:
    """
    Maintains a rolling window of per-step returns and returns a Sharpe-based
    reward signal on each call to `update()`.

    Why Sharpe?
    -----------
    Tran et al. show that DDQN with Sharpe ratio as the reward function
    achieves the best Q-learning trading system (15.96 % monthly return on
    BTC-USDT vs 8.94 % for the raw-return version).  The Sharpe signal naturally
    penalises volatility, steering the agent toward steady consistent gains over
    high-variance windfalls.

    The reward is computed as:
        sharpe_reward = mean(returns) / (std(returns) + ε)
    where 'returns' is the last `window` per-step PnL values.  A small
    market-structure bonus/penalty is *added on top* so timing signals still
    propagate, but the Sharpe term dominates (controlled by `sharpe_weight`).
    """

    def __init__(self, window: int = 60, risk_free: float = 0.0, sharpe_weight: float = 0.7):
        self.window       = window
        self.risk_free    = risk_free
        self.sharpe_weight = sharpe_weight
        self._returns: deque = deque(maxlen=window)

    def reset(self) -> None:
        self._returns.clear()

    def update(
        self,
        step_pnl: float,
        structure_bonus: float = 0.0,
    ) -> float:
        """
        Record `step_pnl`, compute the rolling Sharpe reward, add
        `structure_bonus` (entry/exit quality signal), and return the total.

        Parameters
        ----------
        step_pnl : float
            Realised or unrealised PnL for this timestep (fractional).
        structure_bonus : float
            Small bonus/penalty from market-structure signals (entry near
            support, exit near resistance, etc.).  Should be ≪ the Sharpe term.
        """
        self._returns.append(step_pnl)

        if len(self._returns) < 2:
            # Not enough history yet — fall back to raw PnL
            return float(step_pnl + structure_bonus)

        ret_arr = np.array(self._returns, dtype=np.float64)
        excess  = ret_arr - self.risk_free
        sharpe  = float(excess.mean() / max(excess.std(), 1e-9))

        # Blend: majority Sharpe, minority raw PnL (so sparse rewards still flow)
        raw_weight = 1.0 - self.sharpe_weight
        blended = self.sharpe_weight * sharpe + raw_weight * step_pnl

        return float(blended + structure_bonus)

    @property
    def current_sharpe(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        ret_arr = np.array(self._returns, dtype=np.float64)
        excess  = ret_arr - self.risk_free
        return float(excess.mean() / max(excess.std(), 1e-9))


# ---------------------------------------------------------------------------
# 3b.  Paper-faithful Double-DQN config  (Tran, Pham-Hi & Bui, Algorithms 2023)
# ---------------------------------------------------------------------------
#
# Their headline result: a Double Deep Q-Network with a Sharpe-ratio reward is the BEST
# Q-learning trading system — it beat both Dueling-DDQN and Bayesian Optimization on
# risk-adjusted return and stability. The trainers here default to PPO; this block makes the
# paper's DDQN directly constructible so the DDQN-vs-PPO comparison the paper recommends can
# actually be run. SB3's vanilla DQN does not decouple action selection from evaluation, so we
# use sb3-contrib's QR-DQN (distributional, Double-DQN-style target) — the same "DDQN-style via
# QR-DQN wrapper" the EnsembleAgentSelector docstring already anticipates.
#
# Hyperparameters are Table 2 of the paper (adapted: their ε_step=300 over a short crypto run;
# SB3 expresses exploration as a fraction of total timesteps + a final ε, set by the caller).
TRAN_DDQN_HYPERPARAMS: Dict[str, Any] = {
    "gamma": 0.98,                 # future rewards matter (their γ for volatile series)
    "learning_rate": 0.001,        # ADAM lr
    "batch_size": 40,
    "buffer_size": 10_000,         # replay memory N_B
    "target_update_interval": 10,  # N_u: copy online→target every 10 steps
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.12,
    "exploration_fraction": 0.30,  # ε decays over the first 30% of training
    "learning_starts": 1_000,
}


def ddqn_available() -> bool:
    """True if sb3-contrib (QR-DQN) is importable — the Double-DQN-style agent backend."""
    try:
        import sb3_contrib  # noqa: F401
        return True
    except Exception:
        return False


def make_ddqn_agent(env, n_channels: int = N_CHANNELS, lookback: int = CNN_LOOKBACK,
                    tensorboard_log: Optional[str] = None, **overrides: Any):
    """Construct the paper's Double-DQN-style agent (QR-DQN + CNN extractor + Table-2 params).

    Returns an unfitted SB3 model ready for `.learn(...)`, or raises ImportError with a clear
    message if sb3-contrib is absent. `overrides` win over TRAN_DDQN_HYPERPARAMS so callers /
    tuners can sweep. The reward should be the Sharpe-based signal (SharpeRewardCalculator) —
    that pairing is the paper's best configuration. Requires a Discrete action space."""
    try:
        from sb3_contrib import QRDQN
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "make_ddqn_agent needs sb3-contrib (QR-DQN). Install with `pip install sb3-contrib` "
            "to run the Tran et al. DDQN-vs-PPO comparison."
        ) from exc
    params: Dict[str, Any] = {**TRAN_DDQN_HYPERPARAMS, **overrides}
    return QRDQN(
        "MlpPolicy", env,
        policy_kwargs=get_policy_kwargs(n_channels, lookback),
        tensorboard_log=tensorboard_log,
        verbose=0,
        **params,
    )


# ---------------------------------------------------------------------------
# 4.  Conservative / aggressive action design  (Kong & So, Section 6)
# ---------------------------------------------------------------------------

class ActionDesign:
    """
    Encapsulates the Kong & So finding that the *design of actions* matters
    more than which RL algorithm is selected.

    For each market regime a profile controls:
      * hmax          — maximum shares traded in a single action
      * buy_threshold  — minimum signal strength before buying (0 = instant)
      * sell_threshold — minimum signal strength before selling
      * use_integers   — whether to round share counts to whole numbers

    Conservative profiles (KOSPI/JPX-style) use small hmax, non-zero
    thresholds, and integer rounding.  Aggressive profiles (DJIA-style) allow
    larger position sizes and tighter thresholds.
    """

    CONSERVATIVE = dict(hmax=10, buy_threshold=0.30, sell_threshold=0.10, use_integers=True)
    MODERATE     = dict(hmax=25, buy_threshold=0.20, sell_threshold=0.05, use_integers=True)
    AGGRESSIVE   = dict(hmax=100, buy_threshold=0.0,  sell_threshold=0.0,  use_integers=False)

    REGIME_PROFILES: Dict[str, Dict] = {
        "trending_bull":   AGGRESSIVE,
        "trending_bear":   CONSERVATIVE,
        "range_bound":     CONSERVATIVE,
        "high_volatility": CONSERVATIVE,
        "low_volatility":  MODERATE,
        "unknown":         MODERATE,
    }

    @classmethod
    def get(cls, regime: str) -> Dict:
        return cls.REGIME_PROFILES.get(regime, cls.MODERATE)

    @staticmethod
    def clip_action(
        raw_signal: float,
        profile: Dict,
    ) -> float:
        """
        Convert a raw RL action signal [-1, +1] into a trade signal using the
        profile thresholds.  Returns 0 (hold) if the signal does not clear the
        threshold, otherwise scales by hmax.

        This mimics the Kong & So Step 2 / Step 3 trading mechanism where
        signals below the threshold produce no trade.
        """
        if raw_signal > profile["buy_threshold"]:
            qty = raw_signal * profile["hmax"]
        elif raw_signal < -profile["sell_threshold"]:
            qty = raw_signal * profile["hmax"]
        else:
            qty = 0.0

        if profile["use_integers"]:
            qty = float(int(qty))

        return qty


# ---------------------------------------------------------------------------
# 5.  Ensemble agent selector  (Kong & So, Section 4.4)
# ---------------------------------------------------------------------------

class EnsembleAgentSelector:
    """
    Maintains a rolling Sharpe score for each registered agent and selects
    the best one for live trading at the end of each validation window.

    This is a direct implementation of the Kong & So Ensemble Strategy
    (Section 4.4): every `rebalance_window` steps, re-score all agents on a
    `validation_window` look-back and route signals through the winner.

    Usage
    -----
    selector = EnsembleAgentSelector(agents={"ppo": ppo_model, "ddqn": ddqn_model})
    # After each episode:
    selector.record_return("ppo",  episode_return)
    selector.record_return("ddqn", episode_return)
    # At rebalance time:
    best_name, best_model = selector.select_best()
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        validation_window: int = 63,      # ~3 months trading days
        rebalance_window: int  = 63,
    ):
        self.agents            = agents
        self.validation_window = validation_window
        self.rebalance_window  = rebalance_window
        self._returns: Dict[str, deque] = {
            name: deque(maxlen=validation_window) for name in agents
        }
        self._step = 0
        self._active_name: Optional[str] = None

    def record_return(self, agent_name: str, episode_return: float) -> None:
        if agent_name in self._returns:
            self._returns[agent_name].append(episode_return)
        self._step += 1

    def _sharpe(self, name: str) -> float:
        rets = list(self._returns[name])
        if len(rets) < 2:
            return -999.0
        arr = np.array(rets, dtype=np.float64)
        return float(arr.mean() / max(arr.std(), 1e-9))

    def select_best(self) -> Tuple[str, Any]:
        """Return (name, model) of the agent with the highest validation Sharpe."""
        scored = {name: self._sharpe(name) for name in self.agents}
        best   = max(scored, key=scored.__getitem__)
        self._active_name = best
        return best, self.agents[best]

    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    def should_rebalance(self) -> bool:
        return self._step > 0 and self._step % self.rebalance_window == 0


# ---------------------------------------------------------------------------
# 6.  Dip-reclaim feature set  (shared across all RL agents)
# ---------------------------------------------------------------------------

# Feature count for the dip-reclaim extension (added on top of existing features)
DIP_RECLAIM_N_FEATURES = 8

# Setup-type numeric codes for obs / labeling
SETUP_TYPE_CODES: Dict[str, float] = {
    "dip_reclaim":     1.00,
    "support_reclaim": 0.75,
    "vwap_reclaim":    0.50,
    "breakout":        0.25,
    "none":            0.00,
}


class DipReclaimFeatures:
    """
    Shared feature extraction for the selloff → support → reclaim pattern.

    Computes a fixed 8-feature vector from a single OHLCV + indicator row.
    Designed to be appended to any existing feature vector so existing agents
    can be extended without breaking their observation-space shape contract —
    just update N_FEATURES and pass a compatible obs to the CNN extractor.

    Features (in order):
        0. lower_wick_ratio       — absorption signal at support lows
        1. upper_wick_ratio       — rejection / resistance signal
        2. liquidity_sweep        — 0/1 flag: wick below prior low + closed above
        3. capitulation_bar       — 0/1 flag: high RVOL + large wick + red body
        4. reclaim_vwap           — 0/1 flag: crossed above VWAP this bar
        5. support_absorption     — composite (wick * vol * proximity to day low)
        6. red_streak_norm        — consecutive red bars / 10  (0-1 normalized)
        7. dist_to_day_low        — (close / day_low) - 1, clipped [-0.10, 0.10]
    """

    @staticmethod
    def compute(
        close: float,
        open_: float,
        high: float,
        low: float,
        volume: float,
        avg_volume: float,
        vwap: float,
        prev_close: float,
        prev_vwap: float,
        day_low: float,
        prev_day_low: float,
        red_streak: int = 0,
    ) -> np.ndarray:
        """
        Returns a float32 array of length DIP_RECLAIM_N_FEATURES.

        All inputs should be already-numeric (no NaN). Caller is responsible for
        substituting sensible defaults (e.g. close for missing day_low).
        """
        bar_range = max(high - low, 1e-9)
        body_low = min(open_, close)
        body_high = max(open_, close)

        lower_wick = max(body_low - low, 0.0) / bar_range
        upper_wick = max(high - body_high, 0.0) / bar_range

        vol_ratio = volume / max(avg_volume, 1e-9)
        close_location = (close - low) / bar_range

        # Liquidity sweep: wick punched below prior session low, closed above
        liquidity_sweep = float(
            low < prev_day_low and close > low and close_location > 0.50
        )
        # Capitulation: high volume + large lower wick + red body + closed off lows
        capitulation = float(
            vol_ratio >= 2.0 and lower_wick >= 0.40 and close < open_ and close_location > 0.35
        )
        # VWAP reclaim: was below, now above
        reclaim_vwap = float(close >= vwap and prev_close < prev_vwap)

        # Support absorption composite
        day_low_prox = 1.0 - min((close / max(day_low, 1e-9) - 1.0), 0.05) / 0.05 if day_low > 0 else 0.0
        support_absorption = (
            0.40 * lower_wick
            + 0.30 * min(vol_ratio / 3.0, 1.0)
            + 0.30 * max(min(day_low_prox, 1.0), 0.0)
        )

        dist_to_day_low = float(np.clip(close / max(day_low, 1e-9) - 1.0, -0.10, 0.10)) if day_low > 0 else 0.0

        return np.array(
            [
                float(np.clip(lower_wick, 0.0, 1.0)),
                float(np.clip(upper_wick, 0.0, 1.0)),
                liquidity_sweep,
                capitulation,
                reclaim_vwap,
                float(np.clip(support_absorption, 0.0, 1.0)),
                float(min(red_streak, 10) / 10.0),
                dist_to_day_low,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def classify_setup(
        lower_wick: float,
        capitulation: float,
        liquidity_sweep: float,
        reclaim_vwap: float,
        vol_ratio: float,
        close_location: float,
        breakout_persistence: float = 0.0,
    ) -> str:
        """Return the setup_type label for the current bar."""
        is_dip = (capitulation > 0 or liquidity_sweep > 0) and lower_wick >= 0.35 and reclaim_vwap > 0
        is_vwap = reclaim_vwap > 0 and vol_ratio >= 1.5
        is_support = close_location > 0.5 and lower_wick >= 0.30 and vol_ratio >= 1.3
        is_breakout = breakout_persistence > 0 and vol_ratio >= 1.5
        if is_dip:
            return "dip_reclaim"
        if is_vwap:
            return "vwap_reclaim"
        if is_support:
            return "support_reclaim"
        if is_breakout:
            return "breakout"
        return "none"
