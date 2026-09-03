"""
rl_intraday_structure_trainer.py — QuantumQuant
Trains the Intraday RL Agent to identify and execute dip-reclaim setups:
  SELL-OFF → LIQUIDITY SWEEP → SUPPORT TOUCH → CAPITULATION VOLUME
           → ABSORPTION → RECLAIM → CONTINUATION → 3R TARGET

Features: 25+ engineered channels (see rl_agent_core.CHANNEL_NAMES)
Reward:   Shaped bonuses/penalties covering entry quality, exit timing,
          hold quality, structure validation.
Labels:   setup_type persisted in replay CSV and training diagnostics.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Gymnasium / SB3 — optional at import time; hard-fail only when training
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    gym = None

from rl_agent_core import (
    N_CHANNELS, CNN_LOOKBACK, ACTION_HOLD, ACTION_ENTER_LONG, ACTION_EXIT_LONG,
    CNNFeatureExtractor, SharpeRewardCalculator, PrivateMarketStateBuilder,
    make_observation_space, make_action_space, get_policy_kwargs,
)
from pivot_reversal_scanner import PivotReversalScanner, enrich_bar_from_label
from scalping_execution_layer import (
    ScalpingExecutionLayer, TradeRecord, TradeOutcome,
    compute_execution_metrics, REPLAY_FIELDNAMES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RL_POLICY_MODE       = os.environ.get("RL_POLICY_MODE", "hybrid")          # hybrid|ppo|sac
RL_INTRADAY_TIMESTEPS = int(os.environ.get("RL_INTRADAY_TIMESTEPS", 750_000))
RL_EVAL_FREQ         = int(os.environ.get("RL_EVAL_FREQ", 25_000))
RL_N_EVAL_EPISODES   = int(os.environ.get("RL_N_EVAL_EPISODES", 5))
# Canonical date-stamped output convention — mirror trade.py so producer/consumer tags
# can never drift (trade.py:118-175: DATA_TAG -> OUT_DIR -> RL_MODEL/METRICS paths).
PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from zoneinfo import ZoneInfo
    _TODAY_TAG = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
except Exception:  # zoneinfo/tzdata unavailable — UTC fallback (plan: fall back to UTC %Y%m%d)
    _TODAY_TAG = datetime.now(timezone.utc).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", _TODAY_TAG)).strip()
OUT_DIR  = PROJECT_ROOT / "system-data" / DATA_TAG

RL_SAVE_PATH         = os.environ.get("RL_SAVE_PATH", "models/rl_intraday")  # SB3 working/checkpoint dir (best_model + eval logs) — NOT the canonical artifact
RL_METRICS_PATH      = os.environ.get("RL_METRICS_PATH", str(OUT_DIR / f"{DATA_TAG}_rl_intraday_training_metrics.json"))  # canonical (AC-2)
RL_REPLAY_PATH       = os.environ.get("RL_REPLAY_PATH", "rl_trade_replay.csv")
RL_DATA_PATH         = os.environ.get("RL_DATA_PATH", str(OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv"))  # Alpaca-derived bars (AC-1)
# Reward shaping mode: "sharpe_rolling" (legacy default) | "ddr" (Differential Downside
# Deviation Ratio, Moody & Saffell 2001 — per-step-additive, downside-only). DDR is opt-in;
# empirical A/B vs baseline is deferred to the 03-03 eval harness.
RL_REWARD_MODE       = os.environ.get("RL_REWARD_MODE", "sharpe_rolling").strip().lower()
if RL_REWARD_MODE not in {"sharpe_rolling", "ddr"}:
    print(f"[reward] WARNING: unknown RL_REWARD_MODE={RL_REWARD_MODE!r} -- using sharpe_rolling", flush=True)
    RL_REWARD_MODE = "sharpe_rolling"
RL_REWARD_DDR_ETA    = float(os.environ.get("RL_REWARD_DDR_ETA", 0.02))
RL_MODEL_OUT         = str(OUT_DIR / f"{DATA_TAG}_rl_intraday_structure_model")  # SB3 appends .zip -> canonical (AC-2)

RISK_REWARD_TARGET   = float(os.environ.get("RL_RR_TARGET", 3.0))
MAX_BARS_IN_TRADE    = int(os.environ.get("RL_MAX_BARS_IN_TRADE", 60))
MAX_POSITIONS        = int(os.environ.get("RL_MAX_POSITIONS", 1))

# Reward shaping weights (tunable via env vars)
R_SUPPORT_TOUCH      = float(os.environ.get("R_SUPPORT_TOUCH",       0.20))
R_RECLAIM_SUP        = float(os.environ.get("R_RECLAIM_SUP",         0.25))
R_RECLAIM_VWAP       = float(os.environ.get("R_RECLAIM_VWAP",        0.20))
R_RECLAIM_EMA9       = float(os.environ.get("R_RECLAIM_EMA9",        0.15))
R_RECLAIM_EMA20      = float(os.environ.get("R_RECLAIM_EMA20",       0.15))
R_VOLUME_SPIKE       = float(os.environ.get("R_VOLUME_SPIKE",        0.20))
R_ABSORPTION         = float(os.environ.get("R_ABSORPTION",          0.20))
R_LOWER_WICK         = float(os.environ.get("R_LOWER_WICK",          0.10))
R_NEXT3_POS          = float(os.environ.get("R_NEXT3_POS",           0.15))
R_NEXT5_POS          = float(os.environ.get("R_NEXT5_POS",           0.20))
R_NEXT10_POS         = float(os.environ.get("R_NEXT10_POS",          0.25))
R_MFE_1R             = float(os.environ.get("R_MFE_1R",              0.30))
R_MFE_2R             = float(os.environ.get("R_MFE_2R",              0.50))
R_MFE_3R             = float(os.environ.get("R_MFE_3R",              1.00))
R_EXIT_NEAR_RES      = float(os.environ.get("R_EXIT_NEAR_RES",       0.25))
R_EXIT_3R            = float(os.environ.get("R_EXIT_3R",             0.50))
R_EXIT_NO_GIVEBACK   = float(os.environ.get("R_EXIT_NO_GIVEBACK",    0.20))
R_HOLD_WINNER        = float(os.environ.get("R_HOLD_WINNER",         0.05))
R_CONTINUATION       = float(os.environ.get("R_CONTINUATION",        0.10))
# Weight on the rolling-Sharpe reward term. The OOPS paper argues option strategies need
# utility / full-distribution objectives rather than mean-variance only; raising this default
# makes training more risk-adjusted while preserving the existing structure-shaping rewards.
R_SHARPE_WEIGHT      = float(os.environ.get("RL_SHARPE_REWARD_WEIGHT", 0.50))

P_BUY_INTO_RES       = float(os.environ.get("P_BUY_INTO_RES",       -0.40))
P_PREMATURE_ENTRY    = float(os.environ.get("P_PREMATURE_ENTRY",     -0.30))
P_NO_VOL_CONFIRM     = float(os.environ.get("P_NO_VOL_CONFIRM",     -0.20))
P_NEXT_BAR_BREAKS    = float(os.environ.get("P_NEXT_BAR_BREAKS",    -0.35))
P_EXCESSIVE_MAE      = float(os.environ.get("P_EXCESSIVE_MAE",      -0.40))
P_GIVEBACK           = float(os.environ.get("P_GIVEBACK",           -0.30))
P_MISSED_SETUP       = float(os.environ.get("P_MISSED_SETUP",       -0.15))
P_FAILED_STRUCTURE   = float(os.environ.get("P_FAILED_STRUCTURE",   -0.35))


# ---------------------------------------------------------------------------
# Setup type classifier
# ---------------------------------------------------------------------------

SETUP_TYPES = [
    "dip_reclaim",
    "support_reclaim",
    "vwap_reclaim",
    "breakout",
    "opening_range_reclaim",
    "liquidity_sweep_reclaim",
    "none",
]

def classify_setup(bar: Dict, prev_bar: Optional[Dict] = None) -> str:
    cl    = float(bar.get("close", 0.0))
    o     = float(bar.get("open",  cl))
    vwap  = float(bar.get("vwap",  cl))
    e9    = float(bar.get("ema9",  cl))
    sup   = float(bar.get("support_zone", 0.0))
    liq   = int(bar.get("liquidity_sweep_detected", 0))
    piv   = int(bar.get("pivot_low_reclaim", 0))
    mrr   = int(bar.get("missed_reversal_level_reclaim", 0))
    sbf   = int(bar.get("support_break_failed", 0))
    day_open = float(bar.get("day_open", o))

    if liq and (piv or sbf):
        return "liquidity_sweep_reclaim"
    if mrr:
        return "support_reclaim"
    if piv or sbf:
        return "dip_reclaim" if cl < day_open else "support_reclaim"
    if cl > vwap and (prev_bar and float(prev_bar.get("close", cl)) <= vwap):
        return "vwap_reclaim"
    bar_num = int(bar.get("total_bars_today", 99))
    if bar_num <= 15 and cl > day_open:
        return "opening_range_reclaim"
    return "none"


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

if SB3_AVAILABLE:
    class IntradayRLEnv(gym.Env):
        """
        Single-symbol intraday RL environment.
        Episode = one trading session (one day of 1-min bars).

        Observation : (N_CHANNELS, CNN_LOOKBACK) float32
        Action      : Discrete(3) — HOLD / ENTER_LONG / EXIT_LONG
        Reward      : shaped Sharpe delta + setup quality bonuses/penalties
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            bars_df:         pd.DataFrame,
            symbol:          str   = "UNKNOWN",
            lookback:        int   = CNN_LOOKBACK,
            rr_target:       float = RISK_REWARD_TARGET,
            max_bars_trade:  int   = MAX_BARS_IN_TRADE,
            replay_path:     str   = RL_REPLAY_PATH,
        ):
            super().__init__()
            self.bars_df        = bars_df.reset_index(drop=True)
            self.symbol         = symbol
            self.lookback       = lookback
            self.rr_target      = rr_target
            self.max_bars_trade = max_bars_trade
            self.replay_path    = replay_path

            self.observation_space = make_observation_space(lookback)
            self.action_space      = make_action_space()

            self._state_builder = PrivateMarketStateBuilder(lookback)
            self._sharpe_calc   = SharpeRewardCalculator(mode=RL_REWARD_MODE, eta=RL_REWARD_DDR_ETA)
            self._pivot_scanner = PivotReversalScanner(symbol=symbol)

            # Trade state
            self._in_position      = False
            self._entry_price      = 0.0
            self._stop_price       = 0.0
            self._target_price     = 0.0
            self._bars_in_trade    = 0
            self._mfe              = 0.0
            self._mae              = 0.0
            self._entry_setup_type = "none"
            self._prev_close       = 0.0

            # Session state
            self._t         = 0
            self._session_high = 0.0
            self._day_open     = 0.0
            self._red_streak   = 0
            self._green_streak = 0
            self._prev_bar: Optional[Dict] = None

            # Diagnostics
            self._ep_trades:  List[TradeRecord]  = []
            self._ep_rewards: List[float]        = []
            self.episode_metrics: Dict           = {}

            # Replay CSV header
            if not os.path.exists(self.replay_path):
                os.makedirs(os.path.dirname(self.replay_path) if os.path.dirname(self.replay_path) else ".", exist_ok=True)
                with open(self.replay_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=REPLAY_FIELDNAMES).writeheader()

        # ------------------------------------------------------------------

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._state_builder.reset()
            self._sharpe_calc.reset()
            self._pivot_scanner = PivotReversalScanner(symbol=self.symbol)

            self._in_position   = False
            self._entry_price   = 0.0
            self._stop_price    = 0.0
            self._target_price  = 0.0
            self._bars_in_trade = 0
            self._mfe           = 0.0
            self._mae           = 0.0
            self._entry_setup_type = "none"

            self._t             = self.lookback
            row                 = self.bars_df.iloc[self._t]
            self._day_open      = float(row.get("open", row.get("close", 1.0)))
            self._session_high  = self._day_open
            self._prev_close    = self._day_open
            self._red_streak    = 0
            self._green_streak  = 0
            self._prev_bar      = None

            # Warm up state builder
            for i in range(max(0, self._t - self.lookback), self._t):
                self._push_bar(i)

            self._ep_trades  = []
            self._ep_rewards = []

            obs = self._state_builder.build()
            return obs, {}

        def step(self, action: int):
            row = self.bars_df.iloc[self._t]
            bar = self._build_bar_dict(row)

            cl  = bar["close"]
            h   = bar["high"]
            l   = bar["low"]

            # Advance state builder
            self._push_bar_dict(bar)

            reward     = 0.0
            terminated = False
            truncated  = False
            info: Dict = {}

            setup_type = classify_setup(bar, self._prev_bar)

            # ---- Position management -------------------------------------
            if self._in_position:
                upside   = (h - self._entry_price) / (self._entry_price + 1e-8)
                downside = (self._entry_price - l)  / (self._entry_price + 1e-8)
                self._mfe = max(self._mfe, upside)
                self._mae = max(self._mae, downside)
                self._bars_in_trade += 1

                # Autonomous stop / target (overrides agent action)
                if l <= self._stop_price:
                    reward += self._close_trade(cl=self._stop_price, outcome="loss", bar=bar)
                elif h >= self._target_price:
                    reward += self._close_trade(cl=self._target_price, outcome="win", bar=bar)
                elif self._bars_in_trade >= self.max_bars_trade:
                    reward += self._close_trade(cl=cl, outcome="timeout", bar=bar)
                elif action == ACTION_EXIT_LONG:
                    # Agent-initiated exit
                    reward += self._close_trade(cl=cl, outcome="agent_exit", bar=bar)
                else:
                    # Holding
                    reward += self._hold_reward(bar, cl)

            else:
                # Not in position
                if action == ACTION_ENTER_LONG:
                    reward += self._entry_reward(bar, setup_type, cl)

                    # Detect high-quality missed setup for penalty (agent did NOT enter last bar)
                    # penalty applied in next step if warranted — tracked below
                elif action == ACTION_HOLD:
                    # Missed setup penalty
                    reward += self._missed_setup_penalty(bar, setup_type)

            # Sharpe component
            step_ret = (cl - self._prev_close) / (self._prev_close + 1e-8) if self._in_position else 0.0
            sharpe_r = self._sharpe_calc.update(step_ret)
            reward  += sharpe_r * R_SHARPE_WEIGHT   # Tran et al.: Sharpe reward (weight tunable)

            self._ep_rewards.append(reward)
            self._prev_close = cl
            self._prev_bar   = bar

            # Advance time
            self._t += 1
            if self._t >= len(self.bars_df) - 1:
                terminated = True
                if self._in_position:
                    self._close_trade(cl=cl, outcome="eod", bar=bar)
                self.episode_metrics = self._compute_episode_metrics()

            obs  = self._state_builder.build()
            info = {
                "setup_type":      setup_type,
                "in_position":     self._in_position,
                "sharpe":          self._sharpe_calc.current_sharpe,
                "episode_metrics": self.episode_metrics,
            }
            return obs, reward, terminated, truncated, info

        # ------------------------------------------------------------------
        # Reward helpers
        # ------------------------------------------------------------------

        def _entry_reward(self, bar: Dict, setup_type: str, cl: float) -> float:
            r   = 0.0
            sup = bar.get("support_zone", 0.0)
            res = bar.get("resistance_zone", cl * 1.1)
            vm  = bar.get("volume_ma20", 1.0)
            vol = bar.get("volume", 0.0)
            vwap = bar.get("vwap", cl)
            e9   = bar.get("ema9",  cl)
            e20  = bar.get("ema20", cl)
            lwr  = bar.get("lower_wick_ratio", 0.0)
            abs_score = bar.get("absorption_score", 0.0)

            # Quality bonuses
            if bar.get("support_touch"):            r += R_SUPPORT_TOUCH
            if cl > sup and sup > 0:                r += R_RECLAIM_SUP
            if cl > vwap:                           r += R_RECLAIM_VWAP
            if cl > e9:                             r += R_RECLAIM_EMA9
            if cl > e20:                            r += R_RECLAIM_EMA20
            if vol > vm * 1.5:                      r += R_VOLUME_SPIKE
            if abs_score > 0.5:                     r += R_ABSORPTION
            if lwr > 0.4:                           r += R_LOWER_WICK

            # Penalties
            if res > 0 and (res - cl) / (cl + 1e-8) < 0.005:
                r += P_BUY_INTO_RES
            if not (bar.get("pivot_low_reclaim") or bar.get("support_break_failed")
                    or bar.get("liquidity_sweep_detected")):
                r += P_PREMATURE_ENTRY
            if vol < vm * 0.8:
                r += P_NO_VOL_CONFIRM

            # Open position
            if not self._in_position:
                self._enter_long(bar, cl, sup)

            return r

        def _hold_reward(self, bar: Dict, cl: float) -> float:
            r = 0.0
            risk = (self._entry_price - self._stop_price) + 1e-8
            mfe_r = self._mfe * self._entry_price / risk

            if mfe_r >= 3.0:  r += R_MFE_3R
            elif mfe_r >= 2.0: r += R_MFE_2R
            elif mfe_r >= 1.0: r += R_MFE_1R

            if cl > self._entry_price:  r += R_HOLD_WINNER   # per-bar holding a winner

            # Continuation after reclaim
            if bar.get("pivot_low_reclaim") or bar.get("reclaim_vwap"):
                r += R_CONTINUATION

            # Giveback penalty
            prev_mfe = self._mfe
            if prev_mfe > 0 and (prev_mfe - (cl - self._entry_price) / (self._entry_price + 1e-8)) > 0.01:
                r += P_GIVEBACK

            return r

        def _missed_setup_penalty(self, bar: Dict, setup_type: str) -> float:
            if setup_type in ("liquidity_sweep_reclaim", "dip_reclaim", "support_reclaim"):
                return P_MISSED_SETUP
            return 0.0

        def _close_trade(self, cl: float, outcome: str, bar: Dict) -> float:
            r         = 0.0
            risk      = (self._entry_price - self._stop_price) + 1e-8
            pnl_r     = (cl - self._entry_price) / risk
            res       = bar.get("resistance_zone", cl * 1.1)
            mfe_r     = self._mfe * self._entry_price / risk

            if outcome == "win":
                r += R_EXIT_3R
                r += R_EXIT_NO_GIVEBACK
            if res > 0 and abs(cl - res) / (cl + 1e-8) < 0.005:
                r += R_EXIT_NEAR_RES

            # MAE penalty
            if self._mae * self._entry_price / risk > 1.5:
                r += P_EXCESSIVE_MAE

            if outcome == "loss" and self._in_position:
                # Holding through failed structure
                r += P_FAILED_STRUCTURE

            # Forward bars bonus (approximate — use current price direction)
            if outcome in ("win", "agent_exit") and cl > self._entry_price:
                r += R_NEXT3_POS  # at least 3 bars positive approximation

            # Log trade record
            rec = TradeRecord(
                timestamp            = str(bar.get("timestamp", "")),
                symbol               = self.symbol,
                setup_type           = self._entry_setup_type,
                entry_price          = self._entry_price,
                stop_price           = self._stop_price,
                target_price         = self._target_price,
                sweep_low            = float(bar.get("session_low", self._stop_price)),
                reclaim_price        = self._entry_price,
                volume_spike_ratio   = float(bar.get("volume_spike_ratio", 1.0)),
                support_distance_pct = (self._entry_price - float(bar.get("support_zone", self._entry_price)))
                                        / (self._entry_price + 1e-8),
                reclaim_vwap         = int(cl > float(bar.get("vwap", cl))),
                reclaim_ema9         = int(cl > float(bar.get("ema9", cl))),
                reclaim_ema20        = int(cl > float(bar.get("ema20", cl))),
                mfe_pct              = self._mfe,
                mae_pct              = self._mae,
                outcome              = outcome,
            )
            self._ep_trades.append(rec)
            self._append_replay(rec)

            self._in_position      = False
            self._entry_price      = 0.0
            self._stop_price       = 0.0
            self._target_price     = 0.0
            self._bars_in_trade    = 0
            self._mfe              = 0.0
            self._mae              = 0.0
            return r

        def _enter_long(self, bar: Dict, cl: float, sup: float):
            stop  = max(float(bar.get("session_low", cl * 0.99)), cl * 0.99)
            risk  = max(cl - stop, cl * 0.005)
            self._entry_price      = cl
            self._stop_price       = stop
            self._target_price     = cl + risk * self.rr_target
            self._bars_in_trade    = 0
            self._mfe              = 0.0
            self._mae              = 0.0
            self._in_position      = True
            self._entry_setup_type = classify_setup(bar, self._prev_bar)

        # ------------------------------------------------------------------
        # Bar dict construction
        # ------------------------------------------------------------------

        def _build_bar_dict(self, row) -> Dict:
            cl = float(row.get("close", row.get("Close", 1.0)))
            o  = float(row.get("open",  row.get("Open",  cl)))
            h  = float(row.get("high",  row.get("High",  cl)))
            l  = float(row.get("low",   row.get("Low",   cl)))
            v  = float(row.get("volume",row.get("Volume",1e5)))

            self._session_high = max(self._session_high, h)
            if cl >= o: self._green_streak += 1; self._red_streak = 0
            else:        self._red_streak  += 1; self._green_streak = 0

            # Compute pivot labels
            lbl = self._pivot_scanner.push(dict(
                timestamp = str(row.get("timestamp", row.get("date", self._t))),
                open=o, high=h, low=l, close=cl, volume=v,
            ))

            vol_ma = float(row.get("volume_ma20", v))
            vwap   = float(row.get("vwap",  cl))
            e9     = float(row.get("ema9",  cl))
            e20    = float(row.get("ema20", cl))
            rvol   = v / (vol_ma + 1e-8)

            price_rng = max(h - l, 1e-8)
            body      = abs(cl - o)
            lwr       = (min(o, cl) - l) / price_rng
            uwr       = (h - max(o, cl)) / price_rng
            abs_score = (v / (vol_ma + 1e-8) - 1.0) * (1.0 - body / price_rng)

            bar = dict(
                timestamp               = str(row.get("timestamp", self._t)),
                open                    = o,
                high                    = h,
                low                     = l,
                close                   = cl,
                volume                  = v,
                vwap                    = vwap,
                ema9                    = e9,
                ema20                   = e20,
                session_high            = self._session_high,
                session_low             = float(row.get("session_low",  l)),
                prev_close              = self._prev_close,
                prev_low                = float(row.get("prev_low",     l)),
                prev_high               = float(row.get("prev_high",    h)),
                support_zone            = lbl.nearest_support or float(row.get("support_zone", l * 0.99)),
                resistance_zone         = lbl.nearest_resistance or float(row.get("resistance_zone", h * 1.01)),
                volume_ma20             = vol_ma,
                rvol                    = rvol,
                day_open                = self._day_open,
                total_bars_today        = self._t,
                lower_wick_ratio        = lwr,
                upper_wick_ratio        = uwr,
                body_to_range_ratio     = body / price_rng,
                red_candle_streak       = self._red_streak,
                green_candle_streak     = self._green_streak,
                volume_spike_ratio      = rvol,
                absorption_score        = abs_score,
                # pivot scanner flags
                support_break_failed    = lbl.support_break_failed,
                liquidity_sweep_detected= lbl.liquidity_sweep_detected,
                pivot_low_reclaim       = lbl.pivot_low_reclaim,
                missed_reversal_level_reclaim = lbl.missed_reversal_level_reclaim,
                support_touch           = lbl.support_touch,
                # position state
                in_position             = int(self._in_position),
                unrealised_pnl          = (cl - self._entry_price) if self._in_position else 0.0,
                bars_in_trade           = self._bars_in_trade,
            )
            return bar

        def _push_bar(self, idx: int):
            row = self.bars_df.iloc[idx]
            bar = self._build_bar_dict(row)
            self._state_builder.push(bar)

        def _push_bar_dict(self, bar: Dict):
            self._state_builder.push(bar)

        # ------------------------------------------------------------------
        # Replay / metrics
        # ------------------------------------------------------------------

        def _append_replay(self, rec: TradeRecord):
            try:
                with open(self.replay_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=REPLAY_FIELDNAMES)
                    row = asdict(rec)
                    w.writerow({k: row.get(k, "") for k in REPLAY_FIELDNAMES})
            except Exception as e:
                logger.error("Replay write error: %s", e)

        def _compute_episode_metrics(self) -> Dict:
            trades = self._ep_trades
            if not trades:
                return dict(
                    n_trades=0, win_rate=0.0, profit_factor=0.0,
                    avg_mfe_pct=0.0, avg_mae_pct=0.0, avg_capture_ratio=0.0,
                    dip_reclaim_entries=0, dip_reclaim_win_rate=0.0,
                    support_reclaim_entries=0, support_reclaim_win_rate=0.0,
                    missed_dip_reclaim_count=0,
                    sharpe=self._sharpe_calc.current_sharpe,
                    total_pnl=0.0,
                )
            wins  = [t for t in trades if "win" in t.outcome]
            dips  = [t for t in trades if t.setup_type in ("dip_reclaim","liquidity_sweep_reclaim")]
            sups  = [t for t in trades if t.setup_type == "support_reclaim"]
            dip_w = [t for t in dips if "win" in t.outcome]
            sup_w = [t for t in sups if "win" in t.outcome]
            pnl   = sum((t.mfe_pct if "win" in t.outcome else -t.mae_pct) for t in trades)

            return dict(
                n_trades                = len(trades),
                win_rate                = len(wins) / len(trades),
                profit_factor           = (sum(t.mfe_pct for t in wins) + 1e-8)
                                         / (sum(t.mae_pct for t in trades if "loss" in t.outcome) + 1e-8),
                avg_mfe_pct             = sum(t.mfe_pct for t in trades) / len(trades),
                avg_mae_pct             = sum(t.mae_pct for t in trades) / len(trades),
                avg_capture_ratio       = sum(t.mfe_pct/(t.mfe_pct+t.mae_pct+1e-8) for t in trades) / len(trades),
                dip_reclaim_entries     = len(dips),
                dip_reclaim_win_rate    = len(dip_w) / (len(dips) + 1e-8),
                support_reclaim_entries = len(sups),
                support_reclaim_win_rate= len(sup_w) / (len(sups) + 1e-8),
                missed_dip_reclaim_count= 0,   # populated by monitor
                sharpe                  = self._sharpe_calc.current_sharpe,
                total_pnl               = pnl,
            )


# ---------------------------------------------------------------------------
# Diagnostics callback
# ---------------------------------------------------------------------------

if SB3_AVAILABLE:
    class DiagnosticsCallback(BaseCallback):
        """Logs dip-reclaim metrics to file after each rollout."""

        def __init__(self, metrics_path: str = RL_METRICS_PATH, verbose: int = 0):
            super().__init__(verbose)
            self.metrics_path = metrics_path
            self._all_metrics: List[Dict] = []
            os.makedirs(os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else ".", exist_ok=True)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> bool:
            env = self.training_env
            if hasattr(env, "envs"):
                for e in env.envs:
                    inner = e.env if hasattr(e, "env") else e
                    if hasattr(inner, "episode_metrics") and inner.episode_metrics:
                        self._all_metrics.append(inner.episode_metrics)
            return True

        def _on_training_end(self):
            with open(self.metrics_path, "w") as f:
                json.dump(self._all_metrics[-1] if self._all_metrics else {}, f, indent=2)
            logger.info("Diagnostics saved → %s", self.metrics_path)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def load_bars(path: str) -> pd.DataFrame:
    # Provenance: bars are produced by build_intraday_training_dataset.py (Alpaca IEX). (AC-3)
    logger.info("[data-source] training_csv=alpaca_derived path=%s", path)
    print(f"[data-source] training_csv=alpaca_derived path={path}", flush=True)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported data format: {path}")


def train(
    data_path:  str = RL_DATA_PATH,
    timesteps:  int = RL_INTRADAY_TIMESTEPS,
    save_path:  str = RL_SAVE_PATH,
    policy_mode:str = RL_POLICY_MODE,
):
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 and gymnasium required for training.")

    logger.info("Loading data from %s", data_path)
    df     = load_bars(data_path)
    symbol = os.path.basename(data_path).split("_")[0].upper()

    n      = len(df)
    split  = int(n * 0.85)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    logger.info("Train bars: %d  |  Eval bars: %d", len(train_df), len(eval_df))

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(RL_METRICS_PATH) if os.path.dirname(RL_METRICS_PATH) else ".", exist_ok=True)

    def make_env(bars):
        def _init():
            env = IntradayRLEnv(bars, symbol=symbol, replay_path=RL_REPLAY_PATH)
            return Monitor(env)
        return _init

    train_env = DummyVecEnv([make_env(train_df)])
    if RL_REWARD_MODE == "ddr":
        # DDR is already a normalised downside-risk ratio and can spike on losses (03-01
        # DISCOVERY); don't re-normalise reward, just widen the clip to bound outliers.
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False, clip_reward=100.0)
    else:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env  = DummyVecEnv([make_env(eval_df)])
    eval_env  = VecNormalize(eval_env,  norm_obs=False, norm_reward=False, training=False)

    policy_kwargs = get_policy_kwargs()

    if policy_mode in ("ppo", "hybrid"):
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs   = policy_kwargs,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            verbose         = 1,
            tensorboard_log = "logs/rl_intraday_tb",
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            policy_kwargs   = policy_kwargs,
            learning_rate   = 3e-4,
            buffer_size     = 100_000,
            batch_size      = 256,
            gamma           = 0.99,
            verbose         = 1,
            tensorboard_log = "logs/rl_intraday_tb",
        )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path = save_path,
            log_path             = save_path,
            eval_freq            = RL_EVAL_FREQ,
            n_eval_episodes      = RL_N_EVAL_EPISODES,
            deterministic        = True,
            render               = False,
        ),
        DiagnosticsCallback(metrics_path=RL_METRICS_PATH),
    ]

    logger.info("[reward] mode=%s eta=%s", RL_REWARD_MODE, RL_REWARD_DDR_ETA)
    logger.info("Training %s for %d timesteps …", policy_mode.upper(), timesteps)
    model.learn(total_timesteps=timesteps, callback=callbacks)
    os.makedirs(OUT_DIR, exist_ok=True)          # ensure canonical dir exists before save
    model.save(RL_MODEL_OUT)                      # canonical: system-data/{DATE}/{DATE}_rl_intraday_structure_model.zip (AC-2)
    logger.info("Model saved → %s.zip", RL_MODEL_OUT)

    # Load and print metrics
    try:
        with open(RL_METRICS_PATH) as f:
            m = json.load(f)
        _print_validation_summary(m, save_path)
    except Exception:
        pass

    return model


def _print_validation_summary(m: Dict, save_path: str):
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("  Metrics path          : %s", RL_METRICS_PATH)
    logger.info("  Best params path      : %s", save_path)
    logger.info("  Profit factor         : %.3f", m.get("profit_factor", 0))
    logger.info("  Win rate              : %.1f%%", m.get("win_rate", 0) * 100)
    logger.info("  Avg MFE               : %.2f%%", m.get("avg_mfe_pct", 0) * 100)
    logger.info("  Avg MAE               : %.2f%%", m.get("avg_mae_pct", 0) * 100)
    logger.info("  Capture ratio         : %.2f",   m.get("avg_capture_ratio", 0))
    logger.info("  Dip-reclaim entries   : %d",     m.get("dip_reclaim_entries", 0))
    logger.info("  Dip-reclaim win rate  : %.1f%%", m.get("dip_reclaim_win_rate", 0) * 100)
    logger.info("  Support reclaim WR    : %.1f%%", m.get("support_reclaim_win_rate", 0) * 100)
    logger.info("  Missed dip reclaims   : %d",     m.get("missed_dip_reclaim_count", 0))
    logger.info("  Sharpe                : %.3f",   m.get("sharpe", 0))
    logger.info("  Total PnL (sim)       : %.4f",   m.get("total_pnl", 0))
    logger.info("=" * 60)

    # Self-assessment
    pf   = m.get("profit_factor", 0)
    drwr = m.get("dip_reclaim_win_rate", 0)
    needed = []
    if pf < 1.3:    needed.append("profit_factor too low — increase R_MFE_3R, decrease P_MISSED_SETUP weight")
    if drwr < 0.45: needed.append("dip_reclaim_win_rate low — add more absorption/lower_wick bonuses, check volume filters")
    if m.get("n_trades", 0) < 5: needed.append("model is inactive — decrease P_PREMATURE_ENTRY, increase P_MISSED_SETUP")
    if m.get("avg_mae_pct", 1) > 0.02: needed.append("excessive MAE — tighten stop logic, increase P_EXCESSIVE_MAE penalty")

    if needed:
        logger.warning("Model did NOT fully learn the screenshot-style setup. Improvements needed:")
        for n in needed:
            logger.warning("  → %s", n)
    else:
        logger.info("✓ Model appears to have learned dip-reclaim structure.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()