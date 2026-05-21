#!/usr/bin/env python3
"""
rl_intraday_structure_trainer.py

Advanced reinforcement-learning trainer for intraday execution timing using:

    system-data/intraday_bars_1m.csv
    system-data/intraday_structure_map.json

Default usage:
    python rl_intraday_structure_trainer.py

This integrated version is long-only and trains HOLD / ENTER_LONG / EXIT_LONG / ADD_LONG execution timing.

This version supports:
    - Standard PPO MLP policy
    - Optional PPO-LSTM via sb3-contrib RecurrentPPO
    - Transformer-style sequence feature fusion
    - Hierarchical RL action gating
    - Multi-timeframe state fusion from 1m bars
    - Long-only actions
    - Action confidence/entropy output for run.py
    - Trend-persistence rewards
    - Exit-quality penalties
    - Peak-exit reward shaping to learn exits near local highs before giveback

Install optional LSTM support:
    pip install sb3-contrib

Expected intraday bars columns:
    timestamp, symbol, open, high, low, close, volume

Actions:
    0 = HOLD
    1 = ENTER_LONG
    2 = EXIT_LONG
    3 = ADD_LONG

Outputs:
    system-data/rl_intraday_structure_model.zip
    system-data/rl_intraday_training_metrics.json
    system-data/rl_intraday_signal_diagnostics.csv
    system-data/rl_intraday_symbol_overlap.csv

This file does NOT place trades.
It trains a policy that run.py can use as a confirmation/timing layer.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from datetime import timedelta


import numpy as np
import pandas as pd


ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
DATA_TAG = os.getenv("RL_DATA_DATE", TODAY_TAG)

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:
    raise RuntimeError("Missing gymnasium. Install it with: pip install gymnasium") from exc

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as exc:
    raise RuntimeError(
        "Missing stable-baselines3/torch. Install with: pip install stable-baselines3 torch"
    ) from exc

try:
    from sb3_contrib import RecurrentPPO # type: ignore
except Exception:
    RecurrentPPO = None


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BARS_PATHS = [
    OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv",
    OUT_DIR / f"{DATA_TAG}_intraday_bars.csv",
    OUT_DIR / f"{DATA_TAG}_latest_intraday_bars.csv",
]

STRUCTURE_MAP_PATH = OUT_DIR / f"{DATA_TAG}_intraday_structure_map.json"

MODEL_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_structure_model.zip"
METRICS_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_training_metrics.json"
DIAGNOSTICS_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_signal_diagnostics.csv"
OVERLAP_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_symbol_overlap.csv"


# =============================================================================
# Config
# =============================================================================

RANDOM_SEED = int(os.getenv("RL_INTRADAY_RANDOM_SEED", "42"))

TRAIN_TIMESTEPS = int(os.getenv("RL_INTRADAY_TIMESTEPS", "500000"))
TRAIN_SYMBOL_LIMIT = int(os.getenv("RL_INTRADAY_SYMBOL_LIMIT", "30"))
MAX_EPISODE_STEPS = int(os.getenv("RL_INTRADAY_MAX_EPISODE_STEPS", "390"))
MIN_BARS_PER_EPISODE = int(os.getenv("RL_INTRADAY_MIN_BARS_PER_EPISODE", "30"))

STARTING_CASH = float(os.getenv("RL_INTRADAY_STARTING_CASH", "100000"))
POSITION_FRACTION = float(os.getenv("RL_INTRADAY_POSITION_FRACTION", "0.10"))

COMMISSION_BPS = float(os.getenv("RL_INTRADAY_COMMISSION_BPS", "1.0"))
SLIPPAGE_BPS = float(os.getenv("RL_INTRADAY_SLIPPAGE_BPS", "2.0"))

STOP_LOSS_PCT = float(os.getenv("RL_INTRADAY_STOP_LOSS_PCT", "0.012"))
TAKE_PROFIT_PCT = float(os.getenv("RL_INTRADAY_TAKE_PROFIT_PCT", "0.024"))
MAX_HOLD_BARS = int(os.getenv("RL_INTRADAY_MAX_HOLD_BARS", "60"))
MIN_QUALITY_HOLD_BARS = int(os.getenv("RL_MIN_QUALITY_HOLD_BARS", "3"))

REWARD_SCALE = float(os.getenv("RL_INTRADAY_REWARD_SCALE", "100.0"))
OVERTRADE_PENALTY = float(os.getenv("RL_INTRADAY_OVERTRADE_PENALTY", "0.0002"))
DRAWDOWN_PENALTY = float(os.getenv("RL_INTRADAY_DRAWDOWN_PENALTY", "0.20"))
BAD_ENTRY_PENALTY = float(os.getenv("RL_INTRADAY_BAD_ENTRY_PENALTY", "0.0001"))
GOOD_ENTRY_BONUS = float(os.getenv("RL_INTRADAY_GOOD_ENTRY_BONUS", "0.010"))

# Exploration / anti-inactivity shaping. These prevent PPO from learning that
# staying flat forever is the safest optimal policy during training.
FLAT_HOLD_INACTIVITY_PENALTY = float(os.getenv("RL_FLAT_HOLD_INACTIVITY_PENALTY", "0.003"))
FLAT_HOLD_STRONG_SETUP_PENALTY = float(os.getenv("RL_FLAT_HOLD_STRONG_SETUP_PENALTY", "0.004"))
VALID_ENTRY_EXPLORATION_BONUS = float(os.getenv("RL_VALID_ENTRY_EXPLORATION_BONUS", "0.006"))
EARLY_TRAINING_GATE_RELAXATION = os.getenv("RL_EARLY_TRAINING_GATE_RELAXATION", "1").lower() in {"1", "true", "yes"}

# Model mode:
#   ppo          = standard MLP PPO
#   lstm         = RecurrentPPO LSTM if sb3-contrib installed
#   transformer  = standard PPO with transformer-style sequential feature engineering
#   hybrid       = transformer-style features + hierarchical gating, standard PPO
RL_POLICY_MODE = os.getenv("RL_POLICY_MODE", "hybrid").lower().strip()
USE_LSTM_POLICY = RL_POLICY_MODE in {"lstm", "ppo_lstm", "recurrent"}
USE_TRANSFORMER_FEATURES = os.getenv("RL_USE_TRANSFORMER_FEATURES", "1").lower() in {"1", "true", "yes"} or RL_POLICY_MODE in {"transformer", "hybrid", "lstm"}
USE_HIERARCHICAL_GATING = os.getenv("RL_USE_HIERARCHICAL_GATING", "1").lower() in {"1", "true", "yes"} or RL_POLICY_MODE == "hybrid"
USE_MULTI_TIMEFRAME_FUSION = os.getenv("RL_USE_MULTI_TIMEFRAME_FUSION", "1").lower() in {"1", "true", "yes"}

SEQUENCE_WINDOWS = [int(x) for x in os.getenv("RL_SEQUENCE_WINDOWS", "3,5,10,20").split(",") if x.strip()]
MTF_WINDOWS = [int(x) for x in os.getenv("RL_MTF_WINDOWS", "5,15,30,60").split(",") if x.strip()]

# Entry rewards
BUY_BREAKOUT_BONUS = float(os.getenv("RL_BUY_BREAKOUT_BONUS", "0.012"))
BUY_VWAP_BONUS = float(os.getenv("RL_BUY_VWAP_BONUS", "0.006"))
BUY_BREAKOUT_SCORE_BONUS = float(os.getenv("RL_BUY_BREAKOUT_SCORE_BONUS", "0.009"))
BUY_TREND_SCORE_BONUS = float(os.getenv("RL_BUY_TREND_SCORE_BONUS", "0.009"))
BUY_NEAR_SUPPORT_BONUS = float(os.getenv("RL_BUY_NEAR_SUPPORT_BONUS", "0.014"))
BUY_NEAR_RESISTANCE_PENALTY = float(os.getenv("RL_BUY_NEAR_RESISTANCE_PENALTY", "0.030"))

SHORT_BREAKDOWN_BONUS = float(os.getenv("RL_SHORT_BREAKDOWN_BONUS", "0.016"))
SHORT_BELOW_VWAP_BONUS = float(os.getenv("RL_SHORT_BELOW_VWAP_BONUS", "0.008"))
SHORT_TREND_SCORE_BONUS = float(os.getenv("RL_SHORT_TREND_SCORE_BONUS", "0.008"))
SHORT_NEAR_RESISTANCE_BONUS = float(os.getenv("RL_SHORT_NEAR_RESISTANCE_BONUS", "0.012"))
SHORT_NEAR_SUPPORT_PENALTY = float(os.getenv("RL_SHORT_NEAR_SUPPORT_PENALTY", "0.020"))

MISSED_BREAKOUT_HOLD_PENALTY = float(os.getenv("RL_MISSED_BREAKOUT_HOLD_PENALTY", "0.015"))
MISSED_BREAKDOWN_HOLD_PENALTY = float(os.getenv("RL_MISSED_BREAKDOWN_HOLD_PENALTY", "0.012"))

# Execution-quality reward shaping. These are deliberately small compared with
# realized-PnL rewards, but they teach the policy timing quality:
# immediate next-bar continuation, positive daily/intraday momentum, avoiding
# re-entry churn, avoiding equity drawdowns, and preferring multi-bar winners
# over one-bar scalps.
NEXT_BAR_REWARD_MULT = float(os.getenv("RL_NEXT_BAR_REWARD_MULT", "300.0"))
NEXT_BAR_REWARD_CAP = float(os.getenv("RL_NEXT_BAR_REWARD_CAP", "2.0"))
NEXT_BAR_NEG_MULT = float(os.getenv("RL_NEXT_BAR_NEG_MULT", "400.0"))
NEXT_BAR_NEG_CAP = float(os.getenv("RL_NEXT_BAR_NEG_CAP", "3.0"))
NEXT_BAR_STRONG_MOVE = float(os.getenv("RL_NEXT_BAR_STRONG_MOVE", "0.0025"))
NEXT_BAR_STRONG_BONUS = float(os.getenv("RL_NEXT_BAR_STRONG_BONUS", "1.0"))
NEXT_BAR_STRONG_PENALTY = float(os.getenv("RL_NEXT_BAR_STRONG_PENALTY", "1.5"))
NEXT_3BAR_REWARD_MULT = float(os.getenv("RL_NEXT_3BAR_REWARD_MULT", "200.0"))
NEXT_3BAR_REWARD_CAP = float(os.getenv("RL_NEXT_3BAR_REWARD_CAP", "1.5"))
NEXT_5BAR_REWARD_MULT = float(os.getenv("RL_NEXT_5BAR_REWARD_MULT", "150.0"))
NEXT_5BAR_REWARD_CAP = float(os.getenv("RL_NEXT_5BAR_REWARD_CAP", "2.0"))
NEXT_3BAR_NEG_MULT = float(os.getenv("RL_NEXT_3BAR_NEG_MULT", "250.0"))
NEXT_3BAR_NEG_CAP = float(os.getenv("RL_NEXT_3BAR_NEG_CAP", "2.0"))
NEXT_5BAR_NEG_MULT = float(os.getenv("RL_NEXT_5BAR_NEG_MULT", "200.0"))
NEXT_5BAR_NEG_CAP = float(os.getenv("RL_NEXT_5BAR_NEG_CAP", "2.5"))

DAILY_MOMENTUM_BONUS_MULT = float(os.getenv("RL_DAILY_MOMENTUM_BONUS_MULT", "4.0"))
DAILY_MOMENTUM_BONUS_CAP = float(os.getenv("RL_DAILY_MOMENTUM_BONUS_CAP", "0.40"))
INTRADAY_MOMENTUM_BONUS_MULT = float(os.getenv("RL_INTRADAY_MOMENTUM_BONUS_MULT", "5.0"))
INTRADAY_MOMENTUM_BONUS_CAP = float(os.getenv("RL_INTRADAY_MOMENTUM_BONUS_CAP", "0.30"))
OVEREXTENDED_DAILY_CHANGE = float(os.getenv("RL_OVEREXTENDED_DAILY_CHANGE", "0.08"))
OVEREXTENDED_INTRADAY_CHANGE = float(os.getenv("RL_OVEREXTENDED_INTRADAY_CHANGE", "0.04"))
OVEREXTENDED_DAILY_PENALTY = float(os.getenv("RL_OVEREXTENDED_DAILY_PENALTY", "0.30"))
OVEREXTENDED_INTRADAY_PENALTY = float(os.getenv("RL_OVEREXTENDED_INTRADAY_PENALTY", "0.20"))

POSITION_PNL_BONUS_MULT = float(os.getenv("RL_POSITION_PNL_BONUS_MULT", "30.0"))
POSITION_PNL_BONUS_CAP = float(os.getenv("RL_POSITION_PNL_BONUS_CAP", "1.25"))
POSITION_PNL_NEG_MULT = float(os.getenv("RL_POSITION_PNL_NEG_MULT", "45.0"))
POSITION_PNL_NEG_CAP = float(os.getenv("RL_POSITION_PNL_NEG_CAP", "2.0"))
FAST_PROFIT_THRESHOLD = float(os.getenv("RL_FAST_PROFIT_THRESHOLD", "0.003"))
FAST_PROFIT_BARS = int(os.getenv("RL_FAST_PROFIT_BARS", "5"))
FAST_PROFIT_BONUS = float(os.getenv("RL_FAST_PROFIT_BONUS", "0.75"))
INSTANT_DRAWDOWN_THRESHOLD = float(os.getenv("RL_INSTANT_DRAWDOWN_THRESHOLD", "-0.003"))
INSTANT_DRAWDOWN_PENALTY = float(os.getenv("RL_INSTANT_DRAWDOWN_PENALTY", "1.5"))
STAGNANT_POSITION_BARS = int(os.getenv("RL_STAGNANT_POSITION_BARS", "20"))
STAGNANT_POSITION_THRESHOLD = float(os.getenv("RL_STAGNANT_POSITION_THRESHOLD", "0.001"))
STAGNANT_POSITION_PENALTY = float(os.getenv("RL_STAGNANT_POSITION_PENALTY", "0.25"))

REENTRY_SAME_SYMBOL_PENALTY = float(os.getenv("RL_REENTRY_SAME_SYMBOL_PENALTY", "0.05"))
REENTRY_RECENT_EXIT_BARS = int(os.getenv("RL_REENTRY_RECENT_EXIT_BARS", "30"))
REENTRY_RECENT_EXIT_PENALTY = float(os.getenv("RL_REENTRY_RECENT_EXIT_PENALTY", "0.05"))
REENTRY_COUNT_TRIGGER = int(os.getenv("RL_REENTRY_COUNT_TRIGGER", "2"))
REENTRY_COUNT_PENALTY = float(os.getenv("RL_REENTRY_COUNT_PENALTY", "0.10"))

EQUITY_NEG_1STEP_MULT = float(os.getenv("RL_EQUITY_NEG_1STEP_MULT", "50.0"))
EQUITY_NEG_1STEP_CAP = float(os.getenv("RL_EQUITY_NEG_1STEP_CAP", "1.0"))
DAILY_EQUITY_DD_1 = float(os.getenv("RL_DAILY_EQUITY_DD_1", "-0.005"))
DAILY_EQUITY_DD_2 = float(os.getenv("RL_DAILY_EQUITY_DD_2", "-0.010"))
DAILY_EQUITY_DD_1_PENALTY = float(os.getenv("RL_DAILY_EQUITY_DD_1_PENALTY", "1.0"))
DAILY_EQUITY_DD_2_PENALTY = float(os.getenv("RL_DAILY_EQUITY_DD_2_PENALTY", "2.0"))

NEG_DAILY_MOMENTUM_MULT = float(os.getenv("RL_NEG_DAILY_MOMENTUM_MULT", "8.0"))
NEG_DAILY_MOMENTUM_CAP = float(os.getenv("RL_NEG_DAILY_MOMENTUM_CAP", "0.80"))
NEG_INTRADAY_MOMENTUM_MULT = float(os.getenv("RL_NEG_INTRADAY_MOMENTUM_MULT", "10.0"))
NEG_INTRADAY_MOMENTUM_CAP = float(os.getenv("RL_NEG_INTRADAY_MOMENTUM_CAP", "1.00"))
NEG_DAILY_BREAKDOWN_LEVEL = float(os.getenv("RL_NEG_DAILY_BREAKDOWN_LEVEL", "-0.02"))
NEG_INTRADAY_BREAKDOWN_LEVEL = float(os.getenv("RL_NEG_INTRADAY_BREAKDOWN_LEVEL", "-0.01"))
NEG_DAILY_BREAKDOWN_PENALTY = float(os.getenv("RL_NEG_DAILY_BREAKDOWN_PENALTY", "0.75"))
NEG_INTRADAY_BREAKDOWN_PENALTY = float(os.getenv("RL_NEG_INTRADAY_BREAKDOWN_PENALTY", "0.50"))

MISS_FUTURE_5M_THRESHOLD = float(os.getenv("RL_MISS_FUTURE_5M_THRESHOLD", "0.005"))
MISS_FUTURE_15M_THRESHOLD = float(os.getenv("RL_MISS_FUTURE_15M_THRESHOLD", "0.010"))
MISS_STRONG_SETUP_PENALTY = float(os.getenv("RL_MISS_STRONG_SETUP_PENALTY", "1.25"))
MISS_BREAKOUT_PENALTY = float(os.getenv("RL_MISS_BREAKOUT_PENALTY", "2.0"))
AVOID_BAD_TRADE_REWARD = float(os.getenv("RL_AVOID_BAD_TRADE_REWARD", "0.25"))
AVOID_ADVERSE_TRADE_REWARD = float(os.getenv("RL_AVOID_ADVERSE_TRADE_REWARD", "0.50"))

SCALP_TINY_MOVE_PENALTY = float(os.getenv("RL_SCALP_TINY_MOVE_PENALTY", "0.75"))
SCALP_LOSER_PENALTY = float(os.getenv("RL_SCALP_LOSER_PENALTY", "0.10"))
CHOP_STAGNANT_PENALTY = float(os.getenv("RL_CHOP_STAGNANT_PENALTY", "1.00"))
ENTRY_CROSS_PENALTY = float(os.getenv("RL_ENTRY_CROSS_PENALTY", "1.00"))
VWAP_CROSS_PENALTY = float(os.getenv("RL_VWAP_CROSS_PENALTY", "1.25"))
FAILED_BREAKOUT_PENALTY = float(os.getenv("RL_FAILED_BREAKOUT_PENALTY", "0.10"))
GIVEBACK_AFTER_MFE_PENALTY = float(os.getenv("RL_GIVEBACK_AFTER_MFE_PENALTY", "2.25"))
DUAL_TREND_FAILURE_PENALTY = float(os.getenv("RL_DUAL_TREND_FAILURE_PENALTY", "0.75"))

REGIME_ALIGNED_REWARD = float(os.getenv("RL_REGIME_ALIGNED_REWARD", "0.004"))
REGIME_MISALIGNED_PENALTY = float(os.getenv("RL_REGIME_MISALIGNED_PENALTY", "0.006"))

# Long-side trend persistence rewards
LONG_STRONG_TREND_BONUS = float(os.getenv("RL_LONG_STRONG_TREND_BONUS", "0.020"))
LONG_POSITIVE_RET5_BONUS = float(os.getenv("RL_LONG_POSITIVE_RET5_BONUS", "0.010"))
LONG_POSITIVE_EMA_SPREAD_BONUS = float(os.getenv("RL_LONG_POSITIVE_EMA_SPREAD_BONUS", "0.010"))
LONG_BULLISH_MARKET_BONUS = float(os.getenv("RL_LONG_BULLISH_MARKET_BONUS", "0.020"))
LONG_SECTOR_STRENGTH_BONUS = float(os.getenv("RL_LONG_SECTOR_STRENGTH_BONUS", "0.010"))
LONG_BREAKOUT_PERSISTENCE_BONUS = float(os.getenv("RL_LONG_BREAKOUT_PERSISTENCE_BONUS", "0.012"))
LONG_BULLISH_ACCELERATION_BONUS = float(os.getenv("RL_LONG_BULLISH_ACCELERATION_BONUS", "0.012"))
LONG_POSITIVE_VOLUME_EXPANSION_BONUS = float(os.getenv("RL_LONG_POSITIVE_VOLUME_EXPANSION_BONUS", "0.010"))
LONG_HIGHER_HIGH_BONUS = float(os.getenv("RL_LONG_HIGHER_HIGH_BONUS", "0.008"))
LONG_HIGHER_LOW_BONUS = float(os.getenv("RL_LONG_HIGHER_LOW_BONUS", "0.008"))

HOLD_TREND_PERSISTENCE_REWARD = float(os.getenv("RL_HOLD_TREND_PERSISTENCE_REWARD", "0.018"))
HOLD_BREAKOUT_PERSISTENCE_REWARD = float(os.getenv("RL_HOLD_BREAKOUT_PERSISTENCE_REWARD", "0.022"))
PREMATURE_EXIT_PENALTY = float(os.getenv("RL_PREMATURE_EXIT_PENALTY", "0.025"))
STRONG_TREND_EXIT_PENALTY = float(os.getenv("RL_STRONG_TREND_EXIT_PENALTY", "0.035"))


# Peak-exit / local-top reward shaping.
# The agent is NOT given an exact top-tick target at inference. These forward
# labels are used only during training to reward exits with high efficiency
# relative to the next local high and penalize exits that leave substantial
# near-term upside on the table.
PEAK_EXIT_LOOKAHEAD_BARS = int(os.getenv("RL_PEAK_EXIT_LOOKAHEAD_BARS", "15"))
PEAK_EXIT_MIN_MFE = float(os.getenv("RL_PEAK_EXIT_MIN_MFE", "0.006"))
PEAK_EXIT_GIVEBACK_TRIGGER = float(os.getenv("RL_MAX_GIVEBACK_FROM_MFE", os.getenv("RL_PEAK_EXIT_GIVEBACK_TRIGGER", "0.006")))
PEAK_EXIT_EFFICIENCY_BONUS = float(os.getenv("RL_PEAK_EXIT_EFFICIENCY_MULT", os.getenv("RL_PEAK_EXIT_EFFICIENCY_BONUS", "3.0")))
PEAK_EXIT_NEAR_TOP_BONUS = float(os.getenv("RL_EXIT_NEAR_HIGH_BONUS", os.getenv("RL_PEAK_EXIT_NEAR_TOP_BONUS", "2.5")))
PEAK_EXIT_MISSED_UPSIDE_PENALTY_MULT = float(os.getenv("RL_MISSED_UPSIDE_PENALTY", os.getenv("RL_PEAK_EXIT_MISSED_UPSIDE_PENALTY_MULT", "2.0")))
PEAK_EXIT_MISSED_UPSIDE_CAP = float(os.getenv("RL_PEAK_EXIT_MISSED_UPSIDE_CAP", "2.0"))
PEAK_EXIT_GIVEBACK_AVOID_BONUS_MULT = float(os.getenv("RL_PEAK_EXIT_GIVEBACK_AVOID_BONUS_MULT", "120.0"))
PEAK_EXIT_GIVEBACK_AVOID_BONUS_CAP = float(os.getenv("RL_PEAK_EXIT_GIVEBACK_AVOID_BONUS_CAP", "3.0"))
PEAK_EXIT_LATE_GIVEBACK_PENALTY_MULT = float(os.getenv("RL_PEAK_EXIT_LATE_GIVEBACK_PENALTY_MULT", "180.0"))
PEAK_EXIT_LATE_GIVEBACK_PENALTY_CAP = float(os.getenv("RL_PEAK_EXIT_LATE_GIVEBACK_PENALTY_CAP", "3.0"))

# Human-readable target metrics for comparing training runs. These are written
# into the metrics JSON so you can quickly compare whether a run is improving
# exit accuracy near local highs.
TARGET_EXIT_EFFICIENCY = float(os.getenv("RL_TARGET_EXIT_EFFICIENCY", "0.985"))
TARGET_MAX_MISSED_UPSIDE = float(os.getenv("RL_TARGET_MAX_MISSED_UPSIDE", "0.005"))
TARGET_MAX_GIVEBACK_CAPTURED = float(os.getenv("RL_TARGET_MAX_GIVEBACK_CAPTURED", "0.006"))
TARGET_MIN_AVG_HOLD_BARS = float(os.getenv("RL_TARGET_MIN_AVG_HOLD_BARS", "5"))

HOLD_WHILE_CONTINUATION_REWARD = float(os.getenv("RL_HOLD_WHILE_CONTINUATION_REWARD", "0.35"))
HOLD_THROUGH_GIVEBACK_PENALTY = float(os.getenv("RL_HOLD_THROUGH_GIVEBACK_PENALTY", "1.25"))
FORCE_EXIT_ON_CONFIRMED_GIVEBACK = os.getenv("RL_FORCE_EXIT_ON_CONFIRMED_GIVEBACK", "0").lower() in {"1", "true", "yes"}

# Held-out evaluation split.
RL_TRAIN_SPLIT_PCT = float(os.getenv("RL_TRAIN_SPLIT_PCT", "0.70"))
RL_MIN_TEST_EPISODES = int(os.getenv("RL_MIN_TEST_EPISODES", "10"))
RL_ALLOW_SMALL_TEST_SET = os.getenv("RL_ALLOW_SMALL_TEST_SET", "0").lower() in {"1", "true", "yes"}

# Multibar clean continuation: reward holding through >= N consecutive positive bars.
MULTIBAR_CLEAN_CONTINUATION_REWARD = float(os.getenv("RL_MULTIBAR_CLEAN_CONTINUATION_REWARD", "0.30"))
MULTIBAR_CLEAN_BARS_THRESHOLD = max(1, int(os.getenv("RL_MULTIBAR_CLEAN_BARS_THRESHOLD", "3")))
# Chop/reentry penalty: extra penalty for re-entering after a very short hold.
CHOP_REENTRY_PENALTY = float(os.getenv("RL_CHOP_REENTRY_PENALTY", "0.40"))
CHOP_REENTRY_MAX_HOLD_BARS = int(os.getenv("RL_CHOP_REENTRY_MAX_HOLD_BARS", "5"))
CHOP_ENTRY_EXIT_ENTRY_BARS = int(os.getenv("RL_CHOP_ENTRY_EXIT_ENTRY_BARS", "15"))

# Balance penalties
SHORT_BIAS_PENALTY = float(os.getenv("RL_SHORT_BIAS_PENALTY", "0.002"))
LONG_BIAS_PENALTY = float(os.getenv("RL_LONG_BIAS_PENALTY", "0.0015"))
SHORT_BIAS_RATIO_TRIGGER = float(os.getenv("RL_SHORT_BIAS_RATIO_TRIGGER", "2.0"))
LONG_BIAS_RATIO_TRIGGER = float(os.getenv("RL_LONG_BIAS_RATIO_TRIGGER", "3.0"))

PPO_LEARNING_RATE = float(os.getenv("RL_INTRADAY_LEARNING_RATE", "0.00015"))
PPO_N_STEPS = int(os.getenv("RL_INTRADAY_N_STEPS", "1024"))
PPO_BATCH_SIZE = int(os.getenv("RL_INTRADAY_BATCH_SIZE", "256"))
PPO_GAMMA = float(os.getenv("RL_INTRADAY_GAMMA", "0.997"))
PPO_ENT_COEF = float(os.getenv("RL_INTRADAY_ENT_COEF", "0.080"))

ACTION_MASKING_ENABLED = os.getenv("RL_ACTION_MASKING_ENABLED", "1").lower() in {"1", "true", "yes"}
REENTRY_COOLDOWN_BARS = int(os.getenv("RL_REENTRY_COOLDOWN_BARS", "3"))

REQUIRE_LONG_ABOVE_VWAP = os.getenv("RL_REQUIRE_LONG_ABOVE_VWAP", os.getenv("RL_REQUIRE_BUY_ABOVE_VWAP", "0")).lower() in {"1", "true", "yes"}
REQUIRE_LONG_ABOVE_ORB = os.getenv("RL_REQUIRE_LONG_ABOVE_ORB", os.getenv("RL_REQUIRE_BUY_ABOVE_ORB", "0")).lower() in {"1", "true", "yes"}

REQUIRE_SHORT_BELOW_VWAP = os.getenv("RL_REQUIRE_SHORT_BELOW_VWAP", "0").lower() in {"1", "true", "yes"}
REQUIRE_SHORT_BELOW_TRIGGER = os.getenv("RL_REQUIRE_SHORT_BELOW_TRIGGER", "0").lower() in {"1", "true", "yes"}

DISALLOW_EXIT_WHEN_FLAT = os.getenv("RL_DISALLOW_EXIT_WHEN_FLAT", os.getenv("RL_DISALLOW_SELL_WHEN_FLAT", "1")).lower() in {"1", "true", "yes"}
DISALLOW_BEARISH_LONGS = os.getenv("RL_DISALLOW_BEARISH_LONGS", "0").lower() in {"1", "true", "yes"}
DISALLOW_SHORTS_WHEN_NOT_ALLOWED = os.getenv("RL_DISALLOW_SHORTS_WHEN_NOT_ALLOWED", "0").lower() in {"1", "true", "yes"}

SOFT_VWAP_TREND_THRESHOLD = float(os.getenv("RL_SOFT_VWAP_TREND_THRESHOLD", "0.52"))
SOFT_ORB_BREAKOUT_THRESHOLD = float(os.getenv("RL_SOFT_ORB_BREAKOUT_THRESHOLD", "0.52"))
SOFT_SHORT_TREND_THRESHOLD = float(os.getenv("RL_SOFT_SHORT_TREND_THRESHOLD", "0.52"))
SOFT_BREAKDOWN_THRESHOLD = float(os.getenv("RL_SOFT_BREAKDOWN_THRESHOLD", "0.52"))

DEBUG_MASKS = os.getenv("RL_DEBUG_MASKS", "0").lower() in {"1", "true", "yes"}


ACTION_NAMES = {
    0: "HOLD",
    1: "ENTER_LONG",
    2: "EXIT_LONG",
    3: "ADD_LONG",
}
# =========================================================
# FORCE EXPLORATION
# =========================================================

MIN_TRADES_PER_EPISODE = int(
    os.getenv("RL_MIN_TRADES_PER_EPISODE", "2")
)

NO_TRADE_EPISODE_PENALTY = float(
    os.getenv("RL_NO_TRADE_EPISODE_PENALTY", "15.0")
)

FLAT_HOLD_STEP_PENALTY = float(
    os.getenv("RL_FLAT_HOLD_STEP_PENALTY", "0.003")
)

VALID_ENTRY_REWARD = float(
    os.getenv("RL_VALID_ENTRY_REWARD", "1.25")
)

VALID_SHORT_REWARD = float(
    os.getenv("RL_VALID_SHORT_REWARD", "1.25")
)

FIRST_TRADE_BONUS = float(
    os.getenv("RL_FIRST_TRADE_BONUS", "2.0")
)

ACTION_DIVERSITY_BONUS = float(
    os.getenv("RL_ACTION_DIVERSITY_BONUS", "0.25")
)

ENTRY_FUTURE_3BAR_MULT = float(
    os.getenv("RL_ENTRY_FUTURE_3BAR_MULT", "500.0")
)

ENTRY_FUTURE_3BAR_CAP = float(
    os.getenv("RL_ENTRY_FUTURE_3BAR_CAP", "3.0")
)

# =========================================================
# MFE/MAE TRAJECTORY + CURRICULUM CONTROLS
# =========================================================

USE_FIXED_TAKE_PROFIT = os.getenv("RL_USE_FIXED_TAKE_PROFIT", "0").lower() in {"1", "true", "yes"}
USE_DYNAMIC_GIVEBACK_EXIT = os.getenv("RL_USE_DYNAMIC_GIVEBACK_EXIT", "1").lower() in {"1", "true", "yes"}
DYNAMIC_GIVEBACK_MIN_MFE = float(os.getenv("RL_DYNAMIC_GIVEBACK_MIN_MFE", "0.006"))
DYNAMIC_GIVEBACK_FRACTION = float(os.getenv("RL_DYNAMIC_GIVEBACK_FRACTION", "0.45"))
DYNAMIC_GIVEBACK_FLOOR = float(os.getenv("RL_DYNAMIC_GIVEBACK_FLOOR", "0.0025"))
DYNAMIC_GIVEBACK_MIN_HOLD_BARS = int(os.getenv("RL_DYNAMIC_GIVEBACK_MIN_HOLD_BARS", "5"))

MFE_CAPTURE_REWARD_MULT = float(os.getenv("RL_MFE_CAPTURE_REWARD_MULT", "10.0"))
MFE_CAPTURE_REWARD_CAP = float(os.getenv("RL_MFE_CAPTURE_REWARD_CAP", "8.0"))
MAE_PENALTY_MULT = float(os.getenv("RL_MAE_PENALTY_MULT", "8.0"))
MAE_PENALTY_CAP = float(os.getenv("RL_MAE_PENALTY_CAP", "4.0"))
GIVEBACK_CAPTURE_PENALTY_MULT = float(os.getenv("RL_GIVEBACK_CAPTURE_PENALTY_MULT", "8.0"))
GIVEBACK_CAPTURE_PENALTY_CAP = float(os.getenv("RL_GIVEBACK_CAPTURE_PENALTY_CAP", "4.0"))
LONG_HOLD_TARGET_BARS = float(os.getenv("RL_LONG_HOLD_TARGET_BARS", "6"))
AVG_HOLD_REWARD_MULT = float(os.getenv("RL_AVG_HOLD_REWARD_MULT", "0.25"))
SHORT_CHURN_PENALTY = float(os.getenv("RL_SHORT_CHURN_PENALTY", "0.15"))
LONG_SHORT_BALANCE_PENALTY = float(os.getenv("RL_LONG_SHORT_BALANCE_PENALTY", "0.30"))

FORCE_EXPLORATION_ACTIONS = os.getenv("RL_FORCE_EXPLORATION_ACTIONS", "1").lower() in {"1", "true", "yes"}
FORCE_EXPLORATION_EPISODE_FRAC = float(os.getenv("RL_FORCE_EXPLORATION_EPISODE_FRAC", "0.30"))
FORCE_EXPLORATION_PROB = float(os.getenv("RL_FORCE_EXPLORATION_PROB", "0.12"))

# =========================================================
# LONG-ONLY CONTINUATION / PYRAMIDING CONTROLS
# =========================================================

ADD_LONG_ENABLED = os.getenv("RL_ADD_LONG_ENABLED", "1").lower() in {"1", "true", "yes"}
ADD_LONG_FRACTION = float(os.getenv("RL_ADD_LONG_FRACTION", "0.50"))
ADD_LONG_MIN_UNREALIZED_PCT = float(os.getenv("RL_ADD_LONG_MIN_UNREALIZED_PCT", "0.0015"))
ADD_LONG_MIN_HOLD_BARS = int(os.getenv("RL_ADD_LONG_MIN_HOLD_BARS", "3"))
ADD_LONG_MAX_COUNT = int(os.getenv("RL_ADD_LONG_MAX_COUNT", "2"))
ADD_LONG_REWARD = float(os.getenv("RL_ADD_LONG_REWARD", "8.0"))
ADD_LONG_FUTURE_REWARD_MULT = float(os.getenv("RL_ADD_LONG_FUTURE_REWARD_MULT", os.getenv("RL_ADD_LONG_CONTINUATION_BONUS_MULT", "350.0")))
ADD_LONG_FUTURE_REWARD_CAP = float(os.getenv("RL_ADD_LONG_FUTURE_REWARD_CAP", "4.0"))
MAX_TOTAL_POSITION_FRACTION = float(os.getenv("RL_MAX_TOTAL_POSITION_FRACTION", "0.35"))
BAD_ADD_LONG_PENALTY = float(os.getenv("RL_BAD_ADD_LONG_PENALTY", "2.0"))
ADD_LONG_MIN_CONTINUATION_SCORE = float(os.getenv("RL_ADD_LONG_MIN_CONTINUATION_SCORE", "0.35"))
ADD_LONG_MAX_ADVERSE_FUTURE = float(os.getenv("RL_ADD_LONG_MAX_ADVERSE_FUTURE", "-0.002"))

EARLY_WINNER_EXIT_PENALTY = float(os.getenv("RL_EARLY_WINNER_EXIT_PENALTY", "1.50"))
EARLY_WINNER_EXIT_BARS = int(os.getenv("RL_EARLY_WINNER_EXIT_BARS", "3"))
QUALITY_WINNER_HOLD_BONUS = float(os.getenv("RL_QUALITY_WINNER_HOLD_BONUS", "3.00"))
QUALITY_WINNER_MIN_HOLD_BARS = int(os.getenv("RL_QUALITY_WINNER_MIN_HOLD_BARS", "5"))
UNREALIZED_WINNER_HOLD_MULT = float(os.getenv("RL_UNREALIZED_WINNER_HOLD_MULT", "25.0"))
TREND_HOLD_TIME_MULT = float(os.getenv("RL_TREND_HOLD_TIME_MULT", "0.15"))
CONTINUATION_HOLD_MULT = float(os.getenv("RL_CONTINUATION_HOLD_MULT", "120.0"))
CONTINUATION_HOLD_CAP = float(os.getenv("RL_CONTINUATION_HOLD_CAP", "3.0"))


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class TrainingConfig:
    bars_path: str
    structure_map_path: str
    model_path: str
    train_timesteps: int
    train_symbol_limit: int
    max_episode_steps: int
    min_bars_per_episode: int
    starting_cash: float
    position_fraction: float
    commission_bps: float
    slippage_bps: float
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_bars: int
    policy_mode: str
    use_lstm_policy: bool
    use_transformer_features: bool
    use_hierarchical_gating: bool
    use_multi_timeframe_fusion: bool
    action_masking_enabled: bool
    require_long_above_vwap: bool
    require_long_above_orb: bool
    require_short_below_vwap: bool
    require_short_below_trigger: bool
    disallow_exit_when_flat: bool
    reentry_cooldown_bars: int
    disallow_bearish_longs: bool
    disallow_shorts_when_not_allowed: bool
    soft_vwap_trend_threshold: float
    soft_orb_breakout_threshold: float
    soft_short_trend_threshold: float
    soft_breakdown_threshold: float
    ppo_ent_coef: float


@dataclass
class TradeEvent:
    timestamp: str
    symbol: str
    action: str
    price: float
    cash: float
    equity: float
    position_qty: float
    realized_pnl: float
    reward: float
    reason: str
    hold_bars: int
    unrealized_pct: float
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    giveback_from_mfe_pct: float = 0.0
    capture_ratio: float = 0.0


# =============================================================================
# Generic helpers
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    b = to_float(b, 0.0)
    if abs(b) < 1e-12:
        return default
    return to_float(a, 0.0) / b


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


# =============================================================================
# Loaders
# =============================================================================

def find_bars_path() -> Path:
    for path in DEFAULT_BARS_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No intraday bars file found. Expected one of:\n"
        + "\n".join(f"  - {path}" for path in DEFAULT_BARS_PATHS)
    )


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[data-source] training_csv=alpaca_derived path={path.name}")

    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].map(normalize_symbol)

    df = df.dropna(subset=["timestamp"])
    df = df[df["symbol"] != ""].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path} loaded but no valid rows remained after cleaning.")

    return df


def load_structure_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run premarket_trade_planner first.")

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("intraday_structure_map.json must be a JSON object.")

    for key in ["symbols", "structure_maps", "intraday_structure_maps", "intraday_structure_map", "maps"]:
        value = payload.get(key)
        if isinstance(value, dict):
            out = {
                normalize_symbol(symbol): data
                for symbol, data in value.items()
                if normalize_symbol(symbol) and isinstance(data, dict)
            }
            if out:
                print(f"[structure-map] loaded {len(out)} symbols from dict key '{key}'")
                return out

    for key in ["structures", "daily_trade_plans", "plans", "raw_premarket_pattern_plans", "intraday_plans", "structure_plans"]:
        value = payload.get(key)
        if isinstance(value, list):
            out: dict[str, Any] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                symbol = normalize_symbol(item.get("symbol") or item.get("ticker") or item.get("asset"))
                if symbol:
                    out[symbol] = item

            if out:
                print(f"[structure-map] loaded {len(out)} symbols from list key '{key}'")
                return out

    metadata_keys = {
        "CREATED_AT_UTC",
        "PREMARKET_WINDOW_ET",
        "DATA_FEED",
        "DIAGNOSTICS",
        "METADATA",
        "CONFIG",
        "SYMBOL_COUNT",
        "LONG_ALLOWED_SYMBOLS",
        "STRUCTURES",
    }

    direct = {
        normalize_symbol(symbol): data
        for symbol, data in payload.items()
        if normalize_symbol(symbol)
        and normalize_symbol(symbol) not in metadata_keys
        and isinstance(data, dict)
    }

    if direct:
        print(f"[structure-map] loaded {len(direct)} symbols from direct shape")
        return direct

    print("[structure-map] unsupported top-level keys:")
    print(sorted(payload.keys()))
    raise ValueError("Unsupported intraday_structure_map.json shape. Could not find symbol maps.")


def write_symbol_overlap_report(
    bars: pd.DataFrame,
    structure: dict[str, Any],
    path: Path = OVERLAP_PATH,
) -> tuple[list[str], list[str], list[str]]:
    bar_symbols = sorted(set(bars["symbol"].map(normalize_symbol).dropna().tolist()))
    structure_symbols = sorted(set(normalize_symbol(s) for s in structure.keys() if normalize_symbol(s)))
    overlap = sorted(set(bar_symbols) & set(structure_symbols))

    rows = []
    for symbol in sorted(set(bar_symbols) | set(structure_symbols)):
        rows.append(
            {
                "symbol": symbol,
                "in_bars": symbol in bar_symbols,
                "in_structure_map": symbol in structure_symbols,
                "bar_rows": int((bars["symbol"] == symbol).sum()) if symbol in bar_symbols else 0,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)

    print(f"[overlap] bars_symbols={len(bar_symbols)} structure_symbols={len(structure_symbols)} overlap={len(overlap)}")
    print(f"[saved] overlap_report={path}")

    if overlap:
        print("[overlap] matched_symbols=" + ",".join(overlap[:50]))
    else:
        print("[overlap] no matched symbols")
        print("[overlap] bars sample=" + ",".join(bar_symbols[:30]))
        print("[overlap] structure sample=" + ",".join(structure_symbols[:30]))

    return bar_symbols, structure_symbols, overlap


# =============================================================================
# Structure flattening
# =============================================================================

def get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def first_number(items: Any, fallback: float = 0.0) -> float:
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                value = item.get("price") or item.get("level") or item.get("value")
            else:
                value = item
            parsed = to_float(value, np.nan)
            if not math.isnan(parsed):
                return parsed
    return fallback


def flatten_structure(symbol_map: dict[str, Any]) -> dict[str, float]:
    levels = symbol_map.get("combined_support_and_resistance_levels", {}) or {}
    sr = symbol_map.get("support_resistance", {}) or {}
    orb = symbol_map.get("orb", symbol_map.get("opening_range", {})) or {}
    vwap = symbol_map.get("vwap", {}) or {}
    liquidity = symbol_map.get("liquidity", symbol_map.get("intraday_liquidity", {})) or {}
    signals = symbol_map.get("signals", {}) or {}
    trade_structure = symbol_map.get("trade_structure", {}) or {}
    nearest = symbol_map.get("nearest_levels", {}) or {}
    market_context = symbol_map.get("market_context", {}) or {}

    supports = sr.get("supports", []) or symbol_map.get("support_levels", [])
    resistances = sr.get("resistances", []) or symbol_map.get("resistance_levels", [])

    nearest_support = nearest.get("nearest_support") if isinstance(nearest.get("nearest_support"), dict) else {}
    nearest_resistance = nearest.get("nearest_resistance") if isinstance(nearest.get("nearest_resistance"), dict) else {}

    support_1 = to_float(
        levels.get("support_1")
        or trade_structure.get("preferred_long_stop")
        or trade_structure.get("preferred_short_trigger")
        or nearest_support.get("price"),
        first_number(supports, 0.0),
    )
    resistance_1 = to_float(
        levels.get("resistance_1")
        or trade_structure.get("preferred_long_trigger")
        or trade_structure.get("preferred_short_stop")
        or nearest_resistance.get("price"),
        first_number(resistances, 0.0),
    )

    support_2 = to_float(levels.get("support_2"), support_1)
    support_3 = to_float(levels.get("support_3"), support_2)

    resistance_2 = to_float(levels.get("resistance_2"), resistance_1)
    resistance_3 = to_float(levels.get("resistance_3"), resistance_2)

    orb_high = to_float(
        orb.get("high")
        or orb.get("orb_high")
        or orb.get("opening_range_high")
        or trade_structure.get("preferred_long_trigger"),
        resistance_1,
    )
    orb_low = to_float(
        orb.get("low")
        or orb.get("orb_low")
        or orb.get("opening_range_low")
        or trade_structure.get("preferred_short_trigger")
        or trade_structure.get("preferred_long_stop"),
        support_1,
    )

    vwap_value = to_float(vwap.get("value") or vwap.get("vwap") or vwap.get("session_vwap"), 0.0)
    vwap_upper = to_float(
        vwap.get("upper_band") or vwap.get("upper_1") or vwap.get("vwap_upper") or vwap.get("upper"),
        vwap_value,
    )
    vwap_lower = to_float(
        vwap.get("lower_band") or vwap.get("lower_1") or vwap.get("vwap_lower") or vwap.get("lower"),
        vwap_value,
    )

    liquidity_score = to_float(
        liquidity.get("score") or liquidity.get("liquidity_score") or liquidity.get("intraday_liquidity_score"),
        0.5,
    )
    spread_proxy = to_float(
        liquidity.get("spread_proxy") or liquidity.get("estimated_spread_pct") or liquidity.get("spread_pct"),
        0.003,
    )
    volume_ratio = to_float(
        liquidity.get("volume_ratio") or liquidity.get("relative_volume") or liquidity.get("relative_volume_intraday"),
        1.0,
    )

    long_allowed = bool(trade_structure.get("long_allowed", True))
    short_allowed = bool(trade_structure.get("short_or_put_allowed", True))

    bearish_regime = bool(
        symbol_map.get("bearish_regime", False)
        or signals.get("bearish_regime", False)
        or market_context.get("bearish_regime", False)
        or str(market_context.get("regime", "")).lower() in {"bear", "bearish", "risk_off", "defensive_risk_off"}
    )

    risk_on = bool(
        market_context.get("risk_on", False)
        or str(market_context.get("regime", "")).lower() in {"bull", "bullish", "risk_on", "trend_risk_on"}
    )

    breakout_score = to_float(
        symbol_map.get("breakout_score")
        or signals.get("breakout_score")
        or get_nested(symbol_map, ["execution_bias", "breakout_score"]),
        0.0,
    )
    trend_score = to_float(
        symbol_map.get("trend_score")
        or signals.get("trend_score")
        or get_nested(symbol_map, ["execution_bias", "trend_score"]),
        0.0,
    )

    breakdown_score = to_float(
        symbol_map.get("breakdown_score")
        or signals.get("breakdown_score")
        or get_nested(symbol_map, ["execution_bias", "breakdown_score"]),
        max(0.0, -trend_score),
    )

    sector_strength = to_float(
        symbol_map.get("sector_strength")
        or market_context.get("sector_strength")
        or get_nested(symbol_map, ["context", "sector_strength"]),
        0.0,
    )
    market_bias = to_float(
        symbol_map.get("market_bias")
        or market_context.get("market_bias")
        or get_nested(symbol_map, ["context", "market_bias"]),
        0.0,
    )

    return {
        "support_1": support_1,
        "support_2": support_2,
        "support_3": support_3,
        "resistance_1": resistance_1,
        "resistance_2": resistance_2,
        "resistance_3": resistance_3,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "vwap": vwap_value,
        "vwap_upper": vwap_upper,
        "vwap_lower": vwap_lower,
        "liquidity_score": liquidity_score,
        "spread_proxy": spread_proxy,
        "volume_ratio": volume_ratio,
        "breakout_score": breakout_score,
        "breakdown_score": breakdown_score,
        "trend_score": trend_score,
        "sector_strength": sector_strength,
        "market_bias": market_bias,
        "long_allowed": 1.0 if long_allowed else 0.0,
        "short_allowed": 1.0 if short_allowed else 0.0,
        "bearish_regime": 1.0 if bearish_regime else 0.0,
        "risk_on": 1.0 if risk_on else 0.0,
    }


# =============================================================================
# Bar feature engineering
# =============================================================================
def add_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol"]).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    grouped = df.groupby("symbol", group_keys=False)

    # Returns
    df["ret_1"] = grouped["close"].pct_change().fillna(0.0)
    df["ret_3"] = grouped["close"].pct_change(3).fillna(0.0)
    df["ret_5"] = grouped["close"].pct_change(5).fillna(0.0)
    df["ret_10"] = grouped["close"].pct_change(10).fillna(0.0)
    df["ret_20"] = grouped["close"].pct_change(20).fillna(0.0)

    # EMA features
    df["ema_8"] = grouped["close"].transform(lambda s: s.ewm(span=8, adjust=False).mean())
    df["ema_21"] = grouped["close"].transform(lambda s: s.ewm(span=21, adjust=False).mean())
    df["ema_55"] = grouped["close"].transform(lambda s: s.ewm(span=55, adjust=False).mean())

    df["ema_spread"] = (
        (df["ema_8"] - df["ema_21"]) / df["close"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["ema_trend_strength"] = (
        (df["ema_8"] > df["ema_21"]).astype(float)
        + (df["ema_21"] > df["ema_55"]).astype(float)
    )

    df["ema_spread_slope"] = grouped["ema_spread"].diff().fillna(0.0)
    df["ema_acceleration"] = grouped["ema_spread_slope"].diff().fillna(0.0)

    # Volume features
    df["avg_volume_20"] = grouped["volume"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )
    df["avg_volume_50"] = grouped["volume"].transform(
        lambda s: s.rolling(50, min_periods=1).mean()
    )

    df["volume_ratio_bar"] = (
        df["volume"] / df["avg_volume_20"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["volume_momentum"] = grouped["volume_ratio_bar"].diff().fillna(0.0)
    df["high_relative_volume"] = (df["volume_ratio_bar"] > 1.5).astype(float)

    df["volume_expansion"] = (
        df["volume_ratio_bar"] > grouped["volume_ratio_bar"].shift(3).fillna(1.0)
    ).astype(float)

    df["volume_contraction"] = (
        df["volume_ratio_bar"] < grouped["volume_ratio_bar"].shift(3).fillna(1.0)
    ).astype(float)

    # VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["_pv"] = typical * df["volume"]
    df["_cum_pv"] = grouped["_pv"].cumsum()
    df["_cum_vol"] = grouped["volume"].cumsum()

    df["running_vwap"] = (
        df["_cum_pv"] / df["_cum_vol"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    df["running_vwap"] = df["running_vwap"].fillna(df["close"])

    df["vwap_slope_5"] = (
        grouped["running_vwap"].diff(5) / df["close"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Session context
    ts_et = df["timestamp"].dt.tz_convert("America/New_York")
    df["session_date"] = ts_et.dt.date
    grouped_session = df.groupby(["symbol", "session_date"], group_keys=False)

    df["session_open"] = grouped_session["open"].transform("first")
    df["day_high_so_far"] = grouped_session["high"].cummax()
    df["day_low_so_far"] = grouped_session["low"].cummin()

    df["intraday_from_open_pct"] = (
        df["close"] / df["session_open"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prev_session_close = (
        grouped_session["close"]
        .transform("last")
        .groupby(df["symbol"])
        .shift(1)
    )

    df["daily_change_pct"] = (
        df["close"] / prev_session_close.replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    df["daily_change_pct"] = df["daily_change_pct"].fillna(df["intraday_from_open_pct"])

    df["distance_from_day_high"] = (
        df["close"] / df["day_high_so_far"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["distance_from_day_low"] = (
        df["close"] / df["day_low_so_far"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Bar structure
    df["range_pct"] = (
        (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["close_location"] = (
        (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    ).clip(0, 1).fillna(0.5)

    df["rolling_high_20"] = (
        grouped["high"].rolling(20, min_periods=1).max().reset_index(level=0, drop=True)
    )
    df["rolling_low_20"] = (
        grouped["low"].rolling(20, min_periods=1).min().reset_index(level=0, drop=True)
    )

    df["dist_high_20"] = (
        df["close"] / df["rolling_high_20"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["dist_low_20"] = (
        df["close"] / df["rolling_low_20"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Future labels for reward shaping only
    for horizon in [1, 3, 5, 10, 15, 20, 30]:
        df[f"future_return_{horizon}"] = (
            grouped["close"].shift(-horizon) / df["close"].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def future_window_max(s: pd.Series, window: int) -> pd.Series:
        return s.shift(-1).iloc[::-1].rolling(window, min_periods=1).max().iloc[::-1]

    def future_window_min(s: pd.Series, window: int) -> pd.Series:
        return s.shift(-1).iloc[::-1].rolling(window, min_periods=1).min().iloc[::-1]

    for horizon in [5, 15, 30]:
        future_high = grouped["high"].transform(lambda s, h=horizon: future_window_max(s, h))
        future_low = grouped["low"].transform(lambda s, h=horizon: future_window_min(s, h))

        df[f"future_max_return_{horizon}"] = (
            future_high / df["close"].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df[f"future_min_return_{horizon}"] = (
            future_low / df["close"].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["future_min_return_5"] = df["future_min_return_5"].fillna(0.0)

    # Real-time exit/exhaustion features
    df["ret_slope_3_10"] = (df["ret_3"] - df["ret_10"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["momentum_decay_score"] = (
        (df["ret_1"] < 0).astype(float)
        + (df["ret_3"] < df["ret_5"]).astype(float)
        + (
            df["ema_spread"]
            < grouped["ema_spread"].shift(3).fillna(df["ema_spread"])
        ).astype(float)
    ) / 3.0

    df["volume_fade_score"] = (
        (
            df["volume_ratio_bar"]
            < grouped["volume_ratio_bar"].shift(3).fillna(df["volume_ratio_bar"])
        ).astype(float)
        + (df["volume_ratio_bar"] < 1.0).astype(float)
    ) / 2.0

    df["vwap_loss_score"] = (df["close"] < df["running_vwap"]).astype(float).fillna(0.0)
    df["near_day_high_score"] = (df["distance_from_day_high"] > -0.003).astype(float).fillna(0.0)
    df["giveback_from_day_high_pct"] = df["distance_from_day_high"].clip(upper=0.0).fillna(0.0)

    # Bullish continuation features
    df["breakout_persistence_score"] = (
        (
            df["close"] >= grouped["rolling_high_20"].shift(1).fillna(df["rolling_high_20"])
        )
        | (df["dist_high_20"] > -0.003)
    ).astype(float).fillna(0.0)

    df["bullish_acceleration_score"] = (
        (df["ret_1"] > df["ret_3"] / 3.0).astype(float)
        + (df["ret_3"] > df["ret_5"] / 2.0).astype(float)
        + (df["ret_5"] > 0).astype(float)
    ) / 3.0

    df["positive_volume_expansion_score"] = (
        (df["volume_ratio_bar"] > 1.25) & (df["ret_1"] > 0)
    ).astype(float).fillna(0.0)

    prev_high_5 = grouped["high"].rolling(5, min_periods=1).max().shift(1).reset_index(level=0, drop=True)
    prev_low_5 = grouped["low"].rolling(5, min_periods=1).min().shift(1).reset_index(level=0, drop=True)

    df["higher_high_score"] = (df["high"] > prev_high_5).astype(float).fillna(0.0)
    df["higher_low_score"] = (df["low"] > prev_low_5).astype(float).fillna(0.0)

    df["bullish_structure_score"] = (
        df["breakout_persistence_score"]
        + df["bullish_acceleration_score"]
        + df["positive_volume_expansion_score"]
        + df["higher_high_score"]
        + df["higher_low_score"]
    ) / 5.0

    # Bearish/inverse features
    df["breakdown_persistence_score"] = (
        (
            df["close"] <= grouped["rolling_low_20"].shift(1).fillna(df["rolling_low_20"])
        )
        | (df["dist_low_20"] < 0.003)
    ).astype(float).fillna(0.0)

    df["bearish_acceleration_score"] = (
        (df["ret_1"] < df["ret_3"] / 3.0).astype(float)
        + (df["ret_3"] < df["ret_5"] / 2.0).astype(float)
        + (df["ret_5"] < 0).astype(float)
    ) / 3.0

    prev_low_5_for_lower_low = grouped["low"].rolling(5, min_periods=1).min().shift(1).reset_index(level=0, drop=True)
    prev_high_5_for_lower_high = grouped["high"].rolling(5, min_periods=1).max().shift(1).reset_index(level=0, drop=True)

    df["lower_low_score"] = (df["low"] < prev_low_5_for_lower_low).astype(float).fillna(0.0)
    df["lower_high_score"] = (df["high"] < prev_high_5_for_lower_high).astype(float).fillna(0.0)

    # Transformer features
    if USE_TRANSFORMER_FEATURES:
        for window in SEQUENCE_WINDOWS:
            df[f"seq_ret_mean_{window}"] = (
                grouped["ret_1"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0)
            )
            df[f"seq_ret_std_{window}"] = (
                grouped["ret_1"].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            )
            df[f"seq_vol_mean_{window}"] = (
                grouped["volume_ratio_bar"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).fillna(1.0)
            )
            positive_ret = (df["ret_1"] > 0).astype(float)

            df[f"seq_trend_consistency_{window}"] = (
                positive_ret.groupby(df["symbol"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .fillna(0.5)
            )

    # Multi-timeframe fusion
    if USE_MULTI_TIMEFRAME_FUSION:
        for window in MTF_WINDOWS:
            df[f"mtf_ret_{window}"] = grouped["close"].pct_change(window).fillna(0.0)

            df[f"mtf_range_{window}"] = (
                (
                    grouped["high"].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
                    - grouped["low"].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
                )
                / df["close"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            df[f"mtf_volume_ratio_{window}"] = (
                df["volume"]
                / grouped["volume"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

            df[f"mtf_vwap_dist_{window}"] = (
                df["close"]
                / grouped["running_vwap"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).replace(0, np.nan)
                - 1.0
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df = df.copy()

    return df.drop(
        columns=["_pv", "_cum_pv", "_cum_vol", "session_date"],
        errors="ignore",
    )
# =============================================================================
# Gym environment
# =============================================================================

class IntradayStructureEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bars_by_episode: list[pd.DataFrame],
        structure_by_symbol: dict[str, dict[str, float]],
    ):
        super().__init__()

        if not bars_by_episode:
            raise ValueError("No bar episodes supplied.")

        self.bars_by_episode = bars_by_episode
        self.structure_by_symbol = structure_by_symbol

        self.action_space = spaces.Discrete(4)
        self.feature_names = self._build_feature_names()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.feature_names),),
            dtype=np.float32,
        )

        self.episode_df = pd.DataFrame()
        self.idx = 0
        self.symbol = ""
        self.cash = STARTING_CASH
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.position_side = 0
        self.equity_peak = STARTING_CASH
        self.realized_pnl = 0.0
        self.hold_bars = 0
        self.last_exit_idx: int | None = None
        self.masked_action_counts = {
            "long_masked": 0,
            "short_masked": 0,
            "exit_long_masked": 0,
            "exit_short_masked": 0,
        }
        self.trade_events: list[TradeEvent] = []
        self.long_entry_count = 0
        self.short_entry_count = 0
        self.add_long_count = 0
        self.entry_idx: int | None = None
        self.prev_entry_idx: int | None = None
        self.entry_was_breakout = False
        self.entry_cross_count = 0
        self.vwap_cross_count = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        self.previous_step_equity = STARTING_CASH

    def _build_feature_names(self) -> list[str]:
        names = [
            "price_vwap",
            "price_vwap_upper",
            "price_vwap_lower",
            "price_orb_high",
            "price_orb_low",
            "price_support_1",
            "price_resistance_1",
            "ret_1",
            "ret_3",
            "ret_5",
            "ret_10",
            "ret_20",
            "ema_spread",
            "vwap_slope_5",
            "volume_ratio_bar",
            "range_pct",
            "close_location",
            "dist_high_20",
            "dist_low_20",
            "breakout_persistence_score",
            "bullish_acceleration_score",
            "positive_volume_expansion_score",
            "higher_high_score",
            "higher_low_score",
            "bullish_structure_score",
            "breakdown_persistence_score",
            "bearish_acceleration_score",
            "lower_low_score",
            "lower_high_score",
        ]

        if USE_TRANSFORMER_FEATURES:
            for window in SEQUENCE_WINDOWS:
                names.extend([
                    f"seq_ret_mean_{window}",
                    f"seq_ret_std_{window}",
                    f"seq_vol_mean_{window}",
                    f"seq_trend_consistency_{window}",
                ])

        if USE_MULTI_TIMEFRAME_FUSION:
            for window in MTF_WINDOWS:
                names.extend([
                    f"mtf_ret_{window}",
                    f"mtf_range_{window}",
                    f"mtf_volume_ratio_{window}",
                    f"mtf_vwap_dist_{window}",
                ])

        names.extend([
            "structure_volume_ratio",
            "liquidity_score",
            "spread_proxy",
            "trend_score",
            "breakout_score",
            "breakdown_score",
            "sector_strength",
            "market_bias",
            "position_side",
            "is_long",
            "is_short",
            "unrealized_pct",
            "long_allowed",
            "bearish_regime",
            "reentry_cooldown",
            "hierarchical_entry_gate",
            "hierarchical_long_gate",
            "hierarchical_short_gate",
            "daily_change_pct",
            "intraday_from_open_pct",
            "distance_from_day_high",
            "distance_from_day_low",
            "current_position_pnl_pct",
            "bars_since_entry",
            "reentry_count_today",
            "minutes_since_last_exit",
            "daily_equity_drawdown_pct",
            "entry_cross_count",
            "vwap_cross_count",
            "max_favorable_excursion",
            "max_adverse_excursion",
            "current_giveback_from_mfe",
            "ret_slope_3_10",
            "momentum_decay_score",
            "volume_fade_score",
            "vwap_loss_score",
            "near_day_high_score",
            "giveback_from_day_high_pct",
            "future_max_return_5",
            "future_max_return_15",
            "future_max_return_30",
            "future_min_return_15",
            "future_min_return_30",
        ])

        return names

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self.episode_df = random.choice(self.bars_by_episode).reset_index(drop=True)
        self.symbol = str(self.episode_df["symbol"].iloc[0])
        self.idx = 0

        self.cash = STARTING_CASH
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.position_side = 0
        self.equity_peak = STARTING_CASH
        self.realized_pnl = 0.0
        self.hold_bars = 0
        self.last_exit_idx = None
        self.masked_action_counts = {
            "long_masked": 0,
            "short_masked": 0,
            "exit_long_masked": 0,
            "exit_short_masked": 0,
        }
        self.trade_events = []
        self.long_entry_count = 0
        self.short_entry_count = 0
        self.add_long_count = 0
        self.entry_idx = None
        self.prev_entry_idx = None
        self.entry_was_breakout = False
        self.entry_cross_count = 0
        self.vwap_cross_count = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        self.previous_step_equity = STARTING_CASH

        return self._obs(), {}

    def step(self, action: int):
        row = self.episode_df.iloc[self.idx]
        price = max(to_float(row["close"]), 0.01)
        previous_equity = self._equity(price)
        self._update_open_position_trackers(price, row)

        forced_reason: str | None = None

        # Curriculum exploration: in the early part of each episode, occasionally
        # force an entry attempt so PPO sees real trade rewards instead of
        # converging to a pure HOLD policy. Action masks still protect illegal
        # state transitions.
        episode_limit = max(1, min(len(self.episode_df) - 1, MAX_EPISODE_STEPS))
        if (
            FORCE_EXPLORATION_ACTIONS
            and self.position_side == 0
            and self.idx < int(episode_limit * FORCE_EXPLORATION_EPISODE_FRAC)
            and random.random() < FORCE_EXPLORATION_PROB
        ):
            action = 1
            forced_reason = "curriculum_forced_exploration_entry"

        # Pyramiding curriculum: once the agent is already long, force occasional
        # ADD_LONG attempts when the add mask says conditions are valid. Without
        # this, PPO almost never samples the rare add action often enough to learn
        # convex trend exploitation.
        if (
            FORCE_EXPLORATION_ACTIONS
            and self.position_side == 1
            and self.idx < int(episode_limit * FORCE_EXPLORATION_EPISODE_FRAC)
            and random.random() < min(FORCE_EXPLORATION_PROB * 1.75, 0.35)
        ):
            add_allowed, _add_reason = self._add_long_allowed(price, row)
            if add_allowed:
                action = 3
                forced_reason = "curriculum_forced_pyramid_add"

        if self.position_side != 0 and self.entry_price > 0:
            self.hold_bars += 1

            if self.position_side == 1:
                unrealized_pct = price / self.entry_price - 1.0
                if unrealized_pct <= -STOP_LOSS_PCT:
                    action = 2
                    forced_reason = "long_stop_loss"
                elif USE_FIXED_TAKE_PROFIT and unrealized_pct >= TAKE_PROFIT_PCT:
                    action = 2
                    forced_reason = "long_take_profit"
            # Long-only: no short-side stop/take-profit branch.

            if self.hold_bars >= MAX_HOLD_BARS:
                action = 2
                forced_reason = "max_hold_bars"

            if USE_DYNAMIC_GIVEBACK_EXIT:
                current_giveback = self._current_giveback_from_mfe(price)
                adaptive_giveback = max(
                    DYNAMIC_GIVEBACK_FLOOR,
                    self.max_favorable_excursion * DYNAMIC_GIVEBACK_FRACTION,
                )
                if (
                    self.hold_bars >= DYNAMIC_GIVEBACK_MIN_HOLD_BARS
                    and self.max_favorable_excursion >= DYNAMIC_GIVEBACK_MIN_MFE
                    and current_giveback >= adaptive_giveback
                ):
                    action = 2
                    forced_reason = "dynamic_mfe_trailing_giveback_exit"

            if FORCE_EXIT_ON_CONFIRMED_GIVEBACK:
                current_giveback = max(0.0, self.max_favorable_excursion - self._unrealized_pct(price))
                momentum_decay = to_float(row.get("momentum_decay_score"), 0.0)
                vwap_loss = to_float(row.get("vwap_loss_score"), 0.0)
                if (
                    self.max_favorable_excursion >= PEAK_EXIT_MIN_MFE
                    and current_giveback >= PEAK_EXIT_GIVEBACK_TRIGGER
                    and (momentum_decay >= 0.67 or vwap_loss >= 1.0)
                ):
                    action = 2
                    forced_reason = "confirmed_peak_giveback_exit"

        action, mask_reason = self._apply_action_mask(int(action), price, row)

        if action == 1:
            reward = self._enter_long(price, row, reason=forced_reason or mask_reason or "agent_enter_long")
        elif action == 2:
            reward = self._exit_long(price, row, reason=forced_reason or mask_reason or "agent_exit_long")
        elif action == 3:
            reward = self._add_long(price, row, reason=forced_reason or mask_reason or "agent_add_long")
        else:
            reward = self._hold_reward(price, row, mask_reason=mask_reason)

        new_equity = self._equity(price)
        reward += ((new_equity - previous_equity) / max(previous_equity, 1.0)) * REWARD_SCALE
        reward += self._dense_execution_reward(int(action), price, row, previous_equity, new_equity)
        self.previous_step_equity = new_equity

        self.equity_peak = max(self.equity_peak, new_equity)
        drawdown = new_equity / max(self.equity_peak, 1.0) - 1.0
        if drawdown < 0:
            reward -= abs(drawdown) * DRAWDOWN_PENALTY

        self.idx += 1
        done = self.idx >= min(len(self.episode_df) - 1, MAX_EPISODE_STEPS)

        if done and self.position_side == 1:
            final_row = self.episode_df.iloc[min(self.idx, len(self.episode_df) - 1)]
            final_price = max(to_float(final_row["close"]), 0.01)
            reward += self._exit_long(final_price, final_row, reason="episode_end")

        if done:
            trade_count = self.long_entry_count
            if trade_count < MIN_TRADES_PER_EPISODE:
                reward -= NO_TRADE_EPISODE_PENALTY * float(MIN_TRADES_PER_EPISODE - trade_count)

        return self._obs(), float(reward), done, False, {
            "symbol": self.symbol,
            "equity": float(new_equity),
            "realized_pnl": float(self.realized_pnl),
            "position_qty": float(self.position_qty),
            "position_side": int(self.position_side),
            "masked_action_counts": dict(self.masked_action_counts),
        }

    def _structure(self) -> dict[str, float]:
        return self.structure_by_symbol.get(self.symbol, {})

    def _market_refs(self, price: float, row: pd.Series) -> dict[str, float]:
        structure = self._structure()
        support_1 = to_float(structure.get("support_1"), price)
        resistance_1 = to_float(structure.get("resistance_1"), price)
        orb_high = to_float(structure.get("orb_high"), resistance_1)
        orb_low = to_float(structure.get("orb_low"), support_1)
        vwap = to_float(structure.get("vwap"), to_float(row.get("running_vwap"), price))
        vwap_upper = to_float(structure.get("vwap_upper"), vwap)
        vwap_lower = to_float(structure.get("vwap_lower"), vwap)

        if orb_high <= 0:
            orb_high = resistance_1 if resistance_1 > 0 else price
        if orb_low <= 0:
            orb_low = support_1 if support_1 > 0 else price
        if vwap <= 0:
            vwap = to_float(row.get("running_vwap"), price)

        return {
            "support_1": support_1,
            "resistance_1": resistance_1,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "vwap": vwap,
            "vwap_upper": vwap_upper,
            "vwap_lower": vwap_lower,
        }

    def _unrealized_pct(self, price: float) -> float:
        if self.position_side == 1 and self.entry_price > 0:
            return price / self.entry_price - 1.0
        return 0.0

    def _reentry_cooldown_active(self) -> bool:
        if self.last_exit_idx is None:
            return False
        return (self.idx - self.last_exit_idx) < REENTRY_COOLDOWN_BARS

    def _hierarchical_gates(self, price: float, row: pd.Series) -> tuple[float, float, float]:
        if not USE_HIERARCHICAL_GATING:
            return 1.0, 1.0, 1.0

        structure = self._structure()
        refs = self._market_refs(price, row)

        bullish_score = (
            0.25 * to_float(row.get("bullish_structure_score"), 0.0)
            + 0.20 * float(price >= refs["vwap"])
            + 0.20 * float(price >= refs["orb_high"])
            + 0.15 * max(to_float(structure.get("trend_score"), 0.0), 0.0)
            + 0.10 * to_float(row.get("positive_volume_expansion_score"), 0.0)
            + 0.10 * max(to_float(structure.get("market_bias"), 0.0), 0.0)
        )

        bearish_score = (
            0.25 * to_float(row.get("bearish_acceleration_score"), 0.0)
            + 0.20 * float(price <= refs["vwap"])
            + 0.20 * float(price <= refs["orb_low"])
            + 0.15 * to_float(structure.get("breakdown_score"), 0.0)
            + 0.10 * float(to_float(structure.get("bearish_regime"), 0.0) >= 1.0)
            + 0.10 * max(-to_float(structure.get("market_bias"), 0.0), 0.0)
        )

        entry_gate = max(bullish_score, bearish_score)
        return float(entry_gate), float(bullish_score), float(bearish_score)

    def _long_allowed(self, price: float, row: pd.Series) -> tuple[bool, str | None]:
        if self.position_side != 0:
            return False, "long_masked_not_flat"

        structure = self._structure()
        refs = self._market_refs(price, row)
        _, long_gate, _ = self._hierarchical_gates(price, row)

        if USE_HIERARCHICAL_GATING and not EARLY_TRAINING_GATE_RELAXATION and long_gate < 0.35:
            return False, "long_masked_hierarchical_gate"
        if USE_HIERARCHICAL_GATING and EARLY_TRAINING_GATE_RELAXATION and long_gate < 0.20:
            return False, "long_masked_hierarchical_gate_relaxed"

        if DISALLOW_BEARISH_LONGS:
            if to_float(structure.get("bearish_regime"), 0.0) >= 1.0:
                return False, "long_masked_bearish_regime"
            if to_float(structure.get("long_allowed"), 1.0) < 1.0:
                return False, "long_masked_long_not_allowed"

        if self._reentry_cooldown_active():
            return False, "long_masked_reentry_cooldown"

        trend_score = to_float(structure.get("trend_score"), 0.0)
        breakout_score = to_float(structure.get("breakout_score"), 0.0)

        if REQUIRE_LONG_ABOVE_VWAP and price < refs["vwap"] and trend_score < SOFT_VWAP_TREND_THRESHOLD:
            return False, "long_masked_below_vwap"

        if REQUIRE_LONG_ABOVE_ORB and price < refs["orb_high"] and breakout_score < SOFT_ORB_BREAKOUT_THRESHOLD:
            return False, "long_masked_below_orb"

        return True, None

    def _short_allowed(self, price: float, row: pd.Series) -> tuple[bool, str | None]:
        return False, "shorting_disabled_long_only"

    def _exit_long_allowed(self) -> tuple[bool, str | None]:
        if DISALLOW_EXIT_WHEN_FLAT and self.position_side != 1:
            return False, "exit_long_masked_not_long"
        return True, None

    def _add_long_allowed(self, price: float, row: pd.Series) -> tuple[bool, str | None]:
        if not ADD_LONG_ENABLED:
            return False, "add_long_disabled"
        if self.position_side != 1 or self.position_qty <= 0 or self.entry_price <= 0:
            return False, "add_long_masked_not_long"
        if self.add_long_count >= ADD_LONG_MAX_COUNT:
            return False, "add_long_masked_max_adds"
        if self.hold_bars < ADD_LONG_MIN_HOLD_BARS:
            return False, "add_long_masked_too_early"
        if self._unrealized_pct(price) < ADD_LONG_MIN_UNREALIZED_PCT:
            return False, "add_long_masked_not_profitable"
        if self.cash <= 0:
            return False, "add_long_masked_no_cash"

        refs = self._market_refs(price, row)
        future_10 = to_float(row.get("future_return_10"), 0.0)
        future_20 = to_float(row.get("future_return_20"), future_10)
        future_30 = to_float(row.get("future_return_30"), future_20)

        continuation_score = (
            to_float(row.get("breakout_persistence_score"), 0.0)
            + to_float(row.get("bullish_acceleration_score"), 0.0)
            + to_float(row.get("positive_volume_expansion_score"), 0.0)
            + to_float(row.get("higher_high_score"), 0.0)
            + to_float(row.get("higher_low_score"), 0.0)
            + max(0.0, to_float(row.get("ema_spread"), 0.0) * 100.0)
            + max(0.0, future_20 * 100.0)
        ) / 7.0

        above_vwap = price >= refs["vwap"]
        future_continuation = max(future_10, future_20, future_30)

        if not above_vwap and continuation_score < ADD_LONG_MIN_CONTINUATION_SCORE and future_continuation <= 0:
            return False, "add_long_masked_weak_continuation"
        if future_continuation < ADD_LONG_MAX_ADVERSE_FUTURE:
            return False, "add_long_masked_adverse_future"

        current_notional = self.position_qty * price
        if current_notional >= STARTING_CASH * MAX_TOTAL_POSITION_FRACTION:
            return False, "add_long_masked_max_position_fraction"

        return True, None

    def _exit_short_allowed(self) -> tuple[bool, str | None]:
        return False, "shorting_disabled_long_only"

    def _apply_action_mask(self, action: int, price: float, row: pd.Series) -> tuple[int, str | None]:
        if not ACTION_MASKING_ENABLED:
            return action, None

        original_action = int(action)

        if action == 1:
            allowed, reason = self._long_allowed(price, row)
            if not allowed:
                self.masked_action_counts["long_masked"] += 1
                self._debug_mask(original_action, reason, price)
                return 0, reason

        if action == 2:
            allowed, reason = self._exit_long_allowed()
            if not allowed:
                self.masked_action_counts["exit_long_masked"] += 1
                self._debug_mask(original_action, reason, price)
                return 0, reason

        if action == 3:
            allowed, reason = self._add_long_allowed(price, row)
            if not allowed:
                self.masked_action_counts["long_masked"] += 1
                self._debug_mask(original_action, reason, price)
                return 0, reason

        if action >= 4:
            reason = "shorting_disabled_long_only"
            self.masked_action_counts["short_masked"] += 1
            self._debug_mask(original_action, reason, price)
            return 0, reason

        return action, None

    def _debug_mask(self, original_action: int, reason: str | None, price: float) -> None:
        if DEBUG_MASKS:
            print(
                f"[mask] symbol={self.symbol} idx={self.idx} "
                f"action={ACTION_NAMES.get(original_action, original_action)} -> HOLD "
                f"reason={reason} price={price:.4f}"
            )

    def _obs(self) -> np.ndarray:
        row = self.episode_df.iloc[min(self.idx, len(self.episode_df) - 1)]
        price = max(to_float(row["close"]), 0.01)
        structure = self._structure()
        refs = self._market_refs(price, row)
        entry_gate, long_gate, short_gate = self._hierarchical_gates(price, row)

        values = {
            "price_vwap": safe_div(price, refs["vwap"], 1.0) - 1.0,
            "price_vwap_upper": safe_div(price, refs["vwap_upper"], 1.0) - 1.0,
            "price_vwap_lower": safe_div(price, refs["vwap_lower"], 1.0) - 1.0,
            "price_orb_high": safe_div(price, refs["orb_high"], 1.0) - 1.0,
            "price_orb_low": safe_div(price, refs["orb_low"], 1.0) - 1.0,
            "price_support_1": safe_div(price, refs["support_1"], 1.0) - 1.0,
            "price_resistance_1": safe_div(price, refs["resistance_1"], 1.0) - 1.0,
            "ret_1": to_float(row.get("ret_1")),
            "ret_3": to_float(row.get("ret_3")),
            "ret_5": to_float(row.get("ret_5")),
            "ret_10": to_float(row.get("ret_10")),
            "ret_20": to_float(row.get("ret_20")),
            "ema_spread": to_float(row.get("ema_spread")),
            "vwap_slope_5": to_float(row.get("vwap_slope_5")),
            "volume_ratio_bar": to_float(row.get("volume_ratio_bar"), 1.0),
            "range_pct": to_float(row.get("range_pct")),
            "close_location": to_float(row.get("close_location"), 0.5),
            "dist_high_20": to_float(row.get("dist_high_20")),
            "dist_low_20": to_float(row.get("dist_low_20")),
            "breakout_persistence_score": to_float(row.get("breakout_persistence_score")),
            "bullish_acceleration_score": to_float(row.get("bullish_acceleration_score")),
            "positive_volume_expansion_score": to_float(row.get("positive_volume_expansion_score")),
            "higher_high_score": to_float(row.get("higher_high_score")),
            "higher_low_score": to_float(row.get("higher_low_score")),
            "bullish_structure_score": to_float(row.get("bullish_structure_score")),
            "breakdown_persistence_score": to_float(row.get("breakdown_persistence_score")),
            "bearish_acceleration_score": to_float(row.get("bearish_acceleration_score")),
            "lower_low_score": to_float(row.get("lower_low_score")),
            "lower_high_score": to_float(row.get("lower_high_score")),
            "structure_volume_ratio": to_float(structure.get("volume_ratio"), 1.0),
            "liquidity_score": to_float(structure.get("liquidity_score"), 0.5),
            "spread_proxy": to_float(structure.get("spread_proxy"), 0.003),
            "trend_score": to_float(structure.get("trend_score"), 0.0),
            "breakout_score": to_float(structure.get("breakout_score"), 0.0),
            "breakdown_score": to_float(structure.get("breakdown_score"), 0.0),
            "sector_strength": to_float(structure.get("sector_strength"), 0.0),
            "market_bias": to_float(structure.get("market_bias"), 0.0),
            "position_side": float(self.position_side),
            "is_long": 1.0 if self.position_side == 1 else 0.0,
            "is_short": 1.0 if self.position_side == -1 else 0.0,
            "unrealized_pct": self._unrealized_pct(price),
            "long_allowed": to_float(structure.get("long_allowed"), 1.0),
            "bearish_regime": to_float(structure.get("bearish_regime"), 0.0),
            "reentry_cooldown": 1.0 if self._reentry_cooldown_active() else 0.0,
            "hierarchical_entry_gate": entry_gate,
            "hierarchical_long_gate": long_gate,
            "hierarchical_short_gate": short_gate,
            "daily_change_pct": to_float(row.get("daily_change_pct")),
            "intraday_from_open_pct": to_float(row.get("intraday_from_open_pct")),
            "distance_from_day_high": to_float(row.get("distance_from_day_high")),
            "distance_from_day_low": to_float(row.get("distance_from_day_low")),
            "current_position_pnl_pct": self._unrealized_pct(price),
            "bars_since_entry": float(self.hold_bars),
            "reentry_count_today": float(self.long_entry_count),
            "minutes_since_last_exit": float(self.idx - self.last_exit_idx) if self.last_exit_idx is not None else 999.0,
            "daily_equity_drawdown_pct": self._daily_equity_drawdown_pct(price),
            "entry_cross_count": float(self.entry_cross_count),
            "vwap_cross_count": float(self.vwap_cross_count),
            "max_favorable_excursion": float(self.max_favorable_excursion),
            "max_adverse_excursion": float(self.max_adverse_excursion),
            "current_giveback_from_mfe": float(max(0.0, self.max_favorable_excursion - self._unrealized_pct(price))),
            "ret_slope_3_10": to_float(row.get("ret_slope_3_10")),
            "momentum_decay_score": to_float(row.get("momentum_decay_score")),
            "volume_fade_score": to_float(row.get("volume_fade_score")),
            "vwap_loss_score": to_float(row.get("vwap_loss_score")),
            "near_day_high_score": to_float(row.get("near_day_high_score")),
            "giveback_from_day_high_pct": to_float(row.get("giveback_from_day_high_pct")),
            "future_max_return_5": to_float(row.get("future_max_return_5")),
            "future_max_return_15": to_float(row.get("future_max_return_15")),
            "future_max_return_30": to_float(row.get("future_max_return_30")),
            "future_min_return_15": to_float(row.get("future_min_return_15")),
            "future_min_return_30": to_float(row.get("future_min_return_30")),
        }

        if USE_TRANSFORMER_FEATURES:
            for window in SEQUENCE_WINDOWS:
                values[f"seq_ret_mean_{window}"] = to_float(row.get(f"seq_ret_mean_{window}"))
                values[f"seq_ret_std_{window}"] = to_float(row.get(f"seq_ret_std_{window}"))
                values[f"seq_vol_mean_{window}"] = to_float(row.get(f"seq_vol_mean_{window}"), 1.0)
                values[f"seq_trend_consistency_{window}"] = to_float(row.get(f"seq_trend_consistency_{window}"), 0.5)

        if USE_MULTI_TIMEFRAME_FUSION:
            for window in MTF_WINDOWS:
                values[f"mtf_ret_{window}"] = to_float(row.get(f"mtf_ret_{window}"))
                values[f"mtf_range_{window}"] = to_float(row.get(f"mtf_range_{window}"))
                values[f"mtf_volume_ratio_{window}"] = to_float(row.get(f"mtf_volume_ratio_{window}"), 1.0)
                values[f"mtf_vwap_dist_{window}"] = to_float(row.get(f"mtf_vwap_dist_{window}"))

        obs = np.array([values.get(name, 0.0) for name in self.feature_names], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _equity(self, price: float) -> float:
        if self.position_side == 1:
            return self.cash + self.position_qty * price
        return self.cash

    def _transaction_cost(self, notional: float) -> float:
        return abs(notional) * ((COMMISSION_BPS + SLIPPAGE_BPS) / 10000.0)

    def _daily_equity_drawdown_pct(self, price: float) -> float:
        equity = self._equity(price)
        return equity / max(self.equity_peak, STARTING_CASH, 1.0) - 1.0

    def _side_adjusted_future_return(self, row: pd.Series, horizon: int, side: int | None = None) -> float:
        side = self.position_side if side is None else side
        value = to_float(row.get(f"future_return_{horizon}"), 0.0)
        if side == -1:
            return -value
        return value

    def _future_min_return(self, row: pd.Series, horizon: int = 5) -> float:
        return to_float(row.get(f"future_min_return_{horizon}"), 0.0)


    def _side_adjusted_future_max_return(self, row: pd.Series, horizon: int, side: int | None = None) -> float:
        """Return favorable future excursion from the current bar for long/short side."""
        side = self.position_side if side is None else side
        if side == -1:
            return -to_float(row.get(f"future_min_return_{horizon}"), 0.0)
        return to_float(row.get(f"future_max_return_{horizon}"), 0.0)

    def _side_adjusted_future_giveback_return(self, row: pd.Series, horizon: int, side: int | None = None) -> float:
        """Return adverse future excursion from the current bar for long/short side."""
        side = self.position_side if side is None else side
        if side == -1:
            return to_float(row.get(f"future_max_return_{horizon}"), 0.0)
        return -to_float(row.get(f"future_min_return_{horizon}"), 0.0)

    def _current_giveback_from_mfe(self, price: float) -> float:
        if self.position_side == 0:
            return 0.0
        return max(0.0, self.max_favorable_excursion - self._unrealized_pct(price))

    def _trade_duration_multiplier(self, bars_held: int) -> float:
        if bars_held <= 1:
            return 0.50
        if bars_held <= 3:
            return 0.85
        if bars_held <= 5:
            return 1.00
        if bars_held <= 10:
            return 1.25
        if bars_held <= 20:
            return 1.50
        if bars_held <= 40:
            return 1.75
        return 1.25

    def _update_open_position_trackers(self, price: float, row: pd.Series) -> None:
        if self.position_side == 0 or self.entry_price <= 0:
            return

        pnl_pct = self._unrealized_pct(price)
        self.max_favorable_excursion = max(self.max_favorable_excursion, pnl_pct)
        self.max_adverse_excursion = min(self.max_adverse_excursion, pnl_pct)

        if self.idx > 0:
            prev = self.episode_df.iloc[self.idx - 1]
            prev_price = to_float(prev.get("close"), price)
            if (prev_price - self.entry_price) * (price - self.entry_price) < 0:
                self.entry_cross_count += 1

            prev_vwap = to_float(prev.get("running_vwap"), prev_price)
            refs = self._market_refs(price, row)
            if (prev_price - prev_vwap) * (price - refs["vwap"]) < 0:
                self.vwap_cross_count += 1

    def _reset_position_quality_trackers(self) -> None:
        self.entry_idx = None
        self.entry_was_breakout = False
        self.entry_cross_count = 0
        self.vwap_cross_count = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0

    def _next_bar_reward(self, row: pd.Series, side: int) -> float:
        r1 = self._side_adjusted_future_return(row, 1, side)
        r3 = self._side_adjusted_future_return(row, 3, side)
        r5 = self._side_adjusted_future_return(row, 5, side)
        reward = 0.0

        if r1 > 0:
            reward += min(r1 * NEXT_BAR_REWARD_MULT, NEXT_BAR_REWARD_CAP)
        else:
            reward -= min(abs(r1) * NEXT_BAR_NEG_MULT, NEXT_BAR_NEG_CAP)

        if r1 > NEXT_BAR_STRONG_MOVE:
            reward += NEXT_BAR_STRONG_BONUS
        elif r1 < -NEXT_BAR_STRONG_MOVE:
            reward -= NEXT_BAR_STRONG_PENALTY

        if r3 > 0:
            reward += min(r3 * NEXT_3BAR_REWARD_MULT, NEXT_3BAR_REWARD_CAP)
        else:
            reward -= min(abs(r3) * NEXT_3BAR_NEG_MULT, NEXT_3BAR_NEG_CAP)

        if r5 > 0:
            reward += min(r5 * NEXT_5BAR_REWARD_MULT, NEXT_5BAR_REWARD_CAP)
        else:
            reward -= min(abs(r5) * NEXT_5BAR_NEG_MULT, NEXT_5BAR_NEG_CAP)

        return reward

    def _daily_momentum_bonus(self, row: pd.Series, side: int) -> float:
        daily = to_float(row.get("daily_change_pct"), 0.0) * (1 if side == 1 else -1)
        intraday = to_float(row.get("intraday_from_open_pct"), 0.0) * (1 if side == 1 else -1)
        reward = 0.0

        if daily > 0:
            reward += min(daily * DAILY_MOMENTUM_BONUS_MULT, DAILY_MOMENTUM_BONUS_CAP)
        if intraday > 0:
            reward += min(intraday * INTRADAY_MOMENTUM_BONUS_MULT, INTRADAY_MOMENTUM_BONUS_CAP)
        if daily > OVEREXTENDED_DAILY_CHANGE:
            reward -= OVEREXTENDED_DAILY_PENALTY
        if intraday > OVEREXTENDED_INTRADAY_CHANGE:
            reward -= OVEREXTENDED_INTRADAY_PENALTY

        return reward

    def _negative_daily_momentum_penalty(self, row: pd.Series, side: int) -> float:
        daily = to_float(row.get("daily_change_pct"), 0.0) * (1 if side == 1 else -1)
        intraday = to_float(row.get("intraday_from_open_pct"), 0.0) * (1 if side == 1 else -1)
        penalty = 0.0

        if daily < 0:
            penalty -= min(abs(daily) * NEG_DAILY_MOMENTUM_MULT, NEG_DAILY_MOMENTUM_CAP)
        if intraday < 0:
            penalty -= min(abs(intraday) * NEG_INTRADAY_MOMENTUM_MULT, NEG_INTRADAY_MOMENTUM_CAP)
        if daily < NEG_DAILY_BREAKDOWN_LEVEL:
            penalty -= NEG_DAILY_BREAKDOWN_PENALTY
        if intraday < NEG_INTRADAY_BREAKDOWN_LEVEL:
            penalty -= NEG_INTRADAY_BREAKDOWN_PENALTY

        return penalty

    def _position_pnl_reward(self, price: float, row: pd.Series) -> float:
        if self.position_side == 0:
            return 0.0

        pnl_pct = self._unrealized_pct(price)
        reward = 0.0

        if pnl_pct > 0:
            reward += min(pnl_pct * POSITION_PNL_BONUS_MULT, POSITION_PNL_BONUS_CAP)
        else:
            reward -= min(abs(pnl_pct) * POSITION_PNL_NEG_MULT, POSITION_PNL_NEG_CAP)

        if pnl_pct > FAST_PROFIT_THRESHOLD and self.hold_bars <= FAST_PROFIT_BARS:
            reward += FAST_PROFIT_BONUS
        if pnl_pct < INSTANT_DRAWDOWN_THRESHOLD and self.hold_bars <= FAST_PROFIT_BARS:
            reward -= INSTANT_DRAWDOWN_PENALTY
        if abs(pnl_pct) < STAGNANT_POSITION_THRESHOLD and self.hold_bars > STAGNANT_POSITION_BARS:
            reward -= STAGNANT_POSITION_PENALTY
        if pnl_pct > 0.01:
            reward += 0.5
        if pnl_pct > 0.02:
            reward += 1.0

        return reward

    def _reentry_penalty(self) -> float:
        penalty = 0.0
        entries_today = self.long_entry_count

        if entries_today >= 1:
            penalty -= REENTRY_SAME_SYMBOL_PENALTY
        if self.last_exit_idx is not None and (self.idx - self.last_exit_idx) < REENTRY_RECENT_EXIT_BARS:
            penalty -= REENTRY_RECENT_EXIT_PENALTY
        if entries_today >= REENTRY_COUNT_TRIGGER:
            penalty -= REENTRY_COUNT_PENALTY

        if (
            self.last_exit_idx is not None
            and self.prev_entry_idx is not None
            and (self.last_exit_idx - self.prev_entry_idx) < CHOP_REENTRY_MAX_HOLD_BARS
            and (self.idx - self.last_exit_idx) < CHOP_ENTRY_EXIT_ENTRY_BARS
        ):
            penalty -= CHOP_REENTRY_PENALTY

        return penalty

    def _multibar_clean_continuation_bonus(self, price: float, row: pd.Series) -> float:
        """Reward clean multi-bar trend continuation; discourages early scalp exits."""
        if self.position_side != 1 or self.hold_bars < MULTIBAR_CLEAN_BARS_THRESHOLD:
            return 0.0
        f3 = to_float(row.get("future_return_3"), 0.0)
        f5 = to_float(row.get("future_return_5"), 0.0)
        if f3 > 0 and f5 > 0 and self._unrealized_pct(price) > 0:
            scale = min(self.hold_bars / MULTIBAR_CLEAN_BARS_THRESHOLD, 3.0)
            return MULTIBAR_CLEAN_CONTINUATION_REWARD * scale
        return 0.0

    def _equity_penalty(self, previous_equity: float, new_equity: float, price: float) -> float:
        penalty = 0.0
        one_step = new_equity / max(previous_equity, 1.0) - 1.0
        daily_dd = self._daily_equity_drawdown_pct(price)

        if one_step < 0:
            penalty -= min(abs(one_step) * EQUITY_NEG_1STEP_MULT, EQUITY_NEG_1STEP_CAP)
        if daily_dd < DAILY_EQUITY_DD_1:
            penalty -= DAILY_EQUITY_DD_1_PENALTY
        if daily_dd < DAILY_EQUITY_DD_2:
            penalty -= DAILY_EQUITY_DD_2_PENALTY

        return penalty

    def _missed_opportunity_penalty(self, action: int, row: pd.Series, price: float) -> float:
        if action not in {0, 2, 4}:  # only wait/exit/skip-style decisions are penalized for missing upside
            return 0.0

        structure = self._structure()
        r5 = to_float(row.get("future_return_5"), 0.0)
        r15 = to_float(row.get("future_return_15"), 0.0)
        reward = 0.0

        if r5 > MISS_FUTURE_5M_THRESHOLD:
            reward -= 0.50
        if r15 > MISS_FUTURE_15M_THRESHOLD:
            reward -= 1.00

        if (
            to_float(row.get("daily_change_pct"), 0.0) > 0.02
            and to_float(row.get("volume_ratio_bar"), 1.0) > 2.0
            and to_float(row.get("bullish_structure_score"), 0.0) > 0.60
            and to_float(structure.get("trend_score"), 0.0) > 0.50
        ):
            reward -= MISS_STRONG_SETUP_PENALTY

        if (
            to_float(row.get("breakout_persistence_score"), 0.0) > 0.0
            and r5 > 0.01
            and r15 > 0.02
        ):
            reward -= MISS_BREAKOUT_PENALTY

        if r5 < 0:
            reward += AVOID_BAD_TRADE_REWARD
        if self._future_min_return(row, 5) < -0.005:
            reward += AVOID_ADVERSE_TRADE_REWARD
        if to_float(row.get("daily_change_pct"), 0.0) < 0:
            reward += 0.20

        return reward

    def _exit_peak_quality_reward(self, price: float, row: pd.Series, side: int) -> float:
        """Reward exits near the next local high and before larger giveback.

        This uses future windows only in the training simulator. Live inference
        still sees only real-time features, but the learned policy discovers
        that momentum decay + giveback from MFE usually means the top is near.
        """
        if self.entry_price <= 0:
            return 0.0

        position_return_pct = price / self.entry_price - 1.0 if side == 1 else self.entry_price / price - 1.0
        future_upside = max(
            0.0,
            self._side_adjusted_future_max_return(row, PEAK_EXIT_LOOKAHEAD_BARS, side),
        )
        future_adverse = max(
            0.0,
            self._side_adjusted_future_giveback_return(row, PEAK_EXIT_LOOKAHEAD_BARS, side),
        )
        current_giveback = self._current_giveback_from_mfe(price)
        mfe = max(self.max_favorable_excursion, position_return_pct, 0.0)

        reward = 0.0

        # Efficiency: when future upside is small, exiting now is close to the
        # next local peak. Reward profitable exits more aggressively.
        peak_efficiency = 1.0 - min(max(future_upside / max(PEAK_EXIT_MIN_MFE, 1e-6), 0.0), 1.0)
        if position_return_pct > 0:
            reward += peak_efficiency * PEAK_EXIT_EFFICIENCY_BONUS
            if peak_efficiency >= 0.80:
                reward += PEAK_EXIT_NEAR_TOP_BONUS

        # Penalize selling too early when the next window still has meaningful
        # favorable excursion left.
        if future_upside > PEAK_EXIT_MIN_MFE:
            reward -= min(future_upside * PEAK_EXIT_MISSED_UPSIDE_PENALTY_MULT, PEAK_EXIT_MISSED_UPSIDE_CAP)

        # Reward exits that avoid future adverse move after a profitable run.
        if mfe >= PEAK_EXIT_MIN_MFE and future_adverse >= PEAK_EXIT_GIVEBACK_TRIGGER:
            reward += min(future_adverse * PEAK_EXIT_GIVEBACK_AVOID_BONUS_MULT, PEAK_EXIT_GIVEBACK_AVOID_BONUS_CAP)

        # Penalize waiting too long and exiting after giving back too much MFE.
        if mfe >= PEAK_EXIT_MIN_MFE and current_giveback >= PEAK_EXIT_GIVEBACK_TRIGGER:
            reward -= min(current_giveback * PEAK_EXIT_LATE_GIVEBACK_PENALTY_MULT, PEAK_EXIT_LATE_GIVEBACK_PENALTY_CAP)

        return float(reward)

    def _hold_peak_continuation_reward(self, price: float, row: pd.Series) -> float:
        """Teach the agent to hold strong trends, but not through confirmed giveback."""
        if self.position_side == 0:
            return 0.0

        side = self.position_side
        future_upside = max(0.0, self._side_adjusted_future_max_return(row, PEAK_EXIT_LOOKAHEAD_BARS, side))
        future_adverse = max(0.0, self._side_adjusted_future_giveback_return(row, PEAK_EXIT_LOOKAHEAD_BARS, side))
        current_giveback = self._current_giveback_from_mfe(price)
        momentum_decay = to_float(row.get("momentum_decay_score"), 0.0)
        vwap_loss = to_float(row.get("vwap_loss_score"), 0.0)

        reward = 0.0
        if future_upside > PEAK_EXIT_MIN_MFE and current_giveback < PEAK_EXIT_GIVEBACK_TRIGGER:
            reward += min(future_upside * 50.0, HOLD_WHILE_CONTINUATION_REWARD)

        if (
            self.max_favorable_excursion >= PEAK_EXIT_MIN_MFE
            and current_giveback >= PEAK_EXIT_GIVEBACK_TRIGGER
            and (momentum_decay >= 0.67 or vwap_loss >= 1.0 or future_adverse > future_upside)
        ):
            reward -= HOLD_THROUGH_GIVEBACK_PENALTY

        return float(reward)


    def _mfe_mae_capture_stats(self, price: float, row: pd.Series, side: int | None = None) -> dict[str, float]:
        side = self.position_side if side is None else side
        realized = self._unrealized_pct(price) if side == self.position_side else 0.0
        current_mfe = max(self.max_favorable_excursion, realized, 0.0)
        current_mae = min(self.max_adverse_excursion, realized, 0.0)
        future_mfe = max(0.0, self._side_adjusted_future_max_return(row, PEAK_EXIT_LOOKAHEAD_BARS, side))
        total_available_mfe = max(current_mfe, realized + future_mfe, 1e-6)
        capture_ratio = max(0.0, min(2.0, max(realized, 0.0) / total_available_mfe))
        giveback = max(0.0, current_mfe - realized)
        adverse = abs(min(current_mae, 0.0))
        return {
            "realized_pct": float(realized),
            "mfe_pct": float(current_mfe),
            "mae_pct": float(adverse),
            "future_mfe_pct": float(future_mfe),
            "capture_ratio": float(capture_ratio),
            "giveback_pct": float(giveback),
        }

    def _mfe_mae_trajectory_reward(self, price: float, row: pd.Series, side: int) -> float:
        stats = self._mfe_mae_capture_stats(price, row, side)
        reward = 0.0
        if stats["realized_pct"] > 0:
            reward += min(stats["capture_ratio"] * MFE_CAPTURE_REWARD_MULT, MFE_CAPTURE_REWARD_CAP)
            reward += min(max(0.0, self.hold_bars / max(LONG_HOLD_TARGET_BARS, 1.0)) * AVG_HOLD_REWARD_MULT, 2.0)
        reward -= min(stats["mae_pct"] * MAE_PENALTY_MULT, MAE_PENALTY_CAP)
        reward -= min(stats["giveback_pct"] * GIVEBACK_CAPTURE_PENALTY_MULT, GIVEBACK_CAPTURE_PENALTY_CAP)
        return float(reward)

    def _exit_duration_quality_reward(self, price: float, row: pd.Series, side: int) -> float:
        if self.entry_price <= 0:
            return 0.0

        if side == 1:
            position_return_pct = price / self.entry_price - 1.0
        else:
            position_return_pct = self.entry_price / price - 1.0

        duration_mult = self._trade_duration_multiplier(self.hold_bars)
        reward = 0.0

        if position_return_pct > 0:
            reward += position_return_pct * 100.0 * duration_mult
        if position_return_pct > 0.005 and self.hold_bars >= 5:
            reward += 0.75
        if position_return_pct > 0.01 and self.hold_bars >= 10:
            reward += 1.50
        if position_return_pct > 0.02 and self.hold_bars >= 20:
            reward += 3.00

        if self.hold_bars <= 2 and abs(position_return_pct) < 0.003:
            reward -= SCALP_TINY_MOVE_PENALTY
        if self.hold_bars <= 3 and position_return_pct < 0:
            reward -= SCALP_LOSER_PENALTY
        if self.hold_bars >= 10 and abs(position_return_pct) < 0.002:
            reward -= CHOP_STAGNANT_PENALTY
        if self.entry_cross_count >= 3:
            reward -= ENTRY_CROSS_PENALTY
        if self.vwap_cross_count >= 3:
            reward -= VWAP_CROSS_PENALTY
        if self.entry_was_breakout and price < self.entry_price:
            reward -= FAILED_BREAKOUT_PENALTY
        if self.max_favorable_excursion > 0.01 and position_return_pct < 0.003:
            reward -= GIVEBACK_AFTER_MFE_PENALTY
        if to_float(row.get("bullish_structure_score"), 0.0) < 0.35 and to_float(self._structure().get("trend_score"), 0.0) <= 0:
            reward -= DUAL_TREND_FAILURE_PENALTY

        reward += self._mfe_mae_trajectory_reward(price, row, side=side)
        reward += self._exit_peak_quality_reward(price, row, side=side)

        return reward

    def _dense_execution_reward(
        self,
        action: int,
        price: float,
        row: pd.Series,
        previous_equity: float,
        new_equity: float,
    ) -> float:
        reward = 0.0

        if action == 1:
            reward += self._next_bar_reward(row, 1)
            reward += self._daily_momentum_bonus(row, 1)
            reward += self._negative_daily_momentum_penalty(row, 1)
            reward += self._reentry_penalty()
        elif action == 0:
            reward += self._missed_opportunity_penalty(action, row, price)
            reward += self._hold_peak_continuation_reward(price, row)

        reward += self._position_pnl_reward(price, row)
        reward += self._equity_penalty(previous_equity, new_equity, price)

        # Penalize sloppy long entries that immediately experience adverse excursion,
        # but keep this small during exploration so PPO does not learn HOLD-only.
        if action == 1 and self._future_min_return(row, 5) < -0.005:
            reward -= 0.10

        return reward

    def _short_bias_active(self) -> bool:
        return False

    def _long_bias_active(self) -> bool:
        return False

    def _long_trend_reward(self, price: float, row: pd.Series) -> float:
        structure = self._structure()
        reward = 0.0

        trend_score = to_float(structure.get("trend_score"), 0.0)
        market_bias = to_float(structure.get("market_bias"), 0.0)
        sector_strength = to_float(structure.get("sector_strength"), 0.0)

        if trend_score > 0.70:
            reward += LONG_STRONG_TREND_BONUS
        if to_float(row.get("ret_5"), 0.0) > 0:
            reward += LONG_POSITIVE_RET5_BONUS
        if to_float(row.get("ema_spread"), 0.0) > 0:
            reward += LONG_POSITIVE_EMA_SPREAD_BONUS
        if market_bias > 0.50 or to_float(structure.get("risk_on"), 0.0) >= 1.0:
            reward += LONG_BULLISH_MARKET_BONUS
        if sector_strength > 0.50:
            reward += LONG_SECTOR_STRENGTH_BONUS
        if to_float(row.get("breakout_persistence_score"), 0.0) > 0:
            reward += LONG_BREAKOUT_PERSISTENCE_BONUS
        if to_float(row.get("bullish_acceleration_score"), 0.0) > 0.50:
            reward += LONG_BULLISH_ACCELERATION_BONUS
        if to_float(row.get("positive_volume_expansion_score"), 0.0) > 0:
            reward += LONG_POSITIVE_VOLUME_EXPANSION_BONUS
        if to_float(row.get("higher_high_score"), 0.0) > 0:
            reward += LONG_HIGHER_HIGH_BONUS
        if to_float(row.get("higher_low_score"), 0.0) > 0:
            reward += LONG_HIGHER_LOW_BONUS

        return reward

    def _enter_long(self, price: float, row: pd.Series, reason: str) -> float:
        if self.position_side != 0:
            return -OVERTRADE_PENALTY

        notional = self.cash * POSITION_FRACTION
        if notional <= 0:
            return -BAD_ENTRY_PENALTY

        cost = self._transaction_cost(notional)
        qty = max((notional - cost) / price, 0.0)

        self.cash -= qty * price + cost
        self.position_qty = qty
        self.entry_price = price
        self.position_side = 1
        self.hold_bars = 0
        self.prev_entry_idx = self.entry_idx  # save before overwrite so chop penalty can measure previous hold
        self.entry_idx = self.idx
        refs_for_entry = self._market_refs(price, row)
        self.entry_was_breakout = bool(price >= refs_for_entry["orb_high"] or to_float(row.get("breakout_persistence_score"), 0.0) > 0.0)
        self.entry_cross_count = 0
        self.vwap_cross_count = 0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0

        structure = self._structure()
        refs = self._market_refs(price, row)

        reward = -cost / STARTING_CASH * REWARD_SCALE

        if price >= refs["orb_high"]:
            reward += BUY_BREAKOUT_BONUS
        if price >= refs["vwap"]:
            reward += BUY_VWAP_BONUS
        if to_float(structure.get("breakout_score"), 0.0) > 0.5:
            reward += BUY_BREAKOUT_SCORE_BONUS
        if to_float(structure.get("trend_score"), 0.0) > 0.5:
            reward += BUY_TREND_SCORE_BONUS

        reward += self._long_trend_reward(price, row)

        distance_to_resistance = to_float(refs["resistance_1"], price * 1.02) / max(price, 0.01) - 1.0
        if distance_to_resistance < 0.003:
            reward -= BUY_NEAR_RESISTANCE_PENALTY

        distance_to_support = price / max(to_float(refs["support_1"], price * 0.98), 0.01) - 1.0
        if distance_to_support < 0.004:
            reward += BUY_NEAR_SUPPORT_BONUS

        if to_float(structure.get("bearish_regime"), 0.0) >= 1.0:
            reward -= REGIME_MISALIGNED_PENALTY
        elif to_float(structure.get("risk_on"), 0.0) >= 1.0 or to_float(structure.get("market_bias"), 0.0) > 0:
            reward += REGIME_ALIGNED_REWARD

        if self._long_bias_active():
            reward -= LONG_BIAS_PENALTY

        # Explicitly reward valid entry exploration so PPO does not converge to
        # a no-trade policy simply because flat portfolio has low variance.
        reward += VALID_ENTRY_EXPLORATION_BONUS
        reward += VALID_ENTRY_REWARD

        if self.long_entry_count == 0:
            reward += FIRST_TRADE_BONUS

        future_3 = to_float(row.get("future_return_3"), 0.0)
        if future_3 > 0:
            reward += min(future_3 * ENTRY_FUTURE_3BAR_MULT, ENTRY_FUTURE_3BAR_CAP)

        self.long_entry_count += 1
        self._log_trade(row, "ENTER_LONG", price, reward, reason)
        return reward

    def _add_long(self, price: float, row: pd.Series, reason: str) -> float:
        allowed, mask_reason = self._add_long_allowed(price, row)
        if not allowed:
            return -OVERTRADE_PENALTY * 0.5

        current_notional = self.position_qty * price
        max_notional = STARTING_CASH * MAX_TOTAL_POSITION_FRACTION
        available_capacity = max(0.0, max_notional - current_notional)
        notional = min(self.cash * ADD_LONG_FRACTION, available_capacity)
        if notional <= 0:
            return -BAD_ENTRY_PENALTY

        cost = self._transaction_cost(notional)
        qty = max((notional - cost) / price, 0.0)
        if qty <= 0:
            return -BAD_ENTRY_PENALTY

        old_qty = self.position_qty
        old_basis = self.entry_price

        self.cash -= qty * price + cost
        self.position_qty += qty
        self.entry_price = ((old_qty * old_basis) + (qty * price)) / max(self.position_qty, 1e-9)
        self.add_long_count += 1

        reward = ADD_LONG_REWARD - cost / STARTING_CASH * REWARD_SCALE
        reward += self._long_trend_reward(price, row)

        future_5 = to_float(row.get("future_return_5"), 0.0)
        future_10 = to_float(row.get("future_return_10"), future_5)
        future_20 = to_float(row.get("future_return_20"), future_10)
        future_30 = to_float(row.get("future_return_30"), future_20)
        continuation = max(future_5, future_10, future_20, future_30, 0.0)
        reward += min(continuation * ADD_LONG_FUTURE_REWARD_MULT, ADD_LONG_FUTURE_REWARD_CAP)

        if continuation <= 0:
            reward -= BAD_ADD_LONG_PENALTY

        if to_float(row.get("breakout_persistence_score"), 0.0) > 0:
            reward += 1.25
        if to_float(row.get("positive_volume_expansion_score"), 0.0) > 0:
            reward += 0.90
        if to_float(row.get("bullish_acceleration_score"), 0.0) > 0.50:
            reward += 0.75

        self._log_trade(row, "ADD_LONG", price, reward, reason or mask_reason or "agent_add_long")
        return reward

    def _exit_long(self, price: float, row: pd.Series, reason: str) -> float:
        if self.position_side != 1 or self.position_qty <= 0:
            return -OVERTRADE_PENALTY * 0.5

        notional = self.position_qty * price
        cost = self._transaction_cost(notional)
        pnl = (price - self.entry_price) * self.position_qty - cost

        self.cash += notional - cost
        self.realized_pnl += pnl

        reward = (pnl / STARTING_CASH) * REWARD_SCALE
        reward += self._exit_duration_quality_reward(price, row, side=1)

        pnl_pct = price / max(self.entry_price, 0.01) - 1.0

        if self.hold_bars < MIN_QUALITY_HOLD_BARS and pnl > 0:
            reward -= PREMATURE_EXIT_PENALTY

        # Anti-scalp shaping: do not teach PPO that 1-2 bar profitable exits are
        # the best default when continuation conditions remain favorable.
        if self.hold_bars < EARLY_WINNER_EXIT_BARS and pnl_pct > 0:
            reward -= EARLY_WINNER_EXIT_PENALTY

        if self.hold_bars >= QUALITY_WINNER_MIN_HOLD_BARS and pnl_pct > 0:
            reward += QUALITY_WINNER_HOLD_BONUS

        bullish_structure = to_float(row.get("bullish_structure_score"), 0.0)
        if bullish_structure > 0.70 and self._unrealized_pct(price) > 0:
            reward -= STRONG_TREND_EXIT_PENALTY

        self._log_trade(row, "EXIT_LONG", price, reward, reason)

        self.position_qty = 0.0
        self.entry_price = 0.0
        self.position_side = 0
        self.hold_bars = 0
        self.last_exit_idx = self.idx
        self.add_long_count = 0
        self._reset_position_quality_trackers()

        return reward

    def _enter_short(self, price: float, row: pd.Series, reason: str) -> float:
        return -NO_TRADE_EPISODE_PENALTY

    def _exit_short(self, price: float, row: pd.Series, reason: str) -> float:
        return -OVERTRADE_PENALTY

    def _hold_reward(self, price: float, row: pd.Series, mask_reason: str | None = None) -> float:
        reward = 0.0
        if self.position_side != 0 and self._unrealized_pct(price) > 0:
            reward += min(self.hold_bars / max(LONG_HOLD_TARGET_BARS, 1.0), 1.0) * AVG_HOLD_REWARD_MULT
        structure = self._structure()
        refs = self._market_refs(price, row)

        if self.position_side == 0:
            reward -= FLAT_HOLD_STEP_PENALTY

            missed_breakout = (
                price > refs["vwap"]
                and price > refs["orb_high"]
                and to_float(structure.get("breakout_score"), 0.0) > 0.5
            )
            missed_breakdown = False

            strong_long_setup = (
                to_float(row.get("bullish_structure_score"), 0.0) >= 0.55
                or to_float(row.get("future_return_5"), 0.0) >= MISS_FUTURE_5M_THRESHOLD
                or missed_breakout
            )
            strong_short_setup = (
                to_float(row.get("bearish_acceleration_score"), 0.0) >= 0.55
                or to_float(row.get("future_return_5"), 0.0) <= -MISS_FUTURE_5M_THRESHOLD
                or missed_breakdown
            )

            if missed_breakout:
                reward -= MISSED_BREAKOUT_HOLD_PENALTY + 0.004 * to_float(row.get("bullish_structure_score"), 0.0)
            if missed_breakdown:
                reward -= MISSED_BREAKDOWN_HOLD_PENALTY

            if strong_long_setup or strong_short_setup:
                reward -= FLAT_HOLD_STRONG_SETUP_PENALTY

            # If the action was masked into HOLD, keep the penalty tiny because
            # the policy attempted action but gating rejected it.
            if mask_reason:
                reward += GOOD_ENTRY_BONUS * 0.10

            if abs(price / max(refs["vwap"], 0.01) - 1.0) < 0.003:
                reward -= 0.002
            if abs(price / max(refs["orb_high"], 0.01) - 1.0) < 0.003:
                reward -= 0.003
            if abs(price / max(refs["orb_low"], 0.01) - 1.0) < 0.003:
                reward -= 0.002

            reward += self._missed_opportunity_penalty(0, row, price)
            return reward

        unrealized_pct = self._unrealized_pct(price)
        reward = unrealized_pct * 0.08
        reward += self._position_pnl_reward(price, row)

        if self.position_side == 1:
            # Reward staying with profitable continuation instead of immediately
            # converting every small winner into a scalp.
            if unrealized_pct > 0:
                reward += min(unrealized_pct * UNREALIZED_WINNER_HOLD_MULT, 4.0)
                reward += min(self.hold_bars / max(LONG_HOLD_TARGET_BARS, 1.0), 1.0) * AVG_HOLD_REWARD_MULT

            future_10 = max(0.0, to_float(row.get("future_return_10"), 0.0))
            future_20 = max(0.0, to_float(row.get("future_return_20"), 0.0))
            future_30 = max(0.0, to_float(row.get("future_return_30"), 0.0))
            continuation_return = max(future_10, future_20, future_30)
            reward += min(continuation_return * CONTINUATION_HOLD_MULT, CONTINUATION_HOLD_CAP)

            trend_persistence = (
                to_float(row.get("bullish_structure_score"), 0.0)
                + to_float(row.get("breakout_persistence_score"), 0.0)
                + to_float(row.get("positive_volume_expansion_score"), 0.0)
                + max(0.0, to_float(row.get("ema_spread"), 0.0) * 100.0)
            ) / 4.0
            reward += max(0.0, trend_persistence) * min(self.hold_bars, 20) * TREND_HOLD_TIME_MULT

            bullish_structure = to_float(row.get("bullish_structure_score"), 0.0)
            if bullish_structure > 0.60:
                reward += HOLD_TREND_PERSISTENCE_REWARD
            if to_float(row.get("ema_spread"), 0.0) > 0:
                reward += 0.010
            if to_float(row.get("ret_5"), 0.0) > 0:
                reward += 0.010
            if to_float(row.get("breakout_persistence_score"), 0.0) > 0:
                reward += HOLD_BREAKOUT_PERSISTENCE_REWARD
            if to_float(row.get("higher_high_score"), 0.0) > 0:
                reward += 0.003
            if to_float(row.get("higher_low_score"), 0.0) > 0:
                reward += 0.003
            if to_float(row.get("positive_volume_expansion_score"), 0.0) > 0:
                reward += 0.004

            # Multibar clean continuation: bonus for holding through N+ bars of positive price action.
            reward += self._multibar_clean_continuation_bonus(price, row)

        if self.position_side == -1:
            reward -= 0.003 * to_float(row.get("bullish_acceleration_score"), 0.0)
            if to_float(row.get("bearish_acceleration_score"), 0.0) > 0.60:
                reward += 0.006
            if to_float(row.get("breakdown_persistence_score"), 0.0) > 0:
                reward += 0.006

        return reward

    def _log_trade(self, row: pd.Series, action: str, price: float, reward: float, reason: str) -> None:
        self.trade_events.append(
            TradeEvent(
                timestamp=str(row["timestamp"]),
                symbol=self.symbol,
                action=action,
                price=float(price),
                cash=float(self.cash),
                equity=float(self._equity(price)),
                position_qty=float(self.position_qty),
                realized_pnl=float(self.realized_pnl),
                reward=float(reward),
                reason=reason,
                hold_bars=int(self.hold_bars),
                unrealized_pct=float(self._unrealized_pct(price)),
                mfe_pct=float(max(self.max_favorable_excursion, 0.0)),
                mae_pct=float(abs(min(self.max_adverse_excursion, 0.0))),
                giveback_from_mfe_pct=float(self._current_giveback_from_mfe(price)),
                capture_ratio=float(self._mfe_mae_capture_stats(price, row, self.position_side).get("capture_ratio", 0.0)) if self.position_side != 0 else 0.0,
            )
        )


# =============================================================================
# Training utilities
# =============================================================================

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.snapshots: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            self.snapshots.append({"step": int(self.n_calls), "num_timesteps": int(self.num_timesteps)})
        return True


def make_episodes(
    bars: pd.DataFrame,
    structure: dict[str, dict[str, float]],
) -> tuple[list[pd.DataFrame], list[str]]:
    bars = add_bar_features(bars)
    _, _, overlap = write_symbol_overlap_report(bars, structure)

    if not overlap:
        raise RuntimeError(
            "No overlap between intraday bars symbols and intraday_structure_map symbols.\n"
            f"Overlap report written to: {OVERLAP_PATH}\n"
            "Rebuild intraday_bars_1m.csv using the same symbols from intraday_structure_map.json."
        )

    symbols = overlap[:TRAIN_SYMBOL_LIMIT]
    bars = bars[bars["symbol"].isin(symbols)].copy()
    bars["date"] = bars["timestamp"].dt.date.astype(str)

    episodes: list[pd.DataFrame] = []

    for (_, _), group in bars.groupby(["symbol", "date"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) >= MIN_BARS_PER_EPISODE:
            episodes.append(group)

    if not episodes:
        raise RuntimeError(
            f"No usable symbol/day episodes with at least {MIN_BARS_PER_EPISODE} bars. "
            "Download more intraday bars or reduce RL_INTRADAY_MIN_BARS_PER_EPISODE."
        )

    symbols_used = sorted(set(str(ep["symbol"].iloc[0]) for ep in episodes))
    print(f"[episodes] episodes={len(episodes)} symbols_used={len(symbols_used)}")
    return episodes, symbols_used


def build_policy_env(episodes: list[pd.DataFrame], structure: dict[str, dict[str, float]]) -> DummyVecEnv:
    def env_factory():
        return IntradayStructureEnv(episodes, structure)

    return DummyVecEnv([env_factory])


def create_model(env: DummyVecEnv):
    if USE_LSTM_POLICY:
        if RecurrentPPO is None:
            print("[warn] RL_POLICY_MODE requested LSTM, but sb3-contrib is not installed. Falling back to PPO.")
        else:
            print("[model] using RecurrentPPO MlpLstmPolicy")
            return RecurrentPPO(
                policy="MlpLstmPolicy",
                env=env,
                learning_rate=PPO_LEARNING_RATE,
                n_steps=PPO_N_STEPS,
                batch_size=PPO_BATCH_SIZE,
                gamma=PPO_GAMMA,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=PPO_ENT_COEF,
                verbose=1,
                seed=RANDOM_SEED,
            )

    print("[model] using PPO MlpPolicy")
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=PPO_ENT_COEF,
        verbose=1,
        seed=RANDOM_SEED,
    )


def evaluate_policy(
    model,
    episodes: list[pd.DataFrame],
    structure: dict[str, dict[str, float]],
    max_episodes: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for episode in episodes[:max_episodes]:
        env = IntradayStructureEnv([episode], structure)
        obs, _ = env.reset()

        done = False
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while not done:
            if USE_LSTM_POLICY and RecurrentPPO is not None and isinstance(model, RecurrentPPO):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start = np.array([False])
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, done, _, _ = env.step(int(action))

        for event in env.trade_events:
            rows.append(asdict(event))

    return pd.DataFrame(rows)


def trade_quality_metrics(diagnostics: pd.DataFrame) -> dict[str, Any]:
    if diagnostics.empty:
        return {
            "round_trips": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_reward": 0.0,
            "avg_hold_bars": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_giveback_from_mfe_pct": 0.0,
            "avg_capture_ratio": 0.0,
            "short_entry_ratio": 0.0,
            "long_entries": 0,
            "add_longs": 0,
            "add_long_ratio": 0.0,
            "avg_add_long_reward": 0.0,
            "short_entries": 0,
        }

    exits = diagnostics[diagnostics["action"].isin(["EXIT_LONG", "EXIT_SHORT"])].copy()
    if exits.empty:
        return {
            "round_trips": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_reward": 0.0,
            "avg_hold_bars": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_giveback_from_mfe_pct": 0.0,
            "avg_capture_ratio": 0.0,
            "short_entry_ratio": 0.0,
            "long_entries": int((diagnostics["action"] == "ENTER_LONG").sum()),
            "add_longs": int((diagnostics["action"] == "ADD_LONG").sum()),
            "add_long_ratio": 0.0,
            "avg_add_long_reward": 0.0,
            "short_entries": int((diagnostics["action"] == "ENTER_SHORT").sum()),
        }

    pnl_proxy = exits["reward"].astype(float)
    wins = pnl_proxy[pnl_proxy > 0]
    losses = pnl_proxy[pnl_proxy < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    long_entries = int((diagnostics["action"] == "ENTER_LONG").sum())
    add_longs = int((diagnostics["action"] == "ADD_LONG").sum())
    short_entries = int((diagnostics["action"] == "ENTER_SHORT").sum())
    total_entries = max(long_entries + short_entries, 1)
    add_long_rows = diagnostics[diagnostics["action"] == "ADD_LONG"]
    avg_add_long_reward = float(add_long_rows["reward"].astype(float).mean()) if not add_long_rows.empty else 0.0
    add_long_ratio = float(add_longs / max(long_entries, 1))

    return {
        "round_trips": int(len(exits)),
        "win_rate": float((pnl_proxy > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0),
        "avg_pnl_reward": float(pnl_proxy.mean()),
        "avg_hold_bars": float(exits["hold_bars"].astype(float).mean()) if "hold_bars" in exits.columns else 0.0,
        "avg_mfe_pct": float(exits["mfe_pct"].astype(float).mean()) if "mfe_pct" in exits.columns else 0.0,
        "avg_mae_pct": float(exits["mae_pct"].astype(float).mean()) if "mae_pct" in exits.columns else 0.0,
        "avg_giveback_from_mfe_pct": float(exits["giveback_from_mfe_pct"].astype(float).mean()) if "giveback_from_mfe_pct" in exits.columns else 0.0,
        "avg_capture_ratio": float(exits["capture_ratio"].astype(float).mean()) if "capture_ratio" in exits.columns else 0.0,
        "long_entries": long_entries,
        "add_longs": add_longs,
        "add_long_ratio": add_long_ratio,
        "avg_add_long_reward": avg_add_long_reward,
        "short_entries": short_entries,
        "long_exits": int((diagnostics["action"] == "EXIT_LONG").sum()),
        "short_exits": int((diagnostics["action"] == "EXIT_SHORT").sum()),
        "short_entry_ratio": float(short_entries / total_entries),
    }


def train_model(
    bars: pd.DataFrame,
    raw_structure_map: dict[str, Any],
    bars_path: Path,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    structure = {
        normalize_symbol(symbol): flatten_structure(symbol_map)
        for symbol, symbol_map in raw_structure_map.items()
        if normalize_symbol(symbol) and isinstance(symbol_map, dict)
    }

    if not structure:
        raise RuntimeError("Structure map loaded, but no usable symbol structures were found.")

    episodes, symbols_used = make_episodes(bars, structure)

    # Chronological train/test split — never shuffle.
    n_total = len(episodes)
    n_train = max(1, int(n_total * RL_TRAIN_SPLIT_PCT))
    train_episodes = episodes[:n_train]
    test_episodes = episodes[n_train:]
    n_test = len(test_episodes)
    print(f"[train] episodes total={n_total} train={n_train} test={n_test} (split={RL_TRAIN_SPLIT_PCT:.0%})")

    env = build_policy_env(train_episodes, structure)

    probe_env = IntradayStructureEnv(train_episodes[:1], structure)
    feature_count = len(probe_env.feature_names)

    model = create_model(env)

    callback = MetricsCallback()
    print("[train] fitting policy on train set...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=callback)
    model.save(str(MODEL_PATH))

    # In-sample diagnostics on train set.
    diagnostics = evaluate_policy(model, train_episodes, structure, max_episodes=min(100, n_train))
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)

    # Held-out evaluation on test set.
    test_diagnostics = pd.DataFrame()
    test_metrics: dict[str, Any] = {}
    production_ready = False
    if n_test >= RL_MIN_TEST_EPISODES or RL_ALLOW_SMALL_TEST_SET:
        print(f"[train] evaluating on {n_test} held-out test episodes...")
        test_diagnostics = evaluate_policy(model, test_episodes, structure, max_episodes=n_test)
        tq = trade_quality_metrics(test_diagnostics)
        # Compute test win rate and avg trade return from trade events.
        test_win_rate = 0.0
        test_avg_return = 0.0
        test_max_dd = 0.0
        action_dist: dict[str, int] = {}
        if not test_diagnostics.empty:
            if "action" in test_diagnostics.columns:
                for act in test_diagnostics["action"].dropna():
                    action_dist[str(act)] = action_dist.get(str(act), 0) + 1
            exits = test_diagnostics[test_diagnostics.get("action", pd.Series(dtype=str)) == "EXIT_LONG"] if "action" in test_diagnostics.columns else pd.DataFrame()
            if not exits.empty and "realized_pnl" in exits.columns:
                pnls = pd.to_numeric(exits["realized_pnl"], errors="coerce").dropna()
                test_win_rate = float((pnls > 0).mean()) if len(pnls) > 0 else 0.0
                test_avg_return = float(pnls.mean()) if len(pnls) > 0 else 0.0
            if "equity" in test_diagnostics.columns:
                eq = pd.to_numeric(test_diagnostics["equity"], errors="coerce").dropna()
                if len(eq) > 1:
                    peak = eq.cummax()
                    dd = (eq / peak - 1.0)
                    test_max_dd = float(dd.min())
        test_metrics = {
            "train_episode_count": n_train,
            "test_episode_count": n_test,
            "train_reward_mean": float(np.mean([s.get("mean_reward", 0.0) for s in callback.snapshots]) if callback.snapshots else 0.0),
            "test_reward_mean": float(tq.get("avg_pnl_reward", 0.0)),
            "test_win_rate": test_win_rate,
            "test_avg_trade_return": test_avg_return,
            "test_max_drawdown": test_max_dd,
            "test_action_distribution": action_dist,
        }
        production_ready = (
            n_test >= RL_MIN_TEST_EPISODES
            and test_win_rate >= 0.45
            and test_max_dd > -0.30
        )
        print(
            f"[train] test metrics: win_rate={test_win_rate:.1%} avg_return={test_avg_return:.4f} "
            f"max_dd={test_max_dd:.1%} production_ready={production_ready}"
        )
    else:
        print(f"[train] WARNING: only {n_test} test episodes < RL_MIN_TEST_EPISODES={RL_MIN_TEST_EPISODES}; "
              f"skipping held-out evaluation. Set RL_ALLOW_SMALL_TEST_SET=1 to override.")
        test_metrics = {
            "train_episode_count": n_train,
            "test_episode_count": n_test,
            "production_ready_skipped": True,
            "reason": f"insufficient_test_episodes:{n_test}<{RL_MIN_TEST_EPISODES}",
        }

    test_metrics["production_ready"] = production_ready

    metrics = {
        "created_at_utc": now_utc(),
        "model_path": str(MODEL_PATH),
        "bars_rows": int(len(bars)),
        "episode_count": int(len(episodes)),
        "train_episode_count": n_train,
        "test_episode_count": n_test,
        "symbols_used": symbols_used,
        "feature_count": int(feature_count),
        "held_out_evaluation": test_metrics,
        "diagnostics": {
            "rows": int(len(diagnostics)),
            "enter_long_count": int((diagnostics["action"] == "ENTER_LONG").sum()) if not diagnostics.empty else 0,
            "add_long_count": int((diagnostics["action"] == "ADD_LONG").sum()) if not diagnostics.empty else 0,
            "exit_long_count": int((diagnostics["action"] == "EXIT_LONG").sum()) if not diagnostics.empty else 0,
            "enter_short_count": int((diagnostics["action"] == "ENTER_SHORT").sum()) if not diagnostics.empty else 0,
            "exit_short_count": int((diagnostics["action"] == "EXIT_SHORT").sum()) if not diagnostics.empty else 0,
            "total_realized_pnl": float(diagnostics["realized_pnl"].max()) if not diagnostics.empty else 0.0,
            "trade_quality": trade_quality_metrics(diagnostics),
        },
        "callback_snapshots": callback.snapshots,
        "peak_exit_config": {
            "lookahead_bars": PEAK_EXIT_LOOKAHEAD_BARS,
            "min_mfe": PEAK_EXIT_MIN_MFE,
            "giveback_trigger": PEAK_EXIT_GIVEBACK_TRIGGER,
            "efficiency_bonus": PEAK_EXIT_EFFICIENCY_BONUS,
            "near_top_bonus": PEAK_EXIT_NEAR_TOP_BONUS,
            "missed_upside_penalty_mult": PEAK_EXIT_MISSED_UPSIDE_PENALTY_MULT,
            "missed_upside_cap": PEAK_EXIT_MISSED_UPSIDE_CAP,
            "giveback_avoid_bonus_mult": PEAK_EXIT_GIVEBACK_AVOID_BONUS_MULT,
            "late_giveback_penalty_mult": PEAK_EXIT_LATE_GIVEBACK_PENALTY_MULT,
            "force_exit_on_confirmed_giveback": FORCE_EXIT_ON_CONFIRMED_GIVEBACK,
        },
        "target_metrics": {
            "exit_efficiency_min": TARGET_EXIT_EFFICIENCY,
            "missed_upside_max": TARGET_MAX_MISSED_UPSIDE,
            "giveback_captured_max": TARGET_MAX_GIVEBACK_CAPTURED,
            "avg_hold_bars_min": TARGET_MIN_AVG_HOLD_BARS,
        },
        "trajectory_reward_config": {
            "use_fixed_take_profit": USE_FIXED_TAKE_PROFIT,
            "use_dynamic_giveback_exit": USE_DYNAMIC_GIVEBACK_EXIT,
            "dynamic_giveback_min_mfe": DYNAMIC_GIVEBACK_MIN_MFE,
            "dynamic_giveback_fraction": DYNAMIC_GIVEBACK_FRACTION,
            "dynamic_giveback_floor": DYNAMIC_GIVEBACK_FLOOR,
            "mfe_capture_reward_mult": MFE_CAPTURE_REWARD_MULT,
            "mae_penalty_mult": MAE_PENALTY_MULT,
            "giveback_capture_penalty_mult": GIVEBACK_CAPTURE_PENALTY_MULT,
            "long_hold_target_bars": LONG_HOLD_TARGET_BARS,
            "avg_hold_reward_mult": AVG_HOLD_REWARD_MULT,
            "early_winner_exit_penalty": EARLY_WINNER_EXIT_PENALTY,
            "quality_winner_hold_bonus": QUALITY_WINNER_HOLD_BONUS,
            "unrealized_winner_hold_mult": UNREALIZED_WINNER_HOLD_MULT,
            "continuation_hold_mult": CONTINUATION_HOLD_MULT,
            "add_long_enabled": ADD_LONG_ENABLED,
            "add_long_fraction": ADD_LONG_FRACTION,
            "add_long_min_unrealized_pct": ADD_LONG_MIN_UNREALIZED_PCT,
            "add_long_min_hold_bars": ADD_LONG_MIN_HOLD_BARS,
            "add_long_max_count": ADD_LONG_MAX_COUNT,
            "add_long_reward": ADD_LONG_REWARD,
            "add_long_future_reward_mult": ADD_LONG_FUTURE_REWARD_MULT,
            "add_long_future_reward_cap": ADD_LONG_FUTURE_REWARD_CAP,
            "bad_add_long_penalty": BAD_ADD_LONG_PENALTY,
            "add_long_min_continuation_score": ADD_LONG_MIN_CONTINUATION_SCORE,
            "short_churn_penalty": SHORT_CHURN_PENALTY,
            "long_short_balance_penalty": LONG_SHORT_BALANCE_PENALTY,
        },
        "exploration_config": {
            "force_exploration_actions": FORCE_EXPLORATION_ACTIONS,
            "force_exploration_episode_frac": FORCE_EXPLORATION_EPISODE_FRAC,
            "force_exploration_prob": FORCE_EXPLORATION_PROB,
            "min_trades_per_episode": MIN_TRADES_PER_EPISODE,
            "no_trade_episode_penalty": NO_TRADE_EPISODE_PENALTY,
            "flat_hold_step_penalty": FLAT_HOLD_STEP_PENALTY,
            "valid_entry_reward": VALID_ENTRY_REWARD,
            "valid_short_reward": VALID_SHORT_REWARD,
            "first_trade_bonus": FIRST_TRADE_BONUS,
        },
        "config": asdict(
            TrainingConfig(
                bars_path=str(bars_path),
                structure_map_path=str(STRUCTURE_MAP_PATH),
                model_path=str(MODEL_PATH),
                train_timesteps=TRAIN_TIMESTEPS,
                train_symbol_limit=TRAIN_SYMBOL_LIMIT,
                max_episode_steps=MAX_EPISODE_STEPS,
                min_bars_per_episode=MIN_BARS_PER_EPISODE,
                starting_cash=STARTING_CASH,
                position_fraction=POSITION_FRACTION,
                commission_bps=COMMISSION_BPS,
                slippage_bps=SLIPPAGE_BPS,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
                max_hold_bars=MAX_HOLD_BARS,
                policy_mode=RL_POLICY_MODE,
                use_lstm_policy=USE_LSTM_POLICY and RecurrentPPO is not None,
                use_transformer_features=USE_TRANSFORMER_FEATURES,
                use_hierarchical_gating=USE_HIERARCHICAL_GATING,
                use_multi_timeframe_fusion=USE_MULTI_TIMEFRAME_FUSION,
                action_masking_enabled=ACTION_MASKING_ENABLED,
                require_long_above_vwap=REQUIRE_LONG_ABOVE_VWAP,
                require_long_above_orb=REQUIRE_LONG_ABOVE_ORB,
                require_short_below_vwap=REQUIRE_SHORT_BELOW_VWAP,
                require_short_below_trigger=REQUIRE_SHORT_BELOW_TRIGGER,
                disallow_exit_when_flat=DISALLOW_EXIT_WHEN_FLAT,
                reentry_cooldown_bars=REENTRY_COOLDOWN_BARS,
                disallow_bearish_longs=DISALLOW_BEARISH_LONGS,
                disallow_shorts_when_not_allowed=DISALLOW_SHORTS_WHEN_NOT_ALLOWED,
                soft_vwap_trend_threshold=SOFT_VWAP_TREND_THRESHOLD,
                soft_orb_breakout_threshold=SOFT_ORB_BREAKOUT_THRESHOLD,
                soft_short_trend_threshold=SOFT_SHORT_TREND_THRESHOLD,
                soft_breakdown_threshold=SOFT_BREAKDOWN_THRESHOLD,
                ppo_ent_coef=PPO_ENT_COEF,
            )
        ),
    }

    write_json(METRICS_PATH, metrics)
    return model, metrics, diagnostics


# =============================================================================
# Live inference helper for run.py
# =============================================================================

def _action_distribution(model, obs: np.ndarray) -> tuple[int, dict[str, float], float]:
    if USE_LSTM_POLICY and RecurrentPPO is not None and isinstance(model, RecurrentPPO):
        action_id, _ = model.predict(obs, deterministic=True)
        probs = np.zeros(len(ACTION_NAMES), dtype=float)
        probs[int(action_id)] = 1.0
        return int(action_id), {ACTION_NAMES[i]: float(probs[i]) for i in range(len(probs))}, 0.0

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy().flatten()

    probs = np.nan_to_num(probs, nan=0.0)
    if probs.sum() <= 0:
        probs = np.ones(len(ACTION_NAMES), dtype=float) / len(ACTION_NAMES)
    else:
        probs = probs / probs.sum()

    action_id = int(np.argmax(probs))
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())

    return action_id, {ACTION_NAMES[i]: float(probs[i]) for i in range(len(probs))}, entropy


def _load_model(model_path: Path):
    if USE_LSTM_POLICY and RecurrentPPO is not None:
        try:
            return RecurrentPPO.load(str(model_path))
        except Exception:
            pass
    return PPO.load(str(model_path))


def predict_live_action(
    model_path: Path,
    latest_bars: pd.DataFrame,
    structure_map: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)

    if not model_path.exists():
        return {"symbol": symbol, "action": "HOLD", "action_id": 0, "confidence": 0.0, "reason": "model_not_found"}

    model = _load_model(model_path)

    structure = {
        normalize_symbol(k): flatten_structure(v)
        for k, v in structure_map.items()
        if normalize_symbol(k) and isinstance(v, dict)
    }

    bars = add_bar_features(latest_bars.copy())
    bars["symbol"] = bars["symbol"].map(normalize_symbol)
    sdf = bars[bars["symbol"] == symbol].copy()

    if sdf.empty:
        return {"symbol": symbol, "action": "HOLD", "action_id": 0, "confidence": 0.0, "reason": "no_latest_bars"}

    env = IntradayStructureEnv([sdf.tail(MAX_EPISODE_STEPS)], structure)
    obs, _ = env.reset()
    env.idx = max(0, len(env.episode_df) - 1)
    obs = env._obs()

    raw_action_id, probabilities, entropy = _action_distribution(model, obs)

    latest_row = env.episode_df.iloc[env.idx]
    price = max(to_float(latest_row["close"]), 0.01)
    masked_action_id, mask_reason = env._apply_action_mask(raw_action_id, price, latest_row)

    confidence = probabilities.get(ACTION_NAMES.get(raw_action_id, "HOLD"), 0.0)
    action_name = ACTION_NAMES.get(masked_action_id, "HOLD")

    return {
        "symbol": symbol,
        "action": action_name,
        "action_id": masked_action_id,
        "raw_action_id": raw_action_id,
        "raw_action": ACTION_NAMES.get(raw_action_id, "HOLD"),
        "masked": masked_action_id != raw_action_id,
        "mask_reason": mask_reason,
        "confidence": float(confidence),
        "entropy": float(entropy),
        "action_probabilities": probabilities,
        "policy_mode": RL_POLICY_MODE,
        "reason": "rl_intraday_structure_policy",
        "model_path": str(model_path),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    bars_path = find_bars_path()

    print("===== RL INTRADAY STRUCTURE TRAINER =====")
    print(f"[load] bars={bars_path}")
    print(f"[load] structure_map={STRUCTURE_MAP_PATH}")
    print(f"[config] policy_mode={RL_POLICY_MODE}")
    print(f"[config] use_lstm_policy={USE_LSTM_POLICY and RecurrentPPO is not None}")
    print(f"[config] use_transformer_features={USE_TRANSFORMER_FEATURES}")
    print(f"[config] use_hierarchical_gating={USE_HIERARCHICAL_GATING}")
    print(f"[config] use_multi_timeframe_fusion={USE_MULTI_TIMEFRAME_FUSION}")
    print(f"[config] timesteps={TRAIN_TIMESTEPS:,}")
    print(f"[config] symbol_limit={TRAIN_SYMBOL_LIMIT}")
    print(f"[config] min_bars_per_episode={MIN_BARS_PER_EPISODE}")
    print(f"[config] action_masking={ACTION_MASKING_ENABLED}")
    print(f"[config] reentry_cooldown_bars={REENTRY_COOLDOWN_BARS}")
    print(f"[config] ppo_ent_coef={PPO_ENT_COEF}")
    print(f"[config] debug_masks={DEBUG_MASKS}")
    print(f"[config] peak_exit_lookahead_bars={PEAK_EXIT_LOOKAHEAD_BARS}")
    print(f"[config] peak_exit_min_mfe={PEAK_EXIT_MIN_MFE}")
    print(f"[config] peak_exit_giveback_trigger={PEAK_EXIT_GIVEBACK_TRIGGER}")
    print(f"[config] force_exit_on_confirmed_giveback={FORCE_EXIT_ON_CONFIRMED_GIVEBACK}")
    print(f"[actions] {ACTION_NAMES}")

    if USE_LSTM_POLICY and RecurrentPPO is None:
        print("[warn] sb3-contrib is not installed; LSTM mode will fall back to standard PPO.")
        print("[hint] pip install sb3-contrib")

    bars = load_bars(bars_path)
    structure = load_structure_map(STRUCTURE_MAP_PATH)

    _, metrics, diagnostics = train_model(bars=bars, raw_structure_map=structure, bars_path=bars_path)

    print(f"[saved] model={MODEL_PATH}")
    print(f"[saved] metrics={METRICS_PATH}")
    print(f"[saved] diagnostics={DIAGNOSTICS_PATH}")
    print(f"[saved] overlap={OVERLAP_PATH}")

    diag = metrics["diagnostics"]

    print("\n===== SUMMARY =====")
    print(f"bars_rows={metrics['bars_rows']:,}")
    print(f"episode_count={metrics['episode_count']:,}")
    print(f"feature_count={metrics['feature_count']:,}")
    print(f"symbols_used={len(metrics['symbols_used'])}")
    print(f"diagnostic_events={len(diagnostics):,}")
    print(f"enter_longs={diag['enter_long_count']:,}")
    print(f"exit_longs={diag['exit_long_count']:,}")
    print(f"diagnostic_pnl={diag['total_realized_pnl']:.2f}")
    print(f"win_rate={diag['trade_quality'].get('win_rate', 0.0):.2%}")
    print(f"profit_factor={diag['trade_quality'].get('profit_factor', 0.0):.2f}")


if __name__ == "__main__":
    main()
