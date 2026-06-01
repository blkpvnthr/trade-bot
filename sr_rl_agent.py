"""
sr_rl_agent.py — QuantumQuant Support/Resistance RL Agent
Explicitly recognises:
  • support_bounce
  • vwap_reclaim
  • liquidity_sweep_reclaim
  • missed_reversal_reclaim
  • breakout_retest_continuation

Uses the support/resistance maps from PivotReversalScanner.
Tracks dedicated metrics: dip_reclaim_entries, dip_reclaim_win_rate,
support_reclaim_entries, support_reclaim_win_rate, avg_mfe_pct,
avg_mae_pct, avg_capture_ratio, profit_factor, missed_dip_reclaim_count.

CLI:  python sr_rl_agent.py --train
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR_RL_TIMESTEPS   = int(os.environ.get("SR_RL_TIMESTEPS",  500_000))
SR_SAVE_PATH      = os.environ.get("SR_SAVE_PATH",         "models/sr_rl")
SR_METRICS_PATH   = os.environ.get("SR_METRICS_PATH",      "metrics/sr_rl_metrics.json")
SR_REPLAY_PATH    = os.environ.get("SR_REPLAY_PATH",       "rl_trade_replay.csv")
SR_DATA_PATH      = os.environ.get("SR_DATA_PATH",         "data/intraday_bars.parquet")
SR_EVAL_FREQ      = int(os.environ.get("SR_EVAL_FREQ",     20_000))

RISK_REWARD_SR    = float(os.environ.get("SR_RR_TARGET",   3.0))
MAX_BARS_SR       = int(os.environ.get("SR_MAX_BARS",      60))

# Reward weights (overridable)
SR_R_SUPPORT_BOUNCE   = float(os.environ.get("SR_R_SUPPORT_BOUNCE",  0.30))
SR_R_VWAP_RECLAIM     = float(os.environ.get("SR_R_VWAP_RECLAIM",    0.25))
SR_R_SWEEP_RECLAIM    = float(os.environ.get("SR_R_SWEEP_RECLAIM",   0.35))
SR_R_MISSED_REV       = float(os.environ.get("SR_R_MISSED_REV",      0.30))
SR_R_BK_RETEST        = float(os.environ.get("SR_R_BK_RETEST",       0.25))
SR_R_MFE_1R           = float(os.environ.get("SR_R_MFE_1R",          0.20))
SR_R_MFE_2R           = float(os.environ.get("SR_R_MFE_2R",          0.40))
SR_R_MFE_3R           = float(os.environ.get("SR_R_MFE_3R",          0.80))
SR_P_NO_SETUP         = float(os.environ.get("SR_P_NO_SETUP",       -0.25))
SR_P_INTO_RES         = float(os.environ.get("SR_P_INTO_RES",       -0.35))
SR_P_FAILED           = float(os.environ.get("SR_P_FAILED",         -0.40))
SR_P_MISSED           = float(os.environ.get("SR_P_MISSED",         -0.15))

REPLAY_FIELDNAMES = [
    "timestamp", "symbol", "setup_type",
    "entry_price", "stop_price", "target_price",
    "sweep_low", "reclaim_price",
    "volume_spike_ratio", "support_distance_pct",
    "reclaim_vwap", "reclaim_ema9", "reclaim_ema20",
    "mfe_pct", "mae_pct", "outcome",
]


# ---------------------------------------------------------------------------
# SR setup types
# ---------------------------------------------------------------------------

class SRSetupType(str, Enum):
    SUPPORT_BOUNCE         = "support_bounce"
    VWAP_RECLAIM           = "vwap_reclaim"
    LIQUIDITY_SWEEP_RECLAIM= "liquidity_sweep_reclaim"
    MISSED_REVERSAL        = "missed_reversal_reclaim"
    BREAKOUT_RETEST        = "breakout_retest_continuation"
    NONE                   = "none"


def detect_sr_setup(bar: Dict, prev_bar: Optional[Dict] = None) -> SRSetupType:
    cl     = float(bar.get("close",  0.0))
    o      = float(bar.get("open",   cl))
    l      = float(bar.get("low",    cl))
    vwap   = float(bar.get("vwap",   cl))
    sup    = float(bar.get("support_zone", 0.0))
    res    = float(bar.get("resistance_zone", cl * 1.05))
    liq    = int(bar.get("liquidity_sweep_detected", 0))
    piv    = int(bar.get("pivot_low_reclaim", 0))
    sbf    = int(bar.get("support_break_failed", 0))
    mrr    = int(bar.get("missed_reversal_level_reclaim", 0))
    bkr    = int(bar.get("breakout_retest", 0))

    prev_cl = float(prev_bar.get("close", cl)) if prev_bar else cl

    if liq and (piv or sbf):
        return SRSetupType.LIQUIDITY_SWEEP_RECLAIM
    if mrr:
        return SRSetupType.MISSED_REVERSAL
    if bkr:
        return SRSetupType.BREAKOUT_RETEST
    if piv or sbf:
        return SRSetupType.SUPPORT_BOUNCE
    if prev_cl < vwap <= cl:
        return SRSetupType.VWAP_RECLAIM
    return SRSetupType.NONE


# ---------------------------------------------------------------------------
# Episode metrics tracker
# ---------------------------------------------------------------------------

@dataclass
class SRMetrics:
    n_trades:                   int   = 0
    wins:                       int   = 0
    dip_reclaim_entries:        int   = 0
    dip_reclaim_wins:           int   = 0
    support_reclaim_entries:    int   = 0
    support_reclaim_wins:       int   = 0
    vwap_reclaim_entries:       int   = 0
    vwap_reclaim_wins:          int   = 0
    sweep_reclaim_entries:      int   = 0
    sweep_reclaim_wins:         int   = 0
    missed_dip_reclaim_count:   int   = 0
    total_mfe:                  float = 0.0
    total_mae:                  float = 0.0
    gross_win:                  float = 0.0
    gross_loss:                 float = 0.0

    def record(self, setup: SRSetupType, outcome: str, mfe: float, mae: float):
        self.n_trades   += 1
        win = "win" in outcome
        if win:
            self.wins       += 1
            self.gross_win  += mfe
        else:
            self.gross_loss += mae
        self.total_mfe += mfe
        self.total_mae += mae

        if setup in (SRSetupType.LIQUIDITY_SWEEP_RECLAIM,):
            self.sweep_reclaim_entries += 1
            if win: self.sweep_reclaim_wins += 1
            self.dip_reclaim_entries   += 1
            if win: self.dip_reclaim_wins   += 1
        elif setup == SRSetupType.SUPPORT_BOUNCE:
            self.support_reclaim_entries += 1
            if win: self.support_reclaim_wins += 1
            self.dip_reclaim_entries   += 1
            if win: self.dip_reclaim_wins   += 1
        elif setup == SRSetupType.VWAP_RECLAIM:
            self.vwap_reclaim_entries += 1
            if win: self.vwap_reclaim_wins += 1

    def to_dict(self) -> Dict:
        n = max(self.n_trades, 1)
        return dict(
            n_trades                   = self.n_trades,
            win_rate                   = self.wins / n,
            profit_factor              = (self.gross_win + 1e-8) / (self.gross_loss + 1e-8),
            avg_mfe_pct                = self.total_mfe / n,
            avg_mae_pct                = self.total_mae / n,
            avg_capture_ratio          = sum(
                [self.total_mfe / (self.total_mfe + self.total_mae + 1e-8)]
            ),
            dip_reclaim_entries        = self.dip_reclaim_entries,
            dip_reclaim_win_rate       = self.dip_reclaim_wins / max(self.dip_reclaim_entries, 1),
            support_reclaim_entries    = self.support_reclaim_entries,
            support_reclaim_win_rate   = self.support_reclaim_wins / max(self.support_reclaim_entries, 1),
            vwap_reclaim_entries       = self.vwap_reclaim_entries,
            sweep_reclaim_entries      = self.sweep_reclaim_entries,
            missed_dip_reclaim_count   = self.missed_dip_reclaim_count,
        )


# ---------------------------------------------------------------------------
# Gymnasium Environment (SR-focused)
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    gym = None

from rl_agent_core import (
    N_CHANNELS, CNN_LOOKBACK, ACTION_HOLD, ACTION_ENTER_LONG, ACTION_EXIT_LONG,
    PrivateMarketStateBuilder, SharpeRewardCalculator,
    make_observation_space, make_action_space, get_policy_kwargs,
)
from pivot_reversal_scanner import PivotReversalScanner


if SB3_AVAILABLE:
    class SRRLEnv(gym.Env):
        """
        SR-RL environment. Identical observation space to IntradayRLEnv
        but reward function is shaped around SR setup quality specifically.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            bars_df:        pd.DataFrame,
            symbol:         str   = "UNKNOWN",
            lookback:       int   = CNN_LOOKBACK,
            rr_target:      float = RISK_REWARD_SR,
            max_bars_trade: int   = MAX_BARS_SR,
            replay_path:    str   = SR_REPLAY_PATH,
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

            self._state   = PrivateMarketStateBuilder(lookback)
            self._sharpe  = SharpeRewardCalculator()
            self._scanner = PivotReversalScanner(symbol=symbol)

            self._metrics     = SRMetrics()
            self.ep_metrics: Dict = {}

            # Trade state
            self._in_pos        = False
            self._entry         = 0.0
            self._stop          = 0.0
            self._target        = 0.0
            self._bars_held     = 0
            self._mfe           = 0.0
            self._mae           = 0.0
            self._setup_type    = SRSetupType.NONE
            self._prev_cl       = 0.0
            self._prev_bar: Optional[Dict] = None

            # Session
            self._t          = 0
            self._day_open   = 0.0
            self._sess_high  = 0.0
            self._red_str    = 0
            self._grn_str    = 0

            # Replay CSV
            if not os.path.exists(self.replay_path):
                os.makedirs(
                    os.path.dirname(self.replay_path) if os.path.dirname(self.replay_path) else ".",
                    exist_ok=True,
                )
                with open(self.replay_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=REPLAY_FIELDNAMES).writeheader()

        # ------------------------------------------------------------------

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._state.reset()
            self._sharpe.reset()
            self._scanner  = PivotReversalScanner(symbol=self.symbol)
            self._metrics  = SRMetrics()

            self._in_pos  = False
            self._entry   = 0.0
            self._stop    = 0.0
            self._target  = 0.0
            self._bars_held = 0
            self._mfe     = 0.0
            self._mae     = 0.0
            self._setup_type = SRSetupType.NONE

            self._t         = self.lookback
            row             = self.bars_df.iloc[self._t]
            self._day_open  = float(row.get("open", row.get("close", 1.0)))
            self._sess_high = self._day_open
            self._prev_cl   = self._day_open
            self._red_str   = 0
            self._grn_str   = 0
            self._prev_bar  = None

            for i in range(max(0, self._t - self.lookback), self._t):
                self._push(i)

            obs = self._state.build()
            return obs, {}

        def step(self, action: int):
            row  = self.bars_df.iloc[self._t]
            bar  = self._make_bar(row)
            cl   = bar["close"]
            h    = bar["high"]
            l    = bar["low"]
            setup = detect_sr_setup(bar, self._prev_bar)

            self._state.push(bar)

            reward     = 0.0
            terminated = False
            truncated  = False

            if self._in_pos:
                self._bars_held += 1
                up   = (h - self._entry) / (self._entry + 1e-8)
                down = (self._entry - l)  / (self._entry + 1e-8)
                self._mfe = max(self._mfe, up)
                self._mae = max(self._mae, down)

                risk    = (self._entry - self._stop) + 1e-8
                mfe_r   = self._mfe * self._entry / risk

                if l <= self._stop:
                    reward += SR_P_FAILED
                    reward += self._close("loss", cl, bar, setup)
                elif h >= self._target:
                    reward += SR_R_MFE_3R
                    reward += self._close("win", cl, bar, setup)
                elif self._bars_held >= self.max_bars_trade:
                    reward += self._close("timeout", cl, bar, setup)
                elif action == ACTION_EXIT_LONG:
                    # Agent chooses to exit
                    if mfe_r >= 2.0: reward += SR_R_MFE_2R
                    elif mfe_r >= 1.0: reward += SR_R_MFE_1R
                    reward += self._close("agent_exit", cl, bar, setup)
                else:
                    # Holding
                    if mfe_r >= 3.0:  reward += SR_R_MFE_3R * 0.05
                    elif mfe_r >= 1.0: reward += SR_R_MFE_1R * 0.05

            else:
                if action == ACTION_ENTER_LONG:
                    reward += self._entry_reward(bar, setup, cl)
                    if setup != SRSetupType.NONE:
                        self._open_pos(bar, cl)
                elif action == ACTION_HOLD:
                    # Penalise missing a clear setup
                    if setup in (
                        SRSetupType.LIQUIDITY_SWEEP_RECLAIM,
                        SRSetupType.SUPPORT_BOUNCE,
                        SRSetupType.MISSED_REVERSAL,
                    ):
                        reward += SR_P_MISSED
                        self._metrics.missed_dip_reclaim_count += 1

            # Sharpe
            step_ret = (cl - self._prev_cl) / (self._prev_cl + 1e-8) if self._in_pos else 0.0
            reward  += self._sharpe.update(step_ret) * 0.05

            self._prev_cl  = cl
            self._prev_bar = bar
            self._t       += 1

            if self._t >= len(self.bars_df) - 1:
                terminated = True
                if self._in_pos:
                    self._close("eod", cl, bar, setup)
                self.ep_metrics = self._metrics.to_dict()

            obs = self._state.build()
            info = {"setup_type": setup.value, "in_pos": self._in_pos,
                    "sharpe": self._sharpe.current_sharpe}
            return obs, reward, terminated, truncated, info

        # ------------------------------------------------------------------

        def _entry_reward(self, bar: Dict, setup: SRSetupType, cl: float) -> float:
            r   = 0.0
            sup = float(bar.get("support_zone", 0.0))
            res = float(bar.get("resistance_zone", cl * 1.1))
            vwap= float(bar.get("vwap", cl))
            vol = float(bar.get("volume", 0.0))
            vm  = float(bar.get("volume_ma20", vol)) + 1e-8

            if setup == SRSetupType.LIQUIDITY_SWEEP_RECLAIM: r += SR_R_SWEEP_RECLAIM
            elif setup == SRSetupType.SUPPORT_BOUNCE:        r += SR_R_SUPPORT_BOUNCE
            elif setup == SRSetupType.VWAP_RECLAIM:          r += SR_R_VWAP_RECLAIM
            elif setup == SRSetupType.MISSED_REVERSAL:       r += SR_R_MISSED_REV
            elif setup == SRSetupType.BREAKOUT_RETEST:       r += SR_R_BK_RETEST
            else:
                r += SR_P_NO_SETUP

            if res > 0 and (res - cl) / (cl + 1e-8) < 0.003:
                r += SR_P_INTO_RES

            return r

        def _open_pos(self, bar: Dict, cl: float):
            sess_l  = float(bar.get("session_low", cl * 0.98))
            stop    = max(sess_l * 0.999, cl * 0.99)
            risk    = max(cl - stop, cl * 0.005)
            self._entry       = cl
            self._stop        = stop
            self._target      = cl + risk * self.rr_target
            self._bars_held   = 0
            self._mfe         = 0.0
            self._mae         = 0.0
            self._in_pos      = True
            self._setup_type  = detect_sr_setup(bar, self._prev_bar)

        def _close(self, outcome: str, cl: float, bar: Dict, setup: SRSetupType) -> float:
            self._metrics.record(self._setup_type, outcome, self._mfe, self._mae)
            rec = dict(
                timestamp            = str(bar.get("timestamp", self._t)),
                symbol               = self.symbol,
                setup_type           = self._setup_type.value,
                entry_price          = self._entry,
                stop_price           = self._stop,
                target_price         = self._target,
                sweep_low            = float(bar.get("session_low", self._stop)),
                reclaim_price        = self._entry,
                volume_spike_ratio   = float(bar.get("volume_spike_ratio", 1.0)),
                support_distance_pct = (self._entry - float(bar.get("support_zone", self._entry)))
                                       / (self._entry + 1e-8),
                reclaim_vwap         = int(cl > float(bar.get("vwap", cl))),
                reclaim_ema9         = int(cl > float(bar.get("ema9", cl))),
                reclaim_ema20        = int(cl > float(bar.get("ema20", cl))),
                mfe_pct              = self._mfe,
                mae_pct              = self._mae,
                outcome              = outcome,
            )
            self._write_replay(rec)
            self._in_pos    = False
            self._entry     = 0.0
            self._stop      = 0.0
            self._target    = 0.0
            self._bars_held = 0
            self._mfe       = 0.0
            self._mae       = 0.0
            return 0.0

        def _write_replay(self, rec: Dict):
            try:
                with open(self.replay_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=REPLAY_FIELDNAMES)
                    w.writerow({k: rec.get(k, "") for k in REPLAY_FIELDNAMES})
            except Exception as e:
                logger.error("SR replay write: %s", e)

        def _push(self, idx: int):
            row = self.bars_df.iloc[idx]
            bar = self._make_bar(row)
            self._state.push(bar)

        def _make_bar(self, row) -> Dict:
            cl  = float(row.get("close",  row.get("Close",  1.0)))
            o   = float(row.get("open",   row.get("Open",   cl)))
            h   = float(row.get("high",   row.get("High",   cl)))
            l   = float(row.get("low",    row.get("Low",    cl)))
            v   = float(row.get("volume", row.get("Volume", 1e5)))
            self._sess_high = max(self._sess_high, h)
            if cl >= o: self._grn_str += 1; self._red_str = 0
            else:        self._red_str += 1; self._grn_str = 0

            lbl = self._scanner.push(dict(
                timestamp=str(row.get("timestamp", self._t)),
                open=o, high=h, low=l, close=cl, volume=v,
            ))

            vm   = float(row.get("volume_ma20", v))
            vwap = float(row.get("vwap",  cl))
            e9   = float(row.get("ema9",  cl))
            e20  = float(row.get("ema20", cl))
            rvol = v / (vm + 1e-8)
            pr   = max(h - l, 1e-8)
            body = abs(cl - o)

            return dict(
                timestamp               = str(row.get("timestamp", self._t)),
                open=o, high=h, low=l, close=cl, volume=v,
                vwap=vwap, ema9=e9, ema20=e20,
                session_high            = self._sess_high,
                session_low             = float(row.get("session_low", l)),
                prev_close              = self._prev_cl,
                prev_low                = float(row.get("prev_low", l)),
                prev_high               = float(row.get("prev_high", h)),
                support_zone            = lbl.nearest_support or float(row.get("support_zone", l * 0.99)),
                resistance_zone         = lbl.nearest_resistance or float(row.get("resistance_zone", h * 1.01)),
                volume_ma20             = vm,
                rvol                    = rvol,
                day_open                = self._day_open,
                total_bars_today        = self._t,
                lower_wick_ratio        = (min(o, cl) - l) / pr,
                upper_wick_ratio        = (h - max(o, cl)) / pr,
                body_to_range_ratio     = body / pr,
                red_candle_streak       = self._red_str,
                green_candle_streak     = self._grn_str,
                volume_spike_ratio      = rvol,
                absorption_score        = (rvol - 1.0) * (1.0 - body / pr),
                support_break_failed    = lbl.support_break_failed,
                liquidity_sweep_detected= lbl.liquidity_sweep_detected,
                pivot_low_reclaim       = lbl.pivot_low_reclaim,
                missed_reversal_level_reclaim = lbl.missed_reversal_level_reclaim,
                breakout_retest         = lbl.breakout_retest,
                support_touch           = lbl.support_touch,
                in_position             = int(self._in_pos),
                unrealised_pnl          = (cl - self._entry) if self._in_pos else 0.0,
                bars_in_trade           = self._bars_held,
            )


# ---------------------------------------------------------------------------
# Diagnostics callback
# ---------------------------------------------------------------------------

if SB3_AVAILABLE:
    class SRDiagnosticsCallback(BaseCallback):
        def __init__(self, metrics_path: str = SR_METRICS_PATH, verbose: int = 0):
            super().__init__(verbose)
            self.metrics_path   = metrics_path
            self._last_metrics: Dict = {}
            os.makedirs(
                os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else ".",
                exist_ok=True,
            )

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> bool:
            env = self.training_env
            if hasattr(env, "envs"):
                for e in env.envs:
                    inner = e.env if hasattr(e, "env") else e
                    if hasattr(inner, "ep_metrics") and inner.ep_metrics:
                        self._last_metrics = inner.ep_metrics
            return True

        def _on_training_end(self):
            with open(self.metrics_path, "w") as f:
                json.dump(self._last_metrics, f, indent=2)
            logger.info("SR metrics saved → %s", self.metrics_path)
            _print_sr_summary(self._last_metrics)


def _print_sr_summary(m: Dict):
    logger.info("=" * 60)
    logger.info("SR RL AGENT — VALIDATION SUMMARY")
    logger.info("  Metrics path            : %s", SR_METRICS_PATH)
    logger.info("  Best params path        : %s", SR_SAVE_PATH)
    logger.info("  Profit factor           : %.3f", m.get("profit_factor", 0))
    logger.info("  Win rate                : %.1f%%", m.get("win_rate", 0) * 100)
    logger.info("  Avg MFE                 : %.2f%%", m.get("avg_mfe_pct", 0) * 100)
    logger.info("  Avg MAE                 : %.2f%%", m.get("avg_mae_pct", 0) * 100)
    logger.info("  Capture ratio           : %.2f",   m.get("avg_capture_ratio", 0))
    logger.info("  Dip-reclaim entries     : %d",     m.get("dip_reclaim_entries", 0))
    logger.info("  Dip-reclaim win rate    : %.1f%%", m.get("dip_reclaim_win_rate", 0) * 100)
    logger.info("  Support reclaim entries : %d",     m.get("support_reclaim_entries", 0))
    logger.info("  Support reclaim WR      : %.1f%%", m.get("support_reclaim_win_rate", 0) * 100)
    logger.info("  Missed dip reclaims     : %d",     m.get("missed_dip_reclaim_count", 0))
    logger.info("=" * 60)

    # Self-assessment
    pf   = m.get("profit_factor", 0)
    drwr = m.get("dip_reclaim_win_rate", 0)
    needed = []
    if pf < 1.2:    needed.append("profit_factor low — raise SR_R_SWEEP_RECLAIM, SR_R_SUPPORT_BOUNCE")
    if drwr < 0.40: needed.append("dip_reclaim_win_rate low — tighten setup filter, raise SR_R_MFE_3R")
    if m.get("n_trades", 0) < 3:
        needed.append("too few trades — reduce SR_P_NO_SETUP magnitude, increase SR_R_SUPPORT_BOUNCE")
    if m.get("missed_dip_reclaim_count", 0) > 10:
        needed.append("too many missed setups — increase SR_P_MISSED penalty")

    if needed:
        logger.warning("SR model did NOT fully learn screenshot-style setup. Improvements needed:")
        for n in needed:
            logger.warning("  → %s", n)
    else:
        logger.info("✓ SR model learned dip-reclaim structure.")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train():
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 + gymnasium required")

    logger.info("Loading SR data from %s", SR_DATA_PATH)
    if SR_DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(SR_DATA_PATH)
    else:
        df = pd.read_csv(SR_DATA_PATH, parse_dates=["timestamp"])

    symbol   = os.path.basename(SR_DATA_PATH).split("_")[0].upper()
    split    = int(len(df) * 0.85)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    os.makedirs(SR_SAVE_PATH, exist_ok=True)

    def make_env(bars):
        def _init():
            e = SRRLEnv(bars, symbol=symbol, replay_path=SR_REPLAY_PATH)
            return Monitor(e)
        return _init

    train_env = DummyVecEnv([make_env(train_df)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    eval_env  = DummyVecEnv([make_env(eval_df)])
    eval_env  = VecNormalize(eval_env,  norm_obs=False, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs   = get_policy_kwargs(),
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        verbose         = 1,
        tensorboard_log = "logs/sr_rl_tb",
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path = SR_SAVE_PATH,
            log_path             = SR_SAVE_PATH,
            eval_freq            = SR_EVAL_FREQ,
            n_eval_episodes      = 5,
            deterministic        = True,
        ),
        SRDiagnosticsCallback(metrics_path=SR_METRICS_PATH),
    ]

    logger.info("Training SR RL Agent for %d timesteps …", SR_RL_TIMESTEPS)
    model.learn(total_timesteps=SR_RL_TIMESTEPS, callback=callbacks)
    model.save(os.path.join(SR_SAVE_PATH, "sr_final_model"))
    logger.info("SR model saved → %s/sr_final_model", SR_SAVE_PATH)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    else:
        print("Usage: python sr_rl_agent.py --train")