"""
tjl_signal.py — deterministic Trend Join Long (TJL) signal engine.

This is the SINGLE SOURCE OF TRUTH for TJL trade decisions, ported 1:1 from the
Pine v6 strategy "Trend Join Long (TJL)". No LLM is involved in any decision here:
every entry/exit is a pure function of bar data and config. run.py imports this
module and routes the resulting intents through the existing risk-gated broker.

Pine → Python fidelity notes (the parts that actually matter):

  * ta.ema(src, len): alpha = 2/(len+1); the FIRST value is seeded with src
    (Pine: `na(sum[1]) ? src : ...`), then recursive. Matched exactly in ema_series.
  * ta.sma(src, len): simple mean of the last `len` values; NA until `len` bars.
  * ta.stoch(close, high, low, len): 100*(close - lowest(low,len)) / (highest(high,len)-lowest(low,len));
    NA (None) when the range is 0. Then k = sma(stoch, smoothK), d = sma(k, smoothD).
  * ta.crossover(a, b): a[1] <= b[1] AND a > b  (fast crosses ABOVE slow).
  * Higher-timeframe filters use request.security(..., _expr[1], lookahead=lookahead_on),
    which is the standard NON-REPAINTING idiom: the value seen on any entry bar is the
    one computed on the LAST FULLY COMPLETED HTF bar. Callers therefore pass ONLY
    completed HTF bars and these functions read the last element. This is how we
    guarantee "no lookahead bias" — never the forming bar.

Blocked-entry reason codes (stable strings, used for logging/metrics):
  trend_not_up | momentum_not_ok | no_ema_cross | outside_rth | already_in_position
Exit reason codes:
  stop_loss | take_profit | eod_flat
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence
from zoneinfo import ZoneInfo

logger = logging.getLogger("tjl")

NY = ZoneInfo("America/New_York")

# Blocked-entry reason codes
TREND_NOT_UP = "trend_not_up"
MOMENTUM_NOT_OK = "momentum_not_ok"
NO_EMA_CROSS = "no_ema_cross"
OUTSIDE_RTH = "outside_rth"
ALREADY_IN_POSITION = "already_in_position"
# Entry/exit reason codes
ENTER = "entry"
STOP_LOSS = "stop_loss"
TAKE_PROFIT = "take_profit"
EOD_FLAT = "eod_flat"

__all__ = [
    "TJLConfig", "EntrySignal", "ExitSignal",
    "sma", "ema_series", "sma_series", "stoch_raw_series", "stoch_kd",
    "crossover_last", "trend_up", "momentum_ok", "ema_crossover_up",
    "is_rth", "hit_stop", "hit_take_profit", "is_session_end",
    "evaluate_entry", "evaluate_exit",
    "TREND_NOT_UP", "MOMENTUM_NOT_OK", "NO_EMA_CROSS", "OUTSIDE_RTH",
    "ALREADY_IN_POSITION", "ENTER", "STOP_LOSS", "TAKE_PROFIT", "EOD_FLAT",
]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def _parse_tf_minutes(tf: str) -> int:
    """'60m' -> 60, '15m' -> 15, '60' -> 60, '1h' -> 60, '1D'/'D' -> 1440."""
    s = str(tf or "").strip().lower()
    if not s:
        return 0
    if s in ("d", "1d", "day"):
        return 1440
    mult = 1
    if s.endswith("h"):
        mult, s = 60, s[:-1]
    elif s.endswith("m"):
        mult, s = 1, s[:-1]
    try:
        return int(float(s)) * mult
    except ValueError:
        return 0


def _env_bool(env, key, default: bool) -> bool:
    return str(env.get(key, str(default))).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TJLConfig:
    """All TJL parameters. Defaults match the Pine inputs exactly."""
    enabled: bool = False
    trend_tf: str = "60m"
    trend_sma_len: int = 200
    momo_tf: str = "15m"
    stoch_len: int = 14
    smooth_k: int = 3
    smooth_d: int = 3
    stoch_ob: int = 80
    fast_ema: int = 9
    slow_ema: int = 21
    stop_pct: float = 0.015     # fraction, NOT percent (Pine 1.5/100)
    tp_pct: float = 0.03        # fraction, NOT percent (Pine 3.0/100)
    eod_flat: bool = True

    @property
    def trend_tf_min(self) -> int:
        return _parse_tf_minutes(self.trend_tf)

    @property
    def momo_tf_min(self) -> int:
        return _parse_tf_minutes(self.momo_tf)

    @classmethod
    def from_env(cls, env=None) -> "TJLConfig":
        """Build config from environment. Pure parsing — never reads secrets."""
        env = os.environ if env is None else env
        def _i(k, d):
            try:
                return int(float(env.get(k, d)))
            except (TypeError, ValueError):
                return d
        def _f(k, d):
            try:
                return float(env.get(k, d))
            except (TypeError, ValueError):
                return d
        return cls(
            enabled=_env_bool(env, "TJL_ENABLED", False),
            trend_tf=str(env.get("TJL_TREND_TF", "60m")),
            trend_sma_len=_i("TJL_TREND_SMA_LEN", 200),
            momo_tf=str(env.get("TJL_MOMO_TF", "15m")),
            stoch_len=_i("TJL_STOCH_LEN", 14),
            smooth_k=_i("TJL_SMOOTH_K", 3),
            smooth_d=_i("TJL_SMOOTH_D", 3),
            stoch_ob=_i("TJL_STOCH_OB", 80),
            fast_ema=_i("TJL_FAST_EMA", 9),
            slow_ema=_i("TJL_SLOW_EMA", 21),
            stop_pct=_f("TJL_STOP_PCT", 0.015),
            tp_pct=_f("TJL_TP_PCT", 0.03),
            eod_flat=_env_bool(env, "TJL_EOD_FLAT", True),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Indicator primitives (Pine-faithful, pure)
# ──────────────────────────────────────────────────────────────────────────────

def sma(values: Sequence[float], length: int) -> Optional[float]:
    """SMA of the LAST `length` values; None if insufficient/contains None."""
    if length <= 0 or len(values) < length:
        return None
    window = values[-length:]
    if any(v is None for v in window):
        return None
    return sum(float(v) for v in window) / length


def sma_series(values: Sequence[Optional[float]], length: int) -> List[Optional[float]]:
    """Rolling SMA series; element is None until `length` non-None values precede it."""
    out: List[Optional[float]] = []
    for i in range(len(values)):
        if i + 1 < length:
            out.append(None)
            continue
        window = values[i + 1 - length:i + 1]
        out.append(None if any(v is None for v in window)
                   else sum(float(v) for v in window) / length)
    return out


def ema_series(values: Sequence[float], length: int) -> List[float]:
    """Pine ta.ema: first value seeded with src, then alpha*src + (1-alpha)*prev."""
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out: List[float] = []
    prev: Optional[float] = None
    for v in values:
        cur = float(v) if prev is None else alpha * float(v) + (1.0 - alpha) * prev
        out.append(cur)
        prev = cur
    return out


def stoch_raw_series(highs: Sequence[float], lows: Sequence[float],
                     closes: Sequence[float], length: int) -> List[Optional[float]]:
    """Pine ta.stoch(close, high, low, length). None when range == 0 (Pine NA)."""
    n = len(closes)
    out: List[Optional[float]] = []
    for i in range(n):
        if i + 1 < length:
            out.append(None)
            continue
        hh = max(highs[i + 1 - length:i + 1])
        ll = min(lows[i + 1 - length:i + 1])
        rng = hh - ll
        out.append(None if rng == 0 else 100.0 * (float(closes[i]) - ll) / rng)
    return out


def stoch_kd(highs, lows, closes, length, smooth_k, smooth_d):
    """Return (k_series, d_series): k = sma(rawStoch, smoothK), d = sma(k, smoothD)."""
    raw = stoch_raw_series(highs, lows, closes, length)
    k = sma_series(raw, smooth_k)
    d = sma_series(k, smooth_d)
    return k, d


def crossover_last(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> bool:
    """Pine ta.crossover at the LAST bar: a[1] <= b[1] AND a[0] > b[0]."""
    if len(a) < 2 or len(b) < 2:
        return False
    a0, a1, b0, b1 = a[-1], a[-2], b[-1], b[-2]
    if None in (a0, a1, b0, b1):
        return False
    return a1 <= b1 and a0 > b0


# ──────────────────────────────────────────────────────────────────────────────
# TJL filters (each individually unit-tested)
# ──────────────────────────────────────────────────────────────────────────────

def trend_up(htf_closes: Sequence[float], sma_len: int) -> bool:
    """Trend up = last COMPLETED HTF close > SMA(sma_len) ending at that bar."""
    s = sma(htf_closes, sma_len)
    if s is None or not htf_closes:
        return False
    return float(htf_closes[-1]) > s


def momentum_ok(highs, lows, closes, stoch_len, smooth_k, smooth_d, stoch_ob) -> bool:
    """Momentum OK = on the last COMPLETED momo bar: K > D AND K < stoch_ob."""
    k, d = stoch_kd(highs, lows, closes, stoch_len, smooth_k, smooth_d)
    if not k or not d:
        return False
    kv, dv = k[-1], d[-1]
    if kv is None or dv is None:
        return False
    return (kv > dv) and (kv < stoch_ob)


def ema_crossover_up(entry_closes: Sequence[float], fast_len: int, slow_len: int) -> bool:
    """Fast EMA crosses ABOVE slow EMA on the latest entry bar."""
    if len(entry_closes) < 2:
        return False
    return crossover_last(ema_series(entry_closes, fast_len),
                          ema_series(entry_closes, slow_len))


def is_rth(ts: datetime) -> bool:
    """RTH guard matching Pine time('0930-1600','America/New_York'): TV's session end is
    boundary-INCLUSIVE, so the 16:00 bar IS in-session (EOD flatten then fires on the
    16:05 bar). 16:05+ is out of session."""
    t = ts.astimezone(NY)
    if t.weekday() >= 5:
        return False
    mins = t.hour * 60 + t.minute
    return (9 * 60 + 30) <= mins <= (16 * 60)


def hit_stop(price: float, avg_entry: float, stop_pct: float) -> bool:
    if avg_entry <= 0:
        return False
    return float(price) <= avg_entry * (1.0 - stop_pct)


def hit_take_profit(price: float, avg_entry: float, tp_pct: float) -> bool:
    if avg_entry <= 0:
        return False
    return float(price) >= avg_entry * (1.0 + tp_pct)


def is_session_end(prev_ts: datetime, curr_ts: datetime) -> bool:
    """Pine: isRTH[1] and not isRTH — we were in RTH last bar, not this bar."""
    return is_rth(prev_ts) and not is_rth(curr_ts)


# ──────────────────────────────────────────────────────────────────────────────
# Decision engine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntrySignal:
    enter: bool
    reason: str   # ENTER when enter=True, else a blocked-entry reason code


@dataclass(frozen=True)
class ExitSignal:
    exit: bool
    reason: str   # "" when exit=False, else stop_loss|take_profit|eod_flat


def evaluate_entry(cfg: TJLConfig,
                   trend_closes: Sequence[float],
                   momo_highs: Sequence[float], momo_lows: Sequence[float],
                   momo_closes: Sequence[float],
                   entry_closes: Sequence[float],
                   now_ts: datetime,
                   in_position: bool,
                   *, log: bool = True) -> EntrySignal:
    """Long-only entry decision. Returns the FIRST failing gate's reason, matching
    Pine longCond = trendUp and momoOK and entryX and isRTH (+ no-pyramiding).

    Caller passes ONLY completed HTF bars for trend/momo (no forming bar) so this
    is non-repainting by construction.
    """
    if in_position:                                   # pyramiding = 0
        return _block(ALREADY_IN_POSITION, log)
    if not is_rth(now_ts):
        return _block(OUTSIDE_RTH, log)
    if not trend_up(trend_closes, cfg.trend_sma_len):
        return _block(TREND_NOT_UP, log)
    if not momentum_ok(momo_highs, momo_lows, momo_closes,
                       cfg.stoch_len, cfg.smooth_k, cfg.smooth_d, cfg.stoch_ob):
        return _block(MOMENTUM_NOT_OK, log)
    if not ema_crossover_up(entry_closes, cfg.fast_ema, cfg.slow_ema):
        return _block(NO_EMA_CROSS, log)
    if log:
        logger.info("TJL ENTER (trendUp & momoOK & emaCross & RTH)")
    return EntrySignal(True, ENTER)


def evaluate_exit(cfg: TJLConfig, price: float, avg_entry: float,
                  prev_ts: datetime, now_ts: datetime, in_position: bool) -> ExitSignal:
    """Long-only exit decision: stop-loss, take-profit, or end-of-session flatten."""
    if not in_position:
        return ExitSignal(False, "")
    if hit_stop(price, avg_entry, cfg.stop_pct):
        return ExitSignal(True, STOP_LOSS)
    if hit_take_profit(price, avg_entry, cfg.tp_pct):
        return ExitSignal(True, TAKE_PROFIT)
    if cfg.eod_flat and is_session_end(prev_ts, now_ts):
        return ExitSignal(True, EOD_FLAT)
    return ExitSignal(False, "")


def _block(reason: str, log: bool) -> EntrySignal:
    if log:
        logger.info("TJL entry blocked: %s", reason)
    return EntrySignal(False, reason)
