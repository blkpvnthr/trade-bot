#!/usr/bin/env python3
"""
V2.9.8 momentum-exhaustion / bearish->bullish reversal overlay (pure, shared).
==============================================================================

A SECOND exceptional-exit overlay, independent of loss-replacement. For each held
position it builds a causal (point-in-time) ``momentum_exhaustion_score`` in [0,1]:
0 = healthy trend, 1 = strong evidence a bullish trend is losing strength and
reversal risk is rising. When the score is CONFIRMED above the exit threshold (or a
single CRITICAL reading), the position is rotated into a bearish->bullish
``bullish_reversal_score`` candidate -- an asset near the start of improving
momentum, not one already extended. All risk constraints are re-applied afterward.

Nothing here can know the future. The detector estimates rising reversal *risk*
from information through the decision close; it never uses forward returns,
centered indicators, or retrospective extrema. Forward outcomes are computed only
by the offline diagnostics (``post_exit_outcomes``) for EVALUATION, never as inputs.

Score design (documented, minimal, grouped so correlated indicators do not
double-count -- spec categories trend / momentum / risk / forecast):

  exhaustion = 0.40*trend_decel + 0.20*oscillator_rollover
             + 0.15*volatility_expansion + 0.25*forecast_deterioration
  reversal   = 0.40*momo_improving + 0.20*oscillator_inflection
             + 0.15*price_reclaim   + 0.25*forecast_inflection

Only daily CLOSES are available to the allocator, so DMI/ADX (which need
intraday range) are intentionally omitted; trend strength is captured by the
multi-horizon momentum-deceleration and SMA-slope terms instead. The forecast
term reuses the allocator's own ``momentum_view_panel`` (the repo's view), which
is already lagged and point-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strat_allocator import AllocatorConfig, enforce_constraints, TRADING_DAYS

# Documented composite weights (NOT a config surface -- keep it minimal, spec 10).
EXH_WEIGHTS = {"trend_decel": 0.40, "oscillator": 0.20, "volatility": 0.15,
               "forecast": 0.25}
REV_WEIGHTS = {"momo_improving": 0.40, "oscillator": 0.20, "price_reclaim": 0.15,
               "forecast": 0.25}

# Exhaustion evidence on the trend / oscillator / risk / forecast axes peaks on
# DIFFERENT sessions (a trend rolls over, then MACD crosses, then vol spikes). A
# plain weighted average of the instantaneous axes therefore dilutes real
# exhaustion. Each component is smoothed with a short CAUSAL rolling-max so
# evidence seen within the last few sessions is counted together -- "rising
# reversal probability over a window", not one instant. Backward-looking only.
SMOOTH_WINDOW = 3


def _smooth(comp: dict, index, assets) -> dict:
    out = {}
    for k, v in comp.items():
        df = pd.DataFrame(np.asarray(v, float), index=index, columns=assets)
        out[k] = df.rolling(SMOOTH_WINDOW, min_periods=1).max()
    return out


# ---------------------------------------------------------------------------
# point-in-time indicator primitives (all backward-looking)
# ---------------------------------------------------------------------------
def _ramp(x, lo, hi):
    """Linear 0->1 ramp; NaN-safe."""
    with np.errstate(invalid="ignore"):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(px: pd.DataFrame, fast=12, slow=26, sig=9) -> pd.DataFrame:
    macd = px.ewm(span=fast, adjust=False).mean() - px.ewm(span=slow, adjust=False).mean()
    return macd - macd.ewm(span=sig, adjust=False).mean()


def _state_name(score: float, cfg: AllocatorConfig) -> str:
    if not np.isfinite(score):
        return "UNKNOWN"
    if score >= cfg.momentum_exhaustion_critical:
        return "CRITICAL"
    if score >= cfg.momentum_exhaustion_threshold:
        return "EXHAUSTED"
    if score >= cfg.momentum_exhaustion_warning:
        return "WEAKENING"
    return "HEALTHY"


# ---------------------------------------------------------------------------
# score panels  (index = prices.index, columns = risk assets)  -- point in time
# ---------------------------------------------------------------------------
def exhaustion_panel(prices: pd.DataFrame, cfg: AllocatorConfig,
                     view: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    assets = list(cfg.risk_assets)
    px = prices[assets].astype(float)
    ret = px.pct_change()
    mom5, mom10, mom20 = px / px.shift(5) - 1, px / px.shift(10) - 1, px / px.shift(20) - 1
    vol20 = ret.rolling(20).std()
    sma20 = px.rolling(20).mean()
    slope = (sma20 - sma20.shift(5)) / sma20.shift(5).abs()
    slope_prev = (sma20.shift(5) - sma20.shift(10)) / sma20.shift(10).abs()
    hist = _macd_hist(px)
    rsi = _rsi(px)

    up = (mom20 > 0).astype(float)
    # (A) trend / momentum deceleration: prior uptrend rolling over
    decel = ((mom10 - mom5) + (mom20 - mom10)) / (2.0 * vol20 * np.sqrt(5.0) + 1e-9)
    a = up * (0.6 * _ramp(decel, 0.0, 3.0) + 0.4 * (mom5 < 0).astype(float))
    a = 0.7 * a + 0.3 * up * _ramp(slope_prev - slope, 0.0, 0.02)
    # (B) oscillator rollover: MACD histogram contracting from + / crossing < 0; RSI rolled from OB
    hist_shrink = ((hist > 0) & (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))).astype(float)
    hist_cross = ((hist.shift(1) > 0) & (hist <= 0)).astype(float)
    rsi_roll = ((rsi.rolling(5).max() > 70) & (rsi < rsi.shift(2))).astype(float)
    b = np.clip(0.4 * hist_shrink + 0.35 * hist_cross + 0.25 * rsi_roll, 0.0, 1.0)
    # (C) volatility expansion: short-horizon realized vol elevated vs long
    vol5, vol60 = ret.rolling(5).std(), ret.rolling(60).std()
    c = _ramp(vol5 / (vol60 + 1e-9) - 1.0, 0.0, 1.0)
    # (D) forecast deterioration: allocator view fell from a positive level
    d = _forecast_change(view, assets, px.index, deteriorate=True)

    comp = _smooth({"trend_decel": a, "oscillator": b, "volatility": c, "forecast": d},
                   px.index, assets)
    score = sum(EXH_WEIGHTS[k] * comp[k] for k in EXH_WEIGHTS).clip(0.0, 1.0)
    return score, comp


def reversal_panel(prices: pd.DataFrame, cfg: AllocatorConfig,
                   view: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    assets = list(cfg.risk_assets)
    px = prices[assets].astype(float)
    ret = px.pct_change()
    mom5, mom10, mom20 = px / px.shift(5) - 1, px / px.shift(10) - 1, px / px.shift(20) - 1
    vol20 = ret.rolling(20).std()
    sma20, sma50 = px.rolling(20).mean(), px.rolling(50).mean()
    hist = _macd_hist(px)
    rsi = _rsi(px)

    down = (mom20 < 0).astype(float)
    # (A) momentum improving from negative (bottoming -> recovering)
    accel = ((mom5 - mom10) + (mom10 - mom20)) / (2.0 * vol20 * np.sqrt(5.0) + 1e-9)
    a = down * (0.6 * _ramp(accel, 0.0, 3.0) + 0.4 * (mom5 > 0).astype(float))
    # (B) oscillator inflection: MACD histogram rising from - / crossing up; RSI recovering from OS
    hist_rise = ((hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))).astype(float)
    hist_cross_up = ((hist.shift(1) < 0) & (hist >= 0)).astype(float)
    rsi_rec = ((rsi.rolling(5).min() < 35) & (rsi > rsi.shift(2))).astype(float)
    b = np.clip(0.4 * hist_rise + 0.35 * hist_cross_up + 0.25 * rsi_rec, 0.0, 1.0)
    # (C) price reclaim: back above short MA while long MA still depressed (early recovery)
    c = ((px > sma20) & (sma20 < sma50)).astype(float)
    # (D) forecast inflection: allocator view rising toward/through zero
    d = _forecast_change(view, assets, px.index, deteriorate=False)

    comp = _smooth({"momo_improving": a, "oscillator": b, "price_reclaim": c,
                    "forecast": d}, px.index, assets)
    score = sum(REV_WEIGHTS[k] * comp[k] for k in REV_WEIGHTS).clip(0.0, 1.0)
    return score, comp


def _forecast_change(view, assets, index, *, deteriorate: bool):
    """View-level change over 5 sessions, mapped to [0,1]. All point-in-time."""
    if view is None:
        return pd.DataFrame(0.0, index=index, columns=assets)
    v = view.reindex(index)[assets].astype(float)
    v5 = v.shift(5)
    denom = v5.abs() + 1e-6
    if deteriorate:                       # fell from a positive level
        raw = _ramp((v5 - v) / denom, 0.0, 1.0) * (v5 > 0).astype(float)
    else:                                 # rose from a non-positive level (inflection)
        raw = _ramp((v - v5) / denom, 0.0, 1.0) * ((v5 <= 0) | (v > v5)).astype(float)
    return raw.fillna(0.0)


# ---------------------------------------------------------------------------
# the overlay
# ---------------------------------------------------------------------------
@dataclass
class ExhaustionResult:
    triggered: bool
    weights: np.ndarray
    events: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    exited: list[str] = field(default_factory=list)
    rebalance_reason: str | None = None


def apply_exhaustion_rotation(*,
                              base_weights: np.ndarray,
                              position_returns: np.ndarray,
                              exhaustion_recent: np.ndarray,     # (k, n) last-k scores
                              exhaustion_components: np.ndarray, # (n, ncomp) latest
                              component_names: list[str],
                              reversal_scores: np.ndarray,       # (n,)
                              reversal_rank_tiebreak: np.ndarray,  # (n,) forecast/view for ties
                              cov_adj: np.ndarray,
                              cfg: AllocatorConfig,
                              growth_ceiling: float,
                              vol_budget: float,
                              leverage_cap: float,
                              cooldown_mask: np.ndarray | None = None,
                              eligible_mask: np.ndarray | None = None) -> ExhaustionResult:
    """
    Rotate CONFIRMED-exhausted holdings into bullish-reversal candidates.

    A held asset is exhausted when its score is >= threshold for every one of the
    last ``momentum_exhaustion_confirmations`` sessions (confirmation), OR its
    latest score is >= the critical threshold (immediate). A CONFIRMED (non-critical)
    exhaustion rotates ONLY if a candidate improves the reversal opportunity by at
    least ``minimum_replacement_score_improvement`` (else the winner is left to run,
    spec 11/13). A CRITICAL exhaustion always exits (to cash if no candidate, spec 16).
    """
    assets = list(cfg.risk_assets)
    n = len(assets)
    w = np.asarray(base_weights, float).copy()
    pr = np.asarray(position_returns, float)
    latest = np.asarray(exhaustion_recent, float)[-1]
    thr = float(cfg.momentum_exhaustion_threshold)
    crit = float(cfg.momentum_exhaustion_critical)
    rev_thr = float(cfg.bullish_reversal_score_threshold)
    min_impr = float(cfg.minimum_replacement_score_improvement)
    eps = float(cfg.loss_replacement_hold_eps)
    rev = np.asarray(reversal_scores, float)
    tb = np.asarray(reversal_rank_tiebreak, float)
    if cooldown_mask is None:
        cooldown_mask = np.zeros(n, bool)
    if eligible_mask is None:
        eligible_mask = np.ones(n, bool)
    cooldown_mask = np.asarray(cooldown_mask, bool)
    eligible_mask = np.asarray(eligible_mask, bool)

    held = np.isfinite(pr)
    confirmed = np.all(np.asarray(exhaustion_recent, float) >= thr - 1e-12, axis=0)
    critical = np.isfinite(latest) & (latest >= crit - 1e-12)
    exhausted = held & np.isfinite(latest) & (confirmed | critical)
    ex_idx = [j for j in range(n) if exhausted[j]]
    if not ex_idx:
        return ExhaustionResult(False, w, [], [], [], None)

    def is_candidate(j):
        return (bool(eligible_mask[j]) and not bool(cooldown_mask[j]) and not held[j]
                and not exhausted[j] and np.isfinite(rev[j]) and rev[j] >= rev_thr)

    pool = sorted((j for j in range(n) if is_candidate(j)),
                  key=lambda j: (-float(rev[j]),
                                 -(float(tb[j]) if np.isfinite(tb[j]) else -np.inf),
                                 assets[j]))
    ex_idx.sort(key=lambda j: (-float(latest[j]), assets[j]))    # most exhausted first

    events, skipped, exited, used = [], [], [], set()
    for ex in ex_idx:
        is_crit = bool(critical[ex])
        cur_opp = float(rev[ex]) if np.isfinite(rev[ex]) else 0.0
        repl = next((j for j in pool if j not in used
                     and rev[j] >= cur_opp + min_impr), None)
        comp = {component_names[c]: round(float(exhaustion_components[ex, c]), 4)
                for c in range(len(component_names))}
        if repl is not None:
            used.add(repl)
            freed = float(w[ex])
            w[ex] = 0.0
            w[repl] += freed
            exited.append(assets[ex])
            events.append(_ev(assets[ex], pr[ex], latest[ex], cfg, comp,
                              assets[repl], rev[repl],
                              "confirmed_momentum_exhaustion_and_superior_reversal_candidate"))
        elif is_crit:
            w[ex] = 0.0
            exited.append(assets[ex])
            events.append(_ev(assets[ex], pr[ex], latest[ex], cfg, comp, None, None,
                              "critical_momentum_exhaustion_exit_to_cash"))
        else:
            skipped.append(_ev(assets[ex], pr[ex], latest[ex], cfg, comp, None, None,
                               "no_sufficiently_superior_reversal_candidate", skipped=True))

    if not exited:
        return ExhaustionResult(False, w, [], skipped, [], None)
    w = enforce_constraints(w, cfg, growth_ceiling, np.asarray(cov_adj, float),
                            vol_budget, leverage_cap)
    return ExhaustionResult(True, w, events, skipped, exited, "momentum_exhaustion")


def _ev(asset, pr, score, cfg, comp, repl, rev, reason, *, skipped=False):
    ev = {
        "type": "momentum_exhaustion",
        "exited_asset": None if skipped else asset,
        "asset": asset,
        "position_return": (round(float(pr), 6) if np.isfinite(pr) else None),
        "exhaustion_score": round(float(score), 4),
        "exhaustion_threshold": float(cfg.momentum_exhaustion_threshold),
        "state": _state_name(float(score), cfg),
        "components": comp,
        "replacement_asset": repl,
        "bullish_reversal_score": (round(float(rev), 4) if rev is not None else None),
        "action": "hold" if skipped else ("exit_to_cash" if repl is None else "rotate"),
        "reason": reason,
    }
    return ev


# ---------------------------------------------------------------------------
# offline evaluation ONLY -- forward outcomes never feed a decision
# ---------------------------------------------------------------------------
def post_exit_outcomes(prices: pd.DataFrame, exit_date, symbol: str,
                       horizons=(1, 3, 5, 10)) -> dict:
    """RESEARCH ONLY. Forward returns / excursions after an exhaustion exit."""
    px = prices[symbol].astype(float)
    if exit_date not in px.index:
        return {}
    p0 = float(px.loc[exit_date])
    fut = px.loc[px.index > pd.Timestamp(exit_date)]
    out = {"exit_price": p0}
    for h in horizons:
        seg = fut.iloc[:h]
        out[f"ret_{h}d"] = (float(seg.iloc[-1]) / p0 - 1.0) if len(seg) >= 1 else np.nan
        out[f"mae_{h}d"] = (float(seg.min()) / p0 - 1.0) if len(seg) >= 1 else np.nan
        out[f"mfe_{h}d"] = (float(seg.max()) / p0 - 1.0) if len(seg) >= 1 else np.nan
    return out


def _series(rets):
    return 100.0 * np.cumprod(1.0 + np.asarray(rets, float))


def demo() -> None:
    """Self-check: exhaustion fires on a rolling-over winner; strong trend holds;
    a steep-drop-then-bounce out-scores a falling knife on reversal."""
    cfg = AllocatorConfig(momentum_exhaustion_enabled=True)
    a = list(cfg.risk_assets)
    # Exhaustion: SPY steady constant-% uptrend (no deceleration) vs QQQ strong rise
    # then a sharp 7-day rollover (still up on 20d, short momentum negative).
    T = 140
    idx = pd.bdate_range("2020-01-01", periods=T)
    data = {s: _series([0.004] * T) for s in a}                    # steady uptrend
    data["QQQ"] = _series([0.009] * (T - 7) + [-0.012] * 7)        # rise then rollover
    exh, _ = exhaustion_panel(pd.DataFrame(data, index=idx), cfg)
    last = exh.iloc[-1]
    assert last["QQQ"] > last["SPY"] + 0.15, (last["QQQ"], last["SPY"])
    assert last["QQQ"] > 0.30 and last["SPY"] < 0.20, (last["QQQ"], last["SPY"])

    # Reversal: IWM steep drop then a fresh 5-day bounce (still down on 20d) vs EEM
    # a relentless straight-line decline (falling knife -> must NOT score high).
    T2 = 115
    idx2 = pd.bdate_range("2020-01-01", periods=T2)
    d2 = {s: _series([0.001] * T2) for s in a}
    d2["IWM"] = _series([0.003] * 80 + [-0.012] * 30 + [0.013] * 5)
    d2["EEM"] = _series([-0.006] * T2)
    rev, _ = reversal_panel(pd.DataFrame(d2, index=idx2), cfg)
    r = rev.iloc[-1]
    assert r["IWM"] > r["EEM"], (r["IWM"], r["EEM"])               # bottoming beats knife
    assert r["EEM"] < 0.3, r["EEM"]                                # knife is not a candidate
    print("momentum_exhaustion demo: OK")


if __name__ == "__main__":
    demo()
