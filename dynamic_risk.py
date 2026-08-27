#!/usr/bin/env python3
"""
Dynamic portfolio-risk / exposure controller for V2.9.8 (OPT-IN, default OFF).
==============================================================================

Answers "HOW MUCH portfolio risk should we carry?" -- a single, bounded
multiplier on the strategy's existing volatility budget:

    effective_target_vol = base_target_vol * portfolio_risk_multiplier

It is the complement of the adaptive rotation engine (which answers "WHERE?").
The hypothesis under test: carrying MORE risk when the cross-sectional
opportunity set is unusually strong and materially LESS when it is hostile can
raise CAGR AND lower MaxDD at the same time. This module does NOT place orders,
change any optimizer parameter, or touch raw weights -- it only produces the SAME
kind of per-date ``vol_budget_mult`` series the bond-regime overlay produces
(``bond_regime.regime_frame``), so the optimizer / vol-scaler / leverage / caps
stay authoritative. Default OFF -> multiplier is exactly 1.0 -> byte-identical to
the base strategy (the bond-overlay parity path already proves this pattern).

Design (all point-in-time, deterministic, causal):

  Opportunity O in [0,1] : convex-weighted cross-sectional breadth of the
      growth sleeve (forecast view, 20d momentum, 200d trend) plus the recovery
      state machine's regime. HIGH O = broad, strong opportunity set.
  Stress S in [0,1]      : growth-sleeve covariance stress from the REAL
      ewma_cov -- volatility LEVEL, volatility ACCELERATION (distinguishes
      high-but-falling from rapidly-rising vol), and average correlation, each
      trailing-percentile-ranked. Drawdown is EXCLUDED from S on purpose; it
      stays the independent hard-risk layer (gate/SevereBear). A drawdown-stress
      term is available for ABLATION only (dynamic_risk_dd_stress_weight > 0).
  Asymmetric AND-gate    : A = O*(1-S) (add risk: needs strong opportunity AND
      low stress); C = (1-O)*S (cut risk: needs weak opportunity AND high
      stress); net = clip(A - C + gamma_R*R, -1, 1). Elevated vol alone with
      strong breadth (S~1, O~1) -> A=0, C=0 -> net~0 -> mult~1.0 (does NOT cut a
      broad bull on vol). Weak AND stressed -> max compression.
  Recovery R in [0,1]    : leads the lagging regime levels out of a crash.
      Permits early re-risking on a strong breadth turn EVEN while realized vol
      is still elevated (does not require falling vol). Lifts net and speeds the
      re-risk RATE; NEVER gates de-risk speed.
  Map -> multiplier      : ASYMMETRIC deadband -- risk > 1.0 is much harder to
      earn (wide deadband_up) than risk < 1.0 (narrow deadband_down). Modest
      evidence de-risks; strong broad conviction is required to add risk.
  Stateful recurrence    : eager+fast de-risk, confirmation-gated + rate-limited
      re-risk, hysteresis, SevereBear snap-to-floor. Pure function of the causal
      panels, seeded 1.0, iterated over the identical dates grid the backtest
      uses -> live recompute-and-take-last equals the backtest value at a date.

RESEARCH POLICY: this ships OFF and OWES a held-out falsification pass
(backtest_rotation_overlays.py configs F/G + F1-F5 ablation ladder) before it is
trusted on real money. The controller only earns its place if it adds
information BEYOND the existing market-regime gate (F4/F5 must beat the F1
regime-only baseline). See LIVE_TRADING.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strat_allocator import (
    AllocatorConfig, TRADING_DAYS, ewma_cov_annualized, annualized_vols)

# Regime -> opportunity level (higher = more risk-on). Mirrors the gate states.
STATE_TO_OPP = {"CRISIS": 0.0, "DEFENSIVE": 0.25, "BOTTOMING": 0.45,
                "RECOVERING": 0.70, "BULL": 1.0}
# Recovery only accelerates re-risking when climbing out of a bottom.
R_STATE = {"BOTTOMING": 0.7, "RECOVERING": 1.0}

# Recovery breadth-turn constants (module-level, not config -- they only bite in
# BOTTOMING/RECOVERING, and the recovery term is gated OFF unless gamma_R > 0).
BREADTH_ACCEL_THR = 0.15       # 20-session rise in forecast breadth to start firing
BREADTH_ACCEL_SCALE = 0.25     # ramp width to full strength

DEFAULT_START = "2007-01-02"


# ---------------------------------------------------------------------------
# small causal helpers
# ---------------------------------------------------------------------------
def _shift(a: np.ndarray, n: int) -> np.ndarray:
    """Lag by n with NaN fill (a[i] <- a[i-n]); a[:n] become NaN."""
    out = np.full(len(a), np.nan, dtype=float)
    if 0 <= n < len(a):
        out[n:] = a[:len(a) - n]
    return out


def _trailing_rank(x: np.ndarray, lookback: int, min_periods: int = 60) -> np.ndarray:
    """
    Strictly-trailing percentile rank of x[i] within x[max(0,i-lookback+1):i+1].

    Returns the fraction of the trailing window <= x[i], in (0, 1]. Neutral 0.5
    on cold start (fewer than min_periods finite observations, or x[i] NaN), so
    the controller starts at multiplier 1.0 and only moves once history exists.
    """
    n = len(x)
    out = np.full(n, 0.5)
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        lo = max(0, i - lookback + 1)
        w = x[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_periods:
            continue
        out[i] = float(np.mean(w <= x[i]))
    return out


def _map_net_to_mult(net: np.ndarray, cfg: AllocatorConfig) -> np.ndarray:
    """
    Asymmetric-deadband map from net conviction in [-1,1] to a target multiplier.

    |net| inside [-deadband_down, +deadband_up] -> exactly 1.0 (wide baseline
    zone). net > deadband_up ramps linearly to max_mult; net < -deadband_down
    ramps to min_mult. deadband_up (wide) >> deadband_down (narrow), so adding
    risk needs strong, broad conviction while de-risking triggers on modest
    evidence.
    """
    du = float(cfg.dynamic_risk_deadband_up)
    dd = float(cfg.dynamic_risk_deadband_down)
    net = np.asarray(net, float)
    up = np.clip((net - du) / max(1.0 - du, 1e-9), 0.0, None)
    down = np.clip((-net - dd) / max(1.0 - dd, 1e-9), 0.0, None)
    m = (1.0
         + up * (cfg.dynamic_risk_max_mult - 1.0)
         - down * (1.0 - cfg.dynamic_risk_min_mult))
    return np.clip(m, cfg.dynamic_risk_min_mult, cfg.dynamic_risk_max_mult)


def _risk_state(mult: float, severe: bool, R: float, cfg: AllocatorConfig) -> str:
    """Diagnostic label for the current portfolio risk posture (not load-bearing)."""
    if severe:
        return "CRISIS"
    if mult <= cfg.dynamic_risk_min_mult + 1e-6:
        return "CRISIS"
    if R > 1e-9 and mult < 1.0:
        return "RE_RISKING"
    if mult < 0.60:
        return "DEFENSIVE"
    if mult < 0.90:
        return "CAUTIOUS"
    if mult <= 1.05:
        return "NORMAL"
    return "RISK_ON"


def _reconstruct_risk_multiplier(m_target: np.ndarray, severe: np.ndarray,
                                 recovery: np.ndarray, cfg: AllocatorConfig) -> np.ndarray:
    """
    Stateful hysteresis + rate-limit + confirmation recurrence, seeded at 1.0.

    De-risking is EAGER and FAST (triggers on a small margin_down gap; steps up to
    max_down_change per day). Re-risking is SLOW and CONFIRMED (needs confirm_days
    of sustained up-desire past margin_up, then steps up to
    max_up_change*(1 + recovery_rate_boost*R) per day). SevereBear snaps to
    severe_floor and bypasses the rate limit. A de-risk desire always resets the
    up-run counter, so a single hostile day interrupts a slow climb.
    """
    n = len(m_target)
    out = np.empty(n)
    mult = 1.0
    up_run = 0
    md = float(cfg.dynamic_risk_max_down_change)
    mu = float(cfg.dynamic_risk_max_up_change)
    mg_up = float(cfg.dynamic_risk_margin_up)
    mg_dn = float(cfg.dynamic_risk_margin_down)
    boost = float(cfg.dynamic_risk_recovery_rate_boost)
    kconf = max(1, int(cfg.dynamic_risk_confirm_days))
    lo, hi = float(cfg.dynamic_risk_min_mult), float(cfg.dynamic_risk_max_mult)
    floor = float(cfg.dynamic_risk_severe_floor)

    for i in range(n):
        if bool(severe[i]):
            mult = floor            # snap; bypass rate limit
            up_run = 0
            out[i] = mult
            continue
        tgt = m_target[i]
        if not np.isfinite(tgt):
            up_run = 0              # hold on undefined target
            out[i] = mult
            continue
        if tgt < mult - mg_dn:                     # want to de-risk -> eager + fast
            mult -= min(mult - tgt, md)
            up_run = 0
        elif tgt > mult + mg_up:                   # want to re-risk -> confirm + slow
            up_run += 1
            if up_run >= kconf:
                r = float(recovery[i]) if np.isfinite(recovery[i]) else 0.0
                mult += min(tgt - mult, mu * (1.0 + boost * r))
        else:                                      # inside the band -> hold
            up_run = 0
        mult = float(np.clip(mult, lo, hi))
        out[i] = mult
    return out


# ---------------------------------------------------------------------------
# opportunity / stress feature panels
# ---------------------------------------------------------------------------
def _opportunity_features(prices: pd.DataFrame, cfg: AllocatorConfig,
                          gate: pd.DataFrame, view: pd.DataFrame | None,
                          dates: pd.DatetimeIndex) -> dict:
    """view_breadth, mom_breadth, trend_breadth, regime -- all in [0,1] over dates."""
    g_assets = list(cfg.growth_assets)
    px = prices[list(cfg.risk_assets)].astype(float)

    def _frac_positive(frame: pd.DataFrame) -> np.ndarray:
        pos = (frame > 0).sum(axis=1).to_numpy(float)
        defined = frame.notna().sum(axis=1).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(defined > 0, pos / defined, np.nan)
        return out

    # forecast breadth (view is already lagged .shift(1) -> info through t-1)
    if view is not None:
        vg = view.reindex(dates)[g_assets]
        view_breadth = _frac_positive(vg)
    else:
        view_breadth = np.full(len(dates), np.nan)

    mom20 = (px / px.shift(20) - 1.0).reindex(dates)[g_assets]
    mom_breadth = _frac_positive(mom20)

    sma200 = px.rolling(200).mean()
    above = (px > sma200)
    above = above.where(sma200.notna())          # undefined until 200 obs
    tg = above.reindex(dates)[g_assets].astype(float)
    trend_breadth = _frac_positive(tg)

    reg_state = gate.reindex(dates)["RecoveryState"]
    regime = reg_state.map(STATE_TO_OPP).fillna(0.25).to_numpy(float)

    return {"view_breadth": view_breadth, "mom_breadth": mom_breadth,
            "trend_breadth": trend_breadth, "regime": regime,
            "reg_state": reg_state.to_numpy(object)}


def _stress_features(cfg: AllocatorConfig, dates: pd.DatetimeIndex,
                     full_idx: pd.DatetimeIndex, R_all: np.ndarray) -> dict:
    """
    Growth-sleeve covariance stress from the REAL ewma_cov, one estimate per date
    over the SAME window the loop uses. port_vol (equal-weight sleeve), avg_corr
    (mean off-diagonal of the sleeve correlation block), then trailing-percentile
    ranks -> vol_level, vol_accel, corr_stress. NOT a cheap 2nd estimator: it is
    the allocator's own fast-vol/slow-corr/shrinkage/PSD-repaired covariance, so
    stress tracks the book's actual ex-ante risk.
    """
    g_idx = cfg.growth_idx
    ng = len(g_idx)
    w_eq = np.full(ng, 1.0 / ng)
    lb = cfg.cov_lookback
    pos_of = {d: k for k, d in enumerate(full_idx)}
    iu = np.triu_indices(ng, k=1)

    port_vol = np.full(len(dates), np.nan)
    avg_corr = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        p = pos_of[d]
        window = R_all[max(0, p - lb + 1): p + 1]
        if window.shape[0] < 30:
            continue
        cov = ewma_cov_annualized(window, cfg.cov_halflife_vol,
                                  cfg.cov_halflife_corr, cfg.cov_shrinkage)
        sub = cov[np.ix_(g_idx, g_idx)]
        port_vol[i] = float(np.sqrt(max(w_eq @ sub @ w_eq, 0.0)))
        sd = np.sqrt(np.maximum(np.diag(sub), 1e-16))
        corr = sub / np.outer(sd, sd)
        avg_corr[i] = float(np.mean(corr[iu]))

    lbp = int(cfg.dynamic_risk_vol_percentile_lookback)
    vol_level = _trailing_rank(port_vol, lbp)
    vol_accel = _trailing_rank(port_vol - _shift(port_vol, 20), lbp)
    corr_stress = _trailing_rank(avg_corr, lbp)
    return {"port_vol": port_vol, "avg_corr": avg_corr,
            "vol_level": vol_level, "vol_accel": vol_accel,
            "corr_stress": corr_stress}


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------
def dynamic_risk_frame(prices: pd.DataFrame, cfg: AllocatorConfig, *,
                       gate: pd.DataFrame, view: pd.DataFrame | None,
                       dates: pd.DatetimeIndex, full_idx: pd.DatetimeIndex,
                       R_all: np.ndarray) -> pd.DataFrame:
    """
    Per-date controller frame indexed by ``dates`` (the backtest's traded grid).

    Columns: opportunity_score, stress_score, recovery, net, risk_multiplier,
    risk_state, plus the component features for diagnostics/attribution. Both
    entry points (run_backtest pre-loop, compute_target_portfolio) pass the
    ALREADY-built gate/view/dates/full_idx/R_all so the computation is one shared
    causal function -> the last value on sliced prices equals the backtest value
    at that date (parity).
    """
    opp = _opportunity_features(prices, cfg, gate, view, dates)
    stress = _stress_features(cfg, dates, full_idx, R_all)

    # --- opportunity score O (convex, cfg-weighted, neutral-filled cold start) --
    ow = {k: float(cfg.dynamic_risk_opp_weights.get(k, 0.0))
          for k in ("view_breadth", "mom_breadth", "trend_breadth", "regime")}
    ow_tot = sum(ow.values())
    if ow_tot <= 1e-12:
        O = np.full(len(dates), 0.5)
    else:
        O = np.zeros(len(dates))
        for k, wk in ow.items():
            feat = np.where(np.isfinite(opp[k]), opp[k], 0.5) if k != "regime" else opp[k]
            O += wk * feat
        O = O / ow_tot
    O = np.clip(O, 0.0, 1.0)

    # --- stress score S (3 vol/corr families; drawdown EXCLUDED by default) -----
    sw = dict(cfg.dynamic_risk_stress_weights)
    sw_vol = float(sw.get("vol_level", 0.0))
    sw_acc = float(sw.get("vol_accel", 0.0))
    sw_corr = float(sw.get("corr", 0.0))
    dd_w = float(cfg.dynamic_risk_dd_stress_weight)     # ablation-only; 0 by default
    # drawdown-stress term is built only to keep the ablation path reachable.
    dd252 = gate.reindex(dates)["Drawdown252"].to_numpy(float)
    dd_stress = np.clip(-np.where(np.isfinite(dd252), dd252, 0.0) / 0.20, 0.0, 1.0)
    s_tot = sw_vol + sw_acc + sw_corr + dd_w
    if s_tot <= 1e-12:
        S = np.zeros(len(dates))
    else:
        S = (sw_vol * stress["vol_level"] + sw_acc * stress["vol_accel"]
             + sw_corr * stress["corr_stress"] + dd_w * dd_stress) / s_tot
    S = np.clip(S, 0.0, 1.0)

    # --- recovery R (leads the lagging regime; does NOT require falling vol) ----
    vb = np.where(np.isfinite(opp["view_breadth"]), opp["view_breadth"], 0.5)
    d20_breadth = vb - _shift(vb, 20)
    accel_breadth = d20_breadth - _shift(d20_breadth, 20)
    breadth_strong = np.clip((d20_breadth - BREADTH_ACCEL_THR) / BREADTH_ACCEL_SCALE,
                             0.0, 1.0)
    breadth_strong = np.where(np.isfinite(accel_breadth) & (accel_breadth > 0),
                              breadth_strong, 0.0)
    d20_pv = stress["port_vol"] - _shift(stress["port_vol"], 20)
    falling_vol_rerisk = 0.6 * ((np.nan_to_num(d20_breadth) > 0)
                                & (np.nan_to_num(d20_pv, nan=1.0) < 0)).astype(float)
    r_state = np.array([R_STATE.get(str(s), 0.0) for s in opp["reg_state"]], float)
    R = r_state * np.maximum(breadth_strong, falling_vol_rerisk)
    R = np.clip(np.nan_to_num(R), 0.0, 1.0)

    # --- asymmetric AND-gate + map + stateful recurrence -----------------------
    gamma_R = float(cfg.dynamic_risk_stress_gamma_recovery)
    A = O * (1.0 - S)
    C = (1.0 - O) * S
    net = np.clip(A - C + gamma_R * R, -1.0, 1.0)
    m_target = _map_net_to_mult(net, cfg)
    severe = gate.reindex(dates)["SevereBear"].fillna(False).to_numpy(bool)
    mult = _reconstruct_risk_multiplier(m_target, severe, R, cfg)

    states = [_risk_state(float(mult[i]), bool(severe[i]), float(R[i]), cfg)
              for i in range(len(dates))]

    return pd.DataFrame({
        "opportunity_score": O, "stress_score": S, "recovery": R,
        "net": net, "target_multiplier": m_target, "risk_multiplier": mult,
        "risk_state": states,
        "view_breadth": opp["view_breadth"], "mom_breadth": opp["mom_breadth"],
        "trend_breadth": opp["trend_breadth"], "regime": opp["regime"],
        "vol_level": stress["vol_level"], "vol_accel": stress["vol_accel"],
        "corr_stress": stress["corr_stress"], "port_vol": stress["port_vol"],
        "severe": severe,
    }, index=dates)


def build_risk_frame(prices: pd.DataFrame, cfg: AllocatorConfig,
                     start: str = DEFAULT_START,
                     forecast_hook=None) -> pd.DataFrame:
    """
    Standalone convenience: build gate/view/dates internally and return the
    controller frame. Used by the demo/tests and any caller that does not already
    hold the built panels. The wired backtest/live paths call dynamic_risk_frame
    directly with the panels they already have (no recomputation).
    """
    from strat import build_gate_schedule, momentum_view_panel  # lazy: avoid import cycle
    prices = prices.sort_index()
    gate = build_gate_schedule(prices["SPY"], cfg)
    if forecast_hook is not None:
        view = forecast_hook(prices, cfg)
        if view is None:
            view = momentum_view_panel(prices, cfg)
    else:
        view = momentum_view_panel(prices, cfg)
    assets = list(cfg.risk_assets)
    px = prices[assets].astype(float)
    rets = px.pct_change()
    full_idx = rets.dropna().index
    dates = full_idx.intersection(px.loc[start:].index).intersection(gate.index)
    R_all = rets.loc[full_idx, assets].to_numpy(float)
    return dynamic_risk_frame(prices, cfg, gate=gate, view=view, dates=dates,
                              full_idx=full_idx, R_all=R_all)


def demo() -> None:
    """Self-check: neutral -> 1.0; strong+calm -> >1; weak+stressed -> <1; severe -> floor."""
    from dataclasses import replace
    import sys
    sys.path.insert(0, "tests")
    from strat_fixtures import make_synthetic_prices

    cfg = replace(AllocatorConfig(), dynamic_risk_enabled=True)

    # --- recurrence invariants (the load-bearing state machine) -----------------
    # Cold start with an all-neutral target and no crisis stays exactly 1.0.
    N = 300
    neutral = np.ones(N)
    no_sev = np.zeros(N, bool)
    noR = np.zeros(N)
    m = _reconstruct_risk_multiplier(neutral, no_sev, noR, cfg)
    assert np.allclose(m, 1.0), m[:5]

    # A sustained LOW target de-risks fast (a few days), no confirmation needed.
    low = np.full(N, cfg.dynamic_risk_min_mult)
    m = _reconstruct_risk_multiplier(low, no_sev, noR, cfg)
    assert m[3] < m[0] and m[10] < 0.7, (m[0], m[3], m[10])   # eager + fast down
    assert m[-1] <= cfg.dynamic_risk_min_mult + 1e-9

    # A sustained HIGH target re-risks SLOWLY and only after confirm_days.
    high = np.full(N, cfg.dynamic_risk_max_mult)
    k = int(cfg.dynamic_risk_confirm_days)
    m = _reconstruct_risk_multiplier(high, no_sev, noR, cfg)
    assert m[k - 2] == 1.0        # still neutral before confirmation completes
    assert m[k - 1] > 1.0         # first step the day up_run reaches confirm_days
    assert m[-1] <= cfg.dynamic_risk_max_mult + 1e-9
    # Recovery accelerates the climb: same high target, R=1 takes a bigger first step.
    mR = _reconstruct_risk_multiplier(high, no_sev, np.ones(N), cfg)
    assert mR[k - 1] > m[k - 1]

    # SevereBear snaps to the floor and bypasses the rate limit, even from 1.0.
    sev = no_sev.copy(); sev[5] = True
    m = _reconstruct_risk_multiplier(neutral, sev, noR, cfg)
    assert abs(m[5] - cfg.dynamic_risk_severe_floor) < 1e-9

    # --- map monotonicity + asymmetry ------------------------------------------
    up = _map_net_to_mult(np.array([cfg.dynamic_risk_deadband_up + 0.2]), cfg)[0]
    dn = _map_net_to_mult(np.array([-(cfg.dynamic_risk_deadband_down + 0.2)]), cfg)[0]
    assert up > 1.0 and dn < 1.0
    # a net that would ADD risk if the deadband were symmetric stays at 1.0 because
    # deadband_up is wider than deadband_down -- risk > 1.0 is harder to earn.
    assert _map_net_to_mult(np.array([cfg.dynamic_risk_deadband_down + 0.05]), cfg)[0] == 1.0
    assert _map_net_to_mult(np.array([-(cfg.dynamic_risk_deadband_down + 0.05)]), cfg)[0] < 1.0

    # --- full frame on synthetic data ------------------------------------------
    prices = make_synthetic_prices()
    fr = build_risk_frame(prices, cfg)
    assert {"opportunity_score", "stress_score", "risk_multiplier"}.issubset(fr.columns)
    mm = fr["risk_multiplier"].to_numpy(float)
    assert np.all(mm >= cfg.dynamic_risk_min_mult - 1e-9)
    assert np.all(mm <= cfg.dynamic_risk_max_mult + 1e-9)
    svd = fr["severe"].to_numpy(bool)
    assert np.allclose(mm[svd], cfg.dynamic_risk_severe_floor)   # every crash day at floor
    # OFF config must be exactly neutral everywhere (byte-identical guarantee).
    off = build_risk_frame(prices, replace(cfg, dynamic_risk_enabled=False))
    # (the controller is only READ when enabled; the frame still computes, but the
    # wired paths guard on cfg.dynamic_risk_enabled -- so this just sanity-bounds it.)
    assert off["risk_multiplier"].between(cfg.dynamic_risk_min_mult,
                                          cfg.dynamic_risk_max_mult).all()

    print(f"dynamic_risk demo OK: mult range [{mm.min():.3f}, {mm.max():.3f}], "
          f"mean {mm.mean():.3f}; O mean {fr['opportunity_score'].mean():.2f}, "
          f"S mean {fr['stress_score'].mean():.2f}, severe days {int(svd.sum())}")


if __name__ == "__main__":
    demo()
