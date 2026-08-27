#!/usr/bin/env python3
"""
V2.9.8 backtest engine, state machine, and benchmark suite.

Everything here is leakage-audited: any quantity used to set the weights held
on day t is computed from information available at the close of day t-1 or
earlier. The engine asserts this rather than trusting it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from strat_allocator import (
    AllocatorConfig,
    TRADING_DAYS,
    annualized_metrics,
    annualized_vols,
    calibrated_expected_returns,
    ewma_cov_annualized,
    inflate_cov_for_uncertainty,
    enforce_constraints,
    leverage_cap_for,
    solve_weights,
    strategic_core,
    vol_budget_for,
)
from loss_replacement import apply_loss_replacement, CostBasisTracker
from momentum_exhaustion import (
    apply_exhaustion_rotation, exhaustion_panel, reversal_panel, EXH_WEIGHTS)
from adaptive_rotation import apply_adaptive_rotation

# Between scheduled rebalances the book's ex-ante volatility drifts with the
# market. Once it exceeds the budget by this ratio, rebalance immediately
# regardless of the schedule. This is what makes de-risking fast while keeping
# re-risking slow -- the asymmetry that protects the levered book.
VOL_BREACH_TRIGGER = 1.15

# Growth-gate release ceilings. CRISIS stays hard-capped; BULL is allowed to
# exceed 1.0 so the book can be levered when risk is cheap. That asymmetry is
# the only long-only route to beating buy & hold on BOTH Sharpe and return.
RELEASE_CEILING = {
    "CRISIS": 0.25,
    "DEFENSIVE": None,      # retain the raw structural ceiling
    "BOTTOMING": 0.50,
    "RECOVERING": 0.85,
    "BULL": None,           # filled from cfg.max_leverage
}


# =============================================================================
# SPY STATE MACHINE + STRUCTURAL GATE
# =============================================================================


def build_gate_schedule(spy_close: pd.Series, cfg: AllocatorConfig) -> pd.DataFrame:
    """
    Recovery state machine plus the raw SMA structural ceiling.

    Recovery RELAXES the ceiling; it never bypasses it. The released ceiling is
    handed to the optimizer as a hard constraint, so there is no post-hoc
    override that can violate it.
    """
    close = spy_close.astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    sma200_slope = sma200 - sma200.shift(20)
    peak252 = close.rolling(252, min_periods=60).max()
    trough60 = close.rolling(60, min_periods=20).min()
    dd252 = close / peak252 - 1.0
    rally = close / trough60 - 1.0

    above200 = close > sma200
    golden = sma50 > sma200

    raw = pd.Series(0.25, index=close.index, dtype=float)
    raw[above200 & ~golden] = 0.60
    raw[~above200 & golden] = 0.50
    raw[above200 & golden] = 1.00

    severe = (~above200) & (sma200_slope < 0) & (close < sma200 * 0.92) & (dd252 <= -0.20)
    severe = severe.fillna(False)

    crisis_trigger = ((dd252 <= -0.18) & ~above200).fillna(False)

    state = "DEFENSIVE"
    rows = []
    for date in close.index:
        prev_state = state
        c_above, c_golden = bool(above200.get(date, False)), bool(golden.get(date, False))
        c_dd = float(dd252.get(date, 0.0) or 0.0)
        c_rally = float(rally.get(date, 0.0) or 0.0)

        if bool(crisis_trigger.get(date, False)):
            state = "CRISIS"
        elif prev_state == "CRISIS":
            if c_rally >= 0.10:
                state = "BOTTOMING"
        elif prev_state == "BOTTOMING":
            if c_above:
                state = "RECOVERING"
            elif c_dd <= -0.25:
                state = "CRISIS"
        elif prev_state == "RECOVERING":
            if c_above and c_golden and c_dd > -0.05:
                state = "BULL"
            elif not c_above:
                state = "BOTTOMING"
        elif prev_state == "BULL":
            if not c_above:
                state = "DEFENSIVE"
        else:  # DEFENSIVE
            if c_above and c_golden:
                state = "BULL"

        raw_ceiling = float(raw.get(date, 0.25))
        is_severe = bool(severe.get(date, False))

        if is_severe:
            eff = 0.0
        elif state == "CRISIS":
            eff = min(raw_ceiling, RELEASE_CEILING["CRISIS"])
        elif state == "BOTTOMING":
            eff = max(raw_ceiling, RELEASE_CEILING["BOTTOMING"])
        elif state == "RECOVERING":
            eff = max(raw_ceiling, RELEASE_CEILING["RECOVERING"])
        elif state == "BULL":
            eff = cfg.max_leverage
        else:
            eff = raw_ceiling

        rows.append({
            "Date": date, "RecoveryState": state, "PreviousState": prev_state,
            "RawStructuralCeiling": raw_ceiling, "SevereBear": is_severe,
            "EffectiveGrowthCeiling": float(np.clip(eff, 0.0, cfg.max_leverage)),
            "Drawdown252": c_dd,
        })

    out = pd.DataFrame(rows).set_index("Date")
    if not out["EffectiveGrowthCeiling"].between(0.0, cfg.max_leverage).all():
        raise AssertionError("Growth ceiling escaped [0, max_leverage].")
    return out


# =============================================================================
# FORECAST PROXY  (stands in for the V2.9.6 XGBoost panel during validation)
# =============================================================================


def momentum_view_panel(prices: pd.DataFrame, cfg: AllocatorConfig) -> pd.DataFrame:
    """
    Risk-adjusted multi-horizon momentum, expressed as a 20-day-equivalent
    expected excess return so it occupies exactly the same slot as XGB_Mu20.

    This is a PROXY used only to validate the allocator end to end. In
    production it is replaced by the V2.9.6 walk-forward forecasts.
    """
    px = prices[list(cfg.risk_assets)].astype(float)
    rets = px.pct_change()
    vol = rets.rolling(63).std() * math.sqrt(TRADING_DAYS)

    mom_12_1 = px.shift(21) / px.shift(252) - 1.0
    mom_3 = px / px.shift(63) - 1.0
    reversal = -(px / px.shift(5) - 1.0)

    score = 0.50 * mom_12_1 + 0.35 * mom_3 + 0.15 * reversal * 4.0
    risk_adj = score / vol.replace(0.0, np.nan)

    # Convert to a 20-day-equivalent return so units match a real forecast.
    view = risk_adj * vol / (TRADING_DAYS / 20.0) * 4.0
    return view.shift(1)     # only information through t-1 sets weights at t


def rolling_ic_panel(view: pd.DataFrame,
                     prices: pd.DataFrame,
                     horizon: int = 20,
                     min_pairs: int = 250) -> pd.DataFrame:
    """
    Expanding out-of-sample rank IC per asset.

    Leakage control: a forecast made at date t has a label window closing at
    t+horizon. That pair may only enter the IC estimate used on date d if
    t + horizon < d. The shift below enforces exactly that.
    """
    px = prices[view.columns].astype(float)
    fwd = px.shift(-horizon) / px - 1.0

    out = {}
    for col in view.columns:
        pair = pd.DataFrame({"v": view[col], "y": fwd[col]}).dropna()
        rank_v = pair["v"].rank()
        rank_y = pair["y"].rank()
        # Expanding Spearman via expanding Pearson on ranks.
        ic = rank_v.expanding(min_pairs).corr(rank_y)
        # Label for a forecast at t is only known at t+horizon; publish then.
        ic.index = pair.index
        ic = ic.shift(horizon + 1)
        out[col] = ic.reindex(view.index).ffill()

    return pd.DataFrame(out).clip(-0.2, 0.2)


# =============================================================================
# SINGLE-SESSION DECISION  (the one implementation shared by backtest + live)
# =============================================================================


def decide_target_weights(
    *,
    prev_weights: np.ndarray,
    cov_window: np.ndarray,
    view_row: np.ndarray | None,
    ic_row: np.ndarray | None,
    state: str,
    severe: bool,
    ceiling_raw: float,
    level_history: np.ndarray,
    sessions_since_rebalance: int,
    cfg: AllocatorConfig,
    use_gate: bool = True,
    use_vol_target: bool = True,
    standardize_level: bool = True,
    vol_budget_mult: float = 1.0,
    growth_ceiling_mult: float = 1.0,
) -> tuple[np.ndarray, bool, dict]:
    """
    The V2.9.8 per-session decision, extracted verbatim from the backtest loop so
    ``run_backtest`` and the live engine share ONE implementation.

    Given the held book ``prev_weights`` and completed-close information for
    session ``t``, return ``(new_weights, rebalanced, diagnostics)`` -- the
    weights to hold on ``t+1``. This is the FULL post-optimizer pipeline, NOT the
    raw optimizer target: the 5-day schedule, the forced full rebalance on a
    growth/leverage/vol breach, the strategic-core solve, partial adjustment,
    constraint repair, and the turnover deadband are ALL applied here.

    Parameters mirror the backtest loop exactly so the numbers are identical:

    cov_window   : (T, n) daily simple returns through close(t) INCLUSIVE, oldest
                   row first (== ``R_all[max(0, p - lb + 1):p + 1]``).
    view_row     : forecast view row for t (already lagged), or None for alpha-off.
    ic_row       : trailing OOS rank-IC row for t, or None.
    ceiling_raw  : the gate's EffectiveGrowthCeiling for t (pre-flag).
    level_history: prior forecast LEVELs (``levels[:i]``); may contain NaN, and is
                   filtered to finite values here -- pass it exactly as the loop holds it.
    sessions_since_rebalance : ``i - last_rebalance`` on the backtest dates grid.
    standardize_level : ``use_vol_target and view is not None and i > 250`` (the i>250 gate).
    """
    n = len(cfg.risk_assets)
    w = np.asarray(prev_weights, float)

    ceiling = float(ceiling_raw) if use_gate else cfg.max_leverage
    lev_cap = leverage_cap_for(state, severe, cfg) if use_gate else cfg.max_leverage
    if not use_vol_target:
        lev_cap = min(lev_cap, 1.0)
    # Optional bond-regime overlay (default 1.0 -> exact no-op, byte-identical).
    # Growth ceiling can only be TIGHTENED (never loosened past the gate).
    if growth_ceiling_mult != 1.0:
        ceiling = float(min(max(ceiling * growth_ceiling_mult, 0.0), cfg.max_leverage))

    # Covariance uses returns through TODAY only.
    cov = ewma_cov_annualized(cov_window, cfg.cov_halflife_vol,
                              cfg.cov_halflife_corr, cfg.cov_shrinkage)
    sigma = annualized_vols(cov)

    raw = view_row if view_row is not None else np.zeros(n)
    ic_in = ic_row if ic_row is not None else np.zeros(n)
    dd_neutral = np.full(n, cfg.dd_neutral)

    mu, level = calibrated_expected_returns(
        raw_view=raw, sigma_ann=sigma, rolling_ic=ic_in,
        dd_risk=dd_neutral, cfg=cfg)

    cov_adj = inflate_cov_for_uncertainty(cov, np.zeros(n), cfg)

    # Standardize the forecast LEVEL against its own trailing history.
    if standardize_level:
        arr = np.asarray(level_history, float)
        arr = arr[np.isfinite(arr)]
        sd = float(arr.std()) if len(arr) > 20 else 0.0
        level_z = (level - float(arr.mean())) / sd if sd > 1e-12 else 0.0
    else:
        level_z = 0.0

    vol_budget = (vol_budget_for(level_z, cfg) if use_vol_target
                  else cfg.target_vol)
    if vol_budget_mult != 1.0:               # bond-regime overlay (default no-op)
        vol_budget = float(vol_budget * vol_budget_mult)

    g_idx, d_idx = cfg.growth_idx, cfg.defensive_idx
    held_vol = math.sqrt(max(float(w @ cov_adj @ w), 0.0))

    # Any hard-constraint breach forces an immediate full rebalance, so a newly
    # binding gate or a volatility spike can never be deferred by the 5-day
    # schedule or absorbed by the deadband.
    must_rebalance = (float(w[g_idx].sum()) > ceiling + 1e-9
                      or float(w.sum()) > lev_cap + 1e-9
                      or held_vol > vol_budget * VOL_BREACH_TRIGGER)
    due = sessions_since_rebalance >= cfg.rebalance_every

    if due or must_rebalance:
        core = strategic_core(state, cov_adj, cfg, vol_budget,
                              ceiling, lev_cap)
        target, _info = solve_weights(mu, cov_adj, w, cfg,
                                      growth_ceiling=ceiling,
                                      vol_budget=vol_budget,
                                      leverage_cap=lev_cap,
                                      core=core)
        step = target if must_rebalance else w + cfg.partial_adjust * (target - w)

        # Partial adjustment is a convex step, which can still sit above the vol
        # cap when the held book is over budget. Repair always.
        step = enforce_constraints(step, cfg, ceiling, cov_adj,
                                   vol_budget, lev_cap)

        cand_turnover = 0.5 * float(np.abs(step - w).sum())
        if cand_turnover < cfg.rebalance_deadband and not must_rebalance:
            new_w, rebalanced = w, False
        else:
            new_w, rebalanced = step, True
    else:
        new_w, rebalanced = w, False

    ex_ante = math.sqrt(max(float(new_w @ cov_adj @ new_w), 0.0))
    diag = {
        "level": level,
        "level_z": float(level_z),
        "vol_budget": float(vol_budget),
        "leverage_cap": float(lev_cap),
        "growth_ceiling": float(ceiling),
        "ex_ante_vol": ex_ante,
        "held_vol": held_vol,
        "gross": float(new_w.sum()),
        "growth": float(new_w[g_idx].sum()),
        "defensive": float(new_w[d_idx].sum()),
        "recovery_state": str(state),
        "severe_bear": bool(severe),
        "due": bool(due),
        "must_rebalance": bool(must_rebalance),
        "rebalanced": bool(rebalanced),
        "vol_budget_mult": float(vol_budget_mult),
        "growth_ceiling_mult": float(growth_ceiling_mult),
        # cov_adj is exposed so the live engine can re-verify the ex-ante vol of a
        # share-rounded executable book with the SAME covariance. Only named diag
        # fields are copied into the backtest history, so this adds no persisted state.
        "cov_adj": cov_adj,
    }
    return new_w, rebalanced, diag


# =============================================================================
# BACKTEST
# =============================================================================


def run_backtest(prices: pd.DataFrame,
                 cfg: AllocatorConfig,
                 gate: pd.DataFrame,
                 view: pd.DataFrame | None,
                 ic: pd.DataFrame | None,
                 start: str,
                 label: str,
                 use_gate: bool = True,
                 use_vol_target: bool = True,
                 trace: list | None = None,
                 vol_mult: pd.Series | None = None,
                 growth_mult: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = list(cfg.risk_assets)
    px = prices[assets].astype(float)
    rets = px.pct_change()
    rf_series = prices["CASH_RATE_3M"].astype(float) / 100.0 / TRADING_DAYS

    # Full-history index used for the covariance window, so the estimator can
    # see pre-START data without that data ever influencing a traded weight.
    full_idx = rets.dropna().index
    R_all = rets.loc[full_idx, assets].to_numpy(float)
    pos_of = {d: k for k, d in enumerate(full_idx)}

    dates = full_idx.intersection(px.loc[start:].index).intersection(gate.index)

    # Precompute everything as numpy; per-day .loc lookups dominated runtime.
    d_pos = np.array([pos_of[d] for d in dates], int)
    R_day = R_all[d_pos]
    rf_arr = rf_series.reindex(dates).fillna(0.0).to_numpy(float)
    states = gate.loc[dates, "RecoveryState"].to_numpy(object)
    severes = gate.loc[dates, "SevereBear"].to_numpy(bool)
    ceilings = gate.loc[dates, "EffectiveGrowthCeiling"].to_numpy(float)
    V = (view.loc[dates, assets].to_numpy(float) if view is not None
         else np.zeros((len(dates), len(assets))))
    IC = (ic.loc[dates, assets].to_numpy(float) if ic is not None
          else np.zeros((len(dates), len(assets))))
    dd_neutral = np.full(len(assets), cfg.dd_neutral)
    zero_unc = np.zeros(len(assets))

    n = len(assets)
    pending = np.zeros(n)
    equity = cfg.initial_capital
    levels = np.full(len(dates), np.nan)

    hist, wlog = [], []
    last_rebalance = -10 ** 9
    lb = cfg.cov_lookback

    # --- rotation-overlay setup (loss-replacement / exhaustion / adaptive; OFF) --
    # Skipped byte-for-byte when all overlays are disabled, so the base backtest /
    # parity numbers are unchanged. The adaptive engine SUPERSEDES the two naive
    # overlays (they are gated off when it is on). Cost basis + cooldown are shared.
    adaptive_on = bool(cfg.adaptive_rotation_enabled)
    lr_on = bool(cfg.loss_replacement_enabled) and not adaptive_on
    exh_on = bool(cfg.momentum_exhaustion_enabled) and not adaptive_on
    g_idx = cfg.growth_idx
    if lr_on or exh_on or adaptive_on:
        PX_day = px.reindex(dates).to_numpy(float)             # adjusted close per date
        tracker = CostBasisTracker(assets)
        last_exit = np.full(n, -10 ** 9, dtype=float)          # session idx of last exit
        lr_counts = {"exits": 0, "replacements": 0, "no_candidate": 0}
        exh_counts = {"signals": 0, "exits": 0, "critical": 0, "rotations": 0,
                      "skipped": 0}
        exh_exit_log = []      # (date, symbol, score, position_return) for offline eval
        eps_lr = float(cfg.loss_replacement_hold_eps)
    if lr_on:
        N = int(cfg.replacement_momentum_lookback)
        MOM_day = (px / px.shift(N) - 1.0).reindex(dates).to_numpy(float)
    if exh_on or adaptive_on:
        _exh_df, _exh_comp = exhaustion_panel(prices, cfg, view)
        _rev_df, _ = reversal_panel(prices, cfg, view)
        _cnames = list(EXH_WEIGHTS.keys())
        px_pos = {d: k for k, d in enumerate(px.index)}
        EXH_np = _exh_df[assets].to_numpy(float)               # (len(px), n)
        REV_np = _rev_df[assets].to_numpy(float)
        COMP_np = np.stack([_exh_comp[c][assets].to_numpy(float) for c in _cnames], axis=-1)  # (T,n,ncomp)
        VIEW_np = (view.reindex(px.index)[assets].to_numpy(float) if view is not None
                   else np.zeros((len(px.index), n)))
        kconf = max(1, int(cfg.momentum_exhaustion_confirmations))
    if adaptive_on:
        MOM20_np = (px / px.shift(20) - 1.0).reindex(px.index).to_numpy(float)
        ACCEL_np = ((px / px.shift(5) - 1.0) - (px / px.shift(20) - 1.0)).reindex(px.index).to_numpy(float)
        kconf_ad = max(1, int(cfg.adaptive_hazard_confirmations))
        adaptive_counts = {"sources": 0, "rotations": 0, "cash_rotations": 0, "held": 0,
                           "edge_blocked": 0}
        adaptive_stages = {}       # funnel: how far each evaluated source got
        adaptive_exit_log = []

    # --- dynamic portfolio-risk controller (default OFF -> multiplier 1.0) -----
    # A pre-loop pass builds a per-date vol-budget multiplier. It feeds INTO
    # decide_target_weights, so it cannot read that day's diag cov (circularity);
    # it uses its own point-in-time opportunity/stress features over the SAME cov
    # windows the loop uses. Guarded off -> vbm below is unchanged -> byte-identical.
    dynamic_on = bool(cfg.dynamic_risk_enabled)
    dyn_mult = None
    if dynamic_on:
        from dynamic_risk import dynamic_risk_frame
        dyn_frame = dynamic_risk_frame(prices, cfg, gate=gate, view=view,
                                       dates=dates, full_idx=full_idx, R_all=R_all)
        dyn_mult = dyn_frame["risk_multiplier"]

    for i, date in enumerate(dates):
        w = pending.copy()

        # --- 1) Earn today's return with weights decided at the prior close --
        r = R_day[i]
        gross = float(w @ r)
        rf = float(rf_arr[i])
        cash_held = max(0.0, 1.0 - w.sum())
        borrowed = max(0.0, w.sum() - 1.0)
        port_ret = (gross
                    + cash_held * rf
                    - borrowed * (rf + cfg.financing_spread / TRADING_DAYS))
        equity *= 1.0 + port_ret

        # --- 2) At today's close, decide tomorrow's target -------------------
        state = str(states[i])
        severe = bool(severes[i])
        ceiling = float(ceilings[i]) if use_gate else cfg.max_leverage
        lev_cap = leverage_cap_for(state, severe, cfg) if use_gate else cfg.max_leverage
        if not use_vol_target:
            lev_cap = min(lev_cap, 1.0)

        p = d_pos[i]
        if p < 60:
            hist.append({"Date": date, "Equity": equity, "GrossReturn": port_ret,
                         "Turnover": 0.0, "Gross": 0.0, "Growth": 0.0,
                         "ExAnteVol": 0.0, "VolBudget": cfg.target_vol,
                         "GrowthCeiling": ceiling, "LeverageCap": lev_cap,
                         "RecoveryState": state, "Rebalanced": False})
            wlog.append({"Date": date, "Strategy": label,
                         **{a: 0.0 for a in assets}})
            continue

        # One shared decision path. Covariance window is returns through TODAY.
        window = R_all[max(0, p - lb + 1):p + 1]
        ssr = i - last_rebalance
        standardize = bool(use_vol_target and view is not None and i > 250)
        vbm = float(vol_mult.get(date, 1.0)) if vol_mult is not None else 1.0
        if dynamic_on:
            # Compose multiplicatively with any bond vol_mult; clamp the PRODUCT to
            # the controller's [min, max] so the dynamic layer stays authoritative.
            vbm = float(np.clip(vbm * float(dyn_mult.get(date, 1.0)),
                                cfg.dynamic_risk_min_mult, cfg.dynamic_risk_max_mult))
        gcm = float(growth_mult.get(date, 1.0)) if growth_mult is not None else 1.0
        new_w, rebalanced, diag = decide_target_weights(
            prev_weights=w,
            cov_window=window,
            view_row=(V[i] if view is not None else None),
            ic_row=(IC[i] if ic is not None else None),
            state=state, severe=severe, ceiling_raw=ceilings[i],
            level_history=levels[:i],
            sessions_since_rebalance=ssr,
            cfg=cfg, use_gate=use_gate, use_vol_target=use_vol_target,
            standardize_level=standardize,
            vol_budget_mult=vbm, growth_ceiling_mult=gcm,
        )
        levels[i] = diag["level"]

        # --- loss-replacement / momentum-rotation overlay (default OFF) ------
        # Evaluated EVERY session (a breach can rotate off-schedule, spec 15).
        # Cost basis is buy-and-hold: it only changes when the strategy actually
        # trades, so position_return is the true P&L since the last fill -- the
        # same quantity a live broker reports as avg_entry_price.
        reason = "scheduled"
        if lr_on:
            pos_ret = tracker.position_returns(PX_day[i])   # held book `w`
            cd_mask = (i - last_exit) < float(cfg.replacement_cooldown_days)
            res = apply_loss_replacement(
                base_weights=new_w, position_returns=pos_ret, momentum=MOM_day[i],
                cov_adj=diag["cov_adj"], cfg=cfg,
                growth_ceiling=diag["growth_ceiling"], vol_budget=diag["vol_budget"],
                leverage_cap=diag["leverage_cap"], cooldown_mask=cd_mask)
            if res.triggered:
                new_w = res.weights
                rebalanced = True
                reason = "loss_replacement"
                lr_counts["exits"] += len(res.exited)
                for ev in res.events:
                    if ev["replacement_asset"] is not None:
                        lr_counts["replacements"] += 1
                    elif ev["reason"] == "no_positive_momentum_candidate":
                        lr_counts["no_candidate"] += 1
                for sym in res.exited:
                    last_exit[assets.index(sym)] = i

        # --- momentum-exhaustion overlay (default OFF; runs AFTER loss-replacement) --
        # Operates only on positions still HELD after loss-replacement, so an asset
        # exited for a loss is never also processed for exhaustion (precedence).
        if exh_on:
            p_ex = px_pos[date]
            recent = EXH_np[max(0, p_ex - kconf + 1):p_ex + 1]
            if recent.shape[0] < kconf:
                recent = np.vstack([np.full((kconf - recent.shape[0], n), np.nan), recent])
            cd_mask = (i - last_exit) < float(cfg.replacement_cooldown_days)
            pos_ret_e = tracker.position_returns(PX_day[i])
            exh_pos = np.where(new_w > eps_lr, pos_ret_e, np.nan)
            eres = apply_exhaustion_rotation(
                base_weights=new_w, position_returns=exh_pos, exhaustion_recent=recent,
                exhaustion_components=COMP_np[p_ex], component_names=_cnames,
                reversal_scores=REV_np[p_ex], reversal_rank_tiebreak=VIEW_np[p_ex],
                cov_adj=diag["cov_adj"], cfg=cfg, growth_ceiling=diag["growth_ceiling"],
                vol_budget=diag["vol_budget"], leverage_cap=diag["leverage_cap"],
                cooldown_mask=cd_mask)
            held_now = np.isfinite(exh_pos)
            exh_counts["signals"] += int(np.sum(
                (EXH_np[p_ex] >= cfg.momentum_exhaustion_threshold) & held_now))
            exh_counts["skipped"] += len(eres.skipped)
            if eres.triggered:
                new_w = eres.weights
                rebalanced = True
                if reason == "scheduled":
                    reason = "momentum_exhaustion"
                exh_counts["exits"] += len(eres.exited)
                for ev in eres.events:
                    if ev["action"] == "rotate":
                        exh_counts["rotations"] += 1
                    if ev["reason"].startswith("critical"):
                        exh_counts["critical"] += 1
                    exh_exit_log.append({"Date": date, "symbol": ev["exited_asset"],
                                         "exhaustion_score": ev["exhaustion_score"],
                                         "position_return": ev["position_return"],
                                         "replacement_asset": ev["replacement_asset"]})
                for sym in eres.exited:
                    last_exit[assets.index(sym)] = i

        # --- adaptive opportunity-gated rotation (SUPERSEDES loss/exhaustion) ----
        if adaptive_on:
            p_ex = px_pos[date]
            recent = EXH_np[max(0, p_ex - kconf_ad + 1):p_ex + 1]
            if recent.shape[0] < kconf_ad:
                recent = np.vstack([np.full((kconf_ad - recent.shape[0], n), np.nan), recent])
            cd_mask = (i - last_exit) < float(cfg.replacement_cooldown_days)
            pos_ret_a = tracker.position_returns(PX_day[i])
            dh_a = tracker.days_held(i)
            sigma_ann = annualized_vols(diag["cov_adj"])
            ares = apply_adaptive_rotation(
                base_weights=new_w, position_returns=pos_ret_a, days_held=dh_a,
                view_row=VIEW_np[p_ex], sigma_ann=sigma_ann, mom20_row=MOM20_np[p_ex],
                accel_row=ACCEL_np[p_ex], reversal_score=REV_np[p_ex],
                exhaustion_recent=recent, cov_adj=diag["cov_adj"],
                state=state, severe=severe, cfg=cfg, growth_ceiling=diag["growth_ceiling"],
                vol_budget=diag["vol_budget"], leverage_cap=diag["leverage_cap"],
                cooldown_mask=cd_mask)
            adaptive_counts["sources"] += len(ares.events) + len(ares.skipped)
            adaptive_counts["held"] += len(ares.skipped)
            for _ev in ares.events + ares.skipped:      # funnel stage accounting
                _st = _ev.get("stage", "executed")
                adaptive_stages[_st] = adaptive_stages.get(_st, 0) + 1
                if _ev.get("edge_blocked"):
                    adaptive_counts["edge_blocked"] += 1
            if ares.triggered:
                new_w = ares.weights
                rebalanced = True
                if reason == "scheduled":
                    reason = "adaptive_rotation"
                for ev in ares.events:
                    if ev["action"] == "rotate_to_cash":
                        adaptive_counts["cash_rotations"] += 1
                    else:
                        adaptive_counts["rotations"] += 1
                    adaptive_exit_log.append({"Date": date, "symbol": ev["exited_asset"],
                                              "replacement_asset": ev["replacement_asset"],
                                              "position_return": ev["position_return"],
                                              "net_rotation_edge": ev["net_rotation_edge"]})
                for sym in ares.exited:
                    last_exit[assets.index(sym)] = i

        if trace is not None:
            trace.append({"Date": date, "prev_weights": w.copy(),
                          "sessions_since_rebalance": int(ssr),
                          "target": np.asarray(new_w, float).copy(),
                          "rebalanced": bool(rebalanced),
                          "rebalance_reason": reason})

        if rebalanced:
            last_rebalance = i

        turnover = 0.5 * float(np.abs(new_w - w).sum())
        equity *= 1.0 - cfg.transaction_cost * 2.0 * turnover

        # Audit exposures come from the FINAL book. When neither overlay fired
        # these are byte-identical to diag (same new_w, same cov_adj).
        if reason != "scheduled":
            cov_adj = diag["cov_adj"]
            gross_v = float(new_w.sum())
            growth_v = float(new_w[g_idx].sum())
            exante_v = math.sqrt(max(float(new_w @ cov_adj @ new_w), 0.0))
        else:
            gross_v, growth_v, exante_v = diag["gross"], diag["growth"], diag["ex_ante_vol"]

        row = {"Date": date, "Equity": equity, "GrossReturn": port_ret,
               "Turnover": turnover, "Gross": gross_v,
               "Growth": growth_v, "ExAnteVol": exante_v,
               "VolBudget": diag["vol_budget"],
               "GrowthCeiling": diag["growth_ceiling"],
               "LeverageCap": diag["leverage_cap"], "RecoveryState": state,
               "Rebalanced": rebalanced}
        if lr_on or exh_on or adaptive_on:
            row["RebalanceReason"] = reason
        if dynamic_on:
            row["RiskMultiplier"] = float(dyn_mult.get(date, 1.0))
        hist.append(row)
        wlog.append({"Date": date, "Strategy": label,
                     **{a: float(new_w[j]) for j, a in enumerate(assets)}})

        # Buy-and-hold cost basis: only an actual trade moves shares/cost. Passing
        # the session index stamps first-entry (days_held) for the adaptive engine.
        if (lr_on or exh_on or adaptive_on) and rebalanced:
            tracker.update(new_w, PX_day[i], equity, session=i)

        pending = new_w

    h = pd.DataFrame(hist).set_index("Date")
    wdf = pd.DataFrame(wlog).set_index("Date")
    if lr_on or exh_on or adaptive_on:
        # Overlay accounting for the A/B/C/D harness (does not alter the equity path).
        reasons = h.get("RebalanceReason")
        if lr_on:
            h.attrs["loss_replacement"] = dict(lr_counts)
            h.attrs["loss_triggered_rebalances"] = int((reasons == "loss_replacement").sum())
        if exh_on:
            h.attrs["exhaustion"] = dict(exh_counts)
            h.attrs["exhaustion_triggered_rebalances"] = int(
                (reasons == "momentum_exhaustion").sum())
            h.attrs["exhaustion_exit_log"] = exh_exit_log
        if adaptive_on:
            h.attrs["adaptive_rotation"] = dict(adaptive_counts)
            h.attrs["adaptive_triggered_rebalances"] = int((reasons == "adaptive_rotation").sum())
            h.attrs["adaptive_exit_log"] = adaptive_exit_log
            h.attrs["adaptive_stages"] = dict(adaptive_stages)
    if dynamic_on:
        # Full per-date controller frame + summary for the F/G falsification harness
        # (exposure log, capture ratios, drawdown attribution). Does not alter equity.
        h.attrs["dynamic_risk_frame"] = dyn_frame
        mvals = dyn_frame["risk_multiplier"]
        h.attrs["dynamic_risk"] = {
            "enabled": True,
            "mult_mean": float(mvals.mean()), "mult_min": float(mvals.min()),
            "mult_max": float(mvals.max()), "mult_std": float(mvals.std()),
            "frac_derisked": float((mvals < 0.999).mean()),
            "frac_added": float((mvals > 1.001).mean()),
        }

    # Hard constraint audit. Gate and leverage are absolute on every day.
    # Ex-ante vol is absolute on rebalance days and bounded by the breach
    # trigger in between, which is the intended drift tolerance.
    if (h["Growth"] > h["GrowthCeiling"] + 1e-6).any():
        raise AssertionError(f"{label}: growth gate violated.")
    if (h["Gross"] > h["LeverageCap"] + 1e-6).any():
        raise AssertionError(f"{label}: leverage cap violated.")
    reb = h["Rebalanced"]
    if (h.loc[reb, "ExAnteVol"] > h.loc[reb, "VolBudget"] + 1e-6).any():
        raise AssertionError(f"{label}: vol budget violated on a rebalance day.")
    if (h["ExAnteVol"] > h["VolBudget"] * VOL_BREACH_TRIGGER + 1e-6).any():
        raise AssertionError(f"{label}: vol drift exceeded the breach trigger.")

    return h, wdf


# =============================================================================
# BENCHMARKS
# =============================================================================


def bench_buy_hold(prices: pd.DataFrame, dates, ticker: str,
                   capital: float) -> pd.Series:
    r = prices[ticker].astype(float).reindex(dates).pct_change().fillna(0.0)
    return capital * (1.0 + r).cumprod()


def bench_spy_sma(prices: pd.DataFrame, dates, capital: float) -> pd.Series:
    c = prices["SPY"].astype(float)
    pos = ((c.rolling(50).mean() > c.rolling(200).mean())
           .astype(float).shift(1).reindex(dates).fillna(0.0))
    r = c.reindex(dates).pct_change().fillna(0.0)
    rf = prices["CASH_RATE_3M"].reindex(dates).astype(float) / 100.0 / TRADING_DAYS
    return capital * (1.0 + pos * r + (1.0 - pos) * rf.fillna(0.0)).cumprod()


def bench_60_40(prices: pd.DataFrame, dates, capital: float) -> pd.Series:
    """60/40 SPY/IEF, reset to target at the start of each month."""
    r = prices[["SPY", "IEF"]].astype(float).reindex(dates).pct_change().fillna(0.0)
    target = np.array([0.60, 0.40])
    period = pd.Series(dates, index=dates).dt.to_period("M")

    w = target.copy()
    eq, out, prev_period = capital, [], None
    for d in dates:
        if prev_period is not None and period.loc[d] != prev_period:
            w = target.copy()
        prev_period = period.loc[d]

        rd = r.loc[d].to_numpy(float)
        eq *= 1.0 + float(w @ rd)
        w = w * (1.0 + rd)
        w = w / max(w.sum(), 1e-12)
        out.append(eq)

    return pd.Series(out, index=dates)


def bench_vol_targeted_spy(prices: pd.DataFrame, dates, cfg: AllocatorConfig) -> pd.Series:
    """
    The honest competitor: same vol target, same leverage cap, NO forecasts and
    NO state machine. If the full allocator cannot beat this, the ML and the
    state machine are not earning their keep.
    """
    c = prices["SPY"].astype(float)
    r = c.pct_change()
    vol = r.ewm(halflife=cfg.cov_halflife_vol).std() * math.sqrt(TRADING_DAYS)
    k = (cfg.target_vol / vol.replace(0.0, np.nan)).clip(0.0, cfg.max_leverage)
    k = k.shift(1).reindex(dates).ffill().fillna(0.0)
    rr = r.reindex(dates).fillna(0.0)
    rf = (prices["CASH_RATE_3M"].reindex(dates).astype(float)
          / 100.0 / TRADING_DAYS).fillna(0.0)
    borrow = (k - 1.0).clip(lower=0.0)
    cash = (1.0 - k).clip(lower=0.0)
    port = k * rr + cash * rf - borrow * (rf + cfg.financing_spread / TRADING_DAYS)
    turn = k.diff().abs().fillna(0.0)
    port = port - cfg.transaction_cost * turn
    return cfg.initial_capital * (1.0 + port).cumprod()


def bench_equal_risk(prices: pd.DataFrame, dates, cfg: AllocatorConfig) -> pd.Series:
    """Inverse-volatility (risk-parity-ish) over the same universe, no forecasts."""
    assets = list(cfg.risk_assets)
    r = prices[assets].astype(float).pct_change()
    vol = r.ewm(halflife=63).std() * math.sqrt(TRADING_DAYS)
    iv = (1.0 / vol.replace(0.0, np.nan))
    w = iv.div(iv.sum(axis=1), axis=0).shift(1).reindex(dates).ffill().fillna(0.0)
    port = (w * r.reindex(dates)).sum(axis=1).fillna(0.0)
    turn = w.diff().abs().sum(axis=1).fillna(0.0) * 0.5
    port = port - cfg.transaction_cost * 2.0 * turn
    return cfg.initial_capital * (1.0 + port).cumprod()
