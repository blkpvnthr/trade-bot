#!/usr/bin/env python3
"""
V2.9.8 single-session decision API (pure, no Alpaca).
=====================================================

This is the ONE production entry point into the strategy math. It reconstructs
exactly the state the backtest holds at the close of a session and calls the
SAME ``strat.decide_target_weights`` the backtest calls, so a live decision and
the historical decision for the same date are numerically identical (see
``tests/test_strat_parity.py``).

What it does NOT do: it never runs the whole historical backtest to read off the
last target. It rebuilds the three pieces of cross-day state the decision needs
-- the previous target weights, the sessions-since-last-rebalance counter, and
the trailing forecast-LEVEL history for volatility-budget standardization -- and
does exactly one covariance estimate and one optimizer solve.

Timing contract (identical to the backtest): information through the close of
``as_of_close`` sets the weights to HOLD on ``target_session`` (the next session).
Never feed an unfinished daily bar.

The forecast source is surfaced honestly (``forecast_source``); with no base
module wired in it is the built-in momentum proxy, NOT an ML/XGBoost model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strat_allocator import AllocatorConfig
from strat import (
    build_gate_schedule,
    decide_target_weights,
    momentum_view_panel,
    rolling_ic_panel,
)
from bond_regime import BondOverlayConfig, compute_bond_regime
from loss_replacement import apply_loss_replacement
from momentum_exhaustion import (
    apply_exhaustion_rotation, exhaustion_panel, reversal_panel, EXH_WEIGHTS)
from adaptive_rotation import apply_adaptive_rotation
from strat_allocator import annualized_vols

# The backtest's anchor date for the dates grid and the i>250 standardization
# gate. Live decisions must reconstruct the SAME grid to match historical ones.
DEFAULT_START = "2007-01-02"

# Sentinel for "never rebalanced" so the first live session is always `due`
# (mirrors run_backtest's last_rebalance = -10**9).
COLD_START_SESSIONS = 10 ** 9


@dataclass
class PortfolioDecision:
    """The decision made at close(as_of_close), to be held on target_session."""

    as_of_close: pd.Timestamp
    target_session: pd.Timestamp | None

    recovery_state: str
    severe_bear: bool

    growth_ceiling: float
    leverage_cap: float
    vol_budget: float

    target_weights: dict[str, float]
    prev_weights: dict[str, float]

    gross_exposure: float
    growth_exposure: float
    defensive_exposure: float
    ex_ante_vol: float

    rebalanced: bool
    sessions_since_rebalance: int

    forecast_source: str
    decision_hash: str
    vol_budget_mult: float = 1.0
    growth_ceiling_mult: float = 1.0
    bond_regime: dict | None = None
    portfolio_risk_multiplier: float = 1.0
    risk_state: str | None = None
    opportunity_score: float | None = None
    rebalance_reason: str = "scheduled"
    loss_replacement_events: list = field(default_factory=list)
    exhaustion_events: list = field(default_factory=list)
    adaptive_events: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def target_vector(self, cfg: AllocatorConfig) -> np.ndarray:
        return np.array([self.target_weights[a] for a in cfg.risk_assets], float)

    def rotation_events(self) -> list:
        """All exceptional-exit events (loss / exhaustion / adaptive) for the ledger."""
        return (list(self.loss_replacement_events or []) + list(self.exhaustion_events or [])
                + list(self.adaptive_events or []))


# ---------------------------------------------------------------------------
# Forecast view/IC (reuse the backtest's builders; allow a hook override)
# ---------------------------------------------------------------------------
def build_view_ic(prices: pd.DataFrame,
                  cfg: AllocatorConfig,
                  forecast_hook=None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Return (view, ic, forecast_source). ``forecast_hook(prices, cfg)`` may return
    a dates x assets view DataFrame (already lagged) or None to use the proxy.
    """
    view = None
    source = "momentum_proxy"
    if forecast_hook is not None:
        view = forecast_hook(prices, cfg)
        if view is not None:
            source = getattr(forecast_hook, "forecast_source", "custom_forecast_hook")
    if view is None:
        view = momentum_view_panel(prices, cfg)
        source = "momentum_proxy"
    ic = rolling_ic_panel(view, prices)
    return view, ic, source


# ---------------------------------------------------------------------------
# Exact reconstruction of the backtest's cross-day state
# ---------------------------------------------------------------------------
def _dates_grid(prices: pd.DataFrame, cfg: AllocatorConfig,
                gate: pd.DataFrame, start: str):
    """Reproduce run_backtest's traded-dates grid and full return index."""
    assets = list(cfg.risk_assets)
    px = prices[assets].astype(float)
    rets = px.pct_change()
    full_idx = rets.dropna().index
    dates = full_idx.intersection(px.loc[start:].index).intersection(gate.index)
    R_all = rets.loc[full_idx, assets].to_numpy(float)
    return dates, full_idx, R_all


def _reconstruct_levels(view: pd.DataFrame, cfg: AllocatorConfig,
                        dates: pd.DatetimeIndex, full_idx: pd.DatetimeIndex,
                        upto: int) -> np.ndarray:
    """
    Rebuild ``levels[:upto]`` EXACTLY as the backtest holds it.

    The backtest sets ``levels[i] = mean(finite raw_view row)`` for every day with
    ``d_pos >= 60`` (calibrated_expected_returns returns 0.0 for an all-NaN row,
    a FINITE value that enters the trailing window), and leaves ``levels[i]``
    NaN for ``d_pos < 60`` warmup days (the loop ``continue``s before recording).
    A naive nanmean over the view panel would get both cases wrong.
    """
    assets = list(cfg.risk_assets)
    levels = np.full(upto, np.nan)
    pos_of = {d: k for k, d in enumerate(full_idx)}
    for j in range(upto):
        dd = dates[j]
        if pos_of[dd] < 60:
            continue  # warmup: stays NaN, matching the loop's `continue`
        row = view.loc[dd, assets].to_numpy(float)
        finite = np.isfinite(row)
        levels[j] = float(row[finite].mean()) if finite.any() else 0.0
    return levels


def compute_target_portfolio(prices: pd.DataFrame,
                             prev_weights: np.ndarray,
                             sessions_since_rebalance: int,
                             cfg: AllocatorConfig,
                             *,
                             as_of_close: pd.Timestamp | str | None = None,
                             target_session: pd.Timestamp | str | None = None,
                             forecast_hook=None,
                             start: str = DEFAULT_START,
                             bond_overlay: BondOverlayConfig | None = None,
                             bond_prices: pd.DataFrame | None = None,
                             position_avg_entry: dict[str, float] | None = None,
                             cooldown_state: dict[str, str] | None = None,
                             eligible_symbols: set[str] | None = None,
                             position_entry_dates: dict[str, str] | None = None) -> PortfolioDecision:
    """
    Compute the V2.9.8 target weights to HOLD on the session after ``as_of_close``.

    prices : dates x (>=10 risk assets incl. SPY) of total-return-adjusted closes,
             through close(as_of_close). Anything after as_of_close is ignored
             (sliced off), which is what makes the no-lookahead test meaningful.
    prev_weights : the strategy's previous TARGET weight vector (its internal held
                   book), length == len(cfg.risk_assets). Zeros on cold start,
                   matching run_backtest's initial pending = 0.
    sessions_since_rebalance : trading sessions since the last actual rebalance,
                   counted on THIS dates grid (COLD_START_SESSIONS to force `due`).
    """
    assets = list(cfg.risk_assets)
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise ValueError(f"prices missing risk assets: {missing}")

    prices = prices.sort_index()
    if as_of_close is not None:
        as_of_close = pd.Timestamp(as_of_close)
        prices = prices.loc[:as_of_close]
    if len(prices) == 0:
        raise ValueError("no price rows at//before as_of_close")

    gate = build_gate_schedule(prices["SPY"], cfg)
    view, ic, source = build_view_ic(prices, cfg, forecast_hook)
    dates, full_idx, R_all = _dates_grid(prices, cfg, gate, start)

    decision_date = pd.Timestamp(as_of_close) if as_of_close is not None else prices.index[-1]
    if len(dates) == 0 or dates[-1] != decision_date:
        raise ValueError(
            f"as_of_close {decision_date.date()} is not a valid decision date "
            f"(not in the strategy dates grid; last valid = "
            f"{dates[-1].date() if len(dates) else 'none'}). A NaN/missing price "
            f"or a pre-{start} date excludes it.")

    i_star = len(dates) - 1
    p = full_idx.get_loc(decision_date)
    lb = cfg.cov_lookback
    cov_window = R_all[max(0, p - lb + 1): p + 1]

    view_row = view.loc[decision_date, assets].to_numpy(float)
    ic_row = ic.loc[decision_date, assets].to_numpy(float)
    level_history = _reconstruct_levels(view, cfg, dates, full_idx, i_star)

    state = str(gate.loc[decision_date, "RecoveryState"])
    severe = bool(gate.loc[decision_date, "SevereBear"])
    ceiling_raw = float(gate.loc[decision_date, "EffectiveGrowthCeiling"])

    standardize = bool(view is not None and i_star > 250)
    w_prev = np.asarray(prev_weights, float)

    # Optional bond-regime overlay (default OFF -> 1.0 multipliers -> byte-identical).
    vb_mult, gc_mult, bond_regime = 1.0, 1.0, None
    if bond_overlay is not None and bond_overlay.enabled and bond_prices is not None:
        reg = compute_bond_regime(bond_prices, bond_overlay, as_of=decision_date)
        vb_mult, gc_mult = reg.vol_budget_mult, reg.growth_ceiling_mult
        bond_regime = {"risk_off_z": reg.risk_off_z, "vol_budget_mult": vb_mult,
                       "growth_ceiling_mult": gc_mult,
                       "curve_slope_proxy": reg.curve_slope_proxy, **reg.components}
        source = f"{source}+bond_overlay"

    # Optional dynamic portfolio-risk controller (default OFF -> 1.0 -> byte-identical).
    # Recompute the SAME per-date multiplier series the backtest builds (over the
    # identical dates grid) and take the value at the decision date; it feeds the
    # vol budget, NOT raw weights. Composes multiplicatively with the bond vb_mult;
    # the PRODUCT is clamped to the controller's [min, max] (parity with run_backtest).
    portfolio_risk_multiplier, risk_state, opportunity_score = 1.0, None, None
    if cfg.dynamic_risk_enabled:
        from dynamic_risk import dynamic_risk_frame
        dyn_frame = dynamic_risk_frame(prices, cfg, gate=gate, view=view, dates=dates,
                                       full_idx=full_idx, R_all=R_all)
        last = dyn_frame.iloc[-1]
        portfolio_risk_multiplier = float(last["risk_multiplier"])
        risk_state = str(last["risk_state"])
        opportunity_score = float(last["opportunity_score"])
        vb_mult = float(np.clip(vb_mult * portfolio_risk_multiplier,
                                cfg.dynamic_risk_min_mult, cfg.dynamic_risk_max_mult))
        source = f"{source}+dynamic_risk"

    new_w, rebalanced, diag = decide_target_weights(
        prev_weights=w_prev,
        cov_window=cov_window,
        view_row=view_row,
        ic_row=ic_row,
        state=state, severe=severe, ceiling_raw=ceiling_raw,
        level_history=level_history,
        sessions_since_rebalance=int(sessions_since_rebalance),
        cfg=cfg, use_gate=True, use_vol_target=True,
        standardize_level=standardize,
        vol_budget_mult=vb_mult, growth_ceiling_mult=gc_mult,
    )

    # --- exceptional-exit overlays: loss-replacement THEN momentum-exhaustion ----
    # Both run AFTER the base decision through the SAME diag-supplied risk inputs and
    # re-enforce constraints; default OFF -> byte-identical. Precedence (spec 25):
    # loss-replacement (hard loss) before exhaustion (opportunity rotation); an asset
    # exited for a loss is NOT re-processed for exhaustion (excluded via weight<=eps).
    rebalance_reason = "scheduled"
    lr_events: list[dict] = []
    exh_events: list[dict] = []
    exh_skipped: list[dict] = []
    adaptive_events: list[dict] = []
    adaptive_skipped: list[dict] = []
    adaptive_on = bool(cfg.adaptive_rotation_enabled)
    lr_on = bool(cfg.loss_replacement_enabled) and not adaptive_on
    exh_on = bool(cfg.momentum_exhaustion_enabled) and not adaptive_on
    if lr_on or exh_on or adaptive_on:
        n_a = len(assets)
        px_a = prices[assets]
        last_close = px_a.iloc[-1].to_numpy(float)   # decision close (ALL-adj last row ~ raw)
        eps = float(cfg.loss_replacement_hold_eps)
        # position return vs ACTUAL cost basis at the decision close (idempotent).
        pos_ret = np.full(n_a, np.nan)
        if position_avg_entry:
            for j, a in enumerate(assets):
                ae = float(position_avg_entry.get(a, 0.0) or 0.0)
                if ae > 0 and last_close[j] > 0:
                    pos_ret[j] = last_close[j] / ae - 1.0
        # shared cooldown (trading sessions on THIS grid) + eligibility.
        cd_mask = np.zeros(n_a, bool)
        if cooldown_state:
            for j, a in enumerate(assets):
                le = cooldown_state.get(a)
                if le and int(((dates > pd.Timestamp(le))
                               & (dates <= decision_date)).sum()) < int(cfg.replacement_cooldown_days):
                    cd_mask[j] = True
        elig_mask = (np.array([a in eligible_symbols for a in assets], bool)
                     if eligible_symbols is not None else np.ones(n_a, bool))

    if lr_on:
        N = int(cfg.replacement_momentum_lookback)
        if len(px_a) > N:
            prior = px_a.iloc[-(N + 1)].to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mom_row = np.where(prior > 0, last_close / prior - 1.0, np.nan)
        else:
            mom_row = np.full(n_a, np.nan)
        res = apply_loss_replacement(
            base_weights=new_w, position_returns=pos_ret, momentum=mom_row,
            cov_adj=diag["cov_adj"], cfg=cfg,
            growth_ceiling=diag["growth_ceiling"], vol_budget=diag["vol_budget"],
            leverage_cap=diag["leverage_cap"], cooldown_mask=cd_mask, eligible_mask=elig_mask)
        if res.triggered:
            new_w = res.weights
            rebalanced = True
            rebalance_reason = "loss_replacement"
            lr_events = res.events
            source = f"{source}+loss_replacement"
            # This-session loss exits also cool down exhaustion candidacy, matching
            # run_backtest (which updates last_exit before the exhaustion mask), so
            # exhaustion never rotates INTO a name loss-replacement just exited.
            for ev in lr_events:
                if ev.get("exited_asset"):
                    cd_mask[assets.index(ev["exited_asset"])] = True

    if exh_on:
        k = max(1, int(cfg.momentum_exhaustion_confirmations))
        exh_df, exh_comp = exhaustion_panel(prices, cfg, view)
        rev_df, _ = reversal_panel(prices, cfg, view)
        cnames = list(EXH_WEIGHTS.keys())
        recent = exh_df[assets].to_numpy(float)[-k:]
        if recent.shape[0] < k:                       # not enough history -> never confirms
            recent = np.vstack([np.full((k - recent.shape[0], n_a), np.nan), recent])
        latest_comp = np.column_stack([exh_comp[c][assets].to_numpy(float)[-1] for c in cnames])
        rev_row = rev_df[assets].to_numpy(float)[-1]
        rev_tb = (view.reindex(prices.index)[assets].to_numpy(float)[-1]
                  if view is not None else np.zeros(n_a))
        exh_pos = np.where(new_w > eps, pos_ret, np.nan)   # held AFTER loss-replacement
        eres = apply_exhaustion_rotation(
            base_weights=new_w, position_returns=exh_pos, exhaustion_recent=recent,
            exhaustion_components=latest_comp, component_names=cnames,
            reversal_scores=rev_row, reversal_rank_tiebreak=rev_tb, cov_adj=diag["cov_adj"],
            cfg=cfg, growth_ceiling=diag["growth_ceiling"], vol_budget=diag["vol_budget"],
            leverage_cap=diag["leverage_cap"], cooldown_mask=cd_mask, eligible_mask=elig_mask)
        exh_skipped = eres.skipped
        if eres.triggered:
            new_w = eres.weights
            rebalanced = True
            if rebalance_reason == "scheduled":
                rebalance_reason = "momentum_exhaustion"
            exh_events = eres.events
            source = f"{source}+momentum_exhaustion"

    if adaptive_on:
        k = max(1, int(cfg.adaptive_hazard_confirmations))
        exh_df, _ = exhaustion_panel(prices, cfg, view)
        rev_df, _ = reversal_panel(prices, cfg, view)
        recent = exh_df[assets].to_numpy(float)[-k:]
        if recent.shape[0] < k:
            recent = np.vstack([np.full((k - recent.shape[0], n_a), np.nan), recent])
        rev_row = rev_df[assets].to_numpy(float)[-1]
        px_arr = px_a.to_numpy(float)
        mom20_row = (px_arr[-1] / px_arr[-21] - 1.0 if len(px_arr) > 20 else np.full(n_a, np.nan))
        accel_row = ((px_arr[-1] / px_arr[-6] - 1.0) - mom20_row
                     if len(px_arr) > 20 else np.full(n_a, np.nan))
        view_row_a = (view.reindex(prices.index)[assets].to_numpy(float)[-1]
                      if view is not None else np.zeros(n_a))
        sigma_ann = annualized_vols(diag["cov_adj"])
        # days_held from the persisted first-entry ledger (session-count on THIS grid),
        # the identical parity template as the cooldown block above.
        dh_a = np.full(n_a, np.nan)
        if position_entry_dates:
            for j, a in enumerate(assets):
                fe = position_entry_dates.get(a)
                if fe and np.isfinite(pos_ret[j]):
                    dh_a[j] = float(((dates > pd.Timestamp(fe)) & (dates <= decision_date)).sum())
        ares = apply_adaptive_rotation(
            base_weights=new_w, position_returns=pos_ret, days_held=dh_a,
            view_row=view_row_a, sigma_ann=sigma_ann, mom20_row=mom20_row,
            accel_row=accel_row, reversal_score=rev_row, exhaustion_recent=recent,
            cov_adj=diag["cov_adj"], state=state, severe=severe, cfg=cfg,
            growth_ceiling=diag["growth_ceiling"], vol_budget=diag["vol_budget"],
            leverage_cap=diag["leverage_cap"], cooldown_mask=cd_mask, eligible_mask=elig_mask)
        adaptive_skipped = ares.skipped
        if ares.triggered:
            new_w = ares.weights
            rebalanced = True
            rebalance_reason = "adaptive_rotation"
            adaptive_events = ares.events
            source = f"{source}+adaptive_rotation"

    if lr_events or exh_events or adaptive_events:
        g_idx, d_idx = cfg.growth_idx, cfg.defensive_idx
        cov_adj = np.asarray(diag["cov_adj"], float)
        diag = {**diag,
                "gross": float(new_w.sum()),
                "growth": float(new_w[g_idx].sum()),
                "defensive": float(new_w[d_idx].sum()),
                "ex_ante_vol": float(np.sqrt(max(float(new_w @ cov_adj @ new_w), 0.0))),
                "loss_replacement_events": lr_events,
                "exhaustion_events": exh_events,
                "exhaustion_skipped": exh_skipped,
                "adaptive_events": adaptive_events,
                "adaptive_skipped": adaptive_skipped,
                "rebalance_reason": rebalance_reason}

    tw = {a: float(new_w[j]) for j, a in enumerate(assets)}
    pw = {a: float(w_prev[j]) for j, a in enumerate(assets)}
    dhash = decision_hash(decision_date, tw, cfg, source)

    return PortfolioDecision(
        as_of_close=decision_date,
        target_session=(pd.Timestamp(target_session) if target_session is not None else None),
        recovery_state=state,
        severe_bear=severe,
        growth_ceiling=diag["growth_ceiling"],
        leverage_cap=diag["leverage_cap"],
        vol_budget=diag["vol_budget"],
        target_weights=tw,
        prev_weights=pw,
        gross_exposure=diag["gross"],
        growth_exposure=diag["growth"],
        defensive_exposure=diag["defensive"],
        ex_ante_vol=diag["ex_ante_vol"],
        rebalanced=rebalanced,
        sessions_since_rebalance=int(sessions_since_rebalance),
        forecast_source=source,
        decision_hash=dhash,
        vol_budget_mult=vb_mult,
        growth_ceiling_mult=gc_mult,
        bond_regime=bond_regime,
        portfolio_risk_multiplier=portfolio_risk_multiplier,
        risk_state=risk_state,
        opportunity_score=opportunity_score,
        rebalance_reason=rebalance_reason,
        loss_replacement_events=lr_events,
        exhaustion_events=exh_events,
        adaptive_events=adaptive_events,
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# Deterministic decision identity + Alpaca-safe client order ids (ONE helper)
# ---------------------------------------------------------------------------
def _cfg_signature(cfg: AllocatorConfig) -> str:
    """Stable string of the load-bearing config so a config change changes the hash."""
    keys = ["risk_assets", "growth_assets", "defensive_assets", "target_vol",
            "max_leverage", "crisis_max_leverage", "cov_lookback",
            "cov_halflife_vol", "cov_halflife_corr", "cov_shrinkage",
            "rebalance_every", "partial_adjust", "rebalance_deadband",
            "sizing_tilt", "sizing_tilt_scale", "anchor_strength", "alpha_gain",
            "uncertainty_kappa", "dd_haircut", "dd_neutral", "transaction_cost",
            "forecast_horizon_days",
            "loss_replacement_enabled", "loss_replacement_threshold",
            "replacement_momentum_lookback", "replacement_cooldown_days",
            "loss_replacement_hold_eps",
            "momentum_exhaustion_enabled", "momentum_exhaustion_threshold",
            "momentum_exhaustion_critical", "momentum_exhaustion_warning",
            "momentum_exhaustion_confirmations", "bullish_reversal_score_threshold",
            "minimum_replacement_score_improvement",
            "adaptive_rotation_enabled", "adaptive_minimum_rotation_edge",
            "adaptive_cost_safety_margin", "adaptive_min_hold_days",
            "adaptive_turnover_penalty", "adaptive_concentration_penalty",
            "adaptive_reversal_bonus", "adaptive_diversification_weight",
            "adaptive_diversification_cap", "adaptive_entry_score_threshold",
            "adaptive_hold_score_floor", "adaptive_hazard_confirmations",
            "adaptive_damage_measure", "adaptive_damage_lookback",
            "adaptive_damage_thresholds",
            "dynamic_risk_enabled", "dynamic_risk_min_mult", "dynamic_risk_max_mult",
            "dynamic_risk_severe_floor", "dynamic_risk_deadband_up",
            "dynamic_risk_deadband_down", "dynamic_risk_stress_gamma_recovery",
            "dynamic_risk_vol_percentile_lookback", "dynamic_risk_confirm_days",
            "dynamic_risk_max_up_change", "dynamic_risk_max_down_change",
            "dynamic_risk_recovery_rate_boost", "dynamic_risk_margin_up",
            "dynamic_risk_margin_down", "dynamic_risk_opp_weights",
            "dynamic_risk_stress_weights", "dynamic_risk_dd_stress_weight"]
    payload = {}
    for k in keys:
        v = getattr(cfg, k)
        payload[k] = list(v) if isinstance(v, tuple) else v
    return json.dumps(payload, sort_keys=True, default=str)


def decision_hash(as_of_close: pd.Timestamp | str,
                  target_weights: dict[str, float],
                  cfg: AllocatorConfig,
                  forecast_source: str) -> str:
    """
    Deterministic sha256 over (as_of_close, target_weights, cfg, forecast_source).

    Pure of equity, positions, and wall-clock time so restarts reproduce the same
    identity. Weights are rounded to 1e-9 so platform float repr wobble does not
    change the hash.
    """
    aoc = pd.Timestamp(as_of_close).strftime("%Y-%m-%d")
    w = {k: round(float(v), 9) for k, v in sorted(target_weights.items())}
    payload = json.dumps({"as_of": aoc, "weights": w,
                          "cfg": _cfg_signature(cfg),
                          "forecast": forecast_source}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_client_order_id(as_of_close: pd.Timestamp | str,
                         symbol: str,
                         side: str,
                         decision_hash_hex: str,
                         slice_n: int = 0) -> str:
    """
    The ONE Alpaca-safe deterministic client-order-id builder used everywhere.

    Shape: ``v298-YYYYMMDD-SPY-B-3fa91c2d`` (+ ``-sN`` for a residual slice).
    Uses the SESSION/decision date (never submission wall-clock) so a next-session
    execution run reuses the same id, and Alpaca's duplicate-id rejection gives
    broker-side idempotency for free. Capped at Alpaca's 48-char limit.
    """
    aoc = pd.Timestamp(as_of_close).strftime("%Y%m%d")
    s = "B" if str(side).lower().startswith("b") else "S"
    sym = str(symbol).upper().replace(".", "-")
    coid = f"v298-{aoc}-{sym}-{s}-{decision_hash_hex[:8]}"
    if slice_n:
        coid = f"{coid}-s{int(slice_n)}"
    return coid[:48]
