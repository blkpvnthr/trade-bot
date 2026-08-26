#!/usr/bin/env python3
"""
V2.9.8 allocator library
========================

Self-contained, base-module-independent replacement for the V2.9.7.2
allocation layer. Every function here is pure numpy/pandas/scipy so it can be
unit-tested and validated without the V2.9.6 forecasting stack.

What changed versus V2.9.7.2 and why
------------------------------------
1. Sharpe is measured on EXCESS returns. Cash is a held asset earning the
   T-bill rate; counting that yield as excess return inflated Sharpe, and
   inflated it more for the cash-heavy strategy than for SPY buy & hold.

2. Expected returns are calibrated, not raw. mu = equal-Sharpe PRIOR plus a
   Grinold-Kahn view (IC * sigma * z). Units are annualized, so risk aversion
   is interpretable and the multi-horizon blend self-weights by realized skill.
   An asset with non-positive trailing IC contributes no view and is held for
   diversification only.

3. Forecast uncertainty moved from expected return INTO the covariance
   diagonal. In V2.9.7.2 it was a linear penalty that cash was exempt from,
   which handicapped every risky asset by ~5x the size of the alpha signal.

4. Drawdown probability is a multiplicative alpha haircut, not a linear
   penalty, so it scales with signal instead of dominating it.

5. Sizing is volatility targeting with a LEVERAGE CAP ABOVE 1.0, not a
   discrete state ladder. A strategy that can only de-risk cannot beat buy &
   hold in a bull market. Leverage is hard-killed in CRISIS / severe bear.

6. Covariance is EWMA with fast volatility (halflife 21) and slow correlation
   (halflife 126), shrunk toward identity, PSD-repaired.

7. Turnover is controlled by structure (5-day rebalance, partial adjustment,
   deadband) with TRUE transaction cost in the objective, rather than a
   penalty set 80x above real cost.

8. The structural growth gate is unchanged in spirit: it is a hard cap applied
   as an optimizer constraint. Recovery relaxes the gate; nothing bypasses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252.0


# =============================================================================
# CONFIG
# =============================================================================


@dataclass
class AllocatorConfig:
    # --- Universe -----------------------------------------------------------
    risk_assets: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "IEF", "GLD",
                                    "DBC", "EFA", "EEM", "VNQ")
    growth_assets: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ")
    defensive_assets: tuple[str, ...] = ("TLT", "IEF", "GLD", "DBC")

    asset_max_weights: dict = field(default_factory=lambda: {
        "SPY": 1.20, "QQQ": 0.60, "IWM": 0.30, "EFA": 0.40, "EEM": 0.25,
        "VNQ": 0.20, "TLT": 0.20, "IEF": 0.50, "GLD": 0.25, "DBC": 0.15,
    })
    defensive_combined_max: float = 0.50

    # --- Risk targeting ----------------------------------------------------
    target_vol: float = 0.15
    max_leverage: float = 1.40
    crisis_max_leverage: float = 1.00
    financing_spread: float = 0.0075     # annual, over the cash rate
    sizing_tilt: float = 0.30            # forecast LEVEL modulates the vol target
    sizing_tilt_scale: float = 1.00      # in z-score units, so unit-free

    # --- Strategic core -----------------------------------------------------
    # An equal-Sharpe prior over six highly collinear equity ETFs produces a
    # knife-edge mean-variance solution: it picked QQQ/IWM/EEM/VNQ and gave SPY
    # 1.9%, turning an equity mandate into a levered bond book whose worst
    # drawdown was October 2023. Anchoring composition to an explicit strategic
    # core fixes that, and lets alpha express itself as a bounded TILT.
    equity_core: dict = field(default_factory=lambda: {
        "SPY": 0.75, "QQQ": 0.15, "EFA": 0.10})
    defensive_core: dict = field(default_factory=lambda: {
        "IEF": 0.40, "TLT": 0.25, "GLD": 0.25, "DBC": 0.10})
    state_growth_core: dict = field(default_factory=lambda: {
        "CRISIS": 0.05, "DEFENSIVE": 0.45, "BOTTOMING": 0.55,
        "RECOVERING": 0.80, "BULL": 0.95})
    anchor_strength: float = 3.0     # tracking-error aversion to the core

    # --- Expected returns ---------------------------------------------------
    # Ex-ante capital-market assumptions, NOT fitted to this sample. An equal
    # Sharpe prior across all sleeves made the optimizer prefer whichever asset
    # had the highest volatility within each sleeve (TLT over IEF, DBC over
    # nothing), which is a known artifact rather than a view.
    prior_sharpe: float = 0.30           # fallback for unlisted assets
    asset_prior_sharpe: dict = field(default_factory=lambda: {
        "SPY": 0.38, "QQQ": 0.38, "IWM": 0.30, "EFA": 0.30, "EEM": 0.25,
        "VNQ": 0.28, "IEF": 0.28, "TLT": 0.22, "GLD": 0.20, "DBC": 0.08})
    alpha_gain: float = 1.00             # scales the Grinold-Kahn view
    risk_aversion: float | None = None   # None -> prior_sharpe / target_vol
    uncertainty_kappa: float = 1.00      # forecast variance -> covariance diagonal
    dd_haircut: float = 0.60             # multiplicative alpha haircut
    dd_neutral: float = 0.25

    # --- Covariance ---------------------------------------------------------
    cov_lookback: int = 378
    cov_halflife_vol: int = 21
    cov_halflife_corr: int = 126
    cov_shrinkage: float = 0.20

    # --- Execution ----------------------------------------------------------
    rebalance_every: int = 5
    partial_adjust: float = 0.50
    rebalance_deadband: float = 0.02
    transaction_cost: float = 0.0005     # one-way, per unit traded
    forecast_horizon_days: int = 20

    # --- Loss-replacement / momentum-rotation overlay -----------------------
    # A controlled OVERLAY on the target weights: if an actively held position is
    # more than `loss_replacement_threshold` below its cost basis, exit it and
    # rotate its allocation into the strongest positive-momentum eligible asset,
    # then re-run enforce_constraints so the swapped book still respects every
    # risk limit. DEFAULT OFF: enabling it changes the strategy, so it must not be
    # turned on silently and must clear the A/B (backtest_loss_replacement.py)
    # before production use -- same discipline as the bond-regime overlay. When
    # off, run_backtest / compute_target_portfolio are byte-identical to before.
    loss_replacement_enabled: bool = False
    loss_replacement_threshold: float = -0.005   # inclusive: pr <= -0.5% -> exit
    replacement_momentum_lookback: int = 20      # sessions for P_t/P_{t-N}-1
    replacement_cooldown_days: int = 5           # exited ticker barred as replacement
    loss_replacement_hold_eps: float = 1e-4      # weight below which a leg is "flat"

    # --- Momentum-exhaustion / reversal-rotation overlay --------------------
    # A SECOND, independent overlay: exit a HELD winner when a point-in-time
    # composite exhaustion score (0..1) is CONFIRMED above the exit threshold (or a
    # single CRITICAL reading), rotating into a bearish->bullish reversal candidate
    # if one materially improves the opportunity. DEFAULT OFF (falsify first). Never
    # overrides a hard risk gate or the loss-threshold exit; precedence is
    # hard-risk > loss_replacement > critical-exhaustion > confirmed-exhaustion >
    # scheduled. Shares the replacement cooldown + execution/idempotency plumbing.
    momentum_exhaustion_enabled: bool = False
    momentum_exhaustion_threshold: float = 0.70      # confirmed-exit threshold
    momentum_exhaustion_critical: float = 0.90       # single-reading immediate exit
    momentum_exhaustion_warning: float = 0.55        # WEAKENING state (diagnostic)
    momentum_exhaustion_confirmations: int = 2       # consecutive sessions >= threshold
    bullish_reversal_score_threshold: float = 0.65   # min score to be a replacement
    minimum_replacement_score_improvement: float = 0.10  # rotate only if materially better

    # --- Adaptive opportunity-gated rotation engine -------------------------
    # The UNIFIED rotation engine. A deterioration signal (vol-normalized damage
    # below a regime-conditioned threshold, OR confirmed exhaustion hazard) does NOT
    # auto-sell: it OPENS a hold-vs-rotate evaluation. Capital moves only when a
    # candidate (or CASH) is materially superior on a cross-sectional opportunity
    # edge (rank-units) AND clears a transaction-cost gate (excess-return units),
    # after turnover/hysteresis/cooldown/min-hold. When enabled it SUPERSEDES the
    # naive loss_replacement + momentum_exhaustion overlays (guarded below). DEFAULT
    # OFF -> byte-identical; falsify on real data before enabling.
    adaptive_rotation_enabled: bool = False
    adaptive_minimum_rotation_edge: float = 0.15     # rank-units (Gate 1) primary knob
    adaptive_cost_safety_margin: float = 0.0007      # excess-return units, on top of 2*tc
    adaptive_min_hold_days: int = 10                 # soft; real deterioration bypasses
    adaptive_turnover_penalty: float = 0.10          # rank-units (fresh-position churn)
    adaptive_concentration_penalty: float = 0.10     # rank-units (per unit over-weight)
    adaptive_reversal_bonus: float = 0.25            # candidate-only, rank-units
    adaptive_diversification_weight: float = 0.15    # candidate-only, rank-units
    adaptive_diversification_cap: float = 0.10       # hard cap so div can't dominate
    adaptive_entry_score_threshold: float = 0.60     # absolute candidate quality floor
    adaptive_hold_score_floor: float = 0.40          # hysteresis: only evaluate weak holds
    adaptive_hazard_confirmations: int = 2           # consecutive exhaustion >= threshold
    adaptive_damage_measure: str = "vol"             # "atr" -> vol fallback (no OHLC)
    adaptive_damage_lookback: int = 20               # sessions for vol normalization
    adaptive_damage_thresholds: dict = field(default_factory=lambda: {
        "CRISIS": -1.50, "DEFENSIVE": -1.25, "BOTTOMING": -1.00,
        "RECOVERING": -1.25, "BULL": -1.75})         # vol-units, A/B-tuned (not optimal)

    initial_capital: float = 823.00

    def __post_init__(self) -> None:
        bad = set(self.growth_assets) | set(self.defensive_assets)
        if bad - set(self.risk_assets):
            raise ValueError(f"Unknown assets in sleeves: {bad - set(self.risk_assets)}")
        if set(self.growth_assets) & set(self.defensive_assets):
            raise ValueError("growth_assets and defensive_assets overlap.")
        # The adaptive engine subsumes both naive rotation overlays; running them
        # together would double-count triggers. Force the operator to pick one family.
        if self.adaptive_rotation_enabled and (self.loss_replacement_enabled
                                               or self.momentum_exhaustion_enabled):
            raise ValueError(
                "adaptive_rotation_enabled supersedes loss_replacement / "
                "momentum_exhaustion -- enable only one rotation family at a time.")
        missing = set(self.risk_assets) - set(self.asset_max_weights)
        if missing:
            raise ValueError(f"asset_max_weights missing: {missing}")

    def prior_vector(self) -> np.ndarray:
        return np.array([self.asset_prior_sharpe.get(a, self.prior_sharpe)
                         for a in self.risk_assets], float)

    @property
    def gamma(self) -> float:
        if self.risk_aversion is not None:
            return float(self.risk_aversion)
        return float(float(self.prior_vector().mean()) / max(self.target_vol, 1e-6))

    @property
    def caps(self) -> np.ndarray:
        return np.array([self.asset_max_weights[a] for a in self.risk_assets], float)

    @property
    def growth_idx(self) -> np.ndarray:
        return np.array([self.risk_assets.index(a) for a in self.growth_assets], int)

    @property
    def defensive_idx(self) -> np.ndarray:
        return np.array([self.risk_assets.index(a) for a in self.defensive_assets], int)


# =============================================================================
# RISK ESTIMATION
# =============================================================================


def ewma_cov_annualized(returns: np.ndarray,
                        halflife_vol: int = 21,
                        halflife_corr: int = 126,
                        shrink: float = 0.20) -> np.ndarray:
    """
    Annualized covariance from FAST volatility and SLOW correlation.

    returns : (T, n) daily simple returns, oldest row first, no NaNs.

    Volatility mean-reverts quickly; correlation is more stable. Estimating
    them on one 60-day clock (V2.9.7.2) is both noisy and slow.
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape
    if T < 30:
        raise ValueError(f"Need >=30 observations, got {T}.")

    def weights(halflife: int) -> np.ndarray:
        lam = 0.5 ** (1.0 / max(int(halflife), 1))
        w = lam ** np.arange(T - 1, -1, -1, dtype=float)
        return w / w.sum()

    w_corr = weights(halflife_corr)
    w_vol = weights(halflife_vol)

    centered = returns - (w_corr @ returns)

    slow = (centered * w_corr[:, None]).T @ centered
    d_slow = np.sqrt(np.maximum(np.diag(slow), 1e-16))
    corr = slow / np.outer(d_slow, d_slow)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)

    # Shrink correlation toward identity to control estimation error.
    corr = (1.0 - shrink) * corr + shrink * np.eye(n)

    d_fast = np.sqrt(np.maximum(w_vol @ (centered ** 2), 1e-16))

    cov = np.outer(d_fast, d_fast) * corr * TRADING_DAYS
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def annualized_vols(cov: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.diag(cov), 1e-16))


# =============================================================================
# EXPECTED RETURNS
# =============================================================================


def calibrated_expected_returns(raw_view: np.ndarray,
                                sigma_ann: np.ndarray,
                                rolling_ic: np.ndarray,
                                dd_risk: np.ndarray,
                                cfg: AllocatorConfig,
                                prior_sharpe_vec: np.ndarray | None = None
                                ) -> tuple[np.ndarray, float]:
    """
    mu = equal-Sharpe prior + Grinold-Kahn view, then a drawdown haircut.

    raw_view    : any monotone forecast of relative attractiveness, ANY units.
    sigma_ann   : annualized vol per asset (from the covariance diagonal).
    rolling_ic  : trailing OUT-OF-SAMPLE rank IC per asset, clipped at 0.
                  Must be built only from label windows fully closed by t-1.
    dd_risk     : predicted drawdown probability per asset, in [0, 1].

    Returns (mu_annualized, view_level) where view_level is the cross-sectional
    MEAN of the raw view. The view's cross-section sets COMPOSITION; its level
    is returned separately and modulates the risk budget in size_portfolio.
    """
    raw = np.asarray(raw_view, dtype=float)
    finite = np.isfinite(raw)
    level = float(raw[finite].mean()) if finite.any() else 0.0

    z = np.where(finite, raw - level, 0.0)
    spread = float(np.std(z[finite])) if finite.sum() > 1 else 0.0
    z = z / spread if spread > 1e-12 else np.zeros_like(z)
    z = np.clip(z, -3.0, 3.0)

    ic = np.clip(np.nan_to_num(rolling_ic, nan=0.0), 0.0, None)

    ps = cfg.prior_vector() if prior_sharpe_vec is None else np.asarray(prior_sharpe_vec, float)
    prior = ps * sigma_ann
    view = cfg.alpha_gain * ic * sigma_ann * z

    mu = prior + view

    dd = np.nan_to_num(np.asarray(dd_risk, dtype=float), nan=cfg.dd_neutral)
    haircut = 1.0 - cfg.dd_haircut * np.clip(dd - cfg.dd_neutral, 0.0, 0.5) / 0.5
    mu = mu * np.clip(haircut, 0.0, 1.0)

    return mu, level


def inflate_cov_for_uncertainty(cov: np.ndarray,
                                uncertainty_ann: np.ndarray,
                                cfg: AllocatorConfig) -> np.ndarray:
    """
    Forecast error variance IS variance. Adding it to the covariance diagonal
    is scale-consistent and, unlike a linear penalty on expected return, does
    not hand cash a free advantage over every risky asset.
    """
    unc = np.nan_to_num(np.asarray(uncertainty_ann, dtype=float), nan=0.0)
    out = cov + np.diag(cfg.uncertainty_kappa * unc ** 2)
    out = 0.5 * (out + out.T)
    eigvals, eigvecs = np.linalg.eigh(out)
    return eigvecs @ np.diag(np.clip(eigvals, 1e-10, None)) @ eigvecs.T


# =============================================================================
# THE OPTIMIZER
# =============================================================================


def strategic_core(state: str,
                   cov: np.ndarray,
                   cfg: AllocatorConfig,
                   vol_budget: float,
                   growth_ceiling: float,
                   leverage_cap: float) -> np.ndarray:
    """
    State-dependent strategic direction, scaled to the volatility budget.

    With alpha switched off and anchor_strength large, the allocator collapses
    to exactly this portfolio: a vol-targeted, state-weighted multi-asset core.
    That is the honest neutral baseline the forecast has to improve on.
    """
    n = len(cfg.risk_assets)
    idx = {a: i for i, a in enumerate(cfg.risk_assets)}
    g = float(cfg.state_growth_core.get(str(state).upper(), 0.45))

    core = np.zeros(n)
    for a, share in cfg.equity_core.items():
        core[idx[a]] += g * share
    for a, share in cfg.defensive_core.items():
        core[idx[a]] += (1.0 - g) * share

    total = core.sum()
    if total <= 0:
        return core
    core = core / total

    vol = math.sqrt(max(float(core @ cov @ core), 1e-18))
    scaled = core * (vol_budget / vol)
    return enforce_constraints(scaled, cfg, growth_ceiling, cov,
                               vol_budget, leverage_cap)


def solve_weights(mu: np.ndarray,
                  cov: np.ndarray,
                  prev_weights: np.ndarray,
                  cfg: AllocatorConfig,
                  growth_ceiling: float,
                  vol_budget: float,
                  leverage_cap: float,
                  core: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """
    Maximize  mu'w - (gamma/2) w'Cov w - cost*|w - w_prev|
    subject to
        0 <= w_i <= cap_i
        sum(w)                <= leverage_cap        (cash = 1 - sum(w))
        sqrt(w'Cov w)         <= vol_budget
        sum(w_growth)         <= growth_ceiling      (the structural gate)
        sum(w_defensive)      <= defensive_combined_max

    Single convex program. The volatility constraint does the SIZING, which is
    what lets exposure exceed 1.0 when risk is genuinely cheap, and collapse
    when it is not. The growth gate retains absolute final authority.
    """
    n = len(mu)
    gamma = cfg.gamma
    tau = cfg.anchor_strength
    g_idx, d_idx = cfg.growth_idx, cfg.defensive_idx
    caps = np.minimum(cfg.caps, leverage_cap)
    anchor = np.zeros(n) if core is None else np.asarray(core, float)

    # True cost, amortized over the forecast horizon.
    cost_coef = cfg.transaction_cost * (TRADING_DAYS / max(cfg.forecast_horizon_days, 1))
    eps = 1e-6
    var_budget = float(vol_budget) ** 2

    def f(w):
        d = w - prev_weights
        turn = np.sqrt(d * d + eps).sum()
        a = w - anchor
        return (-(mu @ w)
                + 0.5 * gamma * (w @ cov @ w)
                + 0.5 * tau * (a @ cov @ a)      # tracking error vs the core
                + cost_coef * turn)

    def f_grad(w):
        d = w - prev_weights
        return (-mu + gamma * (cov @ w) + tau * (cov @ (w - anchor))
                + cost_coef * (d / np.sqrt(d * d + eps)))

    constraints = [
        {"type": "ineq",
         "fun": lambda w: leverage_cap - w.sum(),
         "jac": lambda w: -np.ones(n)},
        {"type": "ineq",
         "fun": lambda w: var_budget - float(w @ cov @ w),
         "jac": lambda w: -2.0 * (cov @ w)},
        {"type": "ineq",
         "fun": lambda w: growth_ceiling - w[g_idx].sum(),
         "jac": lambda w: -_onehot(n, g_idx)},
        {"type": "ineq",
         "fun": lambda w: cfg.defensive_combined_max - w[d_idx].sum(),
         "jac": lambda w: -_onehot(n, d_idx)},
    ]

    best_w, best_obj = None, math.inf
    starts = [np.clip(prev_weights, 0.0, caps)]
    if core is not None:
        starts.append(np.clip(anchor, 0.0, caps))
    for x0 in starts:
        res = minimize(f, x0, jac=f_grad, method="SLSQP",
                       bounds=[(0.0, float(c)) for c in caps],
                       constraints=constraints,
                       options={"maxiter": 200, "ftol": 1e-10})
        if not np.all(np.isfinite(res.x)):
            continue
        w = np.clip(res.x, 0.0, caps)
        w = enforce_constraints(w, cfg, growth_ceiling, cov, vol_budget, leverage_cap)
        obj = f(w)
        if obj < best_obj:
            best_obj, best_w = obj, w

    if best_w is None:
        best_w = _fallback(cfg, growth_ceiling, cov, vol_budget, leverage_cap)

    ex_ante_vol = math.sqrt(max(float(best_w @ cov @ best_w), 0.0))
    info = {
        "gross": float(best_w.sum()),
        "growth": float(best_w[g_idx].sum()),
        "defensive": float(best_w[d_idx].sum()),
        "ex_ante_vol": ex_ante_vol,
        "vol_budget": float(vol_budget),
        "leverage_cap": float(leverage_cap),
        "growth_ceiling": float(growth_ceiling),
    }
    return best_w, info


def _onehot(n: int, idx: np.ndarray) -> np.ndarray:
    v = np.zeros(n)
    v[idx] = 1.0
    return v


def _starting_points(prev, caps, cfg, growth_ceiling, cov, vol_budget):
    """Warm start plus a risk-scaled diversified start, for solver robustness."""
    out = []
    p = np.clip(np.asarray(prev, float), 0.0, caps)
    out.append(p)

    inv_vol = 1.0 / np.maximum(annualized_vols(cov), 1e-6)
    u = inv_vol / inv_vol.sum()
    v = math.sqrt(max(float(u @ cov @ u), 1e-18))
    out.append(np.clip(u * (vol_budget / v), 0.0, caps))
    return out


def enforce_constraints(w, cfg, growth_ceiling, cov, vol_budget, leverage_cap):
    """
    Enforce every constraint exactly. SLSQP satisfies constraints only to
    tolerance, and partial adjustment toward a target can leave the executed
    book above the vol cap, so the executed weights are always repaired here.
    De-risking is applied multiplicatively, which preserves composition.
    """
    w = np.clip(np.asarray(w, float), 0.0, np.minimum(cfg.caps, leverage_cap))
    g_idx, d_idx = cfg.growth_idx, cfg.defensive_idx

    for idx, cap in ((g_idx, growth_ceiling), (d_idx, cfg.defensive_combined_max)):
        s = w[idx].sum()
        if s > cap + 1e-12 and s > 0:
            w[idx] *= cap / s

    if w.sum() > leverage_cap + 1e-12 and w.sum() > 0:
        w *= leverage_cap / w.sum()

    v = math.sqrt(max(float(w @ cov @ w), 0.0))
    if v > vol_budget + 1e-12 and v > 0:
        w *= vol_budget / v

    return w


def _fallback(cfg, growth_ceiling, cov, vol_budget, leverage_cap):
    inv_vol = 1.0 / np.maximum(annualized_vols(cov), 1e-6)
    u = inv_vol / inv_vol.sum()
    v = math.sqrt(max(float(u @ cov @ u), 1e-18))
    return enforce_constraints(u * (vol_budget / v), cfg, growth_ceiling, cov,
                   vol_budget, leverage_cap)


def vol_budget_for(level_z: float, cfg: AllocatorConfig) -> float:
    """
    Forecast LEVEL modulates the risk budget by at most +/- sizing_tilt.

    level_z must be a STANDARDIZED level (z-score of the cross-sectional mean
    forecast against its own trailing history). Standardizing makes this
    invariant to the units of whatever forecast engine is plugged in, so an
    XGBoost 20-day excess return and a momentum score behave identically.
    """
    z = float(np.nan_to_num(level_z, nan=0.0))
    tilt = 1.0 + cfg.sizing_tilt * math.tanh(z / max(cfg.sizing_tilt_scale, 1e-9))
    return float(cfg.target_vol * np.clip(tilt, 0.1, 2.0))


def leverage_cap_for(state: str, severe: bool, cfg: AllocatorConfig) -> float:
    """
    Asymmetry is the whole point: lever in calm bull markets, never in crises.
    """
    if severe:
        return 0.0
    if str(state).upper() in {"CRISIS", "DEFENSIVE"}:
        return cfg.crisis_max_leverage
    return cfg.max_leverage


# =============================================================================
# METRICS  (excess-return based)
# =============================================================================


def annualized_metrics(equity: pd.Series,
                       rf_daily: pd.Series | None = None,
                       turnover: pd.Series | None = None) -> dict:
    equity = pd.Series(equity).dropna()
    nan_out = dict.fromkeys(
        ["final_equity", "return", "cagr", "max_drawdown", "vol", "sharpe",
         "sortino", "calmar", "avg_turnover", "annual_turnover"], np.nan)
    if len(equity) < 3:
        return nan_out

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)

    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())

    daily = equity.pct_change().dropna()

    # Excess returns. Cash held inside the portfolio earns the T-bill rate;
    # counting that as excess return is what inflated V2.9.7.2's Sharpe.
    if rf_daily is not None:
        excess = daily - pd.Series(rf_daily).reindex(daily.index).fillna(0.0)
    else:
        excess = daily

    vol = float(daily.std() * math.sqrt(TRADING_DAYS))

    sharpe = 0.0
    if len(excess) > 1 and float(excess.std()) > 0:
        sharpe = float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS))

    downside = excess[excess < 0]
    sortino = 0.0
    if len(downside) > 1 and float(downside.std()) > 0:
        sortino = float(excess.mean() / downside.std() * math.sqrt(TRADING_DAYS))

    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    avg_turnover = float(pd.Series(turnover).mean()) if turnover is not None else 0.0

    return {
        "final_equity": float(equity.iloc[-1]),
        "return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "avg_turnover": avg_turnover,
        "annual_turnover": avg_turnover * TRADING_DAYS,
    }


def block_bootstrap_sharpe(equity: pd.Series,
                           rf_daily: pd.Series | None = None,
                           block: int = 20,
                           draws: int = 1000,
                           seed: int = 7) -> tuple[float, float]:
    """5th/95th percentile Sharpe from a moving-block bootstrap."""
    daily = pd.Series(equity).pct_change().dropna()
    if rf_daily is not None:
        daily = daily - pd.Series(rf_daily).reindex(daily.index).fillna(0.0)
    r = daily.to_numpy()
    n = len(r)
    if n < 3 * block:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(draws, n_blocks))
    out = np.empty(draws)
    for i in range(draws):
        s = np.concatenate([r[j:j + block] for j in starts[i]])[:n]
        sd = s.std()
        out[i] = s.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def deflated_sharpe(observed_sharpe: float,
                    n_trials: int,
                    n_obs: int,
                    skew: float = 0.0,
                    kurt: float = 3.0) -> float:
    """
    Bailey & Lopez de Prado. Probability the observed Sharpe is real given that
    n_trials configurations were searched. Sharpe inputs are ANNUALIZED.
    """
    from scipy.stats import norm

    if n_obs < 10 or n_trials < 1:
        return np.nan
    e = 0.5772156649
    z = norm.ppf(1.0 - 1.0 / max(n_trials, 2))
    z2 = norm.ppf(1.0 - 1.0 / (max(n_trials, 2) * math.e))
    sr_expected_max = (1.0 - e) * z + e * z2      # per-observation units

    sr = observed_sharpe / math.sqrt(TRADING_DAYS)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    stat = (sr - sr_expected_max / math.sqrt(max(n_obs - 1, 1))) \
        * math.sqrt(max(n_obs - 1, 1)) / denom
    return float(norm.cdf(stat))
