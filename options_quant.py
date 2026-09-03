#!/usr/bin/env python3
"""
options_quant.py — research-grounded core library for the long-options engine.

Every formula here traces to a peer-reviewed source (see CITATIONS). It consolidates the
scattered Black-Scholes/greeks code (alpha_call_engine.py, backtest_insider_options.py)
into one tested module and adds the four evidence-backed pillars from the options-research
deep dive (2026-06-07):

  1. CONTRACT SELECTION via embedded leverage Ω = (S/V)·δ  [Frazzini & Pedersen, RAPS 2022].
     CRITICAL: high-embedded-leverage options (short-dated OTM) have NEGATIVE risk-adjusted
     returns — delta-hedged alphas are negative and ~monotonic in leverage. So a directional
     buyer should TILT TOWARD higher-delta / longer-DTE (ITM-ish) contracts and treat a high
     Ω as a COST, not a benefit. (This is the academic explanation of our own backtest result
     that deep-ITM δ≈0.80 calls beat ATM/OTM.)

  2. VOL-AWARE ENTRY via the variance risk premium VRP = IV − RV
     [Bollerslev-Tauchen-Zhou, RFS 2009], which is REGIME-CONDITIONAL — predictive only when
     corr(return, variance innovation) is strongly negative [Pyun, JFE 2019]. Plus IV rank /
     percentile (practitioner standard).

  3. SIZING via Kelly, shrunk for dispersed payoffs. Mean-payoff Kelly OVER-sizes lottery-like
     option bets: f̂ ≤ f* with equality only for constant payoff [Perez-Marco, arXiv:1411.3615].
     We compute the EMPIRICAL variable-payoff Kelly directly (argmax_f E[log(1+f·R)] over
     scenarios) — the rigorous form — then apply a fractional multiplier and a hard cap.
     Risk-Constrained Kelly (drawdown-bounded convex program) [Busseti-Ryu-Boyd] is documented
     in kelly_risk_constrained() as the optional rigorous extension.

  4. GREEKS: minimum-variance delta δ_MV = δ_BS + vega·(a+b·δ_BS+c·δ_BS²)/(S√T)
     [Hull & White, JBF 2017]. BS delta over-states equity directional exposure because price
     and implied vol are negatively correlated. a,b,c MUST be fit per name/sector — the paper's
     S&P-index values are NOT reused (default = no-op until fit_mv_delta_coeffs() is called).

CITATIONS
  - Frazzini, A. & Pedersen, L.H. "Embedded Leverage." Review of Asset Pricing Studies.
    http://docs.lhpedersen.com/EmbeddedLeverage.pdf
  - Bollerslev, T., Tauchen, G., Zhou, H. "Expected Stock Returns and Variance Risk Premia." RFS 2009.
    https://www.federalreserve.gov/pubs/feds/2007/200711/200711pap.pdf
  - Pyun, S. "Variance risk in aggregate stock returns and time-varying return predictability."
    JFE 132(1), 2019. https://www.sciencedirect.com/science/article/abs/pii/S0304405X18302782
  - Perez-Marco, R. "Kelly criterion for variable pay-off." arXiv:1411.3615.
  - Busseti, E., Ryu, E.K., Boyd, S. "Risk-Constrained Kelly Gambling."
    https://stanford.edu/~boyd//papers/pdf/kelly.pdf
  - Hull, J. & White, A. "Optimal Delta Hedging for Options." J. Banking & Finance 82, 2017.
    https://www.sciencedirect.com/science/article/pii/S0378426617301085
  - Hull, J. "Options, Futures, and Other Derivatives" (BSM price + greeks reference).

All functions are pure, dependency-light (math + statistics only), and never raise on bad
input — they return safe defaults so a live engine is never broken by one odd contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import List, Optional, Sequence, Tuple

_N = NormalDist()  # standard normal


def _cdf(x: float) -> float:
    return _N.cdf(x)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BLACK-SCHOLES-MERTON: price + greeks (with continuous dividend yield q)
#    Hull, "Options, Futures, and Other Derivatives". Verified numerically in tests
#    (put-call parity, finite-difference greeks, IV round-trip).
# ─────────────────────────────────────────────────────────────────────────────
def _d1_d2(S: float, K: float, T: float, sigma: float, r: float, q: float) -> Tuple[float, float]:
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
    return d1, d1 - srt


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.04,
             q: float = 0.0, kind: str = "call") -> float:
    """Black-Scholes-Merton price. At/near T→0 or σ→0 returns intrinsic value."""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if kind == "call" else (K - S)
        return max(0.0, intrinsic)
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    if kind == "call":
        return S * math.exp(-q * T) * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    return K * math.exp(-r * T) * _cdf(-d2) - S * math.exp(-q * T) * _cdf(-d1)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float       # per 1.00 (100%) vol move; divide by 100 for per-1%-vol
    theta: float      # per YEAR; divide by 365 for per-calendar-day
    rho: float        # per 1.00 (100%) rate move


def bs_greeks(S: float, K: float, T: float, sigma: float, r: float = 0.04,
              q: float = 0.0, kind: str = "call") -> Greeks:
    """Full BSM greeks. Gamma and vega are identical for calls and puts; delta/theta/rho branch."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        intrinsic_delta = (1.0 if (kind == "call" and S > K) else
                           (-1.0 if (kind == "put" and S < K) else 0.0))
        return Greeks(bs_price(S, K, T, sigma, r, q, kind), intrinsic_delta, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    sqrtT = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = _pdf(d1)
    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT
    if kind == "call":
        delta = disc_q * _cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * disc_r * _cdf(d2) + q * S * disc_q * _cdf(d1))
        rho = K * T * disc_r * _cdf(d2)
    else:
        delta = -disc_q * _cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * disc_r * _cdf(-d2) - q * S * disc_q * _cdf(-d1))
        rho = -K * T * disc_r * _cdf(-d2)
    return Greeks(bs_price(S, K, T, sigma, r, q, kind), delta, gamma, vega, theta, rho)


def implied_vol(price: float, S: float, K: float, T: float, r: float = 0.04, q: float = 0.0,
                kind: str = "call", tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
    """Invert BSM for σ. Newton-Raphson on vega, with a bisection fallback when vega is tiny
    (deep ITM/OTM or near expiry) — the robust hybrid. Returns None if no root in [1e-4, 5.0]
    or the price violates no-arbitrage bounds.
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, (S - K) if kind == "call" else (K - S)) * math.exp(-r * T)
    upper = (S * math.exp(-q * T)) if kind == "call" else (K * math.exp(-r * T))
    if price < intrinsic - tol or price > upper + tol:
        return None  # outside no-arbitrage bounds
    lo, hi = 1e-4, 5.0
    sigma = 0.20  # standard initial guess
    for _ in range(max_iter):
        g = bs_greeks(S, K, T, sigma, r, q, kind)
        diff = g.price - price
        if abs(diff) < tol:
            return sigma
        if g.vega > 1e-8:
            step = diff / g.vega
            sigma_new = sigma - step
            if lo < sigma_new < hi:
                sigma = sigma_new
                continue
        # bisection fallback
        if bs_price(S, K, T, lo, r, q, kind) - price > 0:
            return None
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, r, q, kind) - price > 0:
            hi = mid
        else:
            lo = mid
        sigma = 0.5 * (lo + hi)
        if hi - lo < tol:
            return sigma
    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONTRACT SELECTION — embedded leverage + the Frazzini-Pedersen warning
# ─────────────────────────────────────────────────────────────────────────────
def embedded_leverage(spot: float, premium: float, delta: float) -> float:
    """Ω = (S/V)·|δ| — option elasticity / embedded leverage [Frazzini-Pedersen].
    Percentage change in option value per 1% move in the underlying. 0 on bad input."""
    if spot <= 0 or premium <= 0:
        return 0.0
    return abs(delta) * spot / premium


def leverage_efficiency_score(omega: float, lo: float = 2.0, hi: float = 6.0) -> float:
    """Score a contract's embedded leverage in [0,1], peaking inside the target band [lo,hi]
    and DECAYING for higher Ω — encoding the Frazzini-Pedersen result that high-leverage
    (short-dated OTM) options carry a negative expected premium. Below `lo` is fine (cheap
    directional exposure is scarce); above `hi` is penalised increasingly hard.

    The default band [2,6] is the ITM-to-moderate range our deep-ITM backtest favoured and
    sits well below the >16 leverage of short-dated deep-OTM equity options that FP flag as
    worst. Override per signal strength: a stronger edge can justify a higher band.
    """
    if omega <= 0:
        return 0.0
    if omega <= lo:
        return min(1.0, omega / lo)           # ramp up to the band
    if omega <= hi:
        return 1.0                            # in the sweet spot
    # decay above the band: each unit of excess leverage is a tax (FP negative premium)
    return max(0.0, 1.0 / (1.0 + (omega - hi) / hi))


def select_directional_call(chain: Sequence[dict], spot: float, target_dte: int = 30,
                            min_delta: float = 0.70, lev_lo: float = 2.0, lev_hi: float = 6.0,
                            min_oi: int = 10, max_spread_pct: float = 0.25) -> Optional[dict]:
    """Pick the best long CALL to express a bullish directional signal, research-grounded:
    deep-ITM tilt (min_delta high), embedded-leverage in a sane band, liquid, tight spread,
    expiry near target_dte. Each contract dict needs: symbol, expiration_date, strike_price,
    delta, ask_price, bid_price?, open_interest?, volume?, dte? (else computed elsewhere).

    Returns the chosen contract dict augmented with omega/score, or None.
    """
    best, best_key = None, None
    for c in chain or []:
        try:
            if str(c.get("type", "call")).lower() != "call":
                continue
            delta = float(c.get("delta") or 0.0)
            if delta <= min_delta:
                continue
            ask = float(c.get("ask_price") or 0.0)
            if ask <= 0:
                continue
            bid = float(c.get("bid_price") or 0.0)
            spread_pct = ((ask - bid) / ((ask + bid) / 2)) if (bid > 0 and ask > 0) else max_spread_pct
            if spread_pct > max_spread_pct:
                continue
            if int(c.get("open_interest") or 0) < min_oi and int(c.get("volume") or 0) <= 0:
                continue
            dte = int(c.get("dte") or 0)
            omega = embedded_leverage(spot, ask, delta)
            lev_score = leverage_efficiency_score(omega, lev_lo, lev_hi)
            if lev_score <= 0:
                continue
            # rank: leverage-efficiency first, then nearest DTE, then tightest spread
            key = (-round(lev_score, 3), abs(dte - target_dte), spread_pct)
            if best_key is None or key < best_key:
                best_key = key
                best = {**c, "omega": round(omega, 2), "lev_score": round(lev_score, 3)}
        except Exception:
            continue
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOL-AWARE ENTRY — variance risk premium, IV rank/percentile, regime gate
# ─────────────────────────────────────────────────────────────────────────────
def realized_vol(closes: Sequence[float], annualize: int = 252) -> float:
    """Close-to-close annualized realized volatility from a price series."""
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i] > 0 and closes[i - 1] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(annualize)


def variance_risk_premium(iv: float, rv: float, variance_form: bool = True) -> float:
    """VRP [Bollerslev-Tauchen-Zhou]. variance_form=True → IV²−RV² (their definition);
    False → IV−RV (vol points). Positive VRP (IV>RV) is the normal premium-rich regime."""
    if variance_form:
        return iv * iv - rv * rv
    return iv - rv


def iv_rank(current_iv: float, iv_history: Sequence[float]) -> float:
    """IV rank ∈ [0,1] = (IV − min)/(max − min) over the lookback (e.g. 1y). Practitioner std."""
    hist = [x for x in iv_history if x and x > 0]
    if not hist:
        return 0.5
    lo, hi = min(hist), max(hist)
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (current_iv - lo) / (hi - lo)))


def iv_percentile(current_iv: float, iv_history: Sequence[float]) -> float:
    """IV percentile ∈ [0,1] = fraction of lookback days with IV below current."""
    hist = [x for x in iv_history if x and x > 0]
    if not hist:
        return 0.5
    return sum(1 for x in hist if x < current_iv) / len(hist)


def vrp_regime_active(returns: Sequence[float], variances: Sequence[float],
                      corr_threshold: float = -0.25) -> bool:
    """Pyun (2019): the VRP entry filter is only reliable when corr(return, variance innovation)
    is sufficiently NEGATIVE. Returns True when the rolling correlation ≤ corr_threshold, i.e.
    the leverage-effect regime where VRP predicts returns. Safe-True on insufficient data so the
    gate never silently blocks everything."""
    n = min(len(returns), len(variances))
    if n < 20:
        return True
    r = list(returns)[-n:]
    v = list(variances)[-n:]
    dv = [v[i] - v[i - 1] for i in range(1, n)]   # variance innovations
    rr = r[1:]
    m = min(len(rr), len(dv))
    rr, dv = rr[-m:], dv[-m:]
    mr, mv = sum(rr) / m, sum(dv) / m
    cov = sum((rr[i] - mr) * (dv[i] - mv) for i in range(m)) / m
    sr = math.sqrt(sum((x - mr) ** 2 for x in rr) / m)
    sv = math.sqrt(sum((x - mv) ** 2 for x in dv) / m)
    if sr <= 0 or sv <= 0:
        return True
    return (cov / (sr * sv)) <= corr_threshold


def long_premium_entry_ok(iv: float, rv: float, iv_history: Sequence[float] = (),
                          max_iv_rank: float = 0.85, max_vrp_vol: float = 0.15) -> Tuple[bool, str]:
    """Vol-aware gate for BUYING premium (long options): avoid buying when vol is historically
    rich. Block when IV rank is extreme (you're paying top-of-range premium) or the vol-point VRP
    is very high (IV hugely above RV = expensive). Advisory; returns (ok, reason)."""
    if iv > 0 and iv_history and iv_rank(iv, iv_history) > max_iv_rank:
        return False, f"iv_rank_too_high:{iv_rank(iv, iv_history):.2f}>{max_iv_rank}"
    vrp_v = variance_risk_premium(iv, rv, variance_form=False)
    if vrp_v > max_vrp_vol:
        return False, f"premium_too_rich:vrp_vol={vrp_v:.2f}>{max_vrp_vol}"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIZING — Kelly shrunk for dispersed (lottery-like) option payoffs
# ─────────────────────────────────────────────────────────────────────────────
def kelly_binary(p: float, b: float) -> float:
    """Classical Kelly fraction for a binary bet: f* = (p(b+1) − 1)/b = p − (1−p)/b.
    p = win prob, b = net odds (win amount per unit risked). Clamped to [0,1]."""
    if b <= 0:
        return 0.0
    return max(0.0, min(1.0, (p * (b + 1.0) - 1.0) / b))


def kelly_from_scenarios(returns: Sequence[float], f_max: float = 1.0,
                         grid: int = 400) -> float:
    """EMPIRICAL variable-payoff Kelly: argmax_f E[log(1 + f·R)] over outcome scenarios R
    (each R a per-unit return, e.g. -1.0 = total loss, +2.0 = triple). This is the rigorous
    form for dispersed/asymmetric option payoffs and AUTOMATICALLY satisfies Perez-Marco's
    f̂ ≤ f* shrink (it never over-bets on a fat-tailed distribution the way the mean-payoff
    formula does). Bounded to f ∈ [0, f_max] where 1+f·min(R) stays > 0. 1-D grid search
    (robust, dependency-free). Returns 0 if the bet is not log-favourable.
    """
    rs = [float(r) for r in returns if r is not None]
    if not rs:
        return 0.0
    worst = min(rs)
    # keep 1 + f·worst > 0  → f < -1/worst when worst < 0
    cap = f_max if worst >= 0 else min(f_max, 0.999 / (-worst))
    if cap <= 0:
        return 0.0

    def growth(f: float) -> float:
        tot = 0.0
        for r in rs:
            v = 1.0 + f * r
            if v <= 0:
                return -1e18
            tot += math.log(v)
        return tot / len(rs)

    best_f, best_g = 0.0, growth(0.0)  # = 0
    for i in range(1, grid + 1):
        f = cap * i / grid
        g = growth(f)
        if g > best_g:
            best_g, best_f = g, f
    return best_f if best_g > 0 else 0.0


def position_size_contracts(equity: float, option_premium: float, kelly_f: float,
                            kelly_fraction: float = 0.25, risk_cap_pct: float = 0.02,
                            max_contracts: int = 1000) -> int:
    """Translate a Kelly fraction into a whole-contract count for a long option, applying:
      * fractional Kelly (kelly_fraction, default 1/4 Kelly — standard conservative practice),
      * a hard risk cap (risk_cap_pct of equity at total loss — long-option max loss = premium),
      * whole-contract flooring.
    The binding constraint wins. Long options can lose 100%, so the risk cap is sized on the
    full premium. Returns 0 when nothing fits."""
    if equity <= 0 or option_premium <= 0 or kelly_f <= 0:
        return 0
    cost_per_contract = option_premium * 100.0
    kelly_dollars = equity * kelly_f * max(0.0, min(1.0, kelly_fraction))
    cap_dollars = equity * risk_cap_pct
    budget = min(kelly_dollars, cap_dollars)
    return max(0, min(max_contracts, int(budget // cost_per_contract)))


def kelly_risk_constrained(scenario_returns: Sequence[Sequence[float]], alpha: float = 0.7,
                           beta: float = 0.1) -> Optional[List[float]]:
    """Risk-Constrained Kelly [Busseti-Ryu-Boyd]: maximize E[log(rᵀb)] s.t. 1ᵀb=1, b≥0, and the
    drawdown bound E[(rᵀb)^(−λ)] ≤ 1 with λ = log(β)/log(α), which GUARANTEES Prob(W_min<α)<β.
    This is the rigorous multi-bet sizer; it needs cvxpy (exponential-cone, ECOS). Returns the
    bet-fraction vector, or None if cvxpy is unavailable (caller falls back to fractional Kelly).

    scenario_returns: rows = scenarios, cols = bets (gross return per $1, e.g. 1.0 = break-even).
    """
    try:
        import cvxpy as cp
        import numpy as np
    except Exception:
        return None
    R = np.asarray(scenario_returns, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2:
        return None
    m, n = R.shape
    lam = math.log(beta) / math.log(alpha)
    b = cp.Variable(n, nonneg=True)
    growth = cp.sum(cp.log(R @ b)) / m
    risk = cp.sum(cp.exp(-lam * cp.log(R @ b))) / m   # E[(rᵀb)^-λ]
    prob = cp.Problem(cp.Maximize(growth), [cp.sum(b) == 1, risk <= 1])
    try:
        prob.solve(solver=cp.ECOS)
    except Exception:
        return None
    if b.value is None:
        return None
    return [float(max(0.0, x)) for x in b.value]


# ─────────────────────────────────────────────────────────────────────────────
# 5. GREEKS REFINEMENT — Hull-White minimum-variance delta
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MVDeltaCoeffs:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    fitted: bool = False


def mv_delta(delta_bs: float, vega_bs: float, spot: float, T: float,
             coeffs: MVDeltaCoeffs) -> float:
    """Minimum-variance delta [Hull-White 2017]:
        δ_MV = δ_BS + vega·(a + b·δ_BS + c·δ_BS²)/(S·√T).
    With unfitted coeffs (default a=b=c=0) this is a NO-OP returning δ_BS — deliberately, because
    the paper's S&P-index coefficients must NOT be reused for small/mid-cap single names. Fit per
    name/sector with fit_mv_delta_coeffs() before relying on the correction."""
    if not coeffs.fitted or spot <= 0 or T <= 0:
        return delta_bs
    adj = vega_bs * (coeffs.a + coeffs.b * delta_bs + coeffs.c * delta_bs * delta_bs) / (spot * math.sqrt(T))
    return delta_bs + adj


def fit_mv_delta_coeffs(samples: Sequence[Tuple[float, float, float, float, float]]
                        ) -> MVDeltaCoeffs:
    """Fit (a,b,c) by OLS on Hull-White eq (6): regress realized option P&L f on
        δ_BS·ΔS  +  (vega·ΔS/(√T·S))·(a + b·δ_BS + c·δ_BS²).
    samples: iterable of (f, delta_bs, vega, dS, sqrtT_times_S) per observation, where
        f = realized option price change, dS = underlying change, sqrtT_times_S = √T·S.
    Solves the 3×3 normal equations with pure Python (no numpy). Returns no-op coeffs if the
    system is degenerate. Re-estimate over rolling windows, separately for calls and puts."""
    # design: residual y = f − δ·ΔS ; features x = (vega·ΔS/(√T·S)) · [1, δ, δ²]
    XtX = [[0.0] * 3 for _ in range(3)]
    Xty = [0.0, 0.0, 0.0]
    for f, d, vega, dS, sTS in samples:
        if sTS == 0:
            continue
        base = vega * dS / sTS
        x = [base, base * d, base * d * d]
        y = f - d * dS
        for i in range(3):
            Xty[i] += x[i] * y
            for j in range(3):
                XtX[i][j] += x[i] * x[j]
    coef = _solve3(XtX, Xty)
    if coef is None:
        return MVDeltaCoeffs()
    return MVDeltaCoeffs(a=coef[0], b=coef[1], c=coef[2], fitted=True)


def _solve3(A: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Solve a 3×3 linear system by Gaussian elimination with partial pivoting. None if singular."""
    M = [row[:] + [y[i]] for i, row in enumerate(A)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(3):
            if r != col:
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[col][k] for k in range(4)]
    return [M[i][3] for i in range(3)]


if __name__ == "__main__":
    # quick self-demo
    g = bs_greeks(100, 100, 30 / 365, 0.30, r=0.04, q=0.0, kind="call")
    print("ATM 30D call greeks:", g)
    print("IV round-trip:", implied_vol(g.price, 100, 100, 30 / 365, kind="call"))
    print("Ω(deep-ITM δ0.8, prem $12):", embedded_leverage(100, 12, 0.8),
          "score:", leverage_efficiency_score(embedded_leverage(100, 12, 0.8)))
    print("Kelly scenarios [-1,-1,+3]:", kelly_from_scenarios([-1, -1, 3]))
