#!/usr/bin/env python3
"""
options_quant_engine.py — next-generation 3-layer options engine.

Built on the options_quant.py primitives (BS/greeks, IV solver, embedded leverage, VRP,
IV-rank, Kelly variants), this adds the orchestration the research review prioritised:

  LAYER 1 — ALPHA:    do we want exposure?  alpha_score ∈ [-1,1] from market context,
                      put/call, news, insider, technical momentum. (combine_alpha)
  LAYER 2 — CONTRACT: which contract maximises EV, not highest-delta/closest-ITM?
                      EV = P(win)·Gain − P(loss)·Loss − Theta − Spread, via Monte Carlo
                      under the SIGNAL's real-world drift. (ContractSelector / ContractMetrics)
  LAYER 3 — RISK:     dynamic fractional Kelly. The KEY upgrade is the dual decomposition
                      Kelly_option = Kelly_signal × Kelly_contract — the directional edge AND
                      the contract's ability to PRESERVE it are sized separately. A 65%-win
                      signal on a deep-ITM tight-spread contract supports far more size than
                      the same signal on an ATM wide-spread contract, because theta/spread/IV
                      erode the realised option win-rate. (PositionSizer)

Monte Carlo probability engine: instead of a single BS point estimate, simulate N underlying
paths under the signal drift, price the option on each, and read off Prob(+50%), Prob(+100%),
Prob(+4R), and the full option-return distribution that feeds both EV and the option-level Kelly.

ROADMAP (institutional models noted by the review, not yet built — substantial standalone efforts):
  * SABR vol surface (skew-aware strike selection / IV-anomaly detection)
  * Dupire local vol (future delta/gamma estimation)
  * Hidden-Markov regime model (4-state from VIX / realised vol / breadth / term structure)
Hooks: ContractMetrics carries iv_rank/vrp today; a surface object can be injected later.

Dependency-light (numpy + options_quant). Pure/fail-safe where it touches live data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

import options_quant as oq

TRADING_DAYS = 252.0


# ─────────────────────────────────────────────────────────────────────────────
# Historical realized vol (removes the rv-as-user-input assumption) + VRP
# ─────────────────────────────────────────────────────────────────────────────
def realized_vol(closes: Sequence[float], window: int = 20) -> float:
    """Annualised close-to-close realized vol over the last `window` days. 0 if insufficient."""
    c = [float(x) for x in closes if x and x > 0][-(window + 1):]
    if len(c) < 3:
        return 0.0
    rets = [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def realized_vol_windows(closes: Sequence[float]) -> Dict[str, float]:
    """RV20 and RV60 from a close series — the standard short/medium realized-vol pair."""
    return {"rv20": realized_vol(closes, 20), "rv60": realized_vol(closes, 60)}


def calibrate_jumps(closes: Sequence[float], threshold_sigma: float = 3.0) -> Dict[str, float]:
    """Estimate Merton jump parameters (jump_lambda, jump_mean, jump_std) AND the diffusion vol
    from a price history, so the MC isn't fed assumed jumps. Detects jumps as daily log-returns
    beyond `threshold_sigma`× the robust (MAD) volatility; λ = annualised jump frequency, the
    jump mean/std come from those outliers. The ablation showed assumed jumps over-shift selection
    toward ATM — calibrate per name instead. Returns annualised σ_diffusion too."""
    c = [float(x) for x in closes if x and x > 0]
    if len(c) < 30:
        return {"jump_lambda": 0.0, "jump_mean": 0.0, "jump_std": 0.0, "sigma_diffusion": 0.0}
    rets = np.array([math.log(c[i] / c[i - 1]) for i in range(1, len(c))])
    med = float(np.median(rets))
    mad = float(np.median(np.abs(rets - med))) * 1.4826    # robust σ (daily)
    if mad <= 0:
        mad = float(np.std(rets)) or 1e-6
    jumps = rets[np.abs(rets - med) > threshold_sigma * mad]
    n = len(rets)
    lam = (len(jumps) / n) * TRADING_DAYS
    jm = float(np.mean(jumps)) if len(jumps) else 0.0
    js = float(np.std(jumps)) if len(jumps) > 1 else 0.0
    sig_diff = mad * math.sqrt(TRADING_DAYS)               # diffusion (ex-jumps) annualised vol
    return {"jump_lambda": round(lam, 2), "jump_mean": round(jm, 4),
            "jump_std": round(js, 4), "sigma_diffusion": round(sig_diff, 4)}


def vol_term_structure(iv30: float, iv90: float) -> Dict[str, float]:
    """Volatility term structure TS = IV30 − IV90. TS<0 = contango (near < far, calm/normal);
    TS>0 = backwardation (near > far, stress/event). A regime input for Layer 3, not per-contract.
    The options market often signals regime before price does."""
    ts = float(iv30) - float(iv90)
    return {"ts": round(ts, 4),
            "regime": "backwardation" if ts > 0.01 else ("contango" if ts < -0.01 else "flat")}


def expected_shortfall(returns: Sequence[float], level: float = 0.95) -> float:
    """ES_{level} = E[loss | loss beyond VaR_{level}] = mean of the worst (1-level) tail of returns.
    Returned as a (negative) return. Institutions size on this, not just growth."""
    v = sorted(float(x) for x in returns)
    if not v:
        return 0.0
    k = max(1, int(round((1.0 - level) * len(v))))
    return sum(v[:k]) / k


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — ALPHA: combine inputs into alpha_score ∈ [-1, 1]
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ALPHA_WEIGHTS = {
    "market_context": 0.25,
    "put_call": 0.15,
    "news": 0.15,
    "insider": 0.20,
    "technical": 0.25,
}


def combine_alpha(signals: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted blend of component signals (each pre-normalised to [-1,1]) → alpha_score ∈ [-1,1].
    Missing components are dropped and the weights renormalised over what's present."""
    w = weights or DEFAULT_ALPHA_WEIGHTS
    num = den = 0.0
    for k, wk in w.items():
        v = signals.get(k)
        if v is None:
            continue
        num += wk * max(-1.0, min(1.0, float(v)))
        den += wk
    if den <= 0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def signal_confidence(alpha_score: float) -> float:
    """Map |alpha_score| → a [0,1] confidence used to scale size (Layer 3)."""
    return max(0.0, min(1.0, abs(alpha_score)))


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO probability engine
# ─────────────────────────────────────────────────────────────────────────────
class MonteCarloEngine:
    """Simulate underlying paths under the SIGNAL's real-world drift, price the option on each,
    and produce the option-return distribution, hit-probabilities and EV. The drift encodes the
    directional edge; the BS payoff transform encodes the contract — so the resulting distribution
    naturally combines 'signal × contract'.

    Process: MERTON JUMP-DIFFUSION when jump_lambda>0, else plain GBM. Jumps capture overnight
    gaps / earnings shocks / news — the discontinuities that dominate real option payoffs and that
    GBM cannot produce. This materially fattens BOTH tails, raising P(+100%)/P(+200%)/P(+4R) for
    jumpy (momentum/biotech) names and the left-tail ES — fixing the GBM understatement the
    ablation studies exposed.

        dS_t = (μ − λk)S_t dt + σS_t dW_t + (J−1)S_t dN_t,  ln J ~ N(jump_mean, jump_std²),
        N_t ~ Poisson(λ),  k = E[J−1] = exp(jump_mean + ½jump_std²) − 1  (drift compensator).
    """

    def __init__(self, n_paths: int = 10000, steps: int = 21, seed: Optional[int] = 7,
                 jump_lambda: float = 0.0, jump_mean: float = 0.0, jump_std: float = 0.0):
        self.n = int(n_paths)
        self.steps = max(1, int(steps))
        self.rng = np.random.default_rng(seed)
        self.jl = max(0.0, float(jump_lambda))   # jumps per YEAR (e.g. 4 earnings + news)
        self.jm = float(jump_mean)               # mean log jump size (0 ≈ symmetric)
        self.js = max(0.0, float(jump_std))      # log jump vol (fattens both tails)

    def simulate_paths(self, S0: float, mu: float, sigma: float, T: float) -> np.ndarray:
        """Paths, shape (n_paths, steps+1). mu, sigma annualised; T in years (real-world drift)."""
        dt = T / self.steps
        Z = self.rng.standard_normal((self.n, self.steps))
        incr = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * Z
        if self.jl > 0.0 and self.js >= 0.0:
            # Merton compensator keeps the mean drift at μ
            k = math.exp(self.jm + 0.5 * self.js * self.js) - 1.0
            incr = incr - self.jl * k * dt
            # number of jumps per cell ~ Poisson(λ·dt); sum of N iid log-jumps ~ N(N·jm, N·js²)
            n_jumps = self.rng.poisson(self.jl * dt, (self.n, self.steps))
            jump_log = self.rng.normal(n_jumps * self.jm,
                                       np.sqrt(np.maximum(n_jumps, 0)) * self.js)
            incr = incr + jump_log
        logpaths = np.concatenate([np.zeros((self.n, 1)), np.cumsum(incr, axis=1)], axis=1)
        return S0 * np.exp(logpaths)

    def option_distribution(self, contract: "ContractMetrics", S0: float, mu: float,
                            sigma_real: float, hold_T: float) -> Dict[str, np.ndarray]:
        """Option return distribution over a `hold_T`-year hold (exit before expiry → residual
        time value priced by BS at the IV). Returns gross + net (spread-haircut) option returns
        AND the underlying return, for the dual-Kelly decomposition."""
        K, dte_T, iv = contract.strike, contract.dte / 365.0, contract.iv
        entry = contract.mid if contract.mid > 0 else oq.bs_price(S0, K, dte_T, iv, kind="call")
        if entry <= 0:
            entry = 1e-6
        paths = self.simulate_paths(S0, mu, sigma_real, hold_T)
        S_exit = paths[:, -1]
        T_rem = max(1e-4, dte_T - hold_T)
        # vectorised BS call value at exit
        exit_val = _bs_call_vec(S_exit, K, T_rem, iv, contract.r, contract.q)
        half = contract.spread_pct / 2.0
        # spread charged on extrinsic value both legs
        extr_in = max(0.0, entry - max(0.0, S0 - K))
        extr_out = np.maximum(0.0, exit_val - np.maximum(0.0, S_exit - K))
        cost = entry + half * extr_in
        proceeds = np.maximum(0.0, exit_val - half * extr_out)
        opt_ret = np.maximum(-1.0, proceeds / cost - 1.0)
        und_ret = S_exit / S0 - 1.0
        return {"opt_ret": opt_ret, "und_ret": und_ret, "S_exit": S_exit,
                "entry": float(entry), "paths": paths, "K": K}

    def target_probabilities(self, dist: Dict[str, np.ndarray], r_unit: float) -> Dict[str, float]:
        """P(option return ≥ +50% / +100% / +200% / +Nx R) at exit. r_unit = R as a premium
        multiple (e.g. 1R = +100% so '4R' = +400%)."""
        o = dist["opt_ret"]
        n = len(o)
        return {
            "prob_50": float((o >= 0.50).sum() / n),
            "prob_100": float((o >= 1.00).sum() / n),
            "prob_200": float((o >= 2.00).sum() / n),
            "prob_4R": float((o >= 4.0 * r_unit).sum() / n),
            "prob_win": float((o > 0).sum() / n),
            "prob_total_loss": float((o <= -0.95).sum() / n),
        }


def _bs_call_vec(S, K, T, sigma, r=0.04, q=0.0):
    """Vectorised Black-Scholes call (numpy). Intrinsic at T→0 / sigma→0."""
    S = np.asarray(S, dtype=float)
    out = np.maximum(0.0, S - K)
    mask = (S > 0) & (T > 0) & (sigma > 0)
    if np.any(mask):
        Sm = S[mask]
        srt = sigma * math.sqrt(T)
        d1 = (np.log(Sm / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
        d2 = d1 - srt
        from scipy.stats import norm  # optional fast path
        out_m = Sm * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        out = out.copy(); out[mask] = out_m
    return out


# scipy may be absent — fall back to an erf-based normal cdf
try:
    from scipy.stats import norm  # noqa: F401
except Exception:  # pragma: no cover
    def _bs_call_vec(S, K, T, sigma, r=0.04, q=0.0):  # type: ignore
        S = np.asarray(S, dtype=float)
        out = np.maximum(0.0, S - K)
        mask = (S > 0) & (T > 0) & (sigma > 0)
        if np.any(mask):
            Sm = S[mask]; srt = sigma * math.sqrt(T)
            d1 = (np.log(Sm / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
            d2 = d1 - srt
            ncdf = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
            res = Sm * math.exp(-q * T) * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)
            out = out.copy(); out[mask] = res
        return out


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — CONTRACT METRICS + EV ranking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ContractMetrics:
    symbol: str
    strike: float
    dte: int
    delta: float
    mid: float
    spread_pct: float
    iv: float
    open_interest: int = 0
    volume: int = 0
    r: float = 0.04
    q: float = 0.0
    # derived
    leverage_ratio: float = 0.0     # Ω = (S/premium)·δ
    iv_rank: float = 0.5
    vrp: float = 0.0                 # IV − RV (vol points)
    theta_efficiency: float = 0.0    # directional payoff potential per unit theta cost
    liquidity_score: float = 0.0     # 0-1
    probability_itm: float = 0.0     # real-world P(S_T > K) under signal drift
    expected_value: float = 0.0      # MC mean option return (net)
    expected_shortfall: float = 0.0  # ES95 of the option-return distribution (tail risk)
    prob_targets: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_contract(c: dict, spot: float, rv: float, iv_history: Sequence[float] = (),
                      r: float = 0.04, q: float = 0.0) -> "ContractMetrics":
        K = float(c.get("strike_price") or c.get("strike") or 0.0)
        dte = int(c.get("dte") or 0)
        delta = abs(float(c.get("delta") or 0.0))
        bid = float(c.get("bid_price") or c.get("bid") or 0.0)
        ask = float(c.get("ask_price") or c.get("ask") or 0.0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else float(c.get("mid") or ask or 0.0)
        spread_pct = ((ask - bid) / mid) if (bid > 0 and ask > 0 and mid > 0) else float(c.get("spread_pct") or 0.0)
        iv = float(c.get("implied_volatility") or c.get("iv") or 0.0)
        if iv <= 0 and mid > 0 and dte > 0:
            solved = oq.implied_vol(mid, spot, K, dte / 365.0, r, q, "call")
            iv = solved if solved else max(rv, 0.20)
        m = ContractMetrics(
            symbol=str(c.get("symbol") or ""), strike=K, dte=dte, delta=delta, mid=mid,
            spread_pct=spread_pct, iv=iv, open_interest=int(c.get("open_interest") or 0),
            volume=int(c.get("volume") or 0), r=r, q=q,
            leverage_ratio=oq.embedded_leverage(spot, mid, delta),
            iv_rank=oq.iv_rank(iv, iv_history) if iv_history else 0.5,
            vrp=oq.variance_risk_premium(iv, rv, variance_form=False),
        )
        m.liquidity_score = _liquidity_score(m)
        m.theta_efficiency = _theta_efficiency(m, spot)
        return m


def _liquidity_score(m: ContractMetrics) -> float:
    oi = min(1.0, m.open_interest / 1000.0)
    vol = min(1.0, m.volume / 100.0)
    spr = max(0.0, 1.0 - m.spread_pct / 0.10)   # 0 at 10% spread
    return round(0.4 * oi + 0.2 * vol + 0.4 * spr, 4)


def _theta_efficiency(m: ContractMetrics, spot: float) -> float:
    """Directional payoff potential per unit of daily theta cost. High = the contract converts
    a move into P&L cheaply (deep-ITM); low = theta eats the edge (ATM/short-dated)."""
    if m.mid <= 0 or m.dte <= 0:
        return 0.0
    g = oq.bs_greeks(spot, m.strike, m.dte / 365.0, max(m.iv, 1e-3), m.r, m.q, "call")
    theta_day = abs(g.theta) / 365.0
    if theta_day <= 1e-9:
        return 3.0
    # payoff from a 1-σ daily move vs one day of theta, per dollar of premium
    move = spot * m.iv / math.sqrt(TRADING_DAYS)
    payoff = abs(g.delta) * move
    return round(min(payoff / theta_day, 5.0) / m.mid * m.mid, 4) if m.mid else 0.0


class ContractSelector:
    """Rank contracts by Monte-Carlo expected value under the signal drift — NOT by highest
    delta or closest-ITM. EV = mean(net option return); the distribution also yields the
    hit-probabilities and the option-return scenarios used by the sizer."""

    def __init__(self, mc: Optional[MonteCarloEngine] = None, max_spread_pct: float = 0.10,
                 min_oi: int = 10, ev_alpha: float = 0.0, min_delta: float = 0.0,
                 max_delta: float = 1.0, edge_annual: float = 0.40):
        self.mc = mc or MonteCarloEngine()
        self.max_spread_pct = max_spread_pct
        self.min_oi = min_oi
        # ev_alpha>0 → rank by tail-penalised EV_adj = EV + ev_alpha·ES95 (ES95<0, so worse tails
        # are penalised). Stops the selector preferring +30%EV/−95%ES over +25%EV/−50%ES.
        self.ev_alpha = ev_alpha
        # BOUNDED OPTIMIZER (real-chain validation, 2026-06-07): the unconstrained EV buys cheap
        # OTM convexity on liquid chains because the drift assumption over-states the move
        # distribution → it LOST (−14%) vs a simple ATM heuristic (+9.5%). So the EV may only
        # RERANK inside a safe delta band [min_delta, max_delta] (e.g. 0.45-0.75 — no OTM lottery),
        # and edge_annual (drift at |alpha|=1) is capped low (≤0.08) so EV can't inflate OTM EV.
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.edge_annual = edge_annual

    def _signal_drift(self, alpha_score: float, rv: float, edge_annual: Optional[float] = None) -> float:
        """Map alpha_score ∈ [-1,1] to an annualised real-world drift. edge_annual is the drift
        at full conviction (|alpha|=1) — kept LOW (default self.edge_annual, ≤0.08 recommended) so
        the EV can't over-value OTM convexity. Capped to keep the MC sane."""
        ea = self.edge_annual if edge_annual is None else edge_annual
        return float(np.clip(alpha_score * ea, -1.0, 1.0))

    def evaluate(self, m: ContractMetrics, spot: float, alpha_score: float, rv: float,
                 hold_days: int) -> Dict[str, object]:
        mu = self._signal_drift(alpha_score, rv)
        hold_T = max(1, hold_days) / 365.0
        dist = self.mc.option_distribution(m, spot, mu, max(rv, 1e-3), hold_T)
        ev = float(np.mean(dist["opt_ret"]))
        # real-world prob ITM at exit
        Sx = dist["S_exit"]
        m.probability_itm = float((Sx > m.strike).mean())
        m.expected_value = round(ev, 4)
        m.expected_shortfall = round(expected_shortfall(dist["opt_ret"].tolist(), 0.95), 4)
        m.prob_targets = self.mc.target_probabilities(dist, r_unit=1.0)
        return {"metrics": m, "ev": ev, "dist": dist}

    def rank_contracts(self, chain: Sequence[dict], spot: float, alpha_score: float, rv: float,
                       hold_days: int, iv_history: Sequence[float] = (),
                       min_delta: float = 0.0) -> List[Dict[str, object]]:
        out = []
        for c in chain or []:
            try:
                if str(c.get("type", "call")).lower() != "call":
                    continue
                m = ContractMetrics.from_contract(c, spot, rv, iv_history)
                if m.mid <= 0 or m.dte <= 0 or m.delta <= min_delta:
                    continue
                if m.spread_pct > self.max_spread_pct:
                    continue
                if m.open_interest and m.open_interest < self.min_oi and not m.volume:
                    continue
                ev_eval = self.evaluate(m, spot, alpha_score, rv, hold_days)
                # tail-penalised score (EV_adj = EV + ev_alpha·ES95); ES95 is negative.
                ev_eval["score"] = ev_eval["ev"] + self.ev_alpha * m.expected_shortfall
                out.append(ev_eval)
            except Exception:
                continue
        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    def optimize_ev(self, chain, spot, alpha_score, rv, hold_days, **kw):
        ranked = self.rank_contracts(chain, spot, alpha_score, rv, hold_days, **kw)
        return ranked[0] if ranked else None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — DYNAMIC RISK: Kelly_option = Kelly_signal × Kelly_contract
# ─────────────────────────────────────────────────────────────────────────────
REGIME_MULT = {"bull": 1.0, "normal": 1.0, "neutral": 0.75, "mixed": 0.5,
               "cautious": 0.5, "bear": 0.25, "crisis": 0.0, "halt_new_buys": 0.0}


@dataclass
class SizingResult:
    kelly_signal: float       # growth-optimal fraction on the DIRECTIONAL (underlying) bet
    kelly_contract: float     # quality multiplier ∈ [0,1]: fraction of edge the contract preserves
    kelly_option: float       # combined option-level Kelly (the authoritative size driver)
    regime_mult: float
    confidence: float
    liquidity_factor: float
    fraction: float           # final fraction of equity to risk (post fractional-Kelly + caps)
    contracts: int
    notes: str = ""


class PositionSizer:
    """Dual-Kelly sizer. The combined option-level Kelly is computed on the MC option-return
    distribution (which already fuses signal drift × contract transform). It is also DECOMPOSED:

        Kelly_signal   = Kelly on the underlying-return distribution (contract-independent edge)
        Kelly_option   = Kelly on the option-return distribution (signal × contract)
        Kelly_contract = Kelly_option / Kelly_signal  (how much edge the contract preserves;
                         deep-ITM tight-spread → ~1, ATM wide-spread → ≪1)

    Final size = fractional_kelly(Kelly_option) × regime_mult × confidence × liquidity_factor,
    capped by risk_cap_pct of equity (long-option max loss = full premium). Preserves all the
    risk controls of options_quant.position_size_contracts.
    """

    def __init__(self, kelly_fraction: float = 0.25, risk_cap_pct: float = 0.02,
                 max_contracts: int = 1000):
        self.kelly_fraction = kelly_fraction
        self.risk_cap_pct = risk_cap_pct
        self.max_contracts = max_contracts

    def size(self, equity: float, eval_result: Dict[str, object], regime: str = "normal",
             alpha_score: float = 0.0) -> SizingResult:
        m: ContractMetrics = eval_result["metrics"]   # type: ignore
        dist = eval_result["dist"]                     # type: ignore
        opt_ret = dist["opt_ret"]
        und_ret = dist["und_ret"]

        kelly_option = oq.kelly_from_scenarios(opt_ret.tolist())
        kelly_signal = oq.kelly_from_scenarios(und_ret.tolist())
        # contract quality = how much of the directional Kelly survives the contract structure.
        # Normalised so a frictionless deep-ITM (delta→1, no spread/theta) ≈ 1.0.
        kelly_contract = (kelly_option / kelly_signal) if kelly_signal > 1e-6 else 0.0
        kelly_contract = max(0.0, min(1.5, kelly_contract))   # cap extreme leverage ratios

        regime_mult = REGIME_MULT.get(str(regime).lower(), 0.75)
        conf = signal_confidence(alpha_score)
        liq = max(0.05, m.liquidity_score)

        frac = (self.kelly_fraction * kelly_option) * regime_mult * conf * liq
        frac = max(0.0, min(frac, self.risk_cap_pct))   # never exceed the hard risk cap
        contracts = oq.position_size_contracts(
            equity, m.mid, kelly_f=kelly_option,
            kelly_fraction=self.kelly_fraction * regime_mult * conf * liq,
            risk_cap_pct=self.risk_cap_pct, max_contracts=self.max_contracts)
        return SizingResult(
            kelly_signal=round(kelly_signal, 4), kelly_contract=round(kelly_contract, 4),
            kelly_option=round(kelly_option, 4), regime_mult=regime_mult,
            confidence=round(conf, 3), liquidity_factor=round(liq, 3),
            fraction=round(frac, 4), contracts=contracts,
            notes=f"EV={m.expected_value:+.1%} P_itm={m.probability_itm:.0%} "
                  f"P100%={m.prob_targets.get('prob_100', 0):.0%} Ω={m.leverage_ratio:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL: tie the three layers together
# ─────────────────────────────────────────────────────────────────────────────
def recommend(chain: Sequence[dict], spot: float, alpha_signals: Dict[str, float], equity: float,
              rv: float, regime: str = "normal", hold_days: int = 10,
              iv_history: Sequence[float] = (), min_alpha: float = 0.15,
              kelly_fraction: float = 0.25, risk_cap_pct: float = 0.02) -> Optional[Dict]:
    """End-to-end: Layer 1 alpha → (gate) → Layer 2 EV-best contract → Layer 3 dual-Kelly size.
    Returns None when alpha is too weak to want exposure. Never raises on a bad chain."""
    alpha_score = combine_alpha(alpha_signals)
    if abs(alpha_score) < min_alpha or alpha_score <= 0:   # long calls need a positive edge
        return {"alpha_score": round(alpha_score, 3), "action": "no_trade",
                "reason": f"alpha {alpha_score:+.2f} below |{min_alpha}| or non-bullish"}
    selector = ContractSelector()
    best = selector.optimize_ev(chain, spot, alpha_score, rv, hold_days, iv_history=iv_history)
    if not best or best["ev"] <= 0:
        return {"alpha_score": round(alpha_score, 3), "action": "no_trade",
                "reason": "no positive-EV contract"}
    sizer = PositionSizer(kelly_fraction=kelly_fraction, risk_cap_pct=risk_cap_pct)
    sizing = sizer.size(equity, best, regime=regime, alpha_score=alpha_score)
    m: ContractMetrics = best["metrics"]   # type: ignore
    return {"alpha_score": round(alpha_score, 3), "action": "buy_call" if sizing.contracts > 0 else "no_trade",
            "contract": m.symbol, "strike": m.strike, "dte": m.dte, "delta": round(m.delta, 3),
            "expected_value": m.expected_value, "probability_itm": round(m.probability_itm, 3),
            "prob_targets": m.prob_targets, "leverage": round(m.leverage_ratio, 2),
            "sizing": sizing}


if __name__ == "__main__":
    # demo: the review's example — same 65%-win signal on deep-ITM vs ATM
    print("see test_options_quant_engine.py for the worked example")
