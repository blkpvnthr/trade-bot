#!/usr/bin/env python3
"""
buffett_black_litterman.py — Black-Litterman expected-return engine for the Buffett
Portfolio Optimizer.

Replaces "historical return + conviction/valuation/insider adjustments" as the primary
return estimate with:

    Market-Implied Prior Returns (pi)
      + Black-Litterman Views (from Buffett signals)
      + View Confidence (Omega)
      = Posterior Returns (+ posterior covariance)

Uses PyPortfolioOpt's BlackLittermanModel when available; otherwise a closed-form numpy
implementation. Fully fail-safe: on any failure it falls back to a blended
expected-return model (conviction/valuation/insider) so the Buffett layer never stops.

Final rule (this module's slice):
    Buffett analysis decides WHAT is worth owning.
    Black-Litterman determines EXPECTED RETURNS.
"""
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # PyPortfolioOpt is optional — closed-form fallback used when absent.
    from pypfopt import black_litterman as _pf_bl
    from pypfopt.black_litterman import BlackLittermanModel as _PFBL
    _PYPFOPT = True
except Exception:  # pragma: no cover
    _PYPFOPT = False

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
BL_REPORT = "black_litterman_report.json"


def _research_dir(date_tag: str | None = None) -> Path:
    d = DAILY_OUTPUTS / (date_tag or DATA_TAG)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class BLConfig:
    enabled: bool = os.getenv("ENABLE_BUFFETT_BLACK_LITTERMAN", "true").lower() in {"1", "true"}
    tau: float = float(os.getenv("BUFFETT_BL_TAU", "0.05"))
    risk_aversion: float = float(os.getenv("BUFFETT_BL_RISK_AVERSION", "2.5"))
    risk_free_rate: float = float(os.getenv("BUFFETT_BL_RISK_FREE", os.getenv("OPTIONS_RISK_FREE_RATE", "0.043")))
    equity_premium_base: float = float(os.getenv("BUFFETT_BL_EQUITY_PREMIUM", "0.06"))
    # Tilt magnitudes that turn Buffett signals into absolute-view returns.
    conviction_tilt: float = float(os.getenv("BUFFETT_BL_CONVICTION_TILT", "0.12"))
    valuation_tilt: float = float(os.getenv("BUFFETT_BL_VALUATION_TILT", "0.15"))
    insider_tilt: float = float(os.getenv("BUFFETT_BL_INSIDER_TILT", "0.03"))
    macro_tilt: float = float(os.getenv("BUFFETT_BL_MACRO_TILT", "0.04"))
    min_view: float = float(os.getenv("BUFFETT_BL_MIN_VIEW", "-0.10"))
    max_view: float = float(os.getenv("BUFFETT_BL_MAX_VIEW", "0.40"))
    bucket_view_spread: float = float(os.getenv("BUFFETT_BL_BUCKET_SPREAD", "0.03"))
    default_annual_vol: float = float(os.getenv("BUFFETT_BL_DEFAULT_VOL", "0.28"))
    default_correlation: float = float(os.getenv("BUFFETT_BL_DEFAULT_CORR", "0.20"))


# ── covariance construction ───────────────────────────────────────────────────────
def build_covariance(symbols: list[str], prices: pd.DataFrame | None = None,
                     vol_by_symbol: dict[str, float] | None = None,
                     cfg: BLConfig | None = None) -> pd.DataFrame:
    """
    Annualized covariance matrix. Prefers a real price history (sample covariance of
    daily returns); otherwise builds one from per-name vol estimates + a default
    correlation so the optimizer always has a usable, positive-definite matrix.
    """
    cfg = cfg or BLConfig()
    if prices is not None and not prices.empty and prices.shape[1] >= 2:
        cols = [s for s in symbols if s in prices.columns]
        if len(cols) >= 2:
            rets = np.log(prices[cols] / prices[cols].shift(1)).dropna(how="all")
            cov = rets.cov() * 252.0
            return cov.reindex(index=cols, columns=cols).fillna(0.0)
    vol_by_symbol = vol_by_symbol or {}
    vols = np.array([float(vol_by_symbol.get(s, cfg.default_annual_vol) or cfg.default_annual_vol)
                     for s in symbols])
    rho = cfg.default_correlation
    corr = np.full((len(symbols), len(symbols)), rho)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=symbols, columns=symbols)


def market_implied_prior(symbols: list[str], cov: pd.DataFrame,
                         market_caps: dict[str, float] | None = None,
                         cfg: BLConfig | None = None) -> pd.Series:
    """
    Market-equilibrium (implied) prior returns:  pi = delta * Sigma * w_market.
    Uses market caps for the market portfolio when available, else equal weight.
    """
    cfg = cfg or BLConfig()
    caps = np.array([float((market_caps or {}).get(s, 0.0) or 0.0) for s in symbols])
    if caps.sum() <= 0:
        caps = np.ones(len(symbols))     # equal-weight fallback market portfolio
    w_mkt = caps / caps.sum()
    if _PYPFOPT:
        try:
            mcaps = pd.Series(caps, index=symbols)
            pi = _pf_bl.market_implied_prior_returns(mcaps, cfg.risk_aversion, cov,
                                                     risk_free_rate=cfg.risk_free_rate)
            return pi.reindex(symbols).fillna(cfg.equity_premium_base)
        except Exception as exc:
            logger.debug("pypfopt market_implied_prior failed: %s", exc)
    pi = cfg.risk_aversion * (cov.values @ w_mkt)
    return pd.Series(pi, index=symbols)


# ── views + confidence ─────────────────────────────────────────────────────────────
def _signal(sig: dict, *keys, default=0.0):
    for k in keys:
        if sig.get(k) is not None:
            return sig[k]
    return default


def _absolute_view_return(sig: dict, cfg: BLConfig) -> float:
    conviction = float(_signal(sig, "conviction_score"))
    p2iv = float(_signal(sig, "price_to_intrinsic_value"))
    insider = float(_signal(sig, "insider_buy_score"))
    macro = float(_signal(sig, "macro_score", "macro_cycle_opportunity_score"))
    er = cfg.equity_premium_base + cfg.risk_free_rate
    er += (conviction - 60.0) / 100.0 * cfg.conviction_tilt
    if p2iv > 0:
        er += (1.0 - p2iv) * cfg.valuation_tilt          # deeper discount → higher view
    er += (insider / 100.0) * cfg.insider_tilt
    er += (macro / 100.0 - 0.5) * cfg.macro_tilt
    return round(_clamp(er, cfg.min_view, cfg.max_view), 4)


def _view_confidence(sig: dict, cfg: BLConfig) -> float:
    vc = float(_signal(sig, "valuation_confidence"))
    conv = float(_signal(sig, "conviction_score")) / 100.0
    bal = float(_signal(sig, "BAL_score", "bal_score")) / 100.0
    insider = float(_signal(sig, "insider_buy_score")) / 100.0
    n = float(_signal(sig, "research_sample_size", "valuation_model_count")) or 0.0
    sample_conf = min(n / 20.0, 1.0)
    conf = 0.20 + 0.30 * conv + 0.20 * vc + 0.15 * bal + 0.10 * insider + 0.05 * sample_conf
    return round(_clamp(conf, 0.05, 0.95), 4)


def build_views(signals: list[dict], symbols: list[str], cfg: BLConfig | None = None) -> dict:
    """
    Auto-generate Black-Litterman views from Buffett signals:
      * one ABSOLUTE view per candidate (expected return from valuation/conviction/
        quality/macro/insider),
      * RELATIVE views (high-conviction bucket > low-conviction bucket; best vs worst name).
    Returns {P, Q, confidences, view_labels, absolute_views, rejected_views}.
    """
    cfg = cfg or BLConfig()
    n = len(symbols)
    idx = {s: i for i, s in enumerate(symbols)}
    by_symbol = {str(s.get("symbol", "")).upper(): s for s in signals}
    P_rows, Q, confidences, labels, rejected = [], [], [], [], []
    absolute_views = {}

    for s in symbols:
        sig = by_symbol.get(s)
        if not sig:
            rejected.append({"symbol": s, "reason": "no_signal"})
            continue
        er = _absolute_view_return(sig, cfg)
        conf = _view_confidence(sig, cfg)
        row = [0.0] * n
        row[idx[s]] = 1.0
        P_rows.append(row); Q.append(er); confidences.append(conf)
        labels.append({"type": "absolute", "symbol": s, "view_return": er, "confidence": conf})
        absolute_views[s] = er

    # Relative view: high-conviction bucket outperforms low-conviction bucket.
    conv_sorted = sorted([s for s in symbols if s in by_symbol],
                         key=lambda s: float(_signal(by_symbol[s], "conviction_score")), reverse=True)
    if len(conv_sorted) >= 2:
        third = max(1, len(conv_sorted) // 3)
        high, low = conv_sorted[:third], conv_sorted[-third:]
        if not set(high) & set(low):
            row = [0.0] * n
            for s in high:
                row[idx[s]] = 1.0 / len(high)
            for s in low:
                row[idx[s]] = -1.0 / len(low)
            hi_conf = float(np.mean([_view_confidence(by_symbol[s], cfg) for s in high]))
            P_rows.append(row); Q.append(cfg.bucket_view_spread)
            confidences.append(round(_clamp(hi_conf, 0.05, 0.95), 4))
            labels.append({"type": "relative", "high": high, "low": low,
                           "outperformance": cfg.bucket_view_spread})

    P = np.array(P_rows) if P_rows else np.zeros((0, n))
    return {"P": P, "Q": np.array(Q), "confidences": np.array(confidences),
            "view_labels": labels, "absolute_views": absolute_views, "rejected_views": rejected}


def build_omega(P: np.ndarray, confidences: np.ndarray, cov: pd.DataFrame, tau: float) -> np.ndarray:
    """
    Diagonal Omega (view uncertainty) from per-view confidence:
        omega_k = (1/conf_k - 1) * (P_k · tau·Sigma · P_kᵀ)
    Higher confidence → smaller omega → stronger view influence.
    """
    if P.shape[0] == 0:
        return np.zeros((0, 0))
    S = cov.values
    base = np.array([float(P[k] @ (tau * S) @ P[k].T) for k in range(P.shape[0])])
    base = np.maximum(base, 1e-10)
    conf = np.clip(confidences, 0.01, 0.99)
    omega_diag = (1.0 / conf - 1.0) * base
    return np.diag(np.maximum(omega_diag, 1e-12))


# ── posterior (BL) ─────────────────────────────────────────────────────────────────
def _closed_form_bl(pi: pd.Series, cov: pd.DataFrame, P: np.ndarray, Q: np.ndarray,
                    omega: np.ndarray, tau: float) -> tuple[pd.Series, pd.DataFrame]:
    """Closed-form Black-Litterman posterior (used when PyPortfolioOpt is unavailable)."""
    S = cov.values
    tauS = tau * S
    inv_tauS = np.linalg.pinv(tauS)
    if P.shape[0] == 0:
        return pi.copy(), pd.DataFrame(S + tauS, index=cov.index, columns=cov.columns)
    inv_omega = np.linalg.pinv(omega)
    M = np.linalg.pinv(inv_tauS + P.T @ inv_omega @ P)
    post_mean = M @ (inv_tauS @ pi.values + P.T @ inv_omega @ Q)
    post_cov = S + M
    return (pd.Series(post_mean, index=cov.index),
            pd.DataFrame(post_cov, index=cov.index, columns=cov.columns))


def black_litterman_posterior(pi: pd.Series, cov: pd.DataFrame, views: dict,
                              cfg: BLConfig | None = None) -> dict:
    """Posterior returns + covariance via PyPortfolioOpt (preferred) or closed-form."""
    cfg = cfg or BLConfig()
    P, Q, conf = views["P"], views["Q"], views["confidences"]
    omega = build_omega(P, conf, cov, cfg.tau)
    used = "closed_form"
    if _PYPFOPT and P.shape[0] > 0:
        try:
            bl = _PFBL(cov, pi=pi, P=P, Q=Q, omega=omega, tau=cfg.tau,
                       risk_aversion=cfg.risk_aversion)
            post_ret = bl.bl_returns()
            post_cov = bl.bl_cov()
            return {"posterior_returns": post_ret, "posterior_covariance": post_cov,
                    "omega": omega, "engine": "pypfopt"}
        except Exception as exc:
            logger.debug("pypfopt BlackLittermanModel failed (%s) — closed-form fallback", exc)
    post_ret, post_cov = _closed_form_bl(pi, cov, P, Q, omega, cfg.tau)
    return {"posterior_returns": post_ret, "posterior_covariance": post_cov,
            "omega": omega, "engine": used}


# ── blended fallback expected returns (when BL fails entirely) ──────────────────────
def blended_expected_returns(signals: list[dict], symbols: list[str], cfg: BLConfig | None = None) -> pd.Series:
    """Fallback: the prior conviction/valuation/insider blend as an expected-return vector."""
    cfg = cfg or BLConfig()
    by_symbol = {str(s.get("symbol", "")).upper(): s for s in signals}
    out = {}
    for s in symbols:
        sig = by_symbol.get(s, {})
        out[s] = _absolute_view_return(sig, cfg) if sig else (cfg.equity_premium_base + cfg.risk_free_rate)
    return pd.Series(out)


# ── orchestrator ────────────────────────────────────────────────────────────────────
def _market_state_confidence_multiplier(market_state: dict | None) -> float:
    """Scale BL view confidence by the shared market_state (1.0 when absent).

    Lower market_state confidence (or a high-volatility regime) shrinks view confidence,
    which widens omega and pulls the posterior back toward the equilibrium prior — i.e.
    the optimizer trusts its directional views LESS when the market picture is uncertain.
    """
    if not market_state:
        return 1.0
    try:
        conf = float(market_state.get("confidence") if market_state.get("confidence") is not None else 1.0)
    except Exception:
        conf = 1.0
    mult = max(0.3, min(1.0, 0.5 + 0.5 * conf))
    vol_regime = str(market_state.get("volatility_regime") or market_state.get("vix_state") or "").lower()
    if vol_regime in ("high", "elevated"):
        mult *= 0.8
    return round(mult, 4)


def run_black_litterman(signals: list[dict], prices: pd.DataFrame | None = None,
                        market_caps: dict[str, float] | None = None,
                        vol_by_symbol: dict[str, float] | None = None,
                        cfg: BLConfig | None = None, date_tag: str | None = None,
                        write: bool = True, market_state: dict | None = None) -> dict:
    """
    Full Black-Litterman pass. Returns posterior_returns, posterior_covariance, prior,
    views, confidences, omega, tau, rejected_views, diagnostics, and a `fallback` flag.
    Never raises: on failure it falls back to the blended expected-return model.
    """
    cfg = cfg or BLConfig()
    symbols = [str(s.get("symbol", "")).upper() for s in signals if s.get("symbol")]
    symbols = list(dict.fromkeys(symbols))
    result = {"date_tag": date_tag or DATA_TAG, "generated_at": datetime.now(timezone.utc).isoformat(),
              "fallback": False, "engine": "none", "tau": cfg.tau}
    if len(symbols) < 2:
        fb = blended_expected_returns(signals, symbols, cfg)
        result.update({"fallback": True, "reason": "insufficient_candidates",
                       "posterior_returns": fb.to_dict(),
                       "prior_returns": {}, "views": [], "confidences": [], "diagnostics": {},
                       "_posterior_returns_series": fb,
                       "_posterior_covariance_df": build_covariance(symbols, None, vol_by_symbol, cfg)})
        if write:
            _write_report(result, date_tag)
        return result
    try:
        cov = build_covariance(symbols, prices, vol_by_symbol, cfg)
        pi = market_implied_prior(symbols, cov, market_caps, cfg)
        views = build_views(signals, symbols, cfg)
        # Incorporate the shared market_state: scale view confidence by market confidence
        # / volatility regime before forming the posterior (no-op when market_state is None).
        _ms_mult = _market_state_confidence_multiplier(market_state)
        if _ms_mult != 1.0 and len(views.get("confidences", [])):
            views["confidences"] = [float(c) * _ms_mult for c in views["confidences"]]
        post = black_litterman_posterior(pi, cov, views, cfg)
        post_ret, post_cov = post["posterior_returns"], post["posterior_covariance"]
        result.update({
            "engine": post["engine"],
            "prior_returns": {k: round(float(v), 6) for k, v in pi.items()},
            "posterior_returns": {k: round(float(v), 6) for k, v in post_ret.items()},
            "posterior_covariance": _df_round(post_cov),
            "views": views["view_labels"],
            "confidences": [round(float(c), 4) for c in views["confidences"]],
            "omega": _diag_list(post["omega"]),
            "absolute_views": views["absolute_views"],
            "rejected_views": views["rejected_views"],
            "view_strength": int(views["P"].shape[0]),
            "view_confidence": round(float(np.mean(views["confidences"])), 4) if len(views["confidences"]) else 0.0,
            "market_state_confidence_multiplier": _ms_mult,
            "market_state_confidence": (market_state or {}).get("confidence"),
            "diagnostics": {"n_assets": len(symbols),
                            "posterior_vs_prior_max_shift": round(float(np.max(np.abs(
                                post_ret.values - pi.reindex(post_ret.index).values))), 6),
                            "market_state_confidence_multiplier": _ms_mult,
                            "cov_source": "prices" if (prices is not None and not prices.empty) else "vol_estimate"},
        })
        result["_posterior_returns_series"] = post_ret
        result["_posterior_covariance_df"] = post_cov
    except Exception as exc:
        logger.error("Black-Litterman failed (%s) — blended-return fallback", exc)
        fb = blended_expected_returns(signals, symbols, cfg)
        result.update({"fallback": True, "reason": str(exc)[:200],
                       "posterior_returns": {k: round(float(v), 6) for k, v in fb.items()},
                       "prior_returns": {}, "views": [], "confidences": [], "diagnostics": {},
                       "_posterior_returns_series": fb,
                       "_posterior_covariance_df": build_covariance(symbols, None, vol_by_symbol, cfg)})
    if write:
        _write_report(result, date_tag)
    return result


def _df_round(df: pd.DataFrame, places: int = 6) -> dict:
    try:
        return {str(i): {str(c): round(float(df.loc[i, c]), places) for c in df.columns} for i in df.index}
    except Exception:
        return {}


def _diag_list(omega: np.ndarray) -> list:
    try:
        return [round(float(omega[i, i]), 8) for i in range(omega.shape[0])]
    except Exception:
        return []


def _write_report(result: dict, date_tag: str | None = None) -> None:
    try:
        serializable = {k: v for k, v in result.items() if not k.startswith("_")}
        (_research_dir(date_tag) / BL_REPORT).write_text(
            json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("write black_litterman_report failed: %s", exc)


def run_black_litterman_safe(*args, **kwargs):
    try:
        return run_black_litterman(*args, **kwargs)
    except Exception as exc:
        logger.error("run_black_litterman_safe swallowed: %s", exc)
        return {"fallback": True, "posterior_returns": {}, "reason": str(exc)[:200]}
