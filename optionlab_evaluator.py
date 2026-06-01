"""
optionlab_evaluator.py — Long-Call Payoff / Risk Profile Evaluator
==================================================================

Evaluates the payoff and risk profile of a candidate LONG CALL contract before
Alpha entry, using the `optionlab` package (https://github.com/rgaveiga/optionlab)
when available, and a self-contained Black-Scholes-based payoff calculation as a
fail-safe fallback otherwise.

Role in the Alpha option-scoring ensemble:
    Alpha setup → liquidity filter → Black-Scholes fair value → OOPS utility
    → OptionLab payoff/risk profile → combined score → RiskManager

OptionLab does NOT replace Alpha setup detection, Black-Scholes valuation, OOPS
utility optimization, probability management, or RiskManager. It MAY improve or
reduce a candidate's ranking. It may NEVER bypass liquidity, RiskManager, hard
stops, the kill switch, or the daily-loss halt, and it never places an order.

Per candidate it produces:
    break_even_price, break_even_distance_pct, profitable_price_range, max_loss,
    max_gain_estimate, payoff_asymmetry, payoff_quality, optionlab_greeks,
    optionlab_score, optionlab_reject_reason, source

If optionlab is unavailable, source="fallback_bs" and the closed-form long-call
payoff is used. Any failure degrades to a neutral result — it never raises.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Shared atomic JSON writer (single implementation in opportunity_utils); guarded so
# this module stays import-safe standalone. Falls back to an atomic tmp+rename write.
try:
    from opportunity_utils import write_json_safe as _write_json_safe
except Exception:
    def _write_json_safe(path, data, log_fn=None) -> bool:
        import json as _json, os as _os
        from pathlib import Path as _P
        p = _P(path); tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(_json.dumps(data, indent=2, default=str), encoding="utf-8")
        _os.replace(tmp, p)
        return True


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

SOURCE_OPTIONLAB = "optionlab"
SOURCE_FALLBACK = "fallback_bs"

# Optional optionlab package — probed once (never fatal).
try:  # pragma: no cover - exercised via OPTIONLAB_AVAILABLE in tests
    from optionlab import run_strategy as _ol_run_strategy, Inputs as _OLInputs  # type: ignore
    OPTIONLAB_AVAILABLE = True
except Exception:
    _ol_run_strategy = None
    _OLInputs = None
    OPTIONLAB_AVAILABLE = False
    logger.warning(
        "optionlab package unavailable — OptionLab payoff evaluation uses the "
        "Black-Scholes fallback (install `optionlab` to enable the full engine)."
    )


@dataclass
class OptionLabConfig:
    enabled: bool = os.getenv("ENABLE_OPTIONLAB", "true").lower() in {"1", "true"}
    risk_free_rate: float = float(os.getenv("OPTIONLAB_RFR", "0.04"))
    default_iv: float = float(os.getenv("OPTIONLAB_DEFAULT_IV", "0.30"))
    # Expected-move multiples used to judge break-even reachability.
    near_be_move_mult: float = 0.5     # BE within 0.5σ → excellent
    ok_be_move_mult: float = 1.0       # BE within 1σ → ok
    far_be_move_mult: float = 1.5      # BE beyond 1.5σ → penalised
    unreachable_be_move_mult: float = 2.0  # BE beyond 2σ → reject-flagged
    # Terminal price used for the (bounded) max-gain estimate: S + k·expected_move.
    upside_move_mult: float = 2.0
    max_theta_ratio: float = 0.05      # |theta|/premium/day above this is theta-heavy
    max_records: int = int(os.getenv("OPTIONLAB_MAX_RECORDS", "500"))  # bounded report history


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


def _quality_label(score: float) -> str:
    if score >= 75:
        return "excellent"
    if score >= 60:
        return "good"
    if score >= 45:
        return "fair"
    return "poor"


class OptionLabEvaluator:
    """Evaluate the payoff/risk profile of a long call. Advisory only; never raises."""

    def __init__(self, config: Optional[OptionLabConfig] = None):
        self.config = config or OptionLabConfig()
        self.available = OPTIONLAB_AVAILABLE
        # Bounded ring buffer — keeps only the latest N evaluations so the singleton
        # cannot grow without limit over a long-running session (memory + JSON size).
        self._records: deque = deque(maxlen=max(1, self.config.max_records))
        self._total_evaluated: int = 0   # lifetime count (aggregate metric, not retained rows)

    # ── public API ───────────────────────────────────────────────────────────
    def evaluate_long_call(
        self,
        *,
        underlying_price: float,
        strike: float,
        premium: float,
        dte: float,
        iv: Optional[float] = None,
        contracts: int = 1,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        theta: Optional[float] = None,
        vega: Optional[float] = None,
        expected_move: Optional[float] = None,
        symbol: str = "",
        market_state: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Return a payoff/risk profile dict for a single long-call contract."""
        try:
            return self._evaluate(
                underlying_price=_f(underlying_price), strike=_f(strike),
                premium=_f(premium), dte=max(1.0, _f(dte, 1.0)),
                iv=_f(iv, self.config.default_iv) or self.config.default_iv,
                contracts=max(1, int(contracts or 1)),
                delta=delta, gamma=gamma, theta=theta, vega=vega,
                expected_move=expected_move, symbol=symbol, market_state=market_state,
            )
        except Exception as exc:  # never raise into Alpha
            logger.debug("OptionLab evaluate_long_call failed (%s) — neutral result", exc)
            return self._neutral(reason=f"optionlab_error:{str(exc)[:80]}")

    def _neutral(self, reason: str = "") -> Dict[str, Any]:
        return {
            "break_even_price": None, "break_even_distance_pct": None,
            "profitable_price_range": None, "max_loss": None,
            "max_gain_estimate": None, "payoff_asymmetry": None,
            "payoff_quality": "unknown", "optionlab_greeks": {},
            "optionlab_score": 50.0, "optionlab_reject_reason": reason,
            "source": SOURCE_FALLBACK,
        }

    def _evaluate(self, *, underlying_price, strike, premium, dte, iv, contracts,
                  delta, gamma, theta, vega, expected_move, symbol, market_state) -> Dict[str, Any]:
        cfg = self.config
        S = underlying_price
        if S <= 0 or strike <= 0 or premium <= 0:
            return self._neutral(reason="invalid_inputs")

        T = dte / 365.0
        if expected_move is None or expected_move <= 0:
            expected_move = S * iv * math.sqrt(max(T, 1e-9))
        move_pct = expected_move / S if S > 0 else 0.0

        # Closed-form long-call payoff geometry (independent of optionlab).
        break_even = strike + premium                      # per-share break-even
        be_distance_pct = (break_even - S) / S if S > 0 else 0.0
        max_loss = premium * 100.0 * contracts             # premium fully at risk
        terminal_hi = S + cfg.upside_move_mult * expected_move
        intrinsic_hi = max(0.0, terminal_hi - strike)
        max_gain_estimate = (intrinsic_hi - premium) * 100.0 * contracts
        profitable_low = break_even
        profitable_high = S + 3.0 * expected_move          # realistic upper profitable bound
        payoff_asymmetry = (max_gain_estimate / max_loss) if max_loss > 0 else 0.0

        greeks = {"delta": _f(delta), "gamma": _f(gamma), "theta": _f(theta), "vega": _f(vega)}

        # Optionally refine with the real optionlab engine (best-effort).
        source = SOURCE_FALLBACK
        if cfg.enabled and OPTIONLAB_AVAILABLE:
            ol = self._run_optionlab(S, strike, premium, T, iv, cfg)
            if ol is not None:
                source = SOURCE_OPTIONLAB
                if ol.get("break_even") is not None:
                    break_even = ol["break_even"]
                    be_distance_pct = (break_even - S) / S if S > 0 else be_distance_pct
                if ol.get("profit_ranges"):
                    pr = ol["profit_ranges"][0]
                    profitable_low, profitable_high = pr[0], pr[1]
                if ol.get("max_loss") is not None:
                    max_loss = abs(ol["max_loss"])
                if ol.get("max_gain") is not None and ol["max_gain"] > 0:
                    max_gain_estimate = ol["max_gain"]
                    payoff_asymmetry = (max_gain_estimate / max_loss) if max_loss > 0 else 0.0
                if ol.get("greeks"):
                    greeks.update({k: _f(v) for k, v in ol["greeks"].items()})

        # ── score the payoff profile (0-100) ──────────────────────────────────
        score = 50.0
        reject = ""
        ratio = (be_distance_pct / move_pct) if move_pct > 1e-9 else 999.0
        if be_distance_pct <= 0:
            score += 25.0                                   # already ITM/at-the-money breakeven
        elif ratio <= cfg.near_be_move_mult:
            score += 25.0
        elif ratio <= cfg.ok_be_move_mult:
            score += 15.0
        elif ratio <= cfg.far_be_move_mult:
            score += 5.0
        elif ratio <= cfg.unreachable_be_move_mult:
            score -= 10.0
        else:
            score -= 20.0
            reject = "break_even_unreachable"

        if payoff_asymmetry >= 3.0:
            score += 20.0
        elif payoff_asymmetry >= 2.0:
            score += 12.0
        elif payoff_asymmetry >= 1.0:
            score += 5.0
        elif payoff_asymmetry < 0.5:
            score -= 10.0
            reject = reject or "poor_payoff_asymmetry"

        th = greeks.get("theta")
        if th:
            theta_ratio = abs(th) / premium if premium > 0 else 0.0
            if theta_ratio > cfg.max_theta_ratio:
                score -= 10.0
                reject = reject or "theta_heavy"
            elif theta_ratio > cfg.max_theta_ratio * 0.6:
                score -= 5.0

        d = greeks.get("delta")
        if d and 0.45 <= d <= 0.70:
            score += 5.0

        score = max(0.0, min(100.0, score))
        result = {
            "symbol": symbol,
            "break_even_price": round(break_even, 4),
            "break_even_distance_pct": round(be_distance_pct, 6),
            "profitable_price_range": [round(profitable_low, 4), round(profitable_high, 4)],
            "max_loss": round(max_loss, 2),
            "max_gain_estimate": round(max_gain_estimate, 2),
            "payoff_asymmetry": round(payoff_asymmetry, 4),
            "payoff_quality": _quality_label(score),
            "optionlab_greeks": {k: round(_f(v), 6) for k, v in greeks.items()},
            "optionlab_score": round(score, 2),
            "optionlab_reject_reason": reject,
            "expected_move": round(expected_move, 4),
            "source": source,
        }
        self._records.append(result)
        self._total_evaluated += 1
        return result

    def _run_optionlab(self, S, strike, premium, T, iv, cfg) -> Optional[Dict[str, Any]]:
        """Best-effort call into the real optionlab engine (guarded; returns None on any issue)."""
        try:
            days = max(1, int(round(T * 365.0)))
            inputs = _OLInputs(
                stock_price=S, volatility=iv, interest_rate=cfg.risk_free_rate,
                min_stock=max(0.01, S * 0.5), max_stock=S * 1.5,
                days_to_target_date=days,
                strategy=[{"type": "call", "strike": strike, "premium": premium,
                           "n": 100, "action": "buy"}],
            )
            out = _ol_run_strategy(inputs)
            ranges = list(getattr(out, "profit_ranges", []) or [])
            be = ranges[0][0] if ranges else None
            return {
                "break_even": be,
                "profit_ranges": ranges,
                "max_loss": getattr(out, "minimum_return_in_the_domain", None),
                "max_gain": getattr(out, "maximum_return_in_the_domain", None),
                "greeks": {},  # per-leg greeks are lists; left to the BS-supplied greeks
            }
        except Exception as exc:
            logger.debug("optionlab run_strategy failed (%s) — using fallback geometry", exc)
            return None

    # ── reporting ──────────────────────────────────────────────────────────────
    def write_report(self, out_dir: Any = None) -> Optional[Path]:
        import json
        target = Path(out_dir) if out_dir else (DAILY_OUTPUTS / DATA_TAG)
        try:
            target.mkdir(parents=True, exist_ok=True)
            p = target / "optionlab_evaluation_report.json"
            payload = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "optionlab_available": OPTIONLAB_AVAILABLE,
                "total_evaluated": self._total_evaluated,       # lifetime aggregate
                "retained_count": len(self._records),           # bounded by max_records
                "evaluated_count": len(self._records),          # back-compat
                "evaluations": list(self._records),
            }
            _write_json_safe(p, payload)
            return p
        except Exception as exc:
            logger.debug("OptionLab write_report failed: %s", exc)
            return None


# Module-level singleton (alpha_call_engine reuses one evaluator per process).
_EVALUATOR: Optional[OptionLabEvaluator] = None


def get_evaluator(config: Optional[OptionLabConfig] = None) -> OptionLabEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = OptionLabEvaluator(config)
    return _EVALUATOR
