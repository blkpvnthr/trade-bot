#!/usr/bin/env python3
"""Profit-first daily opportunity routing.

This module is intentionally pure: it does not import Alpaca clients, submit
orders, or read credentials.  It only normalizes/ranks buy opportunities and
emits funnel diagnostics.  The canonical risk and execution substrate remains
``run.py``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _parse_hhmm(value: str) -> time:
    try:
        return time.fromisoformat(str(value))
    except Exception:
        return time(15, 30)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def is_bearish_context(market_context: Optional[dict[str, Any]]) -> bool:
    ctx = market_context or {}
    bias = str(ctx.get("overall_market_bias", ctx.get("market_bias", ""))).lower()
    perm = str(ctx.get("trade_permission", "")).lower()
    risk = str(ctx.get("risk_level", "")).lower()
    return (
        bias in {"bearish", "risk_off", "defensive", "defensive_risk_off"}
        or perm in {"disabled", "halt_new_buys", "risk_off"}
        or risk in {"high", "crisis"}
    )


@dataclass
class TradeOpportunity:
    symbol: str
    asset_class: str = "equity"
    side: str = "buy"
    strategy_id: str = ""
    setup_type: str = ""
    score: float = 0.0
    probability: float = 0.0
    expected_r: float = 0.0
    liquidity_score: float = 0.0
    regime_alignment: float = 0.0
    risk_score: float = 0.0
    freshness_score: float = 1.0
    qty: float = 0.0
    reference_price: float = 0.0
    limit_price: float = 0.0
    reason: str = ""
    overnight_candidate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol or "").upper()
        self.asset_class = str(self.asset_class or "equity").lower()
        self.side = str(self.side or "buy").lower()
        self.score = _finite(self.score)
        self.probability = _clamp(_finite(self.probability))
        self.expected_r = _finite(self.expected_r)
        self.liquidity_score = _clamp(_finite(self.liquidity_score))
        self.regime_alignment = _clamp(_finite(self.regime_alignment))
        self.risk_score = _clamp(_finite(self.risk_score))
        self.freshness_score = _clamp(_finite(self.freshness_score, 1.0))
        self.qty = _finite(self.qty)
        self.reference_price = _finite(self.reference_price)
        self.limit_price = _finite(self.limit_price)


@dataclass
class RejectedOpportunity:
    symbol: str
    blocked_by: str
    reason: str
    opportunity: TradeOpportunity


@dataclass
class RouterConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("DAILY_PROFIT_ENGINE_ENABLED", True))
    paper_only: bool = field(default_factory=lambda: _env_bool("DAILY_PROFIT_ENGINE_PAPER_ONLY", True))
    max_new_entries: int = field(default_factory=lambda: _env_int("DAILY_MAX_NEW_ENTRIES", 15))
    min_expected_r: float = field(default_factory=lambda: _env_float("DAILY_MIN_EXPECTED_R", 1.5))
    max_option_entries: int = field(default_factory=lambda: _env_int("DAILY_MAX_OPTION_ENTRIES", 3))
    max_intraday_tactical_open: int = field(default_factory=lambda: _env_int("DAILY_MAX_INTRADAY_TACTICAL_OPEN", 4))
    entry_cutoff_et: str = field(default_factory=lambda: os.getenv("DAILY_ENTRY_CUTOFF_ET", "15:30"))
    min_rank_score: float = field(default_factory=lambda: _env_float("DAILY_MIN_RANK_SCORE", 0.60))
    min_call_overnight_probability: float = field(
        default_factory=lambda: _env_float("OVERNIGHT_CALL_MIN_PROBABILITY", 0.80)
    )
    max_call_iv_crush: float = field(default_factory=lambda: _env_float("OVERNIGHT_CALL_MAX_IV_CRUSH", 40.0))
    min_call_dte: int = field(default_factory=lambda: _env_int("OVERNIGHT_CALL_MIN_DTE", 10))
    exceptional_call_score: float = field(default_factory=lambda: _env_float("OVERNIGHT_CALL_EXCEPTIONAL_SCORE", 90.0))


@dataclass
class RouteResult:
    candidates: list[TradeOpportunity] = field(default_factory=list)
    approved: list[TradeOpportunity] = field(default_factory=list)
    rejected: list[RejectedOpportunity] = field(default_factory=list)
    safe_mode: bool = False
    safe_mode_reason: str = ""
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def overnight_approved_shares(self) -> int:
        return sum(1 for o in self.approved if o.overnight_candidate and o.asset_class != "option")

    @property
    def overnight_approved_calls(self) -> int:
        return sum(1 for o in self.approved if o.overnight_candidate and o.asset_class == "option")


class DailyOpportunityRouter:
    """Ranks and filters new-buy opportunities.  It never places orders."""

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        self.config = config or RouterConfig()
        self.cutoff_time = _parse_hhmm(self.config.entry_cutoff_et)

    def rank_score(self, opp: TradeOpportunity) -> float:
        setup_evidence = _clamp(opp.score / 100.0)
        expected_r_score = _clamp(opp.expected_r / 4.0)
        return round(
            0.28 * opp.probability
            + 0.22 * expected_r_score
            + 0.18 * setup_evidence
            + 0.12 * opp.liquidity_score
            + 0.10 * opp.regime_alignment
            + 0.06 * (1.0 - opp.risk_score)
            + 0.04 * opp.freshness_score,
            6,
        )

    def exit_allowed(self, side: str, **_: Any) -> bool:
        return str(side or "").lower() != "buy"

    def _reject(self, rejected: list[RejectedOpportunity], opp: TradeOpportunity,
                blocked_by: str, reason: str) -> None:
        rejected.append(RejectedOpportunity(opp.symbol, blocked_by, reason, opp))

    def _entry_window_open(self, now_et: Optional[datetime]) -> bool:
        now = now_et or datetime.now(ET)
        if now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        return now.timetz().replace(tzinfo=None) < self.cutoff_time

    def _validated(self, opp: TradeOpportunity, market_context: Optional[dict[str, Any]]) -> tuple[bool, str]:
        if opp.side != "buy":
            return False, "not_a_buy"
        if not opp.symbol:
            return False, "missing_symbol"
        if opp.expected_r < self.config.min_expected_r:
            return False, f"expected_r={opp.expected_r:.2f}<{self.config.min_expected_r:.2f}"
        if opp.reference_price <= 0 and opp.limit_price <= 0:
            return False, "missing_reference_price"
        if is_bearish_context(market_context):
            return False, "bearish_or_risk_off"
        if self.rank_score(opp) < self.config.min_rank_score:
            return False, "rank_score_below_floor"
        return True, ""

    def overnight_allowed(self, opp: TradeOpportunity,
                          market_context: Optional[dict[str, Any]]) -> tuple[bool, str]:
        if not opp.overnight_candidate:
            return True, ""
        if is_bearish_context(market_context):
            return False, "overnight_gate"
        md = opp.metadata or {}
        if opp.asset_class == "option":
            dte = int(_finite(md.get("dte"), 0))
            iv_crush = _finite(md.get("iv_crush_risk"), 100.0)
            protected = bool(md.get("protected_or_partialed") or md.get("partial_taken"))
            active_exit_ladder = bool(md.get("active_exit_ladder", True))
            exceptional = opp.score >= self.config.exceptional_call_score
            if (
                opp.probability < self.config.min_call_overnight_probability
                or dte < self.config.min_call_dte
                or iv_crush > self.config.max_call_iv_crush
                or not active_exit_ladder
                or (not protected and not exceptional)
            ):
                return False, "overnight_gate"
            return True, ""
        profitable = bool(md.get("profitable")) or _finite(md.get("unrealized_pl"), 0.0) > 0
        thesis_backed = bool(md.get("thesis_backed"))
        trend_intact = bool(md.get("trend_intact"))
        if not ((profitable or thesis_backed) and trend_intact):
            return False, "overnight_gate"
        return True, ""

    def route(
        self,
        opportunities: Iterable[TradeOpportunity],
        market_context: Optional[dict[str, Any]] = None,
        *,
        now_et: Optional[datetime] = None,
        open_intraday_tactical_positions: int = 0,
        stale_state: bool = False,
        feed_timeout: bool = False,
        missing_data: bool = False,
    ) -> RouteResult:
        candidates = list(opportunities or [])
        result = RouteResult(candidates=candidates)
        if stale_state or feed_timeout or missing_data:
            result.safe_mode = True
            if stale_state:
                result.safe_mode_reason = "stale_state"
            elif feed_timeout:
                result.safe_mode_reason = "feed_timeout"
            else:
                result.safe_mode_reason = "missing_data"
            for opp in candidates:
                self._reject(result.rejected, opp, "setup_not_validated", "safe_mode_no_trade")
            return result
        if not self._entry_window_open(now_et):
            for opp in candidates:
                self._reject(result.rejected, opp, "setup_not_validated", "past_entry_cutoff")
            return result

        valid: list[TradeOpportunity] = []
        option_count = 0
        tactical_count = int(open_intraday_tactical_positions or 0)
        for opp in candidates:
            ok, why = self._validated(opp, market_context)
            if not ok:
                blocked = "opportunity_rank" if why == "rank_score_below_floor" else "setup_not_validated"
                self._reject(result.rejected, opp, blocked, why)
                continue
            overnight_ok, overnight_reason = self.overnight_allowed(opp, market_context)
            if not overnight_ok:
                self._reject(result.rejected, opp, overnight_reason, "overnight_not_allowed")
                continue
            if opp.asset_class == "option":
                if option_count >= self.config.max_option_entries:
                    self._reject(result.rejected, opp, "daily_trade_cap", "max_option_entries")
                    continue
                option_count += 1
            if opp.metadata.get("tactical", True):
                if tactical_count >= self.config.max_intraday_tactical_open:
                    self._reject(result.rejected, opp, "daily_trade_cap", "max_intraday_tactical_open")
                    continue
                tactical_count += 1
            valid.append(opp)

        ranked = sorted(valid, key=self.rank_score, reverse=True)
        result.approved = ranked[: max(0, self.config.max_new_entries)]
        for opp in ranked[len(result.approved):]:
            self._reject(result.rejected, opp, "daily_trade_cap", "max_new_entries")
        return result

    def funnel_payload(self, result: RouteResult, *, actual_trades: int = 0,
                       realized_next_day_outcome: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        zero_candidate_day = len(result.candidates) == 0
        no_trade_reason = ""
        if result.safe_mode:
            no_trade_reason = f"safe_mode:{result.safe_mode_reason or 'unknown'}"
        elif zero_candidate_day:
            no_trade_reason = "zero_candidates"
        elif not result.approved:
            no_trade_reason = "no_approved_entries"
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidates": len(result.candidates),
            "approved_entries": len(result.approved),
            "rejected_entries": len(result.rejected),
            "actual_trades": int(actual_trades),
            "overnight_approved_shares": result.overnight_approved_shares,
            "overnight_approved_calls": result.overnight_approved_calls,
            "safe_mode": bool(result.safe_mode),
            "safe_mode_reason": result.safe_mode_reason,
            "zero_candidate_day": zero_candidate_day,
            "no_trade_reason": no_trade_reason,
            "realized_next_day_outcome": realized_next_day_outcome or {"checked": False},
            "approved": [asdict(o) for o in result.approved],
            "rejected": [
                {
                    "symbol": r.symbol,
                    "blocked_by": r.blocked_by,
                    "reason": r.reason,
                    "strategy_id": r.opportunity.strategy_id,
                    "setup_type": r.opportunity.setup_type,
                }
                for r in result.rejected
            ],
        }

    def write_funnel(self, path: Path | str, result: RouteResult, *, actual_trades: int = 0,
                     realized_next_day_outcome: Optional[dict[str, Any]] = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                self.funnel_payload(
                    result,
                    actual_trades=actual_trades,
                    realized_next_day_outcome=realized_next_day_outcome,
                ),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return out
