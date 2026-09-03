#!/usr/bin/env python3
"""
adaptive_capital_allocator.py — Evidence-driven Capital Allocation Engine.

The Portfolio Research Lab measures what works. This module turns that evidence into
a capital ALLOCATION PLAN: it allocates more toward setups / valuation+conviction
profiles that have measurably worked, and less toward what has not.

It consumes:
    daily_system_outputs/<YYYYMMDD>/alpha_expectancy_report.json
    daily_system_outputs/<YYYYMMDD>/buffett_performance_report.json
    daily_system_outputs/<YYYYMMDD>/capital_waterfall_performance.json
    daily_system_outputs/<YYYYMMDD>/research_lab.json

and produces:
    daily_system_outputs/<YYYYMMDD>/capital_allocation_plan.json
    daily_system_outputs/<YYYYMMDD>/allocation_diagnostic.json

HARD INVARIANTS (the allocator is advisory and self-limiting):
    * It never submits orders and never imports/instantiates the broker.
    * It never overrides RiskManager — it only emits recommended sizes that are
      CLAMPED to the same configured caps RiskManager enforces.
    * Recommended sizes can never exceed the configured caps; low-sample evidence is
      dampened toward neutral so small samples can't swing capital.
"""
from __future__ import annotations

import json
import math
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

CAPITAL_ALLOCATION_PLAN = "capital_allocation_plan.json"
ALLOCATION_DIAGNOSTIC = "allocation_diagnostic.json"


def _research_dir(date_tag: str | None = None) -> Path:
    d = DAILY_OUTPUTS / (date_tag or DATA_TAG)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ── configuration (caps mirror run.py / RiskManager — these are CEILINGS) ─────────
@dataclass
class AllocatorConfig:
    enabled: bool = os.getenv("ENABLE_ADAPTIVE_ALLOCATION", "true").lower() in {"1", "true"}
    # Hard caps (never exceeded) — read from the same env as run.py/RiskManager.
    alpha_risk_cap_pct: float = float(os.getenv("OPTIONS_RISK_PER_TRADE_PCT", "0.015"))
    max_options_exposure_pct: float = float(os.getenv("MAX_OPTIONS_EXPOSURE_PCT", "0.15"))
    max_single_stock_weight: float = float(os.getenv("MAX_SINGLE_STOCK_WEIGHT", "0.10"))
    max_stock_exposure_pct: float = float(os.getenv("MAX_STOCK_EXPOSURE_PCT", "0.80"))
    cash_buffer_pct: float = float(os.getenv("CASH_BUFFER_PCT", "0.08"))
    alpha_max_positions: int = int(os.getenv("ALPHA_MAX_POSITIONS", "5"))
    # Evidence handling.
    min_trades: int = int(os.getenv("ALLOC_MIN_TRADES", "5"))   # below this, dampen toward neutral
    # Scaling multiplier bounds (1.0 = neutral). Baseline sits below the cap so a strong
    # score can scale UP toward (never beyond) the cap.
    alpha_min_mult: float = float(os.getenv("ALLOC_ALPHA_MIN_MULT", "0.50"))
    alpha_max_mult: float = float(os.getenv("ALLOC_ALPHA_MAX_MULT", "1.50"))
    bal_min_mult: float = float(os.getenv("ALLOC_BAL_MIN_MULT", "0.50"))
    bal_max_mult: float = float(os.getenv("ALLOC_BAL_MAX_MULT", "1.50"))
    base_fraction: float = float(os.getenv("ALLOC_BASE_FRACTION", "0.70"))   # neutral baseline as a fraction of cap
    floor_fraction: float = float(os.getenv("ALLOC_FLOOR_FRACTION", "0.30"))  # min fraction of cap
    # Capital-waterfall transfer bounds (fraction of realized alpha profit → BAL).
    transfer_base: float = float(os.getenv("ALLOC_TRANSFER_BASE", os.getenv("OPTIONS_PROFIT_TO_BAL_PCT", "0.60")))
    transfer_min: float = float(os.getenv("ALLOC_TRANSFER_MIN", "0.40"))
    transfer_max: float = float(os.getenv("ALLOC_TRANSFER_MAX", "0.80"))
    transfer_spread: float = float(os.getenv("ALLOC_TRANSFER_SPREAD", "0.20"))

    def __post_init__(self) -> None:
        # Keep base/floor coherent and bounds ordered.
        self.base_fraction = _clamp(self.base_fraction, 0.1, 1.0)
        self.floor_fraction = _clamp(self.floor_fraction, 0.0, self.base_fraction)
        self.transfer_min = _clamp(self.transfer_min, 0.0, 1.0)
        self.transfer_max = _clamp(self.transfer_max, self.transfer_min, 1.0)
        self.transfer_base = _clamp(self.transfer_base, self.transfer_min, self.transfer_max)


# ── input loading (consumes the research-lab outputs) ────────────────────────────
def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def load_research(date_tag: str | None = None) -> dict:
    """
    Load the research-lab analytics. Prefers research_lab.json (the combined report);
    falls back to the individual reports. Returns {alpha, buffett, capital_waterfall,
    rankings, source_files}.
    """
    d = _research_dir(date_tag)
    lab = _load(d / "research_lab.json")
    sources = []
    if lab.get("alpha") or lab.get("buffett"):
        sources.append("research_lab.json")
        alpha = lab.get("alpha", {})
        buffett = lab.get("buffett", {})
        waterfall = lab.get("capital_waterfall", {})
        rankings = lab.get("rankings", {})
    else:
        ar = _load(d / "alpha_expectancy_report.json")
        br = _load(d / "buffett_performance_report.json")
        cw = _load(d / "capital_waterfall_performance.json")
        alpha = ar.get("alpha", {})
        buffett = br.get("buffett", {})
        waterfall = cw.get("capital_waterfall", {})
        rankings = {}
        if ar:
            sources.append("alpha_expectancy_report.json")
        if br:
            sources.append("buffett_performance_report.json")
        if cw:
            sources.append("capital_waterfall_performance.json")
    return {"alpha": alpha, "buffett": buffett, "capital_waterfall": waterfall,
            "rankings": rankings, "source_files": sources}


# ── scoring (pure) ───────────────────────────────────────────────────────────────
def _pf(metrics: dict) -> float:
    pf = metrics.get("profit_factor", 0.0)
    try:
        pf = float(pf)
    except (TypeError, ValueError):
        pf = 0.0
    if math.isinf(pf):
        return 1e6
    return pf


def _evidence_score(metrics: dict, cfg: AllocatorConfig) -> tuple[float, dict]:
    """
    0-100 capital score from unit-stable performance metrics, dampened toward the
    neutral 50 when the sample is small. 50 = neutral (no tilt). Sub-scores returned
    for the diagnostic.
    """
    trades = int(metrics.get("trades", 0) or 0)
    if trades == 0:
        return 50.0, {"trades": 0, "note": "no_evidence_neutral"}
    pf = _pf(metrics)
    pf_s = _clamp((pf - 1.0) / 2.0, 0.0, 1.0)                       # pf 1→0, 3→1
    exp = float(metrics.get("expectancy", 0.0) or 0.0)
    exp_s = 1.0 if exp > 0 else (0.0 if exp < 0 else 0.5)          # sign (units vary by group)
    r_s = _clamp((float(metrics.get("avg_r_multiple", 0.0) or 0.0) + 1.0) / 3.0, 0.0, 1.0)  # -1→0, 2→1
    wr_s = _clamp(float(metrics.get("win_rate", 0.0) or 0.0), 0.0, 1.0)
    sharpe_s = _clamp(float(metrics.get("sharpe", 0.0) or 0.0) / 3.0, 0.0, 1.0)
    dd = float(metrics.get("max_drawdown", 0.0) or 0.0)            # negative
    dd_s = _clamp(1.0 + dd, 0.0, 1.0)                              # 0→1, -1→0
    raw = 100.0 * (0.30 * pf_s + 0.20 * exp_s + 0.20 * r_s +
                   0.10 * wr_s + 0.10 * sharpe_s + 0.10 * dd_s)
    confidence = min(trades / max(cfg.min_trades, 1), 1.0)
    score = 50.0 + (raw - 50.0) * confidence                       # dampen small samples toward neutral
    return round(_clamp(score, 0.0, 100.0), 1), {
        "trades": trades, "confidence": round(confidence, 3),
        "pf_s": round(pf_s, 3), "exp_s": exp_s, "r_s": round(r_s, 3),
        "wr_s": round(wr_s, 3), "sharpe_s": round(sharpe_s, 3), "dd_s": round(dd_s, 3),
        "raw": round(raw, 1),
    }


def alpha_capital_score(metrics: dict, cfg: AllocatorConfig | None = None) -> float:
    return _evidence_score(metrics, cfg or AllocatorConfig())[0]


def buffett_capital_score(metrics: dict, cfg: AllocatorConfig | None = None) -> float:
    return _evidence_score(metrics, cfg or AllocatorConfig())[0]


def _multiplier(score: float, lo: float, hi: float) -> float:
    """Map a 0-100 score to a scaling multiplier with 50 → 1.0, clamped to [lo, hi]."""
    if score >= 50.0:
        m = 1.0 + (hi - 1.0) * (score - 50.0) / 50.0
    else:
        m = lo + (1.0 - lo) * (score / 50.0)
    return round(_clamp(m, lo, hi), 4)


def _sized(cap: float, mult: float, cfg: AllocatorConfig) -> float:
    """
    Recommended size = baseline × multiplier, clamped to [floor, cap]. The configured
    `cap` is a hard ceiling that can never be exceeded.
    """
    baseline = cap * cfg.base_fraction
    floor = cap * cfg.floor_fraction
    return round(_clamp(baseline * mult, floor, cap), 6)


# ── allocation builders ──────────────────────────────────────────────────────────
def _alpha_group_allocations(groups: dict, cap: float, cfg: AllocatorConfig) -> dict:
    out = {}
    for key, m in (groups or {}).items():
        score, sub = _evidence_score(m, cfg)
        mult = _multiplier(score, cfg.alpha_min_mult, cfg.alpha_max_mult)
        out[key] = {
            "capital_score": score,
            "multiplier": mult,
            "recommended_risk_pct": _sized(cap, mult, cfg),
            "max_contracts_factor": mult,            # advisory factor on the contract cap
            "trades": m.get("trades", 0),
            "expectancy": m.get("expectancy", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0),
            "_subscores": sub,
        }
    return out


def _buffett_bucket_allocations(buckets: dict, cap: float, cfg: AllocatorConfig) -> dict:
    out = {}
    for key, m in (buckets or {}).items():
        score, sub = _evidence_score(m, cfg)
        mult = _multiplier(score, cfg.bal_min_mult, cfg.bal_max_mult)
        out[key] = {
            "capital_score": score,
            "multiplier": mult,
            "recommended_target_pct": _sized(cap, mult, cfg),
            "trades": m.get("trades", 0),
            "expectancy": m.get("expectancy", 0.0),
            "average_return": m.get("average_return", m.get("total_pnl", 0.0)),
            "_subscores": sub,
        }
    return out


def transfer_recommendation(alpha_score: float, buffett_score: float,
                            waterfall: dict, market_context: dict, cfg: AllocatorConfig) -> dict:
    """
    How much realized Alpha profit should transfer to BAL. Strong Alpha → retain more
    in Alpha (lower transfer); exceptional Buffett opportunity → transfer more to BAL.
    Regime tilts modestly. Always clamped to [transfer_min, transfer_max].
    """
    # Relative strength: buffett stronger than alpha → transfer more.
    delta = ((buffett_score - alpha_score) / 100.0) * cfg.transfer_spread
    bias = str((market_context or {}).get("overall_market_bias") or "").lower()
    regime_adj = 0.0
    reasoning = []
    if any(b in bias for b in ("bear", "risk_off", "downtrend", "crisis", "correction")):
        regime_adj += 0.05
        reasoning.append("bearish/value regime → favour BAL accumulation (+0.05)")
    elif any(b in bias for b in ("bull", "risk_on", "uptrend")):
        regime_adj -= 0.05
        reasoning.append("bullish/momentum regime → retain more in Alpha (-0.05)")
    recommended = _clamp(cfg.transfer_base + delta + regime_adj, cfg.transfer_min, cfg.transfer_max)

    if buffett_score > alpha_score:
        reasoning.append(f"Buffett evidence ({buffett_score:.0f}) > Alpha ({alpha_score:.0f}) "
                         f"→ tilt transfer up ({delta:+.3f})")
    elif alpha_score > buffett_score:
        reasoning.append(f"Alpha evidence ({alpha_score:.0f}) > Buffett ({buffett_score:.0f}) "
                         f"→ retain more in Alpha ({delta:+.3f})")

    alpha_realized = float((waterfall or {}).get("alpha_generated", 0.0) or 0.0)
    transfer_amount = round(recommended * alpha_realized, 2)
    return {
        "current_pct": round(cfg.transfer_base, 4),
        "recommended_pct": round(recommended, 4),
        "bounds": [cfg.transfer_min, cfg.transfer_max],
        "alpha_realized_profit": round(alpha_realized, 2),
        "transfer_amount": transfer_amount,
        "reasoning": reasoning or ["balanced evidence → hold default transfer"],
    }


# ── orchestration ────────────────────────────────────────────────────────────────
def build_allocation_plan(research: dict | None = None, market_context: dict | None = None,
                          market_state: dict | None = None,
                          config: AllocatorConfig | None = None, date_tag: str | None = None,
                          write: bool = True) -> dict:
    """
    Build the capital allocation plan from the research analytics. Pure with respect to
    its `research` input (pass it directly in tests); reads the report files when None.
    """
    cfg = config or AllocatorConfig()
    if research is None:
        research = load_research(date_tag)
    alpha = research.get("alpha", {})
    buffett = research.get("buffett", {})
    waterfall = research.get("capital_waterfall", {})

    alpha_score = alpha_capital_score(alpha.get("totals", {}), cfg)
    buffett_score = buffett_capital_score(buffett.get("totals", {}), cfg)

    setup_allocations = _alpha_group_allocations(alpha.get("by_setup_type", {}),
                                                 cfg.alpha_risk_cap_pct, cfg)
    sector_allocations = _alpha_group_allocations(alpha.get("by_sector", {}),
                                                  cfg.alpha_risk_cap_pct, cfg)
    vol_allocations = _alpha_group_allocations(alpha.get("by_volatility_regime", {}),
                                               cfg.alpha_risk_cap_pct, cfg)
    conviction_allocations = _buffett_bucket_allocations(buffett.get("by_conviction_bucket", {}),
                                                         cfg.max_single_stock_weight, cfg)
    valuation_allocations = _buffett_bucket_allocations(buffett.get("by_valuation_bucket", {}),
                                                        cfg.max_single_stock_weight, cfg)
    insider_allocations = _buffett_bucket_allocations(buffett.get("by_insider_bucket", {}),
                                                      cfg.max_single_stock_weight, cfg)
    tranche_allocations = _buffett_bucket_allocations(buffett.get("by_tranche", {}),
                                                      cfg.max_single_stock_weight, cfg)

    transfer = transfer_recommendation(alpha_score, buffett_score, waterfall, market_context, cfg)

    now = datetime.now(timezone.utc).isoformat()
    reasoning = [
        f"alpha_capital_score={alpha_score:.0f}, buffett_capital_score={buffett_score:.0f}",
        f"all recommendations clamped to hard caps: alpha_risk<= {cfg.alpha_risk_cap_pct:.4f}, "
        f"single_stock<= {cfg.max_single_stock_weight:.4f}",
        f"transfer to BAL recommended at {transfer['recommended_pct']:.0%} "
        f"(bounds {cfg.transfer_min:.0%}-{cfg.transfer_max:.0%})",
    ] + transfer["reasoning"]

    plan = {
        "generated_at": now, "date_tag": date_tag or DATA_TAG,
        "source_files": research.get("source_files", []),
        "alpha_capital_score": alpha_score,
        "buffett_capital_score": buffett_score,
        "setup_allocations": setup_allocations,
        "sector_allocations": sector_allocations,
        "volatility_allocations": vol_allocations,
        "conviction_allocations": conviction_allocations,
        "valuation_allocations": valuation_allocations,
        "insider_allocations": insider_allocations,
        "tranche_allocations": tranche_allocations,
        "transfer_recommendation": transfer,
        "transfer_amount": transfer["transfer_amount"],
        "caps": {
            "alpha_risk_per_trade_pct": cfg.alpha_risk_cap_pct,
            "max_options_exposure_pct": cfg.max_options_exposure_pct,
            "max_single_stock_weight": cfg.max_single_stock_weight,
            "max_stock_exposure_pct": cfg.max_stock_exposure_pct,
            "cash_buffer_pct": cfg.cash_buffer_pct,
            "alpha_max_positions": cfg.alpha_max_positions,
        },
        "reasoning": reasoning,
        "advisory_only": True,
        "note": "Advisory plan. Never submits orders, never overrides RiskManager; "
                "all sizes are clamped to the configured caps.",
    }

    diagnostic = {
        "generated_at": now, "date_tag": date_tag or DATA_TAG,
        "source_files": research.get("source_files", []),
        "min_trades_for_full_confidence": cfg.min_trades,
        "multiplier_bounds": {"alpha": [cfg.alpha_min_mult, cfg.alpha_max_mult],
                              "buffett": [cfg.bal_min_mult, cfg.bal_max_mult]},
        "baseline_fraction_of_cap": cfg.base_fraction,
        "alpha_totals": alpha.get("totals", {}),
        "buffett_totals": buffett.get("totals", {}),
        "setup_subscores": {k: v.get("_subscores") for k, v in setup_allocations.items()},
        "conviction_subscores": {k: v.get("_subscores") for k, v in conviction_allocations.items()},
        "valuation_subscores": {k: v.get("_subscores") for k, v in valuation_allocations.items()},
        "transfer": transfer,
    }

    if write:
        try:
            (_research_dir(date_tag) / CAPITAL_ALLOCATION_PLAN).write_text(
                json.dumps(plan, indent=2, default=str), encoding="utf-8")
            (_research_dir(date_tag) / ALLOCATION_DIAGNOSTIC).write_text(
                json.dumps(diagnostic, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.debug("allocation plan write failed: %s", exc)
    return plan


# ── consumption helpers (engines MAY read these; always within caps) ──────────────
def recommended_alpha_risk_pct(setup_type: str, plan: dict | None = None,
                               config: AllocatorConfig | None = None) -> float:
    """Recommended per-trade risk % for a setup, never exceeding the configured cap."""
    cfg = config or AllocatorConfig()
    plan = plan if plan is not None else _load(_research_dir() / CAPITAL_ALLOCATION_PLAN)
    rec = (plan.get("setup_allocations", {}) or {}).get(setup_type, {})
    val = float(rec.get("recommended_risk_pct", cfg.alpha_risk_cap_pct * cfg.base_fraction) or 0.0)
    return round(min(val, cfg.alpha_risk_cap_pct), 6)   # hard ceiling


def recommended_bal_target_pct(conviction_bucket: str, plan: dict | None = None,
                               config: AllocatorConfig | None = None) -> float:
    """Recommended BAL target weight for a conviction bucket, never exceeding the cap."""
    cfg = config or AllocatorConfig()
    plan = plan if plan is not None else _load(_research_dir() / CAPITAL_ALLOCATION_PLAN)
    rec = (plan.get("conviction_allocations", {}) or {}).get(conviction_bucket, {})
    val = float(rec.get("recommended_target_pct", cfg.max_single_stock_weight * cfg.base_fraction) or 0.0)
    return round(min(val, cfg.max_single_stock_weight), 6)   # hard ceiling


# ── fail-safe wrapper for run.py ──────────────────────────────────────────────────
def build_allocation_plan_safe(*args, **kwargs):
    try:
        return build_allocation_plan(*args, **kwargs)
    except Exception as exc:
        logger.debug("adaptive allocator skipped: %s", exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = build_allocation_plan()
    print(json.dumps({
        "alpha_capital_score": p.get("alpha_capital_score"),
        "buffett_capital_score": p.get("buffett_capital_score"),
        "transfer": p.get("transfer_recommendation", {}).get("recommended_pct"),
        "setups": {k: v["recommended_risk_pct"] for k, v in p.get("setup_allocations", {}).items()},
        "source_files": p.get("source_files"),
    }, indent=2, default=str))
