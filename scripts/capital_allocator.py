#!/usr/bin/env python3
"""
scripts/capital_allocator.py

Sharpe-weighted capital allocation across paper-trading strategy variants.
Hard caps: max 40% per variant, min 5% per active variant.
Paper-only mode: outputs allocation recommendation only, never places orders.

Writes daily_system_outputs/capital_allocation.json.

Usage:
  python scripts/capital_allocator.py [--total-capital DOLLARS]
  python scripts/capital_allocator.py --total-capital 13000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY

OUTPUT_DIR  = PROJECT_ROOT / "daily_system_outputs"

# Hard allocation caps (paper mode — will be re-evaluated before live promotion)
MAX_ALLOCATION_PCT  = 0.40   # 40% max per variant
MIN_ALLOCATION_PCT  = 0.05   # 5% min per active variant (applied after Sharpe weighting)
MIN_SHARPE_FOR_ALLOC = -2.0  # strategies below this get 0 allocation

# Default total paper capital (matches KILL_SWITCH_EQUITY_FLOOR buffer zone)
DEFAULT_TOTAL_CAPITAL = 13_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Load research rankings
# ─────────────────────────────────────────────────────────────────────────────

def _load_rankings(rankings_path: Path) -> list[dict[str, Any]]:
    """Load research_rankings.json. Returns list of variant metric dicts."""
    if not rankings_path.exists():
        return []
    try:
        return json.loads(rankings_path.read_text())
    except Exception:
        return []


def _extract_sharpe(ranking: dict[str, Any]) -> float:
    """Extract Sharpe ratio from a ranking dict, returning 0.0 if unavailable."""
    # Handles nested {"metrics": {"sharpe_ratio": ...}} or flat {"sharpe_ratio": ...}
    metrics = ranking.get("metrics", ranking)
    raw = metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0))
    try:
        val = float(raw)
        return val if math.isfinite(val) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _extract_trade_count(ranking: dict[str, Any]) -> int:
    metrics = ranking.get("metrics", ranking)
    raw = metrics.get("total_trades", metrics.get("trade_count", 0))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Allocation engine
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe_weighted_allocation(
    strategy_sharpes: dict[str, float],
    min_trades: int = 5,
    trade_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Compute Sharpe-weighted fractional allocations with hard caps.
    Strategies with < min_trades get 0 allocation (insufficient data).
    Returns {strategy_id: fraction (0.0–1.0)}.
    """
    tc = trade_counts or {}

    # Filter eligible strategies
    eligible = {
        sid: max(0.0, sharpe)   # treat negative Sharpe as 0 weight
        for sid, sharpe in strategy_sharpes.items()
        if sharpe >= MIN_SHARPE_FOR_ALLOC
        and tc.get(sid, min_trades) >= min_trades   # unknown trade count → assume sufficient
    }

    if not eligible:
        # Equal allocation across all active strategies as fallback
        n = len(strategy_sharpes)
        return {sid: round(1.0 / n, 4) for sid in strategy_sharpes} if n else {}

    total_weight = sum(eligible.values())

    if total_weight == 0:
        # All eligible Sharpes are 0 → equal weight
        n = len(eligible)
        raw = {sid: 1.0 / n for sid in eligible}
    else:
        raw = {sid: w / total_weight for sid, w in eligible.items()}

    # Add zero-weight for ineligible strategies
    for sid in strategy_sharpes:
        if sid not in raw:
            raw[sid] = 0.0

    # Apply MAX cap (clip and redistribute)
    for _ in range(20):   # iterative cap enforcement
        capped: dict[str, float] = {}
        excess = 0.0
        uncapped_ids: list[str] = []
        for sid, frac in raw.items():
            if frac > MAX_ALLOCATION_PCT:
                capped[sid] = MAX_ALLOCATION_PCT
                excess += frac - MAX_ALLOCATION_PCT
            else:
                uncapped_ids.append(sid)

        if not excess or not uncapped_ids:
            raw.update(capped)
            break

        # Redistribute excess proportionally to uncapped
        uncapped_sum = sum(raw[sid] for sid in uncapped_ids)
        if uncapped_sum == 0:
            break
        for sid in uncapped_ids:
            raw[sid] += excess * (raw[sid] / uncapped_sum)
        raw.update(capped)

    # Apply MIN floor to active (non-zero) strategies by trimming max
    active = [sid for sid, frac in raw.items() if frac > 0]
    if active:
        min_required = MIN_ALLOCATION_PCT * len(active)
        if min_required <= 1.0:
            for sid in active:
                if raw[sid] < MIN_ALLOCATION_PCT:
                    raw[sid] = MIN_ALLOCATION_PCT
            # Re-normalize to sum to 1.0
            total = sum(raw.values())
            if total > 0:
                raw = {sid: round(frac / total, 4) for sid, frac in raw.items()}

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Main allocation function
# ─────────────────────────────────────────────────────────────────────────────

def compute_allocation(
    total_capital: float,
    verbose: bool = True,
) -> dict[str, Any]:
    rankings_path = OUTPUT_DIR / "research_rankings.json"
    rankings = _load_rankings(rankings_path)

    strategy_sharpes: dict[str, float] = {}
    trade_counts: dict[str, int] = {}
    data_source = "live_rankings"

    if rankings:
        for r in rankings:
            sid = r.get("strategy_id", "")
            if sid in STRATEGY_REGISTRY:
                strategy_sharpes[sid] = _extract_sharpe(r)
                trade_counts[sid] = _extract_trade_count(r)
    else:
        # No live data — use neutral Sharpe (equal allocation) as bootstrap
        data_source = "bootstrap_equal"
        for sid in STRATEGY_REGISTRY:
            strategy_sharpes[sid] = 0.0

    if verbose:
        print(f"[allocator] data_source={data_source}  capital=${total_capital:,.0f}")
        print(f"[allocator] strategies={len(strategy_sharpes)}")

    fractions = _sharpe_weighted_allocation(strategy_sharpes, trade_counts=trade_counts)

    allocations: list[dict[str, Any]] = []
    for sid in STRATEGY_REGISTRY:
        frac = fractions.get(sid, 0.0)
        dollars = round(total_capital * frac, 2)
        sharpe = strategy_sharpes.get(sid)
        trades = trade_counts.get(sid, 0)
        allocations.append({
            "strategy_id": sid,
            "allocation_pct": round(frac * 100, 2),
            "allocation_dollars": dollars,
            "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
            "trade_count": trades,
            "status": "active" if frac > 0 else "inactive",
        })
        if verbose:
            status_str = f"{frac*100:.1f}%  ${dollars:,.0f}"
            sharpe_str = f"  sharpe={sharpe:.3f}" if sharpe is not None else ""
            print(f"  {sid}: {status_str}{sharpe_str}  trades={trades}")

    # Sanity check: total allocation ≈ 100%
    total_pct = sum(a["allocation_pct"] for a in allocations)
    total_dollars = sum(a["allocation_dollars"] for a in allocations)

    if verbose:
        print(f"\n  Total: {total_pct:.1f}%  ${total_dollars:,.0f}")
        if abs(total_pct - 100.0) > 0.5:
            print(f"  WARNING: allocation does not sum to 100% ({total_pct:.2f}%)")

    summary = {
        "total_capital": total_capital,
        "data_source": data_source,
        "paper_only": True,
        "allocations": allocations,
        "total_allocated_pct": round(total_pct, 2),
        "total_allocated_dollars": round(total_dollars, 2),
        "caps": {
            "max_per_variant_pct": MAX_ALLOCATION_PCT * 100,
            "min_per_active_variant_pct": MIN_ALLOCATION_PCT * 100,
            "min_sharpe_for_allocation": MIN_SHARPE_FOR_ALLOC,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Sharpe-weighted capital allocation")
    parser.add_argument("--total-capital", type=float, default=DEFAULT_TOTAL_CAPITAL,
                        help=f"Total paper capital in dollars (default: {DEFAULT_TOTAL_CAPITAL:,.0f})")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print(f"=== Capital Allocator (paper-only) ===\n")

    summary = compute_allocation(args.total_capital, verbose=not args.quiet)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "capital_allocation.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        print(f"\nOutput: {out_path}")
        print("NOTE: This is a recommendation only. Paper-only mode. No orders placed.")


if __name__ == "__main__":
    main()
