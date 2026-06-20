"""Implausible-results verification gate.

Flags strategies whose stats are *too good to be plausible* (or whose Pine
source has a known repaint/look-ahead vector) for MANUAL REVIEW. It never
rejects or changes RL eligibility -- the goal is to surface statistically
unusual results so a human checks the Pine source before trusting them.

A strategy that lands in the same tier as a known-good one purely on
PF 15 / 82% win / 0.4% drawdown should trigger skepticism, not celebration.
"""
from __future__ import annotations

import math

VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"
SUSPECTED_REPAINT = "suspected_repaint"
SUSPECTED_LOOKAHEAD = "suspected_lookahead"

# Implausibility thresholds (flag, never reject).
IMPLAUSIBLE_PF = 5.0
IMPLAUSIBLE_MAXDD = 0.02        # 2%
IMPLAUSIBLE_WIN = 0.80
IMPLAUSIBLE_WIN_TRADES = 1000
IMPLAUSIBLE_CAGR = 2.0          # 200%/yr


def metric_flags(m) -> list:
    """Statistical-anomaly flags from the aggregate metrics."""
    flags = []
    pf = m.profit_factor
    pf_hi = (not math.isfinite(pf)) or pf > IMPLAUSIBLE_PF
    if pf_hi and m.max_drawdown < IMPLAUSIBLE_MAXDD:
        pf_s = "inf" if not math.isfinite(pf) else f"{pf:.1f}"
        flags.append(f"PF {pf_s} > {IMPLAUSIBLE_PF} with maxDD "
                     f"{m.max_drawdown*100:.1f}% < {IMPLAUSIBLE_MAXDD*100:.0f}%")
    if m.win_rate > IMPLAUSIBLE_WIN and m.trade_count > IMPLAUSIBLE_WIN_TRADES:
        flags.append(f"win_rate {m.win_rate*100:.1f}% > {IMPLAUSIBLE_WIN*100:.0f}% "
                     f"with {m.trade_count} > {IMPLAUSIBLE_WIN_TRADES} trades")
    if m.cagr > IMPLAUSIBLE_CAGR and m.max_drawdown < IMPLAUSIBLE_MAXDD:
        flags.append(f"CAGR {m.cagr*100:.0f}% extreme with maxDD "
                     f"{m.max_drawdown*100:.1f}% tiny")
    return flags


def pine_flags(meta) -> list:
    """Source-level repaint / look-ahead vectors (e.g. old-Pine security())."""
    if meta is None:
        return []
    v = getattr(meta, "pine_version", None)
    uses_sec = getattr(meta, "uses_security", False)
    safe = getattr(meta, "lookahead_off", False)
    if uses_sec and not safe and v is not None and v <= 3:
        return [f"Pine v{v} security() without lookahead_off (repaint/look-ahead)"]
    return []


def verification_status(metrics, meta=None) -> tuple:
    """(status, reasons). Source repaint flags dominate metric anomalies."""
    pf = pine_flags(meta)
    mf = metric_flags(metrics)
    reasons = pf + mf
    if pf:
        return SUSPECTED_LOOKAHEAD, reasons
    if mf:
        return NEEDS_REVIEW, reasons
    return VERIFIED, reasons


def _demo():
    from .types import Metrics, StrategyMeta
    good = Metrics(trade_count=600, win_rate=0.62, profit_factor=1.8, max_drawdown=0.15)
    assert verification_status(good)[0] == VERIFIED
    too_good = Metrics(trade_count=8000, win_rate=0.82, profit_factor=19.0, max_drawdown=0.007)
    assert verification_status(too_good)[0] == NEEDS_REVIEW
    sniper = StrategyMeta("s", "Sniper", "h", "2026-01-01", "s.txt",
                          pine_version=2, uses_security=True, lookahead_off=False)
    assert verification_status(too_good, sniper)[0] == SUSPECTED_LOOKAHEAD
    print("verification._demo OK")


if __name__ == "__main__":
    _demo()
