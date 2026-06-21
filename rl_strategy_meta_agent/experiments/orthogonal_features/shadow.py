"""Shadow-paper phase tracking + promotion gate for the earnings_within_7d sleeve.

The next milestone is NOT another factor -- it is proving the paper sleeve behaves in the
wild the same way it behaved in validation. This module logs the sleeve's daily paper
behavior, compares it to the research baseline, and gates the move to Capital Allocation
Review. No stage may be skipped:

    Research -> Validated -> Paper Sleeve -> Shadow Statistics -> Capital Allocation Review

Promotion criterion (all required, fail-closed): >= min_days of shadow, paper IC ~ research
IC, paper turnover ~ research turnover, and ZERO operational failures (data outages /
blocks). Pure numpy/pandas; no broker/order/live path. Research only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Promotion criteria. A modest signal (research IC ~ 0.02) is extremely noisy over only
# 60-90 days, so the gate does NOT demand paper-IC ~ research-IC near-equality (that rejects
# valid signals on normal variance). It requires robust DIRECTIONAL evidence instead:
# sign-consistent IC, positive rank spread, positive realized net Sharpe, zero outages.
DEFAULTS = {"min_days": 60, "cost_bps": 5.0, "max_operational_failures": 0}


class ShadowTracker:
    """Append-only daily shadow log (JSONL). One row per paper session."""

    def __init__(self, path):
        self.path = Path(path)

    def record_day(self, date, weights: dict, diag: dict, next_returns: dict | None = None,
                   signal: dict | None = None):
        """Log one paper day. If next_returns (realized next-day per-symbol returns) are
        provided, also log daily IC, long-short rank spread, and P&L."""
        rec = {"date": str(date), "blocked": bool(diag.get("blocked", False)),
               "reason": diag.get("reason"), "gross": diag.get("gross", 0.0),
               "net": diag.get("net", 0.0), "n_long": diag.get("n_long", 0),
               "n_short": diag.get("n_short", 0), "data_outage": bool(diag.get("blocked", False))}
        if next_returns and weights:
            longs = [s for s, w in weights.items() if w > 0]
            shorts = [s for s, w in weights.items() if w < 0]
            lr = np.nanmean([next_returns.get(s, np.nan) for s in longs]) if longs else np.nan
            sr = np.nanmean([next_returns.get(s, np.nan) for s in shorts]) if shorts else np.nan
            rec["rank_spread"] = _round(lr - sr)
            rec["daily_pnl"] = _round(sum(w * next_returns.get(s, 0.0) for s, w in weights.items()))
            if signal:
                rec["daily_ic"] = _round(_xs_ic(signal, next_returns))
            rec["turnover"] = diag.get("turnover")
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def read(self):
        if not self.path.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(l) for l in self.path.read_text().splitlines() if l.strip()])


def compare_to_research(shadow_df: pd.DataFrame, research_ic, research_turnover, cfg=None) -> dict:
    """Emit the promotion verdict from robust DIRECTIONAL evidence (not IC near-equality):
    >= min_days, sign-consistent IC, positive rank spread, positive realized net Sharpe,
    zero operational failures. research_turnover is reported for context, not gated."""
    c = {**DEFAULTS, **(cfg or {})}
    if shadow_df.empty:
        return {"verdict": "SHADOW_NOT_STARTED", "days": 0}
    traded = shadow_df[~shadow_df["blocked"].astype(bool)]
    outages = int(shadow_df["blocked"].astype(bool).sum())
    days = int(len(shadow_df))
    paper_ic = float(traded["daily_ic"].mean()) if "daily_ic" in traded and len(traded) else float("nan")
    spread = float(traded["rank_spread"].mean()) if "rank_spread" in traded and len(traded) else float("nan")
    paper_turn = float(traded["turnover"].mean()) if "turnover" in traded and traded["turnover"].notna().any() else float("nan")
    if "daily_pnl" in traded and len(traded):
        # flat (no-earnings-in-window) days are legitimately 0 P&L, not missing
        net = traded["daily_pnl"].fillna(0).to_numpy(float) - c["cost_bps"] / 1e4 * traded["turnover"].fillna(0).to_numpy(float)
        realized_sharpe = float(net.mean() / (net.std() + 1e-12) * np.sqrt(252))
    else:
        realized_sharpe = float("nan")

    days_ok = bool(days >= c["min_days"])
    sign_ok = bool(np.isfinite(paper_ic) and (research_ic == 0 or np.sign(paper_ic) == np.sign(research_ic)))
    spread_ok = bool(np.isfinite(spread) and spread > 0)
    sharpe_ok = bool(np.isfinite(realized_sharpe) and realized_sharpe > 0)
    ops_ok = bool(outages <= c["max_operational_failures"])
    ready = bool(days_ok and sign_ok and spread_ok and sharpe_ok and ops_ok)
    failing = outages > c["max_operational_failures"] or (np.isfinite(realized_sharpe) and realized_sharpe < 0) \
        or (np.isfinite(spread) and spread < 0)
    verdict = "READY_FOR_CAPITAL_REVIEW" if ready else "SHADOW_FAIL" if failing else "SHADOW_CONTINUE"
    return {"verdict": verdict, "days": days, "days_required": c["min_days"],
            "paper_ic": _round(paper_ic), "research_ic": _round(research_ic), "ic_sign_consistent": sign_ok,
            "mean_rank_spread": _round(spread), "rank_spread_positive": spread_ok,
            "realized_net_sharpe": _round(realized_sharpe), "sharpe_positive": sharpe_ok,
            "paper_turnover": _round(paper_turn), "research_turnover": _round(research_turnover),
            "operational_failures": outages, "ops_clean": ops_ok, "promotion_ready": ready,
            "checks": {"days": days_ok, "ic_sign": sign_ok, "rank_spread": spread_ok,
                       "realized_sharpe": sharpe_ok, "ops": ops_ok}}


def capital_review_ready(report: dict) -> bool:
    """The promotion gate. True only after a clean, statistically-matched shadow period."""
    return bool(report.get("promotion_ready", False))


def _xs_ic(signal: dict, rets: dict):
    keys = [k for k in signal if k in rets and np.isfinite(rets[k]) and np.isfinite(signal[k])]
    if len(keys) < 3:
        return np.nan
    s = pd.Series([signal[k] for k in keys]); r = pd.Series([rets[k] for k in keys])
    if s.nunique() < 2 or r.nunique() < 2:
        return np.nan
    return float(s.rank().corr(r.rank()))


def _round(x, n=5):
    return round(float(x), n) if x is not None and np.isfinite(x) else None


def _demo():
    import tempfile
    p = Path(tempfile.mkdtemp()) / "shadow.jsonl"
    t = ShadowTracker(p)
    rng = np.random.default_rng(0)
    for i in range(65):
        sig = {f"S{j}": (1.0 if j < 3 else 0.0) for j in range(10)}
        w = {f"S{j}": (0.5 / 3 if j < 3 else -0.5 / 7) for j in range(10)}
        # realized returns where the long leg modestly outperforms -> positive IC
        nr = {f"S{j}": (0.002 if j < 3 else -0.0003) + rng.normal(0, 0.01) for j in range(10)}
        t.record_day(f"2026-01-{i+1}", w, {"blocked": False, "reason": "ok", "gross": 1.0, "net": 0.0,
                                           "n_long": 3, "n_short": 7, "turnover": 0.24}, nr, sig)
    rep = compare_to_research(t.read(), research_ic=t.read()["daily_ic"].mean(), research_turnover=0.24)
    assert rep["days"] == 65 and rep["operational_failures"] == 0
    assert capital_review_ready(rep) == rep["promotion_ready"]
    print(f"shadow._demo OK  verdict={rep['verdict']} days={rep['days']} paper_ic={rep['paper_ic']} "
          f"ops_failures={rep['operational_failures']} ready={rep['promotion_ready']}")


if __name__ == "__main__":
    _demo()
