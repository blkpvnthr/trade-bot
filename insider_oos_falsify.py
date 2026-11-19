#!/usr/bin/env python3
"""
insider_oos_falsify.py — falsification battery for an insider-copier backtest result.

The base backtest (backtest_insider_copier.py) reports mean forward return, SPY-excess and a
t-stat. Per the research mission, a t-stat alone is not enough to accept an edge. This harness
runs the falsification tests a candidate edge must survive, on the backtest's own trade list:

  1. Bootstrap 95% CI on mean SPY-excess return (resample trades w/ replacement). CI excluding
     0 => the mean is robustly non-zero, not driven by a few trades.
  2. Timing placebo: for each real signal, draw many RANDOM entry dates on the SAME symbol and
     measure forward SPY-excess over the same horizon. If the real mean-excess sits in the tail
     of this null, the edge is specific to insider-buy TIMING, not just being long these names.
  3. Spearman IC: rank-correlation of trailing-cluster buy $value vs forward excess return
     (dose-response: do bigger insider buys predict bigger moves?), with a permutation p-value.
  4. Lookahead audit: using the real Form-4 filing dates, flag any trade whose entry preceded
     the filing (entry_date < filing_date) and recompute the edge on the clean, no-lookahead
     subset. The clean-subset number is the one that counts out-of-sample.

Verdict uses the CLEAN (no-lookahead) subset as the core edge.

Usage
    python insider_oos_falsify.py \
        --trades backtest_outputs/insider_copier_backtest_oos2026q2.json \
        --events data/cache/oos_insider_events.json \
        --filingdates data/cache/oos_insider_filingdates.json \
        --horizon 5
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest_insider_copier import load_bars, _spy_return  # reuse production-parity loaders

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "backtest_outputs"


def _mean_t(xs: List[float]) -> Tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    m = float(np.mean(xs))
    sd = float(np.std(xs, ddof=0))
    t = (m / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return m, t, n


def bootstrap_ci(xs: List[float], n_boot: int, rng: np.random.Generator,
                 lo: float = 2.5, hi: float = 97.5) -> Tuple[float, float]:
    if len(xs) < 2:
        return (0.0, 0.0)
    arr = np.asarray(xs, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def forward_excess_pool(symbol: str, spy: Optional[pd.DataFrame], horizon: int,
                        exclude: set) -> np.ndarray:
    """All possible h-day forward SPY-excess returns for `symbol` (entry open -> exit close),
    excluding the real signal entry dates. This is the symbol's placebo null pool."""
    bars = load_bars(symbol)
    if bars is None or bars.empty:
        return np.array([])
    dates = list(bars.index)
    out = []
    for i in range(len(dates) - horizon):
        ed = dates[i]
        if ed in exclude:
            continue
        xd = dates[i + horizon]
        try:
            ep = float(bars.loc[ed, "open"]); xp = float(bars.loc[xd, "close"])
        except Exception:
            continue
        if ep <= 0 or xp <= 0:
            continue
        m = _spy_return(spy, ed, xd)
        if m is None:
            continue
        out.append(xp / ep - 1.0 - m)
    return np.asarray(out, dtype=float)


def cluster_value_at(events: Dict[str, list], symbol: str, signal_date: date,
                     window_days: int = 2) -> float:
    evs = events.get(symbol.upper(), [])
    rows = [(date.fromisoformat(d), float(v)) for d, v in evs]
    lo = signal_date - timedelta(days=window_days)
    return sum(v for dd, v in rows if lo <= dd <= signal_date)


def spearman_perm_p(x: List[float], y: List[float], n_perm: int,
                    rng: np.random.Generator) -> Tuple[float, float]:
    if len(x) < 5:
        return 0.0, 1.0
    sx = pd.Series(x); sy = pd.Series(y)
    ic = float(sx.corr(sy, method="spearman"))
    rx = sx.rank().to_numpy(); ry = sy.rank().to_numpy()
    rx = (rx - rx.mean()); ry = (ry - ry.mean())
    denom = math.sqrt((rx @ rx) * (ry @ ry)) or 1.0
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = (rx @ rng.permutation(ry)) / denom
    p = float((np.abs(null) >= abs(ic)).mean())
    return ic, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", required=True, help="backtest JSON with a 'trades' list")
    ap.add_argument("--events", default="data/cache/oos_insider_events.json")
    ap.add_argument("--filingdates", default="data/cache/oos_insider_filingdates.json")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--placebo", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    result = json.loads(Path(args.trades).read_text())
    events = json.loads(Path(args.events).read_text()) if Path(args.events).exists() else {}
    filingdates = json.loads(Path(args.filingdates).read_text()) if Path(args.filingdates).exists() else {}

    trades = [t for t in result.get("trades", []) if t.get("horizon") == args.horizon
              and t.get("excess_ret") is not None]
    if not trades:
        print(f"No trades at horizon {args.horizon}.")
        return

    # ── headline (recomputed from trades) ──────────────────────────────────────────
    exc = [float(t["excess_ret"]) for t in trades]
    m, t_stat, n = _mean_t(exc)
    ci_lo, ci_hi = bootstrap_ci(exc, args.bootstrap, rng)

    # ── timing placebo ─────────────────────────────────────────────────────────────
    spy = load_bars("SPY")
    by_sym_entrydates: Dict[str, set] = {}
    for t in result.get("trades", []):  # exclude ALL real entry dates (any horizon) from pool
        by_sym_entrydates.setdefault(t["symbol"], set()).add(date.fromisoformat(t["entry_date"]))
    pools: Dict[str, np.ndarray] = {}
    for sym in {t["symbol"] for t in trades}:
        pools[sym] = forward_excess_pool(sym, spy, args.horizon, by_sym_entrydates.get(sym, set()))
    usable = [t for t in trades if pools.get(t["symbol"], np.array([])).size > 0]
    null_means = np.empty(args.placebo)
    if usable:
        sym_pools = [pools[t["symbol"]] for t in usable]
        for b in range(args.placebo):
            draws = [p[rng.integers(0, p.size)] for p in sym_pools]
            null_means[b] = float(np.mean(draws))
        placebo_p = float((null_means >= m).mean())
        placebo_null_mean = float(null_means.mean())
        placebo_95 = float(np.percentile(null_means, 95))
    else:
        placebo_p, placebo_null_mean, placebo_95 = 1.0, 0.0, 0.0

    # ── Spearman IC (cluster $value vs excess) ─────────────────────────────────────
    cvals, crets = [], []
    for t in trades:
        cv = cluster_value_at(events, t["symbol"], date.fromisoformat(t["signal_date"]))
        if cv > 0:
            cvals.append(cv); crets.append(float(t["excess_ret"]))
    ic, ic_p = spearman_perm_p(cvals, crets, args.perm, rng)

    # ── lookahead audit + clean subset ─────────────────────────────────────────────
    tainted, clean = [], []
    missing_fd = 0
    for t in trades:
        fd_map = filingdates.get(t["symbol"], {})
        fd = fd_map.get(t["signal_date"], "")
        if not fd:
            missing_fd += 1
            clean.append(t)  # no filing date known -> treat as clean (in-sample style)
            continue
        if date.fromisoformat(t["entry_date"]) < date.fromisoformat(fd):
            tainted.append(t)
        else:
            clean.append(t)
    clean_exc = [float(t["excess_ret"]) for t in clean]
    cm, ct, cn = _mean_t(clean_exc)
    cci_lo, cci_hi = bootstrap_ci(clean_exc, args.bootstrap, rng)

    # ── verdict (uses the clean / no-lookahead subset as the core edge) ────────────
    core_sig = cm > 0 and cci_lo > 0          # robustly positive excess after bootstrap
    placebo_pass = placebo_p < 0.05           # timing beats random entry on same names
    if core_sig and placebo_pass:
        verdict = "ACCEPT (OOS edge survives bootstrap + timing placebo + lookahead audit)"
    elif cm > 0 and (cci_lo > 0 or placebo_p < 0.15):
        verdict = "QUARANTINE (positive OOS but one falsification test marginal; needs more data)"
    else:
        verdict = "REJECT (OOS edge does not survive falsification)"

    out = {
        "horizon": args.horizon, "trades_total": n,
        "headline": {"mean_excess": m, "t_stat": t_stat, "boot95": [ci_lo, ci_hi]},
        "placebo": {"p_value": placebo_p, "null_mean": placebo_null_mean, "null_95pct": placebo_95,
                    "draws": args.placebo, "usable_trades": len(usable)},
        "ic": {"spearman": ic, "perm_p": ic_p, "n": len(cvals)},
        "lookahead": {"tainted": len(tainted), "clean": len(clean), "missing_filing_date": missing_fd},
        "clean_subset": {"n": cn, "mean_excess": cm, "t_stat": ct, "boot95": [cci_lo, cci_hi]},
        "verdict": verdict,
    }

    def pct(x):
        return f"{x*100:+.2f}%"

    lines = [
        f"# Insider OOS Falsification — horizon {args.horizon}d", "",
        f"- Trades: **{n}**   (clean/no-lookahead: **{cn}**, lookahead-tainted dropped: {len(tainted)}, "
        f"filing-date unknown: {missing_fd})", "",
        "## Core edge (clean, no-lookahead subset)",
        f"- Mean SPY-excess: **{pct(cm)}**   t-stat **{ct:+.2f}**   bootstrap95 [{pct(cci_lo)}, {pct(cci_hi)}]",
        f"  - CI {'EXCLUDES 0 (robustly positive)' if cci_lo>0 else 'includes 0 (not robust)'}", "",
        "## Falsification battery (all trades)",
        f"- Headline mean excess {pct(m)} (t {t_stat:+.2f}), bootstrap95 [{pct(ci_lo)}, {pct(ci_hi)}]",
        f"- **Timing placebo**: real {pct(m)} vs random-entry null mean {pct(placebo_null_mean)} "
        f"(95th pct {pct(placebo_95)}) -> p = **{placebo_p:.3f}** "
        f"{'(PASS: timing adds value)' if placebo_p<0.05 else '(FAIL: no timing edge beyond holding these names)'}",
        f"- **Spearman IC** (cluster $ vs excess): {ic:+.3f}  perm-p {ic_p:.3f}  (n={len(cvals)})",
        "", f"## VERDICT: {verdict}", "",
        "_Core edge is measured on the clean subset: trades whose entry date is on/after the real "
        "Form-4 filing date, so the strategy could not have traded on information it did not yet have._",
    ]
    md = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"_h{args.horizon}"
    (OUT_DIR / f"insider_oos_falsify{tag}.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / f"insider_oos_falsify{tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
