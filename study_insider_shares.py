#!/usr/bin/env python3
"""
study_insider_shares.py — the INSIDER signal on SHARES (no options).

The insider-buy edge (+2.8% vs SPY @5d, PIT-validated, t=5.82) lives in small-caps whose OPTIONS
were untradeable (spreads 30-56%). But shares have ~0 spread. This is the clean test: does the
strongest signal in the project pay when expressed as STOCK? Parallels study_shares.py (momentum).

Entry: SEC Form-4 cluster (>$1M, 2-day window, 2-day filing lag) → next-day OPEN.
Exit: N-day hold at close. vs unconditional all-days benchmark (same names/horizon).
Metrics per horizon: n, mean, median, Sharpe, win%, MaxDD, profit factor.
"""
from __future__ import annotations

import statistics
from typing import Dict, List

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab

HORIZONS = [3, 5, 10, 20]


def _mdd(rets):
    eq = peak = 1.0; mdd = 0.0
    for r in rets:
        eq *= (1.0 + r)
        if eq <= 0:
            return -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return mdd


def _stat(rows):
    if not rows:
        return None
    v = [r[1] for r in rows]; n = len(v); m = statistics.mean(v)
    sd = statistics.pstdev(v) if n > 1 else 0.0
    wins = [x for x in v if x > 0]; losses = [x for x in v if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    return {"n": n, "mean": m, "median": statistics.median(v), "win": len(wins) / n,
            "sharpe": (m / sd) if sd > 0 else 0.0, "mdd": _mdd([r[1] for r in sorted(rows)]), "pf": pf}


def run(max_names: int = 400):
    pit = base.load_pit_buy_events()
    sig = {h: [] for h in HORIZONS}; allday = {h: [] for h in HORIZONS}
    names = 0
    for sym in sorted(pit.keys()):
        if names >= max_names:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        names += 1
        td = list(bars.index)
        op = [float(bars.loc[d, "open"]) for d in td]
        close = [float(bars.loc[d, "close"]) for d in td]
        ents = set(ab._insider_entry_indices(td, sym, pit))
        for i in range(60, len(td)):
            for h in HORIZONS:
                if i + 1 + h >= len(td):
                    continue
                ep, xp = op[i + 1], close[i + 1 + h]
                if ep <= 0 or xp <= 0:
                    continue
                ret = xp / ep - 1.0
                allday[h].append((td[i + 1].isoformat(), ret))
                if i in ents:
                    sig[h].append((td[i + 1].isoformat(), ret))
    return sig, allday, names


def report(sig, allday, names):
    L = ["# INSIDER signal on SHARES — the strongest project edge, no options", "",
         f"SEC Form-4 >$1M clusters → next-day open → close at horizon. {names} PIT names. NO options. "
         "Benchmark = unconditional all-days forward return (same names/horizon).", ""]
    for h in HORIZONS:
        s = _stat(sig[h]); a = _stat(allday[h])
        L += [f"## {h}-day hold", "", "| Set | n | Mean | Median | Sharpe | Win% | MaxDD | PF |",
              "|---|---|---|---|---|---|---|---|"]
        for name, st in (("Insider entries", s), ("All-days benchmark", a)):
            if not st:
                L.append(f"| {name} | 0 |—|—|—|—|—|—|"); continue
            pf = "∞" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
            L.append(f"| **{name}** | {st['n']} | {st['mean']*100:+.2f}% | {st['median']*100:+.2f}% | "
                     f"{st['sharpe']:+.3f} | {st['win']*100:.0f}% | {st['mdd']*100:.0f}% | {pf} |")
        if s and a:
            L += ["", f"_Insider edge over unconditional: **{(s['mean']-a['mean'])*100:+.2f}%** per "
                  f"{h}-day trade (insider {s['mean']*100:+.2f}% vs market {a['mean']*100:+.2f}%)._", ""]
    L += ["## Interpretation", "",
          "- Insider PF>1, mean>0, beats benchmark by a wide margin → a STRONG share edge worth trading.",
          "- The option spreads (30-56%) that killed insider-options are absent in shares.",
          "_Compare vs momentum-shares: +0.40%/trade edge, PF 1.53 @10d._"]
    return "\n".join(L)


def main():
    print("Running INSIDER signal on shares (local bars)...")
    sig, allday, names = run()
    md = report(sig, allday, names)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "insider_shares.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
