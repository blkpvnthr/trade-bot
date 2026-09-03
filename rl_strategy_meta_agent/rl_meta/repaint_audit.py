"""Repaint / lookahead audit for a PineScript strategy + its exported trades.

Two independent checks, combined into one verdict (safe | suspect | invalid):

  STATIC  -- scan the Pine source for known repaint hazards: pre-v3 security()
             (repaints by default; lookahead_off did not exist), explicit
             lookahead_on, calc_on_every_tick, non-standard chart sources
             (pointfigure/renko/...), barstate gating, and -- the worst case --
             entry signals built directly on a repainting security() call.

  STATS   -- an honest strategy cannot enter at the exact bar extreme every time.
             Implausibly high win rate / profit factor, or a large share of
             trades with ~0 adverse excursion (MAE ~ 0), is a lookahead signature.

Research-only: reads source text and a trades DataFrame; touches no broker.
"""
from __future__ import annotations

import re

import numpy as np

# (regex, severity, description)
HAZARDS = [
    (r"@version\s*=\s*[12]\b", 2, "Pine v1/v2: security() repaints by default (no lookahead_off)"),
    (r"\bsecurity\s*\(", 3, "security() HTF call -- repaints unless lookahead_off (unavailable pre-v3)"),
    (r"lookahead\s*=\s*barmerge\.lookahead_on", 4, "explicit lookahead_on -- consumes future HTF data"),
    (r"calc_on_every_tick\s*=\s*true", 2, "calc_on_every_tick -- intrabar repaint"),
    (r"\b(pointfigure|renko|kagi|linebreak)\s*\(", 2, "non-standard chart source -- repaint-prone"),
    (r"barstate\.(islast|isrealtime)\b", 2, "barstate gating can hindcast on history"),
    (r"request\.security(?![^)\n]*lookahead_off)", 3, "request.security without lookahead_off"),
]
ORDER = {"safe": 0, "suspect": 1, "invalid": 2}


def scan_pine_source(text: str) -> dict:
    found = []
    for pat, sev, desc in HAZARDS:
        n = len(re.findall(pat, text, re.IGNORECASE))
        if n:
            found.append({"severity": sev, "count": n, "description": desc})
    score = sum(h["severity"] * min(h["count"], 3) for h in found)
    signal_repaint = bool(re.search(r"crossover|crossunder|strategy\.entry", text, re.I)) and \
        bool(re.search(r"\bsecurity\s*\(", text, re.I))
    if any(h["severity"] >= 4 for h in found) or (signal_repaint and score >= 5):
        verdict = "invalid"
    elif score >= 3:
        verdict = "suspect"
    else:
        verdict = "safe"
    return {"hazards": found, "score": score,
            "signal_uses_repainting_security": signal_repaint, "static_verdict": verdict}


def signal_implausibility(trades, mae_zero: float = 0.001) -> dict:
    """trades: DataFrame with return_pct (and ideally mae_pct)."""
    ret = trades["return_pct"].to_numpy(float)
    win = float((ret > 0).mean())
    up, dn = ret[ret > 0].sum(), -ret[ret < 0].sum()
    pf = float(up / dn) if dn > 0 else float("inf")
    mae_zero_rate = None
    if "mae_pct" in trades:
        mae = np.abs(trades["mae_pct"].to_numpy(float))
        mae_zero_rate = float(np.mean(mae <= mae_zero))
    flags = []
    if win > 0.85:
        flags.append(f"win_rate {win:.2f} implausibly high")
    if pf > 5:
        flags.append(f"profit_factor {pf:.1f} implausibly high")
    if mae_zero_rate is not None and mae_zero_rate > 0.30:
        flags.append(f"{mae_zero_rate:.0%} of trades have ~0 adverse excursion (entered at the extreme)")
    verdict = "invalid" if len(flags) >= 2 else "suspect" if flags else "safe"
    return {"n": int(len(trades)), "win_rate": round(win, 3),
            "profit_factor": round(pf, 3),
            "mae_zero_rate": round(mae_zero_rate, 3) if mae_zero_rate is not None else None,
            "median_mae_pct": round(float(np.median(np.abs(trades["mae_pct"]))), 4)
            if "mae_pct" in trades else None,
            "flags": flags, "stats_verdict": verdict}


def combine(static: dict, stats: dict) -> str:
    worst = max(static["static_verdict"], stats["stats_verdict"], key=lambda v: ORDER[v])
    # entries built on a repainting security() AND implausible stats -> invalid
    if static["signal_uses_repainting_security"] and stats["stats_verdict"] != "safe":
        worst = "invalid"
    return worst


def _demo():
    repaint_src = ("//@version=2\nch1=security(tickerid,'720',open)\n"
                   "longCondition=crossover(close,ch1)\nstrategy.entry('.',strategy.long)")
    s = scan_pine_source(repaint_src)
    assert s["signal_uses_repainting_security"] and s["static_verdict"] == "invalid"
    clean = "//@version=5\nlongCondition = ta.crossover(close, ta.sma(close,20))\n"
    assert scan_pine_source(clean)["static_verdict"] == "safe"
    import pandas as pd
    perfect = pd.DataFrame({"return_pct": [0.02] * 95 + [-0.01] * 5,
                            "mae_pct": [0.0] * 90 + [0.03] * 10})
    st = signal_implausibility(perfect)
    assert st["stats_verdict"] == "invalid" and len(st["flags"]) >= 2
    assert combine(s, st) == "invalid"
    print("repaint_audit._demo OK  static", s["static_verdict"], "stats", st["stats_verdict"])


if __name__ == "__main__":
    _demo()
