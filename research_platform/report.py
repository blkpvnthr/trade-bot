"""Nightly report: champion vs challengers, ranked by the preregistered verdict, written to markdown."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_ORDER = {"PROMOTE": 0, "CONTINUE": 1, "REJECT": 2}


def _row(label, st, extra=""):
    return (f"| {label} | {st['n']} | {st['ndays']} | {st['sharpe']:+.3f} | "
            f"{st['mean_net']*100:+.2f}% | {st['cagr']*100:+.1f}% | {st['mdd']*100:.0f}%{extra} |")


def nightly_report(experiment_name: str, spec_hash: str, results: List[Dict[str, Any]],
                   out_path: str, as_of: str = "") -> str:
    """results = list of evaluate() outputs (each champion vs one challenger)."""
    champ = results[0]["champion"] if results else None
    ranked = sorted(results, key=lambda r: (_ORDER[r["verdict"]], -r["d_sharpe"]))

    L = [f"# Nightly Champion/Challenger Report — {experiment_name}",
         f"_experiment {spec_hash}{('  ·  as of ' + as_of) if as_of else ''}_", "",
         "## Champion", "",
         "| Portfolio | trades | days | Sharpe | mean-net | CAGR | MaxDD |",
         "|---|---|---|---|---|---|---|"]
    if champ:
        L.append(_row(champ.portfolio_id, champ.stats))
    L += ["", "## Challengers (ranked: PROMOTE → CONTINUE → REJECT)", "",
          "| Challenger | verdict | ΔSharpe full | p | ΔSharpe OOS | trades | days | Sharpe | mean-net | CAGR | MaxDD | failed checks |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ranked:
        ch = r["challenger"]
        failed = ",".join(k for k, v in r["checks"].items() if not v) or "—"
        p = "—" if r["p_value"] != r["p_value"] else f"{r['p_value']:.3f}"
        d_oos = r.get("oos", {}).get("d_oos_sharpe")
        d_oos_s = "—" if d_oos is None else f"{d_oos:+.3f}"
        L.append(f"| {ch.portfolio_id} | **{r['verdict']}** | {r['d_sharpe']:+.3f} | {p} | {d_oos_s} | "
                 f"{ch.stats['n']} | {ch.stats['ndays']} | {ch.stats['sharpe']:+.3f} | "
                 f"{ch.stats['mean_net']*100:+.2f}% | {ch.stats['cagr']*100:+.1f}% | "
                 f"{ch.stats['mdd']*100:.0f}% | {failed} |")
    L += ["", "## Verdict legend",
          "- **PROMOTE**: beats champion on every preregistered criterion (incl. bootstrap significance + per-year robustness).",
          "- **CONTINUE**: improves but not yet significant / insufficient burn-in → keep observing.",
          "- **REJECT**: no Sharpe improvement, or breaks drawdown / per-year robustness.", ""]
    for r in ranked:
        ch = r["challenger"]
        L.append(f"- `{ch.portfolio_id}` — {ch.description}")
    out = "\n".join(L)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(out)
    return out
