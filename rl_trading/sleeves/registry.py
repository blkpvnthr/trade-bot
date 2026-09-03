"""Sleeve registry + status report (dashboard feed).

Every report carries the mandatory honesty banner: no validated live edge, no 1%/day claim, paper
trading is forward-validation only. The status is computed from the gate, not asserted by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .core import Sleeve, SleeveMode

DISCLAIMER = (
    "NO VALIDATED LIVE EDGE YET. NO 1%/day CLAIM. "
    "Paper trading here is forward validation only; live trading is permanently disabled."
)


@dataclass
class SleeveRegistry:
    sleeves: List[Sleeve] = field(default_factory=list)
    champion: Optional[str] = None  # name of the current best research model (may still be scanner-only)
    research_findings: List[str] = field(default_factory=list)  # falsification results for the dashboard

    def add(self, sleeve: Sleeve) -> None:
        if any(s.name == sleeve.name for s in self.sleeves):
            raise ValueError(f"duplicate sleeve name: {sleeve.name}")
        self.sleeves.append(sleeve)

    def get(self, name: str) -> Sleeve:
        for s in self.sleeves:
            if s.name == name:
                return s
        raise KeyError(name)

    def by_mode(self, mode: SleeveMode) -> List[Sleeve]:
        return [s for s in self.sleeves if s.mode() is mode]

    # convenience buckets
    def validated(self) -> List[Sleeve]:
        return [s for s in self.sleeves if s.can_trade()]

    def scanner_only(self) -> List[Sleeve]:
        return self.by_mode(SleeveMode.SCANNER_ONLY)

    def rejected(self) -> List[Sleeve]:
        # "rejected" = has a validation record that was tested and failed (not merely untested/disabled)
        out = []
        for s in self.sleeves:
            r = s.required_validation
            if r is None:
                continue
            tested_and_failed = any(
                v is False for v in (
                    r.placebo_passed, r.oos_passed, r.ood_passed,
                    r.attribution_passed, r.concentration_passed,
                )
            )
            if tested_and_failed and not s.can_trade():
                out.append(s)
        return out


def _sleeve_status(s: Sleeve) -> dict:
    g = s.gate()
    return {
        "name": s.name,
        "signal_source": s.signal_source,
        "native_execution_type": s.native_execution_type,
        "mode": s.mode().value,
        "tradeable": s.can_trade(),
        "paper_enabled": s.paper_enabled,
        "live_enabled": s.live_enabled,  # always False
        "gate_passed": g.passed,
        "gate_failures": list(g.failures),
        "validation_next_step": g.next_step,
        "why_not_tradeable": s.why_not_tradeable(),
        "ledger_id": (s.required_validation.ledger_id if s.required_validation else None),
    }


def status_report(reg: SleeveRegistry) -> Dict[str, object]:
    """Structured status for the dashboard. Pure function of registry state."""
    return {
        "disclaimer": DISCLAIMER,
        "live_status": "DISABLED (permanently)",
        "paper_status": "enabled for validated sleeves only",
        "champion": reg.champion,
        "champion_note": (
            "Best research model to date; remains scanner-only because it does not beat SPY."
        ),
        "counts": {
            "validated": len(reg.validated()),
            "scanner_only": len(reg.scanner_only()),
            "rejected": len(reg.rejected()),
            "total": len(reg.sleeves),
        },
        "validated_sleeves": [_sleeve_status(s) for s in reg.validated()],
        "scanner_only_sleeves": [_sleeve_status(s) for s in reg.scanner_only()],
        "rejected_sleeves": [_sleeve_status(s) for s in reg.rejected()],
        "all_sleeves": [_sleeve_status(s) for s in reg.sleeves],
        "research_findings": list(reg.research_findings),
    }


def render_markdown(reg: SleeveRegistry) -> str:
    """Human-readable status block for /dash or a report file."""
    rep = status_report(reg)
    lines: List[str] = []
    lines.append("# Sleeve Framework Status")
    lines.append("")
    lines.append(f"> **{rep['disclaimer']}**")
    lines.append("")
    lines.append(f"- Live status: **{rep['live_status']}**")
    lines.append(f"- Paper status: {rep['paper_status']}")
    lines.append(f"- Current champion: `{rep['champion']}` — {rep['champion_note']}")
    c = rep["counts"]
    lines.append(
        f"- Sleeves: {c['validated']} validated / {c['scanner_only']} scanner-only / "
        f"{c['rejected']} rejected / {c['total']} total"
    )
    lines.append("")
    lines.append("| Sleeve | Source | Mode | Tradeable | Why not tradeable | Next validation step |")
    lines.append("|---|---|---|---|---|---|")
    for s in rep["all_sleeves"]:
        lines.append(
            f"| {s['name']} | {s['signal_source']} | {s['mode']} | "
            f"{'yes' if s['tradeable'] else 'no'} | {s['why_not_tradeable'] or '—'} | "
            f"{s['validation_next_step']} |"
        )
    if rep.get("research_findings"):
        lines.append("")
        lines.append("## Research findings (falsifications)")
        for f in rep["research_findings"]:
            lines.append(f"- {f}")
    return "\n".join(lines)
