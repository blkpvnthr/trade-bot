#!/usr/bin/env python3
"""
scripts/generate_claude_patch_request.py

Generate a structured Claude patch request markdown file based on
evaluation rankings and diagnostic findings for a strategy variant.

Output: research_requests/YYYYMMDD_{strategy_id}_patch_request.md

These files are intended to be read by a developer or fed to Claude Code
in a future session. They NEVER auto-modify live trading logic.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).strftime("%Y%m%d")

TEMPLATE = """\
# Claude Patch Request — {strategy_id}
**Generated:** {date}
**Variant:** `{strategy_id}`
**Config:** `configs/strategy_experiments.yaml`

---

## Context

{context_block}

---

## Evaluation Summary (last {lookback_days} days)

| Metric | Value |
|--------|-------|
{metrics_table}

---

## Diagnostic Findings

{findings_block}

---

## Requested Changes

{changes_block}

---

## Constraints

- **DO NOT** modify `run.py` order execution, risk controls, or kill switch logic
- **DO NOT** change live (non-paper) credential handling
- **DO NOT** auto-promote to live trading
- **DO NOT** disable exits or stops
- All changes must remain in `strategies/{strategy_id}.py` or `configs/strategy_experiments.yaml`
- After changes: run `python -m py_compile strategies/{strategy_id}.py` to verify syntax

---

## Verification Steps

1. `python -m py_compile strategies/{strategy_id}.py`
2. Backtest against last 30 days of `logs/execution/{strategy_id}/`
3. Run `scripts/evaluate_variants.py` after 5 trading days to compare metrics
4. Confirm no regression in Sharpe or max drawdown vs baseline

---

*This request was generated automatically by `scripts/generate_claude_patch_request.py`.*
*Review all suggestions before applying. Human approval required.*
"""


def load_latest_rankings(out_dir: Path) -> dict[str, Any]:
    """Load the most recent variant_rankings CSV."""
    files = sorted(out_dir.glob("variant_rankings_*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        data = json.load(f)
    return {row["variant_id"]: row for row in data if isinstance(data, list)}


def load_latest_diagnosis(strategy_id: str, research_dir: Path) -> dict[str, Any]:
    """Load the most recent diagnosis JSON for a strategy."""
    files = sorted(research_dir.glob(f"diagnose_{strategy_id}_*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f)


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_patch_request(
    strategy_id: str,
    config_path: Path = PROJECT_ROOT / "configs" / "strategy_experiments.yaml",
) -> str:
    cfg = load_config(config_path)
    variant_cfg = cfg.get("variants", {}).get(strategy_id, {})
    eval_cfg = cfg.get("evaluation", {})
    lookback_days = eval_cfg.get("lookback_days", 30)

    out_dir = PROJECT_ROOT / "system-data"
    research_dir = PROJECT_ROOT / "research_requests"

    rankings = load_latest_rankings(out_dir)
    diagnosis = load_latest_diagnosis(strategy_id, research_dir)

    # ── Context block ─────────────────────────────────────────────────────
    context_block = (
        f"Strategy `{strategy_id}` is a paper-trading variant running in the "
        f"QuantumQuant research cluster (bot credential set: "
        f"`{variant_cfg.get('credential_set', 'unknown')}`). "
        f"\n\nDescription: {variant_cfg.get('description', 'N/A')}"
        f"\n\nCurrent params:\n```yaml\n"
        + yaml.dump(variant_cfg.get("params", {}), default_flow_style=False)
        + "```"
    )

    # ── Metrics table ─────────────────────────────────────────────────────
    metrics = rankings.get(strategy_id, {})
    metric_keys = [
        ("trade_count", "Trade Count"),
        ("total_return", "Total Return ($)"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown ($)"),
        ("profit_factor", "Profit Factor"),
        ("win_rate", "Win Rate"),
        ("mae_avg", "Avg MAE"),
        ("mfe_avg", "Avg MFE"),
        ("exposure_pct", "Avg Exposure %"),
        ("daily_loss_violations", "Daily Loss Violations"),
    ]
    metrics_table = "\n".join(
        f"| {label} | {metrics.get(key, 'N/A')} |"
        for key, label in metric_keys
    )

    # ── Findings block ────────────────────────────────────────────────────
    findings = diagnosis.get("findings", [])
    if findings:
        findings_block = "\n".join(
            f"### {i+1}. `{f['pattern']}` [{f['severity'].upper()}]\n"
            f"- **Observed:** {f['detail']}\n"
            f"- **Recommendation:** {f['recommendation']}\n"
            for i, f in enumerate(findings)
        )
    else:
        findings_block = "_No significant failure patterns detected in the analysis window._"

    # ── Changes block ─────────────────────────────────────────────────────
    changes = []
    for finding in findings:
        if finding["pattern"] == "late_entries":
            changes.append(
                "- Review `strategies/{}.py` signal loop for entry delay. "
                "Consider reducing `max_sweep_age_bars` or `swing_lookback` to catch entries earlier.".format(strategy_id)
            )
        elif finding["pattern"] == "bad_regime":
            changes.append(
                "- Add a `regime` context check gate in `generate_signals()`. "
                "Skip entries when `context['regime']` is `'bearish'` or `'mixed'`."
            )
        elif finding["pattern"] == "weak_volume":
            changes.append(
                "- Increase `volume_confirm_mult` by 0.1–0.2 in `configs/strategy_experiments.yaml` "
                f"under `variants.{strategy_id}.params`."
            )
        elif finding["pattern"] == "excessive_reentry":
            changes.append(
                "- Add `max_entries_per_day` param check in the signal loop. "
                "Track per-symbol entry count and skip if already at limit."
            )
        elif finding["pattern"] == "winners_giving_back":
            changes.append(
                "- Reduce `atr_target_mult` by 0.25. "
                "Or add partial-exit logic: close 50% at 1× ATR, trail remainder."
            )
        elif finding["pattern"] == "stops_too_tight":
            changes.append(
                f"- Increase `atr_stop_mult` in `configs/strategy_experiments.yaml` "
                f"under `variants.{strategy_id}.params` (try current + 0.25)."
            )
        elif finding["pattern"] == "stops_too_wide":
            changes.append(
                f"- Decrease `atr_stop_mult` in `configs/strategy_experiments.yaml` "
                f"under `variants.{strategy_id}.params` (try current - 0.25)."
            )
        elif finding["pattern"] == "poor_symbol_selection":
            changes.append(
                "- Add chronic-loser symbols to a `symbol_blacklist` config field. "
                "Or tighten the TradingView screener RVOL / market cap filters."
            )
        elif finding["pattern"] == "overtrading":
            changes.append(
                "- Add `max_entries_per_day: 3` to the variant config. "
                "Or increase volume/confirmation multipliers to reduce signal frequency."
            )
        elif finding["pattern"] == "undertrading":
            changes.append(
                "- Reduce `volume_confirm_mult` by 0.1. "
                "Or loosen `n_confirm_tfs` / `rsi_buy_min` to generate more signals."
            )

    if not changes:
        changes.append("_No specific code changes requested at this time. Monitor for additional data._")

    changes_block = "\n".join(changes)

    return TEMPLATE.format(
        strategy_id=strategy_id,
        date=datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        context_block=context_block,
        lookback_days=lookback_days,
        metrics_table=metrics_table,
        findings_block=findings_block,
        changes_block=changes_block,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate Claude patch request markdown")
    parser.add_argument("strategy_id", help="Strategy variant ID")
    parser.add_argument("--config", default="configs/strategy_experiments.yaml")
    parser.add_argument("--out-dir", default="research_requests")
    args = parser.parse_args()

    content = build_patch_request(
        args.strategy_id,
        config_path=PROJECT_ROOT / args.config,
    )

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{TODAY}_{args.strategy_id}_patch_request.md"

    out_path.write_text(content)
    print(f"[patch-request] written → {out_path}")


if __name__ == "__main__":
    main()
