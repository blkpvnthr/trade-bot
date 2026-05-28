#!/usr/bin/env python3
"""
scripts/promote_variant.py

Promote a strategy variant by opening a GitHub PR with a summary of its
performance and proposed config/code changes.

SAFETY CONSTRAINTS:
  - NEVER touches run.py order execution or risk controls
  - NEVER modifies live (non-paper) Alpaca credentials or keys
  - NEVER auto-merges the PR — human review and approval required
  - Only operates on strategy params and configs
  - Will refuse to promote if trade_count < min_trades_for_ranking
  - Will refuse to promote if Sharpe < promote_threshold_sharpe

Usage:
  python scripts/promote_variant.py tux_mtf_original
  python scripts/promote_variant.py tux_mtf_original --force   # skip threshold checks (dev only)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).strftime("%Y%m%d")


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_latest_rankings(out_dir: Path) -> dict[str, Any]:
    files = sorted(out_dir.glob("variant_rankings_*.json"), reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        data = json.load(f)
    return {row["variant_id"]: row for row in data if isinstance(data, list)}


def load_latest_patch_request(strategy_id: str, research_dir: Path) -> str | None:
    files = sorted(
        research_dir.glob(f"*{strategy_id}_patch_request.md"), reverse=True
    )
    if not files:
        return None
    return files[0].read_text()


def check_git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_gh_available() -> bool:
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def promote(
    strategy_id: str,
    config_path: Path = PROJECT_ROOT / "configs" / "strategy_experiments.yaml",
    force: bool = False,
) -> None:
    cfg = load_config(config_path)
    eval_cfg = cfg.get("evaluation", {})
    min_trades = eval_cfg.get("min_trades_for_ranking", 5)
    sharpe_threshold = eval_cfg.get("promote_threshold_sharpe", 1.5)
    pf_threshold = eval_cfg.get("promote_threshold_profit_factor", 1.3)

    rankings = load_latest_rankings(PROJECT_ROOT / "system-data")
    metrics = rankings.get(strategy_id)

    if not metrics:
        print(f"[promote] No ranking data found for '{strategy_id}'. "
              "Run scripts/evaluate_variants.py first.")
        sys.exit(1)

    # ── Safety gate ───────────────────────────────────────────────────────
    if not force:
        trade_count = metrics.get("trade_count", 0) or 0
        sharpe = metrics.get("sharpe_ratio") or 0.0
        pf = metrics.get("profit_factor") or 0.0

        errors = []
        if trade_count < min_trades:
            errors.append(
                f"trade_count={trade_count} < min_trades_for_ranking={min_trades}"
            )
        if sharpe < sharpe_threshold:
            errors.append(
                f"sharpe_ratio={sharpe:.3f} < promote_threshold_sharpe={sharpe_threshold}"
            )
        if pf < pf_threshold:
            errors.append(
                f"profit_factor={pf:.3f} < promote_threshold_profit_factor={pf_threshold}"
            )

        if errors:
            print(f"[promote] BLOCKED — variant '{strategy_id}' does not meet promotion thresholds:")
            for e in errors:
                print(f"  - {e}")
            print("Use --force to override (dev/testing only).")
            sys.exit(1)

    print(f"[promote] '{strategy_id}' passed promotion thresholds.")

    # ── Check tooling ─────────────────────────────────────────────────────
    if not check_git_available():
        print("[promote] git not found — cannot create branch or PR.")
        sys.exit(1)
    if not check_gh_available():
        print("[promote] gh CLI not found — install from https://cli.github.com/")
        sys.exit(1)

    # ── Create branch ─────────────────────────────────────────────────────
    branch_name = f"research/promote-{strategy_id}-{TODAY}"
    try:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)
        print(f"[promote] created branch {branch_name}")
    except subprocess.CalledProcessError as exc:
        print(f"[promote] failed to create branch: {exc}")
        sys.exit(1)

    # ── Load patch request for PR body ────────────────────────────────────
    research_dir = PROJECT_ROOT / "research_requests"
    patch_md = load_latest_patch_request(strategy_id, research_dir)

    metrics_summary = "\n".join(
        f"- **{k.replace('_', ' ').title()}**: {v}"
        for k, v in metrics.items()
        if k not in {"variant_id", "strategy_id", "credential_set", "rankable", "rank"}
        and v is not None
    )

    pr_body = f"""\
## Promote Strategy Variant: `{strategy_id}`

**Generated:** {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}
**Branch:** `{branch_name}`

### Performance Summary

{metrics_summary}

### Patch Request

{"See attached patch request below." if patch_md else "_No patch request found. Review metrics above._"}

---

### Safety Checklist

- [ ] Changes are limited to `strategies/` or `configs/`
- [ ] `run.py` execution, risk controls, and kill switch are unchanged
- [ ] No live credentials modified
- [ ] `python -m py_compile` passes on changed strategy files
- [ ] Backtest reviewed by human before merging

**⚠️ DO NOT merge without human review.**
**⚠️ This is a paper-trading variant promotion only — NOT a live trading change.**

---

*Generated by `scripts/promote_variant.py` — review all changes before approving.*
"""

    # ── Commit any config changes staged ─────────────────────────────────
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(["git", "add", "configs/", "strategies/"], check=False)
        subprocess.run(
            ["git", "commit", "-m", f"research: promote variant {strategy_id} ({TODAY})"],
            check=False,
        )

    # ── Push and open PR ──────────────────────────────────────────────────
    try:
        subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[promote] git push failed: {exc}. Check remote config.")
        sys.exit(1)

    try:
        pr_result = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", f"research: promote {strategy_id} paper variant ({TODAY})",
                "--body", pr_body,
                "--label", "research,paper-trading,strategy-variant",
            ],
            capture_output=True, text=True, check=True,
        )
        print(f"[promote] PR created: {pr_result.stdout.strip()}")
    except subprocess.CalledProcessError as exc:
        print(f"[promote] gh pr create failed: {exc.stderr}")
        sys.exit(1)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Promote a strategy variant via PR")
    parser.add_argument("strategy_id", help="Strategy variant ID to promote")
    parser.add_argument("--config", default="configs/strategy_experiments.yaml")
    parser.add_argument(
        "--force", action="store_true",
        help="Skip threshold checks (dev/testing only — not for live promotion)",
    )
    args = parser.parse_args()

    if args.force:
        print("[promote] WARNING: --force bypasses safety thresholds. Dev use only.")

    promote(
        args.strategy_id,
        config_path=PROJECT_ROOT / args.config,
        force=args.force,
    )


if __name__ == "__main__":
    main()
