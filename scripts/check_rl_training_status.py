"""
scripts/check_rl_training_status.py

Diagnose RL model training state: process running, log path, artifact presence,
provenance confirmation, last errors, and recommended next command.

Usage:
  python scripts/check_rl_training_status.py
  python scripts/check_rl_training_status.py --date 20260521
  python scripts/check_rl_training_status.py --strategy tux_mtf_original
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")


def _pgrep(pattern: str) -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True,
        )
        return [int(x) for x in result.stdout.strip().split() if x.isdigit()]
    except Exception:
        return []


def _read_tail(path: Path, n: int = 10) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RL training status")
    parser.add_argument("--date", default="", help="YYYYMMDD date (default: today ET)")
    parser.add_argument("--strategy", default=os.getenv("STRATEGY_ID", "tux_mtf_original"),
                        help="Strategy ID for log path lookup")
    args = parser.parse_args()

    date_tag = args.date or os.getenv("RL_DATA_DATE", "") or datetime.now(ET).strftime("%Y%m%d")
    strategy_id = args.strategy

    out_dir = PROJECT_ROOT / "system-data" / date_tag
    model_path = out_dir / f"{date_tag}_rl_intraday_structure_model.zip"
    metrics_path = out_dir / f"{date_tag}_rl_intraday_training_metrics.json"

    # Log search order: run.py-written log, manual fallback
    log_candidates = [
        PROJECT_ROOT / "logs" / "execution" / strategy_id / f"{date_tag}_rl_trainer.log",
        Path("/tmp/rl_trainer_run.log"),
    ]
    trainer_log = next((p for p in log_candidates if p.exists()), None)

    pids = _pgrep("rl_intraday_structure_trainer")
    trainer_running = bool(pids)

    tail_lines = _read_tail(trainer_log) if trainer_log else []

    provenance_confirmed = False
    error_lines: list[str] = []
    saved_lines: list[str] = []

    if trainer_log and trainer_log.exists():
        try:
            full_text = trainer_log.read_text(encoding="utf-8", errors="replace")
            provenance_confirmed = "[data-source] training_csv=alpaca_derived" in full_text
            error_lines = [l for l in full_text.splitlines() if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()][-5:]
            saved_lines = [l for l in full_text.splitlines() if "[saved]" in l]
        except Exception:
            pass

    # Determine recommendation
    if trainer_running:
        recommendation = f"Wait for trainer to finish (PID {pids})"
    elif model_path.exists() and metrics_path.exists():
        recommendation = "Training complete — model and metrics present."
    elif not (out_dir / f"{date_tag}_intraday_bars_1m.csv").exists():
        recommendation = (
            "Bars CSV missing. Run:\n"
            f"  ALPACA_REGULAR_FEED=iex python build_intraday_training_dataset.py\n"
            f"  python rl_intraday_structure_trainer.py"
        )
    else:
        recommendation = "Run: python rl_intraday_structure_trainer.py 2>&1 | tee /tmp/rl_trainer_run.log"

    sep = "═" * 52
    print(f"\nRL Training Status — {date_tag}")
    print(sep)
    print(f"trainer_process_running : {trainer_running}" + (f" (PID {pids})" if trainer_running else ""))
    print(f"trainer_log_exists      : {trainer_log is not None} ({trainer_log or 'not found'})")
    print(f"model_zip_exists        : {model_path.exists()} ({model_path})")
    print(f"metrics_json_exists     : {metrics_path.exists()} ({metrics_path})")
    print(f"provenance_confirmed    : {provenance_confirmed}")
    print(f"bars_csv_exists         : {(out_dir / (date_tag + '_intraday_bars_1m.csv')).exists()}")

    if saved_lines:
        print(f"\nSaved artifacts (from log):")
        for l in saved_lines:
            print(f"  {l.strip()}")

    if error_lines:
        print(f"\nLast error lines:")
        for l in error_lines:
            print(f"  {l.strip()}")
    elif tail_lines:
        print(f"\nLast {len(tail_lines)} log lines:")
        for l in tail_lines:
            print(f"  {l.rstrip()}")

    print(f"\nRecommended next command:")
    print(f"  {recommendation}")
    print()


if __name__ == "__main__":
    main()
