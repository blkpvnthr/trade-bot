"""
scripts/post_market_retrain_pipeline.py

Run build_intraday_training_dataset → forecast_engine → rl_intraday_structure_trainer
in sequence after market close. Validates all output artifacts and writes a summary report.

Usage:
  python scripts/post_market_retrain_pipeline.py              # today
  python scripts/post_market_retrain_pipeline.py --date 20260521
  RL_DATA_DATE=20260521 python scripts/post_market_retrain_pipeline.py

Exit 0 = all steps OK + all required artifacts present.
Exit 1 = any step failed or artifact missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")


def _run_step(
    name: str,
    script: Path,
    env_overrides: dict[str, str],
    log_fh,
    timeout_seconds: int = 3600,
) -> dict:
    start = time.monotonic()
    log_fh.write(f"\n{'='*72}\n[step] {name}\n[started] {datetime.now(ET).isoformat()}\n{'='*72}\n")
    log_fh.flush()

    if not script.exists():
        msg = f"script_not_found: {script}"
        log_fh.write(f"[error] {msg}\n")
        return {"ok": False, "returncode": -1, "elapsed_s": 0.0, "error": msg, "output": ""}

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in env_overrides.items()})
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        combined = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        elapsed = round(time.monotonic() - start, 2)
        ok = proc.returncode == 0
        log_fh.write(combined + "\n")
        log_fh.write(f"\n[returncode] {proc.returncode} [elapsed] {elapsed}s\n")
        log_fh.flush()
        return {"ok": ok, "returncode": proc.returncode, "elapsed_s": elapsed, "error": "" if ok else combined[-2000:], "output": combined}
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - start, 2)
        msg = f"timeout_after_{timeout_seconds}s"
        log_fh.write(f"[timeout] {msg}\n")
        return {"ok": False, "returncode": -1, "elapsed_s": elapsed, "error": msg, "output": ""}
    except Exception as exc:
        elapsed = round(time.monotonic() - start, 2)
        log_fh.write(f"[exception] {exc}\n")
        return {"ok": False, "returncode": -1, "elapsed_s": elapsed, "error": str(exc), "output": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-market retraining pipeline")
    parser.add_argument("--date", default="", help="YYYYMMDD date tag (default: today ET)")
    args = parser.parse_args()

    date_tag = args.date or os.getenv("RL_DATA_DATE", "") or datetime.now(ET).strftime("%Y%m%d")

    log_dir = PROJECT_ROOT / "logs" / "execution"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"post_market_retrain_{date_tag}.log"

    out_dir = PROJECT_ROOT / "system-data" / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_out = PROJECT_ROOT / "daily_system_outputs"
    daily_out.mkdir(parents=True, exist_ok=True)

    summary_path = daily_out / "post_market_retrain_summary.json"

    print(f"[post-market-retrain] date={date_tag} log={log_path}")

    scripts = {
        "build_intraday_training_dataset": PROJECT_ROOT / "build_intraday_training_dataset.py",
        "forecast_engine": PROJECT_ROOT / "forecast_engine.py",
        "rl_intraday_structure_trainer": PROJECT_ROOT / "rl_intraday_structure_trainer.py",
    }

    env_base = {"RL_DATA_DATE": date_tag, "RUN_DATA_DATE": date_tag, "ALPACA_REGULAR_FEED": "iex"}

    step_results: dict[str, dict] = {}
    combined_output = ""

    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"[post-market-retrain] pipeline start date={date_tag} at {datetime.now(ET).isoformat()}\n")

        for step_name, script_path in scripts.items():
            result = _run_step(step_name, script_path, env_base, log_fh)
            step_results[step_name] = result
            combined_output += result.get("output", "")
            print(f"[post-market-retrain] {step_name}: ok={result['ok']} elapsed={result['elapsed_s']}s")
            if not result["ok"]:
                print(f"[post-market-retrain] step failed — continuing to validate artifacts")

        log_fh.write(f"\n[post-market-retrain] pipeline end at {datetime.now(ET).isoformat()}\n")

    # Artifact validation
    def artifact(filename: str) -> dict:
        p = out_dir / filename
        return {"exists": p.exists(), "path": str(p)}

    artifacts = {
        "bars_csv":          artifact(f"{date_tag}_intraday_bars_1m.csv"),
        "training_dataset":  artifact(f"{date_tag}_intraday_training_dataset.csv"),
        "exit_dataset":      artifact(f"{date_tag}_exit_training_dataset.csv"),
        "rl_model_zip":      artifact(f"{date_tag}_rl_intraday_structure_model.zip"),
        "rl_metrics_json":   artifact(f"{date_tag}_rl_intraday_training_metrics.json"),
    }

    provenance_confirmed = "[data-source] training_csv=alpaca_derived" in combined_output

    all_steps_ok = all(r["ok"] for r in step_results.values())
    all_artifacts_present = all(a["exists"] for a in artifacts.values())
    all_ok = all_steps_ok and all_artifacts_present

    summary = {
        "date": date_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": {k: {kk: vv for kk, vv in v.items() if kk != "output"} for k, v in step_results.items()},
        "artifacts": artifacts,
        "provenance_confirmed": provenance_confirmed,
        "all_steps_ok": all_steps_ok,
        "all_artifacts_present": all_artifacts_present,
        "all_ok": all_ok,
        "log_path": str(log_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[post-market-retrain] summary written to {summary_path}")

    for name, art in artifacts.items():
        status = "OK" if art["exists"] else "MISSING"
        print(f"  [{status}] {name}: {art['path']}")

    print(f"[post-market-retrain] provenance_confirmed={provenance_confirmed}")
    print(f"[post-market-retrain] all_ok={all_ok}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
