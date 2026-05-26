"""
scripts/smoke_test_training_pipeline.py

Static verification of the training data pipeline.
Checks path alignment, log-line presence, and compile status for the three
pipeline files — no Alpaca API calls, no ML imports, no env vars required.

Exit 0 = all checks passed.
Exit 1 = one or more checks failed (details printed inline).
"""

from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILDER  = PROJECT_ROOT / "build_intraday_training_dataset.py"
FORECAST = PROJECT_ROOT / "forecast_engine.py"
RL       = PROJECT_ROOT / "rl_intraday_structure_trainer.py"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = "[PASS]" if passed else "[FAIL]"
    suffix = f": {detail}" if (not passed and detail) else ""
    print(f"{status} {name}{suffix}")


# ── Read source texts ────────────────────────────────────────────────────────

builder_text  = BUILDER.read_text(encoding="utf-8")
forecast_text = FORECAST.read_text(encoding="utf-8")
rl_text       = RL.read_text(encoding="utf-8")

# ── Check 1: path-align builder → forecast_engine ───────────────────────────
# Builder writes: FEATURE_OUTPUT_PATH = ... f"{TODAY_TAG}_intraday_training_dataset.csv"
# Forecast reads: INTRADAY_TRAIN_PATH = ... f"{DATA_TAG}_intraday_training_dataset.csv"

TRAINING_SUFFIX = "_intraday_training_dataset.csv"
builder_has_training  = TRAINING_SUFFIX in builder_text
forecast_has_training = TRAINING_SUFFIX in forecast_text

check(
    "path-align: builder→forecast_engine",
    builder_has_training and forecast_has_training,
    detail=(
        ""
        if builder_has_training and forecast_has_training
        else f"builder={builder_has_training} forecast={forecast_has_training}"
    ),
)

# ── Check 2: path-align builder → rl_intraday_structure_trainer ─────────────
# Builder writes: RAW_OUTPUT_PATH = ... f"{TODAY_TAG}_intraday_bars_1m.csv"
# RL reads:       OUT_DIR / f"{DATA_TAG}_intraday_bars_1m.csv"  (first in list)

BARS_SUFFIX = "_intraday_bars_1m.csv"
builder_has_bars = BARS_SUFFIX in builder_text
rl_has_bars      = BARS_SUFFIX in rl_text

check(
    "path-align: builder→rl_trainer",
    builder_has_bars and rl_has_bars,
    detail=(
        ""
        if builder_has_bars and rl_has_bars
        else f"builder={builder_has_bars} rl_trainer={rl_has_bars}"
    ),
)

# ── Check 3: log line in forecast_engine.py ─────────────────────────────────
LOG_CONSUMER = "[data-source] training_csv=alpaca_derived"

check(
    "log-line: forecast_engine [data-source]",
    LOG_CONSUMER in forecast_text,
    detail="" if LOG_CONSUMER in forecast_text else f"'{LOG_CONSUMER}' not found",
)

# ── Check 4: log line in rl_intraday_structure_trainer.py ───────────────────
check(
    "log-line: rl_trainer [data-source]",
    LOG_CONSUMER in rl_text,
    detail="" if LOG_CONSUMER in rl_text else f"'{LOG_CONSUMER}' not found",
)

# ── Check 5: feed log line in builder ───────────────────────────────────────
LOG_FEED = "[data-feed] regular="

check(
    "log-line: builder [data-feed]",
    LOG_FEED in builder_text,
    detail="" if LOG_FEED in builder_text else f"'{LOG_FEED}' not found",
)

# ── Check 6: compile build_intraday_training_dataset.py ─────────────────────
try:
    py_compile.compile(str(BUILDER), doraise=True)
    check("compile: build_intraday_training_dataset.py", True)
except py_compile.PyCompileError as exc:
    check("compile: build_intraday_training_dataset.py", False, str(exc))

# ── Check 7: compile forecast_engine.py ─────────────────────────────────────
try:
    py_compile.compile(str(FORECAST), doraise=True)
    check("compile: forecast_engine.py", True)
except py_compile.PyCompileError as exc:
    check("compile: forecast_engine.py", False, str(exc))

# ── Check 8: compile rl_intraday_structure_trainer.py ───────────────────────
try:
    py_compile.compile(str(RL), doraise=True)
    check("compile: rl_intraday_structure_trainer.py", True)
except py_compile.PyCompileError as exc:
    check("compile: rl_intraday_structure_trainer.py", False, str(exc))

# ── Check 9: RL model output path suffix ────────────────────────────────────
check(
    "output-path: rl_trainer model_path",
    "_rl_intraday_structure_model" in rl_text,
    detail="" if "_rl_intraday_structure_model" in rl_text else "'_rl_intraday_structure_model' not found",
)

# ── Check 10: RL metrics output path suffix ──────────────────────────────────
check(
    "output-path: rl_trainer metrics_path",
    "_rl_intraday_training_metrics" in rl_text,
    detail="" if "_rl_intraday_training_metrics" in rl_text else "'_rl_intraday_training_metrics' not found",
)

# ── Summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print()
if passed == total:
    print(f"All {total} checks passed.")
    sys.exit(0)
else:
    print(f"{total - passed}/{total} checks failed.")
    sys.exit(1)
