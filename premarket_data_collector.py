#!/usr/bin/env python3
"""
premarket_data_collector.py

Continuously refreshes premarket/session-planning data between 4:00 AM ET
and regular market open.

Default:
    python premarket_data_collector.py

It repeatedly runs:
    1. llm_sentiment_universe_selector.py
    2. market_context_builder.py
    3. premarket_trade_planner.py
    4. build_intraday_training_dataset.py
    5. options_booster.py

Why this exists:
    run.py should not just run planning files once after 4 AM.
    Between 4:00 and 9:30 ET, premarket volume, support/resistance,
    VWAP/ORB context, unusual options flow, and universe selection can change.
    This collector refreshes those files continuously so run.py starts the
    regular session with fresh planning data.

Outputs refreshed by the child scripts include:
    daily_system_outputs/latest_llm_sentiment_universe.json
    locked_trading_universe.txt
    daily_system_outputs/market_context.json
    daily_system_outputs/premarket_trade_plan.json
    daily_system_outputs/intraday_structure_map.json
    daily_system_outputs/intraday_bars_1m.csv
    daily_system_outputs/latest_option_flow_candidates.csv
    daily_system_outputs/latest_option_booster_candidates.csv
    daily_system_outputs/latest_option_booster_ranked_contracts.csv
    daily_system_outputs/premarket_collector_state.json
    logs/execution/premarket_collector_events.jsonl
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from market_context_builder import DAILY_OUTPUT_DIR, TODAY_TAG


# =============================================================================
# Paths
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = (
    DAILY_OUTPUT_DIR / f"{TODAY_TAG}_premarket_collector_state.json"
)
EVENTS_PATH  = (
    DAILY_OUTPUT_DIR / f"{TODAY_TAG}_premarket_collector_events.jsonl"
)
# =============================================================================
# Config
# =============================================================================

ET = ZoneInfo("America/New_York")

PREMARKET_COLLECTOR_ENABLED = os.getenv("PREMARKET_COLLECTOR_ENABLED", "1").lower() in {"1", "true", "yes"}
PREMARKET_START_ET = os.getenv("PREMARKET_COLLECTOR_START_ET", "04:00")
PREMARKET_END_ET = os.getenv("PREMARKET_COLLECTOR_END_ET", "09:30")

# Main cadence.
COLLECTOR_POLL_SECONDS = int(os.getenv("PREMARKET_COLLECTOR_POLL_SECONDS", "300"))

# Script-specific minimum intervals. These prevent expensive modules from being
# run too often while still keeping market/volume-sensitive files fresh.
LLM_UNIVERSE_INTERVAL_SECONDS = int(os.getenv("PREMARKET_LLM_UNIVERSE_INTERVAL_SECONDS", "900"))
MARKET_CONTEXT_INTERVAL_SECONDS = int(os.getenv("PREMARKET_MARKET_CONTEXT_INTERVAL_SECONDS", "300"))
PREMARKET_PLAN_INTERVAL_SECONDS = int(os.getenv("PREMARKET_PLAN_INTERVAL_SECONDS", "300"))
INTRADAY_DATASET_INTERVAL_SECONDS = int(os.getenv("PREMARKET_INTRADAY_DATASET_INTERVAL_SECONDS", "600"))
OPTIONS_FLOW_INTERVAL_SECONDS = int(os.getenv("PREMARKET_OPTIONS_FLOW_INTERVAL_SECONDS", "600"))

SCRIPT_TIMEOUT_SECONDS = int(os.getenv("PREMARKET_SCRIPT_TIMEOUT_SECONDS", "900"))

# Turn individual modules on/off.
RUN_LLM_UNIVERSE = os.getenv("PREMARKET_RUN_LLM_UNIVERSE", "1").lower() in {"1", "true", "yes"}
RUN_MARKET_CONTEXT = os.getenv("PREMARKET_RUN_MARKET_CONTEXT", "1").lower() in {"1", "true", "yes"}
RUN_PREMARKET_PLAN = os.getenv("PREMARKET_RUN_PREMARKET_PLAN", "1").lower() in {"1", "true", "yes"}
RUN_INTRADAY_DATASET = os.getenv("PREMARKET_RUN_INTRADAY_DATASET", "1").lower() in {"1", "true", "yes"}
RUN_OPTIONS_FLOW = os.getenv("PREMARKET_RUN_OPTIONS_FLOW", "1").lower() in {"1", "true", "yes"}

# Run one full cycle even if launched outside the time window.
RUN_ONCE_IF_OUTSIDE_WINDOW = os.getenv("PREMARKET_RUN_ONCE_IF_OUTSIDE_WINDOW", "0").lower() in {"1", "true", "yes"}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CollectorTask:
    name: str
    script: str
    enabled: bool
    interval_seconds: int
    required: bool = False


TASKS = [
    CollectorTask(
        name="llm_sentiment_universe_selector",
        script="llm_sentiment_universe_selector.py",
        enabled=RUN_LLM_UNIVERSE,
        interval_seconds=LLM_UNIVERSE_INTERVAL_SECONDS,
        required=False,
    ),
    CollectorTask(
        name="market_context_builder",
        script="market_context_builder.py",
        enabled=RUN_MARKET_CONTEXT,
        interval_seconds=MARKET_CONTEXT_INTERVAL_SECONDS,
        required=False,
    ),
    CollectorTask(
        name="premarket_trade_planner",
        script="premarket_trade_planner.py",
        enabled=RUN_PREMARKET_PLAN,
        interval_seconds=PREMARKET_PLAN_INTERVAL_SECONDS,
        required=True,
    ),
    CollectorTask(
        name="build_intraday_training_dataset",
        script="build_intraday_training_dataset.py",
        enabled=RUN_INTRADAY_DATASET,
        interval_seconds=INTRADAY_DATASET_INTERVAL_SECONDS,
        required=False,
    ),
    CollectorTask(
        name="options_booster",
        script="options_booster.py",
        enabled=RUN_OPTIONS_FLOW,
        interval_seconds=OPTIONS_FLOW_INTERVAL_SECONDS,
        required=False,
    ),
]


# =============================================================================
# Helpers
# =============================================================================

_SHUTDOWN = False


def request_shutdown(signum=None, frame=None) -> None:
    global _SHUTDOWN
    _SHUTDOWN = True
    print("[collector] shutdown requested")


signal.signal(signal.SIGTERM, request_shutdown)
signal.signal(signal.SIGINT, request_shutdown)


def parse_hhmm(value: str) -> dtime:
    h, m = str(value).strip().split(":")
    return dtime(hour=int(h), minute=int(m))


def now_et() -> datetime:
    return datetime.now(ET)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def market_window_for_today() -> tuple[datetime, datetime]:
    today = now_et().date()
    start = datetime.combine(today, parse_hhmm(PREMARKET_START_ET), tzinfo=ET)
    end = datetime.combine(today, parse_hhmm(PREMARKET_END_ET), tzinfo=ET)
    return start, end


def in_collection_window() -> bool:
    start, end = market_window_for_today()
    current = now_et()
    return start <= current < end


def seconds_until_start() -> int:
    start, _ = market_window_for_today()
    current = now_et()
    if current < start:
        return max(0, int((start - current).total_seconds()))

    tomorrow_start = start + timedelta(days=1)
    return max(0, int((tomorrow_start - current).total_seconds()))


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"last_runs": {}, "cycles": []}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("last_runs", {})
            payload.setdefault("cycles", [])
            return payload
    except Exception:
        pass
    return {"last_runs": {}, "cycles": []}


def write_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = now_utc_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "timestamp_utc": now_utc_iso(),
        "timestamp_et": now_et().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def should_run_task(task: CollectorTask, state: dict[str, Any]) -> bool:
    if not task.enabled:
        return False

    script_path = PROJECT_ROOT / task.script
    if not script_path.exists():
        msg = f"[collector] missing script: {task.script}"
        print(msg)
        write_event("missing_script", {"task": asdict(task), "script_path": str(script_path)})
        return False

    last_runs = state.setdefault("last_runs", {})
    last = last_runs.get(task.name)

    if not last:
        return True

    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)).total_seconds()
        return elapsed >= task.interval_seconds
    except Exception:
        return True


def run_task(task: CollectorTask) -> dict[str, Any]:
    script_path = PROJECT_ROOT / task.script

    print(f"[collector] START {task.name} script={task.script}")
    started = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = {
        "task": task.name,
        "script": task.script,
        "started_at_utc": now_utc_iso(),
        "ok": False,
        "returncode": None,
        "elapsed_seconds": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "error": "",
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SECONDS,
        )

        elapsed = time.time() - started
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        result.update(
            {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "stdout_tail": stdout[-3000:],
                "stderr_tail": stderr[-3000:],
            }
        )

        if stdout.strip():
            print(stdout[-3000:])
        if stderr.strip():
            print(stderr[-3000:])

        status = "DONE" if proc.returncode == 0 else "FAILED"
        print(f"[collector] {status} {task.name} returncode={proc.returncode} elapsed={elapsed:.2f}s")

    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        result.update(
            {
                "ok": False,
                "returncode": -1,
                "elapsed_seconds": round(elapsed, 3),
                "error": f"timeout_after_{SCRIPT_TIMEOUT_SECONDS}s",
                "stdout_tail": (exc.stdout or "")[-3000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-3000:] if isinstance(exc.stderr, str) else "",
            }
        )
        print(f"[collector] TIMEOUT {task.name} elapsed={elapsed:.2f}s")

    except Exception as exc:
        elapsed = time.time() - started
        result.update(
            {
                "ok": False,
                "returncode": -2,
                "elapsed_seconds": round(elapsed, 3),
                "error": str(exc),
            }
        )
        print(f"[collector] ERROR {task.name}: {exc}")

    write_event("task_result", result)
    return result


def run_collection_cycle(state: dict[str, Any]) -> dict[str, Any]:
    cycle = {
        "cycle_started_at_utc": now_utc_iso(),
        "cycle_started_at_et": now_et().isoformat(),
        "results": [],
    }

    print("=" * 80)
    print(f"[collector] cycle start ET={cycle['cycle_started_at_et']}")
    print("=" * 80)

    for task in TASKS:
        if _SHUTDOWN:
            break

        if not should_run_task(task, state):
            continue

        result = run_task(task)
        cycle["results"].append(result)

        if result["ok"]:
            state.setdefault("last_runs", {})[task.name] = now_utc_iso()
            write_state(state)
        elif task.required:
            print(f"[collector] required task failed: {task.name}")

    cycle["cycle_completed_at_utc"] = now_utc_iso()
    cycle["cycle_completed_at_et"] = now_et().isoformat()

    state.setdefault("cycles", []).append(cycle)
    state["cycles"] = state["cycles"][-50:]
    write_state(state)

    write_event("cycle_completed", cycle)
    print(f"[collector] cycle complete results={len(cycle['results'])}")

    return cycle


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("===== PREMARKET DATA COLLECTOR =====")
    print(f"[collector] enabled={PREMARKET_COLLECTOR_ENABLED}")
    print(f"[collector] window={PREMARKET_START_ET}-{PREMARKET_END_ET} ET")
    print(f"[collector] poll_seconds={COLLECTOR_POLL_SECONDS}")
    print(f"[collector] project_root={PROJECT_ROOT}")
    print(f"[collector] state={STATE_PATH}")

    if not PREMARKET_COLLECTOR_ENABLED:
        print("[collector] disabled")
        return

    print("[collector] tasks:")
    for task in TASKS:
        print(
            f"  - {task.name}: enabled={task.enabled} "
            f"interval={task.interval_seconds}s script={task.script}"
        )

    state = read_state()

    if not in_collection_window():
        if RUN_ONCE_IF_OUTSIDE_WINDOW:
            print("[collector] outside collection window; running one cycle because PREMARKET_RUN_ONCE_IF_OUTSIDE_WINDOW=1")
            run_collection_cycle(state)
            return

        wait = seconds_until_start()
        print(f"[collector] outside collection window; next start in {wait}s")
        write_event("outside_collection_window", {"seconds_until_start": wait})
        return

    write_event("collector_started", {"window": [PREMARKET_START_ET, PREMARKET_END_ET]})

    while not _SHUTDOWN and in_collection_window():
        state = read_state()
        run_collection_cycle(state)

        if _SHUTDOWN or not in_collection_window():
            break

        sleep_seconds = max(5, COLLECTOR_POLL_SECONDS)
        print(f"[collector] sleeping {sleep_seconds}s")
        for _ in range(sleep_seconds):
            if _SHUTDOWN or not in_collection_window():
                break
            time.sleep(1)

    print("[collector] stopped; outside collection window or shutdown requested")
    write_event("collector_stopped", {"reason": "outside_window_or_shutdown"})


if __name__ == "__main__":
    main()
