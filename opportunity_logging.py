#!/usr/bin/env python3
"""
opportunity_logging.py

JSONL event logger for the Opportunity Finder system.
Provides a factory function to create bound event writers.
Mirrors the existing event logging pattern used in run.py and spx_price_action_layer.py.

Usage:
    log_fn = make_logger(Path("logs/execution/20260523/20260523_opportunity_finder_events.jsonl"))
    log_fn("opportunity_finder_start", {"data_tag": "20260523", "universe_size": 30})
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _json_default(obj: Any) -> Any:
    """
    JSON serialization fallback for non-standard types.
    Handles float NaN/Inf, datetime/Timestamp objects.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    # Pandas Timestamp
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def make_logger(events_path: Path) -> Callable[[str, dict], None]:
    """
    Factory that returns a bound write_event function.

    Parameters
    ----------
    events_path : Path
        Absolute path to the .jsonl events file.
        Will be created if missing; parent directories created as needed.

    Returns
    -------
    Callable[[str, dict], None]
        Bound write_event(event: str, payload: dict) function.
        Never raises — silently handles path creation failures.
    """

    # Ensure parent directories exist
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we can't create the directory, return a no-op logger
        return lambda event, payload: None

    def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
        """
        Append one JSONL line to the events file.

        Schema:
            {"timestamp_utc": ISO8601, "event": str, "payload": dict}

        Parameters
        ----------
        event : str
            Event name (e.g., "opportunity_finder_start")
        payload : dict | None
            Event payload dict. Defaults to empty dict.
        """
        if payload is None:
            payload = {}

        line = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }

        try:
            json_str = json.dumps(line, default=_json_default)
            # Atomic append: open in append mode, write, flush, close
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(json_str + "\n")
                f.flush()
        except Exception:
            # Silently fail — never raise
            pass

    return write_event
