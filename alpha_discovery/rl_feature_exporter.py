#!/usr/bin/env python3
"""
alpha_discovery/rl_feature_exporter.py — export validated realtime + news
features into the RL options schema.

Bridges the realtime feature builders and the RL feature schema defined in
rl_common.py. Responsibilities:

    * provide the canonical names of realtime/news features the RL state can
      consume (sourced from rl_common so there is a single source of truth);
    * extract a schema-controlled subset from a merged feature dict (no raw API
      payloads, no text — only normalized numeric features);
    * emit an RL feature-schema validation report so the schema can be audited.

Critically, the new feature families enter the RL observation vector ONLY when
their env gate is set (RL_REALTIME_AV_FEATURES=1 / RL_NEWS_FEATURES=1), mirroring
the existing PSC gating — so existing trained models and dataset widths are
unchanged unless explicitly enabled. Never raises.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "alpha_discovery" / "rl_feature_schema.json"


def _rl_schema():
    """Lazily import rl_common to read the canonical realtime/news feature lists."""
    try:
        import rl_common as rc

        return {
            "av_realtime_features": list(getattr(rc, "AV_REALTIME_FEATURES", [])),
            "news_features": list(getattr(rc, "NEWS_FEATURES", [])),
            "analytics_features": list(getattr(rc, "ANALYTICS_FEATURES", [])),
            "options_sentiment_features": list(getattr(rc, "OPTIONS_SENTIMENT_FEATURES", [])),
            "av_realtime_enabled": bool(getattr(rc, "AV_REALTIME_ENABLED", False)),
            "news_enabled": bool(getattr(rc, "NEWS_ENABLED", False)),
            "analytics_enabled": bool(getattr(rc, "ANALYTICS_ENABLED", False)),
            "options_sentiment_enabled": bool(getattr(rc, "OPTIONS_SENTIMENT_ENABLED", False)),
            "alpha_entry_features": list(getattr(rc, "ALPHA_ENTRY_FEATURES", [])),
        }
    except Exception as exc:  # pragma: no cover - rl_common import is optional here
        logger.warning("rl_feature_exporter: rl_common unavailable: %s", exc)
        return {
            "av_realtime_features": [], "news_features": [], "analytics_features": [],
            "options_sentiment_features": [],
            "av_realtime_enabled": False, "news_enabled": False, "analytics_enabled": False,
            "options_sentiment_enabled": False,
            "alpha_entry_features": [],
        }


def export_rl_features(merged: dict[str, Any]) -> dict[str, Any]:
    """Extract the realtime + news feature subset that the RL schema declares.

    Returns a dict containing every declared feature name with a numeric value
    (0.0 when absent), regardless of gating — gating only controls whether they
    are appended to the observation vector, not whether they are recorded.
    """
    schema = _rl_schema()
    names = (list(schema["av_realtime_features"]) + list(schema["news_features"])
             + list(schema["analytics_features"]) + list(schema["options_sentiment_features"]))
    out: dict[str, Any] = {}
    for n in names:
        v = merged.get(n)
        if isinstance(v, bool):
            out[n] = int(v)
        elif isinstance(v, (int, float)):
            out[n] = v
        elif v is None:
            out[n] = 0.0
        else:
            try:
                out[n] = float(v)
            except (TypeError, ValueError):
                out[n] = 0.0
    return out


def write_schema_report(path: Optional[Path] = None) -> dict[str, Any]:
    """Write a snapshot of the RL feature schema (counts + gates) for auditing."""
    schema = _rl_schema()
    report = {
        "av_realtime_feature_count": len(schema["av_realtime_features"]),
        "news_feature_count": len(schema["news_features"]),
        "analytics_feature_count": len(schema["analytics_features"]),
        "options_sentiment_feature_count": len(schema["options_sentiment_features"]),
        "av_realtime_enabled": schema["av_realtime_enabled"],
        "news_enabled": schema["news_enabled"],
        "analytics_enabled": schema["analytics_enabled"],
        "options_sentiment_enabled": schema["options_sentiment_enabled"],
        "alpha_entry_feature_count": len(schema["alpha_entry_features"]),
        "av_realtime_features": schema["av_realtime_features"],
        "news_features": schema["news_features"],
        "analytics_features": schema["analytics_features"],
        "options_sentiment_features": schema["options_sentiment_features"],
    }
    out_path = Path(path) if path else DEFAULT_REPORT
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - non-fatal
        logger.warning("rl_feature_exporter: failed to write schema report")
    return report
