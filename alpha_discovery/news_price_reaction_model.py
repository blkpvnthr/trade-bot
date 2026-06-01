#!/usr/bin/env python3
"""
alpha_discovery/news_price_reaction_model.py — validate news-derived features
against forward price reactions.

Every news (or realtime) feature must earn its place: this module measures how
well a feature predicts forward returns over multiple horizons and emits the
standard validation metric set:

    forward returns : 5m / 15m / 30m / 60m / EOD
    MFE, MAE
    win_rate, expectancy, profit_factor, Sharpe, Sortino

A "feature record" is a dict with the feature value plus realized forward
outcomes, e.g.:
    {"feature": "sentiment_mean_60m", "value": 0.4,
     "ret_5m": 0.001, "ret_15m": 0.002, ..., "ret_eod": 0.01,
     "mfe": 0.012, "mae": -0.004}

The model is direction-aware: a positive feature value is treated as a long
signal; the realized return sign is scored accordingly so both bullish and
bearish features can be validated. Read-only / offline; never raises.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

HORIZONS = ("ret_5m", "ret_15m", "ret_30m", "ret_60m", "ret_eod")


def _to_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else 0.0


def _sharpe(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    return (mean / sd) * math.sqrt(252) if sd > 0 else 0.0


def _sortino(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    downside = [min(0.0, r) for r in returns]
    dd = math.sqrt(sum(d ** 2 for d in downside) / n)
    return (mean / dd) * math.sqrt(252) if dd > 0 else 0.0


def validate_feature(records: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    """Compute the standard validation metrics for a single feature.

    Direction-aware: signal = sign(value); strategy return = signal * forward
    return, so a feature that predicts down-moves validates as strongly as one
    that predicts up-moves.
    """
    vals: list[float] = []
    per_horizon: dict[str, list[float]] = {h: [] for h in HORIZONS}
    signed_returns: list[float] = []   # strategy returns for the primary horizon
    mfes: list[float] = []
    maes: list[float] = []

    primary = "ret_15m"
    for rec in records:
        v = _to_float(rec.get(feature, rec.get("value")))
        if v is None:
            continue
        vals.append(v)
        sign = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        for h in HORIZONS:
            rh = _to_float(rec.get(h))
            if rh is not None:
                per_horizon[h].append(rh)
        rp = _to_float(rec.get(primary))
        if rp is not None and sign != 0.0:
            signed_returns.append(sign * rp)
        mfe = _to_float(rec.get("mfe"))
        mae = _to_float(rec.get("mae"))
        if mfe is not None:
            mfes.append(mfe)
        if mae is not None:
            maes.append(mae)

    n = len(vals)
    if n == 0:
        return {"feature": feature, "n": 0, "status": "no_data"}

    # IC: correlation of feature value with each forward horizon.
    ics = {}
    for h in HORIZONS:
        ys = per_horizon[h]
        if len(ys) == len(vals) and len(ys) >= 2:
            ics[f"ic_{h}"] = round(_pearson(vals, ys), 4)
        else:
            ics[f"ic_{h}"] = 0.0

    wins = [r for r in signed_returns if r > 0]
    losses = [r for r in signed_returns if r < 0]
    win_rate = (len(wins) / len(signed_returns)) if signed_returns else 0.0
    expectancy = (sum(signed_returns) / len(signed_returns)) if signed_returns else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return {
        "feature": feature,
        "n": n,
        "status": "ok",
        **ics,
        "win_rate": round(win_rate, 4),
        "expectancy": round(expectancy, 6),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else 9999.0,
        "sharpe": round(_sharpe(signed_returns), 4),
        "sortino": round(_sortino(signed_returns), 4),
        "mfe_mean": round(sum(mfes) / len(mfes), 6) if mfes else 0.0,
        "mae_mean": round(sum(maes) / len(maes), 6) if maes else 0.0,
    }


def validate_features(records: list[dict[str, Any]], features: Iterable[str]) -> list[dict[str, Any]]:
    """Validate many features over the same record set."""
    return [validate_feature(records, f) for f in features]


def write_validation_report(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write a validation table to CSV (union of keys as columns)."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception:  # pragma: no cover - non-fatal
        logger.warning("failed to write validation report to %s", path)
    return path
