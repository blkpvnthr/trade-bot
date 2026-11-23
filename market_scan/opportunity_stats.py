"""Persisted per-symbol opportunity statistics (Stage A6).

A simple, deterministic, falsifiable heuristic that lets the scanner favor
recently-productive opportunity candidates when allocating the opportunity slots.
It is NOT an ML/alpha model and execution must never depend on it -- a missing or
corrupt stats file degrades to neutral weighting (weight == 1.0 for every symbol),
and the opportunity ranking still works on the live score alone.

Per symbol we keep:
    rank_score_ema       decaying average of the (normalized) rank score
    eligibility_count    times the symbol passed eligibility
    seen_count           distinct observations
    last_seen_ts         logical time of the last observation
    last_observation_id  idempotency guard (same id never double-counts)

``weight(symbol, now)`` returns a BOUNDED multiplier in
[``weight_min``, ``weight_max``], neutral (1.0) for unseen symbols, driven by:
  * the EMA score relative to ``neutral_score`` (above -> >1, below -> <1),
  * observation confidence (ramps to 1 over ``confidence_full_obs``),
  * freshness (decays back to neutral over ``decay_half_life_s``).

Time is always passed in explicitly (``now``) so scoring is deterministic and
unit-testable with no clock. Persistence is atomic (temp + os.replace) and
fail-safe. The tracked set is bounded to ``max_symbols`` (stalest evicted).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def stats_path(mode: str, *, directory: Optional[str | Path] = None) -> Path:
    """Mode-isolated stats file path (mirrors the live_context_{mode} convention)."""
    base = Path(directory) if directory is not None else Path(__file__).resolve().parent
    return base / f"opportunity_stats_{mode}.json"


class OpportunityStats:
    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "paper",
        max_symbols: int = 500,
        ema_alpha: float = 0.3,
        decay_half_life_s: float = 1800.0,
        confidence_full_obs: int = 5,
        weight_min: float = 0.5,
        weight_max: float = 1.5,
        neutral_score: float = 0.5,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.max_symbols = max(1, int(max_symbols))
        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self.decay_half_life_s = max(1e-9, float(decay_half_life_s))
        self.confidence_full_obs = max(1, int(confidence_full_obs))
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.neutral_score = min(1.0, max(0.0, float(neutral_score)))
        self._stats: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _norm(symbol: object) -> str:
        return str(symbol).strip().upper() if symbol is not None else ""

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    # ------------------------------------------------------------------- update
    def update(self, symbol: object, *, score: float, eligible: bool, now: float,
               observation_id: Optional[str] = None) -> None:
        """Record one observation. Fail-safe: never raises on bad input.

        Passing the same ``observation_id`` as the symbol's last observation is a
        no-op (prevents the same rotation/rank tick from double-counting a symbol).
        """
        sym = self._norm(symbol)
        if not sym:
            return
        try:
            sc = self._clamp(float(score), 0.0, 1.0)
        except (TypeError, ValueError):
            sc = self.neutral_score
        rec = self._stats.get(sym)
        if rec is not None and observation_id is not None \
                and rec.get("last_observation_id") == observation_id:
            return  # idempotent: already counted this observation
        if rec is None:
            ema = sc
            seen = 1
            elig = 1 if eligible else 0
        else:
            prev = float(rec.get("rank_score_ema", sc))
            ema = self.ema_alpha * sc + (1.0 - self.ema_alpha) * prev
            seen = int(rec.get("seen_count", 0)) + 1
            elig = int(rec.get("eligibility_count", 0)) + (1 if eligible else 0)
        self._stats[sym] = {
            "rank_score_ema": ema,
            "eligibility_count": elig,
            "seen_count": seen,
            "last_seen_ts": float(now),
            "last_observation_id": observation_id,
        }
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if len(self._stats) <= self.max_symbols:
            return
        # Evict the stalest (oldest last_seen_ts) first; tie-break on lower EMA.
        ordered = sorted(
            self._stats.items(),
            key=lambda kv: (kv[1].get("last_seen_ts", 0.0), kv[1].get("rank_score_ema", 0.0)),
        )
        for sym, _rec in ordered[: len(self._stats) - self.max_symbols]:
            self._stats.pop(sym, None)

    # ------------------------------------------------------------------- weight
    def weight(self, symbol: object, *, now: float) -> float:
        """Bounded historical multiplier; 1.0 (neutral) for unseen symbols."""
        rec = self._stats.get(self._norm(symbol))
        if rec is None:
            return 1.0
        ema = float(rec.get("rank_score_ema", self.neutral_score))
        age = max(0.0, float(now) - float(rec.get("last_seen_ts", now)))
        recency = 0.5 ** (age / self.decay_half_life_s)             # 1 at age 0 -> 0 as age grows
        confidence = min(1.0, int(rec.get("seen_count", 0)) / self.confidence_full_obs)
        # Signal in [-1, 1]: how far EMA sits above/below neutral.
        denom = max(self.neutral_score, 1.0 - self.neutral_score) or 1.0
        signal = (ema - self.neutral_score) / denom
        span = (self.weight_max - 1.0) if signal >= 0 else (1.0 - self.weight_min)
        w = 1.0 + confidence * recency * signal * span
        return self._clamp(w, self.weight_min, self.weight_max)

    def blended_rank(self, symbol: object, normalized_score: float, *, now: float) -> float:
        """normalized_score (clamped to [0,1]) * historical weight."""
        try:
            sc = self._clamp(float(normalized_score), 0.0, 1.0)
        except (TypeError, ValueError):
            sc = 0.0
        return sc * self.weight(symbol, now=now)

    # ------------------------------------------------------------- introspection
    def record(self, symbol: object) -> Optional[dict[str, Any]]:
        rec = self._stats.get(self._norm(symbol))
        return dict(rec) if rec is not None else None

    def size(self) -> int:
        return len(self._stats)

    # -------------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self._stats = {}
            return
        symbols = data.get("symbols") if isinstance(data, dict) else None
        self._stats = symbols if isinstance(symbols, dict) else {}

    def save(self) -> None:
        """Atomically persist (temp + os.replace). Fail-safe: never raises."""
        payload = {"mode": self.mode, "symbols": self._stats}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            body = json.dumps(payload, default=str, sort_keys=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".opportunity_stats.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self.path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except Exception:  # noqa: BLE001 - persistence is best-effort, never fatal
            pass
