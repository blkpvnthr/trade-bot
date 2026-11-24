"""Tests for persisted per-symbol opportunity statistics (Stage A6).

Deterministic: all time is passed in explicitly via `now`. No network, no clock.
Pins: neutral weighting for unseen symbols, bounded weight driven by EMA +
confidence + freshness decay, idempotent observations, atomic persistence +
corrupt-file recovery, and bounded eviction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.opportunity_stats import OpportunityStats


def _stats(tmp_path, **kw):
    kw.setdefault("confidence_full_obs", 1)   # reach full confidence in one obs for tests
    kw.setdefault("weight_min", 0.5)
    kw.setdefault("weight_max", 1.5)
    return OpportunityStats(tmp_path / "opportunity_stats_paper.json", **kw)


def test_unseen_symbol_has_neutral_weight(tmp_path):
    s = _stats(tmp_path)
    assert s.weight("NVDA", now=1000.0) == 1.0


def test_high_score_recent_confident_raises_weight_toward_max(tmp_path):
    s = _stats(tmp_path)
    s.update("NVDA", score=1.0, eligible=True, now=1000.0, observation_id="t1")
    w = s.weight("NVDA", now=1000.0)
    assert 1.0 < w <= 1.5
    assert abs(w - 1.5) < 1e-9        # full score, full confidence, zero age -> max


def test_low_score_recent_confident_lowers_weight_toward_min(tmp_path):
    s = _stats(tmp_path)
    s.update("WEAK", score=0.0, eligible=False, now=1000.0, observation_id="t1")
    w = s.weight("WEAK", now=1000.0)
    assert 0.5 <= w < 1.0
    assert abs(w - 0.5) < 1e-9


def test_midpoint_score_is_neutral(tmp_path):
    s = _stats(tmp_path)
    s.update("MID", score=0.5, eligible=True, now=1000.0, observation_id="t1")
    assert abs(s.weight("MID", now=1000.0) - 1.0) < 1e-9


def test_ema_moves_toward_new_scores(tmp_path):
    s = _stats(tmp_path, ema_alpha=0.5)
    s.update("X", score=1.0, eligible=True, now=1.0, observation_id="a")
    s.update("X", score=0.0, eligible=False, now=2.0, observation_id="b")
    # EMA = 0.5*0.0 + 0.5*1.0 = 0.5 -> neutral weight
    assert abs(s.weight("X", now=2.0) - 1.0) < 1e-9


def test_weight_decays_toward_neutral_as_observation_ages(tmp_path):
    s = _stats(tmp_path, decay_half_life_s=100.0)
    s.update("NVDA", score=1.0, eligible=True, now=1000.0, observation_id="t1")
    fresh = s.weight("NVDA", now=1000.0)         # ~1.5
    half = s.weight("NVDA", now=1100.0)          # one half-life later
    old = s.weight("NVDA", now=10000.0)          # very stale
    assert fresh > half > old
    assert abs(old - 1.0) < 0.05                 # decays back to neutral


def test_duplicate_observation_id_does_not_double_count(tmp_path):
    s = _stats(tmp_path)
    s.update("NVDA", score=1.0, eligible=True, now=1.0, observation_id="same")
    s.update("NVDA", score=1.0, eligible=True, now=2.0, observation_id="same")
    rec = s.record("NVDA")
    assert rec["seen_count"] == 1
    assert rec["eligibility_count"] == 1


def test_distinct_observations_increment_counts(tmp_path):
    s = _stats(tmp_path)
    s.update("NVDA", score=0.8, eligible=True, now=1.0, observation_id="a")
    s.update("NVDA", score=0.8, eligible=False, now=2.0, observation_id="b")
    rec = s.record("NVDA")
    assert rec["seen_count"] == 2
    assert rec["eligibility_count"] == 1          # only one was eligible


def test_persist_and_reload_roundtrip(tmp_path):
    p = tmp_path / "opportunity_stats_paper.json"
    s = OpportunityStats(p, confidence_full_obs=1)
    s.update("NVDA", score=0.9, eligible=True, now=1000.0, observation_id="t1")
    s.save()
    assert p.exists()
    s2 = OpportunityStats(p, confidence_full_obs=1)   # loads from disk
    assert s2.record("NVDA")["seen_count"] == 1
    assert abs(s2.weight("NVDA", now=1000.0) - s.weight("NVDA", now=1000.0)) < 1e-9


def test_corrupt_file_loads_empty_and_is_fail_safe(tmp_path):
    p = tmp_path / "opportunity_stats_paper.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = OpportunityStats(p, confidence_full_obs=1)    # must not raise
    assert s.weight("NVDA", now=1.0) == 1.0           # neutral, empty


def test_eviction_bounds_tracked_symbols_to_max(tmp_path):
    s = _stats(tmp_path, max_symbols=10)
    for i in range(25):
        s.update(f"S{i}", score=0.5, eligible=True, now=float(i), observation_id=f"o{i}")
    assert s.size() <= 10
    # the most-recently-seen survive; the stalest are evicted
    assert s.record("S24") is not None
    assert s.record("S0") is None


def test_weight_is_deterministic(tmp_path):
    s = _stats(tmp_path)
    s.update("NVDA", score=0.7, eligible=True, now=1000.0, observation_id="t1")
    assert s.weight("NVDA", now=1050.0) == s.weight("NVDA", now=1050.0)


def test_save_is_atomic_json(tmp_path):
    p = tmp_path / "opportunity_stats_paper.json"
    s = OpportunityStats(p, confidence_full_obs=1)
    s.update("NVDA", score=0.9, eligible=True, now=1.0, observation_id="t1")
    s.save()
    data = json.loads(p.read_text(encoding="utf-8"))   # valid JSON on disk
    assert "NVDA" in data.get("symbols", data)
