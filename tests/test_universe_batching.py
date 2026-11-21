"""Tests for the canonical universe loader + deterministic batching + coverage
accounting + rate-limit retry + live-scan rotation in universe.py.

Proves the complete-universe-batching acceptance criteria:
  - 55 symbols produce batches of 20, 20, and 15
  - every symbol appears exactly once, original ordering preserved
  - duplicates and blank values are removed
  - invalid batch sizes fail
  - a middle-batch failure is reported and not silently ignored
  - returned frames from all successful batches are combined (no overwrite)
  - universe.txt missing or empty causes startup failure (fail-closed)
  - only transient (429/5xx) failures are retried; permanent 4xx are not
  - the live-scan rotation advances every call (no first-batch starvation)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

import universe


# 55-symbol fixture (duplicate + blank included to exercise normalization).
def _universe_55() -> list[str]:
    return [f"SYM{i:03d}" for i in range(55)]


# ---------------------------------------------------------------- batching
def test_55_symbols_produce_20_20_15():
    syms = _universe_55()
    batches = list(universe.iter_symbol_batches(syms, 20))
    assert [len(b) for b in batches] == [20, 20, 15]


def test_real_universe_txt_matches_canonical_and_batches_cleanly():
    # The root-level universe.txt is the single source of truth and is
    # republished by the real-time scanner, so its exact count is NOT hardcoded
    # here (it was 54, is currently 31, and will keep changing). Instead we
    # validate the batching CONTRACT against whatever the real file holds:
    # all-but-last batches are full (20) and the last is the remainder.
    syms = universe.load_canonical_universe()
    n = len(syms)
    assert n >= 1  # fail-closed loader guarantees non-empty
    batches = list(universe.iter_symbol_batches(syms, 20))
    assert all(len(b) == 20 for b in batches[:-1])
    assert len(batches[-1]) == (n % 20 or 20)
    # every symbol exactly once, original order preserved (no dupes)
    flat = [s for b in batches for s in b]
    assert flat == syms and len(flat) == len(set(flat))


def test_every_symbol_appears_exactly_once_order_preserved():
    syms = _universe_55()
    flat = [s for b in universe.iter_symbol_batches(syms, 20) for s in b]
    assert flat == syms
    assert len(flat) == len(set(flat))


def test_duplicates_and_blanks_removed_preserving_first_order():
    raw = ["aapl", " msft ", "AAPL", "", None, "'tsla'", "TSLA", "  ", "nvda"]
    batches = list(universe.iter_symbol_batches(raw, 2))
    flat = [s for b in batches for s in b]
    assert flat == ["AAPL", "MSFT", "TSLA", "NVDA"]
    assert [len(b) for b in batches] == [2, 2]


def test_invalid_batch_size_raises():
    with pytest.raises(ValueError):
        list(universe.iter_symbol_batches(["AAPL"], 0))
    with pytest.raises(ValueError):
        list(universe.iter_symbol_batches(["AAPL"], -3))


# ---------------------------------------------------------------- fail-closed loading
def test_missing_universe_file_fails_closed(tmp_path):
    with pytest.raises(RuntimeError):
        universe.load_canonical_universe(tmp_path / "does_not_exist.txt")


def test_empty_universe_file_fails_closed(tmp_path):
    p = tmp_path / "universe.txt"
    p.write_text("# only a comment, no symbols\n   \n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        universe.load_canonical_universe(p)


def test_quoted_comma_separated_format_parsed_and_deduped(tmp_path):
    p = tmp_path / "universe.txt"
    p.write_text("'INTC', 'DIA', 'SPY', 'INTC'  # trailing comment\n", encoding="utf-8")
    assert universe.load_canonical_universe(p) == ["INTC", "DIA", "SPY"]


# ---------------------------------------------------------------- retry classification
class _StatusError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def test_transient_errors_are_retried_permanent_are_not():
    assert universe.is_transient_error(_StatusError(429)) is True
    assert universe.is_transient_error(_StatusError(500)) is True
    assert universe.is_transient_error(_StatusError(503)) is True
    assert universe.is_transient_error(_StatusError(401)) is False  # auth
    assert universe.is_transient_error(_StatusError(403)) is False  # entitlement
    assert universe.is_transient_error(_StatusError(400)) is False  # malformed
    assert universe.is_transient_error(Exception("HTTP 503 Service Unavailable")) is True
    assert universe.is_transient_error(Exception("400 Bad Request")) is False


def test_permanent_failure_not_retried():
    calls = {"n": 0}

    def _fetch(batch):
        calls["n"] += 1
        raise _StatusError(403)

    with pytest.raises(_StatusError):
        universe.fetch_with_retry(_fetch, ["A"], max_retries=5, base_delay=0,
                                  sleep_fn=lambda d: None, log_fn=lambda m: None)
    assert calls["n"] == 1  # no retries on a permanent 4xx


def test_transient_failure_retried_then_succeeds():
    calls = {"n": 0}

    def _fetch(batch):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _StatusError(429)
        return pd.DataFrame({"symbol": batch})

    out = universe.fetch_with_retry(_fetch, ["A"], max_retries=5, base_delay=0,
                                    sleep_fn=lambda d: None, log_fn=lambda m: None)
    assert calls["n"] == 3 and not out.empty


# ---------------------------------------------------------------- batched fetch + coverage
def _ok_fetch(batch):
    return pd.DataFrame({"symbol": list(batch), "close": [1.0] * len(batch)})


def test_batched_fetch_combines_all_frames_without_overwrite():
    syms = _universe_55()
    combined, report = universe.fetch_bars_batched(
        _ok_fetch, syms, batch_size=20, sleep_fn=lambda d: None, log_fn=lambda m: None)
    assert report.batch_count == 3
    assert report.attempted_symbol_count == 55
    assert report.successful_symbol_count == 55
    assert report.missing_symbol_count == 0
    assert report.failed_batch_count == 0
    assert report.is_complete is True
    # every symbol present exactly once in the combined frame
    assert len(combined) == 55
    assert sorted(combined["symbol"]) == sorted(syms)


def test_middle_batch_failure_is_reported_not_silently_ignored():
    syms = _universe_55()
    failing_symbol = syms[20]  # in the second (middle) batch

    def _fetch(batch):
        if failing_symbol in batch:
            raise _StatusError(500)  # transient, will exhaust retries
        return _ok_fetch(batch)

    combined, report = universe.fetch_bars_batched(
        _fetch, syms, batch_size=20, max_retries=1, base_delay=0,
        sleep_fn=lambda d: None, log_fn=lambda m: None)

    assert report.failed_batch_count == 1
    assert len(report.failed_symbols) == 20
    assert report.missing_symbol_count == 20
    assert report.is_complete is False               # never claims full coverage
    assert failing_symbol in report.failed_symbols
    # the two good batches (20 + 15) are still combined
    assert len(combined) == 35


def test_no_data_distinguished_from_failure_and_nontradable():
    syms = ["A", "B", "C"]

    def _fetch(batch):
        # return data for A and C only; B returns no row (no-data)
        rows = [{"symbol": s, "close": 1.0} for s in batch if s != "B"]
        return pd.DataFrame(rows)

    combined, report = universe.fetch_bars_batched(
        _fetch, syms, batch_size=10, sleep_fn=lambda d: None, log_fn=lambda m: None)
    assert "B" in report.no_data_symbols
    assert "B" not in report.failed_symbols
    assert report.successful_symbol_count == 2
    report.record_nontradable(["C"])
    assert "C" in report.nontradable_symbols
    d = report.to_dict()
    assert d["no_data_symbols"] == ["B"] and d["nontradable_symbols"] == ["C"]


def test_coverage_summary_lines_format():
    syms = _universe_55()
    _, report = universe.fetch_bars_batched(
        _ok_fetch, syms, batch_size=20, sleep_fn=lambda d: None, log_fn=lambda m: None)
    lines = report.summary_lines()
    assert lines[0] == "[universe] loaded=55 batch_size=20 batches=3"
    assert lines[1] == "[universe] attempted=55 successful=55 missing=0 failed_batches=0"


# ---------------------------------------------------------------- live-scan rotation
def test_rotation_no_first_batch_starvation_and_cycle_completion():
    syms = _universe_55()
    rot = universe.UniverseScanRotation(syms, 20)
    assert rot.batch_count == 3

    b0 = rot.next_batch()
    b1 = rot.next_batch()
    b2 = rot.next_batch()
    # consecutive calls advance through the batches (not stuck on the first)
    assert [b0.batch_index, b1.batch_index, b2.batch_index] == [0, 1, 2]
    assert b0.cycle_id == b1.cycle_id == b2.cycle_id == 1
    assert b2.cycle_complete is True

    # a complete cycle attempts every symbol exactly once
    attempted = b0.symbols + b1.symbols + b2.symbols
    assert attempted == syms

    # next cycle wraps and increments the cycle id
    b3 = rot.next_batch()
    assert b3.cycle_id == 2 and b3.batch_index == 0


def test_rotation_requires_symbols():
    with pytest.raises(ValueError):
        universe.UniverseScanRotation([], 20)
