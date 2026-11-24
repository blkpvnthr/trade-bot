"""Stage B4: 300s same-symbol reentry cooldown + 15% replacement improvement.

Pins the pure cooldown-gate logic and the replacement-path integration: after a
protective liquidation a symbol is on a timed cooldown (allowing earlier reentry
only on a materially-improved signal), and a freed slot is replaced only by a
candidate whose score is meaningfully better than the exited loser (else cash).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import trade


# ------------------------------------------------------------- pure cooldown gate
def test_cooldown_blocks_same_symbol_within_window_without_improvement():
    cds = {"NVDA": {"ts": 1000.0, "score": 0.80}}
    # 100s later (< 300s), new score only 0.85 (< 0.80*1.15=0.92) -> blocked.
    assert trade.reentry_cooldown_blocks(cds, "NVDA", now_ts=1100.0, new_score=0.85,
                                         cooldown_s=300.0, min_improvement=0.15) is True


def test_cooldown_allows_materially_improved_reentry_within_window():
    cds = {"NVDA": {"ts": 1000.0, "score": 0.80}}
    # 0.95 >= 0.80*1.15 -> allowed even within the window.
    assert trade.reentry_cooldown_blocks(cds, "NVDA", now_ts=1100.0, new_score=0.95,
                                         cooldown_s=300.0, min_improvement=0.15) is False


def test_cooldown_elapsed_never_blocks():
    cds = {"NVDA": {"ts": 1000.0, "score": 0.80}}
    assert trade.reentry_cooldown_blocks(cds, "NVDA", now_ts=1400.0, new_score=0.10,
                                         cooldown_s=300.0, min_improvement=0.15) is False


def test_cooldown_unknown_symbol_never_blocks():
    assert trade.reentry_cooldown_blocks({}, "NVDA", now_ts=1.0, new_score=0.1,
                                         cooldown_s=300.0, min_improvement=0.15) is False


def test_cooldown_unknown_prior_score_blocks_full_window():
    # No comparable prior score (0) -> base behavior: hard block for the window.
    cds = {"NVDA": {"ts": 1000.0, "score": 0.0}}
    assert trade.reentry_cooldown_blocks(cds, "NVDA", now_ts=1100.0, new_score=0.99,
                                         cooldown_s=300.0, min_improvement=0.15) is True


# --------------------------------------------------- replacement 15% improvement
def _repl_exec(loser_score=None):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.broker = SimpleNamespace(
        get_all_open_positions=lambda: [],              # loser is CONFIRMED flat
        list_open_orders=lambda: [])
    e.config = SimpleNamespace(
        paper=True,
        scanner=SimpleNamespace(base_target_weight=0.03, enable_fractional_orders=True))
    e.scanner = SimpleNamespace(
        confirm_buy_direction=lambda syms: pd.DataFrame(
            {"symbol": list(syms), "direction_confirmed": [True] * len(list(syms))}))
    import threading as _t
    e._pending_replacements = {"LOSER"}
    e._pending_replacements_lock = _t.RLock()
    return e


def test_replacement_requires_15pct_improvement_over_loser(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    # loser's exit score recorded as 0.80; a 0.85 candidate is NOT 15% better.
    monkeypatch.setattr(trade, "load_reentry_cooldowns",
                        lambda: {"LOSER": {"ts": 0.0, "score": 0.80}})
    e = _repl_exec()
    e._read_scanner_replacement_candidates = lambda: [
        {"symbol": "WEAK", "price": 50.0, "score": 0.85}]
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert out.empty                                    # not materially better -> cash


def test_replacement_accepts_materially_better_candidate(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(trade, "load_reentry_cooldowns",
                        lambda: {"LOSER": {"ts": 0.0, "score": 0.80}})
    e = _repl_exec()
    e._read_scanner_replacement_candidates = lambda: [
        {"symbol": "STRONG", "price": 50.0, "score": 0.95}]   # >= 0.80*1.15
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert list(out["symbol"]) == ["STRONG"]


def test_replacement_fallback_when_loser_score_incomparable(monkeypatch):
    # No recorded loser score -> documented fallback: top confirmed, positive score.
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(trade, "load_reentry_cooldowns", lambda: {})
    e = _repl_exec()
    e._read_scanner_replacement_candidates = lambda: [
        {"symbol": "GOOD", "price": 50.0, "score": 0.30}]
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert list(out["symbol"]) == ["GOOD"]


def test_replacement_rejects_nonpositive_score(monkeypatch):
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "write_event", lambda *a, **k: None)
    monkeypatch.setattr(trade, "load_reentry_cooldowns", lambda: {})
    e = _repl_exec()
    e._read_scanner_replacement_candidates = lambda: [
        {"symbol": "ZERO", "price": 50.0, "score": 0.0}]     # no positive expectancy
    out = e._augment_rebalance_with_replacements(pd.DataFrame(), equity=10000.0)
    assert out.empty
