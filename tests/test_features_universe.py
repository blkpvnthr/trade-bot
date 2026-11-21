"""Tests for features.py's get_download_universe(): the market-data / trading
universe is sourced SOLELY from universe.txt.

The previous behavior combined CORE_SYMBOLS with the full S&P 500 + NASDAQ pool
and then truncated with MAX_SYMBOLS/[:30], which silently dropped most of the
universe. The active path now returns exactly universe.txt (order preserved,
de-duplicated) and does NOT call the S&P 500 / NASDAQ helpers. The broad-market
helpers survive only as an explicit, research-only escape hatch
(include_nasdaq=True).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import features as feat
import universe


def test_download_universe_equals_canonical_universe_txt(tmp_path):
    result = feat.get_download_universe(out_path=tmp_path / "u.csv")
    canonical = universe.load_canonical_universe()
    assert result == canonical
    # The download universe is EXACTLY the canonical universe.txt -- its size is
    # whatever the (scanner-republished) file holds, not a hardcoded count.
    assert len(result) == len(canonical) and len(result) >= 1


def test_default_path_does_not_call_sp500_or_nasdaq(monkeypatch, tmp_path):
    called = {"sp500": False, "nasdaq": False}

    def _sp_tripwire():
        called["sp500"] = True
        return ["ZZZZ"]

    def _nd_tripwire():
        called["nasdaq"] = True
        return ["YYYY"]

    monkeypatch.setattr(feat, "get_sp500", _sp_tripwire)
    monkeypatch.setattr(feat, "get_nasdaq_constituents", _nd_tripwire)

    result = feat.get_download_universe(out_path=tmp_path / "u.csv")
    assert called["sp500"] is False
    assert called["nasdaq"] is False
    assert "ZZZZ" not in result and "YYYY" not in result


def test_no_max_symbols_truncation():
    # The old MAX_SYMBOLS=30 cap is gone entirely; ALL canonical symbols survive
    # (no artificial truncation), regardless of the current universe.txt size.
    assert not hasattr(feat, "MAX_SYMBOLS")
    assert len(feat.get_download_universe()) == len(universe.load_canonical_universe())


def test_order_preserved_and_deduped(tmp_path):
    result = feat.get_download_universe(out_path=tmp_path / "u.csv")
    assert result == list(dict.fromkeys(result))  # no duplicates
    assert result[:3] == universe.load_canonical_universe()[:3]  # file order preserved


def test_context_symbols_not_in_trading_universe_unless_in_universe_txt(tmp_path):
    # QQQ / VIXY are regime-context proxies, not trading candidates; they are in
    # REGIME_SYMBOLS but (if absent from universe.txt) must NOT appear here.
    canonical = set(universe.load_canonical_universe())
    result = set(feat.get_download_universe(out_path=tmp_path / "u.csv"))
    for proxy in feat.REGIME_SYMBOLS:
        norm = feat.normalize_symbol(proxy)
        if norm not in canonical:
            assert norm not in result


def test_include_nasdaq_true_is_research_escape_hatch(monkeypatch, tmp_path):
    monkeypatch.setattr(feat, "get_sp500", lambda: ["AAPL"])
    monkeypatch.setattr(feat, "get_nasdaq_constituents", lambda: ["NASDAQONLYSYM"])

    result = feat.get_download_universe(include_nasdaq=True, out_path=tmp_path / "u.csv")
    assert "NASDAQONLYSYM" in result
    # canonical symbols still present and first (order preserved)
    assert result[: len(universe.load_canonical_universe())] == universe.load_canonical_universe()


def test_deterministic(tmp_path):
    r1 = feat.get_download_universe(out_path=tmp_path / "u1.csv")
    r2 = feat.get_download_universe(out_path=tmp_path / "u2.csv")
    assert r1 == r2
