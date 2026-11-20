"""Tests for features.py's GICS sector extraction (get_sp500_table /
get_sp500_sector_map), used to populate universe.py's equity exposure
groups for Objective 2's covariance-aware sizing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import features as feat

FAKE_SP500_HTML = """
<table id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware</td></tr>
<tr><td>MSFT</td><td>Microsoft</td><td>Information Technology</td><td>Systems Software</td></tr>
<tr><td>JPM</td><td>JPMorgan Chase</td><td>Financials</td><td>Diversified Banks</td></tr>
<tr><td>brk.b</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
</table>
"""


def _mock_response(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.text = FAKE_SP500_HTML
    fake_resp.raise_for_status = MagicMock()
    monkeypatch.setattr(feat.requests, "get", lambda *a, **k: fake_resp)


def test_get_sp500_table_parses_gics_sector_column(monkeypatch):
    _mock_response(monkeypatch)
    df = feat.get_sp500_table()
    assert "GICS Sector" in df.columns
    assert set(df["Symbol"]) == {"AAPL", "MSFT", "JPM", "BRK-B"}


def test_get_sp500_sector_map_returns_correct_mapping(monkeypatch):
    _mock_response(monkeypatch)
    sector_map = feat.get_sp500_sector_map()
    assert sector_map["AAPL"] == "Information Technology"
    assert sector_map["MSFT"] == "Information Technology"
    assert sector_map["JPM"] == "Financials"
    assert sector_map["BRK-B"] == "Financials"


def test_get_sp500_still_returns_symbol_list(monkeypatch):
    _mock_response(monkeypatch)
    symbols = feat.get_sp500()
    assert symbols == sorted({"AAPL", "MSFT", "JPM", "BRK-B"})


def test_get_sp500_table_dedupes_duplicate_symbols(monkeypatch):
    html_with_dupe = FAKE_SP500_HTML.replace(
        "<tr><td>JPM</td>",
        "<tr><td>AAPL</td><td>Apple Inc. dupe</td><td>Information Technology</td><td>dupe</td></tr>\n<tr><td>JPM</td>",
    )
    fake_resp = MagicMock()
    fake_resp.text = html_with_dupe
    fake_resp.raise_for_status = MagicMock()
    monkeypatch.setattr(feat.requests, "get", lambda *a, **k: fake_resp)

    df = feat.get_sp500_table()
    assert (df["Symbol"] == "AAPL").sum() == 1


def test_get_sp500_sector_map_empty_when_column_missing(monkeypatch):
    html_no_sector = """
    <table><tr><th>Symbol</th><th>Security</th></tr>
    <tr><td>AAPL</td><td>Apple Inc.</td></tr></table>
    """
    fake_resp = MagicMock()
    fake_resp.text = html_no_sector
    fake_resp.raise_for_status = MagicMock()
    monkeypatch.setattr(feat.requests, "get", lambda *a, **k: fake_resp)

    assert feat.get_sp500_sector_map() == {}
