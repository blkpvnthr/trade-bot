import pandas as pd
import pytest

from rl_meta import data
from rl_meta.data import (normalize_bars, load_bars, cache_path, COLUMNS,
                          read_universe, DataError)

RAW = [
    {"t": "2023-01-04T05:00:00Z", "o": 10, "h": 11, "l": 9.5, "c": 10.5, "v": 1000},
    {"t": "2023-01-03T05:00:00Z", "o": 10, "h": 11, "l": 9.0, "c": 9.8, "v": 2000},
]


def test_normalize_columns_and_sort():
    df = normalize_bars(RAW, "NVDA")
    assert list(df.columns) == COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert df.iloc[0]["close"] == 9.8          # earlier bar first after sort
    assert (df["symbol"] == "NVDA").all()
    assert df["volume"].tolist() == [2000.0, 1000.0]


def test_normalize_empty():
    df = normalize_bars([], "X")
    assert df.empty and list(df.columns) == COLUMNS


def test_cache_path_format():
    p = cache_path("NVDA", "1Day", "2018-01-01", "2025-01-01")
    assert p.name == "NVDA_1Day_2018-01-01_2025-01-01.csv"


def test_cache_write_then_reuse(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(symbol, timeframe, start, end=None, **kw):
        calls["n"] += 1
        return normalize_bars(RAW, symbol)

    monkeypatch.setattr(data, "fetch_bars", fake_fetch)

    # First call: cache miss -> fetch + write.
    df1 = load_bars("NVDA", "1Day", "2023-01-01", "2023-02-01", bars_dir=tmp_path)
    assert calls["n"] == 1
    assert (tmp_path / "NVDA_1Day_2023-01-01_2023-02-01.csv").exists()

    # Second call: cache hit -> NO fetch.
    df2 = load_bars("NVDA", "1Day", "2023-01-01", "2023-02-01", bars_dir=tmp_path)
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_offline_cache_miss_raises(tmp_path):
    with pytest.raises(DataError):
        load_bars("NVDA", "1Day", "2099-01-01", "2099-02-01",
                  offline=True, bars_dir=tmp_path)


def test_missing_credentials_raises(monkeypatch):
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(data, "_load_env_file", lambda: None)   # no .env fallback
    with pytest.raises(DataError):
        data.get_credentials()


def test_universe_config():
    syms = read_universe()
    assert syms == ["NVDA", "GOOGL", "ASML", "TSM", "KLAC", "FIGS", "TGTX", "YOU"]
