import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import buffett_accumulation_layer as bal


def _cached_payload(symbol: str, hours_ago: float, price: float = 123.45) -> dict:
    return {
        "symbol": symbol,
        "fetched_at_utc": (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(),
        "current_price": price,
        "market_cap": 1_000_000_000.0,
        "revenue_ttm": 100_000_000.0,
    }


def test_get_fundamentals_uses_stale_cache_when_live_fetch_fails(monkeypatch, tmp_path):
    symbol = "AAPL"
    cache_path = tmp_path / f"bal_fd_{symbol}.json"
    cache_path.write_text(json.dumps(_cached_payload(symbol, hours_ago=18)), encoding="utf-8")

    monkeypatch.setattr(bal, "_cache_path", lambda _symbol: cache_path)
    monkeypatch.setattr(
        bal,
        "_fetch_fundamentals",
        lambda _symbol: bal.FundamentalData(
            symbol=_symbol,
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            fetch_error="info_timeout>8s",
        ),
    )

    fd = bal.get_fundamentals(symbol, bal.BALConfig())

    assert fd.current_price == 123.45
    assert fd.market_cap == 1_000_000_000.0
    assert fd.fetch_error.startswith("stale_cache_fallback:")


def test_get_fundamentals_does_not_use_expired_stale_cache(monkeypatch, tmp_path):
    symbol = "MSFT"
    cache_path = tmp_path / f"bal_fd_{symbol}.json"
    cache_path.write_text(json.dumps(_cached_payload(symbol, hours_ago=200, price=222.22)), encoding="utf-8")

    live_fd = bal.FundamentalData(
        symbol=symbol,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        fetch_error="info_timeout>8s",
    )

    monkeypatch.setattr(bal, "_cache_path", lambda _symbol: cache_path)
    monkeypatch.setattr(bal, "_fetch_fundamentals", lambda _symbol: live_fd)

    fd = bal.get_fundamentals(symbol, bal.BALConfig())

    assert fd is live_fd
    assert fd.current_price == 0.0
    assert fd.fetch_error == "info_timeout>8s"
