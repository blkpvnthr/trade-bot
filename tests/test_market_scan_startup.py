"""Production-startup regression tests for MarketScanner construction + lifecycle.

Guards the constructor initialization-order bug (default _make_stream path read
self._now_fn before it was set) that unit tests missed because they always inject
a stream_client. No sockets are opened (StreamClient only connects on run()).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.config import MarketScanConfig
from market_scan.scanner import MarketScanner
from market_scan.stream_client import StreamClient
from market_scan import thread_runner


def _exec():
    return SimpleNamespace(
        broker=SimpleNamespace(api_key="k", secret_key="s",
                               get_all_open_positions=lambda: [], list_open_orders=lambda: []),
        config=SimpleNamespace(paper=True))


def test_default_constructor_builds_real_stream_without_crash():
    # The regression: no injected stream_client -> _make_stream() runs in __init__.
    ms = MarketScanner(_exec(), MarketScanConfig())   # must NOT raise AttributeError
    assert isinstance(ms.stream, StreamClient)
    # clock shared with the stream (consistent is_live semantics), via DI
    assert ms.stream._now_fn is ms._now_fn
    # the production factory is wired (default), and no socket was opened at construct
    assert ms.stream._stream is None


def test_injected_clock_propagates_to_default_stream():
    clk = lambda: 123.0
    ms = MarketScanner(_exec(), MarketScanConfig(), now_fn=clk)
    assert ms._now_fn is clk and ms.stream._now_fn is clk


def test_injected_stream_client_is_used_as_is():
    sentinel = SimpleNamespace(name="injected")
    ms = MarketScanner(_exec(), MarketScanConfig(), stream_client=sentinel)
    assert ms.stream is sentinel


def test_default_scanner_factory_constructs_without_crash():
    # thread_runner's default factory builds a real MarketScanner (the production
    # path exercised at startup) -- must not raise.
    ms = thread_runner._default_scanner_factory(_exec(), MarketScanConfig())
    assert isinstance(ms, MarketScanner)


def test_scanner_init_failure_is_non_fatal_and_releases_lock(tmp_path):
    # A factory that raises must NOT propagate: start_market_scanner returns None,
    # the lock is freed (so a later start can succeed), and the engine survives.
    lock_path = tmp_path / "scanner.lock"

    def boom(executor, config):
        raise RuntimeError("simulated scanner init crash")
    handle = thread_runner.start_market_scanner(
        _exec(), MarketScanConfig(enabled=True), scanner_factory=boom, lock_path=lock_path)
    assert handle is None
    assert not lock_path.exists()   # lock released -> no stuck duplicate-instance guard
