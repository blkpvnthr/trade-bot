"""Real-time Alpaca IEX market scanner package.

Runs in-process inside trade.py as a daemon thread, continuously ranks
eligible US-listed instruments from a single shared IEX websocket, and
atomically republishes the root-level ``universe.txt`` as the authoritative
entry/replacement universe. See
``docs/superpowers/specs/2026-11-21-realtime-iex-market-scanner-integration-design.md``.

Public API is imported lazily (via :func:`start_market_scanner`) so that
``import market_scan`` stays cheap and free of heavy/optional dependencies
(alpaca-py streaming, pandas) until the scanner is actually started.
"""

from __future__ import annotations

from market_scan.config import MarketScanConfig

__all__ = ["MarketScanConfig", "start_market_scanner", "stop_market_scanner"]


def start_market_scanner(executor, config: "MarketScanConfig | None" = None):
    """Start the scanner daemon thread for a running ProductionOrderExecutor.

    Imported lazily to avoid pulling streaming/pandas deps at package import.
    Returns the :class:`market_scan.thread_runner.MarketScanThread` handle, or
    ``None`` if the scanner is disabled.
    """
    from market_scan.thread_runner import start_market_scanner as _start

    return _start(executor, config)


def stop_market_scanner(handle) -> None:
    """Gracefully stop a scanner started by :func:`start_market_scanner`."""
    if handle is not None:
        handle.stop()
