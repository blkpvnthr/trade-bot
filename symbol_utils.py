"""
symbol_utils.py — ticker-format helpers.

Yahoo Finance represents class shares with a dash (BRK-B, BF-B), whereas brokers/Alpaca and
the rest of this system use the canonical dot form (BRK.B, BF.B). `normalize_for_yfinance`
maps a canonical/broker ticker to Yahoo's format and is applied ONLY at yfinance call sites —
broker (Alpaca) symbols everywhere else are left unchanged.
"""

from __future__ import annotations


def normalize_for_yfinance(symbol):
    """Canonical/broker ticker → Yahoo Finance format.

    BRK.B → BRK-B, BF.B → BF-B (class shares use '-' on Yahoo). A no-op for ordinary tickers
    (no '.'), and safe on None/empty. Futures like 'ES=F' contain no '.', so they are returned
    unchanged. Use this ONLY when calling yfinance; never to mutate broker/Alpaca symbols.
    """
    if not symbol:
        return symbol
    return str(symbol).strip().replace(".", "-")
