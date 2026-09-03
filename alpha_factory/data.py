"""Real-data bridge - load the closes/volume panel alpha_factory expects from the adjusted cache.

Thin wrapper over rl_trading.data.load_panel (the leakage-aware, split/dividend-adjusted cache
loader). No broker/order/live imports here or downstream - this only reads parquet bars.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd

from rl_trading import data as _rl

# ponytail: cap the default cross-section so a run finishes (the miner's ts_rank is a python
# rolling-apply). The goal is rejection-rate confirmation, not exhaustive coverage. Pass symbols=
# or universe= to widen/target. Deterministic: sorted, so the same run is reproducible.
DEFAULT_MAX_SYMBOLS = 150

# Named universes. liquid_etf_mega = where trading cost is controllable (the whole point of this
# branch): broad+sector ETFs + mega-cap single names, all deeply liquid (penny-ish spreads).
UNIVERSES = {
    "liquid_etf_mega": [
        "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "SMH", "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AMD", "TSLA",
    ],
}


def load_panel(source: str = "alpaca_cache", universe: Optional[str] = None,
               symbols: Optional[Sequence[str]] = None,
               start: Optional[str] = None, end: Optional[str] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (closes, volumes, report). closes/volumes are [dates x symbols], tz-aware UTC.

    source="alpaca_cache" reads data/cache/alpaca_bars_adj. universe= picks a named symbol set
    (see UNIVERSES). Raises FileNotFoundError if the cache is missing or none of the requested
    symbols are present; missing symbols are reported (report['missing']), not fatal.
    """
    if source != "alpaca_cache":
        raise ValueError(f"unknown source {source!r} (only 'alpaca_cache')")

    avail = _rl.available_daily_symbols("adj")
    if not avail:
        raise FileNotFoundError(
            f"adjusted Alpaca cache is empty/missing at {_rl.ADJ_CACHE}. "
            f"Run the fetcher (data_fetch.py, adjustment=ALL) first.")

    if universe is not None:
        if universe not in UNIVERSES:
            raise ValueError(f"unknown universe {universe!r} (have {sorted(UNIVERSES)})")
        symbols = UNIVERSES[universe]
    elif symbols is None:
        symbols = avail[:DEFAULT_MAX_SYMBOLS]
    symbols = sorted(set(symbols))                         # deterministic filtering

    closes, bars, report = _rl.load_panel(
        symbols, start=start, end=end, field="close", require_all=False, source="adj")
    if closes.empty:
        raise FileNotFoundError(
            f"no cached bars for the requested symbols (missing {len(report['missing'])} of "
            f"{len(symbols)}, e.g. {report['missing'][:8]})")

    vol = pd.DataFrame({s: bars[s]["volume"] for s in closes.columns if s in bars}) \
        .reindex(closes.index).ffill()
    return closes, vol, report
