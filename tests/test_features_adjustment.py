"""Regression test for the corporate-action-handling bug in features.py:
Alpaca bar requests must use split+dividend-adjusted prices (Adjustment.ALL),
not the raw/unadjusted default, or multi-year lookbacks fabricate ~90%+
single-day "crashes" on real stock-split dates.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import features as feat
from alpaca.data.enums import Adjustment, DataFeed


def test_download_chunk_requests_full_adjustment():
    fake_client = MagicMock()
    fake_bars = MagicMock()
    fake_bars.df = pd.DataFrame()
    fake_client.get_stock_bars.return_value = fake_bars

    feat._download_chunk(
        client=fake_client,
        chunk=["AAPL"],
        start=datetime(2022, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc),
        feed=DataFeed.IEX,
    )

    assert fake_client.get_stock_bars.call_count == 1
    sent_request = fake_client.get_stock_bars.call_args[0][0]
    assert sent_request.adjustment == Adjustment.ALL
