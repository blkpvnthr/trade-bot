from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import execution_harness


def _order(symbol, side, qty, price, status="filled"):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value=side),
        filled_qty=qty,
        filled_avg_price=price,
        status=status,
    )


def test_parse_expected_since_prefers_snapshot_timestamp():
    expected = {"date": "2026-06-08", "ts": "2026-06-08T07:00:14.681027-04:00"}

    since = execution_harness._parse_expected_since(expected)

    assert since == datetime.fromisoformat(expected["ts"])


def test_collect_expected_fills_ignores_non_universe_symbols():
    orders = [
        _order("SPY", "buy", 2, 748.0),
        _order("QQQ", "buy", 1, 716.02),
        _order("IWM260617C00284000", "buy", 1, 5.37),
        _order("SQQQ260710C00042500", "buy", 3, 3.63),
    ]

    fills, ignored = execution_harness._collect_expected_fills(
        orders,
        {"SPY": 737.55, "QQQ": 705.06},
        {"SPY", "QQQ"},
    )

    assert set(fills) == {"SPY", "QQQ"}
    assert ignored == ["IWM260617C00284000", "SQQQ260710C00042500"]
    assert fills["SPY"]["buy_qty"] == 2
    assert fills["QQQ"]["buy_qty"] == 1
