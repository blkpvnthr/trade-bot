"""Stage B integration: the executor's protective-exit selection dispatches to the
dynamic monitor ONLY when enabled; otherwise the fixed -0.5% behavior is preserved
exactly. Conservative fallback to the fixed rule when ATR data is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trade


def _exec(dynamic, feats=None):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = SimpleNamespace(scanner=SimpleNamespace(stop_loss_pct=-0.025))
    e._latest_feature_by_symbol = lambda: (feats or {})
    e._dynamic_loss_enabled = lambda: dynamic      # override env read for the test
    return e


def _pos(sym, plpc):
    return SimpleNamespace(symbol=sym, unrealized_plpc=plpc, qty=10)


def test_fixed_behavior_preserved_when_dynamic_off():
    e = _exec(False)
    out = e._breached_positions([_pos("A", -0.006), _pos("B", -0.001)], -0.005)
    assert set(out) == {"A"}            # identical to the legacy fixed -0.5% rule


def test_dynamic_falls_back_to_fixed_without_atr():
    e = _exec(True, feats={})           # no ATR available -> conservative fallback
    out = e._breached_positions([_pos("A", -0.006)], -0.005)
    assert "A" in out


def test_dynamic_holds_small_loss_within_volatility_budget():
    # atr_pct 0.01 -> adaptive soft stop 0.015; a -0.6% loss that the FIXED rule
    # would liquidate is HELD by the dynamic monitor (within the vol budget).
    e = _exec(True, feats={"A": {"atr_pct": 0.01, "5m_above_vwap": True,
                                 "short_mom": 0.01}})
    out = e._breached_positions([_pos("A", -0.006)], -0.005)
    assert out == []


def test_dynamic_hard_stop_still_exits_immediately():
    e = _exec(True, feats={"A": {"atr_pct": 0.01, "5m_above_vwap": True}})
    out = e._breached_positions([_pos("A", -0.03)], -0.005)   # beyond 2.5% hard stop
    assert "A" in out
