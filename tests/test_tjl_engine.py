"""
End-to-end test for tjl_engine.py with a MOCK gateway (no broker, no network).

Proves the wired execution path: a TJL entry signal becomes a risk-gated BUY routed
through gateway.submit_market_order(..., allow_momentum=True); exits sell on stop;
and the whole engine is inert when TJL_ENABLED is false.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tjl_engine  # noqa: E402
import tjl_signal as T  # noqa: E402

UTC = timezone.utc

# Crafted momentum arrays (stoch_len=3, smoothK=1, smoothD=2) -> momentum OK.
_MOMO_HI = [11.0, 12.0, 13.0, 14.0]
_MOMO_LO = [8.0, 9.0, 10.0, 11.0]
_MOMO_CL_OK = [9.0, 11.0, 11.0, 12.5]


class FakeGateway:
    def __init__(self, price=100.0):
        self.calls = []
        self._price = price

    def submit_market_order(self, symbol, side, qty, reason="", limit_price=0.0,
                            order_type="market", reference_price=0.0, notional=0.0,
                            bypass_trend_gate=False, allow_momentum=False):
        self.calls.append(dict(symbol=symbol, side=side, qty=qty, reason=reason,
                               reference_price=reference_price, allow_momentum=allow_momentum))
        return {"sim": True, "symbol": symbol, "side": side, "qty": qty}

    def _resolve_reference_price(self, symbol):
        return self._price


class _Pos:
    def __init__(self, qty, avg):
        self.qty = qty
        self.avg_entry_price = avg


def _bars_with_crossover():
    import pandas as pd
    base = datetime(2026, 6, 12, 14, 0, tzinfo=UTC)   # Fri 10:00 ET -> RTH
    idx = pd.DatetimeIndex([base - timedelta(minutes=30 - i) for i in range(31)], name="timestamp")
    closes = [100.0] * 30 + [110.0]                    # fast EMA crosses above slow on last bar
    return pd.DataFrame({"open": closes, "high": [c * 1.001 for c in closes],
                         "low": [c * 0.999 for c in closes], "close": closes,
                         "volume": [1000] * 31}, index=idx)


class TJLEngineTest(unittest.TestCase):
    _ENV = {
        "TJL_ENABLED": "true", "TJL_TREND_SMA_LEN": "3",
        "TJL_STOCH_LEN": "3", "TJL_SMOOTH_K": "1", "TJL_SMOOTH_D": "2",
        "TJL_ORDER_DOLLARS": "500",
    }

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._ENV}
        os.environ.update(self._ENV)
        # Deterministic HTF: trend up + momentum OK, no network.
        self._orig_fetch = tjl_engine.fetch_htf_ohlc
        self._orig_load = tjl_engine._state_load
        self._orig_save = tjl_engine._state_save
        self._state = {}
        tjl_engine._state_load = lambda: dict(self._state)
        tjl_engine._state_save = lambda s: self._state.clear() or self._state.update(s)

        def _fake_htf(symbol, tf_min, count, now_utc=None):
            if tf_min >= 60:                      # trend (sma_len=3 -> rising closes)
                return ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
            return (_MOMO_HI, _MOMO_LO, _MOMO_CL_OK)   # momentum OK
        tjl_engine.fetch_htf_ohlc = _fake_htf

    def tearDown(self):
        tjl_engine.fetch_htf_ohlc = self._orig_fetch
        tjl_engine._state_load = self._orig_load
        tjl_engine._state_save = self._orig_save
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_entry_places_risk_gated_momentum_buy(self):
        gw = FakeGateway(price=110.0)
        n = tjl_engine.run_tjl_entries(gw, None, ["AMD"], {"AMD": _bars_with_crossover()}, {}, {})
        self.assertEqual(n, 1)
        self.assertEqual(len(gw.calls), 1)
        call = gw.calls[0]
        self.assertEqual(call["symbol"], "AMD")
        self.assertEqual(call["side"], "buy")
        self.assertTrue(call["allow_momentum"], "TJL entries MUST pass allow_momentum=True")
        self.assertTrue(call["reason"].startswith("tjl_entry"))
        self.assertIn("AMD", self._state)   # position recorded

    def test_no_entry_when_already_in_position(self):
        gw = FakeGateway()
        n = tjl_engine.run_tjl_entries(gw, None, ["AMD"], {"AMD": _bars_with_crossover()},
                                       {}, {"AMD": _Pos(5, 100.0)})
        self.assertEqual(n, 0)
        self.assertEqual(gw.calls, [])

    def test_exit_sells_on_stop(self):
        self._state["AMD"] = {"avg_entry": 100.0, "qty": 5, "opened_at": "x"}
        gw = FakeGateway(price=98.0)            # 98 <= 100*(1-0.015)=98.5 -> stop
        n = tjl_engine.run_tjl_exits(gw, {"AMD": _Pos(5, 100.0)})
        self.assertEqual(n, 1)
        self.assertEqual(gw.calls[0]["side"], "sell")
        self.assertTrue(gw.calls[0]["reason"].startswith("tjl_stop"))
        self.assertNotIn("AMD", self._state)    # state cleared after exit

    def test_disabled_is_inert(self):
        os.environ["TJL_ENABLED"] = "false"
        gw = FakeGateway()
        self.assertEqual(tjl_engine.run_tjl_entries(gw, None, ["AMD"],
                         {"AMD": _bars_with_crossover()}, {}, {}), 0)
        self.assertEqual(tjl_engine.run_tjl_exits(gw, {"AMD": _Pos(5, 100.0)}), 0)
        self.assertEqual(gw.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
