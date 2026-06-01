"""
Offline tests for rl_alpaca_backfill (Phase 1 historical replay).
No network: fetch is monkeypatched / replay functions run on synthetic DataFrames.
Verifies the replay produces labelled events, the writer is offline, and the module
never touches execution (no AlpacaGateway / order submission).
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

import rl_alpaca_backfill as bf
import rl_common as rc


def _daily(n=300, start=100.0):
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    closes, c = [], start
    for i in range(n):
        c = max(1.0, c * (1.0 + (0.001 if i % 3 else -0.0015)))   # gentle drift + dips
        closes.append(c)
    return pd.DataFrame({"open": closes, "high": [x * 1.01 for x in closes],
                         "low": [x * 0.99 for x in closes], "close": closes,
                         "volume": [1_000_000] * n}, index=idx)


class BackfillTests(unittest.TestCase):
    def test_replay_buffett_labels_forward_returns(self):
        ev = bf.replay_buffett("AAPL", _daily(), step=5)
        self.assertTrue(ev)
        e = ev[0]
        for k in ("bal_score", "conviction_score", "composite_dca_score",
                  "price_to_intrinsic_value", "fwd_return_60d", "RSI"):
            self.assertIn(k, e)
        self.assertEqual(e["event_type"], "bal_buy")
        self.assertEqual(e["source"], "alpaca_replay")

    def test_replay_alpha_is_safe_on_thin_data(self):
        # Robust: returns a list (possibly empty) and never raises on small input.
        out = bf.replay_alpha("AAPL", _daily(10))
        self.assertIsInstance(out, list)

    def test_fetch_bars_guarded_without_creds(self):
        # No creds → fetch returns {} (no network, no raise).
        orig = bf._creds
        bf._creds = lambda: ("", "")
        try:
            self.assertEqual(bf.fetch_bars(["AAPL"], None, None), {})
        finally:
            bf._creds = orig

    def test_build_backfill_offline_writes_events(self):
        # Monkeypatch fetch_bars so no network is touched.
        orig = bf.fetch_bars
        bf.fetch_bars = lambda syms, s, e, timeframe="day": ({"AAPL": _daily()} if timeframe == "day" else {})
        try:
            with TemporaryDirectory() as d:
                summary = bf.build_backfill(alpha_symbols=["AAPL"], buffett_symbols=["AAPL"],
                                            out_dir=Path(d))
                self.assertGreater(summary["events"], 0)
                self.assertEqual(summary["orders_submitted"], 0)
                self.assertTrue(summary["offline_only"])
                self.assertTrue((Path(d) / "research_events.jsonl").exists())
        finally:
            bf.fetch_bars = orig

    def test_backfill_no_execution_imports(self):
        src = (ROOT / "rl_alpaca_backfill.py").read_text()
        # Instantiation / order calls forbidden anywhere (prose docstring mentions are OK).
        self.assertNotIn("AlpacaGateway(", src)
        for forbidden in ("submit_market_order", "submit_order", "place_order", "TradingClient"):
            self.assertNotIn(forbidden, src)
        # Import statements must touch only read-only market data, never execution.
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        for imp in imports:
            self.assertNotIn("alpaca.trading", imp)
            self.assertNotIn("AlpacaGateway", imp)
        self.assertIn("alpaca.data.historical", src)   # read-only data import present

    def test_dataset_builder_includes_backfill(self):
        self.assertIsInstance(rc.load_backfill_events(), list)


if __name__ == "__main__":
    unittest.main()
