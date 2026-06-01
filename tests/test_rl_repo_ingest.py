"""Offline tests for Source-3 repo dataset ingestion in rl_alpaca_backfill.
Synthetic CSVs only — no network, no big-file reads, no orders. Verifies the quality gate
(invalid entry/stop skipped), realized_R/peak_R labels, and daily-bars → Buffett replay.
"""

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

import rl_alpaca_backfill as bf


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


class TradeReplayIngestTests(unittest.TestCase):
    HEADER = ["timestamp", "symbol", "setup_type", "entry_price", "stop_price", "target_price",
              "sweep_low", "reclaim_price", "volume_spike_ratio", "support_distance_pct",
              "reclaim_vwap", "reclaim_ema9", "reclaim_ema20", "mfe_pct", "mae_pct", "outcome"]

    def test_quality_gate_and_labels(self):
        rows = [
            ["2026-05-01T14:00:00Z", "SPY", "dip_reclaim", 100, 99, 103, 98, 100, 2.0, 0.5, 100, 100, 100, 3.0, 0.5, "win"],
            ["2026-05-01T14:05:00Z", "SPY", "dip_reclaim", 100, 99, 103, 98, 100, 1.5, 0.5, 100, 100, 100, 0.5, 1.2, "loss"],
            # invalid: stop >= entry → must be skipped (not fabricated)
            ["2026-05-01T14:10:00Z", "SPY", "dip_reclaim", 100, 101, 103, 98, 100, 1.0, 0.5, 100, 100, 100, 0.2, 0.3, "win"],
            # invalid: unknown outcome → skipped
            ["2026-05-01T14:15:00Z", "SPY", "dip_reclaim", 100, 99, 103, 98, 100, 1.0, 0.5, 100, 100, 100, 0.2, 0.3, "???"],
        ]
        with TemporaryDirectory() as d:
            p = Path(d) / "replay.csv"
            _write_csv(p, self.HEADER, rows)
            ev = bf.ingest_trade_replay_csv(p)
        # 2 valid rows → 2 (entry+exit) each = 4 events; 2 invalid skipped
        self.assertEqual(len(ev), 4)
        exits = [e for e in ev if e["event_type"] == "alpha_exit"]
        rmap = {round(e["realized_R"], 2) for e in exits}
        self.assertIn(3.0, rmap)    # win → planned_R = (103-100)/(100-99) = 3
        self.assertIn(-1.0, rmap)   # loss → -1
        for e in exits:
            self.assertGreaterEqual(e["peak_R"], e["realized_R"])
            self.assertEqual(e["source"], "repo_trade_replay")

    def test_missing_file_safe(self):
        self.assertEqual(bf.ingest_trade_replay_csv(ROOT / "does_not_exist.csv"), [])


class DailyBarsIngestTests(unittest.TestCase):
    def test_daily_bars_to_buffett_events(self):
        import pandas as pd
        idx = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
        frames = []
        for sym in ("AAA", "BBB"):
            base = 100.0
            closes = [base * (1.0 + 0.0005 * i) for i in range(300)]
            frames.append(pd.DataFrame({"symbol": sym, "timestamp": idx,
                "open": closes, "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes], "close": closes,
                "volume": [1_000_000] * 300}))
        with TemporaryDirectory() as d:
            p = Path(d) / "daily.csv"
            pd.concat(frames).to_csv(p, index=False)
            ev = bf.ingest_daily_bars_csv(p, max_symbols=5, step=10)
        self.assertTrue(ev)
        self.assertTrue(all(e["event_type"] == "bal_buy" for e in ev))
        self.assertIn("fwd_return_60d", ev[0])

    def test_missing_file_safe(self):
        self.assertEqual(bf.ingest_daily_bars_csv(ROOT / "nope.csv"), [])


class RepoBackfillSafetyTests(unittest.TestCase):
    def test_no_execution_in_ingesters(self):
        # Scan IMPORT statements / instantiation only — the safety docstring legitimately
        # names AlpacaGateway/alpaca.trading in prose to state it never imports them.
        src = (ROOT / "rl_alpaca_backfill.py").read_text()
        self.assertNotIn("AlpacaGateway(", src)
        for forbidden in ("submit_market_order", "submit_order", "place_order", "TradingClient"):
            self.assertNotIn(forbidden, src)
        for line in src.splitlines():
            if line.strip().startswith(("import ", "from ")):
                self.assertNotIn("alpaca.trading", line)
                self.assertNotIn("AlpacaGateway", line)
        self.assertTrue(hasattr(bf, "build_from_repo"))
        self.assertTrue(hasattr(bf, "ingest_trade_replay_csv"))
        self.assertTrue(hasattr(bf, "ingest_daily_bars_csv"))


if __name__ == "__main__":
    unittest.main()
