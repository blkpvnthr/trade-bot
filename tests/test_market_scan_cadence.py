"""Scanner cadence tests (Stage A2/A3): subscription rotation is decoupled from
the ~3s rank/publish tick and only churns the websocket on the 60s rotation
cadence (or an immediate mandatory-set change). This is the regression for the
IEX churn that stalled the live stream.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import universe
from market_scan.config import MarketScanConfig
from market_scan.ranking import CandidateFeatures
from market_scan.scanner import MarketScanner


class _CountingStream:
    def __init__(self):
        self.apply_calls = 0
        self._subscribed: set[str] = set()

    async def apply_subscriptions(self, *, to_add, to_remove):
        self.apply_calls += 1
        self._subscribed.update(to_add)
        self._subscribed.difference_update(to_remove)

    @property
    def subscribed(self):
        return set(self._subscribed)

    async def run(self, *a, **k):
        return

    def stop(self):
        pass


class _Broker:
    def __init__(self, positions=(), pending=()):
        self._pos, self._pending = list(positions), list(pending)
        self.api_key = self.secret_key = "x"

    def set_positions(self, positions):
        self._pos = list(positions)

    def get_all_open_positions(self):
        return [SimpleNamespace(symbol=s) for s in self._pos]

    def list_open_orders(self):
        return [SimpleNamespace(symbol=s) for s in self._pending]


def _snap():
    return SimpleNamespace(last_trade_age_s=1.0, last_quote_age_s=1.0, mid=50.0, spread=0.02)


def _feats(syms, now):
    return {
        s: CandidateFeatures(
            symbol=s, price=50.0, avg_daily_volume=5_000_000,
            avg_daily_dollar_volume=250_000_000, rel_volume=1.5, spread_pct=0.0004,
            quote_age_s=1.0, directional_row={"pdi": 30, "mdi": 10, "adx": 25},
            snapshot=_snap(), vwap_dist=0.01, bb_mid_dist=0.01)
        for s in syms
    }


def _scanner(tmp_path, broker, mono_box, discovered):
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=40,
                           rotation_interval_seconds=60.0)
    s = MarketScanner(
        SimpleNamespace(broker=broker, config=SimpleNamespace(paper=True)), cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_CountingStream(),
        feature_provider=_feats,
        discovery_fn=lambda **k: SimpleNamespace(symbols=list(discovered), error=None),
        mono_fn=lambda: mono_box["t"],
        stats_path_override=tmp_path / "opportunity_stats_paper.json",
    )
    s._discovered = list(discovered)
    return s


def test_rotation_does_not_churn_within_the_interval(tmp_path):
    mono = {"t": 0.0}
    s = _scanner(tmp_path, _Broker(), mono, discovered=["AAA", "BBB", "CCC"])

    async def go():
        await s._maybe_rotate_subscriptions(now=1000.0)      # mono=0 -> initial rotation
        assert s.stream.apply_calls == 1
        # Six ~3s rank ticks within the 60s window: publish each, never churn subs.
        for i in range(1, 7):
            mono["t"] = i * 3.0                               # 3,6,...,18s
            await s._maybe_rotate_subscriptions(now=1000.0 + i)
            await s._rank_and_publish(now=1000.0 + i)
        assert s.stream.apply_calls == 1                     # NO resubscribe churn
        # Cross the rotation interval -> one rotation.
        mono["t"] = 61.0
        await s._maybe_rotate_subscriptions(now=1061.0)
        assert s.stream.apply_calls == 2

    asyncio.run(go())


def test_publish_happens_every_tick_without_rotation(tmp_path):
    mono = {"t": 0.0}
    s = _scanner(tmp_path, _Broker(), mono, discovered=["AAA", "BBB"])

    async def go():
        await s._maybe_rotate_subscriptions(now=1000.0)
        base_calls = s.stream.apply_calls
        for i in range(1, 5):
            mono["t"] = i * 3.0
            await s._rank_and_publish(now=1000.0 + i)        # publish only
        assert s.stream.apply_calls == base_calls            # publishing never churns subs
        # Each publish refreshed the context (updated_at advanced to the last now).
        from market_scan.health import read_live_context
        ctx = read_live_context(tmp_path / "live_context.json")
        assert ctx is not None and ctx["updated_at"] == 1004.0

    asyncio.run(go())


def test_mandatory_change_forces_immediate_rotation(tmp_path):
    mono = {"t": 0.0}
    broker = _Broker()
    s = _scanner(tmp_path, broker, mono, discovered=["AAA", "BBB"])

    async def go():
        await s._maybe_rotate_subscriptions(now=1000.0)      # initial
        assert s.stream.apply_calls == 1
        # A new open position changes the mandatory set -> immediate rotation even
        # though only 2s elapsed (well within the 60s rotation interval).
        broker.set_positions(["NEWPOS"])
        s._refresh_positions_snapshot()                      # refresh the cached snapshot
        mono["t"] = 2.0
        await s._maybe_rotate_subscriptions(now=1002.0)
        assert s.stream.apply_calls == 2
        assert "NEWPOS" in s.stream.subscribed               # position pinned immediately

    asyncio.run(go())
