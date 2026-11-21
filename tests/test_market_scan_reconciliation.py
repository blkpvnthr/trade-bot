"""Tests for production subscription reconciliation + candidate eligibility.

Confirmed/warmed/fresh gating, reference-loss -> bounded reconnect, reconnect
recovery not leaving stale candidates eligible, and reconcile telemetry. No
sockets/network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.config import MarketScanConfig
from market_scan.health import read_live_context
from market_scan.ranking import CandidateFeatures
from market_scan.scanner import MarketScanner
from market_scan.stream_client import StreamClient


# ---- StreamClient.request_reconnect ----
def test_request_reconnect_closes_socket_and_counts():
    closed = {"n": 0}

    class _FakeStream:
        def stop(self):
            closed["n"] += 1
    c = StreamClient(api_key="k", secret_key="s", on_trade=lambda *a: None,
                     on_quote=lambda *a: None, stream_factory=lambda *a, **k: _FakeStream())
    c.connect()
    c.request_reconnect()
    assert c.reconnect_requests == 1 and closed["n"] == 1


# ---- scanner reconciliation ----
class _Stream:
    def __init__(self, live=True):
        self._live = live
        self.reconnect_requests = 0

    async def apply_subscriptions(self, *, to_add, to_remove):
        pass

    @property
    def subscribed(self):
        return set()

    def is_live(self, now=None, max_staleness_s=20.0):
        return self._live

    def request_reconnect(self):
        self.reconnect_requests += 1

    def stop(self):
        pass


def _snap(qage=1.0, tage=1.0, ok=True):
    return SimpleNamespace(ok=ok, buy_pressure=0.6, seller_exhaustion=0.4,
                           seller_exhaustion_l1=0.4, cvd_trend=0.3,
                           last_quote_age_s=qage, last_trade_age_s=tage)


def _feat(sym, snap):
    return CandidateFeatures(
        symbol=sym, price=50.0, avg_daily_volume=5_000_000,
        avg_daily_dollar_volume=250_000_000, rel_volume=1.4, spread_pct=0.0004,
        quote_age_s=getattr(snap, "last_quote_age_s", 1.0) if snap else None,
        directional_row={"pdi": 30, "mdi": 10, "adx": 25, "vwap_dist": 0.01,
                         "short_mom": 0.02, "bb_dist": 0.01},
        snapshot=snap, vwap_dist=0.01, bb_mid_dist=0.01)


def _scanner(tmp_path, provider, discovered, stream, cfg=None):
    cfg = cfg or MarketScanConfig(max_subscriptions=30, target_universe_size=10)
    es = SimpleNamespace(
        broker=SimpleNamespace(api_key="k", secret_key="s",
                               get_all_open_positions=lambda: [], list_open_orders=lambda: []),
        config=SimpleNamespace(paper=True))
    ms = MarketScanner(es, cfg, universe_path=tmp_path / "u.txt",
                       context_path=tmp_path / "c.json", stream_client=stream,
                       feature_provider=provider,
                       discovery_fn=lambda **k: SimpleNamespace(symbols=discovered, equities=[],
                                                                etfs=[], error=None, source_counts={}))
    ms._discovered = list(discovered)
    return ms


ALLREF = ["NVDA", "AMD", "SPY", "QQQ", "DIA", "IWM"]


def test_candidate_not_eligible_until_warmed(tmp_path):
    # Fresh data but snapshot.ok=False (not enough history) -> NOT eligible.
    def prov(syms, now):
        return {s: _feat(s, _snap(ok=(s in ("SPY", "QQQ", "DIA", "IWM")))) for s in syms}
    ms = _scanner(tmp_path, prov, ALLREF, _Stream(live=True))
    asyncio.run(ms._rank_tick(1000.0))
    ctx = read_live_context(tmp_path / "c.json")
    # NVDA/AMD fresh but unwarmed -> not eligible
    assert ctx["entry_eligible_count"] == 0
    assert ms._last_reconcile["warmed"] == 4   # only the refs are warmed


def test_reference_loss_forces_bounded_reconnect(tmp_path):
    # All feats stale -> references never confirm -> after max_ref_miss_ticks,
    # force a reconnect and clear accumulated scores.
    def prov(syms, now):
        return {s: _feat(s, _snap(qage=999, tage=999)) for s in syms}
    st = _Stream(live=True)
    cfg = MarketScanConfig(max_ref_miss_ticks=2, target_universe_size=10)
    ms = _scanner(tmp_path, prov, ALLREF, st, cfg=cfg)
    asyncio.run(ms._rank_tick(1000.0))
    assert st.reconnect_requests == 0 and ms._ref_miss_ticks == 1
    asyncio.run(ms._rank_tick(1001.0))
    assert st.reconnect_requests == 1 and ms._ref_miss_ticks == 0   # forced at the bound


def test_refs_delivering_never_forces_reconnect(tmp_path):
    def prov(syms, now):
        return {s: _feat(s, _snap()) for s in syms}
    st = _Stream(live=True)
    ms = _scanner(tmp_path, prov, ALLREF, st)
    for i in range(5):
        asyncio.run(ms._rank_tick(1000.0 + i))
    assert st.reconnect_requests == 0 and ms._ref_miss_ticks == 0
    assert ms._last_reconcile["refs_missing"] == []


def test_reconnect_recovery_clears_stale_candidates(tmp_path):
    # A candidate scored pre-gap must not survive a reconnect recovery.
    st = _Stream(live=True)
    ms = _scanner(tmp_path, lambda syms, now: {}, ALLREF, st)
    ms._scored = {"STALE": (0.9, True, 0.0)}   # a pre-gap eligible candidate
    ms._was_live = False                        # simulate just-recovered stream
    asyncio.run(ms._rank_tick(1000.0))          # live=True & not was_live -> clear
    assert "STALE" not in ms._scored


def test_reconcile_telemetry_published(tmp_path):
    def prov(syms, now):
        return {s: _feat(s, _snap()) for s in syms}
    ms = _scanner(tmp_path, prov, ALLREF, _Stream(live=True))
    asyncio.run(ms._rank_tick(1000.0))
    ctx = read_live_context(tmp_path / "c.json")
    rec = ctx["reconcile"]
    assert set(rec) >= {"requested", "active", "warmed", "missing", "refs_missing"}
    assert rec["active"] >= 4 and rec["refs_missing"] == []
    assert "reconnect_requests" in ctx
