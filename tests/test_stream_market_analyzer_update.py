"""Tests for mid-flight subscription updates in market_scan.stream_market_analyzer.

Pins the new update_symbols capability: incremental subscribe/unsubscribe on the
live socket (no teardown), REFERENCES never dropped, IEX budget never exceeded,
added symbols warmed up / removed symbols pruned. No network, no real websocket.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_scan.stream_market_analyzer as sm

REFS = ["SPY", "QQQ", "DIA", "IWM"]


# ---- pure delta logic ----
def test_delta_no_change_is_noop():
    cur = REFS + ["AAA", "BBB"]
    desired, add, rem = sm.compute_subscription_delta(cur, ["AAA", "BBB"], REFS, budget=30)
    assert add == [] and rem == []
    assert set(desired) == set(cur)


def test_delta_add_and_remove():
    cur = REFS + ["AAA", "BBB"]
    desired, add, rem = sm.compute_subscription_delta(cur, ["BBB", "CCC"], REFS, budget=30)
    assert add == ["CCC"]
    assert rem == ["AAA"]
    assert set(desired) == set(REFS + ["BBB", "CCC"])


def test_delta_never_removes_references():
    # Even if new_pool omits everything, REFS stay and are never in to_remove.
    cur = REFS + ["AAA"]
    desired, add, rem = sm.compute_subscription_delta(cur, [], REFS, budget=30)
    assert set(REFS) <= set(desired)
    assert not any(r in rem for r in REFS)
    assert rem == ["AAA"]


def test_delta_never_exceeds_budget():
    big_pool = [f"C{i}" for i in range(50)]
    desired, add, rem = sm.compute_subscription_delta(REFS, big_pool, REFS, budget=10)
    assert len(desired) == 10               # strict cap
    assert set(REFS) <= set(desired)        # refs prioritized
    assert len(add) == 10 - len(REFS)


def test_delta_references_prioritized_in_desired():
    desired, _a, _r = sm.compute_subscription_delta([], ["Z", "Y"], REFS, budget=30)
    assert desired[: len(REFS)] == REFS


# ---- async update_symbols on a mock socket ----
class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(json.loads(msg))


def _run(coro):
    return asyncio.run(coro)


def test_update_symbols_sends_unsub_before_sub_and_warms_states(monkeypatch):
    # bootstrap must not hit the network: fake it.
    monkeypatch.setattr(sm, "bootstrap", lambda api, syms, session: {s: sm.State() for s in syms})
    ws = _FakeWS()
    states = {s: sm.State() for s in REFS + ["AAA", "BBB"]}
    new_symbols, new_pool = _run(sm.update_symbols(
        ws, api=object(), states=states, session=0.0,
        current=REFS + ["AAA", "BBB"], new_pool=["BBB", "CCC"], refs=REFS, budget=30))
    actions = [m["action"] for m in ws.sent]
    assert actions == ["unsubscribe", "subscribe"]            # unsub first (budget-safe)
    assert ws.sent[0]["trades"] == ["AAA"]                    # removed
    assert ws.sent[1]["trades"] == ["CCC"]                    # added
    assert "AAA" not in states and "CCC" in states           # pruned / warmed
    assert set(new_symbols) == set(REFS + ["BBB", "CCC"])
    assert set(new_pool) == {"BBB", "CCC"}


def test_update_symbols_noop_sends_nothing(monkeypatch):
    monkeypatch.setattr(sm, "bootstrap", lambda api, syms, session: {s: sm.State() for s in syms})
    ws = _FakeWS()
    states = {s: sm.State() for s in REFS + ["AAA"]}
    _run(sm.update_symbols(ws, api=object(), states=states, session=0.0,
                           current=REFS + ["AAA"], new_pool=["AAA"], refs=REFS, budget=30))
    assert ws.sent == []                                      # identical pool -> no frames


def test_update_symbols_never_unsubscribes_references(monkeypatch):
    monkeypatch.setattr(sm, "bootstrap", lambda api, syms, session: {s: sm.State() for s in syms})
    ws = _FakeWS()
    states = {s: sm.State() for s in REFS + ["AAA"]}
    _run(sm.update_symbols(ws, api=object(), states=states, session=0.0,
                           current=REFS + ["AAA"], new_pool=["ZZZ"], refs=REFS, budget=30))
    unsub = next((m for m in ws.sent if m["action"] == "unsubscribe"), {"trades": []})
    assert not any(r in unsub["trades"] for r in REFS)       # refs never dropped
    assert all(r in states for r in REFS)                    # refs' states retained
