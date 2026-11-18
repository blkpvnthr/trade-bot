import asyncio
import json
import terminal_dashboard as td


def test_should_display_modes():
    assert td.should_display("full", "DATA") is True
    assert td.should_display("orders", "DATA") is False
    assert td.should_display("orders", "ORDER") is True
    assert td.should_display("signals", "DATA") is False
    assert td.should_display("signals", "SIGNAL") is True
    assert td.should_display("dashboard", "DATA") is True  # dashboard keeps all for its feed


def test_format_event_contains_category_and_extras():
    line = td.TerminalRenderer.format_event(
        {"category": "ORDER", "ts_local": "10:00:00", "ts_utc": "x", "symbol": "QQQ", "qty": 0.1}
    )
    assert "ORDER" in line and "symbol=QQQ" in line and "qty=0.1" in line


def test_emit_counts_and_payload():
    async def go():
        bus = td.EventBus()
        ev = bus.emit("SIGNAL", {"symbol": "QQQ", "cid": "c1"})
        assert ev["category"] == "SIGNAL" and ev["symbol"] == "QQQ" and "ts_utc" in ev and "ts_local" in ev
        assert bus.counts["SIGNAL"] == 1
    asyncio.run(go())


def test_display_overflow_drops_only_display_not_audit():
    async def go():
        bus = td.EventBus(display_maxlen=2, audit_maxlen=100)
        for _ in range(5):
            bus.emit("DATA", {"px": 1.0})
        assert bus.dropped_display == 3   # only 2 fit the display queue
        assert bus.dropped_audit == 0     # audit queue (100) kept them all
        assert bus.counts["DATA"] == 5
    asyncio.run(go())


def test_audit_logger_writes_jsonl(tmp_path):
    async def go():
        bus = td.EventBus()
        path = tmp_path / "events.jsonl"
        logger = td.AuditLogger(bus, path, batch_timeout=0.05)
        bus.emit("ORDER", {"symbol": "QQQ", "cid": "c1"})
        bus.emit("FILL", {"symbol": "QQQ", "cid": "c1"})
        task = asyncio.create_task(logger.run())
        await asyncio.sleep(0.25)
        logger.stop()
        await asyncio.sleep(0.1)
        task.cancel()
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        cats = [e["category"] for e in lines]
        assert "ORDER" in cats and "FILL" in cats
    asyncio.run(go())
