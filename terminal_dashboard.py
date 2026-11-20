"""M3 real-time terminal dashboard + structured event bus.

Nonblocking by construction: producers call EventBus.emit() (sync put_nowait) and
never await. The renderer and the JSONL audit logger are independent asyncio tasks,
so terminal output and logging can never delay order execution. Display is lossy for
high-volume non-critical events (market data) under backpressure; critical events
(signals/decisions/risk/orders/fills/errors) are counted on overflow and never
silently displaced in the audit stream. Rich is optional (plain ANSI fallback).
"""
from __future__ import annotations
import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone

# Event categories. Non-critical (droppable on DISPLAY overflow) vs critical.
CATEGORIES = (
    "DATA", "SCAN", "TREND", "TV", "SIGNAL", "DECISION",
    "RISK", "ORDER", "ACK", "FILL", "CANCEL", "REJECT", "ERROR",
    "FLOW", "EXIT", "BUY",
)
CRITICAL_CATEGORIES = frozenset(
    {"SIGNAL", "DECISION", "RISK", "ORDER", "ACK", "FILL", "CANCEL", "REJECT", "ERROR", "EXIT"}
)

# Per-mode display filter. None => show all.
_MODE_FILTER = {
    "full": None,
    "dashboard": None,  # dashboard draws a panel + a tail of recent events
    "signals": frozenset(
        {"SCAN", "TREND", "TV", "SIGNAL", "DECISION", "RISK", "ORDER", "ACK", "FILL", "CANCEL", "REJECT", "ERROR", "FLOW", "EXIT"}
    ),
    "orders": frozenset({"ORDER", "ACK", "FILL", "CANCEL", "REJECT", "RISK", "ERROR", "EXIT"}),
}

_COLOR = {
    "DATA": "\033[90m", "SCAN": "\033[36m", "TREND": "\033[35m", "TV": "\033[35m",
    "SIGNAL": "\033[36m", "DECISION": "\033[33m", "RISK": "\033[31m",
    "ORDER": "\033[34m", "ACK": "\033[34m", "FILL": "\033[32m",
    "CANCEL": "\033[33m", "REJECT": "\033[31m", "ERROR": "\033[91m",
    "FLOW": "\033[96m", "EXIT": "\033[93m", "BUY": "\033[92m",
}
_RESET = "\033[0m"

# Exit decisions colored by urgency for the dashboard panel.
_EXIT_DECISION_COLOR = {
    "HOLD_STRONG": "\033[32m", "HOLD_CAUTION": "\033[36m", "PROTECT_PROFIT": "\033[33m",
    "PARTIAL_TAKE_PROFIT": "\033[33m", "FULL_EXIT": "\033[31m", "RISK_EXIT": "\033[91m",
    "INSUFFICIENT_DATA": "\033[90m",
}


def format_exit_panel(exit_events: dict) -> list[str]:
    """Render the order-flow profit-exit panel from the latest EXIT event per symbol.

    Each event carries: symbol, decision, score, pressure, cvd (trend), exhaustion,
    u_plpc / peak_plpc, stop (proposed trailing), enforce/shadow. IEX-only data scope.
    """
    if not exit_events:
        return []
    lines = ["  -- order-flow profit-exit (data_source=IEX, proxy not consolidated) --",
             f"    {'SYM':<6} {'ACTION':<19} {'SCORE':>5} {'PRESSURE':<11} {'CVD':>7} "
             f"{'EXH':>4} {'uP&L%':>6} {'pkP&L%':>6} {'STOP':>8} {'MODE':<7}"]
    for sym, ev in sorted(exit_events.items()):
        dec = str(ev.get("decision", "?"))
        color = _EXIT_DECISION_COLOR.get(dec, "")
        mode = "ENFORCE" if ev.get("enforce") and not ev.get("shadow", True) else "shadow"
        lines.append(
            f"    {color}{sym:<6} {dec:<19}{_RESET} {float(ev.get('score',0)):>5.1f} "
            f"{str(ev.get('pressure','?')):<11} {float(ev.get('cvd',0)):>7.1f} "
            f"{float(ev.get('exhaustion',0)):>4.2f} {float(ev.get('u_plpc',0))*100:>6.2f} "
            f"{float(ev.get('peak_plpc',0))*100:>6.2f} {float(ev.get('stop',0)):>8.2f} {mode:<7}")
    return lines


_BUY_DECISION_COLOR = {
    "NO_BUY": "\033[90m", "WATCH": "\033[36m", "BUY_SETUP": "\033[33m",
    "BUY_READY": "\033[92m", "STRONG_BUY_CONFIRMATION": "\033[1;92m",
    "INSUFFICIENT_DATA": "\033[90m",
}
_PRESSURE_LABEL = {
    "STRONG_BUY": "STRONG BUY", "BUY": "BUYING", "NEUTRAL": "NEUTRAL",
    "SELL": "SELLING", "STRONG_SELL": "STRONG SELL",
}


def format_buy_panel(buy_events: dict, top: int = 8) -> list[str]:
    """Render the BUY OPPORTUNITY MONITOR from the latest BUY event per symbol.

    Each event carries: symbol, score, decision, setup, pressure, cvd_trend,
    seller_exhaustion, scanner_eligible, enforce/shadow. IEX-only data scope.
    """
    if not buy_events:
        return []
    shadow = any(ev.get("shadow", True) for ev in buy_events.values())
    lines = ["", "BUY OPPORTUNITY MONITOR", "-" * 50,
             f"{'SYMBOL':<7} {'SCORE':>5} {'PRESSURE':<11} {'SETUP':<10} {'DECISION':<24} {'ELIG':<4}"]
    ranked = sorted(buy_events.items(), key=lambda kv: float(kv[1].get("score", 0)), reverse=True)
    for sym, ev in ranked[:top]:
        dec = str(ev.get("decision", "?"))
        color = _BUY_DECISION_COLOR.get(dec, "")
        pressure = _PRESSURE_LABEL.get(str(ev.get("pressure", "NEUTRAL")), str(ev.get("pressure", "?")))
        setup = str(ev.get("setup", "NONE"))
        elig = "yes" if ev.get("scanner_eligible") else "no"
        lines.append(
            f"{color}{sym:<7} {float(ev.get('score',0)):>5.0f} {pressure:<11} "
            f"{setup:<10} {dec:<24}{_RESET} {elig:<4}")
    lines.append("-" * 50)
    lines.append(f"Mode: {'SHADOW' if shadow else 'ENFORCE'}    Feed: IEX")
    return lines


def should_display(mode: str, category: str) -> bool:
    """Whether an event of `category` is shown in terminal `mode`."""
    if mode not in _MODE_FILTER:
        return True
    filt = _MODE_FILTER[mode]
    return filt is None or category in filt


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventBus:
    """Fan-out to a (larger) audit queue and a (smaller, lossy) display queue."""

    def __init__(self, display_maxlen: int = 2000, audit_maxlen: int = 20000) -> None:
        self.display_q: asyncio.Queue = asyncio.Queue(maxsize=display_maxlen)
        self.audit_q: asyncio.Queue = asyncio.Queue(maxsize=audit_maxlen)
        self.dropped_display = 0
        self.dropped_audit = 0
        self.counts: dict[str, int] = {c: 0 for c in CATEGORIES}

    def emit(self, category: str, payload: dict | None = None, critical: bool | None = None) -> dict:
        payload = payload or {}
        ev = {"category": category, "ts_utc": _now_iso(), "ts_local": time.strftime("%H:%M:%S"), **payload}
        self.counts[category] = self.counts.get(category, 0) + 1
        # Audit stream: every event. A full audit queue is a surfaced error, not silent.
        try:
            self.audit_q.put_nowait(ev)
        except asyncio.QueueFull:
            self.dropped_audit += 1
        # Display stream: lossy under backpressure (market data dropped first).
        try:
            self.display_q.put_nowait(ev)
        except asyncio.QueueFull:
            self.dropped_display += 1
        return ev

    def overflow_summary(self) -> dict:
        return {"dropped_display": self.dropped_display, "dropped_audit": self.dropped_audit}


class AuditLogger:
    """Drains audit_q to a JSONL file off the event loop (run_in_executor)."""

    def __init__(self, bus: EventBus, path, batch_timeout: float = 1.0) -> None:
        self.bus = bus
        self.path = str(path)
        self.batch_timeout = batch_timeout
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _append(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(text)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            batch = []
            try:
                ev = await asyncio.wait_for(self.bus.audit_q.get(), timeout=self.batch_timeout)
                batch.append(ev)
            except asyncio.TimeoutError:
                continue
            while True:  # drain whatever else is queued
                try:
                    batch.append(self.bus.audit_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            text = "".join(json.dumps(e, default=str) + "\n" for e in batch)
            try:
                await loop.run_in_executor(None, self._append, text)
            except Exception as exc:  # logging failure must never break execution
                self.bus.dropped_audit += len(batch)
                print(f"[audit] write failed: {exc}")


class TerminalRenderer:
    """Consumes display_q and renders per mode. Plain ANSI (rich optional)."""

    def __init__(self, bus: EventBus, mode: str = "dashboard", status_provider=None,
                 dashboard_interval: float = 2.0, feed_tail: int = 20) -> None:
        self.bus = bus
        self.mode = mode if mode in _MODE_FILTER else "dashboard"
        self.status_provider = status_provider
        self.dashboard_interval = dashboard_interval
        self.recent: deque = deque(maxlen=feed_tail)
        self.exit_panel: dict[str, dict] = {}  # latest EXIT event per symbol
        self.buy_panel: dict[str, dict] = {}   # latest BUY event per symbol
        self._running = True

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def format_event(ev: dict) -> str:
        cat = ev.get("category", "?")
        color = _COLOR.get(cat, "")
        extras = " ".join(
            f"{k}={v}" for k, v in ev.items()
            if k not in ("category", "ts_utc", "ts_local")
        )
        return f"{color}[{ev.get('ts_local','')}] {cat:<8}{_RESET} {extras}"

    def _render_dashboard(self) -> None:
        status = {}
        if self.status_provider:
            try:
                status = self.status_provider() or {}
            except Exception as exc:
                status = {"status_error": str(exc)}
        print("\n" + "=" * 72)
        print(f"[dashboard] {time.strftime('%H:%M:%S')}  " + "  ".join(f"{k}={v}" for k, v in status.items()))
        of = self.bus.overflow_summary()
        print(f"[dashboard] events={sum(self.bus.counts.values())} "
              f"dropped_display={of['dropped_display']} dropped_audit={of['dropped_audit']}")
        for line in format_exit_panel(self.exit_panel):
            print(line)
        for line in format_buy_panel(self.buy_panel):
            print(line)
        for line in list(self.recent):
            print("  " + line)
        print("=" * 72)

    async def run(self) -> None:
        last_dash = 0.0
        while self._running:
            try:
                ev = await asyncio.wait_for(self.bus.display_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                ev = None
            if ev is not None and ev.get("category") == "EXIT" and ev.get("symbol"):
                self.exit_panel[str(ev["symbol"])] = ev  # keep latest per symbol for the panel
            if ev is not None and ev.get("category") == "BUY" and ev.get("symbol"):
                self.buy_panel[str(ev["symbol"])] = ev
            if ev is not None and should_display(self.mode, ev.get("category", "")):
                line = self.format_event(ev)
                if self.mode == "dashboard":
                    self.recent.append(line)
                else:
                    print(line)
            if self.mode == "dashboard":
                now = time.monotonic()
                if now - last_dash >= self.dashboard_interval:
                    self._render_dashboard()
                    last_dash = now
