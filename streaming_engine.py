from __future__ import annotations
import asyncio
import time
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
import trade

try:
    from alpaca.data.live.stock import StockDataStream
    from alpaca.trading.stream import TradingStream
    from alpaca.data.enums import DataFeed
except Exception:  # pragma: no cover - streaming deps optional for unit tests
    StockDataStream = None
    TradingStream = None
    DataFeed = None


class OrderState(Enum):
    SIGNAL_CONFIRMED = "SIGNAL_CONFIRMED"
    RISK_APPROVED = "RISK_APPROVED"
    ORDER_SUBMITTING = "ORDER_SUBMITTING"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    BROKER_ACKNOWLEDGED = "BROKER_ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


_ALLOWED = {
    None: {OrderState.SIGNAL_CONFIRMED},
    OrderState.SIGNAL_CONFIRMED: {OrderState.RISK_APPROVED, OrderState.REJECTED},
    OrderState.RISK_APPROVED: {OrderState.ORDER_SUBMITTING, OrderState.REJECTED},
    OrderState.ORDER_SUBMITTING: {OrderState.ORDER_SUBMITTED, OrderState.REJECTED},
    OrderState.ORDER_SUBMITTED: {OrderState.BROKER_ACKNOWLEDGED, OrderState.REJECTED, OrderState.CANCELED},
    OrderState.BROKER_ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED},
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
}


class OrderStateMachine:
    def __init__(self) -> None:
        self._state: dict[str, OrderState] = {}

    def state(self, cid: str) -> OrderState | None:
        return self._state.get(cid)

    def advance(self, cid: str, to_state: OrderState) -> bool:
        cur = self._state.get(cid)
        if to_state in _ALLOWED.get(cur, set()):
            self._state[cid] = to_state
            return True
        return False


def percentile(values, q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


class LatencyTracker:
    def __init__(self, maxlen: int = 1000) -> None:
        self._samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def record(self, stage: str, value: float) -> None:
        self._samples[stage].append(float(value))

    def summary(self) -> dict:
        out = {}
        for stage, xs in self._samples.items():
            out[stage] = {
                "n": len(xs),
                "p50": percentile(xs, 50),
                "p95": percentile(xs, 95),
                "p99": percentile(xs, 99),
            }
        return out


@dataclass
class SymbolState:
    symbol: str
    today_open: float = 0.0
    last_price: float = 0.0
    last_event_ts: float = 0.0
    held: bool = False
    in_flight: bool = False
    done_today: bool = False
    qty: float = 0.0
    avg_entry: float = 0.0


def decide_streaming_exit(state, config):
    """Return (should_exit, reason) for a held position on the daily-green rules."""
    if not state.held or state.qty <= 0:
        return (False, "")
    return trade.daily_green_exit_decision(
        state.today_open,
        state.last_price,
        state.avg_entry,
        config.daily_green_rr,
        config.daily_green_min_risk_pct,
    )


def is_stale(last_event_ts: float, now: float, threshold_s: float) -> bool:
    return (now - last_event_ts) > threshold_s


def decide_streaming_entry(state, equity, no_reentry, config, executor_config):
    if state.held or state.in_flight or state.done_today:
        return None
    if state.today_open <= 0 or state.last_price <= 0:
        return None
    rows = trade.build_daily_green_entry_rows(
        {state.symbol: (state.today_open, state.last_price)},
        held_symbols=set(),
        no_reentry_symbols=set(no_reentry),
        equity=equity,
        config=config,
        executor_config=executor_config,
    )
    return rows[0] if rows else None


class StreamingEngine:
    """Async, event-driven daily-green engine (M2). Flag-gated; paper-only.

    Composes a ProductionOrderExecutor purely to reuse its broker + risk/kill-switch
    helpers (no threshold duplication). Market events drive entries and exits; all
    blocking calls (REST refresh, broker submit) run off the event loop.
    """

    def __init__(self, config) -> None:
        if StockDataStream is None:
            raise RuntimeError("alpaca-py streaming classes unavailable")
        self.config = config
        self.exec = trade.ProductionOrderExecutor(config)
        self.broker = self.exec.broker
        self.scanner = self.exec.scanner
        self.states: dict[str, SymbolState] = {}
        self.sm = OrderStateMachine()
        self.latency = LatencyTracker()
        self.cid_symbol: dict[str, str] = {}
        self._acct = None
        self._acct_ts = 0.0
        self._no_reentry: set[str] = set()
        self._close_dt = None
        self._loop = None
        self._running = True
        feed = DataFeed.IEX if DataFeed is not None else None
        self.data_stream = StockDataStream(self.broker.api_key, self.broker.secret_key, feed=feed)
        self.trading_stream = TradingStream(self.broker.api_key, self.broker.secret_key, paper=config.paper)
        self.staleness_threshold_s = 30.0
        self.acct_refresh_s = 5.0
        self.latency_report_s = 30.0

    @property
    def symbols(self) -> list[str]:
        return list(self.states.keys())

    async def _in_executor(self, fn, *args):
        return await self._loop.run_in_executor(None, fn, *args)

    async def bootstrap(self) -> None:
        self._loop = asyncio.get_running_loop()
        syms = await self._in_executor(self.scanner._universe)
        bars = await self._in_executor(self.scanner._download_bars, syms)
        opx = trade.compute_today_open_and_price(bars)
        await self._refresh_account_and_positions()
        await self._refresh_no_reentry()
        await self._refresh_close_dt()
        now = time.monotonic()
        for s in syms:
            o, p = opx.get(s, (0.0, 0.0))
            st = self.states.get(s) or SymbolState(symbol=s)
            st.today_open, st.last_price, st.last_event_ts = o, p, now
            self.states[s] = st
        self._apply_positions()
        eq = float(self._acct.equity) if self._acct else 0.0
        print(f"[streaming] bootstrap: {len(self.states)} symbols, equity={eq:.2f}")

    async def _refresh_account_and_positions(self) -> None:
        self._acct = await self._in_executor(self.broker.get_account_snapshot)
        self._positions = await self._in_executor(self.broker.get_all_open_positions)
        self._acct_ts = time.monotonic()
        self._apply_positions()

    def _apply_positions(self) -> None:
        pos_by = {trade.normalize_symbol(p.symbol): p for p in getattr(self, "_positions", []) if p.qty > 0}
        for s, st in self.states.items():
            p = pos_by.get(s)
            if p is not None:
                st.held, st.qty, st.avg_entry = True, float(p.qty), float(p.avg_entry_price)
            else:
                st.held, st.qty, st.avg_entry = False, 0.0, 0.0

    async def _refresh_no_reentry(self) -> None:
        self._no_reentry = await self._in_executor(trade.load_no_reentry_symbols)

    async def _refresh_close_dt(self) -> None:
        try:
            info = await self._in_executor(self.broker.get_clock_info)
            nc = info.get("next_close")
            if nc:
                import pandas as pd
                self._close_dt = pd.Timestamp(nc).tz_convert(trade.ET).to_pydatetime()
        except Exception as exc:
            print(f"[streaming] close-time refresh failed: {exc}")

    async def on_trade(self, ev) -> None:
        t0 = time.monotonic()
        sym = trade.normalize_symbol(getattr(ev, "symbol", ""))
        st = self.states.get(sym)
        if st is None:
            return
        st.last_price = float(getattr(ev, "price", st.last_price) or st.last_price)
        st.last_event_ts = time.monotonic()
        if st.held:
            await self._maybe_exit(st)
        else:
            await self._maybe_enter(st, t0)

    async def on_bar(self, ev) -> None:
        sym = trade.normalize_symbol(getattr(ev, "symbol", ""))
        st = self.states.get(sym)
        if st is None:
            return
        st.last_event_ts = time.monotonic()
        if st.today_open <= 0:
            st.today_open = float(getattr(ev, "open", 0.0) or 0.0)

    async def on_quote(self, ev) -> None:
        sym = trade.normalize_symbol(getattr(ev, "symbol", ""))
        st = self.states.get(sym)
        if st is not None:
            st.last_event_ts = time.monotonic()

    def _risk_ok(self, row, equity: float) -> tuple[bool, str]:
        acct = self._acct
        price = float(row["price"])
        est = float(row["delta_shares"]) * price
        allowed, reason, _ = self.exec._kill_switch_allows_execution(equity)
        if not allowed:
            return False, f"kill_switch:{reason}"
        if float(acct.buying_power) < self.config.min_buying_power:
            return False, "buying_power_below_minimum"
        if float(acct.cash) - est < self.exec._cash_buffer(equity):
            return False, "cash_buffer_violation"
        if est > self.exec._max_order_notional(equity):
            return False, "exceeds_max_order_notional"
        if est < trade.build_account_risk_profile(equity).min_notional_to_trade:
            return False, "below_min_notional"
        return True, ""

    async def _maybe_enter(self, st, t0: float) -> None:
        if self._acct is None or st.in_flight:
            return
        # No new entries inside the EOD flatten window.
        if trade.within_eod_flatten_window(trade.datetime.now(trade.ET), self.config.scanner.daily_green_flatten_before_close_min, self._close_dt):
            return
        equity = max(float(self._acct.equity), 1.0)
        row = decide_streaming_entry(st, equity, self._no_reentry, self.config.scanner, self.config)
        if not row:
            return
        cid = trade.run_id()
        self.sm.advance(cid, OrderState.SIGNAL_CONFIRMED)
        t_sig = time.monotonic()
        ok, why = self._risk_ok(row, equity)
        if not ok:
            self.sm.advance(cid, OrderState.REJECTED)
            print(f"[streaming] REJECT entry {st.symbol} cid={cid}: {why}")
            return
        self.sm.advance(cid, OrderState.RISK_APPROVED)
        t_risk = time.monotonic()
        await self._submit(st, "buy", float(row["delta_shares"]), cid, t0, t_sig, t_risk)

    async def _maybe_exit(self, st) -> None:
        if st.in_flight or st.qty <= 0:
            return
        eod = trade.within_eod_flatten_window(trade.datetime.now(trade.ET), self.config.scanner.daily_green_flatten_before_close_min, self._close_dt)
        if eod:
            reason = "eod_flatten"
        else:
            should_exit, reason = decide_streaming_exit(st, self.config.scanner)
            if not should_exit:
                return
        t0 = time.monotonic()
        cid = trade.run_id()
        self.sm.advance(cid, OrderState.SIGNAL_CONFIRMED)
        self.sm.advance(cid, OrderState.RISK_APPROVED)  # exits always allowed (risk reduction)
        print(f"[streaming] EXIT {st.symbol} qty={st.qty} reason={reason} cid={cid}")
        await self._submit(st, "sell", float(st.qty), cid, t0, t0, t0, exit_reason=reason)

    async def _submit(self, st, side: str, qty: float, cid: str, t0: float, t_sig: float, t_risk: float, exit_reason: str = "") -> None:
        st.in_flight = True
        self.cid_symbol[cid] = st.symbol
        self.sm.advance(cid, OrderState.ORDER_SUBMITTING)
        try:
            if side == "buy":
                await self._in_executor(self.broker.submit_market_buy, st.symbol, qty, cid, self.config.time_in_force)
            else:
                await self._in_executor(self.broker.submit_market_sell, st.symbol, qty, cid, self.config.time_in_force)
            self.sm.advance(cid, OrderState.ORDER_SUBMITTED)
        except Exception as exc:
            self.sm.advance(cid, OrderState.REJECTED)
            st.in_flight = False
            print(f"[streaming] submit FAILED {side} {st.symbol} cid={cid}: {exc}")
            return
        t_sub = time.monotonic()
        self.latency.record("event_to_submit_ms", (t_sub - t0) * 1000.0)
        self.latency.record("signal_to_risk_ms", max(0.0, (t_risk - t_sig) * 1000.0))
        print(f"[streaming] SUBMIT {side} {qty} {st.symbol} cid={cid} e2e={(t_sub - t0) * 1000:.1f}ms {exit_reason}")

    async def on_order_update(self, ev) -> None:
        order = getattr(ev, "order", None)
        cid = str(getattr(order, "client_order_id", "") or "") if order is not None else ""
        status = str(getattr(ev, "event", "") or getattr(order, "status", "") or "").lower()
        mapping = {
            "new": OrderState.BROKER_ACKNOWLEDGED,
            "accepted": OrderState.BROKER_ACKNOWLEDGED,
            "pending_new": OrderState.BROKER_ACKNOWLEDGED,
            "partial_fill": OrderState.PARTIALLY_FILLED,
            "fill": OrderState.FILLED,
            "filled": OrderState.FILLED,
            "canceled": OrderState.CANCELED,
            "rejected": OrderState.REJECTED,
        }
        to = mapping.get(status)
        if cid and to:
            self.sm.advance(cid, to)
        sym = self.cid_symbol.get(cid)
        if sym and to in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED):
            st = self.states.get(sym)
            if st:
                st.in_flight = False
                if to == OrderState.FILLED:
                    # Buy fill -> held + one-round-trip-per-day; exit fill -> position closed.
                    side = str(getattr(order, "side", "")).lower()
                    if side == "buy":
                        st.held, st.done_today = True, True
                    else:
                        st.held, st.qty, st.done_today = False, 0.0, True
                        if not self.config.scanner.daily_green_reenter_after_stop:
                            await self._in_executor(trade.mark_no_reentry_symbol, sym, "streaming_exit")
        print(f"[streaming] order_update cid={cid} status={status} -> {to}")

    async def supervisor(self) -> None:
        last_report = time.monotonic()
        while self._running:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            if now - self._acct_ts >= self.acct_refresh_s:
                try:
                    await self._refresh_account_and_positions()
                    await self._refresh_no_reentry()
                except Exception as exc:
                    print(f"[streaming] refresh failed: {exc}")
            stale = [s for s, st in self.states.items() if is_stale(st.last_event_ts, now, self.staleness_threshold_s)]
            if stale:
                print(f"[streaming] STALE (>{self.staleness_threshold_s:.0f}s no events): {len(stale)} symbols e.g. {stale[:5]}")
            if now - last_report >= self.latency_report_s:
                print(f"[streaming] latency: {self.latency.summary()}")
                last_report = now

    async def run(self) -> None:
        await self.bootstrap()
        syms = self.symbols
        self.data_stream.subscribe_trades(self.on_trade, *syms)
        self.data_stream.subscribe_bars(self.on_bar, *syms)
        self.data_stream.subscribe_quotes(self.on_quote, *syms)
        self.trading_stream.subscribe_trade_updates(self.on_order_update)
        print(f"[streaming] subscribed {len(syms)} symbols; starting streams (paper={self.config.paper})")
        # _run_forever() is the awaitable form; run() is blocking and non-composable.
        await asyncio.gather(
            self.data_stream._run_forever(),
            self.trading_stream._run_forever(),
            self.supervisor(),
        )


def run_streaming(config) -> None:
    if StockDataStream is None:
        raise RuntimeError("alpaca-py streaming unavailable; cannot start StreamingEngine")
    engine = StreamingEngine(config)
    asyncio.run(engine.run())
