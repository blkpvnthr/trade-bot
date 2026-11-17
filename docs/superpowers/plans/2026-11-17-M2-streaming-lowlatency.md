# M2 Streaming + Low-Latency Execution - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Add a flag-gated async StreamingEngine that submits daily-green entries on live IEX ticks with sub-100ms internal latency, preserving the sync engine + all risk controls. Paper-only; no live enablement.

**Architecture:** New `streaming_engine.py`. Pure, unit-tested components (state machine, latency tracker, entry decision, staleness) + an async I/O shell that composes a `ProductionOrderExecutor` (reusing its broker + risk + kill-switch) and reuses `build_daily_green_entry_rows` for sizing. `main()` branches on `ExecutorConfig.enable_streaming_engine` (default OFF).

**Tech Stack:** Python 3.11 asyncio, alpaca-py 0.43.2 (StockDataStream/TradingStream), pandas/numpy, pytest.

## Global Constraints
- All new code in `streaming_engine.py`; only `main()` + the config flag change in `trade.py`.
- Flag `enable_streaming_engine` default **False**; flag OFF => `main()` runs the existing sync `run()` unchanged.
- Reuse (do not duplicate): `build_daily_green_entry_rows`, `build_account_risk_profile`, `AlpacaBroker`, and `ProductionOrderExecutor` risk/kill-switch methods.
- Paper-only. No live flag flips. Preserve paper/live separation.
- Nonblocking: blocking calls (REST, broker.submit) run via `loop.run_in_executor`.
- ASCII only. Tests must run with no network (pure components) via `pytest tests/test_streaming_engine.py -v`.
- Latency target: event_recv -> submit_initiated p50 < 100 ms internal (monotonic clock).

---

### Task 1: Order state machine

**Files:** Create `streaming_engine.py`; Test `tests/test_streaming_engine.py`
**Produces:** `OrderState` (enum) and `OrderStateMachine` with `advance(cid, to_state)` returning bool (False on illegal transition), `state(cid)`.

- [ ] **Step 1: Write failing test**
```python
import streaming_engine as se

def test_state_machine_legal_and_illegal():
    m = se.OrderStateMachine()
    cid = "c1"
    assert m.advance(cid, se.OrderState.SIGNAL_CONFIRMED) is True
    assert m.advance(cid, se.OrderState.RISK_APPROVED) is True
    assert m.advance(cid, se.OrderState.ORDER_SUBMITTING) is True
    assert m.advance(cid, se.OrderState.ORDER_SUBMITTED) is True
    assert m.advance(cid, se.OrderState.BROKER_ACKNOWLEDGED) is True
    assert m.advance(cid, se.OrderState.FILLED) is True
    # illegal: cannot jump back to SUBMITTING from FILLED
    assert m.advance(cid, se.OrderState.ORDER_SUBMITTING) is False
    assert m.state(cid) == se.OrderState.FILLED

def test_state_machine_reject_path():
    m = se.OrderStateMachine()
    m.advance("c2", se.OrderState.SIGNAL_CONFIRMED)
    m.advance("c2", se.OrderState.RISK_APPROVED)
    assert m.advance("c2", se.OrderState.REJECTED) is True
```

- [ ] **Step 2: Run -> FAIL** `pytest tests/test_streaming_engine.py -v` (ModuleNotFound / no attr).

- [ ] **Step 3: Implement**
```python
from __future__ import annotations
from enum import Enum


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
```

- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** `git add streaming_engine.py tests/test_streaming_engine.py && git commit -m "feat(streaming): M2 order state machine"`

---

### Task 2: Latency tracker

**Files:** Modify `streaming_engine.py`; Test same file
**Produces:** `LatencyTracker(maxlen=1000)` with `record(stage, seconds)` and `summary()` -> `{stage: {p50,p95,p99,n}}`; `percentile(sorted_list, q)` helper.

- [ ] **Step 1: Failing test**
```python
def test_percentile_and_summary():
    assert se.percentile([1,2,3,4,5], 50) == 3
    assert se.percentile([], 50) == 0.0
    t = se.LatencyTracker()
    for v in [10,20,30,40,50]:
        t.record("e2e_ms", v)
    s = t.summary()["e2e_ms"]
    assert s["n"] == 5 and s["p50"] == 30
```

- [ ] **Step 2: Run -> FAIL**

- [ ] **Step 3: Implement** (append)
```python
from collections import defaultdict, deque


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
```

- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(streaming): M2 latency tracker"`

---

### Task 3: Symbol state, entry decision, staleness, dedupe

**Files:** Modify `streaming_engine.py`; Test same file
**Interfaces:** Consumes `trade.build_daily_green_entry_rows`, `trade.ScannerConfig`, `trade.ExecutorConfig`.
**Produces:**
- `@dataclass SymbolState(symbol, today_open=0.0, last_price=0.0, last_event_ts=0.0, held=False, in_flight=False, done_today=False)`
- `is_stale(last_event_ts, now, threshold_s) -> bool`
- `decide_streaming_entry(state, equity, no_reentry, config, executor_config) -> dict | None`: returns None if `state.held or state.in_flight or state.done_today or state.today_open<=0 or state.last_price<=0`; else calls `trade.build_daily_green_entry_rows({state.symbol:(state.today_open,state.last_price)}, held_symbols=set(), no_reentry_symbols=no_reentry, equity=equity, config=config, executor_config=executor_config)` and returns the first row or None.

- [ ] **Step 1: Failing tests**
```python
import trade

def _st(**kw):
    d = dict(symbol="QQQ", today_open=660.0, last_price=661.0, last_event_ts=100.0, held=False, in_flight=False, done_today=False)
    d.update(kw); return se.SymbolState(**d)

def test_is_stale():
    assert se.is_stale(100.0, 106.0, 5.0) is True
    assert se.is_stale(100.0, 104.0, 5.0) is False

def test_decide_entry_green_and_blocks():
    cfg = trade.ScannerConfig(); ecfg = trade.ExecutorConfig()
    row = se.decide_streaming_entry(_st(), 824.74, set(), cfg, ecfg)
    assert row is not None and row["symbol"] == "QQQ" and row["side"] == "buy" and 0 < row["delta_shares"] < 1
    assert se.decide_streaming_entry(_st(last_price=659.0), 824.74, set(), cfg, ecfg) is None   # red vs open
    assert se.decide_streaming_entry(_st(held=True), 824.74, set(), cfg, ecfg) is None
    assert se.decide_streaming_entry(_st(in_flight=True), 824.74, set(), cfg, ecfg) is None
    assert se.decide_streaming_entry(_st(done_today=True), 824.74, set(), cfg, ecfg) is None
    assert se.decide_streaming_entry(_st(), 824.74, {"QQQ"}, cfg, ecfg) is None   # no-reentry
```

- [ ] **Step 2: Run -> FAIL**

- [ ] **Step 3: Implement** (append; `import trade`, `from dataclasses import dataclass` at top as needed)
```python
@dataclass
class SymbolState:
    symbol: str
    today_open: float = 0.0
    last_price: float = 0.0
    last_event_ts: float = 0.0
    held: bool = False
    in_flight: bool = False
    done_today: bool = False


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
```

- [ ] **Step 4: Run -> PASS**
- [ ] **Step 5: Commit** `git commit -m "feat(streaming): M2 symbol state, entry decision, staleness"`

---

### Task 4: StreamingEngine async shell + config flag + main() branch

**Files:** Modify `streaming_engine.py` (async class) and `trade.py` (flag + main branch).
**Interfaces:** Consumes Tasks 1-3; composes `trade.ProductionOrderExecutor` for broker + `_cash_buffer`/`_max_order_notional`/`_risk_profile`/`_kill_switch_allows_execution`; `trade.build_account_risk_profile`.

Note: the websocket I/O and broker submit are integration-tested on paper (Task 5), not unit-tested. This task is structural; keep the pipeline math delegating to Tasks 1-3 so logic stays covered by their unit tests.

- [ ] **Step 1: Config flag + main() branch (trade.py)**
Add to `ExecutorConfig`: `enable_streaming_engine: bool = False`. In `main()`:
```python
def main() -> None:
    config = build_default_config()
    if config.enable_streaming_engine:
        import streaming_engine
        streaming_engine.run_streaming(config)
        return
    threading.Thread(target=launch_intraday_dataset_builder, daemon=True).start()
    executor = ProductionOrderExecutor(config)
    executor.run()
```

- [ ] **Step 2: Implement `StreamingEngine` + `run_streaming(config)`** in streaming_engine.py:
  - `__init__(config)`: build `ProductionOrderExecutor(config)` (gives `.broker`, risk methods); `self.states: dict[str, SymbolState]`; `OrderStateMachine`; `LatencyTracker`; `StockDataStream(api_key, secret_key, feed=iex)`; `TradingStream(api_key, secret_key, paper=config.paper)`.
  - `async def bootstrap()`: `await loop.run_in_executor` to fetch universe, today opens (daily bars), account, positions; seed `self.states` (today_open, held).
  - `async def on_trade(trade_ev)`: `t0=monotonic`; update state.last_price/last_event_ts; `decide_streaming_entry`; if row: `cid=run_id`; state machine SIGNAL_CONFIRMED; lean risk (kill switch via executor, cash buffer, min_notional, max_order, gross, buying_power using executor helpers + current account snapshot cached/refreshed off hot path); RISK_APPROVED; set `state.in_flight=True`; ORDER_SUBMITTING; `await loop.run_in_executor(None, broker.submit_market_buy, symbol, qty, cid, DAY)`; ORDER_SUBMITTED; record latency stages (t0->submit_initiated).
  - `async def on_quote`/`on_bar`: update spread / bar state (for M3/M4) + last_event_ts.
  - `async def on_order_update(ev)`: map Alpaca order status -> state machine (ACK/PARTIAL/FILLED/CANCELED/REJECTED); on terminal fill/reject clear `in_flight`, set `done_today` on fill.
  - `async def supervisor()`: periodic staleness check (`is_stale`) + reconnect with backoff; periodic latency summary print.
  - `run_streaming(config)`: `asyncio.run(StreamingEngine(config).run())` where `run()` bootstraps, subscribes handlers, and gathers the stream tasks + supervisor.
  - Dedupe: skip `decide_streaming_entry` submit when `state.in_flight`; idempotent `client_order_id=cid`.
  - All blocking calls via `run_in_executor`.

- [ ] **Step 3: Verify import + flag-off parity**
`python -c "import trade, streaming_engine; print('ok')"`; confirm `build_default_config().enable_streaming_engine is False` and main() unchanged path when off.
- [ ] **Step 4: Commit** `git commit -m "feat(streaming): M2 async StreamingEngine + flag-gated entrypoint"`

---

### Task 5: Paper integration validation + latency + review

**Files:** none (validation) unless fixes needed.
- [ ] Run pure suite: `pytest tests/test_streaming_engine.py -v` -> all pass.
- [ ] Market-hours paper run with `enable_streaming_engine=True`: confirm stream connects; a green-vs-open crossing submits ONE entry (correlation id logged); order stream drives state machine to FILLED; latency summary prints with event->submit p50 < 100 ms (report p50/p95/p99).
- [ ] Simulate a disconnect; confirm staleness + reconnect.
- [ ] Confirm no duplicate orders under repeated ticks (dedupe) and risk gates enforced (force kill-switch / tiny buying power).
- [ ] Focused code review (opus) on the whole diff; fix Critical/Important; re-verify.

## Self-Review
Covered: streaming (Task 4), low-latency pipeline + metrics (Task 2/4), state machine (Task 1), entry decision reuse (Task 3), staleness/reconnect (Task 4), flag-off parity (Task 4), paper-only + risk reuse (constraints). Integration/latency validated in Task 5. Terminal/MTF/audit are later milestones.
