# M3 - Real-Time Terminal Dashboard + Env Config - Design

Date: 2026-11-17
Status: Approved (user directive); spec for implementation
Milestone: M3 (+ M2 env-config finalization) of ROADMAP-2026-11-17-trading-engine-evolution.md
Constraints: paper-only (no live enablement), nonblocking (terminal/logging never delay
execution), additive, preserve all M2 safeguards.

## Part A: Environment-based execution configuration (M2 finalization)

- `STREAMING_ENGINE` env: "1" enables the async StreamingEngine; default "0" keeps the
  existing synchronous execution path. Read in build_default_config -> ExecutorConfig.enable_streaming_engine.
- `TERMINAL_MODE` env: dashboard | full | signals | orders (default dashboard) ->
  ExecutorConfig.terminal_mode.
- Live-trading guard: live (paper=False) requires BOTH explicit paper/live config AND a
  separate authorization env `ALLOW_LIVE_TRADING=1`. Otherwise refuse to start (raise).
  Live stays disabled in this program.
- Startup validation `validate_and_print_execution_mode(config)`: validates the mode
  combination and prints a banner: execution path (SYNC/STREAMING), PAPER/LIVE, terminal
  mode, fractional on/off, streaming subscriptions. Called at the top of main() and
  run_streaming().

## Part B: M3 real-time terminal dashboard (new module terminal_dashboard.py)

Rich is OPTIONAL (not installed here) -> plain-stdout renderer is the default; use rich
only if importable. No hard dependency.

### EventBus (pure/testable core)
- Categories: DATA, SCAN, TREND, TV, SIGNAL, DECISION, RISK, ORDER, ACK, FILL, ERROR.
- `emit(category, payload, critical=False)`: stamps ts_utc + ts_local (+ correlation_id,
  exchange_ts when present) and enqueues onto TWO bounded asyncio.Queues:
  - audit_q (larger): every event; if full, increment dropped_audit and emit an ERROR
    (audit-coverage loss is surfaced, never silent).
  - display_q (smaller, lossy): non-critical high-volume events (DATA/quote) are dropped
    first when full (increment dropped_display); critical events (ORDER/RISK/FILL/ACK/
    DECISION/ERROR) are never silently displaced - on display overflow they still count.
- Overflow counters reported periodically.

### AuditLogger (nonblocking JSONL)
- Consumes audit_q; appends structured JSONL to logs/execution/<DATA_TAG>/streaming_events.jsonl
  via run_in_executor (file write off the loop). Flush/bounded buffer. On write failure,
  surface an ERROR event + keep running (logging failure never breaks execution).

### TerminalRenderer
- Consumes display_q; renders per TERMINAL_MODE:
  - full: every event (incl DATA).
  - signals: SCAN/TREND/TV/SIGNAL/DECISION/RISK/ORDER/ACK/FILL/ERROR.
  - orders: ORDER/ACK/FILL/RISK/ERROR.
  - dashboard: periodic status panel (mode, PAPER/LIVE, broker+data conn, candidate count,
    open orders, positions, buying power, exposure, latency p50/p95/p99, overflow counters)
    + a tail of the recent event feed.
- Color via ANSI (no rich dependency); rich Live used only if importable.
- Runs as its own asyncio task; cannot block execution.

### Shown data (per directive)
Live market data per candidate; eligible + rejected candidates; HTF/intraday momentum
state (when available - M4 fills these); every decision + reason; signal confirmations +
correlation ids; risk approvals/rejections; order submissions + acks; partial/complete
fills, cancels, rejects; exchange + local timestamps + latency; positions, buying power,
exposure, open orders; broker + data connection status.

## Part C: StreamingEngine integration
- StreamingEngine builds EventBus + AuditLogger + TerminalRenderer from config and emits
  events at each pipeline stage (DATA on_trade/quote/bar; SCAN eligibility; SIGNAL +
  correlation id; DECISION buy/exit/skip + reason; RISK approve/reject + reason; ORDER
  submit; ACK/FILL/CANCEL/REJECT from order stream with exchange+local ts + latency).
- run() gathers data._run_forever, trading._run_forever, supervisor, renderer, audit.
- Harden on_order_update status extraction: use enum `.value` when present (real
  OrderStatus.FILLED stringifies "orderstatus.filled"); never parse order.side.
- All emits are sync put_nowait (non-blocking); zero added latency to the submit path.

## Part D: Preserve M2 safeguards (no regression) + real-enum regression tests
Add regression tests using ACTUAL alpaca enums (alpaca.trading.enums OrderSide/OrderStatus):
- client_order_id uniqueness (uuid).
- fill classification by recorded cid_side with order.side = real OrderSide.BUY (proves we
  do not parse the enum); status via real OrderStatus.FILLED/PARTIALLY_FILLED (.value).
- in-flight retained across reaper + late terminal (buy not misclassified).
- fractional entry sizing; non-fractionable fallback; position cap; gross-exposure +
  committed-notional; duplicate-order prevention; stop/target/EOD exits.

## Part E: Integrated M2+M3 paper-validation procedure (doc)
Write docs/superpowers/M2-M3-paper-validation.md: preconditions, enable commands, and a
checklist covering connection+bootstrap, continuous data, signal->submit, ack+fill
reconciliation, stop/target/EOD exits, fractional liquidation, no duplicates, risk-reject
logging, terminal visibility of every event, and latency p50/p95/p99 (<100ms p50 target;
record p95/p99 + investigate outliers). Not complete until real market-session evidence
is captured.

## Testing
Unit (EventBus overflow/priority, mode filtering, config env parsing, live guard),
regression (real-enum safeguards), simulated-streaming (drive handlers with fake events
through the bus and assert ordering/latency recording), all no-network.

## Out of scope
M4 MTF gate (TREND/intraday fields are emitted but populated later), M5 TradingView,
M6/M7. Live trading stays disabled.
