# M2 - Streaming Market Data + Low-Latency Execution - Design

Date: 2026-11-17
Status: Approved (architecture); spec for plan + SDD
Milestone: M2 of docs/superpowers/ROADMAP-2026-11-17-trading-engine-evolution.md
Risk: HIGH (sync -> async on a live-capable engine). Additive, flag-gated,
paper-only, sync loop stays as fallback. No live enablement.

## Objective

Add an event-driven streaming path that evaluates candidates on live market data
and submits qualifying daily-green entries with sub-100ms internal latency, while
preserving the existing synchronous engine, all risk controls, and paper/live
separation.

## User-approved architecture
- Coexistence: flag-switched async entrypoint. New flag
  `ExecutorConfig.enable_streaming_engine` (default OFF). `main()` runs the async
  StreamingEngine when ON, the existing sync `run()` loop when OFF (fallback).
- Streams (IEX; SIP is entitlement-gated): StockDataStream -> trades + 1-min bars
  + quotes; TradingStream -> order updates (acks/fills).
- Signal: the EXISTING daily-green signal (price > today's open). The full MTF
  gate is M4; the Rich terminal is M3; full JSONL audit is M6. M2 lays the
  correlation-ID / order-state-machine / latency foundation they build on.

## Module layout (additive)
New file `streaming_engine.py`. trade.py's sync engine is untouched except the
`main()` branch + the config flag.

Pure, unit-testable components (no async, no network):
- `OrderState` enum + `OrderStateMachine`: legal transitions SIGNAL_CONFIRMED ->
  RISK_APPROVED -> ORDER_SUBMITTING -> ORDER_SUBMITTED -> BROKER_ACKNOWLEDGED ->
  PARTIALLY_FILLED -> FILLED / CANCELED / REJECTED; rejects illegal transitions.
- `LatencyTracker`: record stage timestamps per correlation id; report p50/p95/p99
  over a bounded rolling window. Percentiles computed from recorded floats; the
  caller passes timestamps (time.monotonic) so the tracker stays pure.
- `SymbolState`: today_open, last_price, last_trade_ts, last_event_ts, held,
  in_flight (bool), realized round-trip-today flag.
- `decide_streaming_entry(state, equity, held, no_reentry, in_flight, config,
  executor_config) -> EntryDecision | None`: reuses daily_green_entry_signal +
  build_account_risk_profile + fractional sizing; returns symbol, qty, price,
  stop, target, reason, or None. One source of truth with the sync entry builder
  (extract/share the per-symbol sizing so the two paths cannot diverge).
- `staleness` helper: `is_stale(last_event_ts, now, threshold_s) -> bool`.

Async I/O shell (`StreamingEngine`):
- `bootstrap()`: load universe; REST-fetch today's opens (daily bars) + account +
  positions once, off the hot path (run_in_executor). Seed SymbolState.
- stream callbacks: `on_trade` (update last_price, run pipeline), `on_bar`
  (incremental state for M4), `on_quote` (bid/ask/spread for M3),
  `on_order_update` (drive the state machine, fills).
- pipeline per qualifying trade: correlation id -> decide_streaming_entry ->
  lean risk check (kill switch via shared kill-state, cash buffer, min_notional,
  max_order, gross, buying_power) -> dedupe (one in-flight per symbol) ->
  submit via AlpacaBroker (wrapped in run_in_executor so the sync SDK call never
  blocks the event loop) -> record latency stages.
- supervisor: staleness watchdog + reconnect with backoff; on reconnect, re-seed
  state and resubscribe.

## Risk + safety (reuse, do not duplicate thresholds)
- Reuse build_account_risk_profile for cash buffer / min_notional / max_order /
  gross; reuse the kill-switch state + daily-loss logic (shared with the sync
  engine via the same kill-state file + a shared helper).
- Idempotent client_order_id (order_tag_prefix + correlation id); one in-flight
  order per symbol; reconcile ambiguous submits before any retry.
- Paper/live separation preserved; no live flag flips. Flag defaults OFF.
- Nonblocking: all blocking calls (REST refresh, broker submit) off the event
  loop via run_in_executor; the pipeline math is pure and fast.

## Latency (spec section 4)
Measure internal stages with time.monotonic: event_recv -> signal_confirmed ->
risk_approved -> submit_initiated (-> broker_ack -> fill via order stream).
Target: event_recv -> submit_initiated p50 < 100 ms (internal; not a broker
guarantee). Report p50/p95/p99. Print a periodic latency summary.

## Acceptance criteria
- Flag OFF: main() runs the existing sync loop unchanged (byte-identical path).
- Flag ON (paper, market hours): stream connects; a green-vs-open crossing
  triggers one entry; order submitted with correlation id; state machine reaches
  FILLED on the order stream.
- No duplicate orders under repeated ticks (one-in-flight dedupe test).
- All risk gates enforced on the streaming path (kill switch, cash buffer,
  min_notional, max_order, gross, buying_power) - unit tests on decide/risk.
- Staleness detected and reconnect fires on a simulated drop.
- event->submit p50 < 100 ms measured on paper; p95/p99 reported.

## Testing
- Pure unit tests (no network): OrderStateMachine (legal/illegal transitions),
  LatencyTracker percentiles, is_stale, decide_streaming_entry (green/red/held/
  no-reentry/in-flight/sizing/fractional), dedupe one-in-flight.
- Integration (paper, market hours, manual): connect, trigger an entry, observe
  latency summary + state-machine progression; simulate a disconnect.

## Out of scope (later milestones)
- Rich terminal (M3), MTF gate (M4), TradingView (M5), full JSONL audit +
  shutdown/recovery hardening (M6), E2E/benchmark (M7).
