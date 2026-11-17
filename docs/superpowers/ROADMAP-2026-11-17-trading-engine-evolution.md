# Trading Engine Evolution - Program Roadmap

Date: 2026-11-17
Status: Approved (decomposition + sequence); execute sub-project by sub-project
Target: trade.py and new supporting modules
Source spec: user architecture spec (2026-11-17), captured in
docs/superpowers/specs/2026-11-17-fractional-and-intraday-momentum-design.md (M1/M4)

## Guiding constraints (apply to every milestone)

- Preserve existing trading strategies, broker integration, position sizing, risk
  controls, and paper/live separation. Reuse the current scanner/executor.
- DO NOT enable live trading. No unverified production changes. Paper-only
  validation before any milestone is considered done.
- Nonblocking: terminal output, historical analysis, and chart inspection must
  never delay order execution.
- Latency target: under 100 ms internal market-event-to-order-submission,
  measured at p50/p95/p99. A performance target, not a broker guarantee.
- Additive over rewrite: new async/streaming paths are built ALONGSIDE the
  existing synchronous loop behind config flags, and only promoted once proven on
  paper. The working engine remains the fallback until a milestone is validated.
- Every milestone ships with unit tests + measurable acceptance criteria + a
  focused code review before moving on.

## Current state (already shipped this session)

- Daily-green 2:1 strategy in trade.py (config-gated, default on).
- Dynamic account-size risk scaling (build_account_risk_profile caps).
- Synchronous engine: ProductionOrderExecutor loop (~30s poll), MomentumSetupScanner
  (REST daily/intraday bars), AlpacaBroker, kill switch, order plan, JSONL events.

## Architecture direction

Current: single-threaded poll loop; REST bars; integer orders; print logging.
Target: event-driven, nonblocking core where a streaming market-data feed drives
candidate evaluation and a low-latency execution pipeline, with historical/MTF
analysis and terminal rendering on separate async tasks/queues, and a full
audit/latency trail. Built incrementally and flag-gated so the sync loop stays
intact as fallback.

Component map (target):
- market_data_stream: async subscriptions (quotes/trades/bars), reconnect, staleness.
- candidate_state: in-memory per-symbol eligibility + MTF classification + freshness.
- execution_pipeline: event -> eligibility -> momentum -> trigger -> risk -> submit
  -> ack -> fill, with correlation IDs, order state machine, latency timers.
- terminal_console: async Rich dashboard + event feed (modes: full/signals/orders/dashboard).
- audit_log: nonblocking JSONL queue, structured events, reconciliation support.
- mtf_gate: cached higher-TF trend + lower-TF momentum + 1m trigger.
- tv_inspection: out-of-band TradingView chart validation + walk-forward harness.

---

## Milestones (execution order)

### M1. Fractional orders (complete + verify)   [spec: fractional-and-intraday §Part 1]
Scope: exact-dollar fractional sizing (flag default ON), float through
broker/plan/execute, float exit qty, fractionability fallback. Design already
approved (2026-11-17 spec Part 1).
Depends on: none. Risk: low-med (live execution path, but flag-gated).
Acceptance:
- Flag OFF -> integer behavior byte-identical (regression test).
- Flag ON -> $74 budget on a $661 name yields a fractional entry (not 0), order
  clears all risk gates; fractional position sells its full float qty.
- submit rounds to 9 dp; non-fractionable asset falls back to whole shares.
- Live paper check on the current account: QQQ/SPY produce a fractional entry.

### M2. Streaming market-data + low-latency execution   [spec §3, §4, §9]
Scope: async Alpaca streaming (quotes/trades/bars); incremental indicator updates;
staleness/reconnect detection; event-driven pipeline to order submission with
correlation IDs, explicit order state machine, idempotent submission, duplicate
prevention; latency instrumentation (p50/p95/p99). Built as a flag-gated parallel
path; sync loop remains fallback. Pulls in the minimal audit/state-machine
foundation it needs (full hardening in M6).
Depends on: M1. Risk: HIGH (sync -> async rearchitecture on live-capable engine).
Mitigations: additive/flagged; paper-only; reconcile ambiguous submits before
retry; keep all existing risk checks in the pipeline; shadow-run against the sync
loop before promotion.
Acceptance:
- Streaming feed delivers events; staleness + reconnect handled (simulated drop).
- Event-to-submission p50 < 100 ms (measured, paper); p95/p99 reported.
- No duplicate orders under repeated signals (idempotency test); risk gates still
  enforced; paper/live separation intact.

### M3. Real-time terminal monitoring   [spec §5, §6]
Scope: async Rich console; continuous DATA/SCAN/TREND/TV/SIGNAL/DECISION/RISK/
ORDER/ACK/FILL/ERROR feed; system-status panel; modes full/signals/orders/dashboard
(default dashboard); nonblocking render worker; throttling/aggregation under load.
Depends on: M2 (event/audit stream). Risk: med (must not block execution).
Acceptance:
- Every candidate, decision, signal, risk result, order, ack, fill visible with
  ms timestamps; correlation IDs traceable.
- Rendering runs on its own task; a high-frequency event stress test shows no
  measurable increase in execution latency.

### M4. Cached multi-timeframe momentum gate   [spec §1; design §Part 2]
Scope: retrieve up to 180D 5min OHLCV async off the hot path; resample to 30m/60m;
HTF trend classification (EMA20/50 alignment + slope + HH/HL + price vs MAs ->
BULLISH/BEARISH/NEUTRAL) cached with freshness; LTF 5min momentum (session VWAP,
EMA9/21, structure, rel-vol); 1m entry trigger. Long = bullish HTF + bullish 5m +
1m trigger; defer on conflict/neutral. No look-ahead (completed candles only).
Depends on: M2 (streaming feeds the cache incrementally) + existing bar fetch for
initial load. Risk: med. Supersedes the interim simple intraday-uptrend gate.
Acceptance:
- Classification matches hand-computed fixtures on canned bars; no look-ahead
  (decision uses only completed candles at the timestamp).
- Historical load is async and never runs inside an entry decision.
- Gate defers entries on neutral/conflicting trends; unit + paper tests.

### M5. TradingView inspection + validation (out of band)   [spec §2, §10 partial]
Scope: use the TradingView integration to inspect representative 180D:5min charts;
compare computed trend transitions vs visible structure; record inspection
outcomes (symbol/timestamp/status); walk-forward evaluation of parameter changes.
Strictly outside the live execution path; record "unavailable" rather than
fabricate if the integration is absent.
Depends on: M4. Risk: low-med.
Acceptance: inspection runs off the hot path; outcomes persisted; a no-integration
run records unavailable status without blocking trading.

### M6. Reliability, structured logging, integration hardening   [spec §7, §8, §9]
Scope: complete nonblocking JSONL audit pipeline (bounded queues, overflow
detection, priority for order/risk events), full event timestamping + correlation
IDs, controlled shutdown/recovery without losing pending orders, broker-state
reconciliation, no credential leakage.
Depends on: M2, M3. Risk: med.
Acceptance: every material event in both terminal and persistent logs; restart
recovers pending-order state; reconciliation against broker records; logging
failure cannot break execution and is surfaced.

### M7. End-to-end testing, concurrency stress, paper validation   [spec §10, §11]
Scope: compare old vs new gate; backtest w/ costs/slippage; walk-forward; metrics
(win rate, expectancy, profit factor, Sharpe, maxDD, frequency); multi-candidate
concurrency; latency under load; broker reject/partial/stale/disconnect sims;
audit completeness; paper-trading soak before any live consideration.
Depends on: M1-M6. Risk: med.
Acceptance: all §11 final acceptance criteria met on paper; latency benchmarked;
recovery verified. Live remains disabled pending explicit user decision.

---

## Risk register (top)
- R1 sync->async rearchitecture (M2) destabilizes the live-capable engine ->
  additive/flagged parallel path, paper-only, shadow-run before promotion.
- R2 duplicate/again orders under streaming retries -> idempotent client_order_id
  + order state machine + reconcile-before-retry (M2/M6).
- R3 terminal/logging backpressure stalls execution -> separate tasks, bounded
  queues, drop/aggregate render (never drop audit/decisions) (M3/M6).
- R4 look-ahead bias in MTF cache -> completed-candles-only, timestamped decisions,
  walk-forward (M4/M7).
- R5 accidental live enablement -> paper/live guards preserved; no live flag flips
  in this program.

## Milestone gate (per milestone)
spec -> plan -> TDD implementation -> focused code review -> paper validation vs
acceptance criteria -> commit. Do not start milestone N+1 until N's acceptance
criteria pass on paper.
