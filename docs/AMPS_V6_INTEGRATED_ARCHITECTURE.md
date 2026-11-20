# AMPS V6 — Integrated Architecture

_Branch `13K-development` · 2026-11-20 · references the two regime-aware RL papers (architecture only)_

## Pipeline (spec section 2)

```
Market Data
  -> Validation & Normalization
  -> Multi-Timeframe Feature Engine        amps_v6/features/multi_timeframe_engine.py
  -> Market Regime Engine                  amps_v6/regime/regime_engine.py   (context BEFORE candidates)
  -> Specialist Policy Agents              amps_v6/agents/policy_agents.py   (advisory only)
  -> Candidate Signal Engine
  -> Entry-Alpha Agent (MANDATORY GATE)    amps_v6/alpha/entry_alpha.py + evidence_registry.py
  -> Event-Driven Allocator                amps_v6/strategy/allocator.py     (runs on events only)
  -> Independent Risk & Exit Stack         amps_v6/execution/risk_gateway.py + amps_v6/risk/exit_stack.py
  -> Execution Gateway  [STAGED]           reuse alpaca_executor.py / trade.py / streaming_engine.py
  -> Broker (Alpaca)    [STAGED]
  -> Position Manager & Reconciliation     amps_v6/execution/position_manager.py + order_state_machine.py
  -> Monitoring, Audit, Research Feedback  amps_v6/monitoring/decision_trace.py
```
Orchestration: `amps_v6/pipeline.py`. Canonical state: `amps_v6/state/canonical_state.py`.

## Invariants
1. **One canonical state** — every component reads/writes `CanonicalState`; no private copies.
2. **Agents are advisory** — they return `Recommendation` objects; only the execution gateway touches
   the broker.
3. **Regime gates eligibility, never creates trades** — `regime_engine.permissions()`.
4. **Entry needs the full conjunction** (section 16), not a vote — `pipeline.entry_authorized()`.
5. **Exits always available** — `risk/exit_stack.py` runs regardless of entry permissions / regime /
   agent availability / LLM availability.
6. **Fail-closed** — default OBSERVE_ONLY; live needs all three readiness states + operator approval.

## Regime state machine (section 6)
States: BULL / BEAR / TRANSITION / CRISIS / INDETERMINATE. Lock: UNLOCKED -> PRELIMINARY ->
FAST_LOCK (high confidence inside opening window) or CONFIRMED_LOCK (persistent confidence after
window); DETERIORATION / REVERSAL / unlock while locked; CRISIS pre-empts immediately. Permissions:
BULL->long, BEAR->inverse, TRANSITION->no entries, CRISIS->reduce-only, INDETERMINATE->no entries.

## Event-driven allocation (section 10)
Reallocation events (`canonical_state.diff_events`): confirmed regime change, eligible-set change,
new/invalidated alpha candidate, material confidence change, exposure-limit change, position fill,
stop/exit, risk-policy event, EOD. Weight clamp is **signed and reduction-safe**:
```
bounded_delta = clip(target - current, -step_cap, +step_cap)
approved      = clip(current + bounded_delta, min_weight, max_weight)
```
(The papers' `min(...)` form mishandles reductions; this is the spec-mandated correction.)

## Order lifecycle (section 14)
`amps_v6/execution/order_state_machine.py` enumerates all spec states (CREATED..TRADE_BUSTED) with
legal transitions, client-order-id idempotency, and an ambiguity guard: an order in SUBMITTING with
unknown outcome -> RECONCILIATION_REQUIRED, and is never resubmitted until reconciled.

## Long / inverse ETF controls (section 12)
`amps_v6/universe/instrument_registry.py` holds exposure groups + opposing-symbol sets; the risk
gateway blocks simultaneous bull+inverse exposure unless hedge mode; the position manager models
effective (daily) leverage. Inverse entries require BEAR regime + bearish underlying.

## What is staged
Execution gateway/broker I/O, streaming manager, trained PPO, full 6-TF feature build-out, and a
profitability backtest. See `AMPS_V6_INTEGRATED_REPORT.md` section 7. The architecture is complete and
tested; these are the remaining build items, gated behind a validated entry-alpha edge (none yet).
