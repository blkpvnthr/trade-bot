# AMPS V6 — Integrated Regime-Aware Agentic Trading System: Report

_Branch `13K-development` · from HEAD `edb7328` (no reset) · 2026-11-20 · falsification-first per CLAUDE.md_

## 0. Final verdict (spec section 23)

```
ARCHITECTURE_INTEGRATED:   true
PLATFORM_LIVE_READY:       potentially true (deterministic spine built + tested; broker execution
                           gateway, continuous reconciliation, and streaming manager are STAGED reuse)
ENTRY_ALPHA_VALIDATED:     false   (entry-alpha gate found NO validated horizon; re-confirmed below)
STRATEGY_LIVE_ELIGIBLE:    false
LIVE_ACTIVATION_APPROVED:  false
```

The architecture from both papers is integrated and tested. It is **not** validated as profitable:
the mandatory Entry-Alpha gate found no out-of-sample edge, so no strategy is live-eligible. The
papers are architectural references, not evidence of profitability (spec section 20).

## 1. What was integrated (papers -> code)

### From paper 1 (event-driven allocation / risk-constrained execution)
- **Layer separation** prediction -> allocation -> risk -> execution, all reading ONE shared state:
  `amps_v6/state/canonical_state.py` (spec section 3).
- **Event-driven allocation** (reacts only to meaningful change, not every bar):
  `canonical_state.diff_events()` emits the reallocation events; `amps_v6/strategy/allocator.py`
  computes weights only on those events (paper 1 Eq. 3, `w_t = h(C_t, s_t, E_t)`).
- **Advisory models + deterministic execution-time overrides**: policy agents recommend; the
  independent risk gateway + exit stack override (spec sections 13, 16).
- **Decision-to-execution traceability**: `amps_v6/monitoring/decision_trace.py` records intent vs action.

### From paper 2 (regime-aware long-only + inverse ETF)
- **Regimes** BULL / BEAR / TRANSITION (paper) **+ CRISIS / INDETERMINATE** (spec extension) and the
  **fast-lock / confirmed-lock** state machine (+ preliminary / deterioration / reversal / unlock,
  spec extensions): `amps_v6/regime/regime_engine.py`. Proxy set SPY/QQQ/DIA/IWM/VIXY/TQQQ/SQQQ and
  `s_t = f(breadth, opening-range, VWAP alignment, volatility)` are from the paper.
- **Regime -> universe gating** (BULL->long, BEAR->inverse, TRANSITION->none, CRISIS->reduce-only):
  `regime_engine.permissions()`.
- **Trend / Breakout / Defensive branches** (+ OrderFlow, PositionManagement): `amps_v6/agents/policy_agents.py`
  (implemented as interpretable deterministic rule branches; see section 4).
- **Canonical position management** + helpers (drawdown-from-high, run-up, adverse-move):
  `amps_v6/execution/position_manager.py`.
- **Adaptive exposure caps** `E_t`: `allocator.adaptive_exposure_scale()`.
- **Reductions/exits remain available when entries are blocked**: enforced in `amps_v6/risk/exit_stack.py`
  and verified by tests.

### From AMPS V5/V6 research
- **Mandatory Entry-Alpha Agent** (`amps_v6/alpha/entry_alpha.py`): only `ALPHA_CONFIRMED` permits a
  new entry; it cannot size, override risk, or submit (spec section 8).
- **Versioned evidence registry** with OOS fields + strict mode-gating (`amps_v6/alpha/evidence_registry.py`).
- **Matched controls + the V5 finding** that unconditional directional entries lack entry alpha
  (re-confirmed below).

## 2. Processing order (spec section 2) — as built

```
Market Data -> Validation -> Multi-TF Features -> Regime Engine -> Specialist Policy Agents ->
Candidate Signal Engine -> Entry-Alpha Agent -> Event-Driven Allocator ->
Independent Risk & Exit Stack -> Execution Gateway (STAGED) -> Broker (STAGED) ->
Position Manager -> Monitoring/Audit
```
Orchestrated by `amps_v6/pipeline.py`. The regime engine provides context BEFORE candidates are
authorized; policy models recommend but never touch positions or the broker.

## 3. The section-16 decision conjunction (no voting)

```python
entry_authorized = (
    trusted_current_data AND locked_eligible_regime AND eligible_symbol AND valid_policy_setup
    AND entry_alpha == ALPHA_CONFIRMED AND not adversarial_veto AND not security_veto
    AND allocation_above_minimum AND deterministic_risk_approval
)
```
Implemented in `pipeline.entry_authorized()`. A clean **NO_TRADE** (no alpha / no setup / no
allocation) is distinguished from a safety **BLOCKED** (data / regime / eligibility / veto / risk).

## 4. Entry-alpha validation (the gate) — re-confirmed

Re-ran `amps_v6.backtesting.multi_horizon_diagnostic` on split/div-adjusted data (OOS >= 2024-05-01):

| Horizon group | Result |
|---|---|
| intraday 30m / 60m / 120m / 180m / full-day | **REJECTED** (MFE approx |MAE|; drift = beta, random control wins) |
| swing 1-3 days | **REJECTED** |
| swing 5-10 days | **CONTINUE_RESEARCH** (favorable asymmetry 1.15-1.23, beats random+momentum controls +40 to +72 bps, but net expectancy negative) |
| **Any VALIDATED** | **No** |

So `ENTRY_ALPHA_VALIDATED: false`. Details: `backtest_outputs/AMPS_V6_ENTRY_ALPHA_REPORT.md`.

## 5. Component ablation (what I could run without training RL)

The spec's ablation (section 20) compares V5 baseline / PPO-without-alpha / PPO-with-alpha /
deterministic rules / full stack / matched-random. The PPO branches here are implemented as
**interpretable deterministic rule branches**, not trained RL models — so the two PPO rows are
**N/A (no trained model was produced)**; disclosed, not faked. Prior cycles
([[AMPS V5]], RL context-encoder research) found model complexity adds no validated alpha, so
deterministic branches are the honest default. The rows I can report, from the diagnostic:

| Configuration | OOS result |
|---|---|
| V5 baseline (intraday directional-eligible) | net-negative, no edge vs control |
| Deterministic setup rules (regime + directional eligibility) | swing 5-10d beats controls but net-negative |
| Matched random entries | the control the above is measured against |
| Full stack WITH Entry-Alpha gate | **0 entries** (no VALIDATED evidence -> gate blocks all) |

The full stack with the alpha gate correctly authorizes **nothing**, because there is no validated
evidence — which is the intended fail-closed behavior, not a bug.

## 6. Tests (spec section 21)

- New: `tests/test_amps_v6_integrated.py` — **33 tests, all pass**. Covers regime
  classification (all 5 states), fast/confirmed locks, deterioration/reversal, regime permissions,
  event-driven allocation, **signed weight clamping (increase AND reduction)**, adaptive exposure,
  gross-cap scaling, order state machine (transitions/idempotency/ambiguity->reconciliation/resubmit
  guard/persistence), position manager (buy-average/sell-realize/exposure/helpers), exit stack (every
  reason + account-level dominance + exits-available-when-entries-blocked), the section-16
  conjunction, pipeline NO_TRADE/NO_ALPHA/OBSERVE_ONLY/BLOCKED-with-exit-still-running, policy-agent
  contracts, OrderFlow DATA_UNAVAILABLE veto, decision-trace append.
- Plus `tests/test_amps_v6.py` (22) from the prior spine build.
- **AMPS family total: 84 pass.** Full suite: **301 passed, 90 failed** — the 90 are PRE-EXISTING and
  unrelated (stale tests referencing removed `EMBARGO_DAYS`/`make_target_risk_aware`/`dry_run`, plus
  tests of pre-existing working-tree edits to `features.py`/`portfolio_rebalancer.py`). No new
  failures; no `amps_v6` module is imported by any failing test.

## 7. Honest scope / STAGED (spec section 48 disclosure)

Built + tested (deterministic, pure): canonical state, regime engine + locks, policy agents, entry-
alpha + evidence registry, event-driven allocator + signed clamp, risk gateway, exit stack, order
state machine, position manager, decision trace, pipeline.

**Staged (not built here; disclosed, reuse targets named):**
- Broker **execution gateway** + cost-aware limit execution + continuous **reconciliation** + protective-
  order recovery -> reuse `alpaca_executor.py`, `trade.py:ProductionOrderExecutor`,
  `streaming_engine.py`.
- **Streaming manager** (entitlement discovery, capacity-constrained subscriptions, backfill, gap
  detection) -> reuse `streaming_engine.py` + `order_flow_engine.py`.
- Full **multi-timeframe** build-out to all 6 TFs with every indicator + `DATA_UNAVAILABLE` on every
  field (current `features/multi_timeframe_engine.py` covers 5m + adjusted daily; the OrderFlowAgent
  already returns `DATA_UNAVAILABLE` rather than fabricating ticks).
- **Trained PPO** policy branches (deterministic rule branches shipped instead).
- Full backtest of the integrated system with fees/slippage/partial-fills/latency/gaps (the
  entry-alpha gate result makes a profitability backtest premature — nothing is validated to trade).

## 8. Reproduce

```bash
python -m amps_v6.backtesting.multi_horizon_diagnostic --train-end 2024-05-01   # entry-alpha gate
python -m pytest tests/test_amps_v6_integrated.py tests/test_amps_v6.py -q      # architecture tests
```

## 9. Remaining requirements before paper/live promotion

1. A **VALIDATED** entry-alpha horizon (currently none) — the single blocking item for strategy
   eligibility. Next: regime-conditioned, balanced bull/bear swing study with purged walk-forward +
   bootstrap CIs (per `AMPS_V6_ENTRY_ALPHA_REPORT.md`).
2. Build + reconciliation-test the broker **execution gateway** and **streaming manager** (staged).
3. Signed `risk_policy_version`, full security review, tested kill switches + restart recovery.
4. Observe-only -> shadow -> paper soak with live slippage inside tolerance.
5. Explicit operator `LIVE_ACTIVATION_APPROVED` (never set by implementation).
```
