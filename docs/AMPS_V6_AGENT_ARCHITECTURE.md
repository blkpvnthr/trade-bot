# AMPS V6 — Agent Architecture

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20_

## Pipeline (spec §2)

```
Market Data
  -> Dynamic Universe & Liquidity Scanner      (universe/instrument_registry, universe/liquidity_ranker)
  -> Validated Multi-Timeframe Feature Engine   (features/multi_timeframe_engine; adjusted daily + 5m)
  -> Specialist Analysis Agents                 (agents/*  -- analyze/recommend/veto ONLY)
  -> Opportunity Generation
  -> Entry-Alpha Agent (MANDATORY GATE)         (alpha/entry_alpha + alpha/evidence_registry)
  -> Trading Horizon Selection                  (strategy/horizon_classifier)
  -> Deterministic Decision Fusion              (strategy/decision_fusion -- AND of gates, not voting)
  -> Execution-Adjusted Opportunity Ranking
  -> Portfolio Allocation                       (staged)
  -> Independent Risk Gateway                   (execution/risk_gateway -- fail-closed, deterministic)
  -> Order Intent                               (execution/order_intent -- typed, broker-agnostic)
  -> Execution Gateway -> Alpaca                (STAGED: reuse alpaca_executor / trade.py / streaming_engine)
  -> Reconciliation & Audit                     (STAGED)
  -> Continuous Position Management             (STAGED)
```

## Hard invariants (spec §2, §4, §42)

1. **Agents never touch the broker.** They return `Recommendation` objects. Only the execution
   gateway submits/cancels/replaces orders.
2. **Protective exits are independent** of analytical agents and LLMs. (The exit path reuses the
   repo's deterministic exit engines; it is never gated by entry-alpha approval.)
3. **Deterministic by default.** Latency-sensitive decisions use structured numeric features and
   deterministic models. LLM-backed agents sit behind feature flags and are never required per event.
4. **Strict schema or reject.** Every recommendation is validated (`agents/base.validate`); malformed,
   expired, out-of-range, or unsupported recommendations are dropped (`safe_analyze` fails safe).
5. **Entry-alpha is the mandatory gate.** Only `ALPHA_CONFIRMED` (VALIDATED evidence) authorizes
   progression; no other agent can override an alpha rejection (enforced in `decision_fusion`).

## Recommendation contract (spec §4)

`agent_id, recommendation_id, symbol, underlying, timestamp, data_version, strategy_version,
feature_snapshot_id, recommended_action, trading_horizon, confidence, evidence_id, reason_codes,
invalidation_conditions, expiration_timestamp`. An `ENTER_LONG`/`PROMOTE` must cite `evidence_id`
(schema-enforced). Actions: `ENTER_LONG, HOLD, REDUCE, EXIT, NO_TRADE, VETO, PROMOTE,
REJECT_PROMOTION`.

## Specialist agents

| Agent | Status | Backed by |
|---|---|---|
| MarketRegimeAgent | built (deterministic) | `amps_v4_regime.market_regime` |
| EntryAlphaAgent | built (deterministic) | `alpha/entry_alpha` + `alpha/evidence_registry` |
| UniverseSelection / OrderFlow / BullishSetup / BearishInverse / MomentumPersistence / ExecutionEconomics / PortfolioAllocation / PositionManagement / SwingRisk / AdversarialRisk / SecurityAudit / TradingHorizon | interface defined; deterministic wrappers STAGED | existing engines: `order_flow_engine`, `profit_exit_engine`, `buy_opportunity_engine`, `costs`, `amps_v4_regime` |

The two built agents demonstrate the full contract (typed interface, strict validation, fail-safe
execution). The remaining agents are thin deterministic wrappers over already-validated repo engines;
they are staged because the gate found no validated edge, so wiring 12 more agents around an
unvalidated strategy would be premature complexity (spec §36, mission: complexity requires evidence).

## Decision fusion (spec §17)

A candidate entry requires the conjunction: `valid_market_data AND valid_setup AND
eligible_instrument AND eligible_trading_horizon AND alpha_confirmed AND positive_expected_net_return
AND acceptable_execution_cost AND NOT adversarial_veto AND NOT security_veto AND
deterministic_risk_approval`. Inverse-ETF and swing entries add their own mandatory sub-gates. No gate
is weakened to raise trade frequency.
