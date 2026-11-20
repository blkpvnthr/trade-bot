# AMPS V6 — Incident Response

_Branch `13K-development` · 2026-11-20_

## Principle
Fail closed. When in doubt, stop opening risk and keep the deterministic exit path alive. The exit
stack and reduce-only path never depend on agents or LLM availability.

## Immediate actions (any incident)
1. Trip the `global` kill switch (persists across restart):
   ```bash
   python - <<'PY'
   from amps_v6.configuration.platform_state import PlatformState
   st = PlatformState.load(); st.trip("global"); st.save()
   PY
   ```
2. Enter reduce-only; cancel working orders (cancel-all); do not submit new entries.
3. Snapshot `logs/execution/amps_v6_canonical_state.json`, `amps_v6_order_states.json`,
   `amps_v6_positions.json`, and `amps_v6_decision_trace.jsonl` for forensics.

## Scenario playbook

| Incident | Response |
|---|---|
| Broker outage | Reduce-only; orders in SUBMITTING -> RECONCILIATION_REQUIRED (never blind-resubmit); reconcile against broker on recovery before new activity. |
| Market-data outage / stale | `data_health != OK` -> entries blocked by the conjunction (`trusted_current_data`); exits still run on last-good + broker state. |
| Agent timeout / LLM down | Policy agents are advisory; pipeline proceeds with deterministic branches; exit stack unaffected. |
| Process crash / restart | Load order state machine + position manager; resolve every `RECONCILIATION_REQUIRED` before trading; verify protective orders at the broker. |
| Missing protective order | Position manager flags `protective_order_state=MISSING`; re-establish proportionate protection immediately; audit. |
| Duplicate / ambiguous submission | Client-order-id idempotency prevents dup creates; ambiguous SUBMITTING -> RECONCILIATION_REQUIRED; resolve before any resubmit. |
| Daily-loss / drawdown breach | Exit stack returns DAILY_LOSS_SHUTDOWN / DRAWDOWN_SHUTDOWN (account-level, dominates) -> flatten. |
| Regime reversal mid-position | Exit stack REGIME_EXIT fires regardless of entry gate; new entries in the old direction blocked. |
| Clock drift | Halt new entries; timestamps on causal features cannot be trusted; reconcile time source. |
| Suspected credential exposure | Trip `global`; rotate keys; agents never hold credentials (only the execution layer loads them). |

## Recovery verification before resuming
- Zero orders in RECONCILIATION_REQUIRED.
- Every open position has an ACTIVE protective order verified at the broker.
- Canonical state `data_health == OK`, clock drift within tolerance.
- Readiness states reviewed; remain OBSERVE_ONLY until an operator re-approves.

## Post-incident
Write a tamper-evident audit entry; file the forensic snapshots; if evidence deteriorated, demote the
affected entry-alpha evidence (`LIVE_APPROVED -> LIVE_SUSPENDED -> REVALIDATION_REQUIRED`).
