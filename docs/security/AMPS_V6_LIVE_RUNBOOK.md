# AMPS V6 — Live Runbook

_Branch `13K-development` · 2026-11-20 · DO NOT enable live during implementation (spec sections 19, 48)_

## Preconditions (all required before any live submission)
- `PLATFORM_LIVE_READY = true` — execution gateway + reconciliation built and reconciliation-tested.
- `STRATEGY_LIVE_ELIGIBLE = true` — at least one horizon has VALIDATED entry-alpha evidence.
  **Currently false** (no validated horizon — see AMPS_V6_INTEGRATED_REPORT.md).
- `LIVE_ACTIVATION_APPROVED = true` — explicit operator action; never set by code.
- Signed `risk_policy_version`; separate paper/live credentials verified; single active executor.

## Promotion ladder (spec section 20)
```
Backtest -> Purged walk-forward -> Untouched OOS -> Agent ablation ->
Observe-only -> Shadow -> Paper -> Live pilot -> Controlled live
```
Each stage requires the prior stage's evidence. Do not skip. Do not promote on backtest alone.

## Daily operation (America/New_York, exchange calendar)
- ~08:00 premarket regime analysis, universe validation, REST screening, candidate ranking.
- 09:30-09:45 opening observation + regime evidence; NO default auto-entries.
- 09:45-10:10 position review, P&L management, fast-lock review.
- 10:11-13:50 alpha-gated entries, multi-TF monitoring, event-driven allocation.
- 13:50-14:00 overnight/swing decisions.
- Before close: exposure review, exit/carry, reconciliation.
- Continuous: stops, broker updates, data health, risk limits, kill switches, audit.

## Controls (persist across restart)
Inspect/set platform state:
```bash
python - <<'PY'
from amps_v6.configuration.platform_state import PlatformState
st = PlatformState.load(); print(st.to_dict())
PY
```
Kill switches: `global, strategy, new_entries, overnight_entries, position_promotion, leveraged_etf,
data_provider, agent_system` (global tripped by default = fail-safe). Reduce-only mode and cancel-all
are first-class: reductions/exits remain available even with new entries disabled.

## Startup sequence
1. Verify endpoint + credentials match the intended paper/live account (never infer mode from creds).
2. Startup reconciliation: load `order_state_machine` + `position_manager`; any
   `RECONCILIATION_REQUIRED` order must be reconciled against the broker BEFORE new activity.
3. Confirm clock drift within tolerance; confirm data health OK.
4. Confirm the three readiness states; if any is false, remain OBSERVE_ONLY.

## Go-live (only when all preconditions true)
Set `mode=LIVE`, `new_entries_enabled=True`, clear `global` kill, set `live_activation_approved=True`
(operator). `platform_state.can_send_live_order(horizon)` enforces the full conjunction fail-closed.

## Rollback
Trip `global` kill -> cancel-all -> reduce-only -> flatten per policy. Revert to prior commit on the
branch; positions survive restart via the persisted position manager + order state machine.
