# AMPS V6 — Threat Model (draft)

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20_

Scope: the AMPS V6 platform spine built in this cycle. Live execution is not built/activated, so
broker-order threats are partly prospective; they are listed so the staged execution gateway inherits
the controls.

## Assets
- Brokerage credentials (`APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` in `.env`).
- Account capital and open positions.
- Evidence registry + platform state (authorization source of truth).
- Audit trail / journals.

## Trust boundaries
- **Agents / LLMs are UNTRUSTED for control.** They may recommend/veto; they can never submit orders
  and never see brokerage credentials. External market text is untrusted input (prompt-injection
  surface) and must not reach any order path.
- **Only the deterministic execution gateway** (staged) may call the broker, behind the risk gateway.
- **Operator** is the sole authority for `LIVE_ACTIVATION_APPROVED` and a signed `risk_policy_version`.

## Key threats and controls

| Threat | Control |
|---|---|
| Credential exposure to an LLM/agent | Agents receive only feature snapshots; creds load only in the execution layer (`features.load_alpaca_client`), never passed to `agents/*`. |
| Mode inferred from credentials (accidental live) | Mode is explicit in `PlatformState`; `can_send_live_order` requires `mode==LIVE` AND operator approval AND platform+strategy readiness. Fail-closed. |
| Unvalidated strategy reaching live | `STRATEGY_LIVE_ELIGIBLE` derived from per-horizon VALIDATED evidence; evidence registry mode-gates every entry. |
| Runaway / malfunction | 8 persistent kill switches (global tripped by default); risk gateway denies on any active switch; global/strategy switches disable the whole platform. |
| Unauthorized overnight / leverage exposure | `overnight_entries` + `leveraged_etf` kill switches; risk gateway blocks swing entries without overnight permission+eligibility; opposing-exposure conflict blocked unless authorized hedge mode. |
| Position promotion abuse (holding losers) | `position_promotion` kill switch; promotion requires positive executable net profit + tightened protection (enforced in the staged promotion engine; invariants documented). |
| Evidence/version drift | Drift monitor (staged) demotes `LIVE_APPROVED→LIVE_SUSPENDED→REVALIDATION_REQUIRED`; `permits_entry` refuses suspended evidence. |
| Split-contaminated data → false signals | Multi-day analysis requires `adjustment=ALL`; unadjusted multi-day analysis is prohibited (documented; the gate enforced it). |
| Restart losing protective orders / overnight state | Persistent platform state + evidence registry; execution-side reconciliation + protective-order recovery are STAGED reuse of `trade.py`/`streaming_engine.py`. |

## Residual risks (disclosed)
- Broker execution gateway, reconciliation, and order state machine are **not yet implemented** in
  `amps_v6/`; the above order-path controls are specified, not all exercised.
- Full security review, `LIVE_RUNBOOK`, and `INCIDENT_RESPONSE` are pending (staged).
- No live or paper trading has been run.
