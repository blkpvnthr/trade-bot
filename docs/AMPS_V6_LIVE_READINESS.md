# AMPS V6 — Live Readiness

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20_

This document is the authoritative status of the three independent readiness states and the blockers
on each. **Live execution requires ALL THREE true, plus mode=LIVE, new_entries_enabled, no active
kill switch, and a horizon-specific eligibility flag.** None of that is satisfied; nothing live was
built or activated.

## Status block

```
COMMIT:                      (this commit; see git log on branch research/amps-v2-entry-exit-calibration)
FILES_CHANGED:               amps_v6/** (25 modules), tests/test_amps_v6.py, backtest_outputs/AMPS_V6_*.md, docs/AMPS_V6_*.md, docs/security/AMPS_V6_THREAT_MODEL.md
TESTS_PASSED:                22 (tests/test_amps_v6.py) ; 51 amps-family (v2/v4/v5/v6)
TESTS_FAILED:                0 new ; 90 PRE-EXISTING unrelated (stale: EMBARGO_DAYS/make_target_risk_aware/dry_run + pre-existing working-tree edits to features.py/portfolio_rebalancer.py)

PLATFORM_LIVE_READY:         false
STRATEGY_LIVE_ELIGIBLE:      false
LIVE_ACTIVATION_APPROVED:    false

MICRO_SCALP_ELIGIBLE:        false
INTRADAY_MOMENTUM_ELIGIBLE:  false
EXTENDED_INTRADAY_ELIGIBLE:  false
SHORT_SWING_ELIGIBLE:        false
EXTENDED_SWING_ELIGIBLE:     false

OVERNIGHT_TRADING_ELIGIBLE:  false
ENTRY_ALPHA_EVIDENCE_STATUS: no horizon VALIDATED; swing 5-10d = CONTINUE_RESEARCH (ALPHA_WEAK, net-negative); all intraday + swing 1-3d = REJECTED
PAPER_TEST_STATUS:           not started (no validated horizon to paper-test)
SECURITY_REVIEW_STATUS:      threat model drafted (docs/security/AMPS_V6_THREAT_MODEL.md); full review pending
LIVE_BLOCKERS:               (1) no VALIDATED entry-alpha evidence for any horizon; (2) broker execution gateway + order state machine + reconciliation NOT built (staged; reuse alpaca_executor.py / trade.py / streaming_engine.py); (3) overnight risk engine not built; (4) dashboard panels staged; (5) operator authorization required; (6) signed risk-policy version required
```

## Why each state is false

- **PLATFORM_LIVE_READY = false.** The deterministic decision/risk spine is built and tested, but the
  broker-facing execution gateway, order state machine, continuous reconciliation, and protective-order
  recovery are **not** implemented in `amps_v6/` yet. Designated reuse targets: `alpaca_executor.py`
  (`run_execution`, `submit_market_order`, `poll_order_fill`, kill-switch), `trade.py`
  (`ProductionOrderExecutor`, `AlpacaBroker`, `_kill_switch_allows_execution`,
  `build_account_risk_profile`), `streaming_engine.py` (`OrderStateMachine`).
- **STRATEGY_LIVE_ELIGIBLE = false.** The entry-alpha gate validated no horizon (see
  `backtest_outputs/AMPS_V6_ENTRY_ALPHA_REPORT.md`). Swing 5–10d is `CONTINUE_RESEARCH` only.
- **LIVE_ACTIVATION_APPROVED = false.** Operator-only; never set by implementation (spec §48).

## Operator commands (for FUTURE authorized use — do NOT run to go live now)

```bash
# 1. Re-run the decisive gate (safe; read-only research)
python -m amps_v6.backtesting.multi_horizon_diagnostic --train-end 2024-05-01

# 2. Inspect / manage platform state (safe)
python - <<'PY'
from amps_v6.configuration.platform_state import PlatformState
st = PlatformState.load(); print(st.to_dict())
PY

# 3. Trip / clear a kill switch (safe, persists)
python - <<'PY'
from amps_v6.configuration.platform_state import PlatformState
st = PlatformState.load(); st.trip("global"); st.save()   # or st.clear("global")
PY
```

Going live would additionally require, in order: a VALIDATED evidence record promoted through
RESEARCH→PAPER→LIVE_PILOT→LIVE_APPROVED, the execution gateway built and reconciliation-tested, a
signed `risk_policy_version`, `platform_live_ready=True`, the horizon's `*_ELIGIBLE=True`,
`live_activation_approved=True` (operator), `mode=LIVE`, `new_entries_enabled=True`, and the global
kill switch cleared. The code path (`PlatformState.can_send_live_order`) enforces this conjunction,
fail-closed.
