# AMPS V6 — Multi-Horizon, Evidence-Gated Trading Platform: Research & Build Report

_Branch `research/amps-v2-entry-exit-calibration` · from commit `042719e` · 2026-11-20 · falsification-first per CLAUDE.md_

## 0. Executive summary

Two things were delivered, and they are tracked **separately** (as the spec's three-state model demands):

1. **The decisive entry-alpha gate was run first (spec §34), and it governs everything.**
   - **No trading horizon is VALIDATED.** `STRATEGY_LIVE_ELIGIBLE = false`.
   - **Intraday (reproduces V5): REJECTED** at every horizon. Forward MFE ≈ |MAE|; the positive
     full-day drift is pure beta that a random-entry control actually *beats*.
   - **Swing 5–10 trading days: CONTINUE_RESEARCH** — the first time across 7 build/validate cycles
     that directional selection shows a *real* signal: favorable MFE/|MAE| asymmetry (1.15→1.23)
     **and it beats both the random-entry and momentum controls by +40 to +72 bps**. But absolute
     net expectancy is still **negative** (−18 to −57 bps): selection value without profitability.
     Not deployable; worth continuing.
2. **A deterministic, tested platform spine** (registry, adjusted multi-TF features, agent contract,
   evidence registry, deterministic risk gateway, decision fusion, horizon selector, liquidity
   ranker, three-state readiness + kill switches). 25 modules, 22 new tests, all passing. **No live
   order path was built or activated** (spec §44/§48: do not activate live during implementation).

> Honest headline: V6 did not find a deployable edge, but it produced the first genuine selection
> signal (multi-day swing) and a reusable, safety-gated platform to carry any *future* validated
> strategy. Per the mission, nothing is declared profitable without evidence.

## 1. Critical data-integrity finding (and fix)

The multi-day swing question is impossible to answer on the repo's `bars_multi_tf` cache: **that 5m
data is UNADJUSTED for splits.** Leveraged/inverse ETFs split constantly — measured overnight
jumps: SOXS 10.2×/9.6×, SQQQ 4.9×, SPXU 5.0×, NVD 24.3×, TZA 8.1×, DRIP 10.5×, and forward splits
TQQQ 2.0×, SOXL 15.0×. These are harmless within a session (which is **why V6.1 cleanly reproduces
V5**) but create spurious ±hundreds-of-percent "overnight returns" that destroy any multi-day
statistic. The first swing run on unadjusted data produced an absurd **+753 bps random-control
1-day return** — a pure split artifact.

**Fix:** the swing path fetches Alpaca daily bars with `adjustment=ALL` (split+dividend adjusted),
caches them (`data/cache/amps_v6_daily_adj/`), and verifies 0 extreme overnight gaps remain. All
swing numbers below use adjusted data. We **never** approximate a multi-day leveraged-ETF return by
multiplying an underlying return by stated leverage (spec §24). This corroborates the standing data
hazard noted in prior work (legacy caches carry unadjusted splits).

## 2. V6.1 — Intraday (reproduce V5). Data: real 5m bars, 20 ETFs, OOS ≥ 2024-05-01, n=32,403 entries

| Horizon | MFE (bps) | MAE (bps) | MFE/\|MAE\| | entry net | random-ctrl net | target<stop | decision |
|---|---|---|---|---|---|---|---|
| 30 min | +72.6 | −71.0 | 1.024 | −11.9 | −9.6 | 0.305 | NO_ALPHA_EVIDENCE |
| 60 min | +103.1 | −99.8 | 1.032 | −9.7 | −4.5 | 0.308 | NO_ALPHA_EVIDENCE |
| 120 min | +172.8 | −156.0 | 1.107 | +3.6 | +9.2 | 0.309 | NO_ALPHA_EVIDENCE |
| 180 min | +231.1 | −204.9 | 1.128 | +8.3 | +22.4 | 0.309 | NO_ALPHA_EVIDENCE |
| full day | +405.8 | −339.3 | 1.196 | +29.0 | **+52.5** | 0.309 | NO_ALPHA_EVIDENCE |

Reproduces V5 exactly: near-symmetric excursion, and the positive longer-horizon drift is **beta** —
the random-entry control *beats* the directional-eligible entries at every horizon ≥ 2h. Intraday
directional selection adds no value (it slightly subtracts). **All intraday horizons REJECTED.**

## 3. V6 SWING — the new question (daily, split-adjusted). OOS ≥ 2024-05-01, n=1,042 entries, 10 ETFs

| Horizon | MFE (bps) | MAE (bps) | MFE/\|MAE\| | entry net | random-ctrl net | momentum-ctrl net | edge vs ctrl | decision |
|---|---|---|---|---|---|---|---|---|
| 1 day | +329 | −329 | 1.002 | −30.3 | −40.7 | −30.5 | +10.4 | NO_ALPHA_EVIDENCE |
| 2 day | +436 | −421 | 1.036 | −36.2 | −42.8 | −35.3 | +6.6 | NO_ALPHA_EVIDENCE |
| 3 day | +528 | −494 | 1.068 | −36.9 | −51.6 | −36.6 | +14.7 | NO_ALPHA_EVIDENCE |
| **5 day** | +696 | −604 | **1.152** | −18.7 | −90.9 | −34.0 | **+72.2** | **ALPHA_WEAK** |
| **7 day** | +827 | −687 | **1.203** | −28.2 | −88.7 | −31.7 | **+60.5** | **ALPHA_WEAK** |
| **10 day** | +979 | −798 | **1.227** | −56.6 | −96.4 | −27.7 | **+39.8** | **ALPHA_WEAK** |

At 5–10 days the eligible entries develop favorable MFE/|MAE| asymmetry **and** beat both controls
by a wide margin (+40 to +72 bps). This is a genuine *selection* signal — the first in these cycles.
But absolute net expectancy stays negative, so it is graded **ALPHA_WEAK → CONTINUE_RESEARCH**, not
confirmed. It is not rounded up.

**Why net-negative despite beating controls:** the OOS-eligible swing set skewed **inverse-heavy**
(DOG, DRIP, NVD, PSQ, QID, SDS, SH, SPXS, + TECL/TECS) in a 2024–2026 **bull** market — a
structurally disadvantaged book. That selection still beat its own randomly-timed control is the
signal; the inverse skew is the drag. The obvious next experiment (regime-conditioned direction +
balanced bull/bear coverage) is documented as future work, not run here.

## 4. Horizon classification (spec §46)

| Horizon | Classification |
|---|---|
| intraday 30/60/120/180 min, full-day | **REJECTED** (×5) |
| swing 1d, 2d, 3d | **REJECTED** |
| swing 5d, 7d, 10d | **CONTINUE_RESEARCH** |
| **Any VALIDATED** | **No** |

## 5. What was built (the platform spine — infrastructure, not a strategy)

All deterministic, pure, unit-tested; no network in the hot path; no broker calls; agents can only
recommend/veto.

| Module | Spec | Purpose |
|---|---|---|
| `amps_v6/universe/instrument_registry.py` | §5-6 | 43-candidate registry, directional exposure groups, opposing-exposure sets, honest data-coverage report |
| `amps_v6/features/multi_timeframe_engine.py` | §9, §24 | 5m (reuse V2) + **split/div-adjusted daily** resampler/fetcher + daily DMI/ADX/BB/ATR features, no look-ahead |
| `amps_v6/alpha/entry_alpha.py` | §13-15, §34 | horizon excursion/asymmetry/net-drift + matched random control + honest decision grading |
| `amps_v6/alpha/evidence_registry.py` | §14, §44 | versioned evidence lifecycle; strict mode-gating; one-step promotion; drift suspension |
| `amps_v6/strategy/decision_fusion.py` | §17 | deterministic AND-of-gates (not voting); inverse/swing extra gates; any-agent veto |
| `amps_v6/strategy/horizon_classifier.py` | §15 | risk-adjusted net-return horizon selection; NO_TRADE when unprofitable/unevidenced |
| `amps_v6/universe/liquidity_ranker.py` | §7 | execution-aware liquidity ranking (ADV + spread proxy), not raw volume |
| `amps_v6/execution/order_intent.py` | §29 | typed, broker-agnostic order-intent (extended V6 contract) |
| `amps_v6/execution/risk_gateway.py` | §17,§25-26 | independent fail-closed risk approval + fixed-fractional sizing; kill-switch/suppression aware |
| `amps_v6/configuration/platform_state.py` | §3, §42 | three-state readiness, per-horizon eligibility, operating modes, persistent kill switches |
| `amps_v6/agents/base.py` + `deterministic_agents.py` | §4 | typed agent interface + strict schema validation; MarketRegime + EntryAlpha agents |
| `amps_v6/backtesting/multi_horizon_diagnostic.py` | §34-36 | the decisive gate runner (V6.0/V6.1 + swing + controls) |

## 6. Three-state readiness (spec §3)

```
PLATFORM_LIVE_READY:        false  (spine built + tested; execution gateway/reconciliation NOT built; see §7)
STRATEGY_LIVE_ELIGIBLE:     false  (no horizon VALIDATED; swing 5-10d only CONTINUE_RESEARCH)
LIVE_ACTIVATION_APPROVED:   false  (operator-only; never set by implementation)
```

Per-horizon eligibility: all `false`. Overnight trading eligible: `false`. Default mode:
`OBSERVE_ONLY`, `new_entries_enabled=false`, global kill switch tripped.

## 7. Honest scope, limitations, and what is STAGED (spec §48 disclosure)

- **No live execution path was built or run.** The risk gateway + order-intent are the pre-broker
  layer; the broker execution gateway, order state machine, and reconciliation are **staged** —
  the repo's `alpaca_executor.py` / `trade.py ProductionOrderExecutor` / `streaming_engine.py
  OrderStateMachine` are the designated reuse targets. `docs/AMPS_V6_LIVE_READINESS.md` lists this.
- **Order-flow / microstructure features (CVD, bid-ask imbalance) require genuine ticks** and are
  absent/flagged, never fabricated (consistent with V2–V5).
- **Swing universe skew:** OOS-eligible swing entries were inverse-heavy (see §3); the swing result
  is a signal on a structurally-disadvantaged book, not a balanced bull/bear test. Regime-conditioned
  direction is the next experiment.
- **Coverage:** 30/43 candidate ETFs and 11/15 references have cached bars; adjusted daily fetched
  for all 42 swing symbols. Missing single-stock ETFs (MSTU/MSTZ/AMDL/AMDS/CONL/CONI/SPXL/…) are
  disclosed by the registry's `coverage_report()`, never silently assumed.
- **Not yet written (staged, not faked):** `AMPS_V6_MULTI_HORIZON_REPORT.md`,
  `AMPS_V6_EXECUTION_ECONOMICS_REPORT.md`, `AMPS_V6_OVERNIGHT_RISK_REPORT.md`,
  `security/AMPS_V6_LIVE_RUNBOOK.md`, `security/AMPS_V6_INCIDENT_RESPONSE.md`, dashboard panels, and
  experiments V6.2–V6.24 beyond the gate. The gate result (no validated edge) means the right next
  step is swing entry-alpha research, not building live infrastructure around an unvalidated edge.
- **Test suite:** 22 new V6 tests pass; full suite shows 90 failures that are **pre-existing** and
  unrelated (stale tests referencing removed `EMBARGO_DAYS`/`make_target_risk_aware`/`dry_run`, plus
  tests of pre-existing working-tree edits to `features.py`/`portfolio_rebalancer.py`). No V6 module
  is imported by any failing test.

## 8. Reproduce / operate

```bash
# the decisive gate (builds adjusted-daily cache on first run; needs Alpaca creds in .env)
python -m amps_v6.backtesting.multi_horizon_diagnostic --train-end 2024-05-01

# platform-spine tests
python -m pytest tests/test_amps_v6.py -q
```

## 9. Decision

**CONTINUE RESEARCH (swing entry-alpha), platform spine retained as infrastructure.** No horizon is
promoted; nothing is live-eligible; live activation remains operator-gated and false. The one
encouraging thread — a real multi-day selection signal that beats controls — is kept and pointed at
the next experiment (regime-conditioned, balanced-universe swing alpha), consistent with the
session-wide finding that the evidenced edge on this repo remains insider selection, not price/flow
timing.
