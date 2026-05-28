# Project State

## Project Reference

See: .paul/PROJECT.md (updated 2026-05-20)

**Core value:** Autonomous paper-trading engine that turns institutional accumulation signals into structured, risk-controlled intraday trade plans.
**Current focus:** v0.1 Production Foundation — Phase 2: Training Pipeline Consistency
**Objective:** Long-term adaptive compounding via multi-factor optimization (Sharpe, consistency, drawdown, profit factor, Calmar, alpha decay)

## Current Position

Milestone: Production Foundation (v0.1)
Phase: 2 of 4 (Training Pipeline Consistency) — In Progress
Plan: 02-02 ✓, 02-02b ✓, 02-03 not yet planned
Status: Loop closed — ready for 02-03 planning
Last activity: 2026-05-22 — Applied and unified 02-02b (stream_confirmed fix)

Progress:
- Milestone: [████░░░░░░] 42%
- Phase 2: [███████░░░] 75%

## Loop Position

```
PLAN ──▶ APPLY ──▶ UNIFY
  ✓        ✓        ✓     [02-02b loop complete — ready for 02-03 plan]
```

## Performance Metrics

**Velocity:**
- Total plans completed: 4 (3 × Phase 1 + 1 × Phase 2)
- Average duration: ~session
- Total execution time: 2 sessions (Phase 2 so far)

**By Phase:**

| Phase | Plans | Total Time | Avg/Plan |
|-------|-------|------------|----------|
| 01-Data Architecture | 3/3 | 1 session | - |
| 02-Training Pipeline | 2/3 (02-01 ✓, 02-02 ✓) | 2 sessions | - |

## Accumulated Context

### Decisions

| Decision | Phase | Impact |
|----------|-------|--------|
| Alpaca-first, yfinance fallback disabled by default | 1 | All data fetches route to Alpaca; ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0 in .env |
| Split PREMARKET vs REGULAR feed | 1 | Premarket uses SIP/BOATS; intraday uses IEX |
| ALPACA_FAIL_ON_FEED_ERROR=1 for strict mode | 1 | Training scripts fail loudly on feed errors |
| TradingView screener: discovery only | 1 | No OHLCV from TradingView |
| Multi-factor composite scoring for strategy ranking | Cross-cutting | evaluate_variants.py ranks by composite_score (Sharpe+consistency+PF+Calmar+decay penalties); config-driven weights in YAML |
| Optimization objective encoded in YAML + PROJECT.md | Cross-cutting | $400-500/day research target documented as aspirational; system optimizes risk-adjusted returns, not raw P&L |
| Smoke test reads source text, not imports | 2 | Avoids torch/alpaca-py side effects; pattern for all future pipeline checks |
| Orchestration audit is read-only (no hooks in v0.1) | 2 | Hooks reserved for v0.2 Phase 5 after paper trading baseline validated |
| Corrective plans use NN-NNb naming | 2 | Not tracked in ROADMAP; execute between parent plan and next plan in sequence |

### Deferred Issues

| Issue | Origin | Effort | Revisit |
|-------|--------|--------|---------|
| BOATS feed requires paid Alpaca subscription | Phase 1 | S | When upgrading Alpaca plan |
| deeplob_orderbook_direction_layer.py never invoked by run.py — signals always 0.50 | 02-02 audit | M | v0.2 Phase 5 orchestration audit |
| ENV_PATH missing from run.py module-level scope | Paper launch | XS | Fixed 2026-05-22 (added ENV_PATH = PROJECT_ROOT / ".env") |

### Blockers/Concerns

None active.

## Boundaries (Active)

- Never suppress sell orders or exit logic
- APCA_PAPER=true must remain enforced
- PAPER_ONLY=true now in root .env (added 2026-05-22 as permanent safeguard)
- Do not commit system-data/, daily_system_outputs/, logs/ to git
- premarket_trade_planner.py: preserve all risk control paths

## Session Continuity

Last session: 2026-05-22
Stopped at: 02-02b applied and unified; paper trading running (tux_mtf_original, paper mode)
Next action: Plan 02-03 (`/paul:plan` — rl_intraday_structure_trainer.py validation + smoke test)
Resume file: .paul/phases/02-training-pipeline-consistency/02-02b-SUMMARY.md

---
*STATE.md — Updated after every significant action*
