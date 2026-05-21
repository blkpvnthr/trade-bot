# Project State

## Project Reference

See: .paul/PROJECT.md (updated 2026-05-20)

**Core value:** Autonomous paper-trading engine that turns institutional accumulation signals into structured, risk-controlled intraday trade plans.
**Current focus:** v0.1 Production Foundation — Phase 2: Training Pipeline Consistency

## Current Position

Milestone: Production Foundation (v0.1)
Phase: 2 of 4 (Training Pipeline Consistency)
Plan: 02-01 complete
Status: Ready for next PLAN (02-02)
Last activity: 2026-05-20 — Unified 02-01: SUMMARY written, 1 of 3 plans complete in Phase 2

Progress:
- Milestone: [███░░░░░░░] 28%
- Phase 2: [███░░░░░░░] 33%

## Loop Position

```
PLAN ──▶ APPLY ──▶ UNIFY
  ✓        ✓        ✓     [Loop complete — ready for next PLAN]
```

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: ~session
- Total execution time: 1 session

**By Phase:**

| Phase | Plans | Total Time | Avg/Plan |
|-------|-------|------------|----------|
| 01-Data Architecture | 3/3 | 1 session | - |
| 02-Training Pipeline | 0/3 | - | - |

## Accumulated Context

### Decisions

| Decision | Phase | Impact |
|----------|-------|--------|
| Alpaca-first, yfinance fallback disabled by default | 1 | All data fetches route to Alpaca; ALLOW_YFINANCE_MARKET_DATA_FALLBACK=0 in .env |
| Split PREMARKET vs REGULAR feed | 1 | Premarket uses SIP/BOATS; intraday uses IEX |
| ALPACA_FAIL_ON_FEED_ERROR=1 for strict mode | 1 | Training scripts fail loudly on feed errors |
| TradingView screener: discovery only | 1 | No OHLCV from TradingView |

### Deferred Issues

| Issue | Origin | Effort | Revisit |
|-------|--------|--------|---------|
| BOATS feed requires paid Alpaca subscription | Phase 1 | S | When upgrading Alpaca plan |
| forecast_engine.py training CSV provenance not yet validated | Phase 1 | M | Phase 2 plan 02-02 |

### Blockers/Concerns

None active.

## Boundaries (Active)

- Never suppress sell orders or exit logic
- APCA_PAPER=true must remain enforced
- Do not commit system-data/, daily_system_outputs/, logs/ to git
- premarket_trade_planner.py: preserve all risk control paths

## Session Continuity

Last session: 2026-05-20
Stopped at: Plan 02-01 complete and unified. Paused before 02-02.
Next action: `/paul:plan` — Plan 02-02: end-to-end smoke test (build_intraday_training_dataset.py → CSV → consumer log verification)
Resume file: .paul/phases/02-training-pipeline-consistency/02-01-SUMMARY.md

---
*STATE.md — Updated after every significant action*
