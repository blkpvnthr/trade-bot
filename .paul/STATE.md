# Project State

## Project Reference

See: .paul/PROJECT.md (updated 2026-05-20)

**Core value:** Autonomous paper-trading engine that turns institutional accumulation signals into structured, risk-controlled intraday trade plans.
**Current focus:** v0.1 Production Foundation — Phase 2: Training Pipeline Consistency
**Objective:** Long-term adaptive compounding via multi-factor optimization (Sharpe, consistency, drawdown, profit factor, Calmar, alpha decay)

## Current Position

Milestone: Production Foundation (v0.1)
Phase: 2 of 4 (Training Pipeline Consistency) — Planning
Plan: 02-02 ✓, 02-02b ✓, 02-03 created (revised), awaiting approval
Status: PLAN created, ready for APPLY
Last activity: 2026-06-01 — Revised 02-03 plan; coherence check corrected a false premise (trainer writes non-canonical output paths 5+ consumers can't find)

Progress:
- Milestone: [████░░░░░░] 42%
- Phase 2: [███████░░░] 75%

## Loop Position

```
PLAN ──▶ APPLY ──▶ UNIFY
  ✓        ○        ○     [02-03 plan created, awaiting approval]
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
| 02-03: align producer (rl_intraday_structure_trainer.py) to consumers' canonical paths, not vice-versa | 2 | Trainer wrote models/rl_intraday/final_model.zip + metrics/rl_intraday_metrics.json; trade.py/self_awareness/optimizer/scripts expect system-data/{DATE}/{DATE}_rl_intraday_structure_model.zip + _training_metrics.json. Fix the trainer's default paths; consumers untouched. Smoke test exits 0 despite failures — also fixed. |

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

Last session: 2026-06-01
Stopped at: 02-03 plan revised/created (coherence check corrected the trainer-path premise; chose "fix the real mismatch")
Next action: Review and approve plan, then run `/paul:apply 02-03`
Resume file: .paul/phases/02-training-pipeline-consistency/02-03-PLAN.md

---
*STATE.md — Updated after every significant action*
