# Project State

## Project Reference

See: .paul/PROJECT.md (updated 2026-05-20)

**Core value:** Autonomous paper-trading engine that turns institutional accumulation signals into structured, risk-controlled intraday trade plans.
**Current focus:** v0.1 Production Foundation — Phase 3: Forecast & RL Quality
**Objective:** Long-term adaptive compounding via multi-factor optimization (Sharpe, consistency, drawdown, profit factor, Calmar, alpha decay)

## Current Position

Milestone: Production Foundation (v0.1)
Phase: 3 of 4 (Forecast & RL Quality) — Not started (DISCOVERY done)
Plan: Not started
Status: Ready to plan (Phase 2 complete + committed; DISCOVERY.md ready → /paul:plan 03-01)
Last activity: 2026-07-03 — Phase 2 complete: 02-03 unified + transitioned. Trainer I/O canonicalized + provenance; smoke 10/10. AC-5 (live model resolve) deferred to next real session.

Progress:
- Milestone: [█████░░░░░] 50% (2 of 4 phases)
- Phase 3: [░░░░░░░░░░] 0%

## Loop Position

```
PLAN ──▶ APPLY ──▶ UNIFY
  ✓        ✓        ✓     [02-03 loop closed — Phase 2 complete, ready to plan Phase 3]
```

## Performance Metrics

**Velocity:**
- Total plans completed: 7 (3 × Phase 1 + 4 × Phase 2 incl. 02-02b)
- Average duration: ~session
- Total execution time: 3 sessions (Phase 2)

**By Phase:**

| Phase | Plans | Total Time | Avg/Plan |
|-------|-------|------------|----------|
| 01-Data Architecture | 3/3 | 1 session | - |
| 02-Training Pipeline | 4/4 (02-01/02/02b/03 ✓) | 3 sessions | - |

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
| 2026-07-03: 02-03 Task 3 live-run checkpoint deferred (not fixed in code) | 2 | Path fix proven statically (10/10 smoke + canonical defaults); AC-5 live model_exists=True needs a real session. Do NOT force synthetic training on a shared Alpaca conn. |
| 2026-07-03: 02-03 Task 2 was a no-op — smoke-test exit gating already present | 2 | Plan premise stale (02-02 already added sys.exit(0/1)). AC-4 met by existing gating + Task 1; scripts/smoke_test_training_pipeline.py unchanged despite files_modified listing. UNIFY should reconcile. |

### Deferred Issues

| Issue | Origin | Effort | Revisit |
|-------|--------|--------|---------|
| BOATS feed requires paid Alpaca subscription | Phase 1 | S | When upgrading Alpaca plan |
| deeplob_orderbook_direction_layer.py never invoked by run.py — signals always 0.50 | 02-02 audit | M | v0.2 Phase 5 orchestration audit |
| ENV_PATH missing from run.py module-level scope | Paper launch | XS | Fixed 2026-05-22 (added ENV_PATH = PROJECT_ROOT / ".env") |
| AC-5: confirm live-trained model resolves — validate_rl_intraday_structure_model()=True | 02-03 | XS | Next real session: python -c "import trade; print(trade.validate_rl_intraday_structure_model())" |

### Blockers/Concerns

None active.

## Boundaries (Active)

- Never suppress sell orders or exit logic
- APCA_PAPER=true must remain enforced
- PAPER_ONLY=true now in root .env (added 2026-05-22 as permanent safeguard)
- Do not commit system-data/, daily_system_outputs/, logs/ to git
- premarket_trade_planner.py: preserve all risk control paths

## Session Continuity

Last session: 2026-07-03
Stopped at: Phase 2 complete — 02-03 unified + phase transition (PROJECT/ROADMAP evolved, scoped git commit). Ready to plan Phase 3.
Next action: `/paul:plan 03-01` (Phase 3 Forecast & RL Quality — DISCOVERY.md ready: DDR reward shaping). Separately, close AC-5 on next real session.
Resume file: .paul/ROADMAP.md

---
*STATE.md — Updated after every significant action*
