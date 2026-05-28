---
phase: 02-training-pipeline-consistency
plan: 02
subsystem: training-pipeline
tags: [alpaca, data-provenance, smoke-test, orchestration-audit]

requires:
  - phase: 02-01-training-pipeline-consistency
    provides: load_instance_env migration, [data-source]/[data-feed] log lines in builder and consumers

provides:
  - scripts/smoke_test_training_pipeline.py — static path/log/compile verifier for the training CSV chain
  - .paul/phases/02-training-pipeline-consistency/orchestration_audit.md — full run.py orchestration gap analysis
  - Live builder run confirmed: both output CSVs produced with Alpaca IEX feed
affects:
  - 02-03-rl-intraday-structure-trainer-validation
  - 02-02b-stream-confirmed-fix (corrective plan spawned from audit)

tech-stack:
  added: []
  patterns:
    - "Static smoke test pattern: grep source text for log lines + suffixes + py_compile; no imports of heavy deps"
    - "Orchestration audit pattern: grep + targeted reads only; no full-file load; no code changes"

key-files:
  created:
    - scripts/smoke_test_training_pipeline.py
    - .paul/phases/02-training-pipeline-consistency/orchestration_audit.md
    - .paul/phases/02-training-pipeline-consistency/02-02b-PLAN.md
  modified:
    - build_intraday_training_dataset.py (stale ALPACA_DATA_FEED ref fixed in dataset_summary(), Task 3 deviation)

key-decisions:
  - "Smoke test uses grep/regex on source text rather than importing modules — avoids torch/alpaca-py side effects"
  - "Orchestration audit is read-only documentation; no hooks wired yet — reserved for v0.2 Phase 5"
  - "02-02b created as corrective intercalary plan (not in ROADMAP) to fix broken stream_confirmed path before 02-03"

patterns-established:
  - "Corrective plans use NN-NNb naming, autonomous: true, depend on the parent plan, not tracked in ROADMAP phase table"

duration: 2 sessions (2026-05-20 → 2026-05-22)
started: 2026-05-20T00:00:00Z
completed: 2026-05-22T17:30:00Z
---

# Phase 2 Plan 02: Training Pipeline Consistency — Smoke Test + Orchestration Audit

**Closed the Phase 1 deferred issue: training CSV provenance confirmed end-to-end. Smoke test script created (8 static checks, exit 0). Live builder run produced both output CSVs. Orchestration audit written, surfacing the stream_confirmed broken path reference now tracked in corrective plan 02-02b.**

## Performance

| Metric | Value |
|--------|-------|
| Duration | 2 sessions |
| Started | 2026-05-20 |
| Completed | 2026-05-22 |
| Tasks | 4 completed |
| Files created | 3 |
| Files modified | 1 (minor fix, Task 3 deviation) |

## Acceptance Criteria Results

| Criterion | Status | Notes |
|-----------|--------|-------|
| AC-1: Path alignment verified statically | Pass | smoke_test reports [PASS] for both builder→forecast_engine and builder→rl_trainer suffix checks |
| AC-2: Log lines confirmed present | Pass | `[data-source] training_csv=alpaca_derived` confirmed in both forecast_engine.py and rl_intraday_structure_trainer.py; `[data-feed] regular=` in builder |
| AC-3: All three pipeline files compile cleanly | Pass | py_compile passes on all three |
| AC-4: Smoke test exits 0 | Pass | "All 8 checks passed." |
| AC-5: Orchestration audit written | Pass | orchestration_audit.md: 261 lines, 5 sections (integrated inventory, standalone list, pipeline DAG, gap analysis, v0.2 recommendations) |
| AC-6: Live builder run produces expected CSVs | Pass | User confirmed: `[data-feed] regular=iex` present; both CSVs created. Stale `ALPACA_DATA_FEED` reference in `dataset_summary()` fixed during this run. |

## Accomplishments

- `scripts/smoke_test_training_pipeline.py` created: 8 static checks (path suffix alignment × 2, log line presence × 3, py_compile × 3). Runs in <1s, no network/heavy-dep imports.
- Live Alpaca IEX fetch confirmed: `system-data/{DATE}_intraday_training_dataset.csv` and `_intraday_bars_1m.csv` both produced.
- Full orchestration gap analysis delivered: 261-line `orchestration_audit.md` maps every subprocess call, direct import, file-based passive read, and standalone script — with pipeline DAG and v0.2 hook recommendations.
- Critical gap found: `stream_confirmed_movement_gate.py` referenced at 3 sites in run.py but the file does not exist (only `_trade_plan_trailing.py` variant does). Corrective plan 02-02b written and approved.

## Files Created/Modified

| File | Change | Purpose |
|------|--------|---------|
| `scripts/smoke_test_training_pipeline.py` | Created | Static verifier for training CSV provenance chain |
| `.paul/phases/02-training-pipeline-consistency/orchestration_audit.md` | Created | Full run.py orchestration gap analysis |
| `.paul/phases/02-training-pipeline-consistency/02-02b-PLAN.md` | Created | Corrective plan to fix broken stream_confirmed path references |
| `build_intraday_training_dataset.py` | Modified | Stale `ALPACA_DATA_FEED` ref in `dataset_summary()` corrected to `ALPACA_REGULAR_FEED` during Task 3 live run |

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Smoke test reads source text rather than importing | Heavy deps (torch, alpaca-py) and side effects on import make `import forecast_engine` impractical in a lightweight test harness | Pattern established for all future training pipeline checks |
| Orchestration audit is read-only (no hooks wired) | Premature orchestration adds risk before models are stable; hooks belong in v0.2 Phase 5 after v0.1 paper trading baseline is validated | No run.py changes; findings tracked in orchestration_audit.md |
| 02-02b created as corrective intercalary plan | Broken stream_confirmed path is a real bug (silent fallback exists but resolution path is unlogged); fix is surgical and doesn't need a full phase plan | Plan exists, ready for apply before 02-03 |

## Deviations from Plan

### Summary

| Type | Count | Impact |
|------|-------|--------|
| Auto-fixed (in-scope) | 1 | Minimal — stale env var name, same file |
| Scope additions | 1 | 02-02b corrective plan created from audit findings |
| Deferred | 0 | None |

**Total impact:** Essential fix + expected audit output, no scope creep.

### Auto-fixed Issues

**1. Stale `ALPACA_DATA_FEED` reference in `build_intraday_training_dataset.py:dataset_summary()`**
- **Found during:** Task 3 (live builder run)
- **Issue:** `dataset_summary()` logged `os.getenv("ALPACA_DATA_FEED")` instead of `os.getenv("ALPACA_REGULAR_FEED")` — always printed `None` in the feed diagnostic
- **Fix:** Updated to `ALPACA_REGULAR_FEED` to match the Phase 1 env var naming
- **Files:** `build_intraday_training_dataset.py`
- **Verification:** Live run showed correct feed value in output

### Scope Additions

**1. Corrective plan 02-02b: stream_confirmed broken path references**
- **Found during:** Task 4 (orchestration audit)
- **Issue:** `STREAM_CONFIRM_SCRIPT_PATH`, `ScannerConfig.stream_confirmed_movement_script` default, and else-fallback all reference `"stream_confirmed_movement_gate.py"` — a file that does not exist. A silent alt-fallback exists at lines 9774–9777 so execution doesn't hard-fail, but path resolution was unlogged.
- **Action:** Created `.paul/phases/02-training-pipeline-consistency/02-02b-PLAN.md` with 4 targeted edits + `[subprocess-check]` diagnostic. Not tracked in ROADMAP. Executes between 02-02 and 02-03.

## Next Phase Readiness

**Ready:**
- Training CSV provenance chain fully verified: builder → forecast_engine → rl_trainer paths aligned
- `[data-source]` and `[data-feed]` log lines confirmed in all three pipeline files
- Smoke test script ready for regression use in future phases
- run.py orchestration fully mapped: integrated, passive, and standalone components identified

**Pending before 02-03:**
- Apply corrective plan 02-02b (stream_confirmed fix + [subprocess-check] diagnostic) — plan exists, autonomous, takes ~5 min
- 02-03 will validate `rl_intraday_structure_trainer.py` and run the full smoke test cycle

**Concerns:**
- `deeplob_orderbook_direction_layer.py` is PASSIVE (signal file read, but script never invoked by run.py) — signals always default to 0.50. Flagged as high-priority gap in orchestration_audit.md Section 4. Not blocking 02-03 but worth noting for v0.2 Phase 5.

**Blockers:**
- None — 02-02b is ready to apply immediately.

---
*Phase: 02-training-pipeline-consistency, Plan: 02*
*Completed: 2026-05-22*
