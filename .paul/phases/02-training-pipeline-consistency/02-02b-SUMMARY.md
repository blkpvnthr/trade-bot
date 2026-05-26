---
phase: 02-training-pipeline-consistency
plan: 02b
subsystem: orchestration
tags: [run.py, subprocess, stream-confirmed, diagnostics, corrective]

requires:
  - phase: 02-02-training-pipeline-consistency
    provides: orchestration_audit.md identifying the broken stream_confirmed path reference

provides:
  - Correct stream_confirmed script path in run.py (3 stale references fixed)
  - [subprocess-check] diagnostic log line at watcher start
  - orchestration_audit.md and docs/architecture.md updated to reflect resolved status
affects:
  - 02-03-rl-intraday-structure-trainer-validation
  - live execution (stream_confirmed watcher now resolves correct script on first attempt)

tech-stack:
  added: []
  patterns:
    - "[subprocess-check] pattern: emit resolved path + exists= before any subprocess launch for immediate log visibility"

key-files:
  created: []
  modified:
    - run.py (3 stale filename refs → correct; 1 diagnostic log line added)
    - .paul/phases/02-training-pipeline-consistency/orchestration_audit.md (3 entries updated to resolved)
    - docs/architecture.md (DAG: stream_confirmed moved from PASSIVE to MARKET OPEN Popen; filename corrected)

key-decisions:
  - "STREAM_CONFIRM_SCRIPT_PATH constant corrected but NOT wired into launch function — hygiene only; launch function derives path from ScannerConfig"
  - "Existing alt-fallback at lines 9774–9777 preserved — backward compatibility if stale config still names the old file"

patterns-established:
  - "[subprocess-check] log pattern: print resolved path + exists= immediately after fallback resolution, before the existence guard"

duration: ~10 minutes
started: 2026-05-22T17:30:00Z
completed: 2026-05-22T17:40:00Z
---

# Phase 2 Plan 02b: stream_confirmed Broken Path Fix

**Corrected 3 stale `stream_confirmed_movement_gate.py` references in run.py to the actual filename `stream_confirmed_movement_gate_trade_plan_trailing.py`, added `[subprocess-check]` diagnostic log, and updated docs to resolved. Confirmed live: `[subprocess-check]` emitted `exists=True` during same-session paper trading run.**

## Performance

| Metric | Value |
|--------|-------|
| Duration | ~10 minutes |
| Started | 2026-05-22 |
| Completed | 2026-05-22 |
| Tasks | 2 of 2 |
| Files modified | 3 |

## Acceptance Criteria Results

| Criterion | Status | Notes |
|-----------|--------|-------|
| AC-1: No remaining references to old broken script name | Pass | `grep -c 'stream_confirmed_movement_gate\.py"' run.py` → 0 |
| AC-2: All three executable reference sites use correct filename | Pass | Lines 2250, 2716, 9773 all use `stream_confirmed_movement_gate_trade_plan_trailing.py` |
| AC-3: [subprocess-check] diagnostic present | Pass | Line 9781: `print(f"[subprocess-check] stream_confirm_script={script_path} exists={script_path.exists()}")` |
| AC-4: py_compile passes | Pass | `python -m py_compile run.py` → OK |
| AC-5: Docs reflect resolved status | Pass | orchestration_audit.md: 3 entries updated; docs/architecture.md: old name absent, correct name in MARKET OPEN DAG section |

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `run.py` | 4 targeted edits: constant (line 2250), ScannerConfig default (line 2716), else-fallback (line 9773), diagnostic log (line 9781) | Fix broken path; add visibility |
| `.paul/phases/02-training-pipeline-consistency/orchestration_audit.md` | Section 1b row + gap note → resolved; Section 2 row category updated; Section 4 Critical Gap row → resolved | Audit reflects current state |
| `docs/architecture.md` | PASSIVE reads: removed stale `stream_confirmed_movement_gate.py` line; MARKET OPEN: added `stream_confirmed_movement_gate_trade_plan_trailing.py` as Popen background entry | DAG now accurate |

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| STREAM_CONFIRM_SCRIPT_PATH constant corrected but not wired in | Plan boundary: constant is dead code hygiene; launch function derives path from ScannerConfig | No behavior change — diagnostic only |
| Existing alt-fallback (lines 9774–9777) preserved | Provides backward compat if an old config/env still names the pre-fix filename | No regression on stale configs |
| [subprocess-check] inserted after alt-fallback, before existence guard | Logs the fully-resolved path (post-fallback), so the log always shows the final decision | One log line per watcher start, always accurate |

## Deviations from Plan

None — plan executed exactly as specified.

## Live Confirmation

During the same session, `run.py` was launched in paper mode before this plan was formally closed. The execution log confirmed:

```
[subprocess-check] stream_confirm_script=.../stream_confirmed_movement_gate_trade_plan_trailing.py exists=True
[stream_confirm_watcher_started] cmd=[...stream_confirmed_movement_gate_trade_plan_trailing.py --poll]
```

Both lines confirm the fix was effective in production conditions.

## Next Phase Readiness

**Ready:**
- run.py stream_confirmed watcher resolves correct path on first attempt (no fallback needed)
- `[subprocess-check]` diagnostic visible in startup log for every future run
- All docs consistent with actual implementation

**Blockers:**
- None — 02-03 is the next planned item.

---
*Phase: 02-training-pipeline-consistency, Plan: 02b (corrective)*
*Completed: 2026-05-22*
