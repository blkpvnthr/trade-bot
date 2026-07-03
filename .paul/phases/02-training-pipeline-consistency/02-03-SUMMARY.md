---
phase: 02-training-pipeline-consistency
plan: 03
type: execute
status: complete-with-deferral
autonomous: false
executed: 2026-07-03
files_modified: ["rl_intraday_structure_trainer.py"]
files_unchanged_vs_plan: ["scripts/smoke_test_training_pipeline.py"]
---

# SUMMARY: 02-03 — Trainer path canonicalization + provenance

## Outcome

Trainer producer aligned to the consumers' canonical convention. Phase-2 provenance chain
is proven **statically end-to-end** (10/10 smoke + canonical-default checks). The single
live-confirmation AC (AC-5) is **deferred** to the next real trading session — see Deferral.

## Tasks

| Task | Status | Qualify | Notes |
|------|--------|---------|-------|
| 1: Canonicalize trainer I/O + provenance log | DONE | PASS | Only path/provenance changed; training logic byte-equivalent |
| 2: Smoke-test fail-exit gating | DONE_WITH_CONCERNS | PASS | **No edit needed** — gating already present; concern was a stale plan premise |
| 3: Live training confirmation (checkpoint) | DEFERRED | n/a | Today's bars missing + other run.py live on shared Alpaca conn |

## What changed (rl_intraday_structure_trainer.py)

- Added `from datetime import datetime, timezone` and `from pathlib import Path`.
- Added canonical date-stamped derivation mirroring `trade.py:118-175`:
  `PROJECT_ROOT`, ET-based `_TODAY_TAG` (UTC fallback if zoneinfo/tzdata absent),
  `DATA_TAG = RUN_DATA_DATE -> RL_DATA_DATE -> _TODAY_TAG`, `OUT_DIR = PROJECT_ROOT/system-data/DATA_TAG`.
- Changed **defaults only** (env overrides still win):
  - `RL_DATA_PATH`    -> `OUT_DIR/{DATE}_intraday_bars_1m.csv`  (AC-1, Alpaca-derived)
  - `RL_METRICS_PATH` -> `OUT_DIR/{DATE}_rl_intraday_training_metrics.json`  (AC-2)
  - new `RL_MODEL_OUT` = `OUT_DIR/{DATE}_rl_intraday_structure_model`  (SB3 appends `.zip`) (AC-2)
  - `RL_SAVE_PATH` unchanged (SB3 working/checkpoint dir — NOT the canonical artifact)
- Final save: `model.save(os.path.join(save_path,"final_model"))` -> `os.makedirs(OUT_DIR)` + `model.save(RL_MODEL_OUT)`; log points at `RL_MODEL_OUT`.
- `load_bars`: emits literal `[data-source] training_csv=alpaca_derived path=%s` to both logger and stdout (AC-3).

## Verification (all fresh-run, not recalled)

- `ast.parse` OK.
- AC greps: `_intraday_bars_1m.csv`=1, `_rl_intraday_structure_model`=2, `_rl_intraday_training_metrics`=1, provenance literal=2.
- Canonical defaults confirmed via import (RUN_DATA_DATE=20260703): model/metrics/data land in `system-data/20260703/` with correct names; `RL_SAVE_PATH=models/rl_intraday` preserved.
- Env override precedence confirmed: `RL_METRICS_PATH=/tmp/x.json`, `RL_DATA_PATH=/tmp/bars.csv` both win.
- Smoke test: **10/10 [PASS], EXIT=0**. Forced-failure copy (scratchpad, non-destructive): **EXIT=1**. AC-4 satisfied.

## Deviations from plan

1. **Task 2 required no code change.** Plan premise ("smoke test prints failures but exits 0, cannot gate")
   was already false — `import sys` + `sys.exit(0/1)` were present (likely from 02-02). AC-4 is met by the
   existing gating plus Task 1's path edits. No check was weakened. `scripts/smoke_test_training_pipeline.py`
   is therefore unchanged despite being listed in `files_modified`.

## Deferral (AC-5 — live confirmation)

Not run today. Preconditions at execution time (2026-07-03):
- `system-data/20260703/20260703_intraday_bars_1m.csv` **missing** (no bars built for today).
- A `run.py` from the OTHER repo (`/Users/blkpvnthr/trader-paper`) was live; different PROJECT_ROOT so no
  write-race to this repo, but it shares the single Alpaca data connection (build contention risk).
- Chose not to force a synthetic/thin-data 5-20 min training run.

**Follow-up to close AC-5:** on the next real session (or when run.py here builds bars + trains), run:
`python -c "import trade; print(trade.validate_rl_intraday_structure_model())"` and confirm `model_exists=True`
against `system-data/{DATE}/{DATE}_rl_intraday_structure_model.zip`.

## Boundaries respected

No changes to training logic/reward/architecture/timesteps, no consumer edits, no run.py, no risk controls.
