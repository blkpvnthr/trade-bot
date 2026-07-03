---
phase: 03-forecast-rl-quality
plan: 01
subsystem: rl
tags: [reward-shaping, ddr, sortino, moody-saffell, ppo, stable-baselines3]
type: execute
completed: 2026-07-03
files_modified: ["rl_agent_core.py", "rl_intraday_structure_trainer.py"]
files_created: ["tests/test_ddr_reward.py"]
provides:
  - "DDR (differential downside deviation ratio) reward mode, opt-in via RL_REWARD_MODE=ddr"
  - "Per-step-additive downside-only reward fixing the non-additive rolling-Sharpe defect"
affects: ["03-03 model evaluation harness (empirical DDR-vs-baseline A/B)"]
key-decisions:
  - "DDR added as a mode on SharpeRewardCalculator (one param), not a new class/factory"
  - "Default stays sharpe_rolling; DDR opt-in until 03-03 validates it beats baseline"
  - "DDR path disables VecNormalize norm_reward + widens clip (already a normalised ratio)"
---

# Phase 3 Plan 01: RL Reward Shaping (DDR) Summary

**Added a mathematically-correct, per-step-additive Differential Downside Deviation Ratio (Moody & Saffell 2001) reward behind `RL_REWARD_MODE`, defaulting to the unchanged legacy path, proven by 7 data-free unit tests.**

## Acceptance Criteria Results

| Criterion | Status | Evidence |
|-----------|--------|----------|
| AC-1: DDR implemented + correct | Pass | Moody-Saffell recurrence in `_update_ddr`; warm-up raw fallback; finite across streams; downside sensitivity (from identical state, gain +4.99 vs loss -71.2) |
| AC-2: selectable, default preserved | Pass | `RL_REWARD_MODE` unset→sharpe_rolling, `=ddr`→ddr; `RL_REWARD_DDR_ETA` honored; reset() clears A/DD2/n |
| AC-3: parity + hygiene, no new deps | Pass | golden parity to legacy formula at 1e-12; no NaN/inf; deterministic; both modules parse; DDR uses only `math` |

## What Was Built

| File | Change | Detail |
|------|--------|--------|
| `rl_agent_core.py` | Modified | `SharpeRewardCalculator` gains `mode`/`eta`/`downside_target` + DDR online state (`_A`,`_DD2`,`_ddr_n`); `reset()` clears it; `update()` branches; new `_update_ddr()` |
| `rl_intraday_structure_trainer.py` | Modified | `RL_REWARD_MODE`/`RL_REWARD_DDR_ETA` env reads; calculator constructed with mode/eta; VecNormalize clip guard for DDR; `[reward] mode=` log at train() |
| `tests/test_ddr_reward.py` | Created | 7 data-free tests: parity, default-mode, no-NaN, warm-up, downside sensitivity, determinism, reset |

## Verification (fresh runs)

- `python -m pytest tests/test_ddr_reward.py -q` → **7 passed**
- `python -m pytest tests/test_tjl_engine.py -q` → **4 passed** (no regression)
- `ast.parse` clean on both edited modules
- default (unset) → `sharpe_rolling`; `RL_REWARD_MODE=ddr RL_REWARD_DDR_ETA=0.05` → mode=ddr eta=0.05
- DDR method references only `math` (no scipy/pandas/torch added)

## Deviations from Plan

| Type | Item | Note |
|------|------|------|
| Minor | Task 3 tests import `rl_agent_core` which pulls torch/SB3 at module import | Plan said "skip SB3 import entirely"; unavoidable without extracting the class (scope creep). Real intent — no training/data — is satisfied; tests are pure-calculator, run in 1.5s |
| Process (not code) | Task 1 initial qualify probe for downside sensitivity was mis-designed | Compared streams with different PRIOR state (different risk denominators). Corrected to same-prefix/differ-last-step; code was correct all along. This shaped `test_ddr_downside_sensitivity`'s design |

## Notes for 03-03 (eval harness)

- DDR produces large-magnitude penalties on losses when downside deviation is small (loss reward -71 in the probe). The VecNormalize guard (norm_reward off + clip 100) bounds this, but 03-03 should confirm clip 100 isn't swallowing signal, and sweep `eta` (default 0.02).
- Empirical A/B (DDR vs sharpe_rolling on held-out walk-forward) is the deferred validation — this plan only proved correctness + selectability.
- Default remains `sharpe_rolling`; flipping the default is a validated-promotion decision for after 03-03.

## Next Phase Readiness

**Ready:** DDR reward is correct, selectable, unit-proven; trainer wired; no behavior change unless opted in.
**Concerns:** DDR scale vs VecNormalize clip (flagged above) — an empirical question for 03-03.
**Blockers:** None.

---
*Phase: 03-forecast-rl-quality, Plan: 01*
*Completed: 2026-07-03*
