# Target lineage report (Phase 2)

Traces where every RL training label originates and classifies each as **ALLOWED** (computed
from FUTURE bars / real outcomes) or **FORBIDDEN** (derived from features the model also trains on).
Rule: **targets must never be derived from features** (and features must never be derived from the
target). Backed by the Phase-1 audits (`docs/leakage_audit_*.md`).

## Verdict table

| Dataset | Target | Origin | Classification |
|---|---|---|---|
| `alpha_entry` | `realized_R` | `rl_clean_target.py` — forward 2-day underlying return from real daily bars | **ALLOWED** ✓ (clean, post-fix) |
| `alpha_exit` | `realized_R` / `peak_R` | genuine forward bracket (`rl_alpaca_backfill` walks future bars) | target ALLOWED, but **FEATURES LEAK** ⚠ |
| `buffett_dca` | `fwd_return_5d/20d/60d` | `rl_training_dataset_builder` synthesizes `0.5·discount + 0.3·conviction + noise` | **FORBIDDEN** ⚠ (target = f(features)) |
| `capital_allocation` | `*_bucket_return` | `rl_training_dataset_builder`: `alpha_bucket_return = alpha_expectancy·0.01`, etc. | **FORBIDDEN** ⚠ (target = f(features)) |

## Detail & required fix per target

### alpha_entry — `realized_R` — ALLOWED ✓
- Origin: `rl_clean_target.retarget()` → `forward_return = (close[t+H]−close[t])/close[t]` from
  cached daily bars; `realized_R = clip(forward_return/0.005)`. Independent of `setup_score`.
- Phase-1 audit: CLEAN (`probability_of_profit`→win AUC 0.50; no feature >0.55).
- ⚠ Pipeline note: `rl_training_dataset_builder` first writes the *entangled* matched-exit
  bracket value; `rl_clean_target.py` must be run AFTER every rebuild to restore the clean target.

### alpha_exit — `realized_R` genuine, but FEATURES are derived from it ⚠
- Target `realized_R`/`peak_R`: genuine forward bracket (`rl_alpaca_backfill.py:123-133`, walks
  future bars). **ALLOWED as a target.**
- **Leak is on the feature side** — `rl_common.alpha_exit_features()` builds:
  `unrealized_R = r_multiple`; `premium_ratio = 1+0.1·r_multiple`; `stop_distance = max(0,1−0.3·r_multiple)`;
  `trail_high_ratio = max(0,r_multiple/3)` — i.e. **features are invertible functions of the target**
  (Phase-1: R²≈1.0, baseline OOS R² 1.0).
- **Fix:** these "features" must be the ex-ante state KNOWN AT THE EXIT DECISION (unrealized R *so
  far*, trailing-stop distance *at that bar*, MFE-to-date), not the FINAL realized r_multiple. Rebuild
  exit features from the decision-time bar context, not the outcome.

### buffett_dca — `fwd_return_*` — FORBIDDEN ⚠
- Origin: `rl_training_dataset_builder.build_buffett_dca_dataset()`:
  `base = 0.5·intrinsic_value_discount + 0.3·conviction_score + 0.05·noise`, then
  `fwd_return_60d = base`, `20d = 0.6·base`, `5d = 0.25·base`, and `valuation_improvement = discount`.
  The target is a deterministic linear function of two features (Phase-1: RF-regressor OOS R² 0.926).
- **Fix:** use REAL forward returns. The backfill's Buffett replay already computes
  "REAL forward returns over daily bars" (`rl_alpaca_backfill.py:165-168`) — wire those through
  instead of the synthetic `base` formula. (30/90/180/365-day forward returns per Phase-4.)

### capital_allocation — `*_bucket_return` — FORBIDDEN ⚠
- Origin: `build_capital_allocation_dataset()`: `alpha_bucket_return = alpha_expectancy·0.01`,
  `buffett_bucket_return = buffett_quality·0.008`, `cash_bucket_return = 0.0001`. Bucket returns are
  features × constants (Phase-1: `alpha_expectancy` R² 1.0 vs target).
- **Fix:** use realized per-bucket portfolio returns over the forward window (actual alpha/buffett
  sleeve P&L), not the expectancy aggregate. (Also n=169 — far too few samples regardless.)

## Feature-side contamination found in the backfill (must also be cleaned)

- `rl_alpaca_backfill.py:153,447` — `expected_R = pop·3 − (1−pop)`, `pop = setup_score/100`. Fine as
  FEATURES; must NEVER be used as a target (the builder's old `_proxy_realized_R` did exactly that —
  now disabled).
- `rl_alpaca_backfill.py:309` (trade-replay path) — `pop = 0.5 + 0.1·(planned_R if realized_R>0 else
  −0.5)`: **`probability_of_profit` is computed FROM the realized outcome** → a feature derived from
  the target (reverse leak). Any dataset built from this path has a contaminated `probability_of_profit`.
  **Fix:** compute `probability_of_profit` purely ex-ante (e.g., Black-Scholes PoP from delta/IV), never
  from the realized result.

## Allowed target catalogue (for Phase 4 rebuilds)

Compute exclusively from FUTURE bars / real fills:
`future_price_return`, `future_volatility`, `future_drawdown`, `trade_outcome`, `R_multiple`
(from a real stop/target walk), `max_favorable_excursion`, `max_adverse_excursion`, and
30/90/180/365-day forward returns (Buffett). Never `expected_R`, `probability_of_profit`,
`setup_score`, `4·pop−1`, or any feature aggregate.

## Status

- **alpha_entry**: clean ✓ (no model promotion — no proven edge, just leak-free).
- **alpha_exit / buffett_dca / capital_allocation**: **DO NOT TRAIN/PROMOTE** until re-targeted +
  (exit) features rebuilt from decision-time state. Tracked for Phase 4.
