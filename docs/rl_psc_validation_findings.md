# Pre-promotion validation of the causal-state compression — findings

Rigorous validation of the PPO `paper_mode_only` result for the causal-state predictive
compression, run via `psc_validation.py` (leakage-safe, multi-fold). **Bottom line: the headline
result does not survive scrutiny — do NOT promote beyond paper.**

## 1. The paper-mode "win" was TARGET LEAKAGE, not alpha (headline finding)

A leakage-safe multi-fold classifier benchmark on the raw 34 features gave **AUC ≈ 0.999** for
predicting `realized_R > 0` — implausibly perfect for trading. The cause, confirmed by a
univariate contamination scan:

- **`probability_of_profit` has univariate AUC 0.924.** PoP>0.8 ⇒ **99.4%** realized win rate;
  PoP<0.45 ⇒ **3.5%**. In the offline/replay dataset the realized outcome is near-deterministically
  encoded by this feature (and the `_proxy_realized_R = 4·pop − 1` path), so it leaks the target.

With `probability_of_profit` + `expected_R` excluded, **AUC collapses to ~0.53–0.59** (barely above
random) for *every* model. So the PPO `rl_exp 0.81`, the causal `+3.7` decile-spread, and the gate's
`expectancy_drawdown_and_sharpe_improved` verdict were **all riding one leaky feature**. The
deployment gate cannot detect target leakage — a gap worth noting.

## 2. The causal representation does NOT materially beat raw (hypothesis not supported)

Leak-free, 5 expanding folds, embeddings re-fit per-fold on TRAIN only (decile-spread = top−bottom
decile realized_R, threshold-free):

| feature set | AUC (best clf) | decile-spread R |
|---|--:|--:|
| raw34 | 0.587 | 0.13 – 0.96 |
| raw+causal | 0.59 | 0.79 – 1.06 |
| causal_only | 0.58 | 0.80 – 0.85 |
| raw+causal − future_return_embedding | 0.59 | 0.78 – 1.02 |
| raw+pls | 0.59 | 0.13 – 0.94 |

- The causal block gives a **small, noisy** lift (AUC +~0.05 for the *linear* model; decile ~1.0R
  vs ~0.1–0.96R) — not "materially more informative."
- **`future_return_embedding` is NOT the alpha source**: `raw+causal` ≈ `raw+causal−FRE`
  (AUC 0.587 vs 0.587; decile 1.06 vs 1.02). Removing it changes nothing — contradicting the
  expected ablation.
- Honest interpretation: the causal compression is, mechanically, a leakage-safe **stacked return
  predictor** (a HistGBM `E[future R|x]`); its OOS power IS that predictor's OOS power, which here
  is marginal.

## 3. Data span cannot support multi-regime validation

The offline data is **~6 weeks (2026-04-14 → 2026-05-31, 35 trading days, one recent regime)**. The
proposed 2018→2025 cross-regime folds are **impossible without a real multi-year backfill**. The
harness tests *temporal stability within one window* only — a DATA limitation, not a code one.

## 4. Live-compatibility audit

- **No live risk.** `run.py` uses the RL model only via `_rl_shadow_cycle()` → `rl_shadow_mode`,
  which is **OFF by default** (`ENABLE_RL_SHADOW_MODE=true` required) and **advisory-only** ("the
  live trading loop ignores them entirely; no order"). The model never enters the order path.
- **Width coupling is a real landmine.** `rl_env_alpha_entry.FEATURES` is now **43-wide**
  (`ALPHA_ENTRY_FEATURES` 34 + 9 `OOPS_FIELDS` from the in-progress OOPS-utility integration). Every
  saved alpha_entry model (34-wide pre-PSC, 41-wide PSC) is **stale** vs this env and will mismatch.
  `recommend()` catches the shape error and silently returns `"wait"` — inert, not a crash, but a
  silent-failure observability gap.

## Actions taken

- Reverted the canonical `alpha_entry.zip` to the pre-PSC **34-wide** model and the dataset to
  **38 cols** (removed the leaky causal columns I had merged). PSC/41-wide artifacts archived at
  `rl_models/rl_v2/_psc41_archive_20260531/` and `data/rl_training/_psc_archive_20260531/`;
  pre-PSC backups at `*_backup_pre_psc_20260531/`.
- `psc_validation.py` now **excludes outcome-coupled features by default** (`PSC_VAL_INCLUDE_LEAKY=1`
  to reproduce the inflated artifact) and **prints a contamination scan** flagging any feature with
  univariate |AUC| > 0.80.

## Recommendation (before any promotion)

1. **Fix the target leakage first.** Ensure `realized_R` in the offline dataset is a genuine
   forward outcome decoupled from `probability_of_profit`/`expected_R` (the replay/proxy path
   currently couples them). Re-run the contamination scan until no feature exceeds |AUC| 0.80.
2. **Get a multi-year, multi-regime backfill**, then run the multi-window walk-forward.
3. **Re-establish env/model width consistency** — retrain against the current 43-wide env (OOPS
   fields) on a leak-free target; the deployment gate verdict is only meaningful once the target is clean.
4. Only after a leak-free signal clears the classifier bar is a full `run.py` portfolio sim
   (slippage/spread/delays/caps) and multi-week paper trading warranted. A portfolio sim on the
   current marginal/leak-tainted signal would be misleading, so it is deferred.

## UPDATE — leak fixed + retrained (2026-06-01)

**Root cause (refined):** `realized_R` was a genuine forward intraday-bracket outcome, but (a) the
dataset builder's `_proxy_realized_R` FABRICATED it as `expected_R = 4·pop−1` for ~10% of rows
(pure circular leak, AUC 1.0), and (b) `probability_of_profit = setup_score/100`, and `setup_score`
was entangled with the same-day bracket outcome (AUC 0.89 even on genuine rows).

**Fix applied:**
- `rl_training_dataset_builder._proxy_realized_R` now returns `None` — outcomes are NEVER
  synthesized from features (rows tagged `realized_R_source`).
- **`rl_clean_target.py`** re-targets `realized_R` as the **forward 2-day underlying return from
  real cached daily bars** (`realized_R = clip(fwd_return / 0.005)`), keyed by (symbol, date_tag) —
  independent of `setup_score`. Rows without forward bars are dropped (54,652 → **12,164**), never
  fabricated. Prior dataset archived at `data/rl_training/_pre_clean_target_archive/`.

**Result (the leak is gone, and so is the apparent edge):**
- `probability_of_profit` → win AUC **0.924 → 0.515** (random); **no feature exceeds 0.55**.
- Leak-free multi-fold ablation: **every** feature set (raw34 / raw+causal / causal_only /
  raw+causal−FRE / raw+pls) sits at **AUC ~0.48–0.51** — causal/quantum gives NO advantage over raw.
- Retrained `alpha_entry` (12,164 samples) → deployment gate **`do_not_deploy`
  (`does_not_beat_baseline_out_of_sample`)**, rl_exp 1.72 vs baseline 2.14 (both positive only from
  up-window beta, not alpha).

**Conclusion:** confirmed — the paper-mode win was target leakage. On a leak-free target there is no
demonstrable predictive edge in any representation on this ~6-week dataset, and the gate correctly
refuses promotion. Genuine edge work now requires a multi-year, multi-regime dataset with
real (non-synthesized) forward outcomes. NOTE: only `alpha_entry_dataset` was re-targeted; the
exit/dca/allocation datasets still use matched-exit/proxy outcomes and warrant the same audit.
