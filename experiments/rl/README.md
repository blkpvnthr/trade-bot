# experiments/rl — RL experiment ledger (append-only)

Written by `rl_trading/ledger.py`. **Failures are never deleted** (Absolute Rule).

- `rl_experiments.jsonl` — source of truth, one JSON record per line
- `rl_experiments.csv` — flattened view
- `rl_summary.md` — auto-generated rolling summary table
- `universes/<experiment_id>_universe.{json,csv}` — frozen PIT universe per experiment
  (selected + rejected + reasons + scores + config hash + data fingerprint)

## Record schema

`experiment_id, timestamp, git_sha, hypothesis, changed_variable, reward, feature_set,
universe, train_window, validation_window, oos_window, seed, train_metrics,
validation_metrics, oos_metrics, benchmark_comparison, passed, rejection_reason,
next_experiment`

Every experiment is selected/rejected on **untouched OOS** metrics only. Universe
selection and hyperparameter tuning use train/validation windows only — never OOS returns.
