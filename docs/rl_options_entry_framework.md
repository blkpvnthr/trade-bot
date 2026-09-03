# Options entry & placement → RL features (Opening Range + OOPS)

Two papers, one coherent goal: teach the options-entry RL agent (`rl_env_alpha_entry`) *where* to
enter and *which/how-much* contract to place — using signals it previously could not see.

| Paper | Goal | Status before | What was added |
|---|---|---|---|
| Bysshe, *The Opening Range Handbook* (2009) | "find entries into good options positions" | ORB levels existed (`alpha_call_engine._compute_orb`, `scalping_execution_layer._detect_opening_range`) but the **RL had zero OR awareness** | 4 OR features encoding the book's 3 questions |
| Faias & Santa-Clara, *Optimal Option Portfolio Strategies* (2011) | "teach the RLs how to place options contracts" | OOPS fully implemented (`oops_option_optimizer.py`); RL saw only the collapsed scalar `oops_score` | 2 OOPS payoff-shape features (the CRRA core + tail) |

## Opening Range features (Bysshe's three questions)

The handbook reduces "is this a tradable entry?" to three questions about the opening range (first
30 min H/L). Encoded in `rl_common.compute_or_assessment` / `opening_range_features`:

| Question | Feature | Meaning |
|---|---|---|
| Q2 — bullish/bearish bias to the OR? | `or_position` ∈ [−1.5, 1.5] | signed position of current price within/beyond the OR (>1 ≈ at/above OR high) |
| Q2 — breakout state | `or_breakout` ∈ {−1, 0, +1} | price above OR high (+1) / below OR low (−1) / inside (0) |
| Q3 — is OR volume unusually high? | `or_volume_confirm` ∈ [0, 1] | OR-period relative volume (interest → explosive potential) |
| Q1 — well-defined S/R at the extreme? | `or_level_quality` ∈ [0, 1] | fraction of OR bars that tested an extreme (touch count), with OR-tightness-vs-ATR fallback |

Statistical edge the book cites: the day's high/low forms in the first 30 min ~35% of the time vs
~8% random — so OR extremes are high-probability inflection levels. Q1 is the most important
("if you only focus on one question, this should be the one").

**Computed** in `rl_alpaca_bar_enrichment.bar_features` from the session-anchored (09:30–10:00 ET)
slice of the intraday 5m/15m frame (`_opening_range_slice`); H/L/volume/quality from the OR window,
position/breakout vs. the latest price. Or derived at feature time by `rc._or_vals` from an event's
`orb_high`/`orb_low`/`last_price` (which `alpha_call_engine` already produces). Neutral when not
session-anchored.

## OOPS payoff-shape features (the CRRA core the scalar hides)

`oops_option_optimizer.py` maximizes CRRA power-utility (γ=10) over a simulated payoff distribution
and collapses it to a 0–100 `oops_score`. That scalar blends away the two quantities the paper's
utility actually trades off, so they're now exposed directly (`rc._oops_extra_vals`):

| Feature | Source field | Why |
|---|---|---|
| `oops_cert_equiv_return` | `certainty_equivalent_return` | the risk-adjusted core of the OOPS objective (guaranteed-return equivalent of the option's risky payoff) |
| `oops_prob_total_loss` | `probability_total_loss` | the tail the γ=10 utility penalizes hardest — the downside of *placing* this contract |

This lets the entry agent learn contract placement on the same risk-aware terms the paper
optimizes, instead of a single blended score.

## Observation-width change & activation

`ALPHA_ENTRY_FEATURES`: **28 → 34** (`+OR_FEATURES(4) +OOPS_EXTRA_FEATURES(2)`). Vector width
verified to match the name-list (dataset builder maps by index). Neutral defaults keep old events
back-compatible. As with the GA features, **RL models must be retrained** on the new 34-wide
observation; the Sharpe-primary deployment gate guards against regressions. Full activation: rebuild
datasets so the new columns populate (`rl_alpaca_bar_enrichment.py` → `rl_training_dataset_builder.py`),
ensuring events carry `orb_high`/`orb_low` (alpha engine) and OOPS fields (`certainty_equivalent_return`,
`probability_total_loss`) for live setups.

## Validation (this cycle, offline)

- OR math: breakout-bull → `or_position 1.5, or_breakout +1, or_volume_confirm 0.75, or_level_quality 0.6`;
  breakdown-bear → negative; inside/degenerate → neutral.
- Session-anchored enrichment: 5m frame opening at 09:30 ET → OR features computed, `or_breakout`
  fires +1 when the latest price clears the OR-window high.
- Dataset builder emits all 6 new columns (row width 38 incl. metadata); feature vector 34 = 34.
