# RL Enhancements from the Multi-Timeframe TDH-DQN Thesis

Source: Gregory Roy (2024), *Multi-Timeframe Algorithmic Trading Bots using Thick Data
Heuristics with Deep Reinforcement Learning* (MSc thesis). Applied to the offline
two-engine RL stack on 2026-05-31. All changes are offline/advisory — **no
execution-authority change, no RL promotion, deployment gate intact.** This builds on the
earlier [Yasin & Gill enhancements](rl_paper_enhancements.md) (normalization, overfit
guard, paper-aligned hyperparameters).

## The framework

The thesis's "more robust framework" is **TDH-DQN**: judge trend across **multiple
timeframes** (the bots read 5s / 5min / daily / weekly bars and a price-vs-EMA/SMA relation
per timeframe, then trade *with the alignment*), focus on **catalyst stocks** (§3.9 — "bots
will have limited success trading general stocks without a catalyst"), use **DQN with
experience replay + a target network** (§3.6, the two stabilisation methods), and score with
**Accumulated Percent Returns (APR)** (§3.8). Table 1 splits Thick-Data responsibilities
between bot (multi-timeframe TA, volume, price levels, momentum, scanning, volatility, risk)
and human (news, sentiment, fundamentals, Fibonacci) — our architecture already mirrors that
split (engines + market_context do the human-side inputs; the RL agents time within them).

## What the thesis recommends → what we changed

| Thesis element | Section | Our change |
|---|---|---|
| **Multi-timeframe analysis** — trend across fast/mid/slow timeframes, trade with alignment. | title, §3.10–3.11 | `rl_common.mtf_trend_features(closes, windows)` — signed price-vs-SMA over fast/mid/slow windows + an **alignment** score in [-1, 1]. Stateless; thin/absent series → neutral 0. Four features (`mtf_trend_fast/mid/slow`, `mtf_alignment`) **appended** to the alpha-entry, alpha-exit, and buffett-DCA observation sets (extractors kept index-aligned with the feature lists). Populated in `replay_buffett` (10/50/200-day SMAs on daily bars ≈ fast→long-term) and `replay_alpha` (5/20/50 on 5-min bars). |
| **Catalyst stocks** — only stocks with a real catalyst (high relative volume) are worth trading. | §3.9 | `catalyst_strength` ∈ [0,1] feature on **alpha entry** only. Populated from relative volume in `replay_alpha` and from `volume_spike_ratio` in `ingest_trade_replay_csv` (vsr 1→0, 3→1). |
| **APR (Accumulated Percent Returns)** as the performance metric. | §3.8 | `compute_metrics` now reports `apr` = accumulated per-decision return (units = R, since per-trade outcomes are R-normalized — reported under the paper's name). |
| **DQN + experience replay + target network** (the "DQN" in TDH-DQN). | §3.6, Table 6.1 | Optional SB3 `DQN` for the three **discrete** agents (entry/exit/dca) — SB3's DQN provides exactly the replay buffer + frozen target network the thesis describes. Hyperparameters follow Table 6.1 (γ=0.99, lr=1e-3, batch=86, min-ε=0.01). **Default OFF** (`AGENTS` table stays PPO) — opt in via `RL_ALGO_<AGENT>=DQN` or `RL_DISCRETE_ALGO=DQN`. `capital_allocation` is a continuous Box, so DQN is guarded back to PPO. Evaluator loads PPO **or** DQN from the same path. |
| **Risk params** (≤30 min hold, 3% stop, 4% target, no overnight). | Table 5 | Already enforced by the engines/RiskManager + the alpha-exit `no_overnight` action; no change. |

## Measured effect (offline walk-forward, deployment gate applied)

Regenerated the repo backfill (now carrying MTF + catalyst), rebuilt datasets, retrained,
re-evaluated:

| Agent | recommendation | rl expectancy | baseline | note |
|---|---|---|---|---|
| **buffett_dca** | paper_mode_only | **0.320** | 0.213 | MTF features (65.5% of rows populated from daily bars) raised expectancy from ~0.282 → 0.320 — the agent that *has* multi-timeframe data benefits, as the thesis predicts. |
| **alpha_exit** | paper_mode_only | 0.438 | 0.037 | Held its prior level; MTF dims are neutral because the offline trade-replay CSVs carry no bar series. |
| capital_allocation | shadow_mode_only | 0.084 | 0.064 | 46 samples → shadow (too few). |
| alpha_entry | do_not_deploy | 0.0 | 0.037 | Skips too much to beat baseline — unchanged, honest. |

**Training-budget note:** the new features enlarged the observation space (alpha_entry
12→17, alpha_exit 10→14, buffett 12→16). At the old 1024-step budget the larger obs underfit
and alpha_exit briefly regressed to `do_not_deploy`; raising `DEFAULT_TIMESTEPS` to 6000
restored it to ~0.44. This is proportional training for a bigger model, not gate-weakening.

**Honest limitation:** with the *currently cached* offline data sources, alpha-side MTF and
`catalyst_strength` read neutral — the trade-replay CSVs have no intraday bars and a
degenerate `volume_spike_ratio` ≈ 1.0. Both features are wired and unit-tested (catalyst
vsr=3.0 → 1.0; MTF alignment ±1 on trending series) and activate on the Alpaca `replay_alpha`
path and on live signals that carry volume/bar context.

## Safety unchanged

`live_wired=False`, shadow OFF, **no promotion**. The two `paper_mode_only` results are
recommendations awaiting explicit approval. No thresholds weakened; `MIN_DEPLOY_SAMPLES`
unchanged; DQN is opt-in and default-off; no order paths touched. Tests:
`tests/test_rl_thesis_mtf.py` (18). Full suite green.
