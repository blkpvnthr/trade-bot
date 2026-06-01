# Quantum-inspired predictive state compression

Source: Elliott, Yang, Binder, Garner, Thompson & Gu, *"Extreme dimensionality reduction with
quantum modelling"* (arXiv:1909.02817, 2020). Engine: `predictive_state_compression.py`.

## The principle (faithfully adapted)

The paper's framework is **computational mechanics**. The minimal classical model of a process
is the **ε-machine**, whose **causal states** are equivalence classes of pasts with *identical
conditional future distributions* — `f(x⃖)=f(x⃖') ⟺ P(X⃗|x⃖)=P(X⃗|x⃖')` — the **minimal
sufficient statistic of the past for the future**, which the authors note "closely mirror the
belief states of reinforcement learning."

- **Statistical complexity** `C_μ = −Σ P(s) log₂ P(s)` (Shannon entropy of the causal states).
- **Quantum memory** `C_q = −Tr[ρ log₂ ρ]` (von Neumann entropy of `ρ = Σ P(s)|ψ_s⟩⟨ψ_s|`), with
  the amplitude encoding `|ψ_s⟩ = Σ_k √P(k|s)|k⟩` whose overlap `⟨ψ_s|ψ_s'⟩ = Σ_k √(P(k|s)P(k|s'))`
  is the **Bhattacharyya fidelity** of the two future distributions — non-orthogonal exactly when
  futures overlap. Provable advantage: **`C_q ≤ C_μ`** ("efficient isolation of predictive features").

**Trading translation:** thousands of raw features collapse to a handful of latent **causal
market states** carrying the predictive information about *future returns* while discarding
historical noise. Objective: **max `I(latent ; future_return)`, min dimension + irrelevant info.**

## What the engine does (offline, leakage-safe)

Inputs: the rich `alpha_entry_dataset` feature space (34 cols across price/technical, volume,
volatility, news/sentiment, options, market-context, sector-relative, opening-range, premarket
groups) + the **future return** target `realized_R`. Time-ordered 70/30 split; **every map is
fit IN-SAMPLE, predictive information measured OUT-OF-SAMPLE** (no look-ahead).

Compressors (graceful — sklearn/torch when present, numpy fallbacks otherwise):
- `pca` / `incremental_pca` / `sparse_pca` — unsupervised (numpy-SVD fallback).
- `pls` — supervised predictive components (information-bottleneck surrogate; max covariance with future return).
- `autoencoder` — compact torch bottleneck (optional).
- **`causal_states`** — the quantum-inspired centerpiece: belief states `P(future_bin|x)` (HistGradientBoosting, k-NN fallback) → KMeans causal states → `C_μ`, `C_q` (von Neumann), named states.

## Result on the current dataset (honest)

| method | dim | OOS MI (nats) | retain | OOS decile-spread | recommendation |
|---|--:|--:|--:|--:|---|
| pca / sparse / incremental | 4 | ~5.26 | ~1.41 | **−2.0 to −2.4** | shadow_only |
| **pls** | 4 | 5.26 | 1.41 | **+3.85** | **approved** |
| autoencoder | 4 | 5.06 | 1.35 | +1.14 | approved |
| causal_states | 5 | 2.21 | 0.59 | +3.71 | shadow_only |

- **Quantum compression demonstrated** (the paper's core): 8 classical causal states
  `C_μ = 2.42 bits` → `C_q = 0.57 bits` (**ratio 0.235, saving 1.85 bits**, `D_μ=3 → D_q=1`).
  `I(state;future)_oos = 0.56 bits`. The overlapping future distributions collapse to ~1
  effective quantum dimension — `C_q ≤ C_μ` reproduced.
- **Discovered states are strongly differentiated by future return**, e.g. *volatility_contraction*
  (P=0.46, future R −0.89, the big losing cluster) vs. states with future R +2.2/+2.7;
  *momentum_continuation* +1.24; *momentum_exhaustion* −0.08.
- **Honest gating:** unsupervised PCA retains variance but has *negative* OOS decile-spread
  (variance ≠ predictiveness) → shadow_only. Supervised **PLS approved** (+3.85 vs raw −0.07).
  Causal-states is strongly predictive (decile +3.71) and achieves quantum compression, but its
  5-dim belief vector scores low on the dimension-biased summed-MI *retain* metric → shadow_only;
  the report carries both metrics so the trade-off is explicit.

## Outputs

`daily_system_outputs/<tag>/predictive_state_compression.json` (per-method dim / OOS MI / retain
/ decile-spread / recommendation; causal-state `C_μ`,`C_q`,ratio, named states; best approved
method; `predictive_information_score`; `leakage_safe`). Validated OOS embeddings →
`data/rl_training/alpha_entry_psc_embeddings.parquet`.

## RL integration (gated by approval)

Per the brief, the RL should consume **Raw + Approved Alpha + Predictive State Embeddings**.
`rl_common.PSC_FEATURES` (`predictive_state_1..3`, `latent_market_state`,
`future_return_embedding`, `state_confidence`, `predictive_information_score`) append to
`ALPHA_ENTRY_FEATURES` **only when `RL_PSC_FEATURES=1`** (default off → width stays 34; on → 41,
verified). This preserves the validation discipline: no unvalidated latent state enters the
observation. Activation path (executable):

```bash
python3 predictive_state_compression.py --emit-full-embeddings pls   # → alpha_entry_dataset_psc.parquet (leakage-safe: fit IS, transform all)
# point the entry env's dataset at *_psc.parquet, then:
RL_PSC_FEATURES=1 python3 train_rl_agents.py                          # retrain at the 41-wide obs
python3 evaluate_rl_agents.py                                         # Sharpe-primary gate decides promotion
```
A compressed state is **approved only if** it improves OOS predictive power, survives walk-forward,
lifts a downstream metric (decile-spread / Sharpe / expectancy), reduces dimensionality, and is
leakage-free — exactly the gate the engine reports.

## Tunables

`PSC_TARGET`, `PSC_N_BINS`, `PSC_N_STATES`, `PSC_N_COMPONENTS`, `PSC_TRAIN_FRAC`, `PSC_RETAIN_TAU`,
`RL_PSC_FEATURES`.
