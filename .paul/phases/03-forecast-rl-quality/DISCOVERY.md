---
phase: 03-forecast-rl-quality
topic: RL reward shaping (intraday structure agent)
depth: standard
confidence: HIGH
created: 2026-07-03
---

# Discovery: RL Reward Shaping

**Recommendation:** Replace the naive rolling-Sharpe reward term with a **Differential Downside Deviation Ratio (DDR / "differential Sortino")** per-step reward (Moody & Saffell 2001), keep the existing structure/give-back bonuses but demote them to small tiebreakers, and re-weight the shaped terms toward what the codebase has already proven is *learnable* (give-back, MFE/MAE, vol) rather than directional P&L.

**Confidence:** HIGH - the reward mechanics are from a canonical, heavily-cited source and cross-checked against the code actually running in `rl_intraday_structure_trainer.py` / `rl_agent_core.py`. The codebase-fit argument rests on this project's own prior falsification finding (directional return is unpredictable here), which is internal evidence, so MEDIUM-HIGH on that leg.

## Objective

What we needed to settle before planning Phase 3's RL reward work:

- Is the current "Sharpe reward" the right scalar objective, or is it mis-specified?
- Which reward formulation best matches the composite optimization objective (Sharpe + consistency + drawdown control + profit factor + Calmar), *without* adding a pile of tunable knobs?
- Does the reward align with this system's actual edge, given prior research on what is and isn't predictable on this data?
- Does the choice respect the hard constraint that the RL agent is an execution/allocation policy, not an alpha inventor?

## Scope

**Include:**
- The scalar reward signal used by the intraday structure agent (`SharpeRewardCalculator` in `rl_agent_core.py:362`, consumed at `rl_intraday_structure_trainer.py:305`).
- Candidate risk-adjusted reward formulations from the direct-RL-for-trading literature.
- Interaction with the SB3 training stack (PPO default; `VecNormalize(norm_reward=True, clip_reward=10)` at `rl_intraday_structure_trainer.py:673`).

**Exclude:**
- Feature engineering / LSTM architecture (that is the *other* Phase 3 unknown, tracked separately).
- Reward for any other RL surface (`rl_trading/` allocator, alpha_entry, exit agent) - this discovery is scoped to the intraday structure trainer.
- Retraining/tuning execution (that is APPLY work, not discovery).

## Current State (what the code does today)

`SharpeRewardCalculator.update()` keeps a 60-step `deque` of per-step returns and each step returns:

```
sharpe   = mean(window) / (std(window) + eps)          # full rolling ratio, recomputed every step
blended  = 0.7 * sharpe + 0.3 * step_pnl               # arbitrary blend
reward  += blended * R_SHARPE_WEIGHT                   # R_SHARPE_WEIGHT = 0.50 (env-tunable)
```

...on top of ~20 hand-tuned structure bonuses/penalties (`R_MFE_*`, `P_GIVEBACK`, `P_EXCESSIVE_MAE`, entry/exit quality, etc.).

Two structural problems with the scalar term:

1. **Non-additive credit assignment (the real bug).** The reward at step `t` is the *entire* rolling Sharpe of the last 60 steps, not the *marginal* contribution of step `t`. Overlapping windows re-reward the same returns up to 60 times, and the agent is credited/blamed for 59 past actions it can no longer influence. PPO assumes returns decompose across timesteps; a full-window statistic as the per-step reward violates that and muddies the advantage estimates.
2. **Plain Sharpe penalizes upside variance.** This book's entire stated thesis is convex / asymmetric upside capture. Sharpe punishes a big green bar exactly as hard as a big red one, steering the policy *away* from the fat right tail it is supposed to harvest.

The docstring cites Tran et al. 2023 (DDQN + Sharpe reward). That paper's result is real, but it compares Sharpe-reward vs *raw-return* reward - it does not endorse a naive overlapping-window ratio as the per-step signal, and it is a daily/coarse crypto setting, not 1-minute equity bars.

## Findings

### Option A: Differential Sharpe Ratio (DSR)

**Source:** Moody & Saffell, "Learning to Trade via Direct Reinforcement," IEEE Transactions on Neural Networks 12(4), 2001 (and Moody, Wu, Liao, Saffell 1998). Canonical, ~2k citations.

**Summary:** Maintain exponential-moving estimates of the first and second return moments with decay `eta`:
`A_t = A_{t-1} + eta*(R_t - A_{t-1})`, `B_t = B_{t-1} + eta*(R_t^2 - B_{t-1})`.
Reward is the marginal contribution of the current return to the Sharpe ratio:
`D_t = (B_{t-1}*dA - 0.5*A_{t-1}*dB) / (B_{t-1} - A_{t-1}^2)^{3/2}`.

**Pros:**
- Per-step, additive, Markov-consistent - fixes problem (1) directly. Designed as an online RL reward.
- One parameter (`eta`), replaces window+blend+weight (three knobs today).
- Well-scaled and bounded in practice; plays nicely with `norm_reward`.

**Cons:**
- Still symmetric - penalizes upside variance (problem (2) remains).
- Needs a short warm-up before moments stabilize (same as today's 60-step warm-up).

**For our use case:** Correct machinery, wrong risk measure for a convex book. It is the right *base* to build on but not the final answer.

### Option B: Differential Downside Deviation Ratio (DDR / differential Sortino)

**Source:** Same Moody & Saffell 2001 paper - the downside-deviation variant, purpose-built for the "don't punish gains" objection.

**Summary:** Same online-moment trick, but the denominator tracks only *downside* second moment `DD^2` (returns below zero / a target). Reward:
`D_t = (R_t - 0.5*A_{t-1}) / DD_{t-1}` when `R_t > 0`, and a steeper penalty when `R_t < 0`. Upside contributes to numerator only; only losses inflate the risk denominator.

**Pros:**
- Everything DSR gives (per-step, additive, one parameter) **plus** it stops penalizing upside variance - aligns with convex-upside thesis and with the roadmap's explicit "penalize drawdowns" wording.
- Maps almost 1:1 onto the composite objective's Sortino/Calmar/drawdown emphasis.
- Empirically Moody & Saffell found DDR-trained policies matched or beat DSR on risk-adjusted terms.

**Cons:**
- Slightly more state (track downside moment separately). Marginal.
- Target/threshold for "downside" is a (minor) choice; default 0 is fine.

**For our use case:** Best fit. Corrects both defects, adds no knob count over DSR, and is the risk measure the objective actually describes.

### Option C: Keep naive rolling Sharpe (status quo)

**Source:** current `rl_agent_core.py:362`.

**Pros:** Zero work; already wired and env-tunable.

**Cons:** Both defects above. It is the baseline to beat, not a candidate.

**For our use case:** Reject as the target; retain only as the A/B control.

### Option D: Explicit multi-factor composite reward (Sharpe+PF+Calmar+consistency+decay)

**Source:** mirrors `evaluate_variants.py` composite score; ComSIA / OOPS framing in repo memory.

**Pros:** Directly encodes the deploy-gate objective into the reward.

**Cons:** 5+ weights to tune = a curve-fitting surface, in direct tension with global CLAUDE.md ("1 parameter > 10", "complexity requires evidence"). Composite metrics are the right thing to *evaluate and gate on*, the wrong thing to hand the optimizer as a dense per-step reward.

**For our use case:** Keep the composite where it belongs - the promotion gate (`evaluate_variants.py`), not the reward function. Reject as the reward.

## The codebase-specific constraint (decisive)

Prior research on this exact data (memory `project_rl_context_encoder`, 10y minute baseline, ledger `rl-cc44bc2d4af7`):

- Directional next-step return is **unpredictable** here (IC ~0.01).
- Give-back (AUC 0.75), MFE/MAE (IC ~0.30), and volatility (IC ~0.81) **are predictable**.
- Conclusion of that study: *the edge is execution / give-back / vol-sizing, not directional alpha.*

Implication for reward design: a reward dominated by directional P&L Sharpe is optimizing the one quantity the data says is noise. The give-back / MFE / MAE structure terms already in the reward are aimed at the *learnable* quantities - but they are hand-tuned constants currently swamped by `R_SHARPE_WEIGHT`. So the shaping direction is: base scalar = DDR on give-back-adjusted P&L; keep give-back/MFE/MAE as meaningful (not tiny) shaping; drop the directional-P&L emphasis. This also satisfies the hard rule that the agent is an execution policy, not an alpha source (global CLAUDE.md RL Meta-Agent Rules).

## Comparison

| Criterion | A: DSR | B: DDR (Sortino) | C: naive rolling (status quo) | D: composite |
|-----------|--------|------------------|-------------------------------|--------------|
| Per-step additive / Markov-correct | Yes | Yes | **No** | Partial |
| Preserves convex upside | No | **Yes** | No | Depends on weights |
| Knob count vs today (window+blend+weight = 3) | 1 | 1 | 3 | 5+ |
| Matches composite objective (drawdown/Sortino/Calmar) | Partial | **Strong** | Weak | Strong-but-overfit |
| Aligns with "edge = execution not direction" finding | Neutral | Neutral (via shaping) | **Fights it** | Neutral |
| Implementation cost | Low (drop-in) | Low (drop-in) | 0 | High |

## Recommendation

**Choose: Option B - Differential Downside Deviation Ratio (DDR), as a drop-in replacement inside `SharpeRewardCalculator`, with the structure/give-back bonuses retained and re-weighted.**

**Rationale:**
- Fixes the genuine defect (overlapping-window, non-additive credit assignment) that is corrupting PPO's advantage estimates - this is a correctness fix, not a preference.
- Downside-only risk preserves the convex upside the whole system is built to capture; plain Sharpe (Option A / status quo) actively fights it.
- One parameter (`eta`) replaces three, honoring the simplicity mandate; the richer composite (Option D) stays where it is defensible - the evaluation/promotion gate, not the dense reward.
- Re-weighting the shaped terms toward give-back/MFE/MAE puts the reward on the quantities this project has already proven are learnable on this data, and keeps the agent in its lane as an execution policy.

**Caveats:**
- `VecNormalize(norm_reward=True)` will interact with the new reward's scale - validate that `clip_reward=10` is not clipping the bulk of DDR values after the change; may need to disable `norm_reward` for the scalar term or recalibrate the clip.
- DDR needs a warm-up window before moments stabilize; keep the current "fall back to raw PnL for first N steps" guard.
- The give-back-adjusted P&L reframing depends on the `project_rl_context_encoder` finding holding on the *current* feature set / universe - re-confirm the give-back predictability on a fresh slice before leaning on it hard.
- This changes the reward, so any comparison must be A/B (DDR vs status-quo naive Sharpe) on held-out walk-forward windows, reusing the Phase 3 eval harness (03-03), not judged on training reward.

## Open Questions

- Should the downside target be 0 or a small positive MAR (minimum acceptable return per step)? - Impact: low (default 0, revisit only if flat-return churn appears).
- Does `eta` want to be constant or annealed over training? - Impact: low (start constant ~0.01-0.04, tune only if unstable).
- Is give-back-adjusted P&L better fed as the DDR input series, or kept as a separate additive penalty (`P_GIVEBACK`) as today? - Impact: medium (affects how much of the give-back signal the scalar reward carries); resolve in planning with a quick ablation.

## Quality Report

**Sources consulted:**
- Moody & Saffell, "Learning to Trade via Direct Reinforcement," IEEE Trans. Neural Networks 12(4), 2001 - DSR and DDR derivations (knowledge, canonical).
- Moody, Wu, Liao, Saffell, "Performance Functions and Reinforcement Learning for Trading Systems and Portfolios," 1998 - original DSR.
- Tran, Pham-Hi & Bui, "Parameter Optimization / DDQN + Sharpe reward," Algorithms 2023 - already cited in repo (`rl_agent_core.py:369`, `TRAN_DDQN_HYPERPARAMS`).
- Repo code, read 2026-07-03: `rl_agent_core.py:362-431` (SharpeRewardCalculator), `rl_intraday_structure_trainer.py:80-430` (reward assembly), `:673` (VecNormalize).
- Repo memory: `project_rl_context_encoder` (predictability finding), global CLAUDE.md RL Meta-Agent Rules + Simplicity principle.

**Verification:**
- "Current Sharpe reward is a naive overlapping-window ratio, not differential": Verified by reading `SharpeRewardCalculator.update()`.
- "Reward is added per-step on top of structure bonuses with weight 0.50": Verified at `rl_intraday_structure_trainer.py:303-308`.
- "DSR/DDR are per-step additive online estimators": Verified via Moody & Saffell (standard result).
- "Directional return unpredictable, give-back/vol predictable on this data": From this project's own prior study (internal evidence, not independently re-run in this discovery).

**Assumptions (not verified):**
- The `project_rl_context_encoder` predictability numbers still hold on the current feature set/universe.
- SB3 PPO advantage estimation is materially degraded by the non-additive reward (strongly implied by theory; not measured on this specific run).

---
*Discovery completed: 2026-07-03*
*Confidence: HIGH (mechanics) / MEDIUM-HIGH (codebase-fit leg)*
*Ready for: /paul:plan 03-01*
