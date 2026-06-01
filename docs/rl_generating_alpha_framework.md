# "Generating Alpha" framework → RL system integration

Source: Pillai, Ajith & Sumesh, *"Generating Alpha: A Hybrid AI-Driven Trading System
Integrating Technical Analysis, Machine Learning and Financial Sentiment for Regime-Adaptive
Equity Strategies"* (ComSIA 2026; arXiv:2601.19504v1). Reported: Sharpe 1.68, max DD −15.6%,
win rate 61.5%, 135% / 24 months vs. SPX 53%.

The paper's contribution is **not** a new signal — it is a *discipline*: fuse multiple
independent modalities into a confluence score, and treat sentiment + regime as **gates that
suppress trades, never as buy signals**. That discipline maps directly onto this repo's
mandate (CLAUDE.md: "sentiment as a risk shield", "regime protections", "never remove safety
controls to force trades"). Every addition below is **purely suppressive** — it can only
remove a candidate entry, never add or force one — so all live exits, kill switches, and risk
limits are untouched.

## Pillar → implementation map

| Paper pillar (Alg. 1 / §C–§G) | Where it lives now |
|---|---|
| EMA50/EMA200 ratio (interaction term, §C) | `rl_common.ema50_200_ratio` → `ga_ema_ratio` |
| MACD histogram (MACD − signal, §C) | `rl_common.macd_histogram` → `ga_macd_hist` |
| RSI(14) mean-reversion (§C) | `rl_common.rsi` → `ga_rsi` |
| Bollinger %b volatility position (§C) | `rl_common.bollinger_pctb` → `ga_bollinger_pctb` |
| Per-stock regime `Rₜ = SMA₂₀(returns)` on/off switch (§F) | `rl_common.regime_switch` → `ga_regime_switch` |
| FinBERT sentiment `Sₜ = mean(P₊ − P₋) ∈ [−1,1]` (§D) | `rl_common.sentiment_to_st`, `rl_strategy_overlay.load_sentiment_map` → `ga_sentiment_st` |
| Hybrid confluence score, entry needs `score ≥ 2` (Alg. 1 Step 3–4) | `rl_common.generating_alpha_confluence`, gated in `rl_strategy_overlay.passes_gates` |
| Sentiment gate: suspend/exit if `Sₜ < −0.70` (§D) | `passes_gates` → `sentiment_gate` (`STRAT_SENT_GATE_MIN`, default −0.70) |
| Regime gate: long-only off in bear regime (§F) | `passes_gates` → `regime_gate` |
| ATR risk-parity sizing `min(0.01·Cash/ATR, 0.1·Cash/P)` (§G) | already present: `rl_strategy_overlay.position_risk` vol-targeting |

## Data flow

1. **Bars → technicals.** `rl_alpaca_bar_enrichment.bar_features` and `RegimeContext.regimes`
   compute `ga_ema_ratio / ga_macd_hist / ga_rsi / ga_bollinger_pctb / ga_regime_switch` from
   each symbol's **daily closes up to the event timestamp** (no look-ahead). Missing/thin data
   → neutral defaults (`ENRICH_DEFAULTS`), never fabricated.
2. **Sentiment (offline, read-only).** `load_sentiment_map` reads
   `system-data/*/*_llm_universe_sentiment_report.json` (`reports[].sentiment_score ∈ [0,1]`),
   rescales each to the paper's `Sₜ ∈ [−1,1]` via `2·s − 1`, keyed by `(date_tag, symbol)`.
   A **missing** symbol → neutral `0.0` (missing never reads as bearish); an **explicit** 0.0
   LLM score → `Sₜ = −1.0` (genuinely bearish).
3. **Features.** `GA_FEATURES` are appended to `ALPHA_ENTRY_FEATURES`; `alpha_entry_features`
   appends `_ga_vals` in matching order (centered/scaled). Vector width 22 → **28**.
4. **Gates.** `passes_gates` applies sentiment/regime/confluence as suppressive filters inside
   `rl_strategy_overlay.select`; on by default in the *improved* policy, off in *baseline*.

## Validation (offline, IS-tuned / OOS-reported)

`python3 rl_strategy_overlay.py` prints an `[A2] GATE EDGE` block (no-gates vs. gated, R-based,
no leak). Verified this cycle:

- **Gates fire correctly** (adversarial unit tests): bearish `Sₜ`, bear regime, and
  `confluence < 2` are each rejected; a clean confluent row passes.
- **Independent teeth**: applied to all 3,662 OOS rows *without* the quality pre-filter, the
  gates remove **29.8%** (confluence gate). On the *quality-gated* top-9% slice the gates are
  currently a near no-op — because the quality score already concentrates on high-confluence,
  non-bearish setups, so the framework's confluence requirement is already satisfied there.
  This is the intended, honest result: the gates and the quality score agree.

## Activation status / next steps

- The **confluence gate** is active on the current dataset.
- The **sentiment** and **per-stock regime** gates are wired + unit-proven but currently
  near-inert on the *legacy* `alpha_entry_dataset`, because (a) sentiment coverage is thin
  (100 / 3,662 OOS rows matched a report, and the LLM score rarely dips below 0.15), and
  (b) legacy rows predate the `ga_regime_switch` enrichment.
- **To fully activate:** rebuild the datasets so the new `ga_*` columns populate —
  `python3 rl_alpaca_bar_enrichment.py` (regime context) then
  `python3 rl_training_dataset_builder.py`. RL models trained on the old 22-wide observation
  must be **retrained** on the new 28-wide vector (standard append-and-retrain; the deployment
  gate in `rl_common.deployment_recommendation` protects against regressions).

## Tunables (env vars)

| Var | Default | Effect |
|---|---|---|
| `RL_SENT_GATE_MIN` / `STRAT_SENT_GATE_MIN` | `-0.70` | Suspend entries below this `Sₜ` (paper §D) |
| `RL_MIN_CONFLUENCE` / `STRAT_MIN_CONFLUENCE` | `2` | Confirmations required to enter (Alg. 1) |
| `STRAT_SENTIMENT_GATE` / `STRAT_REGIME_GATE` / `STRAT_CONFLUENCE_GATE` | `1` | Toggle each gate (improved policy) |
