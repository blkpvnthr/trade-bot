# Options-Sentiment Alpha Discovery (Research Layer)

Promotes Alpha Vantage options-flow data into the hidden-alpha discovery
framework: a new `options_sentiment` feature family → latent compression →
validation against forward returns → an Approved Alpha Registry → attribution.

> **Two separate responsibilities — do not merge them.**
> - **Live booster overlay** (`options_sentiment_av.py` → `options_booster.py`): a real-time *advisory* layer that re-ranks option contracts. Unchanged by this work.
> - **Alpha discovery** (this document): an offline *research/training* layer that decides which options-derived features earn a place in the RL state. It can never place, size, or gate a live trade.
>
> The research layer **names** the booster only to document the boundary; it never imports it (asserted by `test_research_layer_independent_of_booster`).

---

## 1. Data flow

```
AlphaVantageClient                       (reused: get_put_call_ratio / get_volume_oi_ratio /
  │                                        get_realtime_options → greeks + IV)
  ▼
options_feature_builder.py               build_options_sentiment_features()
  │   options_sentiment family             → candidate factors + interactions
  ▼
latent_feature_discovery.py              discover_options_latent_factors()
  │                                        → latent_options_factor_1..3,
  │                                          latent_options_sentiment_state, latent_gamma_state
  ▼
options_alpha_discovery.py               validate → promote → attribute
  │   ├─ news_price_reaction_model.validate_feature  (forward returns, MFE/MAE, Sharpe, …)
  │   ├─ Approved Alpha Registry  (reports/options/options_approved_features.{json,csv})
  │   └─ Attribution             (reports/options/options_alpha_contribution.csv)
  ▼
rl_common.py                             OPTIONS_SENTIMENT_FEATURES (5 latents),
  │                                        gated RL_OPTIONS_SENTIMENT_FEATURES (default OFF)
  ▼
rl_feature_exporter.py / predictive_state_compression.py
                                         schema export + FEATURE_GROUPS diagnostic
```

`feature_store.py` gains `write_family_report(rows, family, names)` so a
family-scoped snapshot can be emitted to
`reports/alpha_discovery/latest_options_sentiment_features.{csv,json}`.

---

## 2. The `options_sentiment` feature family

Produced by `alpha_discovery.options_feature_builder.build_options_sentiment_features(chain, pcr, voi, spot, as_of, context)`.
All factors are bounded and neutral-0.0 on missing data.

### Candidate factors (`OPTIONS_SENTIMENT_FACTORS`)

| Factor | Range | Meaning |
| --- | --- | --- |
| `put_call_pressure_factor` | [-1,1] | Net call(+) vs put(-) pressure (from PCR signal) |
| `call_pressure_factor` | [0,1] | Call-demand strength |
| `put_pressure_factor` | [0,1] | Put-demand strength |
| `volume_oi_activity_factor` | [0,1] | Volume/OI participation |
| `unusual_options_activity_factor` | [0,1] | Fresh speculative flow (vol/OI ≫ 1) |
| `gamma_pressure_factor` | [-1,1] | OI-weighted net gamma exposure (dealer-flow proxy) |
| `delta_pressure_factor` | [-1,1] | OI-weighted net delta exposure |
| `iv_skew_factor` | ~[-0.5,0.5] | Mean put IV − mean call IV (downside-protection demand) |
| `iv_term_structure_factor` | [-1,1] | Near-term (≤30d) vs far-term IV slope (event/stress premium) |
| `option_sentiment_factor` | [-1,1] | Composite: 0.50·PCR-sentiment + 0.20·delta + 0.15·gamma − 0.15·skew |

### Interaction factors (`OPTIONS_SENTIMENT_INTERACTIONS`)

Element-wise products with a `context` dict (each neutral-0.0 when context absent):

| Interaction | = |
| --- | --- |
| `options_sentiment_x_price_momentum` | `option_sentiment_factor × price_momentum` |
| `options_sentiment_x_news_sentiment` | `option_sentiment_factor × news_sentiment` |
| `options_sentiment_x_relative_volume` | `option_sentiment_factor × relative_volume` |
| `gamma_pressure_x_breakout_strength` | `gamma_pressure_factor × breakout_strength` |
| `put_call_ratio_x_trend_strength` | `put_call_pressure_factor × trend_strength` |
| `volume_oi_ratio_x_volatility_expansion` | `volume_oi_activity_factor × volatility_expansion` |

The aggregate `build_options_features(...)` folds the new family in **additively** —
every legacy `realtime_*` key is preserved (no regression).

---

## 3. Latent options-flow states

`alpha_discovery.latent_feature_discovery.discover_options_latent_factors(records)`
fits PCA in-sample over the options factor block and emits, per record:

- `latent_options_factor_1..3` — PCA of the full 10-factor block
- `latent_options_sentiment_state` — 1st latent of the pressure/sentiment sub-block
- `latent_gamma_state` — 1st latent of the gamma/delta/IV sub-block

`LatentFeatureDiscovery` gained a `prefix` parameter (default `"realtime"`, so the
existing `latent_realtime_factor_*` behavior is unchanged).

---

## 4. Validation, promotion, attribution

Driver: `alpha_discovery.options_alpha_discovery.run(records=..., dataset_path=...)`.

### Validation
Reuses `news_price_reaction_model.validate_feature` — for each candidate it
measures forward-return IC at **5m / 15m / 30m / 60m / EOD**, plus **MFE, MAE,
win-rate, expectancy, profit-factor, Sharpe, Sortino**. Output:
`reports/options/options_validation_report.csv`.

A "labeled record" is a dict with each feature value plus realized outcomes:
```json
{"put_call_pressure_factor": 0.4, "iv_skew_factor": 0.03,
 "ret_5m": 0.001, "ret_15m": 0.002, "ret_30m": 0.0, "ret_60m": 0.004,
 "ret_eod": 0.01, "mfe": 0.012, "mae": -0.004}
```

### Promotion (Approved Alpha Registry)
A **conservative AND-gate** — every condition must pass (env-tunable):

| Gate | Env | Default |
| --- | --- | --- |
| Min samples | `OPTIONS_ALPHA_MIN_SAMPLES` | 30 |
| Per-horizon \|IC\| threshold | `OPTIONS_ALPHA_MIN_ABS_IC` | 0.03 |
| Horizons clearing it | `OPTIONS_ALPHA_MIN_IC_HORIZONS` | 2 |
| IC significance \|IC\|·√n | `OPTIONS_ALPHA_MIN_IC_TSTAT` | 2.0 |
| Min Sharpe | `OPTIONS_ALPHA_MIN_SHARPE` | 0.30 |
| Min profit factor | `OPTIONS_ALPHA_MIN_PROFIT_FACTOR` | 1.10 |
| Min expectancy | `OPTIONS_ALPHA_MIN_EXPECTANCY` | 0.0 |

The √n significance gate is what stops a spurious small-sample correlation from
being promoted (e.g. a noise feature with IC≈0.085 at n=120 → t≈0.93 → rejected,
while a real signal with IC≈0.52 → t≈5.7 → approved). Output:
`reports/options/options_approved_features.{json,csv}`. **Nothing is promoted when
data is insufficient** (`status: no_data`).

### Attribution
`reports/options/options_alpha_contribution.csv`, columns:

| Column | Proxy |
| --- | --- |
| `Feature` | feature name |
| `Sharpe Contribution` | signed-return Sharpe |
| `PnL Contribution` | `expectancy × n` (total signed return, return units) |
| `Trade Selection Contribution` | `win_rate − 0.5` (edge over a coin-flip selector) |
| `Timing Contribution` | `MFE / (MFE + |MAE|)` (favorable-excursion capture), else best \|IC\| |

---

## 5. RL wiring (default-OFF — preserves model width)

`rl_common.py` adds:
```python
OPTIONS_SENTIMENT_FEATURES = ["latent_options_factor_1", "latent_options_factor_2",
                              "latent_options_factor_3", "latent_options_sentiment_state",
                              "latent_gamma_state"]
OPTIONS_SENTIMENT_ENABLED = os.getenv("RL_OPTIONS_SENTIMENT_FEATURES", "0") == "1"
```
Only the **compressed latents** (5) enter the observation — the raw factors stay
research-layer. Appended to `ALPHA_ENTRY_FEATURES` **only when the gate is set**,
mirroring the PSC / AV-realtime / news / analytics pattern. With the gate OFF the
entry width is unchanged (`len(ALPHA_ENTRY_FEATURES) == len(alpha_entry_features({}))`
still holds), so existing trained models load unchanged.

**To activate** (same procedure as the other gated families): set
`RL_OPTIONS_SENTIMENT_FEATURES=1`, rebuild `alpha_entry_dataset` so the latent
columns are populated, and retrain the alpha-entry agent at the new width
(current 44 → 49). Only enable features that the Approved registry has promoted.

`predictive_state_compression.py` gains an `options_sentiment` group in
`FEATURE_GROUPS` (diagnostic coverage only — it does not change PSC width or math).
`rl_feature_exporter.py` reports the family in the RL schema snapshot.

---

## 6. Example run

```python
from alpha_discovery import options_alpha_discovery as oad

summary = oad.run(dataset_path="data/rl_training/alpha_entry_dataset.parquet",
                  horizon_map={"future_return_5m": "ret_5m", "future_return_15m": "ret_15m"},
                  mfe_col="max_favorable_excursion_15m", mae_col="max_adverse_excursion_15m")
# -> {"status": "ok", "approved_count": N, "approved_features": [...],
#     "validation_report": ".../options_validation_report.csv", ...}
```

Until a labeled dataset that contains the `options_sentiment` columns **and**
forward-return/MFE/MAE labels exists (a dataset rebuild), `run()` reports
`no_data` and promotes nothing — by design.
