# Orthogonal feature sources (research branch)

## Why this branch exists

Three independent symbolic-search backbones now agree:

| generator | result |
|---|---|
| AlphaEvolve (evolutionary) | accepted alphas are volatility/giveback-dominated; no ensemble synergy |
| gplearn (symbolic regression) | VOLATILITY_REDISCOVERY: 0 orthogonal alphas, no breadth/combined-IC lift |
| EO 1.3.1 (genetic programming) | finds diverse stable alphas, but synergy ties IC-ranking -- one factor family |

The constraint is no longer **"can we search formulas better?"** It is **"do we have
enough independent input information to discover orthogonal alphas?"** "More optimizer =
more alpha" is now falsified twice. The next edge has to come from new information.

**So: no new evolutionary engines until the feature set is expanded.** AlphaEvolve,
gplearn, and EO stay as the candidate generators; this branch finds the *inputs* worth
giving them.

## The gate (`feature_gate.py`)

Every new feature source must clear one falsifiable test before any generator mines it:

1. **Standalone PIT IC** vs forward return (`fut_ret_1`), with block wild-bootstrap
   p-value + Benjamini-Hochberg FDR (reuses `multiple_testing`).
2. **Orthogonality** -- correlation to the volatility cluster (`atr_pct`, `rel_volume`,
   `volume_acceleration`) and rank-contribution/residual vs the existing accepted-alpha
   space (reuses `alpha_registry`).
3. **Incremental ensemble value** -- effective-breadth lift and **combined OOS IC lift**
   when added to the accepted set, net of turnover (reuses `alpha_synergy`).

Verdict per feature: `ORTHOGONAL_INFORMATIVE` (independent of the vol cluster AND lifts
combined OOS IC above a noise floor) / `REDUNDANT_VOLATILITY` / `WEAK` / `UNINFORMATIVE`.
Breadth alone is **not** enough -- the user's bar is orthogonal *and* a real OOS IC lift.

```bash
python experiments/orthogonal_features/run_feature_study.py   # pilot scan on the cached panel
```

## Pilot scan result (price-derived features, first scan)

| feature | OOS IC | vol corr | rank contrib | combined-OOS-IC lift | verdict |
|---|---|---|---|---|---|
| `idio_reversal` (idiosyncratic short-term reversal) | +0.031 | 0.03 | 0.82 | **+0.0134** | **ORTHOGONAL_INFORMATIVE** |
| `rel_strength_5` | -0.002 | 0.04 | 0.84 | -0.002 | WEAK |
| `downside_ratio` | -0.011 | 0.02 | 0.81 | +0.001 | WEAK |
| `ret_skew_20` | -0.018 | 0.02 | 0.90 | -0.000 | UNINFORMATIVE |
| `parkinson_vol` (volatility control) | -0.006 | 0.91 | 0.40 | -0.017 | REDUNDANT_VOLATILITY |

The volatility control is correctly flagged redundant; noise is rejected. **One**
price-derived feature clears: idiosyncratic short-term reversal -- it is orthogonal to
the volatility cluster and roughly doubles the accepted set's combined OOS IC
(0.012 -> 0.025) at moderate turnover (0.66). It is the classic short-term reversal
anomaly; before integration it still needs a standalone `event_study` (bootstrap +
placebo) and a **net-of-cost breakeven-bps** check (short-horizon reversal is the
textbook gross-positive / net-fragile signal). Everything else price-derived is
redundant or null -- confirming the thesis that durable *new* directional breadth must
come from data outside price/volume.

## Candidate sources (each must clear `feature_gate` before any generator touches it)

| source | data status | priority | note |
|---|---|---|---|
| relative volume / Stocks-in-Play | already in panel (`rel_volume`, `volume_acceleration`) | low | part of the volatility cluster |
| ORB opening range | already in panel (`or_high_dist`, `or_low_dist`) | low | already searched |
| market / sector regime | partly in panel (`spy_trend`, `qqq_trend`, `mkt_regime`) | med | **constant per cross-section** -- needs regime-conditional eval, not the cross-sectional gate |
| intraday microstructure | partly in panel (wicks, body, `vwap_dist`); book = Databento | low | book-skew already net-negative after spread |
| **news / earnings / catalyst** | TESTED -- see below | done | event-timing clears; sentiment fails |
| options flow / put-call / IV rank | TESTED -- see below | done | null vs the validated leads at 15m + 1h |
| spread / slippage / fill-quality | executor journals have slippage | med | the project's known real edge, but a **sizing/timing** axis, not directional alpha |

## News / catalyst / FinBERT scan (the first EXTERNAL source)

`run_news_feature_study.py` aligns leak-free news features to the panel
(`news_features.py`: Alpaca news + FinBERT via `data_sources/`, `av_catalyst.earnings`
for PIT report dates; strictly publish_time < bar t, rolling window + recency decay) and
runs them through the same gate. Real data: **18,043 articles across all 24 symbols**,
2025-10 to 2026-06.

```bash
python experiments/orthogonal_features/run_news_feature_study.py [--backfill]
```

| feature | OOS IC | FDR | vol corr | rank contrib | combined-OOS-IC lift | turnover | verdict |
|---|---|---|---|---|---|---|---|
| `earnings_within_7d` | +0.009 | yes | 0.04 | 0.99 | **+0.0042** | 0.08 | **ORTHOGONAL_INFORMATIVE** |
| `earnings_within_14d` | +0.005 | yes | 0.02 | 1.00 | +0.0018 | 0.08 | WEAK |
| `event_recency_decay` | +0.004 | yes | 0.18 | 0.81 | +0.0015 | 0.16 | WEAK |
| `catalyst_score` / `news_volume_count` / `abnormal_news_volume` | ~0 | yes | <0.3 | >0.76 | ~0 | low | WEAK |
| `news_sentiment_mean/max/min`, `finbert_pos/neg` | ≤0 | mostly no | <0.25 | high | **negative** | low | UNINFORMATIVE |

**Verdict: external information DOES add orthogonal edge -- but it is event *timing*, not
sentiment.** `earnings_within_7d` is FDR-significant, uncorrelated with the volatility
cluster (0.04), fully independent of the accepted set (rank-contrib 0.99), lifts the
ensemble's combined OOS IC (0.012 -> 0.016) at near-zero turnover (0.08). Every FinBERT
**sentiment** feature is uninformative / negative OOS -- the same failure as the
standalone catalyst falsification ([[project_catalyst_stack]]). So unlike price-only
symbolic search (which only rediscovered volatility), an external *event-structure*
feature clears the bar; news *sentiment* does not.

Caveats: the effect is small (+0.0042 lift); `analyst_revision_flag` was excluded as
not PIT-backtestable (AV EARNINGS_ESTIMATES is snapshot-only); earnings dates are taken
from actual `reportedDate` as a proxy for the scheduled date (PIT-safe for "within Nd"
since the date is announced weeks ahead). `earnings_within_7d` is now the validated lead
-- next it needs a standalone `event_study` (bootstrap + placebo) + net-of-cost check
before any generator is allowed to mine the earnings-conditioned panel.

## Event as a CONDITIONER (meta-feature layer)

`run_conditioned_study.py` tests the natural follow-up: does event proximity work as a
CONDITIONING variable -- `base_signal x event_proximity` (e.g. `idio_reversal x
earnings_within_7d`, `volatility x earnings_within_14d`, `ORB x catalyst_window`) --
adding OOS lift BEYOND the parts? Each interaction is scored against an accepted space
that already holds the 8 accepted alphas PLUS every standalone base signal and
conditioner (`feature_gate.accepted_extra`), so a pass means the interaction adds what
neither parent provides alone.

**Verdict: no.** No interaction clears -- the earnings effect is a standalone ADDITIVE
feature, not a useful multiplicative conditioner here. The interactions that do show real
standalone OOS IC (`idio_reversal x event_recency_decay` +0.022, `idio_reversal x
catalyst_score` +0.014) have the LOWEST rank-contribution (0.60-0.72): they merely
re-express `idio_reversal` (already in the book) at higher turnover, adding no orthogonal
signal. Takeaway: add `earnings_within_7d` as its own feature; don't build base*event
interaction terms.

Caveat: this tests *linear additive* ensembling (the combined-IC objective is linear). A
NONLINEAR learner that splits on `earnings_within_7d` and then reads `idio_reversal`
could still extract conditioning value -- untested here, a separate lower-priority probe.

## Options / IV structure (the "positioning" channel)

`options_features.py` + `run_options_study.py`. Options describe how participants are
positioned *before* information arrives -- a different channel from price (what happened)
and earnings timing (when info comes). Source: AV `HISTORICAL_OPTIONS` (as-of-date chain
with IV + greeks + OI). **5,577 symbol-days** fetched across all 24 symbols. ATM found by
delta (~0.5), tenors by DTE. Leak-free: a bar on date D uses only the chain of the latest
date strictly before D (prior-session EOD). Candidates: IV rank, IV percentile, term slope
(IV30-IV7), put/call OI ratio, OI change 1d/5d, 25d skew, and `iv_rank x earnings7`.

**Raised bar:** the accepted space is now the validated leads
`[volatility_factor, idio_reversal, earnings_within_7d]` (+ the 8 alphas), so a pass means
"beats every validated lead," not just "beats volatility."

**Verdict: null at both 15-min (`fut_ret_1`) and 1-hour (`fut_ret_4`) horizons.** No
options/IV feature clears. `iv_rank`/`iv_percentile` are FDR-significant *in-sample*
(IC ~0.012) and genuinely orthogonal (vol-corr 0.14, rank-contrib 0.95) but OOS IC
collapses to ~0. OI/skew are not individually significant. The `iv_rank x earnings7`
composite is FDR-significant but its lift is subsumed by `earnings_within_7d` (rank-contrib
drops to 0.77). 

Gate hardening from this run: at `fut_ret_4`, three non-significant features (skew, P/C OI,
OI-change-5d) briefly showed a tiny positive combined-IC lift (+0.002-0.004 on a 0.006
base) -- combination-weight noise, not signal. `_verdict` now requires **FDR significance
as a necessary condition** for ORTHOGONAL_INFORMATIVE, which correctly reclassifies them as
WEAK. Both horizons then agree: null.

Caveats: (1) liquid mega-cap universe -- exactly where options positioning is most
arbitraged; (2) options-positioning signals are daily/multi-day predictors, and this panel
is 15-min/1-hour forward returns -- a **daily-horizon panel is the cleaner test** and is the
natural follow-up before fully retiring this channel. Prior estimate was ~25-40%; this is
the majority null outcome.

## Net-of-cost validation of the surviving leads

`run_lead_validation.py`. Passing the gate earns the right to this test; it is NOT
deployability. Per lead: dollar-neutral cross-sectional long-short -> daily gross return +
turnover -> cost curve (net Sharpe vs bps), breakeven cost, block-bootstrap significance of
net returns at a realistic 5 bps, and an ADV-based capacity proxy. Plus an earnings event
study.

| lead | gross Sharpe | turnover/day | breakeven | net Sharpe @5bps | bootstrap p(net) | capacity | verdict |
|---|---|---|---|---|---|---|---|
| `idio_reversal` | 2.59 | **48.7** | **0.96 bps** | **-10.9** | 0.0 (on the LOSS) | ~$357k | **KILLED by costs** |
| `earnings_within_7d` | 2.00 | 3.0 | 7.9 bps (OOS 15.6) | +0.73 | **0.57 (n.s.)** | ~$7.1M | survives structurally, **not yet significant** |

**`idio_reversal` is a cost mirage.** A real gross signal (Sharpe 2.6) but ~49x/day
turnover -> breakeven under 1 bps -> at any realistic cost it is a large, bootstrap-significant
LOSER (net Sharpe -10.9 at 5 bps). HFT-only, ~$357k capacity. The "surprising winner" of the
whole search does not survive costs -- the textbook short-term-reversal outcome.

**`earnings_within_7d` is the only structurally-deployable lead** (low turnover, breakeven ~8
bps > realistic 5 bps, ~$7M capacity) and economically coherent -- the event study (70 events)
shows +0.66% drift INTO earnings (d-5..d0) and -3.2% after. BUT its net edge at 5 bps is not
statistically significant (bootstrap p=0.57): real mechanism, too small/short to confirm net of
cost on 8 months. Promising, unproven -- needs longer history.

Methodological note: the gate's per-bar rank-turnover (0.66 for idio_reversal) badly understated
deployment turnover (48.7/day); the cost curve is what exposed the fragility. The net-of-cost
stage is therefore a NECESSARY gate after orthogonality -- a candidate can be FDR-significant,
orthogonal, and OOS-positive yet die entirely on turnover.

## Short interest / short volume (the positioning channel)

`short_interest_features.py` + `run_short_interest_study.py`. Short/borrow pressure is a
genuinely different channel - how participants are positioned before information arrives.
Source: **Massive** (`api.massive.com`, FINRA aggregator) - two channels: bi-weekly **short
interest** (`days_to_cover`, `short_interest` shares, SI change; settlement+~12d publication
lag) and daily **short volume** (`short_volume_ratio`). All 24 symbols, full window. Daily
horizon only (no 15m churn); horizons **1/3/5/10/20d**; leak-free (strict prior-session for
short-volume, settlement+pub-lag for short-interest). Borrow fee / utilization are **not in
this feed** (proprietary securities-lending data) - reported unavailable, not faked.

Raised bar: accepted space = the validated leads `[volatility_factor, idio_reversal,
earnings_within_7d]`. Net-of-cost P&L is **daily mark-to-market** (`fut_ret_1`) regardless of
the signal's predictive horizon - using the h-day forward return as a daily P&L inflates
Sharpe ~sqrt(h)x (overlapping returns); that bug was caught and fixed.

**Verdict: no short-pressure feature is deployable beyond the leads, net-of-cost.** The two
best are **QUARANTINE, not ACCEPT**: `sv_ratio` (daily short-volume, IC 0.139 @20d, orthogonal
vol-corr 0.24, net Sharpe 1.56 @5bps, breakeven 21 bps) and `sv_ratio x volatility` (net Sharpe
1.64, breakeven 29 bps) - both FDR-significant and genuinely orthogonal, but **bootstrap
p~0.2 (net edge not significant)** on 166 days. True short interest (`days_to_cover`, SI change)
fails FDR at the 20-day horizon (~8 independent windows). So short-volume is suggestive and
genuinely independent of the leads, but cannot be confirmed bootstrap-significant net-of-cost on
this sample -> quarantined pending more history, not killed and not accepted.

```bash
python experiments/orthogonal_features/run_short_interest_study.py
```

## Extended-history validation -- the first deployable signal

`daily_bars.py` (Alpaca SIP adjusted, 2016+) + `run_extended_validation.py`. Resolves the
three quarantined signals on a **decade** of daily data instead of 8 months, through the
focused funnel: walk-forward by year, symbol holdout, bootstrap, **placebo labels** (shuffle
forward returns within each day -> the real signal must beat the null), FDR, net-of-cost daily
mark-to-market, breakeven, capacity. Decision: ACCEPT only if bootstrap + FDR + placebo +
holdout + walk-forward all pass net-of-cost; QUARANTINE if positive but underpowered; REJECT
if OOS decays or costs kill it.

| signal | n_obs | net Sharpe @5bps | turnover | breakeven | boot p | placebo p | FDR | +yrs | holdout | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| **earnings_within_7d** | 55,818 (10y) | **1.11** | 0.24 | **56 bps** | 0.0 | 0.0 | yes | 0.73 | +0.32 | **ACCEPT** |
| sv_ratio | 5,592 (2.3y) | 1.67 | 0.63 | 20 bps | 0.108 | 0.017 | no | 1.0 | +0.67 | QUARANTINE |
| sv_ratio x vol | 5,592 | 0.21 | 0.64 | 8 bps | 0.83 | 0.27 | no | 0.5 | +0.37 | REJECT |
| idio_reversal | 55,818 | -1.0 | 1.42 | neg | - | 0.88 | - | 0.09 | -0.62 | REJECT |

**`earnings_within_7d` is the first signal in the whole arc to clear the deployable bar.** On
8 months it was bootstrap p=0.57 (unconfirmable); on a decade it is net Sharpe 1.11 after 5 bps,
breakeven 56 bps (11x the cost), bootstrap AND placebo p=0.0, FDR-significant, positive in 8 of
11 years, survives symbol holdout, low turnover (0.24/day). The pre-earnings drift, robust and
net-positive on this universe. **`sv_ratio` strengthened** (placebo 0.017, walk-forward 2/2,
holdout +0.67) but its net-bootstrap p=0.108 just misses -> still QUARANTINE, sample-limited to
the 2.3y of short-volume history. **`idio_reversal` is dead at daily too** (IC 0.001, no gross
signal -- distinct from the 15m version that had signal but died on turnover); **`sv_ratio x vol`
was noise** revealed by more data. Extending history converted one quarantine to ACCEPT and
collapsed the two weak ones -- exactly what more power is for.

```bash
python experiments/orthogonal_features/run_extended_validation.py
```

## Paper promotion -- earnings_within_7d (the first ACCEPTED signal)

`paper_sleeve.py` + `run_paper_sleeve.py`. The project crosses from research-only
falsification to **controlled paper deployment** of the one signal that cleared the bar.
PAPER ONLY: `paper_sleeve.py` is pure (numpy/pandas) and imports no broker/order/execution
code (AST-enforced by the test).

The sleeve is **market-neutral** (dollar-neutral cross-sectional: long names within ~7d of
earnings, short the rest) so the shadow period measures alpha, not market beta. Safety caps:
max allocation **10%**, per-name **4%**, no leverage (gross scaled down, never up), no
options, daily rebalance. **Fail-safe:** if the earnings-date data is missing or older than 2
days, `target_weights` BLOCKS and returns flat -- it never trades on stale data.

Registry record (`sleeve_registry.jsonl`): `decision=ACCEPT`, `status=PAPER_PROMOTED`,
`promotion_requires_shadow_period=true`, with the frozen validated metrics + caps. Monitoring
(`monitor()` over a daily paper ledger): realized net Sharpe, turnover, slippage, hit rate,
earnings-data freshness, post-earnings exit rate.

Paper-backtest sanity (10% market-neutral book, 1,852 days): realized net Sharpe **1.27**
(reproduces the validated 1.11), turnover 0.026/day (= 0.24 x the 10% gross), post-earnings
exit rate 1.0 (positions flatten after earnings). The "today" book is FLAT until a forward
earnings calendar is wired (historical `reportedDate` has no future dates -> fail-safe flat,
not a guess). Live remains out of scope; promotion still requires the shadow period.

```bash
python experiments/orthogonal_features/run_paper_sleeve.py
```

## Workflow for a new source

```
new source -> align to panel index (leak-free) -> feature_gate.study()
   -> ORTHOGONAL_INFORMATIVE ? -> event_study (bootstrap+placebo) + net-of-cost
        -> only then let AlphaEvolve / gplearn / EO mine over the enriched panel
```

Research only: no broker / order / execution / live code (enforced by
`tests/test_feature_gate.py`).
