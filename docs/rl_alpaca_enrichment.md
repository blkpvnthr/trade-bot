# RL Alpaca Bar-Context Enrichment

Phase added 2026-05-31. Fills the gap where **alpha-side multi-timeframe and catalyst
features were neutral** because the alpha replay datasets were seriesless (symbol + timestamp
+ outcome, no bar history). All work is OFFLINE / advisory — **no RL promotion, no live/shadow
wiring, deployment gate unchanged, risk controls unchanged, no live orders, IEX-only (no SIP
by default).** Builds on [docs/rl_thesis_mtf_enhancements.md](rl_thesis_mtf_enhancements.md).

## Why enrichment was needed

The prior multi-timeframe (TDH-DQN) phase added MTF + catalyst features, but the dominant
alpha data source (`rl_trade_replay.csv`, ~56k rows) has a **corrupt symbol column** (the
date `20260528` instead of a ticker) and no bars, so its MTF/catalyst read 0.0. The "before"
audit (`rl_dataset_audit.py`) confirmed it:

| dataset | mtf_* | catalyst | rel_volume / vwap / atr / regimes |
|---|---|---|---|
| alpha_entry | present, **0% nonzero** | present, **0%** | **absent** |
| alpha_exit | present, **0% nonzero** | absent | absent |
| buffett_dca | 61–66% nonzero | absent | absent |
| capital_allocation | absent (portfolio-level) | absent | absent |

## What data is downloaded / used

Local-first, Alpaca for the rest:

* **Local real 1-minute bars** — `system-data/*/*_intraday_bars_1m.csv` (8 files, 146 real
  tickers, real timestamps). Resampled to 5m/15m/1h to compute the intraday features. No API.
* **Alpaca historical daily bars** (read-only `alpaca.data.historical`, **IEX feed**) for the
  symbol's daily trend + **SPY** market regime + **sector-ETF** regime. One batched call per
  group, cached. Never SIP unless `ALPACA_DATA_FEED=sip` is explicitly set.

## Cache layout

```
data/cache/alpaca_bars/{SYMBOL}_{timeframe}.parquet      # e.g. AAPL_day.parquet
```

`BarCache.get(symbol, timeframe, start, end)` returns the cached slice when the range is
covered (a cache *hit*), otherwise fetches read-only, merges + de-dupes into the cache file,
and returns. 116 daily files were cached on the first enrichment run; subsequent runs are
hits. On missing credentials / network error the call returns an empty frame and the feature
degrades to its neutral default — never fabricated.

## Feature formulas

| feature | formula |
|---|---|
| `mtf_trend_fast` | (close − SMA20) / SMA20 on the **5m** series |
| `mtf_trend_mid` | (close − SMA20) / SMA20 on the **1h** series |
| `mtf_trend_slow` | (close − SMA50) / SMA50 on the **daily** series (Alpaca) |
| `mtf_alignment` | mean of sign(trend) across {5m, 15m, 1h, daily} ∈ [−1, 1] |
| `rel_volume` | mean(last 3 bars vol) / mean(last 20 bars vol); 1.0 = average |
| `vwap_distance` | (close − sessionVWAP) / VWAP, VWAP = Σ(typical·vol)/Σvol |
| `atr_pct` | ATR(14) / close on the 5m series |
| `catalyst_strength` | clip((rel_volume − 1) / 2, 0, 1) — thesis §3.9 |
| `spy_regime` | SPY daily: +0.5 if >SMA50, +0.5 if >SMA200 → {−1,−0.5,0,0.5,1} |
| `sector_regime` | same, on the mapped sector ETF (`SECTOR_ETF`); 0 when unmapped |

Backward-compatible: features are **appended** to the alpha observation sets
(alpha_entry 17→22, alpha_exit 14→19; `rl_common.ENRICH_EXTRA_FEATURES`), extractors stay
index-aligned with the feature lists, and `rel_volume` is centered (value − 1.0) so
"average"/"absent" both map to a neutral 0.0.

## Before → after missingness (alpha datasets)

| field | before | after |
|---|---|---|
| mtf_trend_* / mtf_alignment | 0.0% | **22.3%** nonzero |
| rel_volume / vwap_distance / atr_pct | absent | **22.4%** |
| spy_regime | absent | **26.5%** |
| catalyst_strength (entry) | 0.0% | **8.6%** |
| sector_regime | absent | 1.6% (static ETF map covers few small-caps) |

(`buffett_dca` MTF stays 65% nonzero; the 80k degenerate-symbol alpha rows remain neutral —
they cannot be bar-enriched and are reported as such, not fabricated.)

## Before → after walk-forward (offline, deployment gate applied)

Two split methodologies are reported. The builder now sorts rows by `date_tag` so the
evaluator's 70/30 split is a genuine **chronological walk-forward**; `RL_DATASET_SORT=position`
restores the legacy source-order split for comparison with the prior session.

| agent | prior session (MTF, positional) | enriched — positional | enriched — chronological |
|---|---|---|---|
| buffett_dca | paper_mode_only 0.320 | **paper_mode_only** 0.319 / base 0.213 | **do_not_deploy** 0.300 / base 0.200 — *drawdown* (see below) |
| alpha_exit | paper_mode_only 0.438 | do_not_deploy 0.261 = base | do_not_deploy 0.190 = base |
| alpha_entry | do_not_deploy 0.0 | do_not_deploy 0.004 | do_not_deploy 0.0 |
| capital_allocation | shadow | shadow 0.131 / base 0.122 | shadow (51 samples) |

**Reading:** enrichment populated the alpha features (0% → 22%). buffett (richest
multi-timeframe data) beats its baseline on expectancy and APR under both splits (0.30 vs
0.20, APR 387 vs 258), but it is **borderline at the gate**: under the strict chronological
split its max-drawdown is 0.0099 vs the baseline's 0.0066 — only +50%, but past the gate's
1.15× tolerance, so the gate (correctly) returns `do_not_deploy`; under the legacy positional
split the drawdown is acceptable and it stays `paper_mode_only`. The result is mildly unstable
across dataset rebuilds (the intraday replay sampling shifts the recent-window drawdown a few
basis points either side of the threshold). The alpha agents still don't beat their mechanical
baselines — the entry agent degenerates to skip-all, the exit agent collapses to baseline
full-exit. Only ~22% of alpha rows are enriched (the rest are seriesless), so the alpha signal
is still too sparse/recent to flip the gate. None of this is a safety regression — it is the
gate doing its job, and **no agent is promoted.**

## Simulated account walk-forward (stylized, OOS chronological)

Baseline mechanical policy (no RL promoted), %-risk sizing (0.5%/trade), ≤5 trades/day:

| account | total return | daily avg P/L | max DD | win rate | profit factor | trades | hold | worst day | best day |
|---|---|---|---|---|---|---|---|---|---|
| $25K | +5.62% | $281 | 1.06% | 0.32 | 4.58 | 25 | intraday | −$123 | $1,282 |
| $50K | +5.62% | $562 | 1.06% | 0.32 | 4.58 | 25 | intraday | −$246 | $2,563 |
| $100K | +5.62% | $1,124 | 1.06% | 0.32 | 4.58 | 25 | intraday | −$492 | $5,126 |

Under %-risk sizing, %-return is account-size invariant; only the dollar figures scale.
This is a **stylized** backtest on replay decision-points (5-day OOS window, no commissions /
slippage / fills) — directional, not a P&L promise.

## Safety status

`live_wired=False`, shadow OFF, **no agent promoted**. Deployment gate, `MIN_DEPLOY_SAMPLES`,
overfit guard, and risk controls all unchanged. Enrichment is read-only IEX historical data;
no order path, no credentials, no SIP-by-default touched. Tests:
`tests/test_rl_alpaca_enrichment.py` (17). Full suite: 294 passing.

## Files

`rl_alpaca_bar_enrichment.py` (new), `rl_dataset_audit.py` (new), `rl_account_simulation.py`
(new), `rl_alpaca_backfill.py` (1m-bar ingestion + regime wiring), `rl_common.py`
(+5 enrichment features), `rl_training_dataset_builder.py` (chronological sort),
`train_rl_agents.py` (per-agent timesteps), `tests/test_rl_alpaca_enrichment.py` (new).

## Next recommended step

Grow the enriched alpha share above ~22%: either (a) repair/repopulate the symbol column in
`rl_trade_replay.csv` so its 56k rows can be bar-enriched, or (b) run a longer Alpaca intraday
replay (`build_backfill` over a 90-day IEX window for a large liquid universe) so train and
test both carry dense enrichment — then re-evaluate whether the alpha agents clear the gate.
Expand `SECTOR_ETF` for the small-cap universe to lift `sector_regime` coverage.
