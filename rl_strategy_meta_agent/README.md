# RL Strategy Meta-Agent

A **research-only** hedge-fund-style portfolio manager. It does not trade raw
candles. It **evaluates, ranks, and allocates capital across multiple
PineScript strategies** based on historical performance, rolling performance,
market regime, and risk-adjusted returns.

> Research / backtest / paper-sim only. Alpaca is used for **historical bars
> only** (the Data API). No order submission, no trading API, no live broker,
> no `alpaca-py`. There is no code path to live execution, by design.

## Architecture

```
pinescripts/*.txt
      |
      v
Strategy Parser ........ rl_meta/registry.py
      |
      v
Adapter detection ...... rl_meta/adapters.py   (Pine pattern -> Python strategy)
      |
      v
Strategy Registry ...... data/strategy_registry.json  (+ executable/adapter_type)
      |
      v
Alpaca bars (cache) .... rl_meta/data.py -> data/bars/*.csv   (DATA API ONLY)
      |
      v
TradingView exports .... rl_meta/tv_import.py -> data/tradingview_exports/*.csv
   (source of truth)      [imported trades override the adapter when present]
      |
      v
Multi-symbol backtest .. rl_meta/signals.py + backtest.py + splits.py
      |
      v
Performance DB ......... data/strategy_performance.db  (per symbol/timeframe/split)
      |
      v
Ranking + Walk-forward . rl_meta/ranking.py   (admit >=100 trades; OOS gate)
      |
      v
RL Meta-Agent .......... rl_meta/env.py + train.py + models.py
      |
      v
Risk Engine ............ rl_meta/risk.py   (FINAL authority)
      |
      v
Paper / Simulation     (in-process; never a broker)
```

## Alpaca historical data (research-only)

Set paper-account data credentials in the environment:

```bash
export APCA_API_KEY_ID=...
export APCA_API_SECRET_KEY=...
export APCA_API_BASE_URL=https://paper-api.alpaca.markets   # read for a safety note only
export ALPACA_DATA_FEED=iex                                 # free tier; 'sip' if entitled
```

Only `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` are used, and only against
`https://data.alpaca.markets/v2/stocks/.../bars`. The trading base URL is read
purely to print a paper-safety note; this project never constructs an order.

### Fetch data
```bash
python scripts/fetch_alpaca_bars.py \
    --symbols NVDA GOOGL ASML TSM KLAC FIGS TGTX YOU \
    --timeframe 1Day --start 2018-01-01
```
Bars are cached as `data/bars/{symbol}_{timeframe}_{start}_{end}.csv` and reused
unless `--refresh`. Supported timeframes: `1Day`, `1Hour`, `15Min`, `5Min`.
Normalized columns: `timestamp, open, high, low, close, volume, symbol`.

The ticker universe lives in `config/universe.json`.

## TradingView exports = evaluation source of truth

The adapters (below) are Python *approximations* and can diverge from real
PineScript behavior. When you have the strategy's exact TradingView trades, those
become the source of truth and the adapter is used only as a fallback.

### Exact TradingView export steps
1. Open your strategy on a TradingView chart, set the **symbol** and
   **timeframe** you want to validate.
2. Open the **Strategy Tester** panel -> **List of Trades** tab (use Deep
   Backtesting for the full history).
3. Click the **export / download** icon and save the **List of Trades** CSV.
4. Rename it to the convention `‹strategy_id›__‹symbol›__‹timeframe›.csv` (double
   underscores), e.g. `sample_strategy__NVDA__1Day.csv`, and drop it in
   `data/tradingview_exports/`. The `strategy_id` is the PineScript filename stem
   in `pinescripts/` (lower-cased, non-alphanumerics -> `_`).
5. Import:

```bash
python scripts/parse_pine_strategies.py                       # registry first (for version_hash)
python scripts/import_tradingview_trades.py --dir data/tradingview_exports
# or one file explicitly:
python scripts/import_tradingview_trades.py export.csv \
    --strategy-id sample_strategy --symbol NVDA --timeframe 1Day
```

The importer parses the two-row-per-trade format tolerantly (column names,
currency symbols, thousands separators, parenthesised negatives all handled),
stores trades in `data/strategy_performance.db`, and computes metrics. Each
import is **pinned to the strategy's current `version_hash`** (resolved from the
registry) -- edit the PineScript and its old exports stop matching, so stale
trades are never trusted. **Duplicate protection:** importing identical CSV
content twice is a no-op; re-importing an updated file replaces its prior rows
(no double counting). Use `--force` to override.

### Precedence -- one source per strategy, never mixed
Metrics for a strategy are computed from a **single source of truth**. Mixing
exact TradingView fills with approximate adapter fills in one aggregate would
distort PF / expectancy, so it is not allowed:

- **As soon as any version-matched TradingView import exists** for a strategy,
  the whole strategy is `tradingview_verified`. The adapter is **disabled for
  every symbol** -- even symbols that have Alpaca bars but no export. Those
  symbols are recorded as `missing_tv_validation` (tracked, never back-filled
  with adapter trades). The aggregate pools TradingView trades only.
- **With no TradingView imports**, an executable strategy is `adapter_screening`
  (lower confidence) and a non-executable one is `parsed_but_not_executable`.

`evaluation_source` (per strategy) is one of `tradingview_verified` /
`adapter_screening` / `parsed_but_not_executable`; per-symbol `missing_tv_validation`
marks coverage gaps. All four are persisted in the registry and the
`backtests.source` column; the scoreboard shows `source` + missing symbols.

### RL eligibility and the screening flag
`eligible` (metrics gate) is source-agnostic, but **RL (real-capital)
eligibility** is not:

- `tradingview_verified` strategies that pass are `rl_eligible`.
- `adapter_screening` strategies stay in the research scoreboard but are **not**
  `rl_eligible` by default -- they show as `SCREEN-ONLY`.
- Set `ALLOW_ADAPTER_SCREENING_FOR_RL=true` to let adapter-screened strategies
  receive RL eligibility (default `false`).

`train.build_views_from_pipeline` feeds the RL allocator only `rl_eligible`
strategies, and ranking prefers `tradingview_verified` at equal score.

### Import position-size normalization
TradingView backtests often use unrealistic share counts (500–5000+, even
indices at thousands of "shares"). On import, trades are normalized to a
realistic **reference quantity of 10 shares** so strategies are compared on
roughly equal size:

```
scale_factor = reference_qty / raw_qty   if raw_qty > reference_qty   (default 10)
             = 1.0                        otherwise
pnl (effective) = raw_pnl_usd * scale_factor
```

Only dollar PnL/qty are scaled; **percentage return, direction, win rate, and
timing are never changed**. Raw values (`raw_qty`, `raw_pnl_usd`, `return_pct`)
are preserved for audit and the original CSV is untouched. Flags:
`--normalize-qty` (default), `--reference-qty 10`, `--no-normalize` (raw mode).
Normalized PnL drives expectancy, total PnL, drawdown, comparison, and RL.

### Equity curve & drawdown (corrected)
Metrics build a **valid equity curve** from an explicit `INITIAL_EQUITY` (10,000)
in **chronological** order (by exit date):

```
equity_t = INITIAL_EQUITY + cumulative_normalized_pnl_t   (date-ordered)
peak_t   = max(equity_0 .. equity_t)
drawdown_t = (peak_t - equity_t) / peak_t
max_drawdown = max(drawdown_t)
```

Drawdown is taken from the running **peak**, never from raw cumulative PnL, and
is order-independent. If equity ever hits **<= 0** the strategy is flagged
`blown_up = true`, `max_drawdown` is capped at 100%, and it is rejected from RL.
Metrics expose `initial_equity`, `ending_equity`, `min_equity`, `blown_up`.

### Hard RL risk gates
Beyond `eligible`, RL (real-capital) eligibility requires ALL of:

| gate | reject if |
|---|---|
| blow-up | `blown_up == true` |
| drawdown | `max_drawdown >= 0.50` |
| profit factor | `profit_factor < 1.25` |
| sample size | `trade_count < 500` |
| OOS size | `oos_trade_count < 250` |
| martingale | `strategy_family == "martingale"` OR id contains `martingale` |

Martingale strategies are **never** RL-eligible regardless of win rate; tiering
may still report them. These gates are why a 63%-win strategy that blows the
account (mean_reversion, min equity −$34k) is `SCREEN-ONLY`, not RL-eligible.

### Verification gate (flag, don't reject)
`rl_meta/verification.py` flags strategies whose results are *too good to be
plausible*, or whose Pine source has a known repaint/look-ahead vector, for
**manual review** -- it never rejects or changes RL eligibility. `verification_status`
∈ `verified | needs_review | suspected_repaint | suspected_lookahead`.

Triggers (statistical anomaly → `needs_review`): `PF > 5 AND maxDD < 2%`;
`win_rate > 80% AND trade_count > 1000`; `CAGR > 200% AND maxDD < 2%`. Source
inspection (→ `suspected_lookahead`): old Pine (`@version <= 3`) using
`security()` without `lookahead_off`. The goal is to surface things worth
checking the Pine source for, not to reject profitable strategies. Both
RL-eligible TV strategies here are flagged (`pin_bar` needs_review; `sniper`
suspected_lookahead from a v2 `security()` call) — they pass the gates
mechanically but should not be trusted until re-exported with `lookahead_off`.

### Human-curated labels (WINS/LOSSES/FLAT) -- metadata, not truth
Exports filed under `WINS/LOSSES/FLAT` folders are a human **prior** ("I think
this strategy works / fails / stalls on this ticker"). The importer preserves
that as `human_bucket` per strategy-symbol (`symbol_labels` table) but **never**
uses it to compute win rate, PF, expectancy, drawdown, or RL eligibility —
outcomes come only from normalized Net PnL. Imports are fixed to `15Min`
(`--manifest-timeframe`, `all` to override). `scripts/label_audit.py` reports
per strategy-symbol `human_bucket` vs actual outcome, **bucket accuracy**, a
**mismatch report** (e.g. LOSSES-filed but actually net-positive), and
works/fails by ticker. The label is available as optional metadata for future
supervised trade-outcome learning, kept strictly out of the objective metrics.

### Trade-outcome learning dataset
`rl_meta/trade_features.py` + `scripts/build_trade_outcome_dataset.py` turn each
imported TradingView trade into one labeled example for supervised
entry-/exit-quality learning. Features are hard-separated to prevent look-ahead:

- **pre_entry_features** — only bars with `timestamp <= entry_time` (trend,
  momentum, volume, volatility, VWAP, S/R, entry-bar). Legitimate ENTRY inputs.
- **during_trade_features** — the realized path (MFE/MAE, hold, excursion order).
- **exit_features** — exit quality (`profit_capture_ratio`, `giveback_from_mfe`,
  `exit_efficiency`, ...).
- **context_features** — `strategy_family`, `strategy_confidence`,
  `verification_status`, `independence_score`, `rl_eligible`, `human_bucket`.
- **target_labels** — `normalized_pnl`, `outcome_win/loss/flat`, `return_pct`.

Splits by exit date: train `<2024-01-01`, validation `<2025-01-01`, oos else.
Outcomes come only from normalized PnL; `human_bucket` is context, never truth.
Outputs to `data/trade_outcomes/`: `.parquet` + `.csv`, `feature_groups.json`,
`build_report.json`. Pre-entry + regime features are NaN until 15Min OHLCV bars
are cached (`fetch_alpaca_bars.py --timeframe 15Min`); during/exit/label/context
are real from the exports. This is dataset construction only -- no model has
learned a strategy.

**Context bars:** `scripts/fetch_trade_context_bars.py` pulls 15Min OHLCV for
every dataset symbol (>= 30 days before earliest entry .. latest exit),
fetching only missing ranges and merging with cache (no duplicate rows;
`rl_meta/bar_coverage.py`). Credentials load from the environment or a local
`.env` (gitignored). It writes `data/trade_outcomes/bar_coverage_report.json`,
then re-run the builder to populate the pre-entry features. Data pull only --
no orders, no live trading (the trading base URL, even if live, is never used).

### trade_quality_model (does chart context predict winners?)
`rl_meta/trade_quality.py` + `scripts/train_trade_quality_model.py` train
LogisticRegression / RandomForest / XGBoost / LightGBM on **pre_entry_features
only** to predict `outcome_win`, with SHAP explanations. It evaluates on three
splits of very different honesty: **time** (train pre-2024 → test 2024/2025),
**group** (train on some symbols → test on held-out symbols), and **random**
i.i.d. (optimistic). On this data the honest splits give **AUC ≈ 0.51–0.53
(≈ random)** while the random split inflates to ~0.67 -- the gap is pure
trade-autocorrelation leakage (same symbol/period in train+test). Conclusion:
**pre-entry chart context does not generalizably predict these strategies'
trade outcomes** (consistent with the weak |corr|<0.035 and logreg's 0.53).
The framework is sound; the negative result is honest. No live trading touched.

## strategy_confidence (trust, not quality)

`composite_score` answers "how GOOD does this strategy look?"; `strategy_confidence`
(`rl_meta/confidence.py`, 0..100) answers "how TRUSTWORTHY is the evidence?".
They are deliberately **separate axes** -- a strategy can look excellent on 120
trades from one symbol (high quality, low confidence). Both are fed to the RL
allocator as distinct features.

```
confidence_raw = 0.25*tv_coverage      # tv_symbols / universe   (0 for adapter)
               + 0.20*symbol_coverage  # tested_symbols / universe
               + 0.20*trade_score      # min(trade_count / 500, 1)
               + 0.15*years_score      # min(years_tested / 5, 1)
               + 0.20*oos_stability    # full-vs-OOS PF/WR agreement, scaled by OOS n

strategy_confidence = 100 * confidence_raw * source_cap   # clamped to [0,100]
source_cap: tradingview_verified=1.0, adapter_screening=0.30,
            parsed_but_not_executable / missing / no-trades = 0.0
```

`oos_stability` caps profit factor at 4.0 so an `inf` PF (no losing trades) never
breaks the calc, and shrinks when the OOS sample is thin. The source cap means an
`adapter_screening` strategy can never exceed 30 -- it is screening evidence, not
verified. Examples: broad TV-verified ~92-100, thin one-symbol TV ~9-15, adapter
<=30. It is exposed on the scoreboard (`conf` column), the registry
(`strategy_confidence`), and the `backtests.confidence` DB column.

## strategy_independence (a new edge, or a clone?)

A third, **relational** axis (`rl_meta/correlation.py`). Quality and confidence
judge a strategy in isolation; independence asks whether it is a genuinely new
edge or a correlated clone of one you already have. Five strategies at 0.95
correlation are still one edge. This is a **population-phase diagnostic** -- it
does NOT change the RL observation space, reward, PPO model, or risk engine.

**strategy_family** (coarse tag): explicit `// family: trend` comment override >
`adapter_type` map > strategy-name keywords. Families: `trend`, `breakout`,
`mean_reversion`, `flow`, `volatility`, `unknown`.

**Correlation** is Pearson on **time-aligned return series**: each strategy's
trades are turned into a dated fractional-return series (multiple trades in a
period compound), and pairs are correlated on **overlapping periods only**, with
the overlap count and a reliability flag stored. **Monthly aggregation is the
default** because sparse trade data makes daily correlations noisy; daily is
available. Below the minimum overlap (monthly 6, daily 30) a pair is marked
unreliable and is **not** trusted -- for the breadth calc it is conservatively
filled (same family 0.75, different family 0.25) rather than pretending a
5-point correlation is real.

**Effective breadth** = participation ratio of the correlation-matrix
eigenvalues, `(Σλ)² / Σλ²`, clamped to `[1, N]`. It is the effective number of
independent bets: 5 clones at 0.95 → ~1; 8 independent strategies → ~8. Per
strategy: `independence_score = 1 - mean(|correlation to others|)`. The
scoreboard prints `Effective breadth: X / N strategies` plus the
lowest/highest-independence strategies; `strategy_family`, `independence_score`,
and `avg_correlation` are written to the registry.

Why a diagnostic and not yet an RL feature: it tells you, *as you populate*
`pinescripts/`, whether each new strategy bought diversification or is just
TJL v2. Wiring correlation into the allocator's reward is a deliberate later
step (it would change the frozen RL surfaces).

**Population dashboard.** `scripts/backtest_strategies.py` prints a headline you
watch as the library grows:

```
=== Population ===
Strategies discovered: 24   (+ 1 indicator/study file(s) excluded)
Eligible (metrics):    11
RL eligible:           6
Effective breadth:     2.1 (of 24 with trade history)
```

"24 things built, ~2 independent edges" is exactly the signal you want during
candidate generation. Indicator/study files (no `strategy()` declaration -- e.g.
a Markov regime-probability indicator) are reported separately and never counted
as strategies. (`backtest.population_summary()` is the pure helper behind it.)

## PineScript adapters (which patterns are executable) -- fallback

Full PineScript is **not** executed. `rl_meta/adapters.py` recognises common
patterns from the parsed metadata and maps them to a deterministic Python
implementation run on real bars:

| adapter_type | recognised from | long entry / exit (approximation) |
|---|---|---|
| `sma_crossover` | `ta.sma` + cross | fast SMA crosses slow SMA up / down |
| `ema_crossover` | `ta.ema` + cross | fast EMA crosses slow EMA up / down |
| `rsi_mean_reversion` | `ta.rsi` | RSI reclaims oversold / reaches midline |
| `vwap_reclaim` | `vwap` | close reclaims rolling VWAP / loses it |
| `pdh_pdl_break_retest` | "pdh/pdl/previous day/break+retest" | break above prior-day high / close below prior-day low |
| `tjl_trend_join` | name "TJL"/"trend join" | uptrend (close>SMA200, SMA50>SMA200) + EMA20 reclaim / close<SMA50 |

These are documented approximations of the canonical setups, not bit-exact
PineScript reproductions. A script with **no** recognised pattern is marked
`status = "parsed_but_not_executable"`, `executable = false`, and is **excluded
from RL allocation**. Execution status is stored on each registry entry:
`executable`, `adapter_type`, `unsupported_reason`, `tested_symbols`,
`tested_timeframe`.

## Multi-symbol backtest + walk-forward

```bash
python scripts/parse_pine_strategies.py        # parse + adapter detection + registry
python scripts/fetch_alpaca_bars.py ...        # cache bars (above)
python scripts/backtest_strategies.py --timeframe 1Day --start 2018-01-01
```

Each executable strategy is backtested across the whole universe. Per-symbol
results are kept for reporting; the **aggregate** is computed by **pooling all
trades and computing metrics once** (profit factor / Sharpe do not average
across symbols). Fills are **next-bar open** to avoid look-ahead.

Walk-forward splits (`rl_meta/splits.py`), assigned by trade EXIT date:

| split | range |
|---|---|
| train | earliest data .. 2023-12-31 |
| validation | 2024-01-01 .. 2024-12-31 |
| out-of-sample (oos) | 2025-01-01 .. latest |

**Eligibility for RL** requires BOTH: full-sample admission AND out-of-sample
validation. Only OOS-passing strategies feed the RL allocator.

## Admission rules

Admitted only if `trade_count >= 100` **AND** (`win_rate >= 0.60` **OR**
`profit_factor >= 2.0`). Tiers (win_rate / profit_factor only): **A** = >=60% &
>=2.0, **B** = >=60% & <2.0, **C** = <60% & >=2.0, else **Reject**. Rejected
for: WR<60% & PF<2.0, `trade_count < 100`, not executable, or insufficient
Alpaca data. OOS gate adds: too few OOS trades (<30) or OOS fails the perf limb.

## How the RL agent receives only validated metrics

`train.build_views_from_pipeline` loads cached bars **offline** (no creds
needed), runs the multi-symbol/walk-forward backtest, keeps only
`eligible == True` strategies (executable + admitted + OOS-validated), and turns
their **aggregate** metrics into `StrategyView`s for the env. Unsupported or
un-validated strategies never reach the allocator. If no cached bars exist, the
RL stack falls back to synthetic demo views so `check_env` / training still run.

## Folder structure

```
rl_strategy_meta_agent/
  requirements.txt  README.md  pytest.ini
  config/
    universe.json                # the 8-ticker universe
  pinescripts/
    sample_strategy.txt          # drop your *.txt strategies here
  data/                          # generated artifacts
    strategy_registry.json
    strategy_performance.db       # metrics + imported TradingView trades
    bars/                        # cached Alpaca OHLCV CSVs
    tradingview_exports/         # drop List-of-Trades CSVs here (source of truth)
  scripts/
    parse_pine_strategies.py     # parse + adapter detection -> registry
    fetch_alpaca_bars.py         # download historical bars -> data/bars/
    import_tradingview_trades.py # import TV List-of-Trades CSVs (source of truth)
    backtest_strategies.py       # multi-symbol + walk-forward -> DB + scoreboard
  rl_meta/
    types.py                     # dataclasses + Tier/Status enums
    registry.py                  # PineScript discovery + parser + registry I/O
    adapters.py                  # Pine-pattern -> Python strategy adapters (fallback)
    tv_import.py                 # TradingView List-of-Trades CSV parser (source of truth)
    confidence.py                # strategy_confidence (trust score)
    correlation.py               # family tags + independence diagnostic (population-phase)
    data.py                      # Alpaca Data API bars + CSV cache (DATA ONLY)
    signals.py                   # bar-driven execution engine (-> TradeRecords)
    splits.py                    # walk-forward train/validation/oos
    backtest.py                  # multi-symbol orchestration + eligibility
    performance.py               # metrics, rolling engine, sqlite store + imports
    ranking.py                   # admission, tiering, composite score, ranking
    risk.py                      # hard-limit risk engine (final authority)
    env.py                       # Gymnasium allocation environment
    models.py                    # PPO factory
    train.py                     # build env from real backtests -> PPO
  tests/
    test_loader.py  test_registry.py  test_tiering.py  test_ranking.py
    test_performance.py  test_risk.py  test_env.py  test_data.py
    test_adapters.py  test_splits.py  test_backtest.py  test_no_broker.py
    test_tv_import.py  test_source_separation.py  test_confidence.py
    test_correlation.py
```

## How it works

### PineScript discovery
`registry.discover()` globs `pinescripts/*.txt`. New files are picked up with
no code change. Each file gets a stable `strategy_id` (from the filename) and a
`version_hash` (sha256 of contents) so edits are detected as new versions.

### Strategy registration
`registry.parse_pine()` extracts, via regex (no compilation): strategy name,
inputs, entry/exit/stop/target lines, timeframe hints, and indicator usage.
`scripts/parse_pine_strategies.py` writes `data/strategy_registry.json` with
`strategy_id, strategy_name, version_hash, date_added, extracted_metadata,
status`.

### Evaluation
Existing in `pinescripts/` does **not** make a strategy eligible. Each executable
strategy is backtested on **real Alpaca bars** (`signals.backtest_executable` via
the detected adapter) across the whole universe, and scored
(`performance.compute_metrics`): win rate, profit factor, expectancy, Sharpe,
Sortino, max drawdown, trade count, avg win/loss, avg hold, total return, CAGR --
per symbol, pooled aggregate, and per walk-forward split.

The synthetic seeded-price engine remains only as an **offline fallback** for the
RL stack (no creds / no cache); it never feeds admission. Everything downstream
consumes `TradeRecord`s and is agnostic to their source, so swapping in real
TradingView exports later is a drop-in.

### Admission & tiering (`ranking.py`)
Admitted only if `trade_count >= 100` **AND** (`win_rate >= 60%` **OR**
`profit_factor >= 2.0`). Tier is win-rate/PF only:

| Tier | Win rate | Profit factor |
|------|----------|---------------|
| A | >= 60% | >= 2.0 |
| B | >= 60% | < 2.0 |
| C | < 60% | >= 2.0 |
| REJECT | < 60% | < 2.0 |

Extra rejection gates: `trade_count < 100`, not executable, insufficient Alpaca
data, obvious-overfitting guard (near-perfect win rate below 150 trades), and the
out-of-sample gate (RL eligibility needs OOS to pass too). A non-finite profit
factor (no losing trades) only counts as a pass on a real `>= 100`-trade sample.

### Ranking
`composite_score` (0..1) weights win rate, profit factor, expectancy, *low*
drawdown, lifetime-vs-rolling consistency, and recent (30d) strength. Eligible
strategies always outrank ineligible ones; ranks are contiguous (1..N).

### Rolling metrics
`performance.rolling_metrics` recomputes the full metric set over trailing
30/60/90-day windows ending at a reference date, so the RL agent can see whether
an edge is improving or decaying instead of trusting lifetime numbers.

### RL allocation
`env.StrategyMetaEnv` (Gymnasium):
- **Observation** (Box, dim = 7 market + 5 portfolio + K*14 per-strategy = 82 at
  K=5): regime, SPY/QQQ trend, VIX, ATR, vol regime, time-of-day; cash, equity,
  drawdown, exposure, open positions; and per strategy: active signal,
  confidence (quality), (rolling) win rate / profit factor / expectancy /
  drawdown, recent outcome, rank, tier, and **strategy_confidence (trust)** as a
  distinct 14th feature. (Adding this feature changed obs 77 -> 82, which
  **invalidates any saved PPO weights -- retrain required**; the smoke train
  takes seconds.)
- **Action** (Discrete 7): `0` cash, `1/2` small/medium to top Tier-A, `3` small
  to Tier-B, `4` small to Tier-C, `5` reduce exposure, `6` exit.
- **Reward**: `realized_pnl - drawdown - overtrading - slippage - concentration -
  rule_violation` penalties (never raw return alone).

Strategy step-returns are simulated from each strategy's metrics and the current
regime, so the learnable optimum is: fund high-expectancy Tier-A strategies,
avoid deteriorating ones, sit in cash during bad regimes/drawdowns.

### Risk engine (final authority)
`risk.RiskEngine` clamps or vetoes **every** action. The agent can never breach:
max daily loss, max drawdown (kill switch), max position size, max open
positions, max trades/day, liquidity/spread/slippage/stale-data checks. Hard
limits override the policy unconditionally.

## Adding new PineScript strategies
1. Save the strategy as `pinescripts/<name>.txt`.
2. `python scripts/parse_pine_strategies.py` -> detects an adapter, updates the
   registry with `executable`/`adapter_type`. If unrecognised it is marked
   `parsed_but_not_executable` and excluded from RL.
3. `python scripts/fetch_alpaca_bars.py ...` (once) -> cache the universe bars.
4. `python scripts/backtest_strategies.py --timeframe 1Day --start 2018-01-01`
   -> multi-symbol + walk-forward backtest, writes the performance DB, prints the
   scoreboard.
5. `python -m rl_meta.train --timesteps 5000` -> retrain the allocator on the new
   OOS-eligible set. No code changes needed.

## Running tests
```bash
pip install -r requirements.txt
cd rl_strategy_meta_agent
python -m pytest                 # 186 tests
```
Each module also has a self-check: `python -m rl_meta.<module>`.

## Running PPO smoke training
```bash
python -m rl_meta.train --check              # SB3 check_env only
python -m rl_meta.train --timesteps 5000     # check_env + PPO smoke train
```

## Validation performed
- 71 unit tests pass.
- Adapter detection validated (sample is `sma_crossover`; ADX/stoch rejected).
- Alpaca normalization validated on mocked responses; cache write-then-reuse and
  offline cache-miss behavior validated.
- Multi-symbol aggregation validated (pooled trade count == sum of per-symbol).
- Walk-forward split boundaries + per-split metrics validated.
- `trade_count >= 100` admission gate + OOS gate validated.
- No-broker guard: source scanned for order/trading-API tokens (none).
- `stable_baselines3.common.env_checker.check_env` passes; PPO smoke train runs.

## Known limitations
- **Adapters are approximations, not PineScript** -- mitigated by importing real
  TradingView List-of-Trades exports (now the source of truth). Symbols/timeframes
  without an import still fall back to the approximate adapter.
- **Free Alpaca data = IEX feed** (`adjustment=all`). Recently-listed names
  (e.g. YOU, FIGS) have short history; a symbol with no OOS bars is excluded from
  that split, not counted as a failure.
- **Walk-forward is single-pass + partition-by-date** (params come from the Pine
  inputs and are not re-optimized per split). This is leak-free reporting, not
  per-split hyperparameter optimization.
- **Env market/strategy dynamics are still simulated** from each strategy's real
  aggregate metrics, not a bar-by-bar replay. Good for learning allocation
  behavior; not a PnL claim. (Next step #4.)
- **Single-allocation baseline**: one strategy at a time. Multi-weight allocation
  is designed-for but not implemented (next step #3).
- No transaction-cost model beyond a flat slippage penalty in the risk engine.

## Recommended next steps
1. **Done:** TradingView List-of-Trades import is wired as the evaluation source
   of truth (`tv_import.py` + `scripts/import_tradingview_trades.py`). Next:
   auto-pull via the TradingView MCP tools (`data_get_strategy_results`,
   `data_get_trades`) for a script open on the chart, so exports don't have to be
   downloaded by hand.
2. Validate every adapter against its TradingView export and report the
   divergence (trade-count / win-rate / PF delta) so you know which adapters are
   trustworthy fallbacks and which to retire.
3. Extend the action space to a continuous weight vector (Box -> softmax simplex)
   for simultaneous multi-strategy allocation; the observation layout already
   carries per-strategy features for this.
4. Replay each eligible strategy's actual per-bar trade stream into the env
   (replace the metric-simulated step-returns) and feed real SPY/QQQ/VIX regime
   features instead of the simulated regime.
5. Add per-split rolling re-evaluation so deteriorating strategies are demoted
   from RL eligibility automatically as new OOS data arrives.

## Defensive strategy-family allowlist (paper-only, validated by walk-forward)

`rl_meta/strategy_allowlist.py` blocks structurally-bad strategy families from
selection/execution. It is gated behind `META_FAMILY_ALLOWLIST` (default OFF) and
never touches a broker, order, risk, or position-sizing path.

- Blocked: `martingale`, `pin_bar_magic_v1`, `sniper_strategy`
- Allowed (validated): `linear_regression`, `mean_reversion` -- and nothing else.

This is a **defensive exclusion validated by walk-forward, NOT proof of alpha.**
Why each family is blocked:

- `martingale` -- **drives the drawdown.** `scripts/strategy_family_walkforward.py`
  showed dropping it cuts out-of-sample max drawdown ~74% (0.664 -> 0.169) across
  4/4 chronological folds; `exclude_martingale_only` reproduces the entire DD win.
  It is a structural blow-up strategy.
- `pin_bar_magic_v1` -- **negative out-of-sample expectancy** (and repaint-suspect).
- `sniper_strategy` -- **INVALID due to repaint/lookahead.** `scripts/repaint_audit.py`
  (engine `rl_meta/repaint_audit.py`) found its entries are built on a repainting
  pre-v3 `security()` higher-timeframe call, and its 8,383 exported trades show
  profit factor ~19.9 (impossible for an honest strategy). Its P&L is a repaint
  artifact; do not trust it. Rebuild non-repainting and re-backtest bar-by-bar
  before reconsidering.

**The only validated paper families are `linear_regression` and `mean_reversion`.**
Everything else is structurally bad or invalid. This remains a defensive control
(remove bad strategies), not evidence of alpha.
