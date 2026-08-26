# V2.9.8 Live Trading Engine

Production execution layer that runs the **exact** V2.9.8 multi-asset
volatility-targeted allocator against Alpaca market data and a real Alpaca
brokerage account.

> **LIVE MODE USES REAL MONEY.** Live order submission is deliberately hard to
> reach: it requires three independent gates (see [Live activation](#live-activation)).
> Paper mode uses Alpaca's paper account (no real money). Dry-run places zero
> orders anywhere.

> **FORECAST SOURCE.** The strategy currently uses the built-in **momentum
> proxy**, not a trained ML/XGBoost model (`strat_runner.forecast_panel_from_base`
> returns `None`). The active source is surfaced everywhere as
> `forecast_source=momentum_proxy`. Do not describe this system as ML-driven
> unless a real forecast module is wired in and re-validated.

---

## Architecture

```
Alpaca daily bars (adjustment=ALL)         alpaca_data.py
      -> completed daily closes            (drops the in-progress session)
      -> V2.9.8 decision (ONE function)    strat.decide_target_weights  <- shared with the backtest
      -> target portfolio weights          strat_decision.compute_target_portfolio
      -> target dollars (live equity)      live_engine.py
      -> per-symbol delta vs positions      "  (trade only the delta)
      -> 3 gate layers (all must pass)      "  strategy / operational / broker
      -> kill switches                      "
      -> sell-before-buy, idempotent        "  + alpaca_broker.py
      -> fills reconciled                   "
      -> persistent ledger                  execution_store.py (live_trading.db)
```

**One source of truth for the math.** `strat.decide_target_weights` is the single
per-session decision (5-day schedule, forced risk rebalance, partial adjustment,
constraint repair, deadband). Both `run_backtest` and the live engine call it, so
a live decision is numerically identical to the historical one for the same date
(proven by `tests/test_strat_parity.py`, `np.allclose(..., atol=1e-8)`).

---

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt      # includes alpaca-py
```

`alpaca-py` is the current official SDK. `yfinance` remains research-only and is
never imported by the live path.

---

## Credentials

Read from environment (never hard-coded, never logged in full):

```
APCA_API_KEY_ID       # PK... for paper, AK... for live
APCA_API_SECRET_KEY
```

Loaded from `PROJECT_ROOT/.env` (or `ENV_FILE=<path>`). `.env` and `*.db` are
gitignored. Paper keys start with `PK`; live keys start with `AK`. The broker
refuses to cross environments: a `PK` key can never place a live order, and an
`AK` key can never be used on the paper endpoint.

Optional live safety:

```
EXPECTED_ALPACA_ACCOUNT_ID   # if set, a live account-id mismatch BLOCKS trading
```

---

## Strategy timing (do not violate)

Information through the **close of session t** sets the weights **held on t+1**.
The engine always decides from the most recent **completed** daily session; an
in-progress bar is dropped. Two distinct dates appear everywhere:

```
as_of_close    = 2026-08-21   (the completed close the decision is computed from)
target_session = 2026-08-24   (the next session the weights are held on)
```

The decision hash is a pure function of `as_of_close` + target weights + config +
forecast source — never wall-clock time — so restarts reproduce the same identity.

---

## How targets become orders

1. `target_notional[sym] = target_weight[sym] * live_equity`
2. `delta_notional = target_notional - current_position_notional` (executable price)
3. Trade only the delta. Fractional where the asset supports it; otherwise integer
   shares, then realized weights are recomputed and re-checked against the
   strategy constraints (a rounded book that violates a hard constraint is
   **rejected**, not silently submitted).
4. **Sell-before-buy**: reductions execute first, fills reconcile, buying power
   refreshes, buy sizes are recomputed from current equity/prices, then buys.
5. Order type: **market**, DAY. Chosen for transparency and because the strategy
   is a daily close-to-close allocator on liquid ETFs; there is no reprice loop.

---

## Safety gates (three layers, all must pass — REJECT, never clip)

1. **V2.9.8 strategy constraints** (re-checked on the actual, share-rounded book):
   weights finite and non-negative, per-asset caps, growth <= gate ceiling,
   defensive <= 50%, gross <= strategy leverage cap, ex-ante vol <= budget band.
2. **Operational circuit breakers** (`LIVE_MAX_*`, separate from the optimizer):
   per-order notional, total notional, gross exposure, single-asset weight,
   rebalance turnover. If the strategy asks for more than an operational limit the
   rebalance is **rejected with an explanation**, never clipped.
3. **Broker/account constraints**: account not blocked, every symbol tradable,
   buying power adequate. Broker buying power is **never** treated as permission to
   exceed the strategy leverage cap.

## Kill switches

- **File** (`KILL_SWITCH_PATH`, default `.kill_switch`): presence **halts all new
  orders**.
- **Daily loss** (`LIVE_MAX_DAILY_LOSS_PCT`): if equity has fallen past the
  threshold vs the prior close, **risk-increasing** rebalances are blocked;
  de-risking is still allowed. Classification is by projected **gross exposure**,
  not a naive buy/sell rule.
- **Drawdown** (`LIVE_MAX_ACCOUNT_DRAWDOWN_PCT`): tracked against a persisted
  high-water mark; same block-risk-increasing / allow-de-risking behavior.

The engine **never auto-flattens** on an internal error. Any failure (data,
broker, optimizer, NaN weights, DB, unknown order state) blocks new orders, logs,
and exits nonzero. Flatten is an explicit operator action only.

---

## Commands

```bash
python live_trader.py status                              # read-only
python live_trader.py signal                              # READ-ONLY: compute + persist, 0 orders
python live_trader.py rebalance --dry-run                 # full plan, 0 orders
python live_trader.py rebalance --paper                   # paper account
python live_trader.py rebalance --live --confirm-live LIVE
python live_trader.py cancel-open-orders --paper
python live_trader.py flatten --paper --confirm-live FLATTEN

python run_engine.py                                      # no-arg runner (key/env-driven mode)
```

`signal` is strictly read-only. Only `rebalance` (and the no-arg `run_engine.py`)
place orders.

### Dry-run procedure

```bash
python live_trader.py rebalance --dry-run
```

Computes state, decision, target weights, retrieves broker state (if credentials
exist), builds the full order plan, runs every gate, and prints it — **zero
orders**.

### Paper-trading procedure

```bash
# .env has PK... paper keys
python live_trader.py rebalance --paper
python live_trader.py status              # actual vs target, drift, kill-switch state
```

### No-arg runner

```bash
python run_engine.py
```

Mode is key/env-driven: `PK` key -> paper; `AK` key + full live gates -> live;
`AK` key without the full gates -> **blocked** (never silently falls back to
paper); no credentials -> dry-run. Runs the **same** orchestration path as
`live_trader.py rebalance`. Backtest-faithful timing and idempotent, so it is safe
to run on a schedule (after the close is ideal).

### Live activation

Real money requires **all three**:

```
--live                     (CLI flag)
--confirm-live LIVE        (CLI confirmation token)
ENABLE_LIVE_TRADING=true   (env; ALLOW_LIVE_TRADING also honored)
```

plus the repo invariants `MODE=live` and `ALPACA_PAPER_ONLY=false`, an `AK...`
live key, and (if configured) a matching `EXPECTED_ALPACA_ACCOUNT_ID`. Missing any
gate prints `LIVE TRADING BLOCKED` and exits nonzero.

### Emergency

```bash
python live_trader.py cancel-open-orders --live --confirm-live LIVE
python live_trader.py flatten --live --confirm-live FLATTEN
```

`flatten` cancels open orders and closes all positions. It is never invoked
automatically.

---

## Recovery after crash / restart

Each rebalance is a state machine (`PLANNED -> SELLING -> SELLS_RECONCILED ->
BUYING -> RECONCILING -> COMPLETE/FAILED`) with a **single-active-rebalance lock**
enforced by the database. On restart the engine inspects the persisted state and
the broker **before doing anything else** and completes any interrupted rebalance:
already-submitted legs are reconciled against Alpaca fills; never-submitted legs
(recorded write-ahead as intents) are submitted. Deterministic client-order-ids
(`v298-YYYYMMDD-SYM-SIDE-hash8`, plus `-sN` residual slices) and Alpaca's
duplicate-id rejection prevent double submission. A leg is considered complete
only when cumulative filled quantity reaches the desired quantity — never merely
because an order exists or reached a terminal state.

---

## Logging & database

- Structured logs to stdout (`V298_LOG_LEVEL`, default INFO). Each run reports
  mode, as-of / target session, forecast source, state, vol budget, ex-ante vol,
  leverage cap, growth ceiling, current/target gross, the trade table, and drift.
- Durable ledger: **`live_trading.db`** (SQLite, gitignored). Tables: decisions,
  strategy_state, rebalance_runs, active_rebalance (lock), orders (write-ahead,
  UNIQUE client_order_id), fills, equity_snapshots, high_water_mark,
  kill_switch_events. Override the path with `LIVE_TRADING_DB`.

---

## Alpaca vs research data (parity caveat)

The research backtest uses yfinance `auto_adjust=True` total-return closes. The
live engine uses Alpaca daily bars with `adjustment=ALL` (split + dividend
adjusted) for the **decision**, which is the closest corporate-action-consistent
equivalent — but the two vendors are **not** guaranteed numerically identical
(different dividend timing/sources). Order sizing always uses **raw executable
prices** (latest trade), never adjusted historical pseudo-prices. Validate
splits/dividends/symbol/calendar alignment before trusting live-vs-research parity
on real data; the parity tests prove the *decision math* is identical given the
same price series.

## Bond-regime overlay (opt-in, OFF by default, NOT validated)

An optional bond-market overlay (`bond_regime.py`) tilts the strategy's existing
risk levers from bond ETF proxies:

- flight-to-quality (IEF vs HYG), credit spread (LQD vs HYG), bond volatility (TLT
  realized vol as a MOVE proxy) → a composite risk-off z-score → bounded
  `vol_budget_mult` (de-risk in risk-off, small add in risk-on) and
  `growth_ceiling_mult` (tighten the equity gate in risk-off only).

**Enable with `BOND_REGIME_OVERLAY=true`.** When off (default), the multipliers are
exactly `1.0` — a byte-identical no-op, so the backtest and live parity are
unchanged. When on, `forecast_source` gains `+bond_overlay` and the decision hash
changes.

> **This is not falsification-validated.** It repeats the shape of a macro/liquidity
> regime overlay that was already REJECTED in this repo. Before enabling it on real
> money, run the harness and require a real-data improvement that survives
> walk-forward / holdout / placebo / bootstrap:
>
> ```bash
> python backtest_bond_overlay.py --synthetic          # mechanism check (offline)
> python backtest_bond_overlay.py --prices bonds.csv   # real cached data
> ```
>
> The synthetic run improves Sharpe/drawdown by construction; that is a wiring
> check, not evidence. Keep the overlay OFF until it passes the gate.

## Loss-replacement / momentum-rotation overlay (opt-in, OFF by default, NOT validated)

An optional overlay (`loss_replacement.py`) that, after the allocator sets the
target book, exits any **actively held** position whose return vs its **cost basis**
is at or below `loss_replacement_threshold` (default −0.50%) and rotates that
allocation into the strongest **positive-momentum** eligible asset — then re-runs
the strategy's own `enforce_constraints` so the swapped book still respects every
risk limit (per-asset caps, growth ceiling, leverage, vol budget). It is the SAME
shared helper in the backtest and the live engine.

- **Trigger** uses the actual position cost basis, not a one-day price move. Live
  uses the broker-reported `avg_entry_price`; the backtest tracks a deterministic
  weighted-average cost basis (`CostBasisTracker`).
- **Momentum** is the point-in-time simple return `P_t / P_{t-N} − 1`
  (`replacement_momentum_lookback`, default 20 sessions); only positive-momentum,
  currently-flat, non-cooldown assets are eligible replacements.
- **Churn guard**: an exited ticker is barred as a replacement for
  `replacement_cooldown_days` (default 5), tracked in the ledger.
- Multiple losers are handled worst-first, candidates ranked momentum-desc then
  ticker-asc (fully deterministic). No candidate → the freed weight routes to cash.
- Sell-before-buy, idempotency, and restart recovery are the engine's existing
  guarantees; a rejected exit sell stops its dependent replacement buy (fail-closed).

**Enable in `AllocatorConfig`:** `loss_replacement_enabled=True` (plus
`loss_replacement_threshold`, `replacement_momentum_lookback`,
`replacement_cooldown_days`). When off (default), the overlay is a byte-identical
no-op; when on, `forecast_source` gains `+loss_replacement`, `rebalance_reason`
becomes `loss_replacement`, and the decision hash changes.

> **This is NOT falsification-validated — ship it OFF.** The A/B harness on real
> data (2006–2026) shows the −0.50% threshold HURTS risk-adjusted return: it churns
> (turnover 158% → 402%, cost drag 3.1% → 7.9%) and mostly de-risks to cash, giving
> Sharpe 0.85 → 0.69 (−0.16), CAGR 12.5% → 9.7% (−2.8%) while only trimming max
> drawdown (−25.0% → −19.3%). Same shape as every rejected overlay here.
>
> ```bash
> python backtest_loss_replacement.py --synthetic          # mechanism check (offline)
> python backtest_loss_replacement.py                       # real data (yfinance)
> python backtest_loss_replacement.py --prices px.csv       # real cached data
> ```
>
> The synthetic run improves Sharpe by construction; that is wiring, not evidence.
> Keep it OFF until a real-data improvement survives walk-forward / holdout /
> placebo / bootstrap. Do not tune the threshold on the evaluation period.

## Momentum-exhaustion / reversal-rotation overlay (opt-in, OFF by default, NOT validated)

A SECOND, independent exceptional-exit overlay (`momentum_exhaustion.py`). For each
still-held position it builds a point-in-time composite `momentum_exhaustion_score`
in [0,1] from four grouped, low-redundancy axes (documented weights):

- **trend/momentum deceleration 0.40** — a prior uptrend (20d>0) whose 5d/10d
  momentum is rolling over, plus SMA20-slope flattening.
- **oscillator rollover 0.20** — MACD histogram contracting from positive / crossing
  below zero; RSI rolled down from overbought.
- **volatility expansion 0.15** — short-horizon realized vol elevated vs its trailing level.
- **forecast deterioration 0.25** — the allocator's own view (`momentum_view_panel`)
  falling from a positive level.

Each axis is smoothed with a short causal rolling-max so staggered evidence counts
together. A held asset EXITS when the score is **confirmed** ≥ `momentum_exhaustion_threshold`
(default 0.70) for `momentum_exhaustion_confirmations` (2) sessions, OR a single
reading ≥ `momentum_exhaustion_critical` (0.90). A **confirmed** exit rotates only
into a bearish→bullish `reversal_panel` candidate that improves the reversal
opportunity by ≥ `minimum_replacement_score_improvement` (else the winner is left
to run); a **critical** exit always exits (to cash if no candidate). It runs AFTER
loss-replacement (precedence: hard-risk > loss > critical-exhaustion >
confirmed-exhaustion > scheduled); an asset already exited for a loss is never
re-processed. Same cooldown + execution/idempotency plumbing.

**Enable in `AllocatorConfig`:** `momentum_exhaustion_enabled=True`. Point-in-time
by construction (backward-only indicators; `post_exit_outcomes` is offline
evaluation only, never a decision input). Off by default → byte-identical no-op.

> **Behaviorally INERT on this universe — ship it OFF.** On real data (2006–2026)
> across the whole threshold×confirmation grid (0.60–0.80 × 1–3), the detector
> produced at most **1 confirmed exit in 20 years** (3 raw signals, 0 exits at the
> 0.70/2-day default): config C (exhaustion only) is numerically identical to the
> baseline. Broad, diversified index ETFs rarely show the violent multi-axis
> blow-off-then-reversal the detector (correctly, per "let winners run") requires,
> so it neither helps nor hurts here. It would only act on sharper single-name-style
> reversals this allocator does not hold. The combined A/B/C/D:
>
> ```bash
> python backtest_rotation_overlays.py --synthetic --sensitivity   # offline
> python backtest_rotation_overlays.py --sensitivity                # real (yfinance)
> ```
>
> Do not tune the threshold on the evaluation period. Keep OFF until it clears
> walk-forward / holdout / placebo / bootstrap on a universe where it actually fires.

## Adaptive opportunity-gated rotation engine (opt-in, OFF by default)

The UNIFIED rotation engine (`adaptive_rotation.py`), built on the real-data finding
that a **fixed % stop produces turnover, not alpha**. A deterioration signal here does
NOT auto-sell — it OPENS a *hold-vs-rotate* evaluation, and capital moves only when a
candidate (or CASH) is *materially superior after costs*. When enabled it **supersedes**
the naive `loss_replacement` + `momentum_exhaustion` overlays (a config guard forbids
enabling more than one rotation family). Default OFF → byte-identical.

- **Source (deterioration trigger):** a held asset whose vol-normalized damage
  `(P/avg_entry−1) / (sigma·√(20/252))` ≤ a **regime-conditioned threshold**
  (`adaptive_damage_thresholds`, looser in BULL, tighter in CRISIS), OR a confirmed
  exhaustion hazard. This only *opens* the evaluation.
- **Gate 1 (opportunity, rank-units):** cross-sectional percentile ranks (the repo's
  native currency — the IC is rank-IC) of forecast/accel/risk-adj momentum give a
  common `base_score`; `net_edge = candidate_opportunity − hold_score − turnover_penalty
  − concentration_penalty > adaptive_minimum_rotation_edge`, with a candidate quality
  floor and an asymmetric hysteresis floor (`hold_score_floor`).
- **Gate 2 (cost, excess-return units):** `view[c]−view[s] > 2·transaction_cost +
  adaptive_cost_safety_margin`. Because `view` is an EXCESS return, CASH's forecast is
  identically 0 — cash is a first-class, always-available candidate (no rf needed).
- **Churn control:** `adaptive_min_hold_days` (a real-damage source bypasses it; a
  hazard-only source respects it), the shared cooldown ledger, and a `position_entry`
  ledger for `days_held` (backtest ↔ live parity via the same session-count template as
  cooldowns). Multiple rotations resolve by a deterministic greedy over a `net_edge`
  matrix; the swapped book is re-run through `enforce_constraints`. Same execution /
  idempotency / restart path as the other overlays.

> **REAL-DATA RESULT (2006–2026, backtest_rotation_overlays.py A/B/C/D):**
> the opportunity gate works exactly as intended — of **643 deterioration events it
> executed only 31 (95.2% abstention)**: turnover **170% vs the −0.5% stop's 402%**,
> Sharpe **0.85 = baseline** (vs the stops' 0.69 / 0.79), and it is the only config with
> positive turnover efficiency. It is **stable across a 27-cell parameter grid** and
> across regimes. The decision funnel is caused by the intended gates (hysteresis
> 643→191, quality floor 191→51, cost gate 51→31); `adaptive_minimum_rotation_edge` is
> non-binding on this universe (the quality+cost gates dominate — a redundant backstop).
> **BUT there is no convincing alpha vs baseline** (ΔSharpe +0.003, ΔCAGR +0.05%): split
> apart, risky→risky rotation alpha is coin-flip (≈50% hit) and risky→cash avoidance has
> only a transient 10D edge.
>
> ```bash
> python backtest_rotation_overlays.py --synthetic --sensitivity   # offline
> python backtest_rotation_overlays.py --sensitivity                # real (yfinance)
> ```
>
> **Classification:** PROMOTE the adaptive engine *over the fixed −0.5%/−2.0% stops* as
> the rotation architecture (it strictly dominates them and falsifies their reflexive
> trading). **KEEP OFF by default vs the baseline** pending true holdout validation — it
> reproduces the baseline with controlled churn but does not add return. `Config E`
> (ML) should predict the triple-barrier outcome of the rotation decision itself to
> sharpen *selection* — not built this pass.

## Configuration reference

Strategy parameters live in `AllocatorConfig` (unchanged — do not tune here).
Operational safety lives in `ExecutionConfig` (env `LIVE_MAX_*`). See
`.env.example` for the live-trading section.
