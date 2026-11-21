# AMPS V6 — Automatic Loss-Removal & Replacement Optimizer

**Date:** 2026-11-20
**Status:** DESIGN — awaiting user review
**Scope:** Integrate a continuous loss-removal + replacement optimizer into the **active** AMPS V6 live engine.

---

## 0. Decisions (from brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Alpaca environment | **Both PAPER and LIVE in parallel**, with strict state isolation built from day one |
| 2 | −0.5% protection | **Broker-native stop at Alpaca + continuous engine monitor backstop** (defense in depth) |
| 3 | Seller-exhaustion depth | **Add L1-computable microstructure primitives** (microprice, absorption, bid-replenishment, volume-climax); iceberg/L2 marked UNAVAILABLE (IEX has no L2) |
| 4 | Delivery | **Full spec in one integrated change** |

---

## 1. Objective

Continuously monitor every open position. When its **canonical Total P/L (%)** reaches or falls below **−0.5%**, deterministically close it, then search `universe.txt` for a superior replacement that demonstrates **all** of: confirmed seller exhaustion; positive momentum; DI+ > DI−; price above the Bollinger middle line; `ALPHA_CONFIRMED`; positive expected return after costs; positive marginal contribution to portfolio Sharpe; and deterministic risk approval. If no candidate satisfies every requirement, **hold cash** — never force a replacement.

---

## 2. Reality vs. the written spec (what the codebase actually is)

The written spec assumes richer infrastructure than exists on the **live** path. Confirmed by exploration:

1. **No canonical "Total P/L (%)" exists anywhere** (repo-wide grep = 0). The broker exposes only gross `unrealized_pl` / `unrealized_plpc` / `cost_basis`. There is **no lifecycle realized-P&L-across-reductions, no cost/slippage/financing** accounting. → We build a new **lifecycle ledger** and a single canonical calculation.
2. **The `amps_v6/` full pipeline is dormant.** `trade.py`'s `ProductionOrderExecutor` (the live engine) only imports three V6 primitives: `instrument_registry`, `risk_gateway`, `decision_fusion` (via `_run_amps_v6_overlay`, `trade.py:7003`). `pipeline.py`, `position_manager.py`, `order_state_machine.py`, `exit_stack.py`, `allocator.py` exist but are **not** on the live path. We reuse the dormant pieces where sound, and wire them into the live path through trade.py's own funnels.
3. **The active execution gateway is `AlpacaBroker` (`trade.py:2557`): market-order-only, long-only, no reduce-only, no bracket/stop, no reconciliation beyond fill polling.** → We extend it with stop orders, order query/reconciliation, and pending-order cancellation.
4. **No PAPER/LIVE state isolation today** (`paper` from `APCA_PAPER`; per-day-tagged files; no namespaced locks/cursors/state). → All new state is **environment-namespaced**.
5. **Reusable today:** `order_flow_engine` already computes `seller_exhaustion`, CVD divergence, executed/quote imbalance, spread, pressure (`OrderFlowSnapshot`). `trade.py` already has the completed-5m `di_bb_signal_for_frame`, `dmi_adx`, `bollinger_middle`. `universe.load_canonical_universe` / `UniverseScanRotation` / `classify_symbol().exposure_group`. `costs.py` (bps cost model), `metrics.sharpe_ratio`, `portfolio_optimizer.build_shrinkage_covariance` / `solve_covariance_aware_weights`.

### Live-path hook points (confirmed)
- **Exits** → the `exits` funnel assembled in `run_cycle_from_files` (`trade.py:7204–7274`, alongside `monitor_active_positions` / `_run_profit_exit_overlay` / `_run_di_bb_overlay`), then `run_once_from_frames`. Exits take **no agent input** → structurally un-vetoable.
- **Replacement buys** → the `rebalance` funnel (exactly how `_run_amps_v6_overlay` injects buys, `trade.py:7118–7150`), then the shared risk pipeline + `ALLOW_LIVE_TRADING` guard.
- **Position source of truth** → `broker.get_all_open_positions()` → `BrokerPosition` (`trade.py:254`).

---

## 3. Canonical Total P/L (%) — the foundation

### 3.1 Formula (single definition, used everywhere)

```
position_total_pl =
      realized_pnl_gross_for_lifecycle          # Σ over reductions (exit_px − avg_entry_at_reduction)·qty
    + current_unrealized_pnl_gross              # (valid_mark − avg_entry)·remaining_qty
    − entry_costs                               # Σ entry-side commissions + regulatory fees
    − realized_exit_costs                       # Σ exit-side commissions + fees realized so far
    − estimated_remaining_exit_costs            # cost-model estimate to liquidate remaining_qty
    − realized_slippage                         # Σ |fill_px − reference_px|·qty (entry + exit)
    − applicable_financing_costs                # borrow/overnight financing (0 default for long cash)

total_pl_pct = position_total_pl / original_position_lifecycle_cost_basis
```

- `original_position_lifecycle_cost_basis` = **Σ of all buy-fill notionals over the lifecycle** (opening + adds). It is **fixed at the cumulative capital deployed and never reduced by partial sells** — this is what prevents a partial reduction from hiding earlier losses.
- Lifecycle begins at the **initial opening fill** and includes additions, partial fills, partial reductions, commissions, slippage, current unrealized P&L, realized P&L, and pending quantity.
- Costs: use `costs.TransactionCostConfig` (default `BASELINE_COSTS`: spread 10 / slippage 5 / commission 1 / impact 10 bps) for `estimated_remaining_exit_costs` and expectancy; realized costs/slippage come from fills.

### 3.2 Canonical module: `amps_v6/state/lifecycle_ledger.py`

- `@dataclass FillEvent` — `ts, symbol, side, qty, price, reference_price, commission, fees, client_order_id`.
- `@dataclass PositionLifecycle` — accumulates fills; derives `qty`, `avg_entry`, `realized_pnl_gross`, `entry_costs`, `realized_exit_costs`, `realized_slippage`, `original_cost_basis`, `pending_qty`, `financing_accrued`. Methods: `apply_fill(FillEvent)`, `apply_financing(amount)`.
- `@dataclass CanonicalPnL` — the canonical snapshot: `symbol, position_total_pl, total_pl_pct, realized_pnl, unrealized_pnl, costs_breakdown, cost_basis, qty, avg_entry, mark, quote_age_s, valid: bool, status: "OK"|"UNKNOWN"|"STALE"|"CROSSED"|"MALFORMED"`.
- `compute_canonical_pnl(lifecycle, quote, cost_cfg, now, max_quote_age_s) -> CanonicalPnL` — **pure**, the one place the formula lives.
- `LifecycleLedger` — manages `{symbol: PositionLifecycle}`; `reconcile(broker_position, executable_quote, pending_orders)`; env-namespaced persistence (`save`/`load`).

### 3.3 Validation — what may NOT trigger the rule

The −0.5% rule triggers **only** on a valid canonical computation. The quote is rejected (→ `status != OK`, `valid=False`) when it is **stale** (`age > max_quote_age_s`), **crossed** (`bid ≥ ask`), **future-dated** (`ts > now + skew`), or **malformed** (NaN / ≤0 / missing). Explicitly forbidden trigger sources: previous-close return, daily account return, raw last-trade price, portfolio-wide P&L, gross-unrealized-excluding-costs, an agent's estimated P&L, stale/crossed/future/malformed quotes.

When `valid=False`: the rule **does not fire** (no fabricated trigger) and **existing protection is preserved** (broker-native stop stays in place). Satisfies test #6.

### 3.4 Reconciliation

Reconcile against **broker qty, broker avg entry, internal fill ledger, realized-P&L ledger, current executable quote, pending orders**. Broker qty/avg is adopted as truth on divergence; the discrepancy is flagged and the ledger corrected. Pending exit/increase orders adjust effective exposure.

---

## 4. Deterministic exit authority (−0.5%)

### 4.1 Trigger

`total_pl_pct <= -0.005` (canonical, valid). `−0.49%` no-trigger; `−0.50%` triggers; `−0.51%` triggers. The threshold is **fixed and may not be widened, ignored, or vetoed by any analytical agent.**

### 4.2 Exit-stack extension (`amps_v6/risk/exit_stack.py`)

- `ExitConfig`: add `total_pl_exit_threshold: float = -0.005`.
- `ExitContext`: add `total_pl_pct: float = 0.0`, `total_pl_valid: bool = False`.
- `evaluate(...)`: add `TOTAL_PL_STOP_BREACHED` at **priority 85** — above per-position `HARD_STOP` (80), below account-level `DAILY_LOSS_SHUTDOWN`(90)/`DRAWDOWN_SHUTDOWN`(95)/`EMERGENCY_LIQUIDATION`(100). Fires iff `total_pl_valid and total_pl_pct <= total_pl_exit_threshold`.

Because the exit stack takes no agent input (proven invariant), this is **structurally un-vetoable** by Market-Regime / Trend / Breakout / Defensive-Inverse / Order-Book / Entry-Alpha agents, the portfolio optimizer, the RL policy, or any LLM. Other deterministic rules (HARD_STOP, ATR_STOP, account-level) may still close a position **earlier**.

### 4.3 Exit event

On breach emit `TOTAL_PL_STOP_BREACHED` carrying: symbol, position ID, quantity, average entry price, lifecycle cost basis, realized P&L, unrealized P&L, costs, Total P/L, Total P/L (%), threshold, current valid bid, estimated liquidation price, quote age, regime, timestamp, brokerage environment. Recorded via `decision_trace` + the structured `[position-exit]` log (§12).

---

## 5. Exit execution

Sequence after a breach (orchestrated by `ReplacementStateMachine` + extended `AlpacaBroker`):

1. Set position state `EXIT_REQUIRED`.
2. Block additional exposure to the symbol (suppress new buys/increases).
3. Cancel pending entry/increase orders for the symbol (`AlpacaBroker.cancel_open_orders` filtered to buys).
4. Preserve or safely replace protective orders (keep the broker-native stop until the market exit confirms; cancel-and-replace only atomically).
5. Create a **reduce-only** exit intent (`OrderIntent.reduce_only=True`, new field).
6. Route through the execution gateway (never bypass it).
7. Persist the intent **before** broker submission.
8. Unique **idempotent** `client_order_id`.
9. Reconcile broker acknowledgment.
10. Process partial fills until flat.
11. Treat timeouts as **UNKNOWN**, not failed (reuse `OrderState.RECONCILIATION_REQUIRED`).
12. Query broker state before any retry (`resubmittable()` guard — only from `RISK_APPROVED`).
13. Record final realized Total P/L and execution costs (fills → ledger).

A gap/fast market can realize a loss worse than −0.5%; **record both the trigger value and the realized value** (`[position-exit] trigger_total_pl_pct=… realized_total_pl_pct=… slippage_bps=…`).

---

## 6. Broker-native protection (decision #2)

After every entry fill:

1. Reconcile filled qty + average price (ledger).
2. Compute the −0.5% protection level from canonical cost basis.
3. **Create a broker-native stop/stop-limit SELL at Alpaca** when the order+execution policy supports it (new `AlpacaBroker.submit_stop_order`), reduce-only in effect (sell ≤ held qty).
4. Adjust protection after partial fills or reductions (cancel/replace the stop to track qty and the −0.5% level).
5. **Confirm the protective order exists at the broker** (`AlpacaBroker.get_order` / open-orders query) — `Position.protective_order_state` ∈ `NONE|PENDING|ACTIVE|MISSING`.
6. **Never declare protected on local state alone** — only after broker confirmation (`ACTIVE`).
7. The continuous engine monitor (§4) is a **redundant backstop**: if the broker stop is `MISSING` or the −0.5% canonical trigger fires first, the engine emits the reduce-only market exit.

The `PROTECTION_VERIFIED` terminal state requires a broker-confirmed protective order on the replacement.

---

## 7. Replacement timing & state machine

Do **not** evaluate or submit a replacement until: the exit fill is broker-confirmed; exited qty is zero; unresolved exit orders are reconciled; actual proceeds are known; buying power is refreshed; canonical portfolio state is consistent.

### `amps_v6/execution/replacement_state_machine.py`

States: `HELD → EXIT_REQUIRED → EXIT_SUBMITTED → EXIT_PARTIALLY_FILLED → EXIT_FILLED → REPLACEMENT_EVALUATION → (REPLACEMENT_APPROVED | CASH_HELD) → REPLACEMENT_SUBMITTED → REPLACEMENT_PARTIALLY_FILLED → REPLACEMENT_FILLED → PROTECTION_VERIFIED`.

- Env-namespaced JSON persistence; **restart restores unresolved state** (test #28).
- Per-symbol idempotent `client_order_id`s for exit and replacement legs (test #27).
- Order-level lifecycle delegated to the existing `amps_v6 OrderStateMachine` (idempotency, partial fills, RECONCILIATION_REQUIRED).

---

## 8. Replacement universe & eligibility

### 8.1 Pool

Built **only** from `universe.txt` via `universe.load_canonical_universe()`. **No** S&P/Nasdaq constituents, **no** `bullish_*.csv`, **no** hard-coded fallback, **no** symbols outside `universe.txt`.

**Exclude:** the exited symbol during cooldown; stale/missing data; nontradable; broker-restricted; regime-incompatible; expired signals; unacceptable spreads; insufficient liquidity; concentration-limit names; prohibited leveraged/inverse overlap (`classify_symbol().exposure_group`); non-`ALPHA_CONFIRMED`; nonpositive net expectancy.

### 8.2 The 19-point eligibility gate (all must be true)

1. In `universe.txt`. 2. Not under cooldown. 3. Current/valid data. 4. Seller-exhaustion score passes. 5. Seller-exhaustion persistence passes. 6. Required independent confirmations pass. 7. DI+ > DI−. 8. Price > Bollinger middle. 9. Completed-bar positive momentum. 10. Entry Alpha = `ALPHA_CONFIRMED`. 11. Expected net return > 0 after costs. 12. Marginal portfolio-Sharpe improvement > minimum. 13. Regime permits exposure. 14. Liquidity/spread pass. 15. Concentration/covariance pass. 16. Effective leveraged/inverse exposure valid. 17. Deterministic risk approves. 18. Brokerage account permits. 19. Signal still valid at submission.

Reuses `decision_fusion.fuse()` (conjunction, with `positive_expected_net_return` + `acceptable_execution_cost` + inverse/swing gate blocks) and `risk_gateway.evaluate()` for the deterministic gates.

---

## 9. Seller-exhaustion scorer (decision #3)

### `amps_v6/alpha/seller_exhaustion.py`

Consumes a **window series** of `OrderFlowSnapshot`s (persistence) plus new L1 primitives. Evidence categories (independent): price absorption, failed lower low, executed-flow improvement, declining sell intensity, order-flow reversal, CVD divergence, bid replenishment, spread stabilization, microprice improvement, volume climax, failed breakdown. Iceberg/L2-stacked → **UNAVAILABLE** (no L2 on IEX).

- Normalized score **0..1** from available **independent** categories.
- Require: `AMPS_SELLER_EXHAUSTION_MIN_SCORE`, `AMPS_SELLER_EXHAUSTION_MIN_CONFIRMATIONS` independent confirmations, `AMPS_SELLER_EXHAUSTION_PERSISTENCE_WINDOWS` persistence, current/valid data.
- **Return `UNAVAILABLE`** when required microstructure data is missing/stale (snapshot `status ∈ {INSUFFICIENT_DATA, STALE}`). Never fabricate a neutral/positive state.
- Do **not** confirm from one print / one quote / one transient imbalance / an unfinished bar / missing order-flow / stale data (tests #16, #17).

### `order_flow_engine.py` additions (L1-computable)

Pure functions + snapshot fields: `microprice(bid, ask, bid_size, ask_size)`, `absorption_score(...)` (aggressive sell volume vs realized downside), `bid_replenishment_score(...)` (bid-size recovery after sell executions), `volume_climax_score(...)` (sell-vol spike then decline near support). Keep IEX caveat; mark L2-only categories unavailable.

---

## 10. Positive-momentum gate

### `amps_v6/strategy/momentum_gate.py`

**Mandatory (completed bars only):** (1) DI+ > DI− on completed 5m; (2) price > completed-5m Bollinger middle; (3) latest completed momentum bar closes above the previous completed bar. Reuses `trade.di_bb_signal_for_frame` / `dmi_adx` / `bollinger_middle`.

**Additional confirmation** — configurable subset (min count) of: ROC>0; MACD hist positive/rising; RSI rising; price>VWAP; VWAP slope>0; SMA5>SMA20; positive 1m return; positive 5m return; positive relative strength; volume confirmation; reclaimed support; reclaimed opening-range level.

**Timeframes:** 1–60 s order flow/absorption/CVD/spread/microprice; 1 m exhaustion timing + initial reversal; **5 m required DI+/DI− + BB-middle**; 15 m higher-TF confirmation + contradiction detection. Record timeframe disagreement → reduce confidence.

Seller exhaustion alone does **not** qualify (test #11); positive momentum alone does **not** qualify (test #12).

---

## 11. Portfolio optimizer (marginal Sharpe) & sizing

### `amps_v6/portfolio/marginal_sharpe.py`

For every qualifying candidate `j`:

```
replacement_delta_sharpe[j] =
    expected_sharpe(portfolio_after_candidate_j) − expected_sharpe(portfolio_after_exit_to_cash)
```

Inputs: expected return; forecast uncertainty; covariance with holdings; portfolio volatility; realized loss from the stopped position; entry+exit costs; spread; slippage; market impact; turnover; concentration; sector/factor overlap; effective ETF leverage; regime; remaining session time; overnight eligibility; remaining risk budget; buying power. Reuses `portfolio_optimizer.build_shrinkage_covariance`, `metrics.sharpe_ratio`, `costs`.

**Rank by `delta_sharpe`.** Do **not** rank solely by seller-exhaustion score, momentum score, % change, volume, model confidence, standalone Sharpe, or raw expected return (tests #19, #20).

**Cash is valid** (test #21): if no candidate passes every gate → hold proceeds as cash, emit `NO_QUALIFYING_REPLACEMENT`, continue the 55-symbol scan, reconsider later. Never force a replacement.

**Sizing:** optimizer proposes a target weight from marginal Sharpe / volatility / covariance / liquidity / current exposure / concentration / effective leverage / remaining loss budget / buying power / costs — **not** auto-invest all proceeds. Deterministic risk may reduce or reject the weight.

### Multiple simultaneous stop-outs

Process every required exit → reconcile fills → refresh cash/BP → recompute exposure/covariance → evaluate the canonical universe **once** → one **joint** replacement optimization → prevent BP double-allocation (test #22) → respect turnover + replacement limits.

---

## 12. Anti-churn, continuous-engine priority, isolation, config, diagnostics

### Anti-churn (`amps_v6/execution/anti_churn.py`)
Exited-symbol cooldown; max replacements/cycle; max replacements/symbol/session; min marginal-Sharpe improvement; signal expiration; rebalance hysteresis; turnover budget; spread/slippage limits; no-chase boundaries. **Cooldowns may block reentry but never block a protective exit** (tests #23, #24). Env-namespaced persistence.

### Continuous-engine priority (into `trade.py` `run_cycle_from_files`)
1. Broker events → 2. Reconciliation → 3. Existing protective orders → 4. Total P/L ≤ −0.5% exits → 5. Other deterministic risk exits → 6. Position-deterioration exits → 7. Replacement evaluation → 8. New candidate entries → 9. Ordinary universe scanning. A stop event may interrupt rotation; **resume from the persisted batch cursor** afterward (persist `UniverseScanRotation`, test #29).

### PAPER/LIVE isolation (decision #1)
PAPER and LIVE must **not** share executor locks, position state, order state, reconciliation files, runtime cursors, or audit identifiers. Every new state/ledger/cursor/audit path is suffixed by environment (`…_paper` / `…_live`), derived from the resolved endpoint+credentials. Credential-routed execution: live endpoint+creds → live orders; paper endpoint+creds → paper orders (test #30). Also fix `portfolio_rebalancer.py`'s hardcoded `paper=True` to route from the endpoint.

### Config (`amps_v6/configuration/replacement_config.py`)
Versioned dataclass `ReplacementConfig` (with `config_version`), env-overridable via the ALL-CAPS names, defaults:

| Setting | Default |
|---|---|
| `AMPS_TOTAL_PL_EXIT_THRESHOLD` | `-0.005` (fixed) |
| `AMPS_SELLER_EXHAUSTION_MIN_SCORE` | `0.60` |
| `AMPS_SELLER_EXHAUSTION_MIN_CONFIRMATIONS` | `3` |
| `AMPS_SELLER_EXHAUSTION_PERSISTENCE_WINDOWS` | `3` |
| `AMPS_REPLACEMENT_MIN_SHARPE_IMPROVEMENT` | `0.05` |
| `AMPS_EXITED_SYMBOL_COOLDOWN_SECONDS` | `1800` |
| `AMPS_MAX_REPLACEMENTS_PER_CYCLE` | `2` |
| `AMPS_MAX_SYMBOL_REPLACEMENTS_PER_SESSION` | `2` |
| momentum min-confirmations | `2` |
| cost model | `costs.BASELINE_COSTS` |

### Diagnostics (exact formats)
`[position-exit]` breach/fill lines; `[replacement]` approved/cash lines — per the spec's examples. Every action recorded via `decision_trace` + a structured audit log (test #31). **Final report** covers: positions ≤ −0.5%; trigger vs realized Total P/L; exit costs/slippage; seller-exhaustion candidates; momentum confirmations; DI+/DI− + Bollinger status; Entry-Alpha rejections; portfolio-Sharpe rejections; risk rejections; replacements executed; cash decisions; replacement turnover; projected vs realized Sharpe effect; unresolved exits/replacements; confirmation that all orders used the execution gateway.

---

## 13. Module inventory

**New**
- `amps_v6/configuration/replacement_config.py`
- `amps_v6/state/lifecycle_ledger.py` (canonical Total P/L)
- `amps_v6/alpha/seller_exhaustion.py`
- `amps_v6/strategy/momentum_gate.py`
- `amps_v6/portfolio/marginal_sharpe.py`
- `amps_v6/portfolio/replacement_optimizer.py` (orchestrator)
- `amps_v6/execution/replacement_state_machine.py`
- `amps_v6/execution/anti_churn.py`

**Extended**
- `amps_v6/risk/exit_stack.py` — `TOTAL_PL_STOP_BREACHED` + total-P/L context
- `amps_v6/execution/order_intent.py` — `reduce_only`
- `order_flow_engine.py` — microprice / absorption / bid-replenishment / volume-climax
- `universe.py` — persist the rotation cursor (env-namespaced)
- `portfolio_rebalancer.py` — endpoint-routed paper/live (remove hardcoded `paper=True`)
- `trade.py` — `AlpacaBroker` stop orders + order query/reconcile + pending-buy cancel; `ProductionOrderExecutor` loss-removal/replacement step + priority ordering + ledger feeding + env-namespaced state

**Untouched (hard constraint):** `.claude/worktrees/`, `trader_rewrite/`, `vendor/`, `tradingview-mcp/`, `*.bak`/`*.backup`, generated datasets, historical backtest outputs, unrelated research programs. Preserve all staged/unstaged/untracked work.

---

## 14. Test plan (31 required) → `tests/test_amps_v6_loss_removal_replacement.py` (+ focused companions)

Pure/offline (ledger, canonical P/L, exit stack, seller-exhaustion, momentum, marginal Sharpe, eligibility, state machine, anti-churn, isolation); broker interactions via fakes (as in existing `_Broker`/`_Scanner` tests).

1 −0.49% no-trigger · 2 −0.50% triggers · 3 −0.51% triggers · 4 entry+exit costs affect Total P/L · 5 partial reductions don't hide lifecycle losses · 6 stale data can't fabricate a trigger · 7 agents can't veto the exit · 8 pending increase orders cancelled before exit · 9 partial fills unresolved until flat · 10 replacement waits for confirmed proceeds · 11 exhaustion w/o momentum rejected · 12 momentum w/o exhaustion rejected · 13 DI+ ≤ DI− rejected · 14 price ≤ BB middle rejected · 15 unfinished bars can't confirm momentum · 16 missing order-flow → UNAVAILABLE · 17 single transient imbalance can't confirm · 18 ALPHA_CONFIRMED mandatory · 19 Sharpe-reducing candidate rejected · 20 diversifier wins despite lower standalone return · 21 cash when none qualify · 22 multi-exit no BP double-spend · 23 cooldown blocks reentry · 24 cooldown never blocks exit · 25 leveraged/inverse exposure constrained · 26 broker restriction blocks replacement · 27 idempotency prevents duplicate orders · 28 restart restores unresolved replacement state · 29 rotation resumes after replacement · 30 PAPER/LIVE state isolated · 31 complete audit trail.

---

## 15. Non-claims (CLAUDE.md discipline)

This design does **not** claim seller exhaustion guarantees a reversal, nor that forced replacement improves Sharpe. Any such claim requires net-of-cost, out-of-sample, and operational evidence. The −0.5% deterministic exit is a **risk control**, not an alpha claim; the replacement optimizer only acts on candidates that independently pass every falsification gate, and cash is always a valid outcome.
