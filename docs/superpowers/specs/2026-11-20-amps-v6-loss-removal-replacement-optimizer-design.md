# AMPS V6 — Automatic Loss-Removal & Replacement Optimizer

**Date:** 2026-11-20
**Status:** Approved design (Phase 1 detailed; Phase 2 outlined)
**Author:** QuantumQuant Trader + Claude

---

## 1. Objective

Integrate an automatic loss-removal and replacement optimizer into the active AMPS V6
trading engine.

Continuously monitor every open position. When its **canonical Total P/L (%)** reaches or
falls at/below **−0.5%**, close it deterministically, then search `universe.txt` for a
superior replacement that demonstrates: confirmed seller exhaustion, positive momentum,
DI+ > DI−, price above the Bollinger middle line, `ALPHA_CONFIRMED`, positive net
expected return after costs, positive marginal contribution to portfolio Sharpe, and
deterministic risk approval. If no candidate satisfies **every** requirement, hold cash —
never force a replacement to remain invested.

This document records the design decisions, the as-built codebase reality that reshapes the
original spec, the detailed Phase 1 design, and the Phase 2 outline.

---

## 2. Codebase reality (findings that reshape the spec)

Six parallel exploration passes established the following ground truth.

### 2.1 What already exists and is reusable

- **Deterministic-authority-vs-advisory split is foundational.** `amps_v6/__init__.py:8-10`
  states agents analyze/recommend/veto only; only a deterministic execution gateway may
  talk to the broker; protective exits run independently of any agent or LLM.
- **Exit stack is a non-vetoable priority ladder.** `amps_v6/risk/exit_stack.py:56-93` is a
  short-circuit ladder; the exit path (`amps_v6/pipeline.py:169-183 manage_position`) reads
  **zero** agent output. Account-level dominating checks sit at the top (`:60-66`).
- **Seller-exhaustion machinery mostly exists.** `order_flow_engine.py:319 norm_seller_exhaustion`
  plus CVD / executed-imbalance / quote-imbalance / `cvd_divergence` snapshot fields, consumed
  by `buy_opportunity_engine.py:187 count_buy_confirmations` (a 10-confirmation
  seller-exhaustion→buy scorer) and `buy_opportunity_engine.py:119 buy_score`.
- **Momentum + alpha gates exist.** `trade.py:3088 dmi_adx` (Wilder DI+/DI−/ADX on completed
  bars), `trade.py:3158 bollinger_middle`, `trade.py:3173 di_bb_signal_for_frame` (last
  completed candle). `ALPHA_CONFIRMED` is a well-defined enum returned by
  `amps_v6/alpha/entry_alpha.py:149 decide()` (constant at `:24`, confirmed branch `:165`).
- **Sharpe + cost primitives exist.** `metrics.py:54 sharpe_ratio` (corrected: daily excess
  return mean / std × √periods). Per-bucket cost bps in `backtest_dmi_adx.py:39`
  (majors 1.0 / equities 3.0 / lev-ETF 6.0 per side); generic model in `costs.py`.
- **Order lifecycle primitives exist.** `amps_v6/execution/order_state_machine.py` has a full
  `OrderState` enum (`:22-43`), legal `TRANSITIONS` (`:49-74`), `RECONCILIATION_REQUIRED`,
  `create()` idempotency guard (`:100-102`), `resubmittable()` (`:133-137`),
  `needs_reconciliation()` (`:130`).
- **Canonical universe loader is fail-closed.** `universe.py:277 load_canonical_universe`
  (raises on missing/empty, no fallback); `universe.py:637 UniverseScanRotation`
  (deterministic 20/20/15 batching over the 55-symbol `universe.txt`).

### 2.2 Gaps vs the spec (the real work)

1. **The "active position manager" / "active execution gateway" named by the spec do not
   exist as the spec imagines.** The clean `amps_v6` position manager / exit stack / order
   state machine are **not wired into the live loop**. Live trading runs through
   `trade.py:4978 ProductionOrderExecutor` using broker `unrealized_plpc`, with a
   hand-rolled, **entry-only** V6 overlay (`trade.py:7003 _run_amps_v6_overlay`) that
   bypasses `amps_v6/pipeline.py`.
2. **No canonical Total P/L.** Three divergent per-position return formulas exist, none net
   of costs, none accumulating lifecycle P/L:
   - (A) `price/avg_entry − 1` — `amps_v6/execution/position_manager.py:46`, `exit_stack.py:70`.
   - (B) broker `unrealized_plpc` passed through verbatim — `trade.py` (many sites).
   - (C) `(current_price − avg_entry)/avg_entry` — `profit_exit_engine.py:331`.
3. **No execution gateway.** The "gateway" referenced throughout is an empty slot. Submission
   is market-orders-only (`alpaca_executor.py:912`), **client_order_id is non-idempotent**
   (embeds a run timestamp — `alpaca_executor.py:706-710`, `trade.py:5677`), **no
   pre-submission persistence**, **no reconciliation driver**, **no broker-native protective
   orders** (no stop/bracket/OCO).
4. **PAPER and LIVE state is shared.** All state/log files are keyed by date only, never by
   mode (`alpaca_executor.py:76`; the `amps_v6_*.json` paths at fixed locations). A paper run
   and a live run on the same day collide on kill-switch, order/position state, cursors, and
   audit ids. The spec explicitly forbids this.
5. **No marginal / delta-Sharpe optimizer.** `portfolio_optimizer.py` is score-weighting;
   `amps_v6/strategy/allocator.py` is heuristic weight. Neither uses covariance or a
   portfolio-Sharpe objective. A covariance-aware API is **specified by
   `tests/test_portfolio_optimizer_covariance.py` but not implemented** — a clean extension
   point for Phase 2.
6. **No position state machine** (`HELD → EXIT_REQUIRED → …`). Only an order state machine
   exists.
7. **Batch rotation cursor is in-process-only** (`trade.py:3204`), resets on restart; only
   the coverage report is persisted. `amps_v6/persistence/` is empty.
8. **Missing microstructure primitives** for seller exhaustion: absorption, microprice,
   bid-replenishment, volume-climax, spread *dynamics* (change over time), persistence across
   windows. `failed_breakdown` exists in `trade.py:3141`, not in the order-flow engine.

**Conclusion:** a substantial part of the spec is *building the canonical P/L and
execution-gateway foundation* that the feature then sits on. That motivates the phasing below.

---

## 3. Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Integration architecture | **Fully wire the `amps_v6` pipeline into the live engine** (end state), replacing the hand-rolled overlay. Phase 1 wires the **exit path** (`manage_position`); Phase 2 completes the **entry path** (`evaluate_entry`) for replacement entries. | Most correct long-term; gives one canonical path for P/L, exits, and audit. User-selected. |
| D2 | First deliverable | **Deterministic loss-removal spine** (canonical Total P/L + `TOTAL_PL_STOP` + execution gateway + paper/live isolation + audit + tests). Phase 2 = replacement optimizer. | Pure risk-control half; independently valuable and safe; matches the spec's "exits before replacement" priority. User-selected. |
| D3 | Operating posture | **Paper endpoint; everything armed.** Phase 1: the −0.5% stop **submits real reduce-only exits** in paper. Phase 2: the replacement optimizer **submits buys** in paper. Going LIVE requires explicit flags. | User-selected. Arming in *paper* is consistent with CLAUDE.md falsification-first — it generates forward paper-trading evidence required before any live arming of the (unvalidated) replacement edge. |

### Research-integrity notes (binding)

- The −0.5% stop is a **pure risk control**, not an alpha claim; arming it is unconditionally sound.
- The replacement premise (seller exhaustion + momentum → better net Sharpe) is an
  **unvalidated edge**. Per CLAUDE.md and the spec's own caution, we will **not** claim seller
  exhaustion guarantees a reversal or that forced replacement improves Sharpe without
  net-of-cost out-of-sample and operational evidence. Phase 2 ships armed **in paper only**;
  live arming of replacement is a separate, evidence-gated decision.

---

## 4. Phase 1 — Detailed design

### 4.1 Canonical Total P/L (one calculation, used everywhere)

New module **`amps_v6/execution/canonical_pl.py`**.

```
position_total_pl =
      realized_pnl_for_position_lifecycle
    + current_unrealized_pnl
    − entry_costs
    − realized_exit_costs
    − estimated_remaining_exit_costs
    − realized_slippage
    − applicable_financing_costs

total_pl_pct = position_total_pl / original_position_lifecycle_cost_basis
```

- **Denominator = original lifecycle cost basis** = Σ(entry notional) + entry costs across the
  whole lifecycle (initial open + additions). It is **monotonic — never reset on partial
  reductions**, so earlier losses cannot be hidden.
- **Validity-gated:** the calculation returns a typed `TotalPL` result whose state is one of
  `VALID | UNAVAILABLE | UNKNOWN`. It returns **a number only when the quote is valid**;
  stale / crossed / future-dated / malformed quotes yield `UNAVAILABLE`/`UNKNOWN` and
  **cannot fabricate a trigger**. It uses the **current valid bid** for the liquidation
  estimate and records quote age.
- **Quote guard** (`amps_v6/execution/quote_guard.py` or inline): validates `bid > 0`,
  `ask ≥ bid`, `age ≤ max_quote_age_s`, timestamp not in the future →
  `VALID | STALE | CROSSED | FUTURE | MALFORMED`.
- **Reconciliation inputs:** broker quantity, broker average entry price, internal fill
  ledger, realized-P&L ledger, current executable quote, pending orders. Broker is source of
  truth for qty/avg; the internal ledger owns costs/slippage/financing.
- **Single source of truth:** the exit stack, `terminal_dashboard.py`, the position manager,
  the risk engine, the audit trail, and (Phase 2) the optimizer all call this function. The
  three divergent formulas are replaced at their decision/display points.

**Interface (sketch):**

```python
@dataclass
class TotalPL:
    state: str                 # "VALID" | "UNAVAILABLE" | "UNKNOWN"
    position_total_pl: float    # dollars
    total_pl_pct: float | None  # None unless state == VALID
    realized_pnl: float
    unrealized_pnl: float
    entry_costs: float
    realized_exit_costs: float
    estimated_remaining_exit_costs: float
    realized_slippage: float
    financing_costs: float
    cost_basis: float
    bid: float | None
    quote_age_s: float | None
    reason: str

def compute_total_pl(position, quote, cost_model, financing_fn, *, now, max_quote_age_s) -> TotalPL: ...
```

### 4.2 Lifecycle cost ledger on positions

Extend **`amps_v6/execution/position_manager.py`** `Position` + `apply_fill`:

- New fields: `entry_costs`, `realized_exit_costs`, `realized_slippage`, `financing_costs`,
  `lifecycle_cost_basis` (monotonic).
- `apply_fill` accrues entry costs on buys, exit costs + realized slippage on sells, and
  financing via a pluggable `financing_fn` (0 for cash/non-margin; accrued for
  leveraged/overnight per policy).
- Cost model: `costs.py` + the per-bucket bps (equity 3 / lev-ETF 6 / majors 1), bucketed via
  `backtest_dmi_adx.py:42 bucket_of` / `universe.py classify_symbol`.
- Implement the declared-but-unused `pending_qty`.

### 4.3 `TOTAL_PL_STOP_BREACHED` exit authority

Extend **`amps_v6/risk/exit_stack.py`**:

- `ExitConfig.total_pl_stop_pct: float = -0.005` (new field).
- `ExitContext.total_pl_pct: float | None` + `ExitContext.total_pl_valid: bool` (new fields).
- New rule at the **top of the ladder** (priority ~97: above `DRAWDOWN_SHUTDOWN` 95 and
  `DAILY_LOSS_SHUTDOWN` 90, below `EMERGENCY_LIQUIDATION` 100):

  ```python
  if ctx.total_pl_valid and ctx.total_pl_pct is not None and ctx.total_pl_pct <= cfg.total_pl_stop_pct:
      return ExitDecision("EXIT", "TOTAL_PL_STOP_BREACHED", 97)
  ```

- Comparison is `<=`: −0.49% does **not** trigger; exactly −0.50% triggers; −0.51% triggers.
- **Fail-closed:** invalid data never triggers (and never suppresses an otherwise-valid
  trigger). The threshold may not be widened, ignored, or vetoed.
- **Non-vetoable by construction:** `manage_position`/`exit_stack.evaluate` take no agent
  input. Agents may annotate the trace; they cannot prevent the exit.

### 4.4 Execution gateway (new)

New module **`amps_v6/execution/execution_gateway.py`** — the only component that
submits/cancels/replaces orders for this subsystem.

- **Deterministic idempotent `client_order_id`** = stable hash of
  `symbol + side + qty + position/intent id + session date + mode` (no timestamp).
  `OrderStateMachine.create()` refuses duplicates for the same cid.
- **Write-ahead persistence:** persist the intent + `OrderRecord` **before** broker submit,
  under a mode-namespaced path.
- **Reduce-only exit routing:** the deterministic risk gateway rejects non-buy sides, so
  exits route through this gateway. Reuses `AlpacaBroker.submit_order` (`trade.py:2708`) with
  its long-only / available-qty guard; marks the order reduce-only / sell.
- **State-machine drive:** advance on ack / partial / fill. **Timeouts → `RECONCILIATION_REQUIRED`
  (UNKNOWN, not failed).** A reconciliation driver resolves `needs_reconciliation()` by
  querying broker state **before any retry**. Processes partial fills until the position is
  flat. Records final realized Total P/L + execution costs (trigger value *and* realized value
  — a gap/fast market may realize worse than −0.5%).
- **Broker-native protection:** after an entry fill, reconcile filled qty/avg, compute the
  −0.5% protection level, and **create or verify a broker-native stop when the order type &
  execution policy support it**; confirm it exists at the broker; adjust after partial fills /
  reductions; **never declare a position protected from local state alone.** Where native
  protection is unsupported, the engine-monitored canonical-P/L stop is the documented
  backstop.
  *(Phase 1 scope note: the new subsystem does not open positions until Phase 2, so in Phase 1
  the protection logic attaches to positions opened by the existing engine paths — the scanner
  and the hand-rolled entry overlay. The subsystem monitors **every** open position regardless
  of origin.)*

### 4.5 Position state machine (exit half)

New `PositionState` enum (Phase 1 subset):
`HELD → EXIT_REQUIRED → EXIT_SUBMITTED → EXIT_PARTIALLY_FILLED → EXIT_FILLED`.
(Replacement states `REPLACEMENT_EVALUATION → REPLACEMENT_APPROVED | CASH_HELD →
REPLACEMENT_SUBMITTED → REPLACEMENT_PARTIALLY_FILLED → REPLACEMENT_FILLED → PROTECTION_VERIFIED`
added in Phase 2.)

On a threshold breach: set `EXIT_REQUIRED`; **block additional exposure** to the symbol;
**cancel pending entry/increase orders** for it; preserve or safely replace protective orders;
create a reduce-only exit intent; route it through the gateway.

### 4.6 Paper/live isolation

New **`amps_v6/configuration/runtime_env.py`**:

- One source of truth for mode (`paper | live`), resolving today's ambiguity (`APCA_PAPER`,
  `ALLOW_LIVE_TRADING`, `TradingClient(paper=…)`, `.env` base-url) into a single resolved
  value with an explicit precedence rule.
- **Mode-namespaced paths** for all new state/locks/cursors/audit (e.g.
  `logs/execution/<paper|live>/…`).
- **Per-mode executor lock** so paper and live never share executor locks, position/order
  state, reconciliation files, runtime cursors, or audit ids.
- Legacy date-keyed files are left untouched to limit blast radius; the new subsystem is
  isolated.

### 4.7 Live-loop wiring (full pipeline — exit path)

Wire **`amps_v6/pipeline.manage_position`** into `trade.py:7152 run_cycle_from_files()` in the
exit-assembly window — after `exits = pd.concat([monitor_exits, scanner_exits])`
(`trade.py:7214`) and before `run_once_from_frames(rebalance, exits)` (`trade.py:7300`).

**Continuous-engine priority order** (events processed in this order):
1. Broker events
2. Reconciliation
3. Existing protective orders
4. Total P/L (%) ≤ −0.5% exits
5. Other deterministic risk exits
6. Position deterioration exits
7. Replacement evaluation *(Phase 2)*
8. New candidate entries
9. Ordinary universe scanning

A stop event may interrupt normal rotation; the scan resumes from the persisted batch cursor
(cursor persistence added here so restart/interrupt resumes correctly).

For each open position, the step reconciles canonical state, computes canonical Total P/L,
runs the exit stack (now including `TOTAL_PL_STOP`), and routes any `EXIT` through the new
gateway.

### 4.8 Versioned configuration

Centralized versioned AMPS settings (a dataclass with a `config_version` field + env binding
following the `trade.py:7565` pattern). Phase 1 active settings:

| Setting | Default | Phase |
|---------|---------|-------|
| `AMPS_TOTAL_PL_EXIT_THRESHOLD` | `-0.005` | 1 (active) |
| `AMPS_STOP_ARMED` | `1` | 1 (active) |
| `APCA_PAPER` | `1` | 1 (active) |
| `ALLOW_LIVE_TRADING` | `0` | 1 (active) |
| `AMPS_MAX_QUOTE_AGE_S` | `10` | 1 (active) |
| `AMPS_EXITED_SYMBOL_COOLDOWN_SECONDS` | `900` | 1 (register on exit; consumed P2) |
| `AMPS_SELLER_EXHAUSTION_MIN_SCORE` | `0.60` | 2 (declared, inert) |
| `AMPS_SELLER_EXHAUSTION_MIN_CONFIRMATIONS` | `3` | 2 (declared, inert) |
| `AMPS_SELLER_EXHAUSTION_PERSISTENCE_WINDOWS` | `3` | 2 (declared, inert) |
| `AMPS_REPLACEMENT_MIN_SHARPE_IMPROVEMENT` | `0.02` | 2 (declared, inert) |
| `AMPS_MAX_REPLACEMENTS_PER_CYCLE` | `2` | 2 (declared, inert) |
| `AMPS_MAX_SYMBOL_REPLACEMENTS_PER_SESSION` | `2` | 2 (declared, inert) |
| `AMPS_REPLACEMENT_ARMED` | `0` | 2 (declared, inert) |

Proposed defaults for the "configurable" items are starting points to be tuned/falsified in
Phase 2, not validated values.

### 4.9 Audit & diagnostics

Emit the spec's `[position-exit]` diagnostics on breach and on exit fill, including the full
breach payload (symbol, position id, qty, avg entry, lifecycle cost basis, realized P&L,
unrealized P&L, costs, Total P/L, Total P/L (%), threshold, current valid bid, estimated
liquidation price, quote age, regime, timestamp, brokerage environment). Append to
`amps_v6/monitoring/decision_trace.py` JSONL (mode-namespaced). Record both the **trigger**
Total P/L (%) and the **realized** Total P/L (%) plus slippage bps. Every action leaves an
audit record.

---

## 5. Phase 1 test matrix (TDD)

Phase 1 implements these of the spec's 31 tests:

| # | Test | Covered in P1 |
|---|------|---------------|
| 1 | −0.49% does not trigger | ✅ |
| 2 | Exactly −0.50% triggers | ✅ |
| 3 | −0.51% triggers | ✅ |
| 4 | Entry and exit costs affect Total P/L | ✅ |
| 5 | Partial reductions do not hide lifecycle losses | ✅ |
| 6 | Stale valuation data cannot fabricate a trigger | ✅ |
| 7 | Analytical agents cannot veto the exit | ✅ |
| 8 | Pending increase orders cancelled before exit | ✅ |
| 9 | Partial fills remain unresolved until flat | ✅ |
| 24 | Cooldown never blocks an exit | ✅ |
| 27 | Idempotency prevents duplicate exit orders | ✅ |
| 28 | Restart restores unresolved (exit) state | ✅ |
| 30 | PAPER and LIVE state remain isolated | ✅ |
| 31 | Every action has a complete audit trail | ✅ |
| — | Broker-native protection created/verified/adjusted | ✅ (added) |
| 10 | Replacement waits for confirmed exit proceeds | exit-side half in P1; full in P2 |
| 11–23, 25–26, 29 | Replacement-specific | Phase 2 |

---

## 6. Phase 2 — outline (not implemented now)

- **Seller-exhaustion** new primitives (absorption, microprice, bid-replenishment,
  volume-climax, spread dynamics) + normalized 0–1 score + min-score / min-confirmations /
  persistence-across-windows gating; `UNAVAILABLE` when microstructure data is missing (never
  a fabricated neutral/positive state).
- **Positive-momentum gate:** DI+ > DI−, price above BB middle, completed-bar momentum close
  above the previous completed bar, plus a configurable confirmation subset — reusing
  `trade.py dmi_adx` / `bollinger_middle` / `di_bb_signal_for_frame` / `mtf_gate` /
  `buy_opportunity_engine`.
- **`ALPHA_CONFIRMED`** mandatory gate via `amps_v6/alpha/entry_alpha.decide()`.
- **Marginal-Sharpe optimizer:**
  `replacement_delta_sharpe[j] = sharpe(portfolio_after_candidate_j) − sharpe(portfolio_after_exit_to_cash)`
  using `metrics.sharpe_ratio` + the covariance-aware API sketched in
  `tests/test_portfolio_optimizer_covariance.py`. Rank by marginal Sharpe, not by raw
  score/momentum/%/volume/confidence/standalone Sharpe.
- **Replacement state machine**, anti-churn (cooldown, per-cycle & per-session caps, min
  marginal-Sharpe improvement, signal expiration, hysteresis, turnover budget, spread/slippage
  limits, no-chase).
- **Joint multi-stop-out optimization** (one joint replacement pass; no buying-power
  double-allocation).
- **Cash is valid:** emit `NO_QUALIFYING_REPLACEMENT`, hold cash, continue the 55-symbol scan.
- **Full `evaluate_entry` pipeline wiring** for replacement entries.
- Tests 10–23, 25–26, 29.

---

## 7. Scope guardrails (do not modify)

`.claude/worktrees/`, `trader_rewrite/`, `vendor/`, `tradingview-mcp/`, backup files
(`*.bak`, `*.backup`, `trade.py.pre_killswitch.bak`, etc.), generated datasets, historical
backtest outputs, and unrelated research programs. Preserve all existing staged, unstaged,
and untracked work.

---

## 8. Non-goals (Phase 1)

- No replacement screening, optimizer, or buys (Phase 2).
- No new alpha claims; the −0.5% stop is risk control only.
- No change to the legacy date-keyed file layout (new subsystem is mode-namespaced
  alongside it).
- No live trading by default (`ALLOW_LIVE_TRADING=0`).
