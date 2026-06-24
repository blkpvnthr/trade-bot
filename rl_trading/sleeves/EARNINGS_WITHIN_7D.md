# earnings_within_7d sleeve

Long names with a **forward scheduled** earnings report within the next **7 calendar days**; exit at
the **close of the trading day before the print**. Never holds through an announcement. Long-only,
equal-weight, daily rebalance. **Paper/scanner only — live is impossible.**

## Canonical validation record (do not use the stale one)

Source of truth: `rl_strategy_meta_agent/experiments/orthogonal_features/out/extended_validation.json`
(reproduced 12/12 by `run_extended_validation.py`). Minted mechanically via
`rl_trading.sleeves.promote.from_extended_validation("earnings_within_7d")`.

| metric | value |
|---|---|
| net Sharpe @5bps | **1.11** |
| breakeven | **55.9 bps** |
| turnover/day | **0.242** |
| net bootstrap p | **0.0** |
| placebo p | **0.0** |
| holdout net Sharpe | 0.32 |
| WF years positive | 8/11 (0.73) |
| decision | **ACCEPT_WITH_PROVENANCE_CAVEAT** |

> **Status: `ACCEPT_WITH_PROVENANCE_CAVEAT`** (was plain `ACCEPT`). The metrics above stand
> (net Sharpe 1.11@5bps, breakeven 55.9bps, turnover 0.242/day). Caveat (blocking disclosure, not a
> failure): *Historical validation used actual reportedDate as a near-PIT proxy. Live execution uses
> forward scheduled dates only.* These usually match, but reschedules and announcement-time
> uncertainty mean the historical result is not a perfect PIT replica. Surfaced as
> `promote.CANONICAL_STATUS` / `PROVENANCE_CAVEAT` and in the dashboard payload.

`out/lead_validation.json` is the **superseded intraday run** (turnover 2.98, net Sharpe 0.73,
boot p 0.57). It must NOT drive sleeve design - see `out/lead_validation.SUPERSEDED.md`.
`from_extended_validation` raises if pointed at it.

## Integrated tracker + variants (was `earnings_tracker.py`)

The uploaded `earnings_tracker.py` is integrated as `earnings_features.py` (no longer standalone;
yfinance removed, prices come from cached Alpaca adjusted bars via `rl_trading.data`). It provides
the candidate feature builder, drift/liquidity scoring, the forward AV `EARNINGS_CALENDAR` fetch,
and (fenced) fundamentals/beat scoring.

| variant | signal/ranking | PIT? | default |
|---|---|---|---|
| A | calendar-only, equal weight | yes | **ON** (validated winner) |
| B | calendar + PIT drift/liquidity rank | yes (cached Alpaca bars) | off |
| C | calendar + drift + AV fundamentals/beat | **NO** | off, opt-in only |

Set with `EARNINGS_VARIANT=A|B|C`. C requires `EARNINGS_ALLOW_FUNDAMENTALS=1` and is for live
scanner ranking only.

### PIT Eligibility
- **A is PIT-safe.** Signal = the binary forward-calendar flag; equal weight.
- **B is PIT-safe** because its drift/liquidity inputs are historical cached Alpaca daily bars
  (prices/volume only), and the rank multiplies the calendar flag (gate first, rank second).
- **C is NOT PIT-safe.** AV fundamentals/beat endpoints (OVERVIEW, EARNINGS, INCOME_STATEMENT,
  BALANCE_SHEET, CASH_FLOW, SHARES_OUTSTANDING, EARNINGS_ESTIMATES) are **current snapshots** with
  no historical as-of, so C cannot be backtested PIT and is **excluded from validation and
  promotion** (label `LIVE_ONLY_NON_PIT_BACKTESTABLE`). Disabled by default.

**The firewall:** `earnings_within_7d` is set only by the forward scheduled calendar. Drift and
fundamentals may rank/size a name only after it is already calendar-flagged; they can never create
a signal. `earnings_features.assert_not_signal_source()` enforces this for the fundamentals path.

**Earnings-date provenance caveat (live vs backtest).** The **live** sleeve uses forward-scheduled
dates (AV `EARNINGS_CALENDAR` -> `cal_*.json`). The **historical A/B backtest** uses `reportedDate`
(ex-post) via `NF.load_earnings`, because no historical archive of forward-scheduled dates exists -
this is the **same source the canonical 1.11 record uses** (required for parity) and is a near-PIT
proxy (scheduled dates are announced weeks ahead; only rescheduled events differ). The date-shuffle
placebo (edge collapses to mean -1.45 on shuffled dates) is the guard that the result is not a pure
calendar artifact. Do not read backtest metrics as fully PIT-clean on date provenance; the live
signal path is correct.

## Retrain / A-vs-B validation (canonical parity)

`experiments/orthogonal_features/run_earnings_variants.py` reuses `run_extended_validation`'s
panel + `evaluate` (same frozen bars, same `fut_ret_1`, same `LV.streams` cost model, same
walk-forward / symbol-holdout / placebo / bootstrap) and only changes the signal. Net reported at
5bps (parity) and 10bps round-trip. Result (`out/earnings_variant_validation.json`):

| row | net@5bps | net@10bps | maxDD@10bps | turnover/day | breakeven | placebo p | holdout | WF +yrs |
|---|---|---|---|---|---|---|---|---|
| canonical (raw reportedDate-proxy) | 1.11 | 1.004 | 0.292 | 0.242 | 55.9 | 0.0 | 0.32 | 0.73 |
| A (calendar + liquidity) | 1.01 | 0.896 | 0.257 | 0.225 | 48.0 | 0.0 | 0.33 | 0.82 |
| B (A + drift rank) | 1.04 | 0.916 | 0.253 | 0.232 | 47.9 | 0.0 | 0.35 | 0.91 |

**The canonical raw reportedDate-proxy row reproduces 1.11** (parity check). Variant A is the
liquidity-filtered *live translation* (net 1.01@5bps) - NOT the 1.11 row.

**Winner = A.** B's net@10bps gain (+0.02 Sharpe, within noise) comes with higher turnover, so it
does not clear "strictly better without increasing turnover" -> keep calendar-only A.
**Date-shuffle placebo:** shuffling which rows carry the earnings flag collapses net@10bps to mean
**-1.45** (real +0.90, frac>=real = 0.0) -> the edge genuinely depends on true earnings dates.
**Fundamentals (C):** not PIT-testable -> not retrained, not promoted, default off.

## Daily logic

1. Refresh the forward calendar pre-open (`backtest_cache/earnings_calendar/cal_*.json`, freshest by
   `fetched_at`). Forward-only: earliest scheduled date on/after `asof` per symbol.
2. **Fail closed** (no entries; flatten holds) if: calendar age > 3d, coverage < 0.95, or feed down.
3. Flag = earnings within next 7 calendar days (`1 <= days_to <= 7`).
4. Liquidity filter: price >= $5, 20d dollar ADV >= $5M (preferred $10M).
5. Exclude symbols already held by the insider sleeve.
6. Equal-weight long-only target notionals; gross <= 50% equity; <= 10 names; <= 10% per name;
   8% cash buffer; fractional shares.
7. **Exit before print**: exit when `days_to <= 1` (close day before). A revised-earlier earnings
   date pulls the exit forward automatically. Never hold through `days_to <= 0`.
8. **SPY 200SMA gate** (200d SMA of cached adjusted SPY daily bars): if SPY < 200SMA, default
   `regime_policy="pause"` (block new entries); `"size_down"` and `"off"` are configurable.

## Commands

Scanner-only (default — no orders, shadow log + status only):

```bash
SCANNER_ONLY=1 python -m rl_trading.sleeves.earnings_within_7d --asof 2026-06-24 --json
```

Paper (still requires the gate to pass; live remains impossible):

```bash
SCANNER_ONLY=0 EARNINGS_SLEEVE_PAPER_ENABLED=1 python -m rl_trading.sleeves.earnings_within_7d
```

Paper orders are placed ONLY when `SCANNER_ONLY=0` AND `EARNINGS_SLEEVE_PAPER_ENABLED=1` AND
`sleeve.can_trade()` (gate passed + paper enabled). The `PaperExecutor` independently hard-blocks
anything not tradeable and blocks live unconditionally.

Outputs: shadow CSV `logs/earnings_within_7d/shadow_YYYY-MM.csv` (append-only), dashboard payload
`system-data/earnings_within_7d_status.json`.

## Acceptance criteria (pre-registered — do not tune)

### Shadow stage (>= 4 weeks, SCANNER_ONLY)
- positive realized pre-earnings drift
- drift consistent with the canonical bootstrap CI
- zero PIT/calendar failures
- turnover ~0.2-0.3/day

**Provenance reconciliation (hard, blocks paper — Task 2/3, encoded in `SHADOW_ACCEPTANCE`):**
- `>= 4 weeks` of forward-calendar shadow logs
- **zero holds through the actual reported date** (any `held_through_actual_report=True` is a hard fail)
- calendar-vs-actual **mismatch rate documented**
- **no material degradation** between forward-calendar shadow returns and the reportedDate-proxy expectation
- **every reschedule handled fail-closed or exited early**
- forward-scheduled dates **reconciled against actual reportedDate** for every flagged event

The shadow stage writes `logs/earnings_within_7d/calendar_reconciliation_YYYY-MM.csv` (columns:
`symbol, entry_date, scheduled_earnings_date_at_entry, scheduled_earnings_date_latest,
actual_reported_date, exit_date, held_through_actual_report, date_changed, days_changed,
fail_closed_reason, return_to_exit, return_to_actual_minus_1`) and a summary
`system-data/earnings_within_7d_reconciliation.json` (`{passed, weeks, holds_through_actual_report,
mismatch_rate, ...}`). `reconciliation_gate_passed()` reads it; **missing/incomplete -> paper blocked**.

### Paper stage (>= 8 weeks, after shadow + reconciliation pass)
- >= ~40 trades
- net-of-10bps Sharpe within the canonical bootstrap CI
- acceptable drawdown
- no single-name P&L concentration
- regime split checked for SPY above/below 200SMA

Until shadow **and** the provenance reconciliation pass, the sleeve stays `paper_enabled=False`
(SCANNER_ONLY).

## Final status
- scanner-only: **yes**
- paper-enabled: **no** (blocked until 4wk reconciliation passes)
- live-enabled: **impossible**
- default variant: **A** (calendar-only equal weight)
- variant B: **research-only** unless it clears no-turnover-increase
- variant C: **live-only metadata, non-PIT, disabled** (`LIVE_ONLY_NON_PIT_BACKTESTABLE`)
- canonical status: **`ACCEPT_WITH_PROVENANCE_CAVEAT`**
