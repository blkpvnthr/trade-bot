# earnings_within_7d - Paper Sleeve Implementation Plan (PLAN ONLY, do NOT deploy)

Framing: the insider book is **done**, not bad. The next marginal research hour goes here
because `earnings_within_7d` is the **first ACCEPTED deployable orthogonal signal** of the
whole arc. This document is a plan. No code is written, no orders are placed, live is out of
scope.

## 0. Reconcile the validation record FIRST (blocking precondition)

Two artifacts disagree and must be reconciled before anything is built:

| Source | net Sharpe @5bps | breakeven | turnover/day | boot p |
|---|---|---|---|---|
| Extended-history run (the ACCEPT, canonical per memory) | ~1.11 | ~56 bps | ~0.24 | ~0 |
| `experiments/orthogonal_features/out/lead_validation.json` (older) | ~0.73 | ~7.9 bps | ~2.98 | ~0.57 |

The extended-history run (Alpaca daily 2016+, walk-forward + symbol-holdout + placebo +
bootstrap + FDR) is what accepted the signal; the older `lead_validation.json` is superseded
and must NOT be used as the design basis. **Action: locate/regenerate the extended-history
artifact, pin it as the canonical `ValidationRecord` source, and archive the stale json.** The
56 bps breakeven and 0.24/day turnover (vs 7.9 bps / 2.98) are the numbers that make this
deployable, so the discrepancy is material, not cosmetic.

## 1. Exact signal definition

- Producer exists: `rl_strategy_meta_agent/experiments/orthogonal_features/news_features.py:204`
  (`_earnings_within`) and `earnings_calendar.py:127` (`earnings_within_signal`).
- **Binary, point-in-time**: `earnings_within_7d(sym, t) = 1` if the nearest FUTURE scheduled
  earnings date falls in `(t, t + 7 calendar days]`, else `0`.
- Validated as a **dollar-neutral cross-sectional long-short** on `fut_ret_1` (daily). The edge
  is a **pre-earnings drift**, not an earnings-surprise bet:
  - event profile: **+0.66% cumulative d-5..d-1**, then **-3.24% d+1..d+5**.
  - => the strategy is long the names drifting INTO earnings and is **out before the print**.
- Key translation decision for a small paper account (cannot run a clean LS short book):
  **long-only basket** of flagged names vs **long/short**. Long-only re-introduces market beta
  that the validated LS netted out, so the long-only drift must be re-checked (see acceptance
  criteria). Default plan: **long-only, market-aware** (size down / pause when SPY < 200SMA),
  with LS deferred unless shorting is cleanly available in paper.

## 2. Data dependencies

- **Forward earnings calendar** (live): `earnings_calendar.py:53` `get_upcoming_earnings()` ->
  `CalendarResult{events: {sym: [YYYY-MM-DD]}, coverage, age_days, ok}`. Massive/Benzinga
  primary; Alpha Vantage `EARNINGS_CALENDAR` (CSV, 3-month) fallback.
- **PIT hazard (the single biggest risk)**: use ONLY *scheduled forward dates* (known to the
  market in advance). Do **NOT** source the flag from AV `EARNINGS_ESTIMATES` / `OVERVIEW` -
  those are snapshot-only and would inject lookahead.
- **Prices**: cached Alpaca daily bars `data/cache/alpaca_bars/{SYM}_day.parquet` (entry/exit
  reference + the 200SMA market gate).
- **Fail-closed**: `ok=False` if `age_days > 3` OR `coverage < 0.95` OR feed down -> sleeve
  goes flat (no trades). Correct, but must be surfaced (see failure modes).

## 3. Entry / exit timing

- **Compute** daily, pre-open, from the refreshed forward calendar.
- **Enter** a name when it first flags `earnings_within_7d` (nearest earnings in the next 7
  calendar days), at the next session open. Equal-weight across flagged names.
- **Exit BEFORE the announcement** - target the close on the trading day *before* the scheduled
  earnings date, to capture the +0.66% pre-drift and avoid the -3.24% post-earnings leg. Never
  hold through the print.
- **Re-check the calendar daily**: companies revise earnings dates; the "exit-before" target
  moves with the schedule. A revision that pulls earnings forward must force an earlier exit.
- Average hold ~4 days (consistent with the canonical ~0.24/day turnover).

## 4. Cost / slippage assumptions

- Validation breakeven ~56 bps; budget realistically **5-10 bps round-trip** for liquid names
  (huge margin under breakeven). Plan the sleeve cost model at **10 bps** to stay conservative,
  and **gate the universe on liquidity** so that assumption holds (Section 5).
- Reuse the sleeve fill/cost model (`rl_trading/sleeves/` + `research_platform/fills.py` default
  ~10 bps round-trip). Do NOT inherit the insider book's 100 bps (that was for thin small-caps).

## 5. Universe filters

- Liquidity floor so the 10 bps cost assumption is real: **20d $-ADV >= $5-10M**, **price >= $5**.
- Optional cap on count per day (e.g., top-N by liquidity) to respect small-account granularity.
- Exclude names already held by the insider sleeve to avoid double-booking the same symbol.
- Start with a **liquid mid/large-cap universe** (the validation universe), not micro-caps.

## 6. Risk caps (small account)

Reuse the existing caps; the sleeve framework already enforces hard limits
(`rl_trading/sleeves/core.py:13` `DEFAULT_LIMITS`):

| Cap | Value | Source |
|---|---|---|
| per-order notional | $2,000 (tighten to DYNAMIC_MAX_ORDER_PCT=0.12 of equity ~ small) | core.py / .env |
| max position % | 10% equity | core.py |
| max gross exposure | <= 1.0 (long-only: <= ~0.5 to keep dry powder) | core.py |
| max concurrent | <= 10 names | core.py |
| daily loss kill | -$500 | DAILY_LOSS_LIMIT / core.py |
| cash buffer | 8% | .env CASH_BUFFER_PCT |
| PDT floor | $2,000 equity (multi-day hold, so PDT pressure is low) | .env |

Small-account note: ~$6k equity, 8% buffer, 12% per-order -> roughly 6-10 names at a few
hundred dollars each; use fractional shares. Keep gross <= ~50% long-only initially.

## 7. Paper-trading acceptance criteria (pre-registered)

Two stages, both required, thresholds fixed BEFORE the window:

1. **Shadow (SCANNER_ONLY) >= 4 weeks**: log intended entries/exits + the realized
   pre-earnings drift of flagged names; require (a) drift sign positive, (b) realized d-5..d-1
   drift within the validation CI, (c) zero PIT/calendar integrity failures, (d) turnover
   ~0.2-0.3/day.
2. **Paper (paper_enabled=True) >= 8 weeks / >= ~40 trades**: net-of-10bps paper Sharpe within
   the bootstrap CI of the canonical validation; max drawdown within plan; no single name >
   X% of P&L (concentration check); win/drift consistent across the SPY>200SMA and <200SMA
   regimes (the insider book taught us regime conditioning matters).

Promotion uses the existing gate `rl_trading/sleeves/promote.from_event_study` ->
`ValidationRecord`; `sleeve.gate()` must pass AND `paper_enabled=True` for PAPER mode
(`core.py:70`). Apply the **same anti-overfit discipline as Track 1**: an overlay or variant
(e.g., long-only vs LS, exit-day tweak) is not promoted on Sharpe/return alone.

## 8. Failure modes

- **Calendar staleness / coverage collapse** -> fail-closed flat; correct but silent. Mitigate:
  daily pre-open cache pre-warm cron; dashboard alert on every `ok=False` with age/coverage.
- **PIT contamination** -> using snapshot endpoints instead of forward scheduled dates =
  lookahead = fake edge. Hard rule: forward calendar only; audit the live path.
- **Beta leak** -> long-only drops the LS short leg; a market selloff can swamp the +0.66%
  drift. Mitigate with the SPY-200SMA pause/size-down and the regime acceptance check.
- **Holding through the print** -> captures the -3.24% reversal. Exit-before-announcement logic
  must be robust to date revisions.
- **Earnings-date revisions** -> stale exit target; re-check daily.
- **Crowding / decay** -> pre-earnings drift is a known public anomaly; monitor for edge decay
  over the paper window and beyond.
- **Small-account granularity** -> too few names, lumpy fills; cap per-name weight, use
  fractional shares.

## 9. Commands / files to implement

Reuse (exists):
- `earnings_calendar.py` - `get_upcoming_earnings`, `earnings_within_signal` (signal producer).
- `rl_trading/sleeves/core.py` (`Sleeve`, `RiskLimits`, `gate()`), `promote.py`
  (`from_event_study`), `paper.py` (`PaperExecutor`, live permanently disabled), `initial.py`
  (`build_default_registry`).
- Paper order path + journaling: `run.py` `submit_market_order` -> `logs/execution/.../executor_journal.csv`.

Need-new (to implement when greenlit, not now):
- Mint the canonical `ValidationRecord` from the reconciled extended-history stats (Section 0).
- `earnings_within_7d_sleeve()` factory in `rl_trading/sleeves/initial.py`: `signal_source=
  "earnings_within_7d"`, `native_execution_type="event_hold"`, entry = flag, exit =
  before-announcement, `eligible_symbols` = liquidity-filtered universe, `risk_limits` per
  Section 6, `paper_enabled=False` initially.
- Portfolio builder: binary flag -> equal-weight long-only notionals, market-aware gate.
- Daily driver (cron, pre-open): refresh calendar -> compute signal -> build weights -> route
  paper orders via the sleeve `PaperExecutor`; plus the calendar pre-warm + staleness alert.
- Shadow logger for the SCANNER_ONLY stage.

## 10. What must be proven before live capital

Live is permanently disabled in the sleeve framework by design (`live_enabled` immutable
False, `paper.py:41`). "Before live" therefore means a SEPARATE, deliberate, out-of-framework
decision with its own gate - not flipping a flag. Preconditions:

1. Validation record reconciled and pinned (Section 0).
2. Shadow + paper acceptance both met over the pre-registered window (Section 7).
3. Long-only (or hedged) variant re-validated if LS is not run in paper (beta does not eat the
   drift).
4. Calendar feed reliability demonstrated (coverage/age SLA) over the full window; zero PIT
   findings in a live-path audit.
5. Net-of-realistic-cost paper Sharpe consistent with the canonical validation CI, and edge not
   visibly decaying.
6. Explicit human sign-off on a small, capped live allocation, with the same kill-switch /
   daily-loss / concentration guards as paper.

Until all six hold, this stays paper-only.
