# Autonomous Rotating Active Universe — Design Spec

**Date:** 2026-11-23
**Branch:** 13K-development (+ apply to `.claude/worktrees/market-scanner`, the live trader925 deployment)
**Status:** Approved (design) — pending implementation plan

## Problem

The market scanner subscribes to a rotating 30-of-~6000 symbol window on a single
Alpaca IEX websocket. The current `SubscriptionManager` advances its rotation
cursor and re-subscribes **every ~3s rank tick**. Once the deadlock in
`StreamClient.apply_subscriptions` was fixed (so the loop actually iterates), this
~3s churn disrupts IEX delivery: the stream connects, delivers briefly, then sits
in `connecting` (`stream_live=False`) — even the mandatory SPY/QQQ references stop
delivering — so the scanner never holds `healthy`, the fail-closed entry gate never
opens, and no scanner-path entries can fire.

Separately, the engine should become an **autonomous, continuously rotating
universe**: a persistent baseline plus dynamically discovered/ranked opportunities,
rather than a static 30-symbol watchlist.

## Goals

1. Maintain a **persistent baseline universe** (30 named tickers) that is never
   *permanently* removed by the Opportunity Finder; baseline members may rotate
   out of the active window temporarily and are re-evaluated for reinsertion.
2. Keep a **maximum of 30 unique tickers in the active scanning window** without
   permanently restricting the engine to 30 securities (the master universe is
   larger: baseline ∪ discovered).
3. Allocate the active window **risk-first**, then split the remainder on a
   flexible **20/10 baseline/opportunity** default.
4. Rotate the active window on a **60-second interval** (decoupled from the ~3s
   ranking/publish tick) so the IEX feed stays delivering and the scanner can hold
   `healthy`.
5. Make opportunity selection **ranking-driven + informed by persisted per-symbol
   stats** (a simple, falsifiable heuristic — not an ML alpha source), with a clean
   seam for a future learned allocator.

## Non-goals

- No learned/RL capacity allocator this phase (seam only; future work).
- No change to strategy entry thresholds, order-execution logic, risk controls,
  the fail-closed entry gate, or exit handling.
- LIVE trading stays disabled.

## Constants & defaults

```python
# market_scan/baseline.py
BASELINE_UNIVERSE = (
    "ANT", "GPN", "DVA", "HUM", "UNH",
    "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU",
    "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS",
    "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)
ACTIVE_UNIVERSE_SIZE = 30
```

Config knobs (env-overridable, in `market_scan/config.py`):
- `rotation_interval_seconds: float = 60.0` (`MARKET_SCAN_ROTATION_SECONDS`)
- `baseline_active_slots: int = 20` (`MARKET_SCAN_BASELINE_SLOTS`)
- `opportunity_active_slots: int = 10` (`MARKET_SCAN_OPPORTUNITY_SLOTS`)

`max_subscriptions`/`ACTIVE_UNIVERSE_SIZE` remain the hard cap (30). The 20/10
split applies to the room left **after** risk-first pinning and is scaled to fit.

## Architecture

### 1. `market_scan/baseline.py` (new)
Holds `BASELINE_UNIVERSE`, `ACTIVE_UNIVERSE_SIZE`, and helpers (e.g.
`is_baseline(sym)`, normalized set). Pure data/constants — no I/O.

### 2. Master universe assembly (scanner)
Discovery (`discover_us_listed`, 300s cadence) produces the discovered set.
**Master = BASELINE ∪ discovered ∪ monitoring.** The scanner always merges
`BASELINE_UNIVERSE` into the rankable entry pool so the Opportunity Finder can
never permanently drop baseline names — baseline only rotates out of the *active*
window, never out of the master.

### 3. Slot allocation — `SubscriptionManager.plan_allocated(...)` (new method)
Keeps the existing `plan(priority_groups)` intact (back-compat). New method fills
the 30-slot active window in strict order, hard-capped at `ACTIVE_UNIVERSE_SIZE`:
1. **Pinned / mandatory (never dropped):** open positions, pending orders,
   references (SPY/QQQ/DIA/IWM). Dropping a mandatory symbol over the cap is a loud
   warning (as today) — exit risk is still covered by REST reconciliation.
2. **Baseline allocation:** up to `baseline_active_slots` from baseline members not
   already pinned, selected by a rotation cursor (round-robin) so all baseline
   names get periodic coverage. Baseline is never removed from the master, only
   rotated within its slice.
3. **Opportunity allocation:** up to `opportunity_active_slots` from
   discovered-non-baseline, ordered by **blended score = current rank score ×
   persisted opportunity stat** (see §4), with rotation among equals.

Rotation cursors advance **once per rotation (60s)**, not per rank tick.
Returns the same `SubscriptionPlan` shape (`target/to_add/to_remove/dropped/
mandatory_dropped/overflow/active/available`).

### 4. `market_scan/opportunity_stats.py` (new)
`OpportunityStats`: per-symbol persisted heuristic state.
- Fields per symbol: decaying EMA of rank score, eligibility count, seen count,
  `last_seen_ts`.
- `update(symbol, score, eligible, now)`: update EMA (configurable decay), counts.
- `weight(symbol) -> float`: multiplier (≥ a floor so unseen symbols aren't zeroed
  out) used to blend with the live rank score for opportunity ordering.
- Persistence: atomic JSON write (temp + `os.replace`) to
  `opportunity_stats_{mode}.json`; load on init; survives restarts.
- Bounded: cap tracked symbols (e.g. a few hundred); evict stale/lowest-weight.
- Pure logic + a thin file layer; fully unit-testable with an injected clock and
  tmp path. Never raises into the scanner loop (fail-safe).

### 5. Scanner cadence split (`market_scan/scanner.py`)
Refactor `_rank_tick` into two methods driven by `run()`:
- `_maybe_rotate_subscriptions(now)` — runs when
  `now - _last_rotation >= rotation_interval_seconds` **or** the mandatory set
  (positions/pending) changed since last rotation. Builds the plan via
  `plan_allocated`, `await self.stream.apply_subscriptions(...)`, commits, updates
  `OpportunityStats`, sets `_last_rotation`. **Only place that churns the socket.**
- `_rank_and_publish(now)` — every ~3s: ranks the entry pool from currently
  available features + `_scored` cache, updates `_scored`, computes health, and
  publishes `live_context` + `universe.txt`. No resubscribe.

`run()` seeds an initial rotation before the loop, then calls
`_rank_and_publish` every `rank_refresh_seconds` and `_maybe_rotate_subscriptions`
on its own 60s cadence.

## Data flow

```
discovery (300s) ─┐
BASELINE ─────────┼─► master = BASELINE ∪ discovered ∪ monitoring
monitoring ───────┘
        │
        ├─ every 60s: _maybe_rotate_subscriptions
        │     risk-first pin → 20 baseline (rotating) → 10 opportunity (rank×stat)
        │     → apply_subscriptions (churn here ONLY) → commit → stats.update
        │
        └─ every ~3s: _rank_and_publish
              rank available features → _scored (TTL) → health → publish
              live_context_{mode}.json + universe.txt
                    │
                    └─► fail-closed entry gate (unchanged) → strategy → orders
```

## Error handling

- `OpportunityStats` load/save failures are swallowed (fail-safe); a corrupt stats
  file is ignored and rebuilt.
- Rotation failures (`apply_subscriptions` raising) keep the previous subscription
  set and are recorded in `_errors` (as today); health reflects `connecting`.
- Mandatory-over-cap drops remain a loud warning + recorded error.
- Baseline is always present in the master even if discovery fails entirely.

## Testing

Unit:
- `OpportunityStats`: update/EMA/decay, weight floor, atomic persist + reload,
  bounded eviction, injected clock determinism, fail-safe on bad file.
- `plan_allocated`: risk-first pinning never dropped; 20/10 split honored and
  scaled after pinning; baseline rotates but is never removed from the master;
  hard cap 30; opportunity ordering uses rank×stat; rotation cursor advances only
  per rotation call.
- Scanner cadence: `apply_subscriptions` invoked on the 60s cadence, **not** every
  3s tick (regression for the churn fix); `_rank_and_publish` still publishes every
  tick; positions/pending change forces an immediate rotation.
- Baseline-always-in-pool: baseline symbols are rankable even when discovery
  returns none of them.

Integration:
- End-to-end scanner publish → healthy gate still works with the new cadence
  (extends existing e2e demonstration).
- Churn regression: over N rank ticks within one rotation interval, the websocket
  subscription set is unchanged.

## Rollout

- Implement on 13K-development with TDD; mirror the change into the
  `.claude/worktrees/market-scanner` worktree that trader925 runs.
- Restart trader925; verify the scanner holds `stream_live=True`/`healthy` across
  multiple rotations and that `eligible` can rise above 0 when candidates qualify.
- Do not force trades; LIVE stays disabled; no merge to main without authorization.
