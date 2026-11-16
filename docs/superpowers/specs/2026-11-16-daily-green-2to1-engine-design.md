# Daily-Green 2:1 Engine (trade.py) - Design

Date: 2026-11-16
Status: Approved for implementation
Target file: `trade.py`
Branch context: 13K-options-dev

## 1. Objective

Make the engine powering `trade.py` enter and exit trades on a single, simple
intraday rule set, replacing the existing momentum/setup-score signal entirely:

1. ENTER a symbol (long shares) as soon as its daily PnL is positive, where
   "daily PnL" is measured against today's session open.
2. EXIT at a 2:1 reward-to-risk ratio.
3. If daily PnL turns negative before the 2:1 target is reached, EXIT as soon as
   it turns negative.

Scope: equities only (long shares). Options path unchanged. Long only, no shorts.

## 2. Formalized strategy

Per symbol, universe = `universe.txt` (loaded by `load_trading_universe()`,
trade.py:216):

| Element             | Definition                                                    |
|---------------------|---------------------------------------------------------------|
| Reference ("zero")  | `open_today` = today's session open                           |
| Entry trigger       | `price > open_today` (symbol green vs open); buy on first flip |
| Risk R              | `R = entry_price - open_today`                                 |
| Stop                | exit when `price <= open_today` (day turns red); reason `day_turned_red` |
| Target              | exit when `price >= entry_price + 2*R`; reason `rr_2to1_target` |
| Resolution          | re-evaluated every cycle (POLL_SECONDS = 30, trade.py:189)     |

Whichever of stop or target is hit first wins.

### Degenerate-risk handling
Entering the instant price crosses the open makes `R` near zero, which makes the
2:1 target near zero and causes churn. Mitigation: a minimum risk floor.

```
risk_distance = max(entry_price - open_today, daily_green_min_risk_pct * entry_price)
target_price  = entry_price + daily_green_rr * risk_distance
stop_price     = open_today
```

Default `daily_green_min_risk_pct = 0.0015` (0.15%), `daily_green_rr = 2.0`.
The stop always stays at the open line; only the target distance is floored.

## 3. Architecture - chosen approach (A)

New strategy mode inside the existing scanner/executor, config-gated, default ON.

- `build_orders` (trade.py:3519) branches at the top: when
  `config.daily_green_strategy_enabled` is True it returns
  `self._build_daily_green_orders()` and never runs the momentum scoring path.
- The entire legacy signal path is bypassed in this mode: scoring
  (`_score_latest`), strict/opportunity/relaxed masks, `_annotate_intraday_entries`,
  `predict_next_bar_direction`, `_add_execution_alpha_gate`, `_add_pyramid_orders`,
  `_trigger_options_booster_if_needed`, and the RL/MTF timing gates in the executor.
  Bypassing these is what makes "replace setup-score entirely" real.
- Everything downstream is reused unchanged: account snapshot, sizing, kill switch
  (`_kill_switch_allows_execution`, trade.py:4220), cash buffer, order plan
  (`_build_order_plan`, trade.py:4715), execution (`_execute_order_plan`,
  trade.py:4856), journaling, equity curve, and the main run loop (`run()`,
  trade.py:5687).

### Rejected alternatives
- B: delete the momentum scanner. Destroys working infra (options booster, RL,
  exit-learning), breaks existing tests, hard to revert. Rejected.
- C: standalone new engine file. User asked for changes in `trade.py`; duplicates
  the run loop and broker wiring. Rejected.

## 4. Components (all in trade.py)

### 4.1 Config fields (ScannerConfig, trade.py:282)
| Field                                 | Default | Meaning                                         |
|---------------------------------------|---------|-------------------------------------------------|
| `daily_green_strategy_enabled`        | True    | Master switch for the new mode                  |
| `daily_green_rr`                      | 2.0     | Reward:risk multiple for the target             |
| `daily_green_min_risk_pct`            | 0.0015  | Minimum risk distance as fraction of price      |
| `daily_green_per_position_weight`     | 0.09    | Equal target weight per green name              |
| `daily_green_max_positions`           | 8       | Max concurrent positions in this mode           |
| `daily_green_flatten_before_close_min`| 5       | Minutes before 16:00 ET to flatten all          |
| `daily_green_reenter_after_stop`      | False   | Allow re-entry same day after a red-day stop    |

### 4.2 New entry builder `_build_daily_green_orders()`
Near trade.py:3519. Flow:
1. `symbols = self._universe()`.
2. `bars = self._download_bars(symbols)`; take the last row per symbol (today's
   bar) to get `open_today` and `close` (current price). Skip symbols with no
   today bar or `open_today <= 0`.
3. Candidate entries: `close > open_today`.
4. Filter out symbols already held and (unless `daily_green_reenter_after_stop`)
   symbols in the same-day no-reentry set (`load_no_reentry_symbols()`,
   trade.py:2242).
5. Respect `daily_green_max_positions` (count existing held + new).
6. Size each with `daily_green_per_position_weight` (bounded by existing
   `max_position_weight`, gross exposure, min notional); compute
   `stop_price`/`target_price`/`risk_reward_ratio` per section 2.
7. Build the exits via `_daily_green_exits(...)`.
8. Write rebalance/exit CSVs and return `(rebalance, exits, scores_placeholder)`
   matching the existing 3-tuple contract.

### 4.3 New exit builder `_daily_green_exits(open_today_by_symbol)`
Near trade.py:3473. For each open position with `qty > 0`:
- `open_today` from the passed map; if absent, fetch via a small bars call for
  that symbol.
- `R = avg_entry_price - open_today`, floored by `daily_green_min_risk_pct`.
- `target = avg_entry_price + daily_green_rr * R`.
- Sell if `current_price <= open_today` (reason `day_turned_red`) or
  `current_price >= target` (reason `rr_2to1_target`).
- On a stop-out, mark same-day no-reentry (`mark_no_reentry_symbol`,
  trade.py:2257) unless `daily_green_reenter_after_stop`.

### 4.4 EOD flatten
When within `daily_green_flatten_before_close_min` of 16:00 ET, the exit builder
emits sells for all open positions (reason `eod_flatten`). Reuse existing ET
time helpers.

## 5. Data flow

```
run() (trade.py:5687)
  -> every POLL_SECONDS
    -> run_cycle_from_files (trade.py:5589)
      -> build_orders (trade.py:3519)  [branch: daily_green mode]
        -> _build_daily_green_orders()
            _download_bars(universe) -> last row/symbol -> open_today, close
            entries where close > open_today  (capped, sized)
            _daily_green_exits(open_map):
              get_all_open_positions() x open_today -> stop/target/eod sells
        -> (rebalance, exits)
      -> run_once_from_frames (trade.py:5076)
        -> _build_order_plan (trade.py:4715) -> _execute_order_plan (trade.py:4856)
           (kill switch, cash buffer, broker submit, journal)
```

Current price source: today's daily-bar `close` for entries, `pos.current_price`
for exits. Caveat: a daily-bar `close` can lag a live quote by up to ~1 minute;
if tighter timing is needed later, swap in a latest-trade snapshot. Default keeps
the existing data path (no new API surface).

## 6. Error handling and edge cases
- Missing/empty today bar for a symbol: skip that symbol this cycle.
- `open_today <= 0`: skip.
- Market closed: governed by existing `_market_is_open_or_wait` (trade.py:4174).
- Kill switch, cash buffer, min-notional, max order dollars: all still apply
  unchanged via the executor.
- Held symbol not in `universe.txt`: exit builder fetches its today open
  explicitly so it is still managed.
- Pyramiding/averaging: disabled in this mode; one entry per symbol per day.

## 7. Testing (tests/test_options_bot.py)
Unit tests with mocked bars and the existing `_make_position` helper:
1. Entry fires only when `close > open_today`; no entry when `close <= open`.
2. Entry respects `daily_green_max_positions` and held/no-reentry exclusions.
3. Stop sell emitted when `current_price <= open_today` (reason `day_turned_red`).
4. Target sell emitted when `current_price >= entry + 2R` (reason `rr_2to1_target`).
5. Min-risk floor applied when `entry - open` is below the floor.
6. EOD flatten emits sells inside the close window.
7. No re-entry after a stop-out when `daily_green_reenter_after_stop` is False.
8. Regression: with `daily_green_strategy_enabled = False`, the legacy momentum
   path is unchanged (existing exit tests still pass).

## 8. Out of scope
- Options path (left unchanged).
- Shorting / negative-day entries.
- Any change to the momentum scoring internals beyond gating them off.
- Backtesting this rule set (separate task if desired).
