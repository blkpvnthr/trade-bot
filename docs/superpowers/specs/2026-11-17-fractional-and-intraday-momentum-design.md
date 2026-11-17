# Fractional Orders + Intraday-Momentum Gate (trade.py) - Design

Date: 2026-11-17
Status: Approved for implementation
Target file: trade.py
Builds on: docs/superpowers/specs/2026-11-16-daily-green-2to1-engine-design.md

## Objective

Two additions to the daily-green engine, both on the live execution path:

1. Fractional orders (exact-dollar sizing, default ON) so a small account can buy
   high-priced names (QQQ/SPY) instead of getting 0 integer shares.
2. Add an intraday-uptrend momentum gate to entries: a symbol must be green vs
   today's open AND in an intraday uptrend (price above session VWAP and rising),
   evaluated on every cycle including the first cycle at startup.

## Part 1: Fractional orders

User decisions: exact-dollar fractional ALWAYS; flag default ON.

New config: `ScannerConfig.enable_fractional_orders: bool = True`.

Changes (all gated by the flag; flag OFF = byte-identical integer behavior):
- `AlpacaBroker.submit_market_order`: qty becomes float, `qty = round(float(qty), 9)`
  (Alpaca precision limit) instead of `int(qty)`. Fractional requires TIF=DAY,
  which the config already uses.
- Entry builder `build_daily_green_entry_rows`: when enabled, size
  `delta_shares = delta_dollars / price` (float); keep the dynamic-min-notional
  check; require `delta_shares > 0`. When disabled, `int(delta_dollars // price)`
  and skip if `<= 0` (today's behavior).
- `_build_order_plan`: keep `order_qty` as float when the flag is on (do not
  `astype(int)`); `order_qty > 0` and `<= max_single_order_qty` still hold.
- `_execute_order_plan`: `qty = to_float(order_qty)` when the flag is on; existing
  clip math (`max_order_notional // price`, `gross_room // price`) stays but must
  not floor fractional buys - use float division when enabled.
- Exit builder `build_daily_green_exit_rows`: `int(pos.qty)` -> `float(pos.qty)`
  so a fractional position can be sold (REQUIRED: today `int()` truncates a
  sub-1-share position to 0 and it would never exit).
- Fractionability fallback: add cached `AlpacaBroker.is_fractionable(symbol)`
  (via `trading.get_asset`); at submit, if the asset is not fractionable, floor
  the qty to whole shares and skip if that is 0.

All risk gates (dynamic cash buffer, min-notional, max-order, gross cap, kill
switch, cooldown) are unchanged and still apply to the dollar amounts.

## Part 2: Intraday-uptrend momentum gate

> SUPERSEDED by roadmap milestone M4 (cached multi-timeframe momentum gate) in
> docs/superpowers/ROADMAP-2026-11-17-trading-engine-evolution.md. The simple
> single-timeframe gate below was the interim design before the user expanded
> scope to full MTF confirmation. Implement M4, not this section. Kept for history.

New config:
- `ScannerConfig.enable_intraday_momentum_gate: bool = True`
- `ScannerConfig.intraday_momentum_minutes: int = 5`
- `ScannerConfig.intraday_momentum_min_bars: int = 3`

New pure helper (module-level, matches the repo's close-weighted VWAP convention
in `_intraday_direction_metrics`):

```
intraday_uptrend_ok(frame, min_bars) -> bool:
    sort by timestamp; need >= min_bars bars else False
    vwap = sum(close*volume)/sum(volume)   # close-weighted, repo convention
    above_vwap = last_close >= vwap
    rising = last_close / close[-min(len,6)] - 1 > 0
    return above_vwap and rising
```

Wiring in `_build_daily_green_orders` (when the flag is on):
1. Compute green candidate symbols (price > open_today) from `open_price_map`.
2. `self._download_intraday_bars(green_syms, config.intraday_momentum_minutes)`.
3. Keep only symbols whose frame passes `intraday_uptrend_ok(frame, min_bars)`.
4. Restrict `open_price_map` to that set before calling the entry builder.
When the flag is off, behavior is the pure green-vs-open entry (today's behavior).

On-init / startup: the daily-green branch already runs on the first cycle and
enters every currently-qualifying symbol up to `max_positions`, so a mid-session
start batches all green + uptrend names at once. Early in the session (fewer than
`intraday_momentum_min_bars` bars), a symbol does not qualify yet - conservative
by design; it enters once enough bars exist.

## Testing
- Fractional: entry sizing is fractional when enabled (QQQ $661 on $824 -> ~0.11
  sh, not 0); flag OFF = integer parity; exit builder returns float qty for a
  fractional position; submit rounds to 9 dp.
- Momentum: `intraday_uptrend_ok` True when above VWAP and rising, False when
  below VWAP, falling, or too few bars.
- Live check: on the current account, QQQ/SPY produce a fractional entry when
  they are green + uptrend; order clears all gates.

## Out of scope
- Options path, legacy momentum path internals.
- Non-DAY TIF fractional, extended-hours fractional (Alpaca restricts these).
