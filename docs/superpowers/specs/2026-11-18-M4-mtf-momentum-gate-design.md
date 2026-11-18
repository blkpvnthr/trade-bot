# M4 - Cached Multi-Timeframe Momentum Gate - Design

Date: 2026-11-18
Status: Spec (pending one strictness decision)
Milestone: M4 of ROADMAP-2026-11-17-trading-engine-evolution.md
Constraints: paper-only, nonblocking (historical load + classification off the hot path),
no look-ahead (completed candles only), additive + flag-gated, preserve M1/M2/M3 safeguards.

## Objective
Add a cached multi-timeframe (MTF) momentum confirmation gate to the streaming daily-green
entry path. A long entry requires: price green vs today's open (existing) AND bullish
higher-timeframe trend AND bullish 5-minute momentum AND a valid 1-minute trigger. The
engine is long-only (daily-green), so short logic is out of scope.

## Data (off the hot path)
- Fetch up to 180 days of 5-minute OHLCV per symbol asynchronously at bootstrap (new
  scanner-level fetch with an explicit 180-day window; via run_in_executor). Cache the
  raw 5m frame per symbol.
- Resample completed 5m candles to 30m and 60m (pandas resample, OHLCV agg).
- Refresh incrementally in the supervisor (append newly completed 5m bars; recompute
  resamples + classification). Never fetch 180 days when an entry signal fires.
- No look-ahead: only COMPLETED candles are used (drop the in-progress bar by timestamp).

## Pure, testable components (new module mtf_gate.py)
- `resample_ohlcv(df_5m, rule) -> df`: 5m -> "30min"/"60min" (open=first, high=max,
  low=min, close=last, volume=sum), drop empty buckets.
- `ema(series, span) -> series`.
- `classify_trend(df_htf, ema_fast=20, ema_slow=50, slope_lookback=5, swing_lookback=10)
   -> "BULLISH"|"BEARISH"|"NEUTRAL"`:
   - need >= ema_slow+slope_lookback bars else NEUTRAL (insufficient).
   - fast=EMA(ema_fast), slow=EMA(ema_slow) on close.
   - aligned_up = close[-1] > fast[-1] > slow[-1]; aligned_dn = close[-1] < fast[-1] < slow[-1].
   - slope_up = fast[-1] > fast[-1-slope_lookback]; slope_dn = fast[-1] < fast[-1-slope_lookback].
   - structure_up = higher-high AND higher-low over swing_lookback (recent half vs prior half);
     structure_dn = lower-high AND lower-low.
   - BULLISH iff aligned_up and slope_up and structure_up; BEARISH iff aligned_dn and
     slope_dn and structure_dn; else NEUTRAL.
- `ltf_momentum_bullish(df_5m_today, ema_fast=9, ema_slow=21, ret_lookback=6,
   min_rel_vol=1.0) -> bool`:
   - session VWAP = sum(close*vol)/sum(vol) (repo close-weighted convention).
   - True iff close[-1] > vwap AND EMA9[-1] > EMA21[-1] AND close[-1] > close[-ret_lookback]
     (rising) AND rel_vol (last vol / mean vol) >= min_rel_vol.
- `one_min_trigger(bars_1m, lookback=3) -> bool`: micro-breakout - last 1m close >
  previous 1m close AND last close >= max(high of the prior `lookback` 1m bars). Needs
  >= lookback+1 bars else False.
- `mtf_entry_ok(htf30, htf60, mom5m_ok, trig1m_ok, require_both_htf=True) -> bool`:
   - htf_ok = (htf30 == BULLISH and htf60 == BULLISH) if require_both_htf else
     (BULLISH in {htf30, htf60} and "BEARISH" not in {htf30, htf60}).
   - return htf_ok and mom5m_ok and trig1m_ok.

## Decisions (user, 2026-11-18)
- Gate default ON; HTF strictness LENIENT (either 30m/60m bullish, neither bearish).
- Max concurrent (distinct-symbol) positions = 5.
- Pyramiding: add to ANY profitable, still-bullish position on a minor pullback (not only
  at the cap); Moderate sizing (dip 0.3-1.0% off post-entry high, add = 0.50x base tranche,
  max 2 adds/symbol).
- Reversal profit-exit: SENSITIVE - when a held position is in profit AND 5m momentum is
  lost (price < session VWAP OR EMA9 < EMA21), exit at profit instead of waiting for the
  2:1 target. Losers still ride to the open stop (day_turned_red).

## Part 2: position management (cap + pyramid + reversal exit)

New pure helpers (mtf_gate.py):
- `momentum_lost(df_5m_today, ema_fast=9, ema_slow=21) -> bool`: close[-1] < vwap OR
  EMA9[-1] < EMA21[-1] (the reversal signal; inverse of the entry-momentum check).
- `pullback_add_signal(last_price, avg_entry, peak_price, mom5m_bullish, dip_lo=0.003,
  dip_hi=0.010) -> bool`: in profit (last>avg_entry) AND dip_lo <= (peak-last)/peak <= dip_hi
  AND mom5m_bullish.

SymbolState additions: peak_price (max price since entry; reset on entry/flat), adds_count.

Integration (streaming_engine):
- Position cap applies to DISTINCT held/in-flight symbols only (adds do not count as new
  positions). daily_green_max_positions = 5.
- on_trade (held symbol): update peak_price; evaluate exits first (risk), then pyramid add.
- Reversal profit-exit in _maybe_exit: after the daily_green stop/target check, if NOT
  already exiting AND position in profit (current > avg_entry) AND cached mom5m_lost ->
  exit reason "reversal_profit_exit".
- Pyramid add `_maybe_pyramid(st)`: if enable_pyramiding AND adds_count < pyramid_max_adds
  AND pullback_add_signal(last, avg_entry, peak, cached mom5m_ok) AND not in_flight ->
  submit a BUY add of pyramid_add_weight_mult x base tranche (reuse sizing + full risk
  gates: kill switch, cash buffer/committed, gross, min-notional, fractionable). On fill,
  adds_count += 1; broker refresh updates avg_entry/qty. Emit DECISION action=PYRAMID.

## Config (ScannerConfig)
- enable_mtf_gate: bool = True
- mtf_require_both_htf: bool = False  (lenient: either bullish, none bearish)
- daily_green_max_positions = 5  (changed from 8)
- enable_pyramiding: bool = True
- pyramid_dip_lo: float = 0.003, pyramid_dip_hi: float = 0.010
- pyramid_add_weight_mult: float = 0.50, pyramid_max_adds: int = 2
- enable_reversal_profit_exit: bool = True
- mtf_htf_ema_fast=20, mtf_htf_ema_slow=50, mtf_slope_lookback=5, mtf_swing_lookback=10
- mtf_ltf_ema_fast=9, mtf_ltf_ema_slow=21, mtf_ltf_ret_lookback=6, mtf_min_rel_vol=1.0
- mtf_1m_lookback=3
- mtf_history_days=180, mtf_refresh_seconds=60, mtf_freshness_max_seconds=300

## Integration (streaming_engine.py)
- Per-symbol cached MtfState {htf30, htf60, mom5m_ok, updated_ts}; a rolling deque of
  recent completed 1m bars in SymbolState (fed by on_bar).
- Bootstrap: fetch 180d/5m per symbol (off hot path), compute resamples + classification,
  seed MtfState.
- Supervisor: every mtf_refresh_seconds, append new 5m bars + recompute MtfState (off hot
  path). Mark freshness; if stale (> mtf_freshness_max_seconds) treat as NEUTRAL (no entry).
- on_bar: append completed 1m bar to the symbol's rolling 1m deque.
- _maybe_enter: when enable_mtf_gate, after the green-vs-open decision and before risk,
  require mtf_entry_ok(cached htf30/htf60, cached mom5m_ok, one_min_trigger(1m deque)).
  If not confirmed, emit DECISION skip reason "mtf_not_confirmed" + TREND event and return.
- Emit TREND events (htf30/htf60/mom5m) so the M3 dashboard shows momentum state.
- When enable_mtf_gate is False, entry behavior is exactly today's green-vs-open.

## Acceptance criteria
- Pure functions match hand-computed fixtures; NEUTRAL on insufficient bars; no look-ahead
  (in-progress candle excluded).
- Gate OFF -> streaming entry unchanged (regression).
- Gate ON -> entry requires green + HTF bullish + 5m momentum + 1m trigger; a non-confirming
  symbol is skipped with reason mtf_not_confirmed (test via seeded MtfState + fakes).
- Historical 180d/5m load is async and never runs inside an entry decision.
- MTF state refreshes off hot path; staleness -> NEUTRAL -> no entry.

## Testing
Unit: resample, ema, classify_trend (bullish/bearish/neutral/insufficient/no-look-ahead),
ltf_momentum_bullish, one_min_trigger, mtf_entry_ok (strict + either). Integration (fakes):
_maybe_enter blocked when MtfState not bullish; allowed when seeded bullish + 1m trigger;
gate-off regression. All no-network.

## Out of scope
Short entries; M5 TradingView; M6/M7. Live stays disabled.
