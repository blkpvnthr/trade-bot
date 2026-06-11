# Intraday reversal engine backtest

- Generated: 2026-06-11T19:49:53.751442-04:00
- Universe: 118 cached symbols | symbol-days scanned: 155
- Scan @ 13:30 ET | target = Previous Day High | trail 0.75% | overnight carry: True
- Gates from bars: price>5, vol>3M, chg<0 (RED today), perf_1w<0. Analyst/OS 'strong buy' NEUTRALIZED (no historical ratings in cache) — this tests the price setup + PDH-target management, not the fundamental overlay.

## Realized trade performance (entry -> PDH target / trail / overnight)

| metric | value |
|---|---|
| n | 155 |
| hit_rate | 0.271 |
| avg_ret_bps | -46.9 |
| median_bps | -40.81 |
| profit_factor | 0.358 |
| t_stat | -3.79 |
| avg_win_bps | 96.58 |
| avg_loss_bps | -101.13 |
| pct_hit_pdh | 0.4516 |

- **Exit reasons:** {'trail_stop': 73, 'target_pdh': 66, 'eod_close': 9, 'target_pdh_d2': 4, 'next_day_trail': 3}
- target_pdh = hit the Previous Day High; trail_stop = trailing stop; eod_close = flat-or-down at the close; *_d2 / next_day_trail = overnight carry sold into the next day's high.

## Config sensitivity

| min_score | trail % | n | avg bps | PF | hit-rate | % hit PDH |
|---|---|---|---|---|---|---|
| 60 | 0.50 | 213 | -55.13 | 0.146 | 0.2488 | 0.3991 |
| 60 | 0.75 | 213 | -48.85 | 0.305 | 0.2488 | 0.446 |
| 60 | 1.00 | 213 | -45.06 | 0.387 | 0.2582 | 0.4695 |
| 70 | 0.50 | 192 | -54.86 | 0.152 | 0.25 | 0.401 |
| 70 | 0.75 | 192 | -47.33 | 0.326 | 0.2552 | 0.4531 |
| 70 | 1.00 | 192 | -43.15 | 0.411 | 0.2656 | 0.474 |
| 75 | 0.50 | 155 | -55.99 | 0.166 | 0.2645 | 0.4 |
| 75 | 0.75 | 155 | -46.9 | 0.358 | 0.271 | 0.4516 |
| 75 | 1.00 | 155 | -43.67 | 0.427 | 0.271 | 0.4581 |
| 80 | 0.50 | 143 | -45.61 | 0.205 | 0.2727 | 0.3706 |
| 80 | 0.75 | 143 | -35.24 | 0.444 | 0.2867 | 0.4266 |
| 80 | 1.00 | 143 | -31.21 | 0.529 | 0.2867 | 0.4336 |

## Caveats

- Gross returns (no slippage/fees/borrow); trailing stops fill AT the level, not through overnight/intrabar gaps (optimistic on gap-downs).
- The analyst/OS fundamental gate is not modeled; live, it narrows the universe to fundamentally-liked names, which can change the realized edge materially.
- Entry at the decision-bar close (the live engine uses a marketable limit).