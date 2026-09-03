# Intraday reversal engine backtest

- Generated: 2026-06-11T20:03:59.250979-04:00
- Universe: 118 cached symbols | 58 trading days | avg 2.67 eligible/day
- Scan @ 13:30 ET | target = Previous Day High | trail 0.75% | overnight carry: True | **rel-vol top-N/day: 3**
- Gates from bars: price>5, vol>3M, chg>0 (GREEN today), perf_1w<0. Analyst/OS 'strong buy' NEUTRALIZED (no historical ratings in cache) — this tests the price setup + PDH-target management, not the fundamental overlay.

## Realized trade performance (rel-vol top-N names per day)

| metric | value |
|---|---|
| n | 117 |
| hit_rate | 0.2906 |
| avg_ret_bps | -50.11 |
| median_bps | -40.81 |
| profit_factor | 0.362 |
| t_stat | -3.24 |
| avg_win_bps | 98.0 |
| avg_loss_bps | -112.13 |
| pct_hit_pdh | 0.4786 |

- **Exit reasons:** {'trail_stop': 54, 'target_pdh': 53, 'eod_close': 5, 'target_pdh_d2': 3, 'next_day_trail': 2}
- target_pdh = hit the Previous Day High; trail_stop = trailing stop; eod_close = flat-or-down at the close; *_d2 / next_day_trail = overnight carry sold into the next day's high.

## rel-vol top-N-per-day sensitivity

Keep only the N most-relatively-active (rel_vol_5m desc) eligible names each day.

| top_N/day | n | avg bps | PF | t-stat | hit-rate | % hit PDH |
|---|---|---|---|---|---|---|
| 1 | 58 | -63.51 | 0.311 | -2.47 | 0.3448 | 0.4828 |
| 2 | 94 | -57.55 | 0.29 | -3.37 | 0.2766 | 0.4787 |
| 3 | 117 | -50.11 | 0.362 | -3.24 | 0.2906 | 0.4786 |
| 5 | 142 | -47.02 | 0.374 | -3.5 | 0.2746 | 0.4577 |
| 10 | 155 | -46.9 | 0.358 | -3.79 | 0.271 | 0.4516 |
| uncapped | 155 | -46.9 | 0.358 | -3.79 | 0.271 | 0.4516 |

## Caveats

- Gross returns (no slippage/fees/borrow); trailing stops fill AT the level, not through overnight/intrabar gaps (optimistic on gap-downs).
- The analyst/OS fundamental gate is not modeled; live, it narrows the universe to fundamentally-liked names, which can change the realized edge materially.
- Entry at the decision-bar close (the live engine uses a marketable limit).