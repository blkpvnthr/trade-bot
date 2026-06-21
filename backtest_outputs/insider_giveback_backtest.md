# Insider book - give-back EXIT falsification (Track-1-step-3)

_181 vol-gated trades, 100bps, daily 3d horizon. Trailing give-back stop vs fixed-hold baseline._

| Metric | baseline (fixed 3d) | best give-back |
|---|---|---|
| variant | fixed_hold | mfe0.02_frac0.3 |
| total_return | 0.336 | 0.3772 |
| sharpe_active | 2.569 | 2.849 |
| max_drawdown | -0.074 | -0.0603 |
| pf | 1.684 | 1.862 |
| drop_top5_total | 0.1831 | 0.2078 |
| early_exit_rate | - | 0.735 |

## Tail check (top-10 baseline winners)
- winners cut early: **6/10**, mean retention of their gross: **62%** (baseline +30.2% -> give-back +21.4%)

## Placebo (random early exits, matched frequency)
- give-back best total +37.7% vs placebo q90 +61.5% (mean +51.3%)

## Strict bar

- net_improves: PASS
- sharpe_holds: PASS
- tail_preserved: FAIL
- beats_placebo: FAIL

## VERDICT: FAIL - flat sizing + fixed 3-day hold; do not over-manage the edge