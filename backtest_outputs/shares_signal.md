# Production Signal on SHARES — the clean no-options test

Momentum signal → next-day OPEN → close at horizon. NO options/greeks/IV/theta/spread. Universe: AAPL, MSFT, NVDA, META, AMZN, GOOGL, AVGO, TSLA, AMD, PLTR, QQQ, SPY. Benchmark = UNCONDITIONAL forward return on ALL days (same horizon) — the signal has edge ONLY if it beats simply being long.

## 3-day hold

| Set | n | Mean | Median | Sharpe | Win% | MaxDD | PF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Signal entries** | 210 | +0.41% | +0.00% | +0.076 | 50% | -34% | 1.25 |
| **All-days benchmark** | 15624 | +0.37% | +0.40% | +0.062 | 54% | -100% | 1.22 |

_Signal edge over unconditional: **+0.04%** mean per 3-day trade (signal +0.41% vs market +0.37%)._

## 5-day hold

| Set | n | Mean | Median | Sharpe | Win% | MaxDD | PF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Signal entries** | 210 | +0.97% | +0.57% | +0.150 | 54% | -40% | 1.53 |
| **All-days benchmark** | 15600 | +0.57% | +0.55% | +0.077 | 55% | -100% | 1.28 |

_Signal edge over unconditional: **+0.40%** mean per 5-day trade (signal +0.97% vs market +0.57%)._

## 10-day hold

| Set | n | Mean | Median | Sharpe | Win% | MaxDD | PF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Signal entries** | 210 | +1.49% | +1.32% | +0.133 | 61% | -92% | 1.53 |
| **All-days benchmark** | 15540 | +1.10% | +1.06% | +0.105 | 57% | -100% | 1.39 |

_Signal edge over unconditional: **+0.40%** mean per 10-day trade (signal +1.49% vs market +1.10%)._

## 20-day hold

| Set | n | Mean | Median | Sharpe | Win% | MaxDD | PF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Signal entries** | 207 | +1.66% | +2.01% | +0.110 | 57% | -99% | 1.41 |
| **All-days benchmark** | 15420 | +2.10% | +1.96% | +0.140 | 59% | -100% | 1.55 |

_Signal edge over unconditional: **-0.43%** mean per 20-day trade (signal +1.66% vs market +2.10%)._

## Interpretation

- Signal PF>1 & mean>0 & beats benchmark → the signal has edge; **options destroyed it**.
- Signal PF<1 / mean<0 / ≈benchmark → the **signal itself is weak** (stop optimizing option structures).
_Fixed 1R = raw per-trade share return. MaxDD on chronological 100%-allocation equity._
