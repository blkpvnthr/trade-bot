# Real-Chain ATM Filter Study (which ATM trades are worth taking?)

25 contract-trades over 20 liquid mega-caps. Real chains, enter ASK/exit BID + $0.65/side, ATM δ0.45-0.65, spread ≤10%.
**Baseline (all ATM trades): mean +32.4%, Sharpe +0.242, win 47%, n=251**

## 1. DTE buckets

| DTE | n | Mean | Sharpe | Win% | Median | Avg spread | Avg IV |
|---|---|---|---|---|---|---|---|
| 7-14 | 55 | +61.4% | +0.297 | 47% | -27.8% | 3.2% | 33% |
| 15-30 | 55 | +33.7% | +0.239 | 42% | -16.8% | 3.3% | 32% |
| 31-45 | 54 | +25.9% | +0.272 | 48% | -5.8% | 3.6% | 33% |
| 46-60 | 32 | +10.8% | +0.136 | 41% | -4.4% | 1.9% | 31% |
| 61-90 | 55 | +21.1% | +0.296 | 53% | +1.2% | 2.0% | 33% |

## 2. IV filter

| IV bucket | n | Mean | Sharpe | Win% | Median |
|---|---|---|---|---|---|
| all | 251 | +32.4% | +0.242 | 47% | -4.7% |
| IV<30% | 126 | +6.1% | +0.066 | 43% | -11.6% |
| IV 30-50% | 96 | +75.9% | +0.453 | 57% | +28.2% |
| IV 50-80% | 29 | +2.9% | +0.024 | 28% | -22.8% |
| IV>80% | 0 |—|—|—|—|

## 3. Trend strength (daily proxies; intraday ORB/VWAP not in daily bars)

| Filter | n | Mean | Sharpe | Win% | Median |
|---|---|---|---|---|---|
| breakout_dist >2% | 116 | +67.0% | +0.397 | 48% | -3.9% |
| breakout_dist >5% | 42 | +85.8% | +0.438 | 45% | -6.5% |
| dist>SMA50 >5% | 190 | +41.3% | +0.287 | 48% | -3.6% |
| dist>SMA50 >10% | 126 | +51.9% | +0.324 | 47% | -4.4% |
| relvol >2 | 79 | +73.6% | +0.417 | 52% | +12.3% |
| relvol >3 | 28 | +93.0% | +0.452 | 54% | +22.3% |

## 4. Simplest rule that beats the ATM baseline

| Rule | n | Mean | Sharpe | Win% | Median |
|---|---|---|---|---|---|
| 31-45 DTE only | 54 | +25.9% | +0.272 | 48% | -5.8% |
| 31-45 DTE + IV<50% | 49 | +27.6% | +0.284 | 49% | -4.6% |
| IV<50% | 222 | +36.3% | +0.270 | 49% | -3.5% |
| relvol>2 | 79 | +73.6% | +0.417 | 52% | +12.3% |
| relvol>2 + IV<50% | 72 | +75.4% | +0.414 | 53% | +14.5% |
| breakout>5% + IV<50% | 32 | +102.9% | +0.479 | 50% | +46.4% |
| 31-45 DTE + relvol>2 | 17 | +51.6% | +0.453 | 53% | +12.3% |

_Real fills (ask→bid + commission). Daily-bar trend proxies. Prefer fewer filters; n≥20 for a rule to count. NOT optimized — a robustness scan, not a fit._