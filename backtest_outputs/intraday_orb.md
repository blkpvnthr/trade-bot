# Intraday Option-Entry Test — ATM baseline vs Sweep-Reclaim ORB (real option bars)

REAL Alpaca 5-min OPTION bars (2024+). Baseline enters the ATM call at 09:45 ET; Sweep-Reclaim enters at the engulf-candle close on pattern days. Same liquid names, ATM δ0.45-0.65, spread≤10%, enter ASK / exit BID (½ real spread) + $0.65/side.

| Arm | n | Mean | Median | Sharpe | Win% | MaxDD | avg MAE | avg MFE |
|---|---|---|---|---|---|---|---|---|
| **ATM baseline (09:45)** | 130 | +5.8% | **-11.5%** | +0.062 | 45% | -97% | -52% | +84% |
| **Sweep-Reclaim ORB (engulf close)** | 27 | -5.0% | **-16.1%** | -0.069 | 37% | -75% | -54% | +70% |

_MAE/MFE = mean max adverse/favorable excursion over the hold (intraday option lows/highs vs entry ask). If Sweep-Reclaim improves median+win+DALL+MAE, it becomes the production entry model._