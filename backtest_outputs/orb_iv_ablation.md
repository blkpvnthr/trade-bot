# 2×2 Ablation — ATM / +ORB / +IV / +ORB+IV (real liquid chains)

250 ATM trades, 250 with intraday ORB data. Real chains, ATM δ0.45-0.65, spread≤10%, ask→bid+$0.65/side. ORB=underlying broke 09:30-09:45 high after 09:45 on entry day (Alpaca 1-min). IV gate=[0.3,0.5].

| Arm | n | Mean | Median | Sharpe | Win% | MaxDD |
|---|---|---|---|---|---|---|
| **ATM baseline** | 250 | +12.4% | **-19.3%** | +0.103 | 39% | -97% |
| **ATM + ORB** | 49 | +5.5% | **-8.7%** | +0.077 | 45% | -67% |
| **ATM + IV** | 88 | +9.0% | **-24.5%** | +0.064 | 32% | -92% |
| **ATM + ORB + IV** | 15 | +16.9% | **-8.2%** | +0.209 | 47% | -36% |

## Read
- Judge by MEDIAN + win + MaxDD (means are tail-inflated on long calls).
- KEY: does ATM+ORB+IV beat ATM+ORB? If yes, IV adds value AFTER the structural ORB gate.
  If the ORB column alone already captures most of the median lift, IV is redundant (prefer ORB — not fitted).
_Real fills; ORB is a date-filter (option still priced at the daily chain). Modest sample — robustness scan._