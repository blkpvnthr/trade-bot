# Insider Copier — Options (Black-Scholes) Backtest

PIT signals, 1719 names. Call: 30DTE, entry delta≈0.8, IV=realised×1.3, extrinsic round-trip spread 40%, min ADV$ 0 (dropped 0 illiquid).

| Horizon | n | Win% | **Opt net mean** | net median | t | Wipeouts | gross mean | Share mean | Lev |
|---|---|---|---|---|---|---|---|---|---|
| 5d | 3498 | 45% | **+5.8%** | -4.9% | +3.19 | 1% | +12.2% | +3.15% | 3.9× |
| 10d | 3231 | 45% | **+9.3%** | -6.2% | +3.34 | 2% | +15.0% | +4.06% | 3.7× |

_Spread charged on TIME VALUE (fair to ITM). NET still optimistic (no vol-crush, realised-vol IV). 10d = copier time-exit._