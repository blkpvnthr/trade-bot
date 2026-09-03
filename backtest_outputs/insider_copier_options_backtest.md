# Insider Copier — Options (Black-Scholes) Backtest

PIT signals, 1719 names. Call: 30DTE, entry delta≈0.8, IV=realised×1.3, extrinsic round-trip spread 25%, min ADV$ 5,000,000 (dropped 2811 illiquid).

| Horizon | n | Win% | **Opt net mean** | net median | t | Wipeouts | gross mean | Share mean | Lev |
|---|---|---|---|---|---|---|---|---|---|
| 10d | 420 | 48% | **+0.2%** | -2.7% | +0.07 | 1% | +3.4% | +1.41% | 2.4× |

_Spread charged on TIME VALUE (fair to ITM). NET still optimistic (no vol-crush, realised-vol IV). 10d = copier time-exit._