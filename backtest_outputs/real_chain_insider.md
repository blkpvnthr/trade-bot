# REAL historical chains — EV selector vs ATM heuristic (the ultimate gate)

Signal **insider**. 120 entries on REAL Alpha Vantage chains (fetched 182, 62 had no usable chain). Enter at ASK, exit at BID from the actual exit-date chain — no BS in the P&L.

| Method | n | Mean | Sharpe | Win% | Median | Avg δ | Avg spread |
|---|---|---|---|---|---|---|---|
| Current (ATM δ0.50) | 120 | -35.3% | -0.511 | 21% | -56.9% | 0.50 | 55.7% |
| **EV-max (real greeks)** | 120 | **-18.9%** | **-0.401** | 26% | -14.4% | 0.81 | 30.7% |

_Real IV/greeks/strikes/spreads; realized P&L from the actual chain at exit (enter ASK→exit BID + commission)._