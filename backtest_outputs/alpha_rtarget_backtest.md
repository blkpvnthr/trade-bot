# Alpha Momentum — R-target sweep with TRAILING-STOP ladder

Two exit models per R-target: SIMPLE (target/stop only) vs LADDER (faithful engine: hard stop −28%, +50% partial sells 50%, trail activates +300% / 80% behind peak, full target +R). Max-hold 25d, 35DTE, IV×1.2, spread 3%. n=259.

| R-target | ATM simple | **ATM ladder** | deep-ITM simple | **deep-ITM ladder** | ladder winner |
|---|---|---|---|---|---|
| 4R | +17.1% | **+5.2%** | +20.0% | **+11.7%** | **deep-ITM** |

## Read
- **LADDER** is the real engine: the trailing stop + partial cash in winners early, so it
  shows how much of the simple-target 4R/6R gain actually SURVIVES live management.
- If ladder net is FLAT across 3R→6R, the trailing stop is capping winners before the high
  target → raising r_target alone won't help; raise r_trail_activate too (let winners run).
- If ladder net still RISES with R, the higher target is reachable under trailing → aim higher.

_Daily-close marks (conservative on intraday touches); NET is an optimistic fill ceiling._

## Risk-adjusted view — LADDER (real engine), per-trade

Sharpe = per-trade mean/stdev. MaxDD = sequential 20%-fixed-fraction equity curve (proxy, ignores overlap). PF = profit factor. Spread 3%, trail +300%.

| Instrument | R | Mean | Std | **Sharpe** | Win% | Median | p5 (tail) | **MaxDD** | PF |
|---|---|---|---|---|---|---|---|---|---|
| ATM δ0.5 | 4R | +5.2% | 89% | **+0.059** | 33% | -33.9% | -65% | **-91.1%** | 1.19 |
| deep-ITM δ0.8 | 4R | +11.7% | 69% | **+0.170** | 45% | -29.3% | -53% | **-72.0%** | 1.57 |

## Read (risk-adjusted)
- **Sharpe** (mean/stdev) is the real protective-vs-aggressive decider — not mean alone.
- deep-ITM should show higher Sharpe + shallower MaxDD + higher win% than ATM (its intrinsic
  value dampens variance); ATM is a lower-Sharpe lottery (fat tail, deep p5).
- If Sharpe is ~flat across R-targets, the target is irrelevant risk-adjusted → pick the lowest
  (3R) for simplicity; the trailing stop is doing the real work.

## Risk vs KELLY FRACTION — deep-ITM ladder (the sizing lever)

### deep-ITM δ0.8 — empirical full-Kelly f* = **0.325** (growth-optimal fraction of equity/trade)

| Kelly mult | fraction/trade | Terminal ×equity | **Max DD** | Calmar (growth/|DD|) |
|---|---|---|---|---|
| ¼-Kelly | 8.1% | 8.0× | **-37%** | 18.76 |
| ½-Kelly | 16.2% | 31.9× | **-63%** | 48.86 |
| full-Kelly | 32.5% | 90.2× | **-90%** | 99.40 |
| 1.5×-Kelly | 48.8% | 34.6× | **-98%** | 34.36 |
| 2×-Kelly | 65.0% | 2.1× | **-100%** | 1.12 |

### ATM δ0.5 — empirical full-Kelly f* = **0.075** (growth-optimal fraction of equity/trade)

| Kelly mult | fraction/trade | Terminal ×equity | **Max DD** | Calmar (growth/|DD|) |
|---|---|---|---|---|
| ¼-Kelly | 1.9% | 1.2× | **-14%** | 1.73 |
| ½-Kelly | 3.8% | 1.5× | **-27%** | 1.65 |
| full-Kelly | 7.5% | 1.6× | **-49%** | 1.29 |
| 1.5×-Kelly | 11.2% | 1.5× | **-66%** | 0.72 |
| 2×-Kelly | 15.0% | 1.1× | **-80%** | 0.12 |

## Read (sizing)
- Growth rises toward full-Kelly then the variance drag flattens it; DRAWDOWN keeps exploding past it.
- ¼–½-Kelly is the standard sweet spot: most of the growth, a fraction of the drawdown.
- This is why the R-target is a sideshow — the fraction-of-equity choice dominates the risk profile.
_Sequential one-trade-at-a-time curve (ignores overlap); comparative, not a literal account path._
