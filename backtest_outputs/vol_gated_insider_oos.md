# Vol-Gated Insider SHARES — OOS + Liquidity + Slippage Sweep (hold 5d)

Insider-buy shares, gated on entry-day realized_vol_20 ≥ TRAIN P80 (0.0580). Liquidity filter: 20d avg $-vol ≥ $1,000,000. Train <2023-01-01, test ≥. No lookahead. Hold 5d, enter next open / exit close.

Trades: 952 total, OOS test 382 (48 high-vol gated, 334 low-vol excluded).

## Round-trip slippage 50 bps

| Set | n | Mean | Median | Win% | Sharpe | PF | MaxDD |
|---|---|---|---|---|---|---|---|
| OOS unfiltered | 382 | +1.60% | +0.93% | 55% | +0.211 | 1.85 | -7% |
| **OOS vol-gated (high vol)** | 48 | +4.27% | +2.76% | 62% | +0.304 | 2.34 | -3% |
| OOS excluded (low vol) | 334 | +1.21% | +0.36% | 54% | +0.202 | 1.72 | -6% |

_Gate lift: **+2.68%** mean, PF 2.34 vs 1.85._

## Round-trip slippage 100 bps

| Set | n | Mean | Median | Win% | Sharpe | PF | MaxDD |
|---|---|---|---|---|---|---|---|
| OOS unfiltered | 382 | +1.10% | +0.43% | 51% | +0.145 | 1.52 | -8% |
| **OOS vol-gated (high vol)** | 48 | +3.77% | +2.26% | 62% | +0.268 | 2.12 | -3% |
| OOS excluded (low vol) | 334 | +0.71% | -0.14% | 50% | +0.119 | 1.37 | -8% |

_Gate lift: **+2.68%** mean, PF 2.12 vs 1.52._

## Round-trip slippage 150 bps

| Set | n | Mean | Median | Win% | Sharpe | PF | MaxDD |
|---|---|---|---|---|---|---|---|
| OOS unfiltered | 382 | +0.60% | -0.07% | 49% | +0.079 | 1.25 | -9% |
| **OOS vol-gated (high vol)** | 48 | +3.27% | +1.76% | 58% | +0.233 | 1.92 | -3% |
| OOS excluded (low vol) | 334 | +0.21% | -0.64% | 48% | +0.035 | 1.10 | -14% |

_Gate lift: **+2.68%** mean, PF 1.92 vs 1.25._

## VERDICT (at 100 bps)

- Vol-gated PF ≥ 2.0: YES (PF 2.12)
- Beats unfiltered: YES
- OOS gated n ≥ 30: YES (n=48)

**PASS → vol-gated insider shares is a paper live-candidate. Next: wire a VOLATILITY gate (not KMeans) + liquidity filter into run_insider_trade_copier (shares), keep risk gates, kill if live PF<1.8. NO options overlay.**