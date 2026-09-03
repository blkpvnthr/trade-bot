# Tier-2 Mean-Reversion Universe Expansion — Gauntlet + §A Filter Promotion

Universe: **74** liquidity-screened names (41 ETFs, 33 stocks), >=1000 bars. Core: RSI(2)<10.0 & close>SMA200 → next-open; exit RSI>70.0 or 6d; 5bps. Portfolio sim: 55% deployable, 10%/name, 5 concurrent.

## 1. Core signal OOS (2024+) — does the ETF edge survive on stocks?

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| ETF-only | 808 | +0.43% | +0.66% | 71% | +0.187 | 1.72 | +6.0% | -6% |
| Stock-only | 678 | +0.19% | +0.60% | 68% | +0.074 | 1.26 | +2.8% | -5% |
| Combined | 1,486 | +0.32% | +0.64% | 70% | +0.132 | 1.48 | +6.6% | -3% |

## 2. §A filter promotion (greedy forward selection on TRAIN, BH-FDR q=0.10)

| Step filter | ΔPF | ΔSharpe | ΔCAGR | retained | boot-p | kept | note |
|---|---|---|---|---|---|---|---|
| dip_deep(dist<=P20) | +0.20 | +0.058 | +0.4% | 20% | 0.151 | ❌ | failed keep-rule/FDR (greedy stop) |

**Selected filters:** (none survived — core only)

## 3. Selected config vs core — OOS (2024+)

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| Core only | 1,486 | +0.32% | +0.64% | 70% | +0.132 | 1.48 | +6.6% | -3% |
| Core + selected | 1,486 | +0.32% | +0.64% | 70% | +0.132 | 1.48 | +6.6% | -3% |

## 4. §D regime stratification (selected config, full sample)

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| bull / vol-low | 1,777 | +0.24% | +0.63% | 67% | +0.105 | 1.35 | +3.7% | -5% |
| bull / vol-high | 529 | +0.39% | +0.71% | 70% | +0.157 | 1.54 | +1.8% | -4% |
| bear / vol-low | 17 | -1.02% | -1.10% | 35% | -0.510 | 0.29 | -0.4% | -2% |
| bear / vol-high | 178 | -0.04% | +0.97% | 66% | -0.009 | 0.97 | -0.6% | -8% |

### Per-year stability

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| 2021 | 436 | +0.54% | +0.96% | 70% | +0.217 | 1.75 | +7.7% | -3% |
| 2022 | 165 | -1.18% | +0.20% | 52% | -0.248 | 0.48 | -6.8% | -9% |
| 2023 | 414 | +0.23% | +0.61% | 63% | +0.122 | 1.36 | +3.8% | -3% |
| 2024 | 715 | +0.20% | +0.54% | 66% | +0.090 | 1.29 | +3.3% | -3% |
| 2025 | 541 | +0.61% | +0.77% | 77% | +0.274 | 2.32 | +10.0% | -3% |
| 2026 | 230 | -0.02% | +0.41% | 65% | -0.008 | 0.98 | +7.9% | -2% |

## 5. Diversification
- Corr of daily PnL vs vol-gated INSIDER edge (59 common days): **+0.08**

## VERDICT

- OOS PF>=2.0: NO (1.48)
- OOS n>=30: YES (1486)
- MaxDD>-25%: YES (-3%)
- |insider corr|<0.4: YES (+0.08)
- Non-catastrophic in every regime bucket: NO

**FAIL → keep ETF-only Tier-2; expansion to stocks not validated as configured.**