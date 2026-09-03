# Tier-2 Mean-Reversion Universe Expansion — Gauntlet + §A Filter Promotion

Universe: **150** liquidity-screened names (10 ETFs, 140 stocks), >=1000 bars. Core: RSI(2)<10.0 & close>SMA200 → next-open; exit RSI>70.0 or 6d; 5bps. Portfolio sim: 55% deployable, 10%/name, 5 concurrent.

## 1. Core signal OOS (2024+) — does the ETF edge survive on stocks?

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| ETF-only | 220 | +0.53% | +0.79% | 75% | +0.291 | 2.21 | +4.6% | -2% |
| Stock-only | 2,337 | +0.60% | +1.18% | 66% | +0.101 | 1.37 | +13.2% | -12% |
| Combined | 2,557 | +0.59% | +1.11% | 67% | +0.104 | 1.39 | +8.7% | -12% |

## 2. §A filter promotion (greedy forward selection on TRAIN, BH-FDR q=0.10)

| Step filter | ΔPF | ΔSharpe | ΔCAGR | retained | boot-p | kept | note |
|---|---|---|---|---|---|---|---|
| liquidity(dvol>=P50) | +0.12 | +0.030 | +1.2% | 50% | 0.120 | ❌ | failed keep-rule/FDR (greedy stop) |

**Selected filters:** (none survived — core only)

## 3. Selected config vs core — OOS (2024+)

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| Core only | 2,557 | +0.59% | +1.11% | 67% | +0.104 | 1.39 | +8.7% | -12% |
| Core + selected | 2,557 | +0.59% | +1.11% | 67% | +0.104 | 1.39 | +8.7% | -12% |

## 4. §D regime stratification (selected config, full sample)

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| bull / vol-low | 3,097 | +0.39% | +1.02% | 65% | +0.080 | 1.28 | +1.5% | -13% |
| bull / vol-high | 1,030 | +0.55% | +1.16% | 67% | +0.092 | 1.35 | +4.1% | -7% |
| bear / vol-low | 64 | +0.53% | +0.95% | 64% | +0.153 | 1.50 | +0.1% | -3% |
| bear / vol-high | 570 | +0.44% | +1.42% | 67% | +0.070 | 1.23 | +3.2% | -7% |

### Per-year stability

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| 2021 | 766 | +0.60% | +1.08% | 69% | +0.127 | 1.53 | -7.9% | -11% |
| 2022 | 494 | -0.41% | +1.02% | 61% | -0.072 | 0.82 | +5.3% | -7% |
| 2023 | 944 | +0.32% | +1.13% | 64% | +0.069 | 1.22 | +11.0% | -7% |
| 2024 | 1,214 | +0.71% | +1.06% | 67% | +0.125 | 1.53 | +17.6% | -12% |
| 2025 | 975 | +0.41% | +1.06% | 66% | +0.070 | 1.24 | -1.3% | -11% |
| 2026 | 368 | +0.69% | +1.47% | 68% | +0.131 | 1.46 | +15.4% | -6% |

## 5. Diversification
- Corr of daily PnL vs vol-gated INSIDER edge (92 common days): **+0.20**

## VERDICT

- OOS PF>=2.0: NO (1.39)
- OOS n>=30: YES (2557)
- MaxDD>-25%: YES (-12%)
- |insider corr|<0.4: YES (+0.20)
- Non-catastrophic in every regime bucket: YES

**FAIL → keep ETF-only Tier-2; expansion to stocks not validated as configured.**