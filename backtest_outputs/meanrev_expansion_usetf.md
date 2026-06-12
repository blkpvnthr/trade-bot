# Tier-2 Mean-Reversion Universe Expansion — Gauntlet + §A Filter Promotion

Universe: **40** liquidity-screened names (40 ETFs, 0 stocks), >=1000 bars. Core: RSI(2)<10.0 & close>SMA200 → next-open; exit RSI>70.0 or 6d; 5bps. Portfolio sim: 55% deployable, 10%/name, 5 concurrent.

## 1. Core signal OOS (2024+) — does the ETF edge survive on stocks?

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| ETF-only | 838 | +0.42% | +0.81% | 73% | +0.196 | 1.70 | +5.4% | -3% |
| Stock-only | 0 | — | — | — | — | — | — | — |
| Combined | 838 | +0.42% | +0.81% | 73% | +0.196 | 1.70 | +5.4% | -3% |

## 2. §A filter promotion (greedy forward selection on TRAIN, BH-FDR q=0.10)

| Step filter | ΔPF | ΔSharpe | ΔCAGR | retained | boot-p | kept | note |
|---|---|---|---|---|---|---|---|
| vol_gate(rv20>=P80) | +0.20 | +0.055 | +0.9% | 20% | 0.225 | ❌ | failed keep-rule/FDR (greedy stop) |

**Selected filters:** (none survived — core only)

## 3. Selected config vs core — OOS (2024+)

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| Core only | 838 | +0.42% | +0.81% | 73% | +0.196 | 1.70 | +5.4% | -3% |
| Core + selected | 838 | +0.42% | +0.81% | 73% | +0.196 | 1.70 | +5.4% | -3% |

## 4. §D regime stratification (selected config, full sample)

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| bull / vol-low | 1,040 | +0.30% | +0.79% | 68% | +0.134 | 1.42 | +2.7% | -5% |
| bull / vol-high | 322 | +0.48% | +0.90% | 75% | +0.196 | 1.68 | +1.7% | -4% |
| bear / vol-low | 16 | -1.00% | -0.85% | 38% | -0.484 | 0.31 | -0.4% | -2% |
| bear / vol-high | 98 | +0.17% | +1.17% | 70% | +0.037 | 1.12 | -0.2% | -7% |

### Per-year stability

| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |
|---|---|---|---|---|---|---|---|---|
| 2021 | 290 | +0.71% | +1.24% | 73% | +0.299 | 2.09 | +10.0% | -2% |
| 2022 | 115 | -1.33% | +0.17% | 51% | -0.287 | 0.44 | -7.3% | -8% |
| 2023 | 233 | +0.27% | +0.68% | 64% | +0.131 | 1.39 | +3.9% | -2% |
| 2024 | 423 | +0.37% | +0.78% | 69% | +0.187 | 1.62 | +3.4% | -3% |
| 2025 | 285 | +0.58% | +0.88% | 81% | +0.266 | 2.12 | +6.1% | -3% |
| 2026 | 130 | +0.22% | +0.60% | 67% | +0.089 | 1.29 | +9.2% | -3% |

## 5. Diversification
- Corr of daily PnL vs vol-gated INSIDER edge (45 common days): **+0.07**

## VERDICT

- OOS PF>=2.0: NO (1.70)
- OOS n>=30: YES (838)
- MaxDD>-25%: YES (-3%)
- |insider corr|<0.4: YES (+0.07)
- Non-catastrophic in every regime bucket: YES

**FAIL → keep ETF-only Tier-2; expansion to stocks not validated as configured.**