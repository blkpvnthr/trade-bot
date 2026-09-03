# Options-Positioning Signal — EARNINGS (cross-sectional @ earnings (T+1 after reportedDate), raw pcr & call-share)

Bearish-crowding → mean-reversion-up test. Walk-forward by year (orientation+z fit on PRIOR years only). Stock close→+H close, 15bps round-trip slip. 1076 events, 2019-01-01→2026-06-01. Q5 = signal says LONG, Q1 = avoid/short.

## Horizon 5d — pooled OOS (walk-forward)

Pooled Q5−Q1 **+3.26%** · Q5 mean +4.53% win 46% PF 2.79 (n=171) · Q1 mean +1.27% PF 1.49.
Sign stability: **pcr:UNSTABLE, cshare:STABLE +**.

| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |
|---|---|---|---|---|---|---|---|
| 2021 | 137 | − | + | +0.59% | -0.97% | 41% | 0.60 |
| 2022 | 150 | − | + | +3.75% | +3.48% | 53% | 2.73 |
| 2023 | 152 | − | + | -3.86% | -2.23% | 30% | 0.41 |
| 2024 | 152 | − | + | +1.17% | +2.07% | 47% | 1.67 |
| 2025 | 208 | − | + | -2.21% | +1.83% | 51% | 1.93 |
| 2026 | 57 | + | + | +5.11% | +2.33% | 55% | 2.44 |

## Horizon 10d — pooled OOS (walk-forward)

Pooled Q5−Q1 **-4.76%** · Q5 mean -0.36% win 49% PF 0.91 (n=171) · Q1 mean +4.40% PF 2.48.
Sign stability: **pcr:UNSTABLE, cshare:UNSTABLE**.

| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |
|---|---|---|---|---|---|---|---|
| 2021 | 137 | + | − | -0.27% | -0.89% | 44% | 0.74 |
| 2022 | 150 | + | + | +2.35% | -1.18% | 50% | 0.82 |
| 2023 | 152 | − | + | -5.44% | -1.65% | 30% | 0.66 |
| 2024 | 152 | + | + | -2.67% | +2.61% | 57% | 3.70 |
| 2025 | 208 | − | + | -1.68% | +0.37% | 56% | 1.14 |
| 2026 | 57 | + | + | +4.13% | +1.26% | 64% | 1.14 |

## Horizon 21d — pooled OOS (walk-forward)

Pooled Q5−Q1 **-4.79%** · Q5 mean +0.13% win 49% PF 1.02 (n=171) · Q1 mean +4.91% PF 1.97.
Sign stability: **pcr:UNSTABLE, cshare:UNSTABLE**.

| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |
|---|---|---|---|---|---|---|---|
| 2021 | 137 | + | − | -3.51% | -1.07% | 48% | 0.81 |
| 2022 | 150 | + | + | +5.37% | -2.85% | 43% | 0.63 |
| 2023 | 152 | − | + | -4.69% | +2.24% | 47% | 1.44 |
| 2024 | 152 | + | + | -2.20% | +4.20% | 60% | 3.46 |
| 2025 | 208 | + | + | +0.30% | +1.22% | 61% | 1.21 |
| 2026 | 57 | − | + | -1.21% | +0.12% | 27% | 1.03 |

---
_Honest costs: stock round-trip slippage only; no borrow/short costs modeled for the Q1 (short) leg. NOT wired to the live bot — deploy only if signs stay STABLE across years. Baseline-relative is the fairer test; compare its sign-stability to the cross-sectional run._