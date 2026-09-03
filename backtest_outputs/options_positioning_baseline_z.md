# Options-Positioning Signal — BASELINE (name-relative: pcr & call-share vs each ticker's trailing 26-obs baseline (z), weekly grid)

Bearish-crowding → mean-reversion-up test. Walk-forward by year (orientation+z fit on PRIOR years only). Stock close→+H close, 15bps round-trip slip. 2620 events, 2021-01-01→2026-06-01. Q5 = signal says LONG, Q1 = avoid/short.

## Horizon 5d — pooled OOS (walk-forward)

Pooled Q5−Q1 **-0.00%** · Q5 mean +0.14% win 48% PF 1.07 (n=470) · Q1 mean +0.14% PF 1.06.
Sign stability: **pcr:UNSTABLE, cshare:STABLE +**.

| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |
|---|---|---|---|---|---|---|---|
| 2022 | 554 | + | + | +2.01% | -1.18% | 40% | 0.64 |
| 2023 | 572 | − | + | +0.28% | +1.40% | 56% | 2.20 |
| 2024 | 533 | − | + | -1.92% | +0.11% | 45% | 1.06 |
| 2025 | 520 | − | + | -0.90% | +0.08% | 47% | 1.04 |
| 2026 | 171 | − | + | -0.20% | -0.20% | 35% | 0.92 |

## Horizon 10d — pooled OOS (walk-forward)

Pooled Q5−Q1 **-2.18%** · Q5 mean +0.06% win 49% PF 1.02 (n=470) · Q1 mean +2.24% PF 2.14.
Sign stability: **pcr:UNSTABLE, cshare:UNSTABLE**.

| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |
|---|---|---|---|---|---|---|---|
| 2022 | 554 | + | − | -1.22% | -2.83% | 38% | 0.43 |
| 2023 | 572 | + | + | -1.38% | +2.20% | 64% | 2.28 |
| 2024 | 533 | − | + | -1.63% | +0.75% | 50% | 1.29 |
| 2025 | 520 | − | + | -1.51% | +0.18% | 50% | 1.07 |
| 2026 | 171 | + | + | -0.57% | +3.31% | 56% | 2.85 |

## Horizon 21d — pooled OOS (walk-forward)

Pooled Q5−Q1 **-0.74%** · Q5 mean +1.48% win 58% PF 1.33 (n=470) · Q1 mean +2.21% PF 1.64.
Sign stability: **pcr:STABLE +, cshare:UNSTABLE**.

| Year | n | pcr sign | cshare sign | Q5−Q1 | Q5 mean | Q5 win% | Q5 PF |
|---|---|---|---|---|---|---|---|
| 2022 | 554 | + | − | +0.55% | -2.23% | 41% | 0.66 |
| 2023 | 572 | + | + | -0.52% | +5.05% | 71% | 4.54 |
| 2024 | 533 | + | + | -0.38% | +3.15% | 67% | 1.86 |
| 2025 | 520 | + | − | +3.74% | +4.85% | 67% | 2.83 |
| 2026 | 171 | + | − | -2.06% | +1.56% | 47% | 1.53 |

---
_Honest costs: stock round-trip slippage only; no borrow/short costs modeled for the Q1 (short) leg. NOT wired to the live bot — deploy only if signs stay STABLE across years. Baseline-relative is the fairer test; compare its sign-stability to the cross-sectional run._