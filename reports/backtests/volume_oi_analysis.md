# Volume / OI Ratio Analysis
Tracks Alpha Vantage `HISTORICAL_VOLUME_OPEN_INTEREST_RATIO` across trades to validate whether higher participation predicts better outcomes.

---
## Variant E
- Trades with vol/OI data: **0** / 19
- Average ratio (all): **—**
- Average ratio (winners): **—**
- Average ratio (losers): **—**

---

## Scoring Thresholds Validation

| Score bucket | Ratio range | Raw score | Normalized |
|---|---|---|---|
| Dead contracts | < 0.25 | -15 | 0.00 |
| Below average | 0.25 – 0.75 | +0 | 0.33 |
| Normal activity | 0.75 – 1.50 | +10 | 0.56 |
| Elevated | 1.50 – 3.00 | +20 | 0.78 |
| Strong positioning | ≥ 3.00 | +30 | 1.00 |
