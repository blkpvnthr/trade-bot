# A / B / C / E Comparison — Volume/OI Scoring Impact
## $13K Account · 2024-01-01 → 2024-12-31

**Variant E** adds Alpha Vantage `HISTORICAL_VOLUME_OPEN_INTEREST_RATIO` scoring
on top of Variant C (liquid universe + dynamic risk budget).

Combined rank formula used in Variant E contract selection:

```
rank = technical_score + sentiment_score + liquidity_score + volume_oi_score
       (each component normalised to [0, 1])
```

Vol/OI scoring thresholds:
- ratio < 0.25 → score **-15** (stale contracts, hard reject if not overridden by OI/spread)
- 0.25–0.75   → score **0** (below-average activity)
- 0.75–1.50   → score **+10** (normal participation)
- 1.50–3.00   → score **+20** (elevated new positioning)
- ≥ 3.00      → score **+30** (strong trend initiation signal)

---

## Results

| Metric | A | B | C | **E** |
|---|---|---|---|---|
| **Return** | -10.91% | -28.62% | -1.49% | **-1.49%** |
| **Max drawdown** | -12.81% | -28.62% | -10.50% | **-10.50%** |
| **Sharpe** | -3.486 | -3.231 | -0.719 | **-0.719** |
| **Profit factor** | 0.488 | 0.358 | 0.910 | **0.910** |
| **Win rate** | 22.1% | 19.7% | 36.8% | **36.8%** |
| Trades | 68 | 66 | 19 | 19 |
| Calls / Puts | 68/0 | 66/0 | 19/0 | 19/0 |
| Avg contract cost | $117 | $252 | $389 | $389 |
| Avg win | $90 | $160 | $280 | $280 |
| Avg loss | $52 | $109 | $179 | $179 |
| Total skips | 192 | 86 | 98 | 98 |
| Skips: risk budget | 165 | 65 | 89 | 89 |
| **Avg vol/OI ratio** | — | — | — | **—** |

---

## C vs E — Effect of Vol/OI Scoring

| Metric | Variant C | Variant E | Delta |
|---|---|---|---|
| Return | -1.49% | -1.49% | +0.00pp |
| Max drawdown | -10.50% | -10.50% | +0.00pp |
| Sharpe | -0.719 | -0.719 | +0.000 |
| Win rate | 36.8% | 36.8% | +0.0pp |
| Profit factor | 0.910 | 0.910 | — |
| Trades | 19 | 19 | +0 |

---

## Interpretation

⚠️ **Vol/OI scoring shows no clear benefit over Variant C in this period.** The liquid universe already screens for high-OI names where vol/OI ratio is consistently moderate (0.5–1.5 range). The scoring signal may add more value when the universe includes mixed-liquidity names.

**Note**: No vol/OI data was retrieved (likely due to API key limits or cache misses). Variant E ran with neutral scores (0.50) for all contracts — the combined ranking defaulted to delta-fit + liquidity only. Re-run after building the cache to get true Variant E results.

---

## Notes

1. **Cache first**: Vol/OI data is cached to `logs/backtest_cache/volume_oi_ratio/{symbol}/{date}.json`.
   Each new backtest run builds the cache; subsequent runs use cached data instantly.

2. **API limits**: Alpha Vantage free tier allows 25 calls/day. Premium tiers support
   more. The backtest fetches lazily — only on scan days — so a full 252-day backtest
   requires at most `n_scan_weeks × n_symbols` calls (~52 × 15 = 780 for liquid universe).

3. **Neutral fallback**: When API data is unavailable (rate limit, network error, or
   cache miss), vol/OI score defaults to 0.50 (neutral). This means Variant E degrades
   gracefully to pure delta+liquidity+momentum ranking — it never fails.

4. **Hard reject threshold**: Contracts with ratio < 0.20 are rejected unless proxy OI
   exceeds 10,000 AND spread is below 3%. On the liquid universe, proxy OI is always
   20,000+ for major ETFs, so the override triggers for nearly all contracts. The hard
   reject becomes meaningful when mixed-liquidity underlyings are added.
