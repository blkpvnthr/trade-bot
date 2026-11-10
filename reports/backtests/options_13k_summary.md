# Options-First Backtest — $13K Account
## Configuration

| Parameter | Value |
|---|---|
| Start date | 2024-01-01 |
| End date | 2024-12-31 |
| Starting equity | $13,000 |
| Risk per trade | 1.5% (= $195 at start) |
| Max options exposure | 40% |
| Max open positions | 4 |
| Max contracts/underlying | 2 |
| Target DTE | 21 days |
| Call delta range | [0.35, 0.65] (target 0.5) |
| Put delta range | [-0.65, -0.35] (target -0.5) |
| Hard stop | -35% |
| Take profit | 60% |
| Profit lock trigger | 40% |
| Profit giveback | 25% |
| Close before expiry | 3 days |
| Block new entries | within 5 days of expiry |
| Enable long calls | True |
| Enable long puts | True |
| Assumed spread | 6% of mid |
| Commission | $0.15/contract |
| IV model | 30d realized vol × 1.25 (puts ×1.12) |
| Sizing model | **Strict risk budget**: if 1 contract premium × 100 > risk_budget, trade is **SKIPPED**. |

## Performance Summary

| Metric | Value |
|---|---|
| Starting equity | $13,000.00 |
| Ending equity | $11,540.42 |
| **Total return** | **-11.23%** |
| Annualised return | -11.27% |
| Annualised volatility | 4.67% |
| **Max drawdown** | **-13.12%** |
| **Sharpe ratio** | **-3.551** |
| Total trades | 69 |
| Win rate | 21.7% |
| Profit factor | 0.478 |
| Avg win | $89.28 |
| Avg loss | $51.83 |
| Largest winner | $219.21 |
| Largest loser | $-99.30 |
| Avg holding period | 8.1 days |
| Calls traded | 69 |
| Puts traded | 0 |
| Gross profit | $1339.18 |
| Gross loss | $2798.76 |
| Total commissions | $32.70 |

## Trade Breakdown

| Exit reason | Count |
|---|---|
| hard_stop | 52 |
| take_profit | 13 |
| profit_giveback | 4 |

## Skip Summary

| Skip reason | Count |
|---|---|
| contract_cost_exceeds_risk_budget | 170 |
| no_eligible_contract | 26 |
| vix_too_high | 1 |
| insufficient_cash | 0 |

> **$13K risk budget note**: At 1.5% risk = $195.
> Any contract priced above $1.95/share ($195/contract) was skipped entirely.
> This occurred **170** time(s) during the backtest period.

## Methodology Limitations

1. **Synthetic option pricing**: Black-Scholes assumes log-normal returns. Speculative stocks
   (MARA, RIOT, SOUN, etc.) have fat tails; actual option prices would differ materially.
2. **IV proxy**: 30-day realised vol × 1.25 underestimates IV crush after
   earnings or macro events. Real options have term-structure and event vol.
3. **VIX as PCR proxy**: The live system uses Alpha Vantage real-time PCR. VIX is a coarse
   directional substitute — it misses micro-level sentiment shifts.
4. **Monthly expirations only**: Weekly expirations (available for SPY/QQQ/large-caps) are
   not modelled. Real DTE matching would produce different contracts.
5. **Illiquid universe**: The TV screener universe (top 30 from tv_screener_tickers.csv) is
   heavily weighted toward small/speculative stocks. Many fail the dollar-volume filter and
   produce zero option candidates in practice.
6. **Fill simulation**: Assuming 25% of half-spread as slippage on limit orders at mid is
   optimistic; actual fills depend on market conditions and queue position.
7. **No OI/volume data**: Real OI/volume for historical option contracts is estimated from
   stock dollar volume. Actual liquidity may differ significantly.

## Viability Assessment

❌ **NOT VIABLE** — negative total return. Strategy requires significant revision.

## Top 5 Fixes Before Paper / Live Deployment

1. **Universe mismatch** — The TV screener universe is optimised for a different strategy.
   Replace with liquid options underlyings: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMD, META,
   AMZN, GOOGL, and sector ETFs (XLK, SOXX). These have tight spreads, real OI > 10K, and
   reliable IV data from Alpaca OPRA.

2. **Small-account risk sizing** — At $13K × 1.5% = $195/trade, any contract priced above
   ~$1.95/share exceeds the budget and is silently skipped. This means the system may produce
   zero entries for weeks. Either: (a) lower the delta target to find cheaper OTM contracts,
   (b) raise risk_per_trade_pct to 3-4% for small accounts, or (c) add a minimum-1-contract
   fallback with an explicit cap of $X per trade regardless of risk %.

3. **Real-time IV data required** — The live system must pull actual option snapshots from
   Alpaca OPRA to get real bid/ask/IV/delta. The backtest uses BS with realised vol × 1.25;
   the live system's `AlpacaOptionChainDiscovery` is the correct source but needs the paper
   account 401 issue resolved (API key configuration).

4. **PCR overlay needs historical data** — The Alpha Vantage REALTIME_PUT_CALL_RATIO is only
   available for the current day. For the live system, replace with a cached rolling PCR
   computed from end-of-day data (CBOE historical PCR is free). The current backtest uses VIX
   as a proxy, which is directionally correct but imprecise.

5. **Expiration-day risk** — The current close_before_expiration_days=3 protection works for
   monthlies. For weekly options (SPY, QQQ), the system needs to check expiry every day, not
   just on scan days. Also: add a hard gate to never open a new position if expiry is within
   block_new_entries_within_expiration_days=5 days — currently this gate is applied at
   contract selection but not at the scan level.
