# Buffett Insider Transactions (Alpha Vantage)

Advisory insider-buying/selling signal integrated into the Buffett accumulation
screener. Improves ranking and explainability. **Never triggers a buy. Never
overrides hard valuation, quality, liquidity, or risk gates.**

---

## What the Alpha Vantage API Provides

**Endpoint:** `INSIDER_TRANSACTIONS`

```
GET https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=YOUR_KEY
```

Returns a list of SEC Form 4 insider transactions for the requested symbol:

| Field | Description |
|---|---|
| `transaction_date` | Date of transaction |
| `executive` | Insider's name |
| `executive_title` | Title (CEO, CFO, Director, …) |
| `acquisition_or_disposal` | `A` = acquisition, `D` = disposal |
| `transaction_type` | P-Purchase, S-Sale, S-Sale+OE, … |
| `number_of_shares` | Shares transacted |
| `share_price` | Price per share |
| `value` | Dollar value |
| `ownership_type` | `D` = direct, `I` = indirect |

This is a **per-symbol API** (you query per ticker), unlike SEC EDGAR Form 4 which
requires scraping a daily index across all filers. AV provides a cleaner, rate-limited
interface with both buys and sells.

---

## Environment Variable Setup

```bash
# Required
ALPHA_VANTAGE_API_KEY=your_key_here

# Optional (all have working defaults)
AV_INSIDER_FROM_DAYS=180          # lookback window (days)
AV_INSIDER_CACHE_TTL_DAYS=7       # cache freshness
AV_INSIDER_RATE_LIMIT_SECONDS=15  # seconds between API calls
AV_INSIDER_REQUEST_TIMEOUT=20     # request timeout (seconds)
AV_INSIDER_SCALE_USD=1000000      # $1M net buy → tanh ≈ 0.76 on raw score
AV_INSIDER_SELL_DISCOUNT=0.7      # routine sells weighted at 70%
ENABLE_AV_INSIDER=true            # set to false to disable
```

Never hardcode API keys. The module reads `ALPHA_VANTAGE_API_KEY` from the
environment; if absent, it returns neutral results and logs a warning.

---

## Cache Layout

```
data/cache/alpha_vantage/insider_transactions/
    AAPL_20260531.json
    MSFT_20260531.json
    ...
```

Each file is keyed by symbol + today's date. Cache TTL is `AV_INSIDER_CACHE_TTL_DAYS`
(default 7 days). A cached file prevents a duplicate API call on the same day.

---

## Scoring Formula

### Transaction Classification

| `acquisition_or_disposal` | `transaction_type` | Class |
|---|---|---|
| `A` | `P-Purchase` or similar | `open_market_buy` → bullish |
| `D` | `S-Sale` | `open_market_sell` → bearish |
| `D` | `S-Sale+OE`, `ESPP` | `routine_sell` → mildly bearish |
| `A` | `AWARD`, `GRANT`, `RSU` | `non_qualifying` → ignored |

### Role Weights

| Role | Weight |
|---|---|
| CEO, CFO, Chairman, Founder | 1.5× |
| Director, President, COO, Officer, EVP, SVP | 1.0× |
| Generic Insider | 0.6× |
| Unknown | 0.4× |

### `insider_trend_score` (−1 to +1)

```
weighted_buy  = Σ (value × role_weight) for open_market_buy transactions
weighted_sell = Σ (value × role_weight) for open_market_sell transactions
weighted_routine = Σ (value × role_weight × 0.3) for routine_sell transactions

net_weighted = weighted_buy − weighted_sell − weighted_routine

# tanh-normalized; scale_usd = $1M → score ≈ 0.76
raw_score = tanh(net_weighted / scale_usd)

# Cluster bonus: 3+ distinct buyers amplify bullish signal
if raw_score > 0 and cluster_count ≥ 3: raw_score *= 1.25  (capped at 1.0)
if raw_score > 0 and cluster_count ≥ 2: raw_score *= 1.10  (capped at 1.0)

insider_trend_score = clamp(raw_score, −1.0, +1.0)
```

### `insider_confidence_modifier` (−10 to +10)

Addend to the `insider_buy_score` (0−100) fed into `compute_conviction_score`:

| trend_score | modifier |
|---|---|
| ≥ 0.70 | +10 |
| 0.40 – 0.70 | +6 |
| 0.15 – 0.40 | +3 |
| −0.10 – 0.15 | 0 |
| −0.35 – −0.10 | −3 |
| −0.60 – −0.35 | −6 |
| < −0.60 | −10 |

The modifier is applied **only when `bal_ok` is true** (business already passes the
BAL quality gate). For weak businesses, `av_modifier_applied = 0.0`.

Combined insider score is clamped: `max(0, min(100, sec_score + av_modifier))`.

---

## How the Buffett Screener Uses the Signal

1. **Fetch** `get_insider_trend(symbol)` per-symbol (cached).
2. **Apply modifier** to the `insider_buy_score` that feeds `compute_conviction_score`.
3. **Add advisory fields** to every screener output row.
4. **Rank candidates** — a strong bullish AV signal lifts `conviction_score` through
   the 7% insider weight in `compute_conviction_score`.
5. **No gate bypass** — the modifier is applied inside `insider_for_conviction`, whose
   7% weight in conviction keeps its maximum influence bounded.

The insider conviction weight is 7% of the overall conviction score. Maximum AV
modifier (+10) raises the insider sub-score by at most 10 points, contributing
≤ 0.7 points to conviction — a meaningful nudge, never a trump card.

---

## Output Fields (per Screener Row)

| Field | Type | Description |
|---|---|---|
| `av_insider_from_date` | str | Lookback cutoff (YYYY-MM-DD) |
| `av_insider_net_value` | float | Net $ (buy − sell) |
| `av_insider_buy_value` | float | Total open-market buy $ |
| `av_insider_sell_value` | float | Total discretionary sell $ |
| `av_insider_buy_count` | int | Open-market buy transactions |
| `av_insider_sell_count` | int | Discretionary sell transactions |
| `av_insider_buy_sell_ratio` | float | buy_value / sell_value |
| `av_insider_trend_score` | float | [−1, +1] |
| `av_insider_confidence_modifier` | float | [−10, +10] addend to insider score |
| `av_insider_warning_flags` | list[str] | Caution indicators |
| `av_insider_summary_text` | str | Human-readable one-liner |

---

## RL Dataset Features (rl_common.BUFFETT_DCA_FEATURES)

Two new features appended at positions 12–13 (before MTF features):

| Feature | Source | Default |
|---|---|---|
| `insider_trend_score` | `av_insider_trend_score` | 0.0 |
| `insider_confidence_modifier` | `av_insider_confidence_modifier` / 10 | 0.0 |

Old event rows without these keys default to 0.0 (fully backward-compatible).

---

## Decision Rules Summary

| Situation | Action |
|---|---|
| Multiple senior execs buying, large $ | Strong boost to `conviction_score` |
| CEO/CFO buying, moderate $ | Moderate boost |
| No data | Neutral (0.0 modifier) |
| Only routine option-exercise sells | Mild penalty; flagged |
| Heavy discretionary selling, no buying | Caution penalty |
| Senior exec selling, no one buying | Flagged + penalty |
| Valuation or quality gate fails | AV modifier **not applied** |
| Insider signal alone (no quality pass) | **Never triggers a buy** |

---

## Examples

### Bullish: Multiple Executive Buys

```
CEO bought 20,000 shares @ $100 = $2,000,000
CFO bought 10,000 shares @ $100 = $1,000,000
Director bought 5,000 shares @ $100 = $500,000
─────────────────────────────────────────────
insider_trend_score      = +0.92
insider_confidence_modifier = +10
av_insider_warning_flags = []
av_insider_summary_text  = "3 open-market buy(s) by CEO, CFO, Director totalling
                             $3,500,000. No discretionary selling."
```

### Neutral: No Data

```
insider_trend_score        = 0.0
insider_confidence_modifier = 0.0
av_insider_warning_flags   = ["no_recent_insider_data"]
av_insider_summary_text    = "No insider transactions in past 180 days. Neutral."
```

### Bearish: CEO Selling, No Buying

```
CEO sold 50,000 shares @ $100 = $5,000,000 (S-Sale, discretionary)
─────────────────────────────────────────────────────────────────────
insider_trend_score        = -0.82
insider_confidence_modifier = -10
av_insider_warning_flags   = ["no_open_market_buys",
                               "senior_exec_selling_no_buying"]
av_insider_summary_text    = "No open-market buying. 1 discretionary sell(s)
                               totalling $5,000,000."
```

### Mixed: Buy and Sell

```
Director bought 10,000 shares @ $100 = $1,000,000
CFO sold      8,000 shares @ $100 = $800,000 (S-Sale)
─────────────────────────────────────────────────────
insider_trend_score        = +0.12
insider_confidence_modifier = 0.0
av_insider_summary_text    = "1 buy(s) ($1,000,000) vs 1 discretionary sell(s)
                               ($800,000). Mixed activity."
```

### Routine Only: Option Exercises

```
SVP sold 10,000 shares @ $80 = $800,000 (S-Sale+OE)
────────────────────────────────────────────────────
insider_trend_score        = -0.22
insider_confidence_modifier = 0.0
av_insider_warning_flags   = ["only_routine_option_exercise_sells"]
av_insider_summary_text    = "Only routine/planned transactions (2 exercise sales). Neutral."
```

---

## Limitations

- **Free-tier rate limit:** Alpha Vantage free tier allows ~25 requests/day.
  Use the cache to stay within limits; run the screener daily rather than per-cycle.
- **Coverage:** AV only covers SEC-registered issuers. Non-US or OTC-only stocks
  may return empty data (treated as neutral).
- **Lag:** Form 4 filings are due within 2 business days of the transaction.
  Recent transactions may not yet appear.
- **Classification heuristics:** AV does not always expose a clean open-market vs.
  exercise flag; the `transaction_type` heuristic is conservative.
- **No sell signal:** A negative trend score is a **caution** signal, not a sell
  recommendation. The existing exit logic in run.py governs actual exits.

---

## Safety Rules (Enforced in Code)

1. `get_insider_trend()` never raises — failures return neutral values.
2. The AV modifier is only applied when `bal_ok` (BAL score ≥ min_score_to_buy).
3. Combined insider score is clamped to [0, 100] before entering conviction.
4. Insider conviction weight is 7% — maximum AV influence ≤ 0.7 conviction points.
5. No buy is ever generated from insider data alone.
6. No existing gate (BAL score, valuation, liquidity, RiskManager) is modified.
