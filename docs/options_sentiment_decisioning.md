# Options Sentiment Decisioning (Alpha Vantage)

Advisory realtime options-sentiment signals folded into the options
decision-making process. These signals **adjust confidence and ranking only** —
they are never execution-authoritative on their own and never override hard risk
controls.

Implemented in [`options_sentiment_av.py`](../options_sentiment_av.py) and wired
into [`options_booster.py`](../options_booster.py).

### Architecture

```
data_sources/alpha_vantage_client.py   unified raw AV client: API key, realtime
  (AlphaVantageClient)                  entitlement, retry/timeout, caching, and
                                        request logging. Owns get_put_call_ratio()
                                        and get_volume_oi_ratio().
        ▲ delegates fetching
        │
options_sentiment_av.py                 DECISION OVERLAY (this feature): normalizers
  (OptionsSentimentClient,              + side-aware evaluate_setup_sentiment() that
   evaluate_setup_sentiment)            re-ranks contracts. No HTTP of its own.
        ▲ consumed by
        │
options_booster.py                      live long-call ranking + order rows.

alpha_discovery/options_feature_builder.py   separate consumer: builds PCR/VOI
                                             features for the RL pipeline (gated,
                                             default-off). Not used here.
```

`options_sentiment_av.py` does **no networking** — it calls the unified client, so
realtime entitlement, retries, the request log
(`logs/alpha_vantage_realtime_requests.csv`, API key always redacted) and the
status report (`reports/data_sources/alpha_vantage_realtime_status.json`) are all
inherited.

---

## 1. What the two Alpha Vantage APIs provide

| Function | Provides | Used as |
| --- | --- | --- |
| `REALTIME_PUT_CALL_RATIO` | Chain-level + per-expiration put-call ratio (PCR) | **Directional** read (call vs put demand) |
| `REALTIME_VOLUME_OPEN_INTEREST_RATIO` | Per-contract volume/OI, aggregated to chain + per-side (call / put) means | **Participation** amplifier (liquidity / fresh flow), *not* directional |

Response shapes (verified against the live `demo` key):

```jsonc
// REALTIME_PUT_CALL_RATIO
{
  "symbol": "IBM",
  "put_call_ratio_full_chain": "0.23",
  "put_call_ratio_by_expiration": [
    { "date": "2026-06-19", "value": "1.73" },
    { "date": "2026-07-17", "value": "0.45" }
  ]
}

// REALTIME_VOLUME_OPEN_INTEREST_RATIO  (per-contract; aggregated to chain + per-expiration)
{
  "symbol": "NVDA",
  "data": [
    { "expiration": "2026-05-29", "type": "call", "strike": "100",
      "volume_open_interest_ratio": 0.42 }
  ]
}
```

The client aggregates the per-contract volume/OI rows into a chain mean and a
per-expiration mean.

---

## 2. Environment setup

```bash
# Preferred (per integration spec); falls back to the legacy key if unset.
export ALPHA_VANTAGE_API=your_key_here
# Legacy fallback already present in .env:
# ALPHA_VANTAGE_API_KEY=...
```

Keys are **never hardcoded**. With no key present the client returns neutral
values and the options decision process is unchanged (safe no-op default).

### Tunable env vars (all optional; sane defaults shown)

Networking / key / entitlement are owned by `AlphaVantageClient`:

| Var | Default | Meaning |
| --- | --- | --- |
| `ALPHA_VANTAGE_API_KEY` (alias `ALPHA_VANTAGE_API`) | — | Key, env only, never logged |
| `ALPHA_VANTAGE_ENTITLEMENT` | `realtime` | Realtime entitlement for AV requests |
| `ALPHA_VANTAGE_TIMEOUT_SECONDS` | `12` | HTTP timeout |
| `ALPHA_VANTAGE_MAX_RETRIES` | `2` | Retry attempts on transient errors |
| `ALPHA_VANTAGE_CACHE_SECONDS` | `30` | Realtime cache TTL (avoids rate limits) |

Decision-overlay thresholds (this feature):

| Var | Default | Meaning |
| --- | --- | --- |
| `OPTIONS_SENTIMENT_AV_ENABLED` | `true` | Master toggle for the advisory overlay |
| `OPTIONS_SENTIMENT_PCR_BULLISH_MAX` | `0.60` | PCR ≤ this = bullish call demand |
| `OPTIONS_SENTIMENT_PCR_NEUTRAL_MAX` | `1.00` | PCR in (bullish, this] = neutral/mixed |
| `OPTIONS_SENTIMENT_PCR_EXTREME_LOW` | `0.30` | PCR ≤ this = contrarian-bearish warning |
| `OPTIONS_SENTIMENT_PCR_EXTREME_HIGH` | `1.80` | PCR ≥ this = contrarian-bullish warning |
| `OPTIONS_SENTIMENT_VOI_WEAK` | `0.10` | Vol/OI ≤ this = weak participation |
| `OPTIONS_SENTIMENT_VOI_STRONG` | `0.50` | Vol/OI ≥ this = strong participation |
| `OPTIONS_SENTIMENT_DIRECTION_WEIGHT` | `0.25` | Max directional swing on the multiplier |
| `OPTIONS_SENTIMENT_PARTICIPATION_SWING` | `0.12` | Max participation swing on the multiplier |
| `OPTIONS_SENTIMENT_CONTRARIAN_DAMP` | `0.50` | How much contrarian readings are pulled toward neutral |
| `OPTIONS_SENTIMENT_MULT_FLOOR` | `0.60` | Hard floor on the confidence multiplier |
| `OPTIONS_SENTIMENT_MULT_CEILING` | `1.40` | Hard ceiling on the confidence multiplier |
| `OPTIONS_SENTIMENT_SKIP_LOW_CONFIRMATION` | `false` | If true, drop low-participation/unfavorable contracts (advisory, conservative) |
| `OPTIONS_SENTIMENT_LOW_PARTICIPATION` | `0.20` | Participation below this is "low confirmation" |
| `OPTIONS_SENTIMENT_NAKED_MIN_PARTICIPATION` | `0.45` | Min participation expected for a naked call |

---

## 3. Interpretation rules

### Put-call ratio → `directional_score ∈ [-1, 1]`

| PCR | Band | Read |
| --- | --- | --- |
| ≤ 0.30 | `extreme_low_contrarian_bearish` | Crowded call skew → **contrarian warning**, mild bearish tilt |
| 0.30 – 0.60 | `bullish_call_demand` | Bullish (lower = stronger), `+1.0 … +0.3` |
| 0.60 – 1.00 | `neutral_mixed` | `+0.2 … -0.2` |
| 1.00 – 1.80 | `bearish_put_demand` | Bearish (higher = stronger), `-0.3 … -1.0` |
| ≥ 1.80 | `extreme_high_contrarian_bullish` | Capitulation → **contrarian warning**, mild bullish tilt |

### Volume/OI → `participation_score ∈ [0, 1]`

Linear ramp between `voi_weak` (0) and `voi_strong` (1). High = strong fresh
short-term participation (better confirmation/liquidity); low = weak
confirmation (the move may lack new flow). **Direction-agnostic.**

---

## 4. Contrarian caveats

Extreme PCR readings (`≤ 0.30` or `≥ 1.80`) raise `contrarian_warning = true`.
When set, the advisory multiplier is pulled toward `1.0` by `contrarian_damp`
because extreme readings are unreliable as straight directional signals. The
system does **not** flip a trade on a contrarian reading by itself — technicals
must agree. The warning is journaled so it is auditable.

---

## 5. How confidence is adjusted

For a given contract and side (`call` / `put` / `naked_call`):

```
aligned          = directional_score          (calls)   |  -directional_score (puts)
direction_mult   = 1 + DIRECTION_WEIGHT * aligned
participation_mult = 1 + PARTICIPATION_SWING * (2*participation - 1)
confidence_mult  = direction_mult * participation_mult
if contrarian:  confidence_mult = 1 + (confidence_mult - 1) * CONTRARIAN_DAMP
confidence_mult  = clip(confidence_mult, MULT_FLOOR, MULT_CEILING)   # bounded nudge

option_score    *= confidence_mult            # ranking only
```

The multiplier is **hard-bounded to `[0.60, 1.40]`** so sentiment can re-rank but
never dominate the EV/flow/liquidity score.

### Call setups
- Prefer calls when PCR ≤ 0.60, unless an extreme-low contrarian warning fires.
- Penalize bullish calls when PCR ≥ 1.00 (and warn on conflict with a bullish
  technical trend), unless extremely elevated and technicals strongly agree.
- Boost on strong participation; penalize on weak participation.

### Put setups
- Mirror image: prefer puts when PCR ≥ 1.00 (alignment is `-directional_score`).
- Participation is the same liquidity amplifier.

---

## 6. How naked-call decisions use the signals

Naked calls have stricter advisory handling in `evaluate_setup_sentiment(..., side="naked_call")`:

- Require **bullish technical confirmation** (`technical_bullish=True`) — sentiment
  alone can never justify a naked call.
- Require a **non-hostile** PCR (negative alignment emits a safety warning).
- Require **strong participation** (`participation ≥ naked_call_min_participation`);
  weak participation flags `skip_low_confirmation`.
- Any sentiment-vs-technical conflict attaches a `safety_warning`.

> The current production `options_booster.py` trades **long calls only**. The
> put / naked-call rules are implemented and unit-tested for when those paths are
> enabled.

---

## 7. Safety limits (sentiment is advisory only)

Sentiment may raise or lower confidence and re-rank contracts. It is **forbidden
from overriding**:

- max options exposure / max position size
- stop loss, drawdown limits, no-trade gates
- liquidity, spread, and slippage checks
- account-level risk limits and sizing

This is enforced structurally: the multiplier only scales `option_score`
(ranking). Sizing in `build_booster_orders()` is driven purely by
`risk_budget` / premium-at-risk, which sentiment never touches. The unit test
`test_sentiment_does_not_alter_hard_risk_fields` asserts that
`probability_of_profit`, `premium_at_risk_per_contract`, and EV fields are
identical with and without sentiment.

---

## 8. Example log / journal output

Per-underlying scan line (stdout):

```
[sentiment] NVDA: PCR=0.45 dir=+0.65 participation=1.00 contrarian=False
```

Advisory fields written to the ranked-contracts CSV and order rows:

```jsonc
{
  "symbol": "NVDA260717C00120000",
  "underlying_symbol": "NVDA",
  "expiration": "2026-07-17",
  "option_score": 0.9092,
  "put_call_ratio_chain": 0.45,
  "put_call_ratio_expiration": 0.45,
  "volume_oi_ratio_chain": 0.60,
  "volume_oi_ratio_expiration": 0.60,   // chain-level fallback (per-expiration VOI not provided by the API)
  "sentiment_directional_score": 0.65,
  "participation_score": 1.00,
  "contrarian_warning": false,
  "options_sentiment_confidence": 0.70,
  "options_sentiment_confidence_multiplier": 1.302,
  "options_sentiment_safety_warning": "",
  "options_sentiment_decision_impact":
    "side=call pcr=0.45 dir_score=+0.65 participation=1.00 contrarian=False -> conf_mult=1.302"
}
```

A bearish reading on a bullish technical setup:

```jsonc
{
  "put_call_ratio_chain": 1.40,
  "sentiment_directional_score": -0.65,
  "options_sentiment_confidence_multiplier": 0.938,
  "options_sentiment_safety_warning":
    "options sentiment bearish (PCR=1.4) conflicts with bullish technical trend",
  "options_sentiment_decision_impact":
    "side=call pcr=1.4 dir_score=-0.65 participation=1.00 contrarian=False -> conf_mult=0.938 | WARNING: ..."
}
```

A low-confirmation skip event (only when `OPTIONS_SENTIMENT_SKIP_LOW_CONFIRMATION=true`)
is written to the booster event log:

```jsonc
{ "event": "option_sentiment_low_confirmation_skip",
  "payload": { "underlying": "XYZ", "option_symbol": "...", "impact": "... | skip_low_confirmation" } }
```
