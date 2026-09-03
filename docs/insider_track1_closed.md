# Insider Book - Track 1 CLOSED (execution-quality overlays)

Status: **CLOSED**. The vol-gated insider-shares book is correctly specified as-is. It is not
bad - it is **done**. No further sizing or exit overlays without a pre-registered OOS
hypothesis.

## Canonical baseline (the deployed edge)

- **Flat sizing + fixed 3-day hold.** Entry: open the day after a qualifying Form 4 open-market
  buy cluster; vol gate `rv20 >= train-P80`; exit at the 3rd trading day's close.
- ~169-181 vol-gated trades, 100 bps round-trip, ~58% win, +2.26% avg net/trade,
  Sharpe(active) ~2.57, max DD ~-7.4%, Calmar ~1.03.
- The edge is **right-tail and bear/high-vol concentrated**: bear regime (SPY < 200SMA) mean
  net +4.30% / 63% win vs bull +1.44% / 58%. Insiders buy their beaten-down stock into fear,
  so the payoff is largest exactly when conventional de-risking would cut exposure.

## What was tested and rejected

| Overlay | Result | Why |
|---|---|---|
| Market/regime-vol **exposure scaling** (trend, inverse-vol, combined) | **FAILED** | Every variant worse on Sharpe AND Calmar AND drawdown. The scalars size *down* in bear/high-vol, which is where the edge lives. `backtest_insider_exposure.py` |
| **Per-name vol-sizing** | **REJECTED by mechanism** (not built) | The vol gate already selects high-vol names and the edge is high-vol-concentrated; sizing by vol within the book fights it for the same reason as above. |
| **Give-back exits** (trailing peak-retracement stop, daily 3d horizon, MFE-via-highs grid) | **FAILED despite better headline metrics** | `backtest_insider_giveback.py` |

## Give-back: the instructive failure

The best give-back variant beat the baseline on every headline number - total return
33.6% -> 37.7%, Sharpe 2.57 -> 2.85, max DD -7.4% -> -6.0%, PF 1.68 -> 1.86. It still FAILED
the strict bar on the two guards that matter:

1. **Amputated the right tail.** 6 of the top-10 baseline winners were cut early; only 62% of
   their gross was retained (top winners +30.2% -> +21.4%). On a right-tail edge, that is
   selling the thing you own it for.
2. **Lost to random.** The rule fired on 73.5% of trades (it is mostly "exit early", barely
   "give-back"). A placebo of *random* early exits at the same frequency did better
   (q90 +61.5%, mean +51.3% vs give-back +37.7%). The timing has negative skill - the apparent
   gain is deleveraging/shorter-hold disguised as signal.

## Decision

- **No more sizing or exit overlays on the insider book** unless a NEW out-of-sample
  hypothesis is **pre-registered** before looking at the data.
- Headline Sharpe/return improvement is **not sufficient** to promote an overlay. See the
  guardrail below.
- This is the third independent confirmation of a standing pattern: the meta-agent study
  already found "trade-selection beats every management/sequence/giveback overlay". The edge
  wants to be left alone.

## Promotion guardrail (insider-book overlays)

> Do not promote an insider-book overlay based only on Sharpe/return improvement. To pass it
> must ALSO (a) preserve top-decile winners (mean retention >= 85%, <= 2 of top-10 cut) and
> (b) beat a same-frequency random early-exit placebo. Encoded in
> `tests/test_insider_overlay_guardrail.py` (`backtest_insider_giveback.overlay_promotable`).

## Open lead (NOT acted on)

The placebo's strength (random early exit beat the fixed hold) hints the **3-day hold may be
too long** for these high-vol names - a hold-length question, separate from give-back. It is
in-sample on these trades and carries its own tail-clipping risk, so it is a lead for
disciplined OOS testing only, not a result.

## Artifacts

- `backtest_insider_exposure.py` (+ `backtest_outputs/insider_exposure_backtest.{json,md}`)
- `backtest_insider_giveback.py` (+ `backtest_outputs/insider_giveback_backtest.{json,md}`)
- `tests/test_insider_overlay_guardrail.py`

Next marginal research hour -> `earnings_within_7d` (first ACCEPTED deployable signal), see
`docs/earnings_within_7d_sleeve_plan.md`.
