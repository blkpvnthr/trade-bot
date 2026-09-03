# Insider Paired-Call Overlay — REAL chains (hold 3d, call ~31DTE deep-ITM δ≈0.78)

Vol-gated insider SHARES (entry-day rv20 ≥ TRAIN P80=0.0564, 20d $-vol ≥ $1,000,000), 100bps share slippage. Paired just-ITM ~31DTE long call on REAL AV historical chains (entry ask → exit bid, $0.65/contract). Sizing per name: $1,000 shares + up to $1,500 call premium.

OOS sample: 132 most-recent vol-gated signals (of 132).

Real-chain coverage: 113 tradable calls, 8 no-chain, 1 no-qualifying-contract, 10 premium>budget, 5 exits used intrinsic fallback.

| Strategy | n | Mean | Median | Win% | Sharpe | PF | Total $PnL |
|---|---|---|---|---|---|---|---|
| Shares only (baseline) | 132 | +3.36% | +2.26% | 61% | +0.277 | 2.23 | $4,439 |
| Call only (real fills) | 113 | -8.89% | -10.91% | 35% | -0.191 | 0.60 | $-14,951 |
| **Shares + paired call** | 132 | -3.18% | -3.92% | 40% | -0.113 | 0.74 | $-10,512 |

_Overlay marginal effect: **-6.54%** mean/name, **$-14,951** total $PnL vs shares-only; Sharpe -0.113 vs +0.277._

**VERDICT: the paired-call overlay DOES NOT add value on real fills in this sample.**
_Caveat: long calls bleed theta/spread on real chains (repo prior). Treat a positive result as provisional until confirmed on a larger sample + multiple slippage assumptions._