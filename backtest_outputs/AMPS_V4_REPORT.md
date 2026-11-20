# AMPS V4 — Adaptive Market-Regime & Directional-Eligibility Engine: Research Report

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20 · falsification-first per CLAUDE.md_

## 1. Verdict: **CONTINUE RESEARCH** (kept, forward-paper-testable; not VALIDATED)

The regime + directional-eligibility engine was implemented, integrated (selectable, paper/
shadow), and backtested on Universe V4 (the 20 leveraged/inverse ETFs) with actual IEX 5-min
bars, chronological IS/OOS, a double-cost stress test, and the acceptance criteria applied
verbatim.

**The directional filter + portfolio controls are the first measurable improvement across these
research cycles** — but no variant yet meets the acceptance bar, so the strategy is classified
**CONTINUE RESEARCH**: it is retained (never blacklisted), enabled for live-paper forward
testing, and must not be auto-promoted to real-money. Per the task's own rules and the mission,
I did **not** fabricate a "VALIDATED" label or weaken the criteria.

## 2. What the directional filter does (the new value)
Only go LONG an ETF when (a) the ETF's OWN AMPS long gates pass AND (b) its UNDERLYING momentum
aligns with the ETF's orientation: a bullish ETF needs a bullish underlying; an **inverse ETF
needs a BEARISH underlying** (it rises when the underlying falls). A divergence override allows,
e.g., NVDD when NVDA is strongly bearish even in a bullish broad regime. Reference instruments
(SPY/QQQ/DIA/IWM/SMH/NVDA/TSLA/AAPL/AMZN/MSFT) drive signals only; they are never traded.

## 3. Out-of-sample results (net of costs, bars)
| Variant | OOS trades | Win% | Net exp (bps) | PF | Net Sharpe | Max DD ($) | Accepts? |
|---|---|---|---|---|---|---|---|
| V4.0 baseline (AMPS gates only) | 19,711 | 20.6% | −10.9 | 0.258 | −9.2 | −43,066 | No |
| V4.1 broad-regime filter | 13,766 | 20.2% | −11.1 | 0.252 | −9.4 | −30,570 | No |
| V4.2 underlying filter | 17,351 | 20.8% | −10.6 | 0.268 | −9.1 | −36,896 | No |
| V4.3 combined | 12,385 | 20.6% | −10.9 | 0.261 | −9.2 | −26,912 | No |
| V4.4 + divergence overrides | 16,435 | 20.9% | −10.6 | 0.269 | −9.0 | −34,892 | No |
| **V4.10 combined + portfolio cap** | **1,858** | **29.0%** | **−8.4** | **0.405** | **−5.8** | **−3,119** | No |
| V4.11 V4.10 @ 2× cost | 1,858 | 16.2% | −18.6 | 0.138 | −13.8 | −6,918 | No |

**V4.10 vs V4.0:** profit factor **0.26 → 0.41 (+57%)**, max drawdown **−43,066 → −3,119 (−93%)**,
expectancy **−10.9 → −8.4 bps**, win rate 20.6% → 29.0%. The directional filter + the one-per-
underlying-pair + max-5-concurrent portfolio cap genuinely reduce the bleed and the drawdown.
**But expectancy is still negative and PF (0.41) is far below the required 1.50.**

## 4. Bull vs bear attribution (OOS, net exp bps)
| Variant | Bull ETFs | Bear ETFs |
|---|---|---|
| V4.0 baseline | −11.1 (n=10,946) | −10.7 (n=8,765) |
| V4.10 combined+portfolio | −8.0 (n=758) | −8.6 (n=1,100) |

**The screening-window asymmetry did not replicate.** The cited 4-day baseline (bear ETFs +2.72
bps, bull −7.01 bps) was an artifact of that short window; over the full OOS year both
orientations lose ~−8 to −11 bps. So "always trade the inverse" is not the fix — directionally
correct selection helps symmetrically but doesn't flip either side positive.

## 5. Why it still fails (mechanism, consistent with AMPS V2)
These are high-volatility leveraged ETFs with ~symmetric forward MFE≈MAE and a target-before-
stop probability well under 50%. Directional eligibility removes directionally-wrong trades
(lower drawdown, better PF) but the surviving trades still have negative net expectancy after
~6 bps round-trip costs — the small-bps scalp edge is consumed by cost and symmetric noise.
Double cost is catastrophic (PF 0.14).

## 6. Components delivered
- `amps_v4_regime.py` — instrument registry (20 ETFs → reference/orientation/leverage/category/
  pair), directional score (−100..+100; ADX scales strength, DI sets direction), market-regime
  classification (8 states) from SPY/QQQ + DIA/IWM, **hysteresis** (RegimeState, min-confirm
  bars, activation/deactivation), underlying-momentum alignment, inverse-ETF directional
  eligibility, divergence overrides. Pure + 9 tests.
- `amps_v4_backtest.py` — V4.0–V4.11 experiment sequence with next-bar-open fills, conservative
  intrabar, costs, IS/OOS, portfolio cap + opposing-pair prevention, double-cost stress,
  verbatim acceptance, bull/bear attribution.
- `trade.py` — regime + directional eligibility wired into the AMPS overlay (SHADOW), selectable
  `--strategy amps-v2`, disabled by default; per-symbol eligibility + regime logged to
  `system-data/` for live-paper observability. Zero-arg `python trade.py` unchanged; real-money
  stays behind `ALLOW_LIVE_TRADING`.
- Reuses: amps_v2 engine + feature cache, dmi_adx, costs, existing funnel/risk/dashboard logging.

## 7. Honest scope & limitations
- **Bar-supported research** (sec 16/18): DI/ADX/Bollinger/VWAP/relvol are bar-derivable; the
  order-flow score components + order-flow-divergence setup require genuine ticks and are
  ABSENT/flagged here — never fabricated.
- Portfolio controls tested: max-5-concurrent + one-per-underlying-pair. Full adaptive per-
  instrument suppression and the correlation-threshold matrix are config-ready in the engine but
  the backtest exercised the cap, not the full suppression loop.
- Independent 12-month holdout: the window used is development/validation; it has been inspected,
  so it is NOT an untouched final holdout — disclosed, not misrepresented.

## 8. Decision & next step
**CONTINUE RESEARCH.** The directional-eligibility thesis is partially vindicated (better PF, far
lower drawdown) and is retained + live-paper testable, but historical evidence does not yet show
net profitability, so it is not promoted to real money. The honest signal across six cycles is
that net intraday scalping on this high-vol universe is cost-dominated; the higher-expected-value
path remains the evidenced insider-selection edge, not another intraday variant.

## 9. Reproduce / operate
```bash
python amps_v4_backtest.py                 # V4.0-V4.11 directional-eligibility matrix
python trade.py --strategy amps-v2 --paper # AMPS V2/V4 regime+eligibility in SHADOW (paper)
python -m pytest tests/test_amps_v4.py tests/test_amps_v2.py -q
```
