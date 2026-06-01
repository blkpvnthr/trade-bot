# Strategy Overlay — what can realistically move win-rate / PF / drawdown

Question posed: lift the simulated alpha system from *25 trades / 5 days, +8.27%, win 0.40,
PF 9.1, max DD 0.5%* toward *+20%, win 0.65, PF 25, max DD 0.15%*. This documents the
research, the math, and the **honest, validated** outcome. OFFLINE; no live wiring, no gate or
risk-control changes, no orders.

## 1. The starting numbers were small-sample noise

The original `rl_account_simulation.py` figures came from **25 trades over 5 days**. On the
full **12,206-trade** enriched sample the truth is:

| | win rate | profit factor | mean R |
|---|---|---|---|
| full enriched sample | **0.366** | **1.55** | +0.34 |

So "PF 9.1 / +8.27%" was luck of a tiny window, not an edge to extend.

## 2. The targets are partly impossible (payoff identity)

Outcomes are a fixed ~3:1 bracket (+3R win / −1R loss). Then **PF = 3·w/(1−w)** exactly:

| win rate w | implied PF |
|---|---|
| 0.366 | 1.7 |
| 0.50 | 3.0 |
| **0.65** | **5.6** |
| 0.80 | 12 |
| **0.893** | **25** |

So **win 0.65 and PF 25 cannot both hold** (0.65 forces PF 5.6; PF 25 forces win 0.89).
And lowering the profit target does **not** rescue PF — measured via peak-R:

| target | win rate | PF |
|---|---|---|
| 1.0R | 0.510 | 1.04 |
| 2.0R | 0.391 | 1.29 |
| 3.0R | 0.314 | 1.37 |

Win-rate 0.65 isn't reachable at any target — even a 1R target only hits ~0.51, and PF
collapses to ~1.0. The setups simply don't have a 65% hit rate.

## 3. The legitimate levers (researched) and what they actually did

Literature (VIX/regime filters, fractional-Kelly, volatility targeting, daily loss limits;
see Sources) points to two honest levers. `rl_strategy_overlay.py` implements both, tuned
**in-sample** (first 70% chronological) and reported **out-of-sample** (last 30%) — no
look-ahead (selection uses only `rel_volume`, `mtf_alignment`, `catalyst_strength`,
`sector_regime`, `setup_score`; never `realized_R`/`peak_R`).

### (A) Selection by ex-ante quality — *does not generalise*

| split / policy | trades | win | PF | expectancy |
|---|---|---|---|---|
| in-sample / take-all | 8,544 | 0.350 | 1.42 | 0.265 |
| in-sample / gated (top 9%) | 855 | 0.366 | **1.65** | 0.409 |
| out-of-sample / take-all | 3,662 | 0.403 | **1.89** | 0.523 |
| out-of-sample / gated | 525 | 0.415 | 1.83 | 0.476 |

In-sample, gating lifts PF 1.42→1.65 (+16%). **Out-of-sample it lifts win-rate trivially
(0.403→0.415) but PF *drops* (1.89→1.83) and expectancy drops (0.52→0.48).** The quality
score is in-sample overfitting near the noise floor — it is not a reliable edge on this data.

### (B) Risk management — cuts drawdown, but only as leverage

| account policy (OOS) | return | max DD | PF | Calmar |
|---|---|---|---|---|
| baseline (0.5% risk, no caps) | 27.3% | 4.45% | 1.73 | 6.12 |
| improved (0.25% risk + daily cap + de-risk) | 10.9% | **2.39%** | 1.80 | 4.57 |

Max DD nearly halved — but Calmar *fell* (6.12→4.57): the DD reduction is essentially just
**lower leverage** (0.5%→0.25% risk scales both return and DD), not added skill. Want DD
0.15%? Set risk per trade to ~0.15% — DD scales down ~3×, and so does return. The daily-loss
cap and drawdown-de-risk are genuine *tail* controls, but this benign single-regime 2-week
window has no tail for them to bind on, so they neither help nor hurt here.

## 4. Honest verdict per target

| target | realistic? | why |
|---|---|---|
| +8.27% → +20% | regime/leverage-dependent, not skill | full-sample return is regime-driven; not a repeatable +20% |
| win 0.40 → 0.65 | **no** | setups don't hit 65%; even a 1R target maxes ~0.51 |
| PF 9.1 → 25 | **no** | 9.1 was noise; true ~1.4–1.9; PF 25 needs win 0.89 at 3:1 |
| max DD 0.5% → 0.15% | **yes, trivially** | lower risk-per-trade ~3×; but return scales down too |

What *would* genuinely move risk-adjusted performance (not demonstrable on this 13-day,
single-regime sample): **(1)** a longer multi-regime history so a real **regime/VIX filter**
can cut whole losing stretches (the literature's biggest lever; here `spy_regime` is uniformly
bull so it has no variation); **(2)** a **dynamic/asymmetric exit** instead of the fixed 3:1
bracket (trail winners, scale out) to break the PF = 3w/(1−w) ceiling; **(3)** genuinely
predictive features beyond current enrichment (order-flow, level-2, news) — the current
ex-ante set is near the noise floor out-of-sample.

## Reproduce

```
python rl_strategy_overlay.py          # selection-edge IS/OOS + risk-managed account table
```
Tunables via env: `STRAT_BASE_RISK`, `STRAT_DAILY_LOSS_CAP`, `STRAT_DERISK_K`,
`STRAT_MAX_TRADES_PER_DAY`, `STRAT_QUALITY_SIZING`, `STRAT_VOL_TARGET`.

## Safety

Read-only/offline. No agent promoted, no live/shadow wiring, deployment gate + risk controls
unchanged, IEX-only (no SIP), no order paths, no credentials touched. Tests:
`tests/test_rl_strategy_overlay.py` (11).

## Sources
- [The Easiest Way to Improve a Trading Strategy Without Curve Fitting](https://algotr.substack.com/p/the-easiest-way-to-improve-a-trading)
- [Momentum RSI Strategy Backtest: VIX regime filter (PF 1.64→2.52, lower DD)](https://options.cafe/blog/momentum-rsi-strategy-backtest-results/)
- [Intraday Momentum for ES and NQ — Quantitativo](https://www.quantitativo.com/p/intraday-momentum-for-es-and-nq)
- [Fractional Kelly / VIX-scaled position sizing (arXiv 2508.16598)](https://arxiv.org/html/2508.16598v1)
- [Kelly Criterion position sizing — overbetting/underbetting](https://astuteinvestorscalculus.com/kelly-criterion-position-sizing/)
- [Maximum Drawdown Management — drawdown-based de-risking](https://www.alphaexcapital.com/stocks/stock-portfolio-management/risk-management-for-stock-investors/maximum-drawdown-management)
