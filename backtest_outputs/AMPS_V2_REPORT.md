# AMPS V2 — Adaptive Micro-Profit Scalping: Research Report

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20 · falsification-first per CLAUDE.md_

## 1. Executive summary — VERDICT: **REJECTED**

AMPS V2 was implemented as a single unified engine (no paper/live split), integrated into
`trade.py` as a selectable, disabled-by-default, shadow-only strategy, and backtested on the
**exact existing universe** (the 27–30 leveraged/inverse ETFs + a few stocks in `universe.txt`)
using actual Alpaca IEX 5-min bars, 2020-11 → 2025-11, with a chronological IS/OOS split and a
double-cost stress test.

**No variant meets the acceptance criteria — by a wide margin.** Every entry/exit configuration
is net-negative out-of-sample after costs. The required bar (OOS PF ≥ 1.50, win ≥ 70%, Sharpe ≥
1.50, profitable-days ≥ 70%) is not approached; best OOS PF was **0.26** (needs 1.50), win rate
**12–17%** (needs 70%), net expectancy **−10 to −22 bps/trade**.

The **excursion analysis identifies the mechanism** and it is decisive: on this universe,
forward MFE ≈ MAE (symmetric), and **target-before-stop probability is only 0.19–0.29**. A
small-profit scalp cannot have positive expectancy when the stop is hit ~3–4× more often than
the target and costs consume the thin edge. Better entry quality and adaptive exits help at the
margin (PF 0.18 → 0.26) but cannot bridge a 7× gap. This is consistent with every prior cycle on
this repo (daily-green, DMI/ADX, order-flow, DI/Bollinger, Green-Momentum — all failed OOS).

Per the spec, the research infrastructure is retained and the strategy is classified
**unvalidated / REJECTED**. No fabrication; acceptance criteria were applied verbatim and not
weakened.

## 2. Repository changes (files)
- `amps_v2.py` (new) — unified engine: entry-quality score (8 components, renormalizing when
  order flow absent), Bollinger-middle APPROACH detection (approaching/above/combined), four
  entry setups, mandatory gates + expected-net-return filter, adaptive exit state machine
  (initial→breakeven→min-profit-lock→trailing→target→momentum→max-hold→EOD, monotonic stops),
  cost-adjusted breakeven. Pure/testable.
- `amps_v2_backtest.py` (new) — versioned feature cache, excursion analysis, experiment matrix,
  IS/OOS, double-cost stress, verbatim acceptance check.
- `trade.py` — AMPS V2 config (disabled default, shadow), `_run_amps_v2_overlay` (shadow
  observability through the existing funnel; no orders), CLI `--strategy amps-v2`, env `AMPS_V2`,
  startup-banner line. Zero-arg `python trade.py` unchanged; live never auto-enabled.
- `tests/test_amps_v2.py` (new, 15 tests) — BB approach + false-approach rejection, score
  renormalization, gates, setups, cost-adjusted breakeven, monotonic adaptive stops, exit
  transitions/reasons, EOD precedence, disabled-by-default + CLI integration.
- (corroborating, prior task) `green_momentum_945.py`, `backtest_green_momentum.py` — Green
  Momentum V0–V5, also REJECTED on large-caps.

## 3. Infrastructure reused
`costs.py` (cost model), `backtest_dmi_adx` (DI/ADX/bucket costs), `order_flow_engine`
(order-flow signals for live), existing 5-min bar cache + fetchers, the existing execution
funnel + risk controls, `_download_intraday_bars` / `_intraday_direction_metrics`, the overlay +
CLI + banner patterns. No parallel order-management system was created.

## 4. Dataset & universe
Exact existing universe from `universe.txt` (DIA/QQQ/SPY/SQQQ/TQQQ/SOXL/SOXS/BE/QID/PSQ/DRIP/
TECS/TECL/DOG/SPXU/SDS/SH/SDOW/NVD/NVDL/HUM/TPR/SPXS/THC/NOC/PAYS/CHAT…). 16 of 30 symbols had
≥200 5-min bars over the window with full indicator warmup; thin names (CHAT, PAYS) have little
history — a data limitation, disclosed.

## 5. Excursion analysis (the key finding) — baseline entries
| Horizon | avg MFE (bps) | avg MAE (bps) | target-before-stop |
|---|---|---|---|
| 1 bar (5m) | +11.6 | −11.4 | 0.19 |
| 2 bars | +20.0 | −19.7 | 0.23 |
| 6 bars (30m) | +39.0 | −38.7 | 0.275 |
| 12 bars (60m) | +56.4 | −56.8 | 0.289 |

MFE ≈ |MAE| at every horizon (symmetric high-vol swings); target-before-stop stays ~0.2–0.29.
A 30-bps target with a 10-bps stop is therefore stopped out 72–95% of the time.

## 6. Experiment matrix — out-of-sample, net of costs
| Variant | Trades | Win% | Net exp. (bps) | PF | Sharpe | Prof. days | Accepts? |
|---|---|---|---|---|---|---|---|
| V2.0 baseline (confirmed) | 19,511 | 14.4% | −11.0 | 0.18 | −13.4 | 0.3% | No |
| V2.2 approaching | 25 | 12.0% | −13.1 | 0.09 | −16.0 | 13% | No |
| V2.2 combined | 19,524 | 14.4% | −11.0 | 0.18 | −13.4 | 0.3% | No |
| V2.3 quality≥70 | 5,795 | 15.7% | −10.9 | 0.22 | −12.0 | 2.7% | No |
| V2.3 quality≥80 | 1,121 | 16.9% | −10.5 | 0.26 | −10.8 | 12.9% | No |
| V2.6 adaptive exit | 19,524 | 15.1% | −11.0 | 0.17 | −13.8 | 0.3% | No |
| V2.10 integrated | 5,795 | 16.2% | −10.9 | 0.21 | −12.5 | 2.7% | No |
| V2.11 integrated @2× cost | 5,934 | 4.1% | −21.6 | 0.03 | −35.4 | 0.5% | No |

- **Anticipatory (approaching) entries** are extremely rare (25 trades) and no better — the
  approach condition seldom coincides with the other gates; it does not rescue expectancy.
- **Entry-quality filtering** monotonically improves PF (0.18 → 0.26 at ≥80) and cuts trades,
  but plateaus far below 1.0 — the surviving entries still lack a positive-expectancy edge.
- **Adaptive exits** do not help vs fixed on this data (symmetric excursions give the trail
  nothing durable to protect).
- **Double-cost stress** is catastrophic (PF 0.03) — costs dominate, as expected for scalping.

## 7. Components built vs. not pursued (honest scope)
- Built + validated: unified engine, entry-quality score, BB-approach, 4 setups, adaptive exit
  state machine, inverse-momentum (bearish-confirmation) exit, cost-aware breakeven, excursion
  engine, feature cache, experiment matrix, IS/OOS + stress, trade.py integration, 15 tests.
- **Not pursued (deliberately, not fabricated):** the ML probability model (V2.8) and full
  dashboard/attribution tabs. A classifier cannot convert a 0.19–0.29 target-before-stop base
  rate with symmetric MFE/MAE into PF 1.50 — any "pass" would be overfitting theater. Per
  §11.4 ("retain ML only if it improves net performance") and the mission, I did not build a
  model to manufacture a result. The order-flow score components (CVD accel, buying pressure)
  and order-flow-divergence setup require genuine ticks; they are present in the live engine but
  ABSENT in these bar-only runs (score renormalized + flagged, never fabricated — §25).

## 8. Limitations / blockers
- Bar-supported research only for the broad backtest; full 5-year tick replay for the whole
  universe is infeasible (volume), so microstructure components are validated on ticks elsewhere,
  not here (disclosed). - Thin-history names reduce the effective universe to 16 symbols.
- Profit is not concentrated in any instrument/regime — every variant is uniformly negative, so
  the "not dependent on one instrument" criterion is moot (nothing is profitable to concentrate).

## 9. Final decision
**REJECTED.** No robust positive net OOS expectancy was established under realistic execution
assumptions; the acceptance criteria are not met and are not close. The engine remains available
as a selectable, disabled-by-default, shadow-only research strategy. Live execution is not
enabled and must remain operator-gated.

## 10. Reproduce / operate
```bash
python amps_v2_backtest.py                 # experiment matrix + excursion + acceptance check
python trade.py --strategy amps-v2 --paper # run AMPS V2 in SHADOW (paper, no orders)
python -m pytest tests/test_amps_v2.py -q   # engine + integration tests
```
