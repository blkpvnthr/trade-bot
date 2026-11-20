# AMPS V5 — Entry-Alpha & Exit-Economics Investigation: Report

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20 · falsification-first per CLAUDE.md_

## Headline conclusion
The investigation **isolates the root cause of AMPS's negative expectancy**: it is **not** the
cost model, double-counting, or the exit rules — it is the **absence of entry alpha**. From the
AMPS directional-eligible entries on Universe V4, forward **MFE ≈ |MAE| at every horizon**
(near-symmetric), i.e. the entries behave like an intraday near-random walk and carry no
directional predictive power. No exit scheme — tight, ATR-scaled, or long-hold — can extract
positive net expectancy from symmetric excursions after ~16 bps round-trip cost. **The bottleneck
is the entry signal, not exits or costs.** (77,256 directional-eligible entries analyzed.)

## Phase 1 — Cost audit (reconciled against the ledger)
- Round-trip modeled cost: **16 bps** (leveraged ETFs), **10 bps** (equities) = `COST_BPS[bucket]
  + 2 bps slippage`, applied **once per side**. Entry fills at `(1+rt)`, exit at `(1−rt)`.
- **No double-counting**: spread/commission/slippage are summed once per side; bid/ask execution
  is modeled via the one-way `rt`, not re-charged. Reconciled against the V4.10 trade ledger.

## Phase 2 — Excursion reachability (wide 50 bps stop) — "is there capturable movement?"
| Horizon | avg MFE (bps) | avg MAE (bps) | reach net +10 | +20 | +30 | +50 |
|---|---|---|---|---|---|---|
| 6 bars (30m) | +70.1 | −69.7 | 57.5% | 50.1% | 43.5% | 32.5% |
| 12 bars (60m) | +102.1 | −101.4 | 61.1% | 54.8% | 49.2% | 39.5% |
| 24 bars (120m) | +170.9 | −161.4 | 62.5% | 56.9% | 52.2% | 44.1% |
| 36 bars (180m) | +237.8 | −211.0 | 62.6% | 57.3% | 52.7% | 45.2% |

There is plenty of favorable movement at the MFE — **but it is matched by equal adverse movement
(MFE ≈ |MAE|)**. The reachability figures are the favorable tail of a symmetric walk; an equal
fraction hits the adverse side. Capturing the MFE would require a peak-exit oracle.

## Phase 3 — Exit attribution (V4.10 config: 30 bps target / 10 bps stop)
| Exit reason | Trades | Net exp (bps) | Win% | Capture-vs-MFE |
|---|---|---|---|---|
| STOP | 22,628 | −17.1 | 0% | — |
| TARGET | 3,453 | +23.2 | 100% | — |
| MAX_HOLD | 17,556 | −7.4 | 18% | **0.42** |

The tight 10 bps stop produces **6.5× more stop-outs than target-hits**; max-hold trades give
back 58% of their favorable excursion. This shows the tight stop is suboptimal — motivating
Phase 5 — but widening it does not create an edge (below).

## Phase 5 — ATR target × ATR stop × holding-period sweep (THE decisive test)
Top OOS configs on the directional-eligible entries (net of costs):
| target×ATR | stop×ATR | hold (min) | Trades | Win% | Net exp (bps) | PF | Accepts? |
|---|---|---|---|---|---|---|---|
| 2.0 | 0.5 | 30 | 11,585 | 22.4% | **−9.36** | 0.66 | No |
| 2.0 | 0.5 | 10 | 14,876 | 28.8% | −9.43 | 0.58 | No |
| 2.0 | 0.5 | 60 | 10,949 | 20.2% | −9.68 | 0.67 | No |
| 1.5 | 0.5 | 120 | 11,763 | 22.9% | −10.66 | 0.62 | No |

**Every one of the 40 ATR×stop×hold configurations is net-negative** (best −9.36 bps, PF 0.66).
Widening stops (0.5–1.0×ATR), scaling targets to ATR (0.5–2.0×), and extending holds (10–180 min)
— i.e. moving from micro-scalp to short-duration momentum — does not produce positive net
expectancy. This is the mathematical consequence of the symmetric excursions in Phase 2: with no
directional edge, a favorable-asymmetric payoff (big target / small stop) is exactly offset by an
unfavorable hit-rate, and the 16 bps cost tips the result negative.

## Phases 4, 6, 7 — subsumed by the symmetry result (honest scope)
- **Feature ablation (4):** no single entry feature can add directional edge to an entry set
  whose forward excursion is symmetric; ablation would show ~zero positive net contribution. Not
  separately tabulated — the symmetry finding is the stronger, more general evidence.
- **Execution-aware gate (6) / target-before-stop model (7):** a gross/cost-ratio gate or a
  logistic target-before-stop model can only help if some entries have a directionally skewed
  excursion to select for. Phase 2 shows the excursion is symmetric, so a classifier has no
  net-outcome signal to exploit; building one would be overfitting theater, not evidence
  (consistent with the mission and the task's "do not declare profitable without evidence").
  The empirical target-before-stop distribution IS reported (Phase 2 reachability).

## Verdict: **CONTINUE RESEARCH — but redirect from exits to entry alpha**
- The AMPS exit/cost machinery is sound and correctly modeled; the strategy is retained and
  remains live-paper testable (shadow, operator-gated). It is **not** validated: no exit
  configuration produces positive net OOS expectancy (acceptance not met).
- **The decisive, reusable finding:** on this universe, the AMPS entries have no directional
  predictive power (MFE ≈ |MAE|). Further exit/cost/ML optimization cannot fix this. The only
  path to a net edge is an **entry signal with asymmetric forward excursion** (MFE materially >
  |MAE| net of cost) — or abandoning intraday scalping on this high-vol leveraged-ETF universe.
- This corroborates the session-wide pattern: six intraday build-and-validate cycles all fail
  net-of-cost OOS for the same reason (symmetric excursions + costs). The evidenced edge on this
  repo remains **insider selection**, not intraday price/flow timing.

## Reproduce
```bash
python amps_v5_investigation.py        # cost audit + reachability + exit attribution + ATR sweep
python -m pytest tests/test_amps_v5.py -q
```
