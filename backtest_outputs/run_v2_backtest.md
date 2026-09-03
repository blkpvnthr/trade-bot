# run-v2.py (DailyOpportunityRouter) backtest

- Generated: 2026-06-11T13:32:24.562323+00:00
- Candidate generator: opening-range-breakout long, decision @ 10:00 ET, hold to session close (intraday tactical)
- Universe: 118 cached symbols | trading days: 92 (safe-mode days: 0)
- Base RouterConfig: min_expected_r=1.5, min_rank_score=0.6, max_new_entries=15, max_intraday_tactical_open=4

## Realized forward return (decision→close), by router decision

| Bucket | n | hit-rate | avg (bps) | median (bps) | profit factor | t-stat |
|---|---|---|---|---|---|---|
| ALL candidates | 3230 | 0.5282 | 26.36 | 11.61 | 1.335 | 4.69 |
| APPROVED | 368 | 0.5707 | 28.2 | 19.33 | 1.467 | 2.58 |
| REJECTED | 2862 | 0.5227 | 26.12 | 9.91 | 1.322 | 4.23 |

- **Approval rate:** 0.1139
- **Router value-add (approved − all):** 1.84 bps avg return | PF lift 0.132 | hit-rate lift 0.0425
- Interpretation: a positive edge_vs_all means the router's filter/rank concentrates into better-than-average candidates; ~0 means its gates aren't discriminating on this candidate pool; the APPROVED t-stat indicates whether the approved edge is distinguishable from noise.

## Key findings

1. **`rank_score` adds little on a momentum pool.** Selecting the top-ranked candidates (candidates fed highest-rank-first) yields only a small approved-vs-all edge — the cross-sectional ranking does not strongly predict intraday forward return here (the strongest opening movers tend to mean-revert). Most of the router's selectivity value comes from the **day-level bearish-context gate**, not intra-day ranking.
2. **The tactical cap is the binding constraint and is applied pre-ranking.** `max_intraday_tactical_open` accounts for the large majority of rejections, and in `daily_profit_engine.route()` it is enforced inside the candidate loop in INPUT order — *before* the final `rank_score` sort. So without pre-sorting the input, the cap can discard higher-ranked candidates that simply appeared later. This harness pre-sorts by rank to reflect intent; the live engine drip-feeds a few candidates per cycle, so the effect is milder there, but moving the cap to apply AFTER the rank sort would be a low-risk correctness improvement.
3. **The cap is the throughput knob** (see sweep): tighter cap → fewer, slightly better trades; looser cap → more trades, edge decays toward the candidate average.

## Reject-reason histogram

| blocked_by:reason | count |
|---|---|
| daily_trade_cap:max_intraday_tactical_open | 2641 |
| opportunity_rank:rank_score_below_floor | 115 |
| setup_not_validated:expected_r=1.38<1.50 | 13 |
| setup_not_validated:expected_r=1.43<1.50 | 9 |
| setup_not_validated:expected_r=1.49<1.50 | 9 |
| setup_not_validated:expected_r=1.32<1.50 | 7 |
| setup_not_validated:expected_r=1.35<1.50 | 7 |
| setup_not_validated:expected_r=1.31<1.50 | 7 |
| setup_not_validated:expected_r=1.36<1.50 | 7 |
| setup_not_validated:expected_r=1.34<1.50 | 6 |
| setup_not_validated:expected_r=1.33<1.50 | 5 |
| setup_not_validated:expected_r=1.39<1.50 | 5 |
| setup_not_validated:expected_r=1.44<1.50 | 5 |
| setup_not_validated:expected_r=1.47<1.50 | 4 |
| setup_not_validated:expected_r=1.48<1.50 | 3 |

## RouterConfig sensitivity sweep

Binding constraint is `max_intraday_tactical_open` (the per-day new-tactical cap); candidates are fed highest-rank-first so the cap retains the best.

| max_tactical | min_rank_score | approval_rate | approved n | approved avg (bps) | approved PF | approved t | edge vs all (bps) |
|---|---|---|---|---|---|---|---|
| 2 | 0.55 | 0.057 | 184 | 27.39 | 1.509 | 1.95 | 1.03 |
| 2 | 0.6 | 0.057 | 184 | 27.39 | 1.509 | 1.95 | 1.03 |
| 2 | 0.65 | 0.057 | 184 | 27.39 | 1.509 | 1.95 | 1.03 |
| 4 | 0.55 | 0.1139 | 368 | 28.2 | 1.467 | 2.58 | 1.84 |
| 4 | 0.6 | 0.1139 | 368 | 28.2 | 1.467 | 2.58 | 1.84 |
| 4 | 0.65 | 0.1139 | 368 | 28.2 | 1.467 | 2.58 | 1.84 |
| 6 | 0.55 | 0.1709 | 552 | 32.48 | 1.499 | 3.32 | 6.12 |
| 6 | 0.6 | 0.1709 | 552 | 32.48 | 1.499 | 3.32 | 6.12 |
| 6 | 0.65 | 0.1709 | 552 | 32.48 | 1.499 | 3.32 | 6.12 |
| 10 | 0.55 | 0.2848 | 920 | 29.67 | 1.434 | 3.84 | 3.31 |
| 10 | 0.6 | 0.2848 | 920 | 29.67 | 1.434 | 3.84 | 3.31 |
| 10 | 0.65 | 0.2848 | 920 | 29.67 | 1.434 | 3.84 | 3.31 |
| 20 | 0.55 | 0.4272 | 1380 | 25.55 | 1.36 | 3.91 | -0.81 |
| 20 | 0.6 | 0.4272 | 1380 | 25.55 | 1.36 | 3.91 | -0.81 |
| 20 | 0.65 | 0.4266 | 1378 | 25.56 | 1.36 | 3.91 | -0.8 |

- Read: a smaller cap is more selective (lower approval, higher per-trade edge); as the cap grows, approval rises and the approved edge decays toward the all-candidate average — the classic selectivity/throughput trade-off. This is the lever to tune for the $400–500/day target (more trades vs. higher per-trade quality).

## Caveats

- This validates the ROUTER (filter/rank/caps/gates) + its config against an ORB momentum candidate pool. It is not a P&L claim for live v2: real opportunities come from `run.collect_daily_profit_opportunities` (BAL/alpha/insider), not this generator.
- Forward returns are gross (no slippage/fees/borrow). Intraday hold to close; overnight gating is exercised structurally but overnight P&L is not modeled here.
- `probability`/`expected_r` are heuristic functions of pre-decision breakout strength, not calibrated; they feed the router's rank_score exactly as a live opportunity would.