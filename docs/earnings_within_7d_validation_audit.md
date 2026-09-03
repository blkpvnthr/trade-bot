# earnings_within_7d - Validation Record Audit (report only, no build/deploy)

Objective: determine which `earnings_within_7d` validation record is canonical and whether the
signal still clears the strict deployable bar under honest assumptions. This is an audit. No
sleeve was built, no orders placed. Live is out of scope.

## TL;DR verdict

```
CANONICAL_RECORD = rl_strategy_meta_agent/experiments/orthogonal_features/out/extended_validation.json
                   (generator: run_extended_validation.py; mirrored frozen in paper_sleeve.py)
DEPLOYABLE_STATUS = ACCEPTED  (re-ran the generator; 12/12 key metrics reproduced bit-for-bit)
NEXT_ACTION       = Proceed with the PAPER-only sleeve plan, shadow-first, DAILY rebalance
                    preserved. Before any capital: freeze the bars + EARNINGS caches as part of
                    the record, and resolve the long-only-vs-LS beta translation and the
                    reportedDate PIT subtlety (below). Still paper-only.
```

ACCEPTED, but modest and with load-bearing assumptions - read the caveats; do not over-read 1.11.

## Reproduction test (the decisive step)

Re-ran `python run_extended_validation.py` (cache-backed, deterministic) and diffed the
regenerated `out/extended_validation.json` against the committed copy:

| field | on-disk | re-run | match |
|---|---|---|---|
| n_obs | 55818 | 55818 | OK |
| ic | 0.0305 | 0.0305 | OK |
| ic_bootstrap_p | 0.007 | 0.007 | OK |
| net_sharpe_5bps | 1.11 | 1.11 | OK |
| turnover_day | 0.242 | 0.242 | OK |
| breakeven_bps | 55.9 | 55.9 | OK |
| net_bootstrap_p | 0.0 | 0.0 | OK |
| placebo_p | 0.0 | 0.0 | OK |
| wf_positive_year_frac | 0.73 (8/11) | 0.73 | OK |
| holdout_net_sharpe | 0.32 | 0.32 | OK |
| decision | ACCEPT | ACCEPT | OK |

Exact reproduction = TRUE. The four-signal verdict also reproduced: idio_reversal REJECT,
earnings_within_7d ACCEPT, sv_ratio QUARANTINE, sv_ratio_x_vol REJECT.

Determinism caveat: this holds ONLY while the input caches are frozen -
`backtest_cache/daily_bars/bars_2016-01-01_2026-06-18.parquet` (Alpaca SIP adjusted, cached)
and the `av_catalyst` `EARNINGS` cache. A forced re-fetch could drift (new/revised earnings
rows). Freezing both caches is part of pinning the canonical record.

## Artifact inventory

| # | Path | Type | Date/commit | Universe | Period | Signal def | Weighting | Cost | Turnover/d | Net Sharpe | Breakeven | boot / placebo / holdout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | `experiments/orthogonal_features/out/extended_validation.json` | result (CANONICAL) | commit 2956163, 2026-06-21 | 24 mega/large-caps | 2016-01-01..2026-06-18, 55,818 daily bars | binary earnings in next 7 cal days (PIT) | dollar-neutral LS, equal-weight, **daily** rebalance, fut_ret_1 | 5 bps | 0.242 | **1.11** | **55.9** | net p 0.0 / placebo 0.0 / holdout 0.32, 8/11 yrs +, FDR sig |
| A' | `run_extended_validation.py` | generator | commit 2956163 | same | same | `_earn_flag` lines 59-72 | `LV.streams(fwd="fut_ret_1")` daily | 5 bps | - | - | - | reproduced exactly on re-run |
| A'' | `out/extended_validation.csv` | result summary | 2026-06-21 | same | same | same | same | 5 bps | 0.242 | 1.11 | 55.9 | ACCEPT row |
| B | `out/lead_validation.json` | result (SUPERSEDED) | commit 016f6ba, 2026-06-21 03:33 | 24 (auto) | ~2025-10..2026-06, ~8mo, **15-min intraday** (~105k rows) | same binary | dollar-neutral LS, fut_ret_1 per **15-min bar** | 5 bps | **2.982** | **0.73** | **7.87** | net p **0.5655 (n.s.)**, "unproven, needs longer history" |
| B' | `run_lead_validation.py` | generator | commit 016f6ba | auto from panel | `_alpha_panel_25_8_v2.parquet` | `streams()` lines 44-56 | LS intraday | 5 bps | - | - | - | event study: +0.66% d-5..d-1, -3.2% d+1..d+5 |
| C | `paper_sleeve.py` | frozen config | promoted 2026-06-17 | n/a | n/a | `FROZEN_SIGNAL="earnings_within_7d"` | `target_weights()` market-neutral | n/a | 0.24 | 1.11 | 56.0 | hardcoded mirror of A; cites A's script |
| D | `sleeve_registry.jsonl` | registry record | 2026-06-21 06:08 | n/a | n/a | alpha_id=earnings_within_7d | n/a | n/a | 0.24 | 1.11 | 56.0 | status PAPER_PROMOTED, requires shadow |
| E | `out/sleeve_status.json` | status | 2026-06-21 | n/a | n/a | n/a | n/a | n/a | 0.282 | 1.35 (paper bt) | n/a | shadow READY_FOR_CAPITAL_REVIEW (0 outages after fwd-calendar wiring) |
| F | `out/shadow_replay.jsonl` | daily shadow log | from 2026-02-06 | n/a | live-ish | n/a | n/a | n/a | varies | n/a | n/a | per-day pnl/ic/turnover/outages |
| G | `experiments/orthogonal_features/README.md` | narrative | - | - | - | - | - | - | - | 1.11 | 56 | "first deployable signal" section |
| H | `rl_strategy_meta_agent/tests/test_paper_sleeve.py` | test | - | n/a | n/a | asserts frozen metrics 1.11/56.0/0.03 | n/a | n/a | n/a | n/a | n/a | locks A into the sleeve |
| I | `docs/earnings_within_7d_sleeve_plan.md` | plan | this session | - | - | - | - | - | - | - | - | PLAN ONLY |

## Why A and B disagree

Both runs use the SAME signal, SAME 24-name universe, SAME 5 bps cost, SAME dollar-neutral LS
construction, and BOTH are PIT-clean (forward earnings dates only). The disagreement is NOT
universe, NOT LS-vs-long-only, NOT cost model, NOT a lookahead bug, and NOT a stale/abandoned
artifact (B is a real earlier run that A supersedes). It is two confounded changes:

1. **Implementation / bar frequency (dominant).** B executes on 15-minute intraday bars
   (~26 rebalances/day -> turnover 2.982/day -> ~14.9 bps/day cost @5bps). A executes on DAILY
   bars (1 rebalance/day -> turnover 0.242/day -> ~1.2 bps/day). That ~13.7 bps/day cost swing
   is most of the net-Sharpe lift 0.73 -> 1.11 and nearly all of the breakeven jump 7.9 -> 56 bps.
   The memory note "net P&L must be daily fut_ret_1 not fut_ret_h" points at the same thing; the
   precise driver is intraday->daily execution, not the return-formula label.
2. **Date range / power (secondary).** B is ~8 months; A is ~10 years (55,818 bars). The extra
   history is what collapses the bootstrap p from 0.57 (unconfirmable) to 0.0 and turns IC
   0.0154 -> 0.0305.

Net: A is the legitimate, higher-powered, lower-cost result; B is the underpowered intraday
precursor it replaces. Both on disk; A supersedes B.

## Honest caveats on the ACCEPT (do not over-read 1.11)

- **Daily execution is load-bearing.** The deployable edge depends on the LOW-turnover daily
  rebalance. If the live sleeve drifts toward intraday trading, the 56 bps breakeven collapses
  back toward ~8 bps and the edge with it. Canonical = daily rebalance; preserve it.
- **Holdout is modest.** Symbol-holdout net Sharpe is 0.32 (vs 1.11 full-sample). The edge is
  real and FDR/placebo-clean but small out-of-group; size and expectations accordingly.
- **reportedDate PIT subtlety.** The flag uses `reportedDate` (the date earnings were ACTUALLY
  reported, ex-post) as the "next earnings" date. For liquid names scheduled ~= reported, but
  it is the realized date, a mild lookahead vs the truly scheduled-ahead date. Live must use the
  forward SCHEDULED calendar (the sleeve plan already specifies this).
- **Long-only translation re-opens beta.** Validation is dollar-neutral LS; a small paper
  account that runs long-only must re-confirm the drift survives without the short leg and net
  of beta (already a plan acceptance item).
- **Determinism is cache-bound.** Reproducibility holds against the frozen bars + EARNINGS
  caches; freeze them with the record.

## Discrepancy attribution (checklist)

| Possible cause | Present? |
|---|---|
| different universe | No (same 24) |
| different date range | YES (8mo vs 10y -> power) |
| different implementation | YES (15-min intraday vs daily -> turnover) |
| LS vs long-only | No (both LS) |
| cost model | No (both 5 bps) |
| turnover calculation | YES (consequence of bar frequency) |
| lookahead / PIT issue | No bug; minor reportedDate caveat noted |
| stale artifact | B is superseded, not erroneous |

## Verdict

```
CANONICAL_RECORD = rl_strategy_meta_agent/experiments/orthogonal_features/out/extended_validation.json
DEPLOYABLE_STATUS = ACCEPTED
NEXT_ACTION       = Pin the canonical record by freezing the bars + EARNINGS caches alongside it,
                    then proceed with the PAPER-only sleeve plan (shadow-first, daily rebalance).
                    Resolve long-only-vs-LS beta and the reportedDate->scheduled-calendar swap
                    before any capital review. Do NOT build the sleeve yet.
```
