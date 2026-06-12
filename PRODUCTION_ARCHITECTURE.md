# Production Trading Architecture — Converged Design

_Date: 2026-06-11. Supersedes the "12–15 trades/day" framing._

## Operating objective

Optimize for **trade quality, not trade count**:

- 5–10 A-quality trades/day
- 70%+ win rate
- Portfolio PF > 2
- Average winner > average loser
- Overnight alpha **only when justified** (evidence-based, not time-based)
- Capital deployed into the *best* opportunities, not the *most* opportunities

This is the disciplined path to the CLAUDE.md $400–500/day paper target — not a replacement
for it. Quality-per-trade is how we compound toward the dollar target without violating risk
controls or reviving falsified strategies.

## Evidence ground-truth (from our gauntlet)

| Finding | Evidence |
|---|---|
| Insider edge is validated **ungated** | PF 2.12 OOS, t +3.95 @21d, +3.4% vs SPY |
| Filtering insider **kills** the edge | quality/trend gate: t 3.34→0.33; require>200SMA: dips +2.61% beat trend +1.40% |
| Only vol-gate helps insider | PF 1.44 → 2.20 |
| Insider conviction = vol percentile, **not** deal size | vol Q5 +5.53% vs Q1 −0.40%; $100M+ ≈ $1M+ @5d, negative @21/63d |
| Next-open beats confirmation entries | +2.62% (Sharpe .220) vs vwap-confirm +1.70% (.141); "stop optimizing execution" |
| ETF mean-rev validated minimal | RSI(2)<10 & close>200SMA, exit RSI>70 or 6d, 5bps → PF 2.57 OOS, 75% win, MC 100%, corr −0.06 to insider |
| Momentum/price-patterns lose | sector-rotation PF 0.85, ORB falsified, reversal-B PF 1.15, "price-patterns lose to SPY" |
| Intraday MR leak-free but unproven net | IC 0.03–0.04 @5–15m, 3 OOS days, single regime |

## Tiers

### Tier 1 — Insider copier (LIVE)
- Vol-gated >$1M Form-4 buys, **ungated otherwise** (no trend/quality/confirmation filters — they kill it)
- Broaden mega-cap → broad PIT-clean universe (mega-cap alone has no edge)
- Single **next-open** entry (no scale-in until separately validated)
- Multi-day hold → **overnight by default**
- PF-1.8 rolling kill switch
- Dynamic sizing by **vol-percentile conviction** (+ breadth + freshness), not deal size
- Expected: 0–2 entries/day, highest EV

### Tier 2 — ETF mean-reversion (LIVE — stays NARROW; expansion FALSIFIED 2026-06-11)
- Core (validated): `RSI(2)<10 AND close>200SMA`, exit `RSI>70 or 6-day`, ~8–10 US sector/broad
  ETFs → PF 2.2–2.6 OOS, 75% win.
- **Universe expansion FALSIFIED** (backtest_meanrev_expansion.py): edge is *monotonically
  diluted* by breadth — 150 liquid stocks PF 1.37, 74 broad ETFs PF 1.48 (fragile in bear), 40
  US-equity ETFs PF 1.70. NONE reach PF 2.0, all break in 2022. Mean-reversion is a tight
  US-equity-index phenomenon; correlated dips + 5-concurrent cap mean extra names dilute, not
  diversify. **Keep Tier-2 at the original ~8–10 ETFs.**
- No candidate filter survived §A greedy/BH-FDR on any universe.
- Management: partial at +1R, trail remainder; probabilistic overnight (see below)
- Expected: **~0.5–1 trade/day** (small high-quality sleeve, NOT the trade-count engine)

### Tier 3 — Relative-strength momentum continuation (PAPER until proven)
- Not ORB. Requires: RS leader, sector leadership, expanding RVOL, price>anchored VWAP,
  institutional volume, positive breadth, no major overhead resistance
- Management: quick partials; strongest names become swings
- **Weakest evidence in the stack + adjacent negative evidence** → same promotion gate as Tier 4
- Expected (if promoted): 1–4 trades/day

### Tier 4 — Intraday mean-reversion (PAPER until proven)
- Ex-ante reversion features (atr_pct, dist_sma_5, ret_3m), 5–15m horizon
- Promote only after: 500+ trades, positive expectancy, PF>1.5, survives commissions/slippage,
  survives walk-forward, survives regime changes
- Expected (if promoted): 5–15 trades/day

## Probabilistic overnight hold (Tiers 2/3/4; Tier 1 carries by design)

Carry overnight **only if all hold**:
- market regime still favorable
- closing in upper quartile of day range
- above VWAP
- strong relative strength
- institutional accumulation
- no earnings / no binary catalyst
- VIX stable, positive futures
- stop distance acceptable

Else flatten into the close. Carry a fraction, book the rest.

## Position sizing — conviction buckets

| Score | Allocation |
|---|---|
| A+ | 8% |
| A | 6% |
| B | 4% |
| C | 2% |
| speculative | paper only |

Under existing controls: 10% single-name cap, 45% cash buffer (preserved — "capital always
deployed" means within the risk budget). Conviction score must be empirically calibrated per
sleeve before it drives sizing.

## Build sequence

1. Calibrate insider conviction score (vol-percentile + breadth + freshness) on existing backtest.
2. Mean-rev universe expansion backtest (OOS + walk-forward; test each filter's marginal value). **Gate.**
3. Wire Tier 2 with only passing filters + partial/trail + probabilistic-overnight.
4. Build shared probabilistic-overnight evaluator module.
5. Stand up Tiers 3 & 4 in paper with full shadow logging against the promotion gate.

## Addenda — process framework (2026-06-11)

### A. Filter promotion = multi-metric + hard rule
For every candidate filter report ΔPF, ΔSharpe, ΔCAGR, Δtrade-count, Δcapacity, regime
sensitivity, cross-year stability. Keep-rule is a predicate, not a judgment:
`keep iff ΔSharpe>0 AND ΔCAGR>0 AND retained-trade-fraction ≥ 0.6 AND survives BH-FDR`.
- CAGR/MaxDD measured in a **portfolio sim** (real cash buffer, name cap, granularity at $7k),
  NOT on the trade list — trade-list PF can rise while CAGR falls from idle capital.
- Apply Benjamini-Hochberg FDR across all filters tried (guard against fishing).
- Use **greedy forward selection** (add best marginal, re-score rest conditionally) — filters interact.

### B. Per-sleeve sizing models (do NOT share formulas)
- Insider: f(realized_vol_percentile, breadth, freshness)
- Mean-rev: f(distance_from_mean, realized_vol, liquidity)
- Momentum: f(predicted_edge, liquidity, spread)
- Scalp: fixed until proven
Each sizing model must beat flat sizing OOS before shipping (sizing on a noisy edge estimate
hurts Sharpe). Continuous function clamps into the 8/6/4/2% buckets so 10% cap + 45% buffer bind.
Fractional Kelly (¼–½) + hard caps; validate in portfolio sim at real account size (dual-Kelly
failed at $6–7k on whole-contract granularity).

### C. Continuous shadow portfolios (centerpiece)
Portfolio multiplexer: one signal pass → registry of configs {filters, sizing, overnight_policy}
= PRODUCTION + Shadow A (+candidate filter) + Shadow B (−production filter) + Shadow C (experimental
momentum) + Shadow D (experimental sizing). Each writes its own journal + equity curve; nightly
comparison report (PF, drawdown, turnover, slippage, overnight perf).
- One shared realistic fill/slippage model across all portfolios (else shadows look fake-good).
- Promotion shadow→production requires significance test + min sample + regime coverage + burn-in.
  Never promote on short-run divergence (that would make this the noise-chaser of #E).

### D. Regime-stratified promotion minimums
Not just "500 trades, PF>1.5". Require bull/bear/high-vol/low-vol/trend/chop samples, primarily
from the **historical backtest** (live-forward too slow for a bear sample at $7k; it's confirmation
only). Gate = non-catastrophic in EACH regime bucket, not just positive on average. Reuse existing
regime labels (SPY vs 200SMA, VIX terciles, realized-trend).

### E. Meta-allocation engine — guardrailed
Default = strategic weights from long-run validated Sharpe (risk-parity-ish). Tactical tilts driven
by **observable regime state** (regime predicts forward sleeve perf), NOT trailing realized PnL
(performance-chasing; strategy-timing ≈ market-timing). Heavy shrinkage toward strategic prior +
turnover penalty + slow adaptation. **Asymmetric: cut fast on deterioration, add slow on strength.**
PF<1.8 kill = the defensive version (keep). Allocator runs as a shadow before moving real capital.
The example monthly 2x swings are too large — that magnitude would fit noise.

## Validation log
- **2026-06-11 — Research platform BUILT (research_platform/).** Champion/challenger machine:
  immutable content-addressed registry (config+git+data-snapshot+preregistered criteria), append-only
  decision ledger w/ PIT snapshots + realized/modeled fill tags, ONE shared fill model, filter+weight
  portfolios, preregistered promotion gate (min-trades, burn-in, bootstrap significance, per-year
  robustness, OOS-holdout), nightly report. demo.py self-test reproduced hand-proven verdicts:
  REJECT broad-ETF mean-rev, REJECT insider conviction-bucket sizing, **PROMOTE insider full-book**
  (p=0.003, OOS ΔSharpe +2.53). KEY INSIGHT: *breadth is not automatically diversification* — it
  DILUTES correlated bets (ETF dips, all market-beta) but DIVERSIFIES independent ones (insider buys).
  Insider full-book is a LEAD not a production change (champion's −0.54 Sharpe/−81%DD is a
  concentration/metric artifact needing reconciliation w/ validated per-trade PF 2.12) → route to LIVE
  PAPER SHADOW + add intermediate challengers (gate-as-tilt, gated-but-more-concurrent) before acting.
  Outputs: research_outputs/{registry,events,nightly_insider.md,nightly_meanrev.md}.
- **2026-06-11 — Insider conviction sizing: KEEP FLAT.** backtest_insider_sizing.py. vol (rv20) IS
  per-trade predictive (IC OOS +0.175, Q5−Q1 +4.08%); value = confirmed non-predictor; breadth weak.
  BUT no sizing scheme (vol/breadth/composite/buckets) beats flat OOS on the gated set (flat Sharpe
  +1.34; all tilts worse, p>0.5) — the vol-GATE already captures the vol edge, so sizing-tilt within
  it adds variance not signal. **Insider sleeve stays flat-sized.** Lead for shadow framework:
  full diversified book Sharpe +4.07 vs gated +1.34 → binary gate may over-concentrate (thin n=29).
  Also confirmed: system is on SIP (ALPACA_DATA_FEED=sip, live probe returns SIP data).
  Output: backtest_outputs/insider_sizing_calibration.md.
- **2026-06-11 — Tier-2 expansion: FALSIFIED.** backtest_meanrev_expansion.py {liquid|etf|usetf}.
  Edge dilutes monotonically with breadth (stocks 1.37 < broad-ETF 1.48 < US-ETF 1.70 < original
  2.2–2.6); all break in 2022; no filter survived §A. Implication: trade count must come from NEW
  edges/sleeves, not from widening existing edges → raises priority of the shadow framework.
  Outputs: backtest_outputs/meanrev_expansion_{liquid,etf,usetf}.md. Fetched 64 ETF bars to cache.

## Implementation order (converged)
1. ✅ Expanded mean-rev universe OOS backtest (DONE — expansion falsified; Tier-2 stays narrow)
2. ✅ Per-sleeve sizing calibration (DONE — insider stays FLAT; no conviction scheme beats flat OOS)
2. Per-sleeve conviction/sizing calibration (§B)
3. Probabilistic overnight evaluator (shared module)
4. Continuous shadow portfolios (§C) — highest long-term value
5. Momentum + intraday scalp stay in paper until §D promotion gate passes

## Preserved (never touched)
Exits, sell logic, kill switches, max daily loss, cash buffer, regime exit, PDT floor,
protected symbols. No live trading switch without explicit instruction.
