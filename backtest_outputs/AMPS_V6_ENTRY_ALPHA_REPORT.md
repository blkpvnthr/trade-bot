# AMPS V6 — Entry-Alpha Report (the decisive gate)

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20 · falsification-first per CLAUDE.md §34_

## Question

V5 proved AMPS **intraday** entries have no directional edge (MFE ≈ |MAE| out to 180 min). V6's new
question: does the **multi-day swing** horizon produce genuine directional **asymmetry** (MFE
materially > |MAE|, net-positive after cost, beating matched controls) — or just proportionally
larger symmetric excursions? (Spec §34: "If MFE and |MAE| increase proportionately as holding
duration increases, do not interpret increased MFE alone as evidence of alpha.")

## Method (PIT, conservative, real data)

- **Entries:** AMPS directional-eligibility (ETF own DMI/BB gates + underlying-aligned regime).
  Intraday on real 5m bars; swing on **split/dividend-adjusted** daily bars (see data-integrity note).
- **Measurement per horizon:** average MFE, average MAE, **MFE/|MAE| ratio**, hold-to-horizon **net
  drift** (= forward return entry-open→horizon-close minus round-trip cost), and target-before-stop.
- **Controls (spec §34):** matched **random-entry** (same symbols/period, same count, random dates)
  and **simple-momentum** (close > SMA50). An edge must beat random entry, not just be positive.
- **Costs:** `costs.py` bucket + 2 bps slippage, round-trip per symbol (16 bps leveraged ETFs).
- **Fill:** next bar/session open; forward excursion from there. No look-ahead.
- **Decision grading (honest, never rounded up):**
  - `ALPHA_CONFIRMED` (deployable): ratio ≥ 1.15 **and** net ≥ 0 **and** beats control by ≥ margin.
  - `ALPHA_WEAK` (CONTINUE_RESEARCH): real selection signal (ratio ≥ 1.15 and beats control) **but
    not net-positive**.
  - `NO_ALPHA_EVIDENCE` (REJECTED): neither.
  - `INSUFFICIENT_DATA`: below the minimum sample.

## Data-integrity note (load-bearing)

`bars_multi_tf` 5m data is **unadjusted for splits**; leveraged/inverse ETFs reverse-split
constantly (SOXS 10×, SQQQ 5×, NVD 24×) and bull ETFs forward-split (TQQQ 2×, SOXL 15×). This is
invisible intraday (within-session) but fatal for multi-day analysis — the first swing run produced
a +753 bps/day random-control artifact. All swing results use Alpaca `adjustment=ALL` daily bars
(0 residual extreme gaps verified). We never scale an underlying return by stated leverage (§24).

## Results

See `AMPS_V6_REPORT.md` §2 (intraday) and §3 (swing) for the full tables. Summary:

- **Intraday:** ratio 1.02–1.20, net drift goes positive only at long horizons **but the random
  control beats it** (full-day +52.5 vs +29 bps). → selection subtracts value. **REJECTED (all).**
- **Swing 1–3d:** ratio < 1.15, small positive edge-vs-control. → **REJECTED.**
- **Swing 5–10d:** ratio 1.15–1.23, **beats random + momentum controls by +40 to +72 bps**, but net
  drift −18 to −57 bps. → **ALPHA_WEAK / CONTINUE_RESEARCH.**

## Interpretation

The multi-day swing horizon is the **first** place (across 7 cycles) where directional selection
demonstrably beats a matched random control with favorable excursion asymmetry. That is real
evidence of a *selection* signal. It is **not** evidence of profit: absolute net expectancy is
negative, driven by an inverse-heavy eligible book in a bull-market OOS. Therefore no horizon is
VALIDATED and nothing is live-eligible.

## Next experiments (owed before any swing promotion)

1. Regime-conditioned direction (trade inverse only in confirmed-bearish regime; bull otherwise) on
   a **balanced** bull/bear adjusted universe (fix the inverse skew).
2. Purged walk-forward with embargo ≥ max holding period; bootstrap CIs on net expectancy
   (`metrics.bootstrap_sharpe_ci`, `insider_oos_falsify` battery are the reuse targets).
3. Horizon-specific target-before-stop / conditional-return model only if (1) yields net-positive
   asymmetry to select for (else it is overfitting theater).
4. Cost-stress at 2×/double and an untouched final holdout before any PAPER promotion.

## Verdict

**CONTINUE_RESEARCH (swing).** No fabricated edge; no live promotion; evidence registry entries for
swing 5–10d would be `RESEARCH_CANDIDATE` only.
