# Validated levers: regime gate + dynamic exit (the things that actually move the needle)

Follow-up to [docs/rl_strategy_overlay.md](rl_strategy_overlay.md), which found that on the
2-week, uniformly-bull sample no filter could be tested and ex-ante feature selection did not
generalise out-of-sample. Here the two structural levers are tested on the data that *can*
show them — multi-regime daily history and real 1-minute forward bars. Both work and both
generalise. OFFLINE; read-only IEX (no SIP); no orders, no live wiring, no gate/risk changes.

## Lever 1 — market-regime gate (`rl_regime_study.py`)

A long-only daily-momentum book over **2020-07 → 2026** (31 names, 1,267 days, spanning the
2022 and 2025/26 bears). Both arms use the identical signal (hold names above their 50-DMA);
the **only** difference is arm B goes to **cash when SPY is below its 200-DMA**.

| arm | total return | CAGR | **max DD** | Calmar | Sharpe |
|---|---|---|---|---|---|
| take-all (always long) | +69.0% | 11.0% | **37.95%** | 0.29 | 0.69 |
| **regime-gated** | +63.3% | 10.2% | **12.40%** | **0.83** | **0.86** |

Per-year max drawdown — the gate works exactly where it should and is inert in bull years:

| year | 2021 | **2022** | 2023 | 2024 | **2025** | 2026 |
|---|---|---|---|---|---|---|
| all-in | 4.3 | **37.9** | 12.4 | 10.5 | 15.1 | 4.5 |
| gated | 4.3 | **9.7** | 11.5 | 10.5 | 12.4 | 4.5 |

**Result:** the regime gate cut max drawdown by ~⅔ (37.9%→12.4%), cut the 2022-bear drawdown
by ~74%, and ~tripled Calmar (0.29→0.83) — at a cost of ~6% of total return. This is the
literature's #1 lever, confirmed on real bear-market data. (A concurrent "Generating Alpha"
workstream is wiring an equivalent suppressive regime gate into `rl_strategy_overlay.py`.)

## Lever 2 — dynamic/asymmetric exit (`rl_dynamic_exit.py`)

The fixed +3R/−1R bracket pins `PF = 3·w/(1−w)`. A scale-out + trailing exit (⅓ off at +1R →
stop to breakeven; ⅓ off at +2R → stop to +1R; final ⅓ trails 1R below the peak) is re-derived
on the **same entries** and the **same real 1-minute bars** (9,545 entries), stop checked
before target (path-conservative).

| exit (out-of-sample) | win rate | PF | expectancy | max DD (R) |
|---|---|---|---|---|
| fixed 3:1 | 0.337 | 1.50 | 0.330 | 37.5 |
| **dynamic** | **0.476** | **2.31** | **0.687** | **26.3** |

**Result:** out-of-sample the dynamic exit lifts PF 1.50→2.31, more than doubles expectancy
(0.33→0.69R), raises win-rate to 0.48, and cuts R-drawdown ~30%. In-sample (PF 2.43) ≈
out-of-sample (PF 2.31) → it's a structural rule, not curve-fit. Mechanism: booking ⅓ at +1R
converts many would-be −1R reversals into small net winners, so it legitimately escapes the
fixed-payoff PF ceiling.

## How this maps to the original targets

| metric | requested | honest, validated outcome |
|---|---|---|
| win rate | 0.40 → 0.65 | dynamic exit → **0.48** OOS (real); 0.65 not reachable on these setups |
| profit factor | 9.1 → 25 | the 9.1 was 25-trade noise; dynamic exit → **2.31** OOS (real, generalises); 25 needs win 0.89 — impossible |
| return | +8.27% → +20% | expectancy **doubled** (0.33→0.69R/trade) — the real return lever, vs leverage/regime luck |
| max drawdown | 0.5% → 0.15% | regime gate −⅔ portfolio DD; dynamic exit −30% R-DD; plus lower risk-per-trade scales DD ~linearly |

## What did NOT work / lever 3

- **Ex-ante feature selection** (rel_volume/MTF/catalyst gate): in-sample PF +16% but
  **does not generalise OOS** — near the noise floor. Not a reliable lever.
- **Lever 3 — order-flow / level-2 / news features:** the genuinely-new-signal lever, but it
  is **blocked on data access** (no level-2 / tick / news feed under the current read-only IEX
  subscription). Documented as the next data-acquisition step, not attempted on data we lack.

## Recommended wiring (still gated; no promotion here)

1. **Regime gate** on both engines: suppress new alpha/Buffett entries when SPY < 200-DMA
   (the concurrent GA work already adds the suppressive hook).
2. **Dynamic exit** for the alpha engine: replace/augment the fixed 3R target with scale-out +
   trailing (⅓ at 1R/2R, trail the runner). This is the single biggest validated improvement.
Both remain advisory until they clear the deployment gate on live paper data + explicit approval.

## Wired into the engines (behind the deployment gate, default OFF)

`rl_engine_levers.py` gates both levers; a lever acts only when (a) its `ENABLE_*` env flag is
truthy [default OFF], (b) the process is in paper mode, and (c) the RL deployment gate rates
the relevant agent ≥ `paper_mode_only` (read from `rl_deployment_recommendation.json`;
unknown→blocked). Fail-safe to inactive on any error.

| lever | flag | gated agent | engine hook |
|---|---|---|---|
| Regime gate (alpha) | `ENABLE_RL_REGIME_GATE` | alpha_entry | `alpha_call_engine.evaluate_universe` — `entries_enabled=False` when SPY not confirmed >200-DMA |
| Regime gate (Buffett) | `ENABLE_RL_REGIME_GATE` | buffett_dca | `buffett_accumulation_layer.evaluate_dca` — `_reject("regime_gate…")` before the quality gate |
| Dynamic exit | `ENABLE_RL_DYNAMIC_EXIT` | alpha_exit | `alpha_call_engine._check_active_positions` — re-times partial(+1R)/trail(1R below peak) only |

**Safety invariants (tested):** suppressive-only (can reject a new entry, never forces one;
never touches sells/exits); the dynamic profile re-times *only* the partial/trail triggers —
the hard stop, 3R cap, kill switch, daily-loss halt, and RiskManager are evaluated first and
are byte-for-byte unchanged. With flags OFF (default) the engines behave identically to before.
Current gate state: dynamic-exit BLOCKED (alpha_exit=do_not_deploy); regime-gate would be
permitted for alpha_entry/buffett_dca (paper_mode_only) but stays OFF until its flag is set.
Tests: `tests/test_rl_engine_levers.py` (18).

## Reproduce
```
python rl_regime_study.py     # multi-regime daily book: take-all vs SPY-200DMA gate
python rl_dynamic_exit.py     # fixed 3:1 vs scale-out/trail on real 1m bars
```
Tests: `tests/test_rl_regime_and_exit.py` (6). Full suite 359 passing.
