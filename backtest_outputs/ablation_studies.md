# Ablation Studies — EV selection & Dual-Kelly, sized on LIVE Alpaca equity

**Account equity (Alpaca API): $6,260.00**  ·  98 momentum entries  ·  hold 10d, 35DTE, IV=RV20×1.2 (positive VRP).

## STUDY 1 — SELECTION ONLY (per-trade return, size-independent)

| Method | n | Mean ret | Sharpe | Win% | Median | Avg δ |
|---|---|---|---|---|---|---|
| Current (ATM δ0.50 heuristic) | 98 | +21.8% | +0.127 | 40% | -12.5% | 0.50 |
| **EV-max (ContractSelector)** | 98 | **+23.2%** | **+0.189** | 55% | +5.3% | 0.73 |

Selection delta: EV picks avg δ=0.73 vs the heuristic's 0.50 — does EV move toward deep-ITM, and does it raise Sharpe?

## STUDY 2 — SIZING ONLY (same EV-best contract; equity curve on real account)

- Affordability at $6,260: fixed-fraction funds **38/98** entries, dual-Kelly funds **16/98** (whole-contract flooring at a small account).

| Sizing | Terminal equity | Total return | Max DD |
|---|---|---|---|
| Current fixed-fraction (2%) | $8,028 | +28% | -4% |
| **Dual-Kelly** | $6,277 | +0% | -3% |

## STUDY 3 — FULL STACK (EV-selection + Dual-Kelly vs Current+Current)

| Stack | Terminal equity | Total return | Max DD |
|---|---|---|---|
| Current selection + fixed-fraction | $8,611 | +38% | -4% |
| **EV-selection + Dual-Kelly** | $6,277 | +0% | -3% |

## Attribution read
- Study 1 isolates SELECTION (return space). Study 2 isolates SIZING (same contract). Study 3 = both.
- full_stack − current ≈ (Study1 selection effect) + (Study2 sizing effect) + interaction.
- At $6,260 the affordability constraint is real: only 16/98 entries fund any
  contract, so the SIZING studies are dominated by whole-contract granularity at this account size.
_BS-priced synthetic chains, terminal-exit, modeled spread only (optimistic fills). No-lookahead._