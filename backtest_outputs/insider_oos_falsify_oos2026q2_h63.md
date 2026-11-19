# Insider OOS Falsification — horizon 63d

- Trades: **196**   (clean/no-lookahead: **156**, lookahead-tainted dropped: 40, filing-date unknown: 0)

## Core edge (clean, no-lookahead subset)
- Mean SPY-excess: **+2.02%**   t-stat **+0.91**   bootstrap95 [-2.25%, +6.37%]
  - CI includes 0 (not robust)

## Falsification battery (all trades)
- Headline mean excess +0.18% (t +0.09), bootstrap95 [-3.78%, +4.23%]
- **Timing placebo**: real +0.18% vs random-entry null mean +10.87% (95th pct +42.40%) -> p = **0.719** (FAIL: no timing edge beyond holding these names)
- **Spearman IC** (cluster $ vs excess): +0.046  perm-p 0.515  (n=196)

## VERDICT: REJECT (OOS edge does not survive falsification)

_Core edge is measured on the clean subset: trades whose entry date is on/after the real Form-4 filing date, so the strategy could not have traded on information it did not yet have._