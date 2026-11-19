# Insider OOS Falsification — horizon 21d

- Trades: **213**   (clean/no-lookahead: **167**, lookahead-tainted dropped: 46, filing-date unknown: 0)

## Core edge (clean, no-lookahead subset)
- Mean SPY-excess: **+4.82%**   t-stat **+3.07**   bootstrap95 [+1.86%, +7.98%]
  - CI EXCLUDES 0 (robustly positive)

## Falsification battery (all trades)
- Headline mean excess +3.10% (t +2.31), bootstrap95 [+0.48%, +5.78%]
- **Timing placebo**: real +3.10% vs random-entry null mean +3.95% (95th pct +22.76%) -> p = **0.323** (FAIL: no timing edge beyond holding these names)
- **Spearman IC** (cluster $ vs excess): -0.097  perm-p 0.153  (n=213)

## VERDICT: QUARANTINE (positive OOS but one falsification test marginal; needs more data)

_Core edge is measured on the clean subset: trades whose entry date is on/after the real Form-4 filing date, so the strategy could not have traded on information it did not yet have._