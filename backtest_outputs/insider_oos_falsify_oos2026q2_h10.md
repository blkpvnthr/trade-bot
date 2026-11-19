# Insider OOS Falsification — horizon 10d

- Trades: **236**   (clean/no-lookahead: **185**, lookahead-tainted dropped: 51, filing-date unknown: 0)

## Core edge (clean, no-lookahead subset)
- Mean SPY-excess: **+2.71%**   t-stat **+2.50**   bootstrap95 [+0.70%, +5.03%]
  - CI EXCLUDES 0 (robustly positive)

## Falsification battery (all trades)
- Headline mean excess +1.83% (t +1.90), bootstrap95 [-0.05%, +3.80%]
- **Timing placebo**: real +1.83% vs random-entry null mean +2.64% (95th pct +14.16%) -> p = **0.265** (FAIL: no timing edge beyond holding these names)
- **Spearman IC** (cluster $ vs excess): -0.071  perm-p 0.272  (n=236)

## VERDICT: QUARANTINE (positive OOS but one falsification test marginal; needs more data)

_Core edge is measured on the clean subset: trades whose entry date is on/after the real Form-4 filing date, so the strategy could not have traded on information it did not yet have._