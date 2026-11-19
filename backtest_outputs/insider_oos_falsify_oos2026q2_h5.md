# Insider OOS Falsification — horizon 5d

- Trades: **253**   (clean/no-lookahead: **197**, lookahead-tainted dropped: 56, filing-date unknown: 0)

## Core edge (clean, no-lookahead subset)
- Mean SPY-excess: **+2.66%**   t-stat **+3.16**   bootstrap95 [+1.02%, +4.37%]
  - CI EXCLUDES 0 (robustly positive)

## Falsification battery (all trades)
- Headline mean excess +2.30% (t +2.89), bootstrap95 [+0.74%, +3.90%]
- **Timing placebo**: real +2.30% vs random-entry null mean +1.59% (95th pct +8.27%) -> p = **0.151** (FAIL: no timing edge beyond holding these names)
- **Spearman IC** (cluster $ vs excess): -0.048  perm-p 0.452  (n=253)

## VERDICT: QUARANTINE (positive OOS but one falsification test marginal; needs more data)

_Core edge is measured on the clean subset: trades whose entry date is on/after the real Form-4 filing date, so the strategy could not have traded on information it did not yet have._