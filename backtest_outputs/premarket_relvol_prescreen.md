# Premarket high-relvol → SHARES pre-screen (upper-bound directional test)

46,842 high-relvol day-events. Entry at OPEN, no costs (most generous). open→close = ride-to-close; open→high = best-case 'exit at today's high'; open→low = stop-run proxy.

| Capture | mean | median | win% | PF |
|---|---|---|---|---|
| open→close (realistic) | +0.63% | +0.00% | 50% | 1.28 |
| open→high (best case) | +5.15% | +2.18% | — | — |
| open→low (worst case) | -3.92% | -2.20% | — | — |

Mean gap at open: +1.26% (are these gap-and-fade or continuation?)

## open→close by year (does the momentum continue, regime-stable?)

| Year | n | mean | win% | PF |
|---|---|---|---|---|
| 2021 | 8,184 | +0.52% | 48% | 1.24 |
| 2022 ⚠ | 8,376 | +0.24% | 48% | 1.10 |
| 2023 | 7,152 | +0.47% | 50% | 1.22 |
| 2024 | 7,350 | +0.79% | 51% | 1.37 |
| 2025 | 8,936 | +0.81% | 51% | 1.36 |
| 2026 | 4,244 | +1.05% | 50% | 1.47 |

MC (OOS open→close): P(profit)=67%, median=+46.90%

## VERDICT (shares upper bound)
- open→close PF 1.28, mean +0.63%, per-year UNSTABLE
- **STOP — no directional edge in shares; the 10DTE ATM-call version cannot work (option friction ≫ this).**

Reminder: even if shares show a small edge, an ATM 10DTE call needs the underlying to move ~7%+ (spread) plus cover ~10d of theta just to BREAK EVEN — so the bar for the option version is far higher than for shares.