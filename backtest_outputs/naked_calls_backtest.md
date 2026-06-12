# Naked-Calls Path — Backtest on Screened Universe

Universe (13/14 priced): TDUP, KT, OLLI, EXLS, VOYA, ZGN, FRO, SITE, FAF, BNL, ONB, EPRT, CTRE
Modelled pick: delta≈0.8 (ATM), 35DTE, IV=realised×1.2 (cap 2.5), extrinsic round-trip spread 25%, rolling entry every 10d.

| Hold | n | Win% | **Opt net mean** | net median | t | Wipeouts | gross mean | Share mean | Lev |
|---|---|---|---|---|---|---|---|---|---|
| 10d | 1813 | 43% | **-4.1%** | -8.6% | -3.70 | 0% | -0.9% | +0.38% | -2.2× |

## Read
- **⚠ Survivorship/look-ahead**: universe = TODAY's screen; not a tradeable historical result. Measures naked-calls option mechanics on a fixed quality basket only.
- ATM ~30DTE calls are the naked-calls profile; cross-check backtest_insider_options.py which showed ATM is net-negative once realistic IV+spread are priced. NET here is still an optimistic ceiling.
_No exit logic in naked-calls.py — holds are imposed by this backtest._