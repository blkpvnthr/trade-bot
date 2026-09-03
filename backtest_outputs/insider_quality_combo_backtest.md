# Insider ∩ Quality/Trend — Deep-ITM Call Combo (prototype backtest)

PIT insider universe 1725 names. Deep-ITM call: delta≈0.8, 30DTE, 10d hold, IV=realised×1.3, extrinsic spread 40%.
Quality/trend gate (technical part of the screen, computable PIT): price>$3.99, close>SMA50, rel-vol>1.0.

| Set | n | Win% | **Opt net mean** | net median | t | Wipeouts | Share mean |
|---|---|---|---|---|---|---|---|
| insider ALL (no gate) | 3231 | 45% | **+9.3%** | -6.2% | +3.34 | 2% | +4.06% |
| insider ∩ quality/trend | 470 | 46% | **+1.1%** | -5.1% | +0.33 | 1% | +1.73% |

Gate keeps 470/3231 signals (15%).

## Read
- Does the quality/trend gate RAISE net mean / win% / share edge and CUT wipeouts vs ungated?
  If yes, the combo concentrates the insider edge into better names.
- FUNDAMENTAL screen fields (rev/EPS growth, analyst rating, D/E) are NOT in this backtest (no PIT fundamentals) — they're a live-only gate in run.py, expected to filter further.
- NET is an optimistic ceiling (BS, modelled spread). Universe still has the usual survivorship caveat on the unpriced delisted tail.

_Same no-lookahead entry/exit as the insider backtests._