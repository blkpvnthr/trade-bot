# Insider Entry-Timing Test — next-open vs VWAP-confirmation (real IEX intraday)

Validated vol-gated insider set (rv20≥0.0563, ADV≥$1,000,000), hold 3d, net 1.0% slippage. 180 entries, 120 VWAP-confirmed (67%). VWAP on free IEX feed (partial tape — directional only).

| Entry method | n | Mean | Median | Win% | Sharpe | PF |
|---|---|---|---|---|---|---|
| next_open (LIVE/validated) | 180 | +2.62% | +2.18% | 61% | +0.220 | 1.85 |
| vwap_confirm/momentum (buy higher) | 120 | +1.70% | +1.13% | 58% | +0.141 | 1.51 |
| BVC buy_pressure>0.62 + VWAP | 168 | +2.10% | +1.54% | 58% | +0.170 | 1.65 |
| vwap_SUPPORT pullback (filled 180/180) | 180 | +2.51% | +2.21% | 61% | +0.214 | 1.80 |
| vwap_support_or_open | 180 | +2.51% | +2.21% | 61% | +0.214 | 1.80 |

## Verdict

- VWAP-SUPPORT (pullback, buy lower) beats next-open (mean+PF, n≥30): **NO**
  vwap_support mean +2.51%/PF 1.80 (n=180, filled 180/180) vs next-open +2.62%/PF 1.85

**→ No intraday entry variant beats next-open — support, momentum, AND BVC all lose or tie. KEEP next-open. The validated rule stands; stop optimizing execution on this signal.**