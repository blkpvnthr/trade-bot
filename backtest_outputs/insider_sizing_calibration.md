# Insider Conviction Sizing Calibration (§B) — does conviction sizing beat flat OOS?

hold=5d, min ADV=$1,000,000, vol-gate P80 (=0.0561). Train <2024-01-01, test ≥. Weighted-avg daily series is scale-invariant → same-gross.

## Part 1 — feature predictiveness (Spearman IC, Q5−Q1 spread)

| Feature | IC train | IC OOS | Q5−Q1 train | Q5−Q1 OOS | verdict |
|---|---|---|---|---|---|
| vol | +0.035 | +0.175 | +1.02% | +4.08% | predictive |
| breadth | +0.058 | +0.104 | +1.04% | -0.08% | no/weak |
| value (neg control) | -0.007 | -0.021 | +0.06% | +0.11% | no/weak |

## Part 2 — sizing schemes vs flat (GATED production set, OOS)

| Scheme | days | Sharpe | CAGR | MaxDD | ΔSharpe-p |
|---|---|---|---|---|---|
| flat (baseline) | 95 | +1.338 | +62.9% | -35% |
| vol-tilt | 95 | +1.178 | +52.4% | -35% | p=0.736 |
| breadth-tilt | 95 | +1.150 | +50.1% | -35% | p=0.754 |
| value-tilt (neg ctrl) | 95 | +1.332 | +62.5% | -35% | p=0.554 |
| composite (vol+breadth) | 95 | +1.163 | +51.2% | -35% | p=0.814 |
| buckets A+8/A6/B4/C2 | 95 | +0.888 | +34.1% | -35% | p=0.809 |

## Part 3 — gate vs tilt (does continuous vol sizing beat the binary gate?)

| Config | days | Sharpe | CAGR | MaxDD | ΔSharpe-p |
|---|---|---|---|---|---|
| full set, flat | 252 | +4.069 | +86.6% | -18% |
| full set, vol-tilt | 252 | +3.387 | +82.4% | -22% | p=0.876 |
| gated set, flat | 95 | +1.338 | +62.9% | -35% |

## VERDICT

- Best OOS scheme on gated set: **flat (baseline)** (Sharpe +1.338, vs flat +1.338, p=1.000)
- Gate vs tilt: gated-flat Sharpe +1.338 vs full-vol-tilt +3.387

**KEEP FLAT SIZING — no conviction scheme beats flat at p<0.10 OOS. The vol-GATE already harvests the vol edge as a filter; further sizing tilt adds noise, not signal. Do not poison a validated edge with unproven sizing.**