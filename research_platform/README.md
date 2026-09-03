# research_platform

A reproducible champion/challenger research machine. It turns every trading decision into a
permanently reproducible research observation, and evaluates new ideas against **preregistered**
criteria so good ones are promoted and bad ones are rejected — scientifically, not heuristically.

**Design rule: this layer never generates signals.** It consumes candidate decisions and evaluates them.

## Components
| Module | Responsibility |
|---|---|
| `registry.py` | Immutable, content-addressed `ExperimentSpec` = hash(config + git commit + data snapshot + preregistered search space + promotion criteria). Same inputs → same hash (idempotent). Any change → new experiment. |
| `events.py` | Append-only JSONL decision ledger. Each `DecisionEvent` carries a point-in-time feature snapshot and a `realized` (champion/production) vs `modeled` (challenger counterfactual) fill tag. |
| `fills.py` | The ONE shared `FillModel` (slippage + commission). Every portfolio uses the same instance → differences come from strategy, never fill optimism. |
| `portfolios.py` | A portfolio = `filter_fn` + `weight_fn` over a shared candidate pool. Universe / gate / sizing changes are ALL expressed this way, so every change is compared identically. |
| `promotion.py` | Preregistered `PromotionCriteria`: min-trades, burn-in, bootstrap significance, per-year robustness, **and OOS-holdout improvement**. `evaluate()` → PROMOTE / CONTINUE / REJECT. |
| `report.py` | Nightly markdown ranking challengers by verdict. |
| `demo.py` | End-to-end self-test on real insider + mean-rev decisions (must reproduce hand-proven verdicts). |

## Run
```bash
python -m research_platform.demo
# → research_outputs/{registry, events, nightly_*.md}
```

## Add a challenger (the whole point)
```python
PortfolioConfig(
    portfolio_id="my_idea",
    role="challenger",
    filter_fn=lambda r: ...,      # which candidates this portfolio takes
    weight_fn=my_weights,         # how it sizes them (scale-invariant → same-gross)
    description="what hypothesis this tests")
```
Then `evaluate(champion, my_idea, pool, FILL, CRIT, spec_hash, log=log)`.

## Verdicts (key results, 2026-06-11)
- **Mean-rev:** broad-ETF challenger → REJECT (breadth = correlated beta = dilution).
- **Insider sizing:** conviction buckets → REJECT (flat wins; gate already harvested the vol edge).
- **Insider gate:** full-book → PROMOTE (p=0.003, OOS ΔSharpe +2.53) — breadth = genuine
  diversification for *independent* insider events. **Lead, not a production change:** route to a
  live paper shadow + add intermediate challengers (gate-as-tilt, gated-but-more-concurrent) before acting.

Lesson: *breadth is not automatically diversification* — it dilutes correlated bets (ETF dips) and
diversifies independent ones (insider buys). The platform quantified both with one machine.
