# OLPS allocator: parity vs an external reference

We cross-checked `rl_meta/olps.py` against the external reference implementation
[nglahani/Online-Quantitative-Trading-Strategies](https://github.com/nglahani/Online-Quantitative-Trading-Strategies)
(the OLPS-survey framework). The reference is used for **validation only** and is
cloned read-only under `external_research/` (gitignored, never imported into any
live/allocator-execution path). Our architecture stays authoritative:

> accepted-alpha registry → mega-alpha streams → OLPS allocator

Parity tests live in `tests/test_olps_parity.py` (they skip automatically if the
vendor clone is absent). Run on the vendor's own `price_relative_vectors.csv`
fixture (94 US equities, 1998-2009):

| algorithm | result | parity |
|---|---|---|
| Constant Rebalanced Portfolio | ours 2.0836 vs theirs 2.0836 | **exact** |
| Buy & Hold | identical drift | **exact** |
| Exponential Gradient | ours 2.0894 vs theirs 2.0836 | **within 0.3%** (exp vs linearized update) |
| PAMR (matched ε, C) | per-period return corr **> 0.8** | same passive-aggressive update |
| OLMAR / RMR / Anticor / CORN | valid simplex, sane mean-reversion wealth on real data | behavioral |

## Convention differences found (and respected)
- **best_stock**: the reference's `best_stock` chases the *previous period's* winner
  (a causal momentum strategy). Our `best_stock*` is an explicit **hindsight upper
  bound** (best single asset over the whole path). Different objects -- not compared.
- **Exponential Gradient**: the reference uses the linearized `1 + lr·(x/⟨b,x⟩ − 1)`
  update; ours uses the canonical multiplicative `exp` form. Numerically near-identical.
- **OLMAR / RMR**: the reference takes price *relatives* and an internal MA; ours takes
  prices `P` directly. Behaviorally equivalent (mean-reversion), so we validate
  behavior, not bit-for-bit weights.

## Conclusion
Our EG, ONS, PAMR, OLMAR, RMR, Anticor, CORN, and Aggregating implementations match
or behaviorally agree with an independent reference. This is a validation pass only:
- the accepted-alpha registry and FDR gates are unchanged and remain the promotion authority;
- retired strategies (`martingale`, `pin_bar_magic_v1`, `sniper_strategy`) are never
  reintroduced (`strategy_streams.validate_no_retired` raises; the allocator fails on them);
- no broker / order / execution / live path is touched.
