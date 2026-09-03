# alpha_factory

Autonomous alpha discovery + validation. Research-only: **no broker, no orders, no live execution.**

```
Miner -> Scientist -> Librarian (registry) -> Portfolio export -> RL allocator
```

The point of v0 is the **Scientist gate**: a preregistered, falsifiable test that says "this alpha is
real enough to keep" or "this is trash" without a human reviewing each backtest. The generator is
deliberately thin until the gate is trusted (your own research already falsified "more generator =
more alpha" twice; the bottleneck is information, not optimizers).

## The 5 agents

| Agent | Module | Status |
|---|---|---|
| **Miner** | `miner.py` | thin v0 - Alpha101-style formulas + mutation/combination. ~dozens/run, not 5000/day. Scale = add formulas or wire `rl_strategy_meta_agent` AlphaEvolve/gplearn as candidate sources. |
| **Scientist** | `scientist.py` | full. Wraps `rl_trading.event_study` (forward panels, block-bootstrap, rank-IC, placebo) + adds time-fold OOS IC, net-of-cost LS sim, Benjamini-Hochberg FDR. |
| **Librarian** | `registry.py` + `schema.py` | full. Append-only JSONL, dedupe by **prediction** (not formula), crowding via correlation-to-accepted. |
| **Portfolio** | `cli.py export-allocator-input` | bridge only. Emits `allocator_input.json`; the existing RL allocator consumes it. This layer never sizes or trades. |
| **Evolution** | - | **deferred.** It decides "what to mutate next." Build only after the gate has accepted/rejected enough real alphas to learn from. |

## What the Scientist actually checks

A candidate is a cross-sectional signal panel `[dates x symbols]` (higher = more bullish). Verdict
is `accepted` / `rejected` / `needs_more_data` against preregistered `Criteria`:

- **OOS IC** - mean daily rank-IC across contiguous time folds (walk-forward stability).
- **Bootstrap significance** - 2000-resample CI of the daily IC must exclude 0. (Significance is on
  the IC, not mean excess return - a market-neutral alpha has ~0 mean excess by construction.)
- **Placebo** - random-shift the signal dates; IC must collapse to ~0 (catches lookahead leakage).
- **Net-of-cost** - long/short sim on **1-day** forward returns (per the short-vol method-fix: net
  P&L uses `fut_ret_1`, not the holding-horizon return), after `cost_bps` turnover cost. Net Sharpe
  must clear the bar.
- **FDR** - Benjamini-Hochberg across the whole batch, so testing many candidates does not
  manufacture a false positive.
- **Crowding** - correlation to already-accepted alphas <= `max_corr`.

Composite `score` (transparent, logged in `score_components`): profit_factor .25, expectancy .20,
walk-forward_stability .20, drawdown_safety .15, trade_count .10, independence .10.

## Usage

```bash
# synthetic demo (planted alpha + noise) - watch the gate accept exactly one
python -m alpha_factory.cli evaluate-all --demo
python -m alpha_factory.cli accepted
python -m alpha_factory.cli export-allocator-input   # -> storage/allocator_input.json

# real data, adjusted cache (data/cache/alpaca_bars_adj). 5bps is OPTIMISTIC for small-caps;
# use 15-20bps for promotion decisions (the run prints this warning).
python -m alpha_factory.cli evaluate-all --source alpaca_cache --cost-bps 20

# --panel is a pickled closes DataFrame [dates x symbols, tz-aware UTC]; optional sibling
# <name>_vol.pkl supplies volume.
python -m alpha_factory.cli evaluate-all --panel /path/closes.pkl --cost-bps 15

python -m alpha_factory.cli registry         # all evaluations
python -m alpha_factory.cli rejected         # with reasons
```

### Cost is the gate's softest spot

`--cost-bps` (default 5) sets the round-trip cost used for **registry acceptance**. Every run also
prints a **cost-sensitivity table** (`--cost-sweep-bps`, default `5,15,20,25`) for each accepted
alpha, and tags any alpha `FRAGILE_COST_MIRAGE` if it accepts at the selected cost but its net
Sharpe falls below 0.5 by 15 or 20bps. The default stays 5 for backwards compatibility; real-data
runs warn that 5bps is optimistic for illiquid universes.

**Promotion rule: no paper sleeve for anything with net Sharpe < 1 after 15-20bps.** Such an alpha
is watchlist / research-only, not tradeable.

Tests (no pytest needed): `python -m alpha_factory.tests.test_core`

## Honest expectation

On the synthetic panel the planted oracle is the only accept - by design. **On real liquid US
equities, expect this gate to reject almost everything**, including most Alpha101 formulas, once
costs and FDR apply. That rejection is the product. A passing alpha is rare and worth allocating;
a directory full of "accepted" alphas means the gate is too loose, not that you struck gold.

## Safety

No broker SDK, no order submission, no `trade.py` import - enforced by
`tests/test_core.py::test_no_broker_imports`. Output is validated intellectual property, not trades.
