# EO-backed symbolic alpha miner (research only)

A falsifiable, AlphaGen-style experiment: can symbolic search with the
[EO (Evolving Objects)](https://eodev.sourceforge.net/) C++ evolutionary-computation
library discover **diverse, stable, out-of-sample alphas** from OHLCV features -- and
does combined-mega-alpha **synergy selection beat plain IC-ranking**? It is a third,
independent search backbone next to the AlphaEvolve and gplearn miners; the headline
is the comparison, not a tradeable strategy.

**Research only.** No broker / order / execution / live-trading code is imported
(enforced by `tests/test_eo_alpha_miner.py::test_no_live_trading_imports`).

## Design

- **EO does the evolutionary search** (population, tournament selection, subtree
  crossover, branch mutation, elitist (mu+lambda) replacement, generational loop) over
  genetic-programming expression trees -- `eo_miner.cpp`, built on EO's `eoParseTree`
  (cribbed from EO's own `t-eoSymreg.cpp`). Fitness = mean cross-sectional **rank-IC**
  vs next-period returns, minus a parsimony penalty.
- **Tree nodes are arithmetic** (`+ - * safe_div min max abs sign neg`). The
  time-series operators the spec lists (`rolling_mean/std`, `zscore`, `rank`, `delay`,
  `delta`) are precomputed **leak-free feature columns** in `features.py`, so the C++
  core stays small and panel-agnostic and the search composes them.
- **Python validates** every discovered formula through the existing stack
  (`rl_meta.alphas` AST eval, `spearman_ic`, `rl_meta.alpha_synergy`): train / val / OOS
  IC, hit rate, complexity, correlation to already-accepted alphas, train/val sign
  stability -- then the A/B/C synergy comparison.
- **Fallback:** if the C++ binary is not built (e.g. CI without EO), `python_ga.py` runs
  the same EA mechanics in pure Python over `rl_meta.alphas` trees, so the CLI, the
  experiment, and all non-EO tests run everywhere. EO-only tests skip via
  `backend.available()`.

## Build the C++ EO core

Requires EO 1.3.1 installed locally (default prefix `~/Downloads/EO-1.3.1/local`;
override with `EO_HOME`). Verified on macOS Apple Silicon, clang, `-std=c++14`.

```bash
bash experiments/eo_alpha_miner/build.sh
# -> experiments/eo_alpha_miner/eo_miner
```

The exact link flags (from `build.sh`):

```bash
c++ -std=c++14 -O2 -I"$EO_HOME/include/eo" eo_miner.cpp \
    "$EO_HOME/lib/libeo.a" "$EO_HOME/lib/libeoutils.a" -o eo_miner
```

If EO is absent, `build.sh` exits cleanly and the Python backend is used.

## Run the CLI

```bash
# on the bundled smoke fixture (auto-picks EO backend if built, else python)
python experiments/eo_alpha_miner/run_miner.py

# explicit backend + config
python experiments/eo_alpha_miner/run_miner.py \
    --csv path/to/ohlcv.csv --backend eo \
    --population 300 --generations 40 --max-depth 5 --seed 42 \
    --crossover-rate 0.7 --mutation-rate 0.3 \
    --train-frac 0.6 --val-frac 0.2
```

Input CSV columns: `symbol,timestamp,open,high,low,close,volume` (optional `vwap`).

## Outputs (`experiments/eo_alpha_miner/out/`)

- `alphas.json` / `alphas.csv` -- per-alpha formula, signed polarity, train/val/OOS IC,
  hit rate, complexity, stability, correlation to accepted alphas.
- `report.md` -- top formulas + the A (IC-rank) / B (IC+corr penalty) / C (synergy)
  OOS-IC comparison and the verdict.
- `_train_features.csv`, `_eo_formulas.txt` -- the C++ exchange files.

## Reproducible example run (smoke fixture, EO backend)

Top alpha `mul(rank_ret, vol_sma_10)` (cross-sectional return-reversal times volume),
OOS IC ~0.12; 25 stable alphas. **Synergy verdict: IC-ranking ties/beats synergy OOS --
no diversification gain**, i.e. EO independently rediscovers a single factor family,
matching the gplearn and AlphaGen/synergy baselines. (The fixture has a planted faint
cross-sectional mean-reversion; numbers are illustrative, not a tradeable edge.)

## Tests

```bash
python -m pytest tests/test_eo_alpha_miner.py -q
```

Covers expression evaluation, NaN-safe `safe_div`, fitness sign, deterministic search,
leak-free features, no-live-imports, and (when built) EO determinism.
