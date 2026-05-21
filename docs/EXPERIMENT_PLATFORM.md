# QuantumQuant Experiment Platform

## Overview

Isolated Kubernetes research platform for running multiple paper-trading strategy variants in parallel. Each variant uses a separate Alpaca paper account, logs independently, and is evaluated nightly.

**Core invariants:**
- `run.py` is the sole execution authority — strategies only generate signals
- All deployments default to `PAPER_ONLY=true`
- Alpaca credentials are injected via Kubernetes Secrets (`envFrom`) — never hardcoded
- No auto-promotion to live trading — human approval required for every change

---

## Architecture

```
strategies/           ← signal generation only, no order placement
  base.py             ← BaseStrategy ABC + SIGNAL_COLUMNS + helper indicators
  tux_mtf.py          ← TUX Multi-Timeframe (original + long-only)
  pivot_points.py     ← Floor-trader pivot bounce/breakout
  orb_8am.py          ← 30-minute Opening Range Breakout from 8am ET
  liquidity_sweep.py  ← Smart money sweep reversal
  flawless_victory.py ← All-6-conditions high-conviction entry
  registry.py         ← STRATEGY_REGISTRY + get_strategy()

configs/
  strategy_experiments.yaml  ← variant params + evaluation thresholds

scripts/
  evaluate_variants.py        ← rank by 10 metrics
  diagnose_strategy.py        ← identify 10 failure patterns
  generate_claude_patch_request.py  ← write markdown to research_requests/
  promote_variant.py          ← open GitHub PR (never auto-merges)

k8s/
  namespace.yaml              ← trade-research namespace
  secrets.example.yaml        ← placeholder secrets (REPLACE_ME)
  configmap.yaml              ← shared non-secret config
  deployment-template.yaml    ← base deployment spec
  variants/                   ← one yaml per strategy variant
  cronjobs/nightly_evaluator.yaml

.github/workflows/
  claude-research-loop.yml    ← nightly evaluate + diagnose + patch requests

research_requests/            ← generated patch request markdowns (gitignored)
logs/execution/{strategy_id}/ ← per-variant execution logs
```

---

## Running a Variant Locally

```bash
# Single variant — paper only
STRATEGY_ID=tux_mtf_original \
CONFIG_PATH=configs/strategy_experiments.yaml \
PAPER_ONLY=true \
python run.py

# All variants via Docker Compose
cp .env.bot01.example .env.bot01   # fill in paper credentials
cp .env.bot02.example .env.bot02
cp .env.bot03.example .env.bot03
docker compose up
```

---

## Deploying to Kubernetes

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create secrets (never commit real keys)
kubectl create secret generic alpaca-paper-bot-tux-mtf \
  --from-literal=APCA_API_KEY_ID=YOUR_KEY \
  --from-literal=APCA_API_SECRET_KEY=YOUR_SECRET \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  -n trade-research

kubectl create secret generic alpaca-paper-bot-pivot-orb \
  --from-literal=APCA_API_KEY_ID=YOUR_KEY \
  --from-literal=APCA_API_SECRET_KEY=YOUR_SECRET \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  -n trade-research

kubectl create secret generic alpaca-paper-bot-experimental \
  --from-literal=APCA_API_KEY_ID=YOUR_KEY \
  --from-literal=APCA_API_SECRET_KEY=YOUR_SECRET \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  -n trade-research

# 3. Apply shared config
kubectl apply -f k8s/configmap.yaml

# 4. Deploy variants
kubectl apply -f k8s/variants/

# 5. Deploy nightly evaluator
kubectl apply -f k8s/cronjobs/nightly_evaluator.yaml

# Check status
kubectl get pods -n trade-research
kubectl logs -n trade-research deployment/trader-tux-mtf-original -f
```

---

## Credential Sets

| Bot | K8s Secret | Variants |
|-----|-----------|---------|
| bot-01 (conservative) | `alpaca-paper-bot-tux-mtf` | tux_mtf_original, tux_mtf_long_only |
| bot-02 (aggressive) | `alpaca-paper-bot-pivot-orb` | pivot_points, orb_8am |
| bot-03 (experimental) | `alpaca-paper-bot-experimental` | liquidity_sweep, flawless_victory |

---

## Evaluation Metrics

Variants are ranked nightly by `scripts/evaluate_variants.py` across 10 metrics:

| Metric | Description |
|--------|-------------|
| `total_return` | Net P&L over lookback window |
| `sharpe_ratio` | Annualised risk-adjusted return |
| `max_drawdown` | Worst peak-to-trough loss |
| `profit_factor` | Gross profit / gross loss |
| `win_rate` | % of trades closed positive |
| `mae_avg` | Mean adverse excursion (heat taken) |
| `mfe_avg` | Mean favourable excursion (potential left) |
| `trade_count` | Total trades (variants < 5 are unranked) |
| `exposure_pct` | Avg % of session in a position |
| `daily_loss_violations` | Days where kill switch would have fired |

Rankings are written to `system-data/variant_rankings_{date}.json`.

---

## Diagnostic Patterns

`scripts/diagnose_strategy.py` identifies 10 failure patterns:

1. `late_entries` — entry delay relative to signal
2. `bad_regime` — entries in bearish/mixed market context
3. `weak_volume` — entries with volume_ratio < 1.0
4. `excessive_reentry` — >3 entries per symbol per day
5. `winners_giving_back` — MFE >> close pnl
6. `stops_too_tight` — >50% exits are stop hits
7. `stops_too_wide` — losses > 3× ATR at entry
8. `poor_symbol_selection` — chronic losers in same symbol
9. `overtrading` — >20 trades/day
10. `undertrading` — <0.5 trades/day over 5+ active days

---

## Research Loop

1. **Nightly** (automated): evaluate → diagnose → generate patch requests
2. **Review**: read `research_requests/YYYYMMDD_{strategy_id}_patch_request.md`
3. **Apply** (human): edit `strategies/` or `configs/` based on recommendations
4. **Promote** (human): `python scripts/promote_variant.py {strategy_id}` opens a PR
5. **Merge** (human): review PR, merge after approval — no auto-merge

The GitHub Actions workflow (`.github/workflows/claude-research-loop.yml`) runs steps 1 automatically on weekday nights. Steps 2–5 always require a human.

---

## Adding a New Strategy

1. Create `strategies/{name}.py` extending `BaseStrategy`
2. Register in `strategies/registry.py`
3. Add a variant block in `configs/strategy_experiments.yaml`
4. Create `k8s/variants/{name}.yaml` (copy an existing variant yaml)
5. Verify: `python -m py_compile strategies/{name}.py`

Signal contract — `generate_signals()` must return a DataFrame with `SIGNAL_COLUMNS`:
```
buy_signal, sell_signal, short_signal, cover_signal,
entry_reason, exit_reason, stop_price, target_price,
risk_reward, strategy_id
```

**Strategies must never call Alpaca APIs or place orders. Signals only.**

---

## Security Notes

- Never print, log, or expose `APCA_API_KEY_ID` or `APCA_API_SECRET_KEY`
- `k8s/secrets.example.yaml` contains only `REPLACE_ME` placeholders — safe to commit
- Real secrets are injected via `kubectl create secret` — never via YAML files
- All containers run as non-root user (uid 1000)
- `PAPER_ONLY=true` and `APCA_PAPER=true` are set in both ConfigMap and Dockerfile ENV defaults
