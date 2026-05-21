# QuantumQuant — Runtime Readiness Report

**Generated:** 2026-05-20  
**Status:** Ready for paper-trading deployment  
**Mode:** Kubernetes / Docker Compose (paper only)

---

## 1. Required Local Installs

### macOS (Homebrew)

```bash
# Python 3.11+
brew install python@3.11

# Docker Desktop (includes docker compose)
brew install --cask docker

# kubectl
brew install kubectl

# GitHub CLI (for promote_variant.py)
brew install gh
gh auth login

# Optional: k9s (Kubernetes TUI)
brew install k9s
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Minimum versions (from requirements.txt):**

| Package | Min Version |
|---------|------------|
| alpaca-py | 0.20.0 |
| pandas | 2.0.0 |
| numpy | 1.26.0 |
| pyyaml | 6.0.1 |
| torch | 2.1.0 |
| stable-baselines3 | 2.2.0 |
| yfinance | 0.2.38 |

---

## 2. Docker Desktop Setup

1. Download from https://www.docker.com/products/docker-desktop/
2. Install and open Docker Desktop
3. In Settings → Resources: set Memory ≥ 4 GB, CPUs ≥ 2
4. Verify: `docker version` and `docker compose version`

---

## 3. kubectl Setup (Kubernetes)

### Option A: Docker Desktop built-in Kubernetes
1. Docker Desktop → Settings → Kubernetes → Enable Kubernetes → Apply
2. Wait for the green Kubernetes status indicator
3. Verify: `kubectl cluster-info`
4. Context will be `docker-desktop`

### Option B: kind (Kubernetes in Docker)
```bash
brew install kind
kind create cluster --name quantumquant
kubectl cluster-info --context kind-quantumquant
```

### Option C: Remote cluster (GKE, EKS, AKS)
```bash
# GKE example
gcloud container clusters get-credentials <cluster-name> --region <region>
kubectl cluster-info
```

---

## 4. Kubernetes Deployment Steps

```bash
# 0. Build the image
docker build -t quantumquant:latest .

# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create secrets — NEVER commit real keys
kubectl create secret generic alpaca-paper-bot-tux-mtf \
  --from-literal=APCA_API_KEY_ID=YOUR_KEY_BOT01 \
  --from-literal=APCA_API_SECRET_KEY=YOUR_SECRET_BOT01 \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  --from-literal=APCA_PAPER=true \
  -n trade-research

kubectl create secret generic alpaca-paper-bot-pivot-orb \
  --from-literal=APCA_API_KEY_ID=YOUR_KEY_BOT02 \
  --from-literal=APCA_API_SECRET_KEY=YOUR_SECRET_BOT02 \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  --from-literal=APCA_PAPER=true \
  -n trade-research

kubectl create secret generic alpaca-paper-bot-experimental \
  --from-literal=APCA_API_KEY_ID=YOUR_KEY_BOT03 \
  --from-literal=APCA_API_SECRET_KEY=YOUR_SECRET_BOT03 \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  --from-literal=APCA_PAPER=true \
  -n trade-research

# 3. Apply shared config (no secrets in here)
kubectl apply -f k8s/configmap.yaml

# 4. Deploy all strategy variants
kubectl apply -f k8s/variants/

# 5. Deploy nightly evaluator CronJob
kubectl apply -f k8s/cronjobs/nightly_evaluator.yaml

# Verify everything is up
kubectl get all -n trade-research
```

---

## 5. Secret Creation Reference

| Secret Name | Credential Set | Variants |
|-------------|---------------|---------|
| `alpaca-paper-bot-tux-mtf` | bot-01 (conservative) | tux_mtf_original, tux_mtf_long_only |
| `alpaca-paper-bot-pivot-orb` | bot-02 (aggressive) | pivot_points, orb_8am |
| `alpaca-paper-bot-experimental` | bot-03 (experimental) | liquidity_sweep, flawless_victory |

**Getting Alpaca paper API keys:**
1. Log in at https://app.alpaca.markets (or https://paper.alpaca.markets)
2. Switch to Paper Trading mode
3. API Keys → Generate → copy Key ID and Secret
4. Create 3 separate paper accounts or use the same account for all bots (note: using the same account means all bots share buying power)

**Verify secrets (never prints values):**
```bash
kubectl get secret -n trade-research
kubectl describe secret alpaca-paper-bot-tux-mtf -n trade-research
```

---

## 6. First-Launch Commands

### Docker Compose (local testing)

```bash
# 1. Create env files from examples
cp .env.bot01.example .env.bot01
cp .env.bot02.example .env.bot02
cp .env.bot03.example .env.bot03

# 2. Edit each file and fill in real paper credentials
#    APCA_API_KEY_ID=PK...
#    APCA_API_SECRET_KEY=...

# 3. Start all bots
docker compose up --build

# Start a single variant for testing
STRATEGY_ID=tux_mtf_original \
CONFIG_PATH=configs/strategy_experiments.yaml \
PAPER_ONLY=true \
APCA_PAPER=true \
APCA_API_KEY_ID=YOUR_KEY \
APCA_API_SECRET_KEY=YOUR_SECRET \
python run.py
```

### Kubernetes

```bash
# Scale a specific variant to 0 (pause it)
kubectl scale deployment trader-tux-mtf-original --replicas=0 -n trade-research

# Scale back up
kubectl scale deployment trader-tux-mtf-original --replicas=1 -n trade-research

# Rolling restart (triggers SIGTERM → graceful shutdown → new pod)
kubectl rollout restart deployment/trader-tux-mtf-original -n trade-research

# Watch rollout
kubectl rollout status deployment/trader-tux-mtf-original -n trade-research
```

---

## 7. Smoke-Test Commands

### Startup safety gate test

```bash
# Should abort: PAPER_ONLY not set
APCA_API_KEY_ID=test APCA_API_SECRET_KEY=test python -c "
import os, sys
os.environ['APCA_API_KEY_ID'] = 'test'
os.environ['APCA_API_SECRET_KEY'] = 'test'
# PAPER_ONLY and APCA_PAPER not set — should fail
exec(open('run.py').read().split('def main')[0])
" 2>&1 | grep '\[startup\]'

# Correct invocation:
PAPER_ONLY=true APCA_PAPER=true \
APCA_API_KEY_ID=test APCA_API_SECRET_KEY=test \
STRATEGY_ID=tux_mtf_original \
CONFIG_PATH=configs/strategy_experiments.yaml \
python -c "
import os, sys
# Run just the safety check
exec(open('run.py').read().split('def _startup_safety_checks')[1].split('\ndef ')[0])
" 2>&1 | head -5
```

### Strategy module smoke test

```bash
# Verify all strategies import cleanly and can be instantiated
python3 -c "
from strategies.registry import STRATEGY_REGISTRY, get_strategy
import pandas as pd, numpy as np

for sid, cls in STRATEGY_REGISTRY.items():
    s = get_strategy(sid)
    print(f'OK {sid}: {cls.__name__} params={len(s.params)}')

# Minimal signal generation test
bars = pd.DataFrame({
    'timestamp': pd.date_range('2026-01-02 09:30', periods=100, freq='1min', tz='America/New_York'),
    'open': np.random.uniform(100, 110, 100),
    'high': np.random.uniform(110, 120, 100),
    'low':  np.random.uniform(90, 100, 100),
    'close': np.random.uniform(100, 110, 100),
    'volume': np.random.randint(1000, 50000, 100),
})
for sid in STRATEGY_REGISTRY:
    s = get_strategy(sid)
    try:
        sig = s.generate_signals(bars)
        print(f'  signals OK {sid}: shape={sig.shape}')
    except Exception as e:
        print(f'  WARN {sid}: {e}')
"
```

### Compile check

```bash
python -m py_compile run.py strategies/*.py scripts/*.py && echo "All OK"
```

### K8s dry-run

```bash
kubectl apply --dry-run=client -f k8s/namespace.yaml
kubectl apply --dry-run=client -f k8s/configmap.yaml
kubectl apply --dry-run=client -f k8s/variants/
kubectl apply --dry-run=client -f k8s/cronjobs/
```

### Evaluator smoke test

```bash
# Runs even with no trade data — produces empty rankings
python scripts/evaluate_variants.py --config configs/strategy_experiments.yaml
cat daily_system_outputs/research_rankings.json

# Diagnose a variant
python scripts/diagnose_strategy.py tux_mtf_original
cat daily_system_outputs/strategy_diagnostics.json
```

### Pod health

```bash
# Check pod status
kubectl get pods -n trade-research

# Check readiness (should be 1/1 after main loop starts ~30s)
kubectl get pods -n trade-research -w

# Check liveness probe manually from inside pod
kubectl exec -n trade-research deployment/trader-tux-mtf-original -- \
  python3 -c "import time,sys,os; f='/tmp/trader_heartbeat'; sys.exit(0 if os.path.exists(f) and time.time()-os.path.getmtime(f)<600 else 1)" \
  && echo "LIVE" || echo "NOT LIVE"
```

---

## 8. Troubleshooting Guide

### `[startup] FATAL: PAPER_ONLY is not 'true'`

**Cause:** `PAPER_ONLY` env var is missing or not `true`.  
**Fix:** Add `PAPER_ONLY=true` to your `.env` file or K8s ConfigMap. It is already set in the ConfigMap but may be overridden.

```bash
kubectl get configmap quantumquant-config -n trade-research -o yaml | grep PAPER_ONLY
```

---

### `[startup] FATAL: APCA_API_KEY_ID is not set`

**Cause:** Secret not created or not referenced by the deployment.  
**Fix:**

```bash
# Verify secret exists
kubectl get secret alpaca-paper-bot-tux-mtf -n trade-research

# Verify it has the right keys
kubectl describe secret alpaca-paper-bot-tux-mtf -n trade-research
# Should show: APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, APCA_PAPER

# Verify deployment references the secret
kubectl describe deployment trader-tux-mtf-original -n trade-research | grep -A3 envFrom
```

---

### `[startup] FATAL: STRATEGY_ID=... is set but CONFIG_PATH is missing`

**Cause:** STRATEGY_ID env var is set but CONFIG_PATH is not.  
**Fix:** Both must be set together. Check ConfigMap:

```bash
kubectl get configmap quantumquant-config -n trade-research -o yaml
# CONFIG_PATH should be: configs/strategy_experiments.yaml
```

---

### Pod stays in `Pending` state

**Cause:** Insufficient CPU/memory on node, or secret doesn't exist.

```bash
kubectl describe pod <pod-name> -n trade-research | grep -A10 Events
```

Common causes:
- `secret "alpaca-paper-bot-tux-mtf" not found` → create the secret
- `Insufficient cpu` → reduce resource requests in variant YAML or scale up cluster

---

### Pod restarts repeatedly (CrashLoopBackOff)

```bash
# View crash logs
kubectl logs -n trade-research deployment/trader-tux-mtf-original --previous

# Common causes:
# 1. Bad API keys → Alpaca auth error in logs
# 2. Python import error → check for syntax errors
# 3. Missing dependency → add to requirements.txt and rebuild
```

---

### Liveness probe failing (`/tmp/trader_heartbeat` too old)

**Cause:** Main loop is stuck or taking >10 minutes per cycle.  
**Fix:** Check for blocking operations in the cycle:

```bash
kubectl logs -n trade-research deployment/trader-tux-mtf-original | tail -50 | grep '\[executor\]'
```

The heartbeat is written every `HEARTBEAT_SECONDS=60`. If the cycle takes longer than ~9 minutes, the probe will fail. Increase `failureThreshold` or `periodSeconds` in the variant YAML if needed.

---

### Readiness probe failing (pod never becomes Ready)

**Cause:** Main loop hasn't started yet — startup initialization is taking longer than 30s.  
**Likely cause:** PSO-LSTM optimization or forecast ensemble refresh at startup.

```bash
kubectl logs -n trade-research deployment/trader-tux-mtf-original | grep '\[startup\]\|\[forecast\]\|\[pso\]'
```

**Fix:** Increase `readinessProbe.initialDelaySeconds` to 120 in the variant YAML, or disable slow startup tasks:

```bash
kubectl set env deployment/trader-tux-mtf-original \
  FORECAST_ENSEMBLE_REFRESH_ON_START=0 \
  -n trade-research
```

---

### `[kill-switch] execution blocked`

**Normal behavior** — the daily loss limit was hit. The bot will unblock the next trading day automatically. Check:

```bash
kubectl logs -n trade-research deployment/trader-tux-mtf-original | grep 'kill.switch'
cat logs/execution/tux_mtf_original/skipped_signals.jsonl | tail -5
```

---

### No trades executing (signal count 0)

```bash
# Check if signals are being generated
cat logs/execution/tux_mtf_original/skipped_signals.jsonl

# Check strategy diagnostics
cat logs/execution/tux_mtf_original/strategy_diagnostics.json

# Check market is open
kubectl exec -n trade-research deployment/trader-tux-mtf-original -- \
  python3 -c "
from alpaca.trading.client import TradingClient
import os
tc = TradingClient(os.environ['APCA_API_KEY_ID'], os.environ['APCA_API_SECRET_KEY'], paper=True)
clock = tc.get_clock()
print(f'is_open={clock.is_open} next_open={clock.next_open}')
"
```

---

## 9. Output File Reference

| File | Description |
|------|-------------|
| `logs/execution/{STRATEGY_ID}/skipped_signals.jsonl` | Every skipped signal with reason and gating condition |
| `logs/execution/{STRATEGY_ID}/strategy_diagnostics.json` | Strategy config, params, regime at startup |
| `logs/execution/{STRATEGY_ID}/metrics.json` | Live equity, exposure, skip count (updated every 60s) |
| `logs/execution/{STRATEGY_ID}/{DATE}_executor_journal.csv` | Full trade journal with P&L |
| `logs/execution/{STRATEGY_ID}/{DATE}_execution_events.jsonl` | Structured execution event stream |
| `daily_system_outputs/research_rankings.json` | Nightly variant rankings (all metrics) |
| `daily_system_outputs/strategy_diagnostics.json` | Latest diagnostic run output |
| `research_requests/{DATE}_{STRATEGY_ID}_patch_request.md` | Claude patch request for review |
| `/tmp/trader_heartbeat` | K8s liveness probe file (mtime checked) |

---

## 10. Safety Invariants (Never Modify)

These are preserved unchanged and must never regress:

| Control | File | Constant |
|---------|------|----------|
| Daily loss kill switch | run.py:7908 | `MAX_DAILY_LOSS_PCT=0.015`, `MAX_DAILY_LOSS_DOLLARS=2000` |
| Equity floor | run.py:2287 | `KILL_SWITCH_EQUITY_FLOOR=95000` |
| Cash buffer | run.py:2274 | `MIN_CASH_BUFFER=1500`, `DEFAULT_CASH_BUFFER_PCT=0.12` |
| Max gross exposure | run.py:2276 | `MAX_GROSS_EXPOSURE=0.72` |
| Max position weight | run.py:2277 | `MAX_POSITION_WEIGHT=0.12` |
| Max order dollars | run.py:2279 | `MAX_ORDER_DOLLARS=1500` |
| PDT band | run.py:2992 | `MICRO_UNDER_PDT` dynamically sized |
| Options booster | run.py:2627 | Requires `options_booster_confidence_threshold=0.75` |
| Paper assertion | run.py:main() | `_startup_safety_checks()` aborts if `APCA_PAPER != true` without `ALLOW_LIVE_TRADING=1` |
| Sell before buy | run.py:10104 | `sell_first=True` in `build_default_config()` |

---

*This report was generated as part of the QuantumQuant K8s research platform runtime readiness review.*  
*All variants are paper-trading only. No auto-promotion to live trading.*
