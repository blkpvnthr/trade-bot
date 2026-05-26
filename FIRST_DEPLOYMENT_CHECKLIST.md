# QuantumQuant — First Paper Deployment Checklist

**Purpose:** Exact startup sequence for the first paper-trading deployment.  
**Mode:** Kubernetes paper-only. All bots use separate Alpaca paper credentials.  
**Safety rule:** Never modify risk controls, kill switches, or `PAPER_ONLY=true` setting.

---

## Pre-Flight: Local Smoke Tests

Run these before deploying to K8s or Docker Compose.

```bash
# 1. Compile check
python -m py_compile run.py strategies/*.py scripts/*.py && echo "All OK"

# 2. Strategy signal schema smoke test (all 12 scenarios must pass)
python scripts/smoke_test_variants.py

# 3. Synthetic market replay
python scripts/replay_market_day.py --date $(date +%Y-%m-%d)

# 4. Capital allocation bootstrap
python scripts/capital_allocator.py --total-capital 13000

# 5. Health monitor (expect all WARN before first pod start — that is correct)
python scripts/strategy_health_monitor.py

# 6. Dashboard aggregation
python scripts/generate_dashboard.py
```

Expected output:
- Smoke test: `ALL TESTS PASSED`
- Replay: `ALL PASS` with signals shown per strategy
- Allocator: 6 strategies at ~16.7% each (equal bootstrap weight)
- Health: all WARN (heartbeat absent, no live data) — this clears after first pod starts

---

## Secret Creation (One-Time)

**Never commit real credentials. Never print API key values.**

```bash
# Create the K8s namespace first
kubectl apply -f k8s/namespace.yaml

# Verify namespace exists
kubectl get namespace trade-research

# Bot 01 — conservative (tux_mtf variants)
kubectl create secret generic alpaca-paper-bot-tux-mtf \
  --from-literal=APCA_API_KEY_ID=<YOUR_BOT01_KEY> \
  --from-literal=APCA_API_SECRET_KEY=<YOUR_BOT01_SECRET> \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  --from-literal=APCA_PAPER=true \
  -n trade-research

# Bot 02 — aggressive (pivot + ORB)
kubectl create secret generic alpaca-paper-bot-pivot-orb \
  --from-literal=APCA_API_KEY_ID=<YOUR_BOT02_KEY> \
  --from-literal=APCA_API_SECRET_KEY=<YOUR_BOT02_SECRET> \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  --from-literal=APCA_PAPER=true \
  -n trade-research

# Bot 03 — experimental (liquidity sweep + flawless victory)
kubectl create secret generic alpaca-paper-bot-experimental \
  --from-literal=APCA_API_KEY_ID=<YOUR_BOT03_KEY> \
  --from-literal=APCA_API_SECRET_KEY=<YOUR_BOT03_SECRET> \
  --from-literal=APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  --from-literal=APCA_PAPER=true \
  -n trade-research

# Verify secrets exist (never shows values)
kubectl get secret -n trade-research
kubectl describe secret alpaca-paper-bot-tux-mtf -n trade-research
# Should list keys: APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, APCA_PAPER
```

**Getting Alpaca paper API keys:**
1. Log in at https://app.alpaca.markets (Paper Trading)
2. API Keys → Generate → copy Key ID + Secret
3. Create 3 separate paper accounts OR use one account for all bots
   - Note: one account = shared buying power across all 6 bots

---

## Docker Build

```bash
# Build image
docker build -t quantumquant:latest .

# Quick verify — should print startup diagnostics then exit (no API keys)
docker run --rm quantumquant:latest \
  -e PAPER_ONLY=true \
  -e APCA_PAPER=true \
  -e STRATEGY_ID=tux_mtf_original \
  -e CONFIG_PATH=configs/strategy_experiments.yaml \
  -e APCA_API_KEY_ID=test \
  -e APCA_API_SECRET_KEY=test \
  python -c "
import sys; sys.argv=['run']
# Should print [startup] FATAL: APCA_API_KEY_ID appears to be a placeholder
" 2>&1 | head -5
```

---

## Kubernetes Deployment

```bash
# Apply shared config (no secrets here)
kubectl apply -f k8s/configmap.yaml

# Verify configmap
kubectl get configmap quantumquant-config -n trade-research -o yaml | grep -E "PAPER_ONLY|CONFIG_PATH"

# Deploy all 6 variants
kubectl apply -f k8s/variants/

# Deploy nightly evaluator CronJob
kubectl apply -f k8s/cronjobs/nightly_evaluator.yaml

# Watch startup (first pod becomes Ready after ~30s)
kubectl get pods -n trade-research -w
```

---

## Expected Startup Logs (First 60 Seconds)

Watch logs for the expected sequence:

```bash
kubectl logs -n trade-research deployment/trader-tux-mtf-original -f
```

Expected lines in order:

```
[startup] safety checks passed | paper_only=True | apca_paper=True | strategy=tux_mtf_original
[startup] strategy=tux_mtf_original | class=TuxMTF | params=8 | credential_set=bot01 | timeframe=1Min | regime=any
[metrics] Prometheus endpoint: http://0.0.0.0:8080/metrics
[executor] starting main loop | strategy=tux_mtf_original | paper=True
```

**Red flags (check immediately):**
- `[startup] FATAL:` — misconfigured environment, fix and redeploy
- `CrashLoopBackOff` — check `kubectl logs --previous`
- Pod stuck in `Pending` > 2 min — check `kubectl describe pod`

---

## Smoke Tests After Deployment

```bash
# Confirm readiness probe passes (1/1 READY)
kubectl get pods -n trade-research

# Check liveness probe manually
kubectl exec -n trade-research deployment/trader-tux-mtf-original -- \
  python3 -c "
import time,sys,os
f='/tmp/trader_heartbeat'
age=time.time()-os.path.getmtime(f) if os.path.exists(f) else 9999
print(f'heartbeat age={age:.0f}s')
sys.exit(0 if age < 600 else 1)
" && echo "LIVE" || echo "NOT LIVE"

# Check Prometheus endpoint
kubectl port-forward -n trade-research deployment/trader-tux-mtf-original 8080:8080 &
curl -s http://localhost:8080/metrics
# Expected: qq_equity, qq_gross_exposure, qq_skipped_signals_today

# Check strategy metrics snapshot
kubectl exec -n trade-research deployment/trader-tux-mtf-original -- \
  cat /app/logs/execution/tux_mtf_original/metrics.json

# Run evaluator to verify it produces rankings
kubectl create job --from=cronjob/nightly-evaluator manual-eval-$(date +%s) -n trade-research
kubectl logs -n trade-research job/manual-eval-* -f
```

---

## Monitoring After First Market Day

```bash
# Activity report (run after market close)
python scripts/strategy_activity_report.py

# Check skip reasons
cat logs/execution/tux_mtf_original/skipped_signals.jsonl | tail -20 | python -c "
import sys,json
for line in sys.stdin:
    r=json.loads(line)
    print(r.get('reason',''), r.get('symbol',''), r.get('timestamp',''))
"

# Update dashboard
python scripts/generate_dashboard.py
cat daily_system_outputs/dashboard.json

# Health monitor (should mostly clear to OK once pods run)
python scripts/strategy_health_monitor.py

# Capital allocator (still bootstrap until Sharpe data accumulates)
python scripts/capital_allocator.py
```

---

## Rollback Steps

```bash
# Scale all bots to 0 (immediate stop, no forced pod kill)
kubectl scale deployment --all --replicas=0 -n trade-research

# Or pause a specific variant
kubectl scale deployment trader-tux-mtf-original --replicas=0 -n trade-research

# Rolling restart after a fix (graceful — waits for SIGTERM handler)
kubectl rollout restart deployment/trader-tux-mtf-original -n trade-research
kubectl rollout status deployment/trader-tux-mtf-original -n trade-research

# View last crash logs
kubectl logs -n trade-research deployment/trader-tux-mtf-original --previous
```

---

## Emergency Shutdown

```bash
# Hard stop all bots immediately
kubectl delete deployment --all -n trade-research

# Or scale to 0 (gentler — preserves deployment specs)
for d in $(kubectl get deployment -n trade-research -o name); do
  kubectl scale $d --replicas=0 -n trade-research
done
```

---

## Safety Invariants — Never Modify

These constants must remain unchanged for paper-trading safety:

| Control | File | Value |
|---------|------|-------|
| Daily loss kill switch | run.py | `MAX_DAILY_LOSS_PCT=0.015`, `MAX_DAILY_LOSS_DOLLARS=2000` |
| Equity floor | run.py | `KILL_SWITCH_EQUITY_FLOOR=95000` |
| Cash buffer | run.py | `MIN_CASH_BUFFER=1500` |
| Max gross exposure | run.py | `MAX_GROSS_EXPOSURE=0.72` |
| Max order dollars | run.py | `MAX_ORDER_DOLLARS=1500` |
| Paper assertion | run.py | `_startup_safety_checks()` aborts on non-paper config |
| Placeholder detection | run.py | Aborts if API key starts with YOUR\_, REPLACE_ME, etc. |
| STRATEGY_ID validation | run.py | Validates against config YAML known variants |

---

*Deployment is paper-only. No auto-promotion to live trading. All live-trading promotions require explicit manual PR review.*

---

## Post-Market Retraining Validation

After the first full market day, verify the retraining pipeline works.

### Option A — Automatic (run.py hook, opt-in)

Enable in `.env.botNN` (one instance is enough to own retraining):
```bash
POST_MARKET_RETRAIN_ENABLED=true
POST_MARKET_RETRAIN_START_ET=1605
POST_MARKET_RETRAIN_ONCE_PER_DAY=true
```

After 16:05 ET, check execution events for the instance that had this enabled:
```bash
grep 'post_market_retrain' logs/execution/tux_mtf_original/$(date +%Y%m%d)_execution_events.jsonl
```

### Option B — Manual standalone

```bash
python scripts/post_market_retrain_pipeline.py
```

### Validation

```bash
# Check summary report
cat daily_system_outputs/post_market_retrain_summary.json | python3 -m json.tool

# Expected:
# "all_ok": true
# "provenance_confirmed": true
# "artifacts": { "bars_csv": {"exists": true}, "rl_model_zip": {"exists": true}, ... }

# Check RL model lifecycle
python scripts/check_rl_training_status.py
```
