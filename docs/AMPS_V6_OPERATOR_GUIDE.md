# AMPS V6 — Operator Guide

_Branch `research/amps-v2-entry-exit-calibration` · 2026-11-20_

## What AMPS V6 is

An evidence-gated, multi-horizon trading **platform** (not a strategy). It separates platform
readiness from strategy eligibility from live authorization (three independent states). Agents only
analyze/recommend/veto; only a deterministic execution gateway may ever place orders; protective
exits run independently of agents and LLMs. See `AMPS_V6_AGENT_ARCHITECTURE.md`.

## Operating modes (spec §3)

`BACKTEST → OBSERVE_ONLY → SHADOW → PAPER → LIVE_PILOT → LIVE`. All modes share identical strategy
logic, features, evidence gates, and risk math; only the execution adapter and authorization differ.
**Default: `OBSERVE_ONLY`, `new_entries_enabled=false`, global kill switch tripped.** Mode is never
inferred from credentials.

## Everyday commands

```bash
# Decisive entry-alpha gate (read-only; builds adjusted-daily cache on first run; needs Alpaca creds)
python -m amps_v6.backtesting.multi_horizon_diagnostic --train-end 2024-05-01 --tag mygate

# Tests
python -m pytest tests/test_amps_v6.py -q

# Coverage of the candidate universe (which symbols actually have cached bars)
python -c "from amps_v6.universe import instrument_registry as R; import json; print(json.dumps(R.coverage_report(), indent=2))"

# Liquidity ranking of the candidate ETFs (uses the adjusted-daily cache)
python - <<'PY'
from amps_v6.universe import instrument_registry as R
from amps_v6.universe import liquidity_ranker as L
for sym, score, comp in L.rank(R.tradable_universe())[:10]:
    print(f"{sym:6} {score:.3f} {comp}")
PY
```

## Kill switches (spec §42) — all persist across restarts

`global, strategy, new_entries, overnight_entries, position_promotion, leveraged_etf, data_provider,
agent_system`. The **global** switch is tripped by default (fail-safe). Trip/clear via
`PlatformState.trip(name)` / `.clear(name)` then `.save()`. The risk gateway and the live-order
conjunction both honor them fail-closed.

## Evidence lifecycle (spec §14)

`RESEARCH_CANDIDATE → PAPER_CANDIDATE → LIVE_PILOT_CANDIDATE → LIVE_APPROVED`, with
`LIVE_SUSPENDED` / `REVALIDATION_REQUIRED` as drift demotions. A mode may only act on evidence at or
above its minimum (`PAPER` needs `PAPER_CANDIDATE`+, `LIVE` needs `LIVE_APPROVED`). Promotion is one
step at a time and explicit. Managed via `amps_v6.alpha.evidence_registry.EvidenceRegistry`.

## Current state (as of this build)

No horizon is validated; nothing is paper/live eligible. The only non-REJECTED result is swing
5–10d = CONTINUE_RESEARCH. See `AMPS_V6_LIVE_READINESS.md` for the full status and blockers.

## Reproducing the data integrity guard

The swing analysis **must** use split/dividend-adjusted daily bars. The engine fetches Alpaca
`adjustment=ALL` and caches to `data/cache/amps_v6_daily_adj/`. Never run multi-day analysis on the
unadjusted `bars_multi_tf` cache (leveraged/inverse ETF splits create spurious jumps).
