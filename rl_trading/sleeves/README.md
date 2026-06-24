# Sleeve Framework (`rl_trading.sleeves`)

A **paper-only, validation-gated** framework that turns the research harness into a tradeable
system **without pretending we found alpha**. A sleeve may submit paper orders ONLY after its
information source clears the standing gate. **Live trading is permanently disabled** — there is no
code path that enables it.

## Current truth (enforced, not asserted)
- No deployable SPY-beating edge found yet → **0 sleeves validated, 0 tradeable**.
- The MLP trend ranker is the best research model but is below SPY → **scanner-only**.
- The real asset is the gate: `PIT → event/IC study → bootstrap + placebo → OOS + OOD → native
  execution → sleeve → allocator`.

## The gate (`core.validation_gate`)
A sleeve becomes tradeable only if its `ValidationRecord` clears **every** criterion:
positive bootstrap CI · placebo passed · OOS passed · OOD passed · attribution passed ·
not concentrated in one name/year/domain. Any missing field is treated as a failure
(absence of evidence ≠ evidence). Missing/partial record ⇒ **scanner-only**.

## Minting a record (`promote.from_event_study`)
The only sanctioned way to produce a passing record: feed real `event_study.stats()` output
(real + placebo + per-domain + drop-top-domain). The booleans are derived mechanically from the
statistics, so they cannot be hand-faked. This replays — and rejects — the real PEAD
falsification (placebo ≥ signal).

## Modes
- `DISABLED` — no validation record (event_hold / macro_timing / news_event placeholders).
- `SCANNER_ONLY` — has a record but it does not clear the gate (trend_ranker), or paper not enabled.
  `sleeve.scan()` still emits candidates + PIT timestamp + proposed horizon + reason-not-tradeable +
  required next validation step.
- `PAPER` — gate cleared **and** `paper_enabled=True`. Still never live.

## Paper execution (`paper.PaperExecutor`)
Never calls a broker — a deterministic forward-validation recorder. Every order (accepted or
blocked, with reasons) is appended to a JSONL audit log. Hard blocks, in order: **live (always)**,
sleeve-not-tradeable, kill switch, daily-loss limit, per-order notional, per-symbol position %,
total notional, gross exposure, max concurrent. `record_pnl()` auto-trips the kill switch on the
daily-loss cap. Behavior is deterministic (no wall clock, no RNG).

## Status / dashboard (`registry.status_report`, `status.py`)
`python -m rl_trading.sleeves.status [--json]` → validated / scanner-only / rejected sleeves,
champion, per-sleeve gate results and why-tradeable-or-not, paper/live status, and the mandatory
banner: **no validated live edge, no 1%/day, paper = forward validation only.**

## Tests
`rl_trading/tests/test_sleeves.py` (33): unvalidated cannot trade · missing-ledger-pass blocks ·
live always blocked · paper-only guard · kill switch · every risk cap · scanner works when trading
blocked · audit log written · determinism · promotion bridge (incl. placebo/concentration/OOD
rejections) · end-to-end promoted sleeve becomes paper-tradeable.
