# M2 + M3 Paper Validation Procedure

Status: NOT YET RUN. Paper validation is complete ONLY after real market-session
evidence is collected during regular US market hours (09:30-16:00 ET). The live
engine stays disabled throughout (paper only).

This verifies the streaming execution engine (M2) and the real-time terminal +
structured audit (M3) together, on a paper Alpaca account.

## Preconditions
- `.env` has valid paper Alpaca keys (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`).
- Paper mode: `APCA_PAPER=1` (default). Live stays OFF; `ALLOW_LIVE_TRADING` MUST be unset.
- `universe.txt` present (the symbols to stream).
- Run during regular market hours so trades/quotes/bars actually flow.
- Suggested terminal width >= 120 cols for the dashboard.

## Enable + run (exact commands)

Pre-flight (offline) - run the automated suites first:
```
cd /Users/blkpvnthr/Desktop/trader
python -m pytest tests/test_streaming_engine.py tests/test_terminal_dashboard.py tests/test_execution_mode.py tests/test_daily_green_engine.py -v
python -c "import trade, streaming_engine, terminal_dashboard; print('import OK')"
```

Dry mode banner check (prints the selected execution mode, no live):
```
STREAMING_ENGINE=1 TERMINAL_MODE=dashboard python -c "import trade; c=trade.build_default_config(); trade.validate_and_print_execution_mode(c)"
```

Start the streaming engine + dashboard on paper (market hours):
```
STREAMING_ENGINE=1 TERMINAL_MODE=dashboard python trade.py
```
Other terminal modes: `TERMINAL_MODE=full` (all events incl. DATA),
`TERMINAL_MODE=signals` (candidates/signals/decisions/orders),
`TERMINAL_MODE=orders` (execution + broker events only).

To fall back to the existing synchronous engine at any time, just unset the flag:
```
python trade.py          # STREAMING_ENGINE defaults to 0 -> sync loop (unchanged)
```

Structured audit log (independent of the terminal) is written to:
```
logs/execution/<YYYYMMDD>/<YYYYMMDD>_streaming_events.jsonl
```

## Validation checklist (capture evidence for each)

Connection + bootstrap
- [ ] Startup banner prints EXECUTION PATH=STREAMING, TRADING MODE=PAPER, TERMINAL MODE, FRACTIONAL.
- [ ] `[streaming] bootstrap: N symbols, equity=...` logged; dashboard shows data=up, broker=up.

Continuous market data
- [ ] Dashboard event feed shows continuous DATA updates for subscribed symbols.
- [ ] No STALE warnings during active market (staleness watchdog quiet).

Signal detection + order submission
- [ ] A green-vs-open crossing emits SIGNAL (with correlation id) then DECISION BUY then ORDER.
- [ ] Exactly ONE order per signal; no duplicate ORDER lines for the same crossing.
- [ ] RISK rejections (if any) show an explicit reason (cash_buffer_violation, exceeds_gross_exposure, below_min_notional, buying_power_below_minimum, not_fractionable_qty_below_1_share).

Broker acknowledgment + fill reconciliation
- [ ] ACK event appears after ORDER; then FILL (or PARTIAL then FILL).
- [ ] Alpaca paper dashboard position matches the engine (held + qty).
- [ ] Order state machine progresses SIGNAL->RISK->SUBMITTING->SUBMITTED->ACK->FILLED.

Exits
- [ ] Stop: a held name falling back to its open emits DECISION EXIT reason=day_turned_red then a sell.
- [ ] Target: a held name hitting entry+2R emits DECISION EXIT reason=rr_2to1_target.
- [ ] EOD: within the flatten window all held names emit DECISION EXIT reason=eod_flatten.
- [ ] Fractional liquidation: the sell qty equals the full fractional position (no stranded dust).

Safety
- [ ] No duplicate orders across rapid ticks (one-in-flight dedupe holds).
- [ ] Risk-rejection logging is accurate and visible in both terminal and JSONL.
- [ ] A forced disconnect (kill network briefly) triggers staleness + reconnect; no duplicate or lost orders after recovery.

Terminal + audit visibility
- [ ] Every candidate/decision/signal/risk/order/ack/fill is visible in the terminal (per mode) AND present in the JSONL audit log.
- [ ] Overflow counters (dropped_display/dropped_audit) stay at 0 under normal load; if non-zero, note the event rate.

Latency
- [ ] Dashboard latency panel shows event_to_submit p50 < 100 ms (record p50/p95/p99).
- [ ] Investigate any p95/p99 outliers (note cause: cold cache, REST refresh, broker slowness).

## Evidence to capture
- A screen capture / log excerpt of: the startup banner, a full SIGNAL->FILL lifecycle with correlation id, an exit (stop/target/EOD), a risk rejection, and the latency summary (p50/p95/p99).
- The JSONL audit file for the session.
- The Alpaca paper account positions/orders reconciled against the engine state.

## Sign-off
Paper validation is COMPLETE only when every box above is checked with real
market-session evidence attached. Until then, M2/M3 is code-complete and
tested offline but NOT validated. Do not proceed to M4 until this is signed off.
