# Real-Time IEX Market Scanner → Authoritative `universe.txt` → Hot-Reload → Monitor-Based Loss-Removal / Replacement

**Date:** 2026-11-21
**Branch:** `13K-development`
**Status:** APPROVED (design gate passed 2026-11-21; both scope forks locked by user)
**Governs:** new `market_scan/` package + integration into `trade.py` (the live engine)

---

## 1. Objective

Build a real-time Alpaca **IEX** market scanner that runs automatically inside `trade.py`, continuously
ranks eligible US-listed instruments, atomically republishes the root-level `universe.txt` as the
**authoritative** entry/replacement universe, and is **hot-reloaded** by the live engine each cycle —
without a separate process, without blocking execution/risk, and **fail-closed for new entries** on any
scanner fault while **never** impeding exits or risk management for existing positions.

This is a *candidate-selection layer*, not a replacement for the trading architecture. Every existing
gate, sizing rule, and risk control remains in force.

---

## 2. Locked decisions (this build, `13K-development`)

### 2.1 Section-9 scope — **monitor-based loss removal + dynamic replacement** (defer broker-native stop)
- Continuously monitor broker-reported `unrealized_plpc` for every open position.
- Trigger the **existing** exit execution pipeline immediately when `unrealized_plpc <= -0.005` (−0.5%,
  decimal fraction — confirmed convention, see §6.4).
- The loss threshold is **un-overridable**: no bullish momentum, conflicting strategy signal, or minimum
  holding period may veto it.
- Prevent duplicate exit orders; reconcile existing protective orders.
- **Confirm the position is closed** before initiating a replacement; reconcile actual available buying
  power after execution.
- Reload the latest **validated** `universe.txt`; evaluate replacements in descending rank order.
- Replacement eligibility (ALL required): (1) confirmed seller exhaustion, (2) positive momentum,
  (3) DI+ > DI−, (4) price > Bollinger midpoint, (5) price > session VWAP (intraday), (6) sufficient
  liquidity + acceptable spread, (7) regime/directional alignment, (8) all existing portfolio,
  position-sizing, and execution risk controls.
- No eligible replacement → **remain in cash**. Never force a replacement because a position closed.
- Prevent immediate re-entry into stopped-out symbols unless a fresh, independently confirmed setup
  develops (reuse `load_no_reentry_symbols`/`mark_no_reentry_symbol`).
- Loss-removal is **independent of scanner availability**: a stale/empty/unavailable `universe.txt`
  blocks *replacements* but must **never** block loss-triggered *exits*.
- **Reuse** the existing exit funnel, portfolio optimizer, rebalance pipeline, execution infra. No
  competing optimizer.
- **Deferred to `amps-v6-loss-removal-optimizer`:** broker-native resting stop / stop-limit,
  AlpacaBroker protective-order modifications, canonical Total-P/L ledger, broker-side protective-order
  placement/verification. The monitor is designed to later serve as the **secondary** backstop once the
  broker-native stop merges (see §9 merge notes).

### 2.2 Entry-driving default — **scanner drives entries, ON by default**
- `MARKET_SCAN_ENABLED=1` by default; scanner auto-initializes on `trade.py` startup; **no new CLI args**.
- `universe.txt` is authoritative for new-entry **and** replacement candidates; continuous hot-reload,
  no engine restart; candidates evaluated in descending rank order.
- Preserve every existing strategy-specific entry condition, execution gate, sizing rule, portfolio risk
  control. Support intraday and swing positions. Support live and paper via existing Alpaca credentials
  (operator differentiates accounts by API keys).
- **Operational states:**
  - **Enabled + healthy** → permit qualified entries & replacements from the dynamic universe.
  - **Enabled + unhealthy** → **block** new entries & replacements immediately; continue managing
    existing positions & exits.
  - **Disabled (`=0`)** → stop the scanner; block new entries & replacements; continue monitoring,
    reconciliation, risk. **Do NOT silently revert to a static universe.**
- Before submitting any new entry, independently verify: scanner health, universe freshness, current
  membership, market-data freshness, broker tradability, strategy-specific eligibility, portfolio risk +
  execution approval.
- No mandatory shadow period / paper-only restriction. After technical validation, normal startup
  activates dynamic entry selection automatically.

---

## 3. Grounding facts (from the architectural map; file:line)

- **Production entrypoint is `trade.py` only** (`start.sh:10` → `python -u trade.py`). `run.py` /
  `trade_scanner_fixed.py` are legacy copies — **not modified**.
- Engine is a **single-threaded sync poll loop**: `main()` (7748) → `build_default_config()` (7508, the
  *only* place env flags are read) → `validate_and_print_execution_mode()` (7627) →
  `ProductionOrderExecutor.run()` (7322) → `while True: run_cycle_from_files(); _print_heartbeat();
  time.sleep(poll_seconds=30)` (7426–7436).
- Exactly one background daemon thread today: `launch_intraday_dataset_builder` spawned at
  `trade.py:7757` — mirror this to start the scanner.
- **`trade.py:248 DEFAULT_SCAN_UNIVERSE = load_trading_universe()` is import-time cached** — the sole
  reason universe.txt edits are invisible. Per-cycle choke point: `MomentumSetupScanner._universe()`
  (3219), used by every strategy branch (`_build_daily_green_orders:4355`, `build_orders:4419`).
  Precedence: explicit `config.universe` (3220) > LLM (3223) > SP500/broad (3228) > `DEFAULT_SCAN_UNIVERSE`
  (3232). LLM/SP500 forced OFF in prod (`build_default_config` 7530/7542).
- **Production opens NO websocket** — REST only (`StockHistoricalDataClient`, `DataFeed.IEX` via
  `_feed()` 3270). `StreamingEngine`'s IEX socket is hard-disabled (`main()` raises at 7754–7755). The
  scanner's `StockDataStream` is therefore the **sole** IEX connection. Alpaca permits **one** websocket
  per account/feed; `scanner.py:291`, `deeplob_orderbook_direction_layer.py:126`,
  `stream_confirmed_movement_gate_trade_plan_trailing.py` each open their own → **single-instance lock
  required**.
- **Exits are already universe-independent**: `monitor_active_positions` (3911),
  `_update_high_water_and_exits` (4225), `_run_profit_exit_overlay` (6564) iterate
  `broker.get_all_open_positions()` — dropping a symbol from universe.txt already cannot liquidate it.
  Only the scanner *subscription set* needs positions∪pending∪refs.
- **`universe.load_canonical_universe` is fail-closed (RuntimeError on empty/missing)**
  (`universe.py:289/295`) and is called at import. **No atomic-write helper exists** anywhere. →
  add `write_canonical_universe()` (temp + `os.replace`); never emit an empty file.
- **`_canonical_scan_order` rebuilds its rotation cursor when the symbol list changes** (3344) → write
  universe.txt **only on membership change / discovery cadence**, not on every 3s rank tick.
- **Credentials:** `AlpacaBroker` (2557) reads `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` (no `ALPACA_*`
  fallback). Paper/live via `TradingClient(paper=...)`; `paper=os.getenv("APCA_PAPER","1")` (7588); live
  requires `APCA_PAPER=0` AND `ALLOW_LIVE_TRADING=1` (7633–7639). **Scanner reuses `executor.broker`** →
  inherits mode automatically (in-process design makes paper/live isolation automatic).
- **Order-flow engine** (`order_flow_engine.py`) is REST-only (`poll_rest` 567). Present: CVD, 10/30/60/300s
  windows, pressure, buy/sell-pressure, buyer/seller-exhaustion (CVD-divergence proxy), L1 quote imbalance.
  **Absent (to add):** microprice, absorption, bid-replenishment, volume-climax; VWAP lives outside the
  engine. L2/iceberg **UNAVAILABLE** (IEX has no L2). Mixing WS + REST into the *same* `SymbolOrderFlow`
  double-counts (512-deep dedup deque) → scanner uses its **own** stream-fed `OrderFlowEngine`.
- **Portfolio optimizer is score-weighting**, no covariance/correlation/marginal-Sharpe
  (`expected_sharpe_score` is an input column). Correlation/leverage guard → `universe.exposure_groups()`
  / `SYMBOL_REGISTRY`. `optimize_portfolio` **returns 4 values** despite 3-tuple annotation (trap).
  `portfolio_rebalancer.get_account_and_positions` hardcodes `paper=True` → live loss-removal must read
  positions from `executor.broker`.
- **P/L convention = decimal fraction**: −0.5% = `-0.005` (proven by `stop_loss_pct=-0.025`,
  `failed_breakout_loss_pct=-0.004`, etc.).
- **Regime:** `amps_v6/regime/regime_engine.classify()` (pure, safe @3s) + `permissions()`;
  `amps_v4_regime.directional_score()` ([-100,100], SPY/QQQ rel-strength). **Macro inputs in
  `market_context.json` are DELAYED (yfinance EOD/overnight)** — never surfaced as real-time. SPY/QQQ/
  DIA/IWM/VIXY/TQQQ/SQQQ are **context-only** (never added to ENTRY universe; always in MONITORING).
- **Tests:** no `pytest.ini`/`conftest`/`tests/__init__.py`. Flat `test_<module>.py`, module-level
  `def test_*`, built-in fixtures only, async via `asyncio.run(go())` (no pytest-asyncio). No-order
  isolation = `object.__new__(cls)` + injected `_FakeBroker`. `import trade` requires a valid repo-root
  `universe.txt` at import (fail-closed). New test files add `sys.path.insert(0, parents[1])`;
  `register_equity_sectors` global state must be reset in `try/finally`.

---

## 4. Architecture — new package `market_scan/`

In-process **daemon thread** running its own asyncio event loop (mirrors `trade.py:7757`), started from
`ProductionOrderExecutor.run()` startup block with a reference to the executor (so it reuses
`executor.broker`, inherits paper/live, and can read positions/pending for the monitoring union).

```
market_scan/
  __init__.py              public API: start_market_scanner(executor) / stop / health snapshot
  config.py                MarketScanConfig dataclass + env parsing (Section-4 defaults); MARKET_SCAN_* flags
  thread_runner.py         daemon-thread lifecycle + single-instance lock (pid+mtime in scanner.lock)
  stream_client.py         the ONE IEX StockDataStream; connect, subscribe/unsubscribe, reconnect+backoff
  subscription_manager.py  dynamic allocation under cap; priority positions>pending>refs>candidates
  discovery.py             ~300s web discovery of US-listed symbols + validation (reuse universe.build_universe)
  ranking.py               features (OrderFlowSnapshot + directional_score + liquidity/VWAP/CVD) → descending rank
  seller_exhaustion.py     L1 primitives: microprice, absorption, bid-replenishment, volume-climax
  publisher.py             atomic universe.txt rewrite (membership-change-gated) + live_context.json writer
  health.py                live_context.json schema + resilient load_dynamic_universe() + health states
  replacement.py           −0.005 monitor + ranked replacement candidate selection + gate reuse
  # runtime (gitignored): live_context_<mode>.json, scanner.lock
```

**Modified files:**
- `universe.py` — add `write_canonical_universe(symbols, path=UNIVERSE_TXT)` atomic writer (temp +
  `os.replace`, never empty, normalized/deduped, descending-order-preserving).
- `order_flow_engine.py` — add the four L1 primitives to `OrderFlowSnapshot` + `SymbolOrderFlow`
  (additive, backward-compatible).
- `trade.py` — (a) `build_default_config` (7508): parse `MARKET_SCAN_*` flags into config; (b) `run()`
  startup: `start_market_scanner(self)` if enabled, stop in `finally`; (c) `_universe()` else-branch
  (3232): health-gated hot-reload (see §5); (d) `_augment_exit_orders_with_exit_learning` (5539): −0.5%
  monitor hook + replacement (see §7); (e) `_print_heartbeat` (7310) + startup banner: scanner panels.
- `.gitignore` — `market_scan/live_context*.json`, `market_scan/scanner.lock`.

**Tests:** `tests/test_market_scan_config.py`, `_publisher.py`, `_health.py`, `_ranking.py`,
`_seller_exhaustion.py`, `_subscription_manager.py`, `_thread_runner.py`, `_replacement.py`,
`_trade_integration.py` (hot-reload + fail-closed + monitoring-after-removal).

---

## 5. Hot-reload + operational states (the `_universe()` seam)

`trade.py:_universe()` else-branch (currently `return DEFAULT_SCAN_UNIVERSE`) becomes a **health-gated
dynamic read** that honors the three operational states. Explicit operator `config.universe` and the
opt-in LLM/SP500 paths retain priority above it (they are explicit, not "silent static fallback").

```
entry_universe =
  DISABLED (MARKET_SCAN_ENABLED=0)                     -> []   (block entries; NO static fallback)
  ENABLED + unhealthy                                  -> []   (block entries)
  ENABLED + healthy                                    -> load_dynamic_universe()  (validated, ranked)
```

`health.load_dynamic_universe()` returns `[]` (→ no new entries) unless **all** hold:
file present & non-empty & parseable; `age <= max_age_seconds` (default 15s); `live_context.json` present,
consistent with the file, and `scanner_status == healthy`; market-data freshness within tolerance;
session valid for the strategy. A recently-mtime'd file alone is **insufficient** — metadata must
corroborate. Monitoring, exits, reconciliation, risk are **never** gated by this function.

Per-entry pre-submit verification (§2.2) is enforced where entries are built, not only at `_universe()`:
broker tradability + strategy eligibility + execution approval already exist; scanner-health/freshness/
membership checks are added so a stale universe cannot authorize an entry even within a cycle.

---

## 6. Ranking model (descending rank written to universe.txt)

Per-symbol score blends (reusing existing primitives wherever present):
- Relative strength vs SPY & QQQ (`amps_v4_regime.directional_score`), sector-relative where available.
- Intraday momentum; historical avg dollar volume; intraday volume expansion; time-of-day relative
  volume where data suffices.
- Bid/ask spread; quoted liquidity; trade frequency; session VWAP; estimated CVD; buy/sell pressure;
  **seller exhaustion** (CVD-divergence proxy + new L1 primitives); market-regime alignment
  (`regime_engine.classify` + `permissions`).
- Quote-based trade classification (Lee-Ready) where contemporaneous quotes exist; tick-rule fallback.

**Data-honesty rules:** distinguish historical ADV from current-session volume; account for IEX being a
single exchange; track `source` + timestamp per input; never present delayed index/VIX as real-time;
newly emerging instruments eligible throughout the session.

Context-only proxies (SPY/QQQ/DIA/IWM/VIXY/TQQQ/SQQQ) are **never** added to the ENTRY universe.

---

## 7. Monitor-based loss removal + replacement (§2.1 realized)

Hook point: `_augment_exit_orders_with_exit_learning` (`trade.py:5539`) already iterates open positions
with live `unrealized_plpc` and appends sell rows to the exit frame — the natural, un-vetoable seam.

Flow per cycle:
1. For each open position (`broker.get_all_open_positions`), if `unrealized_plpc <= -0.005` → emit a
   `side='sell'` exit row (routed sell-first; duplicate-guarded by `has_open_order_for_symbol` +
   `_order_lock`). This runs regardless of scanner health.
2. After the exit **fills** (confirmed via reconciliation), reconcile buying power, load the **validated**
   universe.txt (replacements require scanner healthy; else hold cash), rank candidates, and pick the
   highest-ranked symbol satisfying all §2.1 eligibility gates (seller exhaustion, momentum, DI+>DI−,
   >BB-mid, >VWAP, liquidity/spread, regime alignment, risk controls), excluding held/pending and
   no-reentry symbols.
3. Emit the replacement as `side='buy'` into the rebalance frame → passes `_build_order_plan`
   (sell_first, min_notional, max_order_notional, gross_room, cooldown, no-reentry, kill-switch, MTF
   gate) unchanged. No competing optimizer; reuse `portfolio_optimizer.build_optimizer_scores` for the
   candidate ranking.
4. No eligible candidate → cash. Never force a replacement.

**Independence:** exits (step 1) never depend on scanner/universe; replacements (steps 2–4) require
scanner healthy + fresh universe. Designed as a drop-in **secondary** backstop once the broker-native
resting stop merges from `amps-v6-loss-removal-optimizer`.

---

## 8. Fault tolerance (fail-closed for ENTRIES only)

Any of: scanner disconnected/terminated, universe.txt missing/empty/malformed/stale, inconsistent health
metadata, unavailable market references, auth failure, subscription-limit exceeded, market closed for an
RTH-only strategy → **block new entries and replacements**; **never** disable monitoring, stops,
trailing, profit-exits, reconciliation, or risk. **No silent fallback to a hardcoded entry universe.**
Single-instance lock prevents a second IEX websocket / double universe.txt writer. `STREAMING_ENGINE`
must stay `0` (unchanged) so no conflicting socket opens.

---

## 9. Cross-branch integration points & expected merge conflicts

Target reconciliation with `amps-v6-loss-removal-optimizer`:
- **`trade.py:_augment_exit_orders_with_exit_learning` (5539):** both branches add exit logic here →
  **likely conflict**. This branch adds the monitor-based −0.005 exit; the other adds broker-native stop
  placement/verification. Reconcile so the broker-native stop is primary and this monitor is the backstop.
- **`AlpacaBroker` (2557):** other branch adds a stop/stop-limit method; this branch does **not** touch
  `AlpacaBroker` for stops → low conflict, additive.
- **Canonical Total-P/L ledger:** this branch triggers off `unrealized_plpc` (intraday). When the ledger
  lands, swap the trigger source to canonical Total P/L; keep `unrealized_plpc` path as fallback.
- **`universe.py` atomic writer / `order_flow_engine.py` L1 primitives:** additive; expected to merge
  cleanly and be shared by both branches.

---

## 10. Testing & validation plan

Non-ordering throughout (`object.__new__` + `_FakeBroker`). Coverage: scanner startup/shutdown,
duplicate-instance prevention, universe hot-reload, atomic publication round-trip, stale/malformed/empty
rejection, subscription-limit handling, monitoring-after-removal, reconnect, session transitions,
bullish/inverse ETF eligibility, replacement eligibility, duplicate-order prevention, operational-state
transitions (healthy/unhealthy/disabled), −0.005 exit independence from scanner health, and regression of
existing trading modes. Plus: Python syntax validation, full existing suite (no regressions), and
**authenticated read-only IEX connectivity** via existing creds (no orders).

---

## 11. Production startup (unchanged)

```
./start.sh            # == .venv/bin/python -u trade.py
# or: python trade.py
```
Zero new arguments. `MARKET_SCAN_ENABLED` defaults ON. Kill switch: `MARKET_SCAN_ENABLED=0`.

---

## 12. Known limitations / open items

- Trigger uses intraday `unrealized_plpc`, not a canonical Total-P/L ledger (deferred).
- Broker-native resting stop deferred → monitor is currently the sole −0.5% mechanism on this branch.
- Actual IEX subscription cap is account-dependent; default conservative + dynamic allocation; confirmed
  during read-only connectivity test.
- `portfolio_optimizer` has no true covariance model → correlation/leverage guard is a sector+leverage
  proxy via `universe.exposure_groups()`.
- Scanner and engine share one process; a scanner crash is isolated by try/except + watchdog but does not
  get OS-level process isolation.
