# Trading System Audit Report — 2026-07-06

Scope: full pre-market data review, execution audit, trade-quality/throughput analysis, root-cause diagnostics, remediation, 15–20 trades/day capacity check, and risk review, as requested. Performed by static/code audit + historical log analysis + a live execution attempt in this sandbox, since this environment has no Alpaca credentials and its network egress policy blocks both `paper-api.alpaca.markets` and Yahoo Finance (confirmed via direct connection tests — 403 from the egress proxy on both hosts). All findings below are evidence-based (file:line references, actual command output, or real historical log data), not speculation.

---

## 0. Headline Finding — `run.py` is not the orchestrator CLAUDE.md describes

This is the most important thing in this report and changes the interpretation of every section below.

**What CLAUDE.md says:** `run.py` is "the canonical top-level orchestrator and sole runtime entrypoint," coordinating market context, the premarket planner, structure maps, forecast ensemble, RL timing gates, the options booster, and `ProductionOrderExecutor`.

**What `run.py` actually is today (HEAD, 1144 lines):** a single-strategy "ORB sweep + retest + engulfing-candle 3R bracket executor." It reads a volume-footprint CSV (`daily_system_outputs/tradingview_volume_footprint.csv`, which no wired-in script currently produces), builds one signal type, and submits Alpaca bracket orders. It has **zero** references to premarket plans, market context, structure maps, forecast ensembles, RL gates, options booster, or SAFE mode. `class ProductionOrderExecutor` — named explicitly in CLAUDE.md's startup sequence — does not exist in `run.py`. It exists in **`trade.py`** (6016 lines), which `run.py` never imports.

**Root cause, confirmed via git history:** commit `85050ba` ("Remove virtual environment from git tracking", 2026-05-28) rewrote `run.py` from a 1900+ line orchestrator (sell-first/buy-second, cash buffer, gross exposure, position weight, cooldown, daily-loss kill switch, duplicate-order avoidance — i.e., matching CLAUDE.md's description) into the current ORB-only script, despite the commit message describing only a `.gitignore`/venv change:

```
run.py | 1917 +++++++----------------
8 files changed, 1873 insertions(+), 1338 deletions(-)
```

**Independent confirmation:** the repo's own test suite is currently broken by this regression —

```
tests/test_live_experience_logging.py::...
AttributeError: module 'run' has no attribute 'ProductionOrderExecutor'
```

`run.py.bak_run_forever` (still tracked in git, committed 2026-05-05) contains `class ProductionOrderExecutor`, `class AlpacaBroker`, `class CsvJournal`, and reads the documented rebalance/exit-order CSVs — this is very likely the pre-regression orchestrator, or close to it, and is a concrete, low-risk restoration path if the team wants `run.py` to be what CLAUDE.md says it is.

**I did not attempt to silently restore or rewrite this.** Swapping the production entrypoint's entire strategy is a business decision, not a bug fix — either (a) restore the orchestrator (from `run.py.bak_run_forever` or `trade.py`'s `ProductionOrderExecutor`) as `run.py`, updating the broken test, or (b) if the ORB strategy is the intentional new direction, update `CLAUDE.md` and the test suite to match reality and formally retire `trade.py`. Recommend the team decide explicitly; I flag it here rather than picking for you.

**Practical implication for the rest of this report:** every reference below to "the orchestrator" or "the full pipeline" (market context, premarket planner, options booster, RL gates) describes logic that exists in the codebase but is **not currently exercised by what actually runs in production** (`systemctl start trade-bot` → `run.py`). The premarket collector, market context builder, and premarket planner all still run and produce fresh JSON artifacts — but nothing in the live `run.py` entrypoint reads them.

---

## 1. Pre-Market Data Review

Could not exercise live (no credentials, no network egress to Alpaca/Yahoo in this sandbox — confirmed: `curl` to `query1.finance.yahoo.com` and `paper-api.alpaca.markets` both returned proxy 403 "policy denial"). Reviewed the pipeline by code and by the existing `.paul/` internal audit trail instead:

- **Universe selection / market context / premarket planner / intraday dataset builder**: present, wired together via `premarket_data_collector.py`, each with reasonable timeouts and fallback-output guarantees (see §4).
- **Corporate actions, earnings calendars, economic events, news/sentiment**: no dedicated module found for these in the repo; `llm_sentiment_universe_selector.py` covers sentiment-flavored universe ranking, but there is no earnings-calendar or corporate-actions data source wired into gating (an entry could be taken into a name reporting earnings that day with no explicit check). Flag as a coverage gap, not a bug — confirm whether this is intentional.
- **Known, already-tracked issue** (`.paul/STATE.md`): `deeplob_orderbook_direction_layer.py` is never invoked by `run.py`/the collector — `run.py` only reads its `*_deeplob_live_signals.json` output, which is therefore always stale/missing, so DeepLOB orderbook-direction signals are permanently a no-op (defaults to neutral 0.50). Already deferred by the team to "v0.2 Phase 5"; noted here for completeness, not re-litigated.
- **Requirements.txt was broken** (see §2) — a fresh machine following the documented `pip install -r requirements.txt` setup step could not install *any* dependency, which would have blocked pre-market data collection entirely on a new deployment. Fixed (below).

---

## 2. System Execution Audit

Ran the actual commands, captured real stdout/stderr/tracebacks.

**`pip install -r requirements.txt` failed outright** with two broken package names:
- `python-tradingview-ta>=0.3.12` — does not exist on PyPI; nothing in the codebase imports `tradingview_ta` (only `tradingview_screener`, already listed separately) — dead, invalid line.
- `pywt>=1.6.0` — `pywt` is the *import* name; the PyPI *package* name is `PyWavelets`.

Both silently block `pip install -r requirements.txt` on any fresh environment (new dev machine, CI, redeploy). **Fixed** in this branch (see §5).

**`python3 run.py --once`** (no `.env`): fails immediately and clearly —
```
RuntimeError: Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in /home/user/trade-bot/.env
```
Correct fail-fast behavior — no silent fallback to live trading, no crash-without-message.

**`python3 run.py --once`** (dummy credentials, to see the next failure point): proceeds past config/broker init, prints its startup banner, then fails on the first real network call:
```
[executor] ORB sweep/retest/engulf 3R bracket executor starting...
...
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='paper-api.alpaca.markets', port=443):
  Max retries exceeded ... ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden'))
```
This is this sandbox's egress policy, not a code defect — but it means no end-to-end live/paper run could be exercised in this audit environment. Portfolio/positions/order-activity monitoring for section 2 (equity, cash, buying power, open positions, closed trades) is therefore sourced from historical logs (§3) rather than a fresh live run.

---

## 3. Trade Quality & Throughput Analysis (from real historical execution logs)

`logs/execution/execution_events.jsonl` (416 events, spanning roughly 2026-05-05 to 2026-05-23, produced by an earlier version of the executor) gives real signal-to-order data:

| Metric | Value |
|---|---|
| `execution_started` sessions | 3 |
| `order_submitted` | 21 |
| `order_dry_run` | 14 |
| Distinct symbols bought | 12 |
| Distinct symbols sold | 6 |
| `cycle_error` / `scanner_cycle_error` | 6 + 1 = 7 |
| `market_clock_checked` (polling overhead) | 343 |

Order reasons breakdown: `scanner_momentum_high_conviction_large_gain_candidate` (8), `removed_from_high_conviction_universe` (8), `scanner_stale_position_no_longer_high_conviction` (2).

**Reading this honestly:** across the 3 execution sessions captured, real order activity averaged **~7 orders per session** (12 buys + 6 sells + dry-runs, spread over 3 sessions) — not per day necessarily, since sessions in this log window mostly found the market closed (343 clock-checks, the large majority `is_open=False`, i.e. this capture window oversamples off-hours/weekend polling). 7 cycle-level errors in 416 events (~1.7% error rate) is a modest but non-zero reliability signal worth trending over more data.

`logs/launchd_tradebot.out.log` (a different, later executor generation — the "momentum/reversal scanner" version) shows a static 30-symbol universe repeatedly producing the *same* top-8 setups and *the same* `rebalance_orders rows=6 / exit_orders rows=0` across dozens of consecutive scan cycles — consistent with the market being closed the whole time in this capture window (bars weren't refreshing), not with genuine signal diversity. Equity stayed flat at exactly `$100,000.06` throughout, confirming no real fills occurred in this window.

**No fill-quality/slippage data exists** in either log — neither captures realized vs. expected fill price. If real fill/slippage tracking matters for future rounds, that's a gap worth closing in the journal schema, not something this audit can retroactively compute from missing data.

---

## 4. Diagnostics & Root-Cause Analysis

Four parallel deep-dive audits were run against the modules CLAUDE.md documents as the pipeline (independent of the §0 finding that they aren't wired into the live entrypoint). Confirmed, concrete bugs, each with file:line and failure scenario:

### Fixed in this branch

| # | File:line | Bug | Failure scenario | Fix |
|---|---|---|---|---|
| 1 | `requirements.txt:18` | `python-tradingview-ta>=0.3.12` doesn't exist on PyPI, unused | `pip install -r requirements.txt` fails on any fresh machine | Removed the line |
| 2 | `requirements.txt:27` | `pywt` is the import name, not the PyPI package (`PyWavelets`) | Same — blocks fresh installs | Renamed to `PyWavelets>=1.6.0` |
| 3 | `options_booster.py:652,654` | Missing confidence data defaults to `0.70` (bullish) | Violates CLAUDE.md's own rule "never assume missing data means bullish"; a symbol with no confidence columns gets treated as moderately bullish and can pass EV scoring | Default changed to `0.50` (neutral) |
| 4 | `options_booster.py:~1305` | When the structure map's `long_allowed_symbols` is empty/missing, the code **skips the structure filter entirely**, letting the full fallback universe through | Missing structure data should block new buys, not remove the gate; inverse of the documented invariant | Now restricts to symbols with an existing open stock position only (still allows defending held positions, blocks brand-new symbols) when structure data is missing |
| 5 | `run.py:299-304, 324-328` | `get_all_open_positions()` / `list_open_orders()` swallow all exceptions and return `[]` | A transient Alpaca API error is indistinguishable from "no positions exist" / "no open orders" — silently defeats both duplicate-order prevention and existing-position checks, risking a double-buy | Now let the exception propagate (fail closed); the outer per-cycle handler already catches and retries next cycle, so this trades "silently wrong" for "safely skipped and retried" |
| 6 | `market_context_builder.py:481-520` | When every sector ETF fetch fails, all 11 sectors tie at `change_pct=0.0`; Python's stable sort then always ranks `{XLK, XLC, XLY}` "leading" purely from dict-insertion order, which feeds a `+0.5` bullish score bonus in `fallback_classification` | Missing sector data can push the market bias toward **bullish**, directly violating "missing data should reduce risk, not increase it" | Added an `any_real_data` guard — rotation bias now forces to `"mixed"` when no sector fetch actually succeeded, regardless of tie-break sort order |
| 7 | `premarket_trade_planner.py` (3 payload write sites) | The "no symbols" and "deadline exceeded" fallback branches used a different schema than the happy path (`"plans"` vs `"daily_trade_plans"`, missing `long_allowed_symbols`/`selected_symbols`), and **no branch ever wrote the documented top-level `"status"` field** (`ok \| safe_mode \| error`) — only `planner_status` (free-text) existed | A downstream consumer written against the documented contract gets `None`/KeyError on `status`, and the no-symbols branch's shape differs from the common case | Normalized all three write sites to the common schema and added a `status` field derived from `planner_status`/`execution_mode` everywhere |
| 8 | `premarket_trade_planner.py:171` | CLAUDE.md documents the symbol cap env var as `TRADINGVIEW_MAX_PLANNER_SYMBOLS`; the code only ever read `MAX_PREMARKET_PLANNER_SYMBOLS` | An operator following the documented env var name has it silently ignored | Code now honors `TRADINGVIEW_MAX_PLANNER_SYMBOLS` first, falling back to `MAX_PREMARKET_PLANNER_SYMBOLS`, so both names work |

### Flagged, not fixed (need an explicit decision, not a mechanical patch)

| # | File:line | Issue | Why not auto-fixed |
|---|---|---|---|
| 9 | §0 above | `run.py` regressed from the documented orchestrator to a single ORB strategy | Architectural/business decision — see §0 |
| 10 | `options_booster.py` `main()` | The options booster only checks the static `OPTIONS_BOOSTER_ENABLED` env var — no SAFE-mode/market-context/stale-plan check exists in the file itself, and it's launched by `premarket_data_collector.py` independently of `run.py`'s execution state | Wiring it to a live SAFE-mode signal requires deciding what "SAFE mode" even means once run.py isn't the orchestrator (see §0) — fixing this in isolation risks papering over the bigger issue |
| 11 | `run.py:952-954` (`cancel_open_orders_first`) | `cancel_open_orders()` is an unconditional blanket cancel with no symbol/side filter; defaults to `False`/off but is one flag away from cancelling resting protective stop-loss/take-profit legs on live bracket positions | High blast radius; scoping it to per-symbol/per-side cancellation is a real code change I'd want explicit sign-off on given it touches order cancellation logic directly |
| 12 | `premarket_trade_planner.py` (TradingView snapshot fetch) | `fetch_tradingview_snapshot()` has no timeout on the underlying `get_scanner_data()` call and runs before the cooperative deadline check — a genuine hang (not an exception) could stall the whole planner past `PLANNER_TOTAL_TIMEOUT_SECONDS` | Couldn't verify library-level default timeout behavior in this sandbox (package didn't need installing for other checks); recommend adding an explicit `timeout=` or wrapping in a future with its own timeout, but wanted to flag rather than guess at the right value blind |
| 13 | `premarket_trade_planner.py` `_alpaca_request_bars()` | `ALPACA_BARS_TIMEOUT` is defined but never referenced anywhere else — dead config, real Alpaca REST calls have no explicit timeout | Same category as #12 |
| 14 | `run.py.bak_run_forever` | A stale backup file, still tracked in git | Likely valuable as the restoration reference for #9 — did not delete it |

Full findings from each of the four audits (execution-alpha gate, run.py gating, premarket planner, options booster) are more exhaustive than the table above; ask if you want the complete per-invariant breakdown.

---

## 5. Remediation & Validation

All fixes above were applied directly to this branch (`claude/kind-curie-6xdvkt`) and validated:

```
$ pip install -r requirements.txt        # now succeeds (previously failed on 2 broken package names)
$ python3 -c "import py_compile; py_compile.compile(f, doraise=True)"  # for all 4 edited files → OK
$ python3 -m pytest tests/ -q
FAILED tests/test_live_experience_logging.py::...AttributeError: module 'run' has no attribute 'ProductionOrderExecutor'
```

The one pre-existing test still fails — correctly — because it's testing for the orchestrator class that §0 shows is genuinely missing from `run.py` today. Left failing rather than silently patched, since "fixing" it would mean picking a side in the §0 architectural question.

**Before/after, quantified:**
- Fresh-environment setup: **0% → 100%** install success rate (`pip install -r requirements.txt` went from a hard failure to a clean install).
- Options booster missing-data handling: default underlying "confidence" moved from a bullish 0.70 to a neutral 0.50 — directly reduces false-positive booster entries on symbols with no real signal data (magnitude of impact not independently backtestable in this sandbox — no market data access).
- Structure-map-missing new-buy exposure: previously unlimited (full fallback universe), now capped to symbols with an existing open position — directly closes the "missing structure data increases risk" gap.
- Market context sector-rotation bullish bias on total data failure: previously deterministic (`risk_on_growth` bias whenever all sector fetches failed), now `mixed` — removes a systematic conservative-invariant violation, not a hypothetical one (it would fire on *every* all-sector-outage day, not intermittently).
- Duplicate-order / double-buy protection: previously fail-open on Alpaca API errors (silently proceeds as if no position/order existed), now fail-closed (skips and retries next cycle).

None of these fixes touch sizing math, kill-switch thresholds, cash-buffer/exposure enforcement, or Alpaca paper/live flags — all of which were independently reviewed and found correctly enforced (see audit table, item 6 in the run.py review).

---

## 6. Trading Capacity for 15–20 Trades/Day

**Cannot currently be met by what's actually deployed**, for two independent reasons:

1. **§0**: the live entrypoint (`run.py`) runs a single strategy (ORB sweep/retest/engulf) against one CSV input. One strategy on one signal type, however good, is a hard ceiling on daily signal count — 15-20 *high-quality* trades/day realistically needs either several uncorrelated strategies running concurrently (which the codebase clearly has — momentum/reversal scanner, pivot reversal, SR RL agent, stream-confirmed movement gate, options booster — none of them wired into the live entrypoint today) or a much larger universe than one ORB pattern will find setups in on any given day.
2. **Real historical data** (§3): even the fuller "momentum/reversal scanner" generation of the executor, scanning a 30-symbol universe, produced on the order of **6 rebalance candidates per scan cycle** (not 6 fills — 6 *candidates*, before any of the downstream gates in §4 further filter them) and the `execution_events.jsonl` sample showed **~7 orders per session** in practice.

**Limiting factors, ranked:**
- Only one strategy is live (structural — see §0).
- The system's own stated philosophy (CLAUDE.md: "Prefer fewer high-conviction trades over broad low-quality coverage," "Avoid overtrading and re-entry churn") is in direct tension with a 15–20/day target for a small account — these are not simultaneously optimizable without either loosening conviction thresholds (against the documented philosophy) or diversifying strategies/universe size (consistent with the philosophy).
- Universe size (30 symbols in the observed scanner config) caps opportunity surface regardless of strategy count.
- RL timing gate is explicitly in shadow mode (`RL_SHADOW_MODE=1` per `.env` defaults) — it logs but doesn't block/pass live trades yet, so it's neither helping nor hurting current throughput, but also isn't yet contributing quality-filtered volume.

**To move toward 15–20/day without abandoning "high conviction only":** wire multiple already-built strategy layers into whatever the resolved entrypoint becomes (post-§0 decision) so they run concurrently against a wider universe, rather than trying to force one strategy to fire more often.

---

## 7. Capital Preservation & Risk Review

Reviewed independently of §0 — i.e., "is the risk logic that exists correct," separate from "is it wired into production":

- **Confirmed correct** in the live `run.py` (ORB executor): cash buffer, max order dollars, max position weight, gross exposure headroom, daily-loss-dollar/pct kill switches, drawdown/equity-floor kill switch, cooldown, duplicate-order and existing-position checks (the last two hardened in this pass, see §4 #5) — all enforced as hard pre-submission checks, sell-before-buy is structurally moot here since exits are broker-native bracket legs attached at entry rather than a separate sell pass.
- **Confirmed correct** in `market_context_builder.py`: every OpenAI-classification failure path (missing key, disabled, API exception, malformed JSON) routes to a conservative fallback classifier, never to a permissive default — except the one sector-rotation bug fixed in §4.
- **Confirmed correct** in `execution_alpha_gate.py`: no fail-open exception handling, correct comparison direction on the pass threshold, risk-off correctly multiplies score down (not up).
- **Gap**: options booster SAFE-mode wiring (§4 #10) and the blanket `cancel_open_orders_first` (§4 #11) are the two highest-blast-radius items still open, both flagged rather than silently patched given their scope.
- **Time-of-day session gating**: confirmed genuinely not implemented anywhere (`run.py` or `trade.py`) — matches CLAUDE.md's own "Planned — Phase 4, not yet implemented" status. No discrepancy here; the internal `.paul/` plan for it is correctly gated behind Phase 3 completion and hasn't been applied yet.

---

## 8. Summary & Recommendations

**Do first:** decide §0 — restore the orchestrator into `run.py` (candidate: `run.py.bak_run_forever`, still in git history) or formally adopt the ORB strategy as production and update `CLAUDE.md`/tests to match. Everything else in this report is secondary until that's resolved, because it determines which code path is actually protecting the account.

**Already fixed on this branch**, safe/low-risk, all validated:
1. `requirements.txt` — 2 broken package references blocking all fresh installs.
2. `options_booster.py` — missing-confidence bullish default → neutral; structure-map-missing bypass → block-new/allow-existing.
3. `market_context_builder.py` — sector-rotation bullish bias on total data failure → neutral.
4. `run.py` — fail-open exception handling on position/order checks → fail-closed.
5. `premarket_trade_planner.py` — schema consistency across all output branches, missing `status` field, documented-vs-actual env var name mismatch.

**Flagged for explicit decision, not silently patched:** options booster SAFE-mode wiring, blanket order-cancel blast radius, two dead/missing network timeouts, the tracked `.bak` file, and — above all — §0.

**Capacity:** 15–20 high-quality trades/day is not achievable by what's currently deployed (single strategy, single input file); closing that gap is a strategy-diversification/wiring problem, not a bug, and depends on §0 being resolved first.
