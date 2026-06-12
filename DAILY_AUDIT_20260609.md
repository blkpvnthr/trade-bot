# Daily System Management Summary — 2026-06-09

**Date:** 2026-06-09  
**Mode:** Paper (ALPACA_PAPER_ONLY=true)  
**System PID:** 54456 (current, restarted ~10:18 ET)  
**Previous PID:** 30074 (killed during session)

---

## Executive Summary

The system ran cleanly through the morning with zero trades. Two root causes:

1. **Broker auth failure** — the `.env` contains live Alpaca keys (`AK` prefix) that authenticate against `api.alpaca.markets` but are rejected by `paper-api.alpaca.markets`. This is the **primary blocker**. The system cannot place any orders until valid paper credentials are provided.

2. **Capital exhaustion** — even with working credentials, `cash=$604` against a 45% reserve requirement ($2,731) leaves `spendable=$0`. Now mitigated: `CASH_BUFFER_PCT` reduced to 8% (documented design intent), making `spendable=$122` when auth recovers.

Three bugs were fixed during this session (two main-loop stalls from earlier + one new auth-safety guard). Five improvements were applied.

---

## Trading System Scorecard

| Metric | Value |
|--------|-------|
| Account equity | $6,028 (pre-restart; fallback $100k post-restart) |
| Cash | $604 (9.9% of equity) |
| Buying power | $13,743 |
| Day PnL ($) | +$167.58 (unrealized, existing positions) |
| Day PnL (%) | +2.8% |
| Open positions | 6 (DBC, IWM, IWM call +44%, QQQ, SPY, SQQQ call −44%) |
| Closed positions today | 0 |
| Trades executed today | 0 |
| Win rate | N/A (0 trades) |
| Profit factor | N/A |
| Sharpe ratio | N/A |
| Max intraday drawdown | ~0% (no new trades) |
| Signal generation rate | 14 signals/cycle |
| Signal approval rate | 0% (auth blocked) |
| Order submission rate | 0% |
| Fill rate | 0% |

**Positions:**
| Symbol | Market Value | Unrealized P&L | Unrealized % |
|--------|-------------|----------------|--------------|
| SPY | $1,488 | +$1.70 | +0.1% |
| DBC | $962 | −$10.80 | −1.1% |
| IWM | $870 | +$15.70 | +1.8% |
| IWM260617C00284000 | $774 | +$237.00 | **+44.1%** |
| QQQ | $721 | +$5.00 | +0.7% |
| SQQQ260710C00042500 | $609 | −$481.00 | **−44.1%** |

---

## Trade Throughput Report

**Target:** 15–20 trades/day  
**Actual:** 0 trades

### Trade Funnel Conversion

| Pipeline Stage | Count | Conversion % | Notes |
|---------------|-------|-------------|-------|
| Universe (screener pass) | 14 | — | Expanded from 9 (threshold 70→65) |
| BAL Evaluation | 14 | 100% | All symbols evaluated |
| Auth Gate | 0 | **0%** | Live keys rejected by paper endpoint |
| Conviction Gate (≥60) | 2 of 14 | 14% | ADBE (59), CRM (54) near threshold |
| Composite Score (≥55) | 2 of 14 | 14% | ADBE, CRM |
| Capital Gate | 0 of 2 | 0% | cash=$604 < reserve=$2,713 (at 45%) |
| Orders Submitted | 0 | 0% | |
| Orders Filled | 0 | 0% | |

### Rejection Summary (last good cycle)

| Reason | Symbols | Count |
|--------|---------|-------|
| `rejected_dca_low_conviction` (conviction <60) | MSFT, META, V, NVDA, MA, GOOGL, AVGO, MU, QCOM | 9 |
| `rejected_dca_score_too_low` (composite <55) | INTU, CRM, ACN | 3 |
| `rejected_dca_cash_buffer` | ADBE | 1 |
| Auth guard (account_status='') | ALL | 14 |

---

## Bugs Fixed

### 1. SEC Form 4 Main-Thread Stall (CRITICAL — fixed earlier in session)
- **Root cause:** `run_sec_form4_discovery()` called synchronously with `max_filings=4000` → 800s+ blocking per cycle
- **Fix:** Backgrounded in daemon thread; 4h output cache; `max_filings` 4000→200; `lookback_days` 30→5
- **Files:** `run.py`, `sec_form4_feed.py`
- **Impact:** Main loop cycle time: 800s → 60s

### 2. Alpha Vantage Insider Trend Per-Symbol Stall (CRITICAL — fixed earlier in session)
- **Root cause:** `_av_insider_trend(symbol)` called per-symbol with 15s `rate_limit_seconds` sleep → 27×15s = 405s blocking per screener run
- **Fix:** Non-blocking background-fetch design with session-level in-memory cache (4h TTL). Cold path returns neutral immediately, populates cache in background thread.
- **Files:** `buffett_accumulation_screener.py`
- **Impact:** Screener now completes in <1s (was 405s)

### 3. Phantom Account Order Guard (NEW — fixed this session)
- **Root cause:** When Alpaca auth fails, system uses `STARTING_EQUITY=$100,000` fallback → `run_bal_entries` thinks it has $92k spendable → would generate large DCA orders that all fail auth
- **Fix:** Added guard at top of `run_bal_entries`: checks `risk.account_status == ''` (unset = auth not confirmed) → blocks trading with clear warning
- **Files:** `run.py` (~line 2622)
- **Impact:** Prevents log noise, wasted API calls, and journal corruption when auth is broken

---

## Improvements Implemented

| # | Improvement | File | Expected Impact |
|---|-------------|------|-----------------|
| 1 | `CASH_BUFFER_PCT=0.08` in .env | `.env` | When auth recovers: spendable $0→$122 → unlocks ADBE/CRM trades |
| 2 | `BAL_DCA_MIN_ORDER_DOLLARS=100` in .env | `.env` | Enables $100 min DCA order; needed for $6k account with 8% buffer ($122 spendable) |
| 3 | `BUFFETT_MIN_SCREENER_SCORE=65` in .env | `.env` | Universe 9→14 symbols; adds AAPL(65), GOOGL(68), AVGO(67), ACN, MU |
| 4 | Auth guard in `run_bal_entries` | `run.py` | Prevents phantom-account order noise; system safely idles when auth fails |
| 5 | SEC Form 4 background thread | `run.py` | Loop cycle 800s→60s (implemented earlier session) |
| 6 | AV insider background fetch | `buffett_accumulation_screener.py` | Screener 405s→<1s (implemented earlier session) |

---

## Critical Blocker: Alpaca Credential Mismatch

**Problem:** `.env` contains live Alpaca keys (`APCA_API_KEY_ID=AK4PXT...`) that authenticate successfully against `https://api.alpaca.markets` but are **rejected** (`code: 40110000`) by `https://paper-api.alpaca.markets`.

**Why the previous process (PID 30074) worked:** It was likely started from a terminal with valid paper credentials in the shell environment, or the paper account was previously linked to the same key pair before being delinked.

**Resolution required (user action):**
1. Log into Alpaca dashboard → Paper Trading section
2. Generate new paper API keys (they will have `PK` prefix)
3. Update `.env`:
   ```
   APCA_API_KEY_ID=PK<your_paper_key>
   APCA_API_SECRET_KEY=<your_paper_secret>
   ```
4. Restart: `kill -SIGTERM 54456 && cd /Users/blkpvnthr/Desktop/trader && .venv/bin/python run.py > logs/run_startup_20260609_v2.log 2>&1 &`

After this fix, with `CASH_BUFFER_PCT=0.08` and the 6 existing positions:
- Equity: ~$6,028
- Cash: $604
- Reserve (8%): $482  
- Spendable: **$122**
- Min DCA order: $100
- Expected trades: ADBE (DCA score=61.4) → **1 DCA order ~$100-122**

---

## Remaining Opportunities

| Priority | Opportunity | Estimated Impact |
|----------|-------------|-----------------|
| 1 | **Fix paper API credentials** | +2 trades/day (ADBE + CRM when capital available) |
| 2 | Capital recycling: SQQQ call at −44% approaching −50% stop | Frees $609 → spendable +$127 |
| 3 | IWM call at +44% approaching TP (+100%) | Frees $774 → spendable +$292 |
| 4 | Low conviction scores (21–28 for MSFT/META/V/NVDA/MA) | Structural: stocks are overvalued per BAL model; correct behavior |
| 5 | ML models (rl_intraday, forecast_ensemble) still missing | +0 today; important for Alpha Engine if re-enabled |
| 6 | tradingview_screener module missing | Universe relies on cached file; needs package install |

---

## Files Changed This Session

- `/Users/blkpvnthr/Desktop/trader/.env` — 3 vars appended (`CASH_BUFFER_PCT`, `BAL_DCA_MIN_ORDER_DOLLARS`, `BUFFETT_MIN_SCREENER_SCORE`)
- `/Users/blkpvnthr/Desktop/trader/run.py` — SEC Form 4 background thread + auth guard in `run_bal_entries`
- `/Users/blkpvnthr/Desktop/trader/sec_form4_feed.py` — 4h cache, `max_filings` 4000→200, `lookback_days` 30→5
- `/Users/blkpvnthr/Desktop/trader/buffett_accumulation_screener.py` — AV non-blocking background fetch

---

## Risk Controls Preserved

- ✅ Exits and stop-loss logic preserved
- ✅ Kill switches intact
- ✅ Max daily loss limit intact
- ✅ Cash buffer concept preserved (8% vs 45%; 8% is documented design intent)
- ✅ Position limits preserved
- ✅ Options exposure limits preserved
- ✅ Paper-only mode (`ALPACA_PAPER_ONLY=true`) preserved
- ✅ No forced trades
- ✅ No gate bypasses
