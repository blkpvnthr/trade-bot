# Daily-Green 2:1 Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trade.py` enter a long equity position as soon as a symbol is green versus today's open, exit at a 2:1 reward:risk target, and exit immediately if the day turns negative, replacing the momentum/setup-score signal.

**Architecture:** Add pure decision/row-builder functions (no I/O) at module level in `trade.py`, unit-tested directly. Wire them into the scanner via a new `_build_daily_green_orders()` method that `build_orders()` branches to when `ScannerConfig.daily_green_strategy_enabled` is True. Add a dedicated branch in the executor so the daily-green orders go straight to execution, bypassing the momentum/ML/RL/context gates. Reuse the existing sizing, kill switch, order plan, execution, and journaling unchanged.

**Tech Stack:** Python 3, pandas, numpy, alpaca-py (already imported with graceful fallbacks), pytest 9.0.3.

## Global Constraints

- Target file for all production changes: `trade.py` (single file).
- ASCII only in code and files: use plain hyphens, never em-dashes or en-dashes.
- Equities only, long only. Do not touch the options path.
- Reference ("zero line") is TODAY'S SESSION OPEN, not prior close.
- Risk `R = entry_price - open_today`, floored by `daily_green_min_risk_pct * entry_price`.
- Target `= entry_price + daily_green_rr * R` (default rr = 2.0).
- Stop = exit when `current_price <= open_today` (day turns red).
- Reasons are exact strings: entry `daily_green_entry`; exits `day_turned_red`, `rr_2to1_target`, `eod_flatten`.
- Daily-green mode defaults ON; universe source in that mode is `universe.txt` (via `DEFAULT_SCAN_UNIVERSE`).
- Rebalance/exit row dicts must carry at least: `symbol`, `side`, `delta_shares`, `delta_dollars`, `price`, `reason` so `_normalize_order_frame` (trade.py:4326) and `_build_order_plan` (trade.py:4715) consume them.
- Tests must run with no network and no Alpaca credentials. Test the pure functions and row builders directly; construct scanner/executor objects with `object.__new__(...)` plus fakes when a method-level test is needed.
- Run tests with: `python -m pytest tests/test_daily_green_engine.py -v`
- Keep all legacy momentum code in place; it is only gated OFF, never deleted.

---

### Task 1: Config fields and production wiring

**Files:**
- Modify: `trade.py` (ScannerConfig dataclass, after line 326)
- Modify: `trade.py` (`build_default_config`, trade.py:5882-5921)
- Test: `tests/test_daily_green_engine.py`

**Interfaces:**
- Produces: `ScannerConfig` gains fields `daily_green_strategy_enabled: bool`, `daily_green_rr: float`, `daily_green_min_risk_pct: float`, `daily_green_per_position_weight: float`, `daily_green_max_positions: int`, `daily_green_flatten_before_close_min: int`, `daily_green_reenter_after_stop: bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_daily_green_engine.py`:

```python
import trade


def test_scanner_config_has_daily_green_defaults():
    cfg = trade.ScannerConfig()
    assert cfg.daily_green_strategy_enabled is True
    assert cfg.daily_green_rr == 2.0
    assert cfg.daily_green_min_risk_pct == 0.0015
    assert cfg.daily_green_per_position_weight == 0.09
    assert cfg.daily_green_max_positions == 8
    assert cfg.daily_green_flatten_before_close_min == 5
    assert cfg.daily_green_reenter_after_stop is False


def test_build_default_config_uses_universe_txt_in_daily_green():
    cfg = trade.build_default_config().scanner
    assert cfg.daily_green_strategy_enabled is True
    # universe.txt is reached only when all other universe sources are off
    assert cfg.use_llm_sentiment_universe is False
    assert cfg.use_sp500_fallback is False
    assert cfg.broad_market_scan is False
    assert cfg.universe is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: FAIL with `AttributeError: 'ScannerConfig' object has no attribute 'daily_green_strategy_enabled'`

- [ ] **Step 3: Add config fields**

In `trade.py`, immediately after line 326 (`    require_positive_momentum_or_breakout: bool = True`), add:

```python

    # Daily-green 2:1 strategy (equities, long only).
    # ENTER when price > today's open; STOP at today's open (day turns red);
    # TARGET at entry + daily_green_rr * risk, risk = entry - open floored by min_risk_pct.
    daily_green_strategy_enabled: bool = True
    daily_green_rr: float = 2.0
    daily_green_min_risk_pct: float = 0.0015
    daily_green_per_position_weight: float = 0.09
    daily_green_max_positions: int = 8
    daily_green_flatten_before_close_min: int = 5
    daily_green_reenter_after_stop: bool = False
```

- [ ] **Step 4: Wire production config to use universe.txt in daily-green mode**

In `trade.py` `build_default_config` (the `ScannerConfig(...)` call starting at line 5882), change these three kwargs and add the daily-green enable flag. Replace:

```python
        use_llm_sentiment_universe=True,
        refresh_llm_sentiment_universe=True,
        min_llm_universe_confidence=0.0,
```

with:

```python
        use_llm_sentiment_universe=False,
        refresh_llm_sentiment_universe=False,
        min_llm_universe_confidence=0.0,
```

Then replace:

```python
        broad_market_scan=False,
```

with:

```python
        broad_market_scan=False,
        daily_green_strategy_enabled=True,
```

(Leaving `universe=None`, `use_sp500_fallback=False`, and `broad_market_scan=False` makes `_universe()` fall through to `DEFAULT_SCAN_UNIVERSE`, which is `universe.txt`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add trade.py tests/test_daily_green_engine.py
git commit -m "feat(trade): add daily-green 2:1 config fields and production wiring"
```

---

### Task 2: Pure decision functions

**Files:**
- Modify: `trade.py` (module level, after `_atr` at trade.py:2536, before `class MomentumSetupScanner` at trade.py:2549)
- Test: `tests/test_daily_green_engine.py`

**Interfaces:**
- Produces:
  - `daily_green_entry_signal(open_today: float, current_price: float) -> bool`
  - `daily_green_risk_distance(entry_price: float, open_today: float, min_risk_pct: float) -> float`
  - `daily_green_target_price(entry_price: float, open_today: float, rr: float, min_risk_pct: float) -> float`
  - `daily_green_exit_decision(open_today: float, current_price: float, avg_entry_price: float, rr: float, min_risk_pct: float) -> tuple[bool, str]`
  - `within_eod_flatten_window(now_et: datetime, minutes_before_close: int) -> bool`
  - `compute_today_open_and_price(bars: pd.DataFrame) -> dict[str, tuple[float, float]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_daily_green_engine.py`:

```python
from datetime import datetime
import pandas as pd


def test_entry_signal():
    assert trade.daily_green_entry_signal(100.0, 101.0) is True
    assert trade.daily_green_entry_signal(100.0, 100.0) is False
    assert trade.daily_green_entry_signal(100.0, 99.0) is False
    assert trade.daily_green_entry_signal(0.0, 5.0) is False


def test_risk_distance_uses_floor():
    # raw risk dominates
    assert trade.daily_green_risk_distance(101.0, 100.0, 0.0015) == 1.0
    # floor dominates when gap above open is tiny
    assert trade.daily_green_risk_distance(100.05, 100.0, 0.0015) == 100.05 * 0.0015


def test_target_price():
    assert trade.daily_green_target_price(101.0, 100.0, 2.0, 0.0015) == 103.0


def test_exit_decision():
    # day turned red
    assert trade.daily_green_exit_decision(100.0, 99.0, 101.0, 2.0, 0.0015) == (True, "day_turned_red")
    # target reached: target = 101 + 2*1 = 103
    assert trade.daily_green_exit_decision(100.0, 103.0, 101.0, 2.0, 0.0015) == (True, "rr_2to1_target")
    # between open and target: hold
    assert trade.daily_green_exit_decision(100.0, 100.5, 101.0, 2.0, 0.0015) == (False, "")
    # no valid open
    assert trade.daily_green_exit_decision(0.0, 100.5, 101.0, 2.0, 0.0015) == (False, "")


def test_within_eod_flatten_window():
    et = trade.ET
    # Wednesday 2026-11-18, 15:56 ET, 5-min window -> inside
    assert trade.within_eod_flatten_window(datetime(2026, 11, 18, 15, 56, tzinfo=et), 5) is True
    # 15:50 -> before window
    assert trade.within_eod_flatten_window(datetime(2026, 11, 18, 15, 50, tzinfo=et), 5) is False
    # 16:30 -> after close
    assert trade.within_eod_flatten_window(datetime(2026, 11, 18, 16, 30, tzinfo=et), 5) is False
    # Saturday 2026-11-21 -> weekend
    assert trade.within_eod_flatten_window(datetime(2026, 11, 21, 15, 56, tzinfo=et), 5) is False


def test_compute_today_open_and_price():
    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-11-16 14:30", tz="UTC"), "open": 10.0, "close": 10.5},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-11-17 14:30", tz="UTC"), "open": 11.0, "close": 12.0},
            {"symbol": "BBB", "timestamp": pd.Timestamp("2026-11-17 14:30", tz="UTC"), "open": 50.0, "close": 49.0},
        ]
    )
    out = trade.compute_today_open_and_price(bars)
    assert out["AAA"] == (11.0, 12.0)
    assert out["BBB"] == (50.0, 49.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: FAIL with `AttributeError: module 'trade' has no attribute 'daily_green_entry_signal'`

- [ ] **Step 3: Implement the pure functions**

In `trade.py`, after the `_atr` function (ends at trade.py:2548, just before `class MomentumSetupScanner`), insert:

```python
# ======================================================================
# Daily-green 2:1 strategy helpers (pure, no I/O).
# ENTER when price > today's open; STOP at today's open; TARGET at
# entry + rr * risk, risk = max(entry - open, min_risk_pct * entry).
# ======================================================================
def daily_green_entry_signal(open_today: float, current_price: float) -> bool:
    return open_today > 0 and current_price > open_today


def daily_green_risk_distance(entry_price: float, open_today: float, min_risk_pct: float) -> float:
    raw = entry_price - open_today
    floor = min_risk_pct * entry_price
    return max(raw, floor)


def daily_green_target_price(entry_price: float, open_today: float, rr: float, min_risk_pct: float) -> float:
    return entry_price + rr * daily_green_risk_distance(entry_price, open_today, min_risk_pct)


def daily_green_exit_decision(
    open_today: float,
    current_price: float,
    avg_entry_price: float,
    rr: float,
    min_risk_pct: float,
) -> tuple[bool, str]:
    if open_today <= 0 or current_price <= 0:
        return (False, "")
    if current_price <= open_today:
        return (True, "day_turned_red")
    target = daily_green_target_price(avg_entry_price, open_today, rr, min_risk_pct)
    if current_price >= target:
        return (True, "rr_2to1_target")
    return (False, "")


def within_eod_flatten_window(now_et: datetime, minutes_before_close: int) -> bool:
    # Regular US equities session closes at 16:00 ET; skip weekends.
    if now_et.weekday() >= 5:
        return False
    close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    window_start = close_dt - timedelta(minutes=minutes_before_close)
    return window_start <= now_et < close_dt


def compute_today_open_and_price(bars: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if bars is None or bars.empty:
        return out
    for symbol, g in bars.groupby("symbol", sort=False):
        g = g.sort_values("timestamp")
        last = g.iloc[-1]
        open_today = to_float(last.get("open"))
        price = to_float(last.get("close"))
        out[normalize_symbol(symbol)] = (open_today, price)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: PASS (all tests in file pass)

- [ ] **Step 5: Commit**

```bash
git add trade.py tests/test_daily_green_engine.py
git commit -m "feat(trade): add pure daily-green decision helpers"
```

---

### Task 3: Pure row builders

**Files:**
- Modify: `trade.py` (module level, directly after the pure functions from Task 2)
- Test: `tests/test_daily_green_engine.py`

**Interfaces:**
- Consumes: `daily_green_entry_signal`, `daily_green_risk_distance`, `daily_green_target_price`, `daily_green_exit_decision`, `within_eod_flatten_window` (Task 2); `ScannerConfig`, `ExecutorConfig`, `BrokerPosition` (existing).
- Produces:
  - `build_daily_green_entry_rows(open_price_map: dict, held_symbols: set, no_reentry_symbols: set, equity: float, config: ScannerConfig, executor_config: ExecutorConfig) -> list[dict]`
  - `build_daily_green_exit_rows(positions: list, open_by_symbol: dict, now_et: datetime, config: ScannerConfig, equity: float) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_daily_green_engine.py`:

```python
def _cfg(**over):
    return trade.ScannerConfig(**over)


def _ecfg(**over):
    return trade.ExecutorConfig(**over)


def test_entry_rows_selects_only_green_unheld():
    open_price_map = {
        "AAA": (100.0, 101.0),  # green -> entry
        "BBB": (50.0, 49.0),    # red -> skip
        "CCC": (20.0, 20.5),    # green but held -> skip
    }
    rows = trade.build_daily_green_entry_rows(
        open_price_map, held_symbols={"CCC"}, no_reentry_symbols=set(),
        equity=13000.0, config=_cfg(), executor_config=_ecfg(),
    )
    symbols = {r["symbol"] for r in rows}
    assert symbols == {"AAA"}
    r = rows[0]
    assert r["side"] == "buy"
    assert r["reason"] == "daily_green_entry"
    assert r["stop_price"] == 100.0
    # risk = 101-100 = 1, target = 101 + 2*1 = 103
    assert r["target_price"] == 103.0
    assert r["delta_shares"] > 0


def test_entry_rows_respects_no_reentry_and_max_positions():
    open_price_map = {"AAA": (10.0, 11.0), "BBB": (10.0, 11.0), "CCC": (10.0, 11.0)}
    # no-reentry blocks AAA
    rows = trade.build_daily_green_entry_rows(
        open_price_map, held_symbols=set(), no_reentry_symbols={"AAA"},
        equity=13000.0, config=_cfg(), executor_config=_ecfg(),
    )
    assert "AAA" not in {r["symbol"] for r in rows}
    # max_positions caps count (0 held, cap 1 -> 1 row)
    rows2 = trade.build_daily_green_entry_rows(
        open_price_map, held_symbols=set(), no_reentry_symbols=set(),
        equity=13000.0, config=_cfg(daily_green_max_positions=1), executor_config=_ecfg(),
    )
    assert len(rows2) == 1


def test_entry_rows_min_notional_excludes_tiny_account():
    open_price_map = {"AAA": (10.0, 11.0)}
    # equity 500 * 0.09 = 45 < min_notional_to_trade (75) -> excluded
    rows = trade.build_daily_green_entry_rows(
        open_price_map, held_symbols=set(), no_reentry_symbols=set(),
        equity=500.0, config=_cfg(), executor_config=_ecfg(),
    )
    assert rows == []


def test_exit_rows_stop_target_and_eod():
    et = trade.ET
    pos_red = trade.BrokerPosition(symbol="AAA", qty=10, market_value=990.0, avg_entry_price=101.0, current_price=99.0)
    pos_win = trade.BrokerPosition(symbol="BBB", qty=10, market_value=1030.0, avg_entry_price=101.0, current_price=103.0)
    pos_hold = trade.BrokerPosition(symbol="CCC", qty=10, market_value=1005.0, avg_entry_price=101.0, current_price=100.5)
    open_by_symbol = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    midday = datetime(2026, 11, 18, 12, 0, tzinfo=et)

    rows = trade.build_daily_green_exit_rows([pos_red, pos_win, pos_hold], open_by_symbol, midday, _cfg(), 13000.0)
    by_sym = {r["symbol"]: r["reason"] for r in rows}
    assert by_sym["AAA"] == "day_turned_red"
    assert by_sym["BBB"] == "rr_2to1_target"
    assert "CCC" not in by_sym  # held
    for r in rows:
        assert r["side"] == "sell"
        assert r["delta_shares"] == 10

    # EOD window flattens everything
    eod = datetime(2026, 11, 18, 15, 57, tzinfo=et)
    rows_eod = trade.build_daily_green_exit_rows([pos_hold], open_by_symbol, eod, _cfg(), 13000.0)
    assert len(rows_eod) == 1 and rows_eod[0]["reason"] == "eod_flatten"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: FAIL with `AttributeError: module 'trade' has no attribute 'build_daily_green_entry_rows'`

- [ ] **Step 3: Implement the row builders**

In `trade.py`, directly after `compute_today_open_and_price` (from Task 2), insert:

```python
def build_daily_green_entry_rows(
    open_price_map: dict[str, tuple[float, float]],
    held_symbols: set[str],
    no_reentry_symbols: set[str],
    equity: float,
    config: "ScannerConfig",
    executor_config: "ExecutorConfig",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slots = config.daily_green_max_positions - len(held_symbols)
    if slots <= 0:
        return rows

    target_weight = min(config.daily_green_per_position_weight, executor_config.max_position_weight)
    for raw_symbol, (open_today, price) in open_price_map.items():
        symbol = normalize_symbol(raw_symbol)
        if symbol in held_symbols:
            continue
        if not config.daily_green_reenter_after_stop and symbol in no_reentry_symbols:
            continue
        if price <= 0 or not daily_green_entry_signal(open_today, price):
            continue

        delta_dollars = target_weight * equity
        if delta_dollars < executor_config.min_notional_to_trade:
            continue
        delta_shares = int(delta_dollars // price)
        if delta_shares <= 0:
            continue

        target_price = daily_green_target_price(price, open_today, config.daily_green_rr, config.daily_green_min_risk_pct)
        rows.append(
            {
                "symbol": symbol,
                "side": "buy",
                "delta_shares": delta_shares,
                "delta_dollars": float(delta_dollars),
                "price": float(price),
                "market_price": float(price),
                "entry_price": float(price),
                "open_today": float(open_today),
                "stop_price": float(open_today),
                "target_price": float(target_price),
                "risk_reward_ratio": float(config.daily_green_rr),
                "reason": "daily_green_entry",
                "target_weight": float(target_weight),
                "current_weight": 0.0,
                "delta_weight": float(target_weight),
                "trade_confidence": 1.0,
            }
        )
        if len(rows) >= slots:
            break
    return rows


def build_daily_green_exit_rows(
    positions: list["BrokerPosition"],
    open_by_symbol: dict[str, float],
    now_et: datetime,
    config: "ScannerConfig",
    equity: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eod = within_eod_flatten_window(now_et, config.daily_green_flatten_before_close_min)
    for pos in positions:
        if pos.qty <= 0:
            continue
        symbol = normalize_symbol(pos.symbol)

        if eod:
            reason = "eod_flatten"
        else:
            open_today = to_float(open_by_symbol.get(symbol))
            should_exit, exit_reason = daily_green_exit_decision(
                open_today,
                to_float(pos.current_price),
                to_float(pos.avg_entry_price),
                config.daily_green_rr,
                config.daily_green_min_risk_pct,
            )
            reason = exit_reason if should_exit else ""

        if not reason:
            continue
        rows.append(
            {
                "symbol": symbol,
                "side": "sell",
                "delta_shares": int(pos.qty),
                "delta_dollars": float(pos.market_value),
                "price": float(pos.current_price),
                "reason": reason,
                "target_weight": 0.0,
                "current_weight": float(pos.market_value) / max(equity, 1e-9),
                "delta_weight": -float(pos.market_value) / max(equity, 1e-9),
                "trade_confidence": 1.0,
            }
        )
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: PASS (all file tests pass)

- [ ] **Step 5: Commit**

```bash
git add trade.py tests/test_daily_green_engine.py
git commit -m "feat(trade): add daily-green entry/exit row builders"
```

---

### Task 4: Scanner glue and build_orders branch

**Files:**
- Modify: `trade.py` (`MomentumSetupScanner`: new method after `_update_high_water_and_exits` end at trade.py:3517; branch at top of `build_orders` trade.py:3519)
- Test: `tests/test_daily_green_engine.py`

**Interfaces:**
- Consumes: `build_daily_green_entry_rows`, `build_daily_green_exit_rows`, `compute_today_open_and_price`, `within_eod_flatten_window` (Tasks 2-3); `self._universe`, `self._download_bars`, `self.broker.get_account_snapshot`, `self.broker.get_all_open_positions`, `load_no_reentry_symbols`, `mark_no_reentry_symbol` (existing).
- Produces: `MomentumSetupScanner._build_daily_green_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`; `build_orders()` returns it when `self.config.daily_green_strategy_enabled`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_daily_green_engine.py`:

```python
class _FakeBroker:
    def __init__(self, equity, positions):
        self._equity = equity
        self._positions = positions

    def get_account_snapshot(self):
        return trade.AccountSnapshot(buying_power=self._equity, equity=self._equity, cash=self._equity, portfolio_value=self._equity)

    def get_all_open_positions(self):
        return self._positions


def _make_daily_green_scanner(equity, positions, bars, universe):
    s = object.__new__(trade.MomentumSetupScanner)
    s.broker = _FakeBroker(equity, positions)
    s.config = trade.ScannerConfig(daily_green_strategy_enabled=True)
    s.executor_config = trade.ExecutorConfig()
    s.data = None
    s.llm_metadata_by_symbol = {}
    s._universe = lambda: universe
    s._download_bars = lambda syms: bars
    return s


def test_build_orders_branches_to_daily_green(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "DEFAULT_REBALANCE_ORDERS", tmp_path / "rb.csv")
    monkeypatch.setattr(trade, "DEFAULT_EXIT_ORDERS", tmp_path / "ex.csv")
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())
    monkeypatch.setattr(trade, "mark_no_reentry_symbol", lambda *a, **k: None)
    # Pin the clock-dependent EOD check so the test never flakes near the close.
    monkeypatch.setattr(trade, "within_eod_flatten_window", lambda *a, **k: False)

    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-11-17 20:00", tz="UTC"), "open": 100.0, "close": 101.0},
            {"symbol": "BBB", "timestamp": pd.Timestamp("2026-11-17 20:00", tz="UTC"), "open": 50.0, "close": 49.0},
        ]
    )
    s = _make_daily_green_scanner(13000.0, positions=[], bars=bars, universe=["AAA", "BBB"])
    rebalance, exits, scores = s.build_orders()

    assert list(rebalance["symbol"]) == ["AAA"]
    assert rebalance.iloc[0]["reason"] == "daily_green_entry"
    assert exits.empty
    assert scores.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_daily_green_engine.py::test_build_orders_branches_to_daily_green -v`
Expected: FAIL (either `AttributeError: _build_daily_green_orders` or the legacy momentum path runs and raises because `self.data` is None)

- [ ] **Step 3: Add the branch at the top of build_orders**

In `trade.py`, the method starts at trade.py:3519 `def build_orders(self) -> ...:` and its first body line is `symbols = self._universe()`. Insert the branch as the first statements of the method body, before `symbols = self._universe()`:

```python
    def build_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.config.daily_green_strategy_enabled:
            return self._build_daily_green_orders()

        symbols = self._universe()
```

- [ ] **Step 4: Add the `_build_daily_green_orders` method**

In `trade.py`, insert this method inside `MomentumSetupScanner`, directly after `_update_high_water_and_exits` (which ends at trade.py:3517 with `return pd.DataFrame(rows)`) and before `def build_orders`:

```python
    def _build_daily_green_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rebalance_cols = [
            "symbol", "side", "delta_shares", "delta_dollars", "price", "reason",
            "target_weight", "current_weight", "delta_weight", "trade_confidence",
        ]
        symbols = self._universe()
        print(f"[daily-green] scanning universe symbols={len(symbols)}")

        bars = self._download_bars(symbols)
        open_price_map = compute_today_open_and_price(bars)

        acct = self.broker.get_account_snapshot()
        equity = max(float(acct.equity), 1.0)
        positions = [p for p in self.broker.get_all_open_positions() if p.qty > 0]
        held_symbols = {normalize_symbol(p.symbol) for p in positions}

        # Today's open for every held symbol, fetching any not already in the map.
        open_by_symbol = {sym: o for sym, (o, _p) in open_price_map.items()}
        missing = [sym for sym in held_symbols if sym not in open_by_symbol]
        if missing:
            extra = compute_today_open_and_price(self._download_bars(missing))
            for sym, (o, _p) in extra.items():
                open_by_symbol[sym] = o

        now_et = datetime.now(ET)

        exit_rows = build_daily_green_exit_rows(positions, open_by_symbol, now_et, self.config, equity)

        # One round-trip per symbol per day: block same-day re-entry after any exit
        # (stop or target) unless explicitly allowed.
        if not self.config.daily_green_reenter_after_stop:
            for r in exit_rows:
                if r["reason"] in ("day_turned_red", "rr_2to1_target"):
                    mark_no_reentry_symbol(r["symbol"], reason=r["reason"])

        # No new entries inside the end-of-day flatten window.
        if within_eod_flatten_window(now_et, self.config.daily_green_flatten_before_close_min):
            entry_rows: list[dict[str, Any]] = []
        else:
            no_reentry = load_no_reentry_symbols()
            entry_rows = build_daily_green_entry_rows(
                open_price_map, held_symbols, no_reentry, equity, self.config, self.executor_config
            )

        rebalance = pd.DataFrame(entry_rows, columns=rebalance_cols) if entry_rows else pd.DataFrame(columns=rebalance_cols)
        exits = pd.DataFrame(exit_rows) if exit_rows else pd.DataFrame(columns=rebalance_cols)

        rebalance.to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
        exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
        print(f"[daily-green] entries={len(rebalance)} exits={len(exits)} equity={equity:.2f}")
        return rebalance, exits, pd.DataFrame()
```

Note: `pd.DataFrame(entry_rows, columns=rebalance_cols)` keeps only the canonical columns for the plan; the extra per-row keys (`stop_price`, `target_price`, etc.) are intentionally dropped from the frame but preserved in the CSV would require the full set. If you want them in the CSV too, build the frame from `entry_rows` without the `columns=` filter first, write to CSV, then reindex to `rebalance_cols` for the return value. For this plan the canonical columns are sufficient for execution.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_daily_green_engine.py::test_build_orders_branches_to_daily_green -v`
Expected: PASS

- [ ] **Step 6: Run the whole file**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add trade.py tests/test_daily_green_engine.py
git commit -m "feat(trade): wire daily-green orders into scanner build_orders"
```

---

### Task 5: Executor bypasses for daily-green mode

**Files:**
- Modify: `trade.py` (`run_cycle_from_files` trade.py:5589, add branch after market-open check; `run_once_from_frames` trade.py:5105 and trade.py:5107, gate the exit-learning and MTF calls)
- Test: `tests/test_daily_green_engine.py`

**Interfaces:**
- Consumes: `self.config.scanner.daily_green_strategy_enabled`, `self.scanner.build_orders`, `self.run_once_from_frames` (existing).
- Produces: daily-green orders execute without `monitor_active_positions`, regime plan, market-context gate, premarket-plan gate, RL timing gate, exit-learning augmentation, or the MTF buy gate.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_daily_green_engine.py`:

```python
def _make_executor_for_bypass(daily_green_on, calls):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.scanner.daily_green_strategy_enabled = daily_green_on
    e.broker = _FakeBroker(13000.0, positions=[])
    e.account_history = []
    # Stub the heavy helpers run_once_from_frames touches.
    e._record_account_snapshot = lambda label, cid: {"equity": 13000.0, "cash": 13000.0}
    e._gross_exposure = lambda eq: 0.0
    e._market_is_open_or_wait = lambda: True
    e._kill_switch_allows_execution = lambda eq: (True, "", 0.0)
    e._write_account_equity_curve = lambda: None
    e._build_order_plan = lambda rb, ex, eq: pd.DataFrame()  # empty -> early return after bypass checks
    e._in_cooldown = lambda: False

    def spy_augment(ex):
        calls.append("augment")
        return ex if ex is not None else pd.DataFrame()

    def spy_mtf(plan):
        calls.append("mtf")
        return plan

    e._augment_exit_orders_with_exit_learning = spy_augment
    e._apply_mtf_buy_gate_to_plan = spy_mtf
    return e


def test_daily_green_bypasses_momentum_gates(monkeypatch):
    monkeypatch.setattr(trade, "save_df", lambda *a, **k: None)
    monkeypatch.setattr(trade, "load_no_reentry_symbols", lambda: set())

    calls_on = []
    e_on = _make_executor_for_bypass(True, calls_on)
    e_on.run_once_from_frames(pd.DataFrame(), pd.DataFrame())
    assert "augment" not in calls_on
    assert "mtf" not in calls_on

    calls_off = []
    e_off = _make_executor_for_bypass(False, calls_off)
    e_off.run_once_from_frames(pd.DataFrame(), pd.DataFrame())
    assert "augment" in calls_off
    assert "mtf" in calls_off
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_daily_green_engine.py::test_daily_green_bypasses_momentum_gates -v`
Expected: FAIL (both `augment` and `mtf` are called even when daily-green is on)

- [ ] **Step 3: Gate the exit-learning and MTF calls in run_once_from_frames**

In `trade.py`, replace the current lines trade.py:5105-5107:

```python
        exit_orders = self._augment_exit_orders_with_exit_learning(exit_orders if exit_orders is not None else pd.DataFrame())
        plan = self._build_order_plan(rebalance_orders, exit_orders, equity)
        plan = self._apply_mtf_buy_gate_to_plan(plan)
```

with:

```python
        daily_green = self.config.scanner.daily_green_strategy_enabled
        if daily_green:
            exit_orders = exit_orders if exit_orders is not None else pd.DataFrame()
        else:
            exit_orders = self._augment_exit_orders_with_exit_learning(exit_orders if exit_orders is not None else pd.DataFrame())
        plan = self._build_order_plan(rebalance_orders, exit_orders, equity)
        if not daily_green:
            plan = self._apply_mtf_buy_gate_to_plan(plan)
```

- [ ] **Step 4: Add the daily-green branch in run_cycle_from_files**

In `trade.py`, `run_cycle_from_files` starts at trade.py:5589. After the market-closed early-return block (which ends at trade.py:5598 with its `return ...`) and before `if self.scanner is not None:` (trade.py:5600), insert:

```python
        if self.scanner is not None and self.config.scanner.daily_green_strategy_enabled:
            try:
                rebalance, exits, _scores = self.scanner.build_orders()
                if not rebalance.empty or not exits.empty:
                    return self.run_once_from_frames(rebalance, exits)
                cycle_id = run_id()
                results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "daily_green_no_orders", "message": "no green entries and no exits this cycle"}])
                save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
                save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
                self._write_account_equity_curve()
                return pd.DataFrame(), results, self.broker.get_account_snapshot(), pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])
            except Exception as exc:
                print(f"[daily-green] cycle failed: {exc}")
                write_event("daily_green_cycle_error", {"error": str(exc)})
                return pd.DataFrame(), pd.DataFrame(), self.broker.get_account_snapshot(), pd.DataFrame()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_daily_green_engine.py::test_daily_green_bypasses_momentum_gates -v`
Expected: PASS

- [ ] **Step 6: Verify the module still imports and the full file suite passes**

Run: `python -c "import trade; print('import OK')"`
Expected: prints `import OK` (plus the universe-load line)

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add trade.py tests/test_daily_green_engine.py
git commit -m "feat(trade): bypass momentum/ML/RL gates in daily-green execution path"
```

---

### Task 6: Regression and end-to-end verification

**Files:**
- Test: run existing suite; no code changes unless a regression surfaces.

- [ ] **Step 1: Confirm daily-green OFF preserves legacy behavior**

Append to `tests/test_daily_green_engine.py`:

```python
def test_build_orders_legacy_path_untouched_when_disabled(monkeypatch):
    # With the flag off, build_orders must NOT take the daily-green branch.
    # We prove this by asserting the branch guard is the only daily-green entry.
    s = object.__new__(trade.MomentumSetupScanner)
    s.config = trade.ScannerConfig(daily_green_strategy_enabled=False)
    called = {"daily_green": False}
    s._build_daily_green_orders = lambda: called.__setitem__("daily_green", True) or (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    # Make the legacy path fail fast so we do not need network; we only care which branch ran.
    s._universe = lambda: (_ for _ in ()).throw(RuntimeError("legacy path reached"))
    try:
        s.build_orders()
    except RuntimeError as exc:
        assert "legacy path reached" in str(exc)
    assert called["daily_green"] is False
```

- [ ] **Step 2: Run the new file**

Run: `python -m pytest tests/test_daily_green_engine.py -v`
Expected: PASS (all tests, including the legacy-guard test)

- [ ] **Step 3: Run the full existing suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: the pre-existing suite result is unchanged (no new failures attributable to these changes). Note any pre-existing failures separately.

- [ ] **Step 4: Smoke-check the production config path**

Run:

```bash
python -c "import trade; c=trade.build_default_config(); print('daily_green=', c.scanner.daily_green_strategy_enabled, 'rr=', c.scanner.daily_green_rr, 'max_pos=', c.scanner.daily_green_max_positions)"
```

Expected: `daily_green= True rr= 2.0 max_pos= 8`

- [ ] **Step 5: Runtime verification (market-dependent, manual)**

Use the `verify` skill or a supervised run to drive one live/paper cycle during market hours and confirm: entries only on green-vs-open names from `universe.txt`, sells tagged `day_turned_red` / `rr_2to1_target`, and EOD flatten in the final window. If the market is closed, confirm the loop logs `skipped_market_closed` and no orders are produced.

- [ ] **Step 6: Commit**

```bash
git add tests/test_daily_green_engine.py
git commit -m "test(trade): regression guard for legacy path when daily-green disabled"
```

---

## Self-Review

**Spec coverage:**
- Entry on green-vs-open: Task 2 (`daily_green_entry_signal`), Task 3 (entry rows), Task 4 (glue). Covered.
- 2:1 RR target: Task 2 (`daily_green_target_price`), Task 3, exit decision. Covered.
- Exit on day-negative: Task 2 (`daily_green_exit_decision` -> `day_turned_red`). Covered.
- Reference = today's open: `compute_today_open_and_price` + stop_price=open. Covered.
- Equities only / replace setup-score: build_orders branch (Task 4) + executor bypass (Task 5). Covered.
- Config fields and defaults: Task 1. Covered.
- Degenerate-R floor: `daily_green_risk_distance` (Task 2), tested. Covered.
- Re-entry control: glue marks no-reentry on exit (Task 4); entry builder honors it (Task 3). Covered. Note: the plan extends decision #2 to block re-entry after a target win too (not only a red stop), to prevent same-day churn on a still-green name; `daily_green_reenter_after_stop=True` restores re-entry.
- EOD flatten: `within_eod_flatten_window` (Task 2), exit builder + entry suppression (Tasks 3-4). Covered.
- Position cap / sizing: entry builder `daily_green_max_positions` + per-position weight (Task 3). Covered.
- Testing without network/creds: pure functions + `object.__new__` fakes. Covered.

**Placeholder scan:** No TBD/TODO; every code step has complete code and exact commands.

**Type consistency:** `daily_green_exit_decision` returns `(bool, str)` consistently; row dicts use the same keys across builder and consumers; config field names match Task 1 across all tasks; `compute_today_open_and_price` returns `dict[str, tuple[float, float]]` consumed as `(open, price)` in Task 4.
