import trade
from datetime import datetime
import pandas as pd


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


def test_load_trading_universe_parses_file(tmp_path, monkeypatch):
    f = tmp_path / "universe.txt"
    f.write_text("AAPL, msft  nvda # comment\nAAPL\n")
    monkeypatch.setattr(trade, "UNIVERSE_FILE", f)
    assert trade.load_trading_universe() == ["AAPL", "MSFT", "NVDA"]


def test_load_trading_universe_fails_closed_when_missing(tmp_path, monkeypatch):
    # universe.txt is the sole trading-symbol source. A missing file must FAIL
    # CLOSED (RuntimeError) -- there is no hard-coded fallback universe anymore.
    import pytest

    monkeypatch.setattr(trade, "UNIVERSE_FILE", tmp_path / "nope.txt")
    with pytest.raises(RuntimeError):
        trade.load_trading_universe()
    assert not hasattr(trade, "_FALLBACK_SCAN_UNIVERSE")


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


def test_exit_decision_rejects_nonpositive_price():
    assert trade.daily_green_exit_decision(100.0, 0.0, 101.0, 2.0, 0.0015) == (False, "")
    assert trade.daily_green_exit_decision(100.0, -5.0, 101.0, 2.0, 0.0015) == (False, "")


def test_compute_today_open_and_price_sorts_by_timestamp():
    # rows supplied OUT OF chronological order; latest timestamp must win
    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-11-17 20:00", tz="UTC"), "open": 11.0, "close": 12.0},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-11-16 20:00", tz="UTC"), "open": 10.0, "close": 10.5},
        ]
    )
    out = trade.compute_today_open_and_price(bars)
    assert out["AAA"] == (11.0, 12.0)


def test_compute_today_open_and_price_normalizes_before_grouping():
    bars = pd.DataFrame(
        [
            {"symbol": "aapl", "timestamp": pd.Timestamp("2026-11-16 20:00", tz="UTC"), "open": 9.0, "close": 9.5},
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2026-11-17 20:00", tz="UTC"), "open": 10.0, "close": 11.0},
        ]
    )
    out = trade.compute_today_open_and_price(bars)
    assert out == {"AAPL": (10.0, 11.0)}


def test_compute_today_open_and_price_empty_and_none():
    assert trade.compute_today_open_and_price(None) == {}
    assert trade.compute_today_open_and_price(pd.DataFrame()) == {}


def test_within_eod_flatten_window_boundaries():
    et = trade.ET
    # window_start (15:55:00) inclusive -> True
    assert trade.within_eod_flatten_window(datetime(2026, 11, 18, 15, 55, 0, tzinfo=et), 5) is True
    # close (16:00:00) exclusive -> False
    assert trade.within_eod_flatten_window(datetime(2026, 11, 18, 16, 0, 0, tzinfo=et), 5) is False


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


def test_entry_rows_excludes_dust_below_min_notional_floor():
    open_price_map = {"AAA": (10.0, 11.0)}
    # equity 10 * 0.09 = 0.90 < dynamic min_notional floor ($1) -> excluded (dust)
    rows = trade.build_daily_green_entry_rows(
        open_price_map, held_symbols=set(), no_reentry_symbols=set(),
        equity=10.0, config=_cfg(), executor_config=_ecfg(),
    )
    assert rows == []


def test_entry_rows_small_account_now_trades():
    # Regression: an ~$824 account was blocked by the static $75 min_notional
    # (0.09*824 = 74.23 < 75). With the dynamic cap it must now generate an entry.
    open_price_map = {"AAA": (46.0, 47.0)}  # green, cheap enough for >=1 share on $74
    rows = trade.build_daily_green_entry_rows(
        open_price_map, held_symbols=set(), no_reentry_symbols=set(),
        equity=824.74, config=_cfg(), executor_config=_ecfg(),
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["delta_shares"] >= 1


def test_risk_profile_caps_small_account_but_preserves_large():
    small = trade.build_account_risk_profile(824.74)
    assert small.min_cash_buffer <= 824.74 * trade.MAX_CASH_BUFFER_FRACTION + 1e-9
    assert small.min_notional_to_trade <= 824.74 * trade.MAX_MIN_NOTIONAL_FRACTION + 1e-9
    # 9% per-position sizing must clear the dynamic min_notional on a small account
    assert 0.09 * 824.74 > small.min_notional_to_trade

    big = trade.build_account_risk_profile(13_000.0)
    # ~$13K account: fixed floors are below the fractional caps, so unchanged
    assert big.min_cash_buffer == 1_500.0
    assert big.min_notional_to_trade == 75.0


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


def test_direction_confirmation_bypassed_in_daily_green():
    plan = pd.DataFrame([{"symbol": "AAA", "side": "buy"}])
    calls = []

    class _Scanner:
        def confirm_buy_direction(self, syms):
            calls.append(list(syms))
            return pd.DataFrame([{"symbol": "AAA", "direction_confirmed": True}])

    e_on = object.__new__(trade.ProductionOrderExecutor)
    e_on.config = trade.ExecutorConfig()
    e_on.config.scanner.daily_green_strategy_enabled = True
    e_on.scanner = _Scanner()
    assert e_on._direction_confirmations_for_buys(plan) == {}
    assert calls == []

    e_off = object.__new__(trade.ProductionOrderExecutor)
    e_off.config = trade.ExecutorConfig()
    e_off.config.scanner.daily_green_strategy_enabled = False
    e_off.config.scanner.require_direction_confirmation_before_buy = True
    e_off.scanner = _Scanner()
    out = e_off._direction_confirmations_for_buys(plan)
    assert calls == [["AAA"]]
    assert out == {"AAA": {"symbol": "AAA", "direction_confirmed": True}}


def test_within_eod_flatten_window_early_close():
    et = trade.ET
    close = datetime(2026, 11, 27, 13, 0, 0, tzinfo=et)  # early-close day, 13:00 ET
    assert trade.within_eod_flatten_window(datetime(2026, 11, 27, 12, 57, tzinfo=et), 5, close) is True
    assert trade.within_eod_flatten_window(datetime(2026, 11, 27, 12, 50, tzinfo=et), 5, close) is False
    assert trade.within_eod_flatten_window(datetime(2026, 11, 27, 13, 30, tzinfo=et), 5, close) is False
    # default path (no close_dt) still uses 16:00
    assert trade.within_eod_flatten_window(datetime(2026, 11, 18, 15, 56, tzinfo=et), 5) is True


def test_exit_rows_eod_with_early_close():
    et = trade.ET
    close = datetime(2026, 11, 27, 13, 0, 0, tzinfo=et)
    pos = trade.BrokerPosition(symbol="AAA", qty=10, market_value=1005.0, avg_entry_price=101.0, current_price=100.5)
    rows = trade.build_daily_green_exit_rows([pos], {"AAA": 100.0}, datetime(2026, 11, 27, 12, 57, tzinfo=et), trade.ScannerConfig(), 13000.0, close)
    assert len(rows) == 1 and rows[0]["reason"] == "eod_flatten"


# ---------------------------------------------------------------------------
# M1: fractional orders
# ---------------------------------------------------------------------------
def _executor_frac(frac):
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.scanner.enable_fractional_orders = frac
    return e


def test_enable_fractional_orders_default_on():
    assert trade.ScannerConfig().enable_fractional_orders is True


def test_normalize_order_qty():
    assert trade.normalize_order_qty(3.0) == 3
    assert isinstance(trade.normalize_order_qty(3.0), int)
    assert trade.normalize_order_qty(0.1119) == 0.1119
    assert trade.normalize_order_qty(0.12345678912) == round(0.12345678912, 9)


def test_qty_from_dollars_fractional_vs_integer():
    assert _executor_frac(True)._qty_from_dollars(74.23, 661.0) == round(74.23 / 661.0, 9)
    assert _executor_frac(False)._qty_from_dollars(74.23, 661.0) == 0
    assert _executor_frac(True)._qty_from_dollars(100.0, 50.0) == 2.0
    assert _executor_frac(False)._qty_from_dollars(100.0, 50.0) == 2
    assert _executor_frac(True)._qty_from_dollars(50.0, 0.0) == 0


def test_entry_rows_fractional_high_priced_name():
    opx = {"QQQ": (660.0, 661.0)}  # green vs open
    on = trade.build_daily_green_entry_rows(opx, set(), set(), 824.74, _cfg(), _ecfg())
    assert len(on) == 1 and 0 < on[0]["delta_shares"] < 1  # fractional share
    off = trade.build_daily_green_entry_rows(
        opx, set(), set(), 824.74, _cfg(enable_fractional_orders=False), _ecfg()
    )
    assert off == []  # integer shares -> 0 -> excluded


def test_exit_rows_sell_full_fractional_qty():
    midday = datetime(2026, 11, 18, 12, 0, tzinfo=trade.ET)
    frac_pos = trade.BrokerPosition(symbol="QQQ", qty=0.1123, market_value=71.9, avg_entry_price=659.0, current_price=640.0)
    rows = trade.build_daily_green_exit_rows([frac_pos], {"QQQ": 660.0}, midday, _cfg(), 824.74)
    assert len(rows) == 1 and abs(rows[0]["delta_shares"] - 0.1123) < 1e-9  # full fractional qty, not truncated
    int_pos = trade.BrokerPosition(symbol="QQQ", qty=10, market_value=6400.0, avg_entry_price=659.0, current_price=640.0)
    rows2 = trade.build_daily_green_exit_rows([int_pos], {"QQQ": 660.0}, midday, _cfg(), 824.74)
    assert rows2[0]["delta_shares"] == 10.0
