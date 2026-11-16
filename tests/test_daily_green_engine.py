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


def test_load_trading_universe_falls_back_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "UNIVERSE_FILE", tmp_path / "nope.txt")
    out = trade.load_trading_universe()
    assert out == trade._FALLBACK_SCAN_UNIVERSE
    assert "QQQ" in out


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
