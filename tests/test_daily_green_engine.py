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
